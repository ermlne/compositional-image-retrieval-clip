"""Una ROTAZIONE puo' generalizzare dove una matrice libera no?

Cerca in forma chiusa (Procrustes ortogonale) la rotazione `R` che meglio allinea direzioni
testuali e direzioni visive **sui soli 30 attributi visti**, verifica il vincolo
(`R^T R = I` a meno di 1e-6), e la applica alle direzioni della query baseline a varie scale.
L'argomento: una rotazione preserva angoli e norme, quindi non puo' collassare verso il
sottospazio dei 30 come fa una matrice libera.

Riporta l'allineamento prima/dopo su visti e mai visti, e la Recall@10 sui tre strati.
Esito in `docs/piano-interpretabilita.md` sez. 2b: sestuplica l'allineamento sui visti e lo
PEGGIORA sui mai visti in 3 split su 3.

--------------------------------------------------------------------------------------------
NOTA DI PROVENIENZA. Questo file nasce come **fork di
`contributo-candidato/attributi_heldout.py`** e ne conserva l'impalcatura: gli stessi 3 split
leave-attribute-out (`random.Random(--split).sample(attributi_del_devset, 10)`), la stessa
configurazione (tau 0.03, 30 epoche, max_sources 80, corpus di 20.000 immagini del train), i
tre strati di valutazione (visti / MAI VISTI / misti) e il bootstrap accoppiato su R@10.
NON riaddestra e non carica alcun modello: l'intervento e' sulle DIREZIONI
testuali della query baseline `q = normalize(r + c)`.
Fino al 30/08 l'intestazione era ancora quella di `attributi_heldout.py` ("GENERALIZZA AD
ATTRIBUTI MAI VISTI?"), copiata insieme al resto e mai aggiornata: descriveva un esperimento
diverso da quello che il file esegue. Sostituita.

Il benchmark ufficiale non viene toccato.

Uso:    HF_HUB_OFFLINE=1 python esperimenti/procruste.py --seed 2026 --split 0
Output: esperimenti/procruste_seed{SEED}_split{split}.json
--------------------------------------------------------------------------------------------
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark, build_benchmark_from_labels
from src.devset import bootstrap_paired, macro, sample_queries

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--split", type=int, default=0, help="quale split di attributi (0,1,2)")
ap.add_argument("--n-holdout", type=int, default=10)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--tau", type=float, default=0.03)
ap.add_argument("--threads", type=int, default=14)
args = ap.parse_args()
torch.set_num_threads(args.threads)

MODEL_ID = "openai/clip-vit-base-patch32"
SEED = args.seed
K_VALUES = [1, 5, 10]
N_TRAIN_CORPUS, MAX_SOURCES = 20000, 80
QUERY_TARGETS = {1: 80, 2: 80, 3: 60}
BATCH, LR, WD = 256, 3e-4, 1e-4
HIDDEN = 512
OUT = HERE / "multiseed"
OUT.mkdir(exist_ok=True)

t0 = time.time()


def load_split(split):
    d = torch.load(ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"], d


E_train_all, L_train_all, meta = load_split("train")
E_valid, L_valid, _ = load_split("valid")
attr_names = [n for n in meta["attr_names"] if n.strip()]
D, N_ATTR = E_valid.shape[1], L_valid.shape[1]
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)

used = sorted({a for e in devset for a in e["queried"]})
HOLD = set(random.Random(args.split).sample(used, args.n_holdout))
SEEN_POOL = [attr_names[i] for i in range(N_ATTR) if i not in HOLD]
print("=" * 92)
print(f"ATTRIBUTI MAI VISTI — seed {SEED}, split {args.split}")
print("=" * 92)
print("tenuti fuori: " + ", ".join(sorted(attr_names[i] for i in HOLD)))

idx_seen = [i for i, e in enumerate(devset) if not (set(e["queried"]) & HOLD)]
idx_hold = [i for i, e in enumerate(devset) if set(e["queried"]) <= HOLD]
idx_mix = [i for i in range(len(devset)) if i not in idx_seen and i not in idx_hold]
print(f"query di sviluppo: {len(idx_seen)} solo visti, {len(idx_hold)} SOLO MAI VISTI, "
      f"{len(idx_mix)} miste", flush=True)

clip_model = CLIPModel.from_pretrained(MODEL_ID).eval()
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
TEMPLATES = [
    ("a photo of a person who is {}",        "a photo of a person who is not {}"),
    ("a photo of a person with {}",          "a photo of a person without {}"),
    ("a face with {}",                       "a face without {}"),
    ("{}",                                   "not {}"),
    ("a portrait photo of someone with {}",  "a portrait photo of someone with no {}"),
    ("a photo of a {} person",               "a photo of a person that is not {}"),
    ("a cropped photo of a face with {}",    "a cropped photo of a face without {}"),
    ("a close-up photo of a person with {}", "a close-up photo of a person without {}"),
    ("a photo of someone who has {}",        "a photo of someone who has no {}"),
    ("a picture of a face showing {}",       "a picture of a face not showing {}"),
]


@torch.no_grad()
def encode(prompts):
    out = clip_model.get_text_features(**tokenizer(prompts, padding=True, return_tensors="pt"))
    f = out.pooler_output if hasattr(out, "pooler_output") else out
    f = f.float()
    return f / f.norm(dim=-1, keepdim=True)


d_ens = torch.zeros(N_ATTR, D)
for i, name in enumerate(attr_names):
    c = name.replace("_", " ").lower()
    v = encode([t.format(c) for t, _ in TEMPLATES]).mean(0) - \
        encode([t.format(c) for _, t in TEMPLATES]).mean(0)
    d_ens[i] = v / v.norm()
del clip_model, tokenizer

g = torch.Generator().manual_seed(SEED)
sub = torch.randperm(len(E_train_all), generator=g)[:N_TRAIN_CORPUS].sort().values
E_tr, L_tr = E_train_all[sub], L_train_all[sub]
del E_train_all, L_train_all

# le query di addestramento pescano SOLO dagli attributi visti
queries_txt = []
for n_cond, quanti in QUERY_TARGETS.items():
    queries_txt += sample_queries(SEEN_POOL, quanti, n_cond, exclude=queries_txt,
                                  seed=SEED + n_cond)
train_bench = []
for qt in queries_txt:
    train_bench += build_benchmark_from_labels([qt], L_tr, attr_names,
                                               max_sources=MAX_SOURCES, seed=SEED)
assert not any(set(e["queried"]) & HOLD for e in train_bench), "attributo tenuto fuori nel fit"
print(f"addestramento: {len(train_bench)} query valide su {len(queries_txt)}, "
      f"nessuna tocca gli attributi tenuti fuori  ({time.time()-t0:.0f}s)", flush=True)


def cond_tensor(entry, max_n=3):
    u = torch.zeros(max_n, D)
    m = torch.zeros(max_n, dtype=torch.bool)
    k = 0
    for a in entry["pos"][:max_n]:
        u[k] = d_ens[a]; m[k] = True; k += 1
    for b in entry["neg"][:max_n - k]:
        u[k] = -d_ens[b]; m[k] = True; k += 1
    return u, m


def cond_of(entry):
    c = torch.zeros(D)
    for a in entry["pos"]: c = c + d_ens[a]
    for b in entry["neg"]: c = c - d_ens[b]
    return c


def evaluate(build_query, bench, embeddings, k_values, batch_size=512):
    per_source = {k: [] for k in k_values}
    k_max = max(k_values)
    for entry in bench:
        sources = entry["sources"]
        rec = {k: np.zeros(len(sources)) for k in k_values}
        for start in range(0, len(sources), batch_size):
            stop = min(start + batch_size, len(sources))
            bs = sources[start:stop]
            q = build_query(embeddings[bs], entry)
            sim = q @ embeddings.T
            sim[torch.arange(stop - start), bs] = -torch.inf
            for j, retrieved in enumerate(sim.topk(k_max, dim=-1).indices.tolist()):
                gt = entry["gt"][start + j]
                cum = np.array([i in gt for i in retrieved], dtype=bool).cumsum()
                for k in k_values:
                    rec[k][start + j] = float(cum[k - 1] > 0)
        for k in k_values:
            per_source[k].append(rec[k])
    return per_source


def make_builder(model):
    cache = {}
    def f(ref, entry):
        if entry["query"] not in cache:
            cache[entry["query"]] = cond_tensor(entry)
        u, m = cache[entry["query"]]
        with torch.no_grad():
            return model(ref, u.unsqueeze(0).expand(len(ref), -1, -1),
                         m.unsqueeze(0).expand(len(ref), -1))
    return f


def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)



ps_b = evaluate(build_baseline, devset, E_valid, K_VALUES)
res = {"seed": SEED, "split": args.split, "n_holdout": args.n_holdout,
       "holdout": sorted(attr_names[i] for i in HOLD)}

# ==========================================================================================
# VINCOLO DI ROTAZIONE (Procrustes ortogonale) — Arianna
#
# La matrice libera fallisce sugli attributi mai visti perche' COLLASSA: schiaccia le
# direzioni nuove verso la regione di quelle note (Filippo, stessa diagnosi sulla bilineare;
# rango 589 su 1025). Una rotazione non puo' collassare: preserva angoli e distanze, e'
# invertibile, non esiste direzione che venga compressa verso un'altra.
#
# Cerchiamo la rotazione R che meglio allinea le direzioni TESTUALI a quelle VISIVE, usando
# SOLO gli attributi visti. Poi la applichiamo anche a quelli tenuti fuori.
# Soluzione in forma chiusa: R = U V^T dalla SVD di D_txt^T D_vis (problema di Procrustes).
#
# Nessun addestramento, nessun parametro appreso oltre alla rotazione stessa.
# ==========================================================================================
print("\n" + "=" * 92)
print("VINCOLO DI ROTAZIONE (Procrustes) — una rotazione puo' generalizzare dove una matrice no?")
print("=" * 92)

SEEN_IDX = sorted(i for i in range(N_ATTR) if i not in HOLD)

# direzione VISIVA di ogni attributo: centroide delle immagini che lo hanno meno quelle che no
d_vis = torch.zeros(N_ATTR, D)
for a in range(N_ATTR):
    pos = L_tr[:, a] == 1
    neg = ~pos
    if pos.sum() < 20 or neg.sum() < 20:
        continue
    v = E_tr[pos].mean(0) - E_tr[neg].mean(0)
    d_vis[a] = v / v.norm().clamp(min=1e-8)

VALIDI = [a for a in SEEN_IDX if d_vis[a].norm() > 0]
print(f"direzioni visive stimate su {len(VALIDI)} attributi visti (di {len(SEEN_IDX)})")

Dt = d_ens[VALIDI] / d_ens[VALIDI].norm(dim=-1, keepdim=True)
Dv = d_vis[VALIDI]
print(f"allineamento testo-visivo PRIMA della rotazione: coseno medio "
      f"{float((Dt * Dv).sum(-1).mean()):.4f}")

U_, S_, Vt_ = torch.linalg.svd(Dt.T @ Dv)
R = U_ @ Vt_                                    # ortogonale per costruzione
err = float((R.T @ R - torch.eye(D)).abs().max())
print(f"R e' ortogonale? scarto massimo da I: {err:.2e}")
print(f"allineamento testo-visivo DOPO  la rotazione: coseno medio "
      f"{float(((Dt @ R) * Dv).sum(-1).mean()):.4f}")

# quanto e' fedele la rotazione sugli attributi MAI VISTI (che non ha mai visto nel fit)?
HV = [a for a in sorted(HOLD) if d_vis[a].norm() > 0]
if HV:
    Ht = d_ens[HV] / d_ens[HV].norm(dim=-1, keepdim=True)
    print(f"  sugli attributi MAI VISTI: prima {float((Ht * d_vis[HV]).sum(-1).mean()):.4f}"
          f"  dopo {float(((Ht @ R) * d_vis[HV]).sum(-1).mean()):.4f}")

def make_rot(scala):
    def f(ref, entry):
        c = cond_of(entry)
        q = ref + scala * (c @ R)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f

res["procruste"] = {}
print(f"\n{'scala':<10}{'strato':<13}{'baseline':>10}{'ruotato':>10}{'differenza':>13}{'p':>9}")
for scala in (0.5, 1.0, 1.5):
    ps_r = evaluate(make_rot(scala), devset, E_valid, K_VALUES)
    res["procruste"][f"scala={scala}"] = {}
    for nome, idx in (("visti", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix)):
        if not idx: continue
        a_ = [ps_r[10][i] for i in idx]; b_ = [ps_b[10][i] for i in idx]
        st = bootstrap_paired(a_, b_, n_boot=20000, seed=0)
        res["procruste"][f"scala={scala}"][nome] = {"delta_pp": st["delta"]*100,
                                                    "p": st["p"], "metodo": macro(a_)*100}
        print(f"{scala:<10}{nome:<13}{macro(b_)*100:>9.2f}%{macro(a_)*100:>9.2f}%"
              f"{st['delta']*100:>+12.2f}pp{st['p']:>9.4f}")
    print()

with open(ROOT / "esperimenti" / f"procruste_seed{SEED}_split{args.split}.json", "w") as f:
    json.dump(res, f, indent=2)
print("salvato.")
