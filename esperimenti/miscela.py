"""Esiste una "dose giusta" fra direzione testuale grezza e direzione visiva predetta?

Miscela le due direzioni, `c(beta) = (1-beta)*c_grezza + beta*c_predetta`, con la direzione
predetta dalla mappa lineare testo->visione stimata sui tre domini (CelebA visti + AwA2 +
OVAD, ridge in forma chiusa, `esperimenti/frecce_tre_domini.pt`). Si spazza `beta` da 0 a 1 e
il passo `alpha`, e si legge la Recall@10 sui tre strati contro `beta=0` a parita' di alpha.

Nasce dal fatto che la mappa da sola perde fino a 15pp (`mappa_retrieval.py`): la domanda e'
se una dose intermedia recuperi qualcosa. Esito registrato in `docs/piano-interpretabilita.md`
sez. 2e: la curva scende in modo monotono su tutti e tre gli split, nessuna dose e' positiva.

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

Uso:    HF_HUB_OFFLINE=1 python esperimenti/miscela.py --seed 2026 --split 0
Output: esperimenti/miscela_split{split}.json
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



import json
# ==========================================================================================
# LA MAPPA APPRESA SUI TRE DOMINI AIUTA IL RETRIEVAL SUGLI ATTRIBUTI MAI VISTI DI CELEBA?
#
# La mappa lineare direzione-testuale -> direzione-visiva, stimata su 217 attributi (85 AwA2
# + 102 OVAD + i 30 CelebA VISTI di questo split), triplica l'allineamento coseno sugli
# attributi nuovi. Ma il coseno non predice il Recall (smentita del 28/08): questo e' il test
# che conta. Passo normalizzato, per non confondere direzione e lunghezza del passo.
#
# Gli attributi tenuti fuori NON entrano mai nella stima di W.
# ==========================================================================================
FR = torch.load(ROOT / "esperimenti" / "frecce_tre_domini.pt", map_location="cpu")
Tf, Vf, DOMf, ATTf = FR["T"], FR["V"], FR["dom"], FR["attr"]
tenuti = {attr_names[i] for i in HOLD}
usa = [i for i in range(len(ATTf)) if not (DOMf[i] == "celeba" and ATTf[i] in tenuti)]
n_cel = sum(1 for i in usa if DOMf[i] == "celeba")
print(f"\nmappa stimata su {len(usa)} attributi ({n_cel} celeba visti, "
      f"{sum(1 for i in usa if DOMf[i]=='awa2')} awa2, "
      f"{sum(1 for i in usa if DOMf[i]=='ovad')} ovad); "
      f"i {len(tenuti)} tenuti fuori sono esclusi", flush=True)
assert not (set(ATTf[i] for i in usa if DOMf[i] == "celeba") & tenuti), "fuga di attributi!"

Tr, Vr = Tf[usa], Vf[usa]
dt = d_ens / d_ens.norm(dim=-1, keepdim=True)

def mappa(lam):
    W = torch.linalg.solve(Tr.T @ Tr + lam * torch.eye(D), Tr.T @ Vr)
    C = dt @ W
    return C / C.norm(dim=-1, keepdim=True).clamp(min=1e-8)

def make_q(T, alpha):
    cache = {}
    def f(ref, entry):
        k = entry["query"]
        if k not in cache:
            c = torch.zeros(D)
            for a in entry["pos"]: c = c + T[a]
            for b in entry["neg"]: c = c - T[b]
            cache[k] = c / c.norm().clamp(min=1e-8)
        q = ref + alpha * cache[k]
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f

# ==========================================================================================
# MISCELA: la direzione visiva e' una SPINTA o una DESTINAZIONE?
#
# L'MLP del contributo candidato sposta la direzione verso quella visiva (coseno correzione
# <-> visiva +0.239) ma conserva coseno 0.779 con quella testuale di partenza: e' una spinta
# piccola. La mappa ridge invece SOSTITUISCE la direzione, e perde 15pp.
# Se la lettura e' giusta, mescolandole deve esistere un ottimo INTERMEDIO.
#   beta = 0  -> baseline (solo testo)      beta = 1 -> mappa ridge pura (nota: -15pp)
# ==========================================================================================
ALPHAS = (1.0, 1.5)
BETA = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
LAM_FISSO = 1000.0                      # il meno dannoso nella prova precedente
Cmap = mappa(LAM_FISSO)

def miscela(beta):
    M = (1 - beta) * dt + beta * Cmap
    return M / M.norm(dim=-1, keepdim=True).clamp(min=1e-8)

print("\n" + "=" * 100)
print("MISCELA fra direzione testuale e direzione visiva predetta (Recall@10)")
print("=" * 100)
R = {}
for b in BETA:
    T = miscela(b)
    for a in ALPHAS: R[(b, a)] = evaluate(make_q(T, a), devset, E_valid, K_VALUES)[10]

res_out = {}
for strato, idx in (("VISTI", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix)):
    if not idx: continue
    print(f"\n--- {strato} ({len(idx)} query)     differenza vs beta=0, stesso alpha")
    print(f"  {'beta':<8}" + "".join(f"{'a=' + str(a):>24}" for a in ALPHAS))
    for b in BETA:
        parti = []
        for a in ALPHAS:
            v = macro([R[(b, a)][i] for i in idx]) * 100
            if b == 0.0:
                parti.append(f"{v:>23.2f}%")
            else:
                st = bootstrap_paired([R[(b,a)][i] for i in idx],
                                      [R[(0.0,a)][i] for i in idx], n_boot=20000, seed=0)
                parti.append(f"{v:>10.2f}% {st['delta']*100:>+7.2f}({st['p']:.3f})")
            res_out[f"{strato}|b{b}|a{a}"] = v
        print(f"  {b:<8}" + "".join(parti))

with open(ROOT / "esperimenti" / f"miscela_split{args.split}.json", "w") as f:
    json.dump({"seed": SEED, "split": args.split, "lam": LAM_FISSO,
               "holdout": sorted(tenuti), "recall10": res_out}, f, indent=2)
print("\nsalvato.")
