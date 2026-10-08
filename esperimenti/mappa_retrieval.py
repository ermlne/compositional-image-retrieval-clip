"""La mappa testo->visione dei tre domini aiuta il RETRIEVAL sugli attributi mai visti?

La mappa lineare stimata su 217 attributi (85 AwA2 + 102 OVAD + i 30 CelebA visti dello split)
triplica l'allineamento coseno sulle direzioni nuove. Ma il coseno non predice la Recall: qui
si sostituisce la direzione testuale con quella mappata dentro la query baseline e si legge la
Recall@10 sui tre strati, su una griglia passo `alpha` x regolarizzazione `lambda`, a passo
normalizzato per non confondere direzione e lunghezza.

Gli attributi tenuti fuori non entrano mai nella stima di `W`. Legge
`esperimenti/frecce_tre_domini.pt`, che va rigenerato prima con
`esperimenti/relazione_testo_visione.py` (~5 minuti, non passa da git).

Esito in `docs/piano-interpretabilita.md` sez. 2d: nemmeno una casella positiva; il danno e'
uguale sui visti e sui mai visti, quindi il bersaglio e' sbagliato, non la generalizzazione.

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

Uso:    HF_HUB_OFFLINE=1 python esperimenti/mappa_retrieval.py --seed 2026 --split 0
Output: esperimenti/mappa_tre_domini_split{split}.json
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

ALPHAS = (0.5, 1.0, 1.5, 2.0)
LAM = (1.0, 10.0, 100.0, 1000.0)
print("\n" + "=" * 100)
print("RECALL@10 — la mappa dei tre domini applicata alle direzioni di CelebA")
print("=" * 100)
R = {("baseline", a): evaluate(make_q(dt, a), devset, E_valid, K_VALUES)[10] for a in ALPHAS}
for l in LAM:
    C = mappa(l)
    for a in ALPHAS: R[(f"mappa lam={l:g}", a)] = evaluate(make_q(C, a), devset, E_valid, K_VALUES)[10]

res_out = {}
for strato, idx in (("VISTI", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix)):
    if not idx: continue
    print(f"\n--- {strato} ({len(idx)} query)")
    print(f"  {'variante':<18}" + "".join(f"{'a=' + str(a):>21}" for a in ALPHAS))
    for nome in ["baseline"] + [f"mappa lam={l:g}" for l in LAM]:
        parti = []
        for a in ALPHAS:
            v = macro([R[(nome, a)][i] for i in idx]) * 100
            if nome == "baseline":
                parti.append(f"{v:>20.2f}%")
            else:
                st = bootstrap_paired([R[(nome,a)][i] for i in idx],
                                      [R[("baseline",a)][i] for i in idx], n_boot=10000, seed=0)
                parti.append(f"{v:>8.2f}% {st['delta']*100:>+6.2f}({st['p']:.3f})")
            res_out[f"{strato}|{nome}|a{a}"] = v
        print(f"  {nome:<18}" + "".join(parti))

with open(ROOT / "esperimenti" / f"mappa_tre_domini_split{args.split}.json", "w") as f:
    json.dump({"seed": SEED, "split": args.split, "n_fit": len(usa),
               "holdout": sorted(tenuti), "recall10": res_out}, f, indent=2)
print("\nsalvato.")
