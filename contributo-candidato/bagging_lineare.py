"""IL BAGGING SI SCRIVE COME UNA MATRICE SOLA?

===========================================================================================
LA DOMANDA
===========================================================================================
Il bagging media le USCITE di N modelli: delta_ens = media_k(gamma_k * MLP_k([r;c])). Costa
quindi N volte la baseline a inferenza, ed e' l'unica obiezione pratica seria al metodo —
finora il suo pregio era costare esattamente quanto la baseline.

Ma sappiamo (docs/report-27-08-2026-multiseed.md) che ogni MLP e' quasi una mappa lineare:
approssimandolo con `W_r r + W_c c + b` si conserva il 93% del guadagno, R^2 = 0.99. E la
media di N mappe lineari e' una mappa lineare. Quindi, SE l'approssimazione regge anche
sull'uscita mediata, il bagging si collassa in

    q = normalize( r + (W_r r + W_c c + b) )     con W = media_k(gamma_k * W_k)

cioe' due matrici e un bias, al costo della baseline. Non piu' un ensemble: N stime rumorose
della stessa matrice, mediate in una sola.

DUE CONTROLLI, non uno:
  1. NUMERICO — fittare la mappa sull'uscita mediata deve dare la stessa matrice che si
     ottiene mediando le mappe fittate una per una. E' vero per costruzione (la ridge e'
     lineare nel bersaglio) e serve da verifica che il codice faccia quello che dice.
  2. DI RETRIEVAL — la matrice collassata deve valere quanto l'ensemble sul set di sviluppo.
     E' questo il numero che conta: R^2 alto non garantisce Recall uguale, perche' la Recall
     dipende dall'ordinamento (misurato: la stessa approssimazione costa 0.5pp sul singolo).

La mappa e' stimata su reference del TRAIN, mai sul set di sviluppo su cui si valuta.
La ridge (lambda=1e-3) non e' un dettaglio: il sistema e' mal posto (rango 589 su 1025) e
`lstsq` non e' riproducibile — vedi il report del 27/08.

NOTA SUL PUNTO OPERATIVO. Questi checkpoint sono a `max_sources` 80, il regime pre-29/08.
Filippo ha poi mostrato che alzando quel parametro un modello passa a 32.26% e l'ensemble a
36.62%. La domanda qui e' geometrica — la media di mappe quasi lineari e' quasi lineare — e
non dipende da quanti dati abbiano visto i modelli, ma il numero operativo va riverificato
sui checkpoint nuovi (`esperimenti/notte/checkpoints/`, non in repo).

Uso:  HF_HUB_OFFLINE=1 python contributo-candidato/bagging_lineare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark
from src.devset import bootstrap_paired, macro, evaluate_per_source, sample_queries

torch.set_num_threads(12)
SEEDS = [1, 2, 3, 4]          # gli stessi quattro di ensemble_seed.py; 2026 resta fuori
RIDGE = 1e-3
K = [1, 5, 10]
SEED_FIT = 999                # corpus e query per stimare W: non e' il seed di nessun modello
N_CORPUS, PER_QUERY = 20000, 60
QUERY_TARGETS = {1: 80, 2: 80, 3: 60}

# --------------------------------------------------------------------------- modelli
modelli, gammas = [], []
for sd in SEEDS:
    ck = torch.load(HERE / f"checkpoints/models_seed{sd}.pt", map_location="cpu",
                    weights_only=False)
    D, H = ck["config"]["d"], ck["config"]["hidden"]
    net = nn.Sequential(nn.Linear(2 * D, H), nn.GELU(), nn.Linear(H, D))
    net.load_state_dict({k[4:]: v for k, v in ck["A"].items() if k.startswith("net.")})
    net.eval()
    modelli.append(net)
    gammas.append(float(ck["A"]["gamma"]))
d_ens, attr_names = ck["dirs_text"], ck["attr_names"]
print(f"{len(modelli)} modelli, gamma = {[round(g, 3) for g in gammas]}")


def load_split(split):
    d = torch.load(ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"]


E_valid, _ = load_split("valid")
E_train, L_train = load_split("train")
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)
attr_to_idx = {n: i for i, n in enumerate(attr_names)}


def cond_of(entry):
    c = torch.zeros(D)
    for a in entry["pos"]: c = c + d_ens[a]
    for b in entry["neg"]: c = c - d_ens[b]
    return c


def delta_k(k, r, c):
    with torch.no_grad():
        return gammas[k] * modelli[k](torch.cat([r, c], dim=-1))


def delta_ens(r, c):
    return torch.stack([delta_k(k, r, c) for k in range(len(modelli))]).mean(0)


# ------------------------------------------------- coppie (r, c) dal TRAIN per stimare W
g = torch.Generator().manual_seed(SEED_FIT)
sub = torch.randperm(len(E_train), generator=g)[:N_CORPUS].sort().values
E_fit, L_fit = E_train[sub], L_train[sub]
del E_train

qs = []
for n_cond, quanti in QUERY_TARGETS.items():
    qs += sample_queries(list(attr_names), quanti, n_cond, exclude=qs, seed=SEED_FIT + n_cond)

X, Yk = [], [[] for _ in modelli]
for qt in qs:
    pos, neg = [], []
    for tok in qt.split(","):
        tok = tok.strip()
        (pos if tok[0] == "+" else neg).append(attr_to_idx[tok[1:].strip()])
    viol = torch.ones(len(L_fit), dtype=torch.bool)
    for a in pos: viol &= L_fit[:, a] == 0
    for b in neg: viol &= L_fit[:, b] == 1
    src = viol.nonzero(as_tuple=True)[0]
    if len(src) < 10:
        continue
    gs = torch.Generator().manual_seed(SEED_FIT)
    pick = src[torch.randperm(len(src), generator=gs)[:PER_QUERY]]
    R = E_fit[pick]
    c = cond_of({"pos": pos, "neg": neg}).expand(len(R), -1)
    X.append(torch.cat([R, c], 1))
    for k in range(len(modelli)):
        Yk[k].append(delta_k(k, R, c))
X = torch.cat(X)
Yk = [torch.cat(y) for y in Yk]
Y_ens = torch.stack(Yk).mean(0)
print(f"coppie (r, c) per il fit: {len(X)} da {len(qs)} query del train", flush=True)

gl = torch.Generator().manual_seed(0)
idx = torch.randperm(len(X), generator=gl)
tr, te = idx[:int(.8 * len(X))], idx[int(.8 * len(X)):]


def ridge(Xs, Ys, lam=RIDGE):
    A = torch.cat([Xs, torch.ones(len(Xs), 1)], 1).double()
    n = A.shape[1]
    return torch.linalg.solve(A.T @ A + lam * torch.eye(n, dtype=torch.float64),
                              A.T @ Ys.double()).float()


def r2_of(W, Xs, Ys):
    pred = torch.cat([Xs, torch.ones(len(Xs), 1)], 1) @ W
    return (1 - ((Ys - pred) ** 2).sum() / ((Ys - Ys.mean(0)) ** 2).sum()).item()


W_ens = ridge(X[tr], Y_ens[tr])                       # fit sull'uscita MEDIATA
W_sing = [ridge(X[tr], Yk[k][tr]) for k in range(len(modelli))]
W_media = torch.stack(W_sing).mean(0)                 # media delle mappe fittate una per una

print("\n--- controllo 1: fittare la media == mediare i fit? ---")
print(f"  max|W_ens - W_media| = {(W_ens - W_media).abs().max():.2e}   "
      f"(atteso ~0: la ridge e' lineare nel bersaglio)")
print(f"  R^2 della mappa sull'uscita mediata, su dati mai usati per il fit: "
      f"{r2_of(W_ens, X[te], Y_ens[te]):.4f}")
for k in range(len(modelli)):
    print(f"    R^2 del singolo {SEEDS[k]}: {r2_of(W_sing[k], X[te], Yk[k][te]):.4f}")

# --------------------------------------------------------------------------- valutazione
def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)


def build_singolo(k):
    def f(ref, entry):
        q = ref + delta_k(k, ref, cond_of(entry).expand(len(ref), -1))
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


def build_bagging(ref, entry):
    q = ref + delta_ens(ref, cond_of(entry).expand(len(ref), -1))
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def build_lineare(W):
    def f(ref, entry):
        c = cond_of(entry).expand(len(ref), -1)
        q = ref + torch.cat([ref, c, torch.ones(len(ref), 1)], 1) @ W
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


print("\nvaluto sul set di sviluppo...", flush=True)
ps = {"baseline": evaluate_per_source(build_baseline, devset, E_valid, K)}
for k, sd in enumerate(SEEDS):
    ps[f"singolo seed {sd}"] = evaluate_per_source(build_singolo(k), devset, E_valid, K)
ps["BAGGING (4 modelli)"] = evaluate_per_source(build_bagging, devset, E_valid, K)
ps["MATRICE collassata"] = evaluate_per_source(build_lineare(W_ens), devset, E_valid, K)

media_singoli = np.mean([macro(ps[f"singolo seed {sd}"][10]) for sd in SEEDS]) * 100

print(f"\n{'metodo':<24}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'vs baseline':>14}{'p':>9}")
for n, v in ps.items():
    riga = f"{n:<24}{macro(v[1])*100:>7.2f}%{macro(v[5])*100:>7.2f}%{macro(v[10])*100:>7.2f}%"
    if n != "baseline":
        s = bootstrap_paired(v[10], ps["baseline"][10], n_boot=20000, seed=0)
        riga += f"{s['delta']*100:>+13.2f}pp{s['p']:>9.4f}"
    print(riga)
print(f"{'media dei 4 singoli':<24}{'':>8}{'':>8}{media_singoli:>7.2f}%")

print("\n--- controllo 2: la matrice collassata vale quanto l'ensemble? ---")
for a, b in (("MATRICE collassata", "BAGGING (4 modelli)"),
             ("BAGGING (4 modelli)", "singolo seed 1"),
             ("MATRICE collassata", "singolo seed 1")):
    s = bootstrap_paired(ps[a][10], ps[b][10], n_boot=20000, seed=0)
    print(f"  {a:<22} vs {b:<20}{s['delta']*100:>+7.2f}pp  "
          f"IC [{s['lo']*100:+.2f},{s['hi']*100:+.2f}]  p={s['p']:.4f}")

out = {"seeds": SEEDS, "ridge": RIDGE, "n_fit": len(X),
       "r2_ensemble": r2_of(W_ens, X[te], Y_ens[te]),
       "max_diff_fit_media": float((W_ens - W_media).abs().max()),
       "media_singoli_r10": float(media_singoli),
       "r10": {n: macro(v[10]) * 100 for n, v in ps.items()},
       "vs_baseline": {n: bootstrap_paired(v[10], ps["baseline"][10], n_boot=20000,
                                           seed=0)["delta"] * 100
                       for n, v in ps.items() if n != "baseline"}}
torch.save({"W": W_ens, "seeds": SEEDS}, HERE / "checkpoints" / "bagging_lineare.pt")
with open(HERE / "multiseed" / "bagging_lineare.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nmatrice salvata in checkpoints/bagging_lineare.pt, numeri in "
      f"multiseed/bagging_lineare.json")
