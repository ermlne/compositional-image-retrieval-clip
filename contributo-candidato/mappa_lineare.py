"""LA MAPPA LINEARE REGGE? — tre protocolli di fit a confronto, sullo stesso modello.

===========================================================================================
LA DOMANDA
===========================================================================================
`interpretabilita.py` (esp. 4) conclude che l'MLP e' "una mappa lineare travestita": R^2 =
0.94, e sostituendo la rete con due matrici la Recall@10 resta 27.00% contro 27.26%
(p=0.09, non significativo). Da li' la frase del commit 45f6a20: LA RETE NON SERVE.

Ma quella mappa e' stata **fittata sulle reference del set di sviluppo, e poi valutata
sulle stesse**. E' un fit in-sample: si chiede alla mappa di imitare l'MLP esattamente
dove la si misura. Rifittandola sul train (multiseed.py) il risultato cambia di 3pp.

Questo script isola la causa cambiando UNA COSA ALLA VOLTA: stesso modello, stesse
formule, stessa valutazione — solo i dati su cui si stima W cambiano.

  P1  fit su reference del TRAIN                 -> valuta su tutto il dev
  P2  fit sul DEV intero (protocollo originale)  -> valuta su tutto il dev   [in-sample]
  P3  fit su meta' delle QUERY del dev           -> valuta sull'altra meta'  [decisivo]

P3 e' il test che conta: stessa distribuzione di P2, ma la mappa non ha mai visto le query
su cui viene misurata. Se P2 >> P3, il 27.00% era il fit che rincorreva se stesso.

Riporta anche l'R^2 di ciascun protocollo, per mostrare che **R^2 alto non basta**: la
Recall dipende dall'ordinamento, e un residuo del 4% puo' spostarlo.

Uso:  HF_HUB_OFFLINE=1 python contributo-candidato/mappa_lineare.py --seed 2026
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark
from src.devset import bootstrap_paired, macro, sample_queries, evaluate_per_source

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
SEED = args.seed
K = [1, 5, 10]
N_TRAIN_CORPUS, MAX_SOURCES = 20000, 80
QUERY_TARGETS = {1: 80, 2: 80, 3: 60}

ck = torch.load(HERE / f"checkpoints/models_seed{SEED}.pt", map_location="cpu",
                weights_only=False)
D, HIDDEN = ck["config"]["d"], ck["config"]["hidden"]
d_ens, attr_names = ck["dirs_text"], ck["attr_names"]


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def delta(self, r, c):
        return self.net(torch.cat([r, c], dim=-1))

    def forward(self, r, c):
        q = r + self.gamma * self.delta(r, c)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


MOD = Concat(); MOD.load_state_dict(ck["A"]); MOD.eval()
GAMMA = MOD.gamma.item()
print(f"modello seed {SEED} caricato, gamma = {GAMMA:.4f}")


def load_split(split):
    d = torch.load(ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"]


E_valid, L_valid = load_split("valid")
E_train_all, L_train_all = load_split("train")
g = torch.Generator().manual_seed(SEED)
sub = torch.randperm(len(E_train_all), generator=g)[:N_TRAIN_CORPUS].sort().values
E_tr, L_tr = E_train_all[sub], L_train_all[sub]
del E_train_all, L_train_all
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)
attr_to_idx = {n: i for i, n in enumerate(attr_names)}


def cond_of(entry):
    c = torch.zeros(D)
    for a in entry["pos"]: c = c + d_ens[a]
    for b in entry["neg"]: c = c - d_ens[b]
    return c


def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)


def pairs_from(entries, E, per_query):
    """(r, c) e delta dell'MLP per un insieme di entry."""
    X, Y = [], []
    for e in entries:
        c = cond_of(e)
        R = E[e["sources"][:per_query]]
        with torch.no_grad():
            dl = MOD.delta(R, c.expand(len(R), -1))
        X.append(torch.cat([R, c.expand(len(R), -1)], 1)); Y.append(dl)
    return torch.cat(X), torch.cat(Y)


RIDGE = 1e-3

def fit(X, Y, lam=RIDGE):
    """Ridge, non lstsq: la regressione e' MAL POSTA e lstsq non e' riproducibile.

    Misurato (diagnostica in fondo a questo file): la matrice di fit ha rango numerico
    589 su 1025 colonne e numero di condizione 4.7e16, perche' i vettori `c` distinti
    sono ~91 e ne spannano solo 76 dimensioni su 512 — le direzioni di `c` mai
    esercitate non sono identificabili. Su un sistema cosi', `torch.linalg.lstsq`
    restituisce soluzioni DIVERSE a ogni chiamata sugli stessi input (misurato:
    max|W1-W2| = 0.097, R^2 fra 0.935 e 0.995, Recall@10 fra 24.3% e 27.0%).

    Con qualunque lambda fra 1e-4 e 1e-1 la soluzione e' unica, riproducibile
    (max|diff| = 0) e da' R^2 0.9953. Il numero riportato in 45f6a20 (27.00%,
    -0.26pp, p=0.093) e' quello della soluzione regolarizzata: la conclusione regge,
    ma andava ottenuta cosi' invece che per sorteggio.
    """
    A = torch.cat([X, torch.ones(len(X), 1)], 1).double()
    n = A.shape[1]
    return torch.linalg.solve(A.T @ A + lam * torch.eye(n, dtype=torch.float64),
                              A.T @ Y.double()).float()


def r2_of(W, X, Y):
    pred = torch.cat([X, torch.ones(len(X), 1)], 1) @ W
    r2 = (1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()).item()
    cos = torch.nn.functional.cosine_similarity(pred, Y, dim=-1).mean().item()
    return r2, cos


def builder_linear(W):
    def f(ref, entry):
        c = cond_of(entry).expand(len(ref), -1)
        q = ref + GAMMA * (torch.cat([ref, c, torch.ones(len(ref), 1)], 1) @ W)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


def builder_mlp(ref, entry):
    with torch.no_grad():
        return MOD(ref, cond_of(entry).expand(len(ref), -1))


# ---- P1: reference del TRAIN che violano le condizioni (stessa regola del benchmark) ----
print("\ncostruisco le coppie (r, c) del train...", flush=True)
qs = []
for n_cond, quanti in QUERY_TARGETS.items():
    qs += sample_queries(list(attr_names), quanti, n_cond, exclude=qs, seed=SEED + n_cond)
train_entries = []
for qt in qs:
    pos, neg = [], []
    for tok in qt.split(","):
        tok = tok.strip()
        (pos if tok[0] == "+" else neg).append(attr_to_idx[tok[1:].strip()])
    viol = torch.ones(len(L_tr), dtype=torch.bool)
    for a in pos: viol &= L_tr[:, a] == 0
    for b in neg: viol &= L_tr[:, b] == 1
    src = viol.nonzero(as_tuple=True)[0]
    # a CASO, non i primi per indice: CelebA e' ordinato per identita', quindi le prime
    # reference sarebbero quasi duplicate e il blocco `r` della matrice di fit
    # collasserebbe (misurato: R^2 0.60 invece di 0.96). Stessa scelta di
    # build_benchmark_from_labels, che campiona le reference con randperm.
    if len(src) >= 10:
        gsrc = torch.Generator().manual_seed(SEED)
        pick = src[torch.randperm(len(src), generator=gsrc)[:MAX_SOURCES]]
        train_entries.append({"pos": pos, "neg": neg, "sources": pick.sort().values})
X1, Y1 = pairs_from(train_entries, E_tr, 60)
gl = torch.Generator().manual_seed(0)
i1 = torch.randperm(len(X1), generator=gl)
tr1, te1 = i1[:int(.8 * len(X1))], i1[int(.8 * len(X1)):]
W1 = fit(X1[tr1], Y1[tr1])
r2_1, cos_1 = r2_of(W1, X1[te1], Y1[te1])

# ---- P2 e P3: sul dev ------------------------------------------------------------------
X2, Y2 = pairs_from(devset, E_valid, 120)
i2 = torch.randperm(len(X2), generator=torch.Generator().manual_seed(0))
tr2, te2 = i2[:int(.8 * len(X2))], i2[int(.8 * len(X2)):]
W2 = fit(X2[tr2], Y2[tr2])
r2_2, cos_2 = r2_of(W2, X2[te2], Y2[te2])

gq = torch.Generator().manual_seed(0)
permq = torch.randperm(len(devset), generator=gq).tolist()
half_fit = [devset[i] for i in permq[:len(devset) // 2]]
idx_eval = sorted(permq[len(devset) // 2:])
X3, Y3 = pairs_from(half_fit, E_valid, 120)
W3 = fit(X3, Y3)
Xe, Ye = pairs_from([devset[i] for i in idx_eval], E_valid, 120)
r2_3, cos_3 = r2_of(W3, Xe, Ye)

print(f"\n{'protocollo':<44}{'R^2':>8}{'cos':>8}")
print(f"{'P1 fit su reference del TRAIN':<44}{r2_1:>8.4f}{cos_1:>8.4f}")
print(f"{'P2 fit sul DEV (protocollo originale)':<44}{r2_2:>8.4f}{cos_2:>8.4f}")
print(f"{'P3 fit su meta query del DEV':<44}{r2_3:>8.4f}{cos_3:>8.4f}   (R^2 sull altra meta)")

# ---- valutazione ------------------------------------------------------------------------
print("\nvaluto sul set di sviluppo...", flush=True)
ps = {"baseline": evaluate_per_source(build_baseline, devset, E_valid, K),
      "MLP": evaluate_per_source(builder_mlp, devset, E_valid, K),
      "P1 lineare (fit train)": evaluate_per_source(builder_linear(W1), devset, E_valid, K),
      "P2 lineare (fit dev)": evaluate_per_source(builder_linear(W2), devset, E_valid, K),
      "P3 lineare (fit meta dev)": evaluate_per_source(builder_linear(W3), devset, E_valid, K)}

print(f"\n{'metodo':<30}{'R@1':>8}{'R@5':>8}{'R@10':>8}   differenza vs MLP (R@10)")
for n, v in ps.items():
    riga = f"{n:<30}{macro(v[1])*100:>7.2f}%{macro(v[5])*100:>7.2f}%{macro(v[10])*100:>7.2f}%"
    if n != "MLP":
        s = bootstrap_paired(v[10], ps["MLP"][10], n_boot=20000, seed=0)
        riga += f"   {s['delta']*100:>+6.2f}pp  IC [{s['lo']*100:+.2f},{s['hi']*100:+.2f}]  p={s['p']:.4f}"
    print(riga)

print("\nsulle sole query di valutazione di P3 (mai usate per fittare W3):")
sub_ps = {n: [v[10][i] for i in idx_eval] for n, v in ps.items()}
out = {"seed": SEED, "ridge": RIDGE,
       "r2": {"P1_train": r2_1, "P2_dev": r2_2, "P3_meta_dev": r2_3},
       "dev_r10": {n: macro(v[10]) * 100 for n, v in ps.items()},
       "vs_MLP": {}, "meta_dev": {}}
for n, v in ps.items():
    if n != "MLP":
        s = bootstrap_paired(v[10], ps["MLP"][10], n_boot=20000, seed=0)
        out["vs_MLP"][n] = {"delta_pp": s["delta"] * 100, "p": s["p"]}
for n in ("P1 lineare (fit train)", "P2 lineare (fit dev)", "P3 lineare (fit meta dev)"):
    s = bootstrap_paired(sub_ps[n], sub_ps["MLP"], n_boot=20000, seed=0)
    b = bootstrap_paired(sub_ps[n], sub_ps["baseline"], n_boot=20000, seed=0)
    out["meta_dev"][n] = {"vs_MLP_pp": s["delta"] * 100, "vs_MLP_p": s["p"],
                          "vs_baseline_pp": b["delta"] * 100, "vs_baseline_p": b["p"]}
    print(f"  {n:<30} vs MLP {s['delta']*100:>+6.2f}pp p={s['p']:.4f} | "
          f"vs baseline {b['delta']*100:>+6.2f}pp p={b['p']:.4f}")
mb = bootstrap_paired(sub_ps["MLP"], sub_ps["baseline"], n_boot=20000, seed=0)
out["meta_dev"]["MLP"] = {"vs_baseline_pp": mb["delta"] * 100, "vs_baseline_p": mb["p"]}
print(f"  {'MLP':<30} vs baseline {mb['delta']*100:+.2f}pp")

print("\nsensibilita' al lambda della ridge (P1, fit su train):")
for lam in (1e-4, 1e-2, 1e-1):
    Wl = fit(X1[tr1], Y1[tr1], lam)
    v = evaluate_per_source(builder_linear(Wl), devset, E_valid, [10])
    s = bootstrap_paired(v[10], ps["MLP"][10], n_boot=20000, seed=0)
    print(f"  lambda {lam:<7} R@10 {macro(v[10])*100:6.2f}%   vs MLP {s['delta']*100:+6.2f}pp")
    out.setdefault("lambda_sens", {})[str(lam)] = macro(v[10]) * 100

import json
with open(HERE / "multiseed" / f"lineare_seed{SEED}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nsalvato in multiseed/lineare_seed{SEED}.json")
