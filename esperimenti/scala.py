"""Il disallineamento fra frecce testuali e frecce visive e' solo il divario fra le due nuvole?

Tre misure in sequenza, tutte sulle direzioni (nessun modulo appreso):
 1. si toglie il baricentro alla nuvola testuale e a quella visiva, si rinormalizza, e si
    guarda se l'allineamento coseno testo<->visivo cambia (sui 30 visti e sui 10 tenuti fuori);
 2. correlazioni canoniche (CCA regolarizzata) fra i due insiemi di frecce, fittate sui soli
    30 attributi visti, al variare del numero di componenti k: allineamento alto sui visti e
    basso sui mai visti significa che la mappa memorizza i 30 invece di riscalare;
 3. il retrieval: si sostituisce la direzione testuale con quella mappata e si legge la
    Recall@10 sui tre strati, a passo normalizzato.

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

Uso:    HF_HUB_OFFLINE=1 python esperimenti/scala.py --seed 2026 --split 0
Output: esperimenti/scala_split{split}.json
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
T_POS = torch.zeros(N_ATTR, D); T_NEG = torch.zeros(N_ATTR, D)
for i, name in enumerate(attr_names):
    c = name.replace("_", " ").lower()
    T_POS[i] = encode([t.format(c) for t, _ in TEMPLATES]).mean(0)
    T_NEG[i] = encode([t.format(c) for _, t in TEMPLATES]).mean(0)
    v = T_POS[i] - T_NEG[i]
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
SEEN = sorted(i for i in range(N_ATTR) if i not in HOLD)
HELD = sorted(HOLD)

def nrm(X): return X / X.norm(dim=-1, keepdim=True).clamp(min=1e-8)

# --- direzioni visive: dalle etichette, sul train ------------------------------------------
V_POS = torch.zeros(N_ATTR, D); V_NEG = torch.zeros(N_ATTR, D); ok = []
for a in range(N_ATTR):
    p = L_tr[:, a] == 1
    if p.sum() < 20 or (~p).sum() < 20: continue
    V_POS[a] = E_tr[p].mean(0); V_NEG[a] = E_tr[~p].mean(0); ok.append(a)
S = [a for a in SEEN if a in ok]; H = [a for a in HELD if a in ok]

def dirs(centrato):
    """Frecce testuali e visive, con o senza rimozione del divario fra le due nuvole.
    Gli embedding sono normalizzati sulla sfera: sottrarre la media e RInormalizzare non e'
    un'operazione neutra sulle differenze (su vettori non normalizzati si cancellerebbe)."""
    tp, tn, vp, vn = T_POS.clone(), T_NEG.clone(), V_POS.clone(), V_NEG.clone()
    if centrato:
        mt = torch.cat([T_POS, T_NEG]).mean(0)      # baricentro della nuvola TESTUALE
        mv = E_tr.mean(0)                           # baricentro della nuvola VISIVA
        tp, tn = nrm(tp - mt), nrm(tn - mt)
        vp, vn = nrm(vp - mv), nrm(vn - mv)
    return nrm(tp - tn), nrm(vp - vn)

def allin(idxs, T, V):
    return float(torch.tensor([float(T[a] @ V[a]) for a in idxs]).mean())

print("\n" + "=" * 92)
print("1. TOGLIERE IL DIVARIO FRA LE DUE NUVOLE CAMBIA L'ALLINEAMENTO DELLE FRECCE?")
print("=" * 92)
for centrato in (False, True):
    T, V = dirs(centrato)
    et = f"{'DOPO' if centrato else 'PRIMA'} la rimozione del divario"
    print(f"  {et:<36} visti {allin(S,T,V):+.4f}   mai visti {allin(H,T,V):+.4f}")

# ==========================================================================================
# 2. LE DUE MAPPE SONO IN SCALA FRA LORO? (correlazioni canoniche)
#
# Procrustes cercava una ROTAZIONE: conserva le lunghezze per costruzione, quindi non poteva
# trovare un allungamento. Qui si cerca la relazione lineare completa fra le due mappe e si
# guardano i FATTORI DI ALLUNGAMENTO asse per asse (le correlazioni canoniche).
#
# Tutto stimato sui SOLI 30 attributi visti. Con 30 campioni in 512 dimensioni il problema e'
# degenere, quindi si riduce prima a k dimensioni e si fa variare k.
# ==========================================================================================
print("\n" + "=" * 92)
print("2. FATTORI DI ALLUNGAMENTO FRA LE DUE MAPPE (correlazioni canoniche, fit sui 30 visti)")
print("=" * 92)

def cca(T, V, k, lam=1e-3):
    X = T[S] - T[S].mean(0, keepdim=True)
    Y = V[S] - V[S].mean(0, keepdim=True)
    _, _, Pt = torch.linalg.svd(X, full_matrices=False); Pt = Pt[:k]
    _, _, Pv = torch.linalg.svd(Y, full_matrices=False); Pv = Pv[:k]
    Xr, Yr = X @ Pt.T, Y @ Pv.T
    Cxx = Xr.T @ Xr / len(S) + lam * torch.eye(k)
    Cyy = Yr.T @ Yr / len(S) + lam * torch.eye(k)
    Cxy = Xr.T @ Yr / len(S)
    ix = torch.linalg.inv(torch.linalg.cholesky(Cxx))
    iy = torch.linalg.inv(torch.linalg.cholesky(Cyy))
    U_, Sv, Vt_ = torch.linalg.svd(ix @ Cxy @ iy.T)
    # mappa testo -> visivo, riportata nello spazio pieno
    W = Pt.T @ (ix.T @ U_) @ torch.diag(Sv) @ (Vt_ @ iy) @ Pv
    return Sv, W

for centrato in (False, True):
    T, V = dirs(centrato)
    print(f"\n  --- {'con' if centrato else 'senza'} rimozione del divario")
    print(f"  {'k':<5}{'correlazioni canoniche (prime 6)':<44}{'allin VISTI':>13}{'allin MAI VISTI':>17}")
    for k in (3, 5, 8, 12, 20):
        Sv, W = cca(T, V, k)
        TW = nrm(T @ W)
        prime = " ".join(f"{v:.2f}" for v in Sv[:6].tolist())
        print(f"  {k:<5}{prime:<44}{allin(S,TW,V):>13.4f}{allin(H,TW,V):>17.4f}")

print("\n  (allin VISTI alto + allin MAI VISTI basso = la mappa memorizza i 30, non scala.)")
print("  (con 30 campioni e k=20 la memorizzazione e' attesa: serve guardare k piccoli.)")

# ==========================================================================================
# 3. IL GIUDIZIO CHE CONTA: il retrieval. Passo normalizzato.
# ==========================================================================================
print("\n" + "=" * 92)
print("3. E NEL RETRIEVAL?  (Recall@10, passo normalizzato, differenza vs baseline)")
print("=" * 92)

def make_q(T, alpha):
    cache = {}
    def f(ref, entry):
        kk = entry["query"]
        if kk not in cache:
            c = torch.zeros(D)
            for a in entry["pos"]: c = c + T[a]
            for b in entry["neg"]: c = c - T[b]
            cache[kk] = c / c.norm().clamp(min=1e-8)
        q = ref + alpha * cache[kk]
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f

ALPHAS = (1.0, 1.5, 2.0)
VAR = [("baseline", None)]
for centrato in (False, True):
    T0, V0 = dirs(centrato)
    et = "centr." if centrato else "grezzo"
    VAR.append((f"frecce {et}", T0))
    for k in (3, 5, 8):
        _, W = cca(T0, V0, k)
        VAR.append((f"scala k={k} {et}", nrm(T0 @ W)))

R = {}
for nome, T in VAR:
    TT = d_ens if T is None else T
    for a in ALPHAS: R[(nome, a)] = evaluate(make_q(TT, a), devset, E_valid, K_VALUES)[10]

for strato, idx in (("VISTI", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix)):
    if not idx: continue
    print(f"\n--- {strato} ({len(idx)} query)")
    print(f"  {'variante':<22}" + "".join(f"{'a=' + str(a):>21}" for a in ALPHAS))
    for nome, _ in VAR:
        parti = []
        for a in ALPHAS:
            v = macro([R[(nome, a)][i] for i in idx]) * 100
            if nome == "baseline":
                parti.append(f"{v:>20.2f}%")
            else:
                st = bootstrap_paired([R[(nome,a)][i] for i in idx],
                                      [R[("baseline",a)][i] for i in idx], n_boot=10000, seed=0)
                parti.append(f"{v:>8.2f}% {st['delta']*100:>+6.2f}({st['p']:.3f})")
        print(f"  {nome:<22}" + "".join(parti))

with open(ROOT / "esperimenti" / f"scala_split{args.split}.json", "w") as f:
    json.dump({"seed": SEED, "split": args.split,
               "holdout": sorted(attr_names[i] for i in HOLD),
               "recall10": {f"{n}|a{a}": {s: macro([R[(n,a)][i] for i in idx])*100
                            for s, idx in (("visti", idx_seen), ("mai_visti", idx_hold),
                                           ("miste", idx_mix)) if idx}
                            for n, _ in VAR for a in ALPHAS}}, f, indent=2)
print("\nsalvato.")
