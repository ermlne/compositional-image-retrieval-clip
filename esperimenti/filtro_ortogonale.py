"""Somiglianza alla reference AL NETTO degli attributi interrogati.

LA REGOLA (sez. 3.1.1): un risultato e' corretto se soddisfa le condizioni **e** ha distanza
di Hamming <= 2 sugli attributi NON INTERROGATI. La regola chiede esplicitamente di poter
differire su quelli interrogati.

L'ERRORE del filtro precedente: usava `r . e`, la somiglianza COMPLESSIVA, che include gli
attributi interrogati. Penalizzava esattamente la differenza che la query pretende — le
immagini corrette DEVONO differire dalla reference sull'attributo chiesto, perche' la
reference viola ogni condizione.

QUI: si proietta via il sottospazio generato dalle direzioni interrogate, da entrambi i lati,
e si misura la somiglianza in cio' che resta. E' il proxy corretto per "Hamming sui non
interrogati".

    Q       base ortonormale delle direzioni interrogate (QR, al massimo 3)
    x_perp  = x - Q Q^T x,  poi rinormalizzato
    s_perp  = r_perp . e_perp

Tre schemi:
  A  additivo ortogonale   s_perp + a * (c . e)         <- il punteggio, con identita' corretta
  B  filtro ortogonale     top-K per s_perp, poi c . e
  C  filtro ortogonale     top-K per s_perp, poi punteggio additivo standard
"""
import sys, json
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark
from src.devset import bootstrap_paired, macro

ck = torch.load(ROOT / "contributo-candidato/checkpoints/models_seed2026.pt", map_location="cpu")
d_ens = ck["dirs_text"]; attr_names = list(ck["attr_names"]); D = ck["config"]["d"]
dtn = d_ens / d_ens.norm(dim=-1, keepdim=True)

d = torch.load(ROOT / "embeddings/celeba_valid_clip_vitb32.pt", map_location="cpu")
E = d["embeddings"].float(); E = E / E.norm(dim=-1, keepdim=True)
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)
print(f"corpus {len(E)} immagini, {len(devset)} query")


def pezzi(e):
    """direzione della condizione, e base ortonormale degli attributi interrogati"""
    c = torch.zeros(D)
    for a in e["pos"]: c = c + dtn[a]
    for b in e["neg"]: c = c - dtn[b]
    c = c / c.norm().clamp(min=1e-8)
    Dq = torch.stack([dtn[a] for a in e["queried"]], dim=1)      # (D, n_interrogati)
    Q, _ = torch.linalg.qr(Dq)                                    # base ortonormale
    return c, Q


def perp(X, Q):
    Y = X - (X @ Q) @ Q.T
    return Y / Y.norm(dim=-1, keepdim=True).clamp(min=1e-8)


KMAX = 10
ALPHAS = (0.5, 1.0, 1.5, 2.0)
KS = (100, 250, 500, 1000, 2500, 5000)


def valuta(schema, alpha=1.0, K=None):
    per = []
    for e in devset:
        src = e["sources"]; c, Q = pezzi(e)
        Ep = perp(E, Q)                                # corpus proiettato, una volta per query
        s_c_all = Ep @ c if False else E @ c           # la condizione si valuta nello spazio pieno
        rec = np.zeros(len(src))
        for st in range(0, len(src), 512):
            sp = min(st + 512, len(src)); b = src[st:sp]
            r = E[b]
            s_perp = perp(r, Q) @ Ep.T                 # somiglianza AL NETTO degli interrogati
            s_perp[torch.arange(sp - st), b] = -torch.inf
            s_c = s_c_all.expand(sp - st, -1).clone()
            s_c[torch.arange(sp - st), b] = -torch.inf
            q = r + alpha * c
            q = q / q.norm(dim=-1, keepdim=True)
            s_q = q @ E.T
            s_q[torch.arange(sp - st), b] = -torch.inf
            if schema == "additivo standard":
                punteggio = s_q
            elif schema == "additivo ortogonale":
                punteggio = s_perp + alpha * s_c
            else:
                tieni = s_perp.topk(K, dim=-1).indices
                base = s_c if schema == "filtro orto -> condizione" else s_q
                punteggio = torch.full_like(base, -torch.inf)
                punteggio.scatter_(1, tieni, base.gather(1, tieni))
            for j, ret in enumerate(punteggio.topk(KMAX, dim=-1).indices.tolist()):
                rec[st + j] = float(any(i in e["gt"][st + j] for i in ret))
        per.append(rec)
    return per


print("\n" + "=" * 94)
print("SOMIGLIANZA AL NETTO DEGLI ATTRIBUTI INTERROGATI — Recall@10")
print("=" * 94)
base = {a: valuta("additivo standard", a) for a in ALPHAS}
for a in ALPHAS:
    print(f"  additivo standard, a={a}: {macro(base[a])*100:.2f}%")

out = {}
print(f"\n--- A. additivo ORTOGONALE  (identita' al netto degli interrogati)")
for a in ALPHAS:
    p = valuta("additivo ortogonale", a)
    st = bootstrap_paired(p, base[a], n_boot=20000, seed=0)
    out[f"A|a{a}"] = macro(p)*100
    print(f"  a={a:<5}{macro(p)*100:>8.2f}%   {st['delta']*100:>+7.2f}pp  (p={st['p']:.4f})",
          flush=True)

for schema, eti in (("filtro orto -> condizione", "B. filtro orto, poi condizione"),
                    ("filtro orto -> additivo",   "C. filtro orto, poi additivo")):
    print(f"\n--- {eti}")
    print(f"  {'K':<8}" + "".join(f"{'a=' + str(a):>26}" for a in ALPHAS))
    for K in KS:
        parti = []
        for a in ALPHAS:
            p = valuta(schema, a, K)
            st = bootstrap_paired(p, base[a], n_boot=20000, seed=0)
            parti.append(f"{macro(p)*100:>11.2f}% {st['delta']*100:>+7.2f}({st['p']:.3f})")
            out[f"{eti}|K{K}|a{a}"] = macro(p)*100
        print(f"  {K:<8}" + "".join(parti), flush=True)

with open(ROOT / "esperimenti" / "filtro_ortogonale.json", "w") as f:
    json.dump({"baseline": {str(a): macro(base[a])*100 for a in ALPHAS}, "recall10": out}, f,
              indent=2)
print("\nsalvato.")
