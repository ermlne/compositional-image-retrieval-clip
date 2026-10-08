#!/usr/bin/env python3
"""Il singolo modello e' non lineare, o lo e' solo l'ensemble?

`esperimenti/linearita_per_strato.py` misura quanto perde la matrice collassata rispetto
all'ensemble, e trova che la perdita si concentra sui vincoli NEGATIVI. Ma quel divario
confonde due cose:

  a) il singolo modello fa qualcosa che una mappa lineare non sa fare;
  b) sei o dodici modelli diversi non si riassumono in una mappa sola, anche se ognuno
     di loro fosse perfettamente lineare.

Qui si separano: si collassa in una matrice OGNI MEMBRO preso da solo, con lo stesso
protocollo di fase 4 di `scripts/train_colab_v3.py` (ridge in float64 su coppie (r, c)
costruite da un corpus del train disgiunto da quello di addestramento, 80/20 per l'R^2).

Se il divario sui negativi resta anche a livello di singolo membro, e' (a).
Se sparisce, era (b).

Uso:
    python esperimenti/collasso_per_membro.py CACHE CKPT_DIR [--tag v3]

    CACHE     il `cache/dati_pool.pt` della sessione (contiene E_fit, L_fit, d_ens)
    CKPT_DIR  la cartella con i checkpoint dei membri (`*_off*.pt`)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.benchmark import load_benchmark, parse_query_string   # noqa: E402
from src.devset import evaluate_per_source, sample_queries     # noqa: E402

# --- costanti della fase 4 di train_colab_v3.py, copiate perche' il fit deve essere lo stesso
RIDGE = 1e-3
SEED_FIT = 999
PER_QUERY_FIT = 60
POOL_TARGETS = {1: 80, 2: 140, 3: 100}


class Concat(nn.Module):
    """Identica a `Concat` di train_colab_v*.py e di contributo-candidato/multiseed.py."""

    def __init__(self, d=512, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.gamma = nn.Parameter(torch.tensor(1.0))


def ridge(Xs, Ys, lam=RIDGE):
    """Ridge in float64. Non `lstsq`: il sistema e' mal posto (rango 589 su 1025)."""
    A = torch.cat([Xs, torch.ones(len(Xs), 1)], 1).double()
    n = A.shape[1]
    return torch.linalg.solve(A.T @ A + lam * torch.eye(n, dtype=torch.float64),
                              A.T @ Ys.double()).float()


def r2_of(W, Xs, Ys):
    pred = torch.cat([Xs, torch.ones(len(Xs), 1)], 1) @ W
    return float(1 - ((Ys - pred) ** 2).sum() / ((Ys - Ys.mean(0)) ** 2).sum())


def builder_mlp(model, cs):
    def f(ref, entry):
        c = cs[entry["query"]].expand(len(ref), -1)
        with torch.no_grad():
            q = ref + model.gamma * model.net(torch.cat([ref, c], dim=-1))
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


def builder_lineare(W, cs):
    """q = normalize(r + [r; c; 1] @ W). gamma e' GIA' dentro W: il bersaglio del fit
    e' `gamma * net(...)`, rimoltiplicarlo sarebbe un errore silenzioso."""
    def f(ref, entry):
        c = cs[entry["query"]].expand(len(ref), -1)
        q = ref + torch.cat([ref, c, torch.ones(len(ref), 1)], 1) @ W
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


def macro(per_source, indici):
    return 100 * np.mean([per_source[i].mean() for i in indici])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("ckpt_dir")
    ap.add_argument("--tag", default="?")
    args = ap.parse_args()

    torch.set_num_threads(min(8, __import__("os").cpu_count() or 4))

    dati = torch.load(args.cache, map_location="cpu", weights_only=False)
    E_fit, L_fit, d_ens, ATTR = dati["E_fit"], dati["L_fit"], dati["d_ens"], dati["attr"]
    D = E_fit.shape[1]

    ev = torch.load(ROOT / "embeddings/celeba_valid_clip_vitb32.pt", map_location="cpu")
    E_valid = ev["embeddings"].float()
    E_valid = E_valid / E_valid.norm(dim=-1, keepdim=True)
    dev = load_benchmark(str(ROOT / "celeba_devset_valid.json"), ATTR)
    attr_to_idx = {n: i for i, n in enumerate(ATTR)}

    # `c` per ogni query del devset: somma con segno delle direzioni bipolari
    cs = {}
    for e in dev:
        c = torch.zeros(D)
        for a in e["pos"]:
            c += d_ens[a]
        for b in e["neg"]:
            c -= d_ens[b]
        cs[e["query"]] = c.unsqueeze(0)

    n_cond = np.array([len(e["queried"]) for e in dev])
    n_neg = np.array([len(e["neg"]) for e in dev])
    POS1 = np.where((n_cond == 1) & (n_neg == 0))[0]
    NEG1 = np.where((n_cond == 1) & (n_neg == 1))[0]
    TUTTE = np.arange(len(dev))

    # --- le coppie (r, c) del fit, come la fase 4 ------------------------------------------
    qs_fit = []
    for nc, quanti in POOL_TARGETS.items():
        qs_fit += sample_queries(list(ATTR), quanti, nc, exclude=qs_fit, seed=SEED_FIT + nc)
    X = []
    for qt in qs_fit:
        pos, neg = parse_query_string(qt, attr_to_idx)
        viol = torch.ones(len(L_fit), dtype=torch.bool)
        for a in pos:
            viol &= L_fit[:, a] == 0
        for b in neg:
            viol &= L_fit[:, b] == 1
        src = viol.nonzero(as_tuple=True)[0]
        if len(src) < 10:
            continue
        gs = torch.Generator().manual_seed(SEED_FIT)
        pick = src[torch.randperm(len(src), generator=gs)[:PER_QUERY_FIT]]
        R = E_fit[pick]
        c = (sum((d_ens[a] for a in pos), torch.zeros(D))
             - sum((d_ens[b] for b in neg), torch.zeros(D))).expand(len(R), -1)
        X.append(torch.cat([R, c], 1))
    X = torch.cat(X)

    ckpt = sorted(p for p in Path(args.ckpt_dir).glob("*.pt") if re.search(r"_off(_cx)?\.pt$", p.name))
    print(f"sessione {args.tag}: {len(ckpt)} membri, {len(X)} coppie (r, c) per il fit")
    RIS = {"prodotto_da": "esperimenti/collasso_per_membro.py", "tag": args.tag,
           "cache": args.cache, "ckpt_dir": args.ckpt_dir, "ridge": RIDGE,
           "n_fit": len(X), "membri": {}, "ensemble": {}}

    modelli, Yk = [], []
    for p in ckpt:
        d = torch.load(p, map_location="cpu", weights_only=False)
        cfg = d.get("config", {})
        m = Concat(cfg.get("d", D), cfg.get("hidden", 512))
        m.load_state_dict(d["A"])
        m.eval()
        modelli.append((p.name, m))
        with torch.no_grad():
            Yk.append(m.gamma * m.net(X))

    gl = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(X), generator=gl)
    tr, te = idx[:int(.8 * len(X))], idx[int(.8 * len(X)):]

    def valuta(build):
        return evaluate_per_source(build, dev, E_valid, [10])[10]

    print(f"\n{'membro':>34}{'R2':>8}{'MLP':>9}{'lineare':>9}{'divario':>9}"
          f"{'+attr':>9}{'-attr':>9}{'diff':>8}")
    righe = []
    for (nome, m), Y in zip(modelli, Yk):
        W = ridge(X[tr], Y[tr])
        ps_mlp = valuta(builder_mlp(m, cs))
        ps_lin = valuta(builder_lineare(W, cs))
        g = lambda I: macro(ps_mlp, I) - macro(ps_lin, I)   # noqa: E731
        righe.append((nome, r2_of(W, X[te], Y[te]), macro(ps_mlp, TUTTE),
                      macro(ps_lin, TUTTE), g(TUTTE), g(POS1), g(NEG1)))
        n, r2, a, b, gt_, gp, gn = righe[-1]
        RIS["membri"][n] = {"r2": r2, "mlp_r10": a, "lineare_r10": b, "divario_pp": gt_,
                            "divario_positive_pp": gp, "divario_negative_pp": gn,
                            "asimmetria_pp": gn - gp}
        print(f"{n[:34]:>34}{r2:>8.4f}{a:>8.2f}%{b:>8.2f}%{gt_:>+8.2f}pp"
              f"{gp:>+8.2f}pp{gn:>+8.2f}pp{gn - gp:>+7.2f}pp")

    A = np.array([[r[1], r[4], r[5], r[6]] for r in righe])
    print(f"\n{'MEDIA sui membri':>34}{A[:, 0].mean():>8.4f}{'':>9}{'':>9}"
          f"{A[:, 1].mean():>+8.2f}pp{A[:, 2].mean():>+8.2f}pp{A[:, 3].mean():>+8.2f}pp"
          f"{(A[:, 3] - A[:, 2]).mean():>+7.2f}pp")
    d = A[:, 3] - A[:, 2]
    print(f"  membri in cui il divario e' piu' grande sui NEGATIVI: {(d > 0).sum()}/{len(d)}"
          f"   (media {d.mean():+.2f}pp, sd {d.std(ddof=1):.2f})")

    # --- confronto: l'ensemble collassato ---------------------------------------------------
    Y_ens = torch.stack(Yk).mean(0)
    W_ens = ridge(X[tr], Y_ens[tr])

    def build_ens(ref, entry):
        c = cs[entry["query"]].expand(len(ref), -1)
        with torch.no_grad():
            dl = torch.stack([m.gamma * m.net(torch.cat([ref, c], 1)) for _, m in modelli]).mean(0)
        q = ref + dl
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    ps_e, ps_l = valuta(build_ens), valuta(builder_lineare(W_ens, cs))
    g = lambda I: macro(ps_e, I) - macro(ps_l, I)   # noqa: E731
    RIS["ensemble"] = {"r2": r2_of(W_ens, X[te], Y_ens[te]), "mlp_r10": macro(ps_e, TUTTE),
                       "lineare_r10": macro(ps_l, TUTTE), "divario_pp": g(TUTTE),
                       "divario_positive_pp": g(POS1), "divario_negative_pp": g(NEG1),
                       "asimmetria_pp": g(NEG1) - g(POS1)}
    RIS["riassunto"] = {
        "n_membri": len(modelli),
        "membri_con_divario_maggiore_sui_negativi": int((A[:, 3] - A[:, 2] > 0).sum()),
        "asimmetria_media_pp": float((A[:, 3] - A[:, 2]).mean()),
        "asimmetria_sd_pp": float((A[:, 3] - A[:, 2]).std(ddof=1))}
    print(f"\n{'ENSEMBLE':>34}{r2_of(W_ens, X[te], Y_ens[te]):>8.4f}"
          f"{macro(ps_e, TUTTE):>8.2f}%{macro(ps_l, TUTTE):>8.2f}%{g(TUTTE):>+8.2f}pp"
          f"{g(POS1):>+8.2f}pp{g(NEG1):>+8.2f}pp{g(NEG1) - g(POS1):>+7.2f}pp")

    out = HERE / f"collasso_per_membro_{args.tag.split()[0]}.json"
    json.dump(RIS, open(out, "w"), indent=1)
    print(f"scritto {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
