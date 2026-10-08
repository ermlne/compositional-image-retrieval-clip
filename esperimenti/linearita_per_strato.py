#!/usr/bin/env python3
"""Dove la forma lineare del contributo perde, e su quali query.

Il collasso lineare riassume l'uscita di un ensemble in una sola mappa
`W_r r + W_c c + b`. La quota di guadagno che quella mappa NON tiene misura quanto il
modello sia diventato non lineare - ma solo in aggregato. Qui la si stratifica, usando
gli esiti per singola source gia' conservati dalle due sessioni Colab, senza bisogno
dei checkpoint (che non sono in git).

Tre domande, tre risposte:

  1. il divario ensemble-contro-matrice cresce col numero di condizioni?
  2. cresce col numero di condizioni NEGATIVE?
  3. la domanda 2 regge a parita' di numero di condizioni?

La terza e' quella che conta: nel devset le query a una condizione sono 27 positive e
27 negative, quindi il confronto e' bilanciato e non confuso dal numero di condizioni.

Uso:  python esperimenti/linearita_per_strato.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.benchmark import load_benchmark          # noqa: E402
from src.devset import bootstrap_paired           # noqa: E402

SESSIONI = (("v1", ROOT / "scripts/run_colab/esiti_devset.npz"),
            ("v2", ROOT / "scripts/run_colab_v2/esiti_devset.npz"),
            ("v3", ROOT / "scripts/run_colab_v3/esiti_devset.npz"))
N_BOOT = 20000


def sessioni_da_argv(argv):
    """Coppie `tag=percorso` sulla riga di comando, per una sessione non ancora in repo."""
    extra = [a.split("=", 1) for a in argv if "=" in a]
    return tuple((t, Path(p)) for t, p in extra) or SESSIONI


def spacchetta(z, prefisso, k=10):
    """Gli esiti per source di un metodo, una lista di array (uno per query)."""
    piatto, lunghezze = z[f"{prefisso}_k{k}_flat"], z[f"{prefisso}_k{k}_len"]
    ptr = np.concatenate([[0], np.cumsum(lunghezze)])
    return [piatto[ptr[i]:ptr[i + 1]] for i in range(len(lunghezze))]


def livello(per_source, indici):
    """Macro-media sulle query indicate: ogni query pesa uguale."""
    return 100 * np.mean([per_source[i].mean() for i in indici])


def main():
    meta = torch.load(ROOT / "embeddings/celeba_test_clip_vitb32.pt", map_location="cpu")
    attr = [n for n in meta["attr_names"] if n.strip()]
    dev = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr)
    n_cond = np.array([len(e["queried"]) for e in dev])
    n_neg = np.array([len(e["neg"]) for e in dev])
    tutte = np.arange(len(dev))

    rng = np.random.default_rng(0)
    RIS = {"prodotto_da": "esperimenti/linearita_per_strato.py",
           "n_boot": N_BOOT, "sessioni": {}}

    for tag, path in sessioni_da_argv(sys.argv[1:]):
        if not path.exists():
            print(f"[{tag}] {path} non c'e': salto")
            continue
        z = np.load(path)
        base = spacchetta(z, "baseline")
        nome_ens = next(p for p in {k.rsplit("_k", 1)[0] for k in z.files}
                        if p.startswith("ENSEMBLE"))
        ens = spacchetta(z, nome_ens)
        mat = spacchetta(z, "MATRICE_collassata")
        membri = [spacchetta(z, p) for p in sorted({k.rsplit("_k", 1)[0] for k in z.files})
                  if p.startswith("membro")]

        S = RIS["sessioni"][tag] = {
            "file": str(path), "nome_ensemble": nome_ens, "n_membri": len(membri),
            "livelli": {"baseline": livello(base, tutte), "matrice": livello(mat, tutte),
                        "ensemble": livello(ens, tutte)},
            "per_n_condizioni": {}, "per_n_negative": {}, "una_condizione": {}}
        print(f"\n{'=' * 92}\n{tag} — {path}  ({nome_ens})\n{'=' * 92}")
        print(f"controllo: baseline {livello(base, tutte):.2f}%   "
              f"matrice {livello(mat, tutte):.2f}%   ensemble {livello(ens, tutte):.2f}%")

        print(f"\n1) per numero di condizioni\n{'strato':>10}{'n':>5}{'baseline':>10}"
              f"{'membro med':>12}{'matrice':>10}{'ensemble':>10}{'ens-mat':>10}{'IC 95%':>19}{'p':>8}")
        for nome, sel in (("1 cond", n_cond == 1), ("2 cond", n_cond == 2),
                          ("3 cond", n_cond == 3), ("TUTTE", n_cond > 0)):
            idx = np.where(sel)[0]
            r = bootstrap_paired([ens[i] for i in idx], [mat[i] for i in idx],
                                 n_boot=N_BOOT, seed=0)
            mm = np.mean([livello(m, idx) for m in membri])
            S["per_n_condizioni"][nome] = {
                "n": len(idx), "baseline": livello(base, idx), "membro_medio": float(mm),
                "matrice": livello(mat, idx), "ensemble": livello(ens, idx),
                "ens_meno_mat_pp": 100 * r["delta"], "lo": 100 * r["lo"],
                "hi": 100 * r["hi"], "p": r["p"]}
            print(f"{nome:>10}{len(idx):>5}{livello(base, idx):>9.2f}%{mm:>11.2f}%"
                  f"{livello(mat, idx):>9.2f}%{livello(ens, idx):>9.2f}%"
                  f"{100 * r['delta']:>+9.2f}pp"
                  f"{'[%+.2f, %+.2f]' % (100 * r['lo'], 100 * r['hi']):>19}{r['p']:>8.4f}")

        print(f"\n2) per numero di condizioni NEGATIVE\n{'negative':>10}{'n':>5}"
              f"{'ens-mat':>10}{'IC 95%':>19}{'p':>8}")
        for k in (0, 1, 2, 3):
            idx = np.where(n_neg == k)[0]
            if len(idx) < 3:
                continue
            r = bootstrap_paired([ens[i] for i in idx], [mat[i] for i in idx],
                                 n_boot=N_BOOT, seed=0)
            S["per_n_negative"][str(k)] = {
                "n": len(idx), "ens_meno_mat_pp": 100 * r["delta"],
                "lo": 100 * r["lo"], "hi": 100 * r["hi"], "p": r["p"]}
            print(f"{k:>10}{len(idx):>5}{100 * r['delta']:>+9.2f}pp"
                  f"{'[%+.2f, %+.2f]' % (100 * r['lo'], 100 * r['hi']):>19}{r['p']:>8.4f}")

        # --- 3) il confronto pulito: a parita' di numero di condizioni ----------------------
        # Le query a UNA condizione sono 27 '+attributo' e 27 '-attributo': gruppi bilanciati,
        # nessun confondimento col numero di condizioni. I due gruppi NON sono pero' pari in
        # difficolta' (la baseline e' piu' alta sui negativi), e questo va dichiarato.
        pos = np.where((n_cond == 1) & (n_neg == 0))[0]
        neg = np.where((n_cond == 1) & (n_neg == 1))[0]
        divario = np.array([100 * (ens[i].mean() - mat[i].mean()) for i in range(len(dev))])
        d_pos, d_neg = divario[pos], divario[neg]
        osservato = d_neg.mean() - d_pos.mean()

        boot = np.array([rng.choice(d_neg, len(d_neg), True).mean()
                         - rng.choice(d_pos, len(d_pos), True).mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])

        insieme = np.concatenate([d_pos, d_neg])
        estremi = 0
        for _ in range(N_BOOT):
            rng.shuffle(insieme)
            estremi += (insieme[len(d_pos):].mean() - insieme[:len(d_pos)].mean()) >= osservato
        p_perm = (estremi + 1) / (N_BOOT + 1)

        print(f"\n3) SOLO query a una condizione: {len(pos)} positive contro {len(neg)} negative")
        print(f"{'gruppo':>12}{'n':>4}{'baseline':>10}{'matrice':>10}{'ensemble':>10}{'ens-mat':>10}")
        for nome, idx in (("+attributo", pos), ("-attributo", neg)):
            print(f"{nome:>12}{len(idx):>4}{livello(base, idx):>9.2f}%{livello(mat, idx):>9.2f}%"
                  f"{livello(ens, idx):>9.2f}%{livello(ens, idx) - livello(mat, idx):>+9.2f}pp")
        S["una_condizione"] = {
            "n_positive": len(pos), "n_negative": len(neg),
            "baseline_positive": livello(base, pos), "baseline_negative": livello(base, neg),
            "divario_positive_pp": float(d_pos.mean()),
            "divario_negative_pp": float(d_neg.mean()),
            "differenza_pp": float(osservato), "lo": float(lo), "hi": float(hi),
            "p_permutazione": float(p_perm)}
        print(f"   differenza fra i due divari: {osservato:+.2f}pp  "
              f"IC [{lo:+.2f}, {hi:+.2f}]  p(permutazione) = {p_perm:.4f}")

    out = HERE / "linearita_per_strato.json"
    json.dump(RIS, open(out, "w"), indent=1)
    print(f"\nscritto {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
