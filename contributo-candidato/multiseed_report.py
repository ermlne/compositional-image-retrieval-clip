"""Tabella riassuntiva della replica multi-seed. Legge i JSON prodotti da
multiseed.py e mappa_lineare.py; non ricalcola nulla.

Uso:  python contributo-candidato/multiseed_report.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "multiseed"
seeds = sorted(int(p.stem.replace("seed", "")) for p in OUT.glob("seed*.json"))
S = {s: json.load(open(OUT / f"seed{s}.json")) for s in seeds}
L = {s: json.load(open(OUT / f"lineare_seed{s}.json"))
     for s in seeds if (OUT / f"lineare_seed{s}.json").exists()}


def col(f):
    return np.array([f(S[s]) for s in seeds])


print("=" * 100)
print(f"REPLICA MULTI-SEED — {len(seeds)} seed, set di sviluppo a 101 query")
print("=" * 100)
print("Baseline (deterministica): 19.91% di R@10.  Ogni riga e' un addestramento completo")
print("da capo: corpus, query, reference, inizializzazione e ordine dei batch cambiano.\n")

print(f"{'seed':>6}{'A R@10':>9}{'A vs base':>12}{'p':>9}{'B R@10':>9}{'B vs base':>12}"
      f"{'A-B':>9}{'fuori span':>12}{'ctrl pos':>10}")
for s in seeds:
    d = S[s]
    c = d["confronti"]
    print(f"{s:>6}{d['dev']['A']['r10']:>8.2f}%{c['A_vs_baseline']['delta_pp']:>+11.2f}pp"
          f"{c['A_vs_baseline']['p']:>9.4f}{d['dev']['B']['r10']:>8.2f}%"
          f"{c['B_vs_baseline']['delta_pp']:>+11.2f}pp{c['A_vs_B']['delta_pp']:>+8.2f}pp"
          f"{d['fuori_span_energia_pct']:>11.1f}%"
          f"{'ok' if d['controllo_positivo_superato'] else 'FALLITO':>10}")

a = col(lambda d: d["confronti"]["A_vs_baseline"]["delta_pp"])
b = col(lambda d: d["confronti"]["B_vs_baseline"]["delta_pp"])
sem = col(lambda d: d["confronti"]["A_vs_baseline_semplici"]["delta_pp"])
com = col(lambda d: d["confronti"]["A_vs_baseline_composte"]["delta_pp"])
ba = col(lambda d: d["confronti"]["B_vs_A_composte"]["delta_pp"])

print(f"\n{'':>6}{'media':>9}{'sd':>9}{'min':>9}{'max':>9}   (pp sulla differenza accoppiata)")
for nome, v in (("A vs baseline", a), ("B vs baseline", b),
                ("A semplici", sem), ("A composte", com), ("B-A composte", ba)):
    print(f"{nome:<20}{v.mean():>+8.2f}{v.std(ddof=1):>9.2f}{v.min():>+9.2f}{v.max():>+9.2f}")

print(f"\nseed con A significativo (p<0.05): {int((col(lambda d: d['confronti']['A_vs_baseline']['p']) < 0.05).sum())}/{len(seeds)}")
print(f"seed in cui A batte B:             {int((col(lambda d: d['confronti']['A_vs_B']['delta_pp']) > 0).sum())}/{len(seeds)}")
print(f"seed in cui il guadagno e' maggiore sulle COMPOSTE: {int((com > sem).sum())}/{len(seeds)}")
print(f"controllo positivo superato:       "
      f"{int(col(lambda d: d['controllo_positivo_superato']).sum())}/{len(seeds)}")

if L:
    print("\n" + "=" * 100)
    print("LA MAPPA LINEARE (ridge, lambda=1e-3) — quanto della rete serve davvero")
    print("=" * 100)
    print("P1 = W stimata su reference del TRAIN, valutata sul dev: nessun contatto con i")
    print("dati di valutazione. E' il protocollo pulito.\n")
    print(f"{'seed':>6}{'MLP':>9}{'P1 lineare':>12}{'P1 vs MLP':>12}{'p':>9}"
          f"{'R^2 (P1)':>10}{'quota del guadagno':>20}")
    quote = []
    for s in sorted(L):
        d, l = S[s], L[s]
        mlp = d["dev"]["A"]["r10"]
        lin = l["dev_r10"]["P1 lineare (fit train)"]
        q = (lin - 19.91) / (mlp - 19.91) * 100
        quote.append(q)
        print(f"{s:>6}{mlp:>8.2f}%{lin:>11.2f}%"
              f"{l['vs_MLP']['P1 lineare (fit train)']['delta_pp']:>+11.2f}pp"
              f"{l['vs_MLP']['P1 lineare (fit train)']['p']:>9.4f}"
              f"{l['r2']['P1_train']:>10.4f}{q:>19.0f}%")
    print(f"\nquota del guadagno recuperata dalla sola mappa lineare: "
          f"media {np.mean(quote):.0f}%, min {min(quote):.0f}%, max {max(quote):.0f}%")
