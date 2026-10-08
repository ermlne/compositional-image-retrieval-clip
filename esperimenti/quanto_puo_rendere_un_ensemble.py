"""Quanto puo' rendere un ensemble dei modelli gia' addestrati, misurato invece che stimato.

L'ensemble rende in proporzione a quanto i modelli sbagliano in modo DIVERSO. Se cinque seed
azzeccano e sbagliano sulle stesse source, mediarli non produce niente; se sbagliano su
source diverse, c'e' margine.

Non servono i pesi dei modelli: bastano gli esiti per singola source, che
`contributo-candidato/multiseed/per_source_seed*.npz` gia' contiene per tutti e cinque i seed.

COSA SI MISURA
  singolo          la media dei cinque modelli presi uno alla volta (il termine di paragone
                   corretto: NON il seed migliore, che sarebbe selezione sul set di sviluppo)
  migliore/peggiore lo spread fra seed, per sapere quanto vale il rumore di addestramento
  maggioranza      >= 3 modelli su 5 azzeccano la stessa source
  unione           almeno 1 modello su 5 azzecca
  concordanza      quanto spesso due modelli danno lo stesso esito sulla stessa source

L'UNIONE NON E' UN LIMITE SUPERIORE RIGOROSO: un vettore query mediato puo' in linea di
principio azzeccare dove tutti i singoli sbagliano. E' pero' la misura giusta della
DIVERSITA' disponibile, ed e' cio' che governa il guadagno atteso.

Il benchmark ufficiale NON viene toccato.

Uso: python esperimenti/quanto_puo_rendere_un_ensemble.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark

SEEDS = (1, 2, 3, 4, 2026)
K = 10

d = torch.load(ROOT / "embeddings/celeba_valid_clip_vitb32.pt", map_location="cpu")
attr_names = [n for n in d["attr_names"] if str(n).strip()]
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)
n_src = [len(e["sources"]) for e in devset]
bordi = np.cumsum([0] + n_src)
print(f"{len(devset)} query, {bordi[-1]} source")


def carica(seed, chiave):
    z = np.load(ROOT / f"contributo-candidato/multiseed/per_source_seed{seed}.npz",
                allow_pickle=True)
    v = z[chiave]
    assert len(v) == bordi[-1], (seed, chiave, len(v))
    return v


def macro(v):
    """Macro-average: ogni query pesa uguale, come nel protocollo ufficiale."""
    return float(np.mean([v[bordi[i]:bordi[i + 1]].mean() for i in range(len(devset))])) * 100


base = carica(2026, f"baseline_{K}")
A = {s: carica(s, f"A_{K}") for s in SEEDS}
b10 = macro(base)
print(f"baseline R@10 = {b10:.2f}%\n")

print("=" * 88)
print("I CINQUE MODELLI, UNO ALLA VOLTA")
print("=" * 88)
singoli = {s: macro(A[s]) for s in SEEDS}
for s in SEEDS:
    print(f"  seed {s:<6} {singoli[s]:6.2f}%   ({singoli[s] - b10:+.2f}pp)")
media_singolo = float(np.mean(list(singoli.values())))
print(f"\n  media dei singoli   {media_singolo:6.2f}%   ({media_singolo - b10:+.2f}pp)  "
      "<- il termine di paragone corretto")
print(f"  migliore            {max(singoli.values()):6.2f}%")
print(f"  peggiore            {min(singoli.values()):6.2f}%")
print(f"  spread              {max(singoli.values()) - min(singoli.values()):6.2f}pp  "
      "<- rumore di addestramento, non differenza di metodo")

print("\n" + "=" * 88)
print("QUANTO SBAGLIANO IN MODO DIVERSO")
print("=" * 88)
M = np.stack([A[s] for s in SEEDS])                       # [5, n_source]
conc = [float((M[i] == M[j]).mean()) for i, j in combinations(range(len(SEEDS)), 2)]
print(f"  concordanza media fra due modelli   {np.mean(conc) * 100:.2f}%  "
      f"(min {min(conc) * 100:.2f}%, max {max(conc) * 100:.2f}%)")
# concordanza attesa se i modelli fossero indipendenti con lo stesso tasso di successo
p = M.mean()
attesa = p * p + (1 - p) * (1 - p)
print(f"  concordanza se fossero indipendenti {attesa * 100:.2f}%")
verdetto_div = ("molto piu' concordi dell'indipendenza: poca diversita' da sfruttare"
                if np.mean(conc) > attesa + 0.02
                else "vicini all'indipendenza: c'e' diversita' da sfruttare")
print(f"  -> {verdetto_div}")

print("\n" + "=" * 88)
print("COMBINAZIONI DEGLI ESITI")
print("=" * 88)
somma = M.sum(0)
for etichetta, v in (("almeno 1 su 5 azzecca (unione)", (somma >= 1).astype(float)),
                     ("almeno 2 su 5", (somma >= 2).astype(float)),
                     ("maggioranza, >= 3 su 5", (somma >= 3).astype(float)),
                     ("almeno 4 su 5", (somma >= 4).astype(float)),
                     ("tutti e 5", (somma == 5).astype(float))):
    m = macro(v)
    print(f"  {etichetta:<34}{m:6.2f}%   ({m - media_singolo:+.2f}pp vs media singolo)")

print("\n" + "=" * 88)
print("LETTURA")
print("=" * 88)
unione = macro((somma >= 1).astype(float))
magg = macro((somma >= 3).astype(float))
print(f"  Il margine che la diversita' dei seed mette a disposizione sta fra")
print(f"  {media_singolo:.2f}% (un modello solo) e {unione:.2f}% (unione, non raggiungibile).")
print(f"  La maggioranza, che e' il regime in cui un ensemble di solito si colloca, da'")
print(f"  {magg:.2f}%, cioe' {magg - media_singolo:+.2f}pp sul modello singolo medio.")
print()
print("  ATTENZIONE al confronto sbagliato: un ensemble contro il seed PEGGIORE vale")
print(f"  {magg - min(singoli.values()):+.2f}pp, contro il seed MIGLIORE {magg - max(singoli.values()):+.2f}pp.")
print("  Il numero difendibile e' quello contro la media dei singoli, e va replicato con")
print("  ensemble costruiti da seed diversi, non riportato dal primo che si prova.")
