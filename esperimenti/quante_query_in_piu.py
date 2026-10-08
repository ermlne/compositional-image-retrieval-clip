"""Quante query in piu' possiamo davvero dare al modulo? — resa e costo, misurati.

===========================================================================================
LA DOMANDA
===========================================================================================
Il canale delle condizioni e' identificabile solo dentro lo span delle 40 direzioni testuali
(dimensione <= 40, rango effettivo ~20 su 512). Piu' query non allargano quello span, ma
DENTRO lo span aggiungono direzioni distinte, e li' il fit e' limitato dai dati:

    1 condizione     80 direzioni distinte      l'addestramento le usa gia' TUTTE e 80
    2 condizioni     C(40,2) x 4  =    3.120     ne usa 80   (2,6%)
    3 condizioni     C(40,3) x 8  =   79.040     ne usa 60   (0,08%)
    4 condizioni     C(40,4) x 16 = 1.462.320     mai provate

Ma una query campionata non e' automaticamente utilizzabile. `build_benchmark_from_labels`
la tiene solo se esistono reference che violano OGNI condizione e che hanno almeno
`min_targets` target validi (soddisfano le condizioni E Hamming <= 2 sui restanti). Con piu'
condizioni entrambe le richieste si stringono, e il ground truth su CelebA si assottiglia.

Oggi: 182 query valide su 220 campionate, ~24 esempi ciascuna (4407 in totale).

QUESTO SCRIPT MISURA, senza addestrare niente:
  - il tasso di sopravvivenza per numero di condizioni (1, 2, 3, 4, 5)
  - gli esempi (coppie reference-target) prodotti per query campionata
  - quindi il costo reale di portare l'addestramento da 220 a N query
  - e quante direzioni `c` DISTINTE si guadagnano, che e' la cosa che conta

Tutto sul TRAIN split, che e' quello che l'addestramento usa. Il set di sviluppo e il
benchmark ufficiale non vengono toccati.

Uso: HF_HUB_OFFLINE=1 python esperimenti/quante_query_in_piu.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import build_benchmark_from_labels
from src.devset import sample_queries


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


STAMP = datetime.now().strftime("%Y%m%d-%H%M")
LOG_PATH = HERE / f"quante_query_in_piu_{STAMP}.txt"
JSON_PATH = HERE / f"quante_query_in_piu_{STAMP}.json"
_tee = Tee(LOG_PATH)
sys.stdout = _tee

SEED = 2026
N_TRAIN_CORPUS = 20000      # identico a multiseed.py
MAX_SOURCES = 80            # identico a multiseed.py
N_PROVA = 120               # query campionate per ogni numero di condizioni
CONDIZIONI = (1, 2, 3, 4, 5)

print("=" * 100)
print("QUANTE QUERY IN PIU' — resa e costo per numero di condizioni")
print("=" * 100)
print(f"log: {LOG_PATH.name}   json: {JSON_PATH.name}")
print(f"corpus di addestramento {N_TRAIN_CORPUS}, max_sources {MAX_SOURCES}, "
      f"{N_PROVA} query provate per livello (seed {SEED})")

d = torch.load(ROOT / "embeddings/celeba_train_clip_vitb32.pt", map_location="cpu")
L_all = d["labels"]
attr_names = [n for n in d["attr_names"] if str(n).strip()]
N_ATTR = len(attr_names)

# stessa sottoselezione del corpus di multiseed.py, cosi' i numeri sono confrontabili
g = torch.Generator().manual_seed(SEED)
sub = torch.randperm(len(L_all), generator=g)[:N_TRAIN_CORPUS].sort().values
L_tr = L_all[sub]
print(f"corpus: {tuple(L_tr.shape)}\n")

from math import comb

print("=" * 100)
print("SOPRAVVIVENZA E RESA")
print("=" * 100)
print(f"{'cond.':>6}{'spazio totale':>16}{'provate':>9}{'valide':>8}{'resa':>8}"
      f"{'esempi':>9}{'es./provata':>13}{'ref/query':>11}{'sec':>7}")

RIS = {}
for n_cond in CONDIZIONI:
    spazio = comb(N_ATTR, n_cond) * (2 ** n_cond)
    t0 = time.time()
    queries = sample_queries(attr_names, N_PROVA, n_cond, seed=SEED + n_cond)
    bench = []
    for q in queries:
        bench += build_benchmark_from_labels([q], L_tr, attr_names,
                                             max_sources=MAX_SOURCES, seed=SEED)
    n_ex = sum(len(e["sources"]) for e in bench)
    dt = time.time() - t0
    RIS[n_cond] = {
        "spazio_totale": spazio, "provate": len(queries), "valide": len(bench),
        "resa": len(bench) / max(len(queries), 1),
        "esempi": int(n_ex),
        "esempi_per_provata": n_ex / max(len(queries), 1),
        "ref_per_query_valida": n_ex / max(len(bench), 1),
        "secondi": dt,
    }
    r = RIS[n_cond]
    print(f"{n_cond:>6}{spazio:>16,}{len(queries):>9}{len(bench):>8}"
          f"{r['resa'] * 100:>7.1f}%{n_ex:>9}{r['esempi_per_provata']:>13.1f}"
          f"{r['ref_per_query_valida']:>11.1f}{dt:>7.0f}")

print("\n  resa      = frazione delle query campionate che produce almeno una reference con")
print("              >= 5 target validi (la regola dell'assignment applicata al train)")
print("  ref/query = quante coppie (reference, target) porta una query valida")

# ===========================================================================================
# COSA COSTA PORTARE L'ADDESTRAMENTO A PIU' QUERY
# ===========================================================================================
print("\n" + "=" * 100)
print("COSA COSTA PORTARE L'ADDESTRAMENTO A PIU' QUERY")
print("=" * 100)
print("Configurazione attuale di multiseed.py: {1: 80, 2: 80, 3: 60} -> 182 query valide,")
print("4407 esempi, 2,6 minuti per seed su CPU.\n")

ATTUALE = {1: 80, 2: 80, 3: 60}


def stima(targets):
    """Query valide ed esempi attesi per una configurazione, dalle rese misurate."""
    val = ex = 0.0
    for n_cond, quanti in targets.items():
        quanti = min(quanti, comb(N_ATTR, n_cond) * (2 ** n_cond))   # spazio finito
        r = RIS[n_cond]
        val += quanti * r["resa"]
        ex += quanti * r["esempi_per_provata"]
    return val, ex


base_val, base_ex = stima(ATTUALE)
print(f"{'configurazione':<42}{'query valide':>14}{'esempi':>10}{'x attuale':>11}"
      f"{'min/seed':>10}")
PROPOSTE = [
    ("attuale  {1:80, 2:80, 3:60}", ATTUALE),
    ("2x sulle composte  {1:80, 2:160, 3:120}", {1: 80, 2: 160, 3: 120}),
    ("4x sulle composte  {1:80, 2:320, 3:240}", {1: 80, 2: 320, 3: 240}),
    ("4x + quattro condizioni", {1: 80, 2: 320, 3: 240, 4: 240}),
    ("8x sulle composte  {1:80, 2:640, 3:480}", {1: 80, 2: 640, 3: 480}),
]
STIME = {}
for nome, targets in PROPOSTE:
    v, e = stima(targets)
    STIME[nome] = {"targets": targets, "query_valide": v, "esempi": e,
                   "fattore": e / base_ex, "minuti_stimati": 2.6 * e / base_ex}
    print(f"{nome:<42}{v:>14.0f}{e:>10.0f}{e / base_ex:>10.1f}x"
          f"{2.6 * e / base_ex:>10.1f}")
print("\n  I minuti sono una stima lineare negli esempi: il costo per epoca e' dominato dal")
print("  denominatore InfoNCE su tutto il corpus, che scala con il numero di esempi.")

# ===========================================================================================
# QUANTE DIREZIONI c DISTINTE SI GUADAGNANO DAVVERO
# ===========================================================================================
print("\n" + "=" * 100)
print("QUANTE DIREZIONI c DISTINTE SI GUADAGNANO")
print("=" * 100)
print("E' la quantita' che conta: `c` vive nello span delle 40 direzioni (rango effettivo")
print("~20 su 512), quindi piu' query non allargano lo spazio — aggiungono punti dentro.\n")
print(f"{'configurazione':<42}{'direzioni c distinte':>22}{'x attuale':>12}")
base_dir = sum(min(q, comb(N_ATTR, n) * (2 ** n)) for n, q in ATTUALE.items())
for nome, targets in PROPOSTE:
    nd = sum(min(q, comb(N_ATTR, n) * (2 ** n)) for n, q in targets.items())
    STIME[nome]["direzioni_distinte"] = nd
    print(f"{nome:<42}{nd:>22}{nd / base_dir:>11.1f}x")

print("\n  Nota: le query a 1 condizione sono gia' esaurite (80 su 80), quindi tutto il")
print("  guadagno viene dalle composte — che e' anche dove il contributo gia' rende di piu'")
print("  (+8,69pp contro +5,43pp) e dove il bagging puo' avere diversita' da sfruttare.")

# ===========================================================================================
# VERDETTO
# ===========================================================================================
print("\n" + "=" * 100)
print("VERDETTO")
print("=" * 100)
r4 = RIS.get(4)
if r4 and r4["resa"] >= 0.30:
    print(f"  Le query a 4 condizioni sono utilizzabili: resa {r4['resa'] * 100:.0f}%, "
          f"{r4['ref_per_query_valida']:.0f} reference per query valida.")
elif r4:
    print(f"  Le query a 4 condizioni rendono poco: resa {r4['resa'] * 100:.0f}%, "
          f"{r4['esempi_per_provata']:.1f} esempi per query campionata contro "
          f"{RIS[3]['esempi_per_provata']:.1f} a 3 condizioni.")
    print("  Conviene spendere il budget su piu' query a 2 e 3 condizioni.")
print()
print("  Le query a 1 condizione sono esaurite: 80 su 80 gia' usate. Aumentarne il numero")
print("  non fa nulla — e spiega perche' il bagging non puo' avere diversita' li'.")

payload = {"data": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
           "n_prova": N_PROVA, "resa_per_condizioni": RIS, "stime_configurazioni": STIME}
JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSalvato: {LOG_PATH.name} e {JSON_PATH.name}")
sys.stdout = _tee.stdout
_tee.close()
