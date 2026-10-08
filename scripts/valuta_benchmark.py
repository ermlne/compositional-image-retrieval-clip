"""VALUTAZIONE SUL BENCHMARK UFFICIALE — si esegue UNA VOLTA SOLA.

===========================================================================================
LEGGERE PRIMA DI ESEGUIRE
===========================================================================================
Questo script legge `celeba_evaluation.json`, cioe' il ground truth del docente sul test
split. E' l'unico file del progetto che lo fa per valutare un metodo candidato.

Il budget del benchmark e' PER METODO, ed e' gia' stato speso una volta: il 25/08, da
`scripts/zeroshot_filter.py` per il filtro zero-shot, con esito negativo (R@10 13.00% contro
15.92% della baseline). Il contributo ha ancora il suo, e questo script lo spende.

COSA SI RIPORTA, E CON QUALE RUOLO, E' GIA' DECISO: sta in `docs/previsione-benchmark.md`,
scritto PRIMA di eseguire, insieme alla previsione numerica e all'intervallo. Il metodo
PRIMARIO e' la matrice collassata; le altre tre righe sono secondarie e si riportano tutte,
qualunque cosa dicano. Guardare i numeri e poi promuovere a primario quello uscito meglio
sarebbe selezione sul test, ed e' la trappola che ha rovesciato sei conclusioni di questo
progetto.

Lo script rifiuta di partire se quel documento non esiste.

===========================================================================================
COSA VALUTA
===========================================================================================
Le 13 query distinte del JSON (le entry sono 14: `-Young` compare due volte identica),
Recall@K e Precision@K a K = 1, 5, 10, macro-average su query, usando SOLO gli indici di
source forniti dal JSON — la regola della sez. 3.1 dell'assignment:

  "You must restrict your evaluation strictly to the source image indices (keys) provided in
   the JSON for each respective query."

Cinque righe, tutte pre-dichiarate:

  1  solo reference          controllo: nessuna condizione, q = r
  2  baseline dell'assignment  q = normalize(r + d), d = somma delle direzioni testuali
  3  MATRICE COLLASSATA      PRIMARIO: q = normalize(r + [r; c; 1] @ W)
  4  membro singolo          il primo dei 12, forma minima del metodo
  5  ensemble di 4 e di 12   uscite mediate, come in addestramento

Il confronto e' accoppiato (`src/devset.bootstrap_paired`) contro la baseline, source per
source. Con 13 query l'errore standard della macro-media e' 2.58pp: e' scritto anche
nell'uscita, perche' il numero non va letto alla prima cifra decimale.

===========================================================================================
USO
===========================================================================================
    HF_HUB_OFFLINE=1 python scripts/valuta_benchmark.py --conferma

Senza `--conferma` lo script stampa cosa farebbe e si ferma: e' una spesa irreversibile e
non deve poter partire per sbaglio. Serve la cartella `scripts/run_colab_v3/ckpt/` con i 12
membri e `matrice_collassata.pt` (i .pt non sono in git: stanno in locale e su Drive).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark
from src.devset import bootstrap_paired

ap = argparse.ArgumentParser()
ap.add_argument("--conferma", action="store_true",
                help="senza questo lo script mostra cosa farebbe e si ferma")
ap.add_argument("--ckpt", default="scripts/run_colab_v3/ckpt")
args = ap.parse_args()

MODEL_ID = "openai/clip-vit-base-patch32"
K_VALUES = [1, 5, 10]
N_BOOT = 20000
CKPT = ROOT / args.ckpt
PREVISIONE = ROOT / "docs/previsione-benchmark.md"
MEMBRI = [f"ms1400_tau0.01_wd0.0001_h512_q{q}_off_cx.pt" for q in range(11, 23)]

print("=" * 100)
print("BENCHMARK UFFICIALE — spesa unica")
print("=" * 100)

# --- guardie ------------------------------------------------------------------------------
if not PREVISIONE.exists():
    sys.exit(f"MANCA {PREVISIONE.relative_to(ROOT)}.\n"
             "La previsione e il ruolo di ogni riga vanno scritti PRIMA di misurare: e' cio'\n"
             "che impedisce di razionalizzare il risultato a posteriori. Non proseguo.")
mancanti = [m for m in MEMBRI + ["matrice_collassata.pt"] if not (CKPT / m).exists()]
if mancanti:
    sys.exit(f"mancano {len(mancanti)} checkpoint in {CKPT}:\n  " + "\n  ".join(mancanti[:5])
             + "\nI .pt non sono in git: prendili da Drive, in DL-Project-2026/run_v3/ckpt/.")
print(f"previsione registrata : {PREVISIONE.relative_to(ROOT)}")
print(f"checkpoint            : {CKPT.relative_to(ROOT)} (12 membri + matrice)")

if not args.conferma:
    print("\n" + "!" * 100)
    print("PROVA A VUOTO — non e' stato letto nulla del benchmark.")
    print("!" * 100)
    print("Questo script spenderebbe il budget del benchmark ufficiale per il contributo.")
    print("Il budget e' PER METODO ed e' gia' stato speso una volta (filtro zero-shot,")
    print("25/08, esito negativo). Questa e' la seconda e ultima spesa prevista.")
    print("\nValuterebbe, con i ruoli gia' dichiarati in docs/previsione-benchmark.md:")
    print("   1  solo reference                  controllo")
    print("   2  baseline dell'assignment        termine di paragone")
    print("   3  MATRICE COLLASSATA              <- PRIMARIO")
    print("   4  membro singolo                  secondario")
    print("   5  ensemble di 4 e di 12           secondari")
    print("\nPrevisione depositata: ensemble 30-35%, matrice ~2pp sotto, baseline attesa")
    print("15.92%. Errore standard della macro-media su 13 query: 2.58pp (+-5pp al 95%).")
    print("\nQuando siete pronti:  python scripts/valuta_benchmark.py --conferma")
    sys.exit(0)

# ===========================================================================================
# DATI
# ===========================================================================================
print("\ncarico gli embedding del TEST split...")
d = torch.load(ROOT / "embeddings/celeba_test_clip_vitb32.pt", map_location="cpu")
E = d["embeddings"].float()
E = E / E.norm(dim=-1, keepdim=True)
ATTR = [n for n in d["attr_names"] if str(n).strip()]
D, N_ATTR = E.shape[1], len(ATTR)
bench = load_benchmark(str(ROOT / "celeba_evaluation.json"), ATTR)
distinte = {e["query"] for e in bench}
print(f"  test {tuple(E.shape)}, {len(bench)} entry ({len(distinte)} query distinte), "
      f"{sum(len(e['sources']) for e in bench)} reference")

print("costruisco le direzioni testuali (10 template, come in addestramento)...")
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
    f = (out.pooler_output if hasattr(out, "pooler_output") else out).float()
    return f / f.norm(dim=-1, keepdim=True)


d_ens = torch.zeros(N_ATTR, D)
for i, nome in enumerate(ATTR):
    c = nome.replace("_", " ").lower()
    v = encode([t.format(c) for t, _ in TEMPLATES]).mean(0) - \
        encode([t.format(c) for _, t in TEMPLATES]).mean(0)
    d_ens[i] = v / v.norm()
del clip_model, tokenizer

# controllo: le direzioni devono coincidere con quelle salvate nei checkpoint, altrimenti
# il modello riceverebbe in ingresso un `c` diverso da quello su cui e' stato addestrato
_ck0 = torch.load(CKPT / MEMBRI[0], map_location="cpu", weights_only=False)
_dif = (d_ens - _ck0["dirs_text"]).abs().max().item()
print(f"  direzioni identiche a quelle dell'addestramento: max|diff| = {_dif:.2e}"
      f"  {'OK' if _dif < 1e-5 else '*** DIVERSE, NON PROSEGUIRE ***'}")
assert _dif < 1e-5, "le direzioni testuali non coincidono con quelle dei checkpoint"


# ===========================================================================================
# MODELLI
# ===========================================================================================
class Concat(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, r, c):
        q = r + self.gamma * self.net(torch.cat([r, c], dim=-1))
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


modelli = []
for m in MEMBRI:
    ck = torch.load(CKPT / m, map_location="cpu", weights_only=False)
    mod = Concat(ck["config"]["d"], ck["config"]["hidden"])
    mod.load_state_dict(ck["A"])
    mod.eval()
    modelli.append(mod)
W = torch.load(CKPT / "matrice_collassata.pt", map_location="cpu", weights_only=False)["W"]
print(f"caricati {len(modelli)} membri e la matrice {tuple(W.shape)}")


def cond(entry):
    c = torch.zeros(D)
    for a in entry["pos"]:
        c = c + d_ens[a]
    for b in entry["neg"]:
        c = c - d_ens[b]
    return c


def q_solo_ref(ref, entry):
    return ref / ref.norm(dim=-1, keepdim=True)


def q_baseline(ref, entry):
    q = ref + cond(entry)
    return q / q.norm(dim=-1, keepdim=True)


def q_matrice(ref, entry):
    c = cond(entry).expand(len(ref), -1)
    uno = torch.ones(len(ref), 1)
    q = ref + torch.cat([ref, c, uno], dim=1) @ W
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def q_ensemble(k):
    def f(ref, entry):
        c = cond(entry).expand(len(ref), -1)
        with torch.no_grad():
            dl = torch.stack([mo.gamma * mo.net(torch.cat([ref, c], dim=-1))
                              for mo in modelli[:k]]).mean(0)
        q = ref + dl
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


# ===========================================================================================
# VALUTAZIONE — solo sugli indici di source del JSON, come impone la sez. 3.1
# ===========================================================================================
def valuta(build, batch=512):
    """Ritorna {k: [array per query]} di recall e di precision, per il bootstrap."""
    rec = {k: [] for k in K_VALUES}
    pre = {k: [] for k in K_VALUES}
    visti = set()
    for e in bench:
        if e["query"] in visti:      # `-Young` compare due volte identica nel JSON
            continue
        visti.add(e["query"])
        src = e["sources"]
        r_q = {k: np.zeros(len(src)) for k in K_VALUES}
        p_q = {k: np.zeros(len(src)) for k in K_VALUES}
        for st in range(0, len(src), batch):
            sp = min(st + batch, len(src))
            b = src[st:sp]
            sim = build(E[b], e) @ E.T
            sim[torch.arange(sp - st), b] = -torch.inf
            for j, top in enumerate(sim.topk(max(K_VALUES), dim=-1).indices.tolist()):
                gt = e["gt"][st + j]
                cum = np.array([i in gt for i in top], dtype=bool).cumsum()
                for k in K_VALUES:
                    r_q[k][st + j] = float(cum[k - 1] > 0)
                    p_q[k][st + j] = cum[k - 1] / k
        for k in K_VALUES:
            rec[k].append(r_q[k])
            pre[k].append(p_q[k])
    return rec, pre


def macro(per_query):
    return float(np.mean([v.mean() for v in per_query])) * 100


METODI = [("solo reference", q_solo_ref),
          ("baseline dell'assignment", q_baseline),
          ("MATRICE collassata  [PRIMARIO]", q_matrice),
          ("membro singolo", q_ensemble(1)),
          ("ensemble di 4", q_ensemble(4)),
          ("ensemble di 12", q_ensemble(12))]

print("\n" + "=" * 100)
print(f"RISULTATI — {len(distinte)} query distinte, macro-average")
print("=" * 100)
RIS, PS = {}, {}
for nome, f in METODI:
    rec, pre = valuta(f)
    PS[nome] = rec
    RIS[nome] = {"recall": {k: macro(rec[k]) for k in K_VALUES},
                 "precision": {k: macro(pre[k]) for k in K_VALUES}}
    print(f"  {nome:<34}" + "  ".join(f"R@{k}={RIS[nome]['recall'][k]:5.2f}%" for k in K_VALUES)
          + "   " + "  ".join(f"P@{k}={RIS[nome]['precision'][k]:5.2f}%" for k in K_VALUES),
          flush=True)

BASE = "baseline dell'assignment"
b10 = RIS[BASE]["recall"][10]
print(f"\ncontrollo: baseline attesa 15.92%, ottenuta {b10:.2f}%  "
      f"{'OK' if abs(b10 - 15.92) < 0.05 else '*** DISCREPANZA, verificare prima di riportare ***'}")

print("\n" + "=" * 100)
print("CONFRONTI ACCOPPIATI contro la baseline (bootstrap gerarchico, R@10)")
print("=" * 100)
for nome, _ in METODI:
    if nome == BASE:
        continue
    s = bootstrap_paired(PS[nome][10], PS[BASE][10], n_boot=N_BOOT, seed=0)
    RIS[nome]["vs_baseline"] = {"delta_pp": s["delta"] * 100, "lo": s["lo"] * 100,
                                "hi": s["hi"] * 100, "p": s["p"],
                                "query_vinte": s["n_query_win"],
                                "query_perse": s["n_query_loss"]}
    v = RIS[nome]["vs_baseline"]
    print(f"  {nome:<34}{v['delta_pp']:>+7.2f}pp  [{v['lo']:+.2f}, {v['hi']:+.2f}]  "
          f"p = {v['p']:.4f}   ({v['query_vinte']} query vinte / {v['query_perse']} perse)")

print("\n" + "=" * 100)
print("COME LEGGERE QUESTI NUMERI")
print("=" * 100)
print("  Con 13 query l'errore standard della macro-media e' 2.58pp (misurato, src/devset.py):")
print("  circa +-5pp al 95%. Il numero non va letto alla prima cifra decimale, e differenze")
print("  sotto ~6.8pp fra due metodi NON sono risolvibili su questo benchmark — quindi e'")
print("  atteso che matrice ed ensemble risultino indistinguibili fra loro.")
print("  La previsione depositata prima della misura e' in docs/previsione-benchmark.md.")

OUT = ROOT / "scripts" / "benchmark_ufficiale.json"
OUT.write_text(json.dumps(
    {"data": datetime.now().isoformat(timespec="seconds"),
     "n_query_distinte": len(distinte), "checkpoint": str(CKPT.relative_to(ROOT)),
     "membri": MEMBRI, "risultati": RIS,
     "nota": ("spesa unica del budget per metodo; ruoli dichiarati prima della misura in "
              "docs/previsione-benchmark.md")},
    indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nsalvato: {OUT.relative_to(ROOT)}")
