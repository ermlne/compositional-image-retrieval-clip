"""GENERALIZZA AD ATTRIBUTI MAI VISTI? — il secondo asse, quello non ancora verificato.

===========================================================================================
LA DOMANDA
===========================================================================================
Il contributo candidato generalizza a COMBINAZIONI nuove: 46 delle 47 query composte del set
di sviluppo non compaiono in addestramento, e li' il guadagno e' anche piu' grande. Ma tutti
e 40 gli ATTRIBUTI compaiono in addestramento — con 40 attributi e 2 segni le query a una
condizione sono solo 80, e l'addestramento le usa tutte. Sono due assi diversi e finora solo
il primo era verificato (`docs/STATO.md`, sez. 3, punto 3).

IL TEST. Si tolgono 10 dei 29 attributi che il set di sviluppo usa, si addestra su query che
non li nominano mai, e si valuta sulle query di sviluppo che nominano SOLO quelli. Il
modello non ha mai visto quelle direzioni testuali durante l'addestramento.

  strato "visti"       query di sviluppo che non toccano nessun attributo tenuto fuori
  strato "MAI VISTI"   query che toccano SOLO attributi tenuti fuori   <- e' questo che conta
  strato "misti"       le altre

COSA SIGNIFICANO I DUE ESITI.
  Se il guadagno sopravvive sullo strato mai visto, il modello ha imparato una TRASFORMAZIONE
  dello spazio (una correzione testo -> visivo che vale per direzioni qualunque), non una
  tabella di 40 spostamenti memorizzati. E' l'argomento piu' forte disponibile contro
  l'accusa di lookup, ed e' anche cio' che distingue questo metodo dalla pista del filtro per
  attributi, scartata proprio perche' era un lookup su un nome.
  Se il guadagno sparisce, il metodo e' a vocabolario chiuso e va detto: resterebbe valido
  come contributo, ma con un limite dichiarato.

SCELTE PRE-REGISTRATE, prima di guardare qualunque risultato:
  - gli split sono i primi tre generati da `random.Random(0|1|2).sample(attributi_dev, 10)`,
    NON quelli che massimizzano la numerosita' dello strato (sceglierli guardando i conteggi
    sarebbe la stessa trappola di tutta la sez. 2 di STATO.md);
  - metrica primaria: differenza accoppiata su R@10, metodo contro baseline, DENTRO lo strato
    mai visto;
  - tutto il resto della configurazione e' identico a multiseed.py (tau 0.03, 30 epoche).

Il benchmark ufficiale NON viene toccato.

Uso:  HF_HUB_OFFLINE=1 python contributo-candidato/attributi_heldout.py --seed 2026 --split 0
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
for i, name in enumerate(attr_names):
    c = name.replace("_", " ").lower()
    v = encode([t.format(c) for t, _ in TEMPLATES]).mean(0) - \
        encode([t.format(c) for _, t in TEMPLATES]).mean(0)
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


EX = []
for e in train_bench:
    u, m = cond_tensor(e)
    for si, ref in enumerate(e["sources"].tolist()):
        EX.append((ref, u, m, torch.tensor(sorted(e["gt"][si]), dtype=torch.long)))


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, U, M):
        c = (U * M.unsqueeze(-1)).sum(1)
        q = r + self.gamma * self.net(torch.cat([r, c], dim=-1))
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def info_nce_multipos(q, refs, gts, tau):
    sim = (q @ E_tr.T) / tau
    B, N = sim.shape
    ar = torch.arange(B)
    sim = sim.clone()
    sim[ar, refs] = -float("inf")
    pos = torch.zeros(B, N, dtype=torch.bool)
    for i, gt in enumerate(gts):
        pos[i, gt] = True
    pos[ar, refs] = False
    return -(torch.logsumexp(sim.masked_fill(~pos, -float("inf")), 1)
             - torch.logsumexp(sim, 1)).mean()


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


def make_builder(model):
    cache = {}
    def f(ref, entry):
        if entry["query"] not in cache:
            cache[entry["query"]] = cond_tensor(entry)
        u, m = cache[entry["query"]]
        with torch.no_grad():
            return model(ref, u.unsqueeze(0).expand(len(ref), -1, -1),
                         m.unsqueeze(0).expand(len(ref), -1))
    return f


def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)


torch.manual_seed(SEED)
model = Concat()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
gen = torch.Generator().manual_seed(SEED)
for ep in range(args.epochs):
    model.train()
    order = torch.randperm(len(EX), generator=gen)
    tot, nb = 0.0, 0
    for st in range(0, len(EX), BATCH):
        idx = order[st:st + BATCH].tolist()
        refs = torch.tensor([EX[i][0] for i in idx])
        U = torch.stack([EX[i][1] for i in idx]); M = torch.stack([EX[i][2] for i in idx])
        q = model(E_tr[refs], U, M)
        loss = info_nce_multipos(q, refs, [EX[i][3] for i in idx], args.tau)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    if (ep + 1) % 10 == 0:
        print(f"  epoca {ep+1:>3}  loss {tot/nb:6.3f}  gamma {model.gamma.item():5.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
model.eval()

# controllo positivo sulle query di addestramento
pc_b = macro(evaluate(build_baseline, train_bench, E_tr, K_VALUES)[10]) * 100
pc_m = macro(evaluate(make_builder(model), train_bench, E_tr, K_VALUES)[10]) * 100
print(f"controllo positivo (train): baseline {pc_b:.2f}%  modello {pc_m:.2f}%  "
      f"-> {'ok' if pc_m > pc_b else 'FALLITO'}", flush=True)

ps_b = evaluate(build_baseline, devset, E_valid, K_VALUES)
ps_m = evaluate(make_builder(model), devset, E_valid, K_VALUES)

# ------------------------------------------------------------------------------------------
# DIAGNOSI E RIMEDIO — il modello sa di non sapere?
# ------------------------------------------------------------------------------------------
# Ipotesi: il danno sugli attributi mai visti viene dal fatto che `c` ha una componente FUORI
# dallo span delle condizioni viste in addestramento, dove la mappa non e' mai stata
# vincolata (stessa causa del rango 589 su 1025 di docs/report-27-08-2026-multiseed.md).
# La quota in-span si calcola senza etichette e senza iperparametri, quindi si puo' usare
# come guadagno moltiplicativo: g(c) = ||P_span c|| / ||c||. Se l'ipotesi regge, il metodo
# degrada da solo verso la baseline sulle condizioni che non riconosce.
C = torch.stack([cond_of(e) for e in train_bench])
C = C / C.norm(dim=-1, keepdim=True)
U_, S_, _ = torch.linalg.svd(C.T, full_matrices=False)
rank = int((S_ > S_[0] * 1e-6).sum())
BASIS = U_[:, :rank]                                     # [D, rank]
print(f"\nspan delle condizioni viste: rango {rank}")


def in_span(c):
    return float((BASIS.T @ c).norm() / c.norm().clamp(min=1e-9))


def build_gated(ref, entry):
    c = cond_of(entry)
    with torch.no_grad():
        d = model.net(torch.cat([ref, c.expand(len(ref), -1)], dim=-1))
    q = ref + model.gamma * in_span(c) * d
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


ps_g = evaluate(build_gated, devset, E_valid, K_VALUES)

res = {"seed": SEED, "split": args.split, "n_holdout": args.n_holdout,
       "holdout": sorted(attr_names[i] for i in HOLD),
       "n_query": {"visti": len(idx_seen), "mai_visti": len(idx_hold), "misti": len(idx_mix)},
       "controllo_positivo": {"baseline": pc_b, "modello": pc_m, "superato": bool(pc_m > pc_b)},
       "strati": {}}
print(f"\n{'strato':<16}{'n':>5}{'baseline':>11}{'modello':>10}{'differenza':>13}"
      f"{'IC 95%':>18}{'p':>9}")
for nome, idx in (("visti", idx_seen), ("MAI VISTI", idx_hold), ("misti", idx_mix),
                  ("tutte", list(range(len(devset))))):
    a = [ps_m[10][i] for i in idx]; b = [ps_b[10][i] for i in idx]
    s = bootstrap_paired(a, b, n_boot=20000, seed=0)
    res["strati"][nome] = {"n": len(idx), "baseline": macro(b) * 100, "modello": macro(a) * 100,
                           "delta_pp": s["delta"] * 100, "lo": s["lo"] * 100,
                           "hi": s["hi"] * 100, "p": s["p"],
                           "in_span": float(np.mean([in_span(cond_of(devset[i])) for i in idx]))}
    r = res["strati"][nome]
    print(f"{nome:<16}{len(idx):>5}{r['baseline']:>10.2f}%{r['modello']:>9.2f}%"
          f"{r['delta_pp']:>+12.2f}pp  [{r['lo']:+.2f},{r['hi']:+.2f}]{r['p']:>9.4f}")

print(f"\nquota di ||c|| dentro lo span delle condizioni viste (rango {rank}):")
for nome in ("visti", "MAI VISTI", "misti"):
    print(f"  {nome:<12}{res['strati'][nome]['in_span']:.3f}")

print(f"\nrimedio senza iperparametri: delta scalato per ||P_span c|| / ||c||")
print(f"{'strato':<16}{'n':>5}{'baseline':>11}{'con gate':>10}{'differenza':>13}"
      f"{'IC 95%':>18}{'p':>9}")
res["gated"] = {}
for nome, idx in (("visti", idx_seen), ("MAI VISTI", idx_hold), ("misti", idx_mix),
                  ("tutte", list(range(len(devset))))):
    a = [ps_g[10][i] for i in idx]; b = [ps_b[10][i] for i in idx]
    s = bootstrap_paired(a, b, n_boot=20000, seed=0)
    res["gated"][nome] = {"n": len(idx), "modello": macro(a) * 100, "delta_pp": s["delta"] * 100,
                          "lo": s["lo"] * 100, "hi": s["hi"] * 100, "p": s["p"]}
    r = res["gated"][nome]
    print(f"{nome:<16}{len(idx):>5}{macro(b)*100:>10.2f}%{r['modello']:>9.2f}%"
          f"{r['delta_pp']:>+12.2f}pp  [{r['lo']:+.2f},{r['hi']:+.2f}]{r['p']:>9.4f}")
res["rank_span"] = rank

with open(OUT / f"heldout_seed{SEED}_split{args.split}.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"\nsalvato in multiseed/heldout_seed{SEED}_split{args.split}.json "
      f"({(time.time()-t0)/60:.1f} minuti)")
