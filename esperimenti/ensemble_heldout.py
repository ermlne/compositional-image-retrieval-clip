"""L'ensemble aiuta anche dove il metodo fallisce, cioe' sugli attributi mai visti?

Addestra piu' modelli sullo stesso split variando **solo l'inizializzazione** — gli esempi
`EX` sono costruiti una volta sola — e ne media le USCITE (mediare i pesi non ha senso: reti
da inizializzazioni diverse stanno in bacini diversi). Confronta media dei singoli contro
insieme, su strato visti e strato MAI VISTI.

E' il controllo che ha reso interpretabile il bagging: a vocabolario chiuso l'insieme di 4
modelli addestrati su campioni di query DIVERSI vale +3.76pp, qui — dove cambia solo il seed —
vale +0.07 / +0.49 / +0.01pp. Da cui "e' bagging, non ensemble di seed"
(`docs/piano-interpretabilita.md` sez. 6e).

--------------------------------------------------------------------------------------------
NOTA DI PROVENIENZA. Questo file nasce come **fork di
`contributo-candidato/attributi_heldout.py`** e ne conserva l'impalcatura: gli stessi 3 split
leave-attribute-out (`random.Random(--split).sample(attributi_del_devset, 10)`), la stessa
configurazione (tau 0.03, 30 epoche, max_sources 80, corpus di 20.000 immagini del train), i
tre strati di valutazione (visti / MAI VISTI / misti) e il bootstrap accoppiato su R@10.
Riaddestra il modulo da zero a ogni invocazione (nessun checkpoint letto).
Fino al 30/08 l'intestazione era ancora quella di `attributi_heldout.py` ("GENERALIZZA AD
ATTRIBUTI MAI VISTI?"), copiata insieme al resto e mai aggiornata: descriveva un esperimento
diverso da quello che il file esegue. Sostituita.

Il benchmark ufficiale non viene toccato.

Uso:    HF_HUB_OFFLINE=1 python esperimenti/ensemble_heldout.py --seed 2026 --split 0
Output: esperimenti/ensemble_heldout_split{split}.json
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


# ==========================================================================================
# L'ENSEMBLE DI SEED ATTENUA IL DANNO SUGLI ATTRIBUTI MAI VISTI?
#
# Sul dev set a vocabolario chiuso l'ensemble di 4 seed vale +10.61pp contro i +6.76pp del
# singolo medio (+3.76pp, tutti i confronti appaiati p<0.002). Qui la domanda che conta: fa
# qualcosa anche dove il metodo FALLISCE, cioe' sugli attributi mai visti (-4.9 / -4.5 / -7.7)?
#
# Le correzioni si mediano sulle USCITE, non sui pesi: reti da inizializzazioni diverse
# stanno in bacini diversi e mediarne i pesi non ha senso.
# ==========================================================================================
import json
N_SEED = 4
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



class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, r, U, M):
        c = (U * M.unsqueeze(-1)).sum(1)
        q = r + self.gamma * self.net(torch.cat([r, c], dim=-1))
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

EX = []
for e in train_bench:
    u, m = cond_tensor(e)
    for si, ref in enumerate(e["sources"].tolist()):
        EX.append((ref, u, m, torch.tensor(sorted(e["gt"][si]), dtype=torch.long)))

def addestra(seed):
    torch.manual_seed(seed)
    model = Concat()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(EX), generator=gen)
        for st in range(0, len(EX), BATCH):
            ix = order[st:st+BATCH].tolist()
            refs = torch.tensor([EX[i][0] for i in ix])
            U = torch.stack([EX[i][1] for i in ix]); M = torch.stack([EX[i][2] for i in ix])
            loss = info_nce_multipos(model(E_tr[refs], U, M), refs,
                                     [EX[i][3] for i in ix], args.tau)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)

def costr(models):
    cache = {}
    def f(ref, entry):
        if entry["query"] not in cache: cache[entry["query"]] = cond_tensor(entry)
        u, m = cache[entry["query"]]
        U = u.unsqueeze(0).expand(len(ref), -1, -1); M = m.unsqueeze(0).expand(len(ref), -1)
        with torch.no_grad():
            c = (U * M.unsqueeze(-1)).sum(1)
            dl = torch.stack([mo.gamma * mo.net(torch.cat([ref, c], dim=-1))
                              for mo in models]).mean(0)
        q = ref + dl
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f

models = []
for sd in range(N_SEED):
    models.append(addestra(SEED + sd))
    print(f"  addestrato seed {SEED + sd} ({len(models)}/{N_SEED})", flush=True)

ps_b = evaluate(build_baseline, devset, E_valid, K_VALUES)
res = {"seed": SEED, "split": args.split, "holdout": sorted(attr_names[i] for i in HOLD),
       "voci": {}}
print("\n" + "=" * 92)
print("ENSEMBLE SUGLI ATTRIBUTI MAI VISTI — Recall@10")
print("=" * 92)
print(f"{'variante':<22}{'strato':<13}{'baseline':>10}{'modello':>10}{'differenza':>13}{'p':>9}")
voci = [(f"seed {SEED+k} da solo", [models[k]]) for k in range(N_SEED)]
voci.append((f"ENSEMBLE dei {N_SEED}", models))
for nome, ms in voci:
    ps = evaluate(costr(ms), devset, E_valid, K_VALUES)
    res["voci"][nome] = {}
    for strato, ix in (("visti", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix)):
        if not ix: continue
        a_ = [ps[10][i] for i in ix]; b_ = [ps_b[10][i] for i in ix]
        stt = bootstrap_paired(a_, b_, n_boot=20000, seed=0)
        res["voci"][nome][strato] = {"delta_pp": stt["delta"]*100, "p": stt["p"],
                                     "modello": macro(a_)*100}
        print(f"{nome:<22}{strato:<13}{macro(b_)*100:>9.2f}%{macro(a_)*100:>9.2f}%"
              f"{stt['delta']*100:>+12.2f}pp{stt['p']:>9.4f}", flush=True)
    print(flush=True)

with open(ROOT / "esperimenti" / f"ensemble_heldout_split{args.split}.json", "w") as f:
    json.dump(res, f, indent=2)
print("salvato.")
