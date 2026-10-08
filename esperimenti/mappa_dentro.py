"""E se la direzione corretta la si desse ALL'MLP invece di usarla al posto della grezza?

Tre varianti riaddestrate su ogni split, per dare alla rete strettamente piu' informazione e
lasciarle la liberta' di ignorare cio' che non serve:
    `base`        `[r ; c]`            il contributo, invariato
    `[r, c, Wc]`  `[r ; c ; W c]`      entrambe le direzioni, grezza e mappata
    `[r, Wc]`     `[r ; W c]`          solo quella mappata
dove `W` e' la mappa lineare testo->visione dei tre domini (ridge, lambda=1000), stimata
**senza** i 10 attributi tenuti fuori (c'e' un assert a verificarlo).

Esito in `docs/piano-interpretabilita.md` sez. 2f: sui mai visti +0.26 / +0.17 / -0.40, segni
discordi, media zero. Alla rete e' stata offerta la direzione geometricamente migliore, quasi
perpendicolare alla grezza (coseno 0.023), e non l'ha usata.

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

Uso:    HF_HUB_OFFLINE=1 python esperimenti/mappa_dentro.py --seed 2026 --split 0
Output: esperimenti/mappa_dentro_split{split}.json
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
# LA MAPPA DEI TRE DOMINI DENTRO IL MODELLO COMPLETO
#
# Sappiamo (28/08) che: (a) la mappa lineare direzione-testuale -> direzione-visiva stimata su
# 227 attributi di tre domini generalizza ad attributi mai visti (allineamento 0.086 -> 0.295);
# (b) usarla DA SOLA nel retrieval perde fino a 15pp; (c) il 64% del guadagno del contributo
# viene dalla dipendenza dalla reference, che quella mappa non ha.
#
# Sintesi da provare: dare all'MLP ENTRAMBE le direzioni, la grezza e la corretta. Cosi' ha
# strettamente piu' informazione del modello attuale e puo' ignorare la corretta se inutile;
# e sugli attributi mai visti, dove la mappa generalizza, potrebbe avere un appiglio in piu'.
#
# La mappa e' stimata SENZA i 10 attributi tenuti fuori (assert a verificarlo).
# ==========================================================================================
FR = torch.load(ROOT / "esperimenti" / "frecce_tre_domini.pt", map_location="cpu")
Tf, Vf, DOMf, ATTf = FR["T"], FR["V"], FR["dom"], FR["attr"]
tenuti = {attr_names[i] for i in HOLD}
usa = [i for i in range(len(ATTf)) if not (DOMf[i] == "celeba" and ATTf[i] in tenuti)]
assert not (set(ATTf[i] for i in usa if DOMf[i] == "celeba") & tenuti), "fuga di attributi!"
LAM = 1000.0
W = torch.linalg.solve(Tf[usa].T @ Tf[usa] + LAM * torch.eye(D), Tf[usa].T @ Vf[usa])
dtn = d_ens / d_ens.norm(dim=-1, keepdim=True)
d_map = dtn @ W
d_map = d_map / d_map.norm(dim=-1, keepdim=True).clamp(min=1e-8)
d_map = d_map * d_ens.norm(dim=-1, keepdim=True)          # stessa scala delle grezze
print(f"mappa da {len(usa)} attributi; allineamento medio grezza<->mappata "
      f"{float((dtn * (d_map/d_map.norm(dim=-1,keepdim=True))).sum(-1).mean()):.3f}", flush=True)


def cond_due(entry, max_n=3):
    """Come cond_tensor, ma restituisce anche gli stessi slot con le direzioni MAPPATE."""
    u = torch.zeros(max_n, D); w = torch.zeros(max_n, D)
    m = torch.zeros(max_n, dtype=torch.bool)
    k = 0
    for a in entry["pos"][:max_n]:
        u[k], w[k] = d_ens[a], d_map[a]; m[k] = True; k += 1
    for b in entry["neg"][:max_n - k]:
        u[k], w[k] = -d_ens[b], -d_map[b]; m[k] = True; k += 1
    return u, w, m


EX = []
for e in train_bench:
    u, w, m = cond_due(e)
    for si, ref in enumerate(e["sources"].tolist()):
        EX.append((ref, u, w, m, torch.tensor(sorted(e["gt"][si]), dtype=torch.long)))


class Modello(nn.Module):
    """modo='base'      ingresso [r, c]          — il contributo attuale
       modo='piu_mappa' ingresso [r, c, Wc]      — piu' informazione
       modo='solo_mappa' ingresso [r, Wc]        — la direzione corretta al posto della grezza"""
    def __init__(self, modo):
        super().__init__()
        self.modo = modo
        n_in = 3 * D if modo == "piu_mappa" else 2 * D
        self.net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, U, UW, M):
        c = (U * M.unsqueeze(-1)).sum(1)
        cw = (UW * M.unsqueeze(-1)).sum(1)
        x = {"base": (r, c), "piu_mappa": (r, c, cw), "solo_mappa": (r, cw)}[self.modo]
        q = r + self.gamma * self.net(torch.cat(x, dim=-1))
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



def costruttore(model):
    cache = {}
    def f(ref, entry):
        k = entry["query"]
        if k not in cache: cache[k] = cond_due(entry)
        u, w, m = cache[k]
        with torch.no_grad():
            return model(ref, u.unsqueeze(0).expand(len(ref), -1, -1, ),
                         w.unsqueeze(0).expand(len(ref), -1, -1),
                         m.unsqueeze(0).expand(len(ref), -1))
    return f


def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)


def addestra(modo):
    torch.manual_seed(SEED)
    model = Modello(modo)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(EX), generator=gen)
        for st in range(0, len(EX), BATCH):
            idx = order[st:st + BATCH].tolist()
            refs = torch.tensor([EX[i][0] for i in idx])
            U = torch.stack([EX[i][1] for i in idx]); Wt = torch.stack([EX[i][2] for i in idx])
            M = torch.stack([EX[i][3] for i in idx])
            q = model(E_tr[refs], U, Wt, M)
            loss = info_nce_multipos(q, refs, [EX[i][4] for i in idx], args.tau)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


ps_b = evaluate(build_baseline, devset, E_valid, K_VALUES)
res = {"seed": SEED, "split": args.split, "holdout": sorted(tenuti), "varianti": {}}
print("\n" + "=" * 92)
print("LA MAPPA DENTRO IL MODELLO — Recall@10, differenza dalla baseline")
print("=" * 92)
print(f"{'variante':<22}{'strato':<13}{'baseline':>10}{'modello':>10}{'differenza':>13}{'p':>9}")
for modo in ("base", "piu_mappa", "solo_mappa"):
    m = addestra(modo)
    ps = evaluate(costruttore(m), devset, E_valid, K_VALUES)
    res["varianti"][modo] = {}
    for strato, idx in (("visti", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix)):
        if not idx: continue
        a_ = [ps[10][i] for i in idx]; b_ = [ps_b[10][i] for i in idx]
        st = bootstrap_paired(a_, b_, n_boot=20000, seed=0)
        res["varianti"][modo][strato] = {"delta_pp": st["delta"]*100, "p": st["p"],
                                         "modello": macro(a_)*100}
        print(f"{modo:<22}{strato:<13}{macro(b_)*100:>9.2f}%{macro(a_)*100:>9.2f}%"
              f"{st['delta']*100:>+12.2f}pp{st['p']:>9.4f}", flush=True)
    print(flush=True)

with open(ROOT / "esperimenti" / f"mappa_dentro_split{args.split}.json", "w") as f:
    json.dump(res, f, indent=2)
print("salvato.")
