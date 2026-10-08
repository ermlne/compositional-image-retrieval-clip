"""Pesare le condizioni fra loro con l'accordo fra i 10 template (challenge 1).

Ogni attributo ha 10 direzioni, una per template. La CONCENTRAZIONE
    R = || media delle 10 direzioni unitarie ||        (0 = sparse, 1 = tutte allineate)
misura quanto CLIP e' sicuro di un concetto **senza usare etichette**, quindi vale anche per
un attributo mai visto, a differenza dell'AUC.

Due passi:
 1. la premessa, verificata prima del resto: R correla con la qualita' vera della direzione
    (AUC sulle etichette)? Se non correla, il resto e' rumore;
 2. il retrieval con peso `w = R^p`, passo normalizzato, riportando a parte le query composte
    (su una query a UNA condizione la pesatura non puo' fare nulla: dopo la normalizzazione
    resta la stessa direzione — e' il controllo di correttezza incorporato).
Include un ORACOLO di confronto: pesi dall'AUC vera, che non e' un metodo utilizzabile ma da'
il tetto della famiglia.

Esito in `docs/piano-interpretabilita.md` sez. 2g: correlazione R<->AUC solo +0.299, retrieval
dentro il rumore, tetto della pesatura per condizione sotto 1pp.

--------------------------------------------------------------------------------------------
NOTA DI PROVENIENZA. Questo file nasce come **fork di
`contributo-candidato/attributi_heldout.py`** e ne conserva l'impalcatura: gli stessi 3 split
leave-attribute-out (`random.Random(--split).sample(attributi_del_devset, 10)`), la stessa
configurazione (tau 0.03, 30 epoche, max_sources 80, corpus di 20.000 immagini del train), i
tre strati di valutazione (visti / MAI VISTI / misti) e il bootstrap accoppiato su R@10.
NON riaddestra e non carica alcun modello: l'intervento e' sulle DIREZIONI
testuali della query baseline `q = normalize(r + c)`.
Fino al 30/08 l'intestazione era ancora quella di `attributi_heldout.py` ("GENERALIZZA AD
ATTRIBUTI MAI VISTI?"), copiata insieme al resto e mai aggiornata: descriveva un esperimento
diverso da quello che il file esegue. Sostituita.

Il benchmark ufficiale non viene toccato.

Uso:    HF_HUB_OFFLINE=1 python esperimenti/pesi_template.py --seed 2026 --split 0
Output: esperimenti/pesi_oracolo_split{split}.json
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



import json, math
# ==========================================================================================
# PESATURA DINAMICA PER CONDIZIONE, GUIDATA DALL'ACCORDO FRA I TEMPLATE
#
# La challenge 1 chiede che il modello "impari a pesare dinamicamente" le condizioni. I pesi
# APPRESI sono gia' stati provati e sono nulli (+0.1pp e +0.3pp, non significativi): non
# avevano niente a cui aggrapparsi. Qui il peso viene da un segnale misurabile.
#
# L'IDEA. Ogni attributo ha 10 direzioni, una per template. Se CLIP ha chiaro il concetto le
# 10 puntano nella stessa direzione; se e' confuso divergono. La CONCENTRAZIONE
#     R = || media delle 10 direzioni unitarie ||     (0 = sparse, 1 = tutte allineate)
# misura quella chiarezza SENZA usare etichette — quindi vale anche per un attributo mai visto,
# a differenza dell'AUC.
#
# LA PREMESSA DA VERIFICARE PRIMA: R deve correlare con la qualita' vera della direzione
# (l'AUC misurata sulle etichette). Se non correla, l'idea non ha base e il resto e' rumore.
#
# Il peso e' w = R^p; il passo e' normalizzato, quindi conta solo il rapporto fra i pesi.
# NB: su una query a UNA condizione la pesatura non puo' fare nulla (dopo la normalizzazione
# resta la stessa direzione). L'effetto puo' esistere solo sulle query composte, e infatti
# vengono riportate a parte.
# ==========================================================================================
print("\n" + "=" * 92)
print("PREMESSA — l'accordo fra i template predice la qualita' della direzione?")
print("=" * 92)

from transformers import CLIPModel, CLIPTokenizer
cm = CLIPModel.from_pretrained(MODEL_ID).eval()
tk = CLIPTokenizer.from_pretrained(MODEL_ID)

@torch.no_grad()
def enc(frasi):
    o = cm.get_text_features(**tk(frasi, padding=True, return_tensors="pt"))
    f = (o.pooler_output if hasattr(o, "pooler_output") else o).float()
    return f / f.norm(dim=-1, keepdim=True)

R = torch.zeros(N_ATTR)
for i, nome in enumerate(attr_names):
    c = nome.replace("_", " ").lower()
    P, Nn = enc([t.format(c) for t, _ in TEMPLATES]), enc([t.format(c) for _, t in TEMPLATES])
    dpt = P - Nn                                   # una direzione per template
    dpt = dpt / dpt.norm(dim=-1, keepdim=True)
    R[i] = dpt.mean(0).norm()                      # concentrazione: 0 sparse, 1 allineate
del cm, tk

# qualita' vera: AUC della direzione ensemble contro le etichette
g_ = torch.Generator().manual_seed(0)
sub = torch.randperm(len(E_tr), generator=g_)[:20000]
Es, Ls = E_tr[sub], L_tr[sub]
dtn = d_ens / d_ens.norm(dim=-1, keepdim=True)
AUC = torch.full((N_ATTR,), float("nan"))
for a in range(N_ATTR):
    y = Ls[:, a] == 1
    if y.sum() < 50 or (~y).sum() < 50: continue
    sc = Es @ dtn[a]
    r_ = sc.argsort().argsort().float() + 1
    n1, n0 = float(y.sum()), float((~y).sum())
    AUC[a] = (float(r_[y].sum()) - n1*(n1+1)/2) / (n1*n0)

ok = ~torch.isnan(AUC)
x, y = R[ok], AUC[ok]
rho = float(((x-x.mean())*(y-y.mean())).sum() / (x.std(unbiased=False)*y.std(unbiased=False)*len(x)))
print(f"correlazione fra concentrazione R e AUC: {rho:+.3f}  (su {int(ok.sum())} attributi)")
ordine = sorted(range(N_ATTR), key=lambda a: -R[a])
print("\n  piu' concentrati:            meno concentrati:")
for k in range(5):
    a, b = ordine[k], ordine[-1-k]
    print(f"  {attr_names[a]:<20}R={R[a]:.3f} AUC={AUC[a]:.3f}   "
          f"{attr_names[b]:<20}R={R[b]:.3f} AUC={AUC[b]:.3f}")
if rho < 0.15:
    print("\n  ATTENZIONE: correlazione debole. La premessa non regge, i numeri sotto")
    print("  vanno letti come un controllo negativo, non come un metodo.")

print("\n" + "=" * 92)
print("PESATURA w = R^p  (Recall@10, passo normalizzato)")
print("=" * 92)

# ORACOLO: pesi dall'AUC VERA, misurata sulle etichette. Non e' un metodo utilizzabile
# (per un attributo nuovo l'AUC non si puo' calcolare), ma dice qual e' il TETTO della
# pesatura per condizione. Se nemmeno l'oracolo aiuta, la strada e' chiusa per chiunque.
#   qualita' = max(AUC - 0.5, 0) * 2 -> 0 per una direzione inutile o invertita, 1 per perfetta
QUAL = torch.nan_to_num((AUC - 0.5).clamp(min=0.0) * 2, nan=0.0)
print(f"\noracolo: qualita' media {QUAL.mean():.3f}, "
      f"{int((QUAL == 0).sum())} attributi azzerati (AUC <= 0.5)")

def make_q(p, alpha, base=None):
    w = (R if base is None else base).clone() ** p
    w = w / w.mean().clamp(min=1e-8)
    cache = {}
    def f(ref, entry):
        k = entry["query"]
        if k not in cache:
            c = torch.zeros(D)
            for a in entry["pos"]: c = c + w[a] * dtn[a]
            for b in entry["neg"]: c = c - w[b] * dtn[b]
            cache[k] = c / c.norm().clamp(min=1e-8)
        q = ref + alpha * cache[k]
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f

ALPHAS = (1.0, 1.5, 2.0)
PS = (0.0, 1.0, 2.0, 4.0)
Rec = {(p, a): evaluate(make_q(p, a), devset, E_valid, K_VALUES)[10] for p in PS for a in ALPHAS}
for p in (1.0, 2.0, 4.0):
    for a in ALPHAS:
        Rec[(f"oracolo^{p:g}", a)] = evaluate(make_q(p, a, base=QUAL), devset,
                                              E_valid, K_VALUES)[10]
PS = list(PS) + [f"oracolo^{q:g}" for q in (1.0, 2.0, 4.0)]

n_cond = [len(e["queried"]) for e in devset]
strati = [("VISTI", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix),
          ("COMPOSTE (>=2 cond)", [i for i in range(len(devset)) if n_cond[i] >= 2]),
          ("semplici (1 cond)", [i for i in range(len(devset)) if n_cond[i] == 1])]
out = {}
for nome, idx in strati:
    if not idx: continue
    print(f"\n--- {nome} ({len(idx)} query)     differenza vs p=0 (pesi uguali)")
    print(f"  {'p':<6}" + "".join(f"{'a=' + str(a):>24}" for a in ALPHAS))
    for p in PS:
        parti = []
        for a in ALPHAS:
            v = macro([Rec[(p, a)][i] for i in idx]) * 100
            if p == 0.0:
                parti.append(f"{v:>23.2f}%")
            else:
                st = bootstrap_paired([Rec[(p,a)][i] for i in idx],
                                      [Rec[(0.0,a)][i] for i in idx], n_boot=20000, seed=0)
                parti.append(f"{v:>10.2f}% {st['delta']*100:>+7.2f}({st['p']:.3f})")
            out[f"{nome}|p{p}|a{a}"] = v
        print(f"  {str(p):<12}" + "".join(parti))

with open(ROOT / "esperimenti" / f"pesi_oracolo_split{args.split}.json", "w") as f:
    json.dump({"seed": SEED, "split": args.split, "rho_R_AUC": rho,
               "R": R.tolist(), "AUC": AUC.tolist(), "recall10": out}, f, indent=2)
print("\nsalvato.")
