"""Addestrare lo stesso MLP anche su un altro dominio (OVAD) lo rende piu' generale?

Aggiunge un compito ausiliario di retrieval su OVAD con peso `LAMBDA`, e confronta lambda=0
contro lambda=1 sul dev set CelebA. Previsione falsificabile dichiarata prima: se l'MLP impara
geometria generale di CLIP il multi-dominio deve aiutare soprattutto sugli attributi mai
visti ed essere neutro sui visti; se impara cose specifiche dei volti si vedra' un calo sui
visti senza guadagno sui nuovi.

Scelte di progetto dichiarate: AwA2 e' escluso (etichette per CLASSE -> Hamming zero dentro
una classe, il compito degenera in riconoscimento di specie); su OVAD il ground truth e'
definito solo sugli attributi interrogati (il 14% di etichette -1 rende Hamming mal definita);
le direzioni testuali di OVAD sono quelle gia' calcolate in `esperimenti/frecce_tre_domini.pt`.

Salva i checkpoint `contributo-candidato/checkpoints/multidominio_seed{SEED}_lam{lam}.pt`, che
`ensemble_seed.py` e `ovad_eval.py` poi rileggono. Esito in
`docs/piano-interpretabilita.md` sez. 6c: sul guadagno CelebA e' nullo (+0.07pp su 5 seed).

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

Uso:    HF_HUB_OFFLINE=1 python esperimenti/multidominio.py --seed 2026 --split 0
Output: esperimenti/multidominio_seed{SEED}_split{split}.json
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
from src.benchmark import (load_benchmark, build_benchmark_from_labels,
                           parse_query_string)
from src.devset import bootstrap_paired, macro, sample_queries

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--split", type=int, default=0, help="quale split di attributi (0,1,2)")
ap.add_argument("--n-holdout", type=int, default=10)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lambdas", default="0,0.25,0.5,1")
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
# MULTI-DOMINIO: lo stesso MLP addestrato a fare retrieval anche su OVAD (Arianna)
#
# ARGOMENTO. Il 64% del guadagno del contributo viene dalla dipendenza dalla foto di partenza
# (ablazione del 28/08: +7.35pp -> +2.66pp). Quella parte oggi impara SOLO da volti, che in
# CLIP sono una fetta strettissima dello spazio (coseno medio 0.47 fra direzioni visive di
# attributi diversi: tutto ammassato in una regione minima). Contro la memorizzazione — il
# nostro modo di fallire diagnosticato dall'esperimento LayerNorm — la contromisura standard
# non e' un vincolo geometrico ma PIU' DATI DIVERSI. Sette rimedi sulle sole direzioni sono
# falliti; questo e' il primo che tocca l'altra parte.
#
# PREVISIONE FALSIFICABILE. Se l'MLP impara geometria generale di CLIP, il multi-dominio deve
# aiutare SOPRATTUTTO sugli attributi mai visti (dove oggi memorizza) ed essere neutro sui
# visti. Se impara cose specifiche dei volti, si vedra' un calo sui visti senza guadagno sui
# nuovi. I due esiti sono distinguibili.
#
# SCELTE DI PROGETTO, dichiarate:
#  - AwA2 e' escluso: etichette per CLASSE significa Hamming zero dentro una classe, quindi
#    ogni zebra sarebbe ground truth di ogni altra e il compito degenera in riconoscimento di
#    specie.
#  - Su OVAD il ground truth e' definito SOLO sugli attributi interrogati, senza il vincolo di
#    somiglianza sul resto: il 14% di etichette -1 (sconosciute) rende la distanza di Hamming
#    mal definita. Un compito ausiliario non deve essere identico a quello bersaglio, deve
#    avere la stessa FORMA (immagine + direzione testuale -> trova le immagini che
#    corrispondono).
#  - Le direzioni testuali di OVAD sono quelle gia' calcolate con i prompt corretti
#    (esperimenti/frecce_tre_domini.pt), non ricodificate.
#
# RISCHIO: transfer negativo. Si controlla col peso LAMBDA sul compito ausiliario.
# ==========================================================================================
import json

OV = torch.load(ROOT / "embeddings" / "ovad_clip_vitb32.pt", map_location="cpu")
E_ov = OV["embeddings"].float(); E_ov = E_ov / E_ov.norm(dim=-1, keepdim=True)
L_ov = OV["labels"]
nomi_ov = list(OV["attr_names"])

FR = torch.load(ROOT / "esperimenti" / "frecce_tre_domini.pt", map_location="cpu")
pos_ov = {a: i for i, a in enumerate(FR["attr"]) if FR["dom"][i] == "ovad"}
d_ov = torch.zeros(len(nomi_ov), D)
usabili_ov = []
for j, n in enumerate(nomi_ov):
    if n in pos_ov:
        d_ov[j] = FR["T"][pos_ov[n]]
        noti = L_ov[:, j] != -1
        if int(((L_ov[:, j] == 1) & noti).sum()) >= 50 and int(((L_ov[:, j] == 0) & noti).sum()) >= 50:
            usabili_ov.append(j)
print(f"OVAD: {len(E_ov)} oggetti, {len(usabili_ov)} attributi usabili su {len(nomi_ov)}",
      flush=True)


idx_ov = {n: i for i, n in enumerate(nomi_ov)}
nomi_usabili_ov = [nomi_ov[j] for j in usabili_ov]
assert not any(',' in n for n in nomi_usabili_ov), 'una virgola nel nome romperebbe il parsing'


def bench_ovad(n_query=150, max_src=40, seed=SEED):
    """Ground truth: soddisfa le condizioni interrogate, con etichette NOTE su quelle.
    Reference: viola OGNI condizione (stessa regola di CelebA), etichette note."""
    qs = []
    for n_cond, quanti in {1: 60, 2: 60, 3: 30}.items():
        qs += sample_queries(nomi_usabili_ov, quanti, n_cond, exclude=qs, seed=seed + n_cond)
    rng = torch.Generator().manual_seed(seed)
    fuori = []
    for q in qs:
        pos, neg = parse_query_string(q, idx_ov)
        att = list(pos) + list(neg)
        noti = torch.ones(len(E_ov), dtype=torch.bool)
        for a in att: noti &= (L_ov[:, a] != -1)
        ok = noti.clone(); viola = noti.clone()
        for a in pos:
            ok &= (L_ov[:, a] == 1); viola &= (L_ov[:, a] == 0)
        for b in neg:
            ok &= (L_ov[:, b] == 0); viola &= (L_ov[:, b] == 1)
        gt = torch.nonzero(ok).flatten()
        src = torch.nonzero(viola).flatten()
        if len(gt) < 5 or len(src) < 5: continue
        if len(src) > max_src:
            src = src[torch.randperm(len(src), generator=rng)[:max_src]]
        u = torch.zeros(3, D); m = torch.zeros(3, dtype=torch.bool); k = 0
        for a in list(pos)[:3]: u[k] = d_ov[a]; m[k] = True; k += 1
        for b in list(neg)[:3-k]: u[k] = -d_ov[b]; m[k] = True; k += 1
        fuori.append({"src": src, "gt": set(gt.tolist()), "u": u, "m": m})
    return fuori


OVB = bench_ovad()
print(f"compito ausiliario OVAD: {len(OVB)} query valide, "
      f"{sum(len(e['src']) for e in OVB)} coppie", flush=True)
EX_OV = [(int(r), e["u"], e["m"], torch.tensor(sorted(e["gt"] - {int(r)}), dtype=torch.long))
         for e in OVB for r in e["src"].tolist()]
EX_OV = [x for x in EX_OV if len(x[3]) > 0]
print(f"  -> {len(EX_OV)} esempi di addestramento ausiliari", flush=True)

EX = []
for e in train_bench:
    u, m = cond_tensor(e)
    for si, ref in enumerate(e["sources"].tolist()):
        EX.append((ref, u, m, torch.tensor(sorted(e["gt"][si]), dtype=torch.long)))


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



def nce_su(q, refs, gts, corpus, tau):
    sim = (q @ corpus.T) / tau
    B, N = sim.shape
    sim[torch.arange(B), refs] = -torch.inf
    logden = torch.logsumexp(sim, dim=-1)
    perdite = []
    for i, gt in enumerate(gts):
        if len(gt) == 0: continue
        perdite.append(-(sim[i, gt] - logden[i]).mean())
    return torch.stack(perdite).mean() if perdite else q.sum() * 0.0


def make_builder(model):
    cache = {}
    def f(ref, entry):
        if entry["query"] not in cache: cache[entry["query"]] = cond_tensor(entry)
        u, m = cache[entry["query"]]
        with torch.no_grad():
            return model(ref, u.unsqueeze(0).expand(len(ref), -1, -1),
                         m.unsqueeze(0).expand(len(ref), -1))
    return f


def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)


def addestra(lam):
    torch.manual_seed(SEED)
    model = Concat()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(EX), generator=gen)
        ord_ov = torch.randperm(len(EX_OV), generator=gen) if lam > 0 else None
        for bi, st in enumerate(range(0, len(EX), BATCH)):
            idx = order[st:st+BATCH].tolist()
            refs = torch.tensor([EX[i][0] for i in idx])
            U = torch.stack([EX[i][1] for i in idx]); M = torch.stack([EX[i][2] for i in idx])
            loss = info_nce_multipos(model(E_tr[refs], U, M), refs,
                                     [EX[i][3] for i in idx], args.tau)
            if lam > 0:                                   # un batch ausiliario per ogni batch
                j = (bi * BATCH) % max(len(EX_OV) - BATCH, 1)
                io_ = ord_ov[j:j+BATCH].tolist()
                r2 = torch.tensor([EX_OV[i][0] for i in io_])
                U2 = torch.stack([EX_OV[i][1] for i in io_])
                M2 = torch.stack([EX_OV[i][2] for i in io_])
                loss = loss + lam * nce_su(model(E_ov[r2], U2, M2), r2,
                                           [EX_OV[i][3] for i in io_], E_ov, args.tau)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


ps_b = evaluate(build_baseline, devset, E_valid, K_VALUES)
res = {"seed": SEED, "split": args.split, "holdout": sorted(attr_names[i] for i in HOLD),
       "n_aux": len(EX_OV), "lam": {}}
print("\n" + "=" * 92)
print("MULTI-DOMINIO — Recall@10, differenza dalla baseline")
print("=" * 92)
print(f"{'lambda':<12}{'strato':<13}{'baseline':>10}{'modello':>10}{'differenza':>13}{'p':>9}")
PS_LAM = {}
for lam in [float(x) for x in args.lambdas.split(",")]:
    m = addestra(lam)
    ps = evaluate(make_builder(m), devset, E_valid, K_VALUES)
    PS_LAM[lam] = ps
    # i pesi servono per ripetere la caratterizzazione (linearita', quota della reference)
    CK = ROOT / "contributo-candidato" / "checkpoints"
    CK.mkdir(parents=True, exist_ok=True)
    torch.save({"A": m.state_dict(), "seed": SEED, "lambda": lam, "n_aux": len(EX_OV),
                "config": {"hidden": HIDDEN, "d": D, "n_attr": N_ATTR},
                "dirs_text": d_ens, "attr_names": list(attr_names),
                "holdout": sorted(attr_names[i] for i in HOLD)},
               CK / f"multidominio_seed{SEED}_lam{lam:g}.pt")
    res["lam"][str(lam)] = {}
    for strato, idx in (("visti", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix)):
        if not idx: continue
        a_ = [ps[10][i] for i in idx]; b_ = [ps_b[10][i] for i in idx]
        st = bootstrap_paired(a_, b_, n_boot=20000, seed=0)
        res["lam"][str(lam)][strato] = {"delta_pp": st["delta"]*100, "p": st["p"],
                                        "modello": macro(a_)*100}
        eti = f"{lam:g}" + (" (= base)" if lam == 0 else "")
        print(f"{eti:<12}{strato:<13}{macro(b_)*100:>9.2f}%{macro(a_)*100:>9.2f}%"
              f"{st['delta']*100:>+12.2f}pp{st['p']:>9.4f}", flush=True)
    print(flush=True)

print("\n" + "=" * 92)
print("CONFRONTO APPAIATO CONTRO lambda=0 (e' questo il numero che conta)")
print("=" * 92)
res["appaiato"] = {}
if 0.0 in PS_LAM:
    for lam, ps in PS_LAM.items():
        if lam == 0.0: continue
        for strato, idx in (("visti", idx_seen), ("MAI VISTI", idx_hold)):
            if not idx: continue
            a_ = [ps[10][i] for i in idx]; b_ = [PS_LAM[0.0][10][i] for i in idx]
            st = bootstrap_paired(a_, b_, n_boot=20000, seed=0)
            res["appaiato"][f"lam{lam}|{strato}"] = {"delta_pp": st["delta"]*100, "p": st["p"]}
            print(f"  lambda={lam:<6}{strato:<12}{macro(b_)*100:>7.2f}% -> {macro(a_)*100:>7.2f}%"
                  f"   {st['delta']*100:>+6.2f}pp   p={st['p']:.4f}", flush=True)

with open(ROOT / "esperimenti" / f"multidominio_seed{SEED}_split{args.split}.json", "w") as f:
    json.dump(res, f, indent=2)
print("salvato.")
