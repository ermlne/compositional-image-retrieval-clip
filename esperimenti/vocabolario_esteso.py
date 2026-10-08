"""Dare gradiente su molte piu' direzioni riduce il danno sugli attributi mai visti?

Compito ausiliario a **pseudo-etichette**: 600 parole del dizionario di sistema + 85 direzioni
AwA2 + 117 OVAD (787 in tutto) diventano condizioni aggiuntive, con le etichette fornite da
CLIP stesso (decile alto/basso del punteggio zero-shot sulle immagini di CelebA). Corpus e
compito restano quelli veri; cambia solo quante direzioni ricevono gradiente. Peso `lambda`.

Due controlli di disegno inclusi: (a) filtro sulle direzioni lungo cui il corpus varia davvero
(std sopra la mediana), validato contro l'AUC vera sui 40 attributi CelebA; (b) esclusione
delle direzioni con coseno > 0.60 verso un attributo tenuto fuori, per evitare la fuga di
supervisione (regime "stretto" con esclusione, "permissivo" senza).

Esito in `docs/piano-interpretabilita.md` sez. 6d: e' una manopola, non un guadagno
(+1.28pp sui mai visti, -1.05pp sui visti, netto -0.29pp).

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

Uso:    HF_HUB_OFFLINE=1 python esperimenti/vocabolario_esteso.py --seed 2026 --split 0
Output: esperimenti/vocabolario_esteso_split{split}.json
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
# VOCABOLARIO ESTESO CON PSEUDO-ETICHETTE — verso l'aritmetica su direzioni qualunque
#
# IL PROBLEMA, misurato: 30 attributi visti generano un sottospazio di dimensione 30 su 512.
# La mappa e' vincolata solo li'; sulle altre 482 dimensioni contiene quello che ci ha lasciato
# l'inizializzazione. Una direzione mai vista ci cade dentro per circa META' (copertura 0.40-0.51
# sui tre split), e quella meta' produce rumore applicato con sicurezza. Da cui -5pp invece di 0.
# Una query NUOVA fatta di attributi NOTI ha invece copertura 1.0000 esatta — e infatti funziona.
#
# L'IDEA: dare gradiente alla mappa su molte piu' direzioni, SENZA nuove annotazioni e SENZA
# cambiare corpus. Le direzioni testuali costano due encoding; le etichette le fornisce CLIP
# stesso (zero-shot) sulle immagini di CelebA. Il compito resta uno solo: retrieval su CelebA.
#
# PERCHE' E' DIVERSO DAL MULTI-DOMINIO: li' le direzioni OVAD erano vincolate usando IMMAGINI DI
# OGGETTI, quindi la mappa imparava "cosa fare con la direzione striped quando la reference e'
# una sedia". Qui ogni direzione e' vincolata con reference che sono VOLTI — il contesto in cui
# poi va usata. Il vincolo c'era gia', ma nel posto sbagliato.
#
# DUE REGIMI, entrambi riportati:
#   STRETTO    esclude le direzioni troppo simili a quelle tenute fuori (coseno > SOGLIA_FUGA):
#              misura l'estrapolazione vera. Un dizionario contiene "smiling": senza questo
#              filtro si contrabbanderebbe supervisione sull'attributo tenuto fuori.
#   PERMISSIVO nessuna esclusione: misura lo scenario realistico, dove un vocabolario ampio
#              copre anche concetti che nessuno ha annotato.
# ==========================================================================================
import json, random as _rnd

SOGLIA_FUGA = 0.60
N_PAROLE = 600
Q_PSEUDO = 0.10          # decili estremi come pseudo-positivi/negativi
K_VICINI = 40            # quanti pseudo-positivi, fra i piu' vicini alla reference
N_SRC_PSEUDO = 20        # reference per direzione

TEMPL = [("a photo of a person who is {}", "a photo of a person who is not {}"),
         ("a photo of a person with {}",   "a photo of a person without {}"),
         ("a face with {}",                "a face without {}"),
         ("{}",                            "not {}")]

from transformers import CLIPModel, CLIPTokenizer
_m = CLIPModel.from_pretrained(MODEL_ID).eval()
_t = CLIPTokenizer.from_pretrained(MODEL_ID)

@torch.no_grad()
def _enc(fr):
    o = _m.get_text_features(**_t(fr, padding=True, return_tensors="pt"))
    f = (o.pooler_output if hasattr(o, "pooler_output") else o).float()
    return f / f.norm(dim=-1, keepdim=True)

# --- le direzioni extra: dizionario + AwA2 + OVAD ------------------------------------------
parole = [w.strip().lower() for w in open("/usr/share/dict/american-english")
          if w.strip().isalpha() and 4 <= len(w.strip()) <= 12 and w.strip().islower()]
_rnd.Random(0).shuffle(parole); parole = parole[:N_PAROLE]
EXTRA, ETI = [], []
for i in range(0, len(parole), 100):
    lot = parole[i:i+100]
    P = torch.stack([_enc([t.format(w) for t, _ in TEMPL]).mean(0) for w in lot])
    Nn = torch.stack([_enc([t.format(w) for _, t in TEMPL]).mean(0) for w in lot])
    v = P - Nn
    EXTRA.append(v / v.norm(dim=-1, keepdim=True)); ETI += [f"dict:{w}" for w in lot]
del _m, _t
EXTRA = torch.cat(EXTRA)
FR = torch.load(ROOT / "esperimenti" / "frecce_tre_domini.pt", map_location="cpu")
for i, dm in enumerate(FR["dom"]):
    if dm != "celeba":
        EXTRA = torch.cat([EXTRA, FR["T"][i:i+1]]); ETI.append(f"{dm}:{FR['attr'][i]}")
print(f"direzioni extra disponibili: {len(EXTRA)} "
      f"({N_PAROLE} dizionario + {len(EXTRA)-N_PAROLE} AwA2/OVAD)", flush=True)

dtn = d_ens / d_ens.norm(dim=-1, keepdim=True)
sim_hold = (EXTRA @ dtn[sorted(HOLD)].T).abs().max(dim=1).values
print(f"somiglianza massima con un attributo TENUTO FUORI: mediana {sim_hold.median():.3f}, "
      f"max {sim_hold.max():.3f}", flush=True)

# --- pseudo-etichette da CLIP sulle immagini di CelebA -------------------------------------
def pseudo(dirs):
    """positivi = decile alto del punteggio zero-shot, negativi = decile basso.
    Si tengono solo le direzioni lungo cui il corpus VARIA davvero (std sopra la mediana):
    una direzione ortogonale alla variazione delle immagini non separa nulla."""
    S = E_tr @ dirs.T                                    # (n_img, n_dir)
    sd = S.std(dim=0)
    tieni = sd >= sd.median()
    k = max(int(Q_PSEUDO * len(E_tr)), 50)
    alto = S.topk(k, dim=0).indices
    basso = (-S).topk(k, dim=0).indices
    return tieni, alto, basso

print("\n" + "=" * 92)
print("VOCABOLARIO ESTESO — Recall@10 sui tre strati")
print("=" * 92)
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

EX = []
for e in train_bench:
    u, m = cond_tensor(e)
    for si, ref in enumerate(e["sources"].tolist()):
        EX.append((ref, u, m, torch.tensor(sorted(e["gt"][si]), dtype=torch.long)))
print(f"esempi reali (CelebA, etichette vere): {len(EX)}", flush=True)

def esempi_pseudo(stretto):
    idx = [i for i in range(len(EXTRA)) if not (stretto and sim_hold[i] > SOGLIA_FUGA)]
    dirs = EXTRA[idx]
    tieni, alto, basso = pseudo(dirs)
    usabili = [j for j in range(len(idx)) if bool(tieni[j])]
    fuori = []
    for j in usabili:
        pos = alto[:, j]                                   # pseudo-positivi
        src = basso[:, j].tolist()                         # reference: violano la condizione
        _rnd.Random(SEED + j).shuffle(src)
        u = torch.zeros(3, D); m = torch.zeros(3, dtype=torch.bool)
        u[0] = dirs[j]; m[0] = True
        Epos = E_tr[pos]
        for r in src[:N_SRC_PSEUDO]:
            # Il ground truth NON e' tutto il decile alto: sarebbe un compito degenere,
            # risolvibile con q = d ignorando la reference — e insegnerebbe al modello
            # esattamente il contrario di cio' che serve. Come nel compito vero, il risultato
            # deve soddisfare la condizione E somigliare alla reference: si tengono quindi
            # solo i K pseudo-positivi piu' vicini alla reference.
            vicini = (Epos @ E_tr[int(r)]).topk(K_VICINI).indices
            fuori.append((int(r), u, m, pos[vicini].sort().values))
    return fuori, len(idx), len(usabili)

def addestra(EX_AUX, lam):
    torch.manual_seed(SEED)
    model = Concat()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(EX), generator=gen)
        oa = torch.randperm(len(EX_AUX), generator=gen) if (lam > 0 and EX_AUX) else None
        for bi, st in enumerate(range(0, len(EX), BATCH)):
            ix = order[st:st+BATCH].tolist()
            refs = torch.tensor([EX[i][0] for i in ix])
            U = torch.stack([EX[i][1] for i in ix]); M = torch.stack([EX[i][2] for i in ix])
            loss = info_nce_multipos(model(E_tr[refs], U, M), refs,
                                     [EX[i][3] for i in ix], args.tau)
            if lam > 0 and EX_AUX:
                j = (bi * BATCH) % max(len(EX_AUX) - BATCH, 1)
                ia = oa[j:j+BATCH].tolist()
                r2 = torch.tensor([EX_AUX[i][0] for i in ia])
                U2 = torch.stack([EX_AUX[i][1] for i in ia])
                M2 = torch.stack([EX_AUX[i][2] for i in ia])
                loss = loss + lam * info_nce_multipos(model(E_tr[r2], U2, M2), r2,
                                                      [EX_AUX[i][3] for i in ia], args.tau)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

def copertura(base_dirs):
    U_, S_, _ = torch.linalg.svd(base_dirs.T, full_matrices=False)
    Q = U_[:, :int((S_ > 1e-5*S_[0]).sum())]
    return float(((dtn[sorted(HOLD)] @ Q) ** 2).sum(-1).mean())

ps_b = evaluate(build_baseline, devset, E_valid, K_VALUES)
SEEN_IDX = [i for i in range(N_ATTR) if i not in HOLD]
res = {"seed": SEED, "split": args.split, "holdout": sorted(attr_names[i] for i in HOLD),
       "copertura_base": copertura(dtn[SEEN_IDX]), "regimi": {}}
print(f"copertura degli attributi tenuti fuori, con i soli 30 visti: "
      f"{res['copertura_base']:.4f}\n", flush=True)

print(f"{'regime':<22}{'strato':<13}{'baseline':>10}{'modello':>10}{'differenza':>13}{'p':>9}")
CACHE_AUX = {}
for eti, stretto, lam in (("solo CelebA (base)", True, 0.0),
                          ("STRETTO lam=0.1", True, 0.1),
                          ("STRETTO lam=0.25", True, 0.25),
                          ("STRETTO lam=1.0", True, 1.0),
                          ("PERMISSIVO lam=0.25", False, 0.25)):
    if lam == 0:
        AUX, n_dir, n_us = [], 0, 0
    else:
        if stretto not in CACHE_AUX: CACHE_AUX[stretto] = esempi_pseudo(stretto)
        AUX, n_dir, n_us = CACHE_AUX[stretto]
    if lam > 0:
        idx = [i for i in range(len(EXTRA)) if not (stretto and sim_hold[i] > SOGLIA_FUGA)]
        cop = copertura(torch.cat([dtn[SEEN_IDX], EXTRA[idx]]))
        print(f"  [{eti}] {n_dir} direzioni, {n_us} superano il filtro, "
              f"{len(AUX)} esempi; copertura {cop:.4f}", flush=True)
        res["regimi"][eti] = {"n_dir": n_dir, "n_usabili": n_us, "copertura": cop}
    m = addestra(AUX, lam)
    ps = evaluate(make_builder(m), devset, E_valid, K_VALUES)
    res["regimi"].setdefault(eti, {})
    for strato, ix in (("visti", idx_seen), ("MAI VISTI", idx_hold), ("miste", idx_mix)):
        if not ix: continue
        a_ = [ps[10][i] for i in ix]; b_ = [ps_b[10][i] for i in ix]
        st = bootstrap_paired(a_, b_, n_boot=20000, seed=0)
        res["regimi"][eti][strato] = {"delta_pp": st["delta"]*100, "p": st["p"],
                                      "modello": macro(a_)*100}
        print(f"{eti:<22}{strato:<13}{macro(b_)*100:>9.2f}%{macro(a_)*100:>9.2f}%"
              f"{st['delta']*100:>+12.2f}pp{st['p']:>9.4f}", flush=True)
    print(flush=True)

with open(ROOT / "esperimenti" / f"vocabolario_esteso_split{args.split}.json", "w") as f:
    json.dump(res, f, indent=2)
print("salvato.")
