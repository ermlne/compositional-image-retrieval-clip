"""Due rimedi al danno sugli attributi mai visti: il gate in-span e l'interpolazione sui vicini.

Addestra il modulo sullo split e poi prova, sullo stesso modello:
 1. **gate senza iperparametri** — si scala il delta per la quota della condizione che cade
    dentro lo span delle condizioni viste, `g(c) = ||P_span c|| / ||c||`. Si calcola senza
    etichette; se l'ipotesi regge il metodo degrada da solo verso la baseline sulle condizioni
    che non riconosce. Riporta anche la quota in-span per strato e il rango dello span;
 2. **interpolazione sui vicini semantici (kNN)** — si riscrive una condizione nuova come
    combinazione pesata (softmax con temperatura) dei k attributi VISTI piu' vicini nello
    spazio testuale, per non estrapolare mai. Si sostituisce la CONDIZIONE, non il delta, con
    un controllo che isola l'effetto del formato dell'ingresso da quello della proiezione.

Esito in `docs/piano-interpretabilita.md` sez. 2a: media -0.36pp, segno discorde, nessun
effetto.

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

Uso:    HF_HUB_OFFLINE=1 python esperimenti/heldout_knn.py --seed 2026 --split 0
Output: esperimenti/multiseed/knn_seed{SEED}_split{split}.json  (vedi README: i file
        versionati stanno in `esperimenti/`, non nella sottocartella)
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



# ===========================================================================================
# INTERPOLAZIONE FRA ATTRIBUTI VICINI (Arianna) — versione corretta
#
# La mappa globale fallisce sugli attributi mai visti perche' ESTRAPOLA: la condizione `c`
# cade in una regione dello spazio dove la mappa non e' mai stata vincolata. L'idea e'
# PROIETTARE la condizione nuova dentro il guscio delle condizioni note, prima di darla al
# modello: si cercano i k attributi VISTI piu' vicini nello spazio testuale e si sostituisce
# `c` con la loro combinazione pesata.
#
# Il dato che la motiva: le direzioni testuali sono raggruppate per significato (coseno medio
# 0.299 fra attributi diversi contro 0.044 atteso per vettori casuali; Arched/Bushy_Eyebrows
# 0.823, Goatee/Mustache 0.602).
#
# NB: si sostituisce la CONDIZIONE, non il delta. Sostituire il delta (mediandolo su molte
# reference) butterebbe via la dipendenza da `r`, che la caratterizzazione ha mostrato essere
# parte della mappa: si misurerebbe l'errore dell'approssimazione, non il trasferimento.
# Il CONTROLLO qui sotto isola l'effetto del formato dell'input dall'effetto della proiezione.
# ===========================================================================================
print("\n" + "=" * 92)
print("INTERPOLAZIONE FRA ATTRIBUTI VICINI — la correzione si trasferisce ai vicini semantici?")
print("=" * 92)

SEEN_IDX = sorted(i for i in range(N_ATTR) if i not in HOLD)
d_unit = d_ens / d_ens.norm(dim=-1, keepdim=True)
D_seen = d_unit[SEEN_IDX]

def proietta(c, k, temp):
    """Riscrive una condizione come combinazione pesata dei k attributi VISTI piu' vicini,
    conservandone la norma."""
    n = c.norm().clamp(min=1e-8)
    cn = c / n
    sim = D_seen @ cn
    top = sim.abs().topk(min(k, len(SEEN_IDX)))
    w = torch.softmax(top.values / temp, dim=0) * torch.sign(sim[top.indices])
    v = (w.unsqueeze(-1) * D_seen[top.indices]).sum(0)
    return v / v.norm().clamp(min=1e-8) * n

def un_solo_slot(c):
    u = torch.zeros(3, D); m = torch.zeros(3, dtype=torch.bool)
    u[0] = c; m[0] = True
    return u, m

def make_proj(k, temp):
    cache = {}
    def f(ref, entry):
        key = entry["query"]
        if key not in cache:
            cache[key] = un_solo_slot(proietta(cond_of(entry), k, temp))
        u, m = cache[key]
        with torch.no_grad():
            return model(ref, u.unsqueeze(0).expand(len(ref), -1, -1),
                         m.unsqueeze(0).expand(len(ref), -1))
    return f

# CONTROLLO: la condizione gia' sommata, passata in un solo slot invece che spezzata in tre.
# Se questo da' gia' un calo, il calo viene dal formato dell'input, non dalla proiezione.
def build_collassato(ref, entry):
    u, m = un_solo_slot(cond_of(entry))
    with torch.no_grad():
        return model(ref, u.unsqueeze(0).expand(len(ref), -1, -1),
                     m.unsqueeze(0).expand(len(ref), -1))

res["knn"] = {}
print(f"{'variante':<22}{'strato':<13}{'baseline':>10}{'metodo':>9}{'differenza':>13}{'p':>9}")

def riga(nome_var, ps_x):
    res["knn"][nome_var] = {}
    for nome, idx in (("visti", idx_seen), ("MAI VISTI", idx_hold)):
        a_ = [ps_x[10][i] for i in idx]; b_ = [ps_b[10][i] for i in idx]
        st = bootstrap_paired(a_, b_, n_boot=20000, seed=0)
        res["knn"][nome_var][nome] = {"delta_pp": st["delta"]*100, "p": st["p"],
                                      "metodo": macro(a_)*100}
        print(f"{nome_var:<22}{nome:<13}{macro(b_)*100:>9.2f}%{macro(a_)*100:>8.2f}%"
              f"{st['delta']*100:>+12.2f}pp{st['p']:>9.4f}")
    print()

riga("CONTROLLO 1 slot", evaluate(build_collassato, devset, E_valid, K_VALUES))
for k, temp in ((1, 0.1), (3, 0.1), (5, 0.1), (5, 0.05), (10, 0.1)):
    riga(f"proiez. k={k} T={temp}", evaluate(make_proj(k, temp), devset, E_valid, K_VALUES))

with open(OUT / f"knn_seed{SEED}_split{args.split}.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"\nsalvato in multiseed/heldout_seed{SEED}_split{args.split}.json "
      f"({(time.time()-t0)/60:.1f} minuti)")
