"""PIU' QUERY o PIU' ESEMPI? — l'esperimento che separa le due spiegazioni del bagging.

===========================================================================================
DA DOVE NASCE
===========================================================================================
Il bagging di Arianna (4 modelli su campioni di query diversi, uscite mediate) porta il dev
set da 26.76% a 30.53%. L'ensemble di soli seed (stessi dati, inizializzazione diversa) da'
+0.07 / +0.49pp. Quindi il guadagno viene dalle QUERY, non dai pesi iniziali.

Ma "dalle query" ammette due letture diverse, e prescrivono cose opposte:

  H1  COPERTURA DI DIREZIONI. Quattro modelli vedono 4x le direzioni `c` distinte. Se e'
      questo, conviene comprare direzioni al minor costo, cioe' query a molte condizioni.

  H2  MEMORIZZAZIONE. 4407 esempi da 182 query distinte contro 787k parametri, con 67.84%
      sul train e 27.26% sul dev. Piu' esempi e piu' query distinte impediscono al modello
      di imparare una risposta per query. Se e' questo, servono ESEMPI, e le query a 2
      condizioni sono la fonte efficiente.

Una misura fatta ieri esclude una terza lettura che sembrava ovvia: `c` vive nello span
delle 40 direzioni testuali (dimensione <= 40, rango effettivo ~20 su 512) e l'addestramento
usa gia' 220 direzioni distinte li' dentro. Sullo span `W_c` e' SOVRA-determinata: per
fissarla ne bastano ~40 indipendenti. Quindi il guadagno NON puo' essere identificabilita'.

===========================================================================================
IL DISEGNO — tre bracci, un solo seed, tutto il resto identico
===========================================================================================
  0  ATTUALE          {1:80, 2:80, 3:60}, max_sources 80    ~182 query,  ~4400 esempi
  A  PIU' ESEMPI      {1:80, 2:80, 3:60}, max_sources alto   ~182 query,  ~9000 esempi
  B  PIU' QUERY       {1:80, 2:320, 3:240}, max_sources 80   ~473 query,  ~9000 esempi

A e B hanno lo STESSO BUDGET DI ESEMPI e differiscono solo nel numero di query distinte.

  B > A   -> conta la diversita' di query in se' (H1, o H2 nella sua forma "per query")
  B ~ A   -> conta solo la massa di esempi (H2 puro), e la si compra molto piu' a buon
             mercato alzando max_sources, senza campionare niente di nuovo
  A ~ B ~ 0 -> nessuna delle due, e il bagging fa qualcos'altro che non abbiamo capito

PRE-REGISTRAZIONE, prima di guardare i numeri:
  primaria      B contro 0   (piu' query aiuta?)
  meccanismo    B contro A   (e' la diversita' o la massa?)
  controllo     A contro 0   (la massa da sola aiuta?)
  bootstrap gerarchico accoppiato sul set di sviluppo a 101 query, Bonferroni su 3 test.

CONFONDIMENTO DA DICHIARARE. Lo spazio delle query a 1 condizione e' ESAURITO (80 su 80 gia'
usate), quindi non si puo' aumentare i dati mantenendo la composizione del benchmark
(53/32/15%): il braccio B e' 12/50/38%, spostato verso le composte. "Piu' dati" e "piu'
composte" restano legati. Per questo lo script stratifica sempre per numero di condizioni.

Fedelta': pipeline copiata verbatim da `contributo-candidato/multiseed.py` (stessi corpus,
direzioni, loss, iperparametri, valutazione). Si addestra solo il modello A, la
cross-attention non serve. Il seed 1 con la configurazione ATTUALE deve riprodurre 26.53%.

Il benchmark ufficiale NON viene toccato.

Uso:
    HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 python esperimenti/braccio_b_piu_query.py --seed 1
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import build_benchmark_from_labels, load_benchmark
from src.devset import bootstrap_paired, macro, sample_queries

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--tau", type=float, default=0.03)
ap.add_argument("--threads", type=int, default=6)
args = ap.parse_args()
torch.set_num_threads(args.threads)


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
LOG_PATH = HERE / f"braccio_b_seed{args.seed}_{STAMP}.txt"
JSON_PATH = HERE / f"braccio_b_seed{args.seed}_{STAMP}.json"
_tee = Tee(LOG_PATH)
sys.stdout = _tee

MODEL_ID = "openai/clip-vit-base-patch32"
SEED = args.seed
TAU, EPOCHS = args.tau, args.epochs
N_TRAIN_CORPUS = 20000
BATCH, LR, WD, HIDDEN = 256, 3e-4, 1e-4, 512
K_VALUES = [1, 5, 10]
FRAC_SEL = 0.30
N_BOOT = 5000
N_TEST = 3

BRACCI = {
    "0 attuale":     {"targets": {1: 80, 2: 80, 3: 60},   "max_sources": 80},
    "A piu esempi":  {"targets": {1: 80, 2: 80, 3: 60},   "max_sources": 175},
    "B piu query":   {"targets": {1: 80, 2: 320, 3: 240}, "max_sources": 80},
}

t_start = time.time()
print("=" * 104)
print("PIU' QUERY o PIU' ESEMPI? — tre bracci a parita' di budget")
print("=" * 104)
print(f"log  : {LOG_PATH.name}")
print(f"json : {JSON_PATH.name}")
print(f"seed {SEED}, tau {TAU}, {EPOCHS} epoche, {args.threads} thread")
for nome, cfg in BRACCI.items():
    print(f"  {nome:<16} targets {cfg['targets']}  max_sources {cfg['max_sources']}")


def el():
    return f"{time.time() - t_start:6.0f}s"


def load_split(split):
    d = torch.load(ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"], d


print(f"\n[{el()}] carico gli embedding...")
E_train_all, L_train_all, meta = load_split("train")
E_valid, L_valid, _ = load_split("valid")
attr_names = [n for n in meta["attr_names"] if n.strip()]
D, N_ATTR = E_valid.shape[1], L_valid.shape[1]
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)
n_cond_dev = np.array([len(e["queried"]) for e in devset])
print(f"[{el()}] train {tuple(E_train_all.shape)}, valid {tuple(E_valid.shape)}, "
      f"dev {len(devset)} query")

# corpus di addestramento: identico a multiseed.py, e IDENTICO fra i tre bracci
g = torch.Generator().manual_seed(SEED)
sub = torch.randperm(len(E_train_all), generator=g)[:N_TRAIN_CORPUS].sort().values
E_tr, L_tr = E_train_all[sub], L_train_all[sub]
del E_train_all, L_train_all
print(f"[{el()}] corpus di addestramento {tuple(E_tr.shape)} (condiviso dai tre bracci)")

print(f"[{el()}] costruisco le direzioni testuali (10 template)...")
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
for i, name in enumerate(attr_names):
    c = name.replace("_", " ").lower()
    v = encode([t.format(c) for t, _ in TEMPLATES]).mean(0) - \
        encode([t.format(c) for _, t in TEMPLATES]).mean(0)
    d_ens[i] = v / v.norm()
del clip_model, tokenizer
print(f"[{el()}] direzioni {tuple(d_ens.shape)}")

# ===========================================================================================
# DATI DEI TRE BRACCI — costruiti tutti PRIMA di addestrare, cosi' i budget si vedono subito
# ===========================================================================================
print("\n" + "=" * 104)
print("COSTRUZIONE DEI TRE INSIEMI DI ADDESTRAMENTO")
print("=" * 104)


def cond_tensor(entry, max_n=3):
    u = torch.zeros(max_n, D)
    m = torch.zeros(max_n, dtype=torch.bool)
    k = 0
    for a in entry["pos"][:max_n]:
        u[k] = d_ens[a]; m[k] = True; k += 1
    for b in entry["neg"][:max_n - k]:
        u[k] = -d_ens[b]; m[k] = True; k += 1
    return u, m


def make_examples(bench):
    ex = []
    for e in bench:
        u, m = cond_tensor(e)
        for si, ref in enumerate(e["sources"].tolist()):
            ex.append((ref, u, m, torch.tensor(sorted(e["gt"][si]), dtype=torch.long)))
    return ex


DATI = {}
for nome, cfg in BRACCI.items():
    t0 = time.time()
    queries_txt = []
    for n_cond, quanti in cfg["targets"].items():
        queries_txt += sample_queries(list(attr_names), quanti, n_cond,
                                      exclude=queries_txt, seed=SEED + n_cond)
    bench = []
    for qt in queries_txt:
        bench += build_benchmark_from_labels([qt], L_tr, attr_names,
                                             max_sources=cfg["max_sources"], seed=SEED)
    ex = make_examples(bench)
    gg = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(bench), generator=gg).tolist()
    sel = [bench[i] for i in perm[:int(len(bench) * FRAC_SEL)]]
    comp = {k: sum(1 for e in bench if len(e["queried"]) == k) for k in (1, 2, 3)}
    tot = max(len(bench), 1)
    DATI[nome] = {"bench": bench, "ex": ex, "sel": sel,
                  "n_query_campionate": len(queries_txt), "n_query_valide": len(bench),
                  "n_esempi": len(ex), "composizione": comp,
                  "composizione_pct": {k: 100 * v / tot for k, v in comp.items()}}
    p = DATI[nome]["composizione_pct"]
    print(f"  {nome:<16} {len(queries_txt):>4} campionate -> {len(bench):>4} valide, "
          f"{len(ex):>6} esempi  |  composizione {p[1]:.0f}/{p[2]:.0f}/{p[3]:.0f}%  "
          f"({time.time() - t0:.0f}s)", flush=True)

ra, rb = DATI["A piu esempi"]["n_esempi"], DATI["B piu query"]["n_esempi"]
print(f"\n  budget di esempi A vs B: {ra} vs {rb}  "
      f"(rapporto {max(ra, rb) / max(min(ra, rb), 1):.2f}x)")
if max(ra, rb) / max(min(ra, rb), 1) > 1.15:
    print("  *** ATTENZIONE: i budget differiscono di piu' del 15%, il confronto B vs A")
    print("  *** non e' a parita' di esempi. Va corretto max_sources del braccio A.")
print(f"  composizione del set di sviluppo, per confronto: "
      f"{100 * (n_cond_dev == 1).mean():.0f}/{100 * (n_cond_dev == 2).mean():.0f}/"
      f"{100 * (n_cond_dev == 3).mean():.0f}%")

# ===========================================================================================
# MODELLO, LOSS, VALUTAZIONE — verbatim da multiseed.py
# ===========================================================================================
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
    loss = -(torch.logsumexp(sim.masked_fill(~pos, -float("inf")), 1)
             - torch.logsumexp(sim, 1)).mean()
    chance = float(np.mean([np.log(N / max(len(gt), 1)) for gt in gts]))
    return loss, chance


def evaluate(build_query, bench, embeddings, k_values, batch_size=512):
    per_source = {k: [] for k in k_values}
    k_max = max(k_values)
    for entry in bench:
        sources = entry["sources"]
        recall = {k: np.zeros(len(sources)) for k in k_values}
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
                    recall[k][start + j] = float(cum[k - 1] > 0)
        for k in k_values:
            per_source[k].append(recall[k])
    return per_source


def make_builder(model):
    cache = {}
    def f(ref, entry):
        key = entry["query"]
        if key not in cache:
            cache[key] = cond_tensor(entry)
        u, m = cache[key]
        with torch.no_grad():
            q = model(ref, u.unsqueeze(0).expand(len(ref), -1, -1),
                      m.unsqueeze(0).expand(len(ref), -1))
        return q
    return f


def build_baseline(ref, entry):
    d = torch.zeros(D)
    for a in entry["pos"]:
        d = d + d_ens[a]
    for b in entry["neg"]:
        d = d - d_ens[b]
    q = ref + d
    return q / q.norm(dim=-1, keepdim=True)


def train(esempi, bench_track, nome):
    torch.manual_seed(SEED)
    model = Concat()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(SEED)
    n_batch = (len(esempi) + BATCH - 1) // BATCH
    print(f"\n--- {nome}: {len(esempi)} esempi, {n_batch} batch per epoca, "
          f"{EPOCHS} epoche ---", flush=True)
    storia = []
    for ep in range(EPOCHS):
        model.train()
        order = torch.randperm(len(esempi), generator=gen)
        tot_l = tot_c = 0.0
        nb = 0
        for st in range(0, len(esempi), BATCH):
            idx = order[st:st + BATCH].tolist()
            refs = torch.tensor([esempi[i][0] for i in idx])
            U = torch.stack([esempi[i][1] for i in idx])
            M = torch.stack([esempi[i][2] for i in idx])
            gts = [esempi[i][3] for i in idx]
            loss, chance = info_nce_multipos(model(E_tr[refs], U, M), refs, gts, TAU)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_l += loss.item(); tot_c += chance; nb += 1
        model.eval()
        r10 = macro(evaluate(make_builder(model), bench_track, E_tr, K_VALUES)[10]) * 100
        storia.append({"epoca": ep + 1, "loss": tot_l / nb, "caso": tot_c / nb,
                       "gamma": model.gamma.item(), "r10_fit": r10})
        print(f"  epoca {ep + 1:>3}/{EPOCHS}  loss {tot_l / nb:6.3f}  caso {tot_c / nb:6.3f}"
              f"  divario {tot_c / nb - tot_l / nb:6.3f}  gamma {model.gamma.item():6.3f}"
              f"  R@10 fit {r10:5.2f}%  [{el()}]", flush=True)
    return model, storia


# ===========================================================================================
# ADDESTRAMENTO E VALUTAZIONE
# ===========================================================================================
print("\n" + "=" * 104)
print("ADDESTRAMENTO")
print("=" * 104)
PS = {}
STORIE = {}
MODELLI = {}
ps_base = evaluate(build_baseline, devset, E_valid, K_VALUES)
b10 = macro(ps_base[10]) * 100
print(f"baseline sul set di sviluppo: {b10:.2f}%  "
      f"{'OK' if abs(b10 - 19.91) < 0.05 else '*** DISCREPANZA ***'}")

for nome in BRACCI:
    mod, storia = train(DATI[nome]["ex"], DATI[nome]["sel"], nome)
    MODELLI[nome] = mod
    STORIE[nome] = storia
    PS[nome] = evaluate(make_builder(mod), devset, E_valid, K_VALUES)
    m = macro(PS[nome][10]) * 100
    print(f"  -> {nome}: R@10 sul set di sviluppo = {m:.2f}%  ({m - b10:+.2f}pp "
          f"sulla baseline)  [{el()}]", flush=True)

print("\n" + "=" * 104)
print("RISULTATI")
print("=" * 104)
print(f"{'braccio':<16}{'query':>7}{'esempi':>8}{'R@10':>9}{'vs baseline':>13}"
       f"{'1 cond.':>10}{'2 cond.':>10}{'3 cond.':>10}")
RIS = {}
for nome in BRACCI:
    m = macro(PS[nome][10]) * 100
    strati = []
    for k in (1, 2, 3):
        ix = np.where(n_cond_dev == k)[0]
        a = float(np.mean([PS[nome][10][i].mean() for i in ix])) * 100
        bb = float(np.mean([ps_base[10][i].mean() for i in ix])) * 100
        strati.append(a - bb)
    RIS[nome] = {"r10": m, "vs_baseline": m - b10, "strati_vs_baseline": strati,
                 "n_query_valide": DATI[nome]["n_query_valide"],
                 "n_esempi": DATI[nome]["n_esempi"],
                 "composizione_pct": DATI[nome]["composizione_pct"]}
    print(f"{nome:<16}{DATI[nome]['n_query_valide']:>7}{DATI[nome]['n_esempi']:>8}"
          f"{m:>8.2f}%{m - b10:>+12.2f}pp{strati[0]:>+9.2f}pp{strati[1]:>+9.2f}pp"
          f"{strati[2]:>+9.2f}pp")

print("\n" + "=" * 104)
print("CONFRONTI ACCOPPIATI (bootstrap gerarchico, Bonferroni su 3: soglia 0.0167)")
print("=" * 104)
CONFRONTI = [("PRIMARIA   B piu query vs 0 attuale", "B piu query", "0 attuale"),
             ("MECCANISMO B piu query vs A piu esempi", "B piu query", "A piu esempi"),
             ("CONTROLLO  A piu esempi vs 0 attuale", "A piu esempi", "0 attuale")]
CMP = {}
for etichetta, a, b in CONFRONTI:
    bs = bootstrap_paired(PS[a][10], PS[b][10], n_boot=N_BOOT, seed=SEED)
    CMP[etichetta] = {"delta_pp": bs["delta"] * 100, "lo": bs["lo"] * 100,
                      "hi": bs["hi"] * 100, "p": bs["p"],
                      "vinte": bs["n_query_win"], "perse": bs["n_query_loss"]}
    print(f"  {etichetta:<40}{bs['delta'] * 100:>+7.2f}pp  "
          f"[{bs['lo'] * 100:+.2f}, {bs['hi'] * 100:+.2f}]  p = {bs['p']:.4f}  "
          f"({bs['n_query_win']} vinte / {bs['n_query_loss']} perse)")

print("\n" + "=" * 104)
print("LETTURA")
print("=" * 104)
d_ba = CMP["MECCANISMO B piu query vs A piu esempi"]
d_b0 = CMP["PRIMARIA   B piu query vs 0 attuale"]
d_a0 = CMP["CONTROLLO  A piu esempi vs 0 attuale"]
print(f"  piu' query   : {d_b0['delta_pp']:+.2f}pp (p={d_b0['p']:.4f})")
print(f"  piu' esempi  : {d_a0['delta_pp']:+.2f}pp (p={d_a0['p']:.4f})")
print(f"  differenza   : {d_ba['delta_pp']:+.2f}pp (p={d_ba['p']:.4f})")
print()
if d_ba["p"] < 0.0167 and d_ba["delta_pp"] > 0:
    print("  Conta la DIVERSITA' DI QUERY in se', non la massa di esempi. Il bagging e' un")
    print("  sintomo: un modello solo con piu' query dovrebbe bastare, e conviene spingere")
    print("  su 3 e 4 condizioni, che comprano direzioni distinte al minor costo.")
elif d_a0["p"] < 0.0167 and abs(d_ba["delta_pp"]) < 1.0:
    print("  Conta la MASSA DI ESEMPI, non quali query siano. Si compra molto piu' a buon")
    print("  mercato alzando max_sources, senza campionare query nuove.")
elif d_b0["p"] >= 0.0167 and d_a0["p"] >= 0.0167:
    print("  NESSUNO DEI DUE aiuta su un solo modello. Il guadagno del bagging non viene")
    print("  dalla quantita' di dati per modello: viene dal mediare modelli diversi, e la")
    print("  spiegazione resta da trovare.")
else:
    print("  Quadro misto: leggere la tabella per strati prima di concludere.")
print()
print("  Ricorda il confondimento dichiarato: il braccio B e' anche piu' spostato verso le")
print("  query composte, perche' quelle a 1 condizione sono esaurite. La colonna per strati")
print("  serve a questo.")

payload = {"data": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
           "epoche": EPOCHS, "tau": TAU, "baseline_dev_r10": b10,
           "bracci": {k: {kk: vv for kk, vv in v.items()} for k, v in RIS.items()},
           "confronti": CMP,
           "storie": STORIE,
           "composizione_dev_pct": {int(k): float(100 * (n_cond_dev == k).mean())
                                    for k in (1, 2, 3)}}
JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[{el()}] salvato: {LOG_PATH.name} e {JSON_PATH.name}")
sys.stdout = _tee.stdout
_tee.close()
