"""CURVA SU max_sources — dove satura il guadagno da piu' esempi?

===========================================================================================
DA DOVE NASCE
===========================================================================================
`braccio_b_piu_query.py` (seed 1) ha trovato che il contributo e' limitato dagli ESEMPI, non
dalle query:

    braccio           query  esempi     R@10   vs baseline
    0 attuale           182    4433   26.53%       +6.62pp
    A piu esempi        189    9529   30.22%      +10.31pp     max_sources 80 -> 175
    B piu query         481    9079   29.79%       +9.87pp     query 182 -> 481

    B vs A: -0.43pp, p = 0.48   ->  le query in piu' non aggiungono nulla
    A vs 0: +3.69pp, p < 0.0001 ->  gli esempi in piu' si'

Un modello solo a 30.22% eguaglia l'ensemble a quattro modelli di Arianna (30.53%) con un
quarto del costo. E per ottenerlo basta alzare un parametro che era gia' nel codice.

Nessuno ha mai guardato oltre max_sources = 80. Questa curva guarda.

===========================================================================================
IL DISEGNO
===========================================================================================
Quattro addestramenti, un seed nuovo, tutto il resto identico (corpus, direzioni, query,
loss, iperparametri, valutazione).

    max_sources  80    la configurazione storica del progetto
    max_sources 175    il braccio A, gia' misurato su un altro seed
    max_sources 350    mai provato
    controllo passi    max_sources 80, ma con tante epoche da fare gli STESSI passi di
                       gradiente del livello 350

IL CONTROLLO SUI PASSI serve perche' piu' esempi significa anche piu' batch per epoca, e
quindi piu' passi di ottimizzazione a parita' di epoche: il livello 350 ne fa ~4.3 volte
quelli del livello 80. Se il guadagno fosse solo "piu' passi", il controllo lo riprodurrebbe
senza un solo esempio in piu'. C'e' gia' un'evidenza contraria — la curva per epoca misurata
in una sessione precedente e' un plateau piatto fra l'epoca 25 e la 70 — ma quel plateau non
copre le ~129 epoche del controllo, quindi va misurato.

PRE-REGISTRAZIONE, prima di guardare i numeri:
  - il numero riportato per ogni livello e' quello dell'ULTIMA epoca, sempre. La curva sul
    set di sviluppo viene stampata ogni 10 epoche per diagnosi (si vede quando comincia la
    memorizzazione) ma NON viene mai usata per scegliere l'epoca: sarebbe selezione sui dati
    di valutazione;
  - confronti accoppiati di ogni livello contro max_sources 80, piu' il controllo sui passi
    contro 80. Bonferroni sul numero di confronti effettuati.

Il seed e' 5, diverso da tutti quelli gia' usati (1, 2, 3, 4, 2026).
Il benchmark ufficiale NON viene toccato.

Uso:
  HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 python esperimenti/curva_max_sources.py --seed 5
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
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--tau", type=float, default=0.03)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--livelli", type=int, nargs="+", default=[80, 175, 350])
ap.add_argument("--salta-controllo-passi", action="store_true")
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
LOG_PATH = HERE / f"curva_max_sources_seed{args.seed}_{STAMP}.txt"
JSON_PATH = HERE / f"curva_max_sources_seed{args.seed}_{STAMP}.json"
_tee = Tee(LOG_PATH)
sys.stdout = _tee

MODEL_ID = "openai/clip-vit-base-patch32"
SEED, TAU, EPOCHS = args.seed, args.tau, args.epochs
N_TRAIN_CORPUS = 20000
QUERY_TARGETS = {1: 80, 2: 80, 3: 60}
BATCH, LR, WD, HIDDEN = 256, 3e-4, 1e-4, 512
K_VALUES = [1, 5, 10]
FRAC_SEL = 0.30
N_BOOT = 5000
DEV_OGNI = 10

t_start = time.time()


def el():
    return f"{time.time() - t_start:6.0f}s"


print("=" * 108)
print("CURVA SU max_sources — dove satura il guadagno da piu' esempi?")
print("=" * 108)
print(f"log  : {LOG_PATH.name}")
print(f"json : {JSON_PATH.name}")
print(f"seed {SEED} (nuovo), tau {TAU}, {EPOCHS} epoche, {args.threads} thread")
print(f"livelli di max_sources: {args.livelli}"
      f"{'' if args.salta_controllo_passi else '  + controllo sui passi di gradiente'}")
print(f"query: {QUERY_TARGETS} (identiche a tutti i livelli — qui varia SOLO max_sources)")


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

g = torch.Generator().manual_seed(SEED)
sub = torch.randperm(len(E_train_all), generator=g)[:N_TRAIN_CORPUS].sort().values
E_tr, L_tr = E_train_all[sub], L_train_all[sub]
del E_train_all, L_train_all
print(f"[{el()}] corpus di addestramento {tuple(E_tr.shape)}")

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
# LE QUERY, IDENTICHE A TUTTI I LIVELLI
# ===========================================================================================
queries_txt = []
for n_cond, quanti in QUERY_TARGETS.items():
    queries_txt += sample_queries(list(attr_names), quanti, n_cond,
                                  exclude=queries_txt, seed=SEED + n_cond)
print(f"[{el()}] {len(queries_txt)} query campionate (le stesse per ogni livello)")


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


print("\n" + "=" * 108)
print("COSTRUZIONE DEI DATI — tutti i livelli prima di addestrare, per vedere i budget")
print("=" * 108)
DATI = {}
for ms in args.livelli:
    t0 = time.time()
    bench = []
    for qt in queries_txt:
        bench += build_benchmark_from_labels([qt], L_tr, attr_names,
                                             max_sources=ms, seed=SEED)
    ex = make_examples(bench)
    gg = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(bench), generator=gg).tolist()
    sel = [bench[i] for i in perm[:int(len(bench) * FRAC_SEL)]]
    n_batch = (len(ex) + BATCH - 1) // BATCH
    DATI[ms] = {"bench": bench, "ex": ex, "sel": sel, "n_query": len(bench),
                "n_esempi": len(ex), "n_batch": n_batch}
    print(f"  max_sources {ms:>4}: {len(bench):>4} query valide, {len(ex):>6} esempi, "
          f"{n_batch:>3} batch/epoca, {n_batch * EPOCHS:>5} passi   ({time.time() - t0:.0f}s)",
          flush=True)

base_ms = args.livelli[0]
costo = {ms: DATI[ms]["n_esempi"] * EPOCHS for ms in args.livelli}
EP_CTRL = None
if not args.salta_controllo_passi:
    grande = args.livelli[-1]
    EP_CTRL = max(EPOCHS, round(EPOCHS * DATI[grande]["n_batch"] / DATI[base_ms]["n_batch"]))
    costo["controllo"] = DATI[base_ms]["n_esempi"] * EP_CTRL
    print(f"  controllo passi : max_sources {base_ms} per {EP_CTRL} epoche = "
          f"{DATI[base_ms]['n_batch'] * EP_CTRL} passi, contro i "
          f"{DATI[grande]['n_batch'] * EPOCHS} del livello {grande}")

tot = sum(costo.values())
print(f"\n  costo relativo totale: {tot / costo[base_ms]:.1f}x il livello {base_ms}")
print(f"  (il livello {base_ms} da solo e' costato ~10 minuti nelle run precedenti, quindi "
      f"la stima e' ~{10 * tot / costo[base_ms]:.0f} minuti)")

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


ps_base = evaluate(build_baseline, devset, E_valid, K_VALUES)
b10 = macro(ps_base[10]) * 100
print(f"\n[{el()}] baseline sul set di sviluppo: {b10:.2f}%  "
      f"{'OK' if abs(b10 - 19.91) < 0.05 else '*** DISCREPANZA ***'}")


def train(esempi, bench_track, nome, epoche):
    torch.manual_seed(SEED)
    model = Concat()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(SEED)
    n_batch = (len(esempi) + BATCH - 1) // BATCH
    print(f"\n--- {nome}: {len(esempi)} esempi, {n_batch} batch/epoca, {epoche} epoche, "
          f"{n_batch * epoche} passi ---", flush=True)
    storia = []
    for ep in range(epoche):
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
        riga = {"epoca": ep + 1, "loss": tot_l / nb, "caso": tot_c / nb,
                "gamma": model.gamma.item(), "r10_fit": r10}
        coda = ""
        # la curva sul dev e' DIAGNOSTICA: non viene mai usata per scegliere l'epoca
        if (ep + 1) % DEV_OGNI == 0 or ep == epoche - 1:
            rd = macro(evaluate(make_builder(model), devset, E_valid, K_VALUES)[10]) * 100
            riga["r10_dev"] = rd
            coda = f"  |  dev {rd:5.2f}%  divario {r10 - rd:5.2f}"
        storia.append(riga)
        print(f"  epoca {ep + 1:>4}/{epoche}  loss {tot_l / nb:6.3f}  caso {tot_c / nb:6.3f}"
              f"  divario {tot_c / nb - tot_l / nb:6.3f}  gamma {model.gamma.item():6.3f}"
              f"  fit {r10:5.2f}%{coda}  [{el()}]", flush=True)
    return model, storia


# ===========================================================================================
# ADDESTRAMENTO
# ===========================================================================================
print("\n" + "=" * 108)
print("ADDESTRAMENTO")
print("=" * 108)
PS, STORIE, RIS = {}, {}, {}
CONFIG = [(f"max_sources {ms}", DATI[ms]["ex"], DATI[ms]["sel"], EPOCHS, ms)
          for ms in args.livelli]
if EP_CTRL is not None:
    CONFIG.append((f"controllo passi (ms {base_ms}, {EP_CTRL} epoche)",
                   DATI[base_ms]["ex"], DATI[base_ms]["sel"], EP_CTRL, base_ms))

for nome, ex, sel, epoche, ms in CONFIG:
    mod, storia = train(ex, sel, nome, epoche)
    STORIE[nome] = storia
    PS[nome] = evaluate(make_builder(mod), devset, E_valid, K_VALUES)
    m = macro(PS[nome][10]) * 100
    strati = []
    for k in (1, 2, 3):
        ix = np.where(n_cond_dev == k)[0]
        a = float(np.mean([PS[nome][10][i].mean() for i in ix])) * 100
        bb = float(np.mean([ps_base[10][i].mean() for i in ix])) * 100
        strati.append(a - bb)
    RIS[nome] = {"max_sources": ms, "epoche": epoche, "n_esempi": len(ex),
                 "n_passi": ((len(ex) + BATCH - 1) // BATCH) * epoche,
                 "r10": m, "vs_baseline": m - b10, "strati_vs_baseline": strati,
                 "r10_fit_finale": storia[-1]["r10_fit"]}
    print(f"  -> {nome}: dev {m:.2f}%  ({m - b10:+.2f}pp sulla baseline)  [{el()}]",
          flush=True)

# ===========================================================================================
# RISULTATI
# ===========================================================================================
print("\n" + "=" * 108)
print("LA CURVA")
print("=" * 108)
print(f"{'configurazione':<40}{'esempi':>8}{'passi':>8}{'fit':>9}{'dev':>9}"
      f"{'vs base':>10}{'1c':>9}{'2c':>9}{'3c':>9}")
for nome in RIS:
    r = RIS[nome]
    print(f"{nome:<40}{r['n_esempi']:>8}{r['n_passi']:>8}{r['r10_fit_finale']:>8.2f}%"
          f"{r['r10']:>8.2f}%{r['vs_baseline']:>+9.2f}pp{r['strati_vs_baseline'][0]:>+8.2f}"
          f"{r['strati_vs_baseline'][1]:>+8.2f}{r['strati_vs_baseline'][2]:>+8.2f}")

rif = f"max_sources {base_ms}"
CONFRONTI = [n for n in RIS if n != rif]
soglia = 0.05 / max(len(CONFRONTI), 1)
print("\n" + "=" * 108)
print(f"CONFRONTI ACCOPPIATI contro {rif}  "
      f"(Bonferroni su {len(CONFRONTI)}: soglia {soglia:.4f})")
print("=" * 108)
CMP = {}
for nome in CONFRONTI:
    bs = bootstrap_paired(PS[nome][10], PS[rif][10], n_boot=N_BOOT, seed=SEED)
    CMP[nome] = {"delta_pp": bs["delta"] * 100, "lo": bs["lo"] * 100, "hi": bs["hi"] * 100,
                 "p": bs["p"], "vinte": bs["n_query_win"], "perse": bs["n_query_loss"]}
    print(f"  {nome:<42}{bs['delta'] * 100:>+7.2f}pp  "
          f"[{bs['lo'] * 100:+.2f}, {bs['hi'] * 100:+.2f}]  p = {bs['p']:.4f}  "
          f"({bs['n_query_win']} vinte / {bs['n_query_loss']} perse)")

print("\n" + "=" * 108)
print("LETTURA")
print("=" * 108)
liv = [(RIS[f"max_sources {ms}"]["n_esempi"], RIS[f"max_sources {ms}"]["r10"])
       for ms in args.livelli]
for i in range(1, len(liv)):
    d_es = liv[i][0] / liv[i - 1][0]
    d_r = liv[i][1] - liv[i - 1][1]
    print(f"  da {liv[i - 1][0]:>6} a {liv[i][0]:>6} esempi ({d_es:.2f}x): "
          f"{d_r:+.2f}pp")
if len(liv) >= 3:
    p1 = liv[1][1] - liv[0][1]
    p2 = liv[2][1] - liv[1][1]
    if p2 < 0.4 * p1:
        print("\n  LA CURVA STA SATURANDO: il secondo raddoppio rende molto meno del primo.")
        print("  Il valore di max_sources vicino al ginocchio e' quello da usare.")
    else:
        print("\n  LA CURVA NON HA ANCORA SATURATO: vale la pena estendere oltre "
              f"max_sources {args.livelli[-1]}.")
if EP_CTRL is not None:
    c = CMP[f"controllo passi (ms {base_ms}, {EP_CTRL} epoche)"]
    print(f"\n  Controllo sui passi: {c['delta_pp']:+.2f}pp (p={c['p']:.4f}). "
          f"{'I passi in piu' + chr(39) + ' spiegano parte del guadagno.' if c['p'] < soglia and c['delta_pp'] > 0 else 'I passi in piu' + chr(39) + ' da soli NON bastano: e' + chr(39) + ' la quantita' + chr(39) + ' di esempi.'}")

payload = {"data": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
           "epoche": EPOCHS, "tau": TAU, "query_targets": QUERY_TARGETS,
           "n_query_campionate": len(queries_txt), "baseline_dev_r10": b10,
           "livelli": RIS, "confronti": CMP, "storie": STORIE}
JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[{el()}] salvato: {LOG_PATH.name} e {JSON_PATH.name}")
sys.stdout = _tee.stdout
_tee.close()
