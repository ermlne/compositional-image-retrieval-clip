"""PIANO NOTTURNO — la curva su max_sources oltre 350, le repliche, e il bagging in cima.

===========================================================================================
DA DOVE SI PARTE (seed 5, 30 epoche, nessuna selezione)
===========================================================================================
    max_sources  80    dev 26.40%   (+6.49pp)     4422 esempi     540 passi
    max_sources 175    dev 29.22%   (+9.31pp)     9388 esempi    1110 passi
    max_sources 350    dev 32.26%  (+12.35pp)    18222 esempi    2160 passi

I due raddoppi rendono +2.82pp e +3.04pp: il secondo rende PIU' del primo, nessun ginocchio.
Un modello solo a 32.26% batte gia' l'ensemble a 4 modelli di Arianna (30.53%) con un quarto
del costo di inferenza. Il controllo sui passi di gradiente non e' mai stato completato.

===========================================================================================
DISCIPLINA
===========================================================================================
NESSUNA SELEZIONE. 60 epoche fisse per tutti, si riporta SEMPRE l'ultima epoca. Niente early
stopping, niente scelta dell'epoca. Il dev viene registrato ogni 5 epoche solo per diagnosi;
il suo massimo viene salvato ma etichettato TETTO OTTIMISTICO e non e' un risultato (sarebbe
un massimo su 60 alternative preso sui dati di valutazione, la trappola che ha prodotto meta'
della tabella delle esclusioni di STATO.md).

VALIDAZIONE ONESTA. Query nuove, mai addestrate e disgiunte sia dall'addestramento sia dal
set di sviluppo, valutate sul corpus di train. Servono a UNA cosa sola: scegliere quale
livello replicare nella fase 2, senza guardare il dev.

CONTROLLO SUI PASSI. Piu' esempi = piu' batch per epoca = piu' passi di gradiente a parita'
di epoche. Il controllo da' al livello 80 esattamente gli stessi passi del livello piu' alto,
senza un esempio in piu': se recupera il guadagno, non erano i dati ma l'ottimizzazione.

Il benchmark ufficiale NON viene toccato. Nessun comando git, nessuna rete, nessuna
cancellazione, nessuna scrittura fuori da esperimenti/notte/.

===========================================================================================
LE TRE FASI
===========================================================================================
1  CURVA      max_sources 350 (ancora, a 60 epoche), 700, 1400, + controllo sui passi
2  REPLICHE   il livello migliore secondo la VALIDAZIONE, su seed nuovi
3  BAGGING    i modelli della fase 2 mediati: i due guadagni si accumulano?

Ogni modello viene salvato in .pt appena addestrato e il JSON riscritto dopo ognuno: se il
processo muore, quello che e' stato fatto resta.
"""
import argparse
import ctypes
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import build_benchmark_from_labels, load_benchmark
from src.devset import bootstrap_paired, macro, sample_queries

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--tau", type=float, default=0.03)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--seed-curva", type=int, default=5)
ap.add_argument("--livelli", type=int, nargs="+", default=[350, 700, 1400])
ap.add_argument("--seed-repliche", type=int, nargs="+", default=[6, 7, 8])
ap.add_argument("--ore", type=float, default=7.5)
args = ap.parse_args()
torch.set_num_threads(args.threads)

# --- impedisci lo standby SENZA toccare le impostazioni di sistema -------------------------
# SetThreadExecutionState e' una richiesta del singolo processo, temporanea: decade da sola
# quando il processo termina. Non modifica nessuna preferenza di risparmio energetico.
VEGLIA = False
try:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040
    _r = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
    if not _r:
        _r = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    VEGLIA = bool(_r)
except Exception:
    VEGLIA = False


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


STAMP = datetime.now().strftime("%Y%m%d-%H%M")
LOG = HERE / f"notte_{STAMP}.txt"
JSONP = HERE / f"notte_{STAMP}.json"
CKPT = HERE / "checkpoints"
CKPT.mkdir(exist_ok=True)
sys.stdout = Tee(LOG)

MODEL_ID = "openai/clip-vit-base-patch32"
EPOCHS, TAU = args.epochs, args.tau
N_TRAIN_CORPUS = 20000
QUERY_TARGETS = {1: 80, 2: 80, 3: 60}
N_VAL_QUERY = 60
BATCH, LR, WD, HIDDEN = 256, 3e-4, 1e-4, 512
K_VALUES = [1, 5, 10]
DEV_OGNI = 5
N_BOOT = 5000

T0 = time.time()
DEADLINE = T0 + args.ore * 3600
RISULTATI = {"data": datetime.now().isoformat(timespec="seconds"), "epoche": EPOCHS,
             "tau": TAU, "seed_curva": args.seed_curva, "veglia_attiva": VEGLIA,
             "fase1_curva": {}, "fase2_repliche": {}, "fase3_bagging": {}}


def el():
    return f"{(time.time() - T0) / 60:6.1f}m"


def salva():
    JSONP.write_text(json.dumps(RISULTATI, indent=2, ensure_ascii=False, default=float),
                     encoding="utf-8")


def tempo_rimasto():
    return DEADLINE - time.time()


print("=" * 112)
print("PIANO NOTTURNO — curva su max_sources, repliche, bagging")
print("=" * 112)
print(f"avvio      : {datetime.now():%Y-%m-%d %H:%M}")
print(f"scadenza   : {datetime.now() + timedelta(hours=args.ore):%Y-%m-%d %H:%M} "
      f"({args.ore} ore)")
print(f"log        : {LOG.name}")
print(f"json       : {JSONP.name}")
print(f"checkpoint : {CKPT}")
print(f"veglia (standby impedito senza toccare le impostazioni): "
      f"{'ATTIVA' if VEGLIA else 'NON RIUSCITA — il PC potrebbe sospendersi'}")
print(f"{EPOCHS} epoche fisse, nessuna selezione dell'epoca, {args.threads} thread")
print(f"livelli: {args.livelli}   seed repliche: {args.seed_repliche}")


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
dev_query_txt = {e["query"] for e in devset}
n_cond_dev = np.array([len(e["queried"]) for e in devset])
print(f"[{el()}] train {tuple(E_train_all.shape)}, valid {tuple(E_valid.shape)}, "
      f"dev {len(devset)} query")

print(f"[{el()}] costruisco le direzioni testuali...")
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
    concetto = name.replace("_", " ").lower()
    v = encode([t.format(concetto) for t, _ in TEMPLATES]).mean(0) - \
        encode([t.format(concetto) for _, t in TEMPLATES]).mean(0)
    d_ens[i] = v / v.norm()
del clip_model, tokenizer
print(f"[{el()}] direzioni {tuple(d_ens.shape)}")


def cond_tensor(entry, max_n=3):
    u = torch.zeros(max_n, D)
    m = torch.zeros(max_n, dtype=torch.bool)
    k = 0
    for a in entry["pos"][:max_n]:
        u[k] = d_ens[a]
        m[k] = True
        k += 1
    for b in entry["neg"][:max_n - k]:
        u[k] = -d_ens[b]
        m[k] = True
        k += 1
    return u, m


def make_examples(bench):
    ex = []
    for e in bench:
        u, m = cond_tensor(e)
        for si, ref in enumerate(e["sources"].tolist()):
            ex.append((ref, u, m, torch.tensor(sorted(e["gt"][si]), dtype=torch.long)))
    return ex


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


def evaluate(build_query, bench, embeddings, k_values=K_VALUES, batch_size=512):
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


def make_builder(models):
    """Un modello, oppure una lista: in quel caso le CORREZIONI vengono mediate.

    Si mediano le uscite e non i pesi: reti da inizializzazioni diverse stanno in bacini
    diversi. E' la stessa scelta di `esperimenti/ensemble_seed.py`.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    cache = {}

    def f(ref, entry):
        key = entry["query"]
        if key not in cache:
            cache[key] = cond_tensor(entry)
        u, m = cache[key]
        U = u.unsqueeze(0).expand(len(ref), -1, -1)
        M = m.unsqueeze(0).expand(len(ref), -1)
        with torch.no_grad():
            c = (U * M.unsqueeze(-1)).sum(1)
            dl = torch.stack([mo.gamma * mo.net(torch.cat([ref, c], dim=-1))
                              for mo in models]).mean(0)
        q = ref + dl
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


def build_baseline(ref, entry):
    d = torch.zeros(D)
    for a in entry["pos"]:
        d = d + d_ens[a]
    for b in entry["neg"]:
        d = d - d_ens[b]
    q = ref + d
    return q / q.norm(dim=-1, keepdim=True)


ps_base = evaluate(build_baseline, devset, E_valid)
B10 = macro(ps_base[10]) * 100
print(f"[{el()}] baseline sul set di sviluppo: {B10:.2f}%  "
      f"{'OK' if abs(B10 - 19.91) < 0.05 else '*** DISCREPANZA ***'}")
RISULTATI["baseline_dev_r10"] = B10


def prepara(seed, max_sources, val=True):
    """Corpus, query, esempi e set di validazione onesto per un dato seed."""
    g = torch.Generator().manual_seed(seed)
    sub = torch.randperm(len(E_train_all), generator=g)[:N_TRAIN_CORPUS].sort().values
    E_tr, L_tr = E_train_all[sub], L_train_all[sub]

    qtxt = []
    for n_cond, quanti in QUERY_TARGETS.items():
        qtxt += sample_queries(list(attr_names), quanti, n_cond, exclude=qtxt,
                               seed=seed + n_cond)
    bench = []
    for qt in qtxt:
        bench += build_benchmark_from_labels([qt], L_tr, attr_names,
                                             max_sources=max_sources, seed=seed)
    ex = make_examples(bench)

    bench_val = []
    if val:
        # query di validazione: nuove, mai addestrate, e mai presenti nel set di sviluppo
        esclusi = set(qtxt) | dev_query_txt
        qval = []
        for n_cond, quanti in ((2, 40), (3, 40)):
            qval += sample_queries(list(attr_names), quanti, n_cond,
                                   exclude=esclusi | set(qval), seed=seed + 777 + n_cond)
        for qt in qval:
            bench_val += build_benchmark_from_labels([qt], L_tr, attr_names,
                                                     max_sources=60, seed=seed)
        bench_val = bench_val[:N_VAL_QUERY]
        sovr = {e["query"] for e in bench_val} & ({e["query"] for e in bench} | dev_query_txt)
        assert not sovr, sovr
    return E_tr, bench, ex, bench_val


def info_nce(model, E_tr, esempi, idx):
    refs = torch.tensor([esempi[i][0] for i in idx])
    U = torch.stack([esempi[i][1] for i in idx])
    M = torch.stack([esempi[i][2] for i in idx])
    gts = [esempi[i][3] for i in idx]
    q = model(E_tr[refs], U, M)
    sim = (q @ E_tr.T) / TAU
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


def addestra(nome, seed, max_sources, epoche, tag):
    E_tr, bench, ex, bench_val = prepara(seed, max_sources)
    n_batch = (len(ex) + BATCH - 1) // BATCH
    print(f"\n--- {nome}: seed {seed}, max_sources {max_sources}, {len(bench)} query, "
          f"{len(ex)} esempi, {n_batch} batch/epoca, {epoche} epoche, "
          f"{n_batch * epoche} passi ---", flush=True)
    torch.manual_seed(seed)
    model = Concat()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)
    storia = []
    for ep in range(epoche):
        model.train()
        order = torch.randperm(len(ex), generator=gen)
        tl = tc = 0.0
        nb = 0
        for st in range(0, len(ex), BATCH):
            loss, chance = info_nce(model, E_tr, ex, order[st:st + BATCH].tolist())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
            tc += chance
            nb += 1
        model.eval()
        riga = {"epoca": ep + 1, "loss": tl / nb, "caso": tc / nb,
                "gamma": model.gamma.item()}
        coda = ""
        if bench_val and ((ep + 1) % DEV_OGNI == 0 or ep == epoche - 1):
            rv = macro(evaluate(make_builder(model), bench_val, E_tr)[10]) * 100
            rd = macro(evaluate(make_builder(model), devset, E_valid)[10]) * 100
            riga["r10_val"] = rv
            riga["r10_dev"] = rd
            coda = f"  |  val {rv:5.2f}%  dev {rd:5.2f}%"
        storia.append(riga)
        print(f"  epoca {ep + 1:>4}/{epoche}  loss {tl / nb:6.3f}  caso {tc / nb:6.3f}"
              f"  divario {tc / nb - tl / nb:6.3f}  gamma {model.gamma.item():6.3f}{coda}"
              f"  [{el()}]", flush=True)

    ps = evaluate(make_builder(model), devset, E_valid)
    r10 = macro(ps[10]) * 100
    val_fin = storia[-1].get("r10_val", float("nan"))
    devs = [s["r10_dev"] for s in storia if "r10_dev" in s]
    strati = []
    for k in (1, 2, 3):
        ix = np.where(n_cond_dev == k)[0]
        a = float(np.mean([ps[10][i].mean() for i in ix])) * 100
        b = float(np.mean([ps_base[10][i].mean() for i in ix])) * 100
        strati.append(a - b)
    torch.save({"A": model.state_dict(), "seed": seed, "max_sources": max_sources,
                "epoche": epoche, "tau": TAU, "n_esempi": len(ex),
                "config": {"hidden": HIDDEN, "d": D, "n_attr": N_ATTR},
                "dirs_text": d_ens, "attr_names": list(attr_names),
                "dev_r10_ultima_epoca": r10},
               CKPT / f"{tag}.pt")
    out = {"nome": nome, "seed": seed, "max_sources": max_sources, "epoche": epoche,
           "n_query": len(bench), "n_esempi": len(ex), "n_passi": n_batch * epoche,
           "dev_r10": r10, "vs_baseline": r10 - B10, "val_r10": val_fin,
           "strati_vs_baseline": strati,
           "tetto_ottimistico_dev": max(devs) if devs else None,
           "checkpoint": f"{tag}.pt", "storia": storia}
    print(f"  -> {nome}: dev {r10:.2f}% ({r10 - B10:+.2f}pp)  val {val_fin:.2f}%  "
          f"salvato {tag}.pt  [{el()}]", flush=True)
    return model, ps, out


# ===========================================================================================
# FASE 1 — LA CURVA
# ===========================================================================================
print("\n" + "=" * 112)
print("FASE 1 — LA CURVA SU max_sources")
print("=" * 112)
PS1, MOD1 = {}, {}
SEED_C = args.seed_curva
# riferimento di costo: 18222 esempi x 30 epoche = 11 minuti misurati
COSTO_MIN_PER_ESEMPIO_EPOCA = 11.0 / (18222 * 30)


def stima_minuti(max_sources, epoche):
    esempi = 18222 * (max_sources / 350.0)      # sublineare in realta': e' una stima alta
    return COSTO_MIN_PER_ESEMPIO_EPOCA * esempi * epoche


for ms in args.livelli:
    st = stima_minuti(ms, EPOCHS)
    if tempo_rimasto() < st * 60 * 1.4:
        print(f"\n[{el()}] SALTO max_sources {ms}: stimati ~{st:.0f} min, "
              f"restano {tempo_rimasto() / 60:.0f} min")
        continue
    nome = f"max_sources {ms}"
    mod, ps, out = addestra(nome, SEED_C, ms, EPOCHS, f"curva_ms{ms}_seed{SEED_C}")
    MOD1[nome], PS1[nome] = mod, ps
    RISULTATI["fase1_curva"][nome] = out
    salva()

# controllo sui passi: max_sources 80, tante epoche da eguagliare il livello piu' alto fatto
fatti = list(RISULTATI["fase1_curva"])
if fatti:
    grande = RISULTATI["fase1_curva"][fatti[0]]
    _, _, ex80, _ = prepara(SEED_C, 80, val=False)
    nb80 = (len(ex80) + BATCH - 1) // BATCH
    ep_ctrl = max(EPOCHS, int(round(grande["n_passi"] / nb80)))
    st = stima_minuti(80, ep_ctrl)
    if tempo_rimasto() > st * 60 * 1.4:
        nome = f"controllo passi (ms 80, {ep_ctrl} epoche)"
        mod, ps, out = addestra(nome, SEED_C, 80, ep_ctrl, f"curva_ctrlpassi_seed{SEED_C}")
        out["passi_da_eguagliare"] = grande["n_passi"]
        out["livello_eguagliato"] = grande["nome"]
        MOD1[nome], PS1[nome] = mod, ps
        RISULTATI["fase1_curva"][nome] = out
        salva()
    else:
        print(f"\n[{el()}] SALTO il controllo sui passi: stimati ~{st:.0f} min")

print("\n" + "=" * 112)
print("FASE 1 — RIEPILOGO")
print("=" * 112)
print(f"{'configurazione':<44}{'esempi':>8}{'passi':>8}{'val':>9}{'dev':>9}"
      f"{'vs base':>10}{'1c':>8}{'2c':>8}{'3c':>8}")
for n, r in RISULTATI["fase1_curva"].items():
    print(f"{n:<44}{r['n_esempi']:>8}{r['n_passi']:>8}{r['val_r10']:>8.2f}%"
          f"{r['dev_r10']:>8.2f}%{r['vs_baseline']:>+9.2f}pp"
          f"{r['strati_vs_baseline'][0]:>+7.2f}{r['strati_vs_baseline'][1]:>+7.2f}"
          f"{r['strati_vs_baseline'][2]:>+7.2f}")

livelli_veri = {n: r for n, r in RISULTATI["fase1_curva"].items()
                if not n.startswith("controllo")}
if livelli_veri:
    rif = min(livelli_veri, key=lambda n: livelli_veri[n]["max_sources"])
    print(f"\nCONFRONTI ACCOPPIATI contro {rif}")
    for n in RISULTATI["fase1_curva"]:
        if n == rif or n not in PS1:
            continue
        bs = bootstrap_paired(PS1[n][10], PS1[rif][10], n_boot=N_BOOT, seed=SEED_C)
        RISULTATI["fase1_curva"][n]["vs_riferimento"] = {
            "riferimento": rif, "delta_pp": bs["delta"] * 100, "lo": bs["lo"] * 100,
            "hi": bs["hi"] * 100, "p": bs["p"]}
        print(f"  {n:<46}{bs['delta'] * 100:>+7.2f}pp  "
              f"[{bs['lo'] * 100:+.2f}, {bs['hi'] * 100:+.2f}]  p = {bs['p']:.4f}")
    salva()

# ===========================================================================================
# FASE 2 — REPLICHE del livello scelto sulla VALIDAZIONE (mai sul dev)
# ===========================================================================================
print("\n" + "=" * 112)
print("FASE 2 — REPLICHE MULTI-SEED")
print("=" * 112)
MODR, PSR = [], []
if not livelli_veri:
    print("nessun livello completato nella fase 1, salto")
else:
    scelto = max(livelli_veri, key=lambda n: livelli_veri[n]["val_r10"])
    MS = livelli_veri[scelto]["max_sources"]
    migliore_dev = max(livelli_veri, key=lambda n: livelli_veri[n]["dev_r10"])
    print(f"livello scelto sulla VALIDAZIONE (non sul dev): {scelto}, "
          f"val {livelli_veri[scelto]['val_r10']:.2f}%")
    print(f"per trasparenza, il migliore sul dev sarebbe stato: {migliore_dev}")
    RISULTATI["fase2_repliche"] = {"livello": MS, "scelto_su": "validazione onesta",
                                   "livello_scelto": scelto,
                                   "migliore_su_dev": migliore_dev, "modelli": {}}
    if scelto in MOD1:
        MODR.append(MOD1[scelto])
        PSR.append(PS1[scelto])
        RISULTATI["fase2_repliche"]["modelli"][f"seed {SEED_C}"] = livelli_veri[scelto]
    for sd in args.seed_repliche:
        st = stima_minuti(MS, EPOCHS)
        if tempo_rimasto() < st * 60 * 1.4:
            print(f"\n[{el()}] SALTO il seed {sd}: stimati ~{st:.0f} min, "
                  f"restano {tempo_rimasto() / 60:.0f} min")
            continue
        mod, ps, out = addestra(f"replica seed {sd}", sd, MS, EPOCHS,
                                f"replica_ms{MS}_seed{sd}")
        MODR.append(mod)
        PSR.append(ps)
        RISULTATI["fase2_repliche"]["modelli"][f"seed {sd}"] = out
        salva()

    vals = [r["dev_r10"] for r in RISULTATI["fase2_repliche"]["modelli"].values()]
    if vals:
        RISULTATI["fase2_repliche"]["media_singoli"] = float(np.mean(vals))
        RISULTATI["fase2_repliche"]["sd"] = (float(np.std(vals, ddof=1))
                                             if len(vals) > 1 else None)
        print(f"\n  {len(vals)} modelli a max_sources {MS}: media {np.mean(vals):.2f}%, "
              f"min {min(vals):.2f}%, max {max(vals):.2f}%")
        salva()

# ===========================================================================================
# FASE 3 — BAGGING: i due guadagni si accumulano?
# ===========================================================================================
print("\n" + "=" * 112)
print("FASE 3 — BAGGING sopra la configurazione migliore")
print("=" * 112)
if len(MODR) >= 2:
    ps_ens = evaluate(make_builder(MODR), devset, E_valid)
    r_ens = macro(ps_ens[10]) * 100
    media = float(np.mean([r["dev_r10"]
                           for r in RISULTATI["fase2_repliche"]["modelli"].values()]))
    print(f"  media dei {len(MODR)} singoli : {media:.2f}%")
    print(f"  ENSEMBLE dei {len(MODR)}        : {r_ens:.2f}%   "
          f"({r_ens - media:+.2f}pp sul singolo medio)")
    RISULTATI["fase3_bagging"] = {"n_modelli": len(MODR), "media_singoli": media,
                                  "ensemble_dev_r10": r_ens,
                                  "guadagno_su_singolo_medio": r_ens - media,
                                  "confronti": {}}
    for i, ps in enumerate(PSR):
        bs = bootstrap_paired(ps_ens[10], ps[10], n_boot=N_BOOT, seed=0)
        RISULTATI["fase3_bagging"]["confronti"][f"vs modello {i}"] = {
            "delta_pp": bs["delta"] * 100, "p": bs["p"]}
        print(f"    vs modello {i}: {bs['delta'] * 100:+6.2f}pp  p={bs['p']:.4f}")
    print("\n  Riferimento: il bagging di Arianna a max_sources 80 vale +3.76pp sul singolo")
    print(f"  medio. Qui vale {r_ens - media:+.2f}pp. Se e' molto piu' piccolo, i due")
    print("  interventi attaccano la stessa varianza e non si accumulano.")
    salva()
else:
    print("  meno di 2 modelli disponibili, salto")

print("\n" + "=" * 112)
print(f"FINITO — {(time.time() - T0) / 3600:.2f} ore")
print("=" * 112)
print(f"log  : {LOG}")
print(f"json : {JSONP}")
print(f"modelli salvati in: {CKPT}")
salva()
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
except Exception:
    pass
