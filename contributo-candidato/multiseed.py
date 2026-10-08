"""REPLICA MULTI-SEED del contributo candidato — un seed per invocazione.

===========================================================================================
PERCHE'
===========================================================================================
`contributo-candidato/README.md` dichiara il +7.35pp come risultato di UN SOLO SEED, e
mette la replica multi-seed al primo posto fra le cose mancanti. Questo script la esegue.

COSA VARIA CON IL SEED. Tutto cio' che nel run originale dipendeva da SEED=2026:
  - quali 20.000 immagini del train formano il corpus di addestramento;
  - quali query vengono campionate (80/80/60 a 1/2/3 condizioni);
  - quali reference entrano nel ground truth di ogni query;
  - la divisione fit/selezione;
  - l'inizializzazione dei pesi e l'ordine dei batch.
E' la replica conservativa: non isola il solo rumore di ottimizzazione, ma risponde alla
domanda che conta — "se rifacessimo tutto da capo, il guadagno resterebbe?".

COSA NON VARIA. La temperatura resta fissata a 0.03 per entrambi i modelli, il valore
scelto nel run originale su query di addestramento tenute da parte. Una replica valida la
CONFIGURAZIONE: riselezionare la temperatura a ogni seed costerebbe 4x e misurerebbe
un'altra cosa. Il valore e' al bordo inferiore della griglia provata: e' un limite noto,
registrato nel README, e non viene toccato qui.

CONTROLLO DI FEDELTA'. Le definizioni sotto (modelli, loss, valutazione, costruzione dei
dati) sono copiate VERBATIM da cross_attention_v2.py, di proposito: una replica deve
percorrere lo stesso codice, non una riscrittura. La verifica che la copia sia fedele e'
empirica ed e' automatica: con --seed 2026 il risultato deve riprodurre 27.26% di R@10
(saltare la fase di selezione della temperatura non cambia nulla, perche' `train` rifa'
`torch.manual_seed(SEED)` a ogni chiamata).

IN PIU' RISPETTO ALL'ORIGINALE, a costo quasi nullo perche' il modello e' gia' in memoria:
  - esp. 1 del piano di interpretabilita' (quota del delta fuori da span(r, c));
  - esp. 4 (la mappa lineare), ma FITTATA SU REFERENCE DEL TRAIN e non del dev: cosi' la
    mappa non tocca mai i dati su cui viene poi valutata. E' piu' pulito dell'originale,
    che la fittava sul dev stesso.

Il benchmark ufficiale NON viene toccato.

Uso:  HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 python contributo-candidato/multiseed.py --seed 1
"""
import argparse
import json
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
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--tau", type=float, default=0.03)
ap.add_argument("--threads", type=int, default=14)
args = ap.parse_args()
torch.set_num_threads(args.threads)

# --- configurazione, identica a cross_attention_v2.py ---------------------------------
MODEL_ID = "openai/clip-vit-base-patch32"
SEED = args.seed
K_VALUES = [1, 5, 10]
N_TRAIN_CORPUS = 20000
QUERY_TARGETS = {1: 80, 2: 80, 3: 60}
MAX_SOURCES = 80
FRAC_SEL = 0.20
EPOCHS_FINAL = args.epochs
BATCH, LR, WD = 256, 3e-4, 1e-4
HIDDEN, N_HEADS = 512, 4
TAU = args.tau

OUT = HERE / "multiseed"
OUT.mkdir(exist_ok=True)
CKPT = HERE / "checkpoints"
CKPT.mkdir(exist_ok=True)

t_start = time.time()
print("=" * 92)
print(f"REPLICA MULTI-SEED — seed {SEED}, tau {TAU}, {EPOCHS_FINAL} epoche, "
      f"{args.threads} thread")
print("=" * 92, flush=True)


def load_split(split):
    d = torch.load(ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"], d


E_train_all, L_train_all, meta = load_split("train")
E_valid, L_valid, _ = load_split("valid")
attr_names = [n for n in meta["attr_names"] if n.strip()]
D, N_ATTR = E_valid.shape[1], L_valid.shape[1]
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)

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

# --- dati di addestramento, stessa sequenza di RNG dell'originale ----------------------
g = torch.Generator().manual_seed(SEED)
sub = torch.randperm(len(E_train_all), generator=g)[:N_TRAIN_CORPUS].sort().values
E_tr, L_tr = E_train_all[sub], L_train_all[sub]
del E_train_all, L_train_all

queries_txt = []
for n_cond, quanti in QUERY_TARGETS.items():
    queries_txt += sample_queries(list(attr_names), quanti, n_cond,
                                  exclude=queries_txt, seed=SEED + n_cond)

train_bench = []
for qt in queries_txt:
    train_bench += build_benchmark_from_labels([qt], L_tr, attr_names,
                                               max_sources=MAX_SOURCES, seed=SEED)
gg = torch.Generator().manual_seed(SEED)
perm_q = torch.randperm(len(train_bench), generator=gg).tolist()
n_sel = int(len(train_bench) * FRAC_SEL)
BENCH_SEL = [train_bench[i] for i in perm_q[:n_sel]]
print(f"dati: {len(train_bench)} query valide su {len(queries_txt)}, "
      f"{sum(len(e['sources']) for e in train_bench)} coppie  "
      f"({time.time() - t_start:.0f}s)", flush=True)


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


EX_ALL = make_examples(train_bench)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def delta(self, r, c):
        return self.net(torch.cat([r, c], dim=-1))

    def forward(self, r, U, M, want_attn=False):
        c = (U * M.unsqueeze(-1)).sum(1)
        delta = self.net(torch.cat([r, c], dim=-1))
        q = r + self.gamma * delta
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8), delta, None


class CrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, N_HEADS, batch_first=True)
        self.net = nn.Sequential(nn.Linear(2 * D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, U, M, want_attn=False):
        ctx, w = self.attn(r.unsqueeze(1), U, U, key_padding_mask=~M,
                           need_weights=want_attn, average_attn_weights=True)
        delta = self.net(torch.cat([r, ctx.squeeze(1)], dim=-1))
        q = r + self.gamma * delta
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8), delta, w


def cond_of(entry):
    c = torch.zeros(D)
    for a in entry["pos"]: c = c + d_ens[a]
    for b in entry["neg"]: c = c - d_ens[b]
    return c


def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)


def info_nce_multipos(q, refs, gts, tau):
    sim = (q @ E_tr.T) / tau
    B, N = sim.shape
    ar = torch.arange(B)
    sim = sim.clone()
    sim[ar, refs] = -float("inf")
    pos_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, gt in enumerate(gts):
        pos_mask[i, gt] = True
    pos_mask[ar, refs] = False
    num = torch.logsumexp(sim.masked_fill(~pos_mask, -float("inf")), dim=1)
    den = torch.logsumexp(sim, dim=1)
    loss = -(num - den).mean()
    n_pos = pos_mask.sum(1).float().clamp(min=1)
    return loss, torch.log((N - 1) / n_pos).mean().item()


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
            top = sim.topk(k_max, dim=-1).indices.tolist()
            for j, retrieved in enumerate(top):
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
            q, _, _ = model(ref, u.unsqueeze(0).expand(len(ref), -1, -1),
                            m.unsqueeze(0).expand(len(ref), -1))
        return q
    return f


def train(make_model, esempi, tau, epochs, nome, bench_track=None, track_every=10):
    torch.manual_seed(SEED)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(SEED)
    print(f"\n--- {nome} (tau={tau}, {epochs} epoche, {len(esempi)} esempi) ---", flush=True)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(esempi), generator=gen)
        tot_l, tot_c, nb = 0.0, 0.0, 0
        for st in range(0, len(esempi), BATCH):
            idx = order[st:st + BATCH].tolist()
            refs = torch.tensor([esempi[i][0] for i in idx])
            U = torch.stack([esempi[i][1] for i in idx])
            M = torch.stack([esempi[i][2] for i in idx])
            gts = [esempi[i][3] for i in idx]
            r = E_tr[refs]
            q, delta, _ = model(r, U, M)
            loss, chance = info_nce_multipos(q, refs, gts, tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_l += loss.item(); tot_c += chance; nb += 1
        if bench_track is not None and ((ep + 1) % track_every == 0 or ep == epochs - 1):
            model.eval()
            r10 = macro(evaluate(make_builder(model), bench_track, E_tr, K_VALUES)[10]) * 100
            print(f"  epoca {ep + 1:>3}  loss {tot_l / nb:6.3f}  caso {tot_c / nb:6.3f}  "
                  f"divario {tot_c / nb - tot_l / nb:6.3f}  gamma {model.gamma.item():6.3f}  "
                  f"R@10 fit {r10:5.2f}%  ({time.time() - t_start:.0f}s)", flush=True)
    model.eval()
    return model


MOD_A = train(Concat, EX_ALL, TAU, EPOCHS_FINAL, "A: concatenazione", bench_track=BENCH_SEL)
MOD_B = train(CrossAttn, EX_ALL, TAU, EPOCHS_FINAL, "B: cross-attention", bench_track=BENCH_SEL)

torch.save({"A": MOD_A.state_dict(), "B": MOD_B.state_dict(), "seed": SEED,
            "tau": {"A": TAU, "B": TAU},
            "config": {"hidden": HIDDEN, "n_heads": N_HEADS, "d": D, "n_attr": N_ATTR},
            "dirs_text": d_ens, "attr_names": list(attr_names)},
           CKPT / f"models_seed{SEED}.pt")

res = {"seed": SEED, "tau": TAU, "epochs": EPOCHS_FINAL,
       "n_query_train": len(train_bench), "n_examples": len(EX_ALL),
       "gamma": {"A": MOD_A.gamma.item(), "B": MOD_B.gamma.item()}}

# --- controllo positivo: devono battere la baseline IN ADDESTRAMENTO -------------------
pc = {n: macro(evaluate(f, train_bench, E_tr, K_VALUES)[10]) * 100 for n, f in
      (("baseline", build_baseline), ("A", make_builder(MOD_A)), ("B", make_builder(MOD_B)))}
res["controllo_positivo"] = pc
res["controllo_positivo_superato"] = bool(pc["A"] > pc["baseline"] and pc["B"] > pc["baseline"])
print(f"\ncontrollo positivo (R@10 sul train): baseline {pc['baseline']:.2f}%  "
      f"A {pc['A']:.2f}%  B {pc['B']:.2f}%  -> "
      f"{'superato' if res['controllo_positivo_superato'] else 'FALLITO'}", flush=True)

# --- esp. 1: quanta parte del delta esce da span(r, c) ---------------------------------
frac = []
for e in devset[:40]:
    c = cond_of(e)
    R = E_valid[e["sources"][:100]]
    with torch.no_grad():
        dl = MOD_A.delta(R, c.expand(len(R), -1))
    B = torch.stack([R, c.expand_as(R)], dim=1)
    Q, _ = torch.linalg.qr(B.transpose(1, 2))
    proj = torch.einsum('bdk,bk->bd', Q, torch.einsum('bdk,bd->bk', Q, dl))
    rest = dl - proj
    frac += (rest.norm(dim=-1) / dl.norm(dim=-1).clamp(min=1e-9)).tolist()
res["fuori_span_energia_pct"] = float(100 * np.mean(np.array(frac) ** 2))
print(f"esp.1  energia del delta fuori da span(r,c): {res['fuori_span_energia_pct']:.1f}%",
      flush=True)

# --- esp. 4: mappa lineare, FITTATA SU REFERENCE DEL TRAIN -----------------------------
X, Y = [], []
for e in train_bench:
    c = cond_of(e)
    R = E_tr[e["sources"][:60]]
    with torch.no_grad():
        dl = MOD_A.delta(R, c.expand(len(R), -1))
    X.append(torch.cat([R, c.expand(len(R), -1)], 1)); Y.append(dl)
X, Y = torch.cat(X), torch.cat(Y)
gl = torch.Generator().manual_seed(0)
idx = torch.randperm(len(X), generator=gl)
tr, te = idx[:int(.8 * len(X))], idx[int(.8 * len(X)):]
W = torch.linalg.lstsq(torch.cat([X[tr], torch.ones(len(tr), 1)], 1), Y[tr]).solution
pred = torch.cat([X[te], torch.ones(len(te), 1)], 1) @ W
r2 = (1 - ((Y[te] - pred) ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum()).item()
res["mappa_lineare"] = {"r2_heldout_train": r2,
                        "cos_medio": float(torch.nn.functional.cosine_similarity(
                            pred, Y[te], dim=-1).mean())}
torch.save({"W": W, "r2": r2, "gamma": MOD_A.gamma.item()}, CKPT / f"linear_seed{SEED}.pt")
print(f"esp.4  mappa lineare: R^2 (held-out train) {r2:.4f}  "
      f"cos {res['mappa_lineare']['cos_medio']:.4f}", flush=True)

GAMMA_A = MOD_A.gamma.item()
def build_linear(ref, entry):
    c = cond_of(entry).expand(len(ref), -1)
    dl = torch.cat([ref, c, torch.ones(len(ref), 1)], 1) @ W
    q = ref + GAMMA_A * dl
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# --- valutazione sul set di sviluppo ---------------------------------------------------
METODI = {"baseline": build_baseline, "A": make_builder(MOD_A),
          "B": make_builder(MOD_B), "lineare": build_linear}
ps = {n: evaluate(f, devset, E_valid, K_VALUES) for n, f in METODI.items()}

b10 = macro(ps["baseline"][10]) * 100
res["baseline_dev_r10"] = b10
res["baseline_ok"] = bool(abs(b10 - 19.91) < 0.05)

sem = [i for i, e in enumerate(devset) if len(e["queried"]) == 1]
com = [i for i, e in enumerate(devset) if len(e["queried"]) > 1]

def cmp(a, b, idx=None):
    xa = ps[a][10] if idx is None else [ps[a][10][i] for i in idx]
    xb = ps[b][10] if idx is None else [ps[b][10][i] for i in idx]
    s = bootstrap_paired(xa, xb, n_boot=20000, seed=0)
    return {"delta_pp": s["delta"] * 100, "lo": s["lo"] * 100, "hi": s["hi"] * 100,
            "p": s["p"]}

res["dev"] = {n: {f"r{k}": macro(ps[n][k]) * 100 for k in K_VALUES} for n in METODI}
res["confronti"] = {
    "A_vs_baseline": cmp("A", "baseline"),
    "B_vs_baseline": cmp("B", "baseline"),
    "A_vs_B": cmp("A", "B"),
    "lineare_vs_baseline": cmp("lineare", "baseline"),
    "lineare_vs_A": cmp("lineare", "A"),
    "A_vs_baseline_semplici": cmp("A", "baseline", sem),
    "A_vs_baseline_composte": cmp("A", "baseline", com),
    "B_vs_A_composte": cmp("B", "A", com),
}
res["minuti"] = (time.time() - t_start) / 60

print("\n" + "=" * 92)
print(f"SEED {SEED} — set di sviluppo (baseline {b10:.2f}%"
      f"{' OK' if res['baseline_ok'] else ' <-- DISCREPANZA'})")
print("=" * 92)
for n in METODI:
    r = res["dev"][n]
    print(f"  {n:<10}{r['r1']:>7.2f}%{r['r5']:>8.2f}%{r['r10']:>8.2f}%")
for k, v in res["confronti"].items():
    print(f"  {k:<26}{v['delta_pp']:>+7.2f}pp  IC [{v['lo']:+.2f},{v['hi']:+.2f}]  p={v['p']:.4f}")

np.savez_compressed(OUT / f"per_source_seed{SEED}.npz",
                    **{f"{n}_{k}": np.concatenate(ps[n][k]) for n in METODI for k in K_VALUES},
                    n_src=np.array([len(x) for x in ps["baseline"][10]]))
with open(OUT / f"seed{SEED}.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"\nsalvato in {OUT}/seed{SEED}.json  ({res['minuti']:.1f} minuti)")
