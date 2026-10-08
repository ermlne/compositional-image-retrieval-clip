"""Esperimento: la non linearita' dell'MLP serve, o e' dannosa?

Motivazione. L'esp. 4 di interpretabilita.py mostra che l'MLP del contributo candidato e'
una mappa quasi lineare (R^2 = 0.94) e che sostituirlo con due matrici non cambia il
risultato. La causa e' la scala delle pre-attivazioni: |z| mediano 0.064, dove
GELU(x) = 0.5x + O(x^2) e il termine quadratico vale il 5% del lineare.

Restavano due letture: (A) il compito e' lineare, quindi la non linearita' e' inutile;
(B) la rete e' mal scalata e sta lasciando capacita' inutilizzata. Questo script le
distingue aggiungendo una LayerNorm prima della GELU, che porta le pre-attivazioni a
scala unitaria dove la non linearita' e' pienamente attiva.

ESITO (seed 2026): ne' (A) ne' (B). La non linearita' e' DANNOSA.

  metodo                R@10 fit    R@10 dev
  baseline                14.71%      19.91%
  A: concatenazione       67.84%      27.26%
  A + LayerNorm           99.39%      18.53%     -8.73pp vs A, p<0.0001

Il modello con LayerNorm impara quasi perfettamente i dati di addestramento e crolla sul
set di sviluppo, sotto la baseline. Con 4407 esempi e 787k parametri la capacita' extra va
tutta in memorizzazione.

Conseguenza: il regime quasi lineare non e' un difetto ma una REGOLARIZZAZIONE IMPLICITA,
ed e' cio' che permette al metodo di generalizzare. La mappa lineare esplicita non e' una
semplificazione a posteriori: e' la forma giusta del metodo.

Per il fine tuning: piu' capacita' e' una strada chiusa da questa misura. Le direzioni che
restano sono aumentare il segnale (augmentation con i 10 template: la supervisione
effettiva sul lato testo e' di 40 punti soli) o vincolare la mappa (rango ridotto,
regolarizzazione verso l'identita').

Uso:  HF_HUB_OFFLINE=1 python contributo-candidato/esperimento_layernorm.py
Tempo: ~55 minuti su CPU.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark, build_benchmark_from_labels
from src.devset import bootstrap_paired, macro, sample_queries

# ===========================================================================================
# CONFIGURAZIONE — tutto quello che governa l'esperimento, in un posto solo
# ===========================================================================================
MODEL_ID = "openai/clip-vit-base-patch32"
SEED = 2026
K_VALUES = [1, 5, 10]

N_TRAIN_CORPUS = 20000        # galleria di addestramento (sottoinsieme del train split)
QUERY_TARGETS = {1: 80, 2: 80, 3: 60}   # piu' query composte del giro precedente: e' li'
                                        # che vive il meccanismo sotto esame
MAX_SOURCES = 80              # reference per query, prima del filtro di Hamming
FRAC_SEL = 0.20               # quota di QUERY tenute fuori per scegliere la temperatura

TAUS = [0.03, 0.05, 0.10, 0.20]   # griglia della temperatura, scelta su query held-out
EPOCHS_SEL = 10               # epoche nella fase di selezione (corta)
EPOCHS_FINAL = 30             # epoche nell'addestramento finale (piu' lungo)
BATCH, LR, WD = 256, 3e-4, 1e-4
HIDDEN, N_HEADS = 512, 4

ALPHA_SIG, MIN_EFFECT_PP = 0.05, 0.5

print("=" * 92)
print("CONFIGURAZIONE")
print("=" * 92)
for k, v in [("corpus di addestramento", N_TRAIN_CORPUS), ("query per n. condizioni", QUERY_TARGETS),
             ("reference per query", MAX_SOURCES), ("quota query per selezione", FRAC_SEL),
             ("griglia temperatura", TAUS), ("epoche selezione / finale", f"{EPOCHS_SEL} / {EPOCHS_FINAL}"),
             ("batch / lr / weight decay", f"{BATCH} / {LR} / {WD}"),
             ("hidden / teste attenzione", f"{HIDDEN} / {N_HEADS}")]:
    print(f"  {k:<28} {v}")


def load_split(split):
    d = torch.load(ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"], d


print("\nCarico embedding...")
E_train_all, L_train_all, meta = load_split("train")
E_valid, L_valid, _ = load_split("valid")
attr_names = [n for n in meta["attr_names"] if n.strip()]
D, N_ATTR = E_valid.shape[1], L_valid.shape[1]
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)
print(f"  dev: {len(devset)} query, {sum(len(e['sources']) for e in devset)} reference")

# ===========================================================================================
# DIREZIONI TESTUALI — identiche a tutto il resto del progetto
# ===========================================================================================
print(f"\nCarico CLIP ({MODEL_ID})...")
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
print(f"  direzioni costruite: {tuple(d_ens.shape)}")

# ===========================================================================================
# DATI DI ADDESTRAMENTO — query dal train, ground truth con la regola dell'assignment
# ===========================================================================================
g = torch.Generator().manual_seed(SEED)
sub = torch.randperm(len(E_train_all), generator=g)[:N_TRAIN_CORPUS].sort().values
E_tr, L_tr = E_train_all[sub], L_train_all[sub]
del E_train_all
print(f"\nCorpus di addestramento: {len(E_tr)} immagini dal train split")

# sample_queries e' la funzione del GRUPPO: gestisce da sola il fatto che lo spazio delle
# query a una condizione sia finito (80 con 40 attributi). Un primo tentativo scritto a
# mano era un ciclo infinito proprio per questo.
queries_txt = []
for n_cond, quanti in QUERY_TARGETS.items():
    got = sample_queries(list(attr_names), quanti, n_cond, exclude=queries_txt, seed=SEED + n_cond)
    print(f"  {n_cond} condizione/i: chieste {quanti}, ottenute {len(got)}", flush=True)
    queries_txt += got

print(f"\nGround truth di {len(queries_txt)} query (regola dell'assignment, "
      f"build_benchmark_from_labels)...", flush=True)
train_bench, t0 = [], time.time()
for qi, qt in enumerate(queries_txt, 1):
    train_bench += build_benchmark_from_labels([qt], L_tr, attr_names,
                                               max_sources=MAX_SOURCES, seed=SEED)
    if qi % 40 == 0 or qi == len(queries_txt):
        el = time.time() - t0
        print(f"    {qi:3d}/{len(queries_txt)}  ({el:5.1f}s, stimate "
              f"{el / qi * len(queries_txt):5.1f}s, {len(train_bench)} valide)", flush=True)
print(f"  {len(train_bench)} query valide, "
      f"{sum(len(e['sources']) for e in train_bench)} coppie (reference, target)")

# Divisione per QUERY (non per esempio): le query di selezione non compaiono mai
# nell'addestramento, altrimenti la temperatura verrebbe scelta su dati gia' visti.
gg = torch.Generator().manual_seed(SEED)
perm_q = torch.randperm(len(train_bench), generator=gg).tolist()
n_sel = int(len(train_bench) * FRAC_SEL)
IDX_SEL, IDX_FIT = perm_q[:n_sel], perm_q[n_sel:]
BENCH_SEL = [train_bench[i] for i in IDX_SEL]
BENCH_FIT = [train_bench[i] for i in IDX_FIT]
print(f"  divise in {len(BENCH_FIT)} query di fit e {len(BENCH_SEL)} di selezione")


def cond_tensor(entry, max_n=3):
    """Condizioni come insieme di vettori CON SEGNO, piu' maschera di padding."""
    u = torch.zeros(max_n, D)
    m = torch.zeros(max_n, dtype=torch.bool)
    k = 0
    for a in entry["pos"][:max_n]:
        u[k] = d_ens[a]; m[k] = True; k += 1
    for b in entry["neg"][:max_n - k]:
        u[k] = -d_ens[b]; m[k] = True; k += 1
    return u, m


def make_examples(bench):
    """(indice reference, tensore condizioni, maschera, insieme dei ground truth)."""
    ex = []
    for e in bench:
        u, m = cond_tensor(e)
        for si, ref in enumerate(e["sources"].tolist()):
            ex.append((ref, u, m, torch.tensor(sorted(e["gt"][si]), dtype=torch.long)))
    return ex


EX_FIT, EX_SEL = make_examples(BENCH_FIT), make_examples(BENCH_SEL)
gt_sizes = [len(x[3]) for x in EX_FIT]
print(f"  esempi: {len(EX_FIT)} di fit, {len(EX_SEL)} di selezione")
print(f"  ground truth per esempio: mediana {int(np.median(gt_sizes))}, "
      f"min {min(gt_sizes)}, max {max(gt_sizes)}")
print(f"  -> livello di caso atteso della loss: ln(N/|GT|) ~ "
      f"{np.log(N_TRAIN_CORPUS / np.median(gt_sizes)):.2f}")


# ===========================================================================================
# I MODELLI — stessa capacita' a valle, unica differenza il modo di combinare le condizioni
# ===========================================================================================

class Concat(nn.Module):
    """GRADINO A. Le condizioni sono SOMMATE (invariante all'ordine) e concatenate a r.
    Nessuna attenzione: e' il partner di confronto."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        # ultimo strato a zero: all'avvio delta = 0, quindi q = r ESATTO. Il modello parte
        # dall'identita' pura e puo' solo allontanarsene imparando.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, U, M, want_attn=False):
        c = (U * M.unsqueeze(-1)).sum(1)
        delta = self.net(torch.cat([r, c], dim=-1))
        q = r + self.gamma * delta
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8), delta, None


class ConcatLN(nn.Module):
    """Come Concat, ma con LayerNorm PRIMA della GELU. Serve a rispondere a una domanda
    sola: l'MLP risulta lineare perche' il COMPITO e' lineare, o perche' le pre-attivazioni
    sono schiacciate attorno a zero (|z| mediano 0.064) dove la GELU e' indistinguibile da
    una retta? La LayerNorm porta le pre-attivazioni a scala unitaria, dove la non
    linearita' e' pienamente attiva. Se il risultato non migliora, il compito e' lineare."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2 * D, HIDDEN)
        self.ln = nn.LayerNorm(HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, D)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, r, U, M, want_attn=False):
        c = (U * M.unsqueeze(-1)).sum(1)
        delta = self.fc2(nn.functional.gelu(self.ln(self.fc1(torch.cat([r, c], dim=-1)))))
        q = r + self.gamma * delta
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8), delta, None


class CrossAttn(nn.Module):
    """GRADINO B. La reference fa da QUERY, le condizioni da KEY e VALUE. La proiezione dei
    value e' il pezzo NON coperto dall'oracolo sui pesi: permette all'uscita di stare fuori
    dallo span delle direzioni date."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, N_HEADS, batch_first=True)
        self.net = nn.Sequential(nn.Linear(2 * D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, U, M, want_attn=False):
        # key_padding_mask: True = da IGNORARE, quindi il complemento della maschera valida
        ctx, w = self.attn(r.unsqueeze(1), U, U, key_padding_mask=~M,
                           need_weights=want_attn, average_attn_weights=True)
        delta = self.net(torch.cat([r, ctx.squeeze(1)], dim=-1))
        q = r + self.gamma * delta
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8), delta, w


def build_baseline(ref, entry):
    d = torch.zeros(D)
    for a in entry["pos"]: d = d + d_ens[a]
    for b in entry["neg"]: d = d - d_ens[b]
    q = ref + d
    return q / q.norm(dim=-1, keepdim=True)


# ===========================================================================================
# LOSS — InfoNCE MULTI-POSITIVO, con i ground truth mai trattati come negativi
# ===========================================================================================

def info_nce_multipos(q, refs, gts, tau):
    """q: [B, D] normalizzato. refs: [B] indici delle reference. gts: lista di LongTensor.

    numeratore   = logsumexp sulle similarita' con TUTTI i ground truth
    denominatore = logsumexp su tutto il corpus, esclusa la reference stessa

    Cosi' il modello non viene punito per aver messo in cima un ground truth diverso da
    quello 'scelto' — che era il difetto della v1. Restituisce anche il livello di CASO
    della loss, ln(N_validi / |GT|), che e' il riferimento contro cui leggere il numero.
    """
    sim = (q @ E_tr.T) / tau                                # [B, N]
    B, N = sim.shape
    ar = torch.arange(B)
    sim = sim.clone()
    sim[ar, refs] = -float("inf")                           # la reference non e' un target

    pos_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, gt in enumerate(gts):
        pos_mask[i, gt] = True
    pos_mask[ar, refs] = False                              # per sicurezza

    num = torch.logsumexp(sim.masked_fill(~pos_mask, -float("inf")), dim=1)
    den = torch.logsumexp(sim, dim=1)
    loss = -(num - den).mean()

    n_pos = pos_mask.sum(1).float().clamp(min=1)
    chance = torch.log((N - 1) / n_pos).mean()
    return loss, chance.item()


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


# ===========================================================================================
# ADDESTRAMENTO — verboso: loss ACCANTO al suo livello di caso, piu' R@10 tracciata
# ===========================================================================================

def train(make_model, esempi, tau, epochs, nome, bench_track=None, track_every=5):
    torch.manual_seed(SEED)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(SEED)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n--- {nome}  (tau={tau}, {epochs} epoche, {n_par} parametri, "
          f"{len(esempi)} esempi) ---", flush=True)
    if bench_track is not None:
        print(f"{'epoca':>6}{'loss':>9}{'caso':>8}{'divario':>9}"
              f"{'|delta|/|r|':>13}{'gamma':>8}{'R@10 fit':>10}")

    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(esempi), generator=gen)
        tot_l, tot_c, tot_d, nb = 0.0, 0.0, 0.0, 0
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
            tot_d += (delta.norm(dim=-1) / r.norm(dim=-1)).mean().item()

        if bench_track is not None and ((ep + 1) % track_every == 0 or ep == epochs - 1):
            model.eval()
            r10 = macro(evaluate(make_builder(model), bench_track, E_tr, K_VALUES)[10]) * 100
            print(f"{ep + 1:>6}{tot_l / nb:>9.3f}{tot_c / nb:>8.3f}"
                  f"{tot_c / nb - tot_l / nb:>9.3f}{tot_d / nb:>13.4f}"
                  f"{model.gamma.item():>8.3f}{r10:>9.2f}%", flush=True)
    model.eval()
    return model


# ===========================================================================================
# FASE 1 — scelta della temperatura, su query di addestramento tenute da parte
# ===========================================================================================
print("\n" + "=" * 92)
print("FASE 1 — SCELTA DELLA TEMPERATURA")
print("La griglia si valuta su query di addestramento MAI viste nel fit. Il set di")
print("sviluppo non viene toccato: la scelta non consuma budget statistico.")
print("=" * 92)

base_sel = macro(evaluate(build_baseline, BENCH_SEL, E_tr, K_VALUES)[10]) * 100
print(f"\nRiferimento: baseline sulle query di selezione = {base_sel:.2f}% di R@10")

BEST = {}
for nome, ctor in (("A concat", Concat), ("A+LayerNorm", ConcatLN)):
    print(f"\n### {nome}")
    print(f"{'tau':>8}{'loss finale':>14}{'caso':>8}{'divario':>9}{'R@10 selezione':>17}")
    risultati = {}
    for tau in TAUS:
        m = train(ctor, EX_FIT, tau, EPOCHS_SEL, f"{nome} tau={tau}")
        r10 = macro(evaluate(make_builder(m), BENCH_SEL, E_tr, K_VALUES)[10]) * 100
        risultati[tau] = r10
        print(f"{tau:>8}{'(vedi sopra)':>14}{'':>8}{'':>9}{r10:>16.2f}%", flush=True)
    BEST[nome] = max(risultati, key=risultati.get)
    print(f"  -> temperatura scelta per {nome}: {BEST[nome]} "
          f"(R@10 selezione {risultati[BEST[nome]]:.2f}% contro baseline {base_sel:.2f}%)")

# ===========================================================================================
# FASE 2 — addestramento finale, piu' lungo, su tutte le query di addestramento
# ===========================================================================================
print("\n" + "=" * 92)
print(f"FASE 2 — ADDESTRAMENTO FINALE ({EPOCHS_FINAL} epoche, tutte le query)")
print("Colonne: loss e il suo LIVELLO DI CASO ln(N/|GT|). Se il divario resta ~0 il")
print("modello non impara. R@10 fit e' il segnale che conta: se la loss scende ma la")
print("Recall no, l'obiettivo non e' allineato alla metrica.")
print("=" * 92)

EX_ALL = make_examples(train_bench)
MOD_A = train(Concat, EX_ALL, BEST["A concat"], EPOCHS_FINAL, "A: concatenazione",
              bench_track=BENCH_SEL)
MOD_LN = train(ConcatLN, EX_ALL, BEST["A+LayerNorm"], EPOCHS_FINAL, "A+LayerNorm",
               bench_track=BENCH_SEL)
MOD_B = MOD_A   # non riaddestrato: B e' gia' misurato, qui interessa solo A vs A+LN

# ===========================================================================================
# SALVATAGGIO DEI PESI
# ===========================================================================================
# Serve per gli esperimenti di caratterizzazione (docs/piano-interpretabilita.md): senza,
# ogni analisi del modello costa un riaddestramento completo invece di pochi secondi.
CKPT = ROOT / "contributo-candidato" / "checkpoints"
CKPT.mkdir(exist_ok=True)
torch.save({
    "A": MOD_A.state_dict(),
    "B": MOD_B.state_dict(), "LN": MOD_LN.state_dict(),
    "seed": SEED,
    "tau": {"A": BEST["A concat"], "B": BEST["A concat"]},
    "config": {"hidden": HIDDEN, "n_heads": N_HEADS, "d": D, "n_attr": N_ATTR},
    "dirs_text": d_ens,         # le direzioni testuali usate, per riprodurre c
    "attr_names": list(attr_names),
}, CKPT / f"models_LN_seed{SEED}.pt")
print(f"\nPesi salvati in {CKPT / f'models_seed{SEED}.pt'}")

# ===========================================================================================
# CONTROLLO POSITIVO — se fallisce, il dev set NON si interpreta
# ===========================================================================================
print("\n" + "=" * 92)
print("CONTROLLO POSITIVO — i modelli battono la baseline sulle query di ADDESTRAMENTO?")
print("=" * 92)
pc = {"baseline": evaluate(build_baseline, train_bench, E_tr, K_VALUES),
      "A concat": evaluate(make_builder(MOD_A), train_bench, E_tr, K_VALUES),
      "B cross-attn": evaluate(make_builder(MOD_B), train_bench, E_tr, K_VALUES)}
for n, p in pc.items():
    print(f"  {n:<14} R@10 (train) = {macro(p[10]) * 100:6.2f}%")
b_tr = macro(pc["baseline"][10])
ok_pc = all(macro(pc[m][10]) > b_tr for m in ("A concat", "B cross-attn"))
print(f"\n  ESITO: {'superato — si procede' if ok_pc else 'FALLITO'}")
if not ok_pc:
    print("\n  Il setup non impara nemmeno sui dati di addestramento. Come stabilito prima")
    print("  di eseguire, il set di sviluppo NON viene interpretato: qualunque numero")
    print("  misurerebbe un addestramento rotto, non la cross-attention.")

# ===========================================================================================
# VALUTAZIONE SUL SET DI SVILUPPO
# ===========================================================================================
print("\n" + "=" * 92)
print("VALUTAZIONE SUL SET DI SVILUPPO")
print("=" * 92)
METODI = {"baseline (r + d)": build_baseline,
          "A+LayerNorm": make_builder(MOD_LN),
          "A: concatenazione": make_builder(MOD_A),
          "[PRIMARIA] B: cross-attention": make_builder(MOD_B)}
ps = {n: evaluate(f, devset, E_valid, K_VALUES) for n, f in METODI.items()}

BASE = "baseline (r + d)"
b10 = macro(ps[BASE][10])
print(f"\nControllo di implementazione: baseline attesa 19.91%, ottenuta {b10 * 100:.2f}%  "
      f"{'OK' if abs(b10 * 100 - 19.91) < 0.05 else '<-- DISCREPANZA, non fidarsi'}")

print(f"\n{'metodo':<32}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'diff':>10}{'IC 95%':>18}{'p':>9}")
for n in METODI:
    r = {k: macro(ps[n][k]) for k in K_VALUES}
    riga = f"{n:<32}{r[1]*100:>7.2f}%{r[5]*100:>7.2f}%{r[10]*100:>7.2f}%"
    if n != BASE:
        st = bootstrap_paired(ps[n][10], ps[BASE][10], n_boot=20000, seed=0)
        ic = "[{:+.2f},{:+.2f}]".format(st["lo"] * 100, st["hi"] * 100)
        riga += f"{st['delta']*100:>+9.2f}pp{ic:>18}{st['p']:>9.4f}"
    print(riga)

PRIM = "[PRIMARIA] B: cross-attention"
st = bootstrap_paired(ps[PRIM][10], ps[BASE][10], n_boot=20000, seed=0)
passa = st["p"] < ALPHA_SIG and st["delta"] * 100 >= MIN_EFFECT_PP
print(f"\nVERDETTO PRIMARIA: {st['delta']*100:+.2f}pp, p={st['p']:.4f} -> "
      f"{'SUPERATA' if passa else 'non superata'}"
      f"{'' if ok_pc else '   [NON INTERPRETABILE: controllo positivo fallito]'}")

sem = [i for i, e in enumerate(devset) if len(e["queried"]) == 1]
com = [i for i, e in enumerate(devset) if len(e["queried"]) > 1]
print("\n" + "=" * 92)
print(f"VERIFICA DELLA PREVISIONE — {len(sem)} semplici, {len(com)} composte")
print("Con UNA condizione non c'e' nulla su cui attendere: se B porta qualcosa deve")
print("portarlo sulle composte.")
print("=" * 92)
print(f"{'confronto':<34}{'sottoinsieme':<12}{'diff':>10}{'IC 95%':>18}{'p':>9}")
for etichetta, a, rif in (("B vs baseline", PRIM, BASE),
                          ("A vs baseline", "A: concatenazione", BASE),
                          ("B vs A (l'attenzione in se')", PRIM, "A: concatenazione")):
    for et, idx in (("semplici", sem), ("composte", com)):
        s = bootstrap_paired([ps[a][10][i] for i in idx], [ps[rif][10][i] for i in idx],
                             n_boot=20000, seed=0)
        ic = "[{:+.2f},{:+.2f}]".format(s["lo"] * 100, s["hi"] * 100)
        print(f"{etichetta:<34}{et:<12}{s['delta']*100:>+9.2f}pp{ic:>18}{s['p']:>9.4f}")


# ===========================================================================================
# LA DOMANDA DELL'ESPERIMENTO: la non linearita' serve?
# ===========================================================================================
print("\n" + "=" * 92)
print("A+LayerNorm vs A — la non linearita' aggiunge qualcosa?")
print("=" * 92)
ps_a  = evaluate(make_builder(MOD_A),  devset, E_valid, K_VALUES)
ps_ln = evaluate(make_builder(MOD_LN), devset, E_valid, K_VALUES)
for k in K_VALUES:
    b = bootstrap_paired(ps_ln[k], ps_a[k], n_boot=20000, seed=0)
    st = "  significativo" if b["p"] < 0.05 else ""
    print(f"  R@{k:<4}{b['delta']*100:+6.2f}pp  IC [{b['lo']*100:+.2f},{b['hi']*100:+.2f}]  "
          f"p={b['p']:.4f}{st}")
print("\n  se non significativo -> il compito e' LINEARE: la rete non stava sprecando")
print("  capacita', non ce n'era bisogno, e la mappa esplicita e' la soluzione.")
