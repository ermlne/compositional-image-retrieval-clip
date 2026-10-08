"""CROSS-ATTENTION per la fusione delle condizioni — il suggerimento mai esplorato.

L'assignment (sez. 3, punto 2) elenca quattro strade: cross-attention, gating, teste di
proiezione non lineari, allineamento contrastivo. Delle quattro, il gruppo ha gia' provato
e chiuso il gating (PhiDynamic, pesi per condizione) e l'allineamento contrastivo
(InfoNCE). Cross-attention e teste non lineari che producano un EMBEDDING (e non solo
scalari) non sono mai state scritte.

COSA VA ISOLATO, E PERCHE'. La cross-attention fa due cose insieme:

  1. decide quanto pesare ciascuna condizione in funzione della reference (i pesi di
     attenzione);
  2. TRASFORMA le condizioni prima di combinarle (la proiezione dei value).

Il punto 1 e' gia' chiuso: STATO.md, riga "pesi per condizione, per reference" — margine
apparente +3.8pp, margine reale +0.1pp NON significativo, misurato con oracolo onesto.
Anche scegliendo i pesi conoscendo meta' del ground truth non si guadagna nulla.

Il punto 2 non e' coperto da nessun oracolo: la proiezione dei value permette all'uscita di
stare FUORI dallo span delle direzioni date, cioe' di essere una direzione nuova e non una
loro ricombinazione. L'esperimento e' quindi costruito per isolare il punto 2, e il
confronto informativo non e' "attenzione contro baseline" ma ATTENZIONE CONTRO
CONCATENAZIONE A PARITA' DI CAPACITA'.

LA SCALA, tre gradini, una sola implementazione parametrizzata:

  0  baseline  q = normalize(r + d)              controllo di implementazione: 19.91%
  A  MLP con concatenazione, condizionato su r   serve una trasformazione appresa?
  B  cross-attention, stessa capacita' di A      serve ATTENDERE, o basta concatenare?

  Ipotesi PRIMARIA: B contro la baseline. Un solo confronto, alpha=0.05, effetto minimo
  +0.5pp. A e' ablazione descrittiva: dice se l'eventuale merito e' dell'attenzione o
  della semplice capacita' in piu'.

PREVISIONE DEPOSITATA PRIMA DI ESEGUIRE. Se la cross-attention porta qualcosa, deve
portarlo sulle query COMPOSTE (47) e non su quelle semplici (54): con una sola condizione
non c'e' nulla su cui attendere e il meccanismo degenera in una proiezione. Un guadagno
uniforme anche sulle semplici verrebbe dalla capacita' in piu', non dall'attenzione, e va
detto. Mi aspetto inoltre che B non batta la baseline, per la ragione strutturale gia'
misurata (un singolo vettore query puo' solo pesare identita' contro conformita'); se la
batte di piu' di 1pp quell'argomento e' sbagliato.

IL DIFETTO DA NON RIPETERE. Il commit e620b28 documenta che nel primo modulo allenato i
distrattori erano scelti per somiglianza alla reference: dentro quel campione il termine
di identita' ANTICORRELAVA con la correttezza, e la rete imparava (giustamente) a buttarlo
via. Qui il denominatore dell'InfoNCE e' sull'INTERO corpus di addestramento, mai su un
campione selezionato per somiglianza.

CONTROLLO POSITIVO prima di qualunque conclusione: il modello deve battere la baseline
sulle query di ADDESTRAMENTO. Se non impara nemmeno li', il resto non si interpreta.

Il benchmark ufficiale NON viene toccato. Valutazione su celeba_devset_valid.json.

Uso: python experiments/cross_attention.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark, build_benchmark_from_labels
from src.devset import bootstrap_paired, macro

MODEL_ID = "openai/clip-vit-base-patch32"
K_VALUES = [1, 5, 10]
SEED = 2026
ALPHA_SIG, MIN_EFFECT_PP = 0.05, 0.5

# --- addestramento (iperparametri fissati a priori) -----------------------
# NOTA SUL DIMENSIONAMENTO. Un primo tentativo con corpus 40000 / 240 query / 120 source
# si e' piantato senza produrre output oltre il caricamento di CLIP (0.03 s di CPU e 0.1 MB
# di RAM dopo 20 minuti: bloccato, non sovraccarico — causa non diagnosticata). I valori
# sono stati dimezzati per rendere il run tracciabile su CPU. Cio' che conta per la
# correttezza NON cambia: il denominatore dell'InfoNCE resta l'INTERO corpus di
# addestramento, mai un campione di distrattori scelti per somiglianza alla reference,
# che e' il difetto documentato in e620b28.
N_TRAIN_CORPUS = 20000      # sottoinsieme del train usato come galleria di addestramento
N_QUERIES_TRAIN = 150       # query generate dal train (stessa composizione del dev set)
MAX_SOURCES = 80            # reference per query, prima del filtro di Hamming
EPOCHS, BATCH, LR, WD = 10, 256, 3e-4, 1e-4
HIDDEN, N_HEADS = 512, 4
TAU = 0.01                  # temperatura InfoNCE, come in src/phi_dynamic.py


def load_split(split):
    d = torch.load(ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"], d


print("Carico embedding...")
E_train_all, L_train_all, meta = load_split("train")
E_valid, L_valid, _ = load_split("valid")
attr_names = [n for n in meta["attr_names"] if n.strip()]
D, N_ATTR = E_valid.shape[1], L_valid.shape[1]
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)
print(f"  dev: {len(devset)} query, {sum(len(e['sources']) for e in devset)} reference")

# --- direzioni testuali (identiche a tutto il resto del progetto) ---------
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

# --- corpus e query di addestramento, dal TRAIN ---------------------------
g = torch.Generator().manual_seed(SEED)
sub = torch.randperm(len(E_train_all), generator=g)[:N_TRAIN_CORPUS].sort().values
E_tr, L_tr = E_train_all[sub], L_train_all[sub]
del E_train_all
print(f"\nCorpus di addestramento: {len(E_tr)} immagini dal train split")

# Si usa sample_queries di src/devset.py invece di generare le query a mano. Un primo
# tentativo scritto qui si e' rivelato un CICLO INFINITO: chiedeva 81 query a una
# condizione, ma con 40 attributi x 2 segni ne esistono solo 80. La funzione del gruppo
# gestisce gia' questo caso — il suo docstring lo documenta esplicitamente ("lo spazio
# delle query a n_cond condizioni e' finito e, per n_cond = 1, piccolo") e limita sia la
# richiesta allo spazio disponibile sia il numero di tentativi.
from src.devset import sample_queries

pool = list(attr_names)
targets = {1: int(N_QUERIES_TRAIN * 0.54),      # stessa composizione del dev set
           2: int(N_QUERIES_TRAIN * 0.32),
           3: int(N_QUERIES_TRAIN * 0.15)}
queries_txt = []
for n_cond, quanti in targets.items():
    got = sample_queries(pool, quanti, n_cond, exclude=queries_txt, seed=SEED + n_cond)
    print(f"  {n_cond} condizione/i: chieste {quanti}, ottenute {len(got)}", flush=True)
    queries_txt += got

print(f"Costruisco il ground truth di {len(queries_txt)} query di addestramento "
      f"(stessa regola dell'assignment, via build_benchmark_from_labels)...")
# Una query alla volta invece che tutte insieme: la chiamata unica non stampa nulla finche'
# non finisce, e un primo tentativo si e' piantato senza lasciare capire dove. Cosi' si
# vede l'avanzamento e, se si blocca, su quale query.
import time
train_bench, t0 = [], time.time()
for qi, qt in enumerate(queries_txt, 1):
    train_bench += build_benchmark_from_labels([qt], L_tr, attr_names,
                                               max_sources=MAX_SOURCES, seed=SEED)
    if qi % 20 == 0 or qi == len(queries_txt):
        el = time.time() - t0
        print(f"    {qi:3d}/{len(queries_txt)} query  ({el:5.1f}s, "
              f"stimate {el / qi * len(queries_txt):5.1f}s totali, "
              f"{len(train_bench)} valide finora)", flush=True)
n_pairs = sum(len(e["sources"]) for e in train_bench)
print(f"  {len(train_bench)} query valide, {n_pairs} coppie (reference, target)")

# Appiattisce in esempi: (ref_idx, query_idx, un target scelto a caso fra i suoi gt)
esempi = []
for qi, e in enumerate(train_bench):
    for si, ref in enumerate(e["sources"].tolist()):
        esempi.append((ref, qi, sorted(e["gt"][si])))
print(f"  {len(esempi)} esempi di addestramento")


def cond_tensor(entry, max_n=3):
    """Le condizioni come insieme di vettori CON SEGNO, con maschera di padding."""
    u = torch.zeros(max_n, D)
    m = torch.zeros(max_n, dtype=torch.bool)
    for k, a in enumerate(entry["pos"][:max_n]):
        u[k] = d_ens[a]; m[k] = True
    off = len(entry["pos"][:max_n])
    for k, b in enumerate(entry["neg"][:max_n - off]):
        u[off + k] = -d_ens[b]; m[off + k] = True
    return u, m


TRAIN_U = torch.stack([cond_tensor(e)[0] for e in train_bench])
TRAIN_M = torch.stack([cond_tensor(e)[1] for e in train_bench])


# --- i modelli -------------------------------------------------------------

class Concat(nn.Module):
    """Gradino A. Le condizioni sono SOMMATE (l'ordine non conta) e concatenate a r.
    Nessuna attenzione: e' il partner di confronto a parita' di capacita'."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * D, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(-2.0))   # tanh(-2) ~ -0.96 -> parte piccolo
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, U, M):
        c = (U * M.unsqueeze(-1)).sum(1)                # [B, D] somma delle condizioni
        delta = self.net(torch.cat([r, c], dim=-1))
        q = r + torch.tanh(self.gamma).abs() * delta
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class CrossAttn(nn.Module):
    """Gradino B. La reference fa da QUERY, le condizioni da KEY e VALUE. La proiezione
    dei value e' il pezzo che NON e' coperto dall'oracolo sui pesi: permette all'uscita di
    stare fuori dallo span delle direzioni date."""
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, N_HEADS, batch_first=True)
        self.net = nn.Sequential(nn.Linear(2 * D, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, U, M):
        # key_padding_mask: True = da IGNORARE, quindi il complemento della maschera
        ctx, _ = self.attn(r.unsqueeze(1), U, U, key_padding_mask=~M)
        delta = self.net(torch.cat([r, ctx.squeeze(1)], dim=-1))
        q = r + torch.tanh(self.gamma).abs() * delta
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def build_baseline(ref, entry):
    d = torch.zeros(D)
    for a in entry["pos"]: d = d + d_ens[a]
    for b in entry["neg"]: d = d - d_ens[b]
    q = ref + d
    return q / q.norm(dim=-1, keepdim=True)


# --- addestramento ---------------------------------------------------------

def train(model, nome):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gg = torch.Generator().manual_seed(SEED)
    print(f"\nAlleno '{nome}' ({sum(p.numel() for p in model.parameters())} parametri)...")
    for ep in range(EPOCHS):
        model.train()
        order = torch.randperm(len(esempi), generator=gg)
        tot, nb = 0.0, 0
        for st in range(0, len(esempi), BATCH):
            idx = order[st:st + BATCH].tolist()
            refs = torch.tensor([esempi[i][0] for i in idx])
            qis = torch.tensor([esempi[i][1] for i in idx])
            # un positivo a caso fra i ground truth di quella coppia
            pos = torch.tensor([esempi[i][2][int(torch.randint(len(esempi[i][2]), (1,),
                                                               generator=gg))] for i in idx])
            q = model(E_tr[refs], TRAIN_U[qis], TRAIN_M[qis])
            # InfoNCE con denominatore sull'INTERO corpus (mai distrattori scelti per
            # somiglianza alla reference: e' il difetto documentato in e620b28)
            logits = (q @ E_tr.T) / TAU                     # [B, N_corpus]
            loss = nn.functional.cross_entropy(logits, pos)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 3 == 0 or ep == EPOCHS - 1:
            print(f"  epoca {ep + 1:2d}/{EPOCHS}  InfoNCE = {tot / nb:.4f}  "
                  f"|gamma| = {torch.tanh(model.gamma).abs().item():.4f}")
    model.eval()
    return model


def make_builder(model):
    cache = {}
    def f(ref, entry):
        key = entry["query"]
        if key not in cache:
            cache[key] = cond_tensor(entry)
        u, m = cache[key]
        with torch.no_grad():
            return model(ref, u.unsqueeze(0).expand(len(ref), -1, -1),
                         m.unsqueeze(0).expand(len(ref), -1))
    return f


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


torch.manual_seed(SEED)
MOD_A = train(Concat(), "A: concatenazione")
torch.manual_seed(SEED)
MOD_B = train(CrossAttn(), "B: cross-attention")

# --- CONTROLLO POSITIVO: impara sulle query di ADDESTRAMENTO? -------------
print("\n" + "=" * 92)
print("CONTROLLO POSITIVO — il modello batte la baseline sulle query di ADDESTRAMENTO?")
print("Se non impara nemmeno li', il resto non si interpreta.")
print("=" * 92)
pc = {"baseline": evaluate(build_baseline, train_bench, E_tr, K_VALUES),
      "A concat": evaluate(make_builder(MOD_A), train_bench, E_tr, K_VALUES),
      "B cross-attn": evaluate(make_builder(MOD_B), train_bench, E_tr, K_VALUES)}
for nome, p in pc.items():
    print(f"  {nome:<14} R@10 (train) = {macro(p[10]) * 100:6.2f}%")
imparano = all(macro(pc[m][10]) > macro(pc['baseline'][10]) for m in ("A concat", "B cross-attn"))
print(f"\n  {'Entrambi imparano: si procede.' if imparano else 'ATTENZIONE: almeno uno non batte la baseline nemmeno in addestramento.'}")

# --- valutazione sul dev set ----------------------------------------------
METODI = {"baseline (r + d)": build_baseline,
          "A: concatenazione": make_builder(MOD_A),
          "[PRIMARIA] B: cross-attention": make_builder(MOD_B)}
print("\nValuto sul set di sviluppo...")
ps = {n: evaluate(f, devset, E_valid, K_VALUES) for n, f in METODI.items()}

BASE = "baseline (r + d)"
b10 = macro(ps[BASE][10])
print(f"\nControllo di implementazione: baseline attesa 19.91%, ottenuta {b10 * 100:.2f}%  "
      f"{'OK' if abs(b10 * 100 - 19.91) < 0.05 else '<-- DISCREPANZA'}")

print("\n" + "=" * 100)
print(f"RISULTATI — 101 query. Ipotesi primaria: B contro baseline, p < {ALPHA_SIG}, "
      f"effetto minimo +{MIN_EFFECT_PP}pp")
print("=" * 100)
print(f"{'metodo':<32}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'diff':>10}{'IC 95%':>18}{'p':>9}")
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
      f"{'SUPERATA' if passa else 'non superata'}")

# --- la previsione: l'attenzione deve pagare sulle COMPOSTE ---------------
sem = [i for i, e in enumerate(devset) if len(e["queried"]) == 1]
com = [i for i, e in enumerate(devset) if len(e["queried"]) > 1]
print("\n" + "=" * 100)
print(f"VERIFICA DELLA PREVISIONE — {len(sem)} query semplici, {len(com)} composte")
print("Con UNA condizione non c'e' nulla su cui attendere: se B porta qualcosa, deve")
print("portarlo sulle composte. Un guadagno uniforme verrebbe dalla capacita', non")
print("dall'attenzione.")
print("=" * 100)
print(f"{'confronto':<40}{'sottoinsieme':<12}{'diff':>10}{'IC 95%':>18}{'p':>9}")
for nome, a in (("B vs baseline", PRIM), ("A vs baseline", "A: concatenazione"),
                ("B vs A (l'attenzione in se')", PRIM)):
    rif = "A: concatenazione" if "vs A" in nome else BASE
    for et, idx in (("semplici", sem), ("composte", com)):
        s = bootstrap_paired([ps[a][10][i] for i in idx], [ps[rif][10][i] for i in idx],
                             n_boot=20000, seed=0)
        ic = "[{:+.2f},{:+.2f}]".format(s["lo"] * 100, s["hi"] * 100)
        print(f"{nome:<40}{et:<12}{s['delta']*100:>+9.2f}pp{ic:>18}{s['p']:>9.4f}")
