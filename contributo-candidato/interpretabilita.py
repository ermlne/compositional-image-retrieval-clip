"""Caratterizzazione del contributo candidato — esperimenti 1-4 di docs/piano-interpretabilita.md

Non riaddestra nulla: lavora sul checkpoint prodotto da cross_attention_v2.py.
Uso:  HF_HUB_OFFLINE=1 python contributo-candidato/interpretabilita.py

ESITO (seed 2026):
  esp. 1  l'82% dell'energia del delta sta FUORI da span(r, c) — il metodo funziona
          lavorando dove gli oracoli non avevano guardato
  esp. 2  il resto e' piu' allineato alle direzioni visive vere del testo grezzo
          (+0.290 contro +0.199): l'MLP corregge le direzioni testuali
  esp. 3  il RESTO da solo vale +3.74pp; la parte DENTRO span(r,c) da sola vale -2.90pp
  esp. 4  R^2 = 0.995: l'MLP e' una mappa LINEARE travestita, e sostituendolo con due
          matrici il risultato non cambia (27.00% contro 27.26%, p=0.09)
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark
from src.devset import bootstrap_paired, macro, evaluate_per_source

ck = torch.load(ROOT / "contributo-candidato/checkpoints/models_seed2026.pt",
                map_location="cpu", weights_only=False)
D, HIDDEN = ck["config"]["d"], ck["config"]["hidden"]
d_ens, names = ck["dirs_text"], ck["attr_names"]

class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, D))
        self.gamma = nn.Parameter(torch.tensor(1.0))
    def delta(self, r, c):
        return self.net(torch.cat([r, c], dim=-1))

MOD = Concat(); MOD.load_state_dict(ck["A"]); MOD.eval()
print(f"modello caricato | gamma appreso = {MOD.gamma.item():.4f} | tau = {ck['tau']['A']}")

def load_split(split):
    d = torch.load(ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"]

E_val, L_val = load_split("valid")
E_tr, L_tr = load_split("train")
dev = load_benchmark(str(ROOT / "celeba_devset_valid.json"), names)

# concept vector VISIVI dal train (le direzioni "giuste", mai viste dal modello)
V = torch.zeros(len(names), D)
for i in range(len(names)):
    v = E_tr[L_tr[:, i] == 1].mean(0) - E_tr[L_tr[:, i] == 0].mean(0)
    V[i] = v / v.norm()

def cond_of(e):
    c = torch.zeros(D)
    for a in e["pos"]: c = c + d_ens[a]
    for b in e["neg"]: c = c - d_ens[b]
    return c

# ==========================================================================================
print("\n" + "="*88)
print("ESP. 1 — il delta esce da span(r, c)?")
print("="*88)
frac, per_query = [], []
for e in dev:
    c = cond_of(e)
    R = E_val[e["sources"][:200]]
    with torch.no_grad():
        dl = MOD.delta(R, c.expand(len(R), -1))
    # base ortonormale di span(r, c) per ogni reference
    for k in range(len(R)):
        B = torch.stack([R[k], c])
        Q, _ = torch.linalg.qr(B.T)                       # [D, 2]
        proj = Q @ (Q.T @ dl[k])
        rest = dl[k] - proj
        frac.append((rest.norm() / dl[k].norm().clamp(min=1e-9)).item())
    per_query.append(np.mean(frac[-len(R):]))
frac = np.array(frac)
print(f"  ||resto|| / ||delta||   media {frac.mean():.3f}   mediana {np.median(frac):.3f}")
print(f"                          5o-95o percentile [{np.percentile(frac,5):.3f}, {np.percentile(frac,95):.3f}]")
print(f"  energia fuori da span(r,c): {100*(frac**2).mean():.1f}%")
print(f"\n  lettura: <20% = resta nel piano vecchio | >50% = il metodo funziona uscendone")

# ==========================================================================================
print("\n" + "="*88)
print("ESP. 2 — il resto assomiglia alle direzioni VISIVE?")
print("="*88)
print(f"{'query':<42}{'cos(resto, V)':>14}{'cos(testo, V)':>15}{'guadagno':>10}")
gain, cos_rest, cos_text = [], [], []
for e in [x for x in dev if len(x["queried"]) == 1][:14]:
    i = (e["pos"] + e["neg"])[0]
    sgn = 1.0 if e["pos"] else -1.0
    c = cond_of(e)
    R = E_val[e["sources"][:200]]
    with torch.no_grad():
        dl = MOD.delta(R, c.expand(len(R), -1))
    B = torch.stack([R.mean(0)/R.mean(0).norm(), c/c.norm()])
    Q, _ = torch.linalg.qr(B.T)
    rest = dl - (dl @ Q) @ Q.T
    rm = rest.mean(0); rm = rm / rm.norm()
    cr = (rm @ (sgn*V[i])).item()
    ct = ((c/c.norm()) @ (sgn*V[i])).item()
    cos_rest.append(cr); cos_text.append(ct); gain.append(cr-ct)
    print(f"{e['query']:<42}{cr:>+14.3f}{ct:>+15.3f}{cr-ct:>+10.3f}")
print(f"\n  media: resto {np.mean(cos_rest):+.3f}  testo {np.mean(cos_text):+.3f}  "
      f"guadagno {np.mean(gain):+.3f}")
print("  lettura: se il resto e' piu' allineato alla direzione visiva del testo grezzo,")
print("  l'MLP ha imparato la traduzione testo -> visivo.")

# ==========================================================================================
print("\n" + "="*88)
print("ESP. 6 — attributi vicini nel testo ricevono delta vicini?")
print("="*88)
singles = [x for x in dev if len(x["queried"]) == 1]
idx, deltas, texts = [], [], []
ref_fixed = E_val[dev[0]["sources"][:256]]
for e in singles:
    i = (e["pos"] + e["neg"])[0]; sgn = 1.0 if e["pos"] else -1.0
    c = cond_of(e)
    with torch.no_grad():
        dl = MOD.delta(ref_fixed, c.expand(len(ref_fixed), -1)).mean(0)
    idx.append(i); deltas.append(dl/dl.norm()); texts.append(c/c.norm())
Dl, Tx = torch.stack(deltas), torch.stack(texts)
Gd, Gt = (Dl @ Dl.T).numpy(), (Tx @ Tx.T).numpy()
iu = np.triu_indices(len(idx), k=1)
r = np.corrcoef(Gt[iu], Gd[iu])[0, 1]
print(f"  coppie confrontate: {len(iu[0])}  (stessa reference, sole query a 1 condizione)")
print(f"  correlazione fra geometria dei PROMPT e geometria dei DELTA: r = {r:+.3f}")
print(f"  coseno medio fra delta: {Gd[iu].mean():+.3f}   fra prompt: {Gt[iu].mean():+.3f}")
print("  lettura: r alto = l'MLP legge le coordinate; r ~ 0 = tratta gli attributi")
print("  come simboli indipendenti, cioe' lookup.")



K = [1, 5, 10]
GAMMA = MOD.gamma.item()


def decompose(r, c, dl):
    """delta = comp lungo r + comp lungo c + resto (base ortonormale di span(r,c))."""
    B = torch.stack([r, c.expand_as(r)], dim=1)            # [B, 2, D]
    Q, _ = torch.linalg.qr(B.transpose(1, 2))              # [B, D, 2]
    coef = torch.einsum('bdk,bd->bk', Q, dl)
    proj = torch.einsum('bdk,bk->bd', Q, coef)
    return proj, dl - proj, Q, coef

def make(mode):
    """mode decide quale pezzo del delta si tiene."""
    cache = {}
    def build(ref, entry):
        key = entry["query"]
        if key not in cache: cache[key] = cond_of(entry)
        c = cache[key]
        with torch.no_grad():
            dl = MOD.delta(ref, c.expand(len(ref), -1))
            proj, rest, Q, coef = decompose(ref, c, dl)
            if   mode == "completo":   use = dl
            elif mode == "solo_resto": use = rest
            elif mode == "solo_span":  use = proj
            elif mode == "solo_c":     use = torch.einsum('bd,bd->b', dl, (c/c.norm()).expand_as(ref)).unsqueeze(-1) * (c/c.norm())
            elif mode == "solo_r":     use = torch.einsum('bd,bd->b', dl, ref).unsqueeze(-1) * ref
            elif mode == "nessuno":    use = torch.zeros_like(dl)
            q = ref + GAMMA * use
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return build

def build_baseline(ref, entry):
    q = ref + cond_of(entry)
    return q / q.norm(dim=-1, keepdim=True)

print("="*88); print("ESP. 3 — ablazione causale: quale pezzo del delta porta il guadagno")
print("="*88)
ps = {"baseline (r + c)": evaluate_per_source(build_baseline, dev, E_val, K)}
for m, lab in [("completo","delta completo (= il metodo)"), ("solo_resto","solo il RESTO (fuori da span)"),
               ("solo_span","solo la parte DENTRO span(r,c)"), ("solo_c","solo la componente lungo c"),
               ("solo_r","solo la componente lungo r"), ("nessuno","nessuno (= solo reference)")]:
    ps[lab] = evaluate_per_source(make(m), dev, E_val, K)
print(f"{'variante':<36}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
for k, v in ps.items(): print(f"{k:<36}" + "".join(f"{macro(v[j])*100:>7.2f}%" for j in K))

print("\ndifferenze accoppiate contro la baseline (R@10):")
for k in list(ps)[1:]:
    b = bootstrap_paired(ps[k][10], ps["baseline (r + c)"][10], n_boot=20000, seed=0)
    st = "  significativo" if b["p"] < 0.05 else ""
    print(f"  {k:<36}{b['delta']*100:+7.2f}pp  IC [{b['lo']*100:+.2f},{b['hi']*100:+.2f}]  p={b['p']:.4f}{st}")

print("\n" + "="*88); print("ESP. 4 — l'MLP e' una mappa lineare travestita?")
print("="*88)
X, Y = [], []
g = torch.Generator().manual_seed(0)
for e in dev:
    c = cond_of(e); R = E_val[e["sources"][:120]]
    with torch.no_grad(): dl = MOD.delta(R, c.expand(len(R), -1))
    X.append(torch.cat([R, c.expand(len(R), -1)], 1)); Y.append(dl)
X, Y = torch.cat(X), torch.cat(Y)
n = len(X); idx = torch.randperm(n, generator=g)
tr, te = idx[:int(.8*n)], idx[int(.8*n):]
Xtr = torch.cat([X[tr], torch.ones(len(tr), 1)], 1)
W = torch.linalg.lstsq(Xtr, Y[tr]).solution
Xte = torch.cat([X[te], torch.ones(len(te), 1)], 1)
pred = Xte @ W
ss_res = ((Y[te]-pred)**2).sum(); ss_tot = ((Y[te]-Y[te].mean(0))**2).sum()
r2 = (1 - ss_res/ss_tot).item()
cos = torch.nn.functional.cosine_similarity(pred, Y[te], dim=-1)
print(f"  esempi: {n} ({len(tr)} fit / {len(te)} test)")
print(f"  R^2 su dati mai usati per il fit: {r2:.4f}")
print(f"  coseno medio fra delta vero e delta lineare: {cos.mean():.4f} (mediana {cos.median():.4f})")
print(f"\n  lettura: R^2 > 0.9 = l'MLP calcola qualcosa di lineare, e W si puo' scrivere.")
print(f"           R^2 < 0.5 = la non linearita' e' essenziale (regola di stop del piano).")
torch.save({"W": W, "r2": r2}, ROOT / "contributo-candidato/checkpoints/linear_map.pt")
print(f"\nMappa lineare salvata in contributo-candidato/checkpoints/linear_map.pt")
