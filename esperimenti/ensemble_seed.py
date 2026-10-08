"""I seed sono d'accordo DENTRO lo span e in disaccordo FUORI?

Se la parte fuori dal sottospazio vincolato e' residuo di inizializzazione, deve CAMBIARE da
seed a seed; la parte imparata deve essere la stessa. E' il test diretto della diagnosi, e non
richiede di riaddestrare niente.

Poi: l'ensemble (media delle uscite, non dei pesi — reti da inizializzazioni diverse stanno in
bacini diversi e mediare i pesi non ha senso) aiuta sul dev set?
"""
import sys, itertools, statistics as st
from pathlib import Path
import numpy as np, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark
from src.devset import bootstrap_paired, macro

CK = ROOT/"contributo-candidato/checkpoints"
modelli, gammas = [], []
for sd in (1, 2, 3, 4):
    f = CK/f"multidominio_seed{sd}_lam0.pt"
    c = torch.load(f, map_location="cpu")
    D, H = c["config"]["d"], c["config"]["hidden"]
    net = nn.Sequential(nn.Linear(2*D, H), nn.GELU(), nn.Linear(H, D))
    net.load_state_dict({k[4:]: v for k, v in c["A"].items() if k.startswith("net.")})
    net.eval(); modelli.append(net); gammas.append(float(c["A"]["gamma"]))
    dt = c["dirs_text"]; nomi = list(c["attr_names"])
dtn = dt/dt.norm(dim=-1, keepdim=True)
D = dtn.shape[1]
print(f"{len(modelli)} modelli (seed 1-4), gamma = {[round(g,3) for g in gammas]}")

d = torch.load(ROOT/"embeddings/celeba_valid_clip_vitb32.pt", map_location="cpu")
E = d["embeddings"].float(); E = E/E.norm(dim=-1, keepdim=True)
R = E[torch.randperm(len(E), generator=torch.Generator().manual_seed(0))[:400]]

# base ortonormale dello span delle 40 direzioni, e il suo complemento
U, S, _ = torch.linalg.svd(dtn.T, full_matrices=False)
Q = U[:, :int((S > 1e-5*S[0]).sum())]
g = torch.Generator().manual_seed(7)
casuali = torch.randn(40, D, generator=g)
fuori = casuali - (casuali @ Q) @ Q.T                       # ortogonali allo span
fuori = fuori/fuori.norm(dim=-1, keepdim=True)

def accordo(dirs):
    cos = []
    with torch.no_grad():
        for a in range(len(dirs)):
            c = dirs[a].expand(len(R), -1)
            us = [m(torch.cat([R, c], dim=-1)) for m in modelli]
            us = [u/u.norm(dim=-1, keepdim=True) for u in us]
            for i, j in itertools.combinations(range(len(us)), 2):
                cos.append(float((us[i]*us[j]).sum(-1).mean()))
    return st.mean(cos)

print("\n" + "=" * 78)
print("ACCORDO FRA I SEED SULLA CORREZIONE PRODOTTA")
print("=" * 78)
a_in, a_out = accordo(dtn), accordo(fuori)
print(f"  su direzioni DENTRO lo span (i 40 attributi):   coseno {a_in:+.4f}")
print(f"  su direzioni FUORI dallo span (casuali, ortog.): coseno {a_out:+.4f}")
print(f"\n  se fuori fosse rumore di inizializzazione, il secondo sarebbe ~0")

# --- ensemble sul dev set ------------------------------------------------------------------
devset = load_benchmark(str(ROOT/"celeba_devset_valid.json"), nomi)
def cond(e):
    c = torch.zeros(D)
    for a in e["pos"]: c = c + dtn[a]
    for b in e["neg"]: c = c - dtn[b]
    return c
def evaluate(fun):
    per = []
    for e in devset:
        src = e["sources"]; rec = np.zeros(len(src))
        for stt in range(0, len(src), 256):
            sp = min(stt+256, len(src)); b = src[stt:sp]
            q = fun(E[b], e)
            sim = q @ E.T; sim[torch.arange(sp-stt), b] = -torch.inf
            for j, r in enumerate(sim.topk(10, dim=-1).indices.tolist()):
                rec[stt+j] = float(any(i in e["gt"][stt+j] for i in r))
        per.append(rec)
    return per
def baseline(ref, e):
    q = ref + cond(e); return q/q.norm(dim=-1, keepdim=True)
def singolo(k):
    def f(ref, e):
        c = cond(e).expand(len(ref), -1)
        with torch.no_grad(): dl = modelli[k](torch.cat([ref, c], dim=-1))
        q = ref + gammas[k]*dl
        return q/q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f
def ensemble(ref, e):
    c = cond(e).expand(len(ref), -1)
    with torch.no_grad():
        dl = torch.stack([g_*m(torch.cat([ref, c], dim=-1))
                          for m, g_ in zip(modelli, gammas)]).mean(0)
    q = ref + dl
    return q/q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

print("\n" + "=" * 78)
print("L'ENSEMBLE AIUTA?  (Recall@10 sul dev set, 101 query)")
print("=" * 78)
pb = evaluate(baseline)
print(f"  baseline{'':<22}{macro(pb)*100:>7.2f}%")
sing = []
for k in range(len(modelli)):
    p = evaluate(singolo(k)); sing.append(p)
    st_ = bootstrap_paired(p, pb, n_boot=20000, seed=0)
    print(f"  seed {k+1} da solo{'':<16}{macro(p)*100:>7.2f}%   {st_['delta']*100:>+6.2f}pp")
import itertools as _it, json
print("\n  --- quanti modelli servono?")
res_k = {}
for k in (2, 3, 4):
    vals = []
    for comb in _it.combinations(range(len(modelli)), k):
        def ens_k(ref, e, comb=comb):
            c = cond(e).expand(len(ref), -1)
            with torch.no_grad():
                dl = torch.stack([gammas[i]*modelli[i](torch.cat([ref, c], dim=-1))
                                  for i in comb]).mean(0)
            q = ref + dl
            return q/q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vals.append(macro(evaluate(ens_k))*100)
    res_k[k] = vals
    print(f"    {k} modelli: media {st.mean(vals):.2f}%  "
          f"(min {min(vals):.2f}, max {max(vals):.2f}, su {len(vals)} combinazioni)",
          flush=True)
# fix 30/08: `media_sing` era usata nel json.dump qui sotto ma definita solo piu' avanti
# (riga "media_sing = ..." dopo `pe = evaluate(ensemble)`), quindi lo script moriva con
# NameError prima di scrivere qualunque risultato. Definita qui, dove serve la prima volta.
media_sing = st.mean(macro(p)*100 for p in sing)
print(f"    1 modello:  media {media_sing:.2f}%")
with open(ROOT/"esperimenti"/"ensemble_quanti.json", "w") as f:
    json.dump({"singolo": media_sing, "per_k": res_k}, f, indent=2)

pe = evaluate(ensemble)
media_sing = st.mean(macro(p)*100 for p in sing)   # ricalcolo innocuo, stesso valore
st_b = bootstrap_paired(pe, pb, n_boot=20000, seed=0)
print(f"\n  media dei singoli{'':<13}{media_sing:>7.2f}%")
print(f"  ENSEMBLE dei 4{'':<16}{macro(pe)*100:>7.2f}%   {st_b['delta']*100:>+6.2f}pp vs baseline")
print(f"  guadagno dell'ensemble sul singolo medio: {macro(pe)*100 - media_sing:+.2f}pp")
for k, p in enumerate(sing):
    s_ = bootstrap_paired(pe, p, n_boot=20000, seed=0)
    print(f"    vs seed {k+1}: {s_['delta']*100:>+6.2f}pp  p={s_['p']:.4f}")
