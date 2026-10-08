"""L'MLP guadagna perche' corregge le DIREZIONI o perche' si adatta alla FOTO di partenza?

Si congela il modello addestrato e si sostituisce la reference in ingresso all'MLP con una
reference MEDIA fissa, lasciando invariato tutto il resto (la delta si somma sempre alla
reference vera). Se il guadagno sopravvive, il metodo corregge le direzioni; se crolla, il
metodo adatta la query alla foto.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import load_benchmark
from src.devset import bootstrap_paired, macro

ck = torch.load(ROOT / "contributo-candidato/checkpoints/models_seed2026.pt", map_location="cpu")
D, H = ck["config"]["d"], ck["config"]["hidden"]
net = nn.Sequential(nn.Linear(2*D, H), nn.GELU(), nn.Linear(H, D))
net.load_state_dict({k[4:]: v for k, v in ck["A"].items() if k.startswith("net.")})
net.eval()
gamma = float(ck["A"]["gamma"])
d_ens = ck["dirs_text"]
attr_names = list(ck["attr_names"])
print(f"modello caricato: gamma={gamma:.3f}, {len(attr_names)} attributi")

d = torch.load(ROOT / "embeddings/celeba_valid_clip_vitb32.pt", map_location="cpu")
E = d["embeddings"].float(); E = E / E.norm(dim=-1, keepdim=True)
devset = load_benchmark(str(ROOT / "celeba_devset_valid.json"), attr_names)
print(f"dev set: {len(devset)} query")

dtr = torch.load(ROOT / "embeddings/celeba_train_clip_vitb32.pt", map_location="cpu")
Etr = dtr["embeddings"].float(); Etr = Etr / Etr.norm(dim=-1, keepdim=True)
rbar = Etr.mean(0); rbar = rbar / rbar.norm()
del dtr, Etr

def cond(e):
    c = torch.zeros(D)
    for a in e["pos"]: c = c + d_ens[a]
    for b in e["neg"]: c = c - d_ens[b]
    return c

def evaluate(build, bench, emb, ks=(1,5,10), bs=512):
    per = {k: [] for k in ks}; kmax = max(ks)
    for e in bench:
        src = e["sources"]; rec = {k: np.zeros(len(src)) for k in ks}
        for st in range(0, len(src), bs):
            sp = min(st+bs, len(src)); b = src[st:sp]
            q = build(emb[b], e)
            sim = q @ emb.T
            sim[torch.arange(sp-st), b] = -torch.inf
            for j, r in enumerate(sim.topk(kmax, dim=-1).indices.tolist()):
                gt = e["gt"][st+j]
                cum = np.array([i in gt for i in r], dtype=bool).cumsum()
                for k in ks: rec[k][st+j] = float(cum[k-1] > 0)
        for k in ks: per[k].append(rec[k])
    return per

def baseline(ref, e):
    q = ref + cond(e)
    return q / q.norm(dim=-1, keepdim=True)

def modello(usa_ref_vera):
    cache = {}
    def f(ref, e):
        k = e["query"]
        if k not in cache: cache[k] = cond(e)
        c = cache[k].expand(len(ref), -1)
        r_in = ref if usa_ref_vera else rbar.expand(len(ref), -1)
        with torch.no_grad():
            dl = net(torch.cat([r_in, c], dim=-1))
        q = ref + gamma * dl                      # la delta si somma SEMPRE alla ref vera
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f

print("\n" + "=" * 78)
print("ABLAZIONE DELLA DIPENDENZA DALLA REFERENCE (Recall@10)")
print("=" * 78)
pb = evaluate(baseline, devset, E)
pm = evaluate(modello(True), devset, E)
pa = evaluate(modello(False), devset, E)
for k in (1, 5, 10):
    b, m, a = macro(pb[k])*100, macro(pm[k])*100, macro(pa[k])*100
    sm = bootstrap_paired(pm[k], pb[k], n_boot=20000, seed=0)
    sa = bootstrap_paired(pa[k], pb[k], n_boot=20000, seed=0)
    sd = bootstrap_paired(pa[k], pm[k], n_boot=20000, seed=0)
    print(f"\nRecall@{k}")
    print(f"  baseline                      {b:6.2f}%")
    print(f"  modello completo              {m:6.2f}%   {sm['delta']*100:+6.2f}pp vs baseline (p={sm['p']:.4f})")
    print(f"  senza dipendenza dalla ref    {a:6.2f}%   {sa['delta']*100:+6.2f}pp vs baseline (p={sa['p']:.4f})")
    print(f"                                          {sd['delta']*100:+6.2f}pp vs completo (p={sd['p']:.4f})")
    if k == 10:
        quota = 100*(1 - (a-b)/(m-b)) if m > b else float('nan')
        print(f"\n  -> la dipendenza dalla reference vale il {quota:.0f}% del guadagno")
