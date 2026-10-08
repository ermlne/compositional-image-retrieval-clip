"""Lo stesso modello funziona anche su OVAD? Valutato su query MAI VISTE in addestramento."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import parse_query_string
from src.devset import sample_queries, bootstrap_paired, macro

OV = torch.load(ROOT / "embeddings/ovad_clip_vitb32.pt", map_location="cpu")
E = OV["embeddings"].float(); E = E / E.norm(dim=-1, keepdim=True)
L = OV["labels"]; nomi = list(OV["attr_names"])
FR = torch.load(ROOT / "esperimenti/frecce_tre_domini.pt", map_location="cpu")
pos_ov = {a: i for i, a in enumerate(FR["attr"]) if FR["dom"][i] == "ovad"}
D = E.shape[1]
d_ov = torch.zeros(len(nomi), D); usabili = []
for j, n in enumerate(nomi):
    if n in pos_ov:
        d_ov[j] = FR["T"][pos_ov[n]]
        noti = L[:, j] != -1
        if int(((L[:, j] == 1) & noti).sum()) >= 50 and int(((L[:, j] == 0) & noti).sum()) >= 50:
            usabili.append(j)
idx = {n: i for i, n in enumerate(nomi)}
nomi_us = [nomi[j] for j in usabili]

def bench(seed, max_src=40):
    qs = []
    for n_cond, quanti in {1: 60, 2: 60, 3: 30}.items():
        qs += sample_queries(nomi_us, quanti, n_cond, exclude=qs, seed=seed + n_cond)
    rng = torch.Generator().manual_seed(seed)
    out = []
    for q in qs:
        p_, n_ = parse_query_string(q, idx)
        noti = torch.ones(len(E), dtype=torch.bool)
        for a in list(p_) + list(n_): noti &= (L[:, a] != -1)
        ok, viola = noti.clone(), noti.clone()
        for a in p_: ok &= (L[:, a] == 1); viola &= (L[:, a] == 0)
        for b in n_: ok &= (L[:, b] == 0); viola &= (L[:, b] == 1)
        gt = torch.nonzero(ok).flatten(); src = torch.nonzero(viola).flatten()
        if len(gt) < 5 or len(src) < 5: continue
        if len(src) > max_src: src = src[torch.randperm(len(src), generator=rng)[:max_src]]
        u = torch.zeros(3, D); m = torch.zeros(3, dtype=torch.bool); k = 0
        for a in list(p_)[:3]: u[k] = d_ov[a]; m[k] = True; k += 1
        for b in list(n_)[:3-k]: u[k] = -d_ov[b]; m[k] = True; k += 1
        out.append({"src": src, "gt": set(gt.tolist()), "u": u, "m": m, "q": q})
    return out

# 2026 = quello usato in addestramento;  99 = query MAI VISTE
B_train, B_nuovo = bench(2026), bench(99)
condivise = {e["q"] for e in B_train} & {e["q"] for e in B_nuovo}
B_nuovo = [e for e in B_nuovo if e["q"] not in condivise]
print(f"OVAD: {len(B_train)} query di addestramento, {len(B_nuovo)} query NUOVE "
      f"({len(condivise)} in comune, escluse)")

def evaluate(fun, bench_):
    per = []
    for e in bench_:
        src = e["src"]; rec = np.zeros(len(src))
        for st in range(0, len(src), 256):
            sp = min(st+256, len(src)); b = src[st:sp]
            q = fun(E[b], e)
            sim = q @ E.T
            sim[torch.arange(sp-st), b] = -torch.inf
            for j, r in enumerate(sim.topk(10, dim=-1).indices.tolist()):
                rec[st+j] = float(any(i in e["gt"] for i in r))
        per.append(rec)
    return per

def baseline(ref, e):
    c = (e["u"] * e["m"].unsqueeze(-1)).sum(0)
    q = ref + c
    return q / q.norm(dim=-1, keepdim=True)

def modello(f):
    ck = torch.load(f, map_location="cpu")
    H = ck["config"]["hidden"]
    net = nn.Sequential(nn.Linear(2*D, H), nn.GELU(), nn.Linear(H, D))
    net.load_state_dict({k[4:]: v for k, v in ck["A"].items() if k.startswith("net.")})
    net.eval(); gam = float(ck["A"]["gamma"])
    def f_(ref, e):
        c = (e["u"] * e["m"].unsqueeze(-1)).sum(0).expand(len(ref), -1)
        with torch.no_grad(): dl = net(torch.cat([ref, c], dim=-1))
        q = ref + gam * dl
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f_

CK = ROOT / "contributo-candidato/checkpoints"
print("\n" + "=" * 84)
print("RECALL@10 SU OVAD")
print("=" * 84)
print(f"{'modello':<28}{'query di addestr.':>19}{'query NUOVE':>14}{'p (vs baseline)':>18}")
bt, bn = evaluate(baseline, B_train), evaluate(baseline, B_nuovo)
print(f"{'baseline (r + c)':<28}{macro(bt)*100:>18.2f}%{macro(bn)*100:>13.2f}%{'—':>18}")
for seed in (1, 2):
    for lam in (0, 1):
        f = CK / f"multidominio_seed{seed}_lam{lam}.pt"
        if not f.exists(): continue
        fn = modello(f)
        a, b = evaluate(fn, B_train), evaluate(fn, B_nuovo)
        st = bootstrap_paired(b, bn, n_boot=20000, seed=0)
        eti = f"seed {seed}, lambda={lam}" + (" (mai visto OVAD)" if lam == 0 else "")
        print(f"{eti:<28}{macro(a)*100:>18.2f}%{macro(b)*100:>13.2f}%"
              f"{st['delta']*100:>+11.2f} p={st['p']:.3f}", flush=True)
