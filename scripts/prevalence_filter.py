"""Filtro a soglie calibrate sulla prevalenza (dal TRAIN split, disgiunto da
valid/test) con fattore di slack s: per +i si tengono i top s*p_i del corpus
sul punteggio dell'attributo, per -j si esclude il top s*p_j.
Ranking fra i sopravvissuti: query baseline. Sweep di s sul devset.

Inoltre: variante SOFT (penalita' hinge sul percentile, niente filtro duro),
sweep di lambda.
"""
import sys
import numpy as np
import torch

sys.path.append(".")
from src.benchmark import load_benchmark
from src.devset import macro, bootstrap_paired
from transformers import CLIPModel, CLIPTokenizer

K_VALUES = [1, 5, 10]
MODEL_ID = "openai/clip-vit-base-patch32"

def load_split(split):
    d = torch.load(f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"], d

E_valid, L_valid, _ = load_split("valid")
_, L_train, _ = load_split("train")
_, _, meta = load_split("test")
attr_names = [n for n in meta["attr_names"] if n.strip()]
D = E_valid.shape[1]
prev = (L_train == 1).float().mean(0)                     # prevalenza per attributo

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
clip_model = CLIPModel.from_pretrained(MODEL_ID).eval()
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)

def encode(prompts):
    with torch.no_grad():
        out = clip_model.get_text_features(**tokenizer(prompts, padding=True,
                                                       return_tensors="pt"))
    return (out.pooler_output if hasattr(out, "pooler_output") else out).float()

d_ens = torch.zeros(len(attr_names), D)
for i, name in enumerate(attr_names):
    clean = name.replace("_", " ").lower()
    pos = encode([p.format(clean) for p, _ in TEMPLATES])
    neg = encode([n.format(clean) for _, n in TEMPLATES])
    pos = pos / pos.norm(dim=-1, keepdim=True)
    neg = neg / neg.norm(dim=-1, keepdim=True)
    v = pos.mean(0) - neg.mean(0)
    d_ens[i] = v / v.norm()
print("direzioni pronte", flush=True)

def build_baseline(ref, entry):
    d = torch.zeros(D)
    for i in entry["pos"]: d = d + d_ens[i]
    for j in entry["neg"]: d = d - d_ens[j]
    q = ref + d
    return q / q.norm(dim=-1, keepdim=True)

scores = (E_valid @ d_ens.T).numpy()
ranks = scores.argsort(0).argsort(0).astype(np.float32)
pct = torch.from_numpy(ranks.T / (len(E_valid) - 1))      # [40, N]

def eval_set(bench, banned_fn=None, penalty_fn=None):
    per_source = {k: [] for k in K_VALUES}
    k_max = max(K_VALUES)
    for entry in bench:
        banned = banned_fn(entry) if banned_fn else torch.zeros(len(E_valid), dtype=torch.bool)
        pen = penalty_fn(entry) if penalty_fn else None
        sources = entry["sources"]
        n_src = len(sources)
        rec = {k: np.zeros(n_src) for k in K_VALUES}
        for start in range(0, n_src, 2048):
            stop = min(start + 2048, n_src)
            batch_src = sources[start:stop]
            ref = E_valid[batch_src]
            sim = build_baseline(ref, entry) @ E_valid.T
            if pen is not None:
                sim = sim + pen.unsqueeze(0)
            sim[:, banned] = -torch.inf
            sim[torch.arange(stop - start), batch_src] = -torch.inf
            top = sim.topk(k_max, dim=-1).indices.tolist()
            for j, retrieved in enumerate(top):
                gt = entry["gt"][start + j]
                hits = np.array([idx in gt for idx in retrieved], dtype=bool)
                cum = hits.cumsum()
                for k in K_VALUES:
                    rec[k][start + j] = float(cum[k - 1] > 0)
        for k in K_VALUES:
            per_source[k].append(rec[k])
    return per_source

bench = load_benchmark("celeba_devset_valid.json", attr_names)
ps_base = eval_set(bench)
print(f"baseline R@10 {macro(ps_base[10])*100:.2f}%\n", flush=True)

print("filtro duro a soglia di prevalenza, slack s (ranking = query baseline):")
for s in [1.5, 2.0, 3.0, 5.0, 8.0]:
    def banned_fn(entry, s=s):
        keep = torch.ones(len(E_valid), dtype=torch.bool)
        for i in entry["pos"]:
            keep &= pct[i] >= 1 - min(float(prev[i]) * s, 0.98)
        for j in entry["neg"]:
            keep &= pct[j] <= 1 - min(float(prev[j]), 0.9) / max(s / 2, 1)
        return ~keep
    ps = eval_set(bench, banned_fn=banned_fn)
    r = bootstrap_paired(ps[10], ps_base[10], n_boot=20000, seed=0)
    print(f"  s={s:>4}  R@10 {macro(ps[10])*100:6.2f}%  vs baseline {r['delta']*100:+6.2f}pp "
          f"IC [{r['lo']*100:+.2f},{r['hi']*100:+.2f}] p={r['p']:.4f}", flush=True)

print("\npenalita' SOFT: sim + lambda * somma_c log(pct condizione c):")
for lam in [0.01, 0.02, 0.05, 0.1, 0.2]:
    def penalty_fn(entry, lam=lam):
        pen = torch.zeros(len(E_valid))
        eps = 1e-3
        for i in entry["pos"]:
            pen = pen + torch.log(pct[i].clamp(min=eps))
        for j in entry["neg"]:
            pen = pen + torch.log((1 - pct[j]).clamp(min=eps))
        return lam * pen
    ps = eval_set(bench, penalty_fn=penalty_fn)
    r = bootstrap_paired(ps[10], ps_base[10], n_boot=20000, seed=0)
    print(f"  lam={lam:<5} R@10 {macro(ps[10])*100:6.2f}%  vs baseline {r['delta']*100:+6.2f}pp "
          f"IC [{r['lo']*100:+.2f},{r['hi']*100:+.2f}] p={r['p']:.4f}", flush=True)
