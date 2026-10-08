"""Versione ZERO-SHOT del filtro condizioni: percentile di corpus sul punteggio
della direzione ensemblata, filtro duro a quantile q (scelto sul devset),
ranking per similarita' alla reference fra i sopravvissuti.

Nessuna label usata nel metodo. q e' l'unico iperparametro, scelto sulle 101
query di sviluppo; trasferimento valutato UNA volta sul benchmark ufficiale.
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
E_test, L_test, meta = load_split("test")
attr_names = [n for n in meta["attr_names"] if n.strip()]
D = E_valid.shape[1]

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

def percentiles(E):
    """pct[i, n] = percentile dell'immagine n sul punteggio dell'attributo i."""
    scores = (E @ d_ens.T).numpy()                       # [N, 40]
    ranks = scores.argsort(0).argsort(0).astype(np.float32)
    return torch.from_numpy(ranks.T / (len(E) - 1))      # [40, N]

def eval_set(bench, E, pct, q=None, build_query=None, use_ref_rank=True):
    """Filtro duro: tieni v se pct_i(v) >= q per ogni +i e pct_j(v) <= 1-q per ogni -j.
    Ranking: cos alla reference (use_ref_rank) o query passata."""
    per_source = {k: [] for k in K_VALUES}
    k_max = max(K_VALUES)
    for entry in bench:
        if q is not None:
            keep = torch.ones(len(E), dtype=torch.bool)
            for i in entry["pos"]:
                keep &= pct[i] >= q
            for j in entry["neg"]:
                keep &= pct[j] <= 1 - q
            banned = ~keep
        else:
            banned = torch.zeros(len(E), dtype=torch.bool)
        sources = entry["sources"]
        n_src = len(sources)
        rec = {k: np.zeros(n_src) for k in K_VALUES}
        for start in range(0, n_src, 2048):
            stop = min(start + 2048, n_src)
            batch_src = sources[start:stop]
            ref = E[batch_src]
            qv = ref if use_ref_rank else build_query(ref, entry)
            sim = qv @ E.T
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

pct_valid = percentiles(E_valid)
bench = load_benchmark("celeba_devset_valid.json", attr_names)
ps_base = eval_set(bench, E_valid, None, q=None, build_query=build_baseline, use_ref_rank=False)
print(f"\nSET 101 QUERY — baseline: R@10 {macro(ps_base[10])*100:.2f}%", flush=True)

print("\nsweep del quantile q (filtro zero-shot + ranking identita'):")
best_q, best_m, results = None, -1, {}
for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    ps = eval_set(bench, E_valid, pct_valid, q=q)
    m = macro(ps[10])
    results[q] = ps
    r = bootstrap_paired(ps[10], ps_base[10], n_boot=20000, seed=0)
    print(f"  q={q:.1f}  R@10 {m*100:6.2f}%   vs baseline {r['delta']*100:+6.2f}pp "
          f"IC [{r['lo']*100:+.2f},{r['hi']*100:+.2f}] p={r['p']:.4f}", flush=True)
    if m > best_m:
        best_q, best_m = q, m

print(f"\nq* = {best_q} (scelto sul devset). Dettaglio a tutti i K:")
for k in K_VALUES:
    r = bootstrap_paired(results[best_q][k], ps_base[k], n_boot=20000, seed=0)
    print(f"  R@{k:<3} {macro(results[best_q][k])*100:6.2f}%  vs baseline {r['delta']*100:+6.2f}pp "
          f"IC [{r['lo']*100:+.2f},{r['hi']*100:+.2f}] p={r['p']:.4f}", flush=True)

# variante: filtro zero-shot + ranking con la QUERY baseline (non solo identita')
ps = eval_set(bench, E_valid, pct_valid, q=best_q, build_query=build_baseline, use_ref_rank=False)
print("\nfiltro zero-shot + ranking con query baseline (stesso q*):")
for k in K_VALUES:
    r = bootstrap_paired(ps[k], ps_base[k], n_boot=20000, seed=0)
    print(f"  R@{k:<3} {macro(ps[k])*100:6.2f}%  vs baseline {r['delta']*100:+6.2f}pp "
          f"IC [{r['lo']*100:+.2f},{r['hi']*100:+.2f}] p={r['p']:.4f}", flush=True)

# trasferimento sul benchmark ufficiale, una volta sola, con q* scelto sul dev
pct_test = percentiles(E_test)
bench_test = load_benchmark("celeba_evaluation.json", attr_names)
uniq = []
seen = set()
for i, e in enumerate(bench_test):
    if e["query"] not in seen:
        seen.add(e["query"])
        uniq.append(i)
ps_bt = eval_set(bench_test, E_test, None, q=None, build_query=build_baseline, use_ref_rank=False)
ps_ft = eval_set(bench_test, E_test, pct_test, q=best_q)
ps_bt = {k: [ps_bt[k][i] for i in uniq] for k in K_VALUES}
ps_ft = {k: [ps_ft[k][i] for i in uniq] for k in K_VALUES}
print(f"\nBENCHMARK UFFICIALE (13 query distinte), q*={best_q} scelto sul dev:")
for k in K_VALUES:
    r = bootstrap_paired(ps_ft[k], ps_bt[k], n_boot=20000, seed=0)
    print(f"  R@{k:<3} filtro {macro(ps_ft[k])*100:6.2f}%  baseline {macro(ps_bt[k])*100:6.2f}%  "
          f"diff {r['delta']*100:+6.2f}pp IC [{r['lo']*100:+.2f},{r['hi']*100:+.2f}] p={r['p']:.4f}", flush=True)
