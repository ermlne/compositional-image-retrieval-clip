"""Tetti oracolari misurati con le label, PRIMA di implementare qualunque metodo.

Domanda: quanto puo' valere, al massimo, un trattamento perfetto delle condizioni
(o dei soli negativi / soli positivi), tenendo CLIP per l'identita'?

Oracoli (usano le label del valid split -- sono tetti, non metodi):
  O-neg   : query baseline (ensembling, somma grezza, alpha=1), ma i candidati che
            POSSIEDONO un attributo negato sono esclusi dal ranking.
  O-pos   : idem sui positivi (candidati privi di un attributo richiesto esclusi).
  O-cond  : esclusi tutti i violatori di qualunque condizione; ranking per sola
            similarita' alla reference (nessun testo: e' il tetto dell'architettura
            "filtra per condizioni, ordina per identita'").
Confronti: differenza accoppiata + IC (bootstrap_paired), sul set a 101 query.
"""
import sys, json
import numpy as np
import torch

sys.path.append(".")
from src.benchmark import load_benchmark
from src.devset import evaluate_per_source, macro, bootstrap_paired
from transformers import CLIPModel, CLIPTokenizer

K_VALUES = [1, 5, 10]
MODEL_ID = "openai/clip-vit-base-patch32"

def load_split(split):
    d = torch.load(f"embeddings/celeba_{split}_clip_vitb32.pt", map_location="cpu")
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"], d

E_valid, L_valid, _ = load_split("valid")
_, _, meta = load_split("test")
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
print("direzioni ensemblate pronte", flush=True)

def build_baseline(ref, entry):
    d = torch.zeros(D)
    for i in entry["pos"]: d = d + d_ens[i]
    for j in entry["neg"]: d = d - d_ens[j]
    q = ref + d
    return q / q.norm(dim=-1, keepdim=True)

def only_ref(ref, entry):
    return ref

def eval_masked(build_query, bench, embeddings, labels, mask_mode):
    """Come evaluate_per_source, ma con un filtro oracolare sui candidati.
    mask_mode: None | 'neg' | 'pos' | 'cond'"""
    per_source = {k: [] for k in K_VALUES}
    k_max = max(K_VALUES)
    N = len(embeddings)
    for entry in bench:
        banned = torch.zeros(N, dtype=torch.bool)
        if mask_mode in ("neg", "cond"):
            for j in entry["neg"]:
                banned |= labels[:, j] == 1
        if mask_mode in ("pos", "cond"):
            for i in entry["pos"]:
                banned |= labels[:, i] == 0
        sources = entry["sources"]
        n_src = len(sources)
        rec = {k: np.zeros(n_src) for k in K_VALUES}
        for start in range(0, n_src, 2048):
            stop = min(start + 2048, n_src)
            batch_src = sources[start:stop]
            q = build_query(embeddings[batch_src], entry)
            sim = q @ embeddings.T
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

def report(name, ps, ps_base=None):
    line = f"{name:<42}"
    for k in K_VALUES:
        line += f"  R@{k} {macro(ps[k])*100:6.2f}%"
    print(line, flush=True)
    if ps_base is not None:
        for k in K_VALUES:
            r = bootstrap_paired(ps[k], ps_base[k], n_boot=20000, seed=0)
            print(f"    vs baseline  R@{k:<3} {r['delta']*100:+6.2f}pp  "
                  f"IC [{r['lo']*100:+.2f},{r['hi']*100:+.2f}]  p={r['p']:.4f}", flush=True)

for set_name, path in [("SET DI SVILUPPO (101 query, pool filtrato)", "celeba_devset_valid.json"),
                       ("SET POOL COMPLETO (101 query, 40 attributi)", "celeba_devset_valid_full.json")]:
    bench = load_benchmark(path, attr_names)
    print(f"\n=== {set_name} ===", flush=True)
    ps_base = eval_masked(build_baseline, bench, E_valid, L_valid, None)
    ps_ref  = eval_masked(only_ref, bench, E_valid, L_valid, None)
    report("ref-only (lower bound)", ps_ref, ps_base)
    report("baseline ensembling (riprodotta)", ps_base)
    ps = eval_masked(build_baseline, bench, E_valid, L_valid, "neg")
    report("ORACOLO filtro negativi + query baseline", ps, ps_base)
    nneg = [i for i, e in enumerate(bench) if e["neg"]]
    sub = {k: [ps[k][i] for i in nneg] for k in K_VALUES}
    sub_b = {k: [ps_base[k][i] for i in nneg] for k in K_VALUES}
    print(f"  -- sulle sole {len(nneg)} query con almeno un negativo:")
    report("  (subset negativi)", sub, sub_b)
    ps = eval_masked(build_baseline, bench, E_valid, L_valid, "pos")
    report("ORACOLO filtro positivi + query baseline", ps, ps_base)
    ps = eval_masked(only_ref, bench, E_valid, L_valid, "cond")
    report("ORACOLO filtro condizioni + solo identita'", ps, ps_base)
