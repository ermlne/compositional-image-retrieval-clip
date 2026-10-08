# benchmark.py
"""
Valutazione sul benchmark ufficiale dell'assignment (`celeba_evaluation.json`).

Differenza rispetto a `src/metrics.py`: quel modulo *ricalcola* il ground truth
dalle label CelebA (attr+, NOT attr-, Hamming <= 2) su query auto-generate.
Qui invece si usa il ground truth **gia' calcolato dal docente** e la lista di
source image ufficiali, come richiesto dalla sez. 3.1 dell'assignment:

  "You must restrict your evaluation strictly to the source image indices (keys)
   provided in the JSON for each respective query."

Verificato il 24/08/2026: il ground truth del JSON coincide esattamente con quello
ricostruito dalle label del nostro `celeba_test_clip_vitb32.pt` (0 differenze su
un campione di query semplici e composte), quindi gli indici del JSON sono allineati
all'ordine del test split usato per estrarre gli embedding.

Le query del benchmark sono multi-condizione (fino a 3 attributi), a differenza
delle coppie singole attr+/attr- usate nei notebook 02-06.
"""

import json

import numpy as np
import torch


def parse_query_string(query, attr_to_idx):
    """'+Chubby, -Young' -> ([idx_Chubby], [idx_Young])"""
    pos, neg = [], []
    for token in query.split(","):
        token = token.strip()
        if not token:
            continue
        sign, name = token[0], token[1:].strip()
        if sign == "+":
            pos.append(attr_to_idx[name])
        elif sign == "-":
            neg.append(attr_to_idx[name])
        else:
            raise ValueError(f"Segno non riconosciuto in {token!r}")
    return pos, neg


def load_benchmark(json_path, attr_names):
    """Carica le query ufficiali.

    Ritorna una lista di dict, uno per entry del JSON:
      query    : stringa originale, es. '+Chubby, -Young'
      pos, neg : liste di indici di attributo
      queried  : pos + neg
      non_queried : gli altri attributi (quelli su cui vale il vincolo Hamming)
      sources  : LongTensor [n_src] con gli indici delle immagini di riferimento
      gt       : lista di set, gt[i] = ground truth per sources[i]

    Nota: il JSON contiene 14 entry ma '-Young' compare due volte identica
    (indici 4 e 7), quindi le query distinte sono 13. Si mantengono entrambe
    per aderire alla lettera al file ufficiale; `summarize` riporta anche la
    media sulle query distinte.
    """
    attr_to_idx = {name: i for i, name in enumerate(attr_names)}
    with open(json_path) as f:
        raw = json.load(f)

    benchmark = []
    for entry in raw:
        pos, neg = parse_query_string(entry["query"], attr_to_idx)
        queried = pos + neg
        gt_dict = entry["ground_truth"]
        sources = sorted(int(k) for k in gt_dict)
        benchmark.append({
            "query": entry["query"],
            "pos": pos,
            "neg": neg,
            "queried": queried,
            "non_queried": [i for i in range(len(attr_names)) if i not in queried],
            "sources": torch.tensor(sources, dtype=torch.long),
            "gt": [set(gt_dict[str(s)]) for s in sources],
        })
    return benchmark


def evaluate_query(build_query, entry, embeddings, k_values, batch_size=1024):
    """Valuta un metodo su una singola query del benchmark.

    `build_query(ref_embs, entry) -> [n, D]` costruisce i vettori di query per un
    batch di immagini di riferimento; deve restituirli gia' L2-normalizzati.

    Ritorna (recall_per_source, precision_per_source): due dict {k: np.array [n_src]}.
    """
    k_max = max(k_values)
    sources = entry["sources"]
    n_src = len(sources)

    recall = {k: np.zeros(n_src) for k in k_values}
    precision = {k: np.zeros(n_src) for k in k_values}

    for start in range(0, n_src, batch_size):
        stop = min(start + batch_size, n_src)
        batch_src = sources[start:stop]
        q_vecs = build_query(embeddings[batch_src], entry)
        sim = q_vecs @ embeddings.T                     # [b, N]
        # la reference non puo' essere un target: la si esclude prima del topk
        sim[torch.arange(stop - start), batch_src] = -torch.inf
        top = sim.topk(k_max, dim=-1).indices.tolist()  # [b, k_max]

        for j, retrieved in enumerate(top):
            gt = entry["gt"][start + j]
            hits = np.array([idx in gt for idx in retrieved], dtype=bool)
            cum = hits.cumsum()
            for k in k_values:
                recall[k][start + j] = float(cum[k - 1] > 0)
                precision[k][start + j] = cum[k - 1] / k

    return recall, precision


def evaluate_benchmark(build_query, benchmark, embeddings, k_values,
                       batch_size=1024, verbose=True):
    """Valuta un metodo su tutte le query del benchmark.

    Ritorna una lista di dict per query: query, n_src, recall {k: media}, precision {k: media}.
    """
    results = []
    for entry in benchmark:
        recall, precision = evaluate_query(build_query, entry, embeddings,
                                           k_values, batch_size)
        row = {
            "query": entry["query"],
            "n_src": len(entry["sources"]),
            "recall": {k: float(recall[k].mean()) for k in k_values},
            "precision": {k: float(precision[k].mean()) for k in k_values},
        }
        results.append(row)
        if verbose:
            r = "  ".join(f"R@{k}={row['recall'][k] * 100:5.1f}%" for k in k_values)
            print(f"  {row['query']:<45} n={row['n_src']:5d}  {r}")
    return results


def macro_average(results, k_values, unique_only=True):
    """Media delle metriche tra query (macro-average, ogni query pesa uguale).

    Con `unique_only` la query duplicata nel JSON ('-Young') conta una volta sola.
    """
    rows = results
    if unique_only:
        seen, rows = set(), []
        for row in results:
            if row["query"] not in seen:
                seen.add(row["query"])
                rows.append(row)
    return (
        {k: float(np.mean([r["recall"][k] for r in rows])) for k in k_values},
        {k: float(np.mean([r["precision"][k] for r in rows])) for k in k_values},
    )


def print_comparison(named_results, k_values, unique_only=True):
    """Tabella comparativa Recall@K / Precision@K tra piu' metodi.

    `named_results`: dict {nome_metodo: risultati di evaluate_benchmark}.
    """
    names = list(named_results)
    col = max(12, max(len(n) for n in names) + 2)

    for metric, getter in (("Recall@K (hit rate)", 0), ("Precision@K", 1)):
        print(f"\n{metric} — macro-average su "
              f"{'13 query distinte' if unique_only else '14 entry del JSON'}")
        print("-" * (6 + col * len(names)))
        print(f"{'K':<6}" + "".join(f"{n:>{col}}" for n in names))
        avgs = {n: macro_average(named_results[n], k_values, unique_only)[getter]
                for n in names}
        for k in k_values:
            print(f"@{k:<5}" + "".join(f"{avgs[n][k] * 100:>{col - 1}.1f}%" for n in names))


def build_benchmark_from_labels(queries, labels, attr_names, hamming_threshold=2,
                                min_targets=5, max_sources=None, seed=0):
    """Ricostruisce un benchmark con lo stesso protocollo del JSON ufficiale su
    uno split diverso (es. il validation set), a partire dalle sole label.

    Serve per tarare gli iperparametri **fuori** dal test set: il JSON ufficiale
    copre solo il test split, e tararci sopra sarebbe overfitting sul benchmark.

    Replica esattamente la regola della sez. 3.1.1: un target e' valido se
    soddisfa i vincoli +/-, dista <= `hamming_threshold` dalla reference sugli
    attributi non interrogati, e non e' la reference stessa; una source entra nel
    benchmark solo se ha almeno `min_targets` target validi.

    `queries` e' una lista di stringhe nello stesso formato del JSON ('+A, -B').
    """
    attr_to_idx = {name: i for i, name in enumerate(attr_names)}
    n_attr = len(attr_names)
    generator = torch.Generator().manual_seed(seed)

    benchmark = []
    for query in queries:
        pos, neg = parse_query_string(query, attr_to_idx)
        queried = pos + neg
        non_queried = [i for i in range(n_attr) if i not in queried]

        satisfies = torch.ones(len(labels), dtype=torch.bool)
        for i in pos:
            satisfies &= labels[:, i] == 1
        for j in neg:
            satisfies &= labels[:, j] == 0
        candidates = satisfies.nonzero(as_tuple=True)[0]

        # Regola di selezione delle reference, dedotta dal JSON ufficiale e verificata
        # esattamente su tutte le sue query (24/08/2026): la reference deve violare
        # *ogni* condizione, non solo almeno una — per un attributo positivo non deve
        # possederlo, per uno negativo deve possederlo. Con la regola piu' debole
        # ("non soddisfa gia' la query") si ottiene un superset delle source ufficiali.
        violates_all = torch.ones(len(labels), dtype=torch.bool)
        for i in pos:
            violates_all &= labels[:, i] == 0
        for j in neg:
            violates_all &= labels[:, j] == 1
        refs = violates_all.nonzero(as_tuple=True)[0]
        if max_sources is not None and len(refs) > max_sources:
            refs = refs[torch.randperm(len(refs), generator=generator)[:max_sources]]

        cand_nq = labels[candidates][:, non_queried].bool()
        sources, gt = [], []
        for start in range(0, len(refs), 256):   # a blocchi: la matrice Hamming e' [b, n_cand]
            block = refs[start:start + 256]
            ref_nq = labels[block][:, non_queried].bool()
            hamming = (cand_nq.unsqueeze(0) != ref_nq.unsqueeze(1)).sum(dim=2)
            for j, ref in enumerate(block.tolist()):
                valid = candidates[hamming[j] <= hamming_threshold]
                valid = valid[valid != ref]
                if len(valid) >= min_targets:
                    sources.append(ref)
                    gt.append(set(valid.tolist()))

        if sources:
            benchmark.append({
                "query": query,
                "pos": pos,
                "neg": neg,
                "queried": queried,
                "non_queried": non_queried,
                "sources": torch.tensor(sources, dtype=torch.long),
                "gt": gt,
            })
    return benchmark
