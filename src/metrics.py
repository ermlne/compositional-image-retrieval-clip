# metrics.py
"""
Metriche di retrieval condivise da tutti i notebook (02-05).

Il ground truth segue il criterio dell'assignment (sez. 3.1.1): un'immagine
recuperata e' corretta se ha attr+, non ha attr-, e ha distanza di Hamming
<= 2 dalla reference sugli altri 38 attributi non interrogati.

L'assignment (sez. 3.1.3) richiede due metriche distinte, spesso confuse
nei notebook precedenti:

- Recall@K (hit rate, eq. 1): 1 se almeno un'immagine corretta e' tra le
  prime K recuperate, 0 altrimenti. E' la metrica primaria richiesta.
- Precision@K (eq. 2): frazione di immagini corrette tra le prime K.

I notebook 02-05 calcolavano solo la seconda etichettandola come "Recall@K".
"""

import numpy as np


def hamming_correct_mask(retrieved_labels, ref_attrs, non_queried,
                          attr_pos_idx, attr_neg_idx, hamming_threshold=2):
    """Maschera booleana [k] delle immagini recuperate che soddisfano il
    criterio di ground truth dell'assignment (attr+, NOT attr-, Hamming<=T)."""
    has_pos = retrieved_labels[:, attr_pos_idx] == 1
    has_not_neg = retrieved_labels[:, attr_neg_idx] == 0
    hamming = (retrieved_labels[:, non_queried] != ref_attrs[non_queried]).sum(dim=1)
    return has_pos & has_not_neg & (hamming <= hamming_threshold)


def evaluate_retrieval(filtered_top, queries, labels, k_values, hamming_threshold=2):
    """Calcola Recall@K (hit rate) e Precision@K per ogni query.

    filtered_top[i] e' la lista degli indici recuperati per la query i
    (gia' filtrata dal riferimento), lunga almeno max(k_values).

    Ritorna (recall_per_query, precision_per_query): due dict {k: [valori per query]}.
    """
    recall_per_query = {k: [] for k in k_values}
    precision_per_query = {k: [] for k in k_values}

    for i, q in enumerate(queries):
        retrieved_labels = labels[filtered_top[i]]
        ref_attrs = labels[q["ref_idx"]]
        correct = hamming_correct_mask(
            retrieved_labels, ref_attrs, q["non_queried"],
            q["attr_pos_idx"], q["attr_neg_idx"], hamming_threshold,
        )
        for k in k_values:
            correct_k = correct[:k]
            recall_per_query[k].append(float(correct_k.any()))
            precision_per_query[k].append(correct_k.float().mean().item())

    return recall_per_query, precision_per_query


def mean_at_k(metric_per_query):
    """{k: [valori per query]} -> {k: media}"""
    return {k: np.mean(v) for k, v in metric_per_query.items()}


def print_summary(title, recall_per_query, precision_per_query, k_values):
    """Stampa una tabella comparativa Recall@K (hit rate) / Precision@K."""
    mean_recall = mean_at_k(recall_per_query)
    mean_precision = mean_at_k(precision_per_query)

    width = 60
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  {'K':<6}{'Recall@K (hit rate)':>24}{'Precision@K':>20}")
    for k in k_values:
        print(f"  {k:<6}{mean_recall[k] * 100:>22.1f}%{mean_precision[k] * 100:>19.1f}%")
    print("=" * width)
