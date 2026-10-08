# devset.py
"""
Set di sviluppo ampliato e quantificazione dell'incertezza.

MOTIVO. Il benchmark ufficiale ha 13 query distinte. Su quella base una
differenza sotto il punto percentuale fra due metodi non e' distinguibile dal
rumore: e' esattamente cio' che ha prodotto il falso positivo dell'asimmetria
positivi/negativi del notebook 08 (+0.48pp sul dev, -0.20pp sul test).

DOVE STA IL RUMORE. Misurato sul benchmark ufficiale (metodo ortho-text,
alpha=2.0, Recall@10), la macro-average ha due sorgenti di incertezza:

    deviazione standard TRA query      9.30pp
    errore standard della macro-media  2.58pp   <- campionamento delle query
    errore standard da sole source     0.51pp   <- campionamento delle source

Le source sono 27697: l'incertezza che introducono e' cinque volte piu' piccola
di quella dovuta all'avere solo 13 query. Un bootstrap sulle sole source, come
si sarebbe portati a fare, produce quindi intervalli di confidenza troppo
stretti e **non** avrebbe intercettato il falso positivo. L'unita' di
campionamento giusta e' la query, non la source.

COSA FA QUESTO MODULO.
1. Genera un set di sviluppo di ~100 query sul validation split, con la stessa
   distribuzione del benchmark ufficiale, riusando `build_benchmark_from_labels`
   (verificata esatta contro il JSON del docente).
2. Lo serializza nello stesso schema di `celeba_evaluation.json`, cosi' e'
   leggibile con `load_benchmark` e condivisibile via git.
3. Fornisce un bootstrap gerarchico (query, poi source) e la sua versione
   **accoppiata** per confrontare due metodi sulle stesse query.

Il criterio di successo non e' una Recall piu' alta: e' poter affermare che una
differenza fra due metodi e' reale.
"""

import json

import numpy as np
import torch

from .benchmark import build_benchmark_from_labels, evaluate_query

# Composizione del benchmark ufficiale, misurata sulle 13 query distinte
# (7 a una condizione, 4 a due, 2 a tre; 13 condizioni positive su 21).
N_COND_WEIGHTS = (0.54, 0.31, 0.15)
P_POSITIVE = 0.62


def attribute_auc(embeddings, labels, directions, attr_names):
    """AUC della classificazione zero-shot di ogni attributo con la direzione
    testuale di CLIP. Da calcolare sul **train split**, disgiunto da valid e
    test, per non contaminare ne' il set di sviluppo ne' il benchmark.
    """
    scores = embeddings @ directions.T
    auc = {}
    for i, name in enumerate(attr_names):
        y = labels[:, i].numpy().astype(bool)
        s = scores[:, i].numpy()
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        if n_pos < 50 or n_neg < 50:
            auc[name] = float("nan")
            continue
        rank = np.argsort(np.argsort(s)) + 1.0
        auc[name] = float((rank[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return auc


def attribute_pool(auc, min_auc=0.55):
    """Attributi ammessi nel campionamento delle query.

    Si escludono quelli la cui direzione testuale e' scorrelata o
    **anti**-correlata con la label (Sideburns 0.32, Rosy_Cheeks 0.41,
    5_o_Clock_Shadow 0.42, ...): su quelle query ogni metodo resta a zero, e
    query in cui tutti i metodi falliscono allo stesso modo aggiungono rumore
    alla macro-media senza aggiungere potere discriminante.

    La soglia 0.55 e' scelta per includere tutti e 12 gli attributi che il
    docente usa nel benchmark ufficiale: il piu' debole e' Wearing_Lipstick
    (0.568), seguito da Heavy_Makeup (0.598). Il filtro allinea quindi il set
    di sviluppo alla popolazione di query da cui il benchmark e' estratto,
    invece di restringerlo agli attributi facili.

    ATTENZIONE: il criterio dipende dalle direzioni **testuali**. Non falsa il
    confronto fra metodi che condividono quelle direzioni (baseline, ortho,
    varianti di pesatura) — rimuove solo query a pavimento per tutti — ma non
    va usato per confrontare direzioni testuali contro direzioni visive.
    """
    return [name for name, value in auc.items() if value == value and value >= min_auc]


def sample_queries(pool, n_queries, n_cond, exclude=(), seed=2026,
                   p_positive=P_POSITIVE):
    """Query casuali con `n_cond` condizioni ciascuna.

    `exclude` tiene fuori le query gia' presenti altrove (tipicamente le 13
    ufficiali, che vengono aggiunte a parte come sottoinsieme).

    Lo spazio delle query a `n_cond` condizioni e' finito e, per n_cond = 1,
    **piccolo**: con 29 attributi nel pool esistono solo 29 x 2 = 58 query a
    una condizione, meno le 7 ufficiali gia' escluse. Chiederne di piu' di
    quante ne esistano manderebbe il campionamento per rifiuto in stallo, quindi
    la richiesta si limita allo spazio disponibile e c'e' comunque un tetto ai
    tentativi. La funzione puo' restituire meno di `n_queries`: e' il chiamante
    a decidere che farne.
    """
    import random
    from math import comb

    space = comb(len(pool), n_cond) * (2 ** n_cond)
    rng = random.Random(seed)
    seen = set(exclude)
    queries = []
    attempts, max_attempts = 0, 200 * max(n_queries, 1)
    while len(queries) < min(n_queries, space) and attempts < max_attempts:
        attempts += 1
        attrs = rng.sample(pool, n_cond)
        query = ", ".join(
            f"{'+' if rng.random() < p_positive else '-'}{a}" for a in attrs
        )
        if query in seen:
            continue
        seen.add(query)
        queries.append(query)
    return queries


def build_devset(queries, labels, attr_names, min_sources=30, max_sources=1200,
                 max_final_sources=400, seed=7, verbose=True, **kwargs):
    """Costruisce le entry, scarta le query degeneri e pareggia il numero di
    reference per query.

    `max_sources` limita le reference **prima** del filtro di Hamming, quindi
    solo una frazione (~20-30%) sopravvive: e' il parametro che governa il
    costo, O(reference x candidati) per query.

    `max_final_sources` sottocampiona le reference sopravvissute. Non e' una
    perdita: con ~400 reference l'errore standard entro query e' circa 1.8pp,
    che mediato su ~100 query diventa 0.18pp — un ordine di grandezza sotto
    l'incertezza dovuta al campionamento delle query, che e' quella che conta.
    Tenerne di piu' gonfierebbe solo il file e il tempo di valutazione, e
    lascerebbe le query comuni a pesare di piu' nella precisione entro query.

    `min_sources` scarta le query con troppe poche reference valide, le cui
    stime sono troppo rumorose per essere utili. Sul benchmark ufficiale
    ricadono in questa condizione tre query su tredici ('-Male, -Mustache' con
    27 source, '+Wearing_Lipstick, ...' con 34, '-Smiling, ...' con 79).
    """
    bench = build_benchmark_from_labels(queries, labels, attr_names,
                                        max_sources=max_sources, seed=seed, **kwargs)
    kept = [e for e in bench if len(e["sources"]) >= min_sources]
    if max_final_sources is not None:
        generator = torch.Generator().manual_seed(seed)
        for e in kept:
            if len(e["sources"]) > max_final_sources:
                pick = torch.randperm(len(e["sources"]), generator=generator)[:max_final_sources]
                pick = pick.sort().values
                e["sources"] = e["sources"][pick]
                e["gt"] = [e["gt"][i] for i in pick.tolist()]
    if verbose:
        print(f"    {len(bench)}/{len(queries)} query con almeno una reference valida; "
              f"{len(kept)} con >= {min_sources} reference")
    return kept


def build_stratified_devset(pool, labels, attr_names, targets, exclude=(),
                            batch=40, max_rounds=12, seed=2026, verbose=True, **kwargs):
    """Set di sviluppo con la composizione del benchmark ufficiale garantita.

    Campionare secondo la distribuzione ufficiale e *poi* filtrare le query
    degeneri non basta: le query a piu' condizioni hanno molte meno reference
    valide, quindi sopravvivono meno spesso e la composizione finale si sposta
    verso le query semplici (a un tentativo: 51/41/8 invece di 54/31/15). Qui
    si campiona a quote fisse, un gruppo alla volta, finche' ogni strato non
    raggiunge la propria quota.

    `targets` e' un dict {n_condizioni: n_query_volute}. Uno strato puo'
    chiudersi **sotto quota** se il suo spazio di query si esaurisce: succede a
    n_cond = 1, dove le query possibili sono solo 58 e non tutte sopravvivono al
    filtro sulle query degeneri. In quel caso si prende quel che c'e' e la
    composizione finale va riportata come misurata, non come voluta.
    """
    kept = {n: [] for n in targets}
    seen = set(exclude)
    for n_cond, target in targets.items():
        for round_i in range(max_rounds):
            if len(kept[n_cond]) >= target:
                break
            need = target - len(kept[n_cond])
            queries = sample_queries(pool, max(batch, 2 * need), n_cond,
                                     exclude=seen, seed=seed + 100 * n_cond + round_i)
            if not queries:            # spazio esaurito: inutile insistere
                if verbose:
                    print(f"  {n_cond} condizioni: spazio esaurito al giro {round_i + 1}")
                break
            seen.update(queries)
            if verbose:
                print(f"  {n_cond} condizioni, giro {round_i + 1}: {len(queries)} tentate")
            kept[n_cond] += build_devset(queries, labels, attr_names,
                                         seed=seed, verbose=verbose, **kwargs)
        kept[n_cond] = kept[n_cond][:target]
        if verbose:
            flag = "" if len(kept[n_cond]) >= target else "   <- sotto quota"
            print(f"  {n_cond} condizioni: {len(kept[n_cond])}/{target}{flag}")
    return [e for n in sorted(kept) for e in kept[n]]


def save_devset(bench, path):
    """Serializza nello stesso schema di `celeba_evaluation.json`, cosi' che il
    set di sviluppo sia rileggibile con `load_benchmark` e versionabile in git
    (gli embedding `.pt` sono esclusi da `.gitignore`, un JSON no).
    """
    # le source si scrivono ordinate perche' `load_benchmark` le riordina in
    # lettura: cosi' il giro JSON e' esatto, entry per entry.
    payload = [
        {
            "query": e["query"],
            "ground_truth": {
                str(int(src)): sorted(int(t) for t in gt)
                for src, gt in sorted(zip(e["sources"].tolist(), e["gt"]),
                                      key=lambda pair: pair[0])
            },
        }
        for e in bench
    ]
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


# --------------------------------------------------------------------------
# Valutazione con incertezza
# --------------------------------------------------------------------------

def evaluate_per_source(build_query, bench, embeddings, k_values, batch_size=2048):
    """Come `evaluate_benchmark`, ma conserva il valore **per singola source**
    invece di mediarlo subito: e' il dato grezzo che serve al bootstrap.

    Ritorna {k: [array [n_src] per ogni query]}.
    """
    per_source = {k: [] for k in k_values}
    for entry in bench:
        recall, _ = evaluate_query(build_query, entry, embeddings, k_values, batch_size)
        for k in k_values:
            per_source[k].append(recall[k])
    return per_source


def macro(per_source_k):
    """Macro-average: ogni query pesa uguale, come nel protocollo ufficiale."""
    return float(np.mean([v.mean() for v in per_source_k]))


def variance_decomposition(per_source_k):
    """Separa le due sorgenti di incertezza della macro-media.

    Ritorna un dict con la deviazione standard fra query, l'errore standard
    dovuto al campionamento delle query e quello dovuto alle source. Serve a
    mostrare quale delle due domina — e quindi quale va ridotta.
    """
    means = np.array([v.mean() for v in per_source_k])
    n = np.array([len(v) for v in per_source_k])
    q = len(means)
    se_within = np.sqrt(np.sum(means * (1 - means) / n)) / q
    return {
        "n_query": q,
        "n_source": int(n.sum()),
        "macro": float(means.mean()),
        "sd_between_query": float(means.std(ddof=1)),
        "se_query": float(means.std(ddof=1) / np.sqrt(q)),
        "se_source": float(se_within),
    }


def bootstrap_ci(per_source_k, n_boot=5000, alpha=0.05, stage="both", seed=0):
    """Intervallo di confidenza sulla macro-media, per percentili.

    Bootstrap gerarchico: si ricampionano le query con reimmissione e, dentro
    ciascuna, le sue source. Poiche' il valore per source e' binario, il
    ricampionamento delle source ha forma chiusa — la media ricampionata e'
    Binomiale(n_q, r_q)/n_q — quindi non serve materializzare i ricampioni.

    `stage` permette di isolare le due sorgenti:
      'both'   query + source (default, l'incertezza reale)
      'query'  solo query
      'source' solo source (quello che si otterrebbe bootstrappando le sole
               source: sottostima grossolanamente l'incertezza, vedi docstring
               del modulo)
    """
    rng = np.random.default_rng(seed)
    means = np.array([v.mean() for v in per_source_k])
    n = np.array([len(v) for v in per_source_k])
    q = len(means)

    if stage in ("both", "query"):
        idx = rng.integers(0, q, size=(n_boot, q))
        m, nn = means[idx], n[idx]
    else:
        m, nn = np.broadcast_to(means, (n_boot, q)), np.broadcast_to(n, (n_boot, q))

    if stage in ("both", "source"):
        m = rng.binomial(nn, m) / nn

    draws = m.mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"macro": float(means.mean()), "lo": float(lo), "hi": float(hi),
            "se": float(draws.std(ddof=1))}


def bootstrap_paired(per_source_a, per_source_b, n_boot=5000, alpha=0.05, seed=0):
    """Confronto **accoppiato** fra due metodi valutati sulle stesse query.

    E' il test giusto per "A e' meglio di B": la differenza per source vale
    -1, 0 o +1, e ricampionandola si elimina la varianza tra query, che nei
    livelli domina tutto (9.3pp di deviazione standard) ma nella differenza si
    cancella. Il ricampionamento delle source ha di nuovo forma chiusa: i
    conteggi delle tre categorie sono Multinomiali.

    Ritorna la differenza media (A - B), l'intervallo di confidenza e la
    frazione di ricampioni con segno opposto (un p-value bilaterale empirico).
    """
    rng = np.random.default_rng(seed)
    q = len(per_source_a)
    # per ogni query: probabilita' delle tre categorie di (a - b)
    probs = np.array([
        [(d == -1).mean(), (d == 0).mean(), (d == 1).mean()]
        for d in (a - b for a, b in zip(per_source_a, per_source_b))
    ])
    n = np.array([len(v) for v in per_source_a])
    deltas = probs[:, 2] - probs[:, 0]

    idx = rng.integers(0, q, size=(n_boot, q))
    counts = np.stack([
        rng.multinomial(n[idx[:, j]], probs[idx[:, j]])
        for j in range(q)
    ], axis=1)                                        # [n_boot, q, 3]
    resampled = (counts[..., 2] - counts[..., 0]) / n[idx]
    draws = resampled.mean(axis=1)

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"delta": float(deltas.mean()), "lo": float(lo), "hi": float(hi),
            "se": float(draws.std(ddof=1)), "p": float(min(p, 1.0)),
            "n_query_win": int((deltas > 0).sum()), "n_query_loss": int((deltas < 0).sum())}


def print_with_ci(named_per_source, k, n_boot=5000, seed=0, baseline=None):
    """Tabella dei metodi con intervallo di confidenza, e — se `baseline` e'
    indicato — la differenza accoppiata rispetto a quel metodo.
    """
    names = list(named_per_source)
    width = max(len(n) for n in names) + 2
    print(f"Recall@{k} — macro-average, IC 95% (bootstrap gerarchico su "
          f"{len(next(iter(named_per_source.values())))} query)")
    header = f"{'metodo':<{width}}{'R@' + str(k):>9}{'IC 95%':>18}"
    if baseline is not None:
        print(f"confronto accoppiato rispetto a: {baseline}")
        header += f"{'diff':>17}{'IC 95% diff':>18}{'p':>8}"
    print(header)
    print("-" * len(header))
    for name in names:
        ci = bootstrap_ci(named_per_source[name], n_boot=n_boot, seed=seed)
        interval = "[{:.2f}, {:.2f}]".format(ci["lo"] * 100, ci["hi"] * 100)
        row = f"{name:<{width}}{ci['macro'] * 100:>8.2f}%{interval:>18}"
        if baseline is not None and name != baseline:
            pc = bootstrap_paired(named_per_source[name], named_per_source[baseline],
                                  n_boot=n_boot, seed=seed)
            diff = "[{:+.2f}, {:+.2f}]".format(pc["lo"] * 100, pc["hi"] * 100)
            row += f"{pc['delta'] * 100:>+15.2f}pp{diff:>18}{pc['p']:>8.3f}"
        elif baseline is not None:
            row += f"{'—':>16}{'—':>18}{'—':>8}"
        print(row)
