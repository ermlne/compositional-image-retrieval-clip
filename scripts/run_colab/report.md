# Sessione di consolidamento — 29/08/2026 21:00

Device: `cuda`. Baseline sul set di sviluppo: **19.91%** (atteso 19.91%, collaudo superato).
**Questo script non legge mai `celeba_evaluation.json`.** Il budget del benchmark ufficiale e' per metodo ed e' gia' stato speso una volta, il 25/08, per il filtro zero-shot, con esito negativo (`scripts/README.md`): la valutazione del contributo sul benchmark si fa dopo, a mano, a configurazione congelata.
Selezione degli iperparametri: **solo** sulla validazione onesta a 150 query; celle provate: **8**; margine di adozione 0.30pp.

## Risultati sul set di sviluppo (R@10, macro-average su 101 query)

| metodo                                 | R@10   | vs baseline | IC 95%           | p      |
|----------------------------------------|--------|-------------|------------------|--------|
| baseline                               | 19.91% | —           | —                | —      |
| membro ms1400_tau0.01_wd0.0001_q11_off | 33.94% | +14.03pp    | [+11.38, +16.79] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_q12_off | 33.85% | +13.94pp    | [+11.07, +16.92] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_q13_off | 34.21% | +14.29pp    | [+11.41, +17.24] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_q14_off | 32.67% | +12.76pp    | [+10.02, +15.55] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_q15_off | 33.55% | +13.63pp    | [+10.83, +16.56] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_q16_off | 33.82% | +13.91pp    | [+10.98, +16.95] | 0.0000 |
| ENSEMBLE (6 modelli)                   | 36.73% | +16.82pp    | [+14.01, +19.70] | 0.0000 |
| MATRICE collassata                     | 34.60% | +14.69pp    | [+11.79, +17.75] | 0.0000 |

*Ogni confronto e' una differenza accoppiata con bootstrap gerarchico (`src/devset.bootstrap_paired`), non due medie affiancate.*

## Curva dei rendimenti dell'ensemble

| k | R@10   | vs baseline | vs k-1  | p     |
|---|--------|-------------|---------|-------|
| 1 | 33.94% | +14.03pp    | —       | —     |
| 2 | 35.27% | +15.35pp    | +1.33pp | 0.009 |
| 3 | 36.13% | +16.21pp    | +0.86pp | 0.019 |
| 4 | 36.40% | +16.48pp    | +0.27pp | 0.407 |
| 5 | 36.92% | +17.00pp    | +0.52pp | 0.058 |
| 6 | 36.73% | +16.82pp    | -0.19pp | 0.419 |

## Da dove viene la diversita' dell'ensemble

I 6 membri sono addestrati al livello `max_sources 1400` su un pool costruito a `2800`: ognuno vede un campione diverso di query **e** una finestra diversa di reference dentro ogni query.

- sovrapposizione media fra i campioni di due membri (Jaccard sugli esempi, 15 coppie): **0.334** [0.293, 0.430]
- senza l'offset sulle reference sarebbe stata **0.598**: la differenza e' la quota di diversita' che l'offset aggiunge
- su 96/279 query il bacino di reference si esaurisce prima di `1400`, quindi li' tutti i membri vedono le stesse reference: e' inevitabile ed e' contato.

## Ricerca a coordinate (letta sulla validazione onesta)

| max_sources | tau  | wd     | val R@10 (media) | per seed     | dev R@10 (non usato) |
|-------------|------|--------|------------------|--------------|----------------------|
| 1400        | 0.01 | 0.01   | 48.28%           | 47.89, 48.67 | 34.13%               |
| 1400        | 0.01 | 0.0001 | 48.21%           | 47.89, 48.53 | 34.17%               |
| 1400        | 0.01 | 0.001  | 48.21%           | 47.89, 48.53 | 34.17%               |
| 700         | 0.01 | 0.0001 | 43.08%           | 43.30, 42.85 | 33.13%               |
| 700         | 0.02 | 0.0001 | 42.36%           | 41.82, 42.91 | 32.53%               |
| 700         | 0.03 | 0.0001 | 38.12%           | 38.13, 38.12 | 32.34%               |
| 350         | 0.01 | 0.0001 | 37.92%           | 41.56, 34.28 | 30.41%               |
| 700         | 0.05 | 0.0001 | 32.53%           | 32.07, 32.99 | 30.93%               |

Punto scelto: **tau 0.01, max_sources 1400, wd 0.0001**.
Sul set di sviluppo avrebbe vinto tau 0.01, ms 1400, wd 0.001: **non e' stato usato**.

## Confronto accoppiato fra livelli, ai rispettivi punti d'arresto

Le viste sui livelli sono **annidate** (stesse query, prefisso di reference), quindi il confronto e' accoppiato; ogni livello e' fermato dalla propria validazione e gli esiti per source sono quelli dell'epoca scelta.

| confronto      | differenza | IC 95%         | p      | query   |
|----------------|------------|----------------|--------|---------|
| 700_vs_350_s0  | +3.87pp    | [+2.67, +5.12] | 0.0000 | 83V/17P |
| 700_vs_350_s1  | +1.56pp    | [+0.35, +2.72] | 0.0112 | 67V/32P |
| 1400_vs_350_s0 | +4.71pp    | [+3.23, +6.22] | 0.0000 | 84V/11P |
| 1400_vs_350_s1 | +2.81pp    | [+1.10, +4.51] | 0.0016 | 70V/26P |

## Collasso lineare

- `fit(media) == media(fit)`: max|W_ens - W_media| = **1.34e-07** (atteso ~0, la ridge e' lineare nel bersaglio)
- R² della mappa sull'uscita mediata, su dati mai usati per il fit: **0.9441**
- Recall@10 della matrice collassata: **34.60%**, vs ensemble -2.13pp (p = 0.0000)
- coppie (r, c) del fit: 18397, tutte da reference del **train**

## Come rileggere questi file

- `risultati.json` — tutto, comprese le traiettorie per epoca
- `esiti_devset.npz` — esiti per singola source di ogni metodo (`<metodo>_k10_flat` + `<metodo>_k10_len`), per rifare qualunque bootstrap
- `ckpt/*.pt` — un checkpoint per modello, all'**epoca scelta dalla validazione**
- `ckpt/matrice_collassata.pt` — la matrice W (gamma gia' dentro)
