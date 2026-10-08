# Sessione di consolidamento — 29/08/2026 23:00

Device: `cuda`. Baseline sul set di sviluppo: **19.91%** (atteso 19.91%, collaudo superato).
**Questo script non legge mai `celeba_evaluation.json`.** Il budget del benchmark ufficiale e' per metodo ed e' gia' stato speso una volta, il 25/08, per il filtro zero-shot, con esito negativo (`scripts/README.md`): la valutazione del contributo sul benchmark si fa dopo, a mano, a configurazione congelata.
Selezione degli iperparametri: **solo** sulla validazione onesta a 150 query; celle provate: **15**; margine di adozione 0.30pp.

## Risultati sul set di sviluppo (R@10, macro-average su 101 query)

| metodo                                        | R@10   | vs baseline | IC 95%           | p      |
|-----------------------------------------------|--------|-------------|------------------|--------|
| baseline                                      | 19.91% | —           | —                | —      |
| membro ms2800_tau0.003_wd0.05_h512_q11_off_cx | 31.55% | +11.63pp    | [+8.75, +14.65]  | 0.0000 |
| membro ms2800_tau0.003_wd0.05_h512_q12_off_cx | 32.13% | +12.22pp    | [+9.40, +15.23]  | 0.0000 |
| membro ms2800_tau0.003_wd0.05_h512_q13_off_cx | 33.49% | +13.58pp    | [+10.79, +16.62] | 0.0000 |
| membro ms2800_tau0.003_wd0.05_h512_q14_off_cx | 32.97% | +13.05pp    | [+10.02, +16.24] | 0.0000 |
| membro ms2800_tau0.003_wd0.05_h512_q15_off_cx | 29.71% | +9.79pp     | [+7.35, +12.34]  | 0.0000 |
| membro ms2800_tau0.003_wd0.05_h512_q16_off_cx | 32.97% | +13.06pp    | [+10.19, +16.00] | 0.0000 |
| ENSEMBLE (6 modelli)                          | 41.25% | +21.34pp    | [+18.33, +24.49] | 0.0000 |
| MATRICE collassata                            | 36.63% | +16.72pp    | [+13.93, +19.58] | 0.0000 |

*Ogni confronto e' una differenza accoppiata con bootstrap gerarchico (`src/devset.bootstrap_paired`), non due medie affiancate.*

## Curva dei rendimenti dell'ensemble

| k | R@10   | vs baseline | vs k-1  | p     |
|---|--------|-------------|---------|-------|
| 1 | 31.55% | +11.63pp    | —       | —     |
| 2 | 36.61% | +16.69pp    | +5.06pp | 0.000 |
| 3 | 38.65% | +18.73pp    | +2.04pp | 0.003 |
| 4 | 39.98% | +20.07pp    | +1.33pp | 0.006 |
| 5 | 40.57% | +20.66pp    | +0.59pp | 0.121 |
| 6 | 41.25% | +21.34pp    | +0.68pp | 0.052 |

## Da dove viene la diversita' dell'ensemble

I 6 membri sono addestrati al livello `max_sources 2800` su un pool costruito a `4000`, e differiscono su **tre assi**: corpus di addestramento, campione di query, finestra di reference.

- i corpus di due membri condividono in media il **6.6%** delle immagini (Jaccard fra i sottoinsiemi di 20000 estratti dalle 162770 del train)
- le query condivise sono circa il 60% (frazione `BAG_FRAC` = 0.75)
- piu' la finestra circolare di reference, diversa per membro

Nella v1 il corpus era **uno solo per tutti**, e il guadagno del bagging era sceso da +5.16pp a +3.05pp mentre i singoli miglioravano di 2.2pp: e' la decomposizione di ambiguita' (membri migliori e piu' simili, meno da guadagnare mediando). Questo asse e' stato rimesso.

## Ricerca a coordinate (letta sulla validazione onesta)

| max_sources | tau   | wd     | val R@10 (media) | per seed     | dev R@10 (non usato) |
|-------------|-------|--------|------------------|--------------|----------------------|
| 2800        | 0.003 | 0.05   | 52.49%           | 52.20, 52.78 | 33.13%               |
| 2800        | 0.003 | 0.01   | 51.93%           | 51.72, 52.14 | 33.28%               |
| 2800        | 0.003 | 0.0001 | 51.90%           | 51.46, 52.35 | 33.38%               |
| 2800        | 0.003 | 0.05   | 50.36%           | 51.37, 49.34 | 33.53%               |
| 2800        | 0.003 | 0.2    | 50.04%           | 48.77, 51.30 | 34.01%               |
| 1400        | 0.007 | 0.0001 | 49.98%           | 49.13, 50.82 | 33.71%               |
| 2800        | 0.003 | 0.5    | 49.61%           | 49.75, 49.47 | 34.03%               |
| 2800        | 0.003 | 0.05   | 49.43%           | 50.37, 48.50 | 32.39%               |
| 1400        | 0.005 | 0.0001 | 48.83%           | 49.10, 48.56 | 33.44%               |
| 1400        | 0.003 | 0.0001 | 48.65%           | 49.89, 47.41 | 31.81%               |
| 1400        | 0.01  | 0.0001 | 47.37%           | 47.39, 47.35 | 34.67%               |
| 1400        | 0.02  | 0.0001 | 44.84%           | 41.68, 48.00 | 33.53%               |
| 700         | 0.003 | 0.0001 | 44.17%           | 45.20, 43.14 | 30.28%               |
| 1400        | 0.03  | 0.0001 | 39.01%           | 38.38, 39.64 | 32.31%               |
| 350         | 0.003 | 0.0001 | 35.56%           | 35.30, 35.81 | 28.57%               |

Punto scelto: **tau 0.003, max_sources 2800, wd 0.05**.
Sul set di sviluppo avrebbe vinto tau 0.01, ms 1400, wd 0.0001: **non e' stato usato**.

## Confronto accoppiato fra livelli, ai rispettivi punti d'arresto

Le viste sui livelli sono **annidate** (stesse query, prefisso di reference), quindi il confronto e' accoppiato; ogni livello e' fermato dalla propria validazione e gli esiti per source sono quelli dell'epoca scelta.

| confronto      | differenza | IC 95%         | p      | query   |
|----------------|------------|----------------|--------|---------|
| 700_vs_350_s0  | +2.04pp    | [+0.64, +3.41] | 0.0036 | 68V/28P |
| 700_vs_350_s1  | +1.38pp    | [-0.23, +3.00] | 0.0952 | 57V/44P |
| 1400_vs_350_s0 | +3.40pp    | [+1.85, +4.96] | 0.0000 | 76V/22P |
| 1400_vs_350_s1 | +3.09pp    | [+1.18, +4.97] | 0.0020 | 64V/34P |
| 2800_vs_350_s0 | +5.69pp    | [+4.14, +7.30] | 0.0000 | 87V/13P |
| 2800_vs_350_s1 | +3.93pp    | [+1.91, +5.97] | 0.0000 | 66V/30P |

## Collasso lineare

- `fit(media) == media(fit)`: max|W_ens - W_media| = **1.40e-06** (atteso ~0, la ridge e' lineare nel bersaglio)
- R² della mappa sull'uscita mediata, su dati mai usati per il fit: **0.9299**
- Recall@10 della matrice collassata: **36.63%**, vs ensemble -4.62pp (p = 0.0000)
- coppie (r, c) del fit: 18397, tutte da reference del **train**

## Come rileggere questi file

- `risultati.json` — tutto, comprese le traiettorie per epoca
- `esiti_devset.npz` — esiti per singola source di ogni metodo (`<metodo>_k10_flat` + `<metodo>_k10_len`), per rifare qualunque bootstrap
- `ckpt/*.pt` — un checkpoint per modello, all'**epoca scelta dalla validazione**
- `ckpt/matrice_collassata.pt` — la matrice W (gamma gia' dentro)
