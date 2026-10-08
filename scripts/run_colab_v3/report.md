# Sessione di consolidamento — 30/08/2026 07:49

Device: `cuda`. Baseline sul set di sviluppo: **19.91%** (atteso 19.91%, collaudo superato).
**Questo script non legge mai `celeba_evaluation.json`.** Il budget del benchmark ufficiale e' per metodo ed e' gia' stato speso una volta, il 25/08, per il filtro zero-shot, con esito negativo (`scripts/README.md`): la valutazione del contributo sul benchmark si fa dopo, a mano, a configurazione congelata.
**Validazione riparata (v3)**: 148 query valutate su una galleria di 20000 immagini **disgiunta** da quella dell'addestramento (R1), con la composizione del benchmark ufficiale invece che di sole query composte (R2). Nella v2 la galleria era condivisa e lo Spearman validazione-dev era crollato da +0.897 a +0.244.
Nessuna ricerca a coordinate: si sceglie fra **due candidati** gia' misurati, sulla sola validazione. Il dev non entra mai nella scelta.

## Risultati sul set di sviluppo (R@10, macro-average su 101 query)

| metodo                                         | R@10   | vs baseline | IC 95%           | p      |
|------------------------------------------------|--------|-------------|------------------|--------|
| baseline                                       | 19.91% | —           | —                | —      |
| membro ms1400_tau0.01_wd0.0001_h512_q11_off_cx | 32.91% | +13.00pp    | [+10.07, +15.97] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q12_off_cx | 35.28% | +15.37pp    | [+12.56, +18.28] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q13_off_cx | 34.46% | +14.54pp    | [+11.82, +17.60] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q14_off_cx | 33.60% | +13.69pp    | [+10.83, +16.86] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q15_off_cx | 34.47% | +14.56pp    | [+11.89, +17.32] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q16_off_cx | 34.05% | +14.13pp    | [+11.38, +17.06] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q17_off_cx | 35.51% | +15.60pp    | [+12.78, +18.56] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q18_off_cx | 31.94% | +12.03pp    | [+9.69, +14.38]  | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q19_off_cx | 34.55% | +14.63pp    | [+11.76, +17.61] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q20_off_cx | 33.70% | +13.79pp    | [+10.89, +16.78] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q21_off_cx | 33.23% | +13.31pp    | [+10.70, +15.92] | 0.0000 |
| membro ms1400_tau0.01_wd0.0001_h512_q22_off_cx | 33.93% | +14.02pp    | [+11.41, +16.59] | 0.0000 |
| ENSEMBLE (12 modelli)                          | 41.25% | +21.33pp    | [+18.62, +24.16] | 0.0000 |
| MATRICE collassata                             | 39.29% | +19.38pp    | [+16.66, +22.27] | 0.0000 |

*Ogni confronto e' una differenza accoppiata con bootstrap gerarchico (`src/devset.bootstrap_paired`), non due medie affiancate.*

## Curva dei rendimenti dell'ensemble

| k  | R@10   | vs baseline | vs k-1  | p     |
|----|--------|-------------|---------|-------|
| 1  | 32.91% | +13.00pp    | —       | —     |
| 2  | 37.35% | +17.44pp    | +4.44pp | 0.000 |
| 3  | 39.54% | +19.63pp    | +2.19pp | 0.000 |
| 4  | 40.03% | +20.12pp    | +0.49pp | 0.162 |
| 5  | 40.40% | +20.49pp    | +0.36pp | 0.239 |
| 6  | 40.48% | +20.57pp    | +0.08pp | 0.823 |
| 7  | 40.73% | +20.81pp    | +0.25pp | 0.381 |
| 8  | 40.54% | +20.63pp    | -0.19pp | 0.394 |
| 9  | 40.80% | +20.88pp    | +0.25pp | 0.265 |
| 10 | 41.03% | +21.11pp    | +0.23pp | 0.307 |
| 11 | 41.14% | +21.22pp    | +0.11pp | 0.646 |
| 12 | 41.25% | +21.33pp    | +0.11pp | 0.604 |

## Da dove viene la diversita' dell'ensemble

I 12 membri sono addestrati al livello `max_sources 1400` su un pool costruito a `4000`, e differiscono su **tre assi**: corpus di addestramento, campione di query, finestra di reference.

- i corpus di due membri condividono in media il **6.6%** delle immagini (Jaccard fra i sottoinsiemi di 20000 estratti dalle 162770 del train)
- le query condivise sono circa il 60% (frazione `BAG_FRAC` = 0.75)
- piu' la finestra circolare di reference, diversa per membro

Nella v1 il corpus era **uno solo per tutti**, e il guadagno del bagging era sceso da +5.16pp a +3.05pp mentre i singoli miglioravano di 2.2pp: e' la decomposizione di ambiguita' (membri migliori e piu' simili, meno da guadagnare mediando). Questo asse e' stato rimesso.

## I due candidati, e il collaudo della riparazione

Nessuna ricerca a coordinate: in v1 aveva scelto bene, in v2 ha scelto 1.5pp **peggio** del meglio che aveva provato, perche' leggeva uno strumento rotto. Qui si confrontano due soli punti, entrambi gia' misurati, con lo strumento riparato.

| candidato | tau   | max_sources | wd     | val R@10 | val per seed | dev R@10 (non usato) |
|-----------|-------|-------------|--------|----------|--------------|----------------------|
| A         | 0.003 | 2800        | 0.05   | 30.89%   | 30.58, 31.20 | 32.50%               |
| B         | 0.01  | 1400        | 0.0001 | 33.25%   | 34.21, 32.29 | 34.10%               |

- vincitore secondo la **validazione**: **B**
- vincitore secondo il **dev** (non usato per scegliere): B
- **i due strumenti sono D'ACCORDO**; Spearman validazione-dev su tutti i run: -0.086 (v1 +0.897, v2 +0.244)

Punto adottato: **tau 0.01, max_sources 1400, wd 0.0001, hidden 512**.

Differenza fra i due candidati sul set di sviluppo, per seed (non usata per scegliere, serve a dire di quanto differiscono):

| seed   | A - B   | IC 95%         | p      | query   |
|--------|---------|----------------|--------|---------|
| seed11 | +0.31pp | [-1.11, +1.83] | 0.6700 | 43V/50P |
| seed12 | -3.50pp | [-5.14, -1.98] | 0.0000 | 24V/72P |

## Collasso lineare

- `fit(media) == media(fit)`: max|W_ens - W_media| = **1.19e-07** (atteso ~0, la ridge e' lineare nel bersaglio)
- R² della mappa sull'uscita mediata, su dati mai usati per il fit: **0.9673**
- Recall@10 della matrice collassata: **39.29%**, vs ensemble -1.96pp (p = 0.0000)
- coppie (r, c) del fit: 18397, tutte da reference del **train**

## Come rileggere questi file

- `risultati.json` — tutto, comprese le traiettorie per epoca
- `esiti_devset.npz` — esiti per singola source di ogni metodo (`<metodo>_k10_flat` + `<metodo>_k10_len`), per rifare qualunque bootstrap
- `ckpt/*.pt` — un checkpoint per modello, all'**epoca scelta dalla validazione**
- `ckpt/matrice_collassata.pt` — la matrice W (gamma gia' dentro)
