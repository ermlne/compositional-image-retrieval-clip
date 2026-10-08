# Sessione di consolidamento su Colab — 29/08/2026

**Ensemble di 6 modelli: 36,73% di Recall@10 sul set di sviluppo, +16,82pp sulla baseline,
IC 95% [+14,01, +19,70], p < 0,0001.** Ogni numero è una differenza accoppiata con bootstrap
gerarchico, e gli esiti per singola source sono conservati **all'epoca scelta dalla
validazione**, non all'ultima.

Prodotta da `scripts/train_colab.py` (nilo) su GPU T4, 26 addestramenti, ~2 ore. Il
**benchmark ufficiale non è stato letto**: tutto sta sul set di sviluppo a 101 query.

I risultati completi sono in [`report.md`](report.md), generato dallo script. Questo file
aggiunge ciò che il report non contiene: il confronto 1400 contro 700, e tre osservazioni
emerse rileggendo i numeri.

---

## Dove eravamo, dove siamo

| | dev R@10 | vs baseline |
|---|---|---|
| baseline | 19,91% | — |
| contributo candidato, 5 seed (27/08) | 26,12–27,99% | +6,95pp medio |
| bagging di 4 modelli, Arianna (29/08) | 30,53% | +10,62pp |
| `max_sources` 1400 + bagging, sessione notturna (29/08) | 36,62% | +16,71pp, ma **senza IC**, modelli sovra-addestrati |
| **membro singolo, questa sessione** | **33,94%** | **+14,03pp** [+11,38, +16,79] |
| **matrice collassata** | **34,60%** | **+14,69pp** [+11,79, +17,75] |
| **ensemble di 6** | **36,73%** | **+16,82pp** [+14,01, +19,70] |

Il tetto oracolare con le etichette vere di CelebA resta **42,52%**.

---

## Il confronto che il report non contiene: 1400 contro 700

Lo script confronta ogni livello contro il più basso della griglia (350), quindi il passo
700 → 1400 non compare. Ricalcolato dagli esiti per source in [`esiti/`](esiti/), accoppiato,
ai rispettivi punti d'arresto:

```
seed 101:  1400 34,12%   700 33,29%   +0,83pp  [-0,49, +2,15]  p = 0,21   59V/34P
seed 102:  1400 34,22%   700 32,97%   +1,25pp  [-0,26, +2,76]  p = 0,10   55V/41P
                                       Fisher combinato:        p = 0,10
```

Contro il passo precedente, che è invece netto:

```
700  vs 350:  +3,87pp (p<0,0001)  e  +1,56pp (p=0,011)
1400 vs 350:  +4,71pp (p<0,0001)  e  +2,81pp (p=0,0016)
```

**Formulazione difendibile: da 350 a 700 il guadagno è dimostrato; da 700 a 1400 vale circa
1pp, coerente in segno su due seed, ma sotto la soglia di significatività.**

Il punto scelto è comunque 1400, perché la selezione guarda la validazione onesta, che lo
preferisce con un margine molto più largo (48,21% contro 43,08%). Quel margine è però
probabilmente gonfiato per la popolazione del set di sviluppo: **le query di validazione sono
solo a 2 e 3 condizioni**, cioè proprio quelle che beneficiano di più dei dati, mentre il dev
è 53/32/15%.

Riprodurre il calcolo:

```python
import numpy as np, sys; sys.path.insert(0, '.')
from src.devset import bootstrap_paired, macro
def spacchetta(z, pre, k=10):
    flat, lens = z[f"{pre}_k{k}_flat"], z[f"{pre}_k{k}_len"]
    ptr = np.concatenate([[0], np.cumsum(lens)])
    return [flat[ptr[i]:ptr[i+1]] for i in range(len(lens))]
for qs in (101, 102):
    a = np.load(f'scripts/run_colab/esiti/ms1400_tau0.01_wd0.0001_q{qs}.npz')
    b = np.load(f'scripts/run_colab/esiti/ms700_tau0.01_wd0.0001_q{qs}.npz')
    print(qs, bootstrap_paired(spacchetta(a,'dev'), spacchetta(b,'dev'), n_boot=20000, seed=0))
```

---

## La scoperta che nessuno cercava: la temperatura

Il guadagno più grande della ricerca non viene da `max_sources`, viene da **τ**. Il progetto
ha usato τ = 0,03 dal principio; la ricerca la porta a **0,01**:

| τ (a `max_sources` 700) | val R@10 | dev R@10 |
|---|---|---|
| **0,01** | **43,08%** | **33,13%** |
| 0,02 | 42,36% | 32,53% |
| 0,03 (il valore storico) | 38,12% | 32,34% |
| 0,05 | 32,53% | 30,93% |

Cinque punti di validazione fra 0,01 e 0,03, e un crollo a 0,05.

**Va detto nel report, perché ribalta una decisione presa mesi fa.** La prima versione del
contributo (`contributo-candidato/cross_attention.py`) era fallita proprio con τ = 0,01,
importato da un contesto con negativi minati, e il controllo positivo non passava. Con
abbastanza esempi quella scelta è ora la migliore: non era sbagliato il valore, era
sbagliato il regime di dati in cui era stato provato.

---

## Tre osservazioni sulla sessione

### 1. La griglia del weight decay non ha misurato niente

`wd` 1e-4 e 1e-3 danno numeri **identici a due decimali** su validazione *e* dev
(47,89 / 48,53 e 34,12 / 34,22 per entrambi). Non è collisione di cache: `tag_di` include
`wd` e i checkpoint sono file distinti.

È che con AdamW disaccoppiato l'aggiornamento è `p ← p(1 − lr·λ)`, e con `lr = 3e-4` su
~6.500 passi i pesi si contraggono di:

| `wd` | contrazione totale |
|---|---|
| 1e-4 | **0,02%** |
| 1e-3 | **0,19%** |
| 1e-2 | 1,93% |

Sono tutti zero in pratica. Per avere un effetto servirebbe un intervallo 10-100 volte più
alto. Due celle su otto sono state spese per nulla — e il passo C della ricerca a coordinate,
così com'è, non può che restituire l'incumbente.

### 2. A `max_sources` 350 validazione e dev si contraddicono

| seed | epoca scelta | val | dev |
|---|---|---|---|
| 101 | 44 | 41,56% | **29,42%** |
| 102 | 18 | 34,28% | **31,41%** |

L'ordine è **invertito**: il seed con la validazione più alta ha il dev più basso, e la
differenza è di 7,3pp in validazione. Su quel livello la validazione a 150 query non è
affidabile, e i confronti contro 350 vanno letti con questa riserva.

### 3. L'ensemble satura a cinque membri, il sesto peggiora

| k | R@10 | vs k−1 | p |
|---|---|---|---|
| 3 | 36,13% | +0,86pp | 0,019 |
| 4 | 36,40% | +0,27pp | 0,407 |
| 5 | **36,92%** | +0,52pp | 0,058 |
| 6 | 36,73% | −0,19pp | 0,419 |

**Tre modelli prendono il 95% del guadagno a metà del costo di inferenza.** Se il costo conta,
k=3 è il punto giusto; k=6 non si giustifica.

---

## Il tetto sulle epoche era davvero di parte

Alzarlo da 30 a 60 (commit `039a990`) non è stato precauzionale: **sette run su 26 hanno
superato le 30 epoche, e quattro hanno scelto la loro epoca migliore oltre la 30** (34, 38,
40, 44). Con il tetto originale sarebbero stati troncati.

Il caso più netto è proprio `max_sources` 350, che sceglie l'**epoca 44** — esattamente il
livello lento che il confronto a epoca fissa penalizzava. Il tetto a 60 non ha mai legato
nessuno: il massimo raggiunto è 50.

---

## Una nota sul salvataggio

Lo script è finito in `scripts/run_locale` sulla VM di Colab invece che su Drive, e i file
sono stati recuperati a mano prima che la sessione morisse. Causa: `_monta_drive()` chiama
`drive.mount()`, ma lo script gira come `!python ...`, cioè in un **processo separato** dal
kernel di Colab, e `google.colab.drive.mount` ha bisogno del canale RPC del notebook. L'import
riesce, il mount fallisce, e il fallback scrive in locale **con un semplice avviso**.

Per la prossima sessione: `--out /content/drive/MyDrive/DL-Project-2026/run`, che ha la
precedenza e salta del tutto `_monta_drive()`. Con Drive già montato dalla cella del notebook
quel percorso è un normale percorso di filesystem, visibile a qualunque processo della VM.

---

## File

| | |
|---|---|
| [`report.md`](report.md) | il report generato dallo script, con tutte le tabelle |
| `risultati.json` | tutto, comprese le traiettorie per epoca dei 26 addestramenti |
| `esiti_devset.npz` | esiti per singola source di ogni metodo, per rifare qualunque bootstrap |
| `esiti/*.npz` | esiti per source di ogni singolo run, all'epoca scelta |
| `ckpt/*.pt` | **non in git** (`.pt` è in `.gitignore`): 83 MB, restano sulla macchina di Filippo |

I checkpoint includono `matrice_collassata.pt`, cioè la matrice `W` con `gamma` già dentro.
Per rigenerarli servono ~2 ore di T4.
