# Sessione v2 — 30/08/2026: il bagging triplica, la validazione si rompe

**Ensemble di 6 modelli: 41,25% di Recall@10 sul set di sviluppo, +21,34pp sulla baseline,
IC 95% [+18,33, +24,49], p < 0,0001.** È il numero più alto mai misurato nel progetto, contro
un tetto oracolare di 42,52%.

Ma la sessione ha anche **rotto lo strumento di selezione**, e va detto con la stessa
chiarezza: la correlazione fra validazione e set di sviluppo è crollata da **+0,90 a +0,24**,
e la configurazione scelta è ~1,5pp *peggiore* sul dev della migliore che era stata provata.
Il guadagno viene dalla modifica architetturale, non dalla ricerca.

Prodotta da `scripts/train_colab_v2.py` su GPU T4, 40 addestramenti. Il **benchmark ufficiale
non è stato letto**.

> **CORREZIONE del 30/08, segnalata da Arianna.** La media dei sei membri era riportata come
> **32,77%**; quel numero non è la media dei membri, è il `dev_r10` del **modello singolo della
> fase di minimo** di questa stessa sessione (`fm_minimo`, che gira con la configurazione della
> v1: `ms 1400, tau 0,01, wd 1e-4`). Avevo letto il campo sbagliato del JSON. La media vera dei
> sei è **32,14%** (`fasi.f3_finali.media_singoli` = 32,1369), e il guadagno del bagging è
> quindi **+9,12pp**, non +8,48pp. Corretto qui e in tutti i documenti che lo citavano.
> **La conclusione non cambia, si rafforza**: i membri sono ancora più deboli e il bagging rende
> ancora di più. Nello stesso passaggio è stato corretto il guadagno della v1, +3,05 → **+3,06pp**
> (era 3,0556 troncato invece che arrotondato); i `report.md` generati dallo script portano
> ancora +3,05pp perché sono artefatti di esecuzione e non vanno riscritti a mano.

---

## Il risultato

| metodo | R@10 | vs baseline | IC 95% | p |
|---|---|---|---|---|
| baseline | 19,91% | — | — | — |
| membri singoli (6) | 29,71–33,49% | media **32,14%** | | |
| **matrice collassata** | **36,63%** | **+16,72pp** | [+13,93, +19,58] | <0,0001 |
| **ENSEMBLE di 6** | **41,25%** | **+21,34pp** | [+18,33, +24,49] | <0,0001 |

La matrice collassata da sola supera l'**intero ensemble della v1** (36,73%), al costo di
inferenza della baseline.

---

## Cosa ha funzionato: il corpus per membro

Era la modifica principale della v2, ed è quella che ha prodotto il salto.

| sessione | corpus dei membri | membri (media) | ensemble | **guadagno del bagging** |
|---|---|---|---|---|
| notturna 29/08 | uno per seed | 31,46% | 36,62% | +5,16pp |
| Colab v1 29/08 | **condiviso** | 33,67% | 36,73% | +3,06pp |
| **Colab v2 30/08** | **uno per membro** | 32,14% | **41,25%** | **+9,12pp** |

Il pattern è la decomposizione di ambiguità, e si legge in orizzontale: nella v1 i membri
erano i **migliori** delle tre sessioni (33,67%) e l'ensemble il **peggiore** in termini di
guadagno (+3,06pp), perché condividevano tutto. Qui i membri sono individualmente più deboli
(32,14%, 1,53pp sotto la v1) ma il bagging rende **tre volte tanto**.

La diversità è misurata, non asserita: due membri condividono il **6,6%** delle immagini del
corpus (Jaccard fra sottoinsiemi di 20.000 estratti dalle 162.770 del train), circa il 60%
delle query, più finestre di reference diverse.

**E l'ensemble non è ancora saturo.** In v1 il sesto membro peggiorava (−0,19pp); qui rende
ancora +0,68pp (p = 0,052):

| k | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| vs k−1 | +5,06pp | +2,04pp | +1,33pp | +0,59pp | +0,68pp |
| p | <0,001 | 0,003 | 0,006 | 0,121 | 0,052 |

Aggiungere membri è la strada più economica che resta.

---

## Cosa si è rotto: la validazione non seleziona più

```
Spearman validazione–dev, celle della ricerca:   v1 +0,897   ->   v2 +0,244
Spearman su tutti i run:                                          v2 -0,013
```

Il segno di questo si vede a occhio nella tabella della ricerca: **più sale la validazione,
più scende il dev.**

| `max_sources` | τ | `wd` | val | dev |
|---|---|---|---|---|
| 2800 | 0,003 | 0,05 | **52,49%** ← scelto | 33,13% |
| 2800 | 0,003 | 1e-4 | 51,90% | 33,38% |
| 2800 | 0,003 | 0,2 | 50,04% | 34,01% |
| 2800 | 0,003 | 0,5 | 49,61% | 34,03% |
| 1400 | 0,01 | 1e-4 | 47,37% | **34,67%** ← il migliore sul dev, 11° in validazione |

La selezione ha quindi scelto una configurazione **peggiore di ~1,5pp sul dev** di una che
aveva già provato. Lo script lo dichiara da solo (*"Sul set di sviluppo avrebbe vinto tau
0.01, ms 1400, wd 0.0001: non è stato usato"*), ed è giusto che non l'abbia usato — guardare
il dev per scegliere è la trappola che ha rovesciato sei conclusioni di questo progetto. Ma
significa che **lo strumento di selezione va riparato prima della prossima sessione.**

### Perché si è rotta, e perché proprio adesso

La validazione è valutata su `E_tr`, cioè **le stesse 20.000 immagini che stanno nel
denominatore dell'InfoNCE durante l'addestramento**. Le query sono nuove, la galleria no.

Finché gli iperparametri restavano moderati questo non si notava. Ma la v2 ha allargato le
griglie **esattamente nella direzione che sfrutta quella galleria**: τ a 0,003 rende la
softmax dell'InfoNCE estremamente piccata su quel corpus, e `max_sources` 2800 pesca il
massimo numero di reference da quel corpus. La validazione, che vive lì dentro, premia
entrambe. Il dev, che sta su uno split diverso, no.

**La validazione non può rilevare il sovradattamento alla galleria, perché condivide la
galleria.** E i due iperparametri scelti sono finiti entrambi sull'**estremo** della propria
griglia (τ il più basso, `max_sources` il più alto), che è la firma di una ricerca che sta
scivolando invece di trovare un ottimo.

### La riparazione

Due modifiche alla costruzione della validazione, entrambe piccole:

1. **valutarla su una fetta del train tenuta fuori dalla galleria** dell'InfoNCE, invece che
   sulle stesse immagini viste in addestramento. È ciò che rende la validazione capace di
   vedere il sovradattamento al corpus;
2. **comporla 54/31/15%** come il benchmark ufficiale (e come il set di sviluppo), invece
   dell'attuale 0/50/50 di sole query composte. Oggi misura solo lo strato che beneficia di
   più dei dati, ed è il motivo per cui sovrastima sistematicamente i livelli alti di
   `max_sources`.

Finché non sono fatte, **estendere ancora le griglie è controproducente**: la ricerca
camminerebbe più in fretta nella direzione sbagliata.

---

## `max_sources`: la curva continua a salire

Confronti accoppiati sulle viste annidate, ai rispettivi punti d'arresto:

| | seed 101 | seed 102 |
|---|---|---|
| 700 vs 350 | +2,04pp (p=0,004) | +1,38pp (p=0,095) |
| 1400 vs 350 | +3,40pp (p<0,0001) | +3,09pp (p=0,002) |
| 2800 vs 350 | **+5,69pp** (p<0,0001) | **+3,93pp** (p<0,0001) |

E il passo che il report non calcola — ricavato dagli esiti per source, a τ 0,003 e wd 1e-4:

```
2800 vs 1400   seed 101:  +2,29pp  [+0,99, +3,64]  p = 0,0007
               seed 102:  +0,85pp  [-0,54, +2,26]  p = 0,2370
```

Quindi 350 → 700 → 1400 → 2800 è **monotono e ancora in salita**, e il passo 1400 → 2800 è
significativo su un seed. Ma attenzione: a 2800 il bacino di reference si esaurisce su una
query su tre con un corpus di 20.000, quindi il livello successivo richiede prima di alzare
`N_TRAIN_CORPUS`.

---

## Cosa NON ha funzionato

**Il weight decay resta inconcludente**, anche nella gamma giusta. Le celle a `wd` 0,05 / 0,2
/ 0,5 danno val 52,49 / 50,04 / 49,61 e dev 33,13 / 34,01 / 34,03: la validazione preferisce
il più piccolo, il dev il più grande. Con la validazione rotta non se ne può concludere nulla.

**Il passo D sulla larghezza** ha scelto 512, cioè l'incumbente: nessuna delle riduzioni
(256, 128) ha superato il margine di 0,30pp in validazione. Anche questo va riletto quando la
validazione sarà riparata, perché è una delle poche leve anti-overfitting con un argomento
teorico dietro.

---

## Cosa se ne porta via il progetto

1. **Il numero da riportare**: ensemble di 6 a **41,25%**, +21,34pp con IC contro la baseline,
   su un tetto oracolare di 42,52%. E la matrice collassata a **36,63%** per chi guarda al
   costo di inferenza.
2. **La diversità dei dati è la leva più forte del bagging**, e ora è quantificata: stesso
   metodo, stessi iperparametri, corpus condiviso contro corpus per membro → +3,06pp contro
   +9,12pp.
3. **Un set di validazione che condivide la galleria con l'addestramento non può rilevare il
   sovradattamento alla galleria**, e la ricerca a coordinate ci cammina dentro. È un
   risultato metodologico, ed è riportabile.

---

## File

| | |
|---|---|
| `report.md` | il report generato dallo script |
| `risultati.json` | tutto, comprese le traiettorie per epoca dei 40 addestramenti |
| `esiti_devset.npz`, `esiti/*.npz` | esiti per singola source, per rifare qualunque bootstrap |
| `ckpt/*.pt` | **non in git** (119 MB): restano sulla macchina di Filippo e su Drive |

Riprodurre il confronto 2800 contro 1400:

```python
import numpy as np, sys; sys.path.insert(0, '.')
from src.devset import bootstrap_paired, macro
def spac(z, pre, k=10):
    f, l = z[f"{pre}_k{k}_flat"], z[f"{pre}_k{k}_len"]
    p = np.concatenate([[0], np.cumsum(l)])
    return [f[p[i]:p[i+1]] for i in range(len(l))]
for qs in (101, 102):
    a = np.load(f'scripts/run_colab_v2/esiti/ms2800_tau0.003_wd0.0001_h512_q{qs}.npz')
    b = np.load(f'scripts/run_colab_v2/esiti/ms1400_tau0.003_wd0.0001_h512_q{qs}.npz')
    print(qs, bootstrap_paired(spac(a, 'dev'), spac(b, 'dev'), n_boot=20000, seed=0))
```
