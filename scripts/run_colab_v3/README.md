# Sessione v3 — 30/08/2026: la riparazione ha funzionato, e la configurazione è congelata

**Ensemble di 12 modelli: 41,25% di Recall@10, +21,33pp sulla baseline, IC 95%
[+18,62, +24,16], p < 0,0001.** E la **matrice collassata** — una formula, costo di inferenza
della baseline — arriva a **39,29%**, a soli 1,96pp dall'ensemble completo.

Ma il risultato che conta di più non è il livello: è che **lo strumento di selezione è tornato
a funzionare**, e la sua prima decisione è stata ribaltare quella della v2.

Prodotta da `scripts/train_colab_v3.py` su GPU T4, 18 addestramenti. Il **benchmark ufficiale
non è stato letto**.

---

## La riparazione ha funzionato, e si vede da un numero solo

Il divario fra il livello della validazione e quello del set di sviluppo, nelle tre sessioni:

| | validazione | dev | divario |
|---|---|---|---|
| v1 | 43,12% | 32,84% | **+10,29pp** |
| v2 | 48,53% | 32,80% | **+15,73pp** |
| **v3** | **31,78%** | **33,74%** | **−1,96pp** |

Con R1 (galleria disgiunta) e R2 (composizione 54/31/15) la validazione ha smesso di misurare
un compito più facile: adesso misura **quasi esattamente lo stesso** del set di sviluppo. È la
prova diretta che il difetto diagnosticato era quello giusto — la validazione della v2 era
gonfiata perché condivideva la galleria con l'addestramento e guardava solo le query composte.

### Il collaudo pre-dichiarato: superato

La regola era scritta prima di guardare i numeri: **validazione e dev devono scegliere lo
stesso candidato.**

| candidato | τ | `max_sources` | `wd` | validazione | dev *(non usato)* |
|---|---|---|---|---|---|
| A — il punto della v2 | 0,003 | 2800 | 0,05 | 30,89% | 32,50% |
| **B — il migliore sul dev nella v2** | **0,01** | **1400** | **1e-4** | **33,25%** | **34,10%** |

Entrambi scelgono **B**. E qui c'è la parte che vale la pena leggere due volte: **la
validazione riparata ha ribaltato la scelta della v2.** La v2 aveva adottato A perché il suo
strumento rotto lo preferiva di 5pp; il suo dev diceva il contrario, e non era stato ascoltato
— giustamente, perché guardare il dev per scegliere è la trappola che ha rovesciato sei
conclusioni di questo progetto. Ora lo strumento riparato indica **B da solo**, cioè
esattamente ciò che il dev aveva sempre indicato.

Sul set di sviluppo, per seed, B batte A di +3,50pp (p < 0,0001) su un seed e pareggia
sull'altro (+0,31pp a favore di A, p = 0,67).

### Sullo Spearman stampato nel report: non è confrontabile, e non è un fallimento

Il report riporta uno Spearman validazione–dev di **−0,086**, che sembra peggio del +0,244
della v2. **Non lo è, ed è un artefatto della popolazione di run.**

| | configurazioni distinte | Spearman su quelle |
|---|---|---|
| v1 | 9 | +0,950 |
| v2 | 15 | +0,343 |
| **v3** | **2** | **non calcolabile** |

In v1 e v2 la correlazione era calcolata su una griglia che spaziava da τ 0,003 a 0,05 e da
`max_sources` 350 a 2800: differenze vere, e un ordinamento da riprodurre. In v3 **non c'è
griglia**: 16 dei 18 run sono repliche della *stessa* configurazione, che differiscono solo per
il seed. Fra repliche identiche la differenza validazione–dev è puro rumore, quindi una
correlazione di rango prossima a zero è **ciò che ci si deve attendere**.

La prova della riparazione è quindi il divario che si è chiuso (−1,96pp) e l'accordo sulla
decisione, non quel numero.

---

## Il risultato

| metodo | R@10 | vs baseline | IC 95% |
|---|---|---|---|
| baseline | 19,91% | — | — |
| membri singoli (12) | 31,94–35,51% | media **33,97%** | |
| **matrice collassata** | **39,29%** | **+19,38pp** | [+16,66, +22,27] |
| **ENSEMBLE di 12** | **41,25%** | **+21,33pp** | [+18,62, +24,16] |

Tetto oracolare con le etichette vere: 42,52%.

Confronto con la v2, a parità di tutto il resto del protocollo:

| | v2 (τ 0,003, ms 2800) | v3 (τ 0,01, ms 1400) |
|---|---|---|
| membri, media | 32,14% | **33,97%** |
| ensemble | 41,25% | 41,25% |
| **matrice collassata** | 36,63% | **39,29%** |
| R² del collasso | 0,9299 | **0,9673** |
| guadagno del bagging | +9,12pp | +7,28pp |

Stesso ensemble, **membri migliori di 1,83pp**, e soprattutto una **matrice migliore di
2,66pp** — ottenuti a `max_sources` **1400 invece di 2800**, cioè a metà del costo di
addestramento. Il guadagno del bagging scende (+7,28 contro +9,12) perché i membri sono più
bravi e più simili: è la decomposizione di ambiguità, e va nella direzione giusta.

Il compenso è **esatto**, ed è la conferma più netta della decomposizione che abbiamo:
i membri salgono di 1,83pp e l'ambiguità scende di 1,84pp, quindi le due sessioni atterrano
sullo **stesso** ensemble (41,25% entrambe, a meno di 0,004pp). Quello che si guadagna in
qualità dei singoli si restituisce in diversità, uno a uno.

*(La media dei membri v2 è stata corretta il 30/08 da 32,77% a 32,14%: vedi la nota in*
[`../run_colab_v2/README.md`](../run_colab_v2/README.md)*.)*

---

## Quanti membri consegnare: quattro, non dodici

| k | 2 | 3 | **4** | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| R@10 | 37,35% | 39,54% | **40,03%** | 40,40% | 40,48% | 40,54% | 41,03% | 41,25% |
| vs k−1 | +4,44pp | +2,19pp | +0,49pp | +0,36pp | +0,08pp | −0,19pp | +0,23pp | +0,11pp |
| p | <0,001 | <0,001 | 0,162 | 0,239 | 0,823 | 0,394 | 0,307 | 0,604 |

**Dal quarto membro in poi nessun incremento è significativo.** Quattro modelli danno 40,03%,
dodici ne danno 41,25%: **+1,22pp per il triplo del costo di inferenza.**

È la ragione per cui i dodici membri sono stati addestrati — la v2 si fermava a sei con la
curva ancora in salita, e senza arrivare a dodici non si poteva sapere dove finiva. Ora si sa,
ed è una scelta misurata invece che un'abitudine.

---

## Le tre righe da portare alla decisione sul benchmark

| cosa consegnare | dev R@10 | costo di inferenza | argomento |
|---|---|---|---|
| ensemble di 12 | 41,25% | ×12 | il numero più alto |
| **ensemble di 4** | **40,03%** | ×4 | −1,22pp non significativi, un terzo del costo |
| **matrice collassata** | **39,29%** | **×1, come la baseline** | si scrive in una formula; pesa su *originalità* e *chiarezza* |
| membro singolo | 33,97% | ×1 | il metodo nella sua forma minima |

Con 13 query distinte il benchmark risolve solo differenze **sopra ~6,8pp**, quindi tutte e
quattro stanno comodamente sopra la soglia del rilevabile — a patto che il guadagno
trasferisca, e **nulla in questa repo dice quanto ne trasferisca**.

Personalmente riporterei la **matrice collassata** come metodo principale e l'ensemble come
estensione: 39,29% al costo della baseline è un risultato più difendibile di 41,25% a costo
dodici, e la formula `W_r r + W_c c + b` si scrive in una riga.

---

## Cosa resta aperto

- **`max_sources` oltre 1400 non è più la strada.** In v2 sembrava che 2800 fosse meglio, ma
  quella preferenza veniva dalla validazione rotta: con lo strumento riparato 1400 vince, e
  costa metà.
- **Il weight decay e la larghezza dell'MLP non sono mai stati letti con lo strumento buono.**
  In v2 erano inconcludenti perché la validazione era rotta; in v3 non sono stati cercati
  affatto, per scelta. Restano ignoti.
- **La ri-estrazione degli embedding con `transformers`** (conformità alla sez. 3.3) va fatta
  **prima** di spendere il benchmark, non dopo.

---

## File

| | |
|---|---|
| `report.md` | il report generato dallo script |
| `risultati.json` | tutto, comprese le traiettorie per epoca dei 18 addestramenti |
| `esiti_devset.npz` | esiti per source di baseline, ensemble a 12 e matrice collassata |
| `esiti/*.npz` | esiti per source di ogni singolo run, all'epoca scelta dalla validazione |
| `ckpt/*.pt` | **non in git** (52 MB): sulla macchina di Filippo e su Drive in `run_v3/ckpt/` |

I dodici membri sono `ms1400_tau0.01_wd0.0001_h512_q{11..22}_off_cx.pt`; la matrice è
`matrice_collassata.pt`, con `gamma` già dentro.
