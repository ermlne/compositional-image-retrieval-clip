# Il modulo era limitato dai dati, non dal metodo

**In una riga: `max_sources` (quante reference per query entrano in addestramento) era fermo a
80 dal 24/08. Alzandolo, un modello solo passa da 26,40% a 32,26% di Recall@10, e con quattro
modelli mediati si arriva a 36,62% contro il 19,91% della baseline.**

Il benchmark ufficiale non è stato toccato. Tutto sul set di sviluppo a 101 query, con il
protocollo del progetto (bootstrap gerarchico accoppiato).

> **Correzione del 30/08 — la frase qui sopra e' imprecisa.** Il budget del benchmark
> ufficiale e' **per metodo**, ed e' gia' stato speso **una volta**: il 25/08, per il filtro
> zero-shot di `scripts/zeroshot_filter.py` (righe 143-160), con esito **negativo** (R@10
> 13.00% contro 15.92% della baseline, differenza accoppiata -2.92pp, IC [-5.59, +0.07], p=0.055;
> numeri registrati in `scripts/README.md`). Cio' che resta vero, ed e' cio' che questa pagina
> intende dire: **nessuna delle misure riportate qui e' stata fatta sul benchmark ufficiale.**


---

## Da dove viene ogni numero

Ci sono tre run distinti, con **seed e numero di epoche diversi**. Confrontare fra run è
lecito solo dentro la stessa riga di questa tabella, e ogni numero qui sotto è etichettato con
la sua provenienza.

| run | script | seed | epoche | output |
|---|---|---|---|---|
| **A** | `../braccio_b_piu_query.py` | 1 | 30 fisse | `../braccio_b_seed1_20260829-0055.txt` |
| **B** | `../curva_max_sources.py` | 5 | 30 fisse | `../curva_max_sources_seed5_20260829-0133.txt` |
| **C** | `notte.py` | 5 (curva), 6/7/8 (repliche) | 60 fisse | `notte_20260829-0227.txt` |

Il run B è stato interrotto durante il suo controllo sui passi; i tre livelli erano già
completi. Il run C rifà il controllo dall'inizio.

**Controllo di coerenza fra run**: il modello `max_sources` 350, seed 5, all'epoca 30 vale
32,26% sia nel run B (30 epoche fisse) sia nel run C (epoca 30 di 60). È lo stesso modello —
stesso seed, stessa configurazione, addestramento deterministico — e i due run lo confermano.

---

## Cosa conta: gli esempi, non le query

**Run A, seed 1, 30 epoche**, a parità di budget di esempi:

| braccio | `max_sources` | query valide | esempi | dev R@10 | vs baseline |
|---|---|---|---|---|---|
| 0 attuale | **80** | 182 | 4.433 | 26,53% | +6,62pp |
| A più reference | **175** | 189 | 9.529 | **30,22%** | +10,31pp |
| B più query | **80** (640 campionate) | 481 | 9.079 | 29,79% | +9,87pp |

```
PRIMARIA   B vs 0   +3,26pp  [+1,68, +4,86]  p < 0,0001   66 query vinte / 32 perse
CONTROLLO  A vs 0   +3,69pp  [+2,40, +5,11]  p < 0,0001   76 vinte / 21 perse
MECCANISMO B vs A   -0,43pp  [-1,73, +0,87]  p = 0,48     44 vinte / 48 perse
```

Aumentare le reference per query e aumentare le query distinte danno lo stesso risultato, e la
loro differenza è nulla. **Conta la quantità di esempi.**

Una conseguenza di cui tenere conto: lo spazio delle query a 1 condizione è **esaurito** (80 su
80 già usate), quindi il braccio B è per forza spostato verso le composte — composizione
16/54/30% contro 43/36/20% del braccio 0, e 53/32/15% del set di sviluppo. Il braccio A invece
mantiene la composizione (42/37/21%). È un motivo in più per preferire `max_sources`.

---

## La curva su `max_sources`

**Run B, seed 5, 30 epoche fisse:**

| `max_sources` | query valide | esempi | passi | dev R@10 | vs baseline |
|---|---|---|---|---|---|
| 80 | 171 | 4.422 | 540 | 26,40% | +6,49pp |
| 175 | 176 | 9.388 | 1.110 | 29,22% | +9,31pp |
| 350 | 182 | 18.222 | 2.160 | **32,26%** | +12,35pp |

**Run C, seed 5, 60 epoche fisse:**

| `max_sources` | query valide | esempi | passi | dev @ep60 | vs baseline |
|---|---|---|---|---|---|
| 350 | 182 | 18.222 | 4.320 | 31,06% | +11,14pp |
| 700 | 190 | 35.011 | 8.220 | 31,34% | +11,43pp |
| 1400 | 189 | 63.569 | 14.940 | 32,12% | +12,21pp |
| **controllo passi**: `max_sources` **80**, 240 epoche | 171 | 4.422 | 4.320 | **19,66%** | **−0,25pp** |

### Il controllo sui passi

Più esempi significa anche più batch per epoca, quindi più passi di gradiente a parità di
epoche: `max_sources` 350 ne fa 4.320 contro i 540 di `max_sources` 80. Se il guadagno fosse
ottimizzazione e non dati, dare a `max_sources` 80 gli **stessi 4.320 passi** — 240 epoche
invece di 30 — lo riprodurrebbe.

Non lo riproduce: fa **19,66%, sotto la baseline**, e **−11,40pp [−13,10, −9,70] p<0,0001**
contro `max_sources` 350. Lo stesso livello a 30 epoche faceva 26,40%; a 240 si autodistrugge.
Quadruplicare i passi senza aggiungere un solo esempio **peggiora**. Il guadagno è dei dati,
senza ambiguità.

---

## CORREZIONE: la curva NON satura a 350

**La prima versione di questo documento affermava che la curva satura oltre 350, sulla base dei
confronti accoppiati a 60 epoche fisse (+0,28pp p=0,64 per 700, +1,06pp p=0,16 per 1400).
Quell'affermazione era sbagliata, e il confronto su cui si reggeva è confondato.**

Il picco arriva **prima** quando i dati crescono. A epoca 60 tutti i livelli sono oltre il
picco, ma i livelli alti lo sono *di più*: il confronto a epoca fissa penalizza sistematicamente
proprio i livelli che si vogliono valutare.

Le tre letture dello **stesso** addestramento (run C, seed 5):

| `max_sources` | esempi | dev @ep60 | vs 350 | dev @epoca scelta sulla validazione | vs 350 | picco dev | vs 350 |
|---|---|---|---|---|---|---|---|
| 350 | 18.222 | 31,06% | — | 31,29% (ep 55) | — | 32,26% (ep 30) | — |
| 700 | 35.011 | 31,34% | +0,28 | 34,08% (ep 25) | **+2,79** | 34,08% (ep 25) | **+1,82** |
| 1400 | 63.569 | 32,12% | +1,06 | 34,28% (ep 30) | **+2,98** | 35,26% (ep 20) | **+3,00** |

Solo la prima colonna dice "satura". Le altre due dicono che **il passo 350 → 700 vale fra
+1,8 e +2,8pp** ed è reale.

Su **700 → 1400** le due letture non concordano: +0,19pp con l'arresto onesto, +1,18pp al
picco. **Resta aperto.**

### Cosa questa tabella NON permette di concludere

- **Nessun test di significatività** è disponibile sulle due colonne di destra. Il bootstrap
  accoppiato richiede gli esiti per singola source, e lo script li conserva solo per l'ultima
  epoca. Le differenze +2,79 e +2,98 sono differenze fra medie, senza intervallo.
- **La validazione è rumorosa**: 60 query soltanto. Per `max_sources` 350 sceglie l'epoca 55,
  che sul dev vale 31,29% contro i 32,26% dell'epoca 30 — una scelta chiaramente infelice. Il
  divario 350 → 700 letto su quella colonna è quindi probabilmente sovrastimato.
- **Un solo seed.** Lo spread fra seed sulla configurazione storica è 1,87pp.

### L'esperimento che chiude la questione

Serve una curva in cui ogni livello sia valutato al **proprio** punto d'arresto, scelto in
modo onesto, con gli esiti per source conservati **a quell'epoca**:

- livelli `max_sources` **350, 700, 1400, 2800**;
- set di validazione portato da 60 a ~150 query, per rendere stabile la scelta dell'epoca;
- il modello viene **salvato all'epoca scelta dalla validazione**, non all'ultima;
- bootstrap accoppiato fra livelli **ai rispettivi punti d'arresto**;
- almeno 2 seed per livello.

Finché quello non è fatto, la formulazione difendibile è: *da 80 a 350 il guadagno è grande e
misurato; da 350 a 700 c'è un guadagno di circa 2pp visibile in due letture indipendenti su un
seed; oltre 700 non lo sappiamo.*

---

## Le 60 epoche sono troppe, e più dati ci sono peggio è

Traiettoria del dev, registrata ogni 5 epoche a scopo diagnostico (run C, seed 5):

| `max_sources` | epoca del picco | dev al picco | dev a 60 epoche | perdita |
|---|---|---|---|---|
| 350 | 30 | 32,26% | 31,06% | −1,20pp |
| 700 | 25 | 34,08% | 31,34% | −2,74pp |
| 1400 | 20 | 35,26% | 32,12% | −3,14pp |

**I numeri al picco non sono risultati**: sono massimi presi su 12 valutazioni sul set di
valutazione, cioè la trappola che ha prodotto metà della tabella delle esclusioni di
`STATO.md`. Vanno letti solo come tetto ottimistico.

La versione **onesta** usa il set di validazione (60 query nuove, mai addestrate, disgiunte dal
set di sviluppo) per scegliere l'epoca, e legge il dev lì. Sui quattro modelli a
`max_sources` **1400** della fase 2:

| modello | epoca scelta | dev all'epoca scelta | dev a 60 epoche | recuperato |
|---|---|---|---|---|
| seed 5 | 30 | 34,28% | 32,12% | +2,15pp |
| seed 6 | 35 | 32,77% | 30,73% | +2,04pp |
| seed 7 | 40 | 32,07% | 30,99% | +1,09pp |
| seed 8 | 35 | 33,29% | 31,98% | +1,30pp |
| **media** | | **33,10%** (sd 0,93) | 31,46% (sd 0,70) | **+1,65pp** |

Un arresto anticipato onesto vale **+1,65pp in media**, fino a +2,15pp sul singolo modello.

---

## Il bagging e i dati si sommano

Quattro modelli a `max_sources` **1400** (seed 5, 6, 7, 8), **60 epoche**, correzioni mediate:

```
media dei 4 singoli    31,46%
ENSEMBLE dei 4         36,62%    +5,16pp sul singolo medio
   vs seed 5   +4,50pp   p < 0,0001
   vs seed 6   +5,89pp   p < 0,0001
   vs seed 7   +5,63pp   p < 0,0001
   vs seed 8   +4,64pp   p < 0,0001
```

**36,62% contro il 19,91% della baseline: +16,71pp.** Il tetto oracolare con le etichette vere
di CelebA è 42,52%.

Il confronto con il 30,53% del bagging di Arianna **non è appaiato**: lì erano 4 modelli a
`max_sources` 80 con checkpoint `multidominio`, qui 4 modelli a `max_sources` 1400 con
checkpoint `multiseed`. Vanno letti come due configurazioni diverse, non come un miglioramento
misurato dell'una sull'altra.

**Questa è una previsione sbagliata da parte mia, ed è giusto scriverlo.** Avevo argomentato
che i due interventi attaccano la stessa varianza e non si sarebbero accumulati: più dati per
membro → membri più simili → meno ambiguità → meno guadagno dall'ensemble. Il bagging valeva
+3,76pp a `max_sources` 80 e a 1400 vale **+5,16pp**, cioè *di più*. La decomposizione di
ambiguità resta valida in generale, ma qui i campioni dei quattro modelli restano quasi
disgiunti anche a 1400 (1.400 reference su un bacino mediano di ~1.800 per le query a 3
condizioni, e su ~10.000 per quelle a 1 condizione), quindi la diversità non si esaurisce.

**L'ensemble è però fatto di modelli sovra-addestrati**: i suoi quattro membri sono le versioni
a 60 epoche, che valgono 31,46% invece dei 33,10% che varrebbero con l'arresto onesto. Un
ensemble di modelli fermati bene non è stato misurato.

---

## Il bacino di reference, e quando `N_TRAIN_CORPUS` diventa il vincolo

Misurato sul corpus di addestramento da 20.000 immagini (`../quante_query_in_piu.py` e la
misura del bacino), per le 220 query campionate con seed 5:

| condizioni | reference disponibili, mediana | minimo | 10° percentile |
|---|---|---|---|
| 1 | 10.000 | 469 | 1.112 |
| 2 | 5.202 | 0 | 497 |
| 3 | 1.782 | 2 | 129 |

| `max_sources` | query il cui bacino si esaurisce (su 220) |
|---|---|
| 350 | 15 (7%) |
| 700 | 29 (13%) |
| 1400 | 61 → **28%** a 1500 |
| 3000 | 85 (39%) |

Il corpus non è il vincolo fino a `max_sources` ~700, e comincia a mordere **prima sulle query
composte**, che sono quelle dove il contributo rende di più. Per andare oltre 1400 serve o
alzare `N_TRAIN_CORPUS` (20.000 su 162.770 disponibili, ma il costo per batch scala
linearmente perché il denominatore InfoNCE è su tutto il corpus), oppure disaccoppiare i due
ruoli: **galleria a 20.000, bacino di reference su tutte le 162.770**. La seconda toglie il
tetto a costo per esempio invariato.

---

## File

| | |
|---|---|
| `notte.py` | lo script delle tre fasi (curva, repliche, bagging) |
| `notte_20260829-0227.txt` | log completo, una riga per epoca |
| `notte_20260829-0227.json` | risultati e traiettorie complete |
| `checkpoints/*.pt` | un modello per configurazione — **non in git**, `.pt` è in `.gitignore` |
| `collaudo_*`, `*_COLLAUDO1EPOCA.pt` | un run da 1 epoca per verificare il flusso: **non sono risultati** |

Per rigenerare:

```bash
HF_HUB_OFFLINE=1 python esperimenti/notte/notte.py --epochs 60 --livelli 350 700 1400
```

Circa 6 ore su CPU a 8 thread; con `--livelli 350` soltanto, circa 40 minuti.

---

## Nota di riproducibilità

I quattro modelli di `contributo-candidato/multiseed.py` sono stati riaddestrati da zero su
questa macchina per avere i checkpoint (che non sono in git). Riproducono il report di
Arianna: 26,5313 / 26,1173 / 26,4144 / 27,9832 contro 26,53 / 26,12 / 26,41 / 27,99. La sola
differenza è **0,005pp sul seed 4**, rumore numerico da riduzioni BLAS dipendenti dal numero di
thread. I file `contributo-candidato/multiseed/*` **non** sono stati modificati: sono stati
ripristinati alla versione di Arianna.
