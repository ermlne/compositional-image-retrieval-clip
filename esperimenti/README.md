# esperimenti/

*Scritto il 30/08/2026 durante il fact-check pre-consegna. Questa cartella era l'unica del
progetto senza README: i suoi 22 script e i suoi 54 file di risultato (`.json` e `.txt`) erano
leggibili solo aprendoli uno per uno.*

Due famiglie di lavoro, sedimentate in giorni diversi:

- **28/08 — perche' il contributo non generalizza agli attributi mai visti** (Arianna). Undici
  script nati come **fork di `contributo-candidato/attributi_heldout.py`**, che ne condividono
  l'impalcatura: 3 split leave-attribute-out (10 attributi tolti su 40), tau 0.03, 30 epoche,
  `max_sources` 80, corpus di 20.000 immagini del train, tre strati di valutazione
  (visti / MAI VISTI / misti), bootstrap accoppiato su R@10. **Bilancio: nessun guadagno in
  Recall**; il valore e' nella diagnosi. Sintesi in
  `../docs/piano-interpretabilita.md`, sezione
  *"Che cosa fa davvero il contributo, e perche' non generalizza"*.
- **29/08 — il modulo e' limitato dai dati** (Filippo). Gli script su `max_sources`, il bagging
  e l'accordo fra modelli. Sintesi in [`notte/README.md`](notte/README.md).

Tutti si lanciano **dalla radice della repo**, con `HF_HUB_OFFLINE=1`:

```bash
HF_HUB_OFFLINE=1 python esperimenti/scomposizione.py --seed 2026 --split 0
```

Il benchmark ufficiale **non viene toccato da nessuno di questi script**. Nella repo lo
valutano solo `notebooks/11_baseline_e_stato.ipynb` (per la **baseline**, che e' il metro e non
spende budget) e `scripts/zeroshot_filter.py` (per un **metodo candidato**, il 25/08, con esito
negativo): vedi [`../scripts/README.md`](../scripts/README.md).

> **Nota del 30/08 sui percorsi.** Fino a oggi 16 di questi file avevano
> `ROOT = Path("/home/ari/Documents/DLproject/DL-Project-2026")` scritto a mano ed erano quindi
> **non eseguibili da chiunque altro**. Corretto in `ROOT = HERE.parent`. E' la stessa classe di
> bug gia' corretta il 28/08 nei 15 file di `esperimenti-scartati/` (vedi la nota in fondo a
> `../esperimenti-scartati/README.md`).

---

## 1. Perche' il metodo fallisce sugli attributi mai visti (28/08)

Ordinati per valore informativo, non alfabeticamente.

| file | cosa misura | risultato | esito |
|---|---|---|---|
| `scomposizione.py` | congela un ingresso alla volta dell'MLP (reference alla media / condizione alla media): il danno viene dal canale direzioni o dal canale reference? | `scomposizione_split{0,1,2}.json` | **la misura piu' importante della giornata**: il danno sui mai visti sta **tutto** nel canale direzioni (3/3), il guadagno sui visti sta nell'**interazione** fra i due canali |
| `heldout_knn.py` | due rimedi: gate `‖P_span c‖/‖c‖` senza iperparametri, e riscrittura della condizione nuova come combinazione dei k attributi visti piu' vicini | `knn_seed2026_split{0,1,2}.json` | **negativo**: media -0.36pp, segno discorde |
| `procruste.py` | vincola la mappa testo->visione a una **rotazione** ortogonale (Procrustes in forma chiusa) stimata sui soli visti | `procruste_seed2026_split{0,1,2}.json` | **negativo**: allineamento x6 sui visti, **peggiora** sui mai visti 3/3 |
| `mappa_dentro.py` | da' all'MLP **entrambe** le direzioni (grezza e mappata): `[r,c]`, `[r,c,Wc]`, `[r,Wc]` | `mappa_dentro_split{0,1,2}.json` | **nullo**: +0.26 / +0.17 / -0.40 sui mai visti. Alla rete e' stata offerta la direzione migliore e non l'ha usata |
| `mappa_retrieval.py` | applica la mappa lineare dei tre domini alle direzioni CelebA, griglia alpha x lambda | `mappa_tre_domini_split{0,1,2}.json` | **negativo**: nemmeno una casella positiva, fino a -15pp |
| `miscela.py` | esiste una dose intermedia fra direzione grezza e direzione mappata? `beta` da 0 a 1 | `miscela_split{0,1,2}.json` | **negativo**: curva monotona decrescente su 3/3 |
| `scala.py` | il disallineamento testo/visione e' solo il divario fra le due nuvole? centraggio + correlazioni canoniche + retrieval | `scala_split{0,1,2}.json` | **negativo**: la mappa memorizza i 30 visti, non riscala |
| `pesi_template.py` | pesa le condizioni con la concentrazione `R` fra i 10 template (senza etichette); piu' un oracolo dai pesi dell'AUC vera | `pesi_oracolo_split0.json` | **nullo**: correlazione R<->AUC solo +0.299, retrieval dentro il rumore. Tetto della pesatura per condizione **sotto 1pp** |
| `vocabolario_esteso.py` | 787 direzioni extra (dizionario + AwA2 + OVAD) a pseudo-etichette CLIP, come compito ausiliario con peso lambda | `vocabolario_esteso_split{0,1,2}.json` | **manopola, non guadagno**: +1.28pp sui mai visti, -1.05pp sui visti, netto -0.29pp. Unica cosa che riduce il danno in modo replicato 3/3 |
| `multidominio.py` | addestra lo stesso MLP anche su OVAD (compito ausiliario, peso lambda); salva i checkpoint `multidominio_seed*_lam*.pt` | `multidominio_seed{1,2,3,4,2026}_split0.json`, `multidominio_split{0,1,2}.json` | **nullo su CelebA**: +0.07pp su 5 seed, sd 0.554 |
| `ensemble_heldout.py` | ensemble variando **solo l'inizializzazione**, sugli split held-out | `ensemble_heldout_split{0,1,2}.json` | **quasi nullo**: +0.07 / +0.49 / +0.01pp. E' il controllo che ha imposto di dire "bagging", non "ensemble di seed" |
| `relazione_testo_visione.py` | esiste una mappa testo->visione che generalizza? lineare contro rete residua, 227 attributi da tre domini | `relazione_testo_visione.json` **+ `frecce_tre_domini.pt`** | **positivo sull'allineamento**: 0.086 -> 0.295 su 5 partizioni su 5, e la rete non batte la lineare. Ma non si traduce in Recall (vedi `mappa_retrieval.py`) |
| `filtro_ortogonale.py` | somiglianza alla reference **al netto** degli attributi interrogati (proiezione via QR), tre schemi | `filtro_ortogonale.json` | **nullo**: -0.02pp (p=0.87) allo schema migliore |
| `ablazione_r.py` | ablazione della dipendenza dalla reference sul dev set completo (reference in ingresso -> reference media) | **nessun file salvato** | la dipendenza dalla foto vale il **64%** del guadagno (+7.35 -> +2.66pp, p=0.005) |
| `ovad_eval.py` | valuta i modelli multi-dominio sul benchmark OVAD, su query mai viste in addestramento | **nessun file salvato** | il modulo e' **indipendente dal dominio** (+36pp su OVAD, nessun costo su CelebA); il modello mono-dominio **danneggia** OVAD (-7.5 / -8.8pp) |
| `audit_fuga.py` | la soglia 0.60 basta a impedire la fuga di supervisione dal dizionario di `vocabolario_esteso.py`? | **nessun file salvato** | il filtro cattura i casi netti e ne lascia passare di tenui: regime "stretto" = *quasi* pulito |

**Ordine di esecuzione**: `relazione_testo_visione.py` va lanciato **per primo**, perche'
produce `esperimenti/frecce_tre_domini.pt` (direzioni testuali e visive dei tre domini) che
`mappa_retrieval.py`, `miscela.py`, `mappa_dentro.py`, `multidominio.py`, `vocabolario_esteso.py`
e `ovad_eval.py` leggono. Il `.pt` non passa da git (`.gitignore`), si rigenera in ~5 minuti.

## 2. Il modulo e' limitato dai dati (29/08)

| file | cosa misura | risultato | esito |
|---|---|---|---|
| `braccio_b_piu_query.py` | piu' reference per query contro piu' query distinte, a parita' di budget di esempi | `braccio_b_seed1_20260829-0055.{txt,json}` | conta la **quantita' di esempi**, non quali query: +3.69 contro +3.26pp, differenza nulla (-0.43pp, p=0.48) |
| `curva_max_sources.py` | la curva su `max_sources` (80 / 175 / 350) piu' il controllo sui passi di gradiente | `curva_max_sources_seed5_20260829-0133.txt` (il `.json` manca: **il run e' stato interrotto**) | 26.40% -> 29.22% -> **32.26%** |
| `notte/notte.py` | le tre fasi della notte: curva a 60 epoche, repliche, bagging | `notte/notte_20260829-0227.{txt,json}` | 36.62% con 4 modelli mediati a `max_sources` 1400. Vedi [`notte/README.md`](notte/README.md) |
| `quante_query_in_piu.py` | quanto e' grande il bacino di reference per query, e quando `N_TRAIN_CORPUS` diventa il vincolo | `quante_query_in_piu_20260829-0034.{txt,json}` | il corpus non morde fino a `max_sources` ~700, e morde **prima sulle composte** |
| `accordo_canale_c.py` | l'accordo fra modelli fuori dallo span e' reale, o e' il canale della reference? (sottrae `net([r;0])`) | `accordo_canale_c_20260829-0109.{txt,json}` | **ribalta `ensemble_seed.py`**: l'accordo fuori span crolla da +0.7217 a **+0.0776**. Vedi la nota qui sotto |
| `ensemble_seed.py` | accordo fra 4 seed dentro/fuori lo span, e resa dell'ensemble sul dev set | **nessun file salvato** | il `+0.7217` prodotto qui e' **superato** da `accordo_canale_c.py`; la parte sull'ensemble regge (30.53%) |
| `quanto_puo_rendere_un_ensemble.py` | quanto potrebbe rendere un ensemble, dagli esiti per source gia' salvati in `contributo-candidato/multiseed/per_source_seed*.npz` (senza i pesi) | **nessun file salvato** | stima a priori del margine dell'ensemble |

> **Contraddizione risolta, da non riaprire.** `ensemble_seed.py` misurava l'accordo fra
> modelli su `net([r ; c])`, che per il 56.5% e' un termine che dipende **solo** dalla
> reference ed e' quindi condiviso da tutti i modelli. `accordo_canale_c.py` isola il canale
> delle condizioni sottraendo `net([r ; 0])` e trova +0.0776 invece di +0.7217. **Fuori dallo
> span i modelli non concordano.** La conclusione che ne discendeva in
> `docs/piano-interpretabilita.md` sez. 6e — *"spegnere quella componente e' smentito"* — non
> vale piu': l'idea (sez. 6b del piano) e' tornata in gioco.

## 3. Script senza output salvato

Sei script **stampano e basta**: rilanciarli e' l'unico modo di riavere i loro numeri, e i
numeri che li citano nei documenti vengono dal terminale di quella sessione.

```
ablazione_r.py        ensemble_seed.py      ovad_eval.py
audit_fuga.py         quanto_puo_rendere_un_ensemble.py
```

*(Rettifica: `mappa_retrieval.py` era stato incluso in questo elenco altrove, ma **salva**
`mappa_tre_domini_split{0,1,2}.json` alla riga 271. Verificabile aprendo il file.)*

Tre di questi producono numeri **citati nei documenti**: `ablazione_r.py` (il 64% del
guadagno dalla reference), `ovad_eval.py` (+36pp su OVAD) ed `ensemble_seed.py`
(30.53% e il 0.7217 poi ribaltato). Se quei numeri finiscono nel notebook di consegna, vanno
o rigenerati o citati come "misurati il 28-29/08, output solo a terminale".

## 4. JSON senza script produttore identificabile

| file | cosa contiene | chi lo ha prodotto |
|---|---|---|
| `curva_seed2026_split{0,1}_n{5,10,20,30}.json` | curva sul **numero di attributi visti** in addestramento (5/10/20/30), con controllo positivo, quota in-span e differenza per strato | **nessuno script in repo**. E' la fonte della tabella "il vocabolario non e' il collo di bottiglia" (`piano-interpretabilita.md` sez. 1, *sestuplicare il vocabolario vale 1.4pp*). La struttura e' quella dei fork qui sopra, con un parametro `n_seen` che nessun file espone |
| `pesi_template_split0.json` | identico a `pesi_oracolo_split0.json` nei campi letti (stesso `rho_R_AUC`, stesso vettore `R`) | quasi certamente una **versione precedente di `pesi_template.py`**, che scriveva con un altro nome. Oggi lo script produce `pesi_oracolo_split{N}.json` |
| `multidominio_split{0,1,2}.json` | sweep di lambda con `holdout: []` | **versione precedente di `multidominio.py`**: oggi il nome del file include il seed (`multidominio_seed{SEED}_split{N}.json`) |
| `ensemble_quanti.json` | — | **non esiste**: `ensemble_seed.py` moriva con `NameError` prima di scriverlo (`media_sing` usata prima di essere definita). Corretto il 30/08; il file comparira' al prossimo run |

**Regola per il notebook di consegna**: un numero che viene solo da
`curva_seed2026_*` non ha oggi codice che lo riproduca. Va citato come misura del 28/08 con
il JSON come unica evidenza, oppure rigenerato.

## 5. Una svista di percorso ancora aperta

`heldout_knn.py` e `scomposizione.py` scrivono in `OUT = HERE / "multiseed"`, cioe'
`esperimenti/multiseed/` — una cartella che **non esiste** in repo. I loro risultati versionati
(`knn_seed2026_split*.json`, `scomposizione_split*.json`) stanno in `esperimenti/`, quindi sono
stati spostati a mano dopo il run. `OUT` e' un residuo del fork: in
`contributo-candidato/attributi_heldout.py` puntava a `contributo-candidato/multiseed/`, che
esiste ed e' quella giusta. **Non e' stato modificato** per non cambiare il comportamento a
ridosso della consegna; chi rilancia quei due script trovera' l'output in
`esperimenti/multiseed/`.

## 6. La caratterizzazione della sessione v3 (30/08)

Quattro script scritti il 30/08 per rispondere a una domanda che il progetto non si era mai
posta:
**la parte non lineare del modulo che cosa fa, esattamente?** La risposta e' che gestisce i
vincoli **negativi**, e che una mappa lineare non puo' farlo per struttura.

| file | cosa misura | file di risultato | esito |
|---|---|---|---|
| `linearita_per_strato.py` | il divario fra ensemble e matrice collassata, stratificato per numero di condizioni e per numero di condizioni **negative**. Usa i soli esiti per source, quindi non serve nessun checkpoint | `linearita_per_strato.json` | il divario si concentra sui negativi: su query a una condizione, 27 positive contro 27 negative, la differenza fra i due divari e' +2.69pp (v1), +4.08pp (v2), +2.67pp (v3), tutte con p < 0.01 |
| `collasso_per_membro.py` | collassa in una matrice **ogni membro preso da solo**, col protocollo di fase 4 di `scripts/train_colab_v3.py`. Separa "il modello e' non lineare" da "l'ensemble non si riassume" | `collasso_per_membro_v{1,2,3}.json` | **24 membri su 24**, in tre sessioni indipendenti, perdono di piu' sui negativi. E il divario dell'ensemble (+1.96pp) e' pari a quello del singolo membro (+1.80pp): non e' un effetto del mediare |
| `antisimmetria_e_ablazione.py` | **A**: quanto l'MLP si discosta dall'antisimmetria che una matrice gli imporrebbe (`W_c(-c) = -W_c(c)` sempre). **B**: l'ablazione della reference di `ablazione_r.py`, stratificata per segno | `antisimmetria_e_ablazione_v3.json` | **A**: indice 0.2313 sui 12 membri v3 contro 0.0897 sul modello a `max_sources` 80, e 0 esatto per qualunque mappa lineare. **B**: il 58.6% del guadagno viene dalla foto (era 64% a ms 80), ma **83.5% sui negativi** contro 69.6% sui positivi |
| `generalizza_meccanismo.py` | il meccanismo asimmetrico sopravvive su attributi mai visti? Da' al modello congelato le 187 direzioni di AwA2 e OVAD (`frecce_tre_domini.pt`), che hanno la stessa costruzione delle 40 di CelebA | `generalizza_meccanismo_v3.json` | **no**: la risposta dimezza (norma 0.83 contro 1.71) e il comportamento torna quasi lineare (indice 0.105 contro 0.231, p = 9e-20, 12 membri su 12). E fra i mai visti la quota che cade dentro il sottospazio delle 40 viste predice entrambe le cose, rho +0.75 |

Il legame fra i due: sui 12 membri della v3 l'indice di antisimmetria **predice** quanto la
matrice fallisce sui negativi, Spearman +0.678 (p = 0.015).

> **Tre di questi script hanno bisogno di dati che non stanno in git.** I checkpoint (`*.pt`) e la
> cache `dati_pool.pt` delle sessioni Colab pesano centinaia di MB e sono esclusi da
> `.gitignore`; stanno su Drive. `linearita_per_strato.py` gira invece **cosi' com'e'** sulle
> sessioni gia' versionate (`run_colab/`, `run_colab_v2/`, `run_colab_v3/`), perche' gli esiti
> per source sono in repo:
>
> ```bash
> python esperimenti/linearita_per_strato.py                # v1, v2 e v3, dai file in repo
> python esperimenti/collasso_per_membro.py CACHE CKPT_DIR --tag v3
> python esperimenti/antisimmetria_e_ablazione.py CACHE CKPT_DIR --tag v3 --vecchio
> python esperimenti/generalizza_meccanismo.py CACHE CKPT_DIR --tag v3
> ```
>
> `generalizza_meccanismo.py` vuole in piu' `esperimenti/frecce_tre_domini.pt`, che **non e' in
> git** (`*.pt`): si rigenera con `relazione_testo_visione.py`, che a sua volta ha bisogno degli
> embedding di AwA2 e OVAD, anche loro fuori da git e condivisi su Drive.
>
> `CACHE` e' il `cache/dati_pool.pt` della sessione, `CKPT_DIR` la cartella dei suoi
> checkpoint. Verificato il 30/08 che `E_fit`, `L_fit` e `d_ens` sono **bit-identici** fra v2 e
> v3 (stesse costanti `SEED_FIT = 999`, `POOL_TARGETS`, `PER_QUERY_FIT` nei tre script), quindi
> una cache vale per l'altra sessione.
