# Contributo candidato — testa di proiezione non lineare appresa

**Primo metodo del progetto che batte la baseline in modo netto e verificabile: 19,91% →
27,26% di Recall@10 (+7,35pp, IC 95% [+5,14, +9,74], p < 0,0001)** sul set di sviluppo a
101 query.

> **Aggiornamento del 27/08 — replicato su cinque seed.** Il risultato regge: **+6,95pp di
> media** (sd 0,76), intervallo [+6,20, +8,07], tutti e cinque significativi a p<0,0001, e
> reggono anche tutte le affermazioni qualitative qui sotto. Il numero da riportare non è più
> il +7,35pp di questo run — che era leggermente sopra la media — ma la media con il suo
> intervallo. Si è anche scoperto che il metodo **si riduce a due matrici e un bias** senza
> perdere quasi nulla (93% del guadagno), e che la mappa lineare dell'esperimento 4 va
> stimata con una ridge, perché il sistema è mal posto e `lstsq` non è riproducibile.
> Tutto in `../docs/report-27-08-2026-multiseed.md`;
> per rieseguire: `contributo-candidato/multiseed.py`.

~~Il benchmark ufficiale **non è stato toccato**.~~

> **Correzione del 30/08.** La frase sopra è falsa dal 25/08 e va sostituita ovunque compaia.
> Il budget del benchmark è **per metodo**: è stato speso **una volta**, il 25/08, per il
> filtro zero-shot di `scripts/zeroshot_filter.py` (righe 143-160), con **esito negativo** —
> R@10 13,00% contro 15,92% della baseline sulle stesse 13 query, differenza accoppiata
> **−2,92pp**, IC [−5,59, +0,07], p=0,055. Numeri e dettagli in
> [`../scripts/README.md`](../scripts/README.md).
>
> **Quello che resta vero, ed è ciò che conta qui:** *il contributo candidato non è mai stato
> valutato sul benchmark ufficiale.* Tutti i numeri di questa pagina vengono dal set di
> sviluppo a 101 query.

---

## Il metodo, in una riga di differenza

```
baseline:   q = normalize( r  +  c )                    c = somma delle direzioni testuali,
                                                            con segno: +d per T⁺, −d per T⁻
metodo A:   q = normalize( r  +  γ · MLP([r ; c]) )     MLP 1024 → 512 → 512
```

Invece di sommare *direttamente* la direzione testuale, un MLP legge la coppia
**(volto di riferimento, condizioni richieste)** e produce un vettore di spostamento a 512
dimensioni. `γ` è uno scalare appreso; l'ultimo strato dell'MLP è inizializzato a zero,
quindi all'avvio `q = r` esattamente e il modello parte dall'identità pura.

CLIP resta **completamente congelato**, entrambi gli encoder. Il database di embedding
visivi resta intatto e la ricerca resta un prodotto scalare: il costo di inferenza è
identico a quello della baseline.

Rientra nella voce *"non-linear projection heads"* che l'assignment elenca fra le quattro
strade suggerite (sez. 3, punto 2), e che nel progetto **non era mai stata implementata**:
`src/phi_dynamic.py` ha un MLP, ma produce solo **scalari** (pesi per condizione e
lunghezza del passo), quindi la query restava vincolata dentro `span(r, u₁…uₙ)`. Qui l'MLP
produce un **vettore**, che può puntare fuori da quello span. È questa la differenza che
rompe il tetto della famiglia "raffina direzione e passo", chiusa dagli oracoli in
`docs/STATO.md`.

---

## Risultati

Set di sviluppo `celeba_devset_valid.json`, 101 query, confronto accoppiato con bootstrap
gerarchico (`src/devset.bootstrap_paired`, 20.000 ricampionamenti).

| metodo | R@1 | R@5 | R@10 | diff su R@10 | IC 95% | p |
|---|---|---|---|---|---|---|
| baseline (`r + c`) | 4,22% | 13,34% | 19,91% | — | — | — |
| **A: concatenazione** | **5,71%** | **18,22%** | **27,26%** | **+7,35pp** | [+5,14, +9,74] | <0,0001 |
| B: cross-attention | 5,04% | 16,55% | 24,65% | +4,73pp | [+2,78, +6,71] | <0,0001 |

Per confronto, il resto del progetto: prompt ensembling +2,41pp (ma è baseline, non
contributo), filtro con sonda supervisionata +4,87pp (scartato, vedi
`../esperimenti-scartati/01-filtro-per-attributi/`),
tutto il resto ≤ +1pp o negativo.

### Il guadagno è su combinazioni MAI VISTE

Verifica eseguita subito dopo il risultato, perché senza di essa il numero non è
interpretabile: quante query del set di sviluppo il modello aveva già incontrato?

```
query del dev già viste in addestramento: 55/101
   a 1 condizione:  54/54 viste     ← inevitabile: con 40 attributi × 2 segni ne
                                       esistono solo 80, e l'addestramento le usa tutte
   a 2 condizioni:   1/32 viste
   a 3 condizioni:   0/15 viste
```

Quindi **46 delle 47 query composte sono combinazioni che il modello non ha mai visto**.
E il guadagno è più grande proprio lì:

| sottoinsieme | A vs baseline | IC 95% | p |
|---|---|---|---|
| semplici (54, tutte viste) | +6,11pp | [+3,79, +8,58] | <0,0001 |
| **composte (47, 46 mai viste)** | **+8,77pp** | [+4,97, +13,21] | <0,0001 |

Non è un lookup su query note: il modello ha imparato una regola di fusione che si
trasferisce a **combinazioni di attributi nuove**. È la composizionalità che l'assignment
chiede.

---

## Perché funziona, quando tutto il resto ha fallito

**1. Il vettore appreso conosce il criterio di valutazione; la direzione testuale no.**
Probabilmente la ragione principale. I positivi dell'addestramento sono per costruzione le
immagini che soddisfano le condizioni **e** stanno entro Hamming ≤ 2 dalla reference — cioè
la regola dell'assignment, letteralmente (il ground truth è generato da
`src/benchmark.build_benchmark_from_labels`, lo stesso codice che replica il protocollo
ufficiale). Il modello impara quindi a spostarsi in modi che soddisfano la condizione
**senza rompere l'identità**. La direzione testuale `t⁺ − t⁻` non sa nemmeno che esista un
vincolo di identità: sa solo di semantica.

L'ablazione in
`../esperimenti-scartati/01-filtro-per-attributi/`
aveva mostrato che la spinta testuale fa due lavori insieme e che il secondo (allontanare
dall'identità) è puro costo. Qui quel costo viene **ottimizzato via**, invece che subìto.

**2. Può riparare le direzioni rotte.** Sette attributi su 40 hanno la direzione testuale
invertita (AUC < 0,5). La baseline le usa così come sono; l'MLP può imparare a compensarle.

**3. Lo spostamento dipende dal volto.** `delta = MLP([r ; c])` è funzione della reference:
"aggiungi occhiali" viene eseguito diversamente a seconda della faccia. Gli oracoli onesti
di `STATO.md` avevano chiuso il **passo** per reference (margine nullo) e i **pesi** per
reference (+0,1pp, n.s.) — ma una **direzione** per reference è una cosa molto più ricca, e
non era mai stata misurata.

---

## Perché la cross-attention (B) perde

B usa la reference come *query* e le condizioni come *key/value*, con la stessa capacità a
valle di A. Perde, e **perde proprio dove doveva vincere**:

| confronto | sottoinsieme | diff | p |
|---|---|---|---|
| B vs A | semplici | −1,08pp | 0,095 |
| B vs A | **composte** | **−4,37pp** | **0,0001** |

Spiegazione: i pesi di attenzione sono un softmax, quindi **sommano a 1 e mediano** le
condizioni; la baseline e A le **sommano**. Con tre condizioni l'attenzione ne applica
circa un terzo ciascuna, cioè sottoapplica sistematicamente — e l'effetto peggiora al
crescere del numero di condizioni.

Non è che il meccanismo non funzioni: i pesi non sono affatto uniformi (es.
`[0,42 0,088 0,491]` su `+Wavy_Hair, −Black_Hair, −Heavy_Makeup`, contro un uniforme di
0,333). È che *mediare* è l'operazione sbagliata per questo compito.

**Risultato riportabile di per sé**: la cross-attention è una delle quattro strade che
l'assignment suggerisce; poterla escludere con un confronto controllato a parità di
capacità, e con una spiegazione strutturale, vale più che non averla provata.

---

## Come è stato addestrato

- **Dati**: 220 query campionate dal train split con `src/devset.sample_queries` (80 a una
  condizione — tutte quelle che esistono —, 80 a due, 60 a tre), ground truth generato con
  `src/benchmark.build_benchmark_from_labels`, cioè la **regola dell'assignment**. 182 query
  valide, 4.407 coppie (reference, target).
- **Corpus**: 20.000 immagini del train split. `valid` e `test` non entrano mai
  nell'addestramento.
- **Loss**: InfoNCE **multi-positivo** — il numeratore somma su *tutti* i ground truth della
  coppia, la reference è esclusa dal denominatore come in valutazione.
- **Temperatura**: scelta su una griglia `{0,03 0,05 0,10 0,20}` valutata su query di
  addestramento **tenute fuori dal fit**. Il set di sviluppo non entra nella selezione,
  quindi non consuma budget statistico. Scelta: 0,03 per entrambi i modelli.
- **Addestramento finale**: 30 epoche, batch 256, AdamW lr 3e-4, weight decay 1e-4.

### Il controllo positivo

Prima di leggere qualunque numero sul set di sviluppo, i modelli devono battere la baseline
**sulle query di addestramento**. Se non imparano nemmeno lì, il resto non si interpreta.

```
baseline       R@10 (train) = 14,71%
A concat       R@10 (train) = 70,11%
B cross-attn   R@10 (train) = 66,83%
```

Superato. Da notare il divario fra 70,11% in-sample e 27,26% sul dev: c'è memorizzazione
consistente **accanto** a una generalizzazione reale, non al posto suo — lo dimostra il
+8,77pp sulle combinazioni mai viste.

---

## Il primo tentativo, non valido — `cross_attention.py`

Incluso perché fa parte del percorso e perché la lezione è riutilizzabile. Quel run è
arrivato in fondo ma **il suo risultato non è interpretabile**: il controllo positivo era
fallito (13,44% e 12,02% contro 16,31% della baseline, *in addestramento*). Il −7,54pp che
produceva sul dev set misurava un setup rotto, non la cross-attention.

Diagnosi, leggibile nel numero della loss: con denominatore su 20.000 candidati e un solo
positivo, la loss di un modello casuale vale `ln(20000) = 9,90`; quella finale era **9,60**.
Il modello scendeva fino al livello del caso, cioè imparava a rendere la propria uscita
poco informativa. Due cause:

1. **temperatura**: `tau = 0,01` era stata presa da `src/phi_dynamic.py`, dove però il
   denominatore era un piccolo insieme di distrattori estratti col mining;
2. **falsi negativi**: un solo positivo contro 20.000 "negativi", molti dei quali erano
   **altri ground truth validi della stessa query**. Il modello veniva punito per averli
   trovati. `src/phi_dynamic.py` implementa già una InfoNCE multi-positivo: era stata
   ignorata.

**Lezione operativa**: la loss va sempre stampata *accanto al suo livello di caso*. In
`cross_attention_v2.py` quel livello è calcolato per ogni batch (`ln(N/|GT|)`, non `ln(N)`,
perché la loss è multi-positivo) e stampato in colonna a ogni epoca, insieme alla Recall@10
sui dati di fit. Un fallimento come quello sarebbe visibile in tre epoche invece che alla
fine.

---

## Cosa manca

1. ~~**Replica multi-seed.**~~ — fatta il 27/08 su cinque seed, regge: vedi il riquadro in
   testa a questo file e `../docs/report-27-08-2026-multiseed.md`.
2. **La temperatura è al bordo della griglia** (0,03, l'estremo più basso provato).
   Estenderla verso il basso è legittimo — si sceglie su query di addestramento, il set di
   sviluppo resta intatto. La replica multi-seed ha tenuto 0,03 fisso: valida la
   configurazione, non la riseleziona.
3. ~~**Generalizzazione ad attributi mai visti: non dimostrata.**~~ — **MISURATA il 27/08, ed
   è NEGATIVA. Questa pagina è rimasta indietro fino al 30/08: chi la leggeva non scopriva il
   limite principale del metodo.** Il testo originale diceva: *"Il metodo generalizza a
   combinazioni nuove, ma tutti e 40 gli attributi compaiono in addestramento. Sono due assi
   diversi e solo il primo è verificato."* Il secondo asse è ora verificato, e il metodo lo
   fallisce.

   Togliendo 10 attributi dall'addestramento e valutando sulle query di sviluppo che toccano
   **solo** quelli, il metodo **fa peggio della baseline**: **−4,88 / −4,51 / −7,73pp** su tre
   split, tutti significativi (IC in
   `../docs/report-27-08-2026-multiseed.md`, sez. 4;
   codice: `attributi_heldout.py`). Non è capacità né dati — il controllo positivo passa e lo
   strato "visti" rende +5/+7,5pp — e gli attributi tolti non sono più difficili.

   **Il meccanismo è misurato**: le condizioni viste spannano un sottospazio di rango 30 (le
   30 direzioni rimaste), e una condizione nuova ci cade dentro per il 64–70% della sua norma.
   Il modello la tratta quindi come una **miscela degli attributi che conosce**: non la ignora,
   la interpreta male. Un rimedio senza iperparametri (scalare il delta per la quota in-span)
   riduce il danno ma non lo toglie. Altre due famiglie di rimedio — rotazione di Procuste e
   interpolazione sui vicini semantici — sono state provate il 28/08 in `esperimenti/` e sono
   anch'esse negative (vedi [`../esperimenti/README.md`](../esperimenti/README.md)).

   **Conseguenza, da dichiarare nel report**: il metodo è a **vocabolario chiuso**. Non è "una
   trasformazione appresa dello spazio CLIP" ma "una correzione appresa delle 40 direzioni di
   attributo, componibile fra condizioni". Non intacca il +6,95pp — nel setup reale tutti e 40
   gli attributi sono in addestramento e il benchmark ufficiale ne usa 12 — ma cambia ciò che
   si può affermare. E va affrontato di petto il confronto con la pista scartata del filtro per
   attributi, che era stata chiusa **anch'essa** perché a vocabolario chiuso: la distinzione
   che sopravvive (qui il testo resta l'ingresso e il metodo compone condizioni mai viste
   insieme) va argomentata, non data per scontata.

   *(La pista sul vocabolario aperto era già stata tentata e chiusa per un'altra via: vedi
   `../esperimenti-scartati/02-mappa-bilineare/`.)*
4. **Benchmark ufficiale: mai speso per questo metodo.** Da spendere una volta sola, e solo
   quando la configurazione è definitiva. **Nota del 30/08**: il budget è *per metodo*, e per
   un altro metodo è già stato speso una volta — `scripts/zeroshot_filter.py` ha valutato
   `celeba_evaluation.json` il 25/08, con esito negativo. Vedi il riquadro in testa a questo
   file e [`../scripts/README.md`](../scripts/README.md).

---

## Come rieseguire

Serve `torch`, `transformers`, `numpy` e gli embedding in `embeddings/`.

```bash
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 python contributo-candidato/cross_attention_v2.py
```

`HF_HUB_OFFLINE=1` non è un dettaglio: l'hub di HuggingFace ha fatto **appendere due run**
per rate limiting sulle richieste anonime (`CLIPTokenizer.from_pretrained` resta in attesa
di rete a tempo indefinito). I pesi sono già in cache locale, quindi nessuna chiamata è
necessaria. Conviene usarlo per **tutti** gli script del progetto.

Tempo indicativo su CPU: ~40 minuti (selezione della temperatura, due addestramenti da 30
epoche, valutazioni).
