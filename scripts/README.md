# scripts/

Tre gruppi di script che non c'entrano fra loro. Si lanciano tutti dalla radice della repo,
con `HF_HUB_OFFLINE=1` (l'hub di HuggingFace ha gia' fatto appendere due run per rate
limiting: i pesi di CLIP sono in cache locale e nessuna chiamata di rete serve).

## 1. La sessione di consolidamento su Colab — in uso

| file | cosa fa |
|---|---|
| `train_colab.py` | l'intera sessione del contributo candidato in sei fasi: dati, minimo garantito, ricerca a coordinate degli iperparametri, sei modelli finali con bagging, collasso lineare dell'ensemble, report. Pensato per **un solo lancio** su Colab: ogni fase salva su Drive e un rilancio riprende da dov'era |

Non tocca `celeba_evaluation.json`. Seleziona gli iperparametri **solo** sul set di
validazione onesta (query nuove, disgiunte dal pool di addestramento e dal set di sviluppo) e
stampa per trasparenza quale sarebbe stato il vincitore sul set di sviluppo, senza usarlo.
All'avvio verifica che la baseline valga 19.91% sul set di sviluppo e si ferma se non torna:
e' il collaudo di implementazione che il progetto usa dal 24/08.

```bash
python scripts/train_colab.py --prova     # ~1 minuto su CPU: fa esplodere i bug prima di Colab
python scripts/train_colab.py --ore 3     # il lancio vero; le sei righe di cella Colab sono
                                          # nella docstring in testa al file
```

Chiude due questioni rimaste aperte in [`../esperimenti/notte/README.md`](../esperimenti/notte/README.md):
i livelli di `max_sources` sono confrontati **al proprio punto d'arresto** e su viste
annidate, quindi il confronto e' accoppiato e ha un intervallo di confidenza; e i membri
dell'ensemble differiscono sia sulle query sia sulle reference, con la sovrapposizione fra i
loro campioni misurata e riportata invece che asserita.

## 2. Estrazione degli embedding — in uso

Attuano il passo deciso il 28/08 (sez. 4 di `../docs/STATO.md`): portare
il vocabolario da 40 a 242 attributi **non per addestrare ma per misurare** la relazione fra
direzioni testuali e visive.

| file | cosa fa |
|---|---|
| `estrai_awa2.py` | AwA2: 37.322 immagini, 85 attributi. Scarica (~13 GB), riprende da checkpoint |
| `estrai_ovad.py` | OVAD: ~8.800 oggetti ritagliati, 117 attributi. Scarica annotazioni + COCO val2017 (~816 MB) |
| `lancia_awa2.sh` | lancia l'estrazione sulla VM staccata dal terminale, e spegne la VM a fine lavoro |
| `vm.sh` | scorciatoie per la VM: `stato`, `aspetta`, `copia`, `entra`, `log` |

**Gli embedding prodotti non passano da git** (`*.pt` e' in `.gitignore`; AwA2 pesa 78 MB e
OVAD ~18): si condividono su Drive.

Due avvertenze sui dati, per chi li usa e non ha scritto gli script:

- **AwA2 etichetta per CLASSE**, non per immagine: le direzioni visive sono differenze fra
  centroidi di 50 classi e vivono in al piu' 49 dimensioni, non 512.
- **OVAD ha etichette a tre valori**: 1 positivo, 0 negativo, **-1 sconosciuto (16%)**. Va
  sempre mascherato con `labels != -1`; trattarlo come CelebA, dove `0` significa "non ha
  l'attributo", butterebbe un sesto dei dati nella classe sbagliata. L'unita' inoltre e'
  l'**oggetto ritagliato**, non l'immagine intera.

Su una macchina senza GPU usare `--workers 0`: con `forkserver` (predefinito in Python 3.14
su Linux) e poca RAM libera i worker del DataLoader non partono. Su CPU l'estrazione di AwA2
richiede ~22 minuti.

## 3. Prima misura dei tetti oracolari (25/08) — superata

Tre script con cui erano stati misurati per la prima volta i tetti di un approccio "a
vincoli". La versione completa e definitiva di quel lavoro e' in
`../esperimenti-scartati/01-filtro-per-attributi/`,
che aggiunge la sonda supervisionata, l'ablazione sul contributo dell'encoder testuale e la
conclusione: la pista **funziona ma cambia il problema**, quindi e' scartata.

Restano qui perche' il tetto del 42.52% — il numero piu' importante che abbiamo — nasce da
`oracle_ceilings.py`, ed e' citato in `../docs/prompt-implementazione.md`.

| file | cosa misura | esito |
|---|---|---|
| `oracle_ceilings.py` | tetti con le label: filtro sui negativi (+4.30pp), sui positivi (+12.29pp), su tutte le condizioni con ranking di sola identita' (**+22.60pp**) | valido, ripreso in `esperimenti-scartati/01` |
| `zeroshot_filter.py` | filtro duro a quantile globale sul percentile di corpus. **Valuta anche `celeba_evaluation.json` (righe 143-160): e' l'unico posto della repo in cui il benchmark ufficiale e' stato speso per un METODO CANDIDATO** | **negativo** sul set di sviluppo (-3pp a ogni quantile) **e sul benchmark ufficiale** (-2.92pp, vedi sotto) |
| `prevalence_filter.py` | filtro a soglie calibrate sulla prevalenza (dal train) + penalita' soft | +1.02pp p=0.019, poi assorbito e superato dalla sonda di `esperimenti-scartati/01` |

---

# Il benchmark ufficiale E' GIA' STATO SPESO UNA VOLTA (25/08) — registrato il 30/08

**Sette documenti della repo affermavano "il benchmark ufficiale non e' stato toccato".
E' falso, e va corretto ovunque.** `zeroshot_filter.py` alle righe 143-160 carica
`celeba_evaluation.json`, deduplica le 13 query distinte, e confronta il filtro zero-shot a
`q*` scelto sul set di sviluppo contro la baseline, con `bootstrap_paired`. **E' stato eseguito
il 25/08.**

I numeri di quel run esistevano **solo nel terminale di quella sessione**. Registrati qui:

| benchmark ufficiale, 13 query distinte, `q* = 0.5` | R@1 | R@5 | R@10 |
|---|---|---|---|
| filtro zero-shot + ranking di identita' | 2.80% | 9.05% | **13.00%** |
| baseline (`r + c`, prompt ensembling) sulle stesse 13 query | 3.90% | 10.99% | **15.92%** |

```
differenza accoppiata a R@10:  -2.92pp   IC [-5.59, +0.07]   p = 0.055
```

**E' un risultato negativo**: il filtro zero-shot **peggiora** anche sul benchmark ufficiale, in
modo coerente con il -3pp misurato sul set di sviluppo. Il 15.92% della baseline coincide con
quello riportato nella sez. 1 di `../docs/STATO.md`, che e' un controllo di
coerenza.

**Come va detta la cosa d'ora in poi.** Il budget del benchmark e' **per metodo**, non per
progetto: si spende una volta per ciascuna configurazione congelata, e riportare l'esito
qualunque esso sia fa parte del protocollo. Quindi la formulazione corretta e':

> *Il budget del benchmark ufficiale e' per metodo. E' stato speso una volta, il 25/08, per il
> filtro zero-shot di `scripts/zeroshot_filter.py`, con esito negativo (-2.92pp a R@10,
> p=0.055). Il contributo candidato non e' mai stato valutato li'.*

Non: *"il benchmark ufficiale non e' stato toccato"*. Quella frase non e' vera dal 25/08.

**Chi altro legge `celeba_evaluation.json`, e perche' non conta come spesa.**
`notebooks/11_baseline_e_stato.ipynb` lo valuta — ed e' da li' che vengono il 15.92% della
baseline, il 7.62% del solo-reference e il +2.41pp del prompt ensembling. Ma valutare la
**baseline** sul benchmark non spende il budget di un metodo candidato: stabilisce il metro.
`esperimenti-scartati/05-.../passo2_replica_tre_condizioni.py` lo apre soltanto per **escludere**
le sue query da un campionamento, senza valutarci nulla. Lo script `scripts/train_colab.py`
(non versionato al 30/08) dichiara esplicitamente di non leggerlo.

*(Per rigenerare i numeri: `HF_HUB_OFFLINE=1 python scripts/zeroshot_filter.py` — lo script
rifa' l'intero sweep sul dev e poi la valutazione ufficiale. Attenzione: rieseguirlo non
"spende" nuovo budget, perche' il metodo e la scelta di `q*` sono gli stessi; sceglierne un
altro sulla base di questo esito, si'.)*
