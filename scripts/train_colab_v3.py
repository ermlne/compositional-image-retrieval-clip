"""SESSIONE DI CONSOLIDAMENTO SU COLAB — v2 — un solo lancio, sei fasi, tutto ripartibile.

===========================================================================================
COSA FA, IN UNA RIGA
===========================================================================================
Addestra il contributo candidato — `q = normalize(r + gamma * MLP([r; c]))` — con arresto
onesto su un set di validazione mai usato altrove, ne sceglie gli iperparametri con una
ricerca a coordinate, ne fa il bagging su 6 campioni diversi di query, e verifica che
l'ensemble si collassi in una sola mappa lineare.

===========================================================================================
COSA E' QUESTA v3, E PERCHE' NON C'E' PIU' UNA RICERCA
===========================================================================================
E' la sessione di CONGELAMENTO. Non esplora: ripara lo strumento di selezione, sceglie fra
due punti gia' misurati, e spende tutto il resto sulla sola leva che ha pagato in modo
replicato in tutte e tre le sessioni precedenti — la diversita' dei membri.

PERCHE' NIENTE RICERCA. In v1 la ricerca a coordinate aveva scelto bene (tau 0.01). In v2 ha
scelto 1.5pp PEGGIO del meglio che aveva gia' provato, perche' leggeva uno strumento rotto:
la correlazione di Spearman fra validazione e set di sviluppo era crollata da +0.897 a
+0.244, e su tutte le 40 run era -0.013. Con quello strumento, allargare le griglie significa
solo camminare piu' in fretta nella direzione sbagliata.

LE DUE RIPARAZIONI (vedi docs/piano-v3-ultimo-training.md)

R1  LA VALIDAZIONE VIVE SU UNA GALLERIA DISGIUNTA. Nella v2 era valutata sulle STESSE 20000
    immagini che stanno nel denominatore dell'InfoNCE: le query erano nuove, la galleria no.
    Una validazione che condivide la galleria con l'addestramento NON PUO' rilevare il
    sovradattamento alla galleria — ed e' esattamente quello in cui la v2 e' scivolata, perche'
    tau 0.003 e max_sources 2800 sono entrambi modi di sfruttarla piu' a fondo. Qui la
    permutazione del train si taglia in due meta' disgiunte per costruzione: la prima e' la
    galleria di addestramento, la seconda quella di validazione. Vale anche per ogni membro.

R2  LA VALIDAZIONE HA LA COMPOSIZIONE DEL BENCHMARK. Il benchmark ufficiale e' 54/31/15 per
    numero di condizioni; la v2 usava 0/50/50, cioe' misurava solo lo strato che beneficia di
    piu' dei dati, ed e' il motivo per cui sovrastimava i livelli alti di max_sources. Qui si
    punta a 80/46/23 e si riempie STRATO PER STRATO, perche' le rese sono molto diverse (100%
    a una condizione, 83% a due, 52% a tre) e campionare a giri misti fa derivare la
    composizione verso lo strato piu' fertile. Le query a UNA condizione non si possono
    escludere dal pool — lo spazio ha 80 elementi e il pool li usa tutti — e non e' un
    compromesso: nel set di sviluppo tutte e 54 le query a una condizione compaiono in
    addestramento, quindi escluderle misurerebbe una popolazione diversa da quella su cui si
    riporta. Cio' che deve essere inedito e' la GALLERIA, e ci pensa R1.

IL COLLAUDO DELLA RIPARAZIONE, con la regola dichiarata PRIMA di guardare i numeri: la
validazione riparata e il set di sviluppo devono scegliere lo STESSO candidato. E' un test
molto piu' interpretabile di una correlazione su pochi punti, perche' e' esattamente la
decisione che lo strumento deve prendere. Se sono d'accordo si adotta il vincitore della
validazione; se NON lo sono, la riparazione non ha funzionato e si consegna la configurazione
della v2, che e' comunque gia' misurata con intervalli. In nessun caso si sceglie guardando
il dev.

LE FASI
  F0  setup, cache, direzioni, COLLAUDO della baseline (19.91%) e dell'equivalenza numerica
  F1  dati: pool con margine, DUE gallerie disgiunte, validazione stratificata
  FM  minimo garantito: 4 membri al candidato A, ensemble, matrice, tabelle
  F2  i due candidati (A del punto v2, B il migliore sul dev della sua ricerca) + COLLAUDO
  F3  12 membri al punto adottato, curva dei rendimenti per k = 1..12
  F4  collasso lineare in due matrici e un bias
  F5  report

I 12 membri non sono generosita': in v2 il SESTO rendeva ancora +0.68pp (p=0.052), la curva
non era satura, e un addestramento a max_sources 2800 costa 1.7 minuti su T4 — l'intera
sessione v2 e' stata 40 addestramenti in 0.89 ore. La curva fino a k=12 dice poi quanti
membri consegnare, come misura invece che come abitudine.

===========================================================================================
QUESTO SCRIPT NON LEGGE MAI IL BENCHMARK UFFICIALE
===========================================================================================
Il budget del benchmark ufficiale e' PER METODO, e non e' intatto: e' stato speso una volta,
il 25/08, da `scripts/zeroshot_filter.py` per il filtro zero-shot, con esito negativo (R@10
13.00% contro 15.92% della baseline, differenza accoppiata -2.92pp, IC [-5.59, +0.07],
p = 0.055; i numeri sono in `scripts/README.md`). L'affermazione corretta riguarda quindi
questo file, non il progetto.

`celeba_evaluation.json` NON viene letto, NON viene valutato, NON compare in questo script
se non in questa frase. La valutazione del contributo sul benchmark si fa dopo, a mano, a
configurazione congelata. Ogni numero prodotto qui sta sul set di sviluppo a 101 query
(`celeba_devset_valid.json`, valutato sugli embedding del validation split) oppure sul set
di validazione onesta costruito sul corpus di addestramento.

E nessun iperparametro viene scelto sul set di sviluppo: la selezione guarda SOLO la
validazione onesta. Per trasparenza lo script stampa anche quale sarebbe stato il vincitore
sul set di sviluppo, senza usarlo — e' la stessa disciplina di `esperimenti/notte/notte.py`.

===========================================================================================
CELLA COLAB (sei righe, da incollare in una cella e lanciare)
===========================================================================================
    from google.colab import drive; drive.mount('/content/drive')
    %env HF_HOME=/content/drive/MyDrive/DL-Project-2026/hf
    !GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ermlne/compositional-image-retrieval-clip.git /content/repo
    !cd /content/repo && git lfs pull -I embeddings/celeba_train_clip_vitb32.pt
    !pip -q install -U transformers
    !cd /content/repo && python scripts/train_colab.py --ore 3

Riga per riga, perche' e' scritta cosi':
  - `GIT_LFS_SKIP_SMUDGE=1` sul clone: la banda LFS gratuita di GitHub e' 1 GB/mese e il
    solo file `embeddings/celeba_train_clip_vitb32.pt` pesa 385 MB. Senza quella variabile
    ogni clone (anche di prova) se ne mangia un terzo. `valid` e `test` sono blob normali,
    non LFS, quindi arrivano gia' con il clone.
  - `git lfs pull` una volta sola: alla prima esecuzione lo script salva su Drive tutto cio'
    che deriva dal train (corpus, pool, direzioni testuali, corpus per la ridge), quindi a
    una ripresa il file da 385 MB NON serve piu' e si puo' saltare la riga 4.
  - `HF_HOME` su Drive: i pesi di CLIP si scaricano una volta e restano.
  - `--ore` e' il budget: le fasi che non ci stanno vengono saltate, non troncate.

===========================================================================================
LE FASI
===========================================================================================
F0  setup, cache su Drive, direzioni testuali, COLLAUDO della baseline (19.91% sul devset)
F1  dati costruiti UNA volta sola, CON MARGINE sopra la griglia, con i ranghi che danno sia
    le viste annidate (ricerca) sia le finestre sfalsate dei membri (bagging)
FM  MINIMO GARANTITO: 4 membri alla configurazione incumbente, ensemble, matrice, tabelle
F2  ricerca a coordinate (tau -> max_sources -> weight decay), scelta sulla validazione
F3  6 modelli finali al punto vincente, curva dei rendimenti dell'ensemble per k = 1..6
F4  collasso lineare dell'ensemble in due matrici e un bias (ridge, float64, lambda=1e-3)
F5  report: risultati.json, tabelle markdown, esiti per source in .npz

FM esiste per una ragione sola: con un unico lancio disponibile, entro i primi ~40 minuti
deve gia' esistere una consegna completa. Tutto quello che viene dopo la migliora, non la
crea. Se Colab stacca la sessione a meta' della ricerca, su Drive c'e' comunque un ensemble
misurato con tanto di intervalli di confidenza.

===========================================================================================
COSA E' RIUSATO E COSA E' NUOVO
===========================================================================================
Riusati senza modifiche: `src/benchmark.py` (load_benchmark, parse_query_string,
build_benchmark_from_labels come oracolo di riferimento), `src/devset.py` (sample_queries,
macro, bootstrap_paired, evaluate_per_source come oracolo di riferimento), il modello
`Concat` e la media delle uscite di `esperimenti/notte/notte.py`, la `ridge()` in float64 e
`build_lineare()` di `contributo-candidato/bagging_lineare.py`, il blocco TEMPLATES/encode
di `contributo-candidato/multiseed.py`.

Nuovo, e solo perche' senza non si arriva in fondo su 2 vCPU / una GPU:
  1. la distanza di Hamming come prodotto matriciale (vedi `hamming_gemm`), con controllo
     di equivalenza ESATTA contro il percorso CPU e fallback automatico se fallisce;
  2. la valutazione vettorizzata (`valuta`), con controllo di equivalenza contro
     `evaluate_per_source` di `src/devset.py`. Il valore di ritorno e' identico,
     `{k: [np.ndarray[n_src], ...]}`, perche' `macro` e `bootstrap_paired` lo consumano
     cosi' — quelle tre funzioni sono il protocollo del progetto e non si toccano.
Nessuna variante architetturale, nessuna augmentation, nessuna idea nuova: e' una sessione
di consolidamento.

===========================================================================================
USO
===========================================================================================
    python scripts/train_colab.py --prova       # ~4 minuti su CPU, fa esplodere i bug
    python scripts/train_colab.py --ore 3       # il lancio vero

    # collaudo del REGIME (~12 minuti di CPU): un modello a max_sources 80 con 30 epoche e
    # corpus intero deve riportare un dev intorno al 26%, il numero noto di
    # esperimenti/notte/README.md per quel livello. E' la verifica che il percorso di
    # addestramento riscritto per la GPU non abbia cambiato il metodo.
    python scripts/train_colab.py --prova --epoche 30 --corpus 20000 --out /tmp/regime

Se `--prova` fallisce, NON lanciare su Colab: la modalita' esiste apposta per far esplodere
i bug prima, e con un solo lancio disponibile un'eccezione a meta' sessione costa la consegna.
"""
import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ===========================================================================================
# ROOT DELLA REPO
# ===========================================================================================
# `__file__` non esiste quando il contenuto di questo file viene incollato in una cella di
# notebook, ed e' esattamente cosi' che il gruppo lo usera' se qualcosa va storto. Quindi la
# root si cerca, non si deduce, e la ricerca si ferma sul primo candidato che contiene i due
# file senza i quali non si va da nessuna parte.
def _trova_root():
    candidati = []
    if os.environ.get("DL_ROOT"):
        candidati.append(Path(os.environ["DL_ROOT"]))
    try:
        candidati.append(Path(__file__).resolve().parent.parent)
    except NameError:
        pass
    candidati += [Path.cwd(), Path.cwd().parent, Path("/content/repo"),
                  Path("/content/DL-Project-2026"),
                  Path("/content/drive/MyDrive/DL-Project-2026")]
    for c in candidati:
        try:
            if (c / "src" / "benchmark.py").exists() and (c / "celeba_devset_valid.json").exists():
                return c.resolve()
        except OSError:
            continue
    raise SystemExit(
        "Non trovo la root della repo (mi servono src/benchmark.py e celeba_devset_valid.json).\n"
        "Passa la variabile d'ambiente DL_ROOT, oppure lancia lo script da dentro la repo.")


ROOT = _trova_root()
sys.path.insert(0, str(ROOT))
from src.benchmark import (build_benchmark_from_labels, load_benchmark,  # noqa: E402
                           parse_query_string)
from src.devset import (bootstrap_paired, evaluate_per_source, macro,  # noqa: E402
                        sample_queries)

# ===========================================================================================
# ARGOMENTI
# ===========================================================================================
ap = argparse.ArgumentParser()
ap.add_argument("--ore", type=float, default=5.0,
                help="budget di tempo; le fasi che non ci stanno vengono saltate")
ap.add_argument("--prova", action="store_true",
                help="pipeline intera in ~5 minuti su CPU, per far esplodere i bug")
ap.add_argument("--out", type=str, default=None, help="cartella di lavoro (default: Drive)")
ap.add_argument("--threads", type=int, default=0, help="0 = tutti i core disponibili")
ap.add_argument("--cpu", action="store_true", help="forza la CPU anche se c'e' una GPU")
# I due seguenti servono solo a collaudare: `--prova --epoche 30 --corpus 20000` addestra un
# modello nel regime vero (a max_sources 80) e deve riportare il dev intorno al 26%, che e' il
# numero noto di `esperimenti/notte/README.md` per quel livello. E' il modo di verificare che
# il percorso di addestramento riscritto per la GPU non abbia cambiato il metodo.
ap.add_argument("--epoche", type=int, default=0, help="forza le epoche massime (collaudi)")
ap.add_argument("--corpus", type=int, default=0, help="forza N_TRAIN_CORPUS")
ap.add_argument("--corpus-condiviso", action="store_true",
                help="un corpus solo per tutti i membri, come nella v1 (sconsigliato: "
                     "toglie il terzo asse di diversita' al bagging)")
ap.add_argument("--hidden", type=int, default=0, help="forza la larghezza dell'MLP")
# Colab passa argomenti suoi (`-f kernel.json`) quando si esegue un file dentro il notebook:
# `parse_known_args` evita che lo script muoia per quello.
args, _ignoti = ap.parse_known_args()

# stdout riga per riga. In Colab lo script gira come `!python ...`, quindi stdout e' una PIPE
# e Python passa al buffering a blocchi: senza questo si vedono silenzi lunghi seguiti da
# raffiche, ed e' impossibile capire se la sessione stia procedendo o si sia piantata. Solo 3
# print su ~120 avevano flush=True; questa riga li sistema tutti, anche quelli futuri.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

if args.threads:
    torch.set_num_threads(args.threads)
DEV = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
# La distanza di Hamming viene calcolata come prodotto matriciale (vedi `hamming_gemm`):
# gli ingressi sono 0/1 e i risultati interi <= 40, esatti in float32. TF32 arrotonderebbe
# la mantissa a 10 bit — abbastanza per interi fino a 2048, quindi in teoria innocuo, ma su
# un conto che deve essere BIT-IDENTICO al percorso CPU non si scommette.
torch.backends.cuda.matmul.allow_tf32 = False

# ===========================================================================================
# CONFIGURAZIONE E GRIGLIE — dichiarate qui, PRIMA di qualunque risultato
# ===========================================================================================
# Questo blocco e' il preregistro della sessione: chi rilegge lo script deve poter vedere
# cosa era stato deciso di provare prima di sapere come e' andata.

# --- il metodo (invariato rispetto a contributo-candidato/) ---
HIDDEN = 512               # MLP 1024 -> 512 -> 512, GELU, ultimo strato a zero
LR = 3e-4
BATCH = 256
K_VALUES = [1, 5, 10]

# --- punto di partenza della ricerca (dal brief) ---
PARTENZA = {"tau": 0.003, "ms": 2800, "wd": 0.05, "hidden": 512}
# --- configurazione incumbente = quella del 29/08, usata per il MINIMO GARANTITO (FM) ---
INCUMBENTE = {"tau": 0.003, "ms": 2800, "wd": 0.05, "hidden": 512}

# I DUE CANDIDATI DELLA v3. Non c'e' ricerca a coordinate: in v1 aveva scelto bene, in v2 ha
# scelto 1.5pp PEGGIO del meglio che aveva provato, perche' leggeva uno strumento rotto. Cio'
# che paga in modo replicato e' la diversita' dei membri, non la taratura. Quindi si sceglie
# fra due soli punti, entrambi gia' misurati e mai confrontati ad armi pari.
CANDIDATI = {
    # il punto della v2: e' l'unico su cui esista un ensemble con corpus per membro (41.25%)
    "A": {"tau": 0.003, "ms": 2800, "wd": 0.05, "hidden": 512},
    # il migliore sul DEV nella ricerca della v2 (34.67% di singolo) e il vincitore della v1
    "B": {"tau": 0.01, "ms": 1400, "wd": 1e-4, "hidden": 512},
}

# --- le tre griglie della ricerca a coordinate ---
GRIGLIA_A_TAU = [0.003, 0.005, 0.007, 0.01, 0.02, 0.03]   # 0.05 tolto: 32.53%, gia' escluso
# La sessione del 29/08 ha scelto tau 0.01, che era il valore PIU' BASSO della griglia:
# un ottimo su un estremo non e' un ottimo, e' il bordo della ricerca. Qui si estende sotto.
GRIGLIA_B_MS = [350, 700, 1400, 2800]         # al miglior tau
GRIGLIA_C_WD = [1e-2, 5e-2, 2e-1, 5e-1]       # al miglior (tau, ms)
# La griglia del 29/08 (1e-4, 1e-3, 1e-2) non misurava NULLA, ed e' aritmetica: con AdamW
# disaccoppiato l'aggiornamento e' p <- p(1 - lr*wd), e con lr 3e-4 su ~6500 passi i pesi si
# contraggono dello 0.02% (1e-4) e dello 0.19% (1e-3). Infatti quelle due celle diedero
# numeri identici a due decimali su validazione E dev. Serve 5e-2 per il 9%, 2e-1 per il 44%.
GRIGLIA_D_HIDDEN = [512, 256, 128]            # nuovo asse: meno capacita' = meno memorizzazione
# L'esperimento LayerNorm ha misurato che qui la capacita' e' DANNOSA (99.39% train contro
# 18.53% dev). Ridurre la larghezza e' la regolarizzazione piu' diretta e non e' mai stata
# provata in tutto il progetto.

# --- difese contro il sovradattamento alla validazione ---
# Le celle provate saranno ~8-10, ognuna letta su 150 query di validazione: l'errore
# standard della validazione e' circa 0.3pp, quindi una differenza sotto quella soglia e'
# rumore. Da qui le tre regole:
N_SEED_CELLA = 2           # ogni cella e' addestrata su due campioni di dati e si media
MARGINE_PP = 0.30          # l'incumbente si sostituisce solo se il candidato guadagna >= 0.30pp
REGOLA_NON_PERDERE = True  # ... e non perde su nessuno dei due seed
QSEED_RICERCA = [101, 102]           # seed della ricerca
QSEED_FINALI = list(range(11, 23))   # 12 seed: uno per membro, uno per corpus

# --- dati ---
SEED_DATI = 2026           # corpus, query del pool, permutazione delle reference
SEED_FIT = 999             # corpus separato per la ridge della fase 4 (come bagging_lineare)
N_TRAIN_CORPUS = 20000
POOL_TARGETS = {1: 80, 2: 140, 3: 100}
# Il pool si costruisce CON MARGINE sopra il livello massimo della griglia (1400), e il
# margine non e' un lusso: serve a dare a ogni membro dell'ensemble una FINESTRA diversa di
# reference (vedi `seleziona`). Senza margine, al livello 1400 tutti i membri prenderebbero
# le stesse identiche reference di ogni query in comune — e il +5.16pp del bagging misurato
# in esperimenti/notte/ veniva proprio dall'altro asse: li' il seed cambiava anche il
# campionamento delle reference, e il README di notte attribuisce il guadagno al fatto che
# "i campioni dei quattro modelli restano quasi disgiunti anche a 1400".
MS_POOL_GEMM = 4000        # se il percorso GEMM e' attivo: costa poco, e' un prodotto matriciale
MS_POOL_CPU = 3000         # col fallback CPU ogni reference in piu' costa minuti
MS_MAX = MS_POOL_GEMM      # valore vero deciso dopo il collaudo del GEMM, nella fase 1
HAMMING, MIN_TARGETS = 2, 5
N_VAL_QUERY = 150          # validazione: 150 query la rendono stabile (a 60 non lo era)
# composizione del BENCHMARK UFFICIALE (54/31/15 su 13 query distinte), non 0/50/50
VAL_TARGETS = ((1, 80), (2, 46), (3, 23))
VAL_MS = 60
# I due assi da cui viene la diversita' dell'ensemble; vedi `seleziona` per il come e il
# perche'. Solo il primo era presente nella prima stesura, e non bastava: al livello massimo
# della griglia i membri avrebbero condiviso tutte le reference di ogni query in comune.
BAG_FRAC = 0.75            # 1) frazione di query del pool vista da ogni modello
                           # 2) finestra circolare di reference sfalsata per membro (offset)

# --- addestramento ---
EPOCHE_MAX = 60           # tetto, non bersaglio: a fermarsi ci pensa PAZIENZA.
# Era 30. Ma il tetto e' uguale per tutti i livelli mentre i livelli BASSI
# convergono in piu' epoche: in esperimenti/notte/ a max_sources 350 il dev saliva
# ancora all'epoca 30 e la validazione sceglieva la 55, mentre a 1400 il picco era
# all'epoca 20. Un tetto a 30 lega quindi solo i livelli lenti, e introduce a
# specchio lo stesso confondimento del confronto a epoca fissa che quella sessione
# aveva gia' commesso nell'altro verso. Il costo e' nullo quando la patience scatta
# prima, che e' il caso atteso ai livelli alti.
VAL_OGNI = 2               # valutazione sulla validazione onesta ogni 2 epoche
PAZIENZA = 3               # 3 valutazioni senza miglioramento e ci si ferma

# --- bagging ---
N_MEMBRI_MINIMO = 4        # FM
N_MEMBRI_FINALI = 12       # F3
# Dodici e non sei. In v2 il sesto membro rendeva ANCORA +0.68pp (p=0.052) — la curva non
# era satura — e un addestramento a max_sources 2800 costa 1.7 minuti su T4: l'intera
# sessione v2 sono stati 40 addestramenti in 0.89 ore. Aggiungere membri e' la leva piu'
# economica che il progetto abbia, ed e' l'unica che ha pagato in modo replicato in tutte
# e tre le sessioni. La curva dei rendimenti fino a k=12 dice poi quanti consegnarne.

# --- ridge del collasso lineare ---
RIDGE = 1e-3               # NON usare lstsq: il sistema e' mal posto (rango 589 su 1025)
PER_QUERY_FIT = 60
N_BOOT = 5000
BASELINE_ATTESA = 19.91    # il collaudo del progetto, invariato dal 24/08
TOLL_COLLAUDO = 0.05

# --- modalita' prova: tutto piu' piccolo, stesso codice ---
if args.prova:
    N_TRAIN_CORPUS = 6000
    MS_POOL_GEMM = MS_POOL_CPU = MS_MAX = 160   # margine anche in prova: 160 = 2 x 80
    GRIGLIA_A_TAU = [0.01]
    GRIGLIA_B_MS = [80]
    GRIGLIA_C_WD = [5e-2]
    GRIGLIA_D_HIDDEN = [64]
    PARTENZA = {"tau": 0.03, "ms": 80, "wd": 1e-4, "hidden": 128}
    INCUMBENTE = {"tau": 0.03, "ms": 80, "wd": 1e-4, "hidden": 128}
    CANDIDATI = {"A": dict(INCUMBENTE), "B": {"tau": 0.01, "ms": 80, "wd": 1e-4, "hidden": 64}}
    N_SEED_CELLA = 1
    QSEED_RICERCA = [101]
    QSEED_FINALI = [11, 12, 13]
    N_MEMBRI_MINIMO = 2
    N_MEMBRI_FINALI = 3
    EPOCHE_MAX = 2
    N_VAL_QUERY = 30
    VAL_TARGETS = ((1, 16), (2, 9), (3, 5))
    N_BOOT = 500
    PER_QUERY_FIT = 20
    # il set di sviluppo NON si riduce nemmeno in prova: una passata costa ~6 secondi di CPU
    # e tenerlo intero e' cio' che rende la prova una prova vera — i numeri che stampa sono
    # confrontabili con quelli della repo invece di essere di un altro problema.

if args.epoche:
    EPOCHE_MAX = args.epoche
if args.corpus:
    N_TRAIN_CORPUS = args.corpus

# ===========================================================================================
# CARTELLA DI LAVORO — Drive se siamo su Colab, altrimenti locale
# ===========================================================================================
def _monta_drive():
    """Ritorna la cartella di lavoro. Su Colab e' su Drive, cosi' un disconnect non
    cancella niente; fuori da Colab e' locale. Il fallback non e' un caso limite: e' come
    gira `--prova`, ed e' la ragione per cui tutto il resto dello script non sa dove sta.
    """
    if args.out:
        return Path(args.out), "esplicita (--out)"
    if args.prova:
        return ROOT / "scripts" / "run_prova", "locale (--prova)"
    try:
        from google.colab import drive  # noqa: F401
        drive.mount("/content/drive")
        p = Path("/content/drive/MyDrive/DL-Project-2026/run")
        p.mkdir(parents=True, exist_ok=True)
        return p, "Google Drive"
    except Exception as e:
        print(f"  Drive non montato ({type(e).__name__}: {e}); uso una cartella locale.")
        return ROOT / "scripts" / "run_locale", "locale (fallback)"


RUN, DOVE_RUN = _monta_drive()
RUN.mkdir(parents=True, exist_ok=True)
(RUN / "ckpt").mkdir(exist_ok=True)
(RUN / "esiti").mkdir(exist_ok=True)
(RUN / "cache").mkdir(exist_ok=True)
JSONP = RUN / "risultati.json"

T0 = time.time()
DEADLINE = T0 + args.ore * 3600


def el():
    return f"{(time.time() - T0) / 60:6.1f}m"


def tempo_rimasto():
    """Secondi che restano prima della scadenza dichiarata con --ore."""
    return DEADLINE - time.time()


# Stima del costo di un addestramento, in minuti. Il valore iniziale e' un a-priori
# (misurato: 11 minuti per 18222 esempi x 30 epoche su 8 thread di CPU; su GPU circa
# quaranta volte meno), e viene ricalibrato sul tempo vero del primo addestramento: un
# a-priori sbagliato di un fattore 5 farebbe saltare fasi che ci stavano comodamente.
COSTO = {"min_per_esempio_epoca": 11.0 / (18222 * 30) if DEV.type == "cpu" else 5e-7,
         "calibrato": False}


def stima_minuti(n_esempi, epoche):
    return COSTO["min_per_esempio_epoca"] * max(n_esempi, 1) * epoche


def calibra(n_esempi, epoche, minuti):
    if n_esempi * epoche > 0:
        COSTO["min_per_esempio_epoca"] = minuti / (n_esempi * epoche)
        COSTO["calibrato"] = True


# ===========================================================================================
# RISULTATI: si rileggono all'avvio, si riscrivono dopo OGNI run
# ===========================================================================================
# La ripresa e' l'unica difesa contro un disconnect di Colab a meta' sessione: ogni
# configurazione gia' presente qui dentro (e con il suo checkpoint su disco) viene saltata.
if JSONP.exists():
    RISULTATI = json.loads(JSONP.read_text(encoding="utf-8"))
    print(f"riprendo da {JSONP} — {len(RISULTATI.get('run', {}))} addestramenti gia' fatti")
else:
    RISULTATI = {}
RISULTATI.setdefault("meta", {})
RISULTATI["meta"].update({
    "ultimo_avvio": datetime.now().isoformat(timespec="seconds"),
    "device": str(DEV), "prova": args.prova, "ore_budget": args.ore,
    "griglia_A_tau": GRIGLIA_A_TAU, "griglia_B_ms": GRIGLIA_B_MS, "griglia_C_wd": GRIGLIA_C_WD,
    "margine_adozione_pp": MARGINE_PP, "seed_cella": N_SEED_CELLA,
    "qseed_ricerca": QSEED_RICERCA, "qseed_finali": QSEED_FINALI,
    # l'affermazione e' su QUESTO script, non sul progetto: il benchmark ufficiale e' gia'
    # stato speso una volta il 25/08 (filtro zero-shot, esito negativo, scripts/README.md)
    "questo_script_legge_il_benchmark_ufficiale": False,
})
RISULTATI.setdefault("run", {})
RISULTATI.setdefault("fasi", {})


def salva():
    """Scrittura atomica: su Drive un'interruzione a meta' write lascerebbe un JSON
    troncato, cioe' perderebbe tutta la sessione invece dell'ultimo run."""
    tmp = JSONP.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(RISULTATI, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    os.replace(tmp, JSONP)


print("=" * 100)
print("CONSOLIDAMENTO v2 — sei fasi, tutto ripartibile, senza mai leggere il benchmark ufficiale")
print("=" * 100)
print(f"root       : {ROOT}")
print(f"lavoro     : {RUN}   [{DOVE_RUN}]")
print(f"device     : {DEV}")
print(f"avvio      : {datetime.now():%Y-%m-%d %H:%M}   scadenza: "
      f"{datetime.now() + timedelta(hours=args.ore):%H:%M} ({args.ore} ore)")
print(f"modalita'  : {'PROVA (numeri senza valore, serve solo a far girare il codice)' if args.prova else 'lancio vero'}")

# ===========================================================================================
# F0 — SETUP: embedding, direzioni testuali, collaudo della baseline
# ===========================================================================================
print("\n" + "=" * 100)
print("F0 — SETUP E COLLAUDO")
print("=" * 100)


def carica_split(split):
    """Legge un embedding split, preferendo la copia su Drive.

    Il file `train` pesa 385 MB e arriva via git-lfs, che ha una banda mensile gratuita di
    1 GB: scaricarlo a ogni ripresa non e' sostenibile. Alla prima esecuzione lo si legge
    dalla repo; da li' in poi bastano le derivate salvate in `cache/` (corpus, pool,
    direzioni), che pesano insieme meno di un decimo.
    """
    p = ROOT / f"embeddings/celeba_{split}_clip_vitb32.pt"
    if not p.exists():
        raise SystemExit(f"manca {p}. Se hai clonato con GIT_LFS_SKIP_SMUDGE=1, "
                         f"lancia: git lfs pull -I embeddings/celeba_{split}_clip_vitb32.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    E = d["embeddings"].float()
    return E / E.norm(dim=-1, keepdim=True), d["labels"], d


print(f"[{el()}] carico valid...")
E_valid_cpu, L_valid, meta_valid = carica_split("valid")
ATTR = [n for n in meta_valid["attr_names"] if n.strip()]
D, N_ATTR = E_valid_cpu.shape[1], L_valid.shape[1]
devset_cpu = load_benchmark(str(ROOT / "celeba_devset_valid.json"), ATTR)
print(f"[{el()}] valid {tuple(E_valid_cpu.shape)}, {len(ATTR)} attributi, "
      f"devset {len(devset_cpu)} query")

# Il nome NON dipende piu' da MS_MAX: il livello vero del pool si conosce solo dopo il
# collaudo del GEMM, e leggerlo dalla cache e' l'unico modo di saltare la fase 1 senza
# ricaricare i 385 MB del train solo per decidere come chiamare il file.
CACHE_DATI = RUN / "cache" / "dati_pool.pt"

# --- direzioni testuali: identiche a contributo-candidato/multiseed.py ---------------------
TEMPLATES = [
    ("a photo of a person who is {}",        "a photo of a person who is not {}"),
    ("a photo of a person with {}",          "a photo of a person without {}"),
    ("a face with {}",                       "a face without {}"),
    ("{}",                                   "not {}"),
    ("a portrait photo of someone with {}",  "a portrait photo of someone with no {}"),
    ("a photo of a {} person",               "a photo of a person that is not {}"),
    ("a cropped photo of a face with {}",    "a cropped photo of a face without {}"),
    ("a close-up photo of a person with {}", "a close-up photo of a person without {}"),
    ("a photo of someone who has {}",        "a photo of someone who has no {}"),
    ("a picture of a face showing {}",       "a picture of a face not showing {}"),
]


def costruisci_direzioni():
    """Le 40 direzioni bipolari, mediate su 10 coppie di template.

    Il `hasattr(out, "pooler_output")` non e' difensivismo: transformers 5.x fa restituire a
    `get_text_features` un oggetto il cui `pooler_output` e' gia' proiettato, mentre 4.x
    restituisce direttamente il tensore proiettato. Le due strade danno lo stesso vettore, e
    quella riga e' cio' che rende il collaudo a 19.91% indipendente dalla versione
    installata su Colab. Verificato il 29/08 su transformers 5.15: 19.9141%.
    """
    from transformers import CLIPModel, CLIPTokenizer
    mid = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(mid).eval()
    tokenizer = CLIPTokenizer.from_pretrained(mid)

    @torch.no_grad()
    def encode(prompts):
        out = clip_model.get_text_features(
            **tokenizer(prompts, padding=True, return_tensors="pt"))
        f = (out.pooler_output if hasattr(out, "pooler_output") else out).float()
        return f / f.norm(dim=-1, keepdim=True)

    d = torch.zeros(N_ATTR, D)
    for i, name in enumerate(ATTR):
        concetto = name.replace("_", " ").lower()
        v = encode([t.format(concetto) for t, _ in TEMPLATES]).mean(0) - \
            encode([t.format(concetto) for _, t in TEMPLATES]).mean(0)
        d[i] = v / v.norm()
    del clip_model, tokenizer
    return d


_cache_dir = None
if CACHE_DATI.exists():
    _cache_dir = torch.load(CACHE_DATI, map_location="cpu", weights_only=False)
    d_ens_cpu = _cache_dir["d_ens"]
    print(f"[{el()}] direzioni testuali riprese dalla cache (niente CLIP da scaricare)")
else:
    print(f"[{el()}] costruisco le direzioni testuali con CLIP...")
    d_ens_cpu = costruisci_direzioni()
d_ens = d_ens_cpu.to(DEV)
print(f"[{el()}] direzioni {tuple(d_ens.shape)}")


# --- rappresentazione compatta di un benchmark --------------------------------------------
# I ground truth arrivano come liste di `set` Python: al livello 1400 sarebbero ~5 milioni di
# interi in set, cioe' centinaia di MB di oggetti Python e un ciclo per batch. Qui diventano
# un tensore piatto piu' un vettore di offset (rappresentazione "ragged"), che si indicizza
# senza cicli — e' quello che chiude la trappola della maschera dei positivi.
def compatta(bench, device=None):
    device = device or DEV
    out = []
    for e in bench:
        lens = torch.tensor([len(g) for g in e["gt"]], dtype=torch.long)
        ptr = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
        flat = torch.tensor([t for g in e["gt"] for t in sorted(g)], dtype=torch.long)
        c = torch.zeros(D)
        for a in e["pos"]:
            c = c + d_ens_cpu[a]
        for b in e["neg"]:
            c = c - d_ens_cpu[b]
        out.append({"query": e["query"], "pos": list(e["pos"]), "neg": list(e["neg"]),
                    "n_cond": len(e["pos"]) + len(e["neg"]),
                    "sources": e["sources"].to(device), "gt_flat": flat.to(device),
                    "gt_ptr": ptr[:-1].to(device), "gt_len": lens.to(device),
                    "c": c.to(device)})
    return out


def ragged(flat, ptr, lens, idx):
    """Concatena le slice `flat[ptr[i] : ptr[i]+lens[i]]` per ogni i in `idx`, senza cicli.

    Ritorna (righe, colonne) pronte per un `index_put_` su una maschera [len(idx), N].
    E' la sostituzione GPU del ciclo Python su 256 elementi per batch che c'era nella loss:
    quel ciclo, su GPU, costava piu' del prodotto matriciale che doveva preparare.
    """
    l = lens[idx]
    tot = int(l.sum())
    if tot == 0:
        z = torch.zeros(0, dtype=torch.long, device=flat.device)
        return z, z
    righe = torch.repeat_interleave(torch.arange(len(idx), device=flat.device), l)
    inizio = torch.cumsum(l, 0) - l
    offs = torch.arange(tot, device=flat.device) - torch.repeat_interleave(inizio, l)
    col = flat[torch.repeat_interleave(ptr[idx], l) + offs]
    return righe, col


# --- valutazione vettorizzata --------------------------------------------------------------
def valuta(build_query, bench, E, k_values=K_VALUES, batch=512):
    """Come `evaluate_per_source` di src/devset.py, e con lo STESSO valore di ritorno
    `{k: [np.ndarray[n_src], ...]}`, ma senza `.tolist()` ne' lookup su set Python.

    Sul solo set di sviluppo la versione originale fa 27.206 `.tolist()` e ~1.1 milioni di
    lookup su `set`: su CPU e' nascosto dal costo dell'addestramento, su GPU diventerebbe la
    voce principale del tempo, perche' e' l'unica parte che non ci gira sopra.
    L'equivalenza e' verificata numericamente all'avvio (`collaudo_valutazione`).
    """
    kmax = max(k_values)
    per_source = {k: [] for k in k_values}
    N = E.shape[0]
    for entry in bench:
        src = entry["sources"]
        rec = {k: np.zeros(len(src)) for k in k_values}
        for st in range(0, len(src), batch):
            sp = min(st + batch, len(src))
            bs = src[st:sp]
            b = sp - st
            with torch.no_grad():
                q = build_query(E[bs], entry)
                sim = q @ E.T
                # la reference non puo' essere un target: si esclude prima del topk
                sim[torch.arange(b, device=E.device), bs] = -float("inf")
                top = sim.topk(kmax, dim=-1).indices              # [b, kmax]
                mask = torch.zeros(b, N, dtype=torch.bool, device=E.device)
                righe, col = ragged(entry["gt_flat"], entry["gt_ptr"], entry["gt_len"],
                                    torch.arange(st, sp, device=E.device))
                mask[righe, col] = True
                hit = mask.gather(1, top).cummax(dim=1).values    # "almeno un hit entro k"
                hit = hit.float().cpu().numpy()
            for k in k_values:
                rec[k][st:sp] = hit[:, k - 1]
        for k in k_values:
            per_source[k].append(rec[k])
    return per_source


def build_baseline(ref, entry):
    """La baseline dell'assignment: q = normalize(r + somma con segno delle direzioni)."""
    q = ref + entry["c"]
    return q / q.norm(dim=-1, keepdim=True)


def collaudo_valutazione():
    """Verifica che `valuta` dia esattamente quello che darebbe `evaluate_per_source`.

    Si esegue su CPU e su tre entry del devset: `evaluate_per_source` costruisce i suoi
    indici senza `device=`, quindi non gira su GPU, ed e' giusto cosi' — e' codice del
    protocollo, non si tocca. Qui serve solo come oracolo.
    """
    piccolo = []
    for e in devset_cpu[:3]:
        piccolo.append({**e, "sources": e["sources"][:50], "gt": e["gt"][:50]})
    atteso = evaluate_per_source(
        lambda ref, entry: (lambda q: q / q.norm(dim=-1, keepdim=True))(
            ref + sum((d_ens_cpu[a] for a in entry["pos"]), torch.zeros(D))
            - sum((d_ens_cpu[b] for b in entry["neg"]), torch.zeros(D))),
        piccolo, E_valid_cpu, K_VALUES)
    ottenuto = valuta(build_baseline, compatta(piccolo, torch.device("cpu")),
                      E_valid_cpu, K_VALUES)
    for k in K_VALUES:
        for a, b in zip(atteso[k], ottenuto[k]):
            if not np.array_equal(a, b):
                raise SystemExit(
                    "COLLAUDO FALLITO: la valutazione vettorizzata non coincide con "
                    "evaluate_per_source di src/devset.py. Non proseguo: tutti i numeri "
                    "successivi sarebbero incomparabili con quelli della repo.")
    print(f"[{el()}] valutazione vettorizzata == evaluate_per_source  OK")


collaudo_valutazione()

# --- COLLAUDO della baseline: e' il controllo che il progetto usa da sempre -----------------
E_valid = E_valid_cpu.to(DEV)
devset = compatta(devset_cpu)
ps_base_full = valuta(build_baseline, devset, E_valid)
B10_FULL = macro(ps_base_full[10]) * 100
print(f"[{el()}] baseline sul set di sviluppo (101 query): {B10_FULL:.2f}%  "
      f"atteso {BASELINE_ATTESA:.2f}%")
if abs(B10_FULL - BASELINE_ATTESA) > TOLL_COLLAUDO:
    raise SystemExit(
        f"\n*** COLLAUDO FALLITO ***\n"
        f"La baseline vale {B10_FULL:.2f}% invece di {BASELINE_ATTESA:.2f}% "
        f"(+/-{TOLL_COLLAUDO}).\n"
        f"E' il controllo di implementazione che il progetto usa dal 24/08: se non torna,\n"
        f"qualcosa a monte e' cambiato e ogni numero prodotto da qui in poi sarebbe\n"
        f"incomparabile con la tabella di docs/STATO.md. Non proseguo.\n\n"
        f"Da guardare, in quest'ordine:\n"
        f"  1. le direzioni testuali. Con ~14.2% stai usando le feature di testo "
        f"proiettate DUE volte,\n"
        f"     o comunque un'altra uscita di CLIP: vedi `costruisci_direzioni`.\n"
        f"  2. gli embedding: embeddings/celeba_valid_clip_vitb32.pt deve essere il file "
        f"della repo\n"
        f"     (19867 x 512), non una riestrazione.\n"
        f"  3. celeba_devset_valid.json: deve avere 101 query.\n")
RISULTATI["meta"]["baseline_dev_r10"] = B10_FULL
salva()

ps_base = ps_base_full
B10 = B10_FULL

# ===========================================================================================
# F1 — DATI, UNA VOLTA SOLA
# ===========================================================================================
print("\n" + "=" * 100)
print("F1 — DATI (pool con margine + validazione onesta), costruiti una volta sola")
print("=" * 100)


def hamming_gemm(ref_nq, cand_nq):
    """Distanza di Hamming fra due insiemi di vettori binari, come prodotto matriciale.

    Per label in {0,1}:  H[i,j] = sum_a  ref[i,a]*(1-cand[j,a]) + (1-ref[i,a])*cand[j,a]
                              =  ref @ (1-cand).T + (1-ref) @ cand.T

    Con 37-39 colonne i valori sono interi <= 39, esattamente rappresentabili in float32:
    il risultato non e' approssimato, e' lo stesso identico intero che dava il tensore
    booleano 3D di `build_benchmark_from_labels`. Quel tensore, `(cand[None] != ref[:,None])`,
    materializza [b, n_cand, n_attr] valori e su 2 vCPU e' il collo di bottiglia dell'intera
    sessione (~5 minuti per seed); qui e' un GEMM e su GPU sono secondi.
    L'equivalenza e' verificata esattamente da `collaudo_hamming` prima di essere usata.
    """
    uno_c = 1.0 - cand_nq
    return ref_nq @ uno_c.T + (1.0 - ref_nq) @ cand_nq.T


def costruisci(queries, labels_f, ms_max, seed, min_targets=MIN_TARGETS,
               hamming=HAMMING, blocco=256, con_rango=True):
    """Ricostruisce il benchmark con il protocollo della sez. 3.1 (la stessa regola di
    `build_benchmark_from_labels`), ma sul device e con i ranghi delle reference.

    IL RANGO. Per ogni query si permutano TUTTE le reference violanti con un generatore
    fissato e si tiene il prefisso di lunghezza `ms_max`, memorizzando per ognuna la sua
    posizione nella permutazione. La vista a `max_sources = k` e' allora la maschera
    `rango < k`. Questo NON e' l'approssimazione "sottocampiono le source gia' filtrate":
    il prefisso di lunghezza k di una permutazione uniforme e' un sottoinsieme uniforme di
    dimensione k, e il filtro di Hamming e' per-reference, quindi la vista a k coincide
    ESATTAMENTE con la procedura originale (prima si sottocampiona, poi si filtra).
    Non e' bit-identica alla chiamata originale — la sequenza dell'RNG e' diversa, perche'
    li' il generatore e' unico per l'intera chiamata e qui e' per query — ma e'
    distribuzionalmente la stessa cosa.

    Il vantaggio vero e' che i livelli diventano ANNIDATI: 350 e' un sottoinsieme di 700, che
    e' un sottoinsieme di 1400, sulle stesse query. Il confronto 700 vs 1400 diventa quindi
    accoppiato sullo stesso prefisso di reference — che e' esattamente il confronto che il
    gruppo non e' finora riuscito a fare (`esperimenti/notte/README.md`, "l'esperimento che
    chiude la questione").

    Il rango serve pero' a due cose, non a una: il prefisso `rango < ms` da' le viste annidate
    alla ricerca, e la finestra circolare `[off, off + ms)` da' a ogni membro dell'ensemble
    reference diverse. Per questo il pool si costruisce sopra il livello massimo della
    griglia: senza margine la finestra non ha dove scorrere. Vedi `seleziona`.
    """
    n_attr = labels_f.shape[1]
    attr_to_idx = {n: i for i, n in enumerate(ATTR)}
    fuori = []
    # Costruire il pool a ms 2800 su 320 query e' la fase muta piu' lunga dello script: e' il
    # punto in cui chi guarda il log pensa che si sia bloccato.
    # la soglia a 20 tiene muto il collaudo del GEMM, che chiama questa funzione su 5 query
    _t0, _ogni = time.time(), (max(1, len(queries) // 10) if len(queries) >= 20 else 0)
    for qi, query in enumerate(queries):
        if _ogni and qi and qi % _ogni == 0:
            print(f"    ...{qi}/{len(queries)} query costruite, {len(fuori)} valide "
                  f"({time.time() - _t0:.0f}s)", flush=True)
        pos, neg = parse_query_string(query, attr_to_idx)
        queried = pos + neg
        non_queried = [i for i in range(n_attr) if i not in queried]

        soddisfa = torch.ones(len(labels_f), dtype=torch.bool, device=labels_f.device)
        for i in pos:
            soddisfa &= labels_f[:, i] == 1
        for j in neg:
            soddisfa &= labels_f[:, j] == 0
        cand = soddisfa.nonzero(as_tuple=True)[0]

        # la reference deve violare OGNI condizione (regola dedotta e verificata sul JSON
        # ufficiale: vedi il commento in src/benchmark.py)
        viola = torch.ones(len(labels_f), dtype=torch.bool, device=labels_f.device)
        for i in pos:
            viola &= labels_f[:, i] == 0
        for j in neg:
            viola &= labels_f[:, j] == 1
        refs = viola.nonzero(as_tuple=True)[0]
        if len(cand) == 0 or len(refs) == 0:
            continue

        if con_rango:
            g = torch.Generator(device="cpu").manual_seed(seed + 1000 * qi)
            perm = torch.randperm(len(refs), generator=g).to(refs.device)
            refs = refs[perm]
            if ms_max is not None and len(refs) > ms_max:
                refs = refs[:ms_max]
            ranghi = torch.arange(len(refs), device=refs.device)
        else:
            ranghi = torch.arange(len(refs), device=refs.device)

        cand_nq = labels_f[cand][:, non_queried]
        keep_src, keep_rank, gt_parts = [], [], []
        for st in range(0, len(refs), blocco):
            bl = refs[st:st + blocco]
            H = hamming_gemm(labels_f[bl][:, non_queried], cand_nq)
            ok = H <= hamming + 0.5          # interi esatti: la soglia .5 evita ogni dubbio
            ok &= cand.unsqueeze(0) != bl.unsqueeze(1)     # la reference non e' un suo target
            n_ok = ok.sum(1)
            tenuti = (n_ok >= min_targets).nonzero(as_tuple=True)[0]
            if len(tenuti) == 0:
                continue
            ok = ok[tenuti]
            nz = ok.nonzero(as_tuple=False)
            keep_src.append(bl[tenuti])
            keep_rank.append(ranghi[st:st + blocco][tenuti])
            gt_parts.append((cand[nz[:, 1]].to(torch.int32).cpu(), n_ok[tenuti].cpu()))
        if not keep_src:
            continue
        sources = torch.cat(keep_src).cpu()
        rango = torch.cat(keep_rank).cpu()
        gt_flat = torch.cat([p[0] for p in gt_parts])
        gt_len = torch.cat([p[1] for p in gt_parts]).long()
        fuori.append({"query": query, "pos": pos, "neg": neg, "n_cond": len(queried),
                      "sources": sources, "rango": rango,
                      # quante reference sono entrate nella permutazione (dopo il troncamento
                      # a ms_max): e' il modulo delle finestre circolari di `seleziona`, e
                      # dice anche quando il bacino di una query si e' esaurito
                      "n_rango": int(len(refs)),
                      "n_bacino": int(len(viola.nonzero(as_tuple=True)[0])),
                      "gt_flat": gt_flat, "gt_len": gt_len,
                      "gt_ptr": torch.cat([torch.zeros(1, dtype=torch.long),
                                           gt_len.cumsum(0)])[:-1]})
    return fuori


def collaudo_hamming(labels_f_cpu):
    """Equivalenza ESATTA fra il percorso GEMM e `build_benchmark_from_labels`.

    Cinque query, `max_sources=None` (quindi nessun RNG di mezzo, quindi deterministico), su
    una fetta piccola del corpus perche' il percorso CPU con tutte le reference costerebbe
    minuti. Si confrontano gli insiemi di reference sopravvissute e, per ognuna, l'insieme
    dei target. Se non coincidono, non si discute: si torna al percorso CPU.
    """
    fetta = labels_f_cpu[:3000]
    qs = []
    for nc in (1, 2, 3):
        qs += sample_queries(list(ATTR), 2 if nc < 3 else 1, nc, exclude=qs, seed=4242 + nc)
    atteso = build_benchmark_from_labels(qs, fetta.long(), ATTR, hamming_threshold=HAMMING,
                                         min_targets=MIN_TARGETS, max_sources=None)
    ottenuto = costruisci(qs, fetta, ms_max=None, seed=0, con_rango=False)
    da = {e["query"]: e for e in atteso}
    do = {e["query"]: e for e in ottenuto}
    if set(da) != set(do):
        return False, f"query sopravvissute diverse: {set(da) ^ set(do)}"
    for q in da:
        a, b = da[q], do[q]
        if set(a["sources"].tolist()) != set(b["sources"].tolist()):
            return False, f"reference diverse su {q!r}"
        ga = {int(s): set(g) for s, g in zip(a["sources"].tolist(), a["gt"])}
        for i, s in enumerate(b["sources"].tolist()):
            st = int(b["gt_ptr"][i])
            gb = set(b["gt_flat"][st:st + int(b["gt_len"][i])].tolist())
            if ga[int(s)] != gb:
                return False, f"target diversi sulla reference {s} di {q!r}"
    return True, f"{len(qs)} query, {sum(len(e['sources']) for e in atteso)} reference"


def esempi_totali(pool, ms):
    return int(sum(int((e["rango"] < ms).sum()) for e in pool))


def bacini_esauriti(pool, ms):
    """Quante query hanno meno di `ms` reference disponibili.

    Sono le query su cui TUTTI i membri dell'ensemble prendono lo stesso identico insieme di
    reference: la finestra circolare di `seleziona` copre tutto il bacino e l'offset non ha
    niente da spostare. E' inevitabile e va contato, non nascosto — e' la stessa misura della
    tabella "quando N_TRAIN_CORPUS diventa il vincolo" di esperimenti/notte/README.md.
    """
    # si misura sul bacino VERO (`n_bacino`), non su `n_rango` che e' gia' troncato a MS_MAX:
    # altrimenti al livello del pool il conto sarebbe per costruzione il 100%. Col fallback
    # CPU il bacino vero non e' noto e si ripiega su `n_rango`.
    return int(sum(1 for e in pool if (e.get("n_bacino") or e["n_rango"]) <= ms))


# una cache costruita a un livello piu' basso della griglia non e' utilizzabile: darebbe
# finestre di reference tutte uguali, che e' il difetto che il margine esiste per evitare
# una cache della v2 non e' utilizzabile: non ha la galleria di validazione disgiunta
_cache_utile = (_cache_dir is not None and "pool" in _cache_dir
                and _cache_dir.get("versione", 0) >= 3
                and _cache_dir.get("ms_max", 0) > max(GRIGLIA_B_MS))
if _cache_dir is not None and "pool" in _cache_dir and not _cache_utile:
    print(f"[{el()}] cache scartata: costruita a ms_max={_cache_dir.get('ms_max')}, "
          f"serve un margine sopra {max(GRIGLIA_B_MS)}. La rifaccio.")
if _cache_utile:
    MS_MAX = _cache_dir["ms_max"]
    print(f"[{el()}] dati ripresi dalla cache {CACHE_DATI.name} (pool a ms_max={MS_MAX}): "
          f"la fase 1 si salta")
    POOL = _cache_dir["pool"]
    QTXT_POOL = _cache_dir.get("qtxt")
    VAL = _cache_dir["val"]
    sub = _cache_dir["sub"]
    E_tr = _cache_dir["E_tr"].to(DEV)
    L_tr = _cache_dir["L_tr"]
    E_VG = _cache_dir["E_vg"].to(DEV)
    sub_vg = _cache_dir["sub_vg"]
    E_fit_cpu, L_fit_cpu = _cache_dir["E_fit"], _cache_dir["L_fit"]
    GEMM_OK = _cache_dir.get("gemm_ok", True)
else:
    print(f"[{el()}] carico train (385 MB: e' l'unica volta, poi resta la cache)...")
    E_train_all, L_train_all, _ = carica_split("train")

    # R1 — DUE GALLERIE DISGIUNTE, non una.
    # Nella v2 la validazione era valutata sulle STESSE 20000 immagini che stanno nel
    # denominatore dell'InfoNCE: le query erano nuove, la galleria no. Una validazione che
    # condivide la galleria con l'addestramento NON PUO' rilevare il sovradattamento alla
    # galleria — ed e' esattamente in cui la v2 e' scivolata (tau 0.003 e max_sources 2800
    # sono entrambi modi di sfruttarla piu' a fondo). Risultato misurato: la correlazione di
    # Spearman fra validazione e set di sviluppo e' crollata da +0.897 a +0.244.
    # Qui la permutazione si taglia in due meta' disgiunte PER COSTRUZIONE.
    g = torch.Generator().manual_seed(SEED_DATI)
    _perm = torch.randperm(len(E_train_all), generator=g)
    sub = _perm[:N_TRAIN_CORPUS].sort().values
    sub_vg = _perm[N_TRAIN_CORPUS:2 * N_TRAIN_CORPUS].sort().values
    assert not (set(sub.tolist()) & set(sub_vg.tolist())), "le due gallerie si sovrappongono"
    E_tr_cpu, L_tr = E_train_all[sub], L_train_all[sub]
    E_vg_cpu, L_vg = E_train_all[sub_vg], L_train_all[sub_vg]
    # corpus separato per la ridge della fase 4: le coppie (r, c) su cui si stima la mappa
    # devono venire dal train, e conviene che non siano le stesse reference viste in
    # addestramento — e' la scelta di contributo-candidato/bagging_lineare.py (SEED_FIT 999).
    gf = torch.Generator().manual_seed(SEED_FIT)
    subf = torch.randperm(len(E_train_all), generator=gf)[:N_TRAIN_CORPUS].sort().values
    E_fit_cpu, L_fit_cpu = E_train_all[subf], L_train_all[subf]
    # NON si cancella: i contesti per membro (vedi `usa_contesto`) ricampionano il corpus
    # da qui. E' la differenza che ha fatto perdere 2.1pp di bagging nella sessione del 29/08.

    E_tr = E_tr_cpu.to(DEV)
    L_tr_f = L_tr.float().to(DEV)
    E_VG = E_vg_cpu.to(DEV)          # galleria della validazione, mai vista in addestramento
    L_vg_f = L_vg.float().to(DEV)

    print(f"[{el()}] collaudo del percorso GEMM per la distanza di Hamming...")
    GEMM_OK, msg = collaudo_hamming(L_tr.float())
    print(f"[{el()}]   {'OK' if GEMM_OK else 'FALLITO'} — {msg}")
    # il livello del pool si decide QUI, perche' dipende da quale dei due percorsi si usa:
    # col GEMM il margine costa un prodotto matriciale piu' grande, col fallback CPU costa
    # minuti per ogni reference in piu'
    MS_MAX = MS_POOL_GEMM if GEMM_OK else MS_POOL_CPU
    if not GEMM_OK:
        print("  *** fallback al percorso CPU di src/benchmark.py, pool limitato a "
              f"{MS_MAX} invece di {MS_POOL_GEMM}. La costruzione sara' lenta (minuti) e i "
              "livelli NON saranno annidati: il confronto fra livelli restera' non "
              "accoppiato. ***")
    assert MS_MAX > max(GRIGLIA_B_MS), (
        f"il pool ({MS_MAX}) deve stare SOPRA il livello massimo della griglia "
        f"({max(GRIGLIA_B_MS)}): senza margine i membri dell'ensemble vedrebbero le stesse "
        f"reference e resterebbe diverso solo il campione di query")

    # --- pool di query -------------------------------------------------------------------
    # Come in notte.py il pool NON esclude le query del set di sviluppo: lo spazio a una
    # condizione ha esattamente 40x2 = 80 elementi e chiederne 80 lo esaurisce, quindi
    # escluderle sarebbe impossibile. E' la pratica del progetto da sempre (le query sono
    # le stesse, gli esempi no: qui si addestra sul train, li' si valuta sul valid).
    qtxt = []
    for n_cond, quanti in POOL_TARGETS.items():
        qtxt += sample_queries(list(ATTR), quanti, n_cond, exclude=qtxt,
                               seed=SEED_DATI + n_cond)
    print(f"[{el()}] pool: {len(qtxt)} query campionate, costruisco a max_sources={MS_MAX}...")
    t = time.time()
    if GEMM_OK:
        POOL = costruisci(qtxt, L_tr_f, ms_max=MS_MAX, seed=SEED_DATI)
    else:
        POOL = []
        for qi, qt in enumerate(qtxt):
            b = build_benchmark_from_labels([qt], L_tr, ATTR, hamming_threshold=HAMMING,
                                            min_targets=MIN_TARGETS, max_sources=MS_MAX,
                                            seed=SEED_DATI)
            for e in b:
                # senza ranghi veri si assegna una permutazione alle source SOPRAVVISSUTE:
                # e' l'approssimazione della procedura originale (li' il sottocampionamento
                # avviene PRIMA del filtro di Hamming). La si accetta solo in fallback,
                # perche' rifare la costruzione a ogni livello costerebbe ~20 minuti di CPU
                # per livello e i livelli sono tre.
                gg = torch.Generator().manual_seed(SEED_DATI + qi)
                rango = torch.randperm(len(e["sources"]), generator=gg)
                lens = torch.tensor([len(x) for x in e["gt"]], dtype=torch.long)
                POOL.append({
                    "query": e["query"], "pos": e["pos"], "neg": e["neg"],
                    "n_cond": len(e["queried"]), "sources": e["sources"], "rango": rango,
                    # col fallback CPU il bacino vero non e' osservabile: resta None
                    "n_rango": len(e["sources"]), "n_bacino": None,
                    "gt_flat": torch.tensor([t2 for gset in e["gt"] for t2 in sorted(gset)],
                                            dtype=torch.int32),
                    "gt_len": lens,
                    "gt_ptr": torch.cat([torch.zeros(1, dtype=torch.long),
                                         lens.cumsum(0)])[:-1]})
    print(f"[{el()}] pool: {len(POOL)} query valide, "
          f"{esempi_totali(POOL, MS_MAX)} esempi a ms={MS_MAX}, {time.time() - t:.0f}s")
    for _m in sorted(set(GRIGLIA_B_MS) | {MS_MAX}):
        _e = bacini_esauriti(POOL, _m)
        print(f"[{el()}]   a ms {_m:>5}: {esempi_totali(POOL, _m):>7} esempi, bacino "
              f"esaurito su {_e}/{len(POOL)} query ({100 * _e / max(len(POOL), 1):.0f}%)")

    # --- validazione onesta ---------------------------------------------------------------
    # 150 query nuove: mai addestrate, disgiunte dal pool E dal set di sviluppo. Servono a
    # UNA cosa sola — scegliere iperparametri ed epoca di arresto senza guardare il dev.
    # A 60 query (notte.py) la scelta era instabile: per max_sources 350 sceglieva l'epoca
    # 55, che sul dev valeva un punto in meno dell'epoca 30. Da qui le 150.
    # R2 — LA COMPOSIZIONE DEL BENCHMARK, non 0/50/50.
    # Il benchmark ufficiale e' 54/31/15 per numero di condizioni (7 query a una condizione,
    # 4 a due, 2 a tre, su 13 distinte) e il set di sviluppo lo riproduce (53/32/15). La v2
    # usava SOLO query composte, cioe' misurava solo lo strato che beneficia di piu' dei
    # dati: ed e' il motivo per cui sovrastimava sistematicamente i livelli alti di
    # max_sources. Qui si punta a 80/46/23, cioe' 54/31/15 su 149.
    #
    # LE QUERY A UNA CONDIZIONE NON SI POSSONO ESCLUDERE, e non e' un compromesso: lo spazio
    # ha esattamente 40x2 = 80 elementi e il pool li usa tutti, quindi non ne esistono di
    # inedite. Ma e' anche la scelta FEDELE: nel set di sviluppo tutte e 54 le query a una
    # condizione compaiono in addestramento (docs/STATO.md), quindi una validazione che le
    # escludesse misurerebbe una popolazione diversa da quella su cui si riporta. Cio' che
    # deve essere inedito e' la GALLERIA, ed e' garantito da R1.
    dev_query_txt = {e["query"] for e in devset_cpu}
    esclusi = set(qtxt) | dev_query_txt
    # STRATIFICATO PER STRATO, non a giri misti. Gli strati hanno rese molto diverse
    # (a 1 condizione sopravvive il 100% delle query campionate, a 2 l'83%, a 3 il 52%):
    # campionandoli insieme e troncando alla fine, la composizione deriva verso lo strato
    # con la resa piu' alta. Qui ogni strato si riempie fino alla propria quota per conto
    # suo, e la composizione risultante viene stampata come MISURATA, non come voluta.
    VAL, qval = [], []
    for n_cond, quota in VAL_TARGETS:
        strato = []
        for giro in range(12):
            if len(strato) >= quota:
                break
            manca = quota - len(strato)
            # a 1 condizione lo spazio (40 x 2 = 80) e' esaurito dal pool: si riusano le
            # query, mai le immagini. A 2 e 3 condizioni lo spazio e' enorme e si pretendono
            # inedite rispetto al pool e al set di sviluppo.
            escl = set(qval) if n_cond == 1 else (esclusi | set(qval))
            nuove = sample_queries(list(ATTR), max(2 * manca, 8), n_cond, exclude=escl,
                                   seed=SEED_DATI + 777 + 100 * n_cond + giro)
            if not nuove:
                break
            qval += nuove
            if GEMM_OK:
                strato += costruisci(nuove, L_vg_f, ms_max=VAL_MS, seed=SEED_DATI)
            else:
                for qt in nuove:
                    for e in build_benchmark_from_labels([qt], L_vg, ATTR,
                                                         hamming_threshold=HAMMING,
                                                         min_targets=MIN_TARGETS,
                                                         max_sources=VAL_MS, seed=SEED_DATI):
                        lens = torch.tensor([len(x) for x in e["gt"]], dtype=torch.long)
                        strato.append({"query": e["query"], "pos": e["pos"], "neg": e["neg"],
                                       "n_cond": len(e["queried"]), "sources": e["sources"],
                                       "rango": torch.arange(len(e["sources"])),
                                       "n_rango": len(e["sources"]),
                                       "n_bacino": len(e["sources"]),
                                       "gt_flat": torch.tensor(
                                           [t2 for gs in e["gt"] for t2 in sorted(gs)],
                                           dtype=torch.int32),
                                       "gt_len": lens,
                                       "gt_ptr": torch.cat([torch.zeros(1, dtype=torch.long),
                                                            lens.cumsum(0)])[:-1]})
        if len(strato) < quota:
            print(f"[{el()}]   strato a {n_cond} condizioni sotto quota: "
                  f"{len(strato)}/{quota} (spazio o ground truth esauriti)")
        VAL += strato[:quota]
    sovr = {e["query"] for e in VAL if e["n_cond"] > 1} & (set(qtxt) | dev_query_txt)
    assert not sovr, f"query composte di validazione gia' viste: {sovr}"
    _cv = {k: sum(1 for e in VAL if e["n_cond"] == k) for k in (1, 2, 3)}
    _tv = max(len(VAL), 1)
    print(f"[{el()}] validazione: {len(VAL)} query su una GALLERIA DISGIUNTA "
          f"({len(sub_vg)} immagini mai viste in addestramento), composizione "
          f"{100 * _cv[1] / _tv:.0f}/{100 * _cv[2] / _tv:.0f}/{100 * _cv[3] / _tv:.0f}% "
          f"(benchmark ufficiale: 54/31/15)")

    torch.save({"pool": POOL, "val": VAL, "sub": sub, "E_tr": E_tr_cpu, "L_tr": L_tr,
                "qtxt": qtxt, "sub_vg": sub_vg, "E_vg": E_vg_cpu, "L_vg": L_vg,
                "versione": 3,
                "E_fit": E_fit_cpu, "L_fit": L_fit_cpu, "d_ens": d_ens_cpu,
                "attr": ATTR, "gemm_ok": GEMM_OK, "ms_max": MS_MAX,
                "seed": SEED_DATI, "pool_targets": POOL_TARGETS},
               CACHE_DATI)
    QTXT_POOL = qtxt
    print(f"[{el()}] cache scritta: {CACHE_DATI}  "
          f"(a una ripresa la fase 1 si salta e il file da 385 MB non serve piu')")
    del E_tr_cpu, L_tr_f

RISULTATI["fasi"]["f1_dati"] = {
    "n_query_pool": len(POOL), "n_val": len(VAL), "gemm_ok": bool(GEMM_OK),
    "ms_pool": MS_MAX,
    "esempi_per_livello": {str(m): esempi_totali(POOL, m)
                           for m in sorted(set(GRIGLIA_B_MS) | {MS_MAX})},
    # query il cui bacino di reference si esaurisce: li' l'offset dei membri non ha nulla da
    # spostare e tutti vedono le stesse reference
    "bacini_esauriti_per_livello": {str(m): bacini_esauriti(POOL, m)
                                    for m in sorted(set(GRIGLIA_B_MS) | {MS_MAX})},
    "composizione_pool": {str(k): sum(1 for e in POOL if e["n_cond"] == k)
                          for k in (1, 2, 3)},
    "composizione_val": {str(k): sum(1 for e in VAL if e["n_cond"] == k) for k in (1, 2, 3)},
}
salva()
print(f"[{el()}] esempi per livello: {RISULTATI['fasi']['f1_dati']['esempi_per_livello']}")

# la validazione e' gia' in forma ragged (l'ha costruita `costruisci`): basta spostarla sul
# device e aggiungere il vettore delle condizioni
VAL_C = [{"query": e["query"], "pos": e["pos"], "neg": e["neg"], "n_cond": e["n_cond"],
          "sources": e["sources"].to(DEV), "gt_flat": e["gt_flat"].long().to(DEV),
          "gt_ptr": e["gt_ptr"].to(DEV), "gt_len": e["gt_len"].to(DEV),
          "c": (sum((d_ens_cpu[a] for a in e["pos"]), torch.zeros(D))
                - sum((d_ens_cpu[b] for b in e["neg"]), torch.zeros(D))).to(DEV)}
         for e in VAL]
n_cond_dev = np.array([e["n_cond"] for e in devset])

# vettore delle condizioni per ogni query del pool: c = somma con segno delle direzioni
C_POOL = torch.stack([(sum((d_ens_cpu[a] for a in e["pos"]), torch.zeros(D))
                       - sum((d_ens_cpu[b] for b in e["neg"]), torch.zeros(D)))
                      for e in POOL]).to(DEV)


# ===========================================================================================
# CONTESTI PER MEMBRO — il terzo asse di diversita' del bagging, perso nella v1
# ===========================================================================================
# COSA E' SUCCESSO IL 29/08. Nella sessione notturna (`esperimenti/notte/notte.py`) ogni membro
# aveva un seed proprio, e `prepara(seed)` estraeva un CORPUS diverso dalle 162770 immagini di
# train. Nella v1 di questo script il corpus e' costruito una volta sola in F1 e condiviso da
# tutti i membri, che quindi differiscono solo per il 75% di query e per la finestra di
# reference. Misurato: il guadagno del bagging e' passato da +5.16pp a +3.05pp, e i singoli
# nel frattempo erano MIGLIORATI di 2.2pp. E' la decomposizione di ambiguita': membri migliori
# e piu' simili, meno da guadagnare mediando.
#
# Qui l'asse si recupera, e costa ZERO addestramenti in piu': solo la ricostruzione del pool
# per ogni membro (un GEMM, secondi su GPU).
#
# COSA RESTA CONDIVISO, E PERCHE'. La RICERCA (F2) continua a girare sul contesto condiviso:
# li' i livelli devono restare ANNIDATI sulle stesse query e sullo stesso prefisso di
# reference, altrimenti il confronto fra livelli torna a non essere accoppiato — che e'
# esattamente l'esperimento che il gruppo non riusciva a fare. Il corpus per membro riguarda
# solo FM e F3, dove non si confrontano livelli.
QTXT_VAL = [e["query"] for e in VAL]
CTX_CONDIVISO = {"POOL": POOL, "E_tr": E_tr, "L_tr": L_tr, "VAL_C": VAL_C, "C_POOL": C_POOL,
                 "sub": sub, "E_VG": E_VG}
CTX_ATTIVO = SEED_DATI
_ctx_cache = {SEED_DATI: CTX_CONDIVISO}
CORPUS_PER_MEMBRO = (not args.corpus_condiviso) and GEMM_OK and QTXT_POOL is not None
if not CORPUS_PER_MEMBRO:
    print(f"[{el()}] *** corpus CONDIVISO fra i membri: "
          + ("richiesto da --corpus-condiviso" if args.corpus_condiviso
             else "percorso GEMM non disponibile" if not GEMM_OK
             else "le query del pool non sono nella cache")
          + ". Il bagging perdera' il terzo asse di diversita'. ***")


def _contesto(seed):
    """Corpus, pool e validazione ricampionati per un membro dell'ensemble."""
    if seed in _ctx_cache:
        return _ctx_cache[seed]
    t = time.time()
    g = torch.Generator().manual_seed(seed)
    pm = torch.randperm(len(E_train_all), generator=g)
    sb, sb_vg = pm[:N_TRAIN_CORPUS].sort().values, pm[N_TRAIN_CORPUS:2 * N_TRAIN_CORPUS].sort().values
    e_tr_cpu, l_tr = E_train_all[sb], L_train_all[sb]
    e_tr = e_tr_cpu.to(DEV)
    l_tr_f = l_tr.float().to(DEV)
    # R1 anche qui: la validazione del membro vive su una galleria che il membro non vede
    e_vg = E_train_all[sb_vg].to(DEV)
    l_vg_f = L_train_all[sb_vg].float().to(DEV)
    pool = costruisci(QTXT_POOL, l_tr_f, ms_max=MS_MAX, seed=seed)
    val = costruisci(QTXT_VAL, l_vg_f, ms_max=VAL_MS, seed=seed)[:N_VAL_QUERY]
    val_c = [{"query": e["query"], "pos": e["pos"], "neg": e["neg"], "n_cond": e["n_cond"],
              "sources": e["sources"].to(DEV), "gt_flat": e["gt_flat"].long().to(DEV),
              "gt_ptr": e["gt_ptr"].to(DEV), "gt_len": e["gt_len"].to(DEV),
              "c": (sum((d_ens_cpu[a] for a in e["pos"]), torch.zeros(D))
                    - sum((d_ens_cpu[b] for b in e["neg"]), torch.zeros(D))).to(DEV)}
             for e in val]
    c_pool = torch.stack([(sum((d_ens_cpu[a] for a in e["pos"]), torch.zeros(D))
                           - sum((d_ens_cpu[b] for b in e["neg"]), torch.zeros(D)))
                          for e in pool]).to(DEV)
    ctx = {"POOL": pool, "E_tr": e_tr, "L_tr": l_tr, "VAL_C": val_c, "C_POOL": c_pool,
           "sub": sb, "E_VG": e_vg}
    _ctx_cache[seed] = ctx
    cond = len(set(sb.tolist()) & set(CTX_CONDIVISO["sub"].tolist())) / N_TRAIN_CORPUS
    print(f"  contesto del membro {seed}: {len(pool)} query valide, {len(val)} di "
          f"validazione, {100 * cond:.1f}% del corpus in comune col condiviso "
          f"({time.time() - t:.0f}s)", flush=True)
    return ctx


def usa_contesto(seed):
    """Rende attivo il contesto di un seed. Le funzioni leggono i globali, quindi basta
    ricollegarli: nessuna firma cambia e il codice della v1 continua a funzionare."""
    global POOL, E_tr, L_tr, VAL_C, C_POOL, E_VG, CTX_ATTIVO
    if seed == CTX_ATTIVO:
        return
    ctx = _contesto(seed)
    POOL, E_tr, L_tr, VAL_C, C_POOL, E_VG = (ctx["POOL"], ctx["E_tr"], ctx["L_tr"],
                                             ctx["VAL_C"], ctx["C_POOL"], ctx["E_VG"])
    CTX_ATTIVO = seed


def sovrapposizione_corpus(seeds):
    """Jaccard fra i corpus di due membri. Con corpus per membro e' il terzo asse, e con
    20000 immagini su 162770 due estrazioni indipendenti condividono circa il 6.5%."""
    subs = [set(_contesto(q)["sub"].tolist()) for q in seeds]
    v = [len(a & b) / len(a | b) for i, a in enumerate(subs) for b in subs[i + 1:]]
    return float(np.mean(v)) if v else None


# ===========================================================================================
# I DATI DI UN ADDESTRAMENTO: vista a un livello + campione di query
# ===========================================================================================
def seleziona(ms, qseed, offset, frac=BAG_FRAC):
    """Quali query del pool, e quali reference dentro ognuna, vede un modello.

    Ritorna una lista di coppie `(j, m)`: `j` indice della query nel pool, `m` indici locali
    delle reference tenute. E' l'unico posto in cui si decide cosa vede un modello, cosi'
    `Dati` e `chiavi_esempi` non possono divergere.

    DUE ASSI DI DIVERSITA', ed e' il punto delicato di tutto lo script.

    1. LE QUERY. Ogni modello vede una frazione `frac` del pool, scelta da `qseed`.
    2. LE REFERENCE. Le reference di ogni query sono ordinate da una permutazione fissata una
       volta sola (il "rango", vedi `costruisci`). Un modello prende la FINESTRA CIRCOLARE
       `[off, off + ms)` modulo il numero di reference disponibili per quella query. Una
       finestra circolare su una permutazione uniforme e' ancora un sottoinsieme uniforme di
       taglia `ms`, quindi non cambia nulla di statistico — ma due offset diversi danno due
       insiemi di reference che si sovrappongono poco.

    `offset=False` forza `off = 0`, cioe' la finestra `rango < ms`: e' il PREFISSO, che rende
    i livelli ANNIDATI (350 sottoinsieme di 700 sottoinsieme di 1400, sulle stesse query).
    Serve alla ricerca (F2): e' cio' che rende accoppiato il confronto fra livelli, che e'
    l'esperimento che il gruppo non era riuscito a fare. NON si tocca.

    `offset=True` da' a ogni membro dell'ensemble un `off` diverso, derivato dal suo `qseed`.
    Serve al bagging (FM e F3): senza, al livello 1400 con un pool a 1400 tutti i membri
    prenderebbero le stesse identiche reference di ogni query in comune, e resterebbe un solo
    asse di diversita'. Il +5.16pp misurato in esperimenti/notte/ veniva da entrambi gli assi.

    Dove il bacino di una query e' piu' piccolo di `ms`, la finestra copre tutto e ogni
    membro prende tutto: e' inevitabile, e `bacini_esauriti` lo conta.
    """
    n = len(POOL)
    if frac >= 1.0:
        scelte = torch.arange(n)
    else:
        g = torch.Generator().manual_seed(qseed)
        scelte = torch.randperm(n, generator=g)[:max(1, int(round(frac * n)))].sort().values
    if offset:
        go = torch.Generator().manual_seed(qseed + 500003)
        offs = torch.randint(0, 2 ** 30, (n,), generator=go)
    fuori = []
    for j in scelte.tolist():
        e = POOL[j]
        if offset and e["n_rango"] > ms:
            off = int(offs[j]) % e["n_rango"]
            # posizione dentro la finestra circolare che parte da `off`
            rel = (e["rango"] - off) % e["n_rango"]
            m = (rel < ms).nonzero(as_tuple=True)[0]
        else:
            m = (e["rango"] < ms).nonzero(as_tuple=True)[0]
        if len(m):
            fuori.append((j, m))
    return fuori


def chiavi_esempi(ms, qseed, offset):
    """Gli esempi di un modello come interi univoci `query * N + reference`.

    Serve solo a misurare quanto si sovrappongono i campioni di due membri: e' l'unico modo
    di scrivere nel report DA DOVE viene la diversita' dell'ensemble con un numero accanto,
    invece di dire "bagging" e sperare.
    """
    return np.concatenate([(j * N_TRAIN_CORPUS + POOL[j]["sources"][m].numpy())
                           for j, m in seleziona(ms, qseed, offset)])


_memo_chiavi = {}


def sovrapposizione(ms, qseed_a, qseed_b, offset):
    """Jaccard fra gli insiemi di esempi di due membri."""
    for q in (qseed_a, qseed_b):
        if (ms, q, offset) not in _memo_chiavi:
            _memo_chiavi[(ms, q, offset)] = set(chiavi_esempi(ms, q, offset).tolist())
    a, b = _memo_chiavi[(ms, qseed_a, offset)], _memo_chiavi[(ms, qseed_b, offset)]
    return len(a & b) / max(len(a | b), 1)


def diversita(ms, qseeds):
    """Da dove viene la diversita' dell'ensemble, con un numero accanto.

    Riporta la sovrapposizione media a coppie (Jaccard sugli insiemi di esempi) fra i membri
    veri (con offset) e quella che ci sarebbe stata senza offset, cioe' con la sola diversita'
    di query. La differenza fra i due numeri e' la quota che l'offset delle reference aggiunge:
    senza questa misura l'affermazione "la diversita' viene dai dati" resta un'asserzione.
    """
    coppie = [(a, b) for i, a in enumerate(qseeds) for b in qseeds[i + 1:]]
    if not coppie:
        return None
    if CORPUS_PER_MEMBRO:
        # con corpus diversi gli indici delle reference non sono confrontabili fra membri:
        # l'unica sovrapposizione ben definita e' quella dei corpus stessi
        jc = sovrapposizione_corpus(qseeds)
        d = {"n_membri": len(qseeds), "n_coppie": len(coppie), "ms": ms,
             "corpus_per_membro": True, "jaccard_corpus": jc,
             "frazione_query_condivise": BAG_FRAC ** 2 / (2 * BAG_FRAC - BAG_FRAC ** 2),
             "n_query_pool": len(POOL)}
        print(f"  diversita' ({d['n_coppie']} coppie, ms {ms}): i membri condividono il "
              f"{100 * jc:.1f}% del corpus e circa il "
              f"{100 * d['frazione_query_condivise']:.0f}% delle query, piu' finestre di "
              f"reference diverse. Tre assi.")
        return d
    con = [sovrapposizione(ms, a, b, True) for a, b in coppie]
    senza = [sovrapposizione(ms, a, b, False) for a, b in coppie]
    d = {"n_membri": len(qseeds), "n_coppie": len(coppie), "ms": ms,
         "jaccard_medio": float(np.mean(con)), "jaccard_min": float(np.min(con)),
         "jaccard_max": float(np.max(con)),
         "jaccard_medio_senza_offset": float(np.mean(senza)),
         "bacini_esauriti": bacini_esauriti(POOL, ms), "n_query_pool": len(POOL)}
    print(f"  diversita' dei campioni ({d['n_coppie']} coppie, ms {ms}): Jaccard medio "
          f"{d['jaccard_medio']:.3f} [{d['jaccard_min']:.3f}, {d['jaccard_max']:.3f}]")
    print(f"    senza l'offset sulle reference sarebbe {d['jaccard_medio_senza_offset']:.3f} "
          f"— la differenza e' quello che l'offset aggiunge")
    print(f"    bacino esaurito su {d['bacini_esauriti']}/{d['n_query_pool']} query: li' "
          f"tutti i membri vedono le stesse reference, ed e' inevitabile")
    return d


class Dati:
    """Gli esempi (reference, condizioni, target) di un modello, gia' sul device.

    Cosa vede il modello lo decide `seleziona`; qui si impacchetta soltanto.
    """

    def __init__(self, ms, qseed, offset, frac=BAG_FRAC):
        scelte_e_maschere = seleziona(ms, qseed, offset, frac)
        refs, qidx, flats, lens = [], [], [], []
        for j, m in scelte_e_maschere:
            e = POOL[j]
            refs.append(e["sources"][m])
            qidx.append(torch.full((len(m),), j, dtype=torch.long))
            l = e["gt_len"][m]
            lens.append(l)
            # concatena le slice di gt delle sole reference tenute
            p = e["gt_ptr"][m]
            tot = int(l.sum())
            inizio = torch.cumsum(l, 0) - l
            entro = torch.arange(tot) - torch.repeat_interleave(inizio, l)
            flats.append(e["gt_flat"][torch.repeat_interleave(p, l) + entro])
        if not refs:
            raise SystemExit(f"nessun esempio a max_sources={ms}: il pool e' vuoto a "
                             f"quel livello, controlla la fase 1")
        self.refs = torch.cat(refs).to(DEV)
        self.qidx = torch.cat(qidx).to(DEV)
        self.gt_flat = torch.cat(flats).long().to(DEV)
        self.gt_len = torch.cat(lens).to(DEV)
        self.gt_ptr = (torch.cumsum(self.gt_len, 0) - self.gt_len)
        self.n_query = len(scelte_e_maschere)
        self.query_usate = [j for j, _ in scelte_e_maschere]

    def __len__(self):
        return len(self.refs)


# ===========================================================================================
# MODELLO E LOSS — `Concat` di esperimenti/notte/notte.py
# ===========================================================================================
class Concat(nn.Module):
    """q = normalize(r + gamma * MLP([r; c])).

    Unica differenza rispetto a notte.py: `forward` riceve gia' `c` invece di (U, M).
    E' la stessa quantita' — li' `c = (U*M).sum(1)`, cioe' la somma con segno delle
    direzioni — ma precalcolata una volta per query invece che per esempio: al livello 1400
    il tensore U sarebbe [70000, 3, 512], cioe' 400 MB di memoria del device per non
    calcolare una somma di tre vettori. Le chiavi dello state_dict restano identiche
    (`net.0`, `net.2`, `gamma`), quindi i checkpoint sono compatibili con
    `contributo-candidato/bagging_lineare.py` e con gli altri script della repo.
    """

    def __init__(self, d=None, hidden=HIDDEN):
        super().__init__()
        d = d or D
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        # ultimo strato a zero: al passo 0 il modello E' la baseline, quindi ogni guadagno
        # e' guadagno rispetto a essa e non un artefatto dell'inizializzazione
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, c):
        q = r + self.gamma * self.net(torch.cat([r, c], dim=-1))
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def make_builder(models):
    """Un modello, oppure una lista: in quel caso si mediano le CORREZIONI.

    Si mediano le uscite e non i pesi: reti da inizializzazioni diverse stanno in bacini
    diversi. E' la scelta di `esperimenti/notte/notte.py` e di `esperimenti/ensemble_seed.py`.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    def f(ref, entry):
        c = entry["c"].expand(len(ref), -1)
        with torch.no_grad():
            dl = torch.stack([m.gamma * m.net(torch.cat([ref, c], dim=-1))
                              for m in models]).mean(0)
        q = ref + dl
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


def info_nce(model, dati, idx, tau):
    """InfoNCE multi-positivo: denominatore su TUTTO il corpus di addestramento, reference
    esclusa. La maschera dei positivi si costruisce con un solo `index_put_` vettorizzato
    (vedi `ragged`) invece del ciclo Python su 256 elementi per batch che c'era prima: su
    GPU quel ciclo costava piu' del prodotto matriciale che preparava.
    """
    refs = dati.refs[idx]
    c = C_POOL[dati.qidx[idx]]
    q = model(E_tr[refs], c)
    sim = ((q @ E_tr.T) / tau).clone()   # clone: si scrive in-place e autograd deve stare quieto
    B, N = sim.shape
    ar = torch.arange(B, device=DEV)
    sim[ar, refs] = -float("inf")
    pos = torch.zeros(B, N, dtype=torch.bool, device=DEV)
    righe, col = ragged(dati.gt_flat, dati.gt_ptr, dati.gt_len, idx)
    pos[righe, col] = True
    pos[ar, refs] = False
    loss = -(torch.logsumexp(sim.masked_fill(~pos, -float("inf")), 1)
             - torch.logsumexp(sim, 1)).mean()
    caso = float(torch.log(N / dati.gt_len[idx].clamp(min=1).float()).mean())
    return loss, caso


# ===========================================================================================
# ADDESTRAMENTO con arresto onesto
# ===========================================================================================
def tag_di(ms, tau, wd, qseed, offset, hidden=None):
    hidden = HIDDEN if hidden is None else hidden
    # il suffisso non e' decorativo: distingue un membro dell'ensemble (finestra di reference
    # spostata) da una cella della ricerca (prefisso, livelli annidati) anche a parita' di
    # tutto il resto, cosi' la ripresa da disco non puo' scambiarli
    t = f"ms{ms}_tau{tau:g}_wd{wd:g}_h{hidden}_q{qseed}"
    # il contesto entra nel tag: due membri con lo stesso qseed ma corpus condiviso o
    # per membro sono modelli DIVERSI, e la ripresa da disco non deve scambiarli
    return t + ("_off" if offset else "") + ("_cx" if offset and CORPUS_PER_MEMBRO else "")


def impacchetta(ps, prefisso):
    """{k: [array per query]} -> dizionario piatto salvabile in .npz."""
    out = {}
    for k, arrs in ps.items():
        out[f"{prefisso}_k{k}_flat"] = np.concatenate(arrs) if arrs else np.zeros(0)
        out[f"{prefisso}_k{k}_len"] = np.array([len(a) for a in arrs])
    return out


def spacchetta(z, prefisso, k_values=K_VALUES):
    ps = {}
    for k in k_values:
        flat = z[f"{prefisso}_k{k}_flat"]
        lens = z[f"{prefisso}_k{k}_len"]
        off = np.concatenate([[0], np.cumsum(lens)])
        ps[k] = [flat[off[i]:off[i + 1]] for i in range(len(lens))]
    return ps


def addestra(ms, tau, wd, qseed, etichetta, offset, hidden=None):
    """Un modello, fermato sulla validazione onesta.

    ARRESTO. Si valuta sulla validazione ogni 2 epoche; dopo 3 valutazioni senza
    miglioramento ci si ferma; il massimo e' 30 epoche. Vengono salvati lo state_dict E gli
    esiti per singola source ALL'EPOCA MIGLIORE, non all'ultima: e' il pezzo che mancava e
    che impediva ogni test di significativita' fra livelli (`esperimenti/notte/README.md`,
    "Nessun test di significativita' e' disponibile sulle due colonne di destra").

    La scelta dell'epoca guarda SOLO la validazione. Il set di sviluppo viene misurato una
    volta sola, alla fine, sul modello gia' scelto.
    """
    hidden = HIDDEN if hidden is None else hidden
    # I MEMBRI DELL'ENSEMBLE (offset=True) girano ognuno sul proprio corpus; la ricerca
    # (offset=False) resta sul condiviso, perche' li' i livelli devono restare annidati.
    usa_contesto(qseed if (offset and CORPUS_PER_MEMBRO) else SEED_DATI)
    tag = tag_di(ms, tau, wd, qseed, offset, hidden)
    ck = RUN / "ckpt" / f"{tag}.pt"
    npz = RUN / "esiti" / f"{tag}.npz"
    if tag in RISULTATI["run"] and ck.exists() and npz.exists():
        d = torch.load(ck, map_location=DEV, weights_only=False)
        m = Concat(D, d["config"]["hidden"]).to(DEV)
        m.load_state_dict(d["A"])
        m.eval()
        z = np.load(npz)
        print(f"  [ripreso] {tag}: val {RISULTATI['run'][tag]['val_r10']:.2f}%  "
              f"dev {RISULTATI['run'][tag]['dev_r10']:.2f}%")
        return m, spacchetta(z, "dev"), spacchetta(z, "val", [10]), RISULTATI["run"][tag]

    dati = Dati(ms, qseed, offset)
    n_batch = max(1, (len(dati) + BATCH - 1) // BATCH)
    st = stima_minuti(len(dati), EPOCHE_MAX)
    print(f"\n--- {etichetta} [{tag}] — {dati.n_query} query, {len(dati)} esempi, "
          f"{n_batch} batch/epoca, <= {EPOCHE_MAX} epoche, stima ~{st:.1f}m [{el()}]",
          flush=True)

    torch.manual_seed(qseed)
    model = Concat(D, hidden).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    gen = torch.Generator().manual_seed(qseed)
    t_ini = time.time()

    migliore = {"val": -1.0, "epoca": 0, "stato": None, "ps_val": None}
    senza = 0
    storia = []
    for ep in range(1, EPOCHE_MAX + 1):
        model.train()
        ordine = torch.randperm(len(dati), generator=gen).to(DEV)
        tl = tc = 0.0
        _tep = _tbat = time.time()
        for _ib, s in enumerate(range(0, len(dati), BATCH)):
            loss, caso = info_nce(model, dati, ordine[s:s + BATCH], tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss.detach())
            tc += caso
            # battito solo se l'epoca e' lenta: su GPU non stampa mai, su CPU evita il silenzio
            if time.time() - _tbat > 45:
                _tbat = time.time()
                print(f"    epoca {ep}: batch {_ib + 1}/{n_batch} "
                      f"({100 * (_ib + 1) / n_batch:.0f}%), loss corrente "
                      f"{tl / (_ib + 1):6.3f}, {time.time() - _tep:.0f}s", flush=True)
        riga = {"epoca": ep, "loss": tl / n_batch, "caso": tc / n_batch,
                "gamma": float(model.gamma.detach())}
        coda = ""
        if ep % VAL_OGNI == 0 or ep == EPOCHE_MAX:
            model.eval()
            # LA VALIDAZIONE SI VALUTA SULLA GALLERIA DISGIUNTA, non su E_tr (R1)
            ps_val = valuta(make_builder(model), VAL_C, E_VG, [10])
            rv = macro(ps_val[10]) * 100
            riga["r10_val"] = rv
            if rv > migliore["val"]:
                migliore = {"val": rv, "epoca": ep,
                            "stato": copy.deepcopy(model.state_dict()), "ps_val": ps_val}
                senza = 0
                coda = f"  |  val {rv:5.2f}%  <- migliore"
            else:
                senza += 1
                coda = f"  |  val {rv:5.2f}%  ({senza}/{PAZIENZA})"
        storia.append(riga)
        print(f"  epoca {ep:>3}/{EPOCHE_MAX}  loss {riga['loss']:6.3f}  "
              f"caso {riga['caso']:6.3f}  divario {riga['caso'] - riga['loss']:6.3f}  "
              f"gamma {riga['gamma']:6.3f}{coda}  [{el()}]", flush=True)
        if senza >= PAZIENZA:
            print(f"  arresto anticipato all'epoca {ep}: {PAZIENZA} valutazioni senza "
                  f"miglioramento (migliore: epoca {migliore['epoca']})")
            break

    minuti = (time.time() - t_ini) / 60
    calibra(len(dati), storia[-1]["epoca"], minuti)

    # si torna all'epoca scelta dalla validazione, e SOLO ORA si guarda il set di sviluppo
    model.load_state_dict(migliore["stato"])
    model.eval()
    ps_dev = valuta(make_builder(model), devset, E_valid)
    r10 = macro(ps_dev[10]) * 100

    strati = []
    for k in (1, 2, 3):
        ix = np.where(n_cond_dev == k)[0]
        if len(ix) == 0:
            strati.append(float("nan"))
            continue
        a = float(np.mean([ps_dev[10][i].mean() for i in ix])) * 100
        b = float(np.mean([ps_base[10][i].mean() for i in ix])) * 100
        strati.append(a - b)

    out = {"tag": tag, "ms": ms, "tau": tau, "wd": wd, "hidden": hidden,
           "contesto": (qseed if (offset and CORPUS_PER_MEMBRO) else SEED_DATI),
           "qseed": qseed, "offset": offset,
           "etichetta": etichetta, "n_query": dati.n_query, "n_esempi": len(dati),
           "epoca_scelta": migliore["epoca"], "epoche_fatte": storia[-1]["epoca"],
           "val_r10": migliore["val"], "dev_r10": r10, "vs_baseline": r10 - B10,
           "strati_vs_baseline": strati, "minuti": minuti,
           "gamma": float(model.gamma.detach()), "storia": storia}
    torch.save({"A": model.state_dict(), "config": {"hidden": hidden, "d": D,
                                                    "n_attr": N_ATTR},
                "ms": ms, "tau": tau, "wd": wd, "qseed": qseed, "offset": offset,
                "epoca_scelta": migliore["epoca"], "dirs_text": d_ens_cpu,
                "attr_names": list(ATTR), "dev_r10": r10, "val_r10": migliore["val"]},
               RUN / "ckpt" / f"{tag}.pt")
    np.savez_compressed(RUN / "esiti" / f"{tag}.npz",
                        **impacchetta(ps_dev, "dev"), **impacchetta(migliore["ps_val"], "val"))
    RISULTATI["run"][tag] = out
    salva()
    print(f"  -> {tag}: epoca {migliore['epoca']}, val {migliore['val']:.2f}%, "
          f"dev {r10:.2f}% ({r10 - B10:+.2f}pp), {minuti:.1f}m", flush=True)
    return model, ps_dev, migliore["ps_val"], out


def confronto(a, b, seed=0):
    """Differenza accoppiata + intervallo di confidenza + p. Mai due medie affiancate."""
    s = bootstrap_paired(a, b, n_boot=N_BOOT, seed=seed)
    return {"delta_pp": s["delta"] * 100, "lo": s["lo"] * 100, "hi": s["hi"] * 100,
            "p": s["p"], "query_vinte": s["n_query_win"], "query_perse": s["n_query_loss"]}


def per_seed(c):
    return ", ".join(f"{v:.2f}" for v in c["val_seed"])


def riga_confronto(nome, s):
    return (f"  {nome:<44}{s['delta_pp']:>+8.2f}pp  "
            f"[{s['lo']:+.2f}, {s['hi']:+.2f}]  p = {s['p']:.4f}  "
            f"({s['query_vinte']}V/{s['query_perse']}P)")


def curva_ensemble(modelli, nome_fase):
    """Rendimenti dell'ensemble per k = 1..N, mediando le USCITE."""
    fuori = {}
    ps_prec = None
    for k in range(1, len(modelli) + 1):
        ps = valuta(make_builder(modelli[:k]), devset, E_valid)
        r = macro(ps[10]) * 100
        voce = {"k": k, "dev_r10": r, "vs_baseline": confronto(ps[10], ps_base[10])}
        if ps_prec is not None:
            voce["vs_k_precedente"] = confronto(ps[10], ps_prec[10])
        fuori[str(k)] = voce
        print(f"  ensemble k={k}: dev {r:5.2f}%  ({r - B10:+.2f}pp sulla baseline)"
              + (f"   vs k={k-1}: {voce['vs_k_precedente']['delta_pp']:+.2f}pp "
                 f"p={voce['vs_k_precedente']['p']:.3f}" if ps_prec is not None else ""))
        ps_prec = ps
        RISULTATI["fasi"][nome_fase] = {**RISULTATI["fasi"].get(nome_fase, {}), "curva": fuori}
        salva()
    return fuori, ps_prec


# ===========================================================================================
# FM — MINIMO GARANTITO
# ===========================================================================================
print("\n" + "=" * 100)
print(f"FM — MINIMO GARANTITO: {N_MEMBRI_MINIMO} membri alla configurazione incumbente "
      f"(ms {INCUMBENTE['ms']}, tau {INCUMBENTE['tau']})")
print("=" * 100)
print("Serve a garantire che entro i primi minuti esista gia' una consegna completa:")
print("ensemble, matrice collassata, esiti per source, tabelle. Il resto la migliora.")

# FM e' l'unica fase che NON si salta per mancanza di tempo: e' il pavimento della consegna.
# Se il budget non basta si avvisa, ma si prova lo stesso — meglio un ensemble a due membri
# che una sessione senza risultati.
_st_fm = stima_minuti(esempi_totali(POOL, INCUMBENTE["ms"]) * BAG_FRAC, EPOCHE_MAX)
if tempo_rimasto() < _st_fm * 60 * N_MEMBRI_MINIMO:
    print(f"  ATTENZIONE: stimati ~{_st_fm * N_MEMBRI_MINIMO:.0f}m per FM e ne restano "
          f"{tempo_rimasto() / 60:.0f}m. Procedo comunque: FM e' il minimo garantito.")

MODELLI_M, PS_M, INFO_M = [], [], []
for qs in QSEED_FINALI[:N_MEMBRI_MINIMO]:
    # offset=True: i membri devono differire anche sulle reference, non solo sulle query
    m, psd, psv, info = addestra(INCUMBENTE["ms"], INCUMBENTE["tau"], INCUMBENTE["wd"], qs,
                                 f"minimo garantito, campione {qs}", True,
                                 INCUMBENTE["hidden"])
    MODELLI_M.append(m)
    PS_M.append(psd)
    INFO_M.append(info)

RISULTATI["fasi"]["fm_minimo"] = {"config": dict(INCUMBENTE),
                                  "membri": [i["tag"] for i in INFO_M],
                                  "media_singoli": float(np.mean([i["dev_r10"]
                                                                  for i in INFO_M])),
                                  "diversita": diversita(INCUMBENTE["ms"],
                                                         [i["qseed"] for i in INFO_M])}
salva()
if len(MODELLI_M) >= 2:
    print(f"\n  media dei {len(MODELLI_M)} singoli: "
          f"{RISULTATI['fasi']['fm_minimo']['media_singoli']:.2f}%")
    _, PS_ENS_M = curva_ensemble(MODELLI_M, "fm_minimo")

# ===========================================================================================
# F2 — I DUE CANDIDATI, E IL COLLAUDO DELLA RIPARAZIONE
# ===========================================================================================
# NIENTE RICERCA A COORDINATE, ed e' una scelta motivata. In v1 aveva scelto bene (tau 0.01),
# in v2 ha scelto 1.5pp PEGGIO del meglio che aveva gia' provato, perche' leggeva uno
# strumento rotto. Cio' che paga in modo replicato in tutte e tre le sessioni e' la DIVERSITA'
# DEI MEMBRI, non la taratura. Quindi qui si sceglie fra due soli punti, entrambi gia'
# misurati, mai confrontati ad armi pari, e con lo strumento riparato da R1 e R2.
#
# Il candidato A coincide con l'incumbente, quindi i suoi due run sono gia' stati fatti da FM
# e vengono ripresi da disco: il costo reale di questa fase e' due addestramenti.
print("")
print("=" * 100)
print("F2 — I DUE CANDIDATI")
print("=" * 100)
for _n, _c in CANDIDATI.items():
    print(f"  {_n}: tau {_c['tau']:g}, max_sources {_c['ms']}, wd {_c['wd']:g}, "
          f"hidden {_c['hidden']}")
print(f"{N_SEED_CELLA} membri per candidato, corpus per membro, scelta SOLO sulla validazione.")
print("Il dev si stampa per trasparenza e non si usa: e' la disciplina del progetto.")

RIS_CAND, PS_CAND = {}, {}
for nome, cfg in CANDIDATI.items():
    vals, devs, tags = [], [], []
    for qs in QSEED_FINALI[:N_SEED_CELLA]:
        st = stima_minuti(esempi_totali(POOL, cfg["ms"]) * BAG_FRAC, EPOCHE_MAX)
        if (tag_di(cfg["ms"], cfg["tau"], cfg["wd"], qs, True, cfg["hidden"])
                not in RISULTATI["run"] and tempo_rimasto() < st * 60 * 1.4):
            print(f"  SALTO il candidato {nome} seed {qs}: stimati ~{st:.0f}m, restano "
                  f"{tempo_rimasto() / 60:.0f}m")
            continue
        _, psd, _psv, info = addestra(cfg["ms"], cfg["tau"], cfg["wd"], qs,
                                      f"candidato {nome} (seed {qs})", True, cfg["hidden"])
        vals.append(info["val_r10"])
        devs.append(info["dev_r10"])
        tags.append(info["tag"])
        PS_CAND.setdefault(nome, []).append(psd)
    if not vals:
        continue
    RIS_CAND[nome] = {"config": dict(cfg), "val_media": float(np.mean(vals)),
                      "val_seed": vals, "dev_media": float(np.mean(devs)),
                      "dev_seed": devs, "tags": tags}
    print(f"  -> candidato {nome}: val {np.mean(vals):.2f}% {vals}   "
          f"dev {np.mean(devs):.2f}% {devs}", flush=True)
    salva()

# --- il collaudo della riparazione, con la regola dichiarata PRIMA di guardare i numeri -----
# La v2 e' fallita perche' la validazione e il dev ordinavano al contrario. Qui si verifica
# la cosa che conta davvero: i due strumenti scelgono lo STESSO candidato? E' un test molto
# piu' interpretabile di una correlazione su pochi punti, perche' e' esattamente la decisione
# che lo strumento deve prendere.
#
# REGOLA PRE-DICHIARATA: se sono d'accordo, si adotta il vincitore della validazione. Se NON
# sono d'accordo, la riparazione non ha funzionato e si consegna la configurazione della v2
# (il candidato A), che e' comunque gia' misurata con intervalli di confidenza.
SPEAR = None
if len(RIS_CAND) >= 2:
    vinc_val = max(RIS_CAND, key=lambda k: RIS_CAND[k]["val_media"])
    vinc_dev_c = max(RIS_CAND, key=lambda k: RIS_CAND[k]["dev_media"])
    accordo = vinc_val == vinc_dev_c
    tutti = [(v["val_r10"], v["dev_r10"]) for v in RISULTATI["run"].values()]
    if len(tutti) >= 4:
        a = np.array([t[0] for t in tutti]); b = np.array([t[1] for t in tutti])
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        SPEAR = float(np.corrcoef(ra, rb)[0, 1])
    print("")
    print("-" * 100)
    print("COLLAUDO DELLA RIPARAZIONE (R1: galleria disgiunta, R2: composizione 54/31/15)")
    print("-" * 100)
    for n, v in RIS_CAND.items():
        print(f"  {n}: validazione {v['val_media']:6.2f}%   dev {v['dev_media']:6.2f}%")
    print(f"  vincitore secondo la VALIDAZIONE: {vinc_val}")
    print(f"  vincitore secondo il DEV        : {vinc_dev_c}")
    if SPEAR is not None:
        print(f"  Spearman validazione-dev su {len(tutti)} run: {SPEAR:+.3f}   "
              f"(v1 +0.897, v2 +0.244; indicativo con pochi punti)")
    if accordo:
        PUNTO = dict(RIS_CAND[vinc_val]["config"])
        print(f"  -> D'ACCORDO. La riparazione ha funzionato: si adotta {vinc_val}.")
    else:
        PUNTO = dict(CANDIDATI["A"])
        print("  -> IN DISACCORDO. La riparazione NON ha funzionato: per la regola dichiarata")
        print("     prima della sessione si consegna la configurazione della v2 (candidato A),")
        print("     che e' gia' misurata con intervalli. NON si sceglie guardando il dev.")
    RISULTATI["fasi"]["f2_candidati"] = {
        "candidati": RIS_CAND, "vincitore_validazione": vinc_val,
        "vincitore_dev_non_usato": vinc_dev_c, "accordo": bool(accordo),
        "spearman_val_dev": SPEAR, "punto_scelto": PUNTO,
        "regola": ("in caso di disaccordo si adotta il candidato A, cioe' la configurazione "
                   "della v2, gia' misurata: regola dichiarata prima della sessione")}
    salva()
else:
    PUNTO = dict(CANDIDATI["A"])
    print("")
    print("  meno di due candidati completati: si resta sul candidato A (v2)")
    RISULTATI["fasi"]["f2_candidati"] = {"candidati": RIS_CAND, "punto_scelto": PUNTO,
                                         "nota": "confronto non eseguito per mancanza di tempo"}
    salva()

# --- confronto accoppiato fra i due candidati sul dev, per il report ------------------------
# Non serve a scegliere (la scelta e' gia' fatta sulla validazione): serve a dire di quanto
# differiscono, con un intervallo, invece di affiancare due medie.
if len(PS_CAND) == 2 and all(len(v) == N_SEED_CELLA for v in PS_CAND.values()):
    print("")
    print("  differenza fra i due candidati sul set di sviluppo, per seed:")
    RISULTATI["fasi"]["f2_candidati"]["A_vs_B"] = {}
    for i in range(N_SEED_CELLA):
        st = confronto(PS_CAND["A"][i][10], PS_CAND["B"][i][10])
        print(riga_confronto(f"A vs B (seed {QSEED_FINALI[i]})", st))
        RISULTATI["fasi"]["f2_candidati"]["A_vs_B"][f"seed{QSEED_FINALI[i]}"] = st
    salva()

# ===========================================================================================
# F3 — MODELLI FINALI E CURVA DEI RENDIMENTI
# ===========================================================================================
print("\n" + "=" * 100)
print(f"F3 — {N_MEMBRI_FINALI} MODELLI FINALI al punto vincente "
      f"(tau {PUNTO['tau']:g}, ms {PUNTO['ms']}, wd {PUNTO['wd']:g})")
print("=" * 100)
print("Ogni modello vede un campione diverso di query E una finestra diversa di reference:")
print("sono i due assi da cui viene il bagging. Variando solo il seed di inizializzazione il")
print("guadagno misurato era +0.07pp — la diversita' deve stare nei dati, non nell'init.")

MODELLI, PS_DEV, INFO = [], [], []
for qs in QSEED_FINALI[:N_MEMBRI_FINALI]:
    st = stima_minuti(esempi_totali(POOL, PUNTO["ms"]) * BAG_FRAC, EPOCHE_MAX)
    if tag_di(PUNTO["ms"], PUNTO["tau"], PUNTO["wd"], qs, True,
              PUNTO["hidden"]) not in RISULTATI["run"] \
            and tempo_rimasto() < st * 60 * 1.4:
        print(f"  SALTO il campione {qs}: stimati ~{st:.0f}m, restano "
              f"{tempo_rimasto() / 60:.0f}m")
        continue
    m, psd, psv, info = addestra(PUNTO["ms"], PUNTO["tau"], PUNTO["wd"], qs,
                                 f"finale, campione {qs}", True, PUNTO["hidden"])
    MODELLI.append(m)
    PS_DEV.append(psd)
    INFO.append(info)

if not MODELLI and MODELLI_M:
    print("  nessun modello finale nuovo: uso i membri del minimo garantito")
    MODELLI, PS_DEV, INFO = MODELLI_M, PS_M, INFO_M

RISULTATI["fasi"]["f3_finali"] = {
    "config": dict(PUNTO), "membri": [i["tag"] for i in INFO],
    "singoli": {i["tag"]: {"dev_r10": i["dev_r10"], "val_r10": i["val_r10"],
                           "epoca_scelta": i["epoca_scelta"]} for i in INFO},
    "media_singoli": float(np.mean([i["dev_r10"] for i in INFO])) if INFO else None,
    "diversita": diversita(PUNTO["ms"], [i["qseed"] for i in INFO]) if INFO else None}
salva()

PS_ENS = None
if len(MODELLI) >= 2:
    print(f"\n  media dei {len(MODELLI)} singoli: "
          f"{RISULTATI['fasi']['f3_finali']['media_singoli']:.2f}%")
    print("  curva dei rendimenti dell'ensemble (uscite mediate):")
    _, PS_ENS = curva_ensemble(MODELLI, "f3_finali")
    print("\n  confronti accoppiati dell'ensemble completo:")
    s = confronto(PS_ENS[10], ps_base[10])
    print(riga_confronto("ensemble vs baseline", s))
    RISULTATI["fasi"]["f3_finali"]["vs_baseline"] = s
    for i, ps in enumerate(PS_DEV):
        s = confronto(PS_ENS[10], ps[10])
        print(riga_confronto(f"ensemble vs membro {INFO[i]['tag']}", s))
        RISULTATI["fasi"]["f3_finali"].setdefault("vs_membri", {})[INFO[i]["tag"]] = s
    salva()
else:
    print("  meno di due modelli: niente ensemble")

# ===========================================================================================
# F4 — COLLASSO LINEARE
# ===========================================================================================
print("\n" + "=" * 100)
print("F4 — L'ENSEMBLE SI SCRIVE COME UNA MATRICE SOLA?")
print("=" * 100)
print("Se si', il bagging costa quanto la baseline a inferenza invece che N volte tanto.")


def ridge(Xs, Ys, lam=RIDGE):
    """Ridge in float64. NON usare `lstsq`: il sistema e' mal posto (rango 589 su 1025) e
    la sua soluzione non e' riproducibile (report del 27/08)."""
    A = torch.cat([Xs, torch.ones(len(Xs), 1)], 1).double()
    n = A.shape[1]
    return torch.linalg.solve(A.T @ A + lam * torch.eye(n, dtype=torch.float64),
                              A.T @ Ys.double()).float()


def r2_of(W, Xs, Ys):
    pred = torch.cat([Xs, torch.ones(len(Xs), 1)], 1) @ W
    return float(1 - ((Ys - pred) ** 2).sum() / ((Ys - Ys.mean(0)) ** 2).sum())


def build_lineare(W):
    """q = normalize(r + [r; c; 1] @ W). ATTENZIONE: gamma e' GIA' dentro W, perche' il
    bersaglio del fit e' `gamma * net(...)`. Rimoltiplicarlo (come fa multiseed.py) sarebbe
    un errore silenzioso: la Recall crollerebbe senza che nulla segnali il perche'."""
    Wd = W.to(DEV)

    def f(ref, entry):
        c = entry["c"].expand(len(ref), -1)
        q = ref + torch.cat([ref, c, torch.ones(len(ref), 1, device=ref.device)], 1) @ Wd
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return f


if len(MODELLI) >= 2 and tempo_rimasto() > 120:
    # coppie (r, c) da reference del TRAIN — mai dal set di sviluppo su cui si valuta
    attr_to_idx = {n: i for i, n in enumerate(ATTR)}
    qs_fit = []
    for n_cond, quanti in POOL_TARGETS.items():
        qs_fit += sample_queries(list(ATTR), quanti, n_cond, exclude=qs_fit,
                                 seed=SEED_FIT + n_cond)
    E_fit = E_fit_cpu.to(DEV)
    X, Yk = [], [[] for _ in MODELLI]
    for qt in qs_fit:
        pos, neg = parse_query_string(qt, attr_to_idx)
        viol = torch.ones(len(L_fit_cpu), dtype=torch.bool)
        for a in pos:
            viol &= L_fit_cpu[:, a] == 0
        for b in neg:
            viol &= L_fit_cpu[:, b] == 1
        src = viol.nonzero(as_tuple=True)[0]
        if len(src) < 10:
            continue
        gs = torch.Generator().manual_seed(SEED_FIT)
        pick = src[torch.randperm(len(src), generator=gs)[:PER_QUERY_FIT]].to(DEV)
        R = E_fit[pick]
        c = (sum((d_ens[a] for a in pos), torch.zeros(D, device=DEV))
             - sum((d_ens[b] for b in neg), torch.zeros(D, device=DEV))).expand(len(R), -1)
        X.append(torch.cat([R, c], 1).cpu())
        with torch.no_grad():
            for k, m in enumerate(MODELLI):
                Yk[k].append((m.gamma * m.net(torch.cat([R, c], dim=-1))).cpu())
    X = torch.cat(X)
    Yk = [torch.cat(y) for y in Yk]
    Y_ens = torch.stack(Yk).mean(0)
    print(f"  coppie (r, c) dal train per il fit: {len(X)} da {len(qs_fit)} query")

    gl = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(X), generator=gl)
    tr, te = idx[:int(.8 * len(X))], idx[int(.8 * len(X)):]
    W_ens = ridge(X[tr], Y_ens[tr])                  # fit sull'uscita MEDIATA
    W_media = torch.stack([ridge(X[tr], Yk[k][tr]) for k in range(len(MODELLI))]).mean(0)
    diff = float((W_ens - W_media).abs().max())
    r2 = r2_of(W_ens, X[te], Y_ens[te])
    print(f"  controllo 1 — fit(media) == media(fit): max|W_ens - W_media| = {diff:.2e}  "
          f"(atteso ~0: la ridge e' lineare nel bersaglio)")
    print(f"  R^2 della mappa sull'uscita mediata, su dati mai usati per il fit: {r2:.4f}")

    ps_lin = valuta(build_lineare(W_ens), devset, E_valid)
    r_lin = macro(ps_lin[10]) * 100
    print(f"  controllo 2 — retrieval:  matrice collassata {r_lin:.2f}%  "
          f"ensemble {macro(PS_ENS[10]) * 100:.2f}%")
    s_ens = confronto(ps_lin[10], PS_ENS[10])
    s_base = confronto(ps_lin[10], ps_base[10])
    print(riga_confronto("matrice vs ensemble", s_ens))
    print(riga_confronto("matrice vs baseline", s_base))
    torch.save({"W": W_ens, "membri": [i["tag"] for i in INFO], "config": dict(PUNTO)},
               RUN / "ckpt" / "matrice_collassata.pt")
    RISULTATI["fasi"]["f4_lineare"] = {
        "n_fit": len(X), "ridge": RIDGE, "r2": r2, "max_diff_fit_media": diff,
        "dev_r10": r_lin, "vs_ensemble": s_ens, "vs_baseline": s_base,
        "nota": "gamma e' gia' dentro W: il bersaglio del fit e' gamma*net(...)"}
    salva()
else:
    ps_lin = None
    print("  saltata (servono almeno due modelli e un po' di tempo)")

# ===========================================================================================
# F5 — REPORT
# ===========================================================================================
print("\n" + "=" * 100)
print("F5 — REPORT")
print("=" * 100)

metodi = {"baseline": ps_base}
for i in INFO:
    z = np.load(RUN / "esiti" / f"{i['tag']}.npz")
    metodi[f"membro {i['tag']}"] = spacchetta(z, "dev")
if PS_ENS is not None:
    metodi[f"ENSEMBLE ({len(MODELLI)} modelli)"] = PS_ENS
if ps_lin is not None:
    metodi["MATRICE collassata"] = ps_lin

pacco = {}
for nome, ps in metodi.items():
    pacco.update(impacchetta(ps, nome.replace(" ", "_").replace("(", "").replace(")", "")))
np.savez_compressed(RUN / "esiti_devset.npz", **pacco)


def tab(righe, testata):
    larg = [max(len(str(r[i])) for r in [testata] + righe) for i in range(len(testata))]
    out = ["| " + " | ".join(str(t).ljust(larg[i]) for i, t in enumerate(testata)) + " |",
           "|" + "|".join("-" * (l + 2) for l in larg) + "|"]
    for r in righe:
        out.append("| " + " | ".join(str(c).ljust(larg[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


md = [f"# Sessione di consolidamento — {datetime.now():%d/%m/%Y %H:%M}", "",
      f"Device: `{DEV}`. Baseline sul set di sviluppo: **{B10_FULL:.2f}%** "
      f"(atteso {BASELINE_ATTESA}%, collaudo superato).",
      "**Questo script non legge mai `celeba_evaluation.json`.** Il budget del benchmark "
      "ufficiale e' per metodo ed e' gia' stato speso una volta, il 25/08, per il filtro "
      "zero-shot, con esito negativo (`scripts/README.md`): la valutazione del contributo "
      "sul benchmark si fa dopo, a mano, a configurazione congelata.",
      f"**Validazione riparata (v3)**: {len(VAL)} query valutate su una galleria di "
      f"{N_TRAIN_CORPUS} immagini **disgiunta** da quella dell'addestramento (R1), con la "
      f"composizione del benchmark ufficiale invece che di sole query composte (R2). "
      f"Nella v2 la galleria era condivisa e lo Spearman validazione-dev era crollato da "
      f"+0.897 a +0.244.",
      "Nessuna ricerca a coordinate: si sceglie fra **due candidati** gia' misurati, sulla "
      "sola validazione. Il dev non entra mai nella scelta.", ""]

md += ["## Risultati sul set di sviluppo (R@10, macro-average su 101 query)", ""]
righe = []
for nome, ps in metodi.items():
    r = macro(ps[10]) * 100
    if nome == "baseline":
        righe.append([nome, f"{r:.2f}%", "—", "—", "—"])
    else:
        s = confronto(ps[10], ps_base[10])
        righe.append([nome, f"{r:.2f}%", f"{s['delta_pp']:+.2f}pp",
                      f"[{s['lo']:+.2f}, {s['hi']:+.2f}]", f"{s['p']:.4f}"])
md.append(tab(righe, ["metodo", "R@10", "vs baseline", "IC 95%", "p"]))
md.append("")
md.append("*Ogni confronto e' una differenza accoppiata con bootstrap gerarchico "
          "(`src/devset.bootstrap_paired`), non due medie affiancate.*")
md.append("")

if "curva" in RISULTATI["fasi"].get("f3_finali", {}):
    md += ["## Curva dei rendimenti dell'ensemble", ""]
    righe = []
    for k, v in sorted(RISULTATI["fasi"]["f3_finali"]["curva"].items(), key=lambda x: int(x[0])):
        p = v.get("vs_k_precedente")
        righe.append([k, f"{v['dev_r10']:.2f}%", f"{v['vs_baseline']['delta_pp']:+.2f}pp",
                      f"{p['delta_pp']:+.2f}pp" if p else "—",
                      f"{p['p']:.3f}" if p else "—"])
    md.append(tab(righe, ["k", "R@10", "vs baseline", "vs k-1", "p"]))
    md.append("")

_div = RISULTATI["fasi"].get("f3_finali", {}).get("diversita")
if _div and _div.get("corpus_per_membro"):
    md += ["## Da dove viene la diversita' dell'ensemble", "",
           f"I {_div['n_membri']} membri sono addestrati al livello `max_sources "
           f"{_div['ms']}` su un pool costruito a `{MS_MAX}`, e differiscono su **tre assi**: "
           f"corpus di addestramento, campione di query, finestra di reference.", "",
           f"- i corpus di due membri condividono in media il "
           f"**{100 * _div['jaccard_corpus']:.1f}%** delle immagini (Jaccard fra i "
           f"sottoinsiemi di {N_TRAIN_CORPUS} estratti dalle 162770 del train)",
           f"- le query condivise sono circa il "
           f"{100 * _div['frazione_query_condivise']:.0f}% (frazione `BAG_FRAC` = {BAG_FRAC})",
           f"- piu' la finestra circolare di reference, diversa per membro", "",
           "Nella v1 il corpus era **uno solo per tutti**, e il guadagno del bagging era "
           "sceso da +5.16pp a +3.05pp mentre i singoli miglioravano di 2.2pp: e' la "
           "decomposizione di ambiguita' (membri migliori e piu' simili, meno da guadagnare "
           "mediando). Questo asse e' stato rimesso.", ""]
elif _div:
    md += ["## Da dove viene la diversita' dell'ensemble", "",
           f"I {_div['n_membri']} membri sono addestrati al livello `max_sources "
           f"{_div['ms']}` su un pool costruito a `{MS_MAX}`: ognuno vede un campione "
           f"diverso di query **e** una finestra diversa di reference dentro ogni query. "
           f"**Il corpus e' condiviso** (`--corpus-condiviso`), quindi manca il terzo asse.", "",
           f"- sovrapposizione media fra i campioni di due membri (Jaccard sugli esempi, "
           f"{_div['n_coppie']} coppie): **{_div['jaccard_medio']:.3f}** "
           f"[{_div['jaccard_min']:.3f}, {_div['jaccard_max']:.3f}]",
           f"- senza l'offset sulle reference sarebbe stata "
           f"**{_div['jaccard_medio_senza_offset']:.3f}**: la differenza e' la quota di "
           f"diversita' che l'offset aggiunge",
           f"- su {_div['bacini_esauriti']}/{_div['n_query_pool']} query il bacino di "
           f"reference si esaurisce prima di `{_div['ms']}`, quindi li' tutti i membri "
           f"vedono le stesse reference: e' inevitabile ed e' contato.", ""]

_f2 = RISULTATI["fasi"].get("f2_candidati", {})
if _f2.get("candidati"):
    md += ["## I due candidati, e il collaudo della riparazione", "",
           "Nessuna ricerca a coordinate: in v1 aveva scelto bene, in v2 ha scelto 1.5pp "
           "**peggio** del meglio che aveva provato, perche' leggeva uno strumento rotto. "
           "Qui si confrontano due soli punti, entrambi gia' misurati, con lo strumento "
           "riparato.", ""]
    righe = [[n, f"{c['config']['tau']:g}", c["config"]["ms"], f"{c['config']['wd']:g}",
              f"{c['val_media']:.2f}%", ", ".join(f"{v:.2f}" for v in c["val_seed"]),
              f"{c['dev_media']:.2f}%"]
             for n, c in _f2["candidati"].items()]
    md.append(tab(righe, ["candidato", "tau", "max_sources", "wd", "val R@10",
                          "val per seed", "dev R@10 (non usato)"]))
    md += ["",
           f"- vincitore secondo la **validazione**: **{_f2.get('vincitore_validazione')}**",
           f"- vincitore secondo il **dev** (non usato per scegliere): "
           f"{_f2.get('vincitore_dev_non_usato')}",
           "- **i due strumenti sono " + ("D'ACCORDO" if _f2.get("accordo")
                                                 else "IN DISACCORDO") + "**"
           + ("" if _f2.get("spearman_val_dev") is None else
              f"; Spearman validazione-dev su tutti i run: "
              f"{_f2['spearman_val_dev']:+.3f} (v1 +0.897, v2 +0.244)"),
           "", f"Punto adottato: **tau {PUNTO['tau']:g}, max_sources {PUNTO['ms']}, "
           f"wd {PUNTO['wd']:g}, hidden {PUNTO['hidden']}**.", ""]
    if not _f2.get("accordo", True):
        md += ["> **La riparazione non ha funzionato.** Per la regola dichiarata prima della "
               "sessione si adotta il candidato A, cioe' la configurazione della v2, gia' "
               "misurata con intervalli. Non si e' scelto guardando il dev.", ""]
    if _f2.get("A_vs_B"):
        md += ["Differenza fra i due candidati sul set di sviluppo, per seed (non usata per "
               "scegliere, serve a dire di quanto differiscono):", ""]
        md.append(tab([[k, f"{v['delta_pp']:+.2f}pp", f"[{v['lo']:+.2f}, {v['hi']:+.2f}]",
                        f"{v['p']:.4f}", f"{v['query_vinte']}V/{v['query_perse']}P"]
                       for k, v in _f2["A_vs_B"].items()],
                      ["seed", "A - B", "IC 95%", "p", "query"]))
        md.append("")

if "confronti_livelli" in RISULTATI["fasi"].get("f2_ricerca", {}):
    md += ["## Confronto accoppiato fra livelli, ai rispettivi punti d'arresto", "",
           "Le viste sui livelli sono **annidate** (stesse query, prefisso di reference), "
           "quindi il confronto e' accoppiato; ogni livello e' fermato dalla propria "
           "validazione e gli esiti per source sono quelli dell'epoca scelta.", ""]
    righe = [[k, f"{v['delta_pp']:+.2f}pp", f"[{v['lo']:+.2f}, {v['hi']:+.2f}]",
              f"{v['p']:.4f}", f"{v['query_vinte']}V/{v['query_perse']}P"]
             for k, v in RISULTATI["fasi"]["f2_ricerca"]["confronti_livelli"]["vs"].items()]
    md.append(tab(righe, ["confronto", "differenza", "IC 95%", "p", "query"]))
    md.append("")

if "f4_lineare" in RISULTATI["fasi"]:
    f4 = RISULTATI["fasi"]["f4_lineare"]
    md += ["## Collasso lineare", "",
           f"- `fit(media) == media(fit)`: max|W_ens - W_media| = **{f4['max_diff_fit_media']:.2e}** "
           f"(atteso ~0, la ridge e' lineare nel bersaglio)",
           f"- R² della mappa sull'uscita mediata, su dati mai usati per il fit: "
           f"**{f4['r2']:.4f}**",
           f"- Recall@10 della matrice collassata: **{f4['dev_r10']:.2f}%**, "
           f"vs ensemble {f4['vs_ensemble']['delta_pp']:+.2f}pp "
           f"(p = {f4['vs_ensemble']['p']:.4f})",
           f"- coppie (r, c) del fit: {f4['n_fit']}, tutte da reference del **train**", ""]

md += ["## Come rileggere questi file", "",
       "- `risultati.json` — tutto, comprese le traiettorie per epoca",
       "- `esiti_devset.npz` — esiti per singola source di ogni metodo "
       "(`<metodo>_k10_flat` + `<metodo>_k10_len`), per rifare qualunque bootstrap",
       "- `ckpt/*.pt` — un checkpoint per modello, all'**epoca scelta dalla validazione**",
       "- `ckpt/matrice_collassata.pt` — la matrice W (gamma gia' dentro)", ""]

(RUN / "report.md").write_text("\n".join(md), encoding="utf-8")
RISULTATI["meta"]["durata_ore"] = (time.time() - T0) / 3600
RISULTATI["meta"]["fine"] = datetime.now().isoformat(timespec="seconds")
salva()

print("\n".join(md))
print("\n" + "=" * 100)
print(f"FINITO — {(time.time() - T0) / 60:.1f} minuti")
print(f"  {JSONP}")
print(f"  {RUN / 'report.md'}")
print(f"  {RUN / 'esiti_devset.npz'}")
print(f"  {RUN / 'ckpt'}/")
print("Questo script non ha letto celeba_evaluation.json. Il benchmark ufficiale si valuta")
print("dopo, a mano, a configurazione congelata (e il suo budget e' per metodo: una spesa,")
print("il 25/08 sul filtro zero-shot, e' gia' stata fatta — vedi scripts/README.md).")
print("=" * 100)
