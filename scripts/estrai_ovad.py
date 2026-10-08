#!/usr/bin/env python3
"""
Estrae gli embedding CLIP ViT-B/32 di OVAD (Open-Vocabulary Attribute Detection).

Scopo: 117 attributi in piu' rispetto ai 40 di CelebA e agli 85 di AwA2, per caratterizzare
la relazione fra direzioni testuali e direzioni visive nello spazio di CLIP
(docs/piano-interpretabilita.md, diagnosi del commit 90db8b0).

PERCHE' OVAD VALE PIU' DI AwA2, a parita' di numero di attributi. In AwA2 le etichette sono
per CLASSE, quindi le direzioni visive sono differenze fra centroidi di 50 classi e vivono in
al piu' 49 dimensioni. OVAD annota il SINGOLO OGGETTO: 14.300 istanze etichettate una per una,
senza quel tetto. E' il dataset che porta il segnale piu' informativo dei due.

TRE AVVERTENZE DA RIPORTARE NEL REPORT
--------------------------------------
1. L'UNITA' NON E' L'IMMAGINE, E' L'OGGETTO. Si codificano ritagli di bounding box, non
   immagini intere. Gli embedding non sono confrontabili con quelli di CelebA e AwA2 senza
   dirlo: li' l'immagine e' l'oggetto, qui e' una scena che ne contiene molti.

2. LE ETICHETTE SONO A TRE VALORI: 1 positivo (7.4%), 0 negativo (76.4%), **-1 SCONOSCIUTO
   (16.2%)**. Il -1 e' conservato tale e quale nel file. Chi lo legge DEVE mascherarlo: con
   CelebA e AwA2 `labels == 0` significa "non ha l'attributo", qui significherebbe anche
   "non annotato", e un sesto dei dati finirebbe nella classe sbagliata. Il valore giusto e'
   `noti = labels != -1`, poi `positivi = labels == 1`.

3. GLI ATTRIBUTI SONO RAGGRUPPATI IN 24 TIPI di alternative mutuamente esclusive
   ("clothes color:black", "clothes color:blue", ...). Non sono 117 attributi indipendenti
   come i 40 di CelebA: dentro un tipo le direzioni sono per costruzione correlate
   negativamente. Il campo `attr_types` serve a tenerne conto.

DUE SCELTE CHE INFLUENZANO IL RISULTATO, entrambe esposte come opzioni perche' vanno
dichiarate e non subite:

  --margine    quanto contesto attorno al box (default 0.10, cioe' +10%). Un ritaglio
               stretto perde il contesto che CLIP usa; troppo largo include altri oggetti.
  --min-lato   scarta i box con lato corto sotto N pixel (default 32). Non e' pignoleria:
               il 38% delle istanze sta sotto i 32 px, e un ritaglio di 20 px portato a 224
               e' rumore interpolato. Con il default restano ~8.800 istanze, che bastano
               largamente allo scopo (misurare, non addestrare). Il conteggio effettivo
               viene stampato e salvato nel file.

Formato di uscita: stesso schema di embeddings/celeba_*_clip_vitb32.pt e awa2_clip_vitb32.pt,
cosi' il codice esistente lo legge senza modifiche.

Riprende da dove si era interrotto: se il file finale esiste esce subito, se esiste un
checkpoint parziale riparte da li'.

Uso tipico (da radice repo):
  HF_HUB_OFFLINE=1 python scripts/estrai_ovad.py --download --out embeddings
"""
import argparse, json, shutil, subprocess, sys, time, zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

MODEL_ID = "openai/clip-vit-base-patch32"
URL_ANNOTAZIONI = "https://lmb.informatik.uni-freiburg.de/resources/datasets/ovad.zip"
URL_IMMAGINI = "http://images.cocodataset.org/zips/val2017.zip"   # 815 MB

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path.home() / "datasets" / "ovad"),
                help="cartella che contiene (o conterra') ovad2000.json e val2017/")
ap.add_argument("--out", default=str(Path.home() / "embeddings"))
ap.add_argument("--batch", type=int, default=64)
# workers=0 di default: con forkserver (predefinito in Python 3.14 su Linux) e poca RAM
# libera i worker non partono — e' successo davvero durante l'estrazione di AwA2.
ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--download", action="store_true", help="scarica cio' che manca (~816 MB)")
ap.add_argument("--ogni", type=int, default=40, help="salva un checkpoint ogni N batch")
ap.add_argument("--margine", type=float, default=0.10, help="contesto attorno al box")
ap.add_argument("--min-lato", type=int, default=32, help="lato corto minimo, in pixel")
args = ap.parse_args()

ROOT = Path(args.root)
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
FINALE = OUT / "ovad_clip_vitb32.pt"
PARZIALE = OUT / "ovad_clip_vitb32.parziale.pt"
STATO = OUT / "ovad_stato.json"


def stato(fase, **kw):
    STATO.write_text(json.dumps({"fase": fase, "quando": time.strftime("%F %T"), **kw}, indent=2))
    print(f"[{time.strftime('%T')}] {fase} {kw if kw else ''}", flush=True)


if FINALE.exists():
    stato("gia_fatto", file=str(FINALE))
    sys.exit(0)

# ---------------------------------------------------------------- scaricamento
ANNOT = ROOT / "ovad2000.json"
IMMAGINI = ROOT / "val2017"
ROOT.mkdir(parents=True, exist_ok=True)

if not ANNOT.exists() or not IMMAGINI.is_dir():
    if not args.download:
        sys.exit(f"OVAD non trovato in {ROOT}. Rilancia con --download, "
                 f"oppure indica il percorso giusto con --root.")
    libero = shutil.disk_usage(ROOT).free / 2**30
    if libero < 4:
        sys.exit(f"spazio insufficiente: {libero:.1f} GB liberi, ne servono ~4")

if not ANNOT.exists():
    stato("scarico_annotazioni", url=URL_ANNOTAZIONI)
    z = ROOT / "ovad.zip"
    subprocess.run(["curl", "-L", "-C", "-", "-o", str(z), URL_ANNOTAZIONI], check=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(ROOT)
    z.unlink(missing_ok=True)

if not IMMAGINI.is_dir():
    stato("scarico_immagini", url=URL_IMMAGINI, nota="815 MB, immagini COCO val2017")
    z = ROOT / "val2017.zip"
    if not z.exists():
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(z), URL_IMMAGINI], check=True)
    stato("estraggo_immagini")
    subprocess.run(["unzip", "-q", "-n", str(z), "-d", str(ROOT)], check=True)
    if IMMAGINI.is_dir():
        z.unlink(missing_ok=True)

if not ANNOT.exists() or not IMMAGINI.is_dir():
    sys.exit(f"manca ancora qualcosa: {ANNOT} o {IMMAGINI}")

# ---------------------------------------------------------------- annotazioni
d = json.loads(ANNOT.read_text())
attributi = [a["name"] for a in d["attributes"]]
tipi = [a["type"] for a in d["attributes"]]
frequenza = [a.get("freq_set", "") for a in d["attributes"]]
categorie = {c["id"]: c["name"] for c in d["categories"]}
file_di = {im["id"]: im["file_name"] for im in d["images"]}
stato("annotazioni_lette", immagini=len(d["images"]), istanze=len(d["annotations"]),
      attributi=len(attributi), tipi=len(set(tipi)))

istanze, scartate_piccole, scartate_mancanti = [], 0, 0
for a in d["annotations"]:
    x, y, w, h = a["bbox"]
    if min(w, h) < args.min_lato:
        scartate_piccole += 1
        continue
    p = IMMAGINI / file_di[a["image_id"]]
    if not p.exists():
        scartate_mancanti += 1
        continue
    istanze.append({"file": p, "bbox": (x, y, w, h), "att": a["att_vec"],
                    "cat": a["category_id"], "img": a["image_id"], "id": a["id"]})

if not istanze:
    sys.exit("nessuna istanza sopravvissuta ai filtri")
stato("istanze_selezionate", tenute=len(istanze), scartate_piccole=scartate_piccole,
      scartate_immagine_mancante=scartate_mancanti, min_lato=args.min_lato)

etichette = torch.tensor([i["att"] for i in istanze], dtype=torch.int8)   # 1 / 0 / -1

# ---------------------------------------------------------------- modello
from transformers import CLIPModel, CLIPImageProcessor
DEV = "cuda" if torch.cuda.is_available() else "cpu"
stato("carico_modello", device=DEV,
      gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None)
model = CLIPModel.from_pretrained(MODEL_ID).to(DEV).eval()
proc = CLIPImageProcessor.from_pretrained(MODEL_ID)


class Ritagli(Dataset):
    """Un elemento = un oggetto annotato, ritagliato con un margine di contesto."""

    def __init__(self, ist, margine):
        self.ist, self.m = ist, margine

    def __len__(self):
        return len(self.ist)

    def __getitem__(self, i):
        r = self.ist[i]
        im = Image.open(r["file"]).convert("RGB")
        x, y, w, h = r["bbox"]
        dx, dy = w * self.m, h * self.m
        box = (max(0, x - dx), max(0, y - dy),
               min(im.width, x + w + dx), min(im.height, y + h + dy))
        ritaglio = im.crop(tuple(int(round(v)) for v in box))
        if ritaglio.width < 2 or ritaglio.height < 2:      # difesa contro box degeneri
            ritaglio = im
        return proc(images=ritaglio, return_tensors="pt")["pixel_values"][0]


# ---------------------------------------------------------------- ripresa
fatti, pezzi = 0, []
if PARZIALE.exists():
    ck = torch.load(PARZIALE, map_location="cpu")
    pezzi, fatti = [ck["embeddings"]], ck["fatti"]
    stato("riprendo", da=fatti)

loader = DataLoader(Ritagli(istanze[fatti:], args.margine), batch_size=args.batch,
                    shuffle=False, num_workers=args.workers, pin_memory=(DEV == "cuda"))

t0 = time.time()
with torch.no_grad():
    for b, px in enumerate(loader):
        out = model.get_image_features(pixel_values=px.to(DEV, non_blocking=True))
        f = out.pooler_output if hasattr(out, "pooler_output") else out
        pezzi.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        fatti += len(px)
        if (b + 1) % args.ogni == 0:
            E = torch.cat(pezzi); pezzi = [E]
            tmp = PARZIALE.with_suffix(".tmp")
            torch.save({"embeddings": E, "fatti": fatti}, tmp)
            tmp.replace(PARZIALE)              # sostituzione atomica: mai un file mezzo scritto
            vel = fatti / (time.time() - t0)
            stato("in_corso", fatte=fatti, totali=len(istanze),
                  percento=round(100 * fatti / len(istanze), 1),
                  minuti_rimasti=round((len(istanze) - fatti) / max(vel, 1e-9) / 60, 1))

E = torch.cat(pezzi)
assert len(E) == len(istanze), f"attese {len(istanze)} righe, ottenute {len(E)}"

torch.save({
    "embeddings": E,                                     # float32, L2-normalizzati
    "labels": etichette,                                 # int8, [N, 117], valori 1 / 0 / -1
    "attr_names": attributi,
    "attr_types": tipi,                                  # 24 gruppi di alternative esclusive
    "attr_freq": frequenza,                              # head / medium / tail
    "category_of_instance": torch.tensor([i["cat"] for i in istanze], dtype=torch.long),
    "category_names": categorie,
    "image_id": torch.tensor([i["img"] for i in istanze], dtype=torch.long),
    "bbox": torch.tensor([i["bbox"] for i in istanze], dtype=torch.float32),
    "split": "ovad2000",
    "model": MODEL_ID,
    "ritaglio": {"margine": args.margine, "min_lato": args.min_lato},
    "nota_etichette": "TRE VALORI: 1 positivo, 0 negativo, -1 SCONOSCIUTO (16% del totale). "
                      "Mascherare sempre con labels != -1 prima di usarle: a differenza di "
                      "CelebA e AwA2, qui 0 non significa 'non ha l'attributo' per tutti.",
    "nota_unita": "un elemento = un OGGETTO ritagliato, non un'immagine intera: non "
                  "confrontabile con CelebA/AwA2 senza dirlo",
}, FINALE)
PARZIALE.unlink(missing_ok=True)

# ---------------------------------------------------------------- verifica minima
noti = etichette != -1
n_pos = ((etichette == 1) & noti).sum(0)
n_neg = ((etichette == 0) & noti).sum(0)
usabili = int(((n_pos >= 20) & (n_neg >= 20)).sum())
stato("finito", file=str(FINALE), istanze=len(E), attributi=len(attributi),
      attributi_usabili=usabili, sconosciuti_pct=round(100 * float((~noti).float().mean()), 1),
      minuti=round((time.time() - t0) / 60, 1),
      gb=round(FINALE.stat().st_size / 2**30, 3))
