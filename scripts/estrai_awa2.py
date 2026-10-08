#!/usr/bin/env python3
"""
Estrae gli embedding CLIP ViT-B/32 di AwA2 (Animals with Attributes 2).

Scopo: 85 attributi in piu' rispetto ai 40 di CelebA, per caratterizzare la relazione fra
direzioni testuali e direzioni visive nello spazio di CLIP (docs/piano-interpretabilita.md).

AVVERTENZA sui dati, da riportare nel report:
  in AwA2 gli attributi sono annotati per CLASSE (50 classi x 85 attributi), non per immagine.
  Ogni immagine eredita il vettore della sua classe. Conseguenza geometrica: le direzioni
  visive sono differenze fra centroidi di classe, quindi vivono in un sottospazio di al piu'
  49 dimensioni (50 classi - 1), non 512. Restano 85 direzioni distinte — che e' cio' che
  serve — ma il tetto alla caratterizzazione visiva e' 49, non 512.

Formato di uscita: identico a embeddings/celeba_*_clip_vitb32.pt, cosi' il codice esistente
lo legge senza modifiche.

Riprende da dove si era interrotto: se il file finale esiste, esce subito; se esiste un
checkpoint parziale, riparte da li'.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

MODEL_ID = "openai/clip-vit-base-patch32"
URL_DATI = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path.home() / "datasets"),
                help="cartella che contiene (o conterra') Animals_with_Attributes2")
ap.add_argument("--out", default=str(Path.home() / "embeddings"))
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--download", action="store_true", help="scarica AwA2 se manca (~13 GB)")
ap.add_argument("--ogni", type=int, default=40, help="salva un checkpoint ogni N batch")
args = ap.parse_args()

ROOT = Path(args.root)
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
FINALE = OUT / "awa2_clip_vitb32.pt"
PARZIALE = OUT / "awa2_clip_vitb32.parziale.pt"
STATO = OUT / "awa2_stato.json"

def stato(fase, **kw):
    STATO.write_text(json.dumps({"fase": fase, "quando": time.strftime("%F %T"), **kw}, indent=2))
    print(f"[{time.strftime('%T')}] {fase} {kw if kw else ''}", flush=True)

if FINALE.exists():
    stato("gia_fatto", file=str(FINALE))
    sys.exit(0)

# ---------------------------------------------------------------- scaricamento
BASE = ROOT / "Animals_with_Attributes2"
if not (BASE / "JPEGImages").is_dir():
    if not args.download:
        sys.exit(f"AwA2 non trovato in {BASE}. Rilancia con --download, "
                 f"oppure indica il percorso giusto con --root.")
    ROOT.mkdir(parents=True, exist_ok=True)
    libero = shutil.disk_usage(ROOT).free / 2**30
    if libero < 35:
        sys.exit(f"spazio insufficiente: {libero:.1f} GB liberi, ne servono ~35 "
                 f"(13 di zip + 13 di immagini + margine)")
    zipf = ROOT / "AwA2-data.zip"
    if not zipf.exists():
        stato("scarico", url=URL_DATI, gb_liberi=round(libero, 1))
        # -C riprende un download interrotto
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(zipf), URL_DATI], check=True)
    stato("estraggo")
    subprocess.run(["unzip", "-q", "-n", str(zipf), "-d", str(ROOT)], check=True)
    if (BASE / "JPEGImages").is_dir():
        zipf.unlink(missing_ok=True)          # recupera 13 GB
if not (BASE / "JPEGImages").is_dir():
    sys.exit(f"estrazione fallita: {BASE / 'JPEGImages'} non esiste")

# ---------------------------------------------------------------- etichette
classi = [r.split()[-1] for r in (BASE / "classes.txt").read_text().split("\n") if r.strip()]
attributi = [r.split()[-1] for r in (BASE / "predicates.txt").read_text().split("\n") if r.strip()]
M = torch.tensor([[int(float(x)) for x in r.split()]
                  for r in (BASE / "predicate-matrix-binary.txt").read_text().split("\n")
                  if r.strip()], dtype=torch.int8)
assert M.shape == (len(classi), len(attributi)), f"matrice {tuple(M.shape)} inattesa"
stato("etichette_lette", classi=len(classi), attributi=len(attributi))

percorsi, y = [], []
for ci, c in enumerate(classi):
    d = BASE / "JPEGImages" / c
    if not d.is_dir():
        print(f"  attenzione: cartella mancante per la classe {c}", flush=True)
        continue
    for p in sorted(d.glob("*.jpg")):
        percorsi.append(p); y.append(ci)
classe_di = torch.tensor(y, dtype=torch.long)
etichette = M[classe_di]                       # ogni immagine eredita gli attributi della classe
stato("immagini_elencate", n=len(percorsi))
if not percorsi:
    sys.exit("nessuna immagine trovata")

# ---------------------------------------------------------------- modello
from transformers import CLIPModel, CLIPImageProcessor
DEV = "cuda" if torch.cuda.is_available() else "cpu"
stato("carico_modello", device=DEV,
      gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None)
model = CLIPModel.from_pretrained(MODEL_ID).to(DEV).eval()
proc = CLIPImageProcessor.from_pretrained(MODEL_ID)

class Immagini(Dataset):
    def __init__(self, ps): self.ps = ps
    def __len__(self): return len(self.ps)
    def __getitem__(self, i):
        im = Image.open(self.ps[i]).convert("RGB")
        return proc(images=im, return_tensors="pt")["pixel_values"][0]

# ---------------------------------------------------------------- ripresa
fatti, pezzi = 0, []
if PARZIALE.exists():
    ck = torch.load(PARZIALE, map_location="cpu")
    pezzi, fatti = [ck["embeddings"]], ck["fatti"]
    stato("riprendo", da=fatti)

loader = DataLoader(Immagini(percorsi[fatti:]), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=(DEV == "cuda"))

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
            stato("in_corso", fatte=fatti, totali=len(percorsi),
                  percento=round(100 * fatti / len(percorsi), 1),
                  minuti_rimasti=round((len(percorsi) - fatti) / max(vel, 1e-9) / 60, 1))

E = torch.cat(pezzi)
assert len(E) == len(percorsi), f"attese {len(percorsi)} righe, ottenute {len(E)}"

torch.save({
    "embeddings": E,                                    # float32, L2-normalizzati
    "labels": etichette,                                # int8, [N, 85]
    "attr_names": attributi,
    "class_names": classi,
    "class_of_image": classe_di,
    "image_paths": [str(p.relative_to(BASE)) for p in percorsi],
    "split": "all",
    "model": MODEL_ID,
    "nota_etichette": "attributi per CLASSE, ereditati dall'immagine: le direzioni visive "
                      "vivono in un sottospazio di al piu' 49 dimensioni",
}, FINALE)
PARZIALE.unlink(missing_ok=True)

# ---------------------------------------------------------------- verifica minima
n_pos = (etichette == 1).sum(0)
usabili = int(((n_pos >= 20) & ((len(E) - n_pos) >= 20)).sum())
stato("finito", file=str(FINALE), immagini=len(E), attributi=len(attributi),
      attributi_usabili=usabili, minuti=round((time.time() - t0) / 60, 1),
      gb=round(FINALE.stat().st_size / 2**30, 2))
