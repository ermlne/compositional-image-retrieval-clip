# extract_embeddings.py
import torch
from transformers import CLIPModel, CLIPImageProcessor
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("/home/disi/datasets")   # adatta al tuo path
SPLIT = "test"                          # "train" | "valid" | "test"
BATCH_SIZE = 256                        # più alto = più veloce su GPU
MODEL_ID = "openai/clip-vit-base-patch32"   # CLIP ViT-B/32 da HuggingFace, richiesto dall'assignment
OUT_DIR = Path("./embeddings")
OUT_DIR.mkdir(exist_ok=True)
# ────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")

# Carica CLIP da HuggingFace
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
processor = CLIPImageProcessor.from_pretrained(MODEL_ID)

# Il transform applica il preprocessing di CLIP (resize, crop, normalize) a ogni immagine PIL
def preprocess(image):
    return processor(images=image, return_tensors="pt")["pixel_values"][0]

# Dataset con il preprocessing di CLIP
dataset = CelebA(root=DATA_ROOT, split=SPLIT, download=False,
                 transform=preprocess)
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=4, pin_memory=True)

all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(loader, desc=f"Encoding {SPLIT}"):
        images = images.to(DEVICE)
        out = model.get_image_features(pixel_values=images)
        # transformers >=5 restituisce un output strutturato invece del tensore nudo
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        # L2-normalize: CLIP opera su ipersfera unitaria
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu())
        all_labels.append(labels.cpu())

embeddings = torch.cat(all_embeddings, dim=0)  # shape: [N, 512]
labels = torch.cat(all_labels, dim=0)          # shape: [N, 40]

print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape:     {labels.shape}")

torch.save({
    "embeddings": embeddings,   # float32, L2-normalizzati
    "labels": labels,           # int tensor, 40 attributi binari
    "split": SPLIT,
    "model": MODEL_ID,
    "attr_names": dataset.attr_names,
}, OUT_DIR / f"celeba_{SPLIT}_clip_vitb32.pt")

print("Salvato in:", OUT_DIR / f"celeba_{SPLIT}_clip_vitb32.pt")
