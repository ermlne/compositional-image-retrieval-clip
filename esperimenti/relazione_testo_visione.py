#!/usr/bin/env python3
"""
La relazione fra direzioni testuali e direzioni visive in CLIP: lineare o no?

DOMANDA. Per ogni attributo esistono due "frecce": una calcolata dal testo (prompt positivo
meno prompt negativo) e una calcolata dalle immagini (centroide con l'attributo meno centroide
senza). Esiste una trasformazione che porta dalla prima alla seconda, e si trasferisce ad
attributi mai visti? E se esiste, e' lineare o serve qualcosa di piu' ricco?

PERCHE' ORA E NON PRIMA. Su CelebA soli 40 attributi: con 30 esempi in 512 dimensioni se ne
stimano onestamente ~3, e qualunque modello piu' ricco memorizza (esperimento LayerNorm, 99.4%
in addestramento contro 18.5% in sviluppo). Una funzione NON lineare ha piu' parametri, quindi
serve piu' vocabolario, non meno. Con AwA2 (85) e OVAD (117) la domanda diventa ponibile.

DUE GRADINI, stesso protocollo e stessi attributi tenuti fuori:
  1. mappa LINEARE      (ridge, in forma chiusa)
  2. mappa NON LINEARE  (piccola rete)
Il confronto FRA I DUE, sugli attributi mai visti, e' la risposta.

DUE PROTOCOLLI:
  A. attributi tenuti fuori a caso dal pool -> generalizza a concetti nuovi?
  B. un intero dominio tenuto fuori          -> e' una proprieta' di CLIP o del dominio?

METRICA: coseno fra freccia visiva predetta e freccia visiva vera, sugli attributi tenuti
fuori. Qui la domanda E' geometrica (caratterizzare la relazione), quindi il coseno e' la
metrica giusta. Ma vale la regola stabilita il 28/08: se da questa caratterizzazione nascera'
un metodo, quel metodo va validato in Recall su CelebA prima di crederci.
"""
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--fold", type=int, default=5, help="quante partizioni nel protocollo A")
ap.add_argument("--epoche", type=int, default=400)
ap.add_argument("--nascosto", type=int, default=256)
args = ap.parse_args()
torch.manual_seed(args.seed)

MODEL_ID = "openai/clip-vit-base-patch32"
MIN_ESEMPI = 20

# --- template: uno per dominio, stessa logica dei 10 gia' in uso per CelebA ----------------
T_CELEBA = [
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
# gli attributi di AwA2 mescolano aggettivi (furry), parti del corpo (flippers), comportamenti
# (hunter) e habitat (arctic): nessun singolo template li copre. L'ensemble media le forme
# grammaticali, che e' esattamente lo scopo del prompt ensembling.
T_AWA2 = [
    ("a photo of a {} animal",               "a photo of an animal that is not {}"),
    ("a photo of an animal that is {}",      "a photo of an animal that is not {}"),
    ("a photo of an animal with {}",         "a photo of an animal without {}"),
    ("an animal with {}",                    "an animal without {}"),
    ("{}",                                   "not {}"),
    ("a photo of an animal, {}",             "a photo of an animal, not {}"),
]
T_OVAD = [
    ("a photo of a {} object",               "a photo of an object that is not {}"),
    ("a photo of an object that is {}",      "a photo of an object that is not {}"),
    ("an object with {}",                    "an object without {}"),
    ("{}",                                   "not {}"),
    ("a cropped photo of a {} object",       "a cropped photo of an object that is not {}"),
]


def nomi_celeba(n):    return [n.replace("_", " ").lower()]
def nomi_awa2(n):      return [n.replace("+", " ").lower()]
# i nomi OVAD sono "tipo:valore1/valore2". Il TIPO va tenuto: buttarlo via rende
# `color:blue` e `clothes color:blue` la stessa frase, quindi la stessa freccia testuale
# (coseno 1.000) con bersagli visivi diversi — contamina la mappa e falsa il tenuto-fuori.
# Per i tipi composti il cui primo termine e' la parte dell'oggetto, quella parte diventa il
# soggetto: "hair color:yellow" -> "yellow hair", distinto da "color:yellow" -> "yellow".
SOGGETTI = {"clothes", "hair", "face"}

def nomi_ovad(n):
    tipo, valore = n.split(":", 1) if ":" in n else ("", n)
    valori = [s.strip().lower() for s in valore.split("/") if s.strip()]
    parole = tipo.split()
    if len(parole) > 1 and parole[0] in SOGGETTI:
        return [f"{v} {parole[0]}" for v in valori]
    return valori


@torch.no_grad()
def direzioni_testuali(nomi, template, espandi, model, tok):
    """Freccia testuale = media dei prompt positivi meno media dei negativi.
    Se un attributo ha piu' sinonimi (OVAD) si mediano anche quelli."""
    D = model.config.projection_dim
    fuori = torch.zeros(len(nomi), D)
    for i, n in enumerate(nomi):
        varianti = espandi(n)
        pos = [t.format(v) for t, _ in template for v in varianti]
        neg = [t.format(v) for _, t in template for v in varianti]
        e = []
        for frasi in (pos, neg):
            x = tok(frasi, padding=True, return_tensors="pt")
            out = model.get_text_features(**x)
            f = (out.pooler_output if hasattr(out, "pooler_output") else out).float()
            e.append((f / f.norm(dim=-1, keepdim=True)).mean(0))
        v = e[0] - e[1]
        fuori[i] = v / v.norm().clamp(min=1e-8)
    return fuori


def direzioni_visive(E, L):
    """Freccia visiva = centroide con l'attributo meno centroide senza.
    Le etichette a -1 di OVAD sono SCONOSCIUTE: vanno escluse da entrambi i lati, non
    contate come negative (sarebbe il 14% dei dati nella classe sbagliata)."""
    D = E.shape[1]
    fuori = torch.zeros(L.shape[1], D)
    validi = []
    for a in range(L.shape[1]):
        noti = L[:, a] != -1
        pos = (L[:, a] == 1) & noti
        neg = (L[:, a] == 0) & noti
        if int(pos.sum()) < MIN_ESEMPI or int(neg.sum()) < MIN_ESEMPI:
            continue
        v = E[pos].mean(0) - E[neg].mean(0)
        fuori[a] = v / v.norm().clamp(min=1e-8)
        validi.append(a)
    return fuori, validi


# ------------------------------------------------------------------ i tre domini
print("carico CLIP...", flush=True)
from transformers import CLIPModel, CLIPTokenizer
model = CLIPModel.from_pretrained(MODEL_ID).eval()
tok = CLIPTokenizer.from_pretrained(MODEL_ID)

DOMINI = []


def aggiungi(nome, file_pt, template, espandi, chiave_lab="labels"):
    p = ROOT / "embeddings" / file_pt
    if not p.exists():
        print(f"  {nome}: {p} non c'e', salto", flush=True); return
    d = torch.load(p, map_location="cpu")
    E = d["embeddings"].float()
    E = E / E.norm(dim=-1, keepdim=True)
    L = d[chiave_lab]
    nomi = [n for n in d["attr_names"] if str(n).strip()]
    L = L[:, :len(nomi)]
    dv, validi = direzioni_visive(E, L)
    dt = direzioni_testuali(nomi, template, espandi, model, tok)
    for a in validi:
        DOMINI.append({"dominio": nome, "attr": nomi[a], "t": dt[a], "v": dv[a]})
    print(f"  {nome:<8}{len(validi):>4} attributi usabili su {len(nomi)}", flush=True)


print("costruisco le frecce...", flush=True)
aggiungi("celeba", "celeba_train_clip_vitb32.pt", T_CELEBA, nomi_celeba)
aggiungi("awa2",   "awa2_clip_vitb32.pt",         T_AWA2,   nomi_awa2)
aggiungi("ovad",   "ovad_clip_vitb32.pt",         T_OVAD,   nomi_ovad)

if len(DOMINI) < 60:
    sys.exit(f"solo {len(DOMINI)} attributi: servono almeno i tre dataset")

T = torch.stack([r["t"] for r in DOMINI])
V = torch.stack([r["v"] for r in DOMINI])
DOM = [r["dominio"] for r in DOMINI]
N, D = T.shape
print(f"\npool: {N} attributi, {D} dimensioni")
for d in ("celeba", "awa2", "ovad"):
    n = DOM.count(d)
    if n:
        idx = [i for i, x in enumerate(DOM) if x == d]
        c = (T[idx] * V[idx]).sum(-1)
        print(f"  {d:<8}{n:>4} attributi   allineamento grezzo testo<->visivo {c.mean():+.4f}")
c = (T * V).sum(-1)
print(f"  {'TOTALE':<8}{N:>4} attributi   allineamento grezzo {c.mean():+.4f}  "
      f"(caso: +-{1/D**0.5:.4f})")


# ------------------------------------------------------------------ i due gradini
def gradino1_lineare(Tr, Vr, Te, lam):
    """Ridge in forma chiusa: W = (Tr^T Tr + lam I)^-1 Tr^T Vr."""
    A = Tr.T @ Tr + lam * torch.eye(Tr.shape[1])
    W = torch.linalg.solve(A, Tr.T @ Vr)
    P = Te @ W
    return P / P.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def gradino2_rete(Tr, Vr, Te, nascosto, epoche, wd, seed):
    """Piccola rete con la stessa parametrizzazione residua del contributo candidato:
    parte dall'identita' (uscita azzerata) e impara una correzione."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Tr.shape[1], nascosto), nn.GELU(),
                        nn.Linear(nascosto, Tr.shape[1]))
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    for _ in range(epoche):
        p = Tr + net(Tr)
        p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        loss = -(p * Vr).sum(-1).mean()          # massimizza il coseno
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = Te + net(Te)
        return p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def allin(P, Vt):
    return float((P * Vt).sum(-1).mean())


LAMBDE = (1.0, 10.0, 100.0, 1000.0)
WD = (1e-4, 1e-2, 1e-1)


def valuta(tr, te, etichetta):
    Tr, Vr, Te, Vt = T[tr], V[tr], T[te], V[te]
    base = allin(Te, Vt)                                    # identita': la freccia testuale grezza
    lin = {l: allin(gradino1_lineare(Tr, Vr, Te, l), Vt) for l in LAMBDE}
    rete = {w: allin(gradino2_rete(Tr, Vr, Te, args.nascosto, args.epoche, w, args.seed), Vt)
            for w in WD}
    ml, mr = max(lin.values()), max(rete.values())
    print(f"  {etichetta:<26}{len(tr):>5}{len(te):>6}   {base:>+8.4f}{ml:>+10.4f}{mr:>+10.4f}"
          f"   {'LINEARE' if ml >= mr else 'non lineare':>12}")
    return {"etichetta": etichetta, "n_train": len(tr), "n_test": len(te), "identita": base,
            "lineare": lin, "rete": rete, "migliore_lineare": ml, "migliore_rete": mr}


risultati = {"pool": N, "per_dominio": {d: DOM.count(d) for d in set(DOM)}}

print("\n" + "=" * 96)
print("PROTOCOLLO A — attributi tenuti fuori a caso: la relazione arriva a concetti nuovi?")
print("=" * 96)
print(f"  {'partizione':<26}{'train':>5}{'test':>6}   {'identita':>8}{'lineare':>10}{'rete':>10}"
      f"   {'vince':>12}")
perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
risultati["A"] = []
for k in range(args.fold):
    te = perm[k::args.fold].tolist()
    tr = [i for i in range(N) if i not in set(te)]
    risultati["A"].append(valuta(tr, te, f"partizione {k+1}/{args.fold}"))
for nome in ("identita", "migliore_lineare", "migliore_rete"):
    m = sum(r[nome] for r in risultati["A"]) / len(risultati["A"])
    print(f"  media {nome:<20}{m:+.4f}")

print("\n" + "=" * 96)
print("PROTOCOLLO B — un intero dominio tenuto fuori: e' CLIP o e' il dominio?")
print("=" * 96)
print(f"  {'tenuto fuori':<26}{'train':>5}{'test':>6}   {'identita':>8}{'lineare':>10}{'rete':>10}"
      f"   {'vince':>12}")
risultati["B"] = []
for d in ("celeba", "awa2", "ovad"):
    te = [i for i, x in enumerate(DOM) if x == d]
    tr = [i for i, x in enumerate(DOM) if x != d]
    if te and tr:
        risultati["B"].append(valuta(tr, te, f"fuori: {d}"))

# le frecce servono alle analisi di controllo: si salvano per non ricodificare il testo
torch.save({"T": T, "V": V, "dom": DOM, "attr": [r["attr"] for r in DOMINI]},
           ROOT / "esperimenti" / "frecce_tre_domini.pt")

(ROOT / "esperimenti").mkdir(exist_ok=True)
with open(ROOT / "esperimenti" / "relazione_testo_visione.json", "w") as f:
    json.dump(risultati, f, indent=2)
print("\nsalvato in esperimenti/relazione_testo_visione.json")
