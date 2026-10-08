"""L'accordo fra i modelli fuori dallo span e' reale, o e' il canale della reference?

===========================================================================================
LA MISURA CHE SI VUOLE CONTROLLARE
===========================================================================================
`esperimenti/ensemble_seed.py` confronta i quattro modelli su

    net_i([r ; c])   normalizzato,   coseno medio fra coppie di modelli

e trova +0.7217 su direzioni FUORI dallo span delle 40 direzioni testuali, contro +0.5975
DENTRO. Da li' si conclude che la parte fuori span non e' residuo di inizializzazione (se lo
fosse, i modelli non concorderebbero) e si scarta l'idea di spegnerla col weight decay.

IL PROBLEMA. Quel confronto non isola il canale delle condizioni. Il gruppo ha misurato che
la correzione si scompone in **43.5% testo + 56.5% un termine che dipende solo dalla
reference** (`docs/STATO.md`). Se il termine solo-reference domina ed e' condiviso fra i
modelli, due modelli qualunque concordano *comunque*, che `c` stia dentro o fuori dallo span.
E l'accordo piu' BASSO dentro lo span e' proprio cio' che ci si aspetta in quel caso: li' il
canale del testo contribuisce segnale vero e diverso da modello a modello, che ABBASSA la
concordanza.

LA CORREZIONE, una riga:

    contributo del testo  =  net_i([r ; c])  -  net_i([r ; 0])

La sottrazione cancella esattamente il termine che dipende solo da `r` (per una mappa
lineare e' esatta; per la rete e' il controfattuale "stessa reference, nessuna condizione").
Su questo si rifa' lo stesso accordo.

COSA DECIDE.
  Se l'accordo fuori span CROLLA una volta isolato il canale c, allora fuori dallo span i
  modelli NON concordano: quella parte e' rumore specifico del modello, l'idea del weight
  decay torna in gioco, e soprattutto torna in gioco l'idea di vincolare W_c allo span
  (sez. 6b di piano-interpretabilita.md, l'unica casella non provata).
  Se l'accordo RESTA alto, la conclusione di ensemble_seed.py regge e la parte fuori span e'
  una componente sistematica appresa, non rumore.

CONTROLLI INCLUSI
  - la misura ORIGINALE, non isolata, ricalcolata qui: deve riprodurre ~+0.72 / ~+0.60,
    altrimenti non stiamo guardando la stessa cosa;
  - le NORME dei due pezzi, che quantificano direttamente il confondimento;
  - l'accordo sul solo termine di reference, che deve essere altissimo;
  - un calibratore: coseno fra vettori casuali in R^512, che vale ~0.

NOTA SUI MODELLI. ensemble_seed.py usa i checkpoint `multidominio_seed*_lam0.pt`, che non
sono in repo (sono .pt, gitignorati). Qui si usano i quattro modelli di `multiseed.py`
riaddestrati in locale, che sono la stessa architettura addestrata sullo stesso protocollo
con campioni di query diversi — e il seed 1 riproduce 26.53131539% di R@10, identico al
report multi-seed, quindi la replica e' fedele. Se la conclusione dipendesse dalla famiglia
di modelli, andra' rifatta sui suoi.

Il benchmark ufficiale NON viene toccato.

Uso: HF_HUB_OFFLINE=1 python esperimenti/accordo_canale_c.py
"""
import itertools
import json
import statistics as stat
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


STAMP = datetime.now().strftime("%Y%m%d-%H%M")
LOG_PATH = HERE / f"accordo_canale_c_{STAMP}.txt"
JSON_PATH = HERE / f"accordo_canale_c_{STAMP}.json"
_tee = Tee(LOG_PATH)
sys.stdout = _tee

SEEDS = (1, 2, 3, 4)
N_REF = 400
N_FUORI = 40
SEED_RNG = 7

print("=" * 100)
print("ACCORDO FRA I MODELLI — il canale delle condizioni, isolato dal canale della reference")
print("=" * 100)
print(f"log: {LOG_PATH.name}   json: {JSON_PATH.name}")

CKPT = ROOT / "contributo-candidato/checkpoints"
modelli, gammas = [], []
for sd in SEEDS:
    c = torch.load(CKPT / f"models_seed{sd}.pt", map_location="cpu")
    D, H = c["config"]["d"], c["config"]["hidden"]
    net = nn.Sequential(nn.Linear(2 * D, H), nn.GELU(), nn.Linear(H, D))
    net.load_state_dict({k[4:]: v for k, v in c["A"].items() if k.startswith("net.")})
    net.eval()
    modelli.append(net)
    gammas.append(float(c["A"]["gamma"]))
    dirs_text, attr_names = c["dirs_text"], list(c["attr_names"])
D = dirs_text.shape[1]
dtn = dirs_text / dirs_text.norm(dim=-1, keepdim=True)
print(f"\n{len(modelli)} modelli (seed {SEEDS}), gamma = {[round(g, 3) for g in gammas]}")
print(f"direzioni testuali {tuple(dtn.shape)}, D = {D}")

d = torch.load(ROOT / "embeddings/celeba_valid_clip_vitb32.pt", map_location="cpu")
E = d["embeddings"].float()
E = E / E.norm(dim=-1, keepdim=True)
R = E[torch.randperm(len(E), generator=torch.Generator().manual_seed(0))[:N_REF]]
print(f"reference campionate: {tuple(R.shape)}")

# base ortonormale dello span delle 40 direzioni, e direzioni casuali nel complemento
U_, S_, _ = torch.linalg.svd(dtn.T, full_matrices=False)
Q = U_[:, :int((S_ > 1e-5 * S_[0]).sum())]
g = torch.Generator().manual_seed(SEED_RNG)
cas = torch.randn(N_FUORI, D, generator=g)
fuori = cas - (cas @ Q) @ Q.T
fuori = fuori / fuori.norm(dim=-1, keepdim=True)
print(f"span delle direzioni testuali: {Q.shape[1]} dimensioni su {D}")
print(f"residuo massimo di ortogonalita' delle direzioni 'fuori': "
      f"{(fuori @ Q).abs().max():.2e}")

ZERO = torch.zeros(N_REF, D)


@torch.no_grad()
def pezzi(direzione):
    """Per ogni modello: correzione totale, termine solo-reference, contributo del testo."""
    c = direzione.expand(N_REF, -1)
    tot, base, iso = [], [], []
    for m, gm in zip(modelli, gammas):
        t = gm * m(torch.cat([R, c], dim=-1))
        b = gm * m(torch.cat([R, ZERO], dim=-1))
        tot.append(t)
        base.append(b)
        iso.append(t - b)
    return tot, base, iso


def coseno_medio(vs):
    """Coseno medio fra tutte le coppie di modelli, mediato sulle reference."""
    vn = [v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12) for v in vs]
    return stat.mean(float((vn[i] * vn[j]).sum(-1).mean())
                     for i, j in itertools.combinations(range(len(vn)), 2))


def analizza(direzioni, etichetta):
    a_tot, a_base, a_iso, quota = [], [], [], []
    for k in range(len(direzioni)):
        tot, base, iso = pezzi(direzioni[k])
        a_tot.append(coseno_medio(tot))
        a_base.append(coseno_medio(base))
        a_iso.append(coseno_medio(iso))
        quota.append(stat.mean(float((i_.norm(dim=-1) / t_.norm(dim=-1).clamp(min=1e-12)).mean())
                               for i_, t_ in zip(iso, tot)))
    return {"etichetta": etichetta, "accordo_totale": stat.mean(a_tot),
            "accordo_solo_reference": stat.mean(a_base), "accordo_canale_c": stat.mean(a_iso),
            "quota_norma_canale_c": stat.mean(quota), "n_direzioni": len(direzioni)}


print("\n" + "=" * 100)
print("RISULTATO")
print("=" * 100)
RIS = [analizza(dtn, "DENTRO lo span (i 40 attributi)"),
       analizza(fuori, "FUORI dallo span (casuali ortogonali)")]

print(f"{'direzioni':<38}{'totale':>10}{'solo ref':>11}{'canale c':>11}{'|c|/|tot|':>12}")
for r in RIS:
    print(f"{r['etichetta']:<38}{r['accordo_totale']:>+10.4f}"
          f"{r['accordo_solo_reference']:>+11.4f}{r['accordo_canale_c']:>+11.4f}"
          f"{r['quota_norma_canale_c']:>12.3f}")

# calibratore
gc = torch.Generator().manual_seed(123)
rc = torch.randn(2, N_REF, D, generator=gc)
cal = coseno_medio([rc[0], rc[1]])
print(f"\n  calibratore (vettori casuali indipendenti): {cal:+.4f}")

print("\n  totale     = la misura di ensemble_seed.py, non isolata")
print("  solo ref   = accordo sul termine net([r ; 0]), che non dipende dalle condizioni")
print("  canale c   = accordo su net([r ; c]) - net([r ; 0]), il contributo del testo")
print("  |c|/|tot|  = quanta della norma della correzione totale viene dal canale c")

# ===========================================================================================
# VERDETTO
# ===========================================================================================
print("\n" + "=" * 100)
print("VERDETTO")
print("=" * 100)
dentro, fuori_r = RIS[0], RIS[1]
riprodotta = abs(fuori_r["accordo_totale"] - 0.72) < 0.15 and \
             abs(dentro["accordo_totale"] - 0.60) < 0.15
print(f"  la misura originale e' riprodotta (~+0.72 fuori, ~+0.60 dentro)? "
      f"{'SI' if riprodotta else 'NO — attenzione, modelli diversi da quelli di Arianna'}")
crollo = fuori_r["accordo_totale"] - fuori_r["accordo_canale_c"]
print(f"  fuori dallo span, isolando il canale c l'accordo passa da "
      f"{fuori_r['accordo_totale']:+.4f} a {fuori_r['accordo_canale_c']:+.4f} "
      f"({-crollo:+.4f})")
print(f"  e il canale c pesa solo il {fuori_r['quota_norma_canale_c'] * 100:.1f}% della norma"
      f" della correzione, contro il {dentro['quota_norma_canale_c'] * 100:.1f}% dentro")
print()
if fuori_r["accordo_canale_c"] < 0.3 < fuori_r["accordo_totale"]:
    print("  IL CONFONDIMENTO C'ERA. Fuori dallo span i modelli NON concordano sul canale")
    print("  delle condizioni: l'accordo apparente veniva dal termine solo-reference, che e'")
    print("  condiviso. Quella parte e' quindi specifica del modello — l'idea di spegnerla")
    print("  (weight decay, o vincolare W_c allo span) torna in gioco, ed e' la sez. 6b di")
    print("  piano-interpretabilita.md, l'unica casella del piano mai provata.")
elif fuori_r["accordo_canale_c"] >= 0.3:
    print("  LA CONCLUSIONE DI ensemble_seed.py REGGE. Anche isolando il canale c i modelli")
    print("  concordano fuori dallo span: quella componente e' appresa e sistematica, non")
    print("  rumore di modello. Spegnerla non e' 'togliere rumore' ed e' un intervento")
    print("  dall'esito imprevedibile.")
else:
    print("  Quadro ambiguo: l'accordo era gia' basso anche nella misura non isolata.")

payload = {"data": datetime.now().isoformat(timespec="seconds"),
           "seeds": list(SEEDS), "gammas": gammas, "n_reference": N_REF,
           "dim_span": int(Q.shape[1]), "calibratore_casuale": cal, "risultati": RIS}
JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSalvato: {LOG_PATH.name} e {JSON_PATH.name}")
sys.stdout = _tee.stdout
_tee.close()
