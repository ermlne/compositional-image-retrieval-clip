#!/usr/bin/env python3
"""Il meccanismo asimmetrico sopravvive su attributi che il modello non ha mai visto?

`antisimmetria_e_ablazione.py` mostra che il modulo tratta "+X" e "-X" in modo asimmetrico,
e che una mappa lineare non puo' farlo (per lei la risposta a "-X" e' l'opposto esatto di
quella a "+X"). Ma l'asimmetria e' una proprieta' del MODELLO o dei 40 attributi su cui e'
stato addestrato?

Questo non e' il test di generalizzazione in Recall — quello richiede di riaddestrare
tenendo fuori degli attributi (`contributo-candidato/attributi_heldout.py`). Qui si chiede
qualcosa di piu' economico e diverso: dato in pasto al modello congelato un attributo di un
ALTRO dominio, il comportamento asimmetrico si ripresenta o svanisce?

Le direzioni arrivano da `esperimenti/frecce_tre_domini.pt` (prodotto da
`relazione_testo_visione.py`): 40 CelebA + 85 AwA2 + 102 OVAD, tutte unitarie e costruite
con la stessa procedura. Verificato che le 40 CelebA coincidono con le `d_ens`
dell'addestramento (coseno 1.0000), quindi fra i gruppi cambia solo "viste / mai viste".

Uso:
    python esperimenti/generalizza_meccanismo.py CACHE CKPT_DIR [--tag v3]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

N_REF = 500


def carica_membri(ckpt_dir, D=512):
    fuori = []
    for p in sorted(Path(ckpt_dir).glob("*.pt")):
        if not re.search(r"_off(_cx)?\.pt$", p.name):
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)["A"]
        H = sd["net.0.weight"].shape[0]
        net = nn.Sequential(nn.Linear(2 * D, H), nn.GELU(), nn.Linear(H, D))
        net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")})
        net.eval()
        fuori.append((p.name, net, float(sd["gamma"])))
    return fuori


def per_direzione(net, R, dirs):
    """Per ogni direzione: indice di antisimmetria e norma della risposta al testo."""
    zero = torch.zeros(len(R), R.shape[1])
    with torch.no_grad():
        base = net(torch.cat([R, zero], 1))
        idx, norma = [], []
        for k in range(len(dirs)):
            tp = net(torch.cat([R, dirs[k].expand(len(R), -1)], 1)) - base
            tm = net(torch.cat([R, (-dirs[k]).expand(len(R), -1)], 1)) - base
            idx.append(float(((tp + tm).norm(dim=-1)
                              / (tp - tm).norm(dim=-1).clamp(min=1e-9)).mean()))
            norma.append(float(tp.norm(dim=-1).mean()))
    return np.array(idx), np.array(norma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("ckpt_dir")
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()
    torch.set_num_threads(min(8, __import__("os").cpu_count() or 4))

    fr = torch.load(ROOT / "esperimenti/frecce_tre_domini.pt", map_location="cpu",
                    weights_only=False)
    T, dom = fr["T"], np.array(fr["dom"])
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    E_fit, d_ens = cache["E_fit"], cache["d_ens"]
    D = E_fit.shape[1]
    g = torch.Generator().manual_seed(0)
    R = E_fit[torch.randperm(len(E_fit), generator=g)[:N_REF]]

    # quota di ogni direzione che cade DENTRO il sottospazio delle 40 viste (sez. 4ter)
    Q, _ = torch.linalg.qr(d_ens.T)                      # [512, 40]
    in_span = ((T @ Q).norm(dim=-1) / T.norm(dim=-1)).numpy()

    membri = carica_membri(args.ckpt_dir, D)
    print(f"sessione {args.tag}: {len(membri)} membri, {len(T)} direzioni "
          f"({(dom == 'celeba').sum()} viste, {(dom != 'celeba').sum()} mai viste)\n")

    IDX = np.zeros((len(membri), len(T)))
    NRM = np.zeros((len(membri), len(T)))
    for i, (_, net, _) in enumerate(membri):
        IDX[i], NRM[i] = per_direzione(net, R, T)
    idx, nrm = IDX.mean(0), NRM.mean(0)

    RIS = {"prodotto_da": "esperimenti/generalizza_meccanismo.py", "tag": args.tag,
           "cache": args.cache, "ckpt_dir": args.ckpt_dir, "n_membri": len(membri),
           "n_reference": N_REF, "gruppi": {}}
    print("indice di antisimmetria: 0 = il modello si comporta come una matrice")
    print(f"{'gruppo':>26}{'n':>5}{'quota in-span':>15}{'indice':>10}{'norma risposta':>17}")
    for nome, sel in (("CelebA (VISTI)", dom == "celeba"), ("AwA2 (mai visti)", dom == "awa2"),
                      ("OVAD (mai visti)", dom == "ovad"), ("tutti i MAI VISTI", dom != "celeba")):
        RIS["gruppi"][nome] = {"n": int(sel.sum()), "quota_in_span": float(in_span[sel].mean()),
                               "indice_antisimmetria": float(idx[sel].mean()),
                               "norma_risposta": float(nrm[sel].mean())}
        print(f"{nome:>26}{int(sel.sum()):>5}{in_span[sel].mean():>15.3f}"
              f"{idx[sel].mean():>10.4f}{nrm[sel].mean():>17.4f}")

    from scipy.stats import mannwhitneyu, spearmanr
    a, b = idx[dom == "celeba"], idx[dom != "celeba"]
    u = mannwhitneyu(a, b, alternative="greater")
    print(f"\nvisti piu' asimmetrici dei mai visti: p = {u.pvalue:.2e}  "
          f"({a.mean():.4f} contro {b.mean():.4f})")
    m = dom != "celeba"
    r = spearmanr(in_span[m], idx[m])
    print(f"fra i soli MAI VISTI, l'asimmetria segue la quota in-span: "
          f"rho = {r.statistic:+.3f}  p = {r.pvalue:.4f}")
    rn = spearmanr(in_span[m], nrm[m])
    print(f"                        e la norma della risposta:          "
          f"rho = {rn.statistic:+.3f}  p = {rn.pvalue:.4f}")
    RIS["visti_contro_mai_visti"] = {
        "p_mannwhitney": float(u.pvalue), "indice_visti": float(a.mean()),
        "indice_mai_visti": float(b.mean()),
        "in_span_predice_asimmetria": {"rho": float(r.statistic), "p": float(r.pvalue)},
        "in_span_predice_norma": {"rho": float(rn.statistic), "p": float(rn.pvalue)},
        "membri_concordi": int(sum(IDX[i][dom == "celeba"].mean()
                                   > IDX[i][dom != "celeba"].mean() for i in range(len(membri)))),
        "n_membri": len(membri)}
    out = HERE / f"generalizza_meccanismo_{args.tag.split()[0]}.json"
    json.dump(RIS, open(out, "w"), indent=1)
    print(f"scritto {out.relative_to(ROOT)}")
    print(f"\nmembri concordi sul segno (visti > mai visti): "
          f"{int(sum(IDX[i][dom == 'celeba'].mean() > IDX[i][dom != 'celeba'].mean() for i in range(len(membri))))}"
          f"/{len(membri)}")


if __name__ == "__main__":
    main()
