#!/usr/bin/env python3
"""Perche' la forma lineare fallisce sui vincoli negativi, e da dove viene il guadagno.

Due misure, sullo stesso modello e con lo stesso caricamento.

ESPERIMENTO A — ANTISIMMETRIA
La forma collassata e' `W_r r + W_c c + b`, quindi ribaltando il segno della condizione la
correzione si ribalta esattamente: la risposta a "-X" e' OBBLIGATA a essere l'opposto della
risposta a "+X". Il compito pero' non e' simmetrico ("+occhiali" e' una regione, "-occhiali"
e' tutto il resto), quindi una mappa lineare deve sbagliare proprio sui negativi.

Si misura quanto l'MLP si discosta da quella simmetria imposta. Isolando il canale del testo
  t(r, c) = net([r ; c]) - net([r ; 0])
l'indice e'
  asimmetria = || t(r, +d) + t(r, -d) ||  /  || t(r, +d) - t(r, -d) ||
che vale ESATTAMENTE 0 per qualunque mappa lineare, e cresce con la parte simmetrica.

ESPERIMENTO B — ABLAZIONE DELLA REFERENCE
Protocollo di `esperimenti/ablazione_r.py`: si sostituisce la reference IN INGRESSO all'MLP
con la reference media del train, lasciando la correzione sommata alla reference vera. A ms 80
il guadagno crollava da +7.35 a +2.66pp, cioe' il 64% del guadagno veniva dalla dipendenza
dalla foto. Qui si rifa' al punto di lavoro nuovo, e stratificato per segno del vincolo:
se la parte non lineare vive nel canale della reference, l'ablazione deve danneggiare i
vincoli NEGATIVI piu' dei positivi.

Uso:
    python esperimenti/antisimmetria_e_ablazione.py CACHE CKPT_DIR [--tag v3] [--vecchio]
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

from src.benchmark import load_benchmark                       # noqa: E402
from src.devset import bootstrap_paired, evaluate_per_source    # noqa: E402

N_REF = 1000        # reference su cui si misura l'antisimmetria (come sez. 4bis)


def carica_membri(ckpt_dir, D=512):
    """(nome, net, gamma) per ogni membro. `net` e `gamma` separati: il bersaglio del
    collasso lineare e' `gamma * net(...)`, e tenerli distinti evita di moltiplicarlo due
    volte (errore silenzioso gia' capitato nel progetto)."""
    fuori = []
    for p in sorted(Path(ckpt_dir).glob("*.pt")):
        if not re.search(r"_off(_cx)?\.pt$", p.name):
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        sd = d["A"]
        H = sd["net.0.weight"].shape[0]
        net = nn.Sequential(nn.Linear(2 * D, H), nn.GELU(), nn.Linear(H, D))
        net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")})
        net.eval()
        fuori.append((p.name, net, float(sd["gamma"])))
    return fuori


def antisimmetria(net, R, d_ens):
    """Indice di asimmetria e rapporto delle lunghezze, mediati sui 40 attributi."""
    zero = torch.zeros(len(R), R.shape[1])
    with torch.no_grad():
        base = net(torch.cat([R, zero], 1))
        idx, rap = [], []
        for a in range(len(d_ens)):
            tp = net(torch.cat([R, d_ens[a].expand(len(R), -1)], 1)) - base
            tm = net(torch.cat([R, (-d_ens[a]).expand(len(R), -1)], 1)) - base
            idx.append(float(((tp + tm).norm(dim=-1) / (tp - tm).norm(dim=-1)
                              .clamp(min=1e-9)).mean()))
            rap.append(float((tm.norm(dim=-1) / tp.norm(dim=-1).clamp(min=1e-9)).mean()))
    return float(np.mean(idx)), float(np.mean(rap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("ckpt_dir")
    ap.add_argument("--tag", default="?")
    ap.add_argument("--vecchio", action="store_true",
                    help="aggiunge il modello a max_sources 80 come termine di paragone")
    args = ap.parse_args()
    torch.set_num_threads(min(8, __import__("os").cpu_count() or 4))

    dati = torch.load(args.cache, map_location="cpu", weights_only=False)
    E_fit, d_ens, ATTR = dati["E_fit"], dati["d_ens"], dati["attr"]
    D = E_fit.shape[1]
    g = torch.Generator().manual_seed(0)
    R = E_fit[torch.randperm(len(E_fit), generator=g)[:N_REF]]

    membri = carica_membri(args.ckpt_dir, D)
    print(f"sessione {args.tag}: {len(membri)} membri\n")
    RIS = {"prodotto_da": "esperimenti/antisimmetria_e_ablazione.py", "tag": args.tag,
           "cache": args.cache, "ckpt_dir": args.ckpt_dir,
           "n_reference": N_REF, "antisimmetria": {}, "ablazione_reference": {}}

    # ============================ A — ANTISIMMETRIA ========================================
    print("=" * 86)
    print("A — QUANTO L'MLP SI DISCOSTA DALL'ANTISIMMETRIA CHE UNA MATRICE GLI IMPORREBBE")
    print("=" * 86)
    print("indice 0.000 = perfettamente antisimmetrico, cioe' indistinguibile da lineare\n")
    print(f"{'modello':>34}{'indice':>10}{'||t(-d)|| / ||t(+d)||':>24}")
    vals = []
    for nome, net, _ in membri:
        i, r = antisimmetria(net, R, d_ens)
        vals.append(i)
        RIS["antisimmetria"][nome] = {"indice": i, "rapporto_lunghezze": r}
        print(f"{nome[:34]:>34}{i:>10.4f}{r:>24.4f}")
    print(f"{'MEDIA':>34}{np.mean(vals):>10.4f}"
          f"   (sd fra membri {np.std(vals, ddof=1):.4f})")

    if args.vecchio:
        p = ROOT / "contributo-candidato/checkpoints/models_seed2026.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        H = ck["A"]["net.0.weight"].shape[0]
        net = nn.Sequential(nn.Linear(2 * D, H), nn.GELU(), nn.Linear(H, D))
        net.load_state_dict({k[4:]: v for k, v in ck["A"].items() if k.startswith("net.")})
        net.eval()
        i, r = antisimmetria(net, R, ck["dirs_text"])
        RIS["antisimmetria"]["VECCHIO_max_sources_80"] = {"indice": i, "rapporto_lunghezze": r}
        print(f"{'VECCHIO (max_sources 80)':>34}{i:>10.4f}{r:>24.4f}")

    # ============================ B — ABLAZIONE ============================================
    print("\n" + "=" * 86)
    print("B — IL GUADAGNO VIENE DALLA FOTO? (reference in ingresso -> reference media)")
    print("=" * 86)

    ev = torch.load(ROOT / "embeddings/celeba_valid_clip_vitb32.pt", map_location="cpu")
    E = ev["embeddings"].float()
    E = E / E.norm(dim=-1, keepdim=True)
    dev = load_benchmark(str(ROOT / "celeba_devset_valid.json"), ATTR)

    dtr = torch.load(ROOT / "embeddings/celeba_train_clip_vitb32.pt", map_location="cpu")
    Etr = dtr["embeddings"].float()
    Etr = Etr / Etr.norm(dim=-1, keepdim=True)
    rbar = Etr.mean(0)
    rbar = rbar / rbar.norm()
    del dtr, Etr

    cs = {}
    for e in dev:
        c = torch.zeros(D)
        for a in e["pos"]:
            c = c + d_ens[a]
        for b in e["neg"]:
            c = c - d_ens[b]
        cs[e["query"]] = c
    n_cond = np.array([len(e["queried"]) for e in dev])
    n_neg = np.array([len(e["neg"]) for e in dev])
    POS1 = np.where((n_cond == 1) & (n_neg == 0))[0]
    NEG1 = np.where((n_cond == 1) & (n_neg == 1))[0]
    TUTTE = np.arange(len(dev))

    def build_baseline(ref, e):
        q = ref + cs[e["query"]]
        return q / q.norm(dim=-1, keepdim=True)

    def build(usa_ref_vera):
        def f(ref, e):
            c = cs[e["query"]].expand(len(ref), -1)
            r_in = ref if usa_ref_vera else rbar.expand(len(ref), -1)
            with torch.no_grad():
                dl = torch.stack([gm * nt(torch.cat([r_in, c], 1))
                                  for _, nt, gm in membri]).mean(0)
            q = ref + dl                      # la correzione si somma SEMPRE alla ref vera
            return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return f

    def val(build_fn):
        return evaluate_per_source(build_fn, dev, E, [10])[10]

    ps_b, ps_v, ps_a = val(build_baseline), val(build(True)), val(build(False))
    liv = lambda ps, I: 100 * np.mean([ps[i].mean() for i in I])   # noqa: E731

    print(f"{'strato':>12}{'n':>5}{'baseline':>10}{'ablato':>9}{'intero':>9}"
          f"{'guadagno ablato':>18}{'guadagno intero':>18}{'quota persa':>13}")
    for nome, I in (("+attributo", POS1), ("-attributo", NEG1), ("TUTTE", TUTTE)):
        gb, ga, gv = liv(ps_b, I), liv(ps_a, I), liv(ps_v, I)
        persa = 100 * (1 - (ga - gb) / (gv - gb)) if gv > gb else float("nan")
        RIS["ablazione_reference"][nome] = {
            "n": len(I), "baseline": gb, "ablato": ga, "intero": gv,
            "guadagno_ablato_pp": ga - gb, "guadagno_intero_pp": gv - gb,
            "quota_persa_pct": persa}
        print(f"{nome:>12}{len(I):>5}{gb:>9.2f}%{ga:>8.2f}%{gv:>8.2f}%"
              f"{ga - gb:>+17.2f}pp{gv - gb:>+17.2f}pp{persa:>12.1f}%")

    r = bootstrap_paired([ps_v[i] for i in TUTTE], [ps_a[i] for i in TUTTE],
                         n_boot=20000, seed=0)
    print(f"\nintero contro ablato, su tutte le query: {100 * r['delta']:+.2f}pp  "
          f"IC [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]  p={r['p']:.4f}")
    RIS["intero_contro_ablato"] = {"delta_pp": 100 * r["delta"], "lo": 100 * r["lo"],
                                   "hi": 100 * r["hi"], "p": r["p"]}
    RIS["indice_medio_membri"] = float(np.mean(vals))
    print("riferimento a max_sources 80 (esperimenti/ablazione_r.py): "
          "+7.35 -> +2.66pp, cioe' il 64% del guadagno perso")
    out = HERE / f"antisimmetria_e_ablazione_{args.tag.split()[0]}.json"
    json.dump(RIS, open(out, "w"), indent=1)
    print(f"scritto {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
