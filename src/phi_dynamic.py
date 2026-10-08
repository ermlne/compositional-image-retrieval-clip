"""Phi dinamico multi-condizione (notebook 08).

ESITO: risultato negativo. Il modulo impara (la loss InfoNCE scende), ma Recall@10
scende *insieme* alla loss: i due criteri sono anti-correlati su questo task, e il
checkpoint migliore resta l'inizializzazione a pesi uguali. Vedi 08_phi_dynamic.ipynb
per la diagnosi. Il codice resta qui perche' l'esperimento e' riproducibile e la
parametrizzazione e' riutilizzabile per una loss meglio allineata alla metrica.

---

Ogni condizione riceve un peso proprio, predetto da feature geometriche
invarianti all'identita' dell'attributo (cosi' il modulo generalizza a
combinazioni mai viste); la lunghezza del passo e' predetta a parte.

    u_i      direzione della condizione i, con segno, ripulita dallo span
             degli attributi non interrogati
    w_i      = softplus(h(f_i))            peso della condizione
    d        = somma_i w_i * u_i           fusione pesata (non uno stacking)
    alpha    = alpha_max * sigmoid(g(F))   lunghezza del passo
    q        = normalize(r + alpha * d/||d||)

Feature per condizione f_i: allineamento r.u_i, segno, residuo dopo la pulizia,
conflitto massimo con le altre condizioni, numero di condizioni.
Feature globali F: media e minimo degli allineamenti, residuo medio, |T+|, |T-|.
"""
import torch
import torch.nn as nn


class PhiDynamic(nn.Module):
    def __init__(self, n_feat=5, n_gfeat=5, hidden=32, alpha_max=6.0):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(n_feat, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 1))
        self.step = nn.Sequential(nn.Linear(n_gfeat, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 1))
        self.alpha_max = alpha_max
        # partenza neutra: pesi ~1 per ogni condizione, passo ~meta' del massimo
        nn.init.zeros_(self.cond[-1].weight); nn.init.constant_(self.cond[-1].bias, 0.5413)
        nn.init.zeros_(self.step[-1].weight); nn.init.zeros_(self.step[-1].bias)

    def weights(self, feat, mask):
        w = nn.functional.softplus(self.cond(feat).squeeze(-1))     # [B, n]
        return w * mask

    def forward(self, ref, U, feat, mask, gfeat):
        w = self.weights(feat, mask)                                # [B, n]
        d = (w.unsqueeze(-1) * U).sum(1)                            # [B, D]
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha = self.alpha_max * torch.sigmoid(self.step(gfeat))    # [B, 1]
        q = ref + alpha * d
        return q / q.norm(dim=-1, keepdim=True), w, alpha.squeeze(-1)


def info_nce(q, E, pos, neg, tau=0.01):
    """InfoNCE multi-positivo: i ground truth contro i distrattori estratti col mining."""
    sim_p = torch.einsum('bd,bkd->bk', q, E[pos.clamp(min=0)]) / tau
    sim_n = torch.einsum('bd,bkd->bk', q, E[neg.clamp(min=0)]) / tau
    sim_p = sim_p.masked_fill(pos < 0, -1e4)
    sim_n = sim_n.masked_fill(neg < 0, -1e4)
    num = torch.logsumexp(sim_p, dim=1)
    den = torch.logsumexp(torch.cat([sim_p, sim_n], dim=1), dim=1)
    return -(num - den).mean()
