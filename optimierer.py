"""Muon: Optimierer, der Matrizen als Ganzes behandelt.

AdamW skaliert jeden Gewichtswert einzeln anhand seiner eigenen Historie. Bei
einer Matrix, die eine lineare Abbildung darstellt, ist das nicht die
natuerliche Sichtweise - dort zaehlt, in welche Richtungen abgebildet wird.

Muon dreht den Gradienten deshalb so, dass alle Richtungen aehnlich stark
aktualisiert werden (Orthogonalisierung), bevor er ihn anwendet. Das geschieht
ueber eine Newton-Schulz-Iteration: fuenf Matrixmultiplikationen, verglichen
mit dem Vorwaertslauf vernachlaessigbar.

Wichtig und in unseren Messungen der haeufigste Fehler: Muon braucht eine
EIGENE, viel groessere Lernrate. Weil die Orthogonalisierung die Schrittgroesse
vom Gradienten entkoppelt, steckt die gesamte Schrittweite in der Lernrate.
Gemessen: 6e-4 scheitert (+82 %), 2e-2 gewinnt (-11 %).

Eindimensionale Groessen - Normierungen, Embedding - bleiben bei AdamW.
"""

import torch


def orthogonalisiere(G: torch.Tensor, schritte: int = 5) -> torch.Tensor:
    """Naehert eine Matrix an die naechstgelegene orthogonale an."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    quer = X.size(0) > X.size(1)
    if quer:
        X = X.T
    for _ in range(schritte):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return (X.T if quer else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-2, momentum=0.95, weight_decay=0.1):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                z = self.state.setdefault(p, {})
                if "puffer" not in z:
                    z["puffer"] = torch.zeros_like(p)
                z["puffer"].mul_(g["momentum"]).add_(p.grad)
                u = orthogonalisiere(z["puffer"])
                # Ausgleich fuer nicht-quadratische Matrizen
                u = u * max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u, alpha=-g["lr"])


def baue(netz, lr, muon_lr=None, weight_decay=0.1):
    """Gibt die Optimierer und ihre Grundlernraten zurueck.

    Ohne muon_lr ein einzelner AdamW ueber alles - wie bisher.
    """
    if not muon_lr:
        opt = torch.optim.AdamW(netz.parameters(), lr=lr, betas=(0.9, 0.95),
                                weight_decay=weight_decay)
        return [opt], [lr]
    matrizen = [p for n, p in netz.named_parameters()
                if p.ndim == 2 and "tok_emb" not in n and "flash_tab" not in n]
    rest = [p for n, p in netz.named_parameters()
            if not (p.ndim == 2 and "tok_emb" not in n and "flash_tab" not in n)]
    return ([Muon(matrizen, lr=muon_lr, weight_decay=weight_decay),
             torch.optim.AdamW(rest, lr=lr, betas=(0.9, 0.95),
                               weight_decay=weight_decay)],
            [muon_lr, lr])
