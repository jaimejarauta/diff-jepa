"""Sketched Isotropic Gaussian Regularization (SIGReg).
Following Balestriero & LeCun (arXiv:2511.08544).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGRegLoss(nn.Module):
    """Sketched Isotropic Gaussian Regularization with persistent orthogonal projection buffer.
    Guarantees stable, reproducible 1D slices across mini-batches.
    """
    def __init__(self, d: int, n_slices: int = 32):
        super().__init__()
        self.d = d
        self.n_slices = n_slices
        # Generate persistent orthogonal projection slice matrix via QR decomposition
        with torch.no_grad():
            rnd = torch.randn(d, n_slices)
            q, _ = torch.linalg.qr(rnd)
        self.register_buffer("proj", q)
        
        # Test evaluation grid
        t_vals = torch.linspace(0.1, 2.0, 10)
        self.register_buffer("t_vals", t_vals)
        self.register_buffer("exp_std", torch.exp(-0.5 * (t_vals ** 2)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Computes differentiable Epps-Pulley empirical characteristic function distance."""
        # Sketched 1D slices: (batch, n_slices)
        sketched = z @ self.proj.to(z.device, z.dtype)
        # Empirical characteristic function evaluation against standard normal N(0, 1)
        cos_terms = torch.cos(sketched.unsqueeze(-1) * self.t_vals.to(z.device, z.dtype))
        ecf = cos_terms.mean(dim=0)
        diff = (ecf - self.exp_std.to(z.device, z.dtype)).pow(2).mean()
        return diff

def compute_sigreg_epps_pulley(z: torch.Tensor, n_slices: int = 32) -> torch.Tensor:
    """Functional API for differentiable normality test statistic via random 1D projections."""
    d = z.size(-1)
    with torch.no_grad():
        rnd = torch.randn(d, n_slices, device=z.device)
        proj, _ = torch.linalg.qr(rnd)
    sketched = z @ proj
    t_vals = torch.linspace(0.1, 2.0, 10, device=z.device)
    exp_std = torch.exp(-0.5 * (t_vals ** 2))
    cos_terms = torch.cos(sketched.unsqueeze(-1) * t_vals)
    ecf = cos_terms.mean(dim=0)
    return (ecf - exp_std).pow(2).mean()

