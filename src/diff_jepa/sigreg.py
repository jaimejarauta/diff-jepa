"""Sketched Isotropic Gaussian Regularization (SIGReg).
Following Balestriero & LeCun (arXiv:2511.08544).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGRegLoss(nn.Module):
    """Sketched Isotropic Gaussian Regularization with dynamic random projection slices.
    Resamples unit slices dynamically every forward pass to regularize the entire S^{d-1} sphere.
    """
    def __init__(self, d: int, n_slices: int = 32, resample: bool = True):
        super().__init__()
        self.d = d
        self.n_slices = n_slices
        self.resample = resample
        # Initial projection buffer
        self.register_buffer("proj", F.normalize(torch.randn(d, n_slices), dim=0))
        # Test evaluation grid
        t_vals = torch.linspace(0.1, 2.0, 10)
        self.register_buffer("t_vals", t_vals)
        self.register_buffer("exp_std", torch.exp(-0.5 * (t_vals ** 2)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Computes differentiable Epps-Pulley empirical characteristic function distance."""
        # Fresh normalized random directions every call when resample=True
        if self.resample:
            proj = F.normalize(torch.randn(self.d, self.n_slices, device=z.device, dtype=z.dtype), dim=0)
        else:
            proj = self.proj if (self.proj.device == z.device and self.proj.dtype == z.dtype) else self.proj.to(z.device, z.dtype)
        t_vals = self.t_vals if (self.t_vals.device == z.device and self.t_vals.dtype == z.dtype) else self.t_vals.to(z.device, z.dtype)
        exp_std = self.exp_std if (self.exp_std.device == z.device and self.exp_std.dtype == z.dtype) else self.exp_std.to(z.device, z.dtype)

        # Sketched 1D slices: (batch, n_slices)
        sketched = torch.mm(z, proj)
        # Empirical characteristic function evaluation against standard normal N(0, 1)
        cos_terms = torch.cos(sketched.unsqueeze(-1) * t_vals)
        ecf = cos_terms.mean(dim=0)
        diff = (ecf - exp_std).pow(2).mean()
        return diff

def compute_sigreg_epps_pulley(z: torch.Tensor, n_slices: int = 32) -> torch.Tensor:
    """Functional API for differentiable normality test statistic via random 1D projections."""
    d = z.size(-1)
    proj = F.normalize(torch.randn(d, n_slices, device=z.device, dtype=z.dtype), dim=0)
    t_vals = torch.linspace(0.1, 2.0, 10, device=z.device, dtype=z.dtype)
    exp_std = torch.exp(-0.5 * (t_vals ** 2))
    sketched = torch.mm(z, proj)
    cos_terms = torch.cos(sketched.unsqueeze(-1) * t_vals)
    ecf = cos_terms.mean(dim=0)
    return (ecf - exp_std).pow(2).mean()
    sketched = z @ proj
    t_vals = torch.linspace(0.1, 2.0, 10, device=z.device)
    exp_std = torch.exp(-0.5 * (t_vals ** 2))
    cos_terms = torch.cos(sketched.unsqueeze(-1) * t_vals)
    ecf = cos_terms.mean(dim=0)
    return (ecf - exp_std).pow(2).mean()

