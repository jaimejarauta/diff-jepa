"""Sketched Isotropic Gaussian Regularization (SIGReg).
Following Balestriero & LeCun (arXiv:2511.08544).
"""

import torch
import torch.nn.functional as F

def compute_sigreg_epps_pulley(z: torch.Tensor, n_slices: int = 32) -> torch.Tensor:
    """Computes differentiable normality test statistic via random 1D projections."""
    d = z.size(-1)
    # Fixed orthogonal/normalized random projections
    proj = torch.randn(d, n_slices, device=z.device)
    proj = F.normalize(proj, dim=0)
    sketched = z @ proj  # (batch, n_slices)

    # Empirical characteristic function evaluation against standard normal N(0, 1)
    t_vals = torch.linspace(0.1, 2.0, 10, device=z.device)
    exp_std = torch.exp(-0.5 * (t_vals ** 2))
    cos_terms = torch.cos(sketched.unsqueeze(-1) * t_vals)
    ecf = cos_terms.mean(dim=0)
    diff = (ecf - exp_std).pow(2).mean()
    return diff
