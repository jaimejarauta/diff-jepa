"""Representation health checks, effective rank, and saliency contrast metrics."""

import math
import torch

def compute_effective_rank(Z: torch.Tensor) -> float:
    """Computes continuous effective rank via normalized singular value entropy.
    
    Optimized for tall-and-skinny matrices (M >= N): computes the Gram matrix covariance
    eigenvalues on device, eliminating MPS threadgroup memory overflow and CPU fallback.
    """
    with torch.no_grad():
        Z_centered = Z.float() - Z.float().mean(dim=0, keepdim=True)
        m, n = Z_centered.shape
        if m >= n:
            cov = torch.mm(Z_centered.T, Z_centered)
            eigvals = torch.linalg.eigvalsh(cov)
            S = torch.sqrt(torch.clamp(eigvals, min=0.0))
        else:
            cov = torch.mm(Z_centered, Z_centered.T)
            eigvals = torch.linalg.eigvalsh(cov)
            S = torch.sqrt(torch.clamp(eigvals, min=0.0))
        total = torch.sum(S)
        if total == 0:
            return 0.0
        p = S / total
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return math.exp(entropy.item())

def compute_contrast_ratio(saliency: torch.Tensor, fg_mask: torch.Tensor, bg_mask: torch.Tensor) -> float:
    """Computes gradient energy ratio on foreground affordance vs background clutter."""
    with torch.no_grad():
        fg_energy = torch.mean(saliency[fg_mask])
        bg_energy = torch.mean(saliency[bg_mask])
        return (fg_energy / max(1e-8, bg_energy)).item()

