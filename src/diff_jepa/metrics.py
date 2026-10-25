"""Representation health checks, effective rank, and saliency contrast metrics."""

import math
import torch

def compute_effective_rank(Z: torch.Tensor) -> float:
    """Computes continuous effective rank via normalized singular value entropy."""
    with torch.no_grad():
        Z_centered = Z.float() - Z.float().mean(dim=0, keepdim=True)
        S = torch.linalg.svdvals(Z_centered)
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
