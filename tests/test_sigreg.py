"""Unit tests for Sketched Isotropic Gaussian Regularization (SIGReg)."""

import torch
from diff_jepa.sigreg import compute_sigreg_epps_pulley
from diff_jepa.metrics import compute_effective_rank

def test_sigreg_standard_normal():
    # Isotropic Gaussian samples should have small SIGReg loss
    torch.manual_seed(42)
    z_normal = torch.randn(512, 64)
    loss_normal = compute_sigreg_epps_pulley(z_normal).item()
    
    # Degenerate collapsed samples should have much higher SIGReg loss
    z_collapsed = torch.ones(512, 64) * 2.0
    loss_collapsed = compute_sigreg_epps_pulley(z_collapsed).item()
    
    assert loss_normal < loss_collapsed, f"Expected normal loss ({loss_normal}) < collapsed loss ({loss_collapsed})"

def test_effective_rank_computation():
    torch.manual_seed(42)
    # Full rank random Gaussian
    z_full = torch.randn(200, 32)
    erank_full = compute_effective_rank(z_full)
    
    # Rank 1 matrix
    v = torch.randn(200, 1)
    w = torch.randn(1, 32)
    z_rank1 = v @ w
    erank_rank1 = compute_effective_rank(z_rank1)
    
    assert erank_full > 20.0, f"Expected full erank > 20, got {erank_full}"
    assert erank_rank1 < 2.0, f"Expected rank 1 erank < 2, got {erank_rank1}"
