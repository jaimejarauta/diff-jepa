"""Diff-JEPA: Action-Asymmetric Differential Isotropic JEPA."""

from .models import ActionAsymmetricDualPredictor
from .sigreg import compute_sigreg_epps_pulley
from .metrics import compute_effective_rank, compute_contrast_ratio

__all__ = [
    "ActionAsymmetricDualPredictor",
    "compute_sigreg_epps_pulley",
    "compute_effective_rank",
    "compute_contrast_ratio",
]
