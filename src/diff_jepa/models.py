"""Action-Asymmetric Dual-Head Predictor for Diff-JEPA.
See notes/01_causal_identifiability.md for architectural theorems and proofs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualStateEncoder(nn.Module):
    """Visual/sensory encoder mapping raw observations to state latent representations."""
    def __init__(self, d_raw: int, d_state: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_raw, d_state * 2),
            nn.LayerNorm(d_state * 2),
            nn.GELU(),
            nn.Linear(d_state * 2, d_state),
            nn.LayerNorm(d_state)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ActionAsymmetricDualPredictor(nn.Module):
    """Predictor with an action-conditioned causal head and an unconditioned passive background head."""
    def __init__(self, d_state: int, d_action: int, d_out: int,
                 lambda_min: float = 0.20, lambda_max: float = 0.85):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.d_out = d_out
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Head 1 (Causal): Sees full state + action sequence
        self.causal_head = nn.Sequential(
            nn.Linear(d_state + d_action, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out)
        )
        # Head 2 (Passive Background): Sees state + zero-action (isolated from action dynamics)
        self.passive_head = nn.Sequential(
            nn.Linear(d_state + d_action, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out)
        )

        # Learnable clamped lambda parameter (initialized around 0.70)
        self.raw_lambda = nn.Parameter(torch.tensor(1.0))

    def get_lambda(self) -> torch.Tensor:
        """Computes clamped sigmoidal lambda in [lambda_min, lambda_max]."""
        return self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(self.raw_lambda)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Computes action-asymmetric differential representation with standardized variance."""
        zero_action = torch.zeros_like(action)
        z1 = self.causal_head(torch.cat([state, action], dim=-1))
        z2 = self.passive_head(torch.cat([state, zero_action], dim=-1))
        
        lam = self.get_lambda()
        diff = z1 - lam * z2
        
        # Standardize mean and standard deviation per feature so 1D projections match N(0, 1) variance for SIGReg
        # (Avoid spherical F.normalize which collapses 1D projection variance to 1/D)
        diff_mean = diff.mean(dim=0, keepdim=True)
        diff_std = diff.std(dim=0, keepdim=True, correction=0) + 1e-5
        return (diff - diff_mean) / diff_std

