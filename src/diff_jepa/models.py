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

class ConvVisualEncoder(nn.Module):
    """Convolutional visual encoder mapping (B, C, H, W) camera frames or (B, T, C, H, W)
    video sequences to compact physical state latent representations.
    """
    def __init__(self, in_channels: int = 3, d_state: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.d_state = d_state

        # 4-stage convolutional hierarchy
        self.backbone = nn.Sequential(
            # Stage 1: (C, 64, 64) -> (32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            # Stage 2: (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Stage 3: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            # Stage 4: (128, 8, 8) -> (128, 4, 4)
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            # Keep a 4x4 location grid (any input size) instead of global pooling, which discarded position
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(32 * 4 * 4, d_state),
            nn.LayerNorm(d_state)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, H, W) or (B, T, C, H, W) image tensor.
        Returns:
            (B, d_state) or (B, T, d_state) latent representation.
        """
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x_flat = x.view(b * t, c, h, w)
            feats = self.backbone(x_flat)
            out = self.proj(feats)
            return out.view(b, t, -1)
        elif x.dim() == 4:
            feats = self.backbone(x)
            return self.proj(feats)
        else:
            raise ValueError(f"Expected 4D (B, C, H, W) or 5D (B, T, C, H, W) tensor, got {x.shape}")

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
        
        # Raw Euclidean scale preserved; no batch standardization (which distorts geometry and induces batch dependency)
        return diff

