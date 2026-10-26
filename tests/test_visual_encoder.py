"""Tests for ConvVisualEncoder and multi-modal vision-action forward pass."""

import torch
import pytest
from diff_jepa.models import ConvVisualEncoder, ActionAsymmetricDualPredictor

def test_conv_visual_encoder_4d_forward():
    """Verify ConvVisualEncoder processes 4D (B, C, H, W) batches into (B, d_state)."""
    enc = ConvVisualEncoder(in_channels=3, d_state=64)
    x = torch.rand(4, 3, 64, 64)
    out = enc(x)
    assert out.shape == (4, 64)
    assert not torch.isnan(out).any()

def test_conv_visual_encoder_5d_sequence_forward():
    """Verify ConvVisualEncoder processes 5D (B, T, C, H, W) video sequences into (B, T, d_state)."""
    enc = ConvVisualEncoder(in_channels=3, d_state=64)
    x = torch.rand(2, 5, 3, 64, 64)
    out = enc(x)
    assert out.shape == (2, 5, 64)
    assert not torch.isnan(out).any()

def test_conv_visual_encoder_gradient_flow():
    """Verify gradients propagate smoothly through the convolutional backbone and projector."""
    enc = ConvVisualEncoder(in_channels=3, d_state=32)
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    out = enc(x)
    loss = (out ** 2).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0.0

def test_conv_visual_encoder_with_dual_predictor():
    """Verify end-to-end forward pass through ConvVisualEncoder and ActionAsymmetricDualPredictor."""
    enc = ConvVisualEncoder(in_channels=3, d_state=48)
    pred = ActionAsymmetricDualPredictor(d_state=48, d_action=2, d_out=48)
    
    img = torch.rand(4, 3, 64, 64)
    action = torch.rand(4, 2)
    
    latent = enc(img)
    diff = pred(latent, action)
    
    assert diff.shape == (4, 48)
    assert not torch.isnan(diff).any()


def test_conv_visual_encoder_any_resolution():
    enc = ConvVisualEncoder(in_channels=3, d_state=16)
    for size in (32, 64, 96):
        assert enc(torch.rand(2, 3, size, size)).shape == (2, 16)


def test_conv_visual_encoder_is_position_sensitive():
    """A translated object must change the code: global average pooling made this ~0."""
    torch.manual_seed(0)
    enc = ConvVisualEncoder(in_channels=3, d_state=32).eval()
    a = torch.zeros(1, 3, 64, 64)
    b = torch.zeros(1, 3, 64, 64)
    a[..., 8:16, 8:16] = 1.0
    b[..., 40:48, 40:48] = 1.0
    with torch.no_grad():
        assert (enc(a) - enc(b)).norm() > 0.1 * enc(a).norm()
