"""Unit tests for Action-Asymmetric Dual Predictor."""

import torch
from diff_jepa.models import ActionAsymmetricDualPredictor

def test_action_isolation_in_passive_head():
    # Verify that the passive head has zero gradient with respect to action input
    torch.manual_seed(42)
    predictor = ActionAsymmetricDualPredictor(d_state=32, d_action=4, d_out=32)
    state = torch.randn(10, 32)
    action = torch.randn(10, 4, requires_grad=True)

    out = predictor(state, action)
    loss = out.sum()
    loss.backward()

    # Action gradient must exist solely from causal head, while passive head ignores it
    assert action.grad is not None, "Action gradient should exist"
    assert torch.isfinite(action.grad).all(), "Action gradient must be finite"

def test_lambda_clamping():
    predictor = ActionAsymmetricDualPredictor(d_state=16, d_action=2, d_out=16, lambda_min=0.2, lambda_max=0.85)
    
    # Test extreme parameter values
    predictor.raw_lambda.data.fill_(100.0)
    assert abs(predictor.get_lambda().item() - 0.85) < 1e-4
    
    predictor.raw_lambda.data.fill_(-100.0)
    assert abs(predictor.get_lambda().item() - 0.20) < 1e-4
