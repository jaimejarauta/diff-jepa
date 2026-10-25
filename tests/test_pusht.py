"""Unit tests for Push-T trajectory generator, dataset loader, and Ridge linear probe."""

import math
import torch
from diff_jepa.pusht import (
    generate_synthetic_pusht_trajectories,
    create_pusht_dataloader,
    fit_ridge_probe,
    fit_mlp_probe,
    PushTDataset,
)

def test_synthetic_trajectory_generation():
    torch.manual_seed(42)
    num_trajs = 5
    traj_len = 20
    d_causal = 16
    noise_ratio = 0.80

    trajectories = generate_synthetic_pusht_trajectories(
        num_trajectories=num_trajs,
        trajectory_length=traj_len,
        d_causal=d_causal,
        noise_ratio=noise_ratio,
        seed=42
    )

    assert len(trajectories) == num_trajs
    d_noise = int(round(d_causal * noise_ratio / (1.0 - noise_ratio)))
    d_total = d_causal + d_noise

    for traj in trajectories:
        assert traj.agent_pos.shape == (traj_len, 2)
        assert traj.block_pose.shape == (traj_len, 3)
        assert traj.goal_pos.shape == (traj_len, 2)
        assert traj.actions.shape == (traj_len - 1, 2)
        assert traj.obs_raw.shape == (traj_len, d_total)

        # Check finite values
        assert torch.isfinite(traj.obs_raw).all()
        assert torch.isfinite(traj.block_pose).all()
        assert torch.isfinite(traj.actions).all()

        # Check that agent moved
        delta_agent = torch.norm(traj.agent_pos[-1] - traj.agent_pos[0])
        assert delta_agent > 0.01, "Agent should have moved over trajectory"

def test_pusht_dataset_and_dataloader():
    torch.manual_seed(42)
    num_trajs = 4
    traj_len = 15
    trajectories = generate_synthetic_pusht_trajectories(
        num_trajectories=num_trajs,
        trajectory_length=traj_len,
        d_causal=8,
        noise_ratio=0.75,
        seed=101
    )

    dataset = PushTDataset(trajectories)
    expected_transitions = num_trajs * (traj_len - 1)
    assert len(dataset) == expected_transitions

    item = dataset[0]
    assert "obs_curr" in item
    assert "obs_next" in item
    assert "action" in item
    assert "block_pose_next" in item

    loader = create_pusht_dataloader(trajectories, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    assert batch["obs_curr"].shape[0] == 8
    assert batch["action"].shape[0] == 8

def test_ridge_probe_linear_predictability():
    torch.manual_seed(42)
    n_train = 500
    n_val = 100
    d_in = 16
    k_out = 3

    # Generate linear mapping with small noise
    w_true = torch.randn(d_in, k_out)
    b_true = torch.tensor([1.0, -2.0, 0.5])

    train_x = torch.randn(n_train, d_in)
    train_y = train_x @ w_true + b_true + 0.01 * torch.randn(n_train, k_out)

    val_x = torch.randn(n_val, d_in)
    val_y = val_x @ w_true + b_true + 0.01 * torch.randn(n_val, k_out)

    mean_r2, r2_dict = fit_ridge_probe(train_x, train_y, val_x, val_y, alpha=1e-3)
    assert mean_r2 > 0.95, f"Expected high R^2 for linear data, got {mean_r2}"
    assert "x_block" in r2_dict
    assert "y_block" in r2_dict
    assert "theta_block" in r2_dict

def test_ridge_probe_noise_isolation():
    torch.manual_seed(42)
    n_train = 400
    n_val = 100
    d_in = 32
    k_out = 3

    # Pure uncorrelated Gaussian noise
    train_x = torch.randn(n_train, d_in)
    train_y = torch.randn(n_train, k_out)
    val_x = torch.randn(n_val, d_in)
    val_y = torch.randn(n_val, k_out)

    mean_r2, _ = fit_ridge_probe(train_x, train_y, val_x, val_y, alpha=1.0)
    assert mean_r2 < 0.15, f"Expected low R^2 for random noise, got {mean_r2}"

def test_mlp_probe_non_linear_recovery():
    torch.manual_seed(42)
    n_train = 600
    n_val = 150
    d_in = 16

    # Non-linear circular target mapping: y = [cos(2*x0), sin(2*x1)]
    train_x = torch.randn(n_train, d_in)
    train_y = torch.stack([torch.cos(2.0 * train_x[:, 0]), torch.sin(2.0 * train_x[:, 1])], dim=1)

    val_x = torch.randn(n_val, d_in)
    val_y = torch.stack([torch.cos(2.0 * val_x[:, 0]), torch.sin(2.0 * val_x[:, 1])], dim=1)

    # Linear probe fails on circular harmonics
    r2_linear, _ = fit_ridge_probe(train_x, train_y, val_x, val_y, alpha=1.0, dim_names=["cos", "sin"])
    assert r2_linear < 0.40, f"Expected linear probe to struggle on circular non-linearity, got {r2_linear}"

    # 2-layer MLP probe successfully captures circular non-linearity
    r2_mlp, mlp_dict = fit_mlp_probe(train_x, train_y, val_x, val_y, hidden_dim=64, epochs=150, lr=0.02, dim_names=["cos", "sin"])
    assert r2_mlp > 0.80, f"Expected MLP probe to recover non-linear representation, got {r2_mlp}"
    assert "cos" in mlp_dict and "sin" in mlp_dict
