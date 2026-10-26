"""Tests for Visual Push-T rendering, distractor engine, and dataset loader."""

import torch
import pytest
from diff_jepa.pusht_vision import (
    render_pusht_scene,
    generate_distractor_field,
    generate_visual_pusht_trajectories,
    VisualPushTDataset,
    create_visual_pusht_dataloader,
)

def test_render_pusht_scene_occupancy():
    """Verify that rendering T-block and agent produces non-empty smooth masks in [0, 1]."""
    h, w = 64, 64
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    mask_b, mask_a = render_pusht_scene(
        x, y,
        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),
        torch.tensor(0.2), torch.tensor(-0.3),
    )
    assert mask_b.shape == (h, w)
    assert mask_a.shape == (h, w)
    assert (mask_b >= 0.0).all() and (mask_b <= 1.0).all()
    assert (mask_a >= 0.0).all() and (mask_a <= 1.0).all()
    # Masks should have non-zero area
    assert mask_b.sum() > 50.0
    assert mask_a.sum() > 20.0

def test_generate_distractor_field():
    """Verify distractor background clutter is non-empty RGB frame bounded in [0, 1]."""
    bg = generate_distractor_field(64, 64, t_idx=5, num_blobs=3, seed=123)
    assert bg.shape == (3, 64, 64)
    assert (bg >= 0.0).all() and (bg <= 1.0).all()
    assert not torch.isnan(bg).any()

def test_generate_visual_pusht_trajectories():
    """Verify trajectory generation outputs proper shapes and valid physical bounds."""
    trajs = generate_visual_pusht_trajectories(
        num_trajectories=2,
        trajectory_length=10,
        img_size=32,
        noise_ratio=0.80,
        seed=42,
    )
    assert len(trajs) == 2
    traj = trajs[0]
    assert traj.frames.shape == (10, 3, 32, 32)
    assert traj.actions.shape == (9, 2)
    assert traj.states.shape == (10, 6)
    assert not torch.isnan(traj.frames).any()
    assert not torch.isnan(traj.states).any()

def test_visual_pusht_dataset_and_dataloader():
    """Verify VisualPushTDataset wraps trajectories and yields batches of valid items."""
    trajs = generate_visual_pusht_trajectories(
        num_trajectories=2,
        trajectory_length=8,
        img_size=32,
        noise_ratio=0.70,
        seed=100,
    )
    ds = VisualPushTDataset(trajs)
    assert len(ds) == 2 * 7  # 2 trajs * (8 - 1) steps
    loader = create_visual_pusht_dataloader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    assert batch["frame_curr"].shape == (4, 3, 32, 32)
    assert batch["frame_next"].shape == (4, 3, 32, 32)
    assert batch["action"].shape == (4, 2)
    assert batch["target_pose"].shape == (4, 4)
