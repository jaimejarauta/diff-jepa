"""Visual Push-T Trajectory Generator and Dynamic Distractor Rendering Engine.
Renders 2D planar robot manipulation trajectories with dynamic visual distractors to (T, 3, H, W) RGB frames.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class VisualPushTTrajectory:
    """Holds a single visual Push-T trajectory."""
    frames: torch.Tensor       # (T, 3, H, W) in [0, 1]
    actions: torch.Tensor      # (T - 1, 2) in [-1, 1]
    states: torch.Tensor       # (T, 6): [x_b, y_b, cos(th), sin(th), x_a, y_a]


def _box_sdf(points: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    """Computes exact Signed Distance Function (SDF) for an axis-aligned box.
    Args:
        points: (*, 2) coordinates.
        half_extents: (2,) half-width and half-height.
    Returns:
        (*,) signed distance (negative inside, positive outside).
    """
    d = torch.abs(points) - half_extents
    outside = torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(d[..., 0], d[..., 1]), max=0.0)
    return outside + inside


def render_pusht_scene(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    theta: torch.Tensor,
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    smoothness: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Renders the physical Push-T scene (T-block and circular agent) as smooth occupancy masks.
    Args:
        grid_x: (H, W) x-coordinates in [-1, 1].
        grid_y: (H, W) y-coordinates in [-1, 1].
        x_b, y_b: Target block center scalars.
        theta: Target block orientation scalar.
        x_a, y_a: Agent end-effector center scalars.
        smoothness: Edge transition width for anti-aliasing.
    Returns:
        mask_block: (H, W) occupancy in [0, 1].
        mask_agent: (H, W) occupancy in [0, 1].
    """
    # 1. Circular robot agent (radius = 0.10)
    d_agent = torch.sqrt((grid_x - x_a) ** 2 + (grid_y - y_a) ** 2) - 0.10
    mask_agent = torch.sigmoid(-d_agent / smoothness)

    # 2. T-Block geometry (centered at x_b, y_b and rotated by theta)
    # Transform grid to block's local coordinate frame
    dx = grid_x - x_b
    dy = grid_y - y_b
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)
    local_x = dx * cos_t - dy * sin_t
    local_y = dx * sin_t + dy * cos_t
    local_pts = torch.stack([local_x, local_y], dim=-1)  # (H, W, 2)

    # Stem: width 0.12, height 0.35, centered at (0, -0.05)
    stem_pts = local_pts - torch.tensor([0.0, -0.05], device=grid_x.device)
    sdf_stem = _box_sdf(stem_pts, torch.tensor([0.06, 0.175], device=grid_x.device))

    # Bar (roof): width 0.36, height 0.12, centered at (0, 0.18)
    bar_pts = local_pts - torch.tensor([0.0, 0.18], device=grid_x.device)
    sdf_bar = _box_sdf(bar_pts, torch.tensor([0.18, 0.06], device=grid_x.device))

    # Union of stem and bar
    sdf_t_block = torch.minimum(sdf_stem, sdf_bar)
    mask_block = torch.sigmoid(-sdf_t_block / smoothness)

    return mask_block, mask_agent


def generate_distractor_field(
    h: int,
    w: int,
    t_idx: int,
    num_blobs: int = 5,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates high-entropy dynamic background clutter (moving colored Gaussian blobs + textures).
    Args:
        h, w: Image dimensions.
        t_idx: Current timestep index (drives dynamic motion).
        num_blobs: Number of drifting distractor blobs.
        seed: RNG seed for determinism.
    Returns:
        (3, H, W) RGB clutter field in [0, 1].
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    # Pre-generate blob parameters
    blob_centers = torch.randn(num_blobs, 2, generator=gen, device=device) * 0.5
    blob_vels = torch.randn(num_blobs, 2, generator=gen, device=device) * 0.05
    blob_radii = torch.rand(num_blobs, 1, 1, generator=gen, device=device) * 0.25 + 0.15
    blob_colors = torch.rand(num_blobs, 3, 1, 1, generator=gen, device=device)

    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )

    bg = torch.zeros(3, h, w, device=device)
    # Add animated high-frequency wave patterns
    freq1 = 4.0
    freq2 = 6.0
    phase = t_idx * 0.15
    waves = 0.5 + 0.5 * torch.sin(freq1 * x + freq2 * y + phase)
    bg[0] = 0.2 + 0.3 * waves
    bg[1] = 0.2 + 0.3 * torch.cos(freq2 * x - freq1 * y + phase)
    bg[2] = 0.25 + 0.25 * torch.sin(freq1 * x - phase)

    # Accumulate drifting Gaussian blobs
    for i in range(num_blobs):
        bx = blob_centers[i, 0] + blob_vels[i, 0] * t_idx
        by = blob_centers[i, 1] + blob_vels[i, 1] * t_idx
        # Periodic wrap-around within [-1.5, 1.5]
        bx = ((bx + 1.5) % 3.0) - 1.5
        by = ((by + 1.5) % 3.0) - 1.5
        sq_dist = (x - bx) ** 2 + (y - by) ** 2
        blob_mask = torch.exp(-sq_dist / (2.0 * blob_radii[i] ** 2))
        bg = bg + blob_colors[i] * blob_mask

    return torch.clamp(bg, 0.0, 1.0)


def generate_visual_pusht_trajectories(
    num_trajectories: int = 50,
    trajectory_length: int = 40,
    img_size: int = 64,
    noise_ratio: float = 0.80,
    seed: int = 1337,
    device: Optional[torch.device] = None,
    explore_prob: float = 0.0,
) -> List[VisualPushTTrajectory]:
    """Simulates physical planar Push-T interactions and renders full visual image streams
    corrupted by controllable dynamic background clutter.
    Args:
        num_trajectories: Number of trajectory episodes.
        trajectory_length: Sequence length per trajectory (T).
        img_size: Frame resolution (H=W=img_size).
        noise_ratio: Fraction of visual background clutter in [0, 1].
        explore_prob: Per-step probability of a uniform random action instead of the pushing policy;
            > 0 breaks the state -> action dependence a passive head could otherwise exploit.
        seed: Random seed.
        device: Computation device.
    Returns:
        List of VisualPushTTrajectory objects.
    """
    if device is None:
        device = torch.device("cpu")

    gen = torch.Generator(device=device).manual_seed(seed)
    trajectories = []

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1.0, 1.0, img_size, device=device),
        torch.linspace(-1.0, 1.0, img_size, device=device),
        indexing="ij",
    )

    # Visual palettes
    block_color = torch.tensor([0.85, 0.20, 0.20], device=device).view(3, 1, 1)  # Coral Red T-block
    agent_color = torch.tensor([0.20, 0.40, 0.90], device=device).view(3, 1, 1)  # Cobalt Blue End-effector

    for traj_idx in range(num_trajectories):
        traj_seed = seed + traj_idx * 1000

        # Initial physical state
        x_block = (torch.rand(1, generator=gen, device=device).item() - 0.5) * 0.6
        y_block = (torch.rand(1, generator=gen, device=device).item() - 0.5) * 0.6
        theta = (torch.rand(1, generator=gen, device=device).item() - 0.5) * math.pi
        x_agent = x_block + (torch.rand(1, generator=gen, device=device).item() - 0.5) * 0.5
        y_agent = y_block - 0.35

        frames = []
        states = []
        actions = []

        for t in range(trajectory_length):
            states.append(
                torch.tensor([
                    x_block,
                    y_block,
                    math.cos(theta),
                    math.sin(theta),
                    x_agent,
                    y_agent,
                ], device=device)
            )

            # Render scene masks
            mask_block, mask_agent = render_pusht_scene(
                x_grid, y_grid,
                torch.tensor(x_block, device=device),
                torch.tensor(y_block, device=device),
                torch.tensor(theta, device=device),
                torch.tensor(x_agent, device=device),
                torch.tensor(y_agent, device=device),
            )

            # Generate dynamic distractor background
            bg_clutter = generate_distractor_field(
                img_size, img_size, t, num_blobs=4, seed=traj_seed, device=device
            )

            # Clean foreground scene composite
            fg_scene = torch.zeros(3, img_size, img_size, device=device)
            # Paint T-block
            fg_scene = fg_scene * (1.0 - mask_block) + block_color * mask_block
            # Paint Robot Agent
            fg_scene = fg_scene * (1.0 - mask_agent) + agent_color * mask_agent

            # Blend scene with dynamic clutter according to noise_ratio
            total_mask = torch.clamp(mask_block + mask_agent, 0.0, 1.0)
            frame = (1.0 - noise_ratio) * fg_scene + noise_ratio * (
                bg_clutter * (1.0 - total_mask) + fg_scene * total_mask
            )
            frames.append(torch.clamp(frame, 0.0, 1.0))

            # Action dynamics (agent pushes toward block center with exploration noise)
            if t < trajectory_length - 1:
                target_dx = x_block - x_agent
                target_dy = y_block - y_agent
                dist = math.hypot(target_dx, target_dy) + 1e-6
                push_dir_x = target_dx / dist
                push_dir_y = target_dy / dist

                act_noise_x = (torch.rand(1, generator=gen, device=device).item() - 0.5) * 0.08
                act_noise_y = (torch.rand(1, generator=gen, device=device).item() - 0.5) * 0.08
                act_x = 0.04 * push_dir_x + act_noise_x
                act_y = 0.04 * push_dir_y + act_noise_y
                if torch.rand(1, generator=gen, device=device).item() < explore_prob:
                    act_x = (torch.rand(1, generator=gen, device=device).item() - 0.5) * 0.12
                    act_y = (torch.rand(1, generator=gen, device=device).item() - 0.5) * 0.12
                actions.append(torch.tensor([act_x, act_y], device=device))

                # Update robot kinematics
                x_agent += act_x
                y_agent += act_y

                # Contact push mechanics
                rel_x = x_agent - x_block
                rel_y = y_agent - y_block
                rel_dist = math.hypot(rel_x, rel_y)
                if rel_dist < 0.22:
                    push_factor = 0.65
                    x_block += act_x * push_factor
                    y_block += act_y * push_factor
                    # Induced torque rotation
                    torque = rel_x * act_y - rel_y * act_x
                    theta += torque * 1.5

                # Keep coordinates bounded in [-0.85, 0.85]
                x_block = max(-0.85, min(0.85, x_block))
                y_block = max(-0.85, min(0.85, y_block))
                x_agent = max(-0.90, min(0.90, x_agent))
                y_agent = max(-0.90, min(0.90, y_agent))

        trajectories.append(
            VisualPushTTrajectory(
                frames=torch.stack(frames, dim=0),
                actions=torch.stack(actions, dim=0),
                states=torch.stack(states, dim=0),
            )
        )

    return trajectories


class VisualPushTDataset(Dataset):
    """PyTorch Dataset wrapping VisualPushT trajectories into (s_t, s_{t+1}, a_t, target_pose) tuples."""
    def __init__(self, trajectories: List[VisualPushTTrajectory]):
        self.samples = []
        for traj in trajectories:
            t_len = traj.actions.size(0)
            for t in range(t_len):
                self.samples.append({
                    "frame_curr": traj.frames[t],            # (3, H, W)
                    "frame_next": traj.frames[t + 1],        # (3, H, W)
                    "action": traj.actions[t],               # (2,)
                    "target_pose": traj.states[t + 1, :4],   # (4,): [x_b, y_b, cos(th), sin(th)]
                    "state_curr": traj.states[t],            # (6,)
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def create_visual_pusht_dataloader(
    dataset: VisualPushTDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Creates a standard DataLoader for visual Push-T tuples."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
