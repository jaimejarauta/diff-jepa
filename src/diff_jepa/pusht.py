"""Push-T demonstration loader, synthetic trajectory generator, and linear probe evaluators.
Models 2D agent end-effector, 2D T-block position/orientation, and dynamic background clutter.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class PushTTrajectory:
    """Demonstration trajectory container for Push-T affordance learning."""
    agent_pos: torch.Tensor       # (T, 2)
    block_pose: torch.Tensor      # (T, 3): [x, y, theta]
    goal_pos: torch.Tensor        # (T, 2)
    actions: torch.Tensor         # (T - 1, 2)
    obs_raw: torch.Tensor         # (T, d_total): causal features + dynamic clutter

def fit_ridge_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    alpha: float = 1.0,
    dim_names: Optional[List[str]] = None
) -> Tuple[float, Dict[str, float]]:
    """Fits a standardized Ridge linear regression probe on representations to predict T-block pose.
    Returns: (mean_r2, per_dimension_r2_dict).
    """
    device = train_x.device
    dtype = train_x.dtype

    # Standardize input features to ensure well-conditioned Ridge normal equations
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_std = train_x.std(dim=0, keepdim=True).clamp(min=1e-5)
    tr_x_norm = (train_x - x_mean) / x_std
    val_x_norm = (val_x - x_mean) / x_std

    # Add bias term (prepend column of 1s)
    ones_train = torch.ones(tr_x_norm.size(0), 1, device=device, dtype=dtype)
    x_aug = torch.cat([tr_x_norm, ones_train], dim=1)  # (N, D + 1)

    ones_val = torch.ones(val_x_norm.size(0), 1, device=device, dtype=dtype)
    val_x_aug = torch.cat([val_x_norm, ones_val], dim=1)

    d_dim = x_aug.size(1)
    reg = alpha * torch.eye(d_dim, device=device, dtype=dtype)
    # Don't regularize bias term
    reg[-1, -1] = 0.0

    # Normal equations: (X^T X + alpha * I) W = X^T Y
    xtx = x_aug.t() @ x_aug + reg
    xty = x_aug.t() @ train_y
    weights = torch.linalg.solve(xtx, xty)  # (D + 1, K)

    preds = val_x_aug @ weights  # (N_val, K)

    # Compute R^2 per dimension: 1 - SS_res / SS_tot
    ss_res = torch.sum((val_y - preds) ** 2, dim=0)
    val_y_mean = torch.mean(val_y, dim=0, keepdim=True)
    ss_tot = torch.sum((val_y - val_y_mean) ** 2, dim=0)
    r2_per_dim = 1.0 - (ss_res / torch.clamp(ss_tot, min=1e-8))

    if dim_names is None:
        if val_y.size(1) == 3:
            dim_names = ["x_block", "y_block", "theta_block"]
        else:
            dim_names = ["x_block", "y_block", "cos_theta", "sin_theta"]

    r2_dict = {}
    for i, name in enumerate(dim_names):
        if i < r2_per_dim.numel():
            r2_dict[name] = max(-1.0, r2_per_dim[i].item())

    mean_r2 = max(-1.0, r2_per_dim.mean().item())
    return mean_r2, r2_dict

def generate_synthetic_pusht_trajectories(
    num_trajectories: int = 100,
    trajectory_length: int = 50,
    d_causal: int = 16,
    noise_ratio: float = 0.80,
    seed: int = 1337,
    device: Optional[torch.device] = None
) -> List[PushTTrajectory]:
    """Generates synthetic Push-T demonstration trajectories with physical contact dynamics
    and dynamic high-variance background clutter (default: 80% noise dimension ratio).
    """
    if device is None:
        device = torch.device("cpu")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Compute noise dimension based on target ratio: d_noise / (d_causal + d_noise) == noise_ratio
    d_noise = int(round(d_causal * noise_ratio / (1.0 - noise_ratio)))
    d_total = d_causal + d_noise

    trajectories: List[PushTTrajectory] = []

    # Fixed deterministic projection matrix for physical world causal embedding (constant sensor geometry)
    d_phys = 8
    proj_gen = torch.Generator(device=device)
    proj_gen.manual_seed(42)
    w_causal_proj = torch.randn(d_phys, d_causal, generator=proj_gen, device=device) / math.sqrt(d_phys)

    contact_radius = 0.25

    for _ in range(num_trajectories):
        # 1. Initialize random starting states
        agent_pos = torch.zeros(trajectory_length, 2, device=device)
        block_pose = torch.zeros(trajectory_length, 3, device=device)
        goal_pos = torch.zeros(trajectory_length, 2, device=device)
        actions = torch.zeros(trajectory_length - 1, 2, device=device)
        obs_raw = torch.zeros(trajectory_length, d_total, device=device)

        # Fixed goal position for the episode
        goal = (torch.rand(2, generator=gen, device=device) - 0.5) * 1.5
        goal_pos[:] = goal

        # Initial T-block pose
        init_block_pos = (torch.rand(2, generator=gen, device=device) - 0.5) * 1.0
        init_theta = (torch.rand(1, generator=gen, device=device) * 2 * math.pi - math.pi)
        curr_block_pos = init_block_pos.clone()
        curr_theta = init_theta.clone()

        # Initial agent end-effector position (spawned around the block)
        curr_agent_pos = curr_block_pos + (torch.rand(2, generator=gen, device=device) - 0.5) * 0.8

        # Dynamic background clutter generator parameters
        # Correlated low-frequency harmonics + random walk
        n_harmonics = 4
        harmonic_freqs = torch.linspace(0.1, 0.8, n_harmonics, device=device)
        harmonic_phases = torch.rand(n_harmonics, d_noise, generator=gen, device=device) * 2 * math.pi
        harmonic_amps = torch.rand(n_harmonics, d_noise, generator=gen, device=device) * 2.0 + 1.0
        bg_baseline = torch.randn(d_noise, generator=gen, device=device) * 3.0

        for t in range(trajectory_length):
            agent_pos[t] = curr_agent_pos
            block_pose[t, :2] = curr_block_pos
            block_pose[t, 2] = curr_theta.squeeze()

            # Construct 8D physical state: [x_a, y_a, x_b, y_b, cos(th), sin(th), x_g, y_g]
            phys_state = torch.tensor([
                curr_agent_pos[0], curr_agent_pos[1],
                curr_block_pos[0], curr_block_pos[1],
                torch.cos(curr_theta).squeeze(), torch.sin(curr_theta).squeeze(),
                goal[0], goal[1]
            ], device=device)

            causal_embed = phys_state @ w_causal_proj

            # Generate dynamic clutter with high variance
            t_tensor = torch.tensor(float(t), device=device)
            harmonic_term = torch.sum(
                harmonic_amps * torch.sin(harmonic_freqs.unsqueeze(1) * t_tensor + harmonic_phases),
                dim=0
            )
            noise_jitter = torch.randn(d_noise, generator=gen, device=device) * 0.3
            bg_clutter = bg_baseline + harmonic_term + noise_jitter

            obs_raw[t] = torch.cat([causal_embed, bg_clutter], dim=-1)

            if t < trajectory_length - 1:
                # Expert demonstrator policy: move agent toward block, then push block toward goal
                vec_to_block = curr_block_pos - curr_agent_pos
                dist_to_block = torch.norm(vec_to_block)

                vec_block_to_goal = goal - curr_block_pos
                dist_to_goal = torch.norm(vec_block_to_goal)
                dir_to_goal = vec_block_to_goal / (dist_to_goal + 1e-6)

                if dist_to_block > contact_radius:
                    # Approach phase: navigate to a push staging point behind the block
                    staging_point = curr_block_pos - dir_to_goal * (contact_radius * 0.9)
                    desired_move = staging_point - curr_agent_pos
                else:
                    # Push phase: advance along goal direction
                    desired_move = dir_to_goal * 0.12

                step_size = torch.norm(desired_move)
                if step_size > 0.10:
                    desired_move = desired_move / step_size * 0.10

                # Action noise
                act_noise = torch.randn(2, generator=gen, device=device) * 0.015
                action = desired_move + act_noise
                actions[t] = action

                # Step forward agent position
                next_agent_pos = curr_agent_pos + action

                # Contact physics between end-effector and T-block
                new_dist = torch.norm(next_agent_pos - curr_block_pos)
                if new_dist < contact_radius:
                    push_impulse = (contact_radius - new_dist) * (action / (torch.norm(action) + 1e-6))
                    curr_block_pos = curr_block_pos + push_impulse * 1.5 + action * 0.7
                    
                    # Induced torque / rotation based on contact offset
                    torque = (next_agent_pos[0] - curr_block_pos[0]) * action[1] - \
                             (next_agent_pos[1] - curr_block_pos[1]) * action[0]
                    curr_theta = curr_theta + torque * 1.8

                curr_agent_pos = next_agent_pos

        trajectories.append(PushTTrajectory(
            agent_pos=agent_pos,
            block_pose=block_pose,
            goal_pos=goal_pos,
            actions=actions,
            obs_raw=obs_raw
        ))

    return trajectories

class PushTDataset(Dataset):
    """Transition-level dataset for Diff-JEPA affordance pretraining."""
    def __init__(self, trajectories: List[PushTTrajectory]):
        self.transitions = []
        for traj in trajectories:
            T = traj.obs_raw.size(0)
            for t in range(T - 1):
                self.transitions.append({
                    "obs_curr": traj.obs_raw[t],
                    "obs_next": traj.obs_raw[t + 1],
                    "action": traj.actions[t],
                    "block_pose_curr": traj.block_pose[t],
                    "block_pose_next": traj.block_pose[t + 1],
                    "agent_pos_curr": traj.agent_pos[t],
                    "agent_pos_next": traj.agent_pos[t + 1],
                })

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.transitions[idx]

def create_pusht_dataloader(
    trajectories: List[PushTTrajectory],
    batch_size: int = 64,
    shuffle: bool = True
) -> DataLoader:
    """Builds a batched DataLoader from a list of PushT trajectories."""
    dataset = PushTDataset(trajectories)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
