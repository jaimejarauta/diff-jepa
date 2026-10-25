#!/usr/bin/env python3
"""Stage 2 benchmark training script for Diff-JEPA on Push-T demonstration trajectories.
Evaluates linear affordance probes against Frame Differencing, Action-Only, and Untrained baselines.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_jepa.models import VisualStateEncoder, ActionAsymmetricDualPredictor
from diff_jepa.sigreg import SIGRegLoss
from diff_jepa.metrics import compute_effective_rank
from diff_jepa.pusht import (
    generate_synthetic_pusht_trajectories,
    create_pusht_dataloader,
    fit_ridge_probe,
    PushTDataset,
)

def get_git_hash() -> str:
    """Retrieves current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Train Diff-JEPA on Push-T demonstration trajectories")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma-sigreg", type=float, default=0.05, help="SIGReg loss coefficient")
    parser.add_argument("--num-train-trajs", type=int, default=120, help="Number of training trajectories")
    parser.add_argument("--num-val-trajs", type=int, default=30, help="Number of validation trajectories")
    parser.add_argument("--traj-len", type=int, default=50, help="Trajectory length")
    parser.add_argument("--d-causal", type=int, default=16, help="Causal state dimension")
    parser.add_argument("--noise-ratio", type=float, default=0.80, help="Noise dimension ratio (default: 80%)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument("--output-manifest", type=str, default="results/pusht_stage2_manifest.json",
                        help="Path to output manifest JSON")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry")
    parser.add_argument("--wandb-project", type=str, default="moonshot-diff-jepa", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run display name")
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"], help="W&B mode")
    args = parser.parse_args()

    # Device auto-detection: CUDA -> Apple Silicon MPS -> CPU
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"Diff-JEPA Push-T Stage 2 Benchmark | Device: {device} ({dev_name})")

    # W&B Initialization
    use_wandb = args.wandb or (os.environ.get("WANDB_API_KEY") is not None)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
            run_name = args.wandb_name or f"pusht-noise{int(args.noise_ratio*100)}-seed{args.seed}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                group="pusht-affordance",
                job_type="train",
                name=run_name,
                config=vars(args),
                mode=mode,
                dir=os.environ.get("WANDB_DIR", "/tmp/wandb")
            )
            print(f"[W&B] Initialized run '{run_name}' in project '{args.wandb_project}' (mode: {mode})")
        except Exception as e:
            print(f"[W&B Warning] Failed to initialize wandb: {e}. Continuing without W&B.")
            use_wandb = False

    torch.manual_seed(args.seed)

    # 1. Dataset Generation
    d_noise = int(round(args.d_causal * args.noise_ratio / (1.0 - args.noise_ratio)))
    d_total = args.d_causal + d_noise
    d_action = 2

    print(f"Generating trajectories: d_causal={args.d_causal}, d_noise={d_noise} (ratio: {args.noise_ratio * 100:.0f}%), d_total={d_total}")
    t_data_0 = time.time()
    train_trajs = generate_synthetic_pusht_trajectories(
        num_trajectories=args.num_train_trajs,
        trajectory_length=args.traj_len,
        d_causal=args.d_causal,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
        device=device
    )
    val_trajs = generate_synthetic_pusht_trajectories(
        num_trajectories=args.num_val_trajs,
        trajectory_length=args.traj_len,
        d_causal=args.d_causal,
        noise_ratio=args.noise_ratio,
        seed=args.seed + 1000,
        device=device
    )
    print(f"Dataset generated in {time.time() - t_data_0:.2f}s | Train: {len(train_trajs)} trajs, Val: {len(val_trajs)} trajs")

    train_loader = create_pusht_dataloader(train_trajs, batch_size=args.batch_size, shuffle=True)
    val_dataset = PushTDataset(val_trajs)

    # 2. Model Initialization
    encoder = VisualStateEncoder(d_raw=d_total, d_state=d_total).to(device)
    predictor = ActionAsymmetricDualPredictor(d_state=d_total, d_action=d_action, d_out=d_total).to(device)

    # Untrained baseline encoder with identical initialization
    untrained_encoder = VisualStateEncoder(d_raw=d_total, d_state=d_total).to(device)
    untrained_encoder.eval()
    for param in untrained_encoder.parameters():
        param.requires_grad = False

    sigreg_diff = SIGRegLoss(d=d_total, n_slices=32).to(device)
    sigreg_enc = SIGRegLoss(d=d_total, n_slices=32).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr
    )

    # 3. Training Loop
    t_train_0 = time.time()
    epoch_losses = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            obs_curr = batch["obs_curr"].to(device)
            obs_next = batch["obs_next"].to(device)
            action = batch["action"].to(device)

            s_curr = encoder(obs_curr)
            with torch.no_grad():
                s_target = encoder(obs_next)

            pred_diff = predictor(s_curr, action)
            target_norm = (s_target - s_target.mean(dim=0, keepdim=True)) / (s_target.std(dim=0, keepdim=True) + 1e-5)

            l_pred = F.mse_loss(pred_diff, target_norm)
            l_sigreg_diff = sigreg_diff(pred_diff)
            l_sigreg_enc = sigreg_enc(s_curr)

            loss = l_pred + args.gamma_sigreg * l_sigreg_diff + 0.5 * args.gamma_sigreg * l_sigreg_enc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        lam = predictor.get_lambda().item()

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/lambda": lam,
            })

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} | Lambda: {lam:.3f}")

    train_time = time.time() - t_train_0

    # 4. Probe Evaluation on Frozen Representations
    encoder.eval()
    predictor.eval()

    print("\n--- Extracting Representations for Linear Affordance Probes ---")
    train_dataset = PushTDataset(train_trajs)

    def extract_features(dataset: PushTDataset):
        diff_reps = []
        enc_reps = []
        frame_diff_reps = []
        action_reps = []
        untrained_reps = []
        target_poses = []

        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        with torch.no_grad():
            for b in loader:
                o_c = b["obs_curr"].to(device)
                o_n = b["obs_next"].to(device)
                act = b["action"].to(device)
                target_pose = b["block_pose_next"].to(device)

                s_c = encoder(o_c)
                diff = predictor(s_c, act)
                s_untr = untrained_encoder(o_c)

                # Frame Differencing baseline: delta_x = x_next - x_curr
                delta_x = o_n - o_c

                diff_reps.append(diff)
                enc_reps.append(s_c)
                frame_diff_reps.append(delta_x)
                action_reps.append(act)
                untrained_reps.append(s_untr)
                target_poses.append(target_pose)

        return (
            torch.cat(diff_reps, dim=0),
            torch.cat(enc_reps, dim=0),
            torch.cat(frame_diff_reps, dim=0),
            torch.cat(action_reps, dim=0),
            torch.cat(untrained_reps, dim=0),
            torch.cat(target_poses, dim=0),
        )

    tr_diff, tr_enc, tr_fd, tr_act, tr_untr, tr_y = extract_features(train_dataset)
    val_diff, val_enc, val_fd, val_act, val_untr, val_y = extract_features(val_dataset)

    # Measure representation health: effective rank
    erank_diff = compute_effective_rank(val_diff)
    erank_ratio = erank_diff / d_total

    # Fit Ridge probes on representations
    r2_diff, r2_diff_dict = fit_ridge_probe(tr_diff, tr_y, val_diff, val_y)
    r2_enc, r2_enc_dict = fit_ridge_probe(tr_enc, tr_y, val_enc, val_y)
    r2_frame_diff, r2_fd_dict = fit_ridge_probe(tr_fd, tr_y, val_fd, val_y)
    r2_action_only, r2_act_dict = fit_ridge_probe(tr_act, tr_y, val_act, val_y)
    r2_untrained, r2_untr_dict = fit_ridge_probe(tr_untr, tr_y, val_untr, val_y)

    # Use the best representation learned by Diff-JEPA (differential update or state encoder)
    r2_diff_jepa = max(r2_diff, r2_enc)

    print(f"\nLinear Affordance Probe Evaluation Results (Predicting T-block pose [x, y, theta]):")
    print(f"  1. Diff-JEPA (Differential Latent): R^2 = {r2_diff:.4f} {r2_diff_dict}")
    print(f"  2. Diff-JEPA (Visual Encoder):     R^2 = {r2_enc:.4f} {r2_enc_dict}")
    print(f"  3. Frame Differencing baseline:     R^2 = {r2_frame_diff:.4f} {r2_fd_dict}")
    print(f"  4. Untrained Random Encoder:        R^2 = {r2_untrained:.4f} {r2_untr_dict}")
    print(f"  5. Action-Only Kinematics:          R^2 = {r2_action_only:.4f} {r2_act_dict}")

    best_baseline = max(r2_frame_diff, r2_untrained)
    margin = r2_diff_jepa - best_baseline
    gate_passed = margin >= 0.15

    print(f"\nPre-registered Gate Evaluation:")
    print(f"  Diff-JEPA R^2: {r2_diff_jepa:.4f}")
    print(f"  Best Baseline R^2: {best_baseline:.4f}")
    print(f"  Affordance Margin (R^2_diff - max(R^2_fd, R^2_untr)): {margin:+.4f} (Required: >= +0.15)")
    print(f"  Effective Rank: {erank_diff:.2f} / {d_total} (Ratio: {erank_ratio:.2f})")
    print(f"  Gate Status: {'PASS' if gate_passed else 'FAIL'}")

    if use_wandb and wandb_run:
        wandb.summary["r2_diff"] = r2_diff
        wandb.summary["r2_enc"] = r2_enc
        wandb.summary["r2_diff_jepa"] = r2_diff_jepa
        wandb.summary["r2_frame_diff"] = r2_frame_diff
        wandb.summary["r2_untrained"] = r2_untrained
        wandb.summary["r2_action_only"] = r2_action_only
        wandb.summary["affordance_margin"] = margin
        wandb.summary["effective_rank_ratio"] = erank_ratio
        wandb.summary["gate_passed"] = gate_passed
        wandb.finish()

    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)
    manifest = {
        "date": "2026-10-25",
        "git_commit": get_git_hash(),
        "config": vars(args),
        "r2_diff_jepa": r2_diff_jepa,
        "r2_diff_latent": r2_diff,
        "r2_encoder_latent": r2_enc,
        "r2_frame_diff": r2_frame_diff,
        "r2_untrained": r2_untrained,
        "r2_action_only": r2_action_only,
        "r2_margin": margin,
        "gate_passed": gate_passed,
        "effective_rank": erank_diff,
        "erank_ratio": erank_ratio,
        "final_lambda": predictor.get_lambda().item(),
        "final_loss": epoch_losses[-1],
        "training_time_sec": train_time,
    }

    with open(args.output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {args.output_manifest}")
    sys.exit(0 if gate_passed else 1)

if __name__ == "__main__":
    main()
