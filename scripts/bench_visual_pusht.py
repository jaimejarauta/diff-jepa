#!/usr/bin/env python3
"""Stage 2 Pixel-Level Visual Benchmark for Diff-JEPA on Push-T with Dynamic Distractors.
Evaluates visual affordance extraction from 64x64 RGB frames under 80% high-entropy clutter.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from diff_jepa.models import ConvVisualEncoder, ActionAsymmetricDualPredictor
from diff_jepa.sigreg import SIGRegLoss
from diff_jepa.metrics import compute_effective_rank
from diff_jepa.pusht import fit_ridge_probe, fit_mlp_probe
from diff_jepa.pusht_vision import (
    generate_visual_pusht_trajectories,
    VisualPushTDataset,
    create_visual_pusht_dataloader,
)


def get_git_hash() -> str:
    """Retrieves current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Visual Diff-JEPA Push-T Benchmark with Dynamic Distractors")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--gamma-sigreg", type=float, default=0.05, help="SIGReg loss coefficient")
    parser.add_argument("--num-train-trajs", type=int, default=60, help="Number of training trajectories")
    parser.add_argument("--num-val-trajs", type=int, default=20, help="Number of validation trajectories")
    parser.add_argument("--traj-len", type=int, default=35, help="Trajectory sequence length")
    parser.add_argument("--img-size", type=int, default=64, help="Frame resolution (HxW)")
    parser.add_argument("--d-state", type=int, default=64, help="Visual latent representation dimension")
    parser.add_argument("--noise-ratio", type=float, default=0.80, help="Dynamic distractor ratio in [0, 1]")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--explore-prob", type=float, default=0.5, help="Per-step probability of a random action")
    parser.add_argument("--ema", type=float, default=0.99, help="Target-encoder EMA decay")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu)")
    parser.add_argument("--output-manifest", type=str, default="results/pusht_visual_manifest.json",
                        help="Path to provenance manifest JSON")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry")
    parser.add_argument("--wandb-project", type=str, default="moonshot-diff-jepa", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run display name")
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"], help="W&B mode")
    args = parser.parse_args()

    # Device auto-detection
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
    print(f"Visual Diff-JEPA Push-T Benchmark | Device: {device} ({dev_name})")
    print(f"Config: {args.img_size}x{args.img_size} RGB frames, {int(args.noise_ratio*100)}% dynamic background clutter")

    # W&B Initialization
    use_wandb = args.wandb or (os.environ.get("WANDB_API_KEY") is not None)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
            run_name = args.wandb_name or f"visual-pusht-noise{int(args.noise_ratio*100)}-seed{args.seed}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                group="pusht-visual-distractor",
                job_type="train",
                name=run_name,
                config=vars(args),
                mode=mode,
                dir=os.environ.get("WANDB_DIR", "/tmp/wandb"),
            )
            print(f"[W&B] Initialized run '{run_name}' in project '{args.wandb_project}' (mode: {mode})")
        except Exception as e:
            print(f"[W&B Warning] Failed to initialize wandb: {e}. Continuing without W&B.")
            use_wandb = False

    torch.manual_seed(args.seed)

    # 1. Dataset Generation
    t0_data = time.time()
    train_trajs = generate_visual_pusht_trajectories(
        num_trajectories=args.num_train_trajs,
        trajectory_length=args.traj_len,
        img_size=args.img_size,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
        device=device,
        explore_prob=args.explore_prob,
    )
    val_trajs = generate_visual_pusht_trajectories(
        num_trajectories=args.num_val_trajs,
        trajectory_length=args.traj_len,
        img_size=args.img_size,
        noise_ratio=args.noise_ratio,
        seed=args.seed + 9999,
        device=device,
        explore_prob=args.explore_prob,
    )
    train_dataset = VisualPushTDataset(train_trajs)
    val_dataset = VisualPushTDataset(val_trajs)
    train_loader = create_visual_pusht_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_visual_pusht_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Generated Visual Push-T dataset in {time.time() - t0_data:.2f}s | Train samples: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 2. Architecture Setup
    encoder = ConvVisualEncoder(in_channels=3, d_state=args.d_state).to(device)
    predictor = ActionAsymmetricDualPredictor(d_state=args.d_state, d_action=2, d_out=args.d_state).to(device)
    untrained_encoder = ConvVisualEncoder(in_channels=3, d_state=args.d_state).to(device)
    # EMA target encoder (no gradients): an online-encoder target admits the constant-code solution
    target_encoder = copy.deepcopy(encoder).requires_grad_(False)
    untrained_encoder.eval()

    sigreg_loss_fn = SIGRegLoss(d=args.d_state, n_slices=32).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # 3. Training Loop
    t0_train = time.time()
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            f_curr = batch["frame_curr"].to(device)
            f_next = batch["frame_next"].to(device)
            action = batch["action"].to(device)

            s_curr = encoder(f_curr)
            with torch.no_grad():
                delta_target = target_encoder(f_next) - target_encoder(f_curr)
            delta_pred = predictor(s_curr, action)

            # JEPA prediction loss (Smooth L1 / Huber)
            pred_loss = F.smooth_l1_loss(delta_pred, delta_target)

            # SIGReg regularizer preventing dimensional collapse
            sigreg_diff = sigreg_loss_fn(delta_pred)
            sigreg_state = sigreg_loss_fn(s_curr)
            sigreg_total = sigreg_diff + 0.5 * sigreg_state

            loss = pred_loss + args.gamma_sigreg * sigreg_total

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                for p_t, p_o in zip(target_encoder.parameters(), encoder.parameters()):
                    p_t.lerp_(p_o, 1.0 - args.ema)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        current_lambda = predictor.get_lambda().item()

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} | Lambda: {current_lambda:.3f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/lambda": current_lambda,
            })

    train_time = time.time() - t0_train

    # 4. Extract Representations for Downstream Affordance Probes
    print("\n--- Extracting Visual Representations for Affordance Probing ---")
    encoder.eval()
    predictor.eval()

    def extract_features(dataset: VisualPushTDataset):
        loader = create_visual_pusht_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
        diff_reps, enc_reps, fd_reps, act_reps, untr_reps, raw_reps, target_poses = [], [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                f_curr = batch["frame_curr"].to(device)
                f_next = batch["frame_next"].to(device)
                action = batch["action"].to(device)
                target_pose = batch["target_pose"].to(device)

                s_curr = encoder(f_curr)
                diff = predictor(s_curr, action)
                s_untr = untrained_encoder(f_curr)

                # Frame differencing at the same 8x8 grid as the raw-image probe (was 3 global scalars)
                frame_diff = F.adaptive_avg_pool2d(f_next - f_curr, (8, 8)).flatten(start_dim=1)  # (B, 192)
                # Raw frames downsampled
                raw_down = F.adaptive_avg_pool2d(f_curr, (8, 8)).flatten(start_dim=1)  # (B, 3*8*8)

                diff_reps.append(diff.cpu())
                enc_reps.append(s_curr.cpu())
                fd_reps.append(frame_diff.cpu())
                act_reps.append(action.cpu())
                untr_reps.append(s_untr.cpu())
                raw_reps.append(raw_down.cpu())
                target_poses.append(target_pose.cpu())

        return (
            torch.cat(diff_reps, dim=0),
            torch.cat(enc_reps, dim=0),
            torch.cat(fd_reps, dim=0),
            torch.cat(act_reps, dim=0),
            torch.cat(untr_reps, dim=0),
            torch.cat(raw_reps, dim=0),
            torch.cat(target_poses, dim=0),
        )

    tr_diff, tr_enc, tr_fd, tr_act, tr_untr, tr_raw, tr_y = extract_features(train_dataset)
    val_diff, val_enc, val_fd, val_act, val_untr, val_raw, val_y = extract_features(val_dataset)

    # 5. Measure Representation Health: Effective Rank
    erank_diff = compute_effective_rank(val_enc)  # health of the representation under test (encoder)
    erank_ratio = erank_diff / args.d_state

    # 6. Fit Probes
    r2_diff, r2_diff_dict = fit_ridge_probe(tr_diff, tr_y, val_diff, val_y)
    r2_enc, r2_enc_dict = fit_ridge_probe(tr_enc, tr_y, val_enc, val_y)
    r2_frame_diff, r2_fd_dict = fit_ridge_probe(tr_fd, tr_y, val_fd, val_y)
    r2_action_only, r2_act_dict = fit_ridge_probe(tr_act, tr_y, val_act, val_y)
    r2_untrained, r2_untr_dict = fit_ridge_probe(tr_untr, tr_y, val_untr, val_y)
    r2_raw, r2_raw_dict = fit_ridge_probe(tr_raw, tr_y, val_raw, val_y)

    r2_mlp_diff, r2_mlp_diff_dict = fit_mlp_probe(tr_diff, tr_y, val_diff, val_y, hidden_dim=64, epochs=100)
    r2_mlp_enc, r2_mlp_enc_dict = fit_mlp_probe(tr_enc, tr_y, val_enc, val_y, hidden_dim=64, epochs=100)
    r2_mlp_fd, _ = fit_mlp_probe(tr_fd, tr_y, val_fd, val_y, hidden_dim=64, epochs=100)

    r2_diff_jepa = r2_diff
    r2_diff_jepa_mlp = r2_mlp_diff

    print(f"\nLinear Affordance Probe Evaluation Results (Predicting T-block pose [x, y, cos(th), sin(th)] from 64x64 RGB):")
    print(f"  1. Diff-JEPA (Differential Latent): R^2 = {r2_diff:.4f} {r2_diff_dict}")
    print(f"  2. Diff-JEPA (Visual Encoder):     R^2 = {r2_enc:.4f} {r2_enc_dict}")
    print(f"  3. Frame Differencing baseline:     R^2 = {r2_frame_diff:.4f} {r2_fd_dict}")
    print(f"  4. Untrained ConvVisualEncoder:     R^2 = {r2_untrained:.4f} {r2_untr_dict}")
    print(f"  5. Action-Only Kinematics:          R^2 = {r2_action_only:.4f} {r2_act_dict}")
    print(f"  6. Raw Downsampled Image Probe:     R^2 = {r2_raw:.4f} {r2_raw_dict}")

    print(f"\nNon-Linear MLP Affordance Probe (SO(2) Manifold Probing):")
    print(f"  1. Diff-JEPA MLP (Diff):           R^2 = {r2_mlp_diff:.4f} {r2_mlp_diff_dict}")
    print(f"  2. Diff-JEPA MLP (Visual Enc):     R^2 = {r2_mlp_enc:.4f} {r2_mlp_enc_dict}")
    print(f"  3. Frame Differencing MLP:         R^2 = {r2_mlp_fd:.4f}")

    # Gate revised 2026-10-26: the probe target is absolute pose, which a delta code cannot hold, so the
    # representation under test is the encoder. It must beat every baseline, including raw pixels and action-only.
    best_baseline = max(r2_frame_diff, r2_untrained, r2_action_only, r2_raw)
    affordance_baseline = best_baseline
    margin = r2_enc - affordance_baseline
    gate_passed = bool((margin >= 0.15) and (erank_ratio >= 0.70))

    print(f"\nPre-registered Gate Evaluation:")
    print(f"  Diff-JEPA R^2: {r2_diff_jepa:.4f}")
    print(f"  Best Baseline R^2: {best_baseline:.4f}")
    print(f"  Encoder R^2: {r2_enc:.4f}")
    print(f"  Encoder margin over best baseline: {margin:+.4f} (Required: >= +0.15)")
    print(f"  Effective Rank: {erank_diff:.2f} / {args.d_state} (Ratio: {erank_ratio:.2f})")
    print(f"  Gate Status: {'PASS' if gate_passed else 'FAIL'}")

    if use_wandb and wandb_run:
        wandb.summary["r2_diff"] = r2_diff
        wandb.summary["r2_enc"] = r2_enc
        wandb.summary["r2_diff_jepa"] = r2_diff_jepa
        wandb.summary["r2_frame_diff"] = r2_frame_diff
        wandb.summary["r2_untrained"] = r2_untrained
        wandb.summary["r2_action_only"] = r2_action_only
        wandb.summary["r2_raw"] = r2_raw
        wandb.summary["affordance_margin"] = margin
        wandb.summary["effective_rank_ratio"] = erank_ratio
        wandb.summary["gate_passed"] = gate_passed
        wandb.finish()

    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)
    manifest = {
        "date": "2026-10-26",
        "git_commit": get_git_hash(),
        "config": vars(args),
        "r2_diff_jepa": r2_diff_jepa,
        "r2_diff_jepa_mlp": r2_diff_jepa_mlp,
        "r2_diff_latent": r2_diff,
        "r2_encoder_latent": r2_enc,
        "r2_frame_diff": r2_frame_diff,
        "r2_untrained": r2_untrained,
        "r2_action_only": r2_action_only,
        "r2_raw": r2_raw,
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
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
