#!/usr/bin/env python3
"""Diff-JEPA Comprehensive Ablation Study (CoRL / NeurIPS Table 2).
Automates the 4-row architectural ablation matrix proving Theorem 1:
  Row 1: Full Diff-JEPA (Action-Asymmetric Dual Head + SIGReg on Delta z & s_t)
  Row 2: Symmetric JEPA (Lambda = 0, standard joint embedding without asymmetric subtraction)
  Row 3: No SIGReg (Gamma = 0, inducing dimensional rank collapse)
  Row 4: Frame Differencing baseline (spatial subtraction)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, Any, List

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


def run_ablation_arm(
    arm_name: str,
    asymmetric: bool,
    gamma_sigreg: float,
    train_dataset: PushTDataset,
    val_dataset: PushTDataset,
    d_total: int,
    d_causal: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f"Running Ablation Arm: {arm_name}")
    print(f"Config: Asymmetric={asymmetric}, Gamma_SIGReg={gamma_sigreg}, Epochs={epochs}")
    print(f"========================================================")

    encoder = VisualStateEncoder(d_raw=d_total, d_state=d_total).to(device)
    predictor = ActionAsymmetricDualPredictor(d_state=d_total, d_action=2, d_out=d_total).to(device)
    sigreg_loss_fn = SIGRegLoss(d=d_total, n_slices=32).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    train_loader = create_pusht_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        for batch in train_loader:
            s_curr_raw = batch["obs_curr"].to(device)
            s_next_raw = batch["obs_next"].to(device)
            action = batch["action"].to(device)

            s_curr = encoder(s_curr_raw)
            with torch.no_grad():
                s_next = encoder(s_next_raw)

            delta_target = s_next - s_curr
            if asymmetric:
                delta_pred = predictor(s_curr, action)
            else:
                # Symmetric baseline: causal head only without unconditioned subtraction
                delta_pred = predictor.causal_head(torch.cat([s_curr, action], dim=-1))

            pred_loss = F.smooth_l1_loss(delta_pred, delta_target)
            if gamma_sigreg > 0:
                sigreg_loss = sigreg_loss_fn(delta_pred) + 0.5 * sigreg_loss_fn(s_curr)
            else:
                sigreg_loss = torch.tensor(0.0, device=device)

            loss = pred_loss + gamma_sigreg * sigreg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    elapsed = time.time() - t0

    # Evaluation
    encoder.eval()
    predictor.eval()

    def extract_reps(dataset: PushTDataset):
        loader = create_pusht_dataloader(dataset, batch_size=batch_size, shuffle=False)
        reps, targets = [], []
        with torch.no_grad():
            for batch in loader:
                s_curr = encoder(batch["obs_curr"].to(device))
                action = batch["action"].to(device)
                if asymmetric:
                    rep = predictor(s_curr, action)
                else:
                    rep = predictor.causal_head(torch.cat([s_curr, action], dim=-1))
                reps.append(rep.cpu())
                targets.append(batch["target_pose"].cpu())
        return torch.cat(reps, dim=0), torch.cat(targets, dim=0)

    tr_x, tr_y = extract_reps(train_dataset)
    val_x, val_y = extract_reps(val_dataset)

    erank = compute_effective_rank(val_x)
    erank_ratio = erank / d_total
    r2, r2_dict = fit_ridge_probe(tr_x, tr_y, val_x, val_y)

    print(f"Result for [{arm_name}]: R^2={r2:.4f}, Effective Rank={erank:.2f}/{d_total} ({erank_ratio:.2f})")
    return {
        "arm": arm_name,
        "asymmetric": asymmetric,
        "gamma_sigreg": gamma_sigreg,
        "r2_affordance": r2,
        "r2_dict": r2_dict,
        "effective_rank": erank,
        "erank_ratio": erank_ratio,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Diff-JEPA Ablation Matrix Suite")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per ablation arm")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--output-json", type=str, default="results/ablation_study_results.json")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Diff-JEPA Ablation Suite | Device: {device}")

    d_causal, noise_ratio = 16, 0.80
    d_total = int(d_causal / (1.0 - noise_ratio))

    train_trajs = generate_synthetic_pusht_trajectories(num_trajectories=80, trajectory_length=40, d_causal=d_causal, noise_ratio=noise_ratio, seed=42, device=device)
    val_trajs = generate_synthetic_pusht_trajectories(num_trajectories=20, trajectory_length=40, d_causal=d_causal, noise_ratio=noise_ratio, seed=999, device=device)
    train_dataset = PushTDataset(train_trajs)
    val_dataset = PushTDataset(val_trajs)

    results = []

    # Row 1: Full Diff-JEPA
    results.append(run_ablation_arm("Full Diff-JEPA (Ours)", True, 0.05, train_dataset, val_dataset, d_total, d_causal, args.epochs, 64, 1e-3, device))

    # Row 2: Symmetric JEPA (No Asymmetric Subtraction)
    results.append(run_ablation_arm("Symmetric JEPA (Lambda=0)", False, 0.05, train_dataset, val_dataset, d_total, d_causal, args.epochs, 64, 1e-3, device))

    # Row 3: No SIGReg (Gamma=0)
    results.append(run_ablation_arm("No SIGReg (Gamma=0)", True, 0.00, train_dataset, val_dataset, d_total, d_causal, args.epochs, 64, 1e-3, device))

    # Baseline: Frame Differencing
    def extract_fd(dataset):
        loader = create_pusht_dataloader(dataset, batch_size=64, shuffle=False)
        reps, targets = [], []
        for batch in loader:
            fd = batch["obs_next"] - batch["obs_curr"]
            reps.append(fd.cpu())
            targets.append(batch["target_pose"].cpu())
        return torch.cat(reps, dim=0), torch.cat(targets, dim=0)

    tr_fd, tr_y = extract_fd(train_dataset)
    val_fd, val_y = extract_fd(val_dataset)
    r2_fd, r2_fd_dict = fit_ridge_probe(tr_fd, tr_y, val_fd, val_y)
    erank_fd = compute_effective_rank(val_fd)
    results.append({
        "arm": "Frame Differencing Baseline",
        "asymmetric": False,
        "gamma_sigreg": 0.0,
        "r2_affordance": r2_fd,
        "r2_dict": r2_fd_dict,
        "effective_rank": erank_fd,
        "erank_ratio": erank_fd / d_total,
        "elapsed_sec": 0.0,
    })

    print("\n=================== FINAL ABLATION MATRIX ===================")
    print(f"{'Method / Ablation Arm':<35} | {'Affordance R^2':<15} | {'Effective Rank Ratio':<20}")
    print("-" * 75)
    for res in results:
        print(f"{res['arm']:<35} | {res['r2_affordance']:<15.4f} | {res['erank_ratio']:<20.2f}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation matrix saved to {args.output_json}")

if __name__ == "__main__":
    main()
