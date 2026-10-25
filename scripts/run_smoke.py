#!/usr/bin/env python3
"""Execution script for Diff-JEPA Push-T affordance smoke test."""

import argparse
import json
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_jepa.models import ActionAsymmetricDualPredictor
from diff_jepa.sigreg import compute_sigreg_epps_pulley
from diff_jepa.metrics import compute_effective_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke_test.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Diff-JEPA Smoke Test on: {device}")

    d_causal = cfg["model"]["d_causal"]
    d_action = cfg["model"]["d_action"]
    d_noise = cfg["model"]["d_noise"]
    d_total = cfg["model"]["d_total"]
    batch_size = cfg["training"]["batch_size"]
    n_steps = cfg["training"]["steps"]

    predictor = ActionAsymmetricDualPredictor(
        d_state=d_total,
        d_action=d_action,
        d_out=d_total
    ).to(device)

    opt = torch.optim.Adam(predictor.parameters(), lr=cfg["training"]["lr"])

    t0 = time.time()
    losses = []
    latents = []

    for step in range(n_steps):
        # Generate causal transition: s_next_causal = 1.5 * s_causal + 0.8 * action
        s_causal = torch.randn(batch_size, d_causal, device=device)
        action = torch.randn(batch_size, d_action, device=device)
        W_action = torch.randn(d_action, d_causal, device=device) * 0.5
        s_next_causal = 1.5 * s_causal + action @ W_action

        # Correlated background clutter (shared mode across frames)
        bg_shared = 3.0 * torch.randn(batch_size, d_noise, device=device)
        s_noise = bg_shared + 0.2 * torch.randn(batch_size, d_noise, device=device)
        s_next_noise = bg_shared + 0.2 * torch.randn(batch_size, d_noise, device=device)

        s_curr = torch.cat([s_causal, s_noise], dim=-1)
        s_target = torch.cat([s_next_causal, s_next_noise], dim=-1)

        pred_diff = predictor(s_curr, action)
        target_norm = F.normalize(s_target, dim=-1)

        l_pred = F.mse_loss(pred_diff, target_norm)
        l_sigreg = compute_sigreg_epps_pulley(pred_diff)
        loss = l_pred + cfg["training"]["gamma_sigreg"] * l_sigreg

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step >= n_steps - 50:
            latents.append(pred_diff.detach())

        if step % 50 == 0 or step == n_steps - 1:
            lam = predictor.get_lambda().item()
            print(f"Step {step:03d} | Loss: {loss.item():.4f} (Pred: {l_pred.item():.4f}, SIGReg: {l_sigreg.item():.4f}) | Lambda: {lam:.3f}")

    elapsed = time.time() - t0
    final_latents = torch.cat(latents, dim=0)
    erank = compute_effective_rank(final_latents)
    erank_ratio = erank / d_total

    print(f"\nElapsed: {elapsed:.2f}s | Effective Rank: {erank:.2f} / {d_total} (Ratio: {erank_ratio:.2f})")
    
    # Check pre-registered criteria
    passed = erank_ratio >= 0.70 and losses[-1] < losses[0]
    print(f"Rule evaluation (No dimensional collapse & loss decreased): {'PASS' if passed else 'FAIL'}")

    os.makedirs("results", exist_ok=True)
    manifest = {
        "config": args.config,
        "date": cfg["date"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "effective_rank": erank,
        "erank_ratio": erank_ratio,
        "final_lambda": predictor.get_lambda().item(),
        "passed": passed,
        "elapsed_sec": elapsed,
    }
    with open("results/smoke_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
