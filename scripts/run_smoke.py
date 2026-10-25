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

from diff_jepa.models import ActionAsymmetricDualPredictor, VisualStateEncoder
from diff_jepa.sigreg import SIGRegLoss
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

    encoder = VisualStateEncoder(d_raw=d_total, d_state=d_total).to(device)
    predictor = ActionAsymmetricDualPredictor(
        d_state=d_total,
        d_action=d_action,
        d_out=d_total
    ).to(device)

    sigreg_diff = SIGRegLoss(d=d_total, n_slices=32).to(device)
    sigreg_enc = SIGRegLoss(d=d_total, n_slices=32).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg["training"]["lr"]
    )

    t0 = time.time()
    losses = []
    latents = []

    # Fixed synthetic dynamics matrix for causal affordance (T-block position & velocity)
    W_action = torch.randn(d_action, d_causal, device=device) * 0.5

    for step in range(n_steps):
        # Generate causal transition: s_next_causal = 1.2 * s_causal + 0.8 * action
        s_causal = torch.randn(batch_size, d_causal, device=device)
        action = torch.randn(batch_size, d_action, device=device)
        s_next_causal = 1.2 * s_causal + action @ W_action

        # Correlated background clutter (shared mode across frames)
        bg_shared = 3.0 * torch.randn(batch_size, d_noise, device=device)
        s_noise = bg_shared + 0.2 * torch.randn(batch_size, d_noise, device=device)
        s_next_noise = bg_shared + 0.2 * torch.randn(batch_size, d_noise, device=device)

        raw_curr = torch.cat([s_causal, s_noise], dim=-1)
        raw_next = torch.cat([s_next_causal, s_next_noise], dim=-1)

        s_curr = encoder(raw_curr)
        with torch.no_grad():
            s_target = encoder(raw_next)

        pred_diff = predictor(s_curr, action)
        target_norm = (s_target - s_target.mean(dim=0, keepdim=True)) / (s_target.std(dim=0, keepdim=True) + 1e-5)

        l_pred = F.mse_loss(pred_diff, target_norm)
        l_sigreg_diff = sigreg_diff(pred_diff)
        l_sigreg_enc = sigreg_enc(s_curr)

        # Joint loss: prediction + anti-collapse on both differential updates and visual encoder latents
        loss = l_pred + cfg["training"]["gamma_sigreg"] * l_sigreg_diff + 0.5 * cfg["training"]["gamma_sigreg"] * l_sigreg_enc

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step >= n_steps - 50:
            latents.append(pred_diff.detach())

        if step % 50 == 0 or step == n_steps - 1:
            lam = predictor.get_lambda().item()
            print(f"Step {step:03d} | Loss: {loss.item():.4f} (Pred: {l_pred.item():.4f}, SIGReg-Diff: {l_sigreg_diff.item():.4f}, SIGReg-Enc: {l_sigreg_enc.item():.4f}) | Lambda: {lam:.3f}")

    elapsed = time.time() - t0
    final_latents = torch.cat(latents, dim=0)
    erank = compute_effective_rank(final_latents)
    erank_ratio = erank / d_total

    # Baselines comparison:
    # 1. Frame Differencing baseline: delta_raw = raw_next - raw_curr
    delta_raw = raw_next - raw_curr
    bg_raw_norm = delta_raw[:, d_causal:].norm(dim=-1).mean().item()
    causal_raw_norm = delta_raw[:, :d_causal].norm(dim=-1).mean().item()
    frame_diff_snr = causal_raw_norm / max(1e-6, bg_raw_norm)

    # 2. Action-Only baseline: ignores visual observation completely
    action_only_err = F.mse_loss(action @ W_action, s_next_causal).item()

    print(f"\nElapsed: {elapsed:.2f}s | Effective Rank: {erank:.2f} / {d_total} (Ratio: {erank_ratio:.2f})")
    print(f"Baselines | Frame Differencing SNR: {frame_diff_snr:.3f} | Action-Only Causal Error: {action_only_err:.4f}")
    
    # Check pre-registered criteria: high effective rank, loss convergence, encoder non-zero
    passed = erank_ratio >= 0.70 and losses[-1] < losses[0] and s_curr.std().item() > 0.10
    print(f"Rule evaluation (No dimensional/encoder collapse & loss decreased): {'PASS' if passed else 'FAIL'}")

    os.makedirs("results", exist_ok=True)
    manifest = {
        "config": args.config,
        "date": cfg["date"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "effective_rank": erank,
        "erank_ratio": erank_ratio,
        "frame_diff_snr": frame_diff_snr,
        "action_only_err": action_only_err,
        "final_lambda": predictor.get_lambda().item(),
        "passed": passed,
        "elapsed_sec": elapsed,
    }
    with open("results/smoke_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()

