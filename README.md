# Diff-JEPA

Affordance-driven Joint-Embedding Predictive Architecture for robotic manipulation under dynamic visual distractors.

## Overview
Diff-JEPA cancels task-irrelevant background visual clutter in robotic manipulation (such as Push-T) by learning an action-asymmetric dual predictor conditioned on state transitions. Visual representations are regularized using Skew-Information-Geometric Regularization (SIGReg) via random orthogonal QR projections, preventing dimensional and encoder collapse without contrastive negative pairs.

## Pre-Registered Gate Criteria
- **Gate 1 (Smoke):** Effective representation rank ratio $\ge 0.70$ and loss reduction under 80% dynamic background clutter.
- **Gate 2 (Stage 2 Push-T):** Ridge linear probe affordance $R^2_{\text{diff}} - \max(R^2_{\text{frame\_diff}}, R^2_{\text{untrained}}) \ge 0.15$ on held-out manipulation trajectories.

## Quickstart

### Local (Apple Silicon MPS / NVIDIA CUDA):
```bash
# Run smoke test
python scripts/run_smoke.py --config configs/smoke_test.yaml

# Run Stage 2 Benchmark (Push-T 50 trajectories with 80% dynamic clutter)
python scripts/train_pusht.py --trajectories 50 --epochs 200 --wandb
```

### Google Colab:
Open `notebooks/colab_runner.ipynb` in Colab on a T4 GPU. Includes automatic machine teardown (`runtime.unassign()`) to protect compute credits upon completion.

## Telemetry
Native Weights & Biases support via `--wandb` (project: `moonshot-diff-jepa`).

## License
MIT
