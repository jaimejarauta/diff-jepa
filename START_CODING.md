# Getting Started: diff-jepa

This document defines the exact workflow, metrics, and branching protocols for developing **Diff-JEPA** (Idea 06: Action-Asymmetric Differential Noise-Cancelling Latent World Models).

---

## 1. Environment & Setup

```bash
# 1. Create and activate virtual environment via uv
uv venv .venv
source .venv/bin/activate

# 2. Install dependencies (strictly uv pip)
uv pip install -e ".[dev]"

# 3. Verify PyTorch installation and GPU memory
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Git & Branching Protocol

- **Protected Branch:** `main`. **Never commit, rebase, or push directly to `main`.**
- **Active Feature Branch:** `feature/initial-implementation`.
- **Workflow for New Tasks:**
  ```bash
  git checkout feature/initial-implementation
  git checkout -b feat/task-name
  # Implement code with tests (TDD)
  .venv/bin/python -m pytest tests/
  git add <touched_files>
  git commit -m "feat: implement action-asymmetric dual head"
  # Never git push. Notify the lead researcher when ready.
  ```
- **Commit Format:** `<type>: <what changed>` where type is one of `feat`, `fix`, `docs`, `refactor`, `test`, `exp`.
- **Attribution Rule:** **Zero AI or Claude attribution anywhere** (no trailers, no commit notes, no comments).

---

## 3. Core Hypothesis, Metrics & Pre-Registered Criteria

### The Question
Does subtracting an unconditioned passive background head ($z_2(s_t, \mathbf{0})$) from an action-conditioned causal head ($z_1(s_t, a_t)$) with SIGReg regularizing the residual difference $\Delta z$ isolate causal affordances and prevent background clutter from corrupting visual world model rollouts?

### Key Performance Indicators (KPIs)
1. **Effective Rank ($\text{erank}$):** $\text{erank}(\Delta Z) \ge 0.85 \times D$ (confirming zero dimensional collapse across causal updates).
2. **Saliency Contrast Ratio (CR):** Gradient energy on foreground pixels vs background distractors $\text{CR}_{\text{diff}} \ge 4.0 \times \text{CR}_{\text{standard}}$.
3. **Causal T-Block Probe $R^2$:** Linear probe on frozen $\Delta z$ predicting manipulated T-block pose and velocity $(x_b, y_b, \theta_b, \dot{x}_b, \dot{y}_b)$ achieving $R^2 \ge 0.80$ under 80% background clutter (vs $<0.55$ for standard JEPA).
4. **Baselines Contrast:**
   - Outperforms Frame Differencing baseline ($\Delta x_t = x_t - x_{t-1}$) which fails under dynamic background jitter.
   - Outperforms Action-Only kinematic baseline which cannot observe external object interactions.
   - Untrained Control Delta: $R^2_{\text{diff}} - R^2_{\text{untrained}} \ge 0.15$ on T-block pose.

### Pre-Registered Rules
- **Kill Criterion (Day 3 Gate):** If on the noisy Push-T benchmark, the linear probe $R^2$ for T-block object pose on held-out trajectories is not at least $0.15$ higher than both Frame Differencing and Untrained Control ($R^2_{\text{diff}} - \max(R^2_{\text{frame\_diff}}, R^2_{\text{untrained}}) < 0.15$), kill immediately and archive as a controlled null.
- **Pulse Criterion:** If Diff-JEPA achieves $R^2 \ge 0.80$ on manipulated T-block pose while standard JEPA drops to $\le 0.55$ under 80% background noise, double down for nuPlan/Franka scaling.
- **Latent Normalization Invariant:** Differential vectors $\Delta z$ are standardized per feature across mini-batches (mean 0, std 1) rather than spherically normalized, matching $\mathcal{N}(0, 1)$ assumptions of SIGReg Epps-Pulley projections.

---

## 4. Weights & Biases (W&B) Setup

```python
import wandb

wandb.init(
    project="moonshot-diff-jepa",
    group="smoke-test-pusht-clutter",
    job_type="train",
    name="diff-jepa-pusht-seed42",
    config={
        "lambda_diff": 0.75,
        "gamma_sigreg": 0.05,
        "d_causal": 16,
        "d_noise": 64,
        "d_total": 80,
        "asymmetric_conditioning": True,
        "manifest_digest": "sha256_here"
    }
)
```

---

## 5. Day 2 Smoke Test Execution

Before launching on a shared machine:
```bash
# 1. Busy check
ps -eo pid,etime,pcpu,comm,args --sort=-pcpu | head
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
free -g
uptime

# 2. Run unit tests first
.venv/bin/python -m pytest tests/test_sigreg.py -v

# 3. Launch smoke test (under 5 minutes)
.venv/bin/python scripts/run_smoke.py --config configs/smoke_test.yaml
```

