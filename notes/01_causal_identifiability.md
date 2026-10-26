# Note 01: Causal Identifiability and Collapse Prevention in Diff-JEPA

**Author:** Jaime Jarauta  
**Status:** Pre-registered active theory note  
**Target:** `src/diff_jepa/models.py`, `src/diff_jepa/sigreg.py`

---

## 1. The Trivial Cancellation Manifold

Let $z_1 = g_{\theta_1}(h)$ and $z_2 = g_{\theta_2}(h)$ be two projection heads in $\mathbb{R}^D$, and define the differential latent representation as:
$$\Delta z = z_1 - \lambda z_2, \quad \lambda \in (0, 1]$$

Under standard predictive loss $\mathcal{L}_{\text{pred}} = \|\Delta \hat{z}_{t+1} - \Delta z_{t+1}^*\|^2$, unconstrained gradient descent encounters the trivial cancellation manifold:
$$\mathcal{M}_{\text{collapse}} = \{(z_1, z_2) \mid z_1 = \lambda z_2\} \implies \Delta z \to \mathbf{0}$$

### Lemma 1 (Per-Head SIGReg Failure)
Applying SIGReg independently to $z_1$ and $z_2$ ($\mathcal{L}_{\text{SIGReg}}(z_1) + \mathcal{L}_{\text{SIGReg}}(z_2)$) fails to prevent collapse. If $z_1 = z \sim \mathcal{N}(0, \mathbf{I})$ and $z_2 = z$, both heads individually satisfy normality, yet $\Delta z = (1 - \lambda)z \to \mathbf{0}$ as $\lambda \to 1$.

### Theorem 1 (Residual SIGReg Guarantees Full Rank)
Applying SIGReg directly to the residual difference $\Delta z$ ($\mathcal{L}_{\text{SIGReg}}(\Delta z)$) strictly precludes dimensional collapse and bounds head correlation:
$$\text{Cov}(\Delta z) = \text{Cov}(z_1) + \lambda^2 \text{Cov}(z_2) - \lambda \left(\text{Cov}(z_1, z_2) + \text{Cov}(z_2, z_1)\right) = \mathbf{I}_D$$
Complete cancellation ($\Delta z = 0$) or rank collapse ($\text{rank}(\Delta z) < D$) incurs an infinite SIGReg penalty.

---

## 2. Action-Asymmetric Conditioning (The Causal Bottleneck)

While Theorem 1 guarantees full rank, SIGReg alone cannot distinguish whether $\Delta z$ encodes ego-motion or background clutter. If background clutter has higher variance, unconstrained heads could cancel ego-motion and retain the background.

To mathematically enforce semantic directionality:
- **Head 1 (Causal):** $z_1 = P_1(s_t, a_t)$. Receives the action control signal.
- **Head 2 (Passive):** $z_2 = P_2(s_t, \mathbf{0})$. Receives a zero-action token ($a_t \perp z_2$).

Because head 2 is blind to $a_t$, it cannot predict action-induced displacements of the pusher or manipulated object. It is strictly constrained to model passive environment drift and background textures. When subtracted via $\Delta z = z_1 - \lambda z_2$, the common background mode is cancelled, leaving the action-induced affordance intact.


## Run log 2026-10-26: benchmark and gate changes

- State-space generator (`pusht.py`): raw-observation probes on the current rendering score ridge R^2 -0.09 and
  MLP -0.73 (80 train / 20 val trajectories). Sweeping the sine scale: 0.5 -> ridge -0.21; 0.2 -> 0.27; 0.1 -> 0.78
  (linear again). The clutter amplitude (~3x the causal signal) swamps the pose before the sine wraps, so no scale
  is both non-trivial and recoverable. Decision: stop using the state-space benchmark; use the visual one.
- The demonstrator in `pusht.py` is a state-feedback policy: an MLP predicts its action from the true state with
  R^2 0.81, so a "passive" head that sees the state can predict action effects. The visual generator now has
  `explore_prob` (random actions); at 1.0 the state -> action MLP R^2 drops from 0.18 to ~0.
- Visual encoder: global average pooling removed position; replaced by a 4x4 grid (any input size).
- Target: EMA target encoder (0.99), stop-gradient on both target frames.
- Gate (revised): the probe target is absolute pose, which the predictor's delta code cannot represent, so the
  representation under test is the encoder; it must beat untrained, raw 8x8 pixels, 8x8 frame difference and
  action-only by >= 0.15 R^2, with encoder effective-rank ratio >= 0.70.
