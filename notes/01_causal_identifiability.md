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

