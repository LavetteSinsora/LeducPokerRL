# Metrics Glossary

Precise definitions for every evaluation metric used in experiments.
**Why this matters**: several metrics in this project share colloquial names but are
computed differently. For example, "Reward Spearman ρ" has been used for at least
two distinct calculations across experiments — a scalar correlation and a pairwise
distance correlation. Using the wrong variant invalidates comparisons.

Each entry states: what is measured, how it is computed step by step, what it means
to interpret, known limitations, and which experiments have reported it.

---

## Policy / Value Metrics

### `avg_chips_vs_heuristic`

**What it measures**: Average chips won per round against the rule-based heuristic agent.

**Computation**:
1. Run N rounds of Leduc Hold'em, alternating positions every N/2 rounds (first half: agent as P0; second half: agent as P1).
2. Record chip outcome per round for the evaluated agent.
3. Return `mean(chip_outcomes)`.

**Standard N**: 1000 rounds (500 per position). Smaller N (e.g., 100) is acceptable for mid-training eval checkpoints but must be labeled as such.

**Interpretation**: 0.0 = break-even vs heuristic. Positive = wins on average. The heuristic is a reasonable but easily exploitable baseline; value_based typically achieves approximately +0.05 to +0.15 chips/round.

**Limitations**: High variance for REINFORCE-trained agents (standard deviation ≈ 0.3–0.5 chips/round over 1000 rounds). Do not interpret differences < 0.1 as statistically meaningful without confidence intervals.

**Also used as**: `avg_chips_vs_value_based`, `avg_chips_vs_cfr` — same computation, different opponent.

---

### `td_error` / `value_mse`

**What it measures**: Mean squared error of the value network's predictions against TD(0) targets.

**Computation**:
```
target = r + γ * V(s') for non-terminal s'
target = r             for terminal s'
td_error = MSE(V(s), target)
```
γ = 1.0 (Leduc episodes are short; no discounting used).

**Interpretation**: Decreasing TD error indicates the value function is fitting its own bootstrapped targets. Does NOT directly measure value accuracy against true expected returns (see `scalar_spearman_rho` for that).

---

### `policy_entropy`

**What it measures**: Shannon entropy of the policy's action distribution, in bits.

**Computation**: `H = -sum_a π(a|s) * log2(π(a|s))` averaged over a batch of states.

**Interpretation**: Maximum entropy with 3 actions = log2(3) ≈ 1.585 bits (uniform random). Very low entropy (< 0.3 bits) indicates a nearly deterministic policy — may signal premature collapse or overfitting. A healthy REINFORCE policy typically maintains 0.5–1.2 bits throughout training.

---

## Representation Quality Metrics

### `reward_spearman_rho_pairwise` ← **primary Spearman metric**

**What it measures**: How well the embedding space preserves reward distance ordering. States that differ more in expected outcome should be farther apart in embedding space.

**Computation**:
1. Collect a set of N states S = {s_1, ..., s_N} from self-play (N ≥ 500 recommended).
2. For each state s_i, record its terminal reward R_i (actual game outcome, not V(s)).
3. Compute all pairwise embedding distances: `d_ij = ||z_i - z_j||_2` for i < j.
4. Compute all pairwise reward differences: `r_ij = |R_i - R_j|`.
5. Return `scipy.stats.spearmanr(d_ij_all, r_ij_all).correlation`.

**Sample size**: Use at least 500 states → ~125,000 pairs. Subsample to 5,000–10,000 pairs if computation is slow, but sample pairs uniformly at random, not by state.

**Interpretation**: ρ = 1.0 means perfect metric embedding (embedding distances are a monotonic function of reward differences). ρ = 0.0 means no relationship. ρ < 0 means anti-correlated (should not occur for a properly trained encoder).

**Known values from this project**:
- Raw 15-dim features: ρ ≈ 0.106
- contrastive_repr_v1 (L1): ρ ≈ 0.163
- value_based hidden layer (64-dim): ρ ≈ 0.287
- dual_axis_repr_v4 (reward-dominated): ρ ≈ 0.672

**⚠️ Do not confuse with `scalar_spearman_rho`** — they measure different things.

---

### `scalar_spearman_rho`

**What it measures**: How accurately a scalar value function V(s) ranks states by expected outcome. Used for value networks (not embedding spaces).

**Computation**:
1. Collect N states and their terminal rewards R_i.
2. For each state, compute V(s_i) = the network's scalar value output.
3. Return `scipy.stats.spearmanr(V_values, R_values).correlation`.

**Interpretation**: ρ = 1.0 means the value function perfectly ranks states by expected outcome. The value_based agent achieves ρ ≈ 0.626.

**When to use**: Only for networks with a scalar output head (value functions). For embedding spaces, use `reward_spearman_rho_pairwise` instead.

---

### `hand_probe_accuracy`

**What it measures**: How much information about the opponent's private card is linearly decodable from the embedding.

**Computation**:
1. Collect N states from self-play using a data-collection agent (typically frozen value_based).
2. For each state s_i visited by player P, record: embedding z_i and opponent hand label y_i ∈ {0=J, 1=Q, 2=K}.
3. Fit a logistic regression probe: `LogisticRegression(max_iter=1000)` from sklearn.
4. Evaluate via 5-fold cross-validation: `cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')`.
5. Return `mean(cv_scores)`.

**Minimum N**: 500 states. 1000+ recommended.

**Chance baseline**: 1/3 ≈ 0.333 (uniform random over J, Q, K).

**Interpretation**: Accuracy > 0.333 means the embedding contains opponent hand information. Accuracy of 1.0 would mean perfect opponent hand recovery from the embedding alone (not achievable since the opponent's hand is hidden information not in the observation).

**Known values**: contrastive_repr_v1 (reward-only): ~33% (no hand info). hand_identity_repr_v1 (triplet): 62.8%. dual_axis_repr_v2 (SupCon): 63.3%. dual_axis_repr_v5 (subspace): 65.2%.

**⚠️ Probe architecture matters**: a 3-layer MLP probe would achieve higher accuracy than logistic regression by fitting nonlinear structure — but that tests memorization, not linear decodability. Always use logistic regression for comparability.

---

### `effective_dim_80` / `effective_dim_90`

**What it measures**: The intrinsic dimensionality of the embedding space, defined as the minimum number of PCA components needed to explain k% of variance.

**Computation**:
1. Collect N embedding vectors z_1, ..., z_N (N ≥ 500).
2. Fit PCA on the full N × D matrix (D = embedding dimension, typically 8).
3. Compute cumulative explained variance ratio.
4. `effective_dim_k = min{d : sum(explained_variance_ratio[:d]) >= k/100}`.

**Parameters**: k = 80 (primary), k = 90 (secondary). Always report D (total dims) alongside.

**Interpretation**: effective_dim_80 = 1 means one principal component explains 80%+ of the variance — the embedding is essentially 1D (as seen in hand_identity_repr_v1). effective_dim_80 = D means variance is spread evenly — fully multi-dimensional.

**Sample dependency**: This metric depends on what states are in the sample. Compute on a representative distribution (e.g., self-play with a trained agent), not on a trivial subset. Report the number of states used.

---

### `reward_bin_accuracy`

**What it measures**: How well the embedding predicts which reward range a state falls into. A coarser version of `reward_spearman_rho_pairwise` that uses classification instead of correlation.

**Computation**:
1. Discretize terminal rewards into 5 bins using fixed thresholds: `[-∞, -2.0, -0.5, 0.5, 2.0, +∞]` → labels 0–4.
2. Fit logistic regression probe on (embedding → bin label), 5-fold CV.
3. Return mean CV accuracy.

**Chance baseline**: 0.20 (5 classes). Typical bin distribution in Leduc self-play is non-uniform (extreme rewards are rarer), so effective chance may be ~0.25–0.30 due to class imbalance. Always report the actual class distribution.

**When to use**: When `reward_spearman_rho_pairwise` is not available. The two metrics are complementary — Spearman ρ tests continuous metric structure; bin accuracy tests coarse cluster structure.

---

## Convergence Metrics

### `loss_plateau_pct`

**What it measures**: Whether training has converged.

**Computation**:
```
last_20pct_mean  = mean(loss[int(0.8*T):T])
prev_20pct_mean  = mean(loss[int(0.6*T):int(0.8*T)])
plateau_pct      = abs(last_20pct_mean - prev_20pct_mean) / prev_20pct_mean * 100
```

**Convergence threshold**: loss_plateau_pct < 5% means converged. If ≥ 5%, training has not converged and conclusions about the method's effectiveness are premature.

---

## Evaluation Suite Reference

For consistent comparisons, the "standard evaluation suite" in this project is:

| Opponent | Rounds | Position swap |
|---|---|---|
| heuristic | 1000 | Yes (500 each) |
| value_based | 1000 | Yes |
| cfr | 500 | Yes |

Always run vs heuristic at minimum. Results vs value_based and cfr are required before any promotion decision.
