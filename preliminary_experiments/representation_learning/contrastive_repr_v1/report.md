# Contrastive State Representation Learning v1 — Results

## Training Runs

| Formulation | Episodes | Final Loss | Converged? | Checkpoint |
|---|---|---|---|---|
| L0 (TD control) | 40K | ~4.4 (noisy) | Plateaued | `encoder_l0.pt` |
| L1 (Distance Correlation) | 20K | 0.101 | Yes | `encoder_l1.pt` |
| L2 (Rank-N-Contrast) | 40K | 4.14 | Mostly | `encoder_l2.pt` |

L0 loss remained noisy (bouncing 3-7) throughout training, typical of TD(0) with a severe bottleneck. L1 converged cleanly by ~10K episodes. L2 showed slow, steady descent; extended from 20K to 40K where improvement became marginal.

---

## D10: Cross-Formulation Comparison

| Metric | L0 (TD) | L1 (Distance) | L2 (RnC) |
|---|---|---|---|
| **D1: Spearman ρ** | 0.281 | **0.660** | 0.641 |
| **D2: k-NN error (k=10)** | **2.08** | 2.22 | 2.23 |
| **D3: PCA top-1 %** | 99.5% | 50.8% | **31.8%** |
| **D3: Eff dim (80%)** | 1 | 4 | **5** |
| D4: Hand probe acc | 0.772 | 0.728 | **0.781** |
| D4: Round probe acc | **0.941** | 0.574 | 0.897 |
| D4: Reward R² | **0.360** | 0.102 | 0.308 |
| D4: Reward R² (raw input) | 0.213 | 0.193 | 0.150 |
| D5: Dead dims | 0 | 0 | 0 |
| D5: Redundant pairs | **28** | 0 | 0 |
| D7: Separation ratio | **11.2** | 5.2 | 3.3 |
| D9: Hand separation | 1.43 | **1.53** | 1.49 |

### D8: Cross-Trajectory Exclusion Ablation

| Metric | L1 (all) | L1 (cross-traj) | Δ | L2 (all) | L2 (cross-traj) | Δ |
|---|---|---|---|---|---|---|
| Spearman ρ | 0.660 | **0.685** | +0.025 | **0.641** | 0.579 | -0.062 |
| k-NN (k=10) | 2.22 | **2.12** | -0.10 | 2.23 | **1.99** | -0.24 |
| PCA top-1 % | 50.8% | 54.0% | +3.2 | 31.8% | 39.5% | +7.7 |
| Eff dim (80%) | 4 | 4 | 0 | **5** | 4 | -1 |
| Reward R² | 0.102 | 0.097 | -0.005 | **0.308** | 0.254 | -0.054 |
| Hand separation | 1.53 | **1.54** | +0.01 | 1.49 | 1.49 | 0.00 |

**Verdict: No shortcut collapse.** Both L1 and L2 maintain their representation quality when same-trajectory pairs are excluded. L1 actually *improves* slightly, suggesting no trajectory-ID exploitation whatsoever. L2 shows modest degradation in rank correlation (-0.06) but improves in k-NN retrieval — the representation remains sound.

---

## Analysis

### Finding 1: TD(0) collapses to 1D — contrastive losses produce genuinely multi-dimensional representations

The most striking result is L0's PCA spectrum: **99.5% variance in the first principal component**, with 28 out of 28 possible redundant dimension pairs. The 8-dim bottleneck architecture has enough capacity for multi-dimensional representations, but TD(0) does not incentivize using it. The encoder learns V(s) and maps all 8 dimensions to correlated copies of that scalar (all dimensions have |ρ_reward| ≈ 0.62-0.65).

By contrast, L1 (top-1: 50.8%, eff dim: 4) and L2 (top-1: 31.8%, eff dim: 5) spread information across dimensions with zero redundant pairs. **The contrastive loss, not the architecture, drives multi-dimensionality.** This directly confirms the experiment's core hypothesis.

### Finding 2: L2 produces the richest representation; L1 has the best rank correlation

L2 (Rank-N-Contrast) achieves the most distributed representation across 5 effective dimensions with the most even PCA spectrum. It retains the most state information: best hand probe accuracy (0.781), excellent round probe (0.897), strong reward R² (0.308), and is the only formulation where pot size regression produces a positive R² (0.095).

L1 (Distance Correlation) achieves the highest rank correlation (ρ = 0.660) — its explicit distance-matching objective directly optimizes this metric. However, it sacrifices state identity information: round probe accuracy drops to 0.574 (barely above chance for binary classification), and reward R² is only 0.102.

**Interpretation:** L1 organizes the embedding space primarily around reward distance, which is exactly what its loss function demands — but at the cost of discarding contextual state information. L2's ranking-based loss is less prescriptive about exact distances, leaving room for the encoder to encode additional structure alongside reward ordering.

### Finding 3: Per-dimension specialization differs by formulation

**L0:** All 8 dimensions are highly correlated with reward (|ρ| = 0.61-0.65) and with each other. No specialization — just 8 copies of V(s).

**L1:** Dimensions show varying reward correlations: dim 5 has |ρ| = 0.58 (most reward-aligned) while dim 0 has |ρ| = 0.02 (reward-orthogonal). This suggests dim 0 encodes structure unrelated to expected reward — potentially the kind of distributional/strategic information the experiment aimed to discover. VICReg kept all dimensions active (std 0.028-0.033).

**L2:** More heterogeneous: dims 1 and 4 are strongly reward-correlated (|ρ| = 0.56), while dims 0 and 3 are weakly correlated (|ρ| = 0.14, 0.19). The variance spread across dimensions is more even than L1.

### Finding 4: Same-reward states DO separate by strategic features

D9 hand separation ratios are 1.43-1.53 across all formulations (including L0). This means that among states with approximately equal terminal reward, the embedding places states with different hand cards farther apart than states with the same hand card. The effect is modest but consistent.

This is actually expected: even in L0, the encoder must represent the state well enough for the value head to predict V(s), which depends on hand card. The separation exists because the input features require it, not because the contrastive loss induces it. The fact that L1 shows the highest separation (1.53) despite the worst hand probe accuracy (0.728) is interesting — it suggests L1's geometry organizes same-reward clusters by hand identity even though a linear decoder struggles to extract hand from the full embedding.

### Finding 5: D7 variance probe — separation ratios are high but inconclusive

All formulations show high D7 separation ratios (L0: 11.2, L1: 5.2, L2: 3.3), meaning across-group embedding distances are much larger than within-group distances. However, L0 has the *highest* ratio despite being 1D — this is because L0's collapsed 1D geometry naturally separates groups that have different mean rewards by large distances along the single axis. The lower ratios for L1/L2 reflect their more distributed geometry, not worse separation.

This diagnostic, as designed, conflates mean-reward separation with variance-based separation. A more targeted test would hold mean reward constant and vary only the variance — but Leduc's discrete state space makes it difficult to find enough such controlled pairs.

### Finding 6: The "is this just V(s)?" question — partially answered

The experiment's central intellectual concern was whether contrastive learning on reward labels produces anything beyond V(s) in disguise.

**Evidence for "yes, it does more":**
- L1 and L2 use 4-5 effective dimensions vs L0's 1
- L1 has a reward-orthogonal dimension (dim 0, ρ = 0.02)
- L2 retains round, hand, and pot information alongside reward structure
- Zero redundant pairs in L1/L2 vs 28 in L0

**Evidence for "mostly V(s) with dressing":**
- All formulations show similar D9 hand separation (~1.4-1.5)
- The extra dimensions in L1 primarily rearrange the reward signal across axes rather than encoding clearly distinct features
- Linear reward R² is much lower for L1 (0.10) despite higher rank correlation — the information is spread across dimensions but still dominated by reward

**Honest assessment:** L2 comes closest to a representation that captures "more than V(s)" — it achieves strong reward ordering (ρ = 0.64) while simultaneously retaining rich state identity information (hand, round probes). L1 achieves better pure rank correlation but at the cost of state identity. Neither formulation provides strong evidence for capturing *distributional* structure (variance of outcomes) — the training signal (terminal reward proximity) simply doesn't supervise this directly.

---

## Conclusions

### What worked

1. **Contrastive losses produce genuinely multi-dimensional representations** where TD(0) collapses to 1D. This is the cleanest positive result.
2. **L2 (Rank-N-Contrast) produces the most information-rich embeddings**, balancing reward ordering with state identity preservation.
3. **No shortcut exploitation detected** — cross-trajectory exclusion (D8) maintains or improves representation quality.
4. **VICReg anti-collapse works** — zero dead dimensions, zero redundant pairs for L1/L2.

### What didn't work

1. **No clear evidence of distributional structure** (D7) — the representations organize by expected reward, not by outcome variance. The training signal doesn't supervise variance.
2. **L0 (TD control) is actually the best for pure value prediction** (R² = 0.36) — the bottleneck doesn't hurt it because V(s) is inherently 1D for this task.
3. **L1 loses state identity information** (round probe: 0.57) — exact distance matching is too aggressive for noisy continuous labels, confirming the plan's hypothesis that L2 > L1 for noisy settings.

### What we learned

- The **loss function, not the architecture**, determines dimensionality. Same encoder, same data, same capacity — TD(0) gives 1D, contrastive gives 4-5D.
- **Ranking-based contrastive loss (L2) preserves more state information than distance-matching (L1).** L2's less prescriptive objective leaves room for the encoder to encode contextual features alongside reward structure.
- The original motivation ("safe neutral vs risky neutral") requires supervision that explicitly targets outcome variance. Reward-proximity-based contrastive learning, by design, organizes by E[R|s]. Getting distributional structure would require a different training signal — e.g., contrastive loss on rollout variance labels, or a distributional RL objective.

### Recommended next steps (if Phase 2 is pursued)

1. **Freeze L2 encoder, attach value head, evaluate game play.** L2's rich embedding may give faster or better value learning than training from scratch.
2. **Distributional contrastive learning:** Use rollout variance as a second label axis alongside reward, with a multi-objective contrastive loss. This directly targets the "risky neutral vs safe neutral" distinction.
3. **Temperature sweep for L2:** We used τ = 0.5 throughout. τ = 0.1 (stricter ranking) or τ = 1.0 (more lenient) could shift the reward-vs-identity tradeoff.
