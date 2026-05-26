# Report — opp_stats_modulation_v2

**Date**: 2026-03-17
**Experiment**: Two-variant study of ungated residual (v2a) vs state-conditioned gate (v2b)
under a new weighted training recipe (CFR + heuristic 3× oversampled).

---

## Results Summary (checkpoint_best_robust.pt)

| Metric | baseline_v1 | v1_a_td | **v2a_ungated** | **v2b_gated** |
|--------|------------|---------|-----------------|---------------|
| heuristic | +0.295 | +0.008 | **+0.413** | **+0.405** |
| cfr | -0.070 | -0.194 | **+0.044** | **-0.003** |
| tight_passive | +0.745 | +0.704 | +0.187 | +0.199 |
| tight_aggressive | +0.703 | +0.581 | -0.006 | +0.032 |
| loose_passive | +0.841 | +0.797 | +0.408 | +0.371 |
| loose_aggressive | +0.623 | +0.560 | +0.910 | +0.902 |
| maniac | +1.165 | +1.533 | +1.590 | +1.443 |
| random | +1.353 | +1.644 | +1.409 | +1.419 |
| **avg** | +0.580 | +0.539 | **+0.619** | **+0.596** |
| **robustness** | — | — | -0.303 | **-0.284** |
| **worst_case** | -0.070 | -0.194 | -0.006 | -0.003 |

Best robust checkpoint episodes: v2a → ep 50K, v2b → ep 125K.
Full training run: 200K episodes, ~254s (v2a) / ~293s (v2b).

---

## Success Criteria Evaluation

| Criterion | Status |
|-----------|--------|
| Avg beats both baseline_v1 (+0.580) and v1_a_td (+0.539) | ✅ Both variants (v2a: +0.619, v2b: +0.596) |
| Margin over v1_a_td ≥ 0.03 | ✅ v2a: +0.080, v2b: +0.057 |
| Worst-case improves vs baseline | ✅ worst_case −0.070 → −0.006 (v2a), −0.003 (v2b) |
| Robustness positive | ❌ Both still negative (v2a: −0.303, v2b: −0.284) |

---

## Key Findings

### 1. Oversampled training recipe drives the improvement

The most significant change from v1 to v2 is the training recipe: CFR and heuristic each
get 25% of sessions (3× oversampled vs uniform 12.5% in v1). CFR score improved from
−0.194 (v1) to +0.044 (v2a) and −0.003 (v2b). Heuristic improved from +0.008 to +0.413.
This is the primary driver of the +0.08 avg improvement over v1 variant_a.

The architecture (no gate vs state gate) is secondary — v2a (ungated) achieves higher
avg (+0.619 vs +0.596), while v2b (gated) achieves slightly better robustness (−0.284 vs
−0.303) and better worst-case (−0.003 vs −0.006).

### 2. State-conditioned gate is not differentiating

Gate activation analysis for v2b (mean gate per opponent prototype, across 500 game states):

| Opponent | Mean gate | Std gate |
|----------|-----------|---------|
| heuristic | 0.5364 | 0.0465 |
| cfr | 0.5435 | 0.0505 |
| tight_passive | 0.5240 | 0.0390 |
| tight_aggressive | 0.5404 | 0.0493 |
| loose_passive | 0.5177 | 0.0321 |
| loose_aggressive | 0.5635 | 0.0572 |
| maniac | 0.5811 | 0.0627 |
| random | 0.5417 | 0.0501 |

All mean values cluster around 0.50–0.58 with low std (0.03–0.06). The gate has learned
mild uniform modulation rather than state-selective gating. It applies slightly higher
modulation for aggressive opponents (maniac: 0.58, loose_aggressive: 0.56) but the
signal is weak. The gate is not learning to open for opponent-sensitive states and close
for invariant ones as hypothesized.

**Why**: TD(0) doesn't provide direct gradient signal about state-level opponent
sensitivity. The gate receives the same loss signal regardless of whether the state has
high or low EV variance across opponents. Without EV variance information in the gradient
(see v3 direction below), the gate learns a mean-field solution.

### 3. Structured opponents (TP, TA) remain below baseline

The biggest gap vs baseline_v1 is on tight_passive (+0.187 vs +0.745) and
tight_aggressive (−0.006 vs +0.703). These opponents play relatively few hands per 100
rounds, so opponent stats converge more slowly and cold-start noise dominates. Oversampling
CFR/heuristic indirectly reduces the session budget for learning against these opponents.

### 4. v2a (ungated) converged less cleanly; v2b converged

- v2a: converged=False (Δloss=5.3% — just above 5% threshold). Training likely benefits
  from more episodes or a decaying LR.
- v2b: converged=True (Δloss=3.6%). Gating provides implicit regularization — smaller
  effective modulation magnitude reduces gradient variance.

---

## Ablation: Training Recipe vs Architecture

| Configuration | avg | vs v1_a_td |
|---------------|-----|-----------|
| v1_a_td (uniform sampling, no gate) | +0.539 | baseline |
| v2a (oversampled, no gate) | +0.619 | +0.080 |
| v2b (oversampled, state gate) | +0.596 | +0.057 |

Training recipe accounts for the majority of the gain. The state gate provides
marginal benefit on worst-case and robustness but reduces avg performance.

---

## Implications for Follow-up Experiments

### v3 (EV-Weighted Training) — HIGH PRIORITY
The gate analysis reveals the core problem: TD gradients don't carry opponent-sensitivity
signal at the state level. Using `ev_std_cross` from EV_variation_analysis to weight
per-transition TD losses would directly inject this signal, potentially forcing the gate
(or ungated head) to focus on the 51.65% of states where modulation matters.

### v4 (Cold-Start-Fixed Supervised) — MEDIUM PRIORITY
The tight_passive and tight_aggressive performance gap shows cold-start is still
hurting slower-signal opponents. Training the residual head on a curriculum of confidence
levels 0→1 would teach it to gracefully handle the cold-start regime.

### v5 (Street-Specific Modulation) — LOW PRIORITY
With the gate not differentiating, street-specific modulation heads are unlikely to
outperform the current unified head unless paired with EV-variance weighting (v3).

**Recommended next step**: v3 EV-weighted training, applied on top of v2a's ungated
architecture (simpler, higher avg) with the same oversampled recipe.

---

## Artifacts

| File | Description |
|------|-------------|
| `outputs/variant_a_ungated/checkpoint_best_robust.pt` | Best checkpoint for v2a (ep 50K, rob=−0.110 in-training eval) |
| `outputs/variant_a_ungated/checkpoint.pt` | Final checkpoint (avg +0.664) |
| `outputs/variant_b_state_gated/checkpoint_best_robust.pt` | Best checkpoint for v2b (ep 125K, rob=−0.079 in-training eval) |
| `outputs/variant_b_state_gated/checkpoint.pt` | Final checkpoint (avg +0.590) |
| `outputs/variant_*/eval_history.json` | Per-episode eval scores (40 points, 500 rounds/opp) |
| `outputs/variant_*/train_history.json` | Per-batch TD loss trace |
| `outputs/variant_*/results.json` | Final summary metrics |
| `outputs/variant_*/evaluation.json` | Full pool evaluation (5K rounds/opp) |
