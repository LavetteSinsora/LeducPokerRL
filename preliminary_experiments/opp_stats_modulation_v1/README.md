# opp_stats_modulation_v1 — Modulation Head on Frozen Base

**Experiment thread:** OpponentModeling
**Builds on:** `baseline_value_v1`, `opp_stats_input_augmentation_v1`
**Date:** 2026-03-15
**Status:** Complete

---

## Motivation

`opp_stats_input_augmentation_v1` concatenated 7 opponent stats directly as extra input
dimensions to a jointly-trained value network. It failed to beat the baseline (+0.463 vs +0.580).
Primary cause: **cold-start discriminability** — for the first ~20 hands of each 100-hand
session, all opponents map to the same pool-mean prior embedding, injecting noise into ~20% of
training gradients.

This experiment tests a structurally different approach: keep the pretrained value network
**frozen** and train only a lightweight **modulation head** that predicts a residual correction:

```
V(s, opp) = V_base(s)  [frozen]  +  Δ(s, opp_stats)
```

The frozen base provides a stable floor. The modulation head learns to correct for
opponent-specific deviations from the base model's estimates.

**No gate** — gating mechanisms (σ-weighted residuals) previously collapsed to ~0.5
(uninformative). Omitted here; can be revisited.

---

## Architecture

| Component | Details |
|-----------|---------|
| `V_base` | Frozen `agents/value_based/checkpoint.pt` · 15-dim → 64 → 64 → 1 |
| `Δ` (ModulationHead) | Trainable · (15 + 7) → 32 → 32 → 1 · input = game state + opp stats |
| Total trainable params | ~2K (ModulationHead only) |
| Opponent stats | 7-dim Beta-Bernoulli features from `opp_stats_input_augmentation_v1/stats_tracker.py` |

---

## Two Training Variants

### Variant A — `variant_a_td` (TD(0) Online)

Freeze the base. Train Δ during live play sessions via TD(0):

- At **terminal state** `s_T` with reward `r`:
  - Loss: `MSE(V_base(s_T) + Δ(s_T) − r)`  ← Δ directly learns the residual `r − V_base(s_T)`
- At **non-terminal state** `s_t`:
  - Bootstrap target: `V_base(s_{t+1}) + Δ(s_{t+1})`
  - Loss: `MSE(V_base(s_t) + Δ(s_t) − target)`
- Training schedule: `pool_random` (100-hand sessions, opponent sampled uniformly each session)
- 300K episodes, Adam lr=1e-4

**Cold-start note**: Beta-Bernoulli smoothing means session start = pool_mean (not zero).
Low discriminability (~20 hands) is bounded by the frozen base's stability.

### Variant B — `variant_b_supervised` (Supervised Residual)

Train Δ offline to directly predict the oracle residual:

```
residual(s, opp) = EV_infoset(s, opp) − V_base(s)
```

Where `EV_infoset(s, opp)` is the info-set-level expected value, obtained by marginalizing
over the opponent's unknown private hand using card-removal probabilities:

```
EV_infoset(my_hand, board, pot, cp, raises, opp) =
    Σ_h  P(opp_hand=h | my_hand, board) × EV_data(my_hand, h, board, ..., opp)
```

**Data source**: `EV_variation_analysis/data.json` (3,888 records: 486 states × 8 opponents)

**Per-opponent prototype stats**: 500 hands played vs each individual opponent using
`OpponentStatsTracker` to obtain a 7-dim characteristic stat vector per opponent type.
(Different from pool-mean prior, which is the cross-opponent average.)

**Train / Validation split**:
- Train (6 opponents): tight_passive, tight_aggressive, loose_passive, loose_aggressive, maniac, random
- Validation (held out): **heuristic, cfr** — most distinct strategy profiles; tests generalization

**Hyperparameters**: Adam lr=1e-3, L2 weight_decay=1e-4, up to 1000 epochs with early stopping (patience=50)

---

## Preliminary Analysis

Before training either variant, run:

```bash
python preliminary_analysis.py --save
```

This validates:
1. **Base calibration**: Pearson correlation of `V_base(s)` vs `EV(s, cfr)` — should be > 0.5
2. **Residual magnitude**: Mean |residual| per opponent — tells us how much Δ needs to learn
3. **Residual discriminability**: F-ratio of between- vs within-opponent residual variance

---

## Files

| File | Description |
|------|-------------|
| `agent.py` | `StatModValueAgent`: frozen base + trainable `ModulationHead` |
| `train.py` | Variant A (TD) and Variant B (supervised) training logic |
| `eval.py` | Final 5000-round evaluation vs 8 opponents |
| `preliminary_analysis.py` | EV residual sanity check |

---

## Replication

```bash
cd /path/to/PokerRL_Vanilla

# Step 1: check viability of supervised approach
python OpponentModeling/opp_stats_modulation_v1/preliminary_analysis.py --save

# Step 2: Variant B (supervised, recommended first)
python OpponentModeling/opp_stats_modulation_v1/train.py --variant variant_b_supervised

# Step 3: Variant A (TD online)
python OpponentModeling/opp_stats_modulation_v1/train.py --variant variant_a_td

# Smoke-test (quick pipeline check)
python OpponentModeling/opp_stats_modulation_v1/train.py --variant variant_b_supervised --smoke
python OpponentModeling/opp_stats_modulation_v1/train.py --variant variant_a_td --smoke

# Evaluate
python OpponentModeling/opp_stats_modulation_v1/eval.py --variant variant_b_supervised
python OpponentModeling/opp_stats_modulation_v1/eval.py --variant variant_a_td
```

Outputs saved under `outputs/<variant>/`:
- `train_config.json` — hyperparameters
- `train_history.json` — per-step losses
- `eval_history.json` — periodic eval scores (Variant A only)
- `checkpoint_best.pt` — best modulation head weights
- `checkpoint.pt` — final modulation head weights
- `results.json` — training summary
- `evaluation.json` — final 5K-round results
- `pool_priors.json` — pool-mean priors for stats tracker (Variant A)
- `prototype_stats.json` — per-opponent stat prototypes (Variant B)

---

## Baselines to Beat

| Agent | Overall avg | Worst case |
|-------|-------------|------------|
| `baseline_value_v1` | **+0.580** | -0.098 |
| `opp_stats_input_augmentation_v1/pool_random` | +0.463 | -0.375 |

---

## Results

*(To be filled after training)*

| Opponent | baseline_v1 | variant_a_td | variant_b_supervised |
|----------|-------------|--------------|----------------------|
| heuristic | **+0.346** | +0.008 | -0.702 |
| cfr | -0.098 | -0.194 | -0.327 |
| tight_passive | **+0.316** | +0.098 | -0.428 |
| tight_aggressive | **+0.300** | -0.060 | -0.342 |
| loose_passive | **+0.510** | +0.326 | -0.375 |
| loose_aggressive | +0.798 | **+0.955** | +0.377 |
| maniac | +1.165 | **+1.533** | +0.981 |
| random | +1.304 | **+1.644** | +1.406 |
| **Overall avg** | **+0.580** | +0.539 | +0.074 |
| **Worst case** | -0.098 | -0.194 | -0.702 |

### Training Summary

| Variant | Episodes | Converged | Peak vs heuristic | Time |
|---------|----------|-----------|-------------------|------|
| variant_a_td | 300K | ✅ 0.1% | +0.890 | 38 min |
| variant_b_supervised | 218 epochs | ✅ (early stop) | — | 1.6s |

---

## Analysis

### Finding 1: Variant A close to baseline but cannot beat it overall

Variant A reaches +0.539 overall vs baseline +0.580. The frozen base + modulation head
architecture is cleaner, but produces the same qualitative pattern as
`opp_stats_input_augmentation_v1/pool_random`: strong gains on weak, exploitable opponents
(maniac +1.533 vs baseline +1.165; random +1.644 vs baseline +1.304; loose_aggressive +0.955
vs +0.798) but degraded performance on structured opponents (heuristic +0.008 vs +0.346,
tight_passive +0.098 vs +0.316). The cold-start discriminability problem is not resolved
by freezing the base — the modulation head still sees uninformative inputs for the first
~20 hands of each session.

### Finding 2: Variant B fails catastrophically (train/eval distribution mismatch)

Variant B (supervised, +0.074 overall) is dramatically worse than both Variant A and the
baseline. The root cause: training used **fixed, high-confidence prototype stats** (7-dim
vectors from 500-hand calibration, confidence ≈ 0.962), but at evaluation time the stats
tracker starts fresh (confidence = 0, features at pool prior). The modulation head was
trained on a distribution it never sees in practice during early evaluation hands. By the
time the live stats converge toward the prototype values (~500 hands in), the evaluation is
already largely over. This is a fundamental train/eval mismatch, not a model capacity issue.

### Finding 3: Preliminary analysis correctly predicted Variant B's weakness

The F-ratio = 0.009 from `preliminary_analysis.py` flagged that opponent type explains very
little of the EV residual variance. Combined with the distribution mismatch above, Variant B
had two structural disadvantages before training began.

### Finding 4: Supervised residual approach needs distribution-matching

To fix Variant B, supervised training would need to use **noisy/partial stats** matching the
live-accumulation distribution — e.g., training samples with stats drawn from a range of
confidence levels (0 to 0.96) rather than just the high-confidence prototype endpoint.

---

## Hypotheses for Follow-up

1. **Distribution-matched supervised training**: Sample opponent stats at random confidence
   levels (simulate 0–500 hands of observation) during Variant B training to close the
   train/eval gap.
2. **Longer sessions**: Increasing session length (e.g., 500 hands instead of 100) would
   reduce cold-start fraction from 20% to 4% — may help Variant A on structured opponents.
3. **Structured opponent oversampling**: During Variant A TD training, oversample sessions
   against cfr/heuristic to balance the pool_random bias toward weak opponents.
