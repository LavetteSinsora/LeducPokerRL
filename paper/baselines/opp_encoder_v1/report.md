# Opponent Encoder Modulation v1 Report

## Status

Completed. Trained for 40,000 sessions (1.2M hands) and evaluated on the promoted 5-opponent suite. Recommendation: do not promote yet.

## Question

Does a learned opponent embedding, trained partly through next-action prediction, improve the frozen-base modulated value architecture over raw 4-feature stats?

## Control

- `modulated_value`

## Candidate

- `opp_encoder_modulation_v1`

## What changed

- Raw stats conditioning replaced with a learned opponent embedding
- Added auxiliary opponent next-action prediction head
- Kept the frozen `value_based` base and bounded residual structure

## Training setup

- Sessions: 40,000
- Hands per session: 30
- Total hands: 1.2M
- Opponent pool: `heuristic`, `value_based`, `adaptive_value`, `modulated_value`, `cfr`, plus periodic self snapshots
- Runtime: 2315.7 seconds wall-clock (~38.6 minutes)
- Auxiliary loss weight: 0.5

## Results

Primary evaluation against the promoted suite (`1000` rounds per matchup):

| Agent | Avg | Worst | Best | Std | Robustness |
|------|-----|-------|------|-----|------------|
| `opp_encoder_modulation_v1` | +0.099 | -0.075 | +0.333 | 0.158 | **-0.139** |
| `modulated_value` (control) | +0.154 | -0.191 | +0.342 | 0.221 | **-0.177** |

Per-opponent scores:

| Opponent | Candidate | Control | Delta |
|---------|-----------|---------|-------|
| heuristic | -0.075 | +0.342 | -0.417 |
| value_based | +0.333 | +0.266 | +0.067 |
| adaptive_value | +0.160 | +0.295 | -0.135 |
| modulated_value | +0.080 | +0.060 | +0.020 |
| cfr | -0.005 | -0.191 | +0.186 |

Headline:

- The candidate did **not** beat the control on average performance.
- It did beat the control on robustness by a small margin because it avoided the control's large negative CFR matchup.
- The result is mixed, not promotion-worthy.

## Diagnostics

Action-prediction accuracy (`1000` diagnostic hands per opponent):

| Opponent | Accuracy | Majority Baseline | Lift |
|---------|----------|-------------------|------|
| heuristic | 0.403 | 0.569 | -0.166 |
| value_based | 0.651 | 0.623 | +0.028 |
| adaptive_value | 0.672 | 0.662 | +0.010 |
| modulated_value | 0.643 | 0.530 | +0.113 |
| cfr | 0.420 | 0.495 | -0.075 |

The encoder clearly learned something for learned value-style opponents, especially `modulated_value`, but it did not produce a generally strong opponent model. Against `heuristic` and `cfr`, the auxiliary head was worse than a trivial majority-action baseline.

## Follow-up diagnosis

### 1. The modulation path matters

Base-only ablation of the trained checkpoint:

| Agent | Avg | Worst | Best | Std | Robustness |
|------|-----|-------|------|-----|------------|
| full candidate | +0.099 | -0.075 | +0.333 | 0.158 | **-0.139** |
| base-only ablation | -0.113 | -0.228 | +0.229 | 0.195 | **-0.405** |

So the learned modulation path is not decorative. It materially improves the frozen base.

### 2. The new gate lost the parent's safety property

Mean gate and effective modulation fraction of the base value:

| Agent | Mean Gate | Mod / Base Fraction |
|------|-----------|---------------------|
| candidate | ~0.999 across all opponents | 0.24 to 0.36 |
| control `modulated_value` | ~0.40 across all opponents | 0.05 to 0.07 |

This is the most important mechanistic finding in the whole experiment.

The parent agent succeeds because its gate keeps the residual small. This experiment learned the opposite behavior: the gate saturates near 1.0 almost everywhere, so the model effectively trusts the residual all the time. The architecture still benefits from the frozen base, but it no longer has the parent's "first, do no harm" protection.

### 3. The auxiliary head improved faster than value quality

Training-history summary:

- Action loss: `1.091 -> 0.851` (first 100 vs last 100 updates)
- Value loss: `6.025 -> 3.074`

This is consistent with the prediction results: the encoder got better at opponent-action prediction, but the value side did not convert that extra signal into a clean overall win over the control.

## Interpretation

This experiment is a **meaningful but mixed success**.

What worked:

- The encoder-plus-auxiliary setup did learn opponent-discriminative information.
- That information did help the learned residual path relative to a base-only ablation.
- The candidate became more robust than the control against the promoted suite, largely by avoiding the control's large CFR weakness.

What failed:

- The average score still fell short of the promoted `modulated_value` control.
- The learned gate saturated and stopped behaving like a trust mechanism.
- The action-prediction objective mostly helped on learned value-style opponents, not on the full opponent spectrum.

The net result is that representation learning helped, but it also destroyed the most valuable structural property of the parent architecture.

## Next step

Run a follow-up experiment that keeps the learned opponent encoder but restores structural protection explicitly.

The cleanest next step is:

- keep the same encoder and auxiliary head
- regularize the gate toward the parent regime (small residual usage)
- penalize large effective modulation `|gate * delta|`
- keep the same population training setup

That next experiment should test one question only:

**Can a learned opponent embedding help once the residual is forced to remain a small correction instead of becoming the dominant value signal?**
