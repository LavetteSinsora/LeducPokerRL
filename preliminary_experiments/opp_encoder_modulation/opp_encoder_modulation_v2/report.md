# Opponent Encoder Modulation v2 Report

## Status

Completed. Trained for 40,000 sessions (1.2M hands), evaluated on the promoted 5-opponent suite, and diagnosed with a 1000-hand mechanism sweep. Recommendation: do not promote.

## Question

Can explicit gate and effective-modulation regularization preserve the learned opponent encoder from v1 while restoring the small-residual safety behavior of the parent architecture?

## Control

- `opp_encoder_modulation_v1`
- `modulated_value`

## Candidate

- `opp_encoder_modulation_v2`

## What changed

- Kept the v1 encoder and auxiliary head
- Initialized the gate near 0.4
- Added gate-target regularization
- Added effective-modulation regularization

## Training setup

- Sessions: 40,000
- Hands per session: 30
- Total hands: 1.2M
- Opponent pool: `heuristic`, `value_based`, `adaptive_value`, `modulated_value`, `cfr`, plus periodic self snapshots
- Gate target: `0.4`
- Gate regularization weight: `0.5`
- Effective modulation regularization weight: `0.5`
- Auxiliary action loss weight: `0.5`

## Results

Primary evaluation against the promoted suite (`1000` rounds per matchup):

| Agent | Avg | Worst | Best | Std | Robustness |
|------|-----|-------|------|-----|------------|
| `opp_encoder_modulation_v2` | +0.083 | -0.174 | +0.421 | 0.246 | **-0.286** |
| `modulated_value` (control) | +0.110 | -0.167 | +0.427 | 0.223 | **-0.225** |
| `opp_encoder_modulation_v1` | +0.114 | -0.147 | +0.235 | 0.153 | **-0.115** |

Per-opponent scores:

| Opponent | v2 | control | v1 |
|---------|----|---------|----|
| heuristic | +0.421 | +0.217 | +0.235 |
| value_based | +0.246 | +0.427 | +0.184 |
| adaptive_value | -0.174 | +0.031 | +0.190 |
| modulated_value | +0.017 | +0.041 | +0.109 |
| cfr | -0.093 | -0.167 | -0.147 |

Headline:

- v2 did not beat either `modulated_value` or v1 on average.
- v2 improved the `heuristic` matchup substantially, and slightly improved on `cfr` relative to the control.
- v2 lost most clearly on `adaptive_value`, and it did not preserve v1's stronger robustness profile.

## Diagnostics

Action-prediction accuracy and modulation usage (`1000` diagnostic hands per opponent):

| Opponent | Accuracy | Mean Gate | Mean |delta| |
|---------|----------|-----------|------|
| heuristic | 0.376 | 0.557 | 0.453 |
| value_based | 0.671 | 0.564 | 0.426 |
| adaptive_value | 0.706 | 0.564 | 0.419 |
| modulated_value | 0.574 | 0.561 | 0.422 |
| cfr | 0.401 | 0.554 | 0.403 |

Compared with v1:

- The gate is no longer saturated near `1.0`; it now stays around `0.55` to `0.56`.
- Action prediction improved a bit for `value_based` and `adaptive_value`, but got worse for `heuristic`, `modulated_value`, and `cfr`.
- The model still learns opponent-discriminative signal for value-style opponents more than for the full opponent spectrum.

## Follow-up diagnosis

### 1. The gate regularizer worked only superficially

The gate moved from v1's `~1.0` regime down to `~0.56`, so the explicit gate target did change the surface behavior.

### 2. The residual compensated for the smaller gate

Approximate effective modulation stayed close to v1 because the residual grew larger as the gate shrank.

| Opponent | v1 effective modulation | v2 gate | v2 mean |delta| | v2 approx. effective modulation |
|---------|--------------------------|---------|---------|----------------------------------|
| heuristic | 0.253 | 0.557 | 0.453 | 0.252 |
| value_based | 0.303 | 0.564 | 0.426 | 0.240 |
| adaptive_value | 0.303 | 0.564 | 0.419 | 0.237 |
| modulated_value | 0.305 | 0.561 | 0.422 | 0.237 |
| cfr | 0.253 | 0.554 | 0.403 | 0.223 |

This is the central v2 finding. Regularizing the gate alone is not enough if the residual network can simply scale itself up.

### 3. Training shifted toward the auxiliary objective

Training-history summary:

- v1 action loss: `1.091 -> 0.850`
- v2 action loss: `1.083 -> 0.671`
- v1 value loss: `6.025 -> 3.074`
- v2 value loss: `5.954 -> 4.033`

So v2 learned the prediction task more cleanly than v1, but it learned the value objective worse.

## Interpretation

v2 is an informative failure.

What improved:

- It prevented the literal gate saturation seen in v1.
- It improved the `heuristic` matchup materially.
- It improved action prediction on the two strongest value-style opponents.

What failed:

- It still did not restore the parent architecture's small-correction safety regime.
- The residual network compensated for the smaller gate, so effective modulation stayed much larger than the control's.
- The added regularization and prediction pressure appear to have traded away value quality.

The net result is that v2 fixed the visible symptom from v1, but not the underlying mechanism.

## Next step

The next experiment should constrain the *effective residual path itself*, not just the gate scalar.

The cleanest follow-up directions are:

- cap or normalize the residual magnitude directly relative to the base value
- schedule the auxiliary loss so value learning stabilizes first
- make the encoder input richer than 4 macro stats, since the current encoder may be using the prediction loss mainly as a shortcut on coarse behavior frequencies
