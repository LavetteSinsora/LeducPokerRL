# OpponentModeling/shared_data

Reusable data artifacts shared across OpponentModeling experiments.

---

## opponent_prototype_stats.json

### What it is

A 7-dimensional behavioral embedding for each opponent archetype in the pool.
Each entry maps an opponent name to a length-7 list of floats:

```
[preflop_fold_rate, preflop_raise_rate, flop_raise_rate,
 preflop_fold_to_raise, flop_fold_to_raise, raise_after_raise_rate, confidence]
```

### What each feature means

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `preflop_fold_rate` | Fraction of pre-flop decisions where the opponent folded |
| 1 | `preflop_raise_rate` | Fraction of pre-flop decisions where the opponent raised |
| 2 | `flop_raise_rate` | Fraction of flop decisions where the opponent raised |
| 3 | `preflop_fold_to_raise` | Fraction of pre-flop raise-facing decisions where the opponent folded |
| 4 | `flop_fold_to_raise` | Fraction of flop raise-facing decisions where the opponent folded |
| 5 | `raise_after_raise_rate` | Fraction of raise-facing decisions (any round, re-raise legal) where the opponent re-raised |
| 6 | `confidence` | `n / (n + S)` where n = hands observed, S = prior_strength = 20. Reflects how much the data has shifted the prior. At 500 hands ≈ 0.962. |

### How it is produced

Run `generate_prototype_stats.py` (in this directory). For each opponent:

1. A `RandomAgent` learner plays 500 hands as Player 0. The Random agent is used so that all action contexts (facing a raise vs not) are equally represented in the denominator of each rate.
2. The opponent sits at Player 1 and plays normally.
3. An `OpponentStatsTracker` (from `opp_stats_input_augmentation_v1/stats_tracker.py`) records each opponent action and computes Beta-Bernoulli posterior means.

**Smoothing**: `p_hat = (k + α) / (n + α + β)` where `α = pool_mean × S`, `β = (1 − pool_mean) × S`, `S = 20`. Here the prior is initialized at 0.5 (neutral, not the cross-opponent pool mean) so that each opponent's prototype reflects only their own behavior.

**Confidence at 500 hands**: `500 / (500 + 20) ≈ 0.962` — the prior contributes less than 4% of the final estimate.

### How this differs from the pool-mean prior

| | Pool-mean prior | Prototype stats (this file) |
|---|---|---|
| What it represents | Average behavior across all opponents | Each individual opponent's characteristic behavior |
| When to use it | Cold-start: when no observations of the current opponent exist yet | Fixed reference embedding for a known opponent archetype |
| Computed how | Average of raw stats across 500 hands × 8 opponents | Per-opponent 500-hand calibration |
| Dimensionality | 6 scalars (one per rate, no confidence) | 7 per opponent (6 rates + confidence) |

### Opponents included

| Opponent | Type | Notes |
|----------|------|-------|
| `cfr` | Learned (Nash) | Near-optimal; close to unexploitable play |
| `value_based` | Learned (TD) | Greedy 1-step lookahead value agent; trained via self-play |
| `tight_passive` | Rule-based | Calls K/Q; folds J to raises; rarely raises |
| `tight_aggressive` | Rule-based | Raises/re-raises with strong hands; folds weak hands |
| `loose_passive` | Rule-based | Calls frequently; rarely raises |
| `loose_aggressive` | Rule-based | Raises widely; bluffs often |
| `maniac` | Rule-based | Always raises when legal |
| `random` | Rule-based | Uniform random over legal actions |
| `heuristic` | Rule-based | Pot-odds driven; most sophisticated rule-based agent |

### Reproducing

```bash
cd /path/to/PokerRL_Vanilla
python OpponentModeling/shared_data/generate_prototype_stats.py        # full 500-hand run
python OpponentModeling/shared_data/generate_prototype_stats.py --smoke  # quick 50-hand check
```

### First produced by

`opp_stats_modulation_v1` (2026-03-15) — Variant B supervised training needed fixed
per-opponent embeddings as input features for the residual modulation head.

---

## Adding new files

Place any dataset, embedding, or calibration artifact that is useful across multiple
OpponentModeling experiments here. Each file should have a corresponding `_meta` field
(if JSON) or a section in this README explaining its contents and how it was produced.
