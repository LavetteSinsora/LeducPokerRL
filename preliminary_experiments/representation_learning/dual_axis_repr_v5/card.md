# dual_axis_repr_v5: Subspace-Partitioned Dual-Axis Representation

**Status**: Complete
**Date**: 2026-03-12

## One-line summary

Split the 8-dim encoder output into two disjoint 4-dim subspaces and apply each loss exclusively to its own slice, achieving both reward metric structure and hand identity simultaneously for the first time.

## What changed from v4

| | v3 (hybrid) | v4 (EMA norm) | v5 (subspace) |
|---|---|---|---|
| Reward loss applied to | full 8-dim | full 8-dim | **dims 0:4 only** |
| Hand loss applied to | full 8-dim | full 8-dim | **dims 4:8 only** |
| Interference prevention | none | EMA normalization | structural isolation |

## Results (20k episodes)

| Metric | Value |
|---|---|
| Full embedding effective dim (80%) | 3 |
| Reward subspace Spearman rho (dims 0:4) | **0.543** |
| Full embedding Spearman rho | 0.536 |
| Opp hand accuracy (full embedding) | 0.652 |
| Opp hand accuracy (hand subspace, dims 4:8) | **0.676** |

## Targets achieved

- Reward Spearman rho >= 0.163: YES (0.543, 3.3x above target)
- Opp hand accuracy >= 0.620: YES (0.676)
- Both simultaneously: **YES — first time in this research line**

## Config

```
lambda_hand=1.0, lambda_var=0.1, lr=1e-4, batch=256, episodes=20000
```
