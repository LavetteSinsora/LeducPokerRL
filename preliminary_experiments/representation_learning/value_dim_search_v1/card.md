# value_dim_search_v1

## Research Question
What is the minimum hidden width that allows a value network to learn a strong Leduc Hold'em strategy?

## Motivation
The existing value-based agent uses architecture `15 → 64 → 64 → 1`. Bottleneck analysis from
`contrastive_repr_v1` suggests Leduc's strategic value function may have low intrinsic
dimensionality — the contrastive encoder was able to capture meaningful structure in 8 dimensions.
If the value function itself is low-dimensional, we should be able to shrink the network
significantly without sacrificing performance, reducing compute and risk of overfitting.

## Hypothesis
A network as small as `15 → 16 → 1` can match the performance of `15 → 64 → 64 → 1` to within
0.05 chips/round versus the heuristic opponent. Leduc Hold'em has only 6 distinct cards, 2 rounds,
and limited betting — the effective dimensionality of the value function is low.

## Architectures Tested

| Run | Architecture       | Parameters (approx) |
|-----|--------------------|---------------------|
| A   | 15 → 32 → 32 → 1  | ~1,600              |
| B   | 15 → 32 → 16 → 1  | ~1,100              |

Baseline (value_based): `15 → 64 → 64 → 1` (~8,300 parameters)

## Success Criteria
Identify the smallest architecture within **0.05 chips/round** of the full baseline
(`15 → 64 → 64 → 1`), using the same training recipe (TD(0) self-play, 20,000 episodes,
Adam lr=1e-4, batch_size=32).

## Recommended Follow-up
If both Run A and Run B succeed, test `15 → 16 → 16 → 1` and `15 → 16 → 1` to find the absolute
minimum. If Run B fails, Run A's result (`15 → 32 → 32 → 1`) is the recommended floor.
