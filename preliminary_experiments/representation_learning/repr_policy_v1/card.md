# repr_policy_v1: Policy Learning on Contrastive Representation

## Research Question
Does a policy network using the contrastive-learned 8-dim representation as input outperform
vanilla REINFORCE trained on raw 15-dim features?

## Motivation
The `contrastive_repr_v1` experiment trains an encoder `15 → 64 → 64 → 8` using reward-based
contrastive learning. The contrastive objective encourages the encoder to cluster game states
by their eventual outcome (win/loss) and strategic similarity, rather than surface-level feature
proximity. If this encoder has compressed the game state into a more strategically meaningful
8-dim space — filtering out observation noise and emphasizing value-relevant distinctions — then
a REINFORCE policy head trained on top of it should:

1. Receive a smoother loss landscape (less reward noise per gradient step)
2. Learn faster (fewer episodes to convergence)
3. Potentially reach higher final performance (better asymptotic policy)

## Hypothesis
A frozen-encoder policy (`repr_policy_v1/frozen`) matches or beats vanilla REINFORCE
(`repr_policy_v1/baseline`) in average chips per round vs. the heuristic agent after 20,000
training episodes.

## Variants

| Variant   | Encoder     | Policy Head         | Trainable Parameters |
|-----------|-------------|---------------------|----------------------|
| baseline  | None (raw)  | 15 → 64 → 64 → 3   | All (same as legacy PG) |
| frozen    | Fixed (8d)  | 8 → 64 → 64 → 3    | Policy head only |
| finetune  | Unfrozen    | 8 → 64 → 64 → 3    | Encoder + policy head |

The policy head architecture `8 → 64 → 64 → 3` matches the parameter count of the vanilla
baseline `15 → 64 → 64 → 3` as closely as possible, making this a fair comparison.

## Success Criteria
- repr-based policy (frozen or finetune) achieves avg_chips_vs_heuristic ≥ baseline
- OR frozen variant shows faster convergence (reaches baseline performance in fewer episodes)
