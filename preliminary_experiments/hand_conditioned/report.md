# Hand Conditioned Action Model v1 Report

## Status

Completed.

## Question

Can a hand-conditioned action likelihood model produce belief updates that are genuinely more informative than card-removal priors alone?

## Results

Held-out evaluation against `heuristic`, `value_based`, `adaptive_value`, `modulated_value`, and `cfr`, using `modulated_value` as the observing probe, was positive across all tracked belief metrics.

### Aggregate Metrics

- Mean action accuracy: `0.8038`
- Mean belief top-1 accuracy: `0.7424`
- Mean prior top-1 accuracy: `0.5587`
- Belief top-1 lift: `+0.1837`
- Mean true-hand posterior probability: `0.6460`
- Mean true-hand prior probability: `0.4879`
- True-hand probability lift: `+0.1581`
- Mean TVD shift: `0.2527`

### Per-Opponent Highlights

- `value_based`: strongest overall held-out fit, with action accuracy `0.8348` and belief top-1 accuracy `0.8318`
- `adaptive_value`: highest action accuracy at `0.8741`
- `modulated_value`: similarly strong, with action accuracy `0.8786` and posterior true-hand probability `0.6975`
- `heuristic`: still clearly above prior baseline, but weaker than learned value-family opponents
- `cfr`: hardest target, with the lowest action accuracy `0.6646` and lowest belief top-1 accuracy `0.6043`

### Interpretation

This experiment succeeded at its immediate goal. The learned hand-conditioned action model produces substantially more informative beliefs than card-removal priors alone, and it does so on every held-out opponent in the evaluation set.

The gains are largest against learned value-style agents and smallest against `cfr`, which fits the expected story: the model learns clearer hand-conditioned action regularities when the opponent family is behaviorally structured, and struggles more against equilibrium-style play.

The diagnosis traces also show meaningful posterior movement rather than cosmetic updates. Strong actions like early raises frequently push mass sharply toward `K`, while passive lines shift probability toward `J` or `Q` depending on board context and observed stats.
