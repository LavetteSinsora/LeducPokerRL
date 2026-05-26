# value_dim_search_v1 — Results Report

## Research Question
What is the minimum hidden width that allows a value network to learn a strong Leduc Hold'em strategy?

## Results Table

| Run | Architecture      | Params | Episodes | avg_chips vs heuristic (final eval, 500 rounds) |
|-----|-------------------|--------|----------|--------------------------------------------------|
| A   | 15 → 32 → 32 → 1 | 1,601  | 20,000   | **-0.52**                                        |
| B   | 15 → 32 → 16 → 1 | 1,057  | 20,000   | **-1.35**                                        |

## Comparison to value_based Baseline

The reference `value_based` agent (`15 → 64 → 64 → 1`, ~8,300 params) achieves approximately
**-0.075 chips/round vs heuristic** based on evaluation data from `opp_encoder_modulation_v1`
(which uses the same checkpoint at `agents/value_based/checkpoint.pt`). A well-trained
`modulated_value` derived agent achieves +0.34 vs heuristic over 1,000 rounds.

Both Run A and Run B fall significantly below baseline performance under the same 20,000-episode
training budget with identical hyperparameters (TD(0), Adam lr=1e-4, batch_size=32).

## Training Dynamics

During training, both runs showed **high TD loss (10–25) that did not converge**, and
periodic evaluation scores oscillated widely between roughly -1.6 and +0.1 chips/round.
This is characteristic of TD(0) instability in self-play: when the policy changes, the
TD targets shift, causing value estimates to chase a moving target.

Key observations:
- Run A reached +0.07 chips/round at episode 9,000 but regressed badly by episode 20,000.
- Run B touched +0.10 at episode 5,500 but similarly degraded.
- Final evaluations (500 rounds) reflect late-training policy quality, not peak quality.
- The reference `value_based` checkpoint was presumably trained longer or with additional
  stabilization (e.g., target network, replay buffer, or learning rate decay) not captured
  in the 20k-episode recipe used here.

## Key Finding

**Neither architecture is sufficient under the 20,000-episode self-play TD(0) recipe.**
The problem is not hidden width — Run A (32×32, 1,601 params) has more than adequate capacity
to represent Leduc Hold'em value. The bottleneck is **training stability**, not network size:

- The same instability pattern appears for both 32×32 and 32×16 architectures.
- Even Run A, which has 34% of the baseline's parameter count, briefly reached near-baseline
  performance (+0.07 at ep 9k), confirming capacity is not the limiting factor.
- Run B (32×16, 1,057 params) performs slightly worse, suggesting a minor capacity benefit
  from the wider first layer, but the margin is dominated by instability noise.

## Conclusion

The minimum viable capacity question cannot be cleanly answered with the current 20k-episode
TD(0) self-play recipe — training instability swamps the capacity signal. To answer the
research question properly, the experiment should be re-run with a stabilized training recipe.

## Recommended Next Step

Before testing smaller architectures (16×16 or 16×1), first **stabilize training** for the
current recipe:

1. Add a **target network** (hard update every 500 episodes) to prevent TD target drift.
2. Add a **replay buffer** (capacity ~5,000) to decorrelate consecutive updates.
3. Alternatively, increase episodes to 50,000+ and add learning rate decay after episode 15,000.

Once training converges reliably, re-run runs A and B to cleanly measure the capacity effect.
If both converge to within 0.05 chips/round of baseline, proceed to test 16×16 and 16×1.
