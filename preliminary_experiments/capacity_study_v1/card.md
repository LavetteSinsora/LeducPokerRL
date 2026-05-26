# Experiment Card: capacity_study_v1

---

## Research Question

What is the minimum neural network capacity (depth × width) needed to approximate the
value function in Leduc Hold'em with near-optimal performance against a strong heuristic
opponent?

## Parent Agent / Prior Experiment

- Parent: `value_based` (15→64→64→1, TD(0) self-play, Adam lr=1e-4)
- Prior experiment: None — this is a foundational scaling study

## Single Changed Axis

**Multi-axis sweep** (two phases):
- Phase 1: Number of hidden layers (0, 1, 2, 3) at width=32, plus original [64,64] baseline
- Phase 2: Width of the last hidden layer (4, 8, 16, 32, 64), depth fixed to Phase 1 winner

Everything else is held constant: encoding (15-dim), TD(0) objective, self-play,
Adam optimizer, lr=1e-4, batch=32, temperature=1.0.

## Mechanism Story

1. Leduc Hold'em has a small, discrete state space (6 cards × 2 rounds × 3 hands ×
   4 board cards × ~3 betting positions = O(1,000) distinct states).
2. The value function V(s) maps each state to an expected chip outcome — a low-rank
   function over a factored input space.
3. If V is low-rank, a shallow or narrow network should be sufficient; extra capacity
   will only marginally improve fit and may slow convergence.
4. By sweeping depth and width independently, we can identify the "knee" of the
   performance-vs-complexity curve and characterise the effective dimensionality
   of the value function.

## Hypothesis

- **H1 (depth):** Depth-2 [32,32] matches or exceeds the baseline [64,64] within
  ≤5% of its final avg_chips_vs_heuristic, using ≤40% of the parameter count.
- **H2 (width):** A last-layer width of ≥8 retains ≥95% of the depth-2 [32,32]
  performance, suggesting the effective value-function rank is ≤8.

## Success Criteria

- Primary: Identify the smallest architecture achieving avg_chips_vs_heuristic ≥ 90%
  of the [64,64] baseline's final score.
- Secondary: Produce smooth learning curves that clearly show convergence behaviour
  for every configuration, enabling data-driven architecture selection.

## Required Artifacts

- [x] `outputs/capacity_study_v1/<config_id>/checkpoint.pt` (per config)
- [x] `outputs/capacity_study_v1/<config_id>/eval_history.json` (per config)
- [x] `outputs/capacity_study_v1/<config_id>/train_config.json` (per config)
- [x] `outputs/capacity_study_v1/results.json` (combined summary)
- [x] `outputs/capacity_study_v1/figures/depth_comparison.png`
- [x] `outputs/capacity_study_v1/figures/width_comparison.png`
- [x] `outputs/capacity_study_v1/figures/final_bar.png`
- [x] `outputs/capacity_study_v1/report.md`

## Minimum Training Budget

Value/policy network: **50,000 episodes per configuration**

## Metrics to Report

- `avg_chips_vs_heuristic` — 5,000-round position-swapped evaluation
- `n_params` — total trainable parameters
- `final_avg_chips` — mean of last 3 evaluation checkpoints (stability-adjusted)

## Risks

- Risk: Underfitting for very small models (linear, width-4) making the gap look
  larger than reality due to insufficient training epochs per parameter.
  Detection: Check if loss converges early; if so, the model is capacity-limited
  not training-limited.
- Risk: High variance in self-play evaluation masking real differences.
  Detection: 5,000-round evals are large enough to give ~0.01 chips/round SE;
  differences >0.05 are statistically meaningful.
