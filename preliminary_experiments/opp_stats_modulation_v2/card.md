# card.md — opp_stats_modulation_v2

## Hypothesis

`opp_stats_modulation_v1/variant_a_td` (direct residual, no gate) reached +0.539 avg vs
baseline +0.580, leaving a gap of −0.041 chips/round. Two candidate causes:

1. **Training recipe mismatch**: v1 sampled all 8 opponents uniformly, giving CFR and
   heuristic (the most structured and hardest-to-beat agents) only 12.5% of training
   sessions each. Since these dominate the robustness penalty, underexposure biases
   the modulation head toward exploiting weak opponents.

2. **Gate architecture**: The canonical `modulated_value` gate only sees opponent stats
   (4-dim), not the game state. The `EV_variation_analysis` found that 51.65% of states
   have opponent-driven action switching — different states differ in how much opponent
   strategy matters. A state-aware gate could suppress noisy cold-start modulation for
   opponent-invariant states while opening for high-sensitivity ones.

This experiment tests both dimensions independently under a single new training recipe.

## Changed Axis (one per variant)

| Variant | What changes | What stays fixed |
|---------|-------------|-----------------|
| `variant_a_ungated` | Training recipe: CFR + heuristic 3× oversampled, 200K episodes (was 300K + uniform) | Architecture: same 22→32→32→1 direct residual |
| `variant_b_state_gated` | Architecture: gate input = state+stats (22-dim) instead of no gate; total-value TD target | Training recipe: same new recipe as variant_a |

## Success Criteria

- Primary: at least one variant achieves robustness (`avg − 1.5 × std`) > `baseline_value_v1`
- Secondary: either variant beats `opp_stats_modulation_v1/variant_a_td` avg (+0.539) by ≥ 0.03 chips/round
- Stretch: robustness exceeds canonical `modulated_value` robustness score

## Failure Criteria / Exit Conditions

- If both variants fail to beat v1 variant_a avg after training convergence → cold-start
  is the fundamental bottleneck, not training recipe or gate design → proceed to v4
  (cold-start-fixed supervised approach)
- If variant_b gate values don't differentiate high/low EV-variance states after training
  → state-conditioned gating cannot learn the right signal from TD alone

## Training Config

| Parameter | Value | Source |
|-----------|-------|--------|
| Episodes | 200,000 | EVAL_CONFIG.json |
| Learning rate | 1e-4 | EVAL_CONFIG.json |
| Batch size | 32 | EVAL_CONFIG.json |
| Optimizer | Adam | EVAL_CONFIG.json |
| Session length | 100 hands | matches eval |
| CFR/heuristic weight | 3× | this experiment |
| Other opponents weight | 1× | this experiment |

## Evaluation Protocol

- Tool: `OpponentModeling/comparison_protocol.py::evaluate_stat_aware_pool`
- Rounds: 5,000 per opponent (final eval), 500 per opponent (in-training eval every 5K ep)
- Session length: 100 (matches training)
- Position: alternating (both seats covered)
- Key metric for checkpointing: robustness = avg − 1.5 × std

## Baselines to Report

| Baseline | avg | robustness |
|----------|-----|-----------|
| `baseline_value_v1` | +0.580 | — |
| `opp_stats_modulation_v1/variant_a_td` | +0.539 | — |
| canonical `modulated_value` (agents/) | — | best in pool |

## Follow-up Experiments (if this fails)

- **v3**: EV-variance-weighted TD loss (focus gradients on opponent-sensitive states)
- **v4**: Cold-start-fixed supervised (train on curriculum of confidence levels 0→1)
- **v5**: Street-specific modulation heads (separate preflop / flop modules)
