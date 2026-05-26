# Opponent Encoder Modulation v3 Closeout

All three v3 follow-ups were trained for `40,000` sessions (`1.2M` hands) and evaluated on the promoted 5-opponent suite.

## Result summary

| Agent | Avg | Worst | Std | Robustness | Key mechanism |
|------|-----|-------|-----|------------|---------------|
| `opp_encoder_modulation_v1` | +0.099 | -0.075 | 0.158 | **-0.139** | encoder helps, gate saturates |
| `opp_encoder_modulation_v2` | +0.083 | -0.174 | 0.246 | -0.286 | gate reduced, residual compensates |
| `opp_encoder_modulation_v3_relative_cap` | +0.017 | -0.148 | 0.234 | -0.333 | effective residual truly constrained |
| `opp_encoder_modulation_v3_aux_schedule` | -0.023 | -0.327 | 0.221 | -0.355 | optimization improved, play got worse |
| `opp_encoder_modulation_v3_state_gate` | +0.073 | -0.124 | 0.212 | -0.245 | best v3, still below v1 |

## Main conclusions

### 1. The modulation family did not beat v1

That is the headline. v1 remains the best result in this line.

### 2. The v3 experiments clarified the bottleneck

- `relative_cap` proved we can mechanically control the correction path, but doing so removed too much useful exploitative signal
- `aux_schedule` showed that better optimization metrics are not enough; the learned behavior can still regress strategically
- `state_gate` was the strongest v3 repair, but still not enough to make direct value modulation competitive with the earlier simpler v1

### 3. This now looks like a family-level limit, not a small bug

The evidence is increasingly consistent:

- opponent signal is real
- but pushing that signal directly into a value-correction path is difficult to control
- and when we control it aggressively enough, the benefit largely disappears

## Recommendation

Pivot the next novel branch away from direct value modulation and toward **belief/search**:

1. `hand_conditioned_action_model_v1`
2. `belief_lookahead_v1`

The modulation family is still useful as a control family, but it is no longer the strongest place to spend the next research step.
