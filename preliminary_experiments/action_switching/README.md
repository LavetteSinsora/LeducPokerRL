# action_switching_analysis_v1

This package measures whether opponent identity changes the **best current action**
at a fully-specified game state, and whether learned models reproduce that
action-switching behavior.

## What it computes

For each fully-specified state in `EV_variation_analysis` and each handcrafted
opponent archetype:

1. Enumerate legal actions.
2. Force one candidate action.
3. Roll out the rest of the hand with the acting player controlled by
   `ValueBasedAgent` and the opponent controlled by the selected archetype.
4. Estimate `Q(state, action | opponent)` by Monte Carlo.
5. Choose the oracle one-step best action for that opponent.

This yields:

- The fraction of states where the oracle best action differs across opponents.
- Per-state oracle best-action labels across the archetype set.
- Baseline action agreement with the oracle labels.
- Opponent-stat model agreement with the oracle labels.
- Performance specifically on the subset of states where opponent-aware
  action switching is actually required.

## Why this matters

The earlier EV-variation analysis measured whether `EV(state)` varies across
opponents. That is not the same as asking whether the **best action** changes.
For a greedy one-step value agent, the second question is the relevant one.

## Run

```bash
python -m OpponentModeling.action_switching_analysis_v1.analyze
python -m OpponentModeling.action_switching_analysis_v1.analyze --rollouts 50
python -m OpponentModeling.action_switching_analysis_v1.analyze --limit-states 100
```

Outputs are written under `OpponentModeling/action_switching_analysis_v1/outputs/`.
