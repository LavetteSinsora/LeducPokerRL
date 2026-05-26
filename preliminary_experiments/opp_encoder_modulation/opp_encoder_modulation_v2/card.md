# Opponent Encoder Modulation v2

## Origin and motivation

`opp_encoder_modulation_v1` established two things:

- the learned encoder and auxiliary action head do learn some opponent-discriminative signal
- the gate saturates near `1.0`, destroying the parent architecture's protective small-residual behavior

This follow-up asks a narrower question:

- Can we keep the learned opponent embedding, but force the gate and effective residual back toward the parent regime?
- Does explicit gate/residual regularization preserve the encoder's benefits without letting the residual dominate the value estimate?

The aim is not to invent a new family. The aim is to repair the failure mode uncovered in v1.

## Parent and changed axis

- Parent agent: `opp_encoder_modulation_v1`
- Frozen base checkpoint: `agents/value_based/checkpoint.pt`
- Changed axis: add explicit gate-target and effective-modulation regularization to the v1 architecture

## Mechanism story

The intended mechanism is:

1. Keep the v1 encoder and auxiliary prediction path because they showed some real signal.
2. Initialize the gate near the parent regime (`~0.4`) instead of letting it start unconstrained.
3. Add a penalty that keeps the gate near that target.
4. Add a penalty on the effective residual `gate * delta` so the learned correction stays small relative to the base model.
5. If this works, the agent should retain v1's robustness benefits without losing the parent's "do no harm" behavior.

The architecture is:

```text
public/private state from our information set
    -> frozen base value network -> V_base(s)
    -> state encoding h_s

opponent macro stats
    -> learned encoder -> z_opp

[h_s, z_opp]
    -> residual modulation net -> Delta(s, z_opp)
    -> action head -> logits over opponent next action

z_opp
    -> gate -> g(z_opp)

final value = V_base(s) + g(z_opp) * Delta(s, z_opp)
```

## Main risk

Regularizing too hard may collapse the learned residual back into an inert decoration, recovering safety but losing any value from the encoder.

## Training objectives

Same as v1, plus two regularization terms:

- gate-target penalty: `(gate - 0.4)^2`
- effective-modulation penalty: `(gate * delta)^2`

## Control and comparison

Primary controls:

- `opp_encoder_modulation_v1`
- promoted `modulated_value`

What stays fixed relative to the control:

- frozen `value_based` base checkpoint
- value-style action selection
- session-based opponent stats
- evaluation suite

What changes:

- explicit gate/residual regularization is added on top of the v1 setup

## Planned training setup

- Training mode: population-based session training
- Training agent seat: player 0 only
- Opponent pool: `heuristic`, `value_based`, `adaptive_value`, `modulated_value`, `cfr`
- Self snapshots: periodic, frozen copies of the current experiment agent
- Recommended full budget: `40_000` sessions x `30` hands/session
- Initial aux weight: `0.5`
- Gate target: `0.4`
- Gate regularization weight: `0.5`
- Effective-modulation regularization weight: `0.5`

The trainer deliberately uses only the training agent's value chain. This avoids the off-policy opposite-reward issue that already harmed earlier adversarial experiments in this repo.

## What success looks like

Primary success:

- beats or matches `modulated_value` on robustness against the promoted evaluation suite

Secondary success:

- gate profile materially below the v1 saturation regime
- modulation fraction closer to the promoted control than to v1
- no collapse to base-only behavior

## What failure would still teach us

- If gate regularization improves average performance, then v1's failure was mostly about losing structural protection, not about the encoder itself.
- If the model collapses back to base-only behavior, then the current encoder signal is too weak to justify even a controlled residual.

## Files in this experiment folder

- `agent.py`: experiment-only agent definition
- `trainer.py`: population-based joint trainer
- `train.py`: entrypoint for the full run
- `eval.py`: promoted-suite evaluation
- `diagnose.py`: architecture-specific probes
- `report.md`: running report to fill in after results
- `summary.json`: machine-readable experiment record
