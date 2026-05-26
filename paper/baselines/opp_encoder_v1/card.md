# Opponent Encoder Modulation v1

## Origin and motivation

The current best promoted architecture is [`modulated_value`](/Users/chrishe/Downloads/PokerRL_Vanilla/agents/modulated_value/README.md): a frozen pretrained value base plus a bounded opponent-specific residual. Its weakness is that the opponent side is still just the raw 4-feature macro stats vector.

This experiment asks a narrower question:

- Can we replace the raw stats path with a learned opponent embedding without giving up the structural safety of the frozen base?
- Can we force that embedding to carry behaviorally useful information by training it to predict the opponent's next action?

The aim is not to invent a brand new family yet. The aim is to keep the successful transfer-learning story intact and improve only the opponent representation path.

## Parent and changed axis

- Parent agent: `modulated_value`
- Frozen base checkpoint: `agents/value_based/checkpoint.pt`
- Changed axis: raw opponent stats conditioning -> learned opponent embedding trained with an auxiliary action-prediction objective

## Mechanism story

The intended mechanism is:

1. `value_based` already gives a strong generic state value estimate.
2. Session-level opponent stats contain coarse information about how this opponent deviates from equilibrium-like play.
3. An encoder can compress those stats into a learned embedding `z_opp`.
4. A modulation network can use `(state, z_opp)` to produce a bounded correction to the frozen base value.
5. A separate auxiliary head predicts the opponent's next action from our information set plus `z_opp`.
6. If the auxiliary head succeeds, then `z_opp` is carrying behaviorally meaningful information rather than acting as an arbitrary latent.

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

## Why this is a reasonable first step

- It preserves the strongest lesson in the repo so far: do not let training destroy a good pretrained value base.
- It tests a meaningful representation-learning idea without jumping all the way to FiLM, MoE, or self-strategy encoding.
- It creates architecture-specific diagnostics:
  - action-prediction accuracy
  - learned gate usage
  - residual magnitude across opponents

## Main risks

1. Stats may be too low-dimensional for an encoder to help.
2. The auxiliary action head may learn generic public-state regularities rather than opponent-specific tendencies.
3. Pure self-play would likely make the encoder useless because the "opponent" keeps collapsing toward the same policy.

Because of risk 3, this experiment is scaffolded around a fixed rotating opponent pool from day one.

## Training objectives

### 1. Value objective

Train the modulation path so the frozen base plus residual matches empirical return through TD(0)-style bootstrapping on the agent's post-action chain:

`V(s_t) = V_base(s_t) + g(z_t) * Delta(s_t, z_t)`

Target:

- terminal reward on the last transition
- next-step bootstrap value otherwise

### 2. Auxiliary action-prediction objective

When the opponent acts, record our information set just before that action and train:

`P(a_opp | our_state_view, z_opp)`

with cross-entropy against the opponent's actual action.

This keeps the auxiliary target honest:

- input does not leak the opponent's private card
- target is behaviorally meaningful
- the encoder only sees macro stats, not the whole state

## Control and comparison

Primary control:

- promoted `modulated_value`

What stays fixed relative to the control:

- frozen `value_based` base checkpoint
- value-style action selection
- session-based opponent stats
- evaluation suite

What changes:

- raw stats are replaced by a learned embedding
- auxiliary action-prediction loss is added for the embedding path

## Planned training setup

- Training mode: population-based session training
- Training agent seat: player 0 only
- Opponent pool: `heuristic`, `value_based`, `adaptive_value`, `modulated_value`, `cfr`
- Self snapshots: periodic, frozen copies of the current experiment agent
- Recommended full budget: `40_000` sessions x `30` hands/session
- Recommended initial aux weight: `0.5`

The trainer deliberately uses only the training agent's value chain. This avoids the off-policy opposite-reward issue that already harmed earlier adversarial experiments in this repo.

## What success looks like

Primary success:

- beats or matches `modulated_value` on robustness against the promoted evaluation suite

Secondary success:

- better action-prediction accuracy than a trivial majority baseline
- opponent-dependent differences in gate usage or residual magnitude
- no collapse of the residual path into extreme values

## What failure would still teach us

- If action prediction stays weak, the current stats are probably too impoverished.
- If action prediction works but value does not improve, then the representation is behaviorally meaningful but not value-relevant.
- If value gets worse while residual magnitude grows, the encoder path is probably destabilizing the protected base.

## Files in this experiment folder

- `agent.py`: experiment-only agent definition
- `trainer.py`: population-based joint trainer
- `train.py`: entrypoint for the full run
- `eval.py`: promoted-suite evaluation
- `diagnose.py`: architecture-specific probes
- `report.md`: running report to fill in after results
- `summary.json`: machine-readable experiment record
