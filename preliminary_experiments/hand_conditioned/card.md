# Hand Conditioned Action Model v1

## Status

Planned / implementing.

## Core question

Can we model opponent behavior in a more poker-native way by predicting:

`P(opponent_action | public_state, session_stats, candidate_opponent_hand)`

and then use that model as a likelihood function for Bayesian updates over the opponent's private hand?

## Why this is the next branch

The encoder-modulation line consistently showed the same pattern:

- opponent-aware signal exists
- direct value modulation is hard to control

So this branch stops trying to push the signal directly into value prediction. It learns an explicit action likelihood over candidate hands instead.

## Changed axis

- Switch from generic opponent embedding prediction to **hand-conditioned opponent action likelihood**

## Planned data tuple

For each opponent decision during training:

- public state from the observer's perspective
- session stats about the acting opponent
- candidate acting hand (true hand during supervised training)
- observed action

## Success criterion

- belief top-1 accuracy beats the prior-only baseline
- posterior probability on the true hand beats the prior-only baseline
- action prediction beats naive action baselines across multiple opponent types
