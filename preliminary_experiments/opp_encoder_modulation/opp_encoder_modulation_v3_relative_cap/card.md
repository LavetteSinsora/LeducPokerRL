# Opponent Encoder Modulation v3 Relative Cap

## Question

Can we restore the parent architecture's small-correction behavior by capping the **effective residual** directly, instead of only regularizing the gate?

## Parent and changed axis

- Parent agent: `opp_encoder_modulation_v2`
- Changed axis: hard-cap `gate * delta` relative to the frozen base value magnitude

## Why this direction

v2 lowered the gate, but the residual grew to compensate. This experiment tests the most direct fix: bound the effective residual itself.

## What stays fixed

- Frozen `value_based` base checkpoint
- Same encoder and auxiliary next-action head as v2
- Same population trainer
- Same opponent pool

## Success criterion

- Match or beat `modulated_value` and v1 on robustness
- Drive effective residual size materially below v2
- Avoid collapsing to base-only behavior
