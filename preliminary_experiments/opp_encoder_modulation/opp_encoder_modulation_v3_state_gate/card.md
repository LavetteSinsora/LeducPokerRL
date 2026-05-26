# Opponent Encoder Modulation v3 State Gate

## Question

Is the gate too flat because it only sees the opponent embedding and not the strategic state where modulation might actually matter?

## Parent and changed axis

- Parent agent: `opp_encoder_modulation_v2`
- Changed axis: gate input changes from `z_opp` to `[state, z_opp]`

## Why this direction

v2's gate values stayed in a very narrow range across opponents. This experiment tests whether making gating state-aware produces more useful selectivity.

## What stays fixed

- Same encoder
- Same residual network
- Same losses and population training setup as v2

## Success criterion

- More differentiated gate behavior and better robustness than v2
