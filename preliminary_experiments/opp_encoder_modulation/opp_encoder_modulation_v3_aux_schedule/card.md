# Opponent Encoder Modulation v3 Auxiliary Schedule

## Question

Did v2 underperform because the auxiliary action-prediction objective consumed too much gradient budget too early?

## Parent and changed axis

- Parent agent: `opp_encoder_modulation_v2`
- Changed axis: auxiliary action loss is delayed, then ramped in gradually

## Why this direction

v2 improved action prediction but got worse on value loss. This experiment isolates that tradeoff without changing the architecture.

## What stays fixed

- Same agent architecture as v2
- Same gate and modulation regularization
- Same population trainer and opponent pool

## Success criterion

- Recover value-learning quality while keeping enough opponent signal to help modulation
