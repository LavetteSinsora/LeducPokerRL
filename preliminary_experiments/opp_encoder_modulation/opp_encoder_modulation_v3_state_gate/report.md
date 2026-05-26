# Opponent Encoder Modulation v3 State Gate Report

## Status

Planned / running.

## Question

Can state-aware gating produce a more useful trust mechanism than v2's nearly constant embedding-only gate?

## Control

- `opp_encoder_modulation_v2`
- `opp_encoder_modulation_v1`
- `modulated_value`

## Candidate

- `opp_encoder_modulation_v3_state_gate`

## What changed

- Gate now conditions on `[state, z_opp]` instead of `z_opp` alone

## Results

Pending.
