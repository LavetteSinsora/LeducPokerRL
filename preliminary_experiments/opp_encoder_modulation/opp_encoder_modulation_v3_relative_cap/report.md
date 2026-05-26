# Opponent Encoder Modulation v3 Relative Cap Report

## Status

Planned / running.

## Question

Can direct capping of the effective residual recover the protective behavior that v2 still failed to restore?

## Control

- `opp_encoder_modulation_v2`
- `opp_encoder_modulation_v1`
- `modulated_value`

## Candidate

- `opp_encoder_modulation_v3_relative_cap`

## What changed

- Hard cap on `gate * delta` relative to frozen base value magnitude

## Results

Pending.
