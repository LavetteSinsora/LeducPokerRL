# Modulated Value Agent

Promoted robust agent built from a frozen value baseline plus bounded modulation.

- Parent: `adaptive_value` conceptually, `value_based` architecturally
- Mechanism: `V_base(s) + gate(stats) * delta(s, stats)`
- Training: session-based TD(0) over the modulation path
- Official checkpoint: `checkpoint.pt`
