# Modulated Value Agent

> Frozen pretrained base with gated opponent-specific modulation -- the first agent to beat every opponent in the tournament.

| Property | Value |
|----------|-------|
| **ID** | `modulated_value` |
| **Parent** | adaptive_value |
| **Round** | 3 |
| **Rank** | #1 / 17 |
| **Avg Score** | +0.967 |
| **Robustness** | +0.199 |

---

## Motivation

After two rounds of experiments, a clear pattern had emerged: adaptive_value (19-dim obs with opponent stats) consistently ranked at or near the top, but attempts to improve it through feature engineering (adaptive_history), population training (pop_adaptive), or longer training (extended_adaptive) all fell short. The hypothesis behind modulated_value was architectural: instead of concatenating opponent information into the input, use a **gated modulation** mechanism that learns when and how much to adjust a strong base value estimate.

The core design question: can we separate "universal game understanding" (frozen base) from "opponent-specific adjustments" (trainable modulation), and use a confidence gate to control the blend?

---

## Architecture

ModulatedValueAgent uses a **three-network architecture** -- a significant departure from all other agents in the project:

```
V(s, opp) = V_base(s) + gate(opp_stats) * delta(s, opp_stats)
```

### Network 1: Frozen Base (V_base)
```
ValueNetwork(15 -> 64 -> 64 -> 1)
  - Input: 15-dim game state encoding (hand, board, pot, position, round, etc.)
  - Architecture: 2-layer MLP with ReLU, 64 hidden units
  - Pretrained from value_based agent (the #2 overall agent)
  - FROZEN during training -- no gradients flow through this network
  - Parameters: 5,249 (all frozen)
```

### Network 2: Modulation Network (delta)
```
ModulationNetwork(19 -> 32 -> 32 -> 1)
  - Input: 15-dim game state + 4-dim opponent stats = 19 dimensions
  - Architecture: 2-layer MLP with ReLU, 32 hidden units
  - Produces a scalar adjustment (delta) to the base value
  - Parameters: 1,729 (trainable)
```

### Network 3: Gate Network (gate)
```
GateNetwork(4 -> 16 -> 1 -> sigmoid)
  - Input: 4-dim opponent stats only (fold_rate, raise_rate, fold_to_raise, confidence)
  - Architecture: Single hidden layer with ReLU, output through sigmoid
  - Produces a gate value in [0, 1] controlling modulation strength
  - Design intent: low gate when stats are unreliable, high when confident
  - Parameters: 97 (trainable)
```

**Total parameters**: 7,075 (1,826 trainable, 5,249 frozen)

The opponent stats vector (4 dimensions) encodes:
- `fold_rate`: opponent's folding frequency
- `raise_rate`: opponent's raising frequency
- `fold_to_raise`: opponent's fold-to-raise-ratio
- `confidence`: statistical confidence (based on sample count)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Session-based self-play (via ModulatedValueTrainer) |
| Episodes | ~667 sessions x 30 hands = ~20,000 hands |
| Learning Rate | 1e-4 (Adam) |
| Batch Size | 30 hands/session |
| TD Method | TD(0) with modulated value computation |
| Base Model | Pretrained value_based weights (frozen) |
| Trainable Params | 1,826 (mod_net + gate_net only) |
| Training Time | 80.6 seconds |
| Updates | 333 |
| Final Loss | 8.30 |
| Eval vs Heuristic | +0.20 |

Key training detail: the optimizer only updates `mod_net` and `gate_net` parameters. The base network stays in `eval()` mode permanently. TD targets are computed through the full modulated architecture: `V_base + gate * delta`.

---

## Tournament Results

### Overall Performance
ModulatedValueAgent is the **first agent in the entire project to beat every single opponent** with a positive margin. Its worst-case score (+0.126 vs value_based) is positive, making it the only agent with a positive robustness score above the original baselines.

| Opponent | Score | | Opponent | Score |
|----------|-------|-|----------|-------|
| heuristic | +0.186 | | entropy_ac | +0.657 |
| value_based | +0.126 | | pop_adaptive | +1.597 |
| adaptive_value | +0.160 | | adaptive_history | +1.003 |
| aux_value | +0.875 | | target_value | +1.624 |
| actor_critic | +1.173 | | td_variant | +1.504 |
| history_value | +1.347 | | pruned_history | +1.050 |
| decay_adaptive | +1.092 | | curriculum | +1.579 |
| nstep_value | +0.933 | | extended_adaptive | +0.567 |

**Hardest opponent**: value_based (+0.126) -- its own frozen base
**Easiest opponent**: target_value (+1.624)
**Beats all 16 opponents**: Yes

### Performance by Opponent Category
| Category | Avg Score |
|----------|-----------|
| Rule-based (heuristic) | +0.186 |
| Round 0-1 RL agents | +0.796 |
| Round 2 RL agents | +1.163 |
| Round 3 RL agents | +1.175 |

---

## Diagnosis & Findings

Extensive diagnostic experiments revealed that modulated_value's success comes from a fundamentally different mechanism than what was designed.

### 1. Gate Behavior -- Opposite of Design Intent

The gate was designed to increase with confidence ("trust opponent stats more as you see more data"). Instead, it learned the **opposite**:

| Confidence | Gate Value |
|------------|-----------|
| 0.00 (no data) | 0.477 |
| 0.25 | 0.462 |
| 0.50 | 0.447 |
| 0.75 | 0.435 |
| 1.00 (full confidence) | 0.416 |

The gate **decreases** monotonically with confidence. It stays in a narrow band [0.39, 0.48] -- never fully opening or closing. The gate's most attended input is `confidence` (weight 0.283), followed by `raise_rate` (0.268).

Interpretation: the gate learned that the base network is already strong, so the optimal strategy is to **suppress modulation when stats are reliable**, protecting the proven base from unnecessary perturbation.

### 2. Delta Magnitudes -- Tiny Corrections

| Metric | Value |
|--------|-------|
| Average \|V_base\| | 0.780 |
| Average \|delta\| | 0.115 |
| Average gate output | ~0.44 |
| Effective modulation (gate x delta) | ~0.046 |
| Modulation as % of base | **~6%** |

The deltas are predominantly negative corrections, suggesting the modulation network slightly dampens the base's value estimates rather than adjusting them for specific opponents.

### 3. Ablation Study

Tested four variants against heuristic, value_based, and adaptive_value (1000 rounds each):

| Variant | Avg Score |
|---------|-----------|
| No gating (gate=1, delta always applied) | +0.183 |
| Base only (gate=0, no delta at all) | +0.163 |
| Full model (trained gate x delta) | +0.150 |
| Plain pretrained base (original value_based) | +0.068 |

The ablation reveals a striking finding: **base-only slightly outperforms the full model**. The modulation and gating contribute essentially nothing -- in fact, they slightly hurt performance. The real value comes from inheriting the strong pretrained base that is structurally protected from corruption.

### 4. Weight Analysis

| Component | Param Norm |
|-----------|-----------|
| Base network | 20.89 |
| Modulation network | 8.65 |
| Gate network | 4.09 |

The modulation network attends roughly equally to game state features (avg weight 0.115) and opponent stats (avg weight 0.120), ratio 1.05.

---

## Why It Works: "First, Do No Harm"

The modulated_value agent succeeds through a **structural protection mechanism**, not opponent exploitation:

1. **Strong pretrained floor**: The frozen ValueNetwork from value_based (the #2 overall agent) provides an excellent default strategy that cannot be degraded by training.

2. **Bounded perturbation**: With gate in [0.39, 0.48] and |delta| averaging 0.115, the maximum possible deviation from base is ~6%. The architecture is **structurally incapable** of catastrophically deviating from the strong base.

3. **Self-regulating gate**: The gate learned to suppress its own influence as confidence increases, further protecting the base from unnecessary modification.

4. **Robustness maximization**: The robustness metric (avg - 1.5 x std) rewards consistency. By barely deviating from a strong base, modulated_value achieves both high average AND low variance (std = 0.512, lowest among top-3 agents).

This is fundamentally a **transfer learning** success story. The modulation/gating architecture is essentially a very expensive way to implement "freeze most of the model" -- which turned out to be exactly the right approach.

---

## Assumptions & Limitations

1. **Gate operates in a narrow range**: The gate output stays within [0.39, 0.48] regardless of game state, opponent statistics, or confidence level. It never fully opens (trusting modulation entirely) or fully closes (ignoring modulation entirely). This narrow operating range means the gate provides minimal adaptive control -- it is approximately a constant multiplier of ~0.44, making the gating mechanism largely decorative.

2. **Gate decreases with confidence (opposite of design intent)**: The gate was designed so that higher confidence in opponent statistics would increase the gate value (trust the modulation more). Instead, it learned the opposite: gate = 0.477 at zero confidence, decreasing monotonically to 0.416 at full confidence. The architecture learned that the pretrained base is already strong, so reliable opponent data is a signal to SUPPRESS modulation (protect the base) rather than enhance it.

3. **Modulation always negative**: The delta network produces predominantly negative adjustments (~-0.16 preflop, ~-0.03 postflop pair). The effective modulation (gate x delta) averages ~0.046, which is only ~6% of the base value magnitude. The modulation systematically dampens the base's estimates rather than producing opponent-specific adjustments in both directions.

4. **Oracle analysis shows unrealized potential**: An oracle analysis (using true opponent hand information) shows 0.23 MSE improvement potential with a state-aware gate. The current 4-dimensional gate input (fold_rate, raise_rate, fold_to_raise, confidence) cannot distinguish between game states -- the gate produces the same output whether the agent holds J preflop or K with a paired board. A gate that also takes game state features as input could produce meaningfully different modulation strengths across situations.

5. **Success attribution ambiguity**: It remains an open question whether the architecture's tournament success (rank 1, beating all opponents) is due to the quality of the modulation or simply the structural protection of a strong pretrained base. The ablation study shows that base-only (gate=0, no modulation) scores +0.163 vs the full model's +0.150, suggesting modulation actually hurts slightly. The real innovation may be the frozen-base transfer learning paradigm, not the modulation/gating mechanism itself.

---

## Key Insight

Architectural constraints that prevent training from harming a good initialization can be more valuable than any algorithmic improvement -- the best strategy was to take a strong agent and make it almost impossible to break during additional training.

---

## Source Files

- Agent: `src/agents/modulated_value.py`
- Trainer: `src/training/modulated_value_trainer.py`
- Diagnosis: `experiments/diagnose_modulated_value.py`
- Diagnostic Results: `experiments/diagnose_modulated_value_results.json`
