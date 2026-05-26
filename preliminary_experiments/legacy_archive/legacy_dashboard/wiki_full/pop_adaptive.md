# Population-Trained Adaptive Value

> Adaptive value agent trained against a diverse opponent pool instead of pure self-play, giving the opponent-statistics features genuine behavioral diversity.

| Property | Value |
|----------|-------|
| **ID** | `pop_adaptive` |
| **Parent** | AdaptiveValueAgent (`adaptive_value`) |
| **Round** | 2 |
| **Rank (R2)** | #8 / 12 |
| **Rank (R3)** | #14 / 17 |
| **Avg Score (R2)** | -0.64 |
| **Robustness (R2)** | -1.91 |
| **Avg Score (R3)** | -0.557 |
| **Robustness (R3)** | -1.771 |

## Motivation

AdaptiveValueAgent was the top performer in Round 1 (+0.99 avg) thanks to its 4 opponent-statistics features (fold_rate, raise_rate, fold_to_raise_rate, confidence). However, it was trained via self-play, meaning the opponent's behavior was always a mirror of itself. The opponent stats during training reflected only one opponent style, limiting the agent's ability to truly adapt.

The hypothesis was that training against a **diverse pool of opponents** -- each with fundamentally different play patterns -- would force the opponent_stats features to become genuinely useful. If the agent faces a tight player, then an aggressive one, then a passive one, it must learn to read the stats and adjust. Self-play never provides this diversity.

## Architecture

PopAdaptiveAgent inherits directly from AdaptiveValueAgent with **zero architectural changes**:

```python
class PopAdaptiveAgent(AdaptiveValueAgent):
    """Adaptive value agent trained against a diverse opponent population."""
    pass
```

The underlying network is:

```
ValueNetwork:
  Linear(19 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU -> Linear(64 -> 1)
```

- **Input**: 19 dimensions (15 base + 4 opponent stats)
- **Opponent stats**: fold_rate, raise_rate, fold_to_raise_rate, confidence
- **Hidden**: 64 units, 2 layers
- **Output**: 1 scalar value V(s)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 |
| Learning Rate | 1e-4 |
| Hands Per Session | 30 |
| **Opponent Rotation** | **Every 100 episodes** |
| **Self-Snapshot** | **Every 500 episodes** |
| Optimizer | Adam |
| Loss | MSE (TD(0)) |
| Exploration | Boltzmann (temperature=1.0) |

### The Opponent Pool

The initial pool consists of 3 pre-trained agents:

| Pool Member | Type | Style |
|-------------|------|-------|
| `heuristic` | Rule-based | Pot-odds-based, tight-aggressive |
| `value_based` | TD(0) neural | Learned from self-play, balanced |
| `adaptive_value` | TD(0) + stats | Opponent-aware, top performer |

Additional mechanisms:
- **Rotation**: Every 100 episodes, the trainer switches to the next opponent in the pool
- **Self-snapshots**: Every 500 episodes, a frozen copy of the current agent is added to the pool, providing a record of past self-play

### Training Protocol

Unlike AdaptiveTrainer's self-play where both seats are the training agent, PopAdaptiveTrainer assigns:
- **Seat 0**: Training agent (PopAdaptiveAgent, with gradient updates)
- **Seat 1**: Current opponent from the pool (frozen, no gradient updates)

Only Seat 0's post-action chains are collected for TD learning. The opponent plays but does not contribute training data.

## Tournament Results

### Round 2 (12-agent round-robin, 500 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -0.78 |
| value_based | **-1.78** |
| adaptive_value | **-1.57** |
| aux_value | +0.24 |
| actor_critic | +0.13 |
| history_value | -0.65 |
| decay_adaptive | +0.14 |
| nstep_value | -0.54 |
| entropy_ac | **-1.88** |
| adaptive_history | -0.94 |
| target_value | +0.53 |

### Round 3 (17-agent round-robin, 1000 rounds/matchup)

| Stat | Value |
|------|-------|
| Average | -0.557 |
| Worst | -1.961 |
| Best | +0.774 |
| Std | 0.810 |
| Robustness | -1.771 |

## Diagnosis & Findings

An extensive diagnostic study (`diagnose_pop_adaptive_results.json`) compared population training against self-play across multiple configurations.

### Experiment 4: Data Volume

Population training collects fewer training transitions:

| Config | Total Transitions |
|--------|-------------------|
| adaptive_value (self-play) | 105 per batch |
| pop_adaptive (pool) | 71 per batch |
| **Ratio** | **0.676** |

Because only Seat 0 generates training data (the opponent's actions are not recorded), population training receives only 68% as much data per episode.

### Experiment 9: Loss Comparison

| Config | Final Loss |
|--------|-----------|
| adaptive_value | 17.35 |
| pop_adaptive | 8.13 |

The lower loss is deceptive -- it reflects the agent learning to predict outcomes against weaker, more predictable opponents, not better overall play.

### Experiment 5: Adaptive Value Baseline (Fresh Training)

A freshly trained adaptive_value agent evaluated over 667 sessions against heuristic:
- Mean score: approximately -1.07 chips/round
- Highly variable: range [-3.87, +0.61]

### Experiment 6: Pop-Adaptive Fresh Training

A freshly trained pop_adaptive agent evaluated over 667 sessions:
- Mean score: approximately -0.93 chips/round
- High variance: range [-4.23, +1.67]

### Experiment 7: Strong-Only Pool

Training against only strong opponents (value_based + adaptive_value, no heuristic):
- Mean score: approximately -0.56 chips/round
- Better than the mixed pool, suggesting weak opponents dilute learning

### Experiment 8: Head-to-Head Comparison

| Agent | vs adaptive_value | vs heuristic | vs value_based |
|-------|-------------------|--------------|----------------|
| adaptive_value (pretrained) | 0.00 | +0.51 | +0.06 |
| pop_adaptive (pretrained) | -1.51 | -1.87 | -1.64 |
| pop_adaptive (fresh) | -1.92 | -1.39 | -1.93 |
| pop_strong_only | -1.48 | -0.90 | -1.78 |

The pretrained adaptive_value agent dominates all population-trained variants. Even training against only strong opponents cannot close the gap.

### Root Causes of Failure

1. **Weak opponent exploitation**: The pool contained mostly weak agents (heuristic, pre-trained value_based). Training against them taught the agent to exploit weaknesses rather than develop robust play. The heuristic agent in particular has predictable pot-odds-based decisions.

2. **Session stat disruption**: AdaptiveValueAgent's strength comes from accumulating opponent statistics over a session of 30 hands. Population training rotates opponents, which resets the stat accumulation and prevents the stats from becoming meaningful within a session.

3. **Data volume reduction**: By only training on Seat 0's chains, the agent receives 32% fewer training transitions than self-play, reducing sample efficiency.

4. **Non-stationarity mismatch**: The agent learns to play well against the current pool member, but the pool rotation means it never fully adapts to any single opponent style.

## Key Insight

Population diversity only helps if the population includes strong, varied opponents, and the training protocol preserves the mechanisms that make the agent effective. In this case, opponent rotation disrupted the session-based stat accumulation that was AdaptiveValueAgent's primary advantage, and training against weak opponents taught exploitation rather than robustness.

## Source Files

- Agent: `src/agents/pop_adaptive.py`
- Trainer: `src/training/pop_adaptive_trainer.py`
- Parent Agent: `src/agents/adaptive_value.py`
- Parent Trainer: `src/training/adaptive_trainer.py`
- Diagnosis: `experiments/diagnose_pop_adaptive_results.json`
