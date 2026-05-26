# Auxiliary Value Agent

> Value network with an auxiliary head predicting Bellman-consistent pre-action values, intended to provide extra gradient signal but ultimately harmed by the max operator and gradient competition.

| Property | Value |
|----------|-------|
| **ID** | `aux_value` |
| **Parent** | ValueBasedAgent (`value_based`) |
| **Round** | 0 (Gen 1) |
| **Rank (R2)** | #12 / 12 |
| **Rank (R3)** | #8 / 17 |
| **Avg Score (R2)** | -0.92 |
| **Robustness (R2)** | -2.20 |
| **Avg Score (R3)** | -0.211 |
| **Robustness (R3)** | -0.986 |

## Motivation

ValueBasedAgent's TD(0) training only provides gradient signal at post-action states -- the network learns V(s') where s' is the state after an action is taken. At decision nodes (pre-action states), the agent uses 1-step lookahead without direct training signal.

The hypothesis was that adding an **auxiliary loss** enforcing Bellman consistency at pre-action states would provide additional gradient signal, helping the network learn better representations. The auxiliary target is:

```
V(pre-action state) = max_a V(post-action state_a)
```

This is the Bellman optimality equation: the value of a state before acting should equal the value of the best available action.

## Architecture

AuxValueAgent inherits directly from ValueBasedAgent with **zero architectural changes**:

```python
class AuxValueAgent(ValueBasedAgent):
    """Value agent trained with a pre-action Bellman consistency auxiliary loss."""
    pass
```

The modification is entirely in the trainer. The underlying network is:

```
ValueNetwork:
  Linear(15 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU -> Linear(64 -> 1)
```

- **Input**: 15 dimensions (same as ValueBasedAgent)
- **Hidden**: 64 units, 2 layers
- **Output**: 1 scalar value V(s)

Note: Despite the name suggesting an "auxiliary head," the agent uses a **single shared network** for both the main TD loss and the auxiliary loss. There is no separate output head -- the same V(s) output is trained with two different objectives.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 |
| Learning Rate | 1e-4 |
| Batch Size | 30 |
| **Aux Weight** | **0.5** |
| Eval Interval | Every 50 episodes |
| Eval Games | 100 vs heuristic |
| Optimizer | Adam |
| Loss | MSE (main) + 0.5 * MSE (aux) |
| Exploration | Boltzmann (temperature=1.0) |

### Dual Loss Structure

For each episode, two losses are computed per player:

**Main TD(0) Loss** (on post-action states):
```
for t in range(L):
    prediction = V(chain[t])
    if t == L-1:  target = terminal_reward
    else:         target = V(chain[t+1]).detach()
    main_loss += MSE(prediction, target)
```

**Auxiliary Bellman Loss** (on pre-action states):
```
for pre_encoded, post_encodeds in pre_action_data:
    pre_val = V(pre_encoded)
    best_post_val = max(V(post_a) for all legal actions a).detach()
    aux_loss += 0.5 * MSE(pre_val, best_post_val)
```

The auxiliary target is detached (no gradients flow through the max), so it acts as a supervised regression target computed from the network's own current estimates.

### Data Collection

The trainer's `collect_episode()` records both:
1. **Post-action chains** (same as standard TD): chosen post-action states per player
2. **Pre-action data**: (pre-action encoding, list of all post-action encodings) at each decision node

This doubles the data per episode by recording states both before and after action selection.

## Tournament Results

### Round 1 (7-agent, 500 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -0.33 |
| value_based | -1.01 |
| adaptive_value | -1.27 |
| actor_critic | +0.27 |
| history_value | +0.78 |
| decay_adaptive | +0.83 |

### Round 2 (12-agent round-robin, 500 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -0.90 |
| value_based | **-1.70** |
| adaptive_value | -1.25 |
| actor_critic | +0.27 |
| history_value | +0.18 |
| decay_adaptive | +0.07 |
| nstep_value | **-1.67** |
| entropy_ac | **-2.12** |
| pop_adaptive | -0.24 |
| adaptive_history | **-1.60** |
| target_value | -1.16 |

### Round 3 (17-agent round-robin, 1000 rounds/matchup)

| Stat | Value |
|------|-------|
| Average | -0.211 |
| Worst | -1.229 |
| Best | +0.645 |
| Std | 0.517 |
| Robustness | -0.986 |

## Diagnosis & Findings

Three carefully designed experiments (`aux_value_diagnosis_results.json`) confirmed the overestimation hypothesis and revealed deeper structural problems.

### Experiment A: Value Drift Over Training

Tracking mean predicted values on a fixed probe set during training:

| Episode | value_based mean | aux_value mean | Difference |
|---------|------------------|----------------|------------|
| 0 | +0.085 | +0.085 | 0.000 |
| 800 | +0.062 | +0.069 | **+0.007** |
| 1600 | +0.037 | +0.052 | **+0.016** |
| 2400 | +0.014 | +0.037 | **+0.023** |
| 3200 | -0.011 | +0.022 | **+0.033** |
| 4000 | -0.040 | +0.008 | **+0.048** |
| 4800 | -0.074 | -0.008 | **+0.066** |

The aux_value agent's predictions are systematically higher than value_based's, with the gap growing monotonically from 0 to **+0.066** over training. This is the max-operator overestimation bias in action.

### Experiment B: Operator Comparison

Four training variants compared at 4,000 episodes:

| Variant | Final Mean Value | Score vs Heuristic |
|---------|-----------------|-------------------|
| Plain TD(0) | -0.107 | -1.745 |
| Aux (max target) | -0.091 | -1.980 |
| Aux (mean target) | -0.108 | -2.005 |
| Aux (on-policy chain) | -0.097 | **-2.385** |

Replacing `max` with `mean` eliminated the overestimation bias (mean_V matched plain TD(0)), but performance was **still worse** than plain TD(0). Even the on-policy variant performed worst of all. This means the problem is not just the max operator -- the auxiliary loss itself is harmful.

### Experiment C: Prediction Accuracy

Evaluating trained models on fresh games:

| Metric | value_based | aux_value |
|--------|-------------|-----------|
| Mean signed error | +0.300 | -0.521 |
| **Mean absolute error** | **1.684** | **7.186** |
| Mean prediction | +0.014 | -0.213 |
| Mean actual outcome | -0.286 | +0.309 |
| Sample count | 2,140 | 2,573 |

The aux_value agent has **4.3x higher absolute prediction error** than value_based (7.19 vs 1.68). Its predictions are not just biased -- they are dramatically less accurate.

### Three Failure Mechanisms

1. **Max-operator overestimation**: The `max_a V(post_a)` target is systematically biased upward because the max of noisy estimates exceeds the max of true values. This is the same problem that motivated Double DQN, but here it corrupts the auxiliary loss rather than the main TD target.

2. **Gradient budget competition**: The shared network must minimize two losses simultaneously. The auxiliary loss steals gradient budget from the main TD loss. With `aux_weight=0.5`, roughly one-third of all gradient updates serve the auxiliary objective, reducing the effective learning rate for the primary TD task.

3. **Pre-action vs. post-action value conflict**: The main loss trains V(s) on post-action states, while the auxiliary loss trains V(s) on pre-action states. These are fundamentally different value functions (pre-action values should equal the max of post-action values, not the same thing). Forcing a single network to learn both creates an irreconcilable tension.

### The Overestimation Mechanism in Detail

At each decision node with legal actions {a1, a2, a3}:
```
V_true(pre) = max(V_true(post_a1), V_true(post_a2), V_true(post_a3))
V_estimated(pre) -> max(V_est(post_a1) + noise1, V_est(post_a2) + noise2, ...)
```

Since `E[max(X + noise)] >= max(E[X])`, the auxiliary target is biased upward by the noise in the value estimates. As training progresses and the auxiliary loss pushes predictions higher, this creates a positive feedback loop.

## Key Insight

The auxiliary loss is fundamentally counterproductive: even when the max-operator bias is eliminated, the extra loss steals gradient budget from the main TD objective and forces a single network to learn two conflicting value functions (pre-action and post-action states). In this domain, less is more -- plain TD(0) outperforms every auxiliary training modification tested.

## Source Files

- Agent: `src/agents/aux_value.py`
- Trainer: `src/training/aux_value_trainer.py`
- Parent Agent: `src/agents/value_based.py`
- Parent Trainer: `src/training/value_based_trainer.py`
- Diagnosis: `experiments/aux_value_diagnosis_results.json`
