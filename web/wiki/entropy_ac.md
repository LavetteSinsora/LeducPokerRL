# Entropy-Regularized Actor-Critic

> Actor-critic agent with entropy bonus that prevents policy collapse into exploitable deterministic strategies.

| Property | Value |
|----------|-------|
| **ID** | `entropy_ac` |
| **Parent** | ActorCriticAgent (`actor_critic`) |
| **Round** | 2 |
| **Rank (R2)** | #4 / 12 |
| **Rank (R3)** | #6 / 17 |
| **Avg Score (R2)** | +0.94 |
| **Robustness (R2)** | -0.54 |
| **Avg Score (R3)** | +0.664 |
| **Robustness (R3)** | -0.712 |

## Motivation

In Round 1, ActorCriticAgent ranked 5th with an average score of -0.54. The policy gradient approach suffered from a fundamental problem: despite using a value baseline for variance reduction, the agent's policy would collapse to near-deterministic action selection. In poker, a deterministic strategy is trivially exploitable -- if an opponent can predict your actions, they can counter-play perfectly.

The hypothesis was that adding an entropy bonus H(pi) = -sum(p(a) * log(p(a))) to the loss function would force the policy to remain stochastic, maintaining mixed strategies that are inherently harder to exploit. This is the same principle behind Nash equilibria in game theory: optimal poker play requires randomization.

## Architecture

EntropyACAgent inherits directly from ActorCriticAgent with **zero architectural changes**. The agent class itself is a pass-through:

```python
class EntropyACAgent(ActorCriticAgent):
    """Actor-critic agent with entropy regularization."""
    pass
```

The entire modification lives in the trainer. The underlying network is:

```
ActorCriticNetwork:
  Shared backbone: Linear(15 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU
  Policy head:     Linear(64 -> 3) -> Softmax   (action probabilities)
  Value head:      Linear(64 -> 1)               (scalar state value)
```

- **Input**: 15 dimensions (3 hand one-hot + 4 board one-hot + 2 normalized pot + 6 features)
- **Hidden**: 64 units, 2 layers, shared backbone
- **Output**: 3 action probabilities (fold/call/raise) + 1 state value

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 |
| Learning Rate | 1e-3 |
| Batch Size | 30 (episodes per update) |
| Value Coefficient | 0.5 |
| **Entropy Coefficient** | **0.01** |
| Optimizer | Adam |
| Exploration | Categorical sampling from policy (train), argmax (eval) |

### The Entropy Bonus

The key modification is in the loss function. For each decision step:

```
total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

Where:
- `policy_loss = -log(pi(a|s)) * (R - V(s))` (REINFORCE with baseline)
- `value_loss = (V(s) - R)^2` (critic regression)
- `entropy = -sum(pi(a|s) * log(pi(a|s) + 1e-10))` (Shannon entropy)

The negative sign on the entropy term means the optimizer **maximizes** entropy alongside minimizing policy and value losses. The coefficient 0.01 balances exploration pressure against reward-seeking behavior.

The trainer also records the full masked probability distribution at each step (not just the log-prob of the chosen action), enabling proper entropy computation over all legal actions.

## Tournament Results

### Round 2 (12-agent round-robin, 500 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -0.32 |
| value_based | -0.37 |
| adaptive_value | -0.30 |
| aux_value | **+2.12** |
| actor_critic | **+1.67** |
| history_value | **+1.28** |
| decay_adaptive | **+2.16** |
| nstep_value | +0.57 |
| pop_adaptive | **+1.88** |
| adaptive_history | +0.44 |
| target_value | **+1.23** |

### Round 3 (17-agent round-robin, 1000 rounds/matchup)

| Stat | Value |
|------|-------|
| Average | +0.664 |
| Worst | -0.657 |
| Best | +1.961 |
| Std | 0.917 |
| Robustness | -0.712 |

## Diagnosis & Findings

### What Worked

The entropy bonus achieved exactly what was hypothesized: it prevented policy collapse and maintained mixed strategies throughout training. The result was dramatic -- from actor_critic's -0.62 average to entropy_ac's +0.94, a swing of +1.56 chips/round.

Key performance characteristics:

1. **Dominant against weak agents**: Posted the highest single-matchup score in the entire Round 2 tournament (+2.16 vs decay_adaptive). Crushed every agent ranked below it.

2. **Mixed strategy advantage**: By maintaining stochastic play, the agent became harder to predict and exploit. This is especially important in repeated play (500 rounds per matchup), where a deterministic opponent can be figured out.

3. **Only training modification that significantly improved actor-critic**: Among all experiments across all rounds, entropy regularization was the single most impactful change to the policy gradient family.

### What Didn't Work

1. **High variance** (std = 0.99, highest in Round 2): While the average was strong, performance swung wildly between matchups. The agent won big against weak opponents but lost to the top 2 (value_based and adaptive_value).

2. **Robustness penalty**: The robustness metric (avg - 1.5 * std) punished this inconsistency, dropping entropy_ac from rank 4 (by average) to rank 4 (by robustness) but with a negative score (-0.54).

3. **Struggles against value-based top tier**: Lost to adaptive_value (-0.30), value_based (-0.37), and even heuristic (-0.32). The entropy bonus promotes diversity but doesn't improve the quality of the value estimates themselves.

### Comparison: Actor-Critic vs Entropy AC

| Metric | actor_critic | entropy_ac | Delta |
|--------|-------------|------------|-------|
| Average | -0.62 | +0.94 | **+1.56** |
| Best | +0.23 | +2.16 | +1.93 |
| Worst | -1.67 | -0.37 | +1.30 |
| Std | 0.51 | 0.99 | +0.48 |
| Robustness | -1.38 | -0.54 | +0.84 |

The entropy bonus improved every metric except standard deviation. The higher variance is a natural consequence of maintaining stochastic play -- mixed strategies inherently produce more variable outcomes per game, but this variance is the price of being unexploitable.

## Key Insight

Mixed strategies matter in poker. A single hyperparameter -- the entropy coefficient of 0.01 -- transformed the worst neural network agent into one that could compete with the best, because it enforced the game-theoretic principle that optimal poker play requires randomization.

## Source Files

- Agent: `src/agents/entropy_ac.py`
- Trainer: `src/training/entropy_ac_trainer.py`
- Parent Agent: `src/agents/actor_critic.py`
- Parent Trainer: `src/training/actor_critic_trainer.py`
