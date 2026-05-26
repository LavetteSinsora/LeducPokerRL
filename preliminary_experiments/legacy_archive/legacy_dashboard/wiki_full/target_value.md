# Target Value Agent

> Value network using a frozen target network for stable TD learning, inspired by DQN -- but destabilized by self-play non-stationarity.

| Property | Value |
|----------|-------|
| **ID** | `target_value` |
| **Parent** | ValueBasedAgent (`value_based`) |
| **Round** | 2 |
| **Rank (R2)** | #10 / 12 |
| **Rank (R3)** | #11 / 17 |
| **Avg Score (R2)** | -0.31 |
| **Robustness (R2)** | -1.93 |
| **Avg Score (R3)** | -0.296 |
| **Robustness (R3)** | -1.537 |

## Motivation

In DQN and other deep RL methods, a common source of instability is the "moving target" problem: the network being trained is also used to compute the bootstrap targets, creating a feedback loop where both the predictions and the targets shift simultaneously. The standard fix, introduced in the DQN paper (Mnih et al., 2015), is to maintain a **frozen copy** of the network (the "target network") that is only periodically synchronized with the main network.

The hypothesis was that this stabilization technique would help the value-based agent learn more consistent value estimates, especially during the noisy early stages of self-play training.

## Architecture

TargetValueAgent extends ValueBasedAgent with a second copy of the value network:

```python
class TargetValueAgent(ValueBasedAgent):
    def __init__(self, model_path=None, temperature=1.0):
        super().__init__(model_path=model_path, temperature=temperature)
        self.target_model = copy.deepcopy(self.model)  # Frozen copy
        self._freeze_target()
```

```
Main Network (trainable):
  Linear(15 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU -> Linear(64 -> 1)

Target Network (frozen, periodic sync):
  Linear(15 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU -> Linear(64 -> 1)
```

- **Input**: 15 dimensions (same as ValueBasedAgent)
- **Hidden**: 64 units, 2 layers (both networks)
- **Output**: 1 scalar value V(s)
- **Key methods**: `sync_target()` copies main -> target; `get_target_value()` queries the frozen network

At inference time, behavior is identical to ValueBasedAgent (uses main model only). The target model is only used during training for computing bootstrap values.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 |
| Learning Rate | 1e-4 |
| Batch Size | 30 |
| **Target Sync Interval** | **Every 100 gradient steps** |
| Optimizer | Adam |
| Loss | MSE |
| Exploration | Boltzmann (temperature=1.0) |

### The Target Network Mechanism

Standard TD(0):
```
target = V_main(s_{t+1})    # Moving target -- changes every gradient step
```

Target network TD(0):
```
target = V_target(s_{t+1})  # Stable target -- frozen between syncs
```

Every 100 gradient steps, the target network is synchronized:
```python
self.target_model.load_state_dict(self.model.state_dict())
```

## Tournament Results

### Round 2 (12-agent round-robin, 500 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -0.83 |
| value_based | -1.40 |
| adaptive_value | **-1.94** |
| aux_value | +1.16 |
| actor_critic | +0.79 |
| history_value | +0.95 |
| decay_adaptive | +0.91 |
| nstep_value | -0.55 |
| entropy_ac | -1.23 |
| pop_adaptive | -0.53 |
| adaptive_history | -0.70 |

### Round 3 (17-agent round-robin, 1000 rounds/matchup)

| Stat | Value |
|------|-------|
| Average | -0.296 |
| Worst | -1.624 |
| Best | +1.224 |
| Std | 0.828 |
| Robustness | -1.537 |

## Diagnosis & Findings

Two diagnostic experiments were conducted: a detailed mechanism analysis (`diagnose_target_value_results.json`) and a multi-seed confirmation (`diagnose_target_value_multiseed_results.json`).

### Experiment 1: Target Freezing Verification

The target network is confirmed frozen between syncs:
- `target_change`: 0.0 (no parameter drift)
- `main_target_distance`: 0.0065 (networks diverge between syncs)
- `target_frozen`: true

### Experiment 2: Weight Divergence

Between sync points, the main and target networks diverge significantly:
- Mean parameter distance: **0.199**
- Max parameter distance: **0.391**

This means the target is providing bootstrap values from a substantially different function than the one being trained.

### Experiment 3: Sync Frequency Sweep

| Configuration | Score vs Heuristic |
|---------------|-------------------|
| No target (parent) | -0.987 |
| Sync every 1 step | -0.507 |
| Sync every 5 | **-2.393** |
| Sync every 10 | -0.757 |
| Sync every 25 | -1.583 |
| Sync every 50 | -0.563 |
| Sync every 100 | -1.767 |
| Sync every 500 | -0.427 |

No sync interval consistently outperforms no-target training. Performance is erratic and non-monotonic, suggesting the problem is fundamental, not a matter of tuning.

### Experiment 4: Value Divergence Between Networks

Just before a sync at gradient step 49:
- Mean absolute value difference: **0.030**
- Max absolute value difference: **0.049**
- Correlation between networks: 0.783

The target network's values are systematically different from the main network's, and the correlation is only 0.78 -- meaning 22% of the value variance is explained by divergence, not by game state.

### Experiment 5: Chain Length Distribution

| Chain Length | Count | Fraction |
|-------------|-------|----------|
| 1 step | 968 | 26.0% |
| 2 steps | 1,557 | 41.8% |
| 3 steps | 881 | 23.6% |
| 4 steps | 315 | 8.5% |

- Terminal fraction: 46.6% (use reward directly)
- Bootstrap fraction: 53.4% (use target network)

Over half of all training targets come from the target network, so the staleness problem affects a majority of the training signal.

### Experiment 6: Self-Play Non-Stationarity

The main network's value predictions change by approximately **0.0004 per gradient step** on a fixed probe set. Over 100 steps between syncs, this accumulates to ~0.04 total drift -- comparable to the 0.03 mean divergence measured in Experiment 4.

### Multi-Seed Confirmation

5 random seeds, evaluating target_sync_100 vs no-target:

| Configuration | Mean Score | Std | Range |
|---------------|-----------|-----|-------|
| No target | -1.052 | 0.697 | [-1.752, +0.1] |
| Sync every 1 | -1.052 | 0.697 | [-1.752, +0.1] |
| Sync every 100 | -0.915 | 0.709 | [-1.764, +0.1] |
| Sync every 500 | -0.842 | 0.580 | [-1.610, -0.184] |

Across seeds, target networks provide no statistically significant improvement. The best sync interval (500) merely matches the baseline within noise.

### Root Cause: Self-Play Non-Stationarity

The fundamental issue is that DQN's target network was designed for **stationary** environments. In self-play:

1. Both players use the same network
2. As the network updates, the opponent's policy changes
3. A target frozen 100 gradient steps ago reflects a different opponent
4. The frozen target is therefore **systematically wrong**, not just noisy

In a stationary MDP, the target network provides a stable regression target. In self-play, it provides a regression target from a game that no longer exists.

## Key Insight

Stabilization techniques designed for stationary MDPs can actively harm learning in adversarial multi-agent settings. The frozen target network creates a stale reference that is systematically wrong because the agent (and therefore its opponent) has changed since the last sync.

## Source Files

- Agent: `src/agents/target_value.py`
- Trainer: `src/training/target_value_trainer.py`
- Parent Agent: `src/agents/value_based.py`
- Diagnosis: `experiments/diagnose_target_value_results.json`
- Multi-seed: `experiments/diagnose_target_value_multiseed_results.json`
