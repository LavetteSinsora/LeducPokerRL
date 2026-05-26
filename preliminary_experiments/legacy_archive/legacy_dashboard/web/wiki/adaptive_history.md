# Adaptive History Agent

> Combined agent merging opponent statistics from AdaptiveValueAgent with action history encoding from HistoryValueAgent into a single wider-network architecture.

| Property | Value |
|----------|-------|
| **ID** | `adaptive_history` |
| **Parent(s)** | AdaptiveValueAgent (`adaptive_value`) + HistoryValueAgent (`history_value`) |
| **Round** | 2 |
| **Rank (R2)** | #5 / 12 |
| **Rank (R3)** | #7 / 17 |
| **Avg Score (R2)** | +0.22 |
| **Robustness (R2)** | -1.02 |
| **Avg Score (R3)** | +0.134 |
| **Robustness (R3)** | -0.917 |

## Motivation

This agent is unique in the family tree: it has **two parents**, making it a merge rather than a single-aspect change.

- **AdaptiveValueAgent** (Round 0, rank 1): Added 4 opponent statistics features, enabling cross-hand behavioral modeling
- **HistoryValueAgent** (Round 1, rank 6): Added 16 action history features, encoding the within-hand action sequence

The hypothesis was that these two information sources are complementary: opponent stats tell you *who* you're playing against (long-term tendencies), while action history tells you *what happened this hand* (short-term context). Combining them should provide a richer representation than either alone.

## Architecture

AdaptiveHistoryAgent inherits from AdaptiveValueAgent and copies HistoryValueAgent's encoding logic:

```python
class AdaptiveHistoryAgent(AdaptiveValueAgent):
    """Combines opponent stats (4 features) with action history (16 features)."""

    FEATURES_PER_ROUND = 8    # 6 action counts + 2 summary features
    NUM_ROUNDS = 2
    HISTORY_SIZE = 16          # 8 features x 2 rounds

    def __init__(self, model_path=None, temperature=1.0):
        self.input_size = 15 + self.STATS_SIZE + self.HISTORY_SIZE  # 35
        self.model = ValueNetwork(self.input_size, hidden_size=128)  # Wider!
```

```
ValueNetwork (wider):
  Linear(35 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> 1)
```

- **Input**: 35 dimensions (15 base + 4 opponent stats + 16 action history)
- **Hidden**: 128 units (doubled from standard 64)
- **Output**: 1 scalar value V(s)

### Observation Encoding Breakdown

| Feature Block | Dims | Source |
|---------------|------|--------|
| Hand one-hot (J/Q/K) | 3 | Base |
| Board one-hot (J/Q/K/None) | 4 | Base |
| Normalized pot (my/opp) | 2 | Base |
| Game features (turn, pos, round, terminal, pair, raises) | 6 | Base |
| Opponent stats (fold%, raise%, fold-to-raise%, confidence) | 4 | AdaptiveValue |
| Round 0 history (player fold/call/raise, opp fold/call/raise, total, has_raise) | 8 | HistoryValue |
| Round 1 history (same 8 features) | 8 | HistoryValue |

### Action History Encoding Detail

For each of the 2 betting rounds, 8 features are computed:

```
[0] player_fold_count / total_actions
[1] player_call_count / total_actions
[2] player_raise_count / total_actions
[3] opponent_fold_count / total_actions
[4] opponent_call_count / total_actions
[5] opponent_raise_count / total_actions
[6] total_actions / MAX_ACTIONS_PER_ROUND (=6)
[7] 1.0 if any raise occurred, else 0.0
```

The encoding normalizes action counts by total actions in the round, providing a percentage-based representation that is scale-invariant.

### 1-Step Lookahead

The `get_action_evaluations` method carries **both** opponent_stats and action_history forward into simulated post-action states:

```python
def get_action_evaluations(self, obs):
    for action in obs.legal_actions:
        post_obs, done = LeducGame.simulate_action(obs, action)
        # Carry opponent_stats forward
        post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)
        # Carry extended action_history forward
        extended_history = current_history + ((current_p, action_name),)
        post_obs = replace(post_obs, action_history=extended_history)
```

This ensures the value network always receives the full 35-dimensional input during both training and inference.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 (667 sessions x 30 hands) |
| Learning Rate | 1e-4 |
| Hands Per Session | 30 |
| Batch Size | 1 session (30 hands) |
| Optimizer | Adam |
| Loss | MSE (TD(0)) |
| Network Width | 128 hidden units |
| Exploration | Boltzmann (temperature=1.0) |
| Training Mode | Self-play (session-based, stats accumulate across hands) |

The trainer extends AdaptiveTrainer with action_history carry-forward in `collect_episode()`, ensuring both stats and history are present in the encoded post-action states used for TD learning.

## Tournament Results

### Round 2 (12-agent round-robin, 500 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -0.52 |
| value_based | -0.65 |
| adaptive_value | -0.97 |
| aux_value | **+1.60** |
| actor_critic | +0.48 |
| history_value | -0.19 |
| decay_adaptive | **+1.08** |
| nstep_value | +0.39 |
| entropy_ac | -0.44 |
| pop_adaptive | +0.94 |
| target_value | +0.70 |

### Round 3 (17-agent round-robin, 1000 rounds/matchup)

| Stat | Value |
|------|-------|
| Average | +0.134 |
| Worst | -1.003 |
| Best | +0.900 |
| Std | 0.701 |
| Robustness | -0.917 |

## Diagnosis & Findings

### Performance Relative to Parents

| Agent | Round 2 Avg | Round 2 Rank | Robustness |
|-------|-------------|-------------|------------|
| adaptive_value (parent 1) | **+1.06** | **#1** | **+0.12** |
| history_value (parent 2) | -0.50 | #11 | -2.00 |
| **adaptive_history** | +0.22 | #5 | -1.02 |

The combined agent sits between its two parents in performance but closer to the weaker one. The feature combination did not synergize -- it performed worse than its stronger parent.

### Why the Combination Underperformed

1. **Capacity vs. data**: The 35-dim input required doubling the network to 128 hidden units, but the training budget remained 667 sessions (~20K hands). The wider network needed more data to converge than the narrower parent had needed.

2. **Feature redundancy**: In Leduc Hold'em, action sequences are very short (2-4 actions per hand). Much of the strategic information encoded in the action history is already captured by pot sizes and round number in the base features. The 16 history features are largely redundant with existing base features.

3. **Dilution of useful signal**: The opponent stats (4 features) are genuinely informative, but embedding them among 16 largely-redundant history features forces the network to learn which subset matters. With limited training data, this discrimination is incomplete.

4. **Below both parents' strengths**: Lost to adaptive_value (-0.97), the stronger parent, and even lost to history_value (-0.19) in head-to-head. The combination was weaker than either specialized version.

### The Merge Experiment

This agent demonstrates that feature concatenation is not free. Combining two observation spaces requires:
- Proportionally more network capacity
- Proportionally more training data
- Architecture that can learn feature importance (e.g., attention or gating)

Simply stacking features and widening the network is insufficient when the added features are redundant with existing ones.

## Key Insight

Feature concatenation alone is not enough -- the architecture needs mechanisms to learn which features matter. In short games like Leduc Hold'em, action history is largely redundant with pot-size features, and adding 16 redundant dimensions to a 19-dimensional effective input dilutes the signal that makes AdaptiveValueAgent strong.

## Source Files

- Agent: `src/agents/adaptive_history.py`
- Trainer: `src/training/adaptive_history_trainer.py`
- Parent Agent 1: `src/agents/adaptive_value.py`
- Parent Agent 2: `src/agents/history_value.py`
- Parent Trainer: `src/training/adaptive_trainer.py`
