# History Value Agent

> Value-based agent augmented with per-round action history encoding -- an information-rich approach that fails by overwhelming a network too small to process it.

| Property | Value |
|----------|-------|
| **ID** | `history_value` |
| **Parent** | ValueBasedAgent (`value_based`) |
| **Round** | 1 |
| **Rank** | #16 / 17 |
| **Avg Score** | -0.512 |
| **Robustness** | -1.822 |

## Motivation

The Value-Based Agent makes decisions using only the current game state: cards, pot sizes, and round information. It has no memory of what happened earlier in the hand. The History Value Agent tests the hypothesis that **encoding the sequence of actions taken so far** would give the agent richer strategic information -- for example, distinguishing "opponent raised pre-flop then checked the flop" (likely bluffing) from "opponent checked pre-flop then raised the flop" (likely hit the board).

The idea is sound in principle: action sequences carry information about opponent hand strength. In Texas Hold'em (4 rounds, many actions), this information is critical. The question was whether it matters enough in Leduc Hold'em to justify doubling the input dimensionality.

## Architecture

### Value Network

Same `ValueNetwork` architecture as the parent, but with a wider input layer:

```
Input (31) --> Linear(31, 64) --> ReLU --> Linear(64, 64) --> ReLU --> Linear(64, 1)
```

| Layer | Input Dim | Output Dim | Activation | Parameters |
|-------|-----------|------------|------------|------------|
| Linear 1 | 31 | 64 | ReLU | 2,048 |
| Linear 2 | 64 | 64 | ReLU | 4,160 |
| Linear 3 | 64 | 1 | None | 65 |
| **Total** | | | | **6,273** |

### Observation Encoding (31 dimensions)

The first 15 dimensions are inherited from ValueBasedAgent. The additional 16 dimensions encode action history:

| Features | Dims | Source |
|----------|------|--------|
| Base observation | 15 | ValueBasedAgent encoding |
| Round 0 history | 8 | Action counts for pre-flop round |
| Round 1 history | 8 | Action counts for flop round |

### History Encoding Detail (8 features per round)

For each of the 2 betting rounds, the agent encodes:

| Feature | Index | Description |
|---------|-------|-------------|
| Player fold count | +0 | Normalized: count / total_actions_in_round |
| Player call count | +1 | Normalized |
| Player raise count | +2 | Normalized |
| Opponent fold count | +3 | Normalized |
| Opponent call count | +4 | Normalized |
| Opponent raise count | +5 | Normalized |
| Total actions | +6 | Normalized: total / MAX_ACTIONS_PER_ROUND (6) |
| Raise occurred | +7 | Binary: 1.0 if any raise in this round |

The encoding is **perspective-relative**: "player" always refers to the viewing agent, "opponent" to the other player, regardless of seat position.

### Round Boundary Detection

The agent must split the flat action history into per-round segments. This is done by replaying the sequence and detecting round transitions:
- Round 0 ends when betting completes (both players have acted and the last action was called/checked)
- A fold ends the game immediately
- Remaining actions belong to Round 1

### Action Simulation with History

Unlike the parent agent, the history-aware 1-step lookahead must manually extend the action history for each simulated successor state. `LeducGame.simulate_action()` does not carry `action_history` forward, so the agent appends each hypothetical action to the current history before encoding the simulated state:

```python
extended_history = current_history + ((current_p, action_name),)
post_obs = replace(post_obs, action_history=extended_history)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 3,000 |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | MSE |
| Training Method | TD(0) self-play |
| Special | HistoryValueTrainer propagates action_history into simulated states |

Training uses `HistoryValueTrainer`, which extends `SelfPlayTrainer` with correct history propagation. The TD(0) algorithm is identical to the parent -- the only difference is that post-action states now include extended action history in their encoding.

## Tournament Results

### Round 3 -- Full 17-Agent Tournament (1000 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -0.983 |
| value_based | -1.925 |
| adaptive_value | -1.711 |
| aux_value | +0.352 |
| actor_critic | +0.512 |
| decay_adaptive | +0.903 |
| nstep_value | -0.577 |
| entropy_ac | -1.231 |
| pop_adaptive | -0.202 |
| adaptive_history | +0.082 |
| target_value | -1.224 |
| td_variant | -0.923 |
| pruned_history | -0.004 |
| modulated_value | -1.347 |
| curriculum | +0.660 |
| extended_adaptive | -0.578 |

### Performance Profile
- **Wins against:** 5 of 16 opponents
- **Loses to:** 11 of 16 opponents
- **Best matchup:** +0.903 vs decay_adaptive
- **Worst matchup:** -1.925 vs value_based (its own parent)

### Cross-Round Trajectory

| Round | Rank | Avg Score | Robustness | Context |
|-------|------|-----------|------------|---------|
| R1 | #6 / 7 | -0.74 | N/A | Second worst |
| R2 | #11 / 12 | -0.50 | -2.00 | Near bottom |
| R3 | #16 / 17 | -0.512 | -1.822 | Second worst overall |

## Key Findings

1. **Doubled input, same network capacity = underfitting.** The input dimensionality went from 15 to 31, but the hidden layers remained at 64 units each. The first layer must now compress 31 features into 64 dimensions instead of 15 into 64. The network lacks the capacity to learn useful representations from the additional features while also maintaining its base state evaluation ability.

2. **Loses to its own parent by the largest margin in the tournament.** At -1.925 vs value_based, this is the single worst head-to-head matchup for history_value. The parent, using 16 fewer input features and the exact same network width, is dramatically better. The extra features actively hurt performance.

3. **Action history is redundant in Leduc.** Leduc Hold'em games have 2-4 actions per round. The information in "opponent raised" is already captured by the pot sizes (which increase after a raise) and the raises_this_round feature. The 16 additional history features are encoding information that is already present in the base 15 features, but in a less compact form.

4. **The design was meant for Texas Hold'em.** The code explicitly notes `NUM_ROUNDS = 2  # Leduc has 2 rounds; set to 4 for Texas Hold'em`. In Texas Hold'em with 4 rounds and many more actions, the history encoding would carry genuinely new information. But the agent was evaluated on Leduc, where the encoding is oversized for the problem.

5. **Network width must scale with input dimensionality.** Later experiments confirmed this lesson: AdaptiveHistoryAgent (35-dim, 128-wide) and PrunedHistoryAgent (31-dim, 64-wide) both underperformed, but the wider network variant did relatively better. The minimum viable network width grows roughly proportionally with input dimensionality.

6. **Self-play training exacerbates the problem.** With twice as many input features, the network needs more training data to converge. But 3,000 episodes of self-play against an undertrained version of itself means the training signal is both insufficient and non-stationary. The agent never gets a stable enough signal to learn which of its 31 features actually matter.

## Key Insight

Doubling the observation space from 15 to 31 dimensions without increasing network capacity causes underfitting that is worse than having no history at all -- in Leduc Hold'em, the action sequence information is already implicitly captured by pot sizes and round numbers.

## Source Files

- Agent: `src/agents/history_value.py`
- Trainer: `src/training/history_value_trainer.py`
