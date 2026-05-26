# Opponent Response Planning Agent

> Two-ply lookahead with an explicit opponent model -- a principled planning approach that backfires because the self-play-trained opponent model predicts a generic action distribution regardless of the actual opponent.

| Property | Value |
|----------|-------|
| **ID** | `opponent_model` |
| **Parent** | value_based |
| **Round** | 4 |
| **Rank** | ~22 / 22 |
| **Avg Score** | -1.398 |
| **Robustness** | -2.174 |

---

## Motivation

The value_based agent uses 1-ply lookahead: it simulates each of its own actions, evaluates the resulting state, and picks the best. But it does not consider what the opponent will do next. In real poker, skilled players think multiple moves ahead: "if I raise, they'll probably fold their weak hands but call with strong ones, so my expected value depends on their response distribution."

The Opponent Response Planning Agent adds a second ply by training an explicit opponent model P(opponent_action | state) and using it to compute expected values over the opponent's likely responses. This transforms the decision from `max_a V(s')` into `max_a E_{opp~P}[V(s'')]` -- choosing the action that maximizes expected value after accounting for the opponent's most likely response.

---

## Architecture

### Network 1: Value Network
```
ValueNetwork(15 -> 64 -> 64 -> 1)
  - Input: 15-dim game state encoding (same as value_based)
  - Architecture: 2-layer MLP with ReLU, 64 hidden units
  - Output: Scalar state value
  - Trained via TD(0) during self-play
  - Parameters: 5,249
```

### Network 2: Opponent Model
```
OpponentModel(15 -> 32 -> 3)
  - Input: 15-dim game state encoding
  - Architecture: Single hidden layer, 32 units, ReLU
  - Output: P(fold), P(call), P(raise) -- opponent action probabilities
  - Trained via cross-entropy on observed opponent actions during self-play
  - Parameters: 611
```

### 2-Ply Lookahead Decision Procedure

At decision time, the agent:

1. For each legal action `a`:
   - Simulate taking action `a` to reach state `s'`
   - For each possible opponent response `b` in {fold, call, raise}:
     - Simulate opponent taking action `b` to reach state `s''`
     - Evaluate `V(s'')`
   - Compute `Q(s, a) = sum_b P(b|s') * V(s''_b)` using the opponent model
2. Select `argmax_a Q(s, a)`

This gives the agent a **planning horizon of 2 actions** (own action + opponent response) instead of just 1.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Self-play |
| Episodes | 30,000 |
| Learning Rate | 1e-4 (Adam) |
| Value Loss | TD(0) MSE |
| Opponent Model Loss | Cross-entropy on opponent actions |
| Total Parameters | 5,860 (value: 5,249 + opponent: 611) |

Both networks are trained simultaneously during self-play. The value network receives standard TD(0) updates, while the opponent model receives cross-entropy supervision from the opponent's actual actions at each decision point.

---

## Tournament Results

| Opponent | Score |
|----------|-------|
| heuristic | -1.406 |
| value_based | -1.476 |
| adaptive_value | -1.794 |
| modulated_value | -1.304 |
| entropy_ac | -1.642 |
| cfr | -0.768 |

**Best matchup**: -0.768 vs cfr
**Worst matchup**: -1.794 vs adaptive_value
**Wins against**: 0 of 6 core opponents
**Dead last**: Ranks last (or near-last) in the full tournament

---

## Diagnosis & Findings

### 1. Opponent Model Learns a Single Generic Distribution

Despite being designed to model specific opponents, the opponent model predicts nearly identical action distributions regardless of who it faces:

| Actual Opponent | Model's Predicted Distribution |
|----------------|-------------------------------|
| value_based | ~17% fold, ~47% call, ~37% raise |
| adaptive_value | ~17% fold, ~47% call, ~37% raise |
| modulated_value | ~17% fold, ~47% call, ~37% raise |
| entropy_ac | ~17% fold, ~47% call, ~37% raise |
| heuristic | ~17% fold, ~47% call, ~37% raise |
| cfr | ~17% fold, ~47% call, ~37% raise |

The model has learned the **average self-play action distribution** and applies it uniformly to every opponent. It cannot distinguish between a passive opponent who folds frequently and an aggressive one who always raises.

### 2. Opponent Model Accuracy is Misleadingly High

| Opponent | Accuracy |
|----------|----------|
| value_based | 76% |
| adaptive_value | 76% |
| modulated_value | 76% |
| entropy_ac | 74% |
| heuristic | 66% |
| cfr | 58% |

These accuracy numbers look reasonable, but they are inflated by the base rate. Since the model predicts "call" ~47% of the time and many opponents do call frequently, the model gets a decent accuracy score simply by predicting the most common action. The accuracy drops against opponents with unusual distributions (heuristic: 66%, cfr: 58%).

### 3. 2-Ply Amplifies Bias Toward Aggression

The 2-ply lookahead changes 18.3% of decisions compared to 1-ply. The direction of these changes is overwhelmingly toward aggression:

| Decision Shift | Count |
|---------------|-------|
| CALL to RAISE | 241 |
| FOLD to RAISE | 134 |
| RAISE to CALL | 47 |
| FOLD to CALL | 23 |
| Other | minimal |

The mechanism: the opponent model predicts ~17% fold probability for every state. This means the agent believes raising gives it a ~17% chance of winning the pot immediately (the opponent folds). This overestimated fold equity makes raising look better than it actually is against opponents who rarely fold, leading to systematically excessive aggression.

### 4. Why It's the Worst Agent

The combination is catastrophic:
1. The opponent model overestimates fold equity (~17% predicted vs often near 0% actual)
2. The 2-ply lookahead converts this bias into aggressive RAISE decisions
3. Against strong opponents who never fold to weak raises, the agent bleeds chips
4. The more it raises, the bigger the pots when it loses

This makes the agent worse than its parent (value_based) and worse than agents with no opponent model at all. The planning horizon amplifies a biased model's errors rather than correcting them.

---

## Key Insight

Two-ply planning amplifies rather than corrects opponent model bias -- a self-play-trained opponent model produces a single-mode generic prediction that overestimates fold equity, and the lookahead dutifully converts this into excessive aggression that loses against every real opponent.

---

## Source Files

- Agent: `src/agents/opponent_model.py`
- Trainer: `src/training/opponent_model_trainer.py`
