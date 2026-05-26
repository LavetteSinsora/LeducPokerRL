# Decay Adaptive Agent

> Adaptive value agent with exponential moving average (EMA) opponent statistics -- a plausible idea that fails because EMA recency bias is meaningless in self-play.

| Property | Value |
|----------|-------|
| **ID** | `decay_adaptive` |
| **Parent** | AdaptiveValueAgent (`adaptive_value`) |
| **Round** | 1 |
| **Rank** | #12 / 17 |
| **Avg Score** | -0.881 |
| **Robustness** | -1.657 |

## Motivation

The Adaptive Value Agent (parent) tracks opponent behavior using uniform running averages: every observed action contributes equally to the opponent statistics (fold_rate, raise_rate, fold_to_raise_rate, confidence). The Decay Adaptive Agent tests a natural hypothesis: **what if recent observations matter more than old ones?**

In real poker, opponents adjust their strategy over time. An exponential moving average (EMA) gives more weight to recent behavior, allowing the agent to track a shifting opponent strategy. The hypothesis was that EMA-weighted stats would make the agent more responsive to opponents who change their play mid-session.

This is the textbook "obvious improvement" that turns out to be wrong.

## Architecture

### Network Architecture

Identical to AdaptiveValueAgent -- the agent class is a thin subclass that inherits everything:

```
Input (19) --> Linear(19, 64) --> ReLU --> Linear(64, 64) --> ReLU --> Linear(64, 1)
```

| Layer | Input Dim | Output Dim | Activation | Parameters |
|-------|-----------|------------|------------|------------|
| Linear 1 | 19 | 64 | ReLU | 1,280 |
| Linear 2 | 64 | 64 | ReLU | 4,160 |
| Linear 3 | 64 | 1 | None | 65 |
| **Total** | | | | **5,505** |

### Observation Encoding (19 dimensions)

Same as AdaptiveValueAgent:

| Features | Dims | Source |
|----------|------|--------|
| Base observation | 15 | ValueBasedAgent encoding |
| fold_rate | 1 | EMA-weighted opponent fold frequency |
| raise_rate | 1 | EMA-weighted opponent raise frequency |
| fold_to_raise_rate | 1 | EMA-weighted fold-when-raised frequency |
| confidence | 1 | Effective sample size (decayed) |

### The Difference: EMA vs. Uniform Averaging

The **only** difference from AdaptiveValueAgent is in the training infrastructure. The `DecayAdaptiveAgent` class itself is literally:

```python
class DecayAdaptiveAgent(AdaptiveValueAgent):
    pass  # Everything inherited -- the difference is in the trainer/session
```

The behavioral change comes from `DecayPokerSession`, which computes opponent statistics using EMA instead of uniform counts:

- **Uniform (parent):** `stat = total_count / total_observations` -- all observations weighted equally
- **EMA (this agent):** `stat = alpha * new_observation + (1 - alpha) * old_stat` -- recent observations weighted more heavily, with decay factor `alpha=0.1`

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes (sessions) | 100 |
| Hands per Session | 30 |
| Total Hands | ~3,000 |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| EMA Alpha | 0.1 |
| Optimizer | Adam |
| Training Method | TD(0) session-based self-play |

Training uses `DecayAdaptiveTrainer`, which extends `AdaptiveTrainer` by swapping `PokerSession` for `DecayPokerSession`. The only configuration addition is the `alpha` parameter controlling EMA decay rate.

The session-based training structure is identical to AdaptiveValueAgent:
1. Reset session and opponent stats
2. Play 30 hands against yourself, accumulating EMA-weighted stats
3. Collect post-action state chains with stats embedded in the encoding
4. Apply TD(0) updates on the batch

## Tournament Results

### Round 3 -- Full 17-Agent Tournament (1000 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -1.177 |
| value_based | -1.294 |
| adaptive_value | -1.559 |
| aux_value | -0.215 |
| actor_critic | -0.149 |
| history_value | -0.903 |
| nstep_value | -0.994 |
| entropy_ac | -1.867 |
| pop_adaptive | +0.054 |
| adaptive_history | -0.881 |
| target_value | -0.738 |
| td_variant | -1.313 |
| pruned_history | -0.632 |
| modulated_value | -1.092 |
| curriculum | -0.834 |
| extended_adaptive | -0.495 |

### Performance Profile
- **Wins against:** Only 1 of 16 opponents (pop_adaptive, barely at +0.054)
- **Loses to:** 15 of 16 opponents
- **Best matchup:** +0.054 vs pop_adaptive
- **Worst matchup:** -1.867 vs entropy_ac

### Cross-Round Trajectory

| Round | Rank | Avg Score | Robustness | Context |
|-------|------|-----------|------------|---------|
| R1 | #7 / 7 | -0.79 | N/A | Dead last |
| R2 | #9 / 12 | -0.97 | -1.92 | Among the worst |
| R3 | #12 / 17 | -0.881 | -1.657 | Consistently bottom tier |

## Key Findings

1. **EMA is meaningless in self-play.** The fundamental flaw: during training, both players are the same agent. There is no shifting opponent strategy to track. The "opponent" changes every episode because the network weights update, but EMA tracks within-session behavior of a fixed policy. In self-play, your opponent's within-session strategy is constant (it is you), so recency weighting adds noise without information.

2. **Worse than its parent across all rounds.** AdaptiveValueAgent averages +1.012 in Round 3; DecayAdaptiveAgent averages -0.881. That is a **1.89 chip/round** degradation from a single change (uniform to EMA stats). This is the largest parent-to-child performance drop in the project.

3. **The confidence signal is corrupted.** With uniform averaging, confidence = hands_observed / max_hands is a clean measure of sample size. With EMA, the effective sample size depends on alpha and decays over time, making the confidence feature unreliable. Since the parent agent learned to use low confidence as a fallback signal, corrupting it breaks the graceful degradation mechanism.

4. **Alpha sensitivity is a red flag.** The EMA approach introduces a hyperparameter (alpha=0.1) that must be tuned. In the parent, uniform averaging has no hyperparameters. The added complexity buys nothing in the self-play training regime.

5. **The idea is not wrong in general.** EMA opponent modeling would likely help in a setting with a fixed, non-stationary opponent (e.g., a human player who tilts). The problem is specifically that self-play training cannot teach the agent to exploit EMA's recency bias, because there is nothing recent to bias toward.

## Key Insight

Exponential moving average opponent tracking is a solution to a problem that does not exist in self-play training -- there is no shifting opponent strategy to adapt to when your opponent is yourself, so recency bias adds noise that uniformly degrades learning.

## Source Files

- Agent: `src/agents/decay_adaptive.py`
- Trainer: `src/training/decay_adaptive_trainer.py`
