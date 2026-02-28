# Actor-Critic Agent

> Policy gradient agent with a learned value baseline for variance reduction -- a principled approach that underperforms due to policy collapse in self-play.

| Property | Value |
|----------|-------|
| **ID** | `actor_critic` |
| **Parent** | PolicyGradientAgent (`policy_gradient`) |
| **Round** | 1 |
| **Rank** | #10 / 17 |
| **Avg Score** | -0.568 |
| **Robustness** | -1.197 |

## Motivation

The Actor-Critic Agent tests whether policy gradient methods can compete with value-based approaches in poker. Pure REINFORCE (the parent PolicyGradientAgent) has notoriously high variance because it reinforces actions with raw terminal rewards. The actor-critic modification adds a learned value baseline V(s) to reduce this variance: instead of asking "was this a good outcome?", the agent asks "was this outcome better than expected?" -- the advantage signal `A = R - V(s)`.

The hypothesis was that variance reduction from the value baseline would enable faster, more stable policy learning than pure REINFORCE, potentially matching or exceeding the value-based agent.

## Architecture

### Actor-Critic Network (`ActorCriticNetwork`)

A shared-backbone network with separate policy and value heads:

```
Input (15) --> Linear(15, 64) --> ReLU --> Linear(64, 64) --> ReLU
                                                               |
                                                     +---------+---------+
                                                     |                   |
                                            Linear(64, 3)       Linear(64, 1)
                                                     |                   |
                                               Softmax              Value V(s)
                                                     |
                                          P(fold), P(call), P(raise)
```

| Component | Input Dim | Output Dim | Activation | Parameters |
|-----------|-----------|------------|------------|------------|
| Backbone Linear 1 | 15 | 64 | ReLU | 1,024 |
| Backbone Linear 2 | 64 | 64 | ReLU | 4,160 |
| Policy Head | 64 | 3 | Softmax | 195 |
| Value Head | 64 | 1 | None | 65 |
| **Total** | | | | **5,444** |

### Observation Encoding (15 dimensions)

Same encoding scheme as the value-based agent:

| Features | Dims | Encoding |
|----------|------|----------|
| Player hand | 3 | One-hot (J/Q/K) |
| Board card | 4 | One-hot (J/Q/K/None) |
| Pot sizes | 2 | Normalized by MAX_CHIPS=13 |
| Current player | 1 | Float: player ID (0 or 1) |
| Round | 1 | Float: current round (0 or 1) |
| Terminal | 1 | Binary |
| Has pair | 1 | Binary |
| Raises normalized | 1 | raises_this_round / 2.0 |
| Can raise | 1 | Binary: 1.0 if RAISE is legal |

Note: The actor-critic encoding includes a "can raise" feature (dim 15) that the value-based agent does not have. This is because the policy head needs to know which actions are available to output meaningful probabilities.

### Action Selection

Unlike the value-based agent's 1-step lookahead, the actor-critic directly outputs action probabilities:

1. Forward pass produces `probs = [P(fold), P(call), P(raise)]` and `V(s)`
2. Illegal actions are masked (probability set to 0) and probabilities renormalized
3. **Training:** Sample from the categorical distribution
4. **Evaluation:** Pick the highest-probability legal action (greedy)

This is fundamentally different from value-based: no simulation, no lookahead. The policy must learn to map states directly to good action distributions.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 3,000 |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| Optimizer | Adam |
| Value Coefficient | 0.5 |
| Training Method | REINFORCE + value baseline (self-play) |

### Training Algorithm: REINFORCE with Learned Baseline

The `ActorCriticTrainer` uses advantage-weighted policy gradients:

1. **Episode collection:** Play a complete game. For each decision by player `p`, record:
   - `log_prob`: log probability of the chosen action
   - `value`: V(s) prediction from the critic head
   - The same terminal `reward` applies to all decisions by that player

2. **Loss computation:** For each (log_prob, value) pair:
   - `advantage = reward - V(s).detach()` (no gradient through baseline for policy)
   - `policy_loss = -log_prob * advantage` (REINFORCE with baseline)
   - `value_loss = (V(s) - reward)^2` (MSE to train the critic)
   - `total_loss = policy_loss + 0.5 * value_loss`

3. **Key design choice:** All decisions within an episode receive the same terminal reward. This means every action in the game is reinforced equally based on the final outcome -- there is no temporal credit assignment within an episode.

### Why the Learning Rate is 10x Higher

The actor-critic uses lr=1e-3 vs value-based's lr=1e-4. Policy gradient methods require larger learning rates because the gradient signal `log_prob * advantage` is inherently noisier than TD error signals. The value baseline reduces but does not eliminate this noise.

## Tournament Results

### Round 3 -- Full 17-Agent Tournament (1000 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | -0.619 |
| value_based | -1.044 |
| adaptive_value | -1.050 |
| aux_value | -0.225 |
| history_value | -0.512 |
| decay_adaptive | +0.149 |
| nstep_value | -0.639 |
| entropy_ac | -1.426 |
| pop_adaptive | -0.180 |
| adaptive_history | -0.537 |
| target_value | -0.532 |
| td_variant | -0.412 |
| pruned_history | -0.151 |
| modulated_value | -1.173 |
| curriculum | -0.415 |
| extended_adaptive | -0.327 |

### Performance Profile
- **Wins against:** Only 1 of 16 opponents (decay_adaptive)
- **Loses to:** 15 of 16 opponents
- **Best matchup:** +0.149 vs decay_adaptive
- **Worst matchup:** -1.426 vs entropy_ac

### Cross-Round Trajectory

| Round | Rank | Avg Score | Robustness | Context |
|-------|------|-----------|------------|---------|
| R1 | #5 / 7 | -0.54 | N/A | Below heuristic, above history_value and decay_adaptive |
| R2 | #7 / 12 | -0.62 | -1.38 | Steady decline relative to field |
| R3 | #10 / 17 | -0.568 | -1.197 | Beaten by its own descendant entropy_ac |

## Key Findings

1. **Policy collapse is the core failure.** During training, the policy converges to near-deterministic strategies (e.g., always call). Once the policy becomes deterministic, the gradient signal vanishes -- `log_prob` for the chosen action approaches 0, and all other actions have near-zero probability and receive no learning signal. This is a well-known failure mode of policy gradient methods.

2. **The credit assignment problem is catastrophic.** In poker, the agent makes 1-3 decisions per game, and all receive the same terminal reward. If the agent calls with a strong hand and then folds to a bluff, both decisions get the same negative reinforcement. The value baseline helps (it tells the agent "losing 2 chips is worse than expected") but cannot distinguish which specific action caused the loss.

3. **No 1-step lookahead is a structural disadvantage.** The value-based agent explicitly simulates actions and evaluates successors. The actor-critic must learn the mapping from state to action purely from reward signals. This means the policy must implicitly learn what the value-based agent gets for free from the game model.

4. **Entropy regularization partially fixes this.** The descendant entropy_ac (rank 6, +0.664 avg) adds an entropy bonus to prevent policy collapse, and it dramatically outperforms the vanilla actor-critic. This confirms that policy collapse, not the actor-critic architecture itself, is the bottleneck.

5. **Shared backbone is a mixed blessing.** The policy and value heads share features, which means value learning can help the policy learn useful representations. But it also means policy gradient noise can destabilize the value predictions, creating a feedback loop of degradation.

## Key Insight

Policy gradient methods fail in self-play poker not because the algorithm is wrong, but because 3,000 episodes with a single terminal reward per game provides insufficient signal for the policy to converge before it collapses to a deterministic (and exploitable) strategy.

## Source Files

- Agent: `src/agents/actor_critic.py`
- Trainer: `src/training/actor_critic_trainer.py`
