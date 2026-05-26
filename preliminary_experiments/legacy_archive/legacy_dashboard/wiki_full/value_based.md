# Value-Based Agent

> TD(0) self-play value network with 1-step lookahead and Boltzmann exploration -- the foundational RL agent from which most descendants inherit.

| Property | Value |
|----------|-------|
| **ID** | `value_based` |
| **Parent** | None -- baseline |
| **Round** | 0 |
| **Rank** | #2 / 17 |
| **Avg Score** | +0.970 |
| **Robustness** | +0.049 |

## Motivation

The Value-Based Agent is the foundational RL agent of the entire project. The core hypothesis: can a simple neural network learn to evaluate poker states through self-play, using only a scalar value prediction and temporal-difference learning? No policy head, no opponent modeling, no game-tree search -- just "how good is this state for me?" combined with 1-step lookahead to pick actions.

This agent was designed to be the simplest possible RL approach that could plausibly work, establishing a baseline for all future RL experiments. It ended up being far more than a baseline -- it is the second-best agent in the entire 17-agent tournament and the direct ancestor of 11 other agents in the family tree.

## Architecture

### Value Network (`ValueNetwork`)

A 3-layer MLP that estimates the scalar state value V(s):

```
Input (15) --> Linear(15, 64) --> ReLU --> Linear(64, 64) --> ReLU --> Linear(64, 1)
```

| Layer | Input Dim | Output Dim | Activation | Parameters |
|-------|-----------|------------|------------|------------|
| Linear 1 | 15 | 64 | ReLU | 1,024 |
| Linear 2 | 64 | 64 | ReLU | 4,160 |
| Linear 3 | 64 | 1 | None | 65 |
| **Total** | | | | **5,249** |

### Observation Encoding (15 dimensions)

The 15-dimensional input vector encodes the game state relative to the viewing player:

| Features | Dims | Encoding |
|----------|------|----------|
| Player hand | 3 | One-hot (J/Q/K) |
| Board card | 4 | One-hot (J/Q/K/None) |
| Pot sizes | 2 | Normalized by MAX_CHIPS=13, relative to viewer |
| My turn | 1 | Binary: 1.0 if it is the viewer's turn |
| Position | 1 | Float: viewer's player ID (0 or 1) |
| Round | 1 | Float: current betting round (0 or 1) |
| Terminal | 1 | Binary: 1.0 if game is finished |
| Has pair | 1 | Binary: 1.0 if hand matches board card |
| Raises normalized | 1 | raises_this_round / 2.0 |

### Action Selection: 1-Step Lookahead

The agent does not directly output action probabilities. Instead, it simulates each legal action, evaluates the resulting state with the value network, and selects based on those evaluations:

1. For each legal action `a` in {FOLD, CALL, RAISE}:
   - Simulate `obs' = simulate_action(obs, a)`
   - If `a == FOLD` and game ends: `V = -pot_contribution` (known loss)
   - Otherwise: `V = ValueNetwork(encode(obs'))`
2. **Training:** Boltzmann (softmax) selection with temperature -- `P(a) = softmax(V(s') / T)`
3. **Evaluation:** Greedy selection -- pick the action with highest V(s')

This 1-step lookahead is critical: the network only needs to learn state values, not action values. The search over actions is explicit, which dramatically simplifies what the network must learn.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 3,000 |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | MSE |
| Exploration | Boltzmann (temperature=1.0) |
| Training Method | TD(0) self-play |

### Training Algorithm: TD(0) Self-Play

The `SelfPlayTrainer` trains the agent by playing games against itself:

1. **Episode collection:** Play a complete game. Both players use the same network. For each action taken by player `p`, record the encoded post-action state in `chains[p]`.

2. **TD(0) update:** For each player's chain of post-action states:
   - For non-terminal transitions: `target = V(s_{t+1})` (bootstrap from next state)
   - For the last state: `target = terminal_reward` (actual game outcome)
   - Loss: `MSE(V(s_t), target)`

3. **Batch accumulation:** Collect 32 episodes, compute all TD losses, take one gradient step.

4. **Self-play dynamics:** Because both players share the same network, the training signal is inherently non-stationary -- the opponent improves as the agent improves. TD(0) bootstrapping provides implicit temporal smoothing that stabilizes learning in this setting.

### Why TD(0) Works So Well Here

Leduc Hold'em games last 2-6 decision steps. With such short episodes:
- TD(0) bootstrapping smooths out the high variance of terminal rewards
- The bootstrap target `V(s_{t+1})` acts as an exponential moving average of future outcomes
- In contrast, Monte Carlo methods (n-step with large n) use raw terminal rewards, which are noisy +/- chip values

## Tournament Results

### Round 3 -- Full 17-Agent Tournament (1000 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | +0.092 |
| adaptive_value | +0.143 |
| aux_value | +0.812 |
| actor_critic | +1.044 |
| history_value | +1.925 |
| decay_adaptive | +1.294 |
| nstep_value | +1.027 |
| entropy_ac | +0.597 |
| pop_adaptive | +1.550 |
| adaptive_history | +0.870 |
| target_value | +1.247 |
| td_variant | +1.420 |
| pruned_history | +1.560 |
| modulated_value | -0.126 |
| curriculum | +1.680 |
| extended_adaptive | +0.392 |

### Performance Profile
- **Wins against:** 15 of 16 opponents (positive score)
- **Loses to:** Only modulated_value (-0.126)
- **Best matchup:** +1.925 vs history_value
- **Worst matchup:** -0.126 vs modulated_value

### Cross-Round Trajectory

| Round | Rank | Avg Score | Robustness | Context |
|-------|------|-----------|------------|---------|
| R1 | #2 / 7 | +0.91 | N/A | Behind only adaptive_value |
| R2 | #2 / 12 | +0.98 | +0.06 | Second-highest robustness |
| R3 | #2 / 17 | +0.970 | +0.049 | Beaten only by modulated_value (which is built on top of it) |

## Key Findings

1. **The most successful architecture in the project.** Rank 2 across all three tournament rounds, with positive robustness in every evaluation. 11 of the 17 agents descend from it either directly or through adaptive_value.

2. **Simplicity is the strength.** With only 5,249 parameters and a 15-dimensional input, the network is small enough to converge reliably in 3,000 episodes of self-play. Larger networks and richer inputs consistently fail to match it.

3. **TD(0) bootstrapping is uniquely suited to self-play.** The bootstrap target provides implicit temporal smoothing that stabilizes learning when the opponent (yourself) is constantly changing. Monte Carlo and n-step variants remove this smoothing and perform worse.

4. **1-step lookahead compensates for a simple value function.** By explicitly simulating actions and evaluating successors, the agent gets action-level discrimination without needing a Q-network or policy head. This is a form of model-based RL that works because the game model (simulate_action) is exact.

5. **Boltzmann exploration prevents reward hacking.** During training, softmax action selection ensures the agent samples suboptimal actions proportionally to their estimated value, maintaining exploration without epsilon-greedy's crude randomness.

6. **It is the hidden backbone of the #1 agent.** Modulated_value (rank 1) works by freezing a pretrained value_based network and adding a small modulation layer. The value_based network IS the actual decision-maker; modulation just fine-tunes it by ~6%.

## Key Insight

A simple 5,249-parameter value network with TD(0) self-play and 1-step lookahead outperforms every algorithmic innovation tested across 3 rounds of experimentation -- the only agent to beat it is one that starts from its own pretrained weights and is architecturally constrained not to deviate far.

## Source Files

- Agent: `src/agents/value_based.py`
- Trainer: `src/training/value_based_trainer.py`
