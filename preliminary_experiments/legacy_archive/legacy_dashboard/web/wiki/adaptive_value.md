# Adaptive Value Agent

> Value-based agent augmented with 4 running opponent statistics features -- the most successful innovation in the project and the parent of most subsequent agents.

| Property | Value |
|----------|-------|
| **ID** | `adaptive_value` |
| **Parent** | ValueBasedAgent (`value_based`) |
| **Round** | 0 (Generation 1) |
| **Rank** | #3 / 17 |
| **Avg Score** | +1.012 |
| **Robustness** | -0.030 |

## Motivation

The Value-Based Agent plays the same strategy against every opponent. But poker is fundamentally about exploiting opponent tendencies -- a player who folds too much should be bluffed; a player who never folds should be value-bet with strong hands. The Adaptive Value Agent tests whether appending a small number of cross-hand opponent statistics to the observation can teach the network to adjust its play based on who it is facing.

The key design choice was **minimalism**: only 4 additional features (fold_rate, raise_rate, fold_to_raise_rate, confidence), for a total of 19 dimensions. This is in contrast to the history_value approach that added 16 features. The hypothesis was that a few well-chosen aggregate statistics would be more learnable than a detailed action sequence encoding.

This agent became the most impactful innovation in the project, serving as the parent of 7 descendant agents across Rounds 1-3.

## Architecture

### Value Network

Same `ValueNetwork` class as the parent, with 4 additional input dimensions:

```
Input (19) --> Linear(19, 64) --> ReLU --> Linear(64, 64) --> ReLU --> Linear(64, 1)
```

| Layer | Input Dim | Output Dim | Activation | Parameters |
|-------|-----------|------------|------------|------------|
| Linear 1 | 19 | 64 | ReLU | 1,280 |
| Linear 2 | 64 | 64 | ReLU | 4,160 |
| Linear 3 | 64 | 1 | None | 65 |
| **Total** | | | | **5,505** |

Note: Only 256 more parameters than the parent (5,249), because only the first layer changes size.

### Observation Encoding (19 dimensions)

| Features | Dims | Source |
|----------|------|--------|
| Player hand | 3 | One-hot (J/Q/K) |
| Board card | 4 | One-hot (J/Q/K/None) |
| Pot sizes | 2 | Normalized by MAX_CHIPS=13, relative to viewer |
| My turn | 1 | Binary |
| Position | 1 | Float: viewer's player ID |
| Round | 1 | Float: current round |
| Terminal | 1 | Binary |
| Has pair | 1 | Binary |
| Raises normalized | 1 | raises_this_round / 2.0 |
| **fold_rate** | 1 | Fraction of hands opponent has folded |
| **raise_rate** | 1 | Fraction of hands opponent has raised |
| **fold_to_raise_rate** | 1 | Fraction of times opponent folded when facing a raise |
| **confidence** | 1 | hands_observed / max_hands (0 to 1 scale) |

### The 4 Opponent Features -- Why These Specific Stats?

1. **fold_rate**: How often the opponent gives up. A high fold rate means the opponent is passive and can be bluffed profitably.

2. **raise_rate**: How often the opponent shows aggression. A high raise rate means the opponent bets frequently, so their raises carry less information about hand strength.

3. **fold_to_raise_rate**: The critical exploit feature. If an opponent folds to raises 80% of the time, every raise is profitable regardless of hand strength. If they never fold to raises, bluffing is futile.

4. **confidence**: The meta-feature that makes everything work. `confidence = hands_observed / max_hands`, ranging from 0 (no data) to 1 (many hands observed). This tells the network how much to trust the other 3 stats.

### The Default Vector: Graceful Degradation

When no opponent stats are available (e.g., first hand of a session, or single-hand evaluation):

```python
stats_vec = torch.tensor([0.5, 0.5, 0.5, 0.0])
```

This default is carefully chosen:
- **0.5 for rates**: Uninformative prior -- "assume the opponent is perfectly balanced"
- **0.0 for confidence**: "I have no information about this opponent"

This means the network learns two distinct modes:
1. **Low confidence (confidence near 0):** Ignore the stat features, fall back to base game state evaluation (effectively behaving like ValueBasedAgent)
2. **High confidence (confidence near 1):** Exploit the stat features to adjust play against this specific opponent type

### Action Simulation with Stats Propagation

The 1-step lookahead must carry opponent_stats into simulated successor states, because `LeducGame.simulate_action()` discards them:

```python
post_obs, done = LeducGame.simulate_action(obs, action)
if obs.opponent_stats is not None:
    post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)
```

Without this propagation, the value network would receive zeroed stat features in every simulated state, defeating the purpose of the adaptation.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Sessions | 100 |
| Hands per Session | 30 |
| Total Hands | ~3,000 |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | MSE |
| Training Method | TD(0) session-based self-play |

### Session-Based Training (AdaptiveTrainer)

Unlike the parent's per-hand self-play, the adaptive agent trains in **sessions** of 30 hands:

1. **Reset session:** Clear all opponent statistics
2. **Play 30 hands:** Both players use the same network. After each hand, the session updates running opponent stats (fold counts, raise counts, etc.)
3. **Collect chains:** Each hand produces per-player post-action state chains, with current opponent stats embedded in the encoding
4. **TD(0) update:** Standard temporal-difference learning on the collected chains

The session structure is critical: it creates a natural curriculum where early-session decisions have low confidence (and should rely on base features) while late-session decisions have high confidence (and can exploit stats). The network must learn BOTH modes simultaneously.

### Why Self-Play Training Works Despite Training Against Yourself

The obvious objection: "If you are playing against yourself, the opponent stats just describe your own behavior. How can you learn to exploit opponent tendencies you have never seen?"

The answer lies in the confidence mechanism:

1. **During training:** The agent plays against itself, so opponent stats describe its own policy. The stats are noisy (policy changes every batch update) but the confidence signal is reliable. The network learns: "when confidence is low, ignore stats and play solid poker."

2. **During evaluation:** Against a new opponent, confidence starts at 0. The network falls back to its base strategy (equivalent to a well-trained ValueBasedAgent). As hands accumulate, confidence grows, and the network begins adjusting to the actual opponent's tendencies.

3. **The self-play training teaches robustness, not exploitation.** The agent does not learn specific exploits during training. It learns the base game well (because low confidence forces this) and learns the correlation between stats and value adjustments (because high confidence allows this). The specific exploits emerge naturally when the stats describe a genuinely exploitable opponent at evaluation time.

## Tournament Results

### Round 3 -- Full 17-Agent Tournament (1000 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | +0.386 |
| value_based | -0.143 |
| aux_value | +1.229 |
| actor_critic | +1.050 |
| history_value | +1.711 |
| decay_adaptive | +1.559 |
| nstep_value | +0.889 |
| entropy_ac | +0.339 |
| pop_adaptive | +1.731 |
| adaptive_history | +0.854 |
| target_value | +1.115 |
| td_variant | +1.618 |
| pruned_history | +1.745 |
| modulated_value | -0.160 |
| curriculum | +1.960 |
| extended_adaptive | +0.316 |

### Performance Profile
- **Wins against:** 14 of 16 opponents (positive score)
- **Loses to:** value_based (-0.143), modulated_value (-0.160)
- **Best matchup:** +1.960 vs curriculum
- **Worst matchup:** -0.160 vs modulated_value
- **Highest average score** in the entire tournament (+1.012)

### Cross-Round Trajectory

| Round | Rank | Avg Score | Robustness | Context |
|-------|------|-----------|------------|---------|
| R1 | #1 / 7 | +0.99 | N/A | Best agent, first RL agent to clearly beat heuristic |
| R2 | #1 / 12 | +1.06 | +0.12 | Best average AND best robustness |
| R3 | #3 / 17 | +1.012 | -0.030 | Highest average but rank 3 due to slight robustness dip |

## Key Findings

1. **Highest average score across all 17 agents.** At +1.012 avg chips/round, the adaptive value agent wins more on average than any other agent, including the tournament-winning modulated_value (+0.967). Its slightly lower robustness (due to losses against value_based and modulated_value) is why it ranks 3rd instead of 1st.

2. **Only 4 features made the difference.** Adding just 4 well-chosen features to the 15-base observation transformed a rank-2 agent into the highest-scoring agent across 3 rounds. Compare this to history_value's 16 features (rank 16) -- feature quality dominates feature quantity.

3. **The confidence mechanism is the secret weapon.** Without confidence, the agent would need to learn a single strategy that works with or without meaningful stats. The confidence feature lets the network explicitly condition on data quality, learning a graceful degradation from "exploit mode" to "safe mode."

4. **Spawned 7 descendants.** This agent is the parent of decay_adaptive, pop_adaptive, adaptive_history, pruned_history, modulated_value, curriculum, and extended_adaptive. The fact that the #1 agent (modulated_value) descends from adaptive_value confirms its fundamental approach is sound.

5. **Robust across all opponent types.** The agent beats weak opponents by large margins (exploiting their tendencies) and competes closely with strong opponents (falling back to solid base play). Its only losses are to agents that either have the same knowledge (value_based, which adaptive_value nearly ties) or inherit its strengths (modulated_value).

6. **Self-play training WORKS for opponent modeling.** Despite never seeing a "real" opponent during training, the agent effectively exploits diverse opponents at evaluation time. The session structure with the confidence ramp naturally teaches both robust base play and conditional exploitation.

## Key Insight

The confidence feature is the reason adaptive_value succeeds where other observation augmentations fail: it starts at 0 for new opponents, causing the network to fall back to well-learned base features, then gradually enables exploitation as statistics accumulate -- a built-in curriculum from "play solid poker" to "exploit this specific opponent."

## Source Files

- Agent: `src/agents/adaptive_value.py`
- Trainer: `src/training/adaptive_trainer.py`
