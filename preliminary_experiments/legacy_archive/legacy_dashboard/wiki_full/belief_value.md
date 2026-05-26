# Bayesian Belief Agent

> Bayesian hand inference that updates opponent hand probabilities after each action -- a theoretically elegant approach undermined by self-play training that produces a degenerate likelihood model.

| Property | Value |
|----------|-------|
| **ID** | `belief_value` |
| **Parent** | BaseAgent (NEW LINEAGE) |
| **Round** | 4 |
| **Rank** | ~18 / 22 |
| **Avg Score** | -0.434 |
| **Robustness** | -1.024 |

---

## Motivation

All previous agents in the project treat the opponent as a black box -- they observe outcomes and statistics but never explicitly reason about what the opponent is holding. In real poker, skilled players constantly update their mental model of the opponent's hand range based on each action: "they raised preflop, so they probably don't have a Jack." The Bayesian Belief Agent attempts to formalize this reasoning by maintaining an explicit probability distribution P(opponent_hand) that is updated via Bayes' rule after each opponent action.

The hypothesis: if the agent can accurately infer what the opponent is holding, it can make substantially better fold/call/raise decisions by computing expected value against the inferred hand distribution rather than the prior.

---

## Architecture

BeliefValueAgent introduces a **two-network architecture** with a novel Bayesian belief-tracking pipeline:

### Network 1: Likelihood Model
```
LikelihoodModel(10 -> 32 -> 32 -> 3)
  - Input: 10-dim context (game state features excluding own hand)
  - Architecture: 2-layer MLP with ReLU, 32 hidden units
  - Output: P(action | opponent_hand) for each of 3 possible hands (J, Q, K)
  - Trained via cross-entropy on observed opponent actions during self-play
```

### Network 2: Belief Value Network
```
BeliefValueNetwork(14 -> 64 -> 64 -> 1)
  - Input: 14-dim observation (hand 3 + board 4 + pot 2 + belief 3 + round 1 + raises 1)
  - Architecture: 2-layer MLP with ReLU, 64 hidden units
  - Output: Scalar state value
  - Trained via TD(0) on self-play episodes
```

### Bayesian Belief Update

At each opponent action, the agent performs a Bayesian update:

```
P(hand | actions) = P(action | hand) * P(hand | prev_actions) / Z
```

Where `P(action | hand)` comes from the likelihood model. The belief vector (3 probabilities over J/Q/K) is included in the observation fed to the value network, giving it access to the agent's current inference about the opponent's hand.

### Observation Encoding (14 dimensions)

| Features | Dims | Encoding |
|----------|------|----------|
| Player hand | 3 | One-hot (J/Q/K) |
| Board card | 4 | One-hot (J/Q/K/None) |
| Pot sizes | 2 | Normalized |
| Belief vector | 3 | P(opp=J), P(opp=Q), P(opp=K) |
| Round | 1 | Current round (0 or 1) |
| Raises | 1 | Raises this round / 2.0 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Self-play with Bayesian belief tracking |
| Episodes | 20,000 |
| Learning Rate | 1e-4 (Adam) |
| Value Loss | TD(0) |
| Likelihood Loss | Cross-entropy on opponent actions |
| Belief Prior | Uniform [1/3, 1/3, 1/3] at episode start |

The likelihood model and value network are trained simultaneously during self-play. At each opponent decision, the likelihood model is supervised on the observed action, and beliefs are updated before the next value network forward pass.

---

## Tournament Results

| Opponent | Score |
|----------|-------|
| heuristic | -0.338 |
| value_based | -0.708 |
| adaptive_value | -0.626 |
| modulated_value | -0.862 |
| entropy_ac | +0.240 |
| cfr | -0.310 |

**Best matchup**: +0.240 vs entropy_ac
**Worst matchup**: -0.862 vs modulated_value
**Wins against**: 1 of 6 core opponents

---

## Diagnosis & Findings

### 1. Likelihood Model is Broken

The likelihood model, which is the foundation of the entire Bayesian approach, fails catastrophically:

| Action | Prediction Accuracy |
|--------|-------------------|
| FOLD | 2.7% |
| CALL | 32.2% |
| RAISE | 88.3% |

The model predicts RAISE with high confidence regardless of the opponent's hand, and almost never predicts FOLD. This makes the Bayesian updates nearly meaningless -- if P(action|hand) is roughly the same for all hands, the posterior barely moves from the prior.

### 2. Self-Play Produces Degenerate Training Data

The root cause is a **training data distribution problem**. During self-play, the agent plays against copies of itself. Early in training, the agent (like most RL agents) learns an aggressive strategy that overwhelmingly selects RAISE. This means the likelihood model is trained on data where ~80% of observed actions are RAISE, fold is extremely rare, and the action distribution barely varies across different hands.

The likelihood model faithfully learns this distribution: "everyone raises, almost no one folds." This is accurate for the self-play training distribution but useless for actual opponent modeling.

### 3. Beliefs Barely Shift

Because the likelihood model outputs nearly uniform predictions across hands for any given action, the Bayesian update produces negligible belief shifts. The belief vector stays close to [0.33, 0.33, 0.33] throughout most games, which means the value network receives essentially no useful information from the belief features.

### 4. The Agent Never Folds

A downstream consequence of the broken beliefs: without meaningful hand inference, the agent cannot identify situations where folding is optimal. It defaults to an aggressive strategy similar to a poorly-trained value-based agent, never folding when it should.

---

## Assumptions & Limitations

1. **Self-play bias in likelihood model**: The likelihood model is trained exclusively on self-play data, so it learns the agent's own policy -- not the opponent's. This produces catastrophically skewed prediction accuracy: 88% for RAISE (the dominant self-play action), 32% for CALL, but only 3% for FOLD. The model is useless for distinguishing between opponent hands because it predicts the same action distribution regardless of what the opponent holds.

2. **Belief reset at round transitions**: When the board card is revealed (transitioning from preflop to postflop), all preflop belief accumulation is discarded. The belief vector resets to the card-removal prior. This means that information gained from preflop actions (e.g., "the opponent raised preflop, suggesting a strong hand") does not carry forward. In a 2-round game like Leduc, this wastes roughly half of the available evidence.

3. **Small-game limitation (3 possible opponent hands)**: In Leduc Hold'em, there are only 3 card ranks. Card removal alone narrows the opponent to 2 possible hands (e.g., if you hold Q, the opponent has either J or K). This means the prior is already very informative -- the belief system has relatively little room to add value beyond what card removal provides. In larger games (e.g., Texas Hold'em with 1,326 possible hole card combinations), the belief system would have much more room to contribute.

4. **Beliefs barely shift**: Because P(RAISE | any_hand) is nearly uniform across hands in the self-play-trained likelihood model, Bayes' update produces negligible belief shifts. The belief vector stays close to the card-removal prior [0.5, 0.0, 0.5] throughout most games. The value network effectively receives no useful information from the belief features, making the 3 belief dimensions wasted capacity.

---

## Key Insight

Bayesian opponent modeling is conceptually sound for poker but fundamentally incompatible with self-play training -- the likelihood model learns from a single policy's action distribution, which is far too narrow to capture the diversity of opponent behaviors needed for meaningful hand inference. The fix would require training the likelihood model against a diverse population of opponents, not against copies of itself.

---

## Source Files

- Agent: `src/agents/belief_value.py`
- Trainer: `src/training/belief_value_trainer.py`
