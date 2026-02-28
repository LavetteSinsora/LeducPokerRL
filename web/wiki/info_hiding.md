# Information-Hiding Agent

> Adversarial training against a spy network to learn information-hiding policies -- a game-theoretically motivated approach that fails because discrete action sampling blocks gradient flow from the adversary.

| Property | Value |
|----------|-------|
| **ID** | `info_hiding` |
| **Parent** | actor_critic (ActorCriticAgent) |
| **Round** | 4 |
| **Rank** | ~20 / 22 |
| **Avg Score** | -0.626 |
| **Robustness** | -1.133 |

---

## Motivation

In poker, information leakage is costly. If an opponent can infer your hand from your actions, they can exploit you perfectly. Strong human players deliberately act unpredictably -- sometimes betting big with weak hands (bluffing) and sometimes checking with strong hands (slow-playing) -- specifically to hide information.

The Information-Hiding Agent formalizes this through **adversarial training**. A "spy network" is trained to infer the agent's hand from its observed actions. The agent is then penalized when the spy succeeds, creating an incentive to adopt policies that are hard to read. This is directly analogous to GAN training: the spy is the discriminator, and the agent's policy is the generator.

The game-theoretic motivation is clear: in a Nash equilibrium, a player's mixed strategy ensures that their actions reveal no exploitable information about their private cards. Adversarial training should push the policy toward this information-hiding property.

---

## Architecture

### Main Network: Actor-Critic
```
ActorCriticNetwork(15 -> 64 -> 64 -> {policy: 3, value: 1})
  - Input: 15-dim game state encoding
  - Shared backbone: 2-layer MLP with ReLU, 64 hidden units
  - Policy head: Linear(64, 3) -> Softmax (action probabilities)
  - Value head: Linear(64, 1) (scalar state value)
  - Parameters: 5,444
```

### Adversary: Spy Network
```
SpyNetwork(20 -> 32 -> 3)
  - Input: 20-dim observation (15-dim game state + 5-dim action history features)
  - Architecture: Single hidden layer, 32 units, ReLU
  - Output: P(hand = J), P(hand = Q), P(hand = K)
  - Trained to predict the agent's true hand from observable actions
  - Parameters: 771
```

### Training Loss

The combined loss balances three objectives:

```
total_loss = policy_gradient_loss + 0.5 * value_loss - lambda * cross_entropy(spy_pred, true_hand)
```

| Component | Purpose |
|-----------|---------|
| `policy_gradient_loss` | Standard REINFORCE with value baseline -- maximize reward |
| `0.5 * value_loss` | Critic regression -- accurate state value estimates |
| `-lambda * cross_entropy(spy, hand)` | Adversarial term -- penalize when spy can identify hand |

The negative sign on the spy loss means the optimizer tries to **maximize** the spy's cross-entropy (make the spy's predictions as wrong as possible), while the spy's own optimizer minimizes it. This creates the adversarial dynamic.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Self-play with adversarial spy |
| Episodes | 40,000 |
| Learning Rate | 1e-3 (Adam) |
| Value Coefficient | 0.5 |
| Lambda (spy penalty) | 0.1 |
| Spy Learning Rate | 1e-3 |
| Total Parameters | 6,215 (actor-critic: 5,444 + spy: 771) |

The spy network is updated on every batch alongside the actor-critic, creating a minimax game: the spy tries to read the agent's hand, the agent tries to be unreadable.

---

## Tournament Results

| Opponent | Score |
|----------|-------|
| heuristic | -0.644 |
| value_based | -0.590 |
| adaptive_value | -0.514 |
| modulated_value | -0.892 |
| entropy_ac | -0.070 |
| cfr | -1.046 |

**Best matchup**: -0.070 vs entropy_ac
**Worst matchup**: -1.046 vs cfr
**Wins against**: 0 of 6 core opponents

---

## Diagnosis & Findings

### 1. Spy Accuracy Reduction is Real but Insufficient

The adversarial training does reduce how readable the agent is:

| Agent | Spy Accuracy |
|-------|-------------|
| value_based | 88.0% |
| entropy_ac | 75.0% |
| **info_hiding** | **73.3%** |
| cfr | 61.8% |

The info_hiding agent is marginally harder to read than entropy_ac, but still far more readable than CFR (which achieves near-optimal information hiding through its Nash equilibrium mixed strategy).

### 2. Action Distributions Tell a Different Story

Despite the spy accuracy improvement, the agent's actual action distributions reveal that it has not learned meaningful information hiding:

**Raise Gap (King vs Jack)**: The difference in raise probability between the strongest hand (King) and weakest hand (Jack). Lower is better for information hiding.

| Agent | Raise Gap (K - J) |
|-------|-------------------|
| value_based | 0.463 |
| **info_hiding** | **0.804** |

The info_hiding agent has a **larger** raise gap than value_based -- it raises more with Kings and less with Jacks, making it **easier** to read, not harder. The spy accuracy improvement comes from other factors (e.g., slightly noisier overall play), not from genuine information hiding.

### 3. The Agent Never Folds

Fold rates for the info_hiding agent:

| Hand | Fold Rate |
|------|-----------|
| Jack | 0% |
| Queen | 0% |
| King | 0% |

The agent has learned to never fold, regardless of hand strength. This is a symptom of the policy collapsing to a narrow action space, not a deliberate strategic choice.

### 4. Lambda Sweep Shows Non-Monotonic, Ineffective Response

| Lambda | Spy Accuracy |
|--------|-------------|
| 0.0 | 74.1% |
| 0.05 | 76.2% |
| 0.1 | 73.3% |
| 0.5 | 75.8% |
| 1.0 | 72.9% |

There is no consistent relationship between lambda (the adversarial penalty strength) and spy accuracy. Increasing the penalty does not systematically reduce readability. This is the clearest evidence that the adversarial gradient is not reaching the policy.

### 5. Root Cause: Non-Differentiable Action Sampling

The fundamental failure is in the gradient path. The actor-critic selects actions via `torch.multinomial(probs)` -- discrete sampling from the policy distribution. This sampling operation is **non-differentiable**: gradients cannot flow backward through a discrete sample.

The adversarial gradient from the spy loss never reaches the policy parameters. The policy is optimized only by the REINFORCE signal, which knows nothing about information hiding. The spy loss term in the total loss is effectively dead weight.

### 6. The Fix

Two approaches could restore gradient flow:

1. **Gumbel-Softmax**: Replace discrete sampling with a differentiable relaxation that allows gradients to flow through the action selection
2. **REINFORCE-based spy loss**: Instead of backpropagating through the spy, use the spy's prediction accuracy as a reward signal for the policy: `spy_reward = -log(spy_accuracy)`, applied via REINFORCE

Either approach would allow the policy to actually receive gradient signal about its information leakage.

---

## Key Insight

Adversarial information hiding is conceptually sound for poker but requires differentiable action selection -- the discrete sampling step (`torch.multinomial`) blocks gradient flow from the spy network to the policy, making the adversarial objective invisible to the policy optimizer. The fix requires either Gumbel-Softmax relaxation or a REINFORCE-based spy loss.

---

## Source Files

- Agent: `src/agents/info_hiding.py`
- Trainer: `src/training/info_hiding_trainer.py`
