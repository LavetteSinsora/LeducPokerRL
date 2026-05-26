# Nash Value Network

> A neural network trained to approximate CFR's counterfactual values via supervised learning -- a theoretically motivated approach that fails because Nash equilibria require mixed strategies, but the agent plays pure (argmax) strategies.

| Property | Value |
|----------|-------|
| **ID** | `nash_value` |
| **Parent** | BaseAgent (NEW LINEAGE, uses CFR as teacher) |
| **Round** | 4 |
| **Rank** | ~21 / 22 |
| **Avg Score** | -0.984 |
| **Robustness** | -1.221 |

---

## Motivation

The CFR agent computes a provably optimal Nash equilibrium for Leduc Hold'em, but it uses a tabular strategy store that maps information set keys to action probabilities. This is not scalable -- in larger poker variants, the number of information sets explodes. The Nash Value Network asks: can we **distill** CFR's game-theoretic knowledge into a neural network that generalizes across states?

The approach is knowledge distillation: run CFR to convergence, extract the counterfactual values at every information set, then train a standard ValueNetwork via supervised regression (MSE) to predict these values. If the network can accurately approximate the Nash values, it should inherit CFR's game-theoretic optimality while being compact and generalizable.

---

## Architecture

### Teacher: CFR Solver

```
CFR+ Solver
  - Iterations: 10,000
  - Information sets: 288
  - Final exploitability: 0.003136 (near-zero = near-Nash)
  - Output: counterfactual values for each (infoset, action) pair
```

### Student: Standard Value Network

```
ValueNetwork(15 -> 64 -> 64 -> 1)
  - Input: 15-dim game state encoding (same as value_based agent)
  - Architecture: 2-layer MLP with ReLU, 64 hidden units
  - Output: Scalar value estimate
  - Trained via MSE on CFR counterfactual values
  - Parameters: 5,249
```

### Training Pipeline

1. Run CFR for 10K iterations to convergence
2. For each information set, extract the counterfactual value of each action
3. Convert each information set to a 15-dim observation vector
4. Train the neural network to predict CFR values via supervised MSE loss

### Encoding Collision Problem

The 15-dim observation encoding maps 288 CFR information sets to only **180 unique encodings**. This means multiple distinct game states (with different optimal strategies) collide into the same input vector. The network cannot distinguish them, creating an **irreducible approximation error**.

| Metric | Value |
|--------|-------|
| CFR information sets | 288 |
| Unique 15-dim encodings | 180 |
| Collision rate | 37.5% |
| Irreducible MSE minimum | 0.317 |
| Achieved MSE | 0.324 |

The network essentially achieves the best possible MSE given the encoding -- it cannot do better without a richer state representation.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Supervised regression on CFR values |
| CFR Iterations | 10,000 |
| CFR Exploitability | 0.003136 |
| Loss Function | MSE(predicted_value, cfr_value) |
| Learning Rate | 1e-4 (Adam) |
| Epochs | 1,000 |
| Final MSE | 0.324 |
| Irreducible MSE | 0.317 |

---

## Tournament Results

| Opponent | Score |
|----------|-------|
| heuristic | -1.181 |
| value_based | -0.914 |
| adaptive_value | -1.126 |
| modulated_value | -0.903 |
| entropy_ac | -1.026 |
| cfr | -0.756 |

**Best matchup**: -0.756 vs cfr (its own teacher)
**Worst matchup**: -1.181 vs heuristic
**Wins against**: 0 of 6 core opponents

The agent loses to every opponent in the tournament, including (ironically) the CFR agent whose values it was trained to approximate.

---

## Diagnosis & Findings

### 1. Two Compounding Failures

The Nash Value Network suffers from two independent but compounding problems:

**Failure 1: Encoding Collisions (Approximation Error)**

288 information sets map to only 180 unique input vectors. When two information sets with different optimal values collide, the network can only learn their average. This introduces irreducible error that no amount of training can fix.

The achieved MSE (0.324) is barely above the theoretical minimum (0.317), confirming the network has learned everything it can from the data -- the bottleneck is the encoding, not the model capacity.

**Failure 2: Argmax on Nash Values (Decision Procedure)**

This is the more fundamental problem. Nash equilibrium strategies are **mixed** -- they require randomization over actions. The CFR agent samples from its mixed strategy, playing FOLD with probability 0.3, CALL with 0.4, RAISE with 0.3 (for example). This randomization is essential to unexploitability.

The Nash Value Network uses argmax: it picks the action with the highest predicted value. This converts the Nash mixed strategy into a **pure** (deterministic) strategy. A pure strategy derived from Nash values is not a Nash equilibrium -- it is a specific, predictable strategy that opponents can exploit.

### 2. Nash + Greedy is Fundamentally Incompatible

This is the key theoretical insight. Nash equilibrium values encode the expected payoff of a mixed strategy. Taking the argmax of these values does not recover the mixed strategy -- it selects one action from the support and plays it 100% of the time. This is like reading a recipe that says "use salt OR sugar" and always choosing salt. The recipe's value depends on mixing; committing to one ingredient defeats the purpose.

### 3. Comparison with CFR Agent

| Property | CFR Agent | Nash Value Network |
|----------|-----------|-------------------|
| Strategy type | Mixed (probabilistic) | Pure (deterministic) |
| Action selection | Sample from distribution | Argmax on values |
| State representation | Exact (information set key) | Lossy (15-dim encoding) |
| Exploitability | Near-zero (0.003) | High (deterministic) |
| Game knowledge | Complete (full tree) | Approximated (neural net) |

The CFR agent's strength comes from two properties the Nash Value Network discards: exact state representation and mixed strategy execution.

---

## Key Insight

The decision procedure (argmax) is the bottleneck, not the approximation quality. Even a perfect neural network that exactly reproduced every CFR value would fail, because Nash equilibria fundamentally require mixed strategies -- taking the greedy action from Nash values produces a pure strategy that is exploitable and loses to opponents that a properly mixed Nash strategy would beat.

---

## Source Files

- Agent: `src/agents/nash_value.py`
- Trainer: `src/training/nash_value_trainer.py`
- CFR Solver: `src/cfr/solver.py`
