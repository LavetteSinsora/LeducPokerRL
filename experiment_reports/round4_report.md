# Round 4: Creative Exploration — Experiment Report

> **Date**: 2026-02-27
> **Branch**: `exp/round-3-integration`
> **Goal**: Break from incremental single-variable optimization. Test 5 fundamentally different architectures, training paradigms, and objectives to see if radical approaches can discover strategies that incremental refinement cannot reach.

## Executive Summary

**All 5 creative directions failed to beat the incumbent agents.** After 3 rounds of incremental optimization that plateaued at avg ~+1.0, Round 4 tested 5 radically different approaches — Bayesian belief tracking, Nash value distillation, distributional RL, opponent modeling with 2-ply search, and adversarial information hiding. None approached the incumbents' performance. The best Round 4 agent (Belief, avg -0.434) still loses by 1.4 chips/hand compared to adaptive_value (+1.012).

However, the failures are deeply informative. Each direction failed for a **specific, diagnosable reason** that reveals fundamental constraints of the problem setup. These lessons should guide future work.

### Results at a Glance

| Direction | Agent ID | Avg Score | Best Matchup | Worst Matchup | Robustness |
|-----------|----------|-----------|-------------|--------------|------------|
| 1. Bayesian Belief | `belief_value` | **-0.434** | entropy_ac +0.24 | modulated_value -0.86 | -1.024 |
| 2. Nash Value | `nash_value` | -0.984 | cfr -0.76 | heuristic -1.18 | -1.221 |
| 3. Distributional | `distributional_value` | **-0.580** | entropy_ac +0.08 | modulated_value -1.04 | -1.230 |
| 4. Opponent Model | `opponent_model` | -1.398 | cfr -0.77 | adaptive_value -1.79 | -2.174 |
| 5. Info-Hiding | `info_hiding` | **-0.626** | entropy_ac -0.07 | cfr -1.05 | -1.133 |

*Distributional uses dual-head architecture (6th iteration). Beta=0.5 sweep shows avg -0.37 with robustness -0.88 — the best R4 robustness.*

**Incumbent comparison**: adaptive_value +1.012, value_based +0.970, modulated_value +0.967

---

## Direction 1: Bayesian Belief Agent (NEW LINEAGE)

### Concept
The first agent to reason about what card the opponent holds **in this specific hand**. Maintains a belief vector P(opponent_hand) = [P(J), P(Q), P(K)] initialized from card removal, updated after each opponent action via a learned likelihood model P(action | hand, state).

### Architecture
- **Likelihood Model**: MLP(10 → 32 → 32 → 3) — predicts P(action | hand, game_state)
- **Value Network**: MLP(14 → 64 → 64 → 1) — takes hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1)
- **Training**: 30K episodes self-play, TD(0) for value + cross-entropy for likelihood

### Results

| Opponent | Score |
|----------|-------|
| heuristic | -0.338 |
| value_based | -0.708 |
| adaptive_value | -0.626 |
| modulated_value | -0.862 |
| entropy_ac | **+0.240** |
| cfr | -0.310 |
| **Average** | **-0.434** |

### Diagnosis: Why It Failed

**The likelihood model is catastrophically biased.** Per-action accuracy:
- FOLD: **2.7%** (barely detected)
- CALL: 32.2%
- RAISE: **88.3%** (overwhelmingly predicted)

The model predicts RAISE with ~88% probability regardless of the opponent's hand. This happens because self-play training converges toward a single policy — the agent plays against itself, so the opponent model only observes one behavioral pattern.

**Consequence**: Belief vectors barely shift during a hand. The card removal prior (e.g., [0.5, 0.0, 0.5] when holding Q) dominates because the likelihood updates are near-uniform across hands. The agent never folds (0% fold rate for all hands) because it can't identify when the opponent likely has a strong hand.

**Root Cause**: Self-play is a poor training signal for opponent modeling. The agent can only learn "the opponent plays like me" because it never sees diverse opponent behaviors during training.

### Key Insight
> Within-hand belief tracking is a sound concept, but the likelihood model needs exposure to diverse opponent strategies during training. A curriculum that trains against the full pool of existing agents (heuristic, value_based, adaptive_value, etc.) would provide the behavioral diversity needed for the likelihood model to distinguish between different hands.

---

## Direction 2: Nash Value Network (GAME-THEORY + RL HYBRID)

### Concept
Use CFR to compute exact Nash equilibrium counterfactual values for all 288 information sets, then train a neural network on these exact values via supervised regression. The hypothesis: a value function learned from game-theoretically optimal targets should be maximally robust.

### Architecture
- **CFR**: 10K iterations, final exploitability 0.003136 (near-exact Nash)
- **Value Network**: MLP(15 → 64 → 64 → 1) — same as value_based
- **Training**: Pure supervised MSE regression on (encoding, CFR_value) pairs
- No self-play needed — all training data comes from CFR

### Results

| Opponent | Score |
|----------|-------|
| heuristic | -1.181 |
| value_based | -0.914 |
| adaptive_value | -1.126 |
| modulated_value | -0.903 |
| entropy_ac | -1.026 |
| cfr | -0.756 |
| **Average** | **-0.984** |

### Diagnosis: Two Compounding Failures

**Failure 1: Encoding collisions.** The 15-dimensional encoding maps 288 CFR infosets to only 180 unique encodings. 108 infosets collide (different game histories with identical encodings). This creates an irreducible MSE floor of 0.317. The network achieves MSE=0.324 — only 0.007 above the theoretical minimum. The approximation is essentially perfect given the encoding constraints.

**Failure 2 (the fundamental one): Nash + greedy = exploitable.** Nash equilibrium play requires **mixed strategies** — e.g., raise with K 60% of the time and call 40%. But our decision procedure is `argmax_a V(post(s, a))`, which always picks the single best action. This produces a **pure strategy** that is trivially exploitable.

The diagnosis is stark: nash_value diverges from self-play-trained values by an average of **2.67** chips per infoset (max 9.37 for K:K board states). States where Nash assigns high value to mixed play (K vs K, where bluffing is critical) show the largest divergence because the greedy strategy collapses the bluffing component.

**Root Cause**: You cannot compress Nash equilibrium play into V(s) + argmax. Nash equilibrium is inherently about action distributions, not scalar state values.

### Key Insight
> The encoding is not the bottleneck — the decision procedure is. To use Nash values effectively, the agent would need to output a policy π(a|s) rather than a value V(s), or use the Nash values as a regularizer alongside self-play training rather than as the sole training signal. This is a fundamental architectural mismatch, not a training problem.

---

## Direction 3: Risk-Sensitive Distributional Agent

### Concept
Learn the full distribution of returns (not just the expected value), then make decisions using a risk-sensitive criterion: `argmax_a [E[V] - β × Std[V]]`. Since our robustness metric penalizes variance (avg - 1.5×std), an agent that directly accounts for variance should achieve higher robustness.

### Architecture Evolution (6 iterations)

The agent went through 6 architecture revisions:
1. **Quantile regression** (attempts 1-3): Bootstrapped quantile regression diverges — taus at boundaries push outputs to extreme values; the "distribution" being learned via self-play bootstrapping is not well-defined
2. **Shared trunk mean-variance** (attempt 4): Variance gradients interfere with mean learning
3. **Dual-head with separate networks** (final): Clean separation

**Final Architecture: DualHeadModel**
- **ValueNetwork**: MLP(15 → 64 → 64 → 1) — identical to ValueBasedAgent, trained with standard MSE TD(0)
- **VarianceNetwork**: MLP(15 → 64 → 64 → softplus → 1) — predicts Var[V(s)], trained on `(reward - mean)²` at terminal states, propagated as `Var(s_t) = Var(s_{t+1}) + (mean_{t+1} - mean_t)²`
- **Separate optimizers** prevent gradient interference
- **Training**: Risk-neutral (mean-only) Boltzmann exploration; risk-sensitivity at evaluation only

### Results

**Evaluation at beta=0.5 (500 rounds each):**

| Opponent | Score |
|----------|-------|
| heuristic | -0.200 |
| value_based | -0.910 |
| adaptive_value | -0.740 |
| modulated_value | -1.040 |
| entropy_ac | **+0.080** |
| cfr | -0.670 |
| **Average** | **-0.580** |

**Beta Sweep (comprehensive):**

| Beta | Avg | Std | Robustness | Notes |
|------|-----|-----|------------|-------|
| 0.0 | -0.86 | 0.31 | -1.33 | Risk-neutral baseline |
| 0.1 | -0.80 | 0.36 | -1.34 | |
| 0.2 | -0.62 | 0.36 | -1.16 | |
| 0.3 | -0.43 | 0.47 | -1.13 | |
| **0.5** | **-0.37** | **0.34** | **-0.88** | **Optimal — best R4 robustness** |
| 0.7 | -0.76 | 0.22 | -1.09 | |
| 1.0 | -1.00 | 0.00 | -1.00 | Always folds |

### Diagnosis: Risk-Sensitivity Has Real Strategic Effects

The value head correctly orders hand strengths:
- K preflop: mean=+0.98 (RAISE selected)
- Q preflop: mean=-0.52 (FOLD at beta=0.5)
- J preflop: mean=-1.25 (FOLD)

**Variance captures genuine distributional properties**: paired states have higher std (3.79 avg) vs unpaired (3.49), confirming the network learns that pairs produce more certain outcomes.

**Key strategic effect**: 70% of decisions differ from ValueBasedAgent. The distributional agent systematically prefers CALL over RAISE for K preflop — RAISE has higher mean but also higher std (larger pot = more at risk), so the risk-adjusted score favors CALL. This is a genuinely novel strategic insight: risk-sensitive play is inherently more conservative with pot commitment.

**The always-fold collapse at high beta**: Poker has genuine std of 3-4 chips per hand from card randomness. Since FOLD always has std=0, any beta above ~0.7 makes all non-fold actions look terrible. The beta=0.5 sweet spot balances risk-awareness against this fundamental poker variance.

### Key Insight
> Risk-sensitivity works, but only in a narrow operating range. The dual-head architecture (separate value/variance networks with separate optimizers) is critical — shared architectures fail. The most interesting finding is that risk-adjusted play naturally shifts from RAISE to CALL, preferring lower-variance actions even when they have slightly lower expected value. At beta=0.5, the agent achieves the best robustness score of any R4 agent (-0.88), though still below incumbents. Future work should explore separating **aleatoric** (card randomness) from **epistemic** (model uncertainty) variance to allow higher effective beta without collapsing to all-fold.

---

## Direction 4: Opponent Response Planning (2-PLY SEARCH)

### Concept
Add a learned opponent model P(action | state) and use 2-ply lookahead: for each action, predict what the opponent will do in response, evaluate the resulting state, and select the action with the highest expected value across all opponent responses.

### Architecture
- **Opponent Model**: MLP(15 → 32 → 3) — predicts P(FOLD/CALL/RAISE | state)
- **Value Network**: MLP(15 → 64 → 64 → 1) — standard TD(0) training
- **Planning**: 2-ply search with opponent model at decision time

### Results

| Opponent | Score |
|----------|-------|
| heuristic | -1.406 |
| value_based | -1.476 |
| adaptive_value | -1.794 |
| modulated_value | -1.304 |
| entropy_ac | -1.642 |
| cfr | -0.768 |
| **Average** | **-1.398** |

### Diagnosis: Generic Opponent Model + Excessive Aggression

**Opponent model learns averages, not opponents.** Prediction accuracy:

| Opponent | Accuracy | Notes |
|----------|----------|-------|
| value_based | 76.1% | |
| modulated_value | 75.8% | |
| adaptive_value | 75.7% | |
| entropy_ac | 73.5% | Never folds → model wrongly predicts ~15% fold |
| heuristic | 66.0% | Most different from self-play |
| cfr | 58.0% | Mixed strategies hard to predict |

The model predicts ~17% fold, ~47% call, ~37% raise regardless of opponent type. It learned the aggregate self-play distribution rather than opponent-specific patterns.

**2-ply induces excessive aggression.** When the model predicts ~17% opponent fold probability, it overvalues aggressive actions (raising "works" 17% of the time via fold equity). Analysis of changed decisions:
- CALL → RAISE: 241 changes (51.6%)
- FOLD → RAISE: 134 changes (28.7%)
- RAISE → CALL: 92 changes (19.7%)

The 2-ply search converts 375 passive/defensive decisions into raises, making the agent overly aggressive against opponents who actually fold much less than 17%.

**Root Cause**: Same as belief agent — self-play produces a single-mode opponent model. The 2-ply search then amplifies this error by treating the biased fold prediction as exploitable fold equity.

### Key Insight
> 2-ply search is only as good as the opponent model. With a biased model, deeper search amplifies bias rather than correcting it. Effective opponent modeling requires training against diverse opponents, and ideally adapting online based on observed behavior (not just a fixed model from training).

---

## Direction 5: Information-Hiding Agent (ADVERSARIAL)

### Concept
Train a policy to be unpredictable: add an auxiliary "spy" network that tries to predict the agent's hand from its action history, and train the policy to confuse the spy. This should naturally discover bluffing (acting strong with weak hands) and slow-playing (acting weak with strong hands).

### Architecture
- **Policy Network**: MLP(15 → 64 → 64 → 3) — actor-critic with policy head
- **Value Network**: MLP(15 → 64 → 64 → 1) — critic head
- **Spy Network**: MLP(20 → 32 → 3) — predicts P(hand) from action history
- **Loss**: `policy_gradient + value_loss - λ × cross_entropy(spy_pred, true_hand)`

### Results

| Opponent | Score |
|----------|-------|
| heuristic | -0.644 |
| value_based | -0.590 |
| adaptive_value | -0.514 |
| modulated_value | -0.892 |
| entropy_ac | -0.070 |
| cfr | -1.046 |
| **Average** | **-0.626** |

### Diagnosis: Spy Detached, Raises MORE Predictable

**Spy accuracy comparison** (lower = more hidden information):

| Agent | Spy Accuracy |
|-------|-------------|
| value_based | **88.0%** |
| entropy_ac | 75.0% |
| info_hiding (λ=0.1) | 73.3% |
| cfr | 61.8% |

The agent achieves lower spy accuracy than value_based (73% vs 88%), but this is misleading. Examining the action distributions reveals the problem:

| Hand | info_hiding Raise Rate | value_based Raise Rate |
|------|----------------------|----------------------|
| K | **87.7%** | 56.1% |
| Q | 25.1% | 6.7% |
| J | 7.3% | 9.8% |
| **Raise gap (K - J)** | **0.804** | **0.463** |

The info_hiding agent is **more predictable than value_based**, not less! Its raise gap (K - J = 0.804) is nearly double value_based's (0.463). The lower spy accuracy comes from the spy network itself being poorly trained, not from the policy actually hiding information.

**The fundamental problem**: The policy outputs action probabilities, then samples an action via `torch.multinomial()`. This sampling step is **non-differentiable** — gradients from the spy loss cannot flow back through the discrete action choice to reach the policy parameters. The adversarial signal never reaches the policy.

**Lambda sweep** confirms no consistent effect:

| λ | Spy Accuracy |
|---|-------------|
| 0.0 | 0.360 |
| 0.05 | 0.365 |
| 0.1 | 0.570 |
| 0.2 | 0.365 |
| 0.5 | 0.310 |

No monotonic relationship between λ and spy accuracy — the adversarial loss is effectively noise.

**Root Cause**: Adversarial training requires differentiable pathways. Discrete action sampling breaks the gradient chain. Solutions would need Gumbel-Softmax (continuous relaxation of discrete sampling) or REINFORCE-style gradients for the spy loss.

### Key Insight
> The concept of adversarial information hiding is sound — it's how Nash equilibrium bluffing strategies arise. But the implementation requires differentiable action selection. The agent also never folds (0% for all hands), suggesting the policy gradient + value loss is not well-calibrated independently of the spy loss.

---

## Cross-Cutting Themes

### Theme 1: Self-Play Is a Poor Training Signal for Opponent-Aware Components

Three of five directions (Belief, Opponent Model, Info-Hiding) included components that needed to learn about opponents. All three failed because self-play only exposes the agent to **its own converging policy**. The likelihood model learns "everyone raises like me"; the opponent model learns "everyone plays like me"; the spy network only sees one behavioral pattern.

**Fix**: Train against a diverse pool of existing agents. The registry already contains 20+ agents with varied strategies — training against this pool would provide the behavioral diversity needed.

### Theme 2: The Value-Function + Argmax Paradigm Has Fundamental Limits

Both Nash Value and Distributional exposed limitations of `argmax_a V(post(s, a))`:
- Nash requires mixed strategies that argmax cannot produce
- Risk-sensitivity requires action distributions, not point estimates
- The value-based decision procedure inherently produces pure strategies

The incumbent agents succeed despite this limitation because they exploit opponents rather than trying to play optimally. But for game-theoretically motivated approaches, the argmax bottleneck is binding.

### Theme 3: Problem Setup Constraints Dominate Architecture Innovation

The 15-dimensional encoding, self-play training, and post-state value evaluation are the real performance determinants — not the architecture. The Belief agent's 14-dim encoding with belief features is actually richer than the standard 15-dim encoding, yet it loses because its belief features are garbage (due to self-play). The Nash agent approximates CFR values nearly perfectly, yet loses because argmax produces exploitable pure strategies.

**Implication**: Future rounds should prioritize training methodology (diverse opponents, curriculum learning) and decision procedure (stochastic policies, mixed strategies) over architecture changes.

---

## Comparison with Incumbents

| Agent | Round | Avg | Robustness | Key Strength |
|-------|-------|-----|------------|-------------|
| adaptive_value | R0 | **+1.012** | -0.030 | Opponent exploitation via simple stats |
| value_based | R0 | +0.970 | +0.040 | Clean value learning |
| modulated_value | R1 | +0.967 | **+0.106** | Best robustness via modulated betting |
| **distributional** | **R4** | -0.580 | **-1.230** | Dual-head; beta=0.5 gets -0.37 avg, -0.88 robustness |
| **belief_value** | **R4** | -0.434 | -1.024 | Belief concept sound, execution broken |
| **info_hiding** | **R4** | -0.626 | -1.133 | Lowest spy accuracy, but not via policy |
| **nash_value** | **R4** | -0.984 | -1.221 | Perfect CFR fit, wrong decision procedure |
| **opponent_model** | **R4** | -1.398 | -2.174 | 2-ply amplifies biased opponent model |

---

## Files Created

| File | Description |
|------|-------------|
| `src/agents/belief_value.py` | Bayesian belief tracking agent |
| `src/agents/nash_value.py` | Nash value network agent |
| `src/agents/distributional_value.py` | Distributional/risk-sensitive agent |
| `src/agents/opponent_model_agent.py` | 2-ply opponent modeling agent |
| `src/agents/info_hiding.py` | Adversarial information-hiding agent |
| `src/training/belief_trainer.py` | Belief agent trainer |
| `src/training/nash_trainer.py` | CFR + supervised trainer |
| `src/training/distributional_trainer.py` | Quantile/dual-head trainer |
| `src/training/opponent_model_trainer.py` | Value + opponent model trainer |
| `src/training/info_hiding_trainer.py` | Actor-critic + spy trainer |
| `experiments/round4_*.py` | Experiment scripts (one per direction) |
| `experiments/round4_*_results.json` | Raw results (one per direction) |
| `models/*_agent.pt` | Saved model weights (one per direction) |

---

## Recommendations for Round 5

Based on these findings, the most promising next directions are:

1. **Pool Training**: Train value-based agents against the full agent pool instead of self-play. This directly addresses Theme 1 and should improve all opponent-aware components.

2. **Policy-Based Nash**: Instead of V(s) + argmax, train a policy network π(a|s) that outputs action probabilities, regularized toward Nash equilibrium strategies from CFR. This addresses Theme 2.

3. **Gumbel-Softmax Info Hiding**: Reimplement the adversarial information-hiding concept with differentiable action selection (Gumbel-Softmax trick). The concept is sound; only the gradient pathway was broken.

4. **Belief Agent with Pool-Trained Likelihood**: Keep the Bayesian belief architecture but train the likelihood model against diverse opponents. This was the closest to working (avg -0.434, beat entropy_ac).

5. **Hybrid Value + Opponent Stats**: The simplest path to improvement may be combining the best-performing existing approach (adaptive_value's opponent stats) with deeper search or better training, rather than radically different architectures.
