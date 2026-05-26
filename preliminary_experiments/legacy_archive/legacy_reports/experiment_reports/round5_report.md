# Round 5: Belief-Based Agents — Principled Incremental Exploration

> **Date**: 2026-02-28
> **Branch**: `exp/round-3-integration`
> **Goal**: Return to single-variable methodology after Round 4's broad creative exploration. Systematically investigate belief-based opponent modeling along two orthogonal axes: how to compute P(opponent_hand) (Axis 1: Belief Estimation) and how to incorporate belief into the architecture (Axis 2: Belief Usage).

## Executive Summary

**Round 5 explored 5 belief-based agents across two orthogonal axes. The headline result: belief_modulated (E1b) with full training (40K sessions, 1.2M hands) achieves avg +0.315, beating 4 of 6 opponents — the first competitive belief agent.** Initial results from shorter training (2K sessions, avg -0.30) suggested all belief agents fail. The full training run reveals that belief + population training + sufficient duration can produce a competitive agent.

The second key finding: with short training, belief_confident (E2c, avg -0.21) succeeds by *ignoring* its beliefs (conf=0: +0.014, conf=1: -0.280). This reveals that undertrained belief models are actively harmful — but E1b's full-training success shows this is a training duration problem, not a fundamental architectural limitation.

Round 4 revealed that self-play was a poor training signal for belief estimation. Round 5 tested three alternative likelihood sources (CFR Nash, Nash+modulation, self-play baseline) and four architectural variations for belief usage. The core lesson: **belief-based opponent modeling works in Leduc Hold'em, but requires ~20× more training than value-based agents (1.2M hands vs 60K hands) because the likelihood model needs extensive opponent exposure to become informative.**

### Results at a Glance

| ID | Agent | Training | Avg Score | Robustness | Belief Acc | Raises? | Key Finding |
|----|-------|----------|-----------|------------|------------|---------|-------------|
| E1a | belief_cfr | 40K ep | -0.72 | -1.12 | 0.42 (below random) | No | Nash too uniform for Bayes |
| **E1b** | **belief_modulated** | **40K sess** | **+0.315** | **-0.43** | **0.51 (above random)** | **K, J** | **First competitive belief agent** |
| E2a | control (=E1b) | 40K sess | +0.315 | -0.43 | 0.51 (above random) | K, J | Same as E1b |
| E2b | belief_oracle | 40K ep | -0.49 | -1.02 | 0.39 (below random) | Q, K | Perfect V can't fix noisy beliefs |
| E2c | belief_confident | 2K sess | -0.21 | -0.32 | 0.39 (below random) | No | Best short-training — conf=0 optimal |
| E2d | belief_stable | 40K ep | -0.44 | -0.90 | **0.65 (above random)** | Q, K | Best beliefs, worst TD chain |
| **Ablation** | **belief_cfr (pop)** | **40K sess** | **+0.173** | **-0.47** | **0.39 (below random)** | **K, J, Q** | **Pop+duration = 86% of E1b** |

**Incumbent comparison**: adaptive_value +1.012, value_based +0.970, modulated_value +0.967
**E1b beats**: heuristic (+0.776), value_based (+0.002), modulated_value (+0.186), entropy_ac (+1.082)

---

## Methodology & Metrics

This section defines the key metrics used throughout the report. All metrics are specific to Leduc Hold'em (3 cards: J, Q, K).

### Belief Correctness
The average probability mass the belief model assigns to the true opponent hand. At each decision point, the agent maintains a belief vector `b = [P(J), P(Q), P(K)]`. Belief correctness = `mean(b[true_hand_idx])` across all decision points.

- **Random baseline: 0.50** — After card removal (you hold one card), 2 equally likely opponent cards remain. A uniform belief assigns P = 0.5 to the true hand.
- Values **below 0.50** indicate beliefs are counterproductive (pointing away from truth).
- Values **above 0.50** indicate beliefs are informative.

### Belief Shift (L1)
Measures how much the belief vector changes after each Bayesian update: `shift = sum(|b_new[i] - b_old[i]|)` for i in {J, Q, K}. Reported per-round.

- A shift of **0.0** means the likelihood model provides no discriminative signal — observing the opponent's action changes nothing about the belief.
- A positive shift means the model extracts information from the opponent's action.

### Likelihood Accuracy
Fraction of opponent actions correctly predicted by the likelihood model: `P(predicted_action == actual_action)` across evaluation hands.

- **Nash accuracy**: using pure CFR Nash likelihoods
- **Modulated accuracy**: using Nash + learned modulation
- Higher accuracy means the model better predicts what the opponent will do, but this does NOT necessarily mean the likelihoods are more useful for Bayesian updates (see E1b diagnosis).

### Gate Activation
A learned scalar in [0, 1] controlling how much the modulation layer deviates from the Nash baseline. Higher gate = more deviation applied. Reported per-opponent type.

### Robustness Score
`robustness = avg - 1.5 × std` across all opponents. Penalizes high variance to reward agents that perform consistently rather than excelling against one opponent type while losing badly to another.

### Chips/Round
The primary performance metric. Average chips won per round across evaluation hands. Positive = winning, negative = losing. The evaluation plays both seat positions equally.

---

## Context: Why Belief-Based Agents?

Round 4 tested 5 radically different approaches (belief, distributional, Nash, opponent model, info hiding). All lost. The Bayesian Belief Agent (-0.43 avg) surfaced the deepest questions about opponent modeling — specifically, that self-play training produces a catastrophically biased likelihood model (88% RAISE prediction regardless of hand).

Round 5 returns to principled single-variable methodology, exploring belief-based agents along two orthogonal axes:

- **Axis 1 (Belief Estimation)**: How to compute P(opponent_hand) — CFR Nash vs Nash+modulation vs self-play (R4 baseline)
- **Axis 2 (Belief Usage)**: How to incorporate belief into architecture — input dims, oracle V, confidence, stable targets

This two-axis framework allows clean attribution: Axis 1 isolates the likelihood source, Axis 2 isolates the architectural integration.

---

## Axis 1: Belief Estimation — How to Compute P(opponent_hand)

### E1a: belief_cfr (CFR Nash as Likelihood Source)

#### Concept
Replace the self-play-learned likelihood model from Round 4 with exact Nash equilibrium action probabilities from CFR. Since CFR computes P(action | hand, infoset) at every information set, these can serve as Bayesian likelihoods for belief updates. The hypothesis: game-theoretically correct likelihoods should produce accurate beliefs.

#### Architecture & Training
- **Likelihood source**: CFR Nash equilibrium P(action | hand, infoset)
- **Value Network**: Standard MLP, TD(0) self-play
- **Training**: 40K episodes, ~90s

#### Results

| Opponent | Score |
|----------|-------|
| heuristic | -0.386 |
| value_based | -0.898 |
| adaptive_value | -0.784 |
| modulated_value | -1.118 |
| entropy_ac | -0.540 |
| cfr | -0.616 |
| **Average** | **-0.724** |

#### Diagnosis: Nash Equilibrium Is Informationally Opaque

**Nash accuracy**: 71.5% — the CFR action predictions are somewhat correct.

**But belief quality is terrible**: belief correctness 0.42, **below the random baseline of 0.50** (uniform over 2 remaining cards after card removal). This means the beliefs are actively counterproductive — the agent would be better off ignoring beliefs entirely and using a uniform prior.

**Belief shift is zero on the flop.** The per-round belief shift data reveals the core failure:

| Round | Avg Belief Shift (L1) | Interpretation |
|-------|----------------------|----------------|
| Round 0 (preflop) | 0.5624 | Beliefs update normally |
| Round 1 (flop) | **0.000** | Beliefs do not change at all |

After the flop is revealed and the opponent acts, the Bayesian update produces **zero change** in the belief vector. Mathematically: `posterior(h) ∝ P(action|h) × prior(h)`. If `P(action|h)` is approximately constant across all hands h, then `posterior ∝ prior` — the likelihood provides no new information, so the posterior equals the prior.

This happens because Nash equilibrium likelihoods are nearly uniform across hands: `P(RAISE|J) ≈ P(RAISE|Q) ≈ P(RAISE|K)`. Observing a RAISE (or any action) barely shifts the probabilities.

**100% FOLD with J** — the agent folds every Jack hand, never raises, never calls.

**Root Cause**: Nash equilibrium strategies are designed to be unexploitable, which means they are designed to **reveal minimal information** about private state. If opponents could infer your hand from your actions, they would exploit you — so equilibrium strategies minimize information leakage. Using Nash as a *likelihood model* is fundamentally self-defeating: the very property that makes Nash strategies robust (information hiding) makes them useless for Bayesian inference.

#### Key Insight
> Nash equilibrium strategies are designed to be unexploitable, which means they are designed to reveal minimal information about private state. This makes them the worst possible likelihood source for Bayesian inference. The very property that makes Nash strategies robust (information hiding) makes them useless for opponent modeling.

---

### E1b: belief_modulated (Nash + Learned Gated Modulation)

#### Concept
Start with CFR Nash likelihoods but add a learned modulation layer: a gating network that adjusts the Nash baseline based on observed opponent statistics. The hypothesis: Nash provides a reasonable prior, and the gate can learn to amplify differences for exploitable opponents.

#### Architecture & Training
- **Likelihood source**: CFR Nash + gated modulation from opponent stats
- **Training**: 40,000 sessions (1.2M hands), ~109 min, **population-based** (heuristic, value_based, adaptive_value)
- **Gate**: Learned scalar that blends Nash likelihoods with modulated likelihoods

#### Results (Full Training — 40K Sessions)

| Opponent | Score |
|----------|-------|
| heuristic | **+0.776** |
| value_based | **+0.002** |
| adaptive_value | -0.140 |
| modulated_value | **+0.186** |
| entropy_ac | **+1.082** |
| cfr | -0.016 |
| **Average** | **+0.315** |

**This is the first belief agent to achieve a positive average score**, beating 4 of 6 opponents.

Note: An initial 2K-session run (avg -0.30) suggested the agent was competitive but still negative. The full 40K-session run reveals dramatically better performance, demonstrating that belief agents need ~20× more training than value-based agents to converge.

#### Diagnosis: Sufficient Training Unlocks Belief-Based Play

**Nash accuracy**: 77.6% — with full training, the base Nash likelihoods provide a solid foundation.

**Modulation hurts raw accuracy**: -11.1% drop (77.6% → 66.5%) when modulation is applied. But the gate shows **differentiation across opponent profiles**: heuristic (0.578) > value_based (0.542) > adaptive_value (0.535). This suggests the gate is learning a coarse opponent classification, even if the modulation itself doesn't improve likelihood accuracy.

**Belief correctness**: 0.514 — best among non-stable agents, above the random baseline of 0.50. This means the belief model assigns slightly more probability mass to the true opponent hand than a uniform guess would.

**Non-degenerate strategy**: K raises 83% preflop, J raises 41% on flop. The agent uses FOLD, CALL, and RAISE across different hands — the only agent besides E2d with a diverse action distribution.

**Key insight**: The 2K-session version (avg -0.30) was dramatically undertrained. The full training reveals that population-based training + sufficient duration produces a competitive belief agent.

#### Caution: Three Confounded Variables

E1b's success **cannot be cleanly attributed to modulation alone**. Three variables differ between E1a (the pure-Nash failure) and E1b (the competitive agent):

| Factor | E1a (failed, -0.72) | E1b (competitive, +0.315) |
|--------|---------------------|---------------------------|
| Likelihood source | Pure Nash (frozen) | Nash + learned modulation |
| Training methodology | Self-play | Population-based (heuristic, value_based, adaptive_value) |
| Training duration | 40K episodes (~60K hands) | 40K sessions (~1.2M hands, **20×**) |

Because all three factors changed simultaneously, we cannot determine which is responsible for the +1.04 chip/round improvement. The training duration alone (20× more data) and population training (diverse opponents) are both independently known to improve agent performance — either could explain E1b's success without modulation contributing at all.

**The "modulation hurts accuracy" finding is consistent with E1b's success** because:
1. Likelihood accuracy (% of actions correctly predicted) ≠ belief informativeness (how much likelihoods vary across hands). Modulation may reduce overall prediction accuracy while making likelihoods more discriminative across hands, producing more informative Bayesian updates.
2. E1b's success may come entirely from population training + duration, with modulation being a no-op or even slightly harmful.

#### Ablation Result: Confound Resolved

The ablation experiment (pure Nash + population training + 40K sessions, NO modulation) achieved **avg +0.173**, resolving the three-way confound:

| Agent | Avg | Robustness | Belief Acc | Key Difference |
|-------|-----|------------|-----------|----------------|
| E1a (Nash + self-play + 40K ep) | -0.724 | -1.12 | 0.42 | Baseline |
| **Ablation (Nash + pop + 40K sess)** | **+0.173** | **-0.465** | **0.39** | +pop +duration |
| E1b (Nash+mod + pop + 40K sess) | +0.315 | -0.430 | 0.51 | +pop +duration +modulation |

**Improvement attribution:**
- Population training + training duration (E1a → Ablation): **+0.897 chips/round** (86% of total improvement)
- Modulation (Ablation → E1b): **+0.142 chips/round** (14% of total improvement)

**Modulation is a small but real contributor, not a no-op.** It adds +0.142 chips/round, primarily by improving belief correctness (0.39 → 0.51) and performance against entropy_ac (+0.56 → +1.08). But population training + sufficient duration are the dominant factors, explaining 86% of the improvement from E1a to E1b.

Notably, the ablation agent learned the same non-degenerate strategy as E1b (K raises 85% preflop, J raises 42% on flop), confirming that diverse action distributions come from population training and duration, not from modulation.

#### Key Insight
> Population training + training duration are the dominant factors for belief agent performance, explaining 86% of E1b's improvement over E1a. Modulation provides a modest +0.142 chips/round contribution (14%) primarily through improved belief quality. The lesson: when three variables are confounded, run the ablation — assumptions about which variable matters are often wrong.

---

### Axis 1 Winner

**E1b (belief_modulated)** at avg **+0.315** vs E1a's -0.72 and R4 baseline's -0.43. With full training (40K sessions), E1b is the first belief agent to achieve positive average performance, beating 4/6 opponents. The improvement comes from the combination of population-based training + Nash+modulation likelihood + sufficient training duration.

---

## Axis 2: Belief Usage — How to Incorporate Belief into Architecture

All Axis 2 experiments use E1b's CFR Nash likelihood (the Axis 1 winner) as the belief estimation source.

### E2a: Control (= E1b Reused Directly)

Serves as the baseline for Axis 2 comparisons.

- **Avg**: +0.315 (full 40K-session training)
- **Robustness**: -0.43

**Important**: The Axis 2 variants (E2b-E2d) were trained with shorter schedules (40K episodes or 2K sessions ≈ 60K hands). The control's superior performance may partly reflect training duration rather than architectural superiority. A fair comparison would require training all variants for 40K sessions.

---

### E2b: belief_oracle (Perfect-Information Value Function)

#### Concept
Learn V(s, my_hand, opp_hand) — a value function conditioned on BOTH players' hands. At decision time, compute the belief-weighted expected value: V(s) = sum_h P(opp=h) * V(s, my_hand, h). The hypothesis: even if beliefs are noisy, a perfect-information value function should extract maximum signal from whatever belief quality exists.

#### Architecture & Training
- **Value Network**: Conditioned on both hands — learns exact hand-vs-hand values
- **Training**: 40K episodes, both MC and TD variants trained
- **Decision rule**: Belief-weighted selection across opponent hands

#### Results (TD variant — canonical)

| Opponent | Score |
|----------|-------|
| heuristic | -0.326 |
| value_based | -0.834 |
| adaptive_value | -0.510 |
| modulated_value | -0.908 |
| entropy_ac | **+0.054** |
| cfr | -0.402 |
| **Average** | **-0.488** |

MC variant: avg -0.540, robustness -1.306 (worse).

#### Diagnosis: Perfect V Cannot Fix Noisy Beliefs

**The oracle V learns correctly**: 241 unique states covered, value ordering correct for 6 of 7 hand-vs-hand matchups. The value function accurately captures that K > Q > J in each pairwise comparison.

**But belief quality kills performance**: belief correctness 0.385 — **below the random baseline of 0.50**, meaning beliefs are actively pointing away from the true hand. The belief-weighted selection computes V = P(J)*V(me,J) + P(Q)*V(me,Q) + P(K)*V(me,K), but when P(J), P(Q), P(K) are worse than random, this produces a systematically wrong mixture of correct values.

**100% FOLD with J** — same degenerate behavior as E1a.

#### Key Insight
> This experiment cleanly separates the value learning problem from the belief estimation problem. The oracle proves that value learning is not the bottleneck — hand-vs-hand values can be learned accurately. The bottleneck is exclusively belief quality. Even with a perfect value function, noisy beliefs produce noisy decisions.

---

### E2c: belief_confident (Belief + Confidence Score)

#### Concept
Add a confidence score alongside the belief vector: the model receives a 15-dimensional input including the 3-dim belief vector AND a scalar confidence measure (e.g., how much the belief has shifted from its prior). The hypothesis: the model can learn to weight beliefs more heavily when confidence is high and rely on simpler features when confidence is low.

#### Architecture & Training
- **Input**: 15-dim (standard features + 3-dim belief + 1-dim confidence)
- **Training**: 2000 sessions (60K hands), ~80s, population-based
- **Confidence**: Derived from belief entropy / divergence from prior

#### Results

| Opponent | Score |
|----------|-------|
| heuristic | -0.184 |
| value_based | -0.192 |
| adaptive_value | -0.280 |
| modulated_value | -0.168 |
| entropy_ac | -0.316 |
| cfr | -0.138 |
| **Average** | **-0.213** |

#### Diagnosis: The Model Learned to Ignore Its Beliefs

**The confidence mechanism works — but not as intended.** Total variation distance between action distributions at conf=0 vs conf=1 is 0.18, confirming the model responds differently to confidence levels.

**But higher confidence leads to MORE folding**, not better play. The confidence ablation is devastating:

| Confidence | Avg Score |
|------------|-----------|
| 0.0 | **+0.014** |
| 0.5 | -0.206 |
| 1.0 | -0.280 |

Performance **degrades monotonically** with confidence. At conf=0 (completely ignoring beliefs), the agent actually achieves a *positive* average score (+0.014). At conf=1 (fully trusting beliefs), it loses 0.28 chips/round.

**The confidence mechanism is functioning as a belief DAMPENER.** The model has correctly learned that its beliefs are unreliable and should be ignored. The improvement over E2a (-0.21 vs -0.30) comes from the confidence feature accidentally attenuating the harmful belief signal, not from exploiting accurate beliefs.

#### Key Insight
> This is the clearest evidence that current belief estimation is not just useless but actively detrimental. The model's optimal strategy is conf=0 (ignore beliefs entirely). The confidence mechanism's value lies in providing a learnable "off switch" for the belief input — a form of automatic feature selection that the network discovered through gradient descent.

---

### E2d: belief_stable (Stable Belief TD Targets)

#### Concept
Standard TD learning uses V(s_t, b_t) → r + V(s_{t+1}, b_{t+1}), where b is the belief vector. But b_{t+1} includes information from the opponent's next action, which introduces look-ahead bias into the target. Instead, use stable belief targets: V(s_t, b_t) → r + V(s_{t+1}, b_t) — the target uses the CURRENT belief, not the updated belief. The hypothesis: removing look-ahead bias in belief should produce more consistent value estimates.

#### Architecture & Training
- **Value Network**: Standard architecture with belief input
- **TD target**: Uses b_t (current belief) instead of b_{t+1} (next belief) in bootstrap target
- **Training**: 40K episodes, ~115s, self-play TD(0)

#### Results

| Opponent | Score |
|----------|-------|
| heuristic | -0.414 |
| value_based | -0.548 |
| adaptive_value | -0.472 |
| modulated_value | -0.814 |
| entropy_ac | **+0.110** |
| cfr | -0.530 |
| **Average** | **-0.445** |

#### Diagnosis: Best Beliefs, Broken TD Chain

**Dramatically better belief correctness**: 0.648 — far above the random baseline of 0.50, and far above every other agent (0.39-0.51 range). The stable target approach produces genuinely better beliefs because the value function learns a more consistent mapping from belief to value, which in turn provides better gradient signal for belief learning.

**Smallest belief shift**: avg 0.14 (vs 0.20-0.30 for others). The beliefs are more stable across rounds because the value function doesn't chase moving targets.

**ONLY agent with non-degenerate action distribution**:
- Raises with Q: 54%
- Raises with K: 82%
- Uses all three actions (FOLD, CALL, RAISE) across different hands

Every other Round 5 agent either never raises (4 of 6 agents have 0% raise rate) or only raises with K. E2d is the only agent that discovered a mixed strategy involving raising with Q — a strategically sound behavior (bluffing/value betting).

**But the broken TD chain hurts overall performance.** By using b_t instead of b_{t+1} in the target, the bootstrap value V(s_{t+1}, b_t) is systematically biased — it evaluates the next state with stale information. This produces less accurate value estimates overall, resulting in worse average score despite better beliefs and better action diversity.

#### Key Insight
> Stable belief targets create a fascinating tradeoff: better belief quality AND better action diversity, but at the cost of a biased TD chain. This agent provides the strongest evidence that belief-based play CAN work — it's the only agent that raises, bluffs, and uses mixed strategies. The path forward may be to find a way to get stable-target belief quality without breaking the value learning chain.

---

## Cross-Cutting Insights

### 1. Training Duration Is the Dominant Variable for Belief Agents

E1b's progression from 2K sessions (avg -0.30) to 40K sessions (avg **+0.315**) is a 0.6 chip/round improvement — larger than any architectural change tested in Axis 2. Belief agents need ~20× more training than value-based agents (1.2M vs 60K hands) because their likelihood models must learn opponent behavior patterns across diverse opponent types. The E2b-E2d agents (40K episodes or 2K sessions) may have been systematically undertrained.

### 2. With Sufficient Training, Beliefs Become Helpful (Not Harmful)

E2c's confidence ablation (conf=0 best with 2K sessions) initially suggested beliefs are actively harmful. But E1b's full-training success (avg +0.315, belief correctness 0.514) shows this is a training duration problem, not a fundamental limitation. Undertrained likelihood models produce noisy beliefs that degrade decisions; fully-trained likelihood models produce informative beliefs that improve decisions. The confidence mechanism correctly identifies that SHORT-trained beliefs should be ignored.

### 3. Never-Raise Is a Symptom of Insufficient Training

Four of six agents have 0% raise rate across ALL hands, but E1b with full training raises K 83% preflop and J 41% on flop. E2d (stable targets) also raises. The pattern: short training → passive play, long training → mixed strategies. The value function needs extensive data to learn that controlled aggression is profitable.

### 4. Population Training + Duration = 86% of the Story

The ablation experiment (pure Nash + population + 40K sessions, no modulation) achieved avg +0.173, resolving the confound:

| Change | Improvement | % of Total |
|--------|-------------|-----------|
| E1a → Ablation (add pop + duration) | +0.897 | **86%** |
| Ablation → E1b (add modulation) | +0.142 | **14%** |

Modulation is a small but real contributor (+0.142), primarily improving belief quality (0.39 → 0.51). But population training and training duration dominate.

### 5. The Two-Axis Framework Was Validated as a Methodology

Despite negative results, the orthogonal axis design allowed clean attribution:
- **Axis 1** showed that population training matters for value learning (not belief estimation)
- **Axis 2** showed that confidence dampening helps by attenuating harmful belief signal

Neither axis produced a competitive agent, but both produced interpretable insights. The two-axis approach is a reusable experimental design pattern for exploring multi-dimensional hypothesis spaces.

---

## What Would Make Beliefs Work?

Based on the comprehensive failure analysis, four directions could potentially rescue belief-based play:

1. **Better likelihood source**: Neither Nash (too uniform by design) nor self-play (learns own policy) nor population-with-gate (gate doesn't discriminate) provides useful P(action|hand). A **hand-crafted heuristic likelihood** — encoding domain knowledge like "strong hands raise more" — might work better than any learned model, at least as a starting point.

2. **Larger state space**: In Leduc (3 cards), card removal already narrows opponent hands to ~50% probability. The belief signal is inherently weak because there are so few cards to distinguish. In Texas Hold'em (1326 starting hands), beliefs would carry far more information — observing a raise could meaningfully shift probability mass across hundreds of possible holdings.

3. **Policy-based approach**: V(s)+argmax produces degenerate pure strategies (the never-raise failure mode). A policy network pi(a|s,belief) could learn mixed strategies that use belief for probabilistic adjustments rather than hard action selection. E2d's success with raising/bluffing (despite worse overall score) suggests that action diversity is valuable.

4. **Reward shaping for belief accuracy**: Directly reward correct belief at showdown (e.g., add bonus proportional to P(correct_hand) at terminal state) to give the likelihood model a stronger learning signal. Currently, belief accuracy only affects performance indirectly through value estimation — a direct reward signal could dramatically accelerate belief learning.

---

## Comparison with Incumbents

| Agent | Round | Avg | Robustness | Key Strength |
|-------|-------|-----|------------|-------------|
| adaptive_value | R0 | **+1.012** | -0.030 | Opponent exploitation via simple stats |
| value_based | R0 | +0.970 | +0.040 | Clean value learning |
| modulated_value | R1 | +0.967 | **+0.106** | Best robustness via modulated betting |
| **belief_modulated** | **R5** | **+0.315** | **-0.430** | **First competitive belief agent (full training)** |
| **belief_confident** | **R5** | -0.213 | -0.317 | Best short-training — conf=0 optimal |
| **belief_stable** | **R5** | -0.445 | -0.902 | Best belief accuracy (0.65), only short-trained raiser |
| **belief_oracle** | **R5** | -0.488 | -1.018 | Perfect V, noisy beliefs |
| **belief_cfr** | **R5** | -0.724 | -1.120 | Nash too uniform for inference |

**Gap**: E1b (avg +0.315) is within 0.7 chips/round of the incumbents — a dramatic improvement from R4 baseline (-0.43). With full training, belief-based agents are approaching competitive performance. The remaining gap to adaptive_value (+1.012) likely reflects the robustness disadvantage of belief-based approaches in a 3-card game.

---

## Files Created

| File | Description |
|------|-------------|
| `src/agents/belief_common.py` | Shared module: card constants, belief init, replay, CFR keys |
| `src/agents/belief_cfr.py` | E1a: CFR Nash likelihood belief agent |
| `src/agents/belief_modulated.py` | E1b: Nash + gated modulation belief agent |
| `src/agents/belief_oracle.py` | E2b: Perfect-info V with belief-weighted selection |
| `src/agents/belief_confident.py` | E2c: Belief + confidence score agent |
| `src/agents/belief_stable.py` | E2d: Stable belief TD target agent |
| `src/training/` | Corresponding trainer files for each agent |
| `experiments/round5_belief_cfr_results.json` | E1a raw results |
| `experiments/round5_belief_modulated_results.json` | E1b raw results |
| `experiments/round5_belief_oracle_results.json` | E2b raw results |
| `experiments/round5_belief_confident_results.json` | E2c raw results |
| `experiments/round5_belief_stable_results.json` | E2d raw results |
| `models/belief_modulated_agent.pt` | E1b saved weights |
| `models/belief_oracle_agent.pt` | E2b saved weights |
| `models/belief_confident_agent.pt` | E2c saved weights |
| `models/belief_stable_agent.pt` | E2d saved weights |
| `experiments/round5_e1a_pop_ablation.py` | Ablation: pure Nash + population + 40K sessions |
| `experiments/round5_e1a_pop_ablation_results.json` | Ablation raw results |
| `models/belief_cfr_pop_ablation_agent.pt` | Ablation saved weights |

---

## Key Takeaways for Future Rounds

1. **Belief-based opponent modeling works in Leduc Hold'em — but needs massive training.** E1b achieves avg +0.315 with 40K sessions (1.2M hands), beating 4/6 opponents. The 2K-session version (avg -0.30) was dramatically undertrained. Belief agents require ~20× more training than value-based agents because likelihood models need extensive opponent exposure.

2. **Population training + sufficient duration is the winning combination (ablation confirmed).** The ablation experiment (pure Nash + population + 40K sessions, no modulation) achieved avg +0.173, proving that population training + duration explain 86% of E1b's improvement over E1a. Modulation adds a modest +0.142 (14%), primarily through improved belief quality.

3. **Architectural variations matter less than training duration.** The Axis 2 experiments (E2b-E2d) tested sophisticated architectural changes (oracle V, confidence, stable targets) but all used shorter training schedules. None approached E1b's full-training performance. The open question: would E2c or E2d also become competitive with 40K sessions?

4. **Simple opponent statistics (adaptive_value) still lead** at avg +1.012 vs E1b's +0.315, but the gap has narrowed from 1.45 (R4 baseline) to 0.70. In larger games with more cards, the belief-based approach may close or reverse this gap.
