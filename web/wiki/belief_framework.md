# Belief-Based Agent Framework

> A systematic analysis of how to build agents that reason about opponent hands -- decomposing the problem into belief estimation and belief usage, with documented assumptions and open questions.

---

## 1. Two-Axis Framework

Belief-based agents must solve two orthogonal problems:

**Axis 1 -- Belief Estimation**: How do we compute P(H_opp | observations)?

| Method | Description | Status |
|--------|-------------|--------|
| **Self-play likelihood** | Train P(a \| H, state) from self-play data. Model learns own policy, not opponent's. | Tested (Round 4). Failed: 88% RAISE accuracy, 3% FOLD accuracy. |
| **CFR Nash policy** | Use the converged CFR strategy as P(a \| H, state). Game-theoretically calibrated but assumes rational opponent. | Untested. Available from `CFRAgent`. |
| **Nash + modulated** | Start from CFR policy, apply state-agnostic modulation based on opponent stats: `P_mod(a \| H) = P_nash(a \| H, state) * (1 + delta(opp_stats))`. | Untested. Combines rational baseline with observed deviation. |

**Axis 2 -- Belief Usage**: How do we incorporate the belief into the agent's architecture?

| Method | Description | Status |
|--------|-------------|--------|
| **Extra input dims** | Concatenate belief vector [P(J), P(Q), P(K)] to observation. Let the value network learn how to use it. | Tested (Round 4). Failed due to garbage beliefs, not architecture. |
| **Oracle V + weighting** | Train V(s, my_hand, opp_hand) with full information. At decision time, compute E[V] = sum_h belief(h) * V(s, my_hand, h). | Untested. Requires training state coverage of evaluation distribution. |
| **Belief + confidence** | Add belief vector AND a scalar confidence (e.g., n_sessions or belief entropy) so the network can discount uncertain beliefs. | Untested. Addresses the "how much to trust" problem. |
| **Stable belief** | Use belief_t (pre-action) instead of belief_{t+1} (post-action) in TD targets. Avoids noisy belief updates in bootstrap targets. | Untested. Trades TD chain correctness for target stability. |

**Why these axes are orthogonal**: The estimation method determines the *quality* of the belief vector -- how accurately it reflects the true opponent hand distribution. The usage method determines the *architecture* for incorporating that belief into decisions. A better likelihood model improves all usage methods equally; a better architecture improves performance for any likelihood model. They can be tested independently by holding one axis fixed while varying the other.

### Experiment Space (2x4 grid)

| | Extra Input | Oracle V + Weight | Belief + Confidence | Stable Belief |
|---|---|---|---|---|
| **Self-play likelihood** | Round 4 (failed) | -- | -- | -- |
| **CFR Nash policy** | -- | -- | -- | -- |
| **Nash + modulated** | -- | -- | -- | -- |

Only one cell has been explored. The remaining 11 combinations represent the systematic investigation space for Round 5.

---

## 2. Documented Assumptions

### Assumption 1: State-Agnostic Modulation

When modulating the Nash likelihood based on opponent stats (fold_rate, raise_rate, etc.), the modulation factor is applied uniformly across all game states:

```
P_mod(a | H, state) = P_nash(a | H, state) * (1 + delta(opp_stats))
```

The deviation `delta` depends only on aggregate opponent statistics, not on the current game state.

**Justification**: With ~30 hands of data per session, state-conditional modulation has far too high variance. Leduc Hold'em has ~180 unique states; reliable per-state statistics would require ~300+ samples per state (at minimum ~10 observations per state-action pair), meaning ~54,000+ hands total. At 30 hands/session, that is 1,800 sessions -- well beyond what a single evaluation provides.

**Limitation**: Cannot capture opponents who play differently in different game states. For example, an opponent who is aggressive in high-pot situations but passive in low-pot situations would have their behavior averaged into a single modulation factor.

**Important nuance**: The base Nash policy IS state-conditional -- CFR produces different action probabilities for different information sets. Only the deviation from Nash is state-agnostic. So the combined model captures "rational play (state-conditional) + constant personality shift (state-agnostic)." This is a reasonable first-order approximation: most behavioral differences between players are consistent tendencies (e.g., "plays tight" or "plays loose") rather than state-specific deviations.

### Assumption 2: Macro Features as Lossy Compression

Opponent statistics (fold_rate, raise_rate, fold_to_raise, confidence) are fixed-dimensional summaries of the opponent's full action history. This compression discards:

1. **Ordering of actions**: A player who folds early then raises late looks identical to one who raises early then folds late.
2. **Temporal dynamics**: Style drift within a session (e.g., tilting after a bad beat) is averaged away.
3. **State-conditional behavior**: Different play in different game states is collapsed to aggregate rates.

The most information-preserving alternative is the raw action history -- a variable-length sequence of (state, action) pairs. Processing this requires sequence models (RNN, Transformer) that can handle variable-length inputs, adding significant architectural complexity.

**When macro features are sufficient**: When the opponent's strategy is approximately stationary and state-independent (i.e., consistent tendencies). This holds reasonably well for the trained RL agents in our pool, which converge to near-fixed policies.

**When richer representations are needed**: Against adaptive opponents who change strategy mid-session, or against sophisticated opponents with state-conditional play. Future work should investigate the empirical boundary between these regimes.

### Assumption 3: TD(0) with Belief Coupling

The value network is trained via TD(0) on augmented states (s, belief):

```
V(s_t, b_t) <- r_t + gamma * V(s_{t+1}, b_{t+1})
```

The belief changes between t and t+1 because the opponent acts (updating our belief via Bayes' rule). This is theoretically sound during training: beliefs are calibrated for the self-play opponent, so b_{t+1} is a valid posterior given the training policy.

**Generalization risk**: At evaluation time against a different opponent, beliefs may become miscalibrated. If the likelihood model was trained on self-play but evaluated against a heuristic agent, the posterior b_{t+1} may be systematically wrong. The value network has never seen these miscalibrated belief states during training, so its outputs in those regions are unreliable.

**The "stable belief" alternative**: Use b_t (pre-opponent-action belief) instead of b_{t+1} in the TD target:

```
V(s_t, b_t) <- r_t + gamma * V(s_{t+1}, b_t)   [stable variant]
```

This avoids incorporating noisy/miscalibrated belief updates into the bootstrap target. However, it breaks the TD chain: the value at s_{t+1} is evaluated with a stale belief, so the Bellman equation no longer holds exactly. The tradeoff is between target stability (stable belief) and theoretical correctness (standard TD).

### Assumption 4: Training State Distribution Coverage

For the oracle V approach, where V(s, my_hand, opp_hand) is trained with full information and beliefs are applied at decision time via:

```
E[V] = sum_h belief(h) * V(s, my_hand, h)
```

the training trajectories must cover the state distribution encountered during evaluation. If the agent reaches states at evaluation time that were never visited during training, V(s, my_hand, h) is undefined in those regions.

**Valid in Leduc**: With only ~180 unique game states, self-play training easily covers the full state space. Even with 20K training episodes, each state is visited thousands of times.

**Would not hold in larger games**: Texas Hold'em has ~10^14 information sets. No amount of training covers the full state space, making the oracle V approach impractical without function approximation that generalizes across unseen states.

---

## 3. Open Research Questions

### Q1: Confidence-Gated Belief Trust

Can we give the value network a signal for how much to trust the current beliefs? Candidate confidence scores:

- **n_sessions**: Simple count of opponent actions observed. More data = more reliable beliefs.
- **Belief entropy**: H(belief) = -sum_h b(h) log b(h). Low entropy means beliefs are concentrated (confident); high entropy means uncertain. But low entropy could also mean confidently wrong.
- **KL from prior**: KL(belief || card_removal_prior). Large divergence means beliefs have moved far from the uninformative prior -- either because of strong evidence or because of miscalibrated updates.

The value network could learn to weight belief features by confidence: use beliefs heavily when confidence is high, fall back to a belief-agnostic strategy when confidence is low.

### Q2: Oracle V vs Direct Belief Input

Is V(s, my_hand, opp_hand) with belief-weighted expectation at decision time better than V(s, belief) as direct input?

- **Oracle V advantage**: The value function is trained with complete information (no belief noise). Beliefs only enter at decision time, so miscalibrated beliefs affect action selection but not value estimation.
- **Direct input advantage**: The value function can learn nonlinear interactions between game state and belief (e.g., "when I hold K and believe opponent has Q, this state is worth X"). Oracle V assumes linear belief aggregation.

### Q3: Raw Action Histories

Can variable-length opponent behavior data replace macro feature summaries? An RNN or Transformer processing the sequence [(state_1, action_1), (state_2, action_2), ...] preserves ordering, temporal dynamics, and state-conditional information that macro features discard.

**Practical challenge**: Variable-length inputs complicate batching and require architectural changes throughout the pipeline.

### Q4: Modulated Value Gate Anomalies

From Round 3's modulated_value diagnosis, several unexplained behaviors:

- **Modulation always negative**: The delta network produces mostly negative adjustments (~-0.16 preflop, ~-0.03 postflop pair). Why? Is the base systematically overestimating values?
- **Narrow gate range [0.39, 0.48]**: The gate never fully opens or closes. Is this a local optimum, or is the architecture unable to learn more expressive gating?
- **Gate decreases with confidence**: Opposite of design intent. The gate was supposed to trust modulation MORE with more data, but it trusts it LESS. Interpretation: the base is so strong that confident stats are a signal to suppress modulation, not enhance it. But is this truly optimal, or just what this training run converged to?

### Q5: Stable Belief TD Tradeoff

Does stripping belief-change value from TD targets improve robustness against novel opponents, or does breaking the TD chain hurt learning more than it helps?

**Key diagnostic**: Measure |b_{t+1} - b_t| (belief shift magnitude) and correlate with TD error magnitude. If large belief shifts correspond to large TD errors, stable belief could reduce target noise. If belief shifts are small (as in Round 4, where beliefs barely moved), the stable variant would have negligible effect.

---

## 4. Bayesian Update Derivation

### Notation

- **H**: Random variable for opponent's hand (values: J, Q, K)
- **h_t**: History of all actions and observations up to time t
- **a_t**: New opponent action observed at time t
- **belief_t(h)**: P(H = h | h_t), the current belief about the opponent's hand

### Goal

Compute P(H | h_t, a_t) from P(H | h_t) and P(a_t | H, h_t).

### Recursive Bayesian Update

By Bayes' theorem:

```
belief_{t+1}(h) = P(H = h | h_t, a_t)
                = P(a_t | H = h, h_t) * P(H = h | h_t) / P(a_t | h_t)
                = P(a_t | H = h, h_t) * belief_t(h) / Z
```

where the normalizing constant Z ensures the posterior sums to 1:

```
Z = sum_{h'} P(a_t | H = h', h_t) * belief_t(h')
```

### Batch Form

If the agent observes a sequence of opponent actions a_1, a_2, ..., a_n, the full posterior is:

```
P(H = h | all actions) = [prod_i P(a_i | H = h, h_i)] * P(H = h | card_removal) / Z
```

where the product runs over all observed actions and the prior P(H = h | card_removal) accounts for the known cards (the agent's own hand removes one card from the deck).

### Common Error in the Denominator

The normalizing constant Z must weight each likelihood by the **prior belief**, not use raw likelihoods:

```
CORRECT:   Z = P(a|J) * b(J) + P(a|Q) * b(Q) + P(a|K) * b(K)
INCORRECT: Z = P(a|J) + P(a|Q) + P(a|K)
```

The incorrect version treats all hands as equally likely regardless of prior information. With card removal (e.g., agent holds Q, so P(opp=Q) is reduced), the prior is NOT uniform, and ignoring it produces wrong posteriors.

### Connection to Opponent Policy

The likelihood term P(a_t | H = h, h_t) is exactly the opponent's policy:

```
P(a_t | H = h, h_t) = pi_opp(a_t | hand = h, state)
```

The likelihood model IS the opponent's policy. This means the quality of belief updates is entirely determined by how well we can model/approximate the opponent's decision-making.

### Belief Smoothing Variant

To reduce sensitivity to a single noisy likelihood estimate:

```
b_{t+1} = alpha * bayes_update(b_t, a_t) + (1 - alpha) * b_t
```

where alpha in [0, 1] controls update aggressiveness. At alpha = 1, this is standard Bayes. At alpha = 0, beliefs never change. Intermediate values provide robustness against miscalibrated likelihood models at the cost of slower convergence to the true posterior.

### Card Removal Initialization

The prior at the start of each hand accounts for the agent's known hand:

```
If agent holds Q:
  P(opp = J) = 1/2,  P(opp = Q) = 0,  P(opp = K) = 1/2

If agent holds J:
  P(opp = J) = 0,  P(opp = Q) = 1/2,  P(opp = K) = 1/2
```

In Leduc Hold'em with one card per rank, card removal is fully deterministic (the opponent cannot hold the same card). With multiple cards per rank, card removal adjusts probabilities proportionally.

---

## 5. Future Directions (Beyond Round 5)

### State-Conditional Modulation

With sufficient data (hundreds of samples per game state), the modulation factor could become state-dependent:

```
P_mod(a | H, state) = P_nash(a | H, state) * (1 + delta(opp_stats, state))
```

This would capture opponents who vary their play by game state -- e.g., tighter preflop but looser postflop. Requires either much longer evaluation sessions or a hierarchical model that shares statistical strength across similar states.

### Raw Action History via RNN/Transformer

Replace macro features with a sequence encoder that processes the full action history. A small LSTM or 2-layer Transformer could learn temporal patterns (style drift, tilt) and state-conditional behavior that aggregate statistics discard. The key design question is how to integrate the sequence encoding with the value network -- concatenation, attention, or gating.

### Policy-Based Nash

Round 4's Nash Value agent showed that V(s) + argmax collapses mixed strategies to pure (exploitable) strategies. The fix is to output a policy directly:

```
pi(a | s) instead of argmax_a V(post(s, a))
```

Train the policy network to match CFR's Nash equilibrium action probabilities via KL divergence, while also incorporating self-play value learning. This preserves the mixed-strategy structure that Nash equilibrium requires.

### Gumbel-Softmax for Differentiable Information Hiding

Round 4's info-hiding agent failed because `torch.multinomial()` is non-differentiable -- gradients from the spy network cannot flow back through discrete action sampling. The Gumbel-Softmax trick provides a continuous relaxation:

```
y = softmax((log(pi) + g) / tau)    where g ~ Gumbel(0, 1)
```

At low temperature tau, this approximates discrete sampling while remaining differentiable. The spy network's adversarial gradients can then directly influence the policy, enabling genuine information hiding.

### Epistemic Uncertainty

Bayesian neural networks or deep ensembles could provide uncertainty estimates over value predictions. This separates **aleatoric uncertainty** (irreducible card randomness) from **epistemic uncertainty** (model's own ignorance), enabling risk-sensitivity that targets only the reducible component -- addressing the distributional agent's collapse-to-fold problem at high beta.
