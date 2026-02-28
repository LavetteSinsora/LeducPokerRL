# PokerRL Research Overview

> A journey through 4 rounds of agent development for Leduc Hold'em, from hand-crafted heuristics through the discovery that protecting a strong model from corruption beats algorithmic innovation, to creative explorations of belief tracking, game theory, and adversarial training.

## The Evolution Story

This project began with a simple question: can reinforcement learning produce a poker agent that outperforms a hand-crafted baseline in Leduc Hold'em? The answer turned out to be yes -- but the path to getting there revealed far more about RL's failure modes than its successes.

**Round 0** established the baselines. A rule-based HeuristicAgent used pot odds and hand strength to make decisions. A TD(0) ValueBasedAgent learned to estimate state values through self-play. A REINFORCE PolicyGradientAgent tried direct policy optimization. And a tabular CFR+ agent solved for the Nash equilibrium as a game-theoretic reference.

**Round 1** explored five single-aspect modifications. The results were sobering: only two of the five new agents (adaptive_value and value_based) outperformed the heuristic. The others -- aux_value, actor_critic, history_value, and decay_adaptive -- all failed, each for a distinct reason. The big winner was adaptive_value, which added just 4 opponent statistic features to the observation space and immediately became the top-ranked agent. This taught the first critical lesson: *opponent modeling matters more than algorithmic sophistication*.

**Round 2** scaled up to 20,000 training episodes and introduced a proper robustness metric (avg - 1.5 x std) to penalize inconsistency. Five new agents were tested. Entropy-regularized actor-critic (entropy_ac) proved that mixed strategies prevent policy collapse. But population training (pop_adaptive), target networks (target_value), and feature concatenation (adaptive_history) all underperformed their parents, revealing that techniques designed for stationary environments can backfire in adversarial self-play.

**Round 3** was hypothesis-driven, with each agent designed to test a specific diagnostic finding from Round 2. The headline result: **modulated_value** became the first agent to beat every opponent in the tournament, achieving the highest robustness score ever (+0.199). But the mechanism was not what we designed -- instead of exploiting opponent models through gated modulation, the architecture simply prevented a strong pretrained base from being corrupted during training. Meanwhile, curriculum training with both-player chains proved catastrophically wrong, TD(3) turned out to be pure Monte Carlo in Leduc's short chains, and extended training confirmed that more data helps but architecture matters more.

**Round 4** broke from incremental optimization to test 5 radically different architectures: Bayesian belief tracking, Nash value distillation, distributional RL, 2-ply opponent modeling, and adversarial information hiding. All 5 failed to beat the incumbents, but each failure was deeply informative. The belief agent proved that self-play training produces a degenerate likelihood model (88% RAISE accuracy, 3% FOLD). The Nash value agent achieved near-perfect approximation of CFR values (99.2% variance explained) but argmax on Nash values produces exploitable pure strategies -- Nash requires mixed strategies. The distributional agent discovered genuine risk-sensitivity effects (RAISE-to-CALL shifts) but poker's inherent variance (std 3-4 chips) creates a collapse-to-fold boundary at beta > 0.5. The info-hiding agent showed that adversarial training through discrete sampling is non-differentiable -- gradients never reach the policy. Three cross-cutting themes emerged: self-play is a poor training signal for opponent-aware components, the V(s)+argmax paradigm has fundamental limits for game-theoretic play, and problem setup constraints (training methodology, decision procedure) dominate architecture innovation.

The overarching narrative: **in a domain as small and adversarial as Leduc Hold'em, the most powerful approach is to learn a strong value function via TD(0) self-play, augment it with opponent statistics, and then structurally protect it from degradation. Radically different paradigms (game theory, distributional RL, adversarial training) are promising but require solving fundamental implementation challenges before they can compete.**

## Key Concepts

### What Does the Value Function Represent?

The value function is **on-policy value** -- V(s) estimates the expected cumulative reward when following the agent's current Boltzmann policy from state s onward. This is NOT optimal-policy value (as in Q-learning). Training bootstraps V(s_t) toward V(s_{t+1}) using the agent's own value estimates, and terminal states use actual game rewards (+/- chips won or lost). Since both players are the same agent in self-play, the value represents "my expected payoff when I play my current policy against myself."

This matters because the value function co-evolves with the policy. As the policy improves, the value function must track a moving target. This co-evolution is why TD(0) bootstrapping works so well -- it provides a smooth, slowly-changing learning signal that naturally tracks policy changes.

### Self-Play Dynamics

In self-play training, the agent plays against a copy of itself. This creates a non-stationary environment: as the agent improves, its opponent simultaneously improves. This has profound implications:

- **Generalization is surprising**: An agent trained only against itself somehow learns to beat agents it has never seen. This works because self-play forces the agent to learn robust strategies that don't exploit specific opponent weaknesses.
- **Adaptive features enable transfer**: The adaptive_value agent's opponent statistics (fold rate, raise rate, call rate, aggression) allow it to read opponent behavior at test time, even against novel opponents. This bridges the gap between training (vs self) and evaluation (vs diverse opponents).
- **Non-stationarity breaks standard techniques**: Target networks, which work well in stationary Atari environments, create stale references in self-play. The frozen target quickly becomes outdated as both the agent and its training opponent evolve.

### TD Bootstrapping vs Monte Carlo

TD(0) bootstraps: V(s_t) is updated toward r_t + gamma * V(s_{t+1}), using the network's own estimate of the next state's value. Monte Carlo uses the actual cumulative return from the episode. In Leduc Hold'em, where games last only 1-3 player decisions:

- **TD(0) provides temporal smoothing**: The bootstrap target V(s_{t+1}) changes slowly as the network updates, creating a damped learning signal. This is critical in self-play where the "true" value of a state shifts with every policy update.
- **N-step returns converge to MC**: With n=3 and mean chain length 1.3, 99.1% of transitions use terminal reward. N-step is functionally identical to Monte Carlo.
- **MC produces noisy gradients**: Terminal rewards depend on the opponent's cards, the board card, and the full action sequence -- all sources of variance that TD(0) bootstrapping smooths away.

The lesson: for short-horizon games, the bias of TD(0) is negligible but its variance reduction is substantial.

### The Exploration-Exploitation Tradeoff

Pure actor-critic agents collapse to deterministic policies that are easily exploitable in poker. A player who always raises with strong hands and folds with weak hands is trivially predictable. Entropy regularization adds a bonus term H(pi) to the loss, rewarding policy diversity:

- **Without entropy**: actor_critic learns a near-deterministic policy, loses to any opponent that reads its patterns (-0.568 avg).
- **With entropy**: entropy_ac maintains mixed strategies, bluffing occasionally and varying bet sizes (+0.664 avg, best actor-critic variant).

In poker, randomization is not a weakness -- it is a fundamental requirement for unexploitable play. The Nash equilibrium itself is a mixed strategy.

### On-Policy vs Off-Policy in Population Training

When training against a pool of frozen opponents (population training), a critical question arises: whose experience do you learn from?

- **P0-only chains (on-policy)**: Train only on states visited and rewards received by the learning agent (Player 0). This is on-policy -- the training data matches the agent's own policy.
- **Both-player chains (off-policy)**: Also train on the frozen opponent's (Player 1's) experience. In zero-sum games, P1's rewards are the negative of P0's. This introduces destructive gradient interference: the value network receives "this state is worth +X" from P0 and "similar state is worth -X" from P1.

The curriculum agent's catastrophic failure (-0.822 avg, worst in tournament) was traced entirely to both-player chain collection. The P0-only ablation improved performance by +1.2 chips/round -- the single largest ablation effect in the entire project.

## Agent Lineage

### Round 0: Baselines

The project started with four fundamentally different approaches to poker:

- **Heuristic** -- A hand-crafted rule-based agent using pot odds and hand strength. Serves as the "beat this" baseline for all RL agents. Surprisingly resilient: rank 4 across all three rounds.
- **Value Based** -- TD(0) value network with Boltzmann exploration. The foundational RL agent and ancestor of most successful descendants. Consistently ranks in the top 2-3.
- **Actor-Critic** -- REINFORCE with a value baseline. Ancestor of the entropy_ac line. Struggles without entropy regularization.
- **CFR+** -- Tabular counterfactual regret minimization. Converges to Nash equilibrium but operates in a fundamentally different paradigm from the RL agents (not included in RL tournaments).

### Round 1: Foundations (3K training episodes)

Five single-aspect modifications tested the effect of individual design choices:

- **Adaptive Value** (+0.99 avg) -- Added 4 opponent statistics to the observation space. Immediately became the #1 agent, proving that opponent modeling is the single most impactful feature.
- **Aux Value** (-0.12 avg) -- Added a Bellman consistency auxiliary loss. The max operator in the auxiliary target caused value overestimation, and the extra loss stole gradient budget from the primary TD objective.
- **History Value** (-0.74 avg) -- Doubled input dimensionality (15 to 31) with action history encoding but kept the same 64-unit network. Underfitting from capacity mismatch.
- **Decay Adaptive** (-0.79 avg) -- EMA-weighted opponent stats. In self-play, there is no shifting opponent to track, so recency bias provides no advantage over uniform averaging.

### Round 2: Refinements (20K training episodes)

Five more agents, now with proper training budgets and a robustness metric:

- **Entropy AC** (+0.664 avg, -0.712 rob) -- Entropy bonus prevents policy collapse. Highest single-matchup score in the tournament (+2.16 vs decay_adaptive) but high variance hurts robustness.
- **N-Step Value** (+0.072 avg, -1.007 rob) -- N-step returns (n=3) provided noisier gradients than TD(0) in Leduc's short chains.
- **Adaptive History** (+0.134 avg, -0.917 rob) -- Merged adaptive and history features (35-dim). Feature concatenation without architectural support for feature selection.
- **Target Value** (-0.296 avg, -1.537 rob) -- Frozen target network created stale bootstrap targets in non-stationary self-play.
- **Pop Adaptive** (-0.557 avg, -1.771 rob) -- Population training against weak opponents taught exploitation over robustness.

### Round 3: Advanced Techniques (agent-specific configs)

Five diagnosis-informed agents, each testing a specific hypothesis:

- **Modulated Value** (+0.967 avg, +0.199 rob) -- Frozen pretrained base with gated modulation. First agent to beat every opponent. Success came from structural protection of the pretrained model, not opponent exploitation.
- **Extended Adaptive** (+0.329 avg, -0.406 rob) -- 3x training budget null hypothesis control. More training helps monotonically (no overfitting) but cannot match architectural innovation.
- **TD Variant** (-0.216 avg, -1.729 rob) -- Calibrated TD(3). Functionally pure Monte Carlo in Leduc (99.1% terminal targets). Noisy gradients + slow learning rate = non-convergence.
- **Pruned History** (-0.691 avg, -1.781 rob) -- Pruned action history features with 31-dim input. Insufficient training budget (667 sessions) and undersized network (64 hidden units).
- **Curriculum** (-0.822 avg, -1.976 rob) -- Block training with rehearsal buffer. Both-player chain collection caused catastrophic gradient interference. Worst agent in the entire tournament.

### Round 4: Creative Exploration (5 new lineages)

Five radically different approaches, breaking from incremental optimization:

- **Belief Value** (-0.434 avg, -1.024 rob) -- Bayesian belief tracking over opponent hand. Likelihood model trained on self-play produces degenerate predictions (88% RAISE accuracy, 3% FOLD). Beliefs barely shift from card-removal prior. NEW LINEAGE.
- **Distributional Value** (-0.580 avg, -1.230 rob) -- Dual-head mean+variance network with risk-adjusted decisions. Beta=0.5 optimal (avg -0.37, rob -0.88). Genuine RAISE-to-CALL strategic shift, but poker's inherent variance creates collapse-to-fold at beta > 0.5.
- **Nash Value** (-0.984 avg, -1.221 rob) -- Neural net trained on exact CFR Nash values (99.2% variance explained). Argmax on Nash values produces exploitable pure strategies; Nash requires mixed strategies. NEW LINEAGE.
- **Opponent Model** (-1.398 avg, -2.174 rob) -- 2-ply lookahead with learned opponent model. Model learns generic self-play distribution; search amplifies biased fold prediction into excessive aggression.
- **Info Hiding** (-0.626 avg, -1.133 rob) -- Adversarial spy network for deceptive play. Gradient cannot flow through discrete sampling (torch.multinomial). Policy is MORE predictable than value_based despite lower spy accuracy.

## Cross-Round Comparison

All 22 RL agents + heuristic, sorted by robustness score (avg - 1.5 x std). Round 4 agents evaluated against 6 core opponents (heuristic, value_based, adaptive_value, modulated_value, entropy_ac, cfr).

| Rank | Agent | Round | Avg Score | Std | Robustness | Tier |
|------|-------|-------|-----------|-----|------------|------|
| 1 | Modulated Value | R3 | +0.967 | 0.512 | **+0.199** | Gold |
| 2 | Value Based | R0 | +0.970 | 0.614 | **+0.049** | Gold |
| 3 | Adaptive Value | R0 | +1.012 | 0.695 | -0.030 | Green |
| 4 | Heuristic | R0 | +0.604 | 0.594 | -0.287 | Green |
| 5 | Extended Adaptive | R3 | +0.329 | 0.490 | -0.406 | Green |
| 6 | Entropy AC | R2 | +0.664 | 0.917 | -0.712 | Green |
| 7 | Adaptive History | R2 | +0.134 | 0.701 | -0.917 | Green |
| 8 | Aux Value | R0 | -0.211 | 0.517 | -0.986 | Gray |
| 9 | N-Step Value | R2 | +0.072 | 0.720 | -1.007 | Green |
| 10 | Belief Value | R4 | -0.434 | 0.393 | -1.024 | Gray |
| 11 | Info Hiding | R4 | -0.626 | 0.338 | -1.133 | Red |
| 12 | Actor-Critic | R1 | -0.568 | 0.419 | -1.197 | Red |
| 13 | Nash Value | R4 | -0.984 | 0.158 | -1.221 | Red |
| 14 | Distributional | R4 | -0.580 | 0.433 | -1.230 | Red |
| 15 | Target Value | R2 | -0.296 | 0.828 | -1.537 | Gray |
| 16 | Decay Adaptive | R1 | -0.881 | 0.518 | -1.657 | Red |
| 17 | TD Variant | R3 | -0.216 | 1.008 | -1.729 | Gray |
| 18 | Pop Adaptive | R2 | -0.557 | 0.810 | -1.771 | Red |
| 19 | Pruned History | R3 | -0.691 | 0.727 | -1.781 | Red |
| 20 | History Value | R1 | -0.512 | 0.873 | -1.822 | Red |
| 21 | Curriculum | R3 | -0.822 | 0.770 | -1.976 | Red |
| 22 | Opponent Model | R4 | -1.398 | 0.517 | -2.174 | Red |

Notable patterns:
- The top 3 agents are all value-based with TD(0) at their core (modulated_value inherits a frozen value_based network). No Round 4 agent breaks into the top 9.
- The only two agents with positive robustness are both gold-tier: one from Round 0 (value_based) and one from Round 3 (modulated_value).
- The heuristic baseline (rank 4) outperforms 18 of 21 RL agents, underscoring how hard it is to improve on simple strategies in small games.
- Round 4's best agent (belief_value, rank 10) sits in the gray tier -- better than the worst R1-R3 agents but nowhere near the incumbents.
- Round 4's worst agent (opponent_model, rank 22) replaces curriculum as the overall worst, due to 2-ply search amplifying biased opponent predictions.

## Hypothesis Matrix

| Hypothesis | Agent | Round | Result | Evidence |
|-----------|-------|-------|--------|----------|
| Opponent stats improve generalization | adaptive_value | R1 | **Confirmed** | +4 features yielded #1 in R1, #3 overall. Opponent modeling bridges self-play to diverse evaluation. |
| Auxiliary losses improve value estimation | aux_value | R1 | **Rejected** | Max-operator causes overestimation; aux loss steals gradient budget. 4x higher prediction error than value_based. |
| Action history provides strategic information | history_value | R1 | **Rejected** | Doubled input space without capacity increase. In Leduc's short games, history is largely redundant with pot sizes. |
| EMA adapts to shifting opponents | decay_adaptive | R1 | **Rejected** | In self-play, no shifting opponent exists. Recency bias provides no advantage over uniform averaging. |
| Value baseline reduces REINFORCE variance | actor_critic | R1 | **Partially supported** | Baseline helps but cannot overcome fundamental credit assignment problem with terminal-only rewards. |
| Entropy prevents policy collapse | entropy_ac | R2 | **Confirmed** | Mixed strategies essential for poker. Entropy bonus yields highest single-matchup scores but high variance hurts robustness. |
| N-step returns reduce TD bias | nstep_value | R2 | **Rejected** | n=3 is pure MC in Leduc (99.1% terminal). Removes beneficial bootstrapping smoothing. |
| Target networks stabilize TD learning | target_value | R2 | **Rejected** | Frozen targets become stale in non-stationary self-play. Designed for stationary MDPs. |
| Diverse training opponents improve robustness | pop_adaptive | R2 | **Rejected** | Weak opponent pool teaches exploitation. Opponent rotation disrupts stat accumulation. |
| Feature concatenation synergizes | adaptive_history | R2 | **Partially supported** | Positive average but below both parents. Needs architecture that learns feature importance. |
| Calibrated n-step/MC matches TD(0) | td_variant | R3 | **Rejected** | n=3 is functionally pure MC. TD(0) bootstrapping provides critical temporal smoothing for self-play. |
| Gated modulation outperforms concatenation | modulated_value | R3 | **Confirmed (unexpected)** | Succeeds via structural protection of pretrained base, not opponent exploitation. Gate suppresses itself. |
| Block training + rehearsal beats self-play | curriculum | R3 | **Rejected** | Both-player chains cause catastrophic gradient interference. P0-only ablation improves by +1.2 chips/round. |
| Pruned features + training recovers benefit | pruned_history | R3 | **Partially supported** | More training helps (+0.78) but still needs wider network. Feature pruning itself is benign. |
| 3x training is sufficient improvement | extended_adaptive | R3 | **Rejected** | More training helps monotonically (no overfitting), but architecture matters more. modulated_value vastly outperforms with 1/3 the budget. |
| Bayesian belief tracking improves decisions | belief_value | R4 | **Rejected (implementation)** | Concept sound, but self-play likelihood model is degenerate (88% RAISE, 3% FOLD). Beliefs barely shift. Needs diverse training opponents. |
| Nash values produce robust play | nash_value | R4 | **Rejected (fundamental)** | Near-perfect neural fit (99.2% R^2) but argmax collapses mixed strategies to exploitable pure strategies. Nash requires policy output, not V+argmax. |
| Risk-sensitivity improves robustness | distributional_value | R4 | **Partially supported** | Genuine strategic effect (RAISE->CALL shift). Best R4 robustness at beta=0.5 (-0.88). But poker's inherent variance creates fold-collapse at beta>0.5. |
| 2-ply search improves decisions | opponent_model | R4 | **Rejected** | Self-play opponent model learns generic distribution. 2-ply amplifies biased fold prediction into excessive aggression (+375 passive->aggressive changes). |
| Adversarial training discovers bluffing | info_hiding | R4 | **Rejected (implementation)** | Gradient cannot flow through discrete sampling (torch.multinomial). Policy is MORE predictable (raise gap 0.804 vs 0.463). Needs differentiable relaxation. |

## Cumulative Key Insights

1. **TD(0) bootstrapping is optimal for Leduc's short chains (1-3 steps).** The game's brevity makes n-step and MC returns functionally identical, and both produce noisier gradients than TD(0). The bootstrap target provides implicit temporal smoothing that stabilizes self-play training.

2. **Opponent modeling via running statistics enables generalization beyond self-play.** Four features (fold rate, raise rate, call rate, aggression) let the agent read opponent behavior at evaluation time, even against agents it has never trained against. This is the single most impactful observation-space modification.

3. **Population training must preserve on-policy guarantees (P0-only chains).** In zero-sum games against frozen opponents, training on both players' experience introduces opposite-sign rewards that create destructive gradient interference. This was the largest single failure mode discovered in the project.

4. **Transfer learning via frozen base + gated modulation provides "first, do no harm" safety.** The modulated_value architecture is structurally incapable of catastrophically deviating from its strong pretrained base. The gate learned to suppress its own influence, making the architecture a sophisticated way to do "mostly freeze the model."

5. **Entropy regularization is essential for actor-critic to prevent policy collapse.** In poker, deterministic policies are trivially exploitable. Entropy bonus forces mixed strategies that approximate the randomization required for unexploitable play.

6. **More data beats more complexity, but architecture beats more data.** Extended training (3x budget) monotonically improves performance with no overfitting, but modulated_value outperforms it dramatically with 1/3 the training budget. The hierarchy is: good architecture > more training > more features.

7. **Self-play non-stationarity makes target networks counterproductive.** The frozen target quickly becomes stale as the agent's policy and its self-play opponent co-evolve. Stabilization techniques designed for stationary MDPs backfire in adversarial settings.

8. **Auxiliary objectives steal gradient budget from the primary loss.** Even when the auxiliary target is unbiased, the extra loss forces the shared network to learn conflicting value functions, degrading primary task performance.

9. **Architecture changes must be capacity-matched to input dimensionality changes.** Doubling the input from 15 to 31 dimensions while keeping 64 hidden units creates underfitting. When expanding observations, network width must scale proportionally.

10. **Self-play is a poor training signal for opponent-aware components.** Three Round 4 directions (belief, opponent model, info-hiding) included components that needed to learn about opponents. All failed because self-play only exposes the agent to its own converging policy. Likelihood models, opponent models, and spy networks all learn "everyone plays like me" -- accurate for self-play but useless for generalization.

11. **V(s) + argmax cannot implement mixed strategies.** Nash equilibrium play requires randomization (e.g., raise 60%, call 40%). The argmax decision procedure always selects the single best action, collapsing mixed strategies to pure (exploitable) ones. Game-theoretically motivated agents need policy output pi(a|s), not scalar values.

12. **Risk-sensitivity in poker has a narrow viable range.** Poker's inherent outcome variance (std 3-4 chips from random deals) dominates strategic variance. Any risk penalty strong enough to matter will also make FOLD (the only zero-variance action) dominate. The viable beta range is approximately [0.3, 0.6] -- a fragile operating point.

13. **Adversarial training requires differentiable pathways.** Gradient-based adversarial objectives (like the info-hiding spy loss) cannot influence parameters upstream of non-differentiable operations (like discrete action sampling). Continuous relaxations (Gumbel-Softmax) or policy-gradient estimators are needed.

## What's Next

See [Belief Framework](belief_framework.md) for the theoretical analysis motivating Round 5's systematic investigation of belief-based agents -- covering the two-axis framework (estimation vs usage), documented assumptions, and open research questions.
