# Agent Family Tree

> **Experiment Protocol**: All architecture experiments MUST use `TournamentCheckpointer`
> from `agents/tournament_eval.py`. See `CLAUDE.md` for the full protocol and
> `paper/evaluation/meta/EVAL_CONFIG.json` for frozen evaluation parameters.

> **Start here for insights**: Skip to **Key Insights — Cumulative** (search `## Key Insights`)
> for the accumulated mechanistic understanding across all rounds. Read these before
> designing any new experiment.

This document tracks the lineage of all agents in the PokerRL project.
Each agent is an incremental change from its parent, changing exactly one aspect.

## Family Tree

```
BaseAgent (ABC)
├── HeuristicAgent .................. Rule-based baseline (hand-crafted strategy)
│
├── ValueBasedAgent ................ TD(0) value network, 15-dim obs, Boltzmann exploration
│   ├── AuxValueAgent .............. +Bellman consistency auxiliary loss (changes: training objective)
│   ├── HistoryValueAgent .......... +Action history encoding, 31-dim obs (changes: observation space)
│   ├── NStepValueAgent ............ +N-step returns instead of TD(0) (changes: TD target)     [Round 2]
│   ├── TargetValueAgent ........... +Frozen target network for stable TD (changes: bootstrap)  [Round 2]
│   ├── TDVariantAgent ............. +Calibrated n-step/MC comparison (changes: TD target)      [Round 3]
│   └── AdaptiveValueAgent ......... +Opponent stats, 19-dim obs (changes: observation space)
│       ├── DecayAdaptiveAgent ..... +EMA opponent stats (changes: stat accumulation method)
│       ├── PopAdaptiveAgent ....... +Diverse opponent pool training (changes: training opponents)  [Round 2]
│       ├── AdaptiveHistoryAgent ... +Action history combo, 35-dim (changes: observation space)     [Round 2]
│       ├── PrunedHistoryAgent ..... +Pruned history (12 feat), 31-dim (changes: obs space)        [Round 3]
│       ├── ModulatedValueAgent .... +Frozen base + gated modulation (changes: architecture) ★     [Round 3]
│       ├── CurriculumAgent ........ +Block training + rehearsal (changes: training opponents)      [Round 3]
│       └── ExtendedAdaptiveAgent .. +3× training budget (changes: training duration)               [Round 3]
│
├── PolicyGradientAgent ............ REINFORCE, 15-dim obs, categorical sampling
│   └── ActorCriticAgent ........... +Value baseline for variance reduction (changes: loss function)
│       └── EntropyACAgent ......... +Entropy regularization for mixed strategies (changes: loss)   [Round 2]
│
├── CFRAgent ....................... Tabular CFR+, game-theoretic Nash equilibrium
│
├── BeliefValueAgent .............. Bayesian belief tracking over opponent hand          [Round 4, NEW]
│   ├── BeliefCfrAgent ............ +CFR Nash as likelihood source (frozen)              [Round 5]
│   ├── BeliefModulatedAgent ...... +Nash + gated modulation from opponent stats         [Round 5]
│   ├── BeliefOracleAgent ......... +Perfect-info V + belief-weighted selection          [Round 5]
│   ├── BeliefConfidentAgent ...... +Confidence score (15-dim input)                    [Round 5]
│   └── BeliefStableAgent ......... +Stable belief TD targets (b_t not b_{t+1})         [Round 5]
├── NashValueAgent ................ Neural net trained on CFR Nash values                [Round 4, NEW]
├── DistributionalValueAgent ...... Dual-head mean+variance, risk-sensitive              [Round 4]
├── OpponentModelAgent ............ 2-ply search with learned opponent model             [Round 4]
└── InfoHidingAgent ............... Adversarial spy network for deceptive play           [Round 4]
```

## Agent Details

### Generation 0 — Baselines

| Agent | ID | Obs Dims | Algorithm | Key Feature |
|-------|----|----------|-----------|-------------|
| HeuristicAgent | `heuristic` | N/A | Rule-based | Hand-crafted strategy with pot odds |
| ValueBasedAgent | `value_based` | 15 | TD(0) | 1-step lookahead value estimation |
| PolicyGradientAgent | `policy_gradient` | 15 | REINFORCE | Direct policy optimization |
| CFRAgent | `cfr` | tabular | CFR+ | Nash equilibrium convergence |

### Generation 1 — Single-Aspect Changes

| Agent | ID | Parent | What Changed | Obs Dims |
|-------|----|--------|-------------|----------|
| AuxValueAgent | `aux_value` | value_based | Training: added Bellman consistency aux loss | 15 |
| AdaptiveValueAgent | `adaptive_value` | value_based | Obs: +4 opponent stat features | 19 |

### Generation 2 — Round 1 Experiments (3K episodes)

| Agent | ID | Parent | What Changed | Obs Dims |
|-------|----|--------|-------------|----------|
| ActorCriticAgent | `actor_critic` | policy_gradient | Loss: REINFORCE → REINFORCE + value baseline | 15 |
| HistoryValueAgent | `history_value` | value_based | Obs: +16 per-round action count features | 31 |
| DecayAdaptiveAgent | `decay_adaptive` | adaptive_value | Stats: uniform averaging → EMA | 19 |

### Generation 3 — Round 2 Experiments (20K episodes)

| Agent | ID | Parent | What Changed | Obs Dims |
|-------|----|--------|-------------|----------|
| NStepValueAgent | `nstep_value` | value_based | TD target: TD(0) → n-step returns (n=3) | 15 |
| EntropyACAgent | `entropy_ac` | actor_critic | Loss: +entropy bonus H(π) for mixed strategies | 15 |
| PopAdaptiveAgent | `pop_adaptive` | adaptive_value | Training: self-play → diverse opponent pool | 19 |
| AdaptiveHistoryAgent | `adaptive_history` | adaptive_value | Obs: +16 action history features, wider network | 35 |
| TargetValueAgent | `target_value` | value_based | Bootstrap: same network → frozen target network | 15 |

## Round 1 Evaluation Results (3K training episodes)

Agents ranked by average chips/round across all opponents (500 rounds/matchup):

| Rank | Agent | Avg Chips/Round | vs Heuristic |
|------|-------|----------------|-------------|
| 1 | adaptive_value | +0.99 | +0.68 |
| 2 | value_based | +0.91 | +0.44 |
| 3 | heuristic | +0.29 | — |
| 4 | aux_value | -0.12 | -0.33 |
| 5 | actor_critic | -0.54 | -0.71 |
| 6 | history_value | -0.74 | -1.28 |
| 7 | decay_adaptive | -0.79 | -0.52 |

### Head-to-Head Matrix (row agent's avg chips/round vs column agent)

|  | heuristic | value_based | adaptive | aux_value | actor_critic | history | decay_adapt |
|--|-----------|-------------|----------|-----------|-------------|---------|-------------|
| **heuristic** | — | -0.44 | -0.68 | +0.33 | +0.71 | +1.28 | +0.52 |
| **value_based** | +0.44 | — | -0.10 | +1.01 | +1.02 | +1.97 | +1.14 |
| **adaptive** | +0.68 | +0.10 | — | +1.27 | +0.86 | +1.63 | +1.43 |
| **aux_value** | -0.33 | -1.01 | -1.27 | — | +0.27 | +0.78 | +0.83 |
| **actor_critic** | -0.71 | -1.02 | -0.86 | -0.27 | — | -0.51 | +0.13 |
| **history** | -1.28 | -1.97 | -1.63 | -0.78 | +0.51 | — | +0.69 |
| **decay_adapt** | -0.52 | -1.14 | -1.43 | -0.83 | -0.13 | -0.69 | — |

## Round 2 Evaluation Results (20K training episodes)

12-agent round-robin tournament (500 rounds/matchup). Primary metric: **Robustness Score** = avg - 1.5 × std.

### Robustness Leaderboard

| Rank | Agent | Avg | Worst | Best | Std | Robustness |
|------|-------|-----|-------|------|-----|------------|
| 1 | adaptive_value | +1.06 | -0.18 | +1.94 | 0.63 | **+0.12** |
| 2 | value_based | +0.98 | +0.03 | +1.78 | 0.61 | **+0.06** |
| 3 | heuristic | +0.47 | -0.76 | +1.23 | 0.59 | -0.41 |
| 4 | **entropy_ac** | +0.94 | -0.37 | +2.16 | 0.99 | -0.54 |
| 5 | adaptive_history | +0.22 | -0.97 | +1.60 | 0.83 | -1.02 |
| 6 | nstep_value | +0.28 | -0.95 | +1.67 | 0.91 | -1.08 |
| 7 | actor_critic | -0.62 | -1.67 | +0.23 | 0.51 | -1.38 |
| 8 | pop_adaptive | -0.64 | -1.88 | +0.53 | 0.85 | -1.91 |
| 9 | decay_adaptive | -0.97 | -2.16 | -0.07 | 0.63 | -1.92 |
| 10 | target_value | -0.31 | -1.94 | +1.16 | 1.08 | -1.93 |
| 11 | history_value | -0.50 | -1.82 | +1.03 | 1.00 | -2.00 |
| 12 | aux_value | -0.92 | -2.12 | +0.27 | 0.85 | -2.20 |

### Full Head-to-Head Matrix

|  | heur | vb | adapt | aux | ac | hist | decay | nstep | ent_ac | pop | ah | target |
|--|------|----|-------|-----|----|------|-------|-------|--------|-----|----| -------|
| **heur** | — | -0.03 | -0.76 | +0.90 | +0.42 | +1.23 | +1.08 | -0.07 | +0.32 | +0.78 | +0.52 | +0.83 |
| **vb** | +0.03 | — | +0.18 | +1.70 | +1.04 | +1.61 | +1.09 | +0.95 | +0.37 | +1.78 | +0.65 | +1.40 |
| **adapt** | +0.76 | -0.18 | — | +1.25 | +1.01 | +1.82 | +1.34 | +0.87 | +0.30 | +1.57 | +0.97 | +1.94 |
| **aux** | -0.90 | -1.70 | -1.25 | — | +0.27 | +0.18 | +0.07 | -1.67 | -2.12 | -0.24 | -1.60 | -1.16 |
| **ac** | -0.42 | -1.04 | -1.01 | -0.27 | — | -0.61 | +0.23 | -0.59 | -1.67 | -0.13 | -0.48 | -0.79 |
| **hist** | -1.23 | -1.61 | -1.82 | -0.18 | +0.61 | — | +1.03 | -0.90 | -1.28 | +0.65 | +0.19 | -0.95 |
| **decay** | -1.08 | -1.09 | -1.34 | -0.07 | -0.23 | -1.03 | — | -1.56 | -2.16 | -0.14 | -1.08 | -0.91 |
| **nstep** | +0.07 | -0.95 | -0.87 | +1.67 | +0.59 | +0.90 | +1.56 | — | -0.57 | +0.54 | -0.39 | +0.55 |
| **ent_ac** | -0.32 | -0.37 | -0.30 | +2.12 | +1.67 | +1.28 | +2.16 | +0.57 | — | +1.88 | +0.44 | +1.23 |
| **pop** | -0.78 | -1.78 | -1.57 | +0.24 | +0.13 | -0.65 | +0.14 | -0.54 | -1.88 | — | -0.94 | +0.53 |
| **ah** | -0.52 | -0.65 | -0.97 | +1.60 | +0.48 | -0.19 | +1.08 | +0.39 | -0.44 | +0.94 | — | +0.70 |
| **target** | -0.83 | -1.40 | -1.94 | +1.16 | +0.79 | +0.95 | +0.91 | -0.55 | -1.23 | -0.53 | -0.70 | — |

## Diagnosis: Round 1 Failures

### Actor-Critic (Round 1 rank 5, -0.54 avg)
- **Problem**: REINFORCE inherently has high variance even with a baseline; 3000 episodes is insufficient for policy gradient convergence in self-play
- **Root cause**: The value baseline helps but can't compensate for the fundamental credit assignment problem — the agent only gets one reward signal per episode (terminal), applied to ALL actions

### History Value (Round 1 rank 6, -0.74 avg)
- **Problem**: Doubled the input space (15→31) without increasing network capacity, causing underfitting
- **Root cause**: In Leduc Hold'em, action sequences are very short (2-4 actions), and most of the strategic information is already captured by pot sizes and round number

### Decay Adaptive (Round 1 rank 7, -0.79 avg)
- **Problem**: Slightly worse than its parent (adaptive_value, rank 1)
- **Root cause**: In self-play training, both players are the same agent — there's no shifting opponent strategy to adapt to, so EMA's recency bias provides no advantage

## Diagnosis: Round 2 Results

### Entropy AC — Best New Agent (rank 4, +0.94 avg)
- **What worked**: Entropy bonus kept the policy diverse, preventing collapse to exploitable deterministic strategies
- **Strengths**: Highest single matchup score in the tournament (+2.16 vs decay_adaptive). Beat every weak agent by large margins
- **Weakness**: High variance (std=0.99). Strong against weak opponents but struggles vs top 2. Robustness penalized by inconsistency
- **Key insight**: Mixed strategies matter — entropy regularization is one of the few modifications that actually improved actor-critic performance

### N-Step Value (rank 6, +0.28 avg)
- **What happened**: Modest positive average but high variance. N-step returns didn't provide the expected bias reduction
- **Why**: In Leduc (2-6 steps), n=3 makes most transitions use terminal reward directly, but this also removes the bootstrapping smoothing effect that stabilizes TD(0)
- **Lesson**: The bias-variance tradeoff of n-step returns doesn't favor n>1 in very short games

### Adaptive History (rank 5, +0.22 avg)
- **What happened**: Positive average but below both parents. The feature combination didn't synergize
- **Why**: 35-dim input with 128-wide network trained for only 667 sessions (~20K hands). The wider network needed more training data to converge. The history features are redundant with pot-size features in short Leduc games
- **Lesson**: Feature concatenation alone isn't enough — need architecture that can learn which features matter

### Target Value (rank 10, -0.31 avg)
- **What happened**: Target network made things worse, not better
- **Why**: In self-play, both players' value estimates shift together. The frozen target network creates a stale reference that's systematically wrong. DQN's target network works when the environment is stationary — self-play is non-stationary
- **Lesson**: Stabilization techniques designed for stationary MDPs can backfire in adversarial multi-agent settings

### Pop Adaptive (rank 8, -0.64 avg)
- **What happened**: Diverse training opponents didn't translate to better evaluation
- **Why**: The opponent pool contained mostly weak agents (heuristic, pretrained value_based). Training against weak opponents taught the agent to exploit weaknesses rather than play robustly. Also, opponent rotation disrupted the session-based stat accumulation that makes adaptive_value strong
- **Lesson**: Population diversity only helps if the population includes strong, varied opponents

## Aux-Value Deep Diagnosis

Three experiments confirmed the max-operator overestimation hypothesis:

1. **Value drift**: aux_value's mean predicted values were consistently +0.066 higher than value_based's on a fixed probe set, growing monotonically over training
2. **Operator comparison**: Replacing max with mean in the aux target eliminated the bias (mean_V matched plain TD(0)), but still didn't improve performance
3. **Prediction error**: Trained aux_value had 4x higher absolute prediction error than value_based (7.19 vs 1.68)

**Conclusion**: The aux loss itself is the problem, not just the max operator. Even unbiased aux targets (mean, on-policy) fail because the extra loss steals gradient budget from the main TD loss and forces the shared network to learn two conflicting value functions (pre-action and post-action states)

### Generation 4 — Round 3 Experiments (agent-specific configs)

| Agent | ID | Parent | What Changed | Obs Dims |
|-------|----|--------|-------------|----------|
| TDVariantAgent | `td_variant` | value_based | TD target: TD(0) → n=3 calibrated lr | 15 |
| PrunedHistoryAgent | `pruned_history` | adaptive_value | Obs: +12 pruned action history features | 31 |
| ModulatedValueAgent | `modulated_value` | adaptive_value | Arch: frozen base + gated modulation | 15+4 split |
| CurriculumAgent | `curriculum` | adaptive_value | Training: block scheduling + rehearsal buffer | 19 |
| ExtendedAdaptiveAgent | `extended_adaptive` | adaptive_value | Training: 3× budget (null hypothesis control) | 19 |

## Round 3 Evaluation Results (agent-specific training)

17-agent round-robin tournament (1000 rounds/matchup). Primary metric: **Robustness Score** = avg - 1.5 × std.

### Robustness Leaderboard

| Rank | Agent | Avg | Worst | Best | Std | Robustness | Round |
|------|-------|-----|-------|------|-----|------------|-------|
| 1 | **modulated_value** ★ | **+0.967** | **+0.126** | +1.624 | 0.512 | **+0.199** | R3 |
| 2 | value_based | +0.970 | -0.126 | +1.925 | 0.614 | +0.049 | R0 |
| 3 | adaptive_value | +1.012 | -0.160 | +1.960 | 0.695 | -0.030 | R0 |
| 4 | heuristic | +0.604 | -0.386 | +1.526 | 0.594 | -0.287 | R0 |
| 5 | **extended_adaptive** | +0.329 | -0.567 | +1.335 | 0.490 | -0.406 | R3 |
| 6 | entropy_ac | +0.664 | -0.657 | +1.961 | 0.917 | -0.712 | R2 |
| 7 | adaptive_history | +0.134 | -1.003 | +0.900 | 0.701 | -0.917 | R2 |
| 8 | aux_value | -0.211 | -1.229 | +0.645 | 0.517 | -0.986 | R0 |
| 9 | nstep_value | +0.072 | -1.027 | +1.064 | 0.720 | -1.007 | R2 |
| 10 | actor_critic | -0.568 | -1.426 | +0.149 | 0.419 | -1.197 | R1 |
| 11 | target_value | -0.296 | -1.624 | +1.224 | 0.828 | -1.537 | R2 |
| 12 | decay_adaptive | -0.881 | -1.867 | +0.054 | 0.518 | -1.657 | R1 |
| 13 | **td_variant** | -0.216 | -1.618 | +1.313 | 1.008 | -1.729 | R3 |
| 14 | pop_adaptive | -0.557 | -1.961 | +0.774 | 0.810 | -1.771 | R2 |
| 15 | **pruned_history** | -0.691 | -1.745 | +0.632 | 0.727 | -1.781 | R3 |
| 16 | history_value | -0.512 | -1.925 | +0.903 | 0.873 | -1.822 | R1 |
| 17 | **curriculum** | -0.822 | -1.960 | +0.834 | 0.770 | -1.976 | R3 |

★ modulated_value is the first agent to beat EVERY opponent (positive worst-case).

## Diagnosis: Round 3 Results

### Modulated Value — New #1 (rank 1, +0.967 avg, +0.199 robustness) ★

- **What worked**: "First, do no harm" — a frozen pretrained base (value_based) with structurally bounded perturbations
- **Mechanism**: Gate stays in narrow [0.39, 0.48] range; effective modulation is only ~6% of base value. The architecture is structurally incapable of catastrophically deviating from the strong base
- **Surprise**: Gate DECREASES with confidence (opposite of design intent). The network learned to suppress modulation when stats are reliable, protecting the pretrained base
- **Ablation**: Base-only (gate=0) performs comparably to full model. The success is fundamentally a **transfer learning** story — a strong pretrained model protected from degradation
- **Key insight**: Architectural constraints that prevent training from harming a good initialization can be more valuable than any algorithmic improvement

### TD Variant (rank 13, -0.216 avg)

- **Root cause**: n=3 is functionally pure Monte Carlo in Leduc (99.1% of transitions use terminal reward)
- **Mechanism**: MC targets produce 2.5× higher loss and 2.4× larger gradients than TD(0). lr=5e-5 only produces 50% of TD(0)'s param updates, creating a slow-learning + noisy-gradient combination
- **Key insight**: TD(0) bootstrapping provides implicit temporal smoothing uniquely suited to self-play. MC removes this smoothing entirely

### Curriculum (rank 17, -0.822 avg — worst agent)

- **Root cause**: Both-player chain collection introduces destructive gradient interference
- **Mechanism**: In pool training, P1 is a frozen opponent. P1's chains are off-policy AND have opposite-sign rewards (zero-sum). P0-only ablation improves by +1.2 chips/round
- **Key insight**: "More data is better" only when data is on-policy. In adversarial settings, opponent-side training data has opposite rewards that destroy learning

### Pruned History (rank 15, -0.691 avg)

- **Root cause**: Compound — received only 667 sessions (planned 2000) AND 64 hidden units insufficient for 31-dim input
- **Budget fix**: 2000 sessions improves from -1.32 to -0.54, but still far below adaptive_value (+0.02)
- **Key insight**: When expanding observation space, network width must scale proportionally

### Extended Adaptive (rank 5, +0.329 avg)

- **Result**: No overfitting — performance monotonically improves with more training
- **Null hypothesis**: Partially supported — more training helps (from -1.67 at 667 to -0.33 at 2000) but can't match architectural innovations (modulated_value +0.97 with same budget)

### Generation 5 — Round 4 Creative Exploration (30K-40K episodes)

| Agent | ID | Parent | What Changed | Obs Dims |
|-------|----|--------|-------------|----------|
| BeliefValueAgent | `belief_value` | BaseAgent (NEW) | Obs: +3 belief features, Bayesian hand inference | 14 |
| NashValueAgent | `nash_value` | BaseAgent (NEW) | Training: CFR Nash values as supervised targets | 15 |
| DistributionalValueAgent | `distributional_value` | value_based | Arch: dual-head mean+variance, risk-sensitive decisions | 15 |
| OpponentModelAgent | `opponent_model` | value_based | Planning: 2-ply search with learned opponent model | 15 |
| InfoHidingAgent | `info_hiding` | actor_critic | Loss: adversarial spy network penalizing predictability | 15 |

## Round 4 Evaluation Results (Creative Exploration)

22-agent round-robin tournament (500 rounds/matchup). All 5 new agents tested against 6 representative opponents.

### Results

| Rank | Agent | Avg | Robustness | Key Finding |
|------|-------|-----|------------|-------------|
| 1 | **belief_value** | -0.434 | -1.024 | Likelihood model broken (88% RAISE bias from self-play) |
| 2 | **distributional_value** | -0.580 | -1.230 | Dual-head works; beta=0.5 optimal (sweep: -0.37 avg, -0.88 robust) |
| 3 | **info_hiding** | -0.626 | -1.133 | Spy gradient detached (non-differentiable sampling) |
| 4 | **nash_value** | -0.984 | -1.221 | Perfect fit, wrong decision procedure (argmax ≠ mixed strategy) |
| 5 | **opponent_model** | -1.398 | -2.174 | Generic opponent model + 2-ply amplifies aggression bias |

**All 5 agents lose to incumbents** (adaptive_value +1.01, value_based +0.97, modulated_value +0.97).

### Diagnosis: Round 4 Results

**Belief Value** (avg -0.434): Bayesian belief tracking is conceptually sound but the likelihood model, trained via self-play, learns "everyone raises" (88% RAISE accuracy, 2.7% FOLD). Beliefs barely shift because P(action|hand) is near-uniform. Needs diverse opponent training to work.

**Distributional Value** (avg -0.580): Required 6 architecture iterations. Quantile regression diverges in bootstrapped RL; dual-head (separate value + variance networks with separate optimizers) is stable. Beta=0.5 achieves best robustness (-0.88) among R4 agents. Key insight: risk-sensitivity shifts RAISE→CALL (lower pot variance), a genuine strategic effect.

**Info-Hiding** (avg -0.626): Spy accuracy lowered to 73% (vs 88% value_based), but raise gap K-J = 0.804 (worse than value_based's 0.463). The adversarial gradient cannot flow through discrete action sampling (torch.multinomial). Needs Gumbel-Softmax or REINFORCE-based spy loss.

**Nash Value** (avg -0.984): CFR gives near-exact Nash values (exploitability 0.003). Network fits them nearly perfectly (MSE=0.324, irreducible=0.317). But argmax V(post(s,a)) produces a pure strategy — Nash requires mixed strategies. This is a fundamental architectural mismatch.

**Opponent Model** (avg -1.398): 2-ply search changes 18.3% of decisions, mostly CALL→RAISE (241) and FOLD→RAISE (134). But opponent model predicts generic 17%/47%/37% fold/call/raise regardless of opponent. Self-play produces a single-mode opponent model that amplifies aggression bias.

### Cross-Cutting Themes

1. **Self-play is poor for opponent-aware components** — belief, opponent model, and info-hiding all needed diverse opponent data
2. **Value + argmax has fundamental limits** — Nash and distributional both expose that argmax cannot produce mixed strategies or risk-sensitive play without careful tuning
3. **Training methodology > architecture innovation** — the 15-dim encoding, self-play, and TD(0) are the real performance determinants

## Key Insights — Cumulative

1. **TD(0) value learning remains king** — simple, stable, effective. Neither n-step returns, target networks, nor auxiliary losses improved on it in this domain
2. **Opponent statistics are the strongest augmentation** — adaptive_value has been #1 across Rounds 1-2
3. **Structural protection of pretrained models is the strongest technique** — modulated_value's "don't break what works" approach achieved the highest robustness ever (#1 across all 3 rounds)
4. **Entropy regularization is the best training modification** — entropy_ac is the only agent to come close to the top 2 in average performance
5. **Self-play is non-stationary** — techniques designed for stationary environments (target networks, population training) can backfire
6. **On-policy data is non-negotiable in adversarial settings** — opponent-side training data has opposite rewards that destroy learning
7. **Feature engineering needs matched capacity and training** — more features need wider networks and more training data
8. **Robustness ≠ average performance** — entropy_ac ranks high in avg but low in robustness due to variance
9. **Transfer learning > algorithmic sophistication** — a well-initialized, structurally protected model outperforms all algorithmic innovations tested in 3 rounds
10. **Self-play is poor for opponent modeling** — belief tracking, opponent prediction, and spy networks all learn "everyone plays like me" because self-play only exposes the agent to its own converging policy [Round 4]
11. **Nash equilibrium + greedy ≠ Nash play** — compressing Nash into V(s) + argmax produces pure (exploitable) strategies; Nash requires mixed strategies that argmax cannot produce [Round 4]
12. **Risk-sensitivity needs careful calibration** — poker has irreducible variance (std ~3-4) from hidden cards; risk penalty collapses to all-fold if beta too high, but beta=0.5 genuinely shifts RAISE→CALL (lower variance action) [Round 4]
13. **Adversarial training needs differentiable actions** — discrete sampling (torch.multinomial) breaks gradient flow from spy loss to policy; Gumbel-Softmax or REINFORCE-based spy rewards needed [Round 4]

### Generation 6 — Round 5 Belief-Based Agents (Two-Axis Framework)

| Agent | ID | Parent | What Changed | Obs Dims |
|-------|----|--------|-------------|----------|
| BeliefCfrAgent | `belief_cfr` | belief_value | Likelihood: self-play learned → frozen CFR Nash | 14 |
| BeliefModulatedAgent | `belief_modulated` | belief_value | Likelihood: self-play → Nash + gated modulation | 14 |
| BeliefOracleAgent | `belief_oracle` | belief_value | Architecture: V(s,belief) → V(s,my_hand,opp_hand) + belief weighting | 14 |
| BeliefConfidentAgent | `belief_confident` | belief_value | Obs: +1 confidence feature (n_sessions/30) | 15 |
| BeliefStableAgent | `belief_stable` | belief_value | Training: TD target uses b_t instead of b_{t+1} | 14 |

## Round 5 Evaluation Results (Belief-Based Agents)

Two-axis experiment design: Axis 1 (belief estimation) then Axis 2 (belief usage).

### Axis 1 — Belief Estimation

| Rank | Agent | Avg | Worst | Best | Std | Robustness |
|------|-------|-----|-------|------|-----|------------|
| 1 | **belief_modulated** | **+0.315** | -0.14 | +1.082 | 0.496 | **-0.430** |
| 2 | **belief_cfr (pop ablation)** | **+0.173** | -0.27 | +0.82 | 0.425 | **-0.465** |
| 3 | belief_value (R4 baseline) | -0.434 | -1.024 | — | — | -1.024 |
| 4 | belief_cfr | -0.724 | -1.118 | -0.386 | 0.264 | -1.120 |

Winner: **belief_modulated** — first belief agent with positive avg score. Beats 4/6 opponents (heuristic +0.776, value_based +0.002, modulated_value +0.186, entropy_ac +1.082). Only loses to adaptive_value (-0.14) and cfr (-0.016). Trained for 40K sessions (1.2M hands, ~109 min).

### Axis 2 — Belief Usage (using belief_modulated likelihood)

| Rank | Agent | Avg | Worst | Best | Std | Robustness | Key Finding |
|------|-------|-----|-------|------|-----|------------|-------------|
| 1 | **belief_modulated** (control, full training) | **+0.315** | -0.14 | +1.082 | 0.496 | **-0.430** | First competitive belief agent |
| 2 | belief_confident (2K sessions) | -0.213 | -0.316 | -0.138 | 0.069 | -0.317 | Best short-training; conf=0 optimal |
| 3 | belief_stable (40K episodes) | -0.445 | -0.814 | +0.110 | 0.304 | -0.901 | Best beliefs (0.65), only raiser |
| 4 | belief_oracle (40K episodes) | -0.488 | -0.908 | +0.054 | 0.353 | -1.018 | TD > MC, correct value ordering |

Note: E2b-E2d used shorter training schedules (40K episodes or 2K sessions ≈ 60K hands) vs E1b's 40K sessions (1.2M hands). Training duration may confound architectural comparisons.

### Per-Opponent Details

**belief_modulated (E1b, full training) — Best Overall:**
| vs heuristic | vs value_based | vs adaptive | vs modulated | vs entropy_ac | vs cfr |
|-------------|---------------|-------------|-------------|---------------|--------|
| **+0.776** | **+0.002** | -0.140 | **+0.186** | **+1.082** | -0.016 |

**belief_confident (E2c, 2K sessions) — Best Short-Training:**
| vs heuristic | vs value_based | vs adaptive | vs modulated | vs entropy_ac | vs cfr |
|-------------|---------------|-------------|-------------|---------------|--------|
| -0.184 | -0.192 | -0.280 | -0.168 | -0.316 | -0.138 |

**belief_stable (E2d) — Best Belief Quality:**
| vs heuristic | vs value_based | vs adaptive | vs modulated | vs entropy_ac | vs cfr |
|-------------|---------------|-------------|-------------|---------------|--------|
| -0.414 | -0.548 | -0.472 | -0.814 | +0.110 | -0.530 |

### Diagnosis: Round 5 Results

**Belief Confident (E2c, avg -0.213 — best short-training belief agent):**
- Belief correctness 0.39 — below random baseline (0.50), confirming beliefs are counterproductive at this training level
- Confidence mechanism works: TVD=0.18 between conf=0 and conf=1 strategies
- But performance DEGRADES with confidence: conf=0 → +0.014, conf=0.5 → -0.206, conf=1.0 → -0.280
- The model correctly learned that its belief is harmful and should be ignored
- The improvement comes from accidentally dampening the bad belief signal, not from exploiting beliefs

**Belief Stable (E2d, avg -0.445 — most interesting diagnostically):**
- BEST belief correctness: 0.648 — well above random baseline (0.50), far ahead of all others (0.39-0.51)
- ONLY agent that raises with Q (54%) and K (82%) — non-degenerate strategy
- But broken TD chain means value estimates are less accurate
- Trade-off: better beliefs + worse value learning → net negative

**Belief CFR (E1a, avg -0.724 — informative failure):**
- Nash equilibrium likelihoods are informationally opaque: P(a|J) ≈ P(a|Q) ≈ P(a|K) at equilibrium
- Belief correctness 0.42 — **below random baseline (0.50)**, meaning beliefs are counterproductive
- Belief shift = 0.000 in round 1 (flop): after observing opponent's action, `posterior ∝ P(action|h) × prior` reduces to `posterior ≈ prior` because Nash likelihoods are constant across hands
- 100% FOLD with J — exploitable degenerate strategy

**Belief Modulated (E1b, avg +0.315 — first competitive belief agent):**
- With full training (40K sessions, 1.2M hands, ~109 min), beats 4/6 opponents
- Gate shows differentiation: heuristic (0.578) > value_based (0.542) > adaptive_value (0.535)
- Non-degenerate strategy: K raises 83% preflop, J raises 41% on flop
- Belief correctness 0.514 — above random baseline (0.50), best among non-stable agents
- Modulation hurts raw Nash accuracy (-11.1%), but may improve discriminativeness across hands
- **Ablation resolved**: pure Nash + population + 40K sessions (no modulation) achieved avg +0.173. Population+duration explain 86% of improvement over E1a; modulation adds +0.142 (14%), primarily through improved belief quality (0.39 → 0.51).
- **Key**: 2K-session version (avg -0.30) was dramatically undertrained. Population training + duration are the dominant variables.

### Cross-Cutting Themes — Round 5

**E1b (belief_modulated) is the first belief agent to achieve positive avg (+0.315), beating 4/6 opponents.** With full training (40K sessions), it approaches incumbent performance. Short-trained variants (E2b-E2d) still lose to all opponents.

| Theme | Evidence |
|-------|----------|
| Training duration is critical for belief agents | E1b: 2K sessions → -0.30, 40K sessions → **+0.315** |
| Population+duration = 86% of improvement | Ablation (Nash+pop+40K): +0.173 vs E1b (Nash+mod+pop+40K): +0.315 |
| Beliefs are harmful when undertrained | E2c conf=0 (+0.014) > conf=1 (-0.280) with 2K sessions |
| Nash equilibrium = informationally opaque | E1a likelihoods too uniform for Bayesian discrimination |
| Population training + sufficient duration = competitive | E1b full (pop, +0.315) > E1a (self-play, -0.72) |
| Stable belief fixes beliefs but breaks values | E2d: belief acc 0.648 but avg -0.445 |

## Key Insights — Cumulative

14. **Bayesian belief tracking needs massive training to work** — with 2K sessions (60K hands), belief agents lose to all opponents. But with 40K sessions (1.2M hands), belief_modulated achieves avg +0.315, beating 4/6 opponents. Nash equilibrium likelihoods alone are too uniform (E1a), but Nash + modulation + population training + sufficient duration produces the first competitive belief agent. [Round 5]
15. **Confidence mechanisms can protect against bad beliefs** — adding a "trust knob" lets the model learn to ignore unreliable beliefs. E2c's monotonically decreasing performance with confidence (conf=0 best) is diagnostic of harmful beliefs. [Round 5]
16. **Stable TD targets improve belief quality but break value propagation** — using b_t instead of b_{t+1} in TD targets produces 0.648 belief correctness (vs ~0.40 baseline) and non-degenerate strategies, but the broken TD chain degrades value estimation. [Round 5]
17. **Two-axis experiment design enables clean attribution** — testing belief estimation and usage independently revealed that the estimation problem (Axis 1) is more fundamental than the usage architecture (Axis 2). [Round 5]
18. **Training duration matters more than architecture for belief agents** — E1b at 2K sessions (-0.30) vs 40K sessions (+0.315) is a 0.6 chip/round swing, larger than any Axis 2 architectural change. Belief agents need ~20× more training than value-based agents to converge, likely because the likelihood model requires extensive opponent exposure. [Round 5]
