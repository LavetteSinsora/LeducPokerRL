# Agent Family Tree

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
│   └── AdaptiveValueAgent ......... +Opponent stats, 19-dim obs (changes: observation space)
│       └── DecayAdaptiveAgent ..... +EMA opponent stats (changes: stat accumulation method)
│
├── PolicyGradientAgent ............ REINFORCE, 15-dim obs, categorical sampling
│   └── ActorCriticAgent ........... +Value baseline for variance reduction (changes: loss function)
│
└── CFRAgent ....................... Tabular CFR+, game-theoretic Nash equilibrium
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

### Generation 2 — Round 1 Experiments

| Agent | ID | Parent | What Changed | Obs Dims |
|-------|----|--------|-------------|----------|
| ActorCriticAgent | `actor_critic` | policy_gradient | Loss: REINFORCE → REINFORCE + value baseline | 15 |
| HistoryValueAgent | `history_value` | value_based | Obs: +16 per-round action count features | 31 |
| DecayAdaptiveAgent | `decay_adaptive` | adaptive_value | Stats: uniform averaging → EMA | 19 |

## Round 1 Evaluation Results

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

## Diagnosis: Why Round 1 Agents Underperformed

### Actor-Critic (rank 5, -0.54 avg)
- **Problem**: REINFORCE inherently has high variance even with a baseline; 3000 episodes is insufficient for policy gradient convergence in self-play
- **Root cause**: The value baseline helps but can't compensate for the fundamental credit assignment problem — the agent only gets one reward signal per episode (terminal), applied to ALL actions
- **What works**: Marginally better than raw policy gradient (which doesn't even train stably)

### History Value (rank 6, -0.74 avg)
- **Problem**: Doubled the input space (15→31) without increasing network capacity, causing underfitting
- **Root cause**: In Leduc Hold'em, action sequences are very short (2-4 actions), and most of the strategic information is already captured by pot sizes and round number. The 16 extra features are mostly redundant noise
- **What works**: The scalable encoding design is sound — it will matter more in Texas Hold'em where sequences are longer and pot sizes alone don't capture the betting context

### Decay Adaptive (rank 7, -0.79 avg)
- **Problem**: Slightly worse than its parent (adaptive_value, rank 1)
- **Root cause**: In self-play training, both players are the same agent — there's no shifting opponent strategy to adapt to, so EMA's recency bias provides no advantage over uniform averaging
- **What works**: The EMA mechanism is correct and would likely help against a population of diverse opponents

## Key Insights for Future Work

1. **Value-based TD(0) is the best learning algorithm** for this domain — it provides stable gradients per step rather than noisy episode-level signals
2. **Opponent statistics genuinely help** (adaptive_value is #1) — the network can learn to exploit opponent tendencies
3. **More input features need more training** — simply adding features to a fixed-size network can hurt performance
4. **Self-play limits opponent modeling** — session-based stats only help when there's genuine diversity in opponent behavior
5. **Policy gradient methods need stronger variance reduction** — beyond simple baselines (e.g., GAE, multi-step returns, entropy regularization)
