# Heuristic Agent

> Hand-crafted rule-based strategy using hand strength, pot odds, and position awareness.

| Property | Value |
|----------|-------|
| **ID** | `heuristic` |
| **Parent** | None -- baseline |
| **Round** | 0 |
| **Rank** | #4 / 17 |
| **Avg Score** | +0.604 |
| **Robustness** | -0.287 |

## Motivation

The Heuristic Agent serves as the non-learning baseline for the entire project. Before training any neural network, we need a competent opponent that plays reasonable poker. This agent encodes human poker knowledge -- hand strength evaluation, pot odds calculation, position awareness, and balanced bluffing frequencies -- into explicit decision rules. It provides the initial yardstick against which all RL agents are measured: if an RL agent cannot beat a hand-crafted strategy, it has not learned anything meaningful.

This is one of only two non-RL agents in the tournament (alongside CFR). Unlike every other agent, it has zero trainable parameters and requires no training episodes.

## Architecture

**No neural network.** The Heuristic Agent uses pure rule-based decision logic with two main strategy branches:

### Pre-flop Strategy (Round 0)
The agent evaluates hand strength using card rank alone (K > Q > J) since no community card is visible:

- **Facing a raise:**
  - King: Re-raise if possible, otherwise call
  - Queen: Call (decent equity, ~50% vs random hand)
  - Jack: Fold unless pot odds are very favorable (< 35% of total pot to call)

- **Not facing a raise:**
  - King: Always raise for value
  - Queen: Raise as a thin value bet (also balances the raising range)
  - Jack: Check/call by default, but bluff-raise ~20% of the time for balance

### Flop Strategy (Round 1)
With the community card visible, the agent branches on pair status:

- **Has pair (hand == board card):** Always raise regardless of pair rank. Even Jack pairs are strong enough to value-raise in Leduc.

- **No pair, not facing a raise:**
  - King high: Raise for value (likely the best unpaired hand)
  - Queen high on J board: Raise thinly (beats J high)
  - Queen high on K board: Check/call (vulnerable)
  - Jack high: Bluff-raise ~25% of the time, otherwise check/call

- **No pair, facing a raise:** Uses pot odds calculation:
  - King high: Call to catch bluffs
  - Queen high: Call only with good pot odds (< 20-25% of pot), otherwise fold
  - Jack high: Call only with excellent pot odds (< 18%), otherwise fold

### Key Design Features
- **Pot odds integration:** `pot_odds = to_call / (pot_total + to_call)` -- thresholds vary by hand strength
- **Bluffing balance:** Jacks bluff 20% pre-flop and 25% on the flop to prevent opponents from always folding against raises
- **Position awareness:** Different strategies for acting first vs. responding to opponent aggression

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 0 |
| Learning Rate | N/A |
| Batch Size | N/A |
| Special | Pure rule-based, no training |

No training is required. The strategy is hand-coded based on Leduc Hold'em game theory. The agent uses Python's `random` module for bluffing decisions (20% pre-flop, 25% on flop with weak hands).

## Tournament Results

### Round 3 -- Full 17-Agent Tournament (1000 rounds/matchup)

| Opponent | Score |
|----------|-------|
| value_based | -0.092 |
| adaptive_value | -0.386 |
| aux_value | +0.348 |
| actor_critic | +0.619 |
| history_value | +0.983 |
| decay_adaptive | +1.177 |
| nstep_value | +0.560 |
| entropy_ac | +0.529 |
| pop_adaptive | +0.637 |
| adaptive_history | +0.278 |
| target_value | +1.020 |
| td_variant | +1.526 |
| pruned_history | +1.454 |
| modulated_value | -0.186 |
| curriculum | +1.233 |
| extended_adaptive | -0.033 |

### Performance Profile
- **Wins against:** 12 of 16 opponents (positive score)
- **Loses to:** value_based (-0.092), adaptive_value (-0.386), modulated_value (-0.186), extended_adaptive (-0.033)
- **Best matchup:** +1.526 vs td_variant
- **Worst matchup:** -0.386 vs adaptive_value

### Cross-Round Trajectory

| Round | Rank | Avg Score | Context |
|-------|------|-----------|---------|
| R1 | #3 / 7 | +0.29 | Above aux_value, actor_critic, history_value, decay_adaptive |
| R2 | #3 / 12 | +0.47 | Consistent mid-tier performance |
| R3 | #4 / 17 | +0.604 | Still top 4 despite 13 RL competitors |

## Key Findings

1. **Remarkably competitive.** The heuristic agent has never dropped below rank 4 across three tournament rounds. Most RL agents trained with thousands of episodes fail to beat it.

2. **Robustness through simplicity.** With std=0.594, the heuristic has lower variance than most RL agents. It never catastrophically fails because its rules are always reasonable, even if not optimal.

3. **Exploitable by strong learners.** The top 3 agents (modulated_value, value_based, adaptive_value) all beat the heuristic, showing that RL can surpass hand-coded play -- but only with well-designed architectures.

4. **The bluffing frequencies matter.** The 20% pre-flop and 25% post-flop bluff rates were hand-tuned. Without bluffing, the agent would be trivially exploitable (opponents could always fold to raises with weak hands).

5. **Pot odds prevent costly mistakes.** The threshold-based calling logic means the agent rarely makes hugely negative-EV calls, which is why it outperforms many RL agents that over-call or over-fold.

## Key Insight

A well-designed rule-based agent is harder to beat than most people expect -- it took multiple rounds of RL experimentation before more than 3 out of 17 agents could consistently defeat it.

## Source Files

- Agent: `src/agents/heuristic.py`
- Trainer: None (no training required)
