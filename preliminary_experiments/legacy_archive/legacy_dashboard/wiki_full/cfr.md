# CFR Agent

> Counterfactual Regret Minimization -- a game-theoretic agent that computes the Nash equilibrium through tabular regret minimization.

| Property | Value |
|----------|-------|
| **ID** | `cfr` |
| **Parent** | None (game-theoretic, not RL) |
| **Round** | 0 (baseline) |
| **Rank** | N/A (reference agent) |
| **Avg Score** | N/A |
| **Robustness** | N/A |

---

## Motivation

All other agents in the PokerRL project learn through reinforcement learning -- trial-and-error interaction with the game environment. But poker has a well-known theoretical solution: the **Nash equilibrium**, a mixed strategy from which no player can unilaterally improve by deviating.

The CFR agent exists as a **reference point** -- the theoretically optimal strategy against which all RL agents can be compared. It represents the ceiling of what is achievable if the game's structure is fully known and exploited.

Important caveat: comparing CFR to RL agents is fundamentally **apples-to-oranges**. CFR has complete knowledge of the game tree, card probabilities, and payout structure. RL agents must learn everything from experience. CFR computes the exact Nash equilibrium; RL agents approximate value functions. The comparison is useful for understanding how close RL methods come to theoretical optimality, not for declaring a "winner."

---

## Algorithm: Counterfactual Regret Minimization (CFR+)

CFR is fundamentally different from all other agents in the project. It does not use neural networks, gradient descent, or experience replay. Instead, it uses **tabular regret minimization** over the full game tree.

### Core Concepts

**Information Set**: A set of game states that are indistinguishable to the acting player. In Leduc Hold'em, an information set is defined by:
- The player's own hand card
- The board card (if revealed)
- The sequence of actions taken so far

Example keys: `"Q:cr"` (holding Queen, preflop: check then raise), `"K:J:cc/r"` (holding King, board is Jack, preflop: check-check, flop: raise)

**Regret**: After each iteration, CFR computes how much better each alternative action would have been compared to the action actually taken (weighted by the strategy). Positive regret means "I should have played this action more."

**Regret Matching**: Convert accumulated regrets into a strategy by normalizing positive regrets into a probability distribution. Actions with more accumulated regret get more probability mass.

### CFR+ Algorithm

Each iteration:

1. **Traverse the full game tree** for every possible deal (all combinations of P0 hand, P1 hand, and board card)
2. At each information set, compute the **counterfactual value** of each action -- the expected value weighted by the opponent's reach probability
3. Compute **regret** for each action: `regret[a] = counterfactual_value[a] - node_value`
4. Buffer regret deltas and apply them after the full traversal
5. **Floor regrets to 0** (the "+" in CFR+ -- negative regrets are discarded, accelerating convergence)
6. **Update strategy sum** with linear weighting (later iterations weighted more heavily)

### Convergence to Nash Equilibrium

The **average strategy** across all iterations provably converges to a Nash equilibrium. The rate of convergence is measured by **exploitability**: how much a best-responding opponent can gain against the current strategy. As iterations increase, exploitability approaches 0.

```
Exploitability = BR_value(P0 best-responds) - BR_value(P1 best-responds)
```

where BR_value computes the expected value when one player plays the optimal best response against the other's average strategy.

---

## Architecture

CFR uses no neural networks. Its "model" is a **TabularStrategyStore** -- a dictionary mapping information set keys to regret and strategy accumulators:

```
TabularStrategyStore
  - data: Dict[str, InfoSetData]
  - Each InfoSetData contains:
    - regret_sum: np.ndarray[3]    (accumulated regrets for FOLD, CALL, RAISE)
    - strategy_sum: np.ndarray[3]  (cumulative strategy weights)
```

For Leduc Hold'em, there are approximately **~300 information sets** (the exact number depends on the action sequences that can be reached). Each information set stores 6 floating-point numbers (3 regrets + 3 strategy weights).

### Game Tree Enumeration

The solver enumerates all possible deals:
- 3 possible hands for P0 (J, Q, K)
- 3 possible hands for P1 (J, Q, K)
- Remaining cards for the board (depending on which cards are dealt)
- Total: 30 unique deal combinations (weighted by probability)

For each deal, the solver traverses the complete game tree, computing counterfactual values at every decision node.

### Showdown Evaluation

Terminal values are computed by comparing hands:
- **Pair beats non-pair**: If one player's hand matches the board, they win
- **Higher card wins**: If neither (or both) have a pair, the higher card wins
- **Tie**: If both have the same rank, the pot is split

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Full game tree traversal (not experience-based) |
| Iterations | Configurable (typically 10,000+) |
| Convergence Metric | Exploitability |
| Strategy Averaging | Linear (later iterations weighted more) |
| Regret Flooring | CFR+ (negative regrets clipped to 0) |
| Information Sets | ~300 (for Leduc Hold'em) |
| Storage | ~1,800 floating-point numbers |

Unlike RL agents, CFR training is **deterministic** -- the same number of iterations always produces the same strategy. There is no random sampling, no stochastic gradient descent, and no exploration-exploitation tradeoff.

---

## Implementation Details

### Agent (Inference)

The `CFRAgent` wraps the strategy store for play:

```python
class CFRAgent(BaseAgent):
    def select_action(self, obs: Observation) -> Action:
        key = self._obs_to_key(obs)
        strategy = self.strategy_store.get_average_strategy(key, obs.legal_actions)
        return random.choices(legal_actions, weights=strategy_weights, k=1)[0]
```

At each decision point, the agent:
1. Converts the observation to an information set key
2. Looks up the average (converged) strategy for that key
3. **Samples** an action according to the mixed strategy probabilities

The sampling is critical: Nash equilibrium strategies are **mixed** (probabilistic), not deterministic. Playing deterministically would make the agent exploitable.

### Observation to Key Conversion

The `_obs_to_key` method replays the action history to determine round boundaries (preflop vs flop), then constructs the key in the solver's format:
- Preflop: `"{hand}:{preflop_actions}"` (e.g., `"Q:cr"`)
- Flop: `"{hand}:{board}:{preflop_actions}/{flop_actions}"` (e.g., `"K:J:cc/r"`)

### Trainer

The `CFRTrainer` overrides the standard RL training loop entirely:

```python
class CFRTrainer(BaseTrainer):
    def train(self, num_episodes, ...):
        for i in range(num_episodes):
            self.solver.run_iteration(iteration)
            # Periodically compute exploitability and evaluate
```

Each "episode" is one full CFR+ iteration over the complete game tree. The trainer reports exploitability as its "loss" metric, allowing it to integrate with the same monitoring infrastructure as RL agents.

---

## Theoretical Properties

### Nash Equilibrium Guarantees

In a two-player zero-sum game like Leduc Hold'em, the Nash equilibrium has strong theoretical properties:

1. **Minimax optimality**: The Nash strategy maximizes the minimum expected value against any opponent strategy
2. **Unexploitability**: No opponent can gain more than 0 expected value against a perfect Nash strategy
3. **Convergence guarantee**: CFR's average strategy is proven to converge to Nash at rate O(1/sqrt(T)) where T is the number of iterations

### Why Nash is Not Always "Best"

While Nash is theoretically optimal in the minimax sense, it is not necessarily the best strategy against specific, imperfect opponents:

- Nash plays **defensively** -- it guarantees no loss but does not exploit opponent weaknesses
- Against a player who folds too often, Nash will not bluff enough to maximize profit
- Against a player who calls too much, Nash will not value-bet aggressively enough
- RL agents can potentially **exploit** specific opponents more effectively than Nash

This is why RL agents like modulated_value can outperform a Nash strategy in practice: they adapt to the specific opponent they face, while Nash plays the same mixed strategy regardless.

---

## Why CFR is Included

CFR serves several important roles in the project:

1. **Theoretical benchmark**: Establishes the upper bound of what is achievable with perfect game knowledge
2. **Strategy reference**: The Nash equilibrium strategy can be compared against RL agent policies to understand how they deviate from optimal play
3. **Algorithmic diversity**: Demonstrates that the same game can be approached from completely different paradigms (game theory vs reinforcement learning)
4. **Exploitability analysis**: CFR's best-response computation can theoretically be used to measure how exploitable RL agents are

### Apples-to-Oranges Caveat

Direct performance comparisons between CFR and RL agents are misleading because:
- CFR has **complete game knowledge** (all card probabilities, exact payoffs, full game tree)
- RL agents must **learn from experience** (no access to hidden information distributions)
- CFR requires **game-specific implementation** (the solver is hardcoded for Leduc rules)
- RL agents use **general-purpose architectures** (the same value network approach could work for many games)

CFR is the "specialist" that knows everything about one game; RL agents are "generalists" that learn any game from scratch.

---

## Key Insight

CFR computes the theoretically optimal mixed strategy for Leduc Hold'em through tabular regret minimization over the full game tree -- it serves as a game-theoretic reference point, but comparing it to RL agents is apples-to-oranges since CFR has perfect knowledge of the game's structure while RL agents must learn everything from experience.

---

## Source Files

- Agent: `src/agents/cfr_agent.py`
- Trainer: `src/training/cfr_trainer.py`
- Strategy Store: `src/cfr/strategy.py`
- Solver: `src/cfr/solver.py`
