# Round 4 Follow-Up Context: Non-Belief Agents

> Briefing document for a new Claude session to explore follow-up experiments on the four non-belief Round 4 agents (Distributional, Nash Value, Opponent Model, Info Hiding).

---

## Project Overview

This is a Leduc Hold'em poker RL research project. Over 4 rounds of experiments, 22+ agents have been designed, trained, and evaluated in round-robin tournaments. The project uses self-play training with TD(0) value learning as the core paradigm, and has explored opponent modeling, game theory, distributional RL, adversarial training, and Bayesian inference.

The top 3 agents after Round 3 were adaptive_value (avg +1.012), value_based (+0.970), and modulated_value (+0.967, best robustness +0.199). Round 4 tested 5 radically different directions; all underperformed the incumbents but produced deeply informative failures.

This document focuses on the 4 non-belief Round 4 agents. For the belief agent's follow-up framework, see `web/wiki/belief_framework.md`.

---

## Relevant Files to Read First

| File | What It Contains |
|------|-----------------|
| `AGENTS.md` | Full agent family tree, all results across all rounds, cumulative insights |
| `experiment_reports/round4_report.md` | Comprehensive Round 4 analysis with diagnosis for all 5 directions |
| `experiments/round4_distributional_results.json` | Distributional agent detailed results (beta sweep, action distributions) |
| `experiments/round4_nash_results.json` | Nash value agent results (encoding analysis, CFR value comparison) |
| `experiments/round4_opponent_model_results.json` | Opponent model results (accuracy by opponent, decision changes) |
| `experiments/round4_info_hiding_results.json` | Info hiding results (spy accuracy, lambda sweep, action distributions) |
| `src/agents/distributional_value.py` | Distributional agent source code |
| `src/agents/nash_value.py` | Nash value agent source code |
| `src/agents/opponent_model.py` | Opponent model agent source code |
| `src/agents/info_hiding.py` | Info hiding agent source code |
| `src/training/distributional_trainer.py` | Distributional trainer (dual-head, separate optimizers) |
| `src/training/nash_trainer.py` | Nash trainer (CFR + supervised regression) |
| `src/training/opponent_model_trainer.py` | Opponent model trainer (value + opponent model) |
| `src/training/info_hiding_trainer.py` | Info hiding trainer (actor-critic + spy) |

---

## Round 4 Agent Summaries

### 1. Distributional Value Agent (avg -0.58, robustness -1.23)

**What it does**: Dual-head architecture with separate value and variance networks (separate optimizers). Makes risk-adjusted decisions: `V_risk(s,a) = E[V(s')] - beta * Var[V(s')]`. Beta=0.5 is optimal (avg -0.37, robustness -0.88 -- best R4 robustness).

**Why it's interesting**: Genuine risk-sensitivity effect. 70% of decisions differ from value_based. Systematic shift from RAISE to CALL -- raising increases pot (higher variance), calling keeps pot small (lower variance). The dual-head with separate optimizers was critical; 5 prior architectures failed (quantile regression x3, shared trunk, shared optimizer).

**Why it failed**: Poker has inherent std of 3-4 chips per hand from card randomness. At beta > 0.7, the variance penalty makes FOLD (zero variance) dominate every other action. The viable beta range is narrow: [0.3, 0.6].

**Follow-up directions**:
- Better variance estimation: separate aleatoric (card randomness) from epistemic (model uncertainty). Only penalize epistemic variance, allowing higher effective beta without fold-collapse.
- Dynamic beta: adjust risk sensitivity based on game state (e.g., more conservative when behind, more aggressive when ahead).
- Conditional variance: variance conditioned on action choice, not just state.

### 2. Nash Value Agent (avg -0.98, robustness -1.22)

**What it does**: Trains a neural network via supervised regression on exact CFR Nash equilibrium values. 10K CFR iterations produce exploitability of 0.003136. The network achieves MSE = 0.324, only 0.007 above the theoretical minimum (encoding collisions create an irreducible MSE floor of 0.317). The neural network explains 99.2% of variance in CFR values.

**Why it's interesting**: Near-perfect approximation of game-theoretically optimal values. Proves that a standard MLP can represent Nash equilibrium values for Leduc Hold'em with negligible error.

**Why it failed**: `argmax_a V(post(s, a))` produces pure strategies. Nash equilibrium requires MIXED strategies (e.g., raise with K 60%, call 40%). Argmax always picks the single best action, collapsing the mixed strategy to a deterministic one that is trivially exploitable. States where Nash assigns high value to mixed play (K vs K board) show the largest divergence from self-play values (up to 9.37 chips per infoset).

**Follow-up directions**:
- Policy-based Nash: train pi(a|s) to match CFR action probabilities via KL divergence, not V(s) + argmax. This preserves mixed strategies.
- Nash-regularized self-play: use Nash values/policies as a regularizer during standard self-play training. Loss = TD_loss + lambda * KL(pi, pi_nash).
- Nash as initialization: pretrain value network on CFR values, then fine-tune via self-play. Tests whether Nash provides a better starting point than random initialization.

### 3. Opponent Model Agent (avg -1.40, robustness -2.17)

**What it does**: Learns an opponent action model P(a|state) during self-play. At decision time, uses 2-ply lookahead: for each action, predicts opponent response from the model, evaluates the resulting state, and picks the highest expected value action.

**Why it's interesting**: First agent to do explicit forward planning beyond 1-step lookahead. The 2-ply search is conceptually sound -- it should capture "if I raise, they'll probably fold" reasoning.

**Why it failed**: The opponent model learns the aggregate self-play distribution (~17% fold, ~47% call, ~37% raise) regardless of opponent type. The 2-ply search then amplifies this bias: it treats the 17% fold prediction as exploitable fold equity, converting 375 passive/defensive decisions into raises (51.6% CALL->RAISE, 28.7% FOLD->RAISE). The agent becomes excessively aggressive against opponents who actually fold much less than 17%.

**Accuracy by opponent**: value_based 76.1%, modulated_value 75.8%, adaptive_value 75.7%, entropy_ac 73.5%, heuristic 66.0%, cfr 58.0%. Note that accuracy is highest for opponents most similar to the training distribution (self-play).

**Follow-up directions**:
- Diverse training opponents: train against the full agent pool (20+ agents) to learn varied behavioral patterns.
- Online model adaptation: update the opponent model during evaluation based on observed actions (not just training).
- Opponent-conditioned model: separate models per opponent type, or a conditional model P(a|state, opponent_embedding).

### 4. Info Hiding Agent (avg -0.63, robustness -1.13)

**What it does**: Actor-critic with an adversarial spy network. The spy tries to predict the agent's hand from its action history. The policy is trained to maximize value while MINIMIZING spy accuracy (hiding information). Loss = policy_gradient + value_loss - lambda * cross_entropy(spy_pred, true_hand).

**Why it's interesting**: Adversarial information hiding is how Nash equilibrium bluffing strategies arise. The concept directly targets the mechanism behind optimal poker play: making your actions uninformative about your hand.

**Why it failed**: The policy outputs action probabilities, then samples via `torch.multinomial()`. This sampling is non-differentiable -- gradients from the spy loss CANNOT flow back through the discrete action choice to the policy. The adversarial signal never reaches the policy. Evidence: the info_hiding agent has a LARGER raise gap (K-J = 0.804) than value_based (0.463), meaning it is MORE predictable, not less. The lower spy accuracy (73% vs 88%) comes from the spy itself being poorly trained, not from the policy hiding information. Lambda sweep shows no monotonic relationship between lambda and spy accuracy.

**Follow-up directions**:
- Gumbel-Softmax: replace `torch.multinomial()` with the Gumbel-Softmax continuous relaxation, enabling gradient flow through action selection. This is the most direct fix.
- REINFORCE-style spy loss: instead of backpropagating through actions, use REINFORCE to provide policy gradients from the spy loss. The spy's prediction error becomes a reward signal.
- Separate training phases: alternate between training the policy (with spy frozen) and training the spy (with policy frozen), similar to GAN training.

---

## What the User Wants

The user wants to discuss these 4 directions in a new session. The session should be:

- **Exploratory and collaborative**: Ask follow-up questions about mechanisms, discuss tradeoffs, propose alternatives.
- **Design-oriented**: Propose concrete experiment designs (architecture, training procedure, hyperparameters).
- **Grounded in evidence**: Reference specific numbers from the Round 4 results, not generic RL advice.
- **Focused on feasibility**: These experiments run in a small Leduc Hold'em environment. Keep proposals practical.

---

## Key Codebase Conventions

| Convention | Details |
|------------|---------|
| Agent base classes | `BaseAgent`, `ValueBasedAgent`, `ActorCriticAgent` in `src/agents/` |
| Trainer base class | `BaseTrainer` in `src/training/` |
| Agent registry | `src/agents/registry.py` -- all agents must be registered here |
| Experiment scripts | `experiments/` folder, one script per experiment |
| Results format | JSON files in `experiments/` |
| Reports | Markdown files in `experiment_reports/` |
| Wiki pages | `web/wiki/` with standard template: quote, property table, motivation, architecture, results, diagnosis, key insight |
| Training episodes | Typically 20,000 (can vary) |
| Evaluation | Round-robin tournament, 500-1000 rounds per matchup |
| Robustness metric | avg - 1.5 * std across all matchups |
| Game | Leduc Hold'em (2 players, 3 card ranks J/Q/K, 2 rounds, ante 1 chip) |
| State encoding | 15 dimensions standard (hand 3, board 4, pot 2, position 1, round 1, raises_left 2, actions 2) |
