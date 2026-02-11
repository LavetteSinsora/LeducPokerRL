# PokerRL Cookbook

A step-by-step guide to implementing & training your own agents in this repo.

By the end of this guide, you will have:
- Built a working Policy Gradient agent from scratch
- Written a training loop that teaches it to play poker
- Connected everything to the web dashboard (so you can train it and play against it)
- Added your trained agent to the `agent pool` (a place where all agents trained in this project will be stored)
---

## 1. Project Overview

I know that this repo looks intimidating and has a lot of files. On a high level, this project has 3 main components:
1. Agents: The AI that plays poker
    - Our main goal in this project is to explore different model architectures to building poker bots
2. Trainers: The code needed to train our agents
3. Web dashboard: The web interface for training agents and playing against them (human vs AI)

Below explains what each directory contains (for your reference). 
**tl;dr: implement your agent in `src/agents/` and its corresponding trainer in `src/training/`.**

A list of quick definitions:
1. Game state/state: The current situation in the game, including:
    - Your hand (J, Q, or K)
    - The board card (J, Q, K, or None if no public card has been dealt yet)
    - The pot (the number of chips each player has put into the pot)
    - Who's turn it is
    - etc.
2. Action: A move that a player can make (fold, call, or raise)

### `src/agents/` — Where agents live

This is where you'll spend most of your time. Every poker-playing AI in this project is an "agent" — a Python class that looks at the current game state (e.g., I got Q. The board is a J. Opponent just called. etc.) and decides what to do (fold, call, or raise).

- **`base.py`** — The `BaseAgent` abstract class. All agents inherit from this. The only method you *must* implement is `select_action(obs) -> Action`.
- **`value_based.py`** — An agent I trained that already beats me on average. High-level idea: looks at what will happen if it takes each action (fold/call/raise). Use a neural network to rate each action, and select the action that takes it to the state with highest value. A good reference for what a trainable agent look like.
- **`heuristic.py`** — A rule-based agent with hand-crafted poker strategy (no neural network). Used as the default opponent during training evaluation as a benchmark/baseline.
- **`registry.py`** — All agents should be registered here. Registered agents automatically appears in the web dashboard. In other words, only registered agents can be trained and accessed via the dashboard.

### `src/training/` — Where trainers live

Each trainable agent needs a "trainer" — a class that defines how the agent learns from experience. The training infrastructure handles the repetitive parts (running episodes, batching, evaluation, saving) so you can focus on the learning algorithm.

- **`base.py`** — The `BaseTrainer` class. Provides a ready-made training loop. You just implement two hooks: `collect_episode()` (play one game and record what happened) and `update_model()` (learn from a batch of games).
- **`value_based_trainer.py`** — The trainer for the Value Network agent. A concrete example of how to implement those hooks.
- **`training_manager.py`** — Manages training sessions for the web dashboard (start/stop/reset). You don't need to modify this.
- **`evaluation.py`** — Evaluates agents by playing them against benchmark agents in the agent pool. Evaluation metric is chips won per round. Used by `BaseTrainer` during training to periodically evaluate the agent's performance.

### `src/engine/` — The game itself

- **`leduc_game.py`** — The Leduc Hold'em game engine. Handles dealing cards, managing rounds, enforcing rules, and determining winners. You'll use `LeducGame` in your trainer to play episodes.
- **`observation.py`** — The `Observation` dataclass that represents what a player can see: their hand, the board card, the pot, legal actions, etc. This is essentially agent's input, containing all information the agent needs to make a decision/select an action.

### `src/server/` — The web server

- **`app.py`** — A simple HTTP server that connects the web frontend to the Python backend. Serves the dashboard pages and provides API endpoints for playing games, training, and analysis. You don't need to modify this.

### `web/` — The browser dashboard

HTML/CSS/JS files for the web interface. Three main pages:
- **Play mode** (`index.html`) — Play poker against any registered agent
- **Training dashboard** (`dashboard.html`) — Start/stop training, watch loss and performance charts update in real time
- **Analyzer** (`analyzer.html`) — Inspect what the agent "thinks" about each possible action

---

## 2. Implementing Your Agent

Let's build a **Policy Gradient agent** — one of the simplest RL approaches. A policy gradient agent directly learns a *policy*: a function that maps game states to action probabilities.

The idea is intuitive:
- The agent plays a game and records every action it took
- If it won chips, those actions were probably good — make them more likely next time
- If it lost chips, those actions were probably bad — make them less likely

### The neural network

Our policy network takes in a game state (encoded as a vector) and outputs a probability for each action (fold, call, raise):

```
Game state (14 numbers) → Neural Network → [P(fold), P(call), P(raise)]
```

Create a new file `src/agents/policy_gradient.py`:

```python
import torch
import torch.nn as nn
from src.engine.leduc_game import Action
from src.engine.observation import Observation
from .base import BaseAgent


class PolicyNetwork(nn.Module):
    """Maps a game state to action probabilities."""
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # 3 actions: fold, call, raise
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class PolicyGradientAgent(BaseAgent):
    """
    A policy gradient agent that directly learns action probabilities.

    During training, it samples actions from its learned distribution
    (so it explores different strategies). During evaluation, it picks
    the highest-probability action (greedy).
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None):
        self.input_size = 15  # 3 (hand) + 4 (board) + 2 (pot) + 6 (features)
        self.train_mode = False

        self.model = PolicyNetwork(self.input_size)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def encode_observation(self, obs: Observation, **kwargs) -> torch.Tensor:
        """Turn a game observation into a tensor the network can process."""
        # One-hot encode the player's hand card (J/Q/K)
        hand_vec = torch.zeros(3)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # One-hot encode the board card (J/Q/K/None)
        board_vec = torch.zeros(4)
        board_idx = self.CARD_MAP.get(obs.board, 3)  # 3 = no board card yet
        board_vec[board_idx] = 1.0

        # Pot sizes, normalized
        pot_vec = torch.tensor(obs.pot, dtype=torch.float32) / self.MAX_CHIPS

        # Extra features
        features = torch.tensor([
            float(obs.current_player),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board and obs.player_hand == obs.board) else 0.0,  # pair?
            obs.raises_this_round / 2.0,
            1.0 if Action.RAISE in obs.legal_actions else 0.0,  # can raise?
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)

    def select_action(self, obs: Observation) -> Action:
        """Pick an action based on the policy network's output."""
        encoded = self.encode_observation(obs)

        with torch.no_grad():
            probs = self.model(encoded).squeeze(0)  # [P(fold), P(call), P(raise)]

        # Mask out illegal actions (set their probability to 0)
        legal_mask = torch.zeros(3)
        for action in obs.legal_actions:
            legal_mask[action.value] = 1.0
        probs = probs * legal_mask

        # Re-normalize so probabilities sum to 1
        probs = probs / probs.sum()

        if self.train_mode:
            # Sample from the distribution (exploration)
            action_idx = torch.multinomial(probs, 1).item()
        else:
            # Pick the most probable action (greedy)
            action_idx = probs.argmax().item()

        return Action(action_idx)

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
```

That's the complete agent. Let's break down what each piece does:

- **`encode_observation()`** converts the game state (cards, pot, round info) into a flat tensor of 15 numbers. The neural network needs numeric input, not strings like `'K'` or `'Q'`.
- **`select_action()`** feeds the encoded state through the network, masks out illegal actions, and either samples (training) or picks the best (evaluation).
- **`save_model()` / `load_model()`** save and restore the network weights so training progress isn't lost.

---

## 3. Implementing Your Trainer

Now we need to teach the agent how to play. The trainer defines *how* the agent learns from its experience.

### How policy gradient training works

The core idea is simple. After each game:

1. Look at every action the agent took during the game
2. Multiply each action's log-probability by the game's outcome (chips won or lost)
3. Use gradient ascent to make winning actions more probable and losing actions less probable

This is called the **REINFORCE** algorithm. If the agent won +5 chips, all actions it took get nudged upward. If it lost -3 chips, they all get nudged downward. Over thousands of games, the agent learns which actions lead to wins.

### The trainer code

Create a new file `src/training/policy_gradient_trainer.py`:

```python
import torch
import torch.optim as optim
from typing import Dict, List
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer


class PolicyGradientTrainer(BaseTrainer):
    """
    Trains a PolicyGradientAgent using the REINFORCE algorithm.

    Each episode:
      1. Play a full game, recording every (state, action) pair
      2. Use the final reward to compute the policy gradient loss
      3. Update the network to make winning actions more likely
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-3):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.game = LeducGame()

    def collect_episode(self) -> dict:
        """Play one game of poker and record everything that happened.

        Returns a dict with:
          - log_probs: the log-probability of each action the agent chose
          - reward: the final chips won/lost by the agent (playing as both players)
        """
        self.game.reset()

        # We'll collect log-probs separately for each player seat,
        # since the reward is different for each side
        log_probs = [[], []]  # log_probs[0] = player 0's actions, etc.

        self.agent.set_train_mode(True)

        while not self.game.is_finished:
            player = self.game.current_player
            obs = self.game.get_observation(viewer_id=player)
            encoded = self.agent.encode_observation(obs)

            # Get action probabilities from the policy network
            probs = self.agent.model(encoded).squeeze(0)

            # Mask illegal actions
            legal_mask = torch.zeros(3)
            for action in obs.legal_actions:
                legal_mask[action.value] = 1.0
            probs = probs * legal_mask
            probs = probs / probs.sum()

            # Sample an action and record its log-probability
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_probs[player].append(dist.log_prob(action_idx))

            self.game.step(Action(action_idx.item()))

        rewards = self.game.get_reward()  # [player_0_chips, player_1_chips]
        return {"log_probs": log_probs, "rewards": rewards}

    def update_model(self, batch_data: list) -> float:
        """Learn from a batch of games using the REINFORCE algorithm.

        For each game, for each player seat:
          loss = -sum(log_prob * reward)

        This pushes the policy toward actions that led to positive rewards
        and away from actions that led to negative rewards.
        """
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for episode in batch_data:
            log_probs = episode["log_probs"]
            rewards = episode["rewards"]

            for player in [0, 1]:
                if not log_probs[player]:
                    continue

                reward = rewards[player]
                # REINFORCE: loss = -log_prob * reward
                # Negative because we want to maximize reward (gradient ascent)
                for lp in log_probs[player]:
                    total_loss = total_loss - lp * reward

        total_loss = total_loss / len(batch_data)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def update_params(self, params: Dict):
        """Allow the dashboard to adjust learning rate during training."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate updated to: {new_lr}")
```

Notice how little code this is. `BaseTrainer` already handles:
- The episode loop (play N games, stop if requested)
- Batching (collect `batch_size` episodes before updating)
- Periodic evaluation (test against the heuristic opponent every 50 episodes)
- Callbacks (send loss and evaluation data to the dashboard charts)
- Model saving (save weights when training finishes)

Your trainer only needs to define what data to collect (`collect_episode`) and how to learn from it (`update_model`).

---

## 4. Connecting to the Dashboard

To make your agent appear in the web dashboard, you need to register it in `src/agents/registry.py`. This is the single place that tells the system "here are all the available agents and their trainers."

### Register your agent

Open `src/agents/registry.py` and add your imports and registration inside the `_register_builtin_agents()` function:

```python
def _register_builtin_agents():
    """Register all built-in agents with the registry."""
    from .heuristic import HeuristicAgent
    from .value_based import ValueBasedAgent
    from src.training.value_based_trainer import SelfPlayTrainer

    # --- Add these two imports ---
    from .policy_gradient import PolicyGradientAgent
    from src.training.policy_gradient_trainer import PolicyGradientTrainer

    # ... existing registrations for heuristic and value_based ...

    # --- Add this registration block ---
    registry.register(
        id="policy_gradient",
        agent_class=PolicyGradientAgent,
        metadata=AgentMetadata(
            id="policy_gradient",
            display_name="Policy Gradient AI",
            description="RL agent using REINFORCE policy gradients",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=PolicyGradientTrainer,
        )
    )
```

The key fields:
- **`id`** — A unique string identifier. This is used internally and as the model filename (`models/policy_gradient_agent.pt`).
- **`display_name`** — What users see in the dropdown menus.
- **`is_trainable`** — Set to `True` so the training tab knows this agent can be trained.
- **`trainer_class`** — Points to your trainer. The training manager uses this to create a trainer instance when the user clicks "Start Training."

### Launch the dashboard and start training

1. Start the server:
   ```
   python -m src.server.app
   ```

2. Open your browser to `http://localhost:8000`

3. Go to the **Training** tab (`dashboard.html`):
   - Select "Policy Gradient AI" from the agent dropdown
   - Click **Reset Agent** to initialize it (this clears any old model and creates a fresh agent)
   - Adjust the training settings if you want (episodes, batch size, learning rate)
   - Click **Start Training**
   - Watch the loss chart and performance chart update in real time

4. Go to the **Play** tab (`index.html`) to play against your trained agent:
   - Select "Policy Gradient AI" from one of the player dropdowns
   - The game will load the trained model from `models/policy_gradient_agent.pt`

5. Run the tests to make sure nothing is broken:
   ```
   python -m pytest tests/ -x -v
   ```

---

## 5. Adding Your Trained Agent to the Opponent Pool

Once your agent is trained, other agents can play against it. This is useful because training against a variety of opponents (not just the heuristic) produces stronger agents.

The system already supports this out of the box. When any agent is selected as an opponent in play mode, the server loads its saved weights from `models/<agent_id>_agent.pt`. So as long as your agent has been trained and its model file exists, it's already in the opponent pool.

But what if you want to use your trained agent as the **evaluation opponent during training** — so a new agent is measured against your Policy Gradient agent instead of the default heuristic?

You can do this by passing an `eval_opponent_factory` when creating your trainer. For example, here's how a future trainer could evaluate against your trained Policy Gradient agent:

```python
from src.agents.policy_gradient import PolicyGradientAgent

def make_pg_opponent():
    """Create a trained Policy Gradient agent to use as opponent."""
    agent = PolicyGradientAgent(model_path="models/policy_gradient_agent.pt")
    agent.set_train_mode(False)
    return agent

class MyNewTrainer(BaseTrainer):
    def __init__(self, agent):
        super().__init__(
            agent,
            eval_interval=50,
            eval_num_games=100,
            eval_opponent_factory=make_pg_opponent,  # evaluate against PG agent
        )
        # ... rest of init
```

Now during training, the periodic evaluation will pit the new agent against your trained Policy Gradient agent instead of the heuristic. The dashboard charts will reflect performance against this stronger opponent.

You can also evaluate any two agents against each other from a Python script:

```python
from src.training.evaluation import evaluate_agents
from src.agents.policy_gradient import PolicyGradientAgent
from src.agents.heuristic import HeuristicAgent

pg_agent = PolicyGradientAgent(model_path="models/policy_gradient_agent.pt")
heuristic = HeuristicAgent()

result = evaluate_agents(pg_agent, heuristic, num_rounds=1000)
print(f"Policy Gradient: {result.agent_0_avg_chips:+.2f} chips/round")
print(f"Heuristic:       {result.agent_1_avg_chips:+.2f} chips/round")
```

---

## Quick Reference

### Action enum

There are only 3 possible actions in Leduc Hold'em:

```python
class Action(IntEnum):
    FOLD  = 0   # Give up the hand. You lose whatever chips you've put in.
    CALL  = 1   # Match the opponent's bet (or check if bets are equal).
    RAISE = 2   # Put in more chips than the opponent. Max 2 raises per round.
```

### Observation fields

The `Observation` is a frozen dataclass (immutable) that contains everything your agent can see when it's time to make a decision. It is located in `src/engine/observation.py`.

Here's what each field means:

- **`player_hand`** (`str`) — Your private card. Always one of `'J'`, `'Q'`, or `'K'`. This is your secret card that only you can see.

- **`board`** (`str | None`) — The community card shared by both players. During pre-flop (round 0), this is `None` because no community card has been dealt yet. During flop (round 1), it's `'J'`, `'Q'`, or `'K'`. If your hand matches the board, you have a **pair**, which is very strong.

- **`pot`** (`list[int]`) — How many chips each player has put into the pot so far, as `[player_0_chips, player_1_chips]`. At the start of the game, both players put in 1 chip as an ante, so the initial pot is `[1, 1]`. The pot grows as players call and raise.

- **`current_player`** (`int`) — Whose turn it is to act: `0` or `1`. Your agent plays both sides during self-play training, so you'll see both values.

- **`current_round`** (`int`) — Which betting round we're in. `0` = pre-flop (no community card yet), `1` = flop (community card is visible). Each round has its own betting limit: 2 chips per raise in pre-flop, 4 chips per raise in flop.

- **`legal_actions`** (`list[Action]`) — The actions your agent is allowed to take right now. Always includes `FOLD` and `CALL`. Includes `RAISE` unless the max raises (2 per round) have been reached.

- **`is_finished`** (`bool`) — Whether the hand is over. If `True`, no more actions can be taken. This happens after a fold or after the final round of betting.

- **`raises_this_round`** (`int`) — How many raises have happened in the current round (0, 1, or 2). Once this reaches 2, `RAISE` is removed from `legal_actions`.

### Example: what observations look like during a game

Imagine a game where Player 0 has a Queen and Player 1 has a King. The deck deals a Jack as the community card.

**Turn 1 — Pre-flop, Player 0 acts first:**
```python
Observation(
    player_hand='Q',         # Player 0's private card
    board=None,              # No community card yet (pre-flop)
    pot=[1, 1],              # Both players anted 1 chip
    current_player=0,        # Player 0's turn
    current_round=0,         # Pre-flop
    legal_actions=[FOLD, CALL, RAISE],
    is_finished=False,
    raises_this_round=0,
)
```
Player 0 decides to CALL (check).

**Turn 2 — Pre-flop, Player 1 acts:**
```python
Observation(
    player_hand='K',         # Player 1's private card
    board=None,              # Still pre-flop
    pot=[1, 1],              # No chips added yet
    current_player=1,        # Player 1's turn
    current_round=0,
    legal_actions=[FOLD, CALL, RAISE],
    is_finished=False,
    raises_this_round=0,
)
```
Player 1 decides to CALL (check). Both players checked, so the round ends and we move to the flop. A community card (Jack) is dealt.

**Turn 3 — Flop, Player 0 acts first:**
```python
Observation(
    player_hand='Q',         # Still Player 0's Queen
    board='J',               # Community card is now visible!
    pot=[1, 1],              # Pot unchanged from pre-flop
    current_player=0,
    current_round=1,         # Now in flop round
    legal_actions=[FOLD, CALL, RAISE],
    is_finished=False,
    raises_this_round=0,
)
```
Player 0 decides to RAISE (bets 4 chips in the flop round).

**Turn 4 — Flop, Player 1 responds:**
```python
Observation(
    player_hand='K',         # Player 1's King
    board='J',               # Same community card
    pot=[5, 1],              # Player 0 put in 4 more chips (1 ante + 4 raise = 5)
    current_player=1,
    current_round=1,
    legal_actions=[FOLD, CALL, RAISE],  # Can still raise (only 1 raise so far)
    is_finished=False,
    raises_this_round=1,     # One raise has happened
)
```
Player 1 decides to CALL (matches Player 0's bet). The round ends and we go to showdown. Player 1 wins because K > Q (neither has a pair).

### BaseAgent API

```python
class BaseAgent(ABC):
    # --- You MUST implement ---
    @abstractmethod
    def select_action(self, obs: Observation) -> Action: ...

    # --- Optional overrides (have sensible defaults) ---
    def encode_observation(self, obs, **kwargs) -> Any: ...  # default: returns obs
    def get_action_evaluations(self, obs) -> list: ...       # default: []
    def save_model(self, path: str) -> None: ...             # default: no-op
    def load_model(self, path: str) -> None: ...             # default: no-op
    def set_train_mode(self, mode: bool) -> None: ...        # default: no-op
```

### BaseTrainer API

```python
class BaseTrainer(ABC):
    def __init__(self, agent, eval_interval=50, eval_num_games=100,
                 eval_opponent_factory=None): ...

    # --- You MUST implement ---
    @abstractmethod
    def collect_episode(self): ...                 # play one game, return data
    @abstractmethod
    def update_model(self, batch_data) -> float: ...  # learn from batch
    @abstractmethod
    def update_params(self, params: dict): ...     # handle live param changes

    # --- Already provided (you inherit these for free) ---
    def train(self, num_episodes, batch_size=32, save_path=None,
              callback=None, start_episode=0): ...
    def request_stop(self): ...
    def evaluate(self, num_games=None) -> float: ...
    def debug_episode(self) -> dict: ...  # raises NotImplementedError by default
```

---

## Troubleshooting

**Agent doesn't appear in dropdowns**
Make sure `registry.register()` runs inside `_register_builtin_agents()` in `registry.py`, and that the server is restarted after changes.

**"No trainer registered for agent type: X"**
Your `AgentMetadata` is missing `trainer_class`, or `is_trainable` is `False`.

**Training dashboard shows no data**
Make sure your `collect_episode()` and `update_model()` are working. The `BaseTrainer` loop calls the `callback` automatically — you don't need to call it yourself.

**Model not loading in play mode**
The server looks for model files at `models/<agent_id>_agent.pt`. Make sure training has completed (or been stopped after enough episodes) so the model file exists.

**"NotImplementedError: ... does not implement debug_episode()"**
Override `debug_episode()` in your trainer if you want the analyzer's episode trace feature. This is optional.
