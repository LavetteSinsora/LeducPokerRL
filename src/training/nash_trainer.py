"""
Nash Value Trainer -- supervised training from exact CFR equilibrium values.

Instead of learning from noisy self-play TD(0) targets, this trainer:
1. Runs CFR for N iterations to compute Nash equilibrium strategies
2. Traverses the full game tree under Nash play to get exact expected values
   for every information set
3. Converts each infoset key to the same 15-dim encoding used by ValueBasedAgent
4. Trains the value network via supervised MSE regression on (encoding, value) pairs

Since Leduc Hold'em has only ~288 information sets, the entire dataset fits in
memory and we can train for many epochs until convergence.
"""

import os
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.cfr.solver import LeducCFRSolver, _make_key, _legal_actions, _showdown
from src.cfr.solver import CARDS, CARD_VALUES, BET_AMOUNTS, MAX_RAISES, _generate_deals
from src.cfr.strategy import TabularStrategyStore, NUM_ACTIONS
from src.engine.leduc_game import Action
from src.engine.observation import Observation
from src.agents.nash_value import NashValueAgent


class NashTrainer:
    """Supervised trainer that teaches a value network exact Nash values.

    Usage:
        agent = NashValueAgent()
        trainer = NashTrainer(agent, cfr_iterations=10000)
        trainer.run_cfr()                 # Step 1: compute Nash equilibrium
        trainer.extract_nash_values()     # Step 2: get V(infoset) for all infosets
        trainer.build_dataset()           # Step 3: convert to (encoding, value) pairs
        trainer.train(epochs=10000)       # Step 4: supervised regression
        agent.save_model("models/nash_value_agent.pt")
    """

    def __init__(
        self,
        agent: NashValueAgent,
        cfr_iterations: int = 10000,
        learning_rate: float = 1e-3,
    ):
        self.agent = agent
        self.cfr_iterations = cfr_iterations
        self.learning_rate = learning_rate

        # CFR components
        self.store = TabularStrategyStore()
        self.solver = LeducCFRSolver(self.store)

        # Nash value storage: {infoset_key: expected_value_for_acting_player}
        # We store values for BOTH players separately
        self.nash_values: Dict[str, float] = {}  # key -> value for player who owns key

        # Dataset
        self.X: Optional[torch.Tensor] = None  # (N, 15)
        self.y: Optional[torch.Tensor] = None  # (N, 1)
        self.dataset_keys: List[str] = []       # for diagnostics

        # Optimizer
        self.optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Step 1: Run CFR to convergence
    # ------------------------------------------------------------------

    def run_cfr(self, callback: Optional[Callable] = None) -> Dict:
        """Run CFR+ for self.cfr_iterations iterations.

        Returns dict with exploitability trajectory.
        """
        print(f"Running CFR+ for {self.cfr_iterations} iterations...")
        exploitability_log = []

        for i in range(1, self.cfr_iterations + 1):
            game_val = self.solver.run_iteration(i)

            if i % 1000 == 0 or i == 1:
                expl = self.solver.compute_exploitability()
                exploitability_log.append({"iteration": i, "exploitability": expl})
                print(f"  Iteration {i:>6d}  |  exploitability = {expl:.6f}  |  game_val = {game_val:.6f}")

                if callback:
                    callback({"type": "cfr_progress", "iteration": i,
                              "exploitability": expl})

        final_expl = self.solver.compute_exploitability()
        num_infosets = self.store.num_info_sets()
        print(f"CFR complete: {num_infosets} infosets, exploitability = {final_expl:.6f}")

        return {
            "final_exploitability": final_expl,
            "num_infosets": num_infosets,
            "exploitability_log": exploitability_log,
        }

    # ------------------------------------------------------------------
    # Step 2: Extract Nash values for every information set
    # ------------------------------------------------------------------

    def extract_nash_values(self) -> Dict[str, float]:
        """Traverse the full game tree under Nash play, recording expected values.

        For each information set, we compute the expected value for the player
        who owns that infoset (the acting player), weighted by chance and
        opponent strategy probabilities.

        The value represents: "What is my expected payoff from this infoset
        onwards, if both players follow the Nash equilibrium strategy?"

        Returns:
            Dict mapping infoset keys to Nash expected values.
        """
        self.nash_values.clear()

        # We need to accumulate weighted values per infoset.
        # Each infoset may be reached via multiple deals (different opponent hands).
        # V(infoset) = sum_deals [ P(deal | reaching infoset) * V(deal, infoset) ]
        #
        # We accumulate (weighted_value, weight) and normalize at the end.
        value_accum: Dict[str, float] = {}
        weight_accum: Dict[str, float] = {}

        deals = _generate_deals()

        for p0_hand, p1_hand, board_card, chance_prob in deals:
            self._nash_traverse(
                p0_hand, p1_hand, board_card, chance_prob,
                "", "", 0, 0, 1, 1, 0,
                1.0, 1.0,  # reach probabilities
                value_accum, weight_accum,
            )

        # Normalize
        for key in value_accum:
            if weight_accum[key] > 0:
                self.nash_values[key] = value_accum[key] / weight_accum[key]
            else:
                self.nash_values[key] = 0.0

        print(f"Extracted Nash values for {len(self.nash_values)} information sets")
        return self.nash_values

    def _nash_traverse(
        self,
        p0_hand: str, p1_hand: str, board: str, chance_prob: float,
        preflop: str, flop: str,
        rnd: int, player: int,
        pot0: int, pot1: int, raises: int,
        r0: float, r1: float,
        value_accum: Dict[str, float],
        weight_accum: Dict[str, float],
    ) -> float:
        """Traverse game tree under Nash strategies, recording per-infoset values.

        Returns expected value for Player 0 from this node onwards.
        """
        hand = p0_hand if player == 0 else p1_hand
        key = _make_key(hand, board, preflop, flop, rnd)
        legal = _legal_actions(raises)

        # Get Nash (average) strategy
        strategy = self.store.get_average_strategy(key, legal)

        action_vals = np.zeros(NUM_ACTIONS, dtype=np.float64)
        node_val = 0.0

        for action in legal:
            a = action.value
            new_r0 = r0 * (strategy[a] if player == 0 else 1.0)
            new_r1 = r1 * (strategy[a] if player == 1 else 1.0)

            v = self._nash_apply(
                action, p0_hand, p1_hand, board, chance_prob,
                preflop, flop, rnd, player,
                pot0, pot1, raises,
                new_r0, new_r1,
                value_accum, weight_accum,
            )
            action_vals[a] = v
            node_val += strategy[a] * v

        # Record value for the acting player at this infoset.
        # The "value for the acting player" is:
        #   - node_val (if acting player is P0, node_val is already P0's value)
        #   - -node_val (if acting player is P1, since node_val is P0's value)
        acting_player_value = node_val if player == 0 else -node_val

        # Weight by chance probability and opponent's reach probability
        # This gives us the probability of reaching this specific game state
        # from the perspective of the acting player.
        opp_reach = r1 if player == 0 else r0
        weight = chance_prob * opp_reach

        if key not in value_accum:
            value_accum[key] = 0.0
            weight_accum[key] = 0.0

        value_accum[key] += weight * acting_player_value
        weight_accum[key] += weight

        return node_val

    def _nash_apply(
        self,
        action: Action,
        p0_hand: str, p1_hand: str, board: str, chance_prob: float,
        preflop: str, flop: str,
        rnd: int, player: int,
        pot0: int, pot1: int, raises: int,
        r0: float, r1: float,
        value_accum: Dict[str, float],
        weight_accum: Dict[str, float],
    ) -> float:
        """Apply action under Nash traversal. Mirrors solver._apply logic."""
        code = "fcr"[action.value]
        pf = preflop + code if rnd == 0 else preflop
        fl = flop + code if rnd == 1 else flop

        # Fold
        if action == Action.FOLD:
            return -pot0 if player == 0 else pot1

        other_pot = pot1 if player == 0 else pot0
        new_pot0, new_pot1 = pot0, pot1

        # Raise
        if action == Action.RAISE:
            new_my = other_pot + BET_AMOUNTS[rnd]
            if player == 0:
                new_pot0 = new_my
            else:
                new_pot1 = new_my
            return self._nash_traverse(
                p0_hand, p1_hand, board, chance_prob,
                pf, fl, rnd, 1 - player,
                new_pot0, new_pot1, raises + 1,
                r0, r1,
                value_accum, weight_accum,
            )

        # Call / Check
        my_pot = pot0 if player == 0 else pot1
        round_ended = False
        if other_pot > my_pot:
            if player == 0:
                new_pot0 = other_pot
            else:
                new_pot1 = other_pot
            round_ended = True
        elif player == 1:
            round_ended = True

        if not round_ended:
            return self._nash_traverse(
                p0_hand, p1_hand, board, chance_prob,
                pf, fl, rnd, 1 - player,
                new_pot0, new_pot1, raises,
                r0, r1,
                value_accum, weight_accum,
            )

        # Round ended
        if rnd == 0:
            return self._nash_traverse(
                p0_hand, p1_hand, board, chance_prob,
                pf, "", 1, 0,
                new_pot0, new_pot1, 0,
                r0, r1,
                value_accum, weight_accum,
            )

        return _showdown(p0_hand, p1_hand, board, new_pot0, new_pot1)

    # ------------------------------------------------------------------
    # Step 3: Build supervised dataset
    # ------------------------------------------------------------------

    def build_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Nash infoset values to (encoding, value) pairs.

        Each infoset key is parsed to reconstruct the game state, then
        encoded using ValueBasedAgent's 15-dim encoding.

        Returns:
            (X, y) tensors of shape (N, 15) and (N, 1).
        """
        if not self.nash_values:
            raise RuntimeError("Must call extract_nash_values() first")

        encodings = []
        values = []
        keys = []

        for key, value in self.nash_values.items():
            # Parse infoset key to reconstruct observation
            obs, viewer_id = self._key_to_observation(key)
            if obs is None:
                continue

            enc = self.agent.encode_observation(obs, viewer_id=viewer_id)
            encodings.append(enc.squeeze(0))
            values.append(value)
            keys.append(key)

        self.X = torch.stack(encodings)  # (N, 15)
        self.y = torch.tensor(values, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        self.dataset_keys = keys

        print(f"Built dataset: {self.X.shape[0]} samples, encoding dim = {self.X.shape[1]}")
        return self.X, self.y

    def _key_to_observation(self, key: str) -> Tuple[Optional[Observation], int]:
        """Parse a CFR infoset key back into an Observation + viewer_id.

        Key formats:
            Pre-flop: "{hand}:{preflop_actions}"
            Flop:     "{hand}:{board}:{preflop_actions}/{flop_actions}"

        We need to:
        1. Extract hand, board, and action sequences
        2. Replay the action sequence to reconstruct pot sizes, round, etc.
        3. Determine who the acting player (viewer) is
        """
        try:
            parts = key.split(":")
            hand = parts[0]

            if len(parts) == 2:
                # Pre-flop: "H:actions"
                board = None
                preflop_actions = parts[1]
                flop_actions = ""
                current_round = 0
            elif len(parts) == 3:
                # Flop: "H:B:preflop/flop"
                board = parts[1]
                action_parts = parts[2].split("/")
                preflop_actions = action_parts[0]
                flop_actions = action_parts[1] if len(action_parts) > 1 else ""
                current_round = 1
            else:
                return None, 0

            # Replay action sequence to get pot, player, raises
            pot0, pot1 = 1, 1
            player = 0
            rnd = 0
            raises = 0
            action_history = []

            all_actions = preflop_actions + ("/" + flop_actions if current_round == 1 else "")

            for ch in all_actions:
                if ch == "/":
                    rnd = 1
                    player = 0
                    raises = 0
                    continue

                code_to_name = {"f": "FOLD", "c": "CALL", "r": "RAISE"}
                action_name = code_to_name.get(ch)
                if action_name is None:
                    continue

                action_history.append((player, action_name))

                if action_name == "FOLD":
                    break

                if action_name == "RAISE":
                    other_pot = pot1 if player == 0 else pot0
                    new_my = other_pot + BET_AMOUNTS[rnd]
                    if player == 0:
                        pot0 = new_my
                    else:
                        pot1 = new_my
                    raises += 1
                    player = 1 - player
                else:  # CALL/CHECK
                    other_pot = pot1 if player == 0 else pot0
                    my_pot = pot0 if player == 0 else pot1
                    round_ended = False
                    if other_pot > my_pot:
                        if player == 0:
                            pot0 = other_pot
                        else:
                            pot1 = other_pot
                        round_ended = True
                    elif player == 1:
                        round_ended = True

                    if round_ended and rnd == 0:
                        rnd = 1
                        player = 0
                        raises = 0
                    elif not round_ended:
                        player = 1 - player

            # The viewer_id is the current acting player at this infoset
            viewer_id = player

            # Determine legal actions
            legal = [Action.FOLD, Action.CALL]
            if raises < MAX_RAISES:
                legal.append(Action.RAISE)

            obs = Observation(
                player_hand=hand,
                board=board,
                pot=[pot0, pot1],
                current_player=player,
                current_round=current_round,
                legal_actions=legal,
                is_finished=False,
                raises_this_round=raises,
                action_history=tuple(tuple(h) for h in action_history),
            )

            return obs, viewer_id

        except Exception as e:
            print(f"Warning: Could not parse key '{key}': {e}")
            return None, 0

    # ------------------------------------------------------------------
    # Step 4: Train neural network via supervised regression
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int = 10000,
        log_interval: int = 1000,
        callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Train the value network on the supervised dataset.

        Since we have only ~288 samples, we use full-batch gradient descent.
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Must call build_dataset() first")

        self.agent.model.train()
        loss_log = []

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            predictions = self.agent.model(self.X)
            loss = self.criterion(predictions, self.y)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()

            if epoch % log_interval == 0 or epoch == 1:
                loss_log.append({"epoch": epoch, "loss": loss_val})
                print(f"  Epoch {epoch:>6d}  |  MSE = {loss_val:.8f}")

                if callback:
                    callback({"type": "training_progress", "epoch": epoch, "loss": loss_val})

        self.agent.model.eval()

        final_loss = loss_log[-1]["loss"] if loss_log else 0.0
        print(f"Training complete. Final MSE = {final_loss:.8f}")

        return loss_log

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        cfr_iterations: int = None,
        epochs: int = 10000,
        save_path: str = "models/nash_value_agent.pt",
        callback: Optional[Callable] = None,
    ) -> Dict:
        """Run the complete Nash training pipeline.

        1. CFR to convergence
        2. Extract Nash values
        3. Build dataset
        4. Supervised training
        5. Save model
        """
        if cfr_iterations:
            self.cfr_iterations = cfr_iterations

        # Step 1
        cfr_result = self.run_cfr(callback=callback)

        # Step 2
        self.extract_nash_values()

        # Step 3
        self.build_dataset()

        # Step 4
        loss_log = self.train(epochs=epochs, callback=callback)

        # Step 5
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

        return {
            "cfr": cfr_result,
            "num_infosets": len(self.nash_values),
            "dataset_size": self.X.shape[0] if self.X is not None else 0,
            "final_mse": loss_log[-1]["loss"] if loss_log else None,
            "loss_log": loss_log,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_value_comparison(self, other_agent) -> List[Dict]:
        """Compare Nash values vs another agent's value predictions.

        Returns per-infoset comparison for analysis.
        """
        if self.X is None:
            raise RuntimeError("Must call build_dataset() first")

        comparisons = []
        self.agent.model.eval()
        other_agent.model.eval()

        for i, key in enumerate(self.dataset_keys):
            enc = self.X[i].unsqueeze(0)
            nash_true = self.y[i].item()

            with torch.no_grad():
                nash_pred = self.agent.model(enc).item()
                other_pred = other_agent.model(enc).item()

            comparisons.append({
                "key": key,
                "nash_true": round(nash_true, 6),
                "nash_pred": round(nash_pred, 6),
                "other_pred": round(other_pred, 6),
                "nash_error": round(abs(nash_pred - nash_true), 6),
                "divergence": round(abs(nash_true - other_pred), 6),
            })

        # Sort by divergence (most different first)
        comparisons.sort(key=lambda x: -x["divergence"])

        return comparisons
