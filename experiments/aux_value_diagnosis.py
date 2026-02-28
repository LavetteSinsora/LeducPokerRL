"""
Aux-Value Overestimation Diagnosis Experiments.

Three experiments to test whether the max operator in aux_value
causes systematic overestimation:

Experiment A: Track mean predicted values over training on a fixed probe set.
Experiment B: Compare max vs mean vs on-policy aux targets.
Experiment C: Measure signed prediction error at convergence.

Plus visualization of value estimates over training time.
"""

import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.value_based import ValueBasedAgent, ValueNetwork
from src.agents.aux_value import AuxValueAgent
from src.training.value_based_trainer import SelfPlayTrainer
from src.training.aux_value_trainer import AuxValueTrainer
from src.training.base import BaseTrainer
from src.training.evaluation import quick_evaluate
from src.agents.heuristic import HeuristicAgent


# ──────────────────────────────────────────────
# Probe Set Generation
# ──────────────────────────────────────────────

def generate_probe_set(n_states=50, seed=42):
    """Generate a fixed set of diverse game states for value tracking.

    Plays random games and collects states at various points.
    Returns a list of (encoded_tensor, description) pairs.
    """
    torch.manual_seed(seed)
    import random
    random.seed(seed)

    game = LeducGame()
    encoder = ValueBasedAgent()  # just for encoding
    probes = []
    descriptions = []

    while len(probes) < n_states:
        game.reset()
        while not game.is_finished:
            player = game.current_player
            obs = game.get_observation(viewer_id=player)

            # Record this state
            encoded = encoder.encode_observation(obs, viewer_id=player)
            desc = f"hand={obs.player_hand} board={obs.board} pot={obs.pot} round={obs.current_round}"
            probes.append(encoded)
            descriptions.append(desc)

            if len(probes) >= n_states:
                break

            # Take random action
            action = random.choice(obs.legal_actions)
            game.step(action)

    return probes[:n_states], descriptions[:n_states]


def evaluate_probe_set(model, probes):
    """Compute mean and individual V(s) for probe states."""
    model.eval()
    values = []
    with torch.no_grad():
        for enc in probes:
            v = model(enc).item()
            values.append(v)
    return values


# ──────────────────────────────────────────────
# Variant Trainers for Experiment B
# ──────────────────────────────────────────────

class MeanAuxTrainer(AuxValueTrainer):
    """Aux trainer that uses mean_a V(post_a) instead of max_a."""

    def update_model(self, batch_data: list) -> float:
        self.optimizer.zero_grad()
        total_losses = []

        for chains, pre_action_data, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                # Main TD(0) loss (identical)
                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]).squeeze(0)
                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = self.agent.model(chain[t + 1]).squeeze(0)
                    total_losses.append(self.criterion(prediction, target))

                # Aux loss: MEAN instead of MAX
                for pre_encoded, post_encodeds in pre_action_data[p_idx]:
                    with torch.no_grad():
                        post_vals = torch.stack([
                            self.agent.model(enc).squeeze() for enc in post_encodeds
                        ])
                        mean_post_val = post_vals.mean().unsqueeze(0)  # <-- MEAN not MAX

                    pre_val = self.agent.model(pre_encoded).squeeze(0)
                    aux_loss = self.criterion(pre_val, mean_post_val) * self.aux_weight
                    total_losses.append(aux_loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0


class OnPolicyAuxTrainer(BaseTrainer):
    """Your proposed architecture: unified chain pre→post→pre→...→reward.

    Single loss: V(pre_t) → V(post_t_chosen), V(post_t) → V(pre_{t+1}).
    No max operator. On-policy action only.
    """

    def __init__(self, agent, learning_rate=1e-4):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()

    def collect_episode(self):
        """Play one episode. Return full chain of alternating pre/post states + rewards."""
        self.game.reset()

        # Build per-player chains of (pre_encoded, post_encoded_chosen) pairs
        chains = [[], []]  # chains[p] = list of (pre_encoded, post_encoded)

        while not self.game.is_finished:
            player = self.game.current_player
            obs = self.game.get_observation(viewer_id=player)

            pre_encoded = self.agent.encode_observation(obs, viewer_id=player)

            # Select action (Boltzmann)
            evaluations = self.agent.get_action_evaluations(obs)
            values = torch.tensor([e["value"] for e in evaluations])
            if self.agent.train_mode:
                probs = torch.softmax(values / self.agent.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = int(values.argmax().item())

            selected = evaluations[idx]
            post_encoded = selected["encoded"]
            action = selected["action"]

            chains[player].append((pre_encoded, post_encoded))
            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards

    def update_model(self, batch_data):
        """Unified chain loss: pre→post→pre→...→reward."""
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                reward = rewards[p_idx]

                for t in range(len(chain)):
                    pre_enc, post_enc = chain[t]

                    # Loss 1: V(pre_t) → V(post_t_chosen) [detached target]
                    pre_val = self.agent.model(pre_enc).squeeze(0)
                    with torch.no_grad():
                        post_target = self.agent.model(post_enc).squeeze(0)
                    total_losses.append(self.criterion(pre_val, post_target))

                    # Loss 2: V(post_t) → V(pre_{t+1}) or terminal reward
                    post_val = self.agent.model(post_enc).squeeze(0)
                    if t == len(chain) - 1:
                        target = torch.FloatTensor([reward])
                    else:
                        with torch.no_grad():
                            next_pre_enc = chain[t + 1][0]
                            target = self.agent.model(next_pre_enc).squeeze(0)
                    total_losses.append(self.criterion(post_val, target))

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def update_params(self, params):
        if "lr" in params:
            for pg in self.optimizer.param_groups:
                pg["lr"] = params["lr"]


# ──────────────────────────────────────────────
# Experiment A: Value Drift Tracking
# ──────────────────────────────────────────────

def experiment_a(num_episodes=5000, probe_interval=100, seed=42):
    """Track mean predicted values on fixed probe set during training.

    Compares: plain value_based vs aux_value (max)
    to see if aux_value exhibits systematic overestimation.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT A: Value Drift Tracking")
    print("  Does aux_value overestimate compared to value_based?")
    print("=" * 60 + "\n")

    torch.manual_seed(seed)
    probes, descriptions = generate_probe_set(n_states=50, seed=seed)

    results = {}

    for label, agent_cls, trainer_cls in [
        ("value_based", ValueBasedAgent, SelfPlayTrainer),
        ("aux_value_max", AuxValueAgent, AuxValueTrainer),
    ]:
        print(f"\n--- Training {label} ---")
        torch.manual_seed(seed)
        agent = agent_cls()

        if trainer_cls == SelfPlayTrainer:
            trainer = trainer_cls(agent, learning_rate=1e-4)
        else:
            trainer = trainer_cls(agent, learning_rate=1e-4, aux_weight=0.5)

        value_history = []
        episode_counter = [0]

        # Measure initial values
        initial_vals = evaluate_probe_set(agent.model, probes)
        value_history.append({
            "episode": 0,
            "mean_value": sum(initial_vals) / len(initial_vals),
            "max_value": max(initial_vals),
            "min_value": min(initial_vals),
            "std_value": (sum((v - sum(initial_vals)/len(initial_vals))**2 for v in initial_vals) / len(initial_vals)) ** 0.5
        })

        def track_callback(data):
            if data["type"] == "batch_update":
                ep = data["episode"]
                if ep % probe_interval == 0:
                    vals = evaluate_probe_set(agent.model, probes)
                    mean_v = sum(vals) / len(vals)
                    value_history.append({
                        "episode": ep,
                        "mean_value": mean_v,
                        "max_value": max(vals),
                        "min_value": min(vals),
                        "std_value": (sum((v - mean_v)**2 for v in vals) / len(vals)) ** 0.5
                    })
                    if ep % 500 == 0:
                        print(f"  Ep {ep:5d}: mean_V = {mean_v:+.4f}, range = [{min(vals):+.4f}, {max(vals):+.4f}]")

        trainer.train(
            num_episodes=num_episodes,
            batch_size=32,
            callback=track_callback,
        )

        results[label] = value_history

    # Print comparison
    print("\n" + "-" * 60)
    print("  VALUE DRIFT COMPARISON")
    print("-" * 60)
    print(f"{'Episode':>8s}  {'value_based':>14s}  {'aux_value_max':>14s}  {'Difference':>12s}")
    print("-" * 52)

    vb_points = {h["episode"]: h["mean_value"] for h in results["value_based"]}
    av_points = {h["episode"]: h["mean_value"] for h in results["aux_value_max"]}

    all_episodes = sorted(set(vb_points.keys()) | set(av_points.keys()))
    for ep in all_episodes:
        if ep in vb_points and ep in av_points:
            vb = vb_points[ep]
            av = av_points[ep]
            print(f"{ep:>8d}  {vb:>+14.4f}  {av:>+14.4f}  {av - vb:>+12.4f}")

    return results


# ──────────────────────────────────────────────
# Experiment B: Max vs Mean vs On-Policy
# ──────────────────────────────────────────────

def experiment_b(num_episodes=5000, seed=42):
    """Compare three aux target operators: max, mean, on-policy.

    All start from same random init. Tracks probe values + final eval.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT B: Max vs Mean vs On-Policy Aux Targets")
    print("  Which operator produces the most accurate value estimates?")
    print("=" * 60 + "\n")

    torch.manual_seed(seed)
    probes, _ = generate_probe_set(n_states=50, seed=seed)

    # Save initial weights for identical initialization
    init_agent = AuxValueAgent()
    torch.manual_seed(seed)
    init_weights = copy.deepcopy(init_agent.model.state_dict())

    results = {}
    opponent = HeuristicAgent()

    for label, trainer_cls in [
        ("plain_td0", None),  # No aux loss baseline
        ("aux_max", AuxValueTrainer),
        ("aux_mean", MeanAuxTrainer),
        ("on_policy_chain", OnPolicyAuxTrainer),
    ]:
        print(f"\n--- Training {label} ---")
        torch.manual_seed(seed)

        if label == "plain_td0":
            agent = ValueBasedAgent()
        else:
            agent = AuxValueAgent()

        agent.model.load_state_dict(copy.deepcopy(init_weights))

        if label == "plain_td0":
            trainer = SelfPlayTrainer(agent, learning_rate=1e-4)
        elif label == "on_policy_chain":
            trainer = OnPolicyAuxTrainer(agent, learning_rate=1e-4)
        else:
            trainer = trainer_cls(agent, learning_rate=1e-4, aux_weight=0.5)

        value_snapshots = []

        def make_callback(ag, snaps):
            def cb(data):
                if data["type"] == "batch_update" and data["episode"] % 500 == 0:
                    vals = evaluate_probe_set(ag.model, probes)
                    mean_v = sum(vals) / len(vals)
                    snaps.append({"episode": data["episode"], "mean_value": mean_v})
                    print(f"  Ep {data['episode']:5d}: mean_V = {mean_v:+.4f}")
            return cb

        trainer.train(
            num_episodes=num_episodes,
            batch_size=32,
            callback=make_callback(agent, value_snapshots),
        )

        # Final evaluation
        agent.set_train_mode(False)
        eval_score = quick_evaluate(agent, opponent, num_rounds=200)
        final_vals = evaluate_probe_set(agent.model, probes)
        final_mean = sum(final_vals) / len(final_vals)

        results[label] = {
            "value_snapshots": value_snapshots,
            "final_mean_value": final_mean,
            "eval_vs_heuristic": eval_score,
        }

        print(f"  Final: mean_V = {final_mean:+.4f}, eval = {eval_score:+.4f}")

    # Print comparison table
    print("\n" + "-" * 60)
    print("  EXPERIMENT B RESULTS")
    print("-" * 60)
    print(f"{'Variant':>20s}  {'Final mean_V':>14s}  {'Eval vs Heur':>14s}")
    print("-" * 52)
    for label, r in results.items():
        print(f"{label:>20s}  {r['final_mean_value']:>+14.4f}  {r['eval_vs_heuristic']:>+14.4f}")

    return results


# ──────────────────────────────────────────────
# Experiment C: Signed Prediction Error
# ──────────────────────────────────────────────

def experiment_c(num_games=500, seed=42):
    """Measure signed prediction error for trained agents.

    Plays games and compares V(s) predictions to actual rewards.
    Positive mean error = overestimation.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT C: Signed Prediction Error")
    print("  Do trained agents overestimate or underestimate?")
    print("=" * 60 + "\n")

    game = LeducGame()
    results = {}

    agents_to_test = {}

    # Load value_based
    vb = ValueBasedAgent()
    vb_path = "models/value_based_agent.pt"
    if os.path.exists(vb_path):
        vb.load_model(vb_path)
    vb.set_train_mode(False)
    agents_to_test["value_based"] = vb

    # Load aux_value
    av = AuxValueAgent()
    av_path = "models/aux_value_agent.pt"
    if os.path.exists(av_path):
        # aux_value model might not exist if not trained in this round
        # Check for older model or use the value_based path
        av.load_model(av_path)
    av.set_train_mode(False)
    agents_to_test["aux_value"] = av

    for label, agent in agents_to_test.items():
        print(f"\n--- {label} ---")
        signed_errors = []
        abs_errors = []
        predictions = []
        actuals = []

        torch.manual_seed(seed)
        import random
        random.seed(seed)

        for game_num in range(num_games):
            game.reset()
            game_predictions = []

            while not game.is_finished:
                player = game.current_player
                obs = game.get_observation(viewer_id=player)

                # Record prediction for the chosen action
                evals = agent.get_action_evaluations(obs)
                best_eval = max(evals, key=lambda x: x["value"])
                game_predictions.append({
                    "player": player,
                    "value": best_eval["value"],
                })

                # Take greedy action
                game.step(best_eval["action"])

            rewards = game.get_reward()

            for pred in game_predictions:
                actual = rewards[pred["player"]]
                error = pred["value"] - actual
                signed_errors.append(error)
                abs_errors.append(abs(error))
                predictions.append(pred["value"])
                actuals.append(actual)

        mean_signed = sum(signed_errors) / len(signed_errors)
        mean_abs = sum(abs_errors) / len(abs_errors)
        mean_pred = sum(predictions) / len(predictions)
        mean_actual = sum(actuals) / len(actuals)

        results[label] = {
            "mean_signed_error": round(mean_signed, 4),
            "mean_abs_error": round(mean_abs, 4),
            "mean_prediction": round(mean_pred, 4),
            "mean_actual": round(mean_actual, 4),
            "n_samples": len(signed_errors),
        }

        print(f"  Mean signed error:  {mean_signed:+.4f} (positive = overestimation)")
        print(f"  Mean abs error:     {mean_abs:.4f}")
        print(f"  Mean prediction:    {mean_pred:+.4f}")
        print(f"  Mean actual reward: {mean_actual:+.4f}")
        print(f"  Samples:            {len(signed_errors)}")

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AUX-VALUE OVERESTIMATION DIAGNOSIS")
    print("=" * 60)

    all_results = {}

    # Experiment A: Value drift tracking
    all_results["experiment_a"] = experiment_a(num_episodes=5000)

    # Experiment B: Max vs Mean vs On-Policy
    all_results["experiment_b"] = experiment_b(num_episodes=5000)

    # Experiment C: Signed prediction error on trained models
    all_results["experiment_c"] = experiment_c(num_games=500)

    # Save results
    results_path = "experiments/aux_value_diagnosis_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    main()
