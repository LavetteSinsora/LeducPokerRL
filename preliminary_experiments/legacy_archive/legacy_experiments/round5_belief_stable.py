"""
Round 5 — E2d: Belief Stable Agent

Implements and evaluates an agent that uses the OLD belief (b_t) instead
of the UPDATED belief (b_{t+1}) in the TD target, stripping out the
"value of belief change" from the learning signal.

Protocol:
  1. Train for 40K episodes via self-play
  2. Evaluate against all opponents (500 rounds each, both positions)
  3. Diagnostics:
     - Belief jump magnitude: |b_{t+1} - b_t| per game, correlated with TD error
     - TD chain integrity: predict value 2 steps ahead and compare
     - Training loss curves: stable vs standard belief TD (train standard too)
     - Action distribution per hand
  4. Save results + model
"""

import json
import os
import sys
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.belief_stable import BeliefStableAgent
from src.agents.belief_value import BeliefValueAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.belief_stable_trainer import BeliefStableTrainer
from src.training.belief_trainer import BeliefTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# -----------------------------------------------
# Configuration
# -----------------------------------------------

TRAIN_EPISODES = 40000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LIKELIHOOD_LR = 5e-4
EVAL_ROUNDS = 500
MODEL_PATH = "models/belief_stable_agent.pt"
RESULTS_PATH = "experiments/round5_belief_stable_results.json"

OPPONENTS = {
    "heuristic": {"class": "HeuristicAgent", "model_path": None},
    "value_based": {"class": "ValueBasedAgent", "model_path": "models/value_based_agent.pt"},
    "adaptive_value": {"class": "AdaptiveValueAgent", "model_path": "models/adaptive_value_agent.pt"},
    "modulated_value": {"class": "ModulatedValueAgent", "model_path": "models/modulated_value_agent.pt"},
    "entropy_ac": {"class": "EntropyACAgent", "model_path": "models/entropy_ac_agent.pt"},
    "cfr": {"class": "CFRAgent", "model_path": "models/cfr_agent.pt"},
}


# -----------------------------------------------
# Training
# -----------------------------------------------

def train_belief_stable_agent():
    """Train the Belief Stable Agent via self-play."""
    print("=" * 60)
    print("  TRAINING: Belief Stable Agent")
    print(f"  Episodes: {TRAIN_EPISODES}, Batch: {BATCH_SIZE}")
    print(f"  Value LR: {LEARNING_RATE}, Likelihood LR: {LIKELIHOOD_LR}")
    print("=" * 60)

    agent = BeliefStableAgent(temperature=1.0)
    trainer = BeliefStableTrainer(agent, learning_rate=LEARNING_RATE,
                                  likelihood_lr=LIKELIHOOD_LR)

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append({"episode": data["episode"], "loss": data["loss"]})
            if len(losses) % 100 == 0:
                print(f"    Update {len(losses)}: loss={data['loss']:.6f}")
        elif data["type"] == "evaluation":
            eval_scores.append({
                "episode": data["episode"],
                "avg_chips": data["avg_chips_per_round"]
            })

    start_time = time.time()
    trainer.train(
        num_episodes=TRAIN_EPISODES,
        batch_size=BATCH_SIZE,
        save_path=MODEL_PATH,
        callback=callback,
    )
    elapsed = time.time() - start_time

    print(f"\n  Training complete in {elapsed:.1f}s")
    if losses:
        print(f"  Final loss: {losses[-1]['loss']:.6f}")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]['avg_chips']:+.3f}")

    return {
        "training_time_s": round(elapsed, 1),
        "num_updates": len(losses),
        "final_loss": losses[-1]["loss"] if losses else None,
        "final_eval_vs_heuristic": eval_scores[-1]["avg_chips"] if eval_scores else None,
        "loss_curve": [l["loss"] for l in losses],
        "eval_history": eval_scores[-10:] if eval_scores else [],
    }


# -----------------------------------------------
# Evaluation
# -----------------------------------------------

def load_opponent(name: str, config: dict):
    """Dynamically load an opponent agent."""
    if name == "heuristic":
        return HeuristicAgent()
    elif name == "value_based":
        agent = ValueBasedAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        agent.set_train_mode(False)
        return agent
    elif name == "adaptive_value":
        from src.agents.adaptive_value import AdaptiveValueAgent
        agent = AdaptiveValueAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        agent.set_train_mode(False)
        return agent
    elif name == "modulated_value":
        from src.preliminary_experiments.promoted_registry.modulated_value import ModulatedValueAgent
        agent = ModulatedValueAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        agent.set_train_mode(False)
        return agent
    elif name == "entropy_ac":
        from src.agents.entropy_ac import EntropyACAgent
        agent = EntropyACAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        agent.set_train_mode(False)
        return agent
    elif name == "cfr":
        from src.agents.cfr_agent import CFRAgent
        agent = CFRAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        return agent
    else:
        raise ValueError(f"Unknown opponent: {name}")


def evaluate_against_opponents(agent: BeliefStableAgent):
    """Evaluate the belief stable agent against all opponents."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION ({EVAL_ROUNDS} rounds per matchup, both positions)")
    print("=" * 60)

    agent.set_train_mode(False)
    results = {}

    for name, config in OPPONENTS.items():
        try:
            opponent = load_opponent(name, config)
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}")
            continue

        result = evaluate_agents(agent, opponent, num_rounds=EVAL_ROUNDS)
        avg_chips = result.agent_0_avg_chips
        results[name] = round(avg_chips, 4)
        print(f"  vs {name:20s}: {avg_chips:+.4f} chips/round")

    robustness = compute_robustness_metrics(results)
    print(f"\n  Robustness metrics:")
    print(f"    Avg:       {robustness['avg']:+.4f}")
    print(f"    Worst:     {robustness['worst_case']:+.4f}")
    print(f"    Best:      {robustness['best_case']:+.4f}")
    print(f"    Std:       {robustness['std']:.4f}")
    print(f"    Robustness:{robustness['robustness']:+.4f}")

    return results, robustness


# -----------------------------------------------
# Diagnostics
# -----------------------------------------------

def diagnose_belief_jumps(agent: BeliefStableAgent, num_games: int = 1000):
    """
    Measure belief jump magnitudes |b_{t+1} - b_t| per game,
    and correlate with TD error to see if stable targets help.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 1: Belief Jump Magnitude vs TD Error")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    all_belief_jumps = []
    all_td_errors = []
    per_game_jumps = []

    for game_idx in range(num_games):
        game.reset()
        game_jumps = []
        prev_belief = None
        prev_value = None

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                # Belief agent's perspective
                belief = agent.compute_belief_from_history(obs)
                encoded = agent.encode_observation(obs, viewer_id=0, belief=belief)
                with torch.no_grad():
                    value = agent.model(encoded).item()

                if prev_belief is not None:
                    jump = np.linalg.norm(belief - prev_belief)
                    game_jumps.append(jump)
                    all_belief_jumps.append(jump)

                if prev_value is not None:
                    # TD error: |V(s_t) - V(s_{t+1})|
                    td_error = abs(prev_value - value)
                    all_td_errors.append(td_error)

                prev_belief = belief.copy()
                prev_value = value

                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            game.step(action)

        if game_jumps:
            per_game_jumps.append({
                "game": game_idx,
                "mean_jump": float(np.mean(game_jumps)),
                "max_jump": float(np.max(game_jumps)),
                "n_jumps": len(game_jumps),
            })

    # Compute statistics
    mean_jump = float(np.mean(all_belief_jumps)) if all_belief_jumps else 0
    std_jump = float(np.std(all_belief_jumps)) if all_belief_jumps else 0
    mean_td = float(np.mean(all_td_errors)) if all_td_errors else 0

    # Correlation between belief jumps and TD errors (aligned pairs)
    n_pairs = min(len(all_belief_jumps), len(all_td_errors))
    if n_pairs > 10:
        correlation = float(np.corrcoef(
            all_belief_jumps[:n_pairs],
            all_td_errors[:n_pairs]
        )[0, 1])
    else:
        correlation = 0.0

    print(f"  Mean belief jump:    {mean_jump:.4f}")
    print(f"  Std belief jump:     {std_jump:.4f}")
    print(f"  Mean TD error:       {mean_td:.4f}")
    print(f"  Jump-TD correlation: {correlation:.4f}")
    print(f"  Total jump events:   {len(all_belief_jumps)}")

    return {
        "mean_belief_jump": round(mean_jump, 4),
        "std_belief_jump": round(std_jump, 4),
        "mean_td_error": round(mean_td, 4),
        "jump_td_correlation": round(correlation, 4),
        "n_jump_events": len(all_belief_jumps),
    }


def diagnose_td_chain_integrity(agent: BeliefStableAgent, num_games: int = 500):
    """
    TD chain integrity: predict value 2 steps ahead and compare to actual.

    Does stable belief hurt multi-step predictions? Compare V(s_t) to
    the 2-step-ahead actual value to test whether the broken chain matters.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 2: TD Chain Integrity (2-step ahead)")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    two_step_errors = []
    one_step_errors = []

    for _ in range(num_games):
        game.reset()

        # Collect agent's value predictions as player 0
        agent_values = []
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                belief = agent.compute_belief_from_history(obs)
                encoded = agent.encode_observation(obs, viewer_id=0, belief=belief)
                with torch.no_grad():
                    val = agent.model(encoded).item()
                agent_values.append(val)
                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            game.step(action)

        rewards = game.get_reward()
        final_value = rewards[0]

        # 1-step errors: each prediction vs next prediction or terminal
        for i in range(len(agent_values)):
            if i == len(agent_values) - 1:
                target = final_value
            else:
                target = agent_values[i + 1]
            one_step_errors.append(abs(agent_values[i] - target))

        # 2-step errors: each prediction vs 2-ahead prediction or terminal
        for i in range(len(agent_values)):
            if i >= len(agent_values) - 2:
                target = final_value
            else:
                target = agent_values[i + 2]
            two_step_errors.append(abs(agent_values[i] - target))

    mean_1step = float(np.mean(one_step_errors)) if one_step_errors else 0
    mean_2step = float(np.mean(two_step_errors)) if two_step_errors else 0
    degradation = mean_2step / mean_1step if mean_1step > 0 else 0

    print(f"  Mean 1-step prediction error: {mean_1step:.4f}")
    print(f"  Mean 2-step prediction error: {mean_2step:.4f}")
    print(f"  Degradation ratio (2-step/1-step): {degradation:.4f}")
    print(f"  (Ratio > 2.0 suggests chain breakage)")

    return {
        "mean_1step_error": round(mean_1step, 4),
        "mean_2step_error": round(mean_2step, 4),
        "degradation_ratio": round(degradation, 4),
        "n_1step": len(one_step_errors),
        "n_2step": len(two_step_errors),
    }


def diagnose_loss_comparison(num_episodes: int = 5000):
    """
    Compare training loss curves: stable vs standard belief TD.

    Train both agents for a short period and compare loss curves.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 3: Training Loss Comparison (Stable vs Standard)")
    print("=" * 60)

    # Train standard belief agent
    print("\n  Training standard BeliefValueAgent...")
    standard_agent = BeliefValueAgent(temperature=1.0)
    standard_trainer = BeliefTrainer(standard_agent, learning_rate=LEARNING_RATE,
                                     likelihood_lr=LIKELIHOOD_LR)

    standard_losses = []

    def standard_callback(data):
        if data["type"] == "batch_update":
            standard_losses.append(data["loss"])

    standard_trainer.train(num_episodes=num_episodes, batch_size=BATCH_SIZE,
                          callback=standard_callback)

    # Train stable belief agent
    print("  Training stable BeliefStableAgent...")
    stable_agent = BeliefStableAgent(temperature=1.0)
    stable_trainer = BeliefStableTrainer(stable_agent, learning_rate=LEARNING_RATE,
                                         likelihood_lr=LIKELIHOOD_LR)

    stable_losses = []

    def stable_callback(data):
        if data["type"] == "batch_update":
            stable_losses.append(data["loss"])

    stable_trainer.train(num_episodes=num_episodes, batch_size=BATCH_SIZE,
                        callback=stable_callback)

    # Compare final losses
    std_final = np.mean(standard_losses[-10:]) if len(standard_losses) >= 10 else (np.mean(standard_losses) if standard_losses else 0)
    stb_final = np.mean(stable_losses[-10:]) if len(stable_losses) >= 10 else (np.mean(stable_losses) if stable_losses else 0)

    print(f"\n  Standard belief TD — final avg loss: {std_final:.6f}")
    print(f"  Stable belief TD   — final avg loss: {stb_final:.6f}")
    print(f"  Delta (stable - standard):           {stb_final - std_final:+.6f}")

    return {
        "standard_final_loss": round(float(std_final), 6),
        "stable_final_loss": round(float(stb_final), 6),
        "delta": round(float(stb_final - std_final), 6),
        "standard_loss_curve_last20": [round(l, 6) for l in standard_losses[-20:]],
        "stable_loss_curve_last20": [round(l, 6) for l in stable_losses[-20:]],
        "num_updates_standard": len(standard_losses),
        "num_updates_stable": len(stable_losses),
    }


def diagnose_action_distribution(agent: BeliefStableAgent, num_games: int = 2000):
    """
    Action distribution per hand for the stable agent.

    Break down what the agent does with each hand card (J, Q, K).
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 4: Action Distribution Per Hand")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    action_by_hand = {
        'J': {"FOLD": 0, "CALL": 0, "RAISE": 0},
        'Q': {"FOLD": 0, "CALL": 0, "RAISE": 0},
        'K': {"FOLD": 0, "CALL": 0, "RAISE": 0},
    }

    for _ in range(num_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                action = agent.select_action(obs)
                hand = obs.player_hand
                if hand in action_by_hand:
                    action_by_hand[hand][action.name] += 1
            else:
                action = heuristic.select_action(obs)

            game.step(action)

    # Normalize
    distributions = {}
    for hand, counts in action_by_hand.items():
        total = sum(counts.values())
        dist = {a: round(c / total, 4) if total > 0 else 0
                for a, c in counts.items()}
        distributions[hand] = dist
        print(f"  Hand {hand}: FOLD={dist['FOLD']:.3f}  CALL={dist['CALL']:.3f}  RAISE={dist['RAISE']:.3f}")

    return distributions


# -----------------------------------------------
# Main
# -----------------------------------------------

def main():
    print("=" * 60)
    print("  ROUND 5 — E2d: Belief Stable Agent")
    print("=" * 60)

    all_results = {
        "agent": "belief_stable",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "likelihood_lr": LIKELIHOOD_LR,
            "eval_rounds": EVAL_ROUNDS,
        },
    }

    # Phase 1: Training
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1: TRAINING")
    print(f"{'#' * 60}")
    training_results = train_belief_stable_agent()
    all_results["training"] = training_results

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")
    agent = BeliefStableAgent(model_path=MODEL_PATH)
    agent.set_train_mode(False)
    eval_results, robustness = evaluate_against_opponents(agent)
    all_results["evaluation"] = eval_results
    all_results["robustness"] = robustness

    # Phase 3: Diagnostics
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: DIAGNOSTICS")
    print(f"{'#' * 60}")

    diag1 = diagnose_belief_jumps(agent)
    all_results["diagnostics"] = {"belief_jumps": diag1}

    diag2 = diagnose_td_chain_integrity(agent)
    all_results["diagnostics"]["td_chain_integrity"] = diag2

    diag3 = diagnose_loss_comparison(num_episodes=5000)
    all_results["diagnostics"]["loss_comparison"] = diag3

    diag4 = diagnose_action_distribution(agent)
    all_results["diagnostics"]["action_distribution"] = diag4

    # Phase 4: Save results
    os.makedirs(os.path.dirname(RESULTS_PATH) if os.path.dirname(RESULTS_PATH) else ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Training time: {training_results['training_time_s']}s")
    print(f"  Final loss: {training_results.get('final_loss', 'N/A')}")
    print(f"\n  Evaluation (avg chips/round):")
    for name, score in eval_results.items():
        print(f"    vs {name:20s}: {score:+.4f}")
    print(f"\n  Robustness: {robustness['robustness']:+.4f}")
    print(f"  Avg: {robustness['avg']:+.4f}, Worst: {robustness['worst_case']:+.4f}")
    print(f"\n  Belief jumps (mean): {diag1['mean_belief_jump']:.4f}")
    print(f"  Jump-TD correlation: {diag1['jump_td_correlation']:.4f}")
    print(f"  TD chain degradation: {diag2['degradation_ratio']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
