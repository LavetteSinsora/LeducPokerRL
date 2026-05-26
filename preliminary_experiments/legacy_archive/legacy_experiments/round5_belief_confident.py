"""
Round 5 — E2c: Belief Confident Agent

Implements and evaluates an agent that extends BeliefValueAgent with a
confidence dimension encoding how reliable the belief estimate is.

Protocol:
  1. Train for 40K episodes via self-play
  2. Evaluate against all opponents (500 rounds each, both positions)
  3. Diagnostics:
     - Strategy change with confidence: compare action distributions at n_games=0 vs 30
     - Value predictions at different confidence levels for fixed game states
     - Ablation: performance when confidence is always 0 (belief ignored)
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
from src.agents.belief_confident import BeliefConfidentAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.belief_confident_trainer import BeliefConfidentTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# -----------------------------------------------
# Configuration
# -----------------------------------------------

TRAIN_EPISODES = 40000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LIKELIHOOD_LR = 5e-4
EVAL_ROUNDS = 500
MODEL_PATH = "models/belief_confident_agent.pt"
RESULTS_PATH = "experiments/round5_belief_confident_results.json"

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

def train_belief_confident_agent():
    """Train the Belief Confident Agent via self-play."""
    print("=" * 60)
    print("  TRAINING: Belief Confident Agent")
    print(f"  Episodes: {TRAIN_EPISODES}, Batch: {BATCH_SIZE}")
    print(f"  Value LR: {LEARNING_RATE}, Likelihood LR: {LIKELIHOOD_LR}")
    print("=" * 60)

    agent = BeliefConfidentAgent(temperature=1.0)
    trainer = BeliefConfidentTrainer(agent, learning_rate=LEARNING_RATE,
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


def evaluate_against_opponents(agent: BeliefConfidentAgent):
    """Evaluate the belief confident agent against all opponents."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION ({EVAL_ROUNDS} rounds per matchup, both positions)")
    print("=" * 60)

    agent.set_train_mode(False)
    # Set moderate confidence for evaluation (simulates mid-session)
    agent.set_game_count(15)
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

def diagnose_confidence_effect(agent: BeliefConfidentAgent, num_games: int = 1000):
    """
    Compare action distributions at confidence=0 (n_games=0) vs confidence=1 (n_games=30).

    For the same game states, does the agent behave differently when it
    trusts vs doesn't trust its belief?
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 1: Strategy Change with Confidence Level")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    # Track action distributions at different confidence levels
    action_counts_low = {"FOLD": 0, "CALL": 0, "RAISE": 0}
    action_counts_high = {"FOLD": 0, "CALL": 0, "RAISE": 0}

    for _ in range(num_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                # Test at low confidence
                agent.set_game_count(0)
                evals_low = agent.get_action_evaluations(obs)
                best_low = max(evals_low, key=lambda x: x["value"])
                action_counts_low[best_low["action"].name] += 1

                # Test at high confidence
                agent.set_game_count(30)
                evals_high = agent.get_action_evaluations(obs)
                best_high = max(evals_high, key=lambda x: x["value"])
                action_counts_high[best_high["action"].name] += 1

                # Actually play with high confidence
                action = best_high["action"]
            else:
                action = heuristic.select_action(obs)

            game.step(action)

    # Normalize to distributions
    total_low = sum(action_counts_low.values())
    total_high = sum(action_counts_high.values())

    dist_low = {k: round(v / total_low, 4) if total_low > 0 else 0
                for k, v in action_counts_low.items()}
    dist_high = {k: round(v / total_high, 4) if total_high > 0 else 0
                 for k, v in action_counts_high.items()}

    print(f"  Action distribution at confidence=0 (no trust):")
    for a, p in dist_low.items():
        print(f"    {a:>5s}: {p:.4f}")
    print(f"  Action distribution at confidence=1 (full trust):")
    for a, p in dist_high.items():
        print(f"    {a:>5s}: {p:.4f}")

    # Compute total variation distance
    tvd = sum(abs(dist_low[a] - dist_high[a]) for a in dist_low) / 2
    print(f"\n  Total variation distance: {tvd:.4f}")
    print(f"  (0 = identical strategies, 1 = completely different)")

    return {
        "action_dist_confidence_0": dist_low,
        "action_dist_confidence_1": dist_high,
        "total_variation_distance": round(tvd, 4),
        "total_decisions": total_low,
    }


def diagnose_value_by_confidence(agent: BeliefConfidentAgent):
    """
    For fixed game states, plot value predictions at different confidence levels.

    Shows how the value function responds to confidence for representative states.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 2: Value Predictions at Different Confidence Levels")
    print("=" * 60)

    agent.set_train_mode(False)

    # Create representative game states
    test_states = [
        {"hand": "K", "board": None, "pot": [1, 1], "round": 0, "raises": 0, "label": "K preflop, no action"},
        {"hand": "J", "board": None, "pot": [1, 1], "round": 0, "raises": 0, "label": "J preflop, no action"},
        {"hand": "K", "board": "K", "pot": [3, 3], "round": 1, "raises": 0, "label": "K pair, round 2"},
        {"hand": "J", "board": "K", "pot": [3, 3], "round": 1, "raises": 0, "label": "J vs K board, round 2"},
        {"hand": "Q", "board": None, "pot": [1, 3], "round": 0, "raises": 1, "label": "Q facing raise preflop"},
    ]

    confidence_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for state in test_states:
        obs = Observation(
            player_hand=state["hand"],
            board=state["board"],
            pot=state["pot"],
            current_player=0,
            current_round=state["round"],
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            raises_this_round=state["raises"],
        )

        belief = agent.compute_belief_from_history(obs)
        values_by_conf = {}

        print(f"\n  State: {state['label']}")
        print(f"    Belief: [J:{belief[0]:.3f}, Q:{belief[1]:.3f}, K:{belief[2]:.3f}]")

        for conf in confidence_levels:
            encoded = agent.encode_observation(obs, viewer_id=0, belief=belief,
                                                confidence=conf)
            with torch.no_grad():
                val = agent.model(encoded).item()
            values_by_conf[str(conf)] = round(val, 4)
            print(f"    conf={conf:.2f}: V={val:+.4f}")

        value_range = max(values_by_conf.values()) - min(values_by_conf.values())
        results.append({
            "label": state["label"],
            "belief": belief.tolist(),
            "values_by_confidence": values_by_conf,
            "value_range": round(value_range, 4),
        })

    return results


def diagnose_ablation_no_confidence(agent: BeliefConfidentAgent, num_games: int = 500):
    """
    Ablation: evaluate performance when confidence is always 0.

    This tests whether the agent has learned a reasonable base strategy
    when it doesn't trust the belief (confidence=0).
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 3: Ablation — Confidence Always 0")
    print("=" * 60)

    agent.set_train_mode(False)
    heuristic = HeuristicAgent()

    # Test with confidence = 0 (no trust in belief)
    agent.set_game_count(0)
    result_no_conf = evaluate_agents(agent, heuristic, num_rounds=num_games)
    avg_no_conf = result_no_conf.agent_0_avg_chips

    # Test with confidence = 1 (full trust)
    agent.set_game_count(30)
    result_full_conf = evaluate_agents(agent, heuristic, num_rounds=num_games)
    avg_full_conf = result_full_conf.agent_0_avg_chips

    # Test with confidence = 0.5 (moderate trust)
    agent.set_game_count(15)
    result_mid_conf = evaluate_agents(agent, heuristic, num_rounds=num_games)
    avg_mid_conf = result_mid_conf.agent_0_avg_chips

    print(f"  vs Heuristic ({num_games} rounds):")
    print(f"    Confidence=0.0 (ignore belief): {avg_no_conf:+.4f} chips/round")
    print(f"    Confidence=0.5 (moderate):      {avg_mid_conf:+.4f} chips/round")
    print(f"    Confidence=1.0 (trust belief):   {avg_full_conf:+.4f} chips/round")
    print(f"    Delta (high - low):              {avg_full_conf - avg_no_conf:+.4f}")

    return {
        "confidence_0": round(avg_no_conf, 4),
        "confidence_0.5": round(avg_mid_conf, 4),
        "confidence_1": round(avg_full_conf, 4),
        "delta_high_minus_low": round(avg_full_conf - avg_no_conf, 4),
        "num_games": num_games,
    }


# -----------------------------------------------
# Main
# -----------------------------------------------

def main():
    print("=" * 60)
    print("  ROUND 5 — E2c: Belief Confident Agent")
    print("=" * 60)

    all_results = {
        "agent": "belief_confident",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "likelihood_lr": LIKELIHOOD_LR,
            "eval_rounds": EVAL_ROUNDS,
            "confidence_cap": BeliefConfidentAgent.CONFIDENCE_CAP,
        },
    }

    # Phase 1: Training
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1: TRAINING")
    print(f"{'#' * 60}")
    training_results = train_belief_confident_agent()
    all_results["training"] = training_results

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")
    agent = BeliefConfidentAgent(model_path=MODEL_PATH)
    agent.set_train_mode(False)
    eval_results, robustness = evaluate_against_opponents(agent)
    all_results["evaluation"] = eval_results
    all_results["robustness"] = robustness

    # Phase 3: Diagnostics
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: DIAGNOSTICS")
    print(f"{'#' * 60}")

    diag1 = diagnose_confidence_effect(agent)
    all_results["diagnostics"] = {"confidence_effect": diag1}

    diag2 = diagnose_value_by_confidence(agent)
    all_results["diagnostics"]["value_by_confidence"] = diag2

    diag3 = diagnose_ablation_no_confidence(agent)
    all_results["diagnostics"]["ablation_no_confidence"] = diag3

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
    print(f"\n  Confidence effect (TVD): {diag1['total_variation_distance']:.4f}")
    print(f"  Ablation delta (high-low conf): {diag3['delta_high_minus_low']:+.4f}")

    return all_results


if __name__ == "__main__":
    main()
