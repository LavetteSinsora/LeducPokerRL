"""
Round 4 — Direction 1: Bayesian Belief Agent

Implements and evaluates an agent that maintains an explicit belief
distribution over the opponent's hand card, updated within each hand
using a learned likelihood model P(action | hand, state).

Protocol:
  1. Train for 30K episodes (1000 sessions x 30 hands) via self-play
  2. Evaluate against 6 opponents (500 rounds each, both positions)
  3. Diagnose: likelihood accuracy, belief evolution, bluff-catching
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
from src.agents.belief_value import BeliefValueAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.belief_trainer import BeliefTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TRAIN_EPISODES = 30000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LIKELIHOOD_LR = 5e-4
EVAL_ROUNDS = 500
MODEL_PATH = "models/belief_value_agent.pt"
RESULTS_PATH = "experiments/round4_belief_results.json"

# Opponents for evaluation
OPPONENTS = {
    "heuristic": {"class": "HeuristicAgent", "model_path": None},
    "value_based": {"class": "ValueBasedAgent", "model_path": "models/value_based_agent.pt"},
    "adaptive_value": {"class": "AdaptiveValueAgent", "model_path": "models/adaptive_value_agent.pt"},
    "modulated_value": {"class": "ModulatedValueAgent", "model_path": "models/modulated_value_agent.pt"},
    "entropy_ac": {"class": "EntropyACAgent", "model_path": "models/entropy_ac_agent.pt"},
    "cfr": {"class": "CFRAgent", "model_path": "models/cfr_agent.pt"},
}


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_belief_agent():
    """Train the Bayesian Belief Agent via self-play."""
    print("=" * 60)
    print("  TRAINING: Bayesian Belief Agent")
    print(f"  Episodes: {TRAIN_EPISODES}, Batch: {BATCH_SIZE}")
    print(f"  Value LR: {LEARNING_RATE}, Likelihood LR: {LIKELIHOOD_LR}")
    print("=" * 60)

    agent = BeliefValueAgent(temperature=1.0)
    trainer = BeliefTrainer(agent, learning_rate=LEARNING_RATE,
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
        "eval_history": eval_scores[-10:] if eval_scores else [],  # Last 10 evals
    }


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

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


def evaluate_against_opponents(belief_agent: BeliefValueAgent):
    """Evaluate the belief agent against all opponents."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION ({EVAL_ROUNDS} rounds per matchup, both positions)")
    print("=" * 60)

    belief_agent.set_train_mode(False)
    results = {}

    for name, config in OPPONENTS.items():
        try:
            opponent = load_opponent(name, config)
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}")
            continue

        result = evaluate_agents(belief_agent, opponent, num_rounds=EVAL_ROUNDS)
        avg_chips = result.agent_0_avg_chips
        results[name] = round(avg_chips, 4)
        print(f"  vs {name:20s}: {avg_chips:+.4f} chips/round")

    # Compute robustness
    robustness = compute_robustness_metrics(results)
    print(f"\n  Robustness metrics:")
    print(f"    Avg:       {robustness['avg']:+.4f}")
    print(f"    Worst:     {robustness['worst_case']:+.4f}")
    print(f"    Best:      {robustness['best_case']:+.4f}")
    print(f"    Std:       {robustness['std']:.4f}")
    print(f"    Robustness:{robustness['robustness']:+.4f}")

    return results, robustness


# ──────────────────────────────────────────────
# Diagnosis
# ──────────────────────────────────────────────

def diagnose_likelihood_accuracy(agent: BeliefValueAgent, num_games: int = 500):
    """
    Test how accurate the likelihood model is on held-out games.

    Play games with random agents, then check if the model correctly
    predicts the actions that were actually taken.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 1: Likelihood Model Accuracy")
    print("=" * 60)

    game = LeducGame()
    correct = 0
    total = 0
    per_action_correct = {0: 0, 1: 0, 2: 0}
    per_action_total = {0: 0, 1: 0, 2: 0}

    # Use heuristic as the test opponent (unseen during self-play training)
    heuristic = HeuristicAgent()

    for _ in range(num_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            # Let the heuristic choose
            action = heuristic.select_action(obs)

            # Now test: can our likelihood model predict this action?
            hand = game.player_hands[cp]
            hand_idx = agent.CARD_MAP.get(hand)
            if hand_idx is not None:
                inp = agent._encode_likelihood_input(hand_idx, obs)
                with torch.no_grad():
                    log_probs = agent.likelihood_model(inp)
                    predicted = torch.argmax(log_probs, dim=-1).item()

                actual = int(action)
                total += 1
                per_action_total[actual] = per_action_total.get(actual, 0) + 1
                if predicted == actual:
                    correct += 1
                    per_action_correct[actual] = per_action_correct.get(actual, 0) + 1

            game.step(action)

    accuracy = correct / total if total > 0 else 0
    print(f"  Overall accuracy: {accuracy:.3f} ({correct}/{total})")

    action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}
    per_action_acc = {}
    for a_idx in range(3):
        t = per_action_total.get(a_idx, 0)
        c = per_action_correct.get(a_idx, 0)
        acc = c / t if t > 0 else 0
        per_action_acc[action_names[a_idx]] = round(acc, 4)
        print(f"  {action_names[a_idx]:>5s}: {acc:.3f} ({c}/{t})")

    return {
        "overall_accuracy": round(accuracy, 4),
        "per_action_accuracy": per_action_acc,
        "total_predictions": total,
    }


def diagnose_belief_evolution(agent: BeliefValueAgent, num_examples: int = 5):
    """
    Show how belief vectors evolve during sample hands.

    Plays hands against a heuristic opponent and tracks the belief
    vector at each decision point.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 2: Belief Evolution During Hands")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()
    examples = []

    attempts = 0
    while len(examples) < num_examples and attempts < 200:
        attempts += 1
        game.reset()
        hand_trace = {
            "belief_agent_hand": game.player_hands[0],
            "opponent_hand": game.player_hands[1],
            "board": None,
            "steps": [],
        }

        # Belief agent is player 0
        action_history = []
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                # Belief agent's turn
                belief = agent.compute_belief_from_history(obs)
                evaluations = agent.get_action_evaluations(obs)
                best_eval = max(evaluations, key=lambda x: x["value"])
                action = best_eval["action"]

                hand_trace["steps"].append({
                    "player": "belief_agent",
                    "belief": belief.tolist(),
                    "action": action.name,
                    "values": {e["action"].name: round(e["value"], 3)
                               for e in evaluations},
                })
            else:
                # Opponent's turn
                action = heuristic.select_action(obs)
                # Show what belief agent WOULD compute after this action
                hand_trace["steps"].append({
                    "player": "opponent",
                    "action": action.name,
                })

            action_history.append((cp, action.name))
            game.step(action)

        hand_trace["board"] = game.board
        rewards = game.get_reward()
        hand_trace["rewards"] = rewards

        # Only include hands with at least 3 actions (interesting ones)
        if len(hand_trace["steps"]) >= 3:
            examples.append(hand_trace)

    # Print examples
    for i, ex in enumerate(examples):
        print(f"\n  --- Example {i+1} ---")
        print(f"  Agent hand: {ex['belief_agent_hand']}, "
              f"Opponent hand: {ex['opponent_hand']}, "
              f"Board: {ex['board']}")

        for step in ex["steps"]:
            if step["player"] == "belief_agent":
                belief_str = "[" + ", ".join(
                    f"{c}:{b:.3f}" for c, b in
                    zip(["J", "Q", "K"], step["belief"])
                ) + "]"
                print(f"    Belief agent: belief={belief_str} -> {step['action']}")
                if "values" in step:
                    vals = ", ".join(f"{a}:{v}" for a, v in step["values"].items())
                    print(f"      Values: {vals}")
            else:
                print(f"    Opponent: {step['action']}")

        print(f"  Rewards: agent={ex['rewards'][0]:+d}, opp={ex['rewards'][1]:+d}")

    return examples


def diagnose_bluff_catching(agent: BeliefValueAgent, num_games: int = 1000):
    """
    Analyze bluff-catching: when opponent raises, does belief help
    the agent decide fold vs call/raise better than a non-belief agent?

    Compare belief agent's decisions when facing a raise against
    a baseline value agent.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 3: Bluff-Catching Analysis")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()

    # Track decisions when facing opponent raises
    facing_raise_decisions = {
        "fold_when_behind": 0,  # Correct fold (opponent has better hand)
        "fold_when_ahead": 0,   # Missed value (folded winning hand)
        "call_when_ahead": 0,   # Good call (caught bluff or beat)
        "call_when_behind": 0,  # Bad call (paid off)
        "raise_when_ahead": 0,
        "raise_when_behind": 0,
        "total_facing_raise": 0,
    }

    belief_correct_decisions = 0
    total_decisions_facing_raise = 0

    for _ in range(num_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                # Belief agent
                # Check if facing a raise
                facing_raise = obs.pot[1] > obs.pot[0]

                action = agent.select_action(obs)

                if facing_raise:
                    facing_raise_decisions["total_facing_raise"] += 1
                    total_decisions_facing_raise += 1

                    # Determine who is actually ahead
                    card_values = {'J': 0, 'Q': 1, 'K': 2}
                    my_val = card_values[game.player_hands[0]]
                    opp_val = card_values[game.player_hands[1]]

                    my_pair = game.board is not None and game.player_hands[0] == game.board
                    opp_pair = game.board is not None and game.player_hands[1] == game.board

                    if my_pair and not opp_pair:
                        ahead = True
                    elif opp_pair and not my_pair:
                        ahead = False
                    elif my_pair and opp_pair:
                        ahead = my_val >= opp_val
                    else:
                        ahead = my_val >= opp_val

                    if action == Action.FOLD:
                        if ahead:
                            facing_raise_decisions["fold_when_ahead"] += 1
                        else:
                            facing_raise_decisions["fold_when_behind"] += 1
                            belief_correct_decisions += 1
                    elif action == Action.CALL:
                        if ahead:
                            facing_raise_decisions["call_when_ahead"] += 1
                            belief_correct_decisions += 1
                        else:
                            facing_raise_decisions["call_when_behind"] += 1
                    elif action == Action.RAISE:
                        if ahead:
                            facing_raise_decisions["raise_when_ahead"] += 1
                            belief_correct_decisions += 1
                        else:
                            facing_raise_decisions["raise_when_behind"] += 1
            else:
                # Opponent (heuristic)
                action = heuristic.select_action(obs)

            game.step(action)

    # Print results
    total = facing_raise_decisions["total_facing_raise"]
    accuracy = belief_correct_decisions / total if total > 0 else 0

    print(f"  Total decisions facing raise: {total}")
    print(f"  Correct decisions: {belief_correct_decisions} ({accuracy:.3f})")
    print(f"\n  Breakdown:")
    print(f"    Fold when behind (good): {facing_raise_decisions['fold_when_behind']}")
    print(f"    Fold when ahead  (bad):  {facing_raise_decisions['fold_when_ahead']}")
    print(f"    Call when ahead  (good): {facing_raise_decisions['call_when_ahead']}")
    print(f"    Call when behind (bad):  {facing_raise_decisions['call_when_behind']}")
    print(f"    Raise when ahead (good): {facing_raise_decisions['raise_when_ahead']}")
    print(f"    Raise when behind (bad): {facing_raise_decisions['raise_when_behind']}")

    # Compare with value_based if available
    vb_results = None
    try:
        vb_agent = ValueBasedAgent(model_path="models/value_based_agent.pt")
        vb_agent.set_train_mode(False)
        vb_correct = 0
        vb_total = 0

        for _ in range(num_games):
            game.reset()
            while not game.is_finished:
                cp = game.current_player
                obs = game.get_observation(viewer_id=cp)

                if cp == 0:
                    facing_raise = obs.pot[1] > obs.pot[0]
                    action = vb_agent.select_action(obs)

                    if facing_raise:
                        vb_total += 1
                        card_values = {'J': 0, 'Q': 1, 'K': 2}
                        my_val = card_values[game.player_hands[0]]
                        opp_val = card_values[game.player_hands[1]]
                        my_pair = game.board is not None and game.player_hands[0] == game.board
                        opp_pair = game.board is not None and game.player_hands[1] == game.board

                        if my_pair and not opp_pair:
                            ahead = True
                        elif opp_pair and not my_pair:
                            ahead = False
                        elif my_pair and opp_pair:
                            ahead = my_val >= opp_val
                        else:
                            ahead = my_val >= opp_val

                        if action == Action.FOLD and not ahead:
                            vb_correct += 1
                        elif action == Action.CALL and ahead:
                            vb_correct += 1
                        elif action == Action.RAISE and ahead:
                            vb_correct += 1
                else:
                    action = heuristic.select_action(obs)
                game.step(action)

        vb_accuracy = vb_correct / vb_total if vb_total > 0 else 0
        print(f"\n  Comparison with ValueBasedAgent:")
        print(f"    Belief agent accuracy:  {accuracy:.3f}")
        print(f"    Value agent accuracy:   {vb_accuracy:.3f}")
        print(f"    Delta:                  {accuracy - vb_accuracy:+.3f}")

        vb_results = {
            "value_based_accuracy": round(vb_accuracy, 4),
            "value_based_total": vb_total,
        }
    except Exception as e:
        print(f"  (Could not compare with ValueBasedAgent: {e})")

    results = {
        "belief_accuracy": round(accuracy, 4),
        "total_facing_raise": total,
        "decisions": {k: v for k, v in facing_raise_decisions.items()},
    }
    if vb_results:
        results.update(vb_results)

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ROUND 4 — Direction 1: Bayesian Belief Agent")
    print("=" * 60)

    all_results = {
        "agent": "belief_value",
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
    training_results = train_belief_agent()
    all_results["training"] = training_results

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")
    agent = BeliefValueAgent(model_path=MODEL_PATH)
    agent.set_train_mode(False)
    eval_results, robustness = evaluate_against_opponents(agent)
    all_results["evaluation"] = eval_results
    all_results["robustness"] = robustness

    # Phase 3: Diagnostics
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: DIAGNOSTICS")
    print(f"{'#' * 60}")

    diag1 = diagnose_likelihood_accuracy(agent)
    all_results["diagnostics"] = {"likelihood_accuracy": diag1}

    diag2 = diagnose_belief_evolution(agent)
    # Convert examples to serializable format
    serializable_examples = []
    for ex in diag2:
        serializable_examples.append({
            "belief_agent_hand": ex["belief_agent_hand"],
            "opponent_hand": ex["opponent_hand"],
            "board": ex["board"],
            "rewards": ex["rewards"],
            "num_steps": len(ex["steps"]),
            "steps": ex["steps"],
        })
    all_results["diagnostics"]["belief_evolution_examples"] = serializable_examples

    diag3 = diagnose_bluff_catching(agent)
    all_results["diagnostics"]["bluff_catching"] = diag3

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
    print(f"\n  Likelihood accuracy: {diag1['overall_accuracy']:.3f}")
    print(f"  Bluff-catching accuracy: {diag3['belief_accuracy']:.3f}")

    return all_results


if __name__ == "__main__":
    main()
