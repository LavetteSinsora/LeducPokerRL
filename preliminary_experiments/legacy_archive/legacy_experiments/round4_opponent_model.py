"""
Round 4 — Direction 4: Opponent Response Planning Agent (2-Ply Lookahead)

This experiment trains and evaluates an OpponentModelAgent that performs
2-ply lookahead with a learned opponent model. Instead of the standard 1-ply
approach ("if I raise, the resulting state is worth X"), this agent asks:
"if I raise, the opponent will probably call (70%) or reraise (30%), and
accounting for their likely response, the expected value is Y."

Protocol:
  1. Train OpponentModelAgent for 30K episodes via self-play
     - Value network: TD(0) on post-action chains
     - Opponent model: cross-entropy on (state, action) pairs
  2. Evaluate against 6 opponents (500 rounds each, play as both P0 and P1)
  3. Diagnose:
     a) Opponent model accuracy: predicted action distribution vs actual
     b) 1-ply vs 2-ply comparison: how often does 2-ply change the decision?
     c) Which states benefit most from 2-ply lookahead?
  4. Save results to experiments/round4_opponent_model_results.json
  5. Save model to models/opponent_model_agent.pt
"""

import json
import os
import sys
import time
import random
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.engine.leduc_game import LeducGame, Action
from src.agents.opponent_model_agent import OpponentModelAgent
from src.training.opponent_model_trainer import OpponentModelTrainer
from src.training.evaluation import evaluate_agents, quick_evaluate

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TRAIN_EPISODES = 30000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EVAL_ROUNDS = 500
MODEL_PATH = "models/opponent_model_agent.pt"
RESULTS_PATH = "experiments/round4_opponent_model_results.json"

OPPONENTS = {
    "heuristic": {"class": "src.agents.heuristic.HeuristicAgent", "model_path": None},
    "value_based": {"class": "src.agents.value_based.ValueBasedAgent", "model_path": "models/value_based_agent.pt"},
    "adaptive_value": {"class": "src.agents.adaptive_value.AdaptiveValueAgent", "model_path": "models/adaptive_value_agent.pt"},
    "modulated_value": {"class": "src.preliminary_experiments.promoted_registry.modulated_value.ModulatedValueAgent", "model_path": "models/modulated_value_agent.pt"},
    "entropy_ac": {"class": "src.agents.entropy_ac.EntropyACAgent", "model_path": "models/entropy_ac_agent.pt"},
    "cfr": {"class": "src.agents.cfr_agent.CFRAgent", "model_path": "models/cfr_agent.pt"},
}


# ──────────────────────────────────────────────
# Utility: Load opponent agent
# ──────────────────────────────────────────────

def load_opponent(name, config):
    """Dynamically load an opponent agent."""
    module_path, class_name = config["class"].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if config["model_path"] and os.path.exists(config["model_path"]):
        agent = cls(model_path=config["model_path"])
    else:
        agent = cls()

    if hasattr(agent, 'set_train_mode'):
        agent.set_train_mode(False)
    return agent


# ──────────────────────────────────────────────
# Phase 1: Training
# ──────────────────────────────────────────────

def train_agent():
    """Train the OpponentModelAgent via self-play."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1: TRAINING OpponentModelAgent")
    print(f"  Episodes: {TRAIN_EPISODES}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    agent = OpponentModelAgent(temperature=1.0)
    trainer = OpponentModelTrainer(agent, learning_rate=LEARNING_RATE)

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
        print(f"  Final eval vs heuristic: {eval_scores[-1]['avg_chips']:+.3f} chips/round")

    training_result = {
        "training_time_s": round(elapsed, 1),
        "num_updates": len(losses),
        "final_loss": losses[-1]["loss"] if losses else None,
        "final_eval_vs_heuristic": eval_scores[-1]["avg_chips"] if eval_scores else None,
        "eval_history": eval_scores[-10:],  # Last 10 eval points
    }

    return agent, training_result


# ──────────────────────────────────────────────
# Phase 2: Evaluation vs opponents
# ──────────────────────────────────────────────

def evaluate_against_opponents(agent):
    """Evaluate the trained agent against all configured opponents."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2: EVALUATION ({EVAL_ROUNDS} rounds per opponent)")
    print(f"{'='*60}\n")

    agent.set_train_mode(False)
    results = {}

    for opp_name, opp_config in OPPONENTS.items():
        try:
            opponent = load_opponent(opp_name, opp_config)
            result = evaluate_agents(agent, opponent, num_rounds=EVAL_ROUNDS)
            avg_chips = result.agent_0_avg_chips
            results[opp_name] = round(avg_chips, 4)
            print(f"  vs {opp_name:20s}: {avg_chips:+.4f} chips/round")
        except Exception as e:
            print(f"  vs {opp_name:20s}: ERROR - {e}")
            results[opp_name] = None

    # Summary stats
    valid = [v for v in results.values() if v is not None]
    if valid:
        print(f"\n  Average across opponents: {sum(valid)/len(valid):+.4f}")
        print(f"  Best:  {max(valid):+.4f}")
        print(f"  Worst: {min(valid):+.4f}")

    return results


# ──────────────────────────────────────────────
# Phase 3: Diagnostics
# ──────────────────────────────────────────────

def diagnose_opponent_model_accuracy(agent):
    """
    Test how accurate the opponent model is against each opponent type.
    Play games and compare predicted action distribution vs actual actions.
    """
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS 1: Opponent Model Accuracy")
    print(f"{'='*60}\n")

    agent.set_train_mode(False)
    game = LeducGame()
    num_test_games = 200

    accuracy_results = {}

    for opp_name, opp_config in OPPONENTS.items():
        try:
            opponent = load_opponent(opp_name, opp_config)
        except Exception as e:
            print(f"  Skipping {opp_name}: {e}")
            continue

        # Collect opponent actions and our predictions
        correct = 0
        total = 0
        action_counts = defaultdict(int)
        predicted_probs_sum = defaultdict(float)

        for _ in range(num_test_games):
            game.reset()
            # Assign agent to player 0, opponent to player 1
            agents = [agent, opponent]

            while not game.is_finished:
                cp = game.current_player
                obs = game.get_observation(viewer_id=cp)

                if cp == 1:  # Opponent is acting
                    # Get our model's prediction BEFORE the opponent acts
                    opp_obs = game.get_observation(viewer_id=1)
                    opp_probs = agent._get_opponent_action_probs(opp_obs, opponent_id=1)

                    # Record actual action
                    actual_action = opponent.select_action(obs)
                    action_counts[actual_action.name] += 1
                    total += 1

                    # Check if highest-probability prediction matches
                    predicted_action_idx = opp_probs.argmax().item()
                    if predicted_action_idx == actual_action.value:
                        correct += 1

                    # Accumulate predicted probabilities
                    for a in Action:
                        predicted_probs_sum[a.name] += opp_probs[a.value].item()

                    game.step(actual_action)
                else:
                    action = agent.select_action(obs)
                    game.step(action)

        if total > 0:
            accuracy = correct / total
            actual_dist = {a: action_counts[a] / total for a in action_counts}
            predicted_dist = {a: predicted_probs_sum[a] / total for a in predicted_probs_sum}

            accuracy_results[opp_name] = {
                "accuracy": round(accuracy, 4),
                "total_predictions": total,
                "actual_distribution": {k: round(v, 4) for k, v in sorted(actual_dist.items())},
                "predicted_distribution": {k: round(v, 4) for k, v in sorted(predicted_dist.items())},
            }

            print(f"  vs {opp_name:20s}: accuracy={accuracy:.1%} ({total} decisions)")
            print(f"    Actual:    {', '.join(f'{k}={v:.1%}' for k, v in sorted(actual_dist.items()))}")
            print(f"    Predicted: {', '.join(f'{k}={v:.1%}' for k, v in sorted(predicted_dist.items()))}")

    return accuracy_results


def diagnose_1ply_vs_2ply(agent):
    """
    Compare 1-ply and 2-ply decisions: how often does 2-ply change the action?
    Which states benefit most from 2-ply planning?
    """
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS 2: 1-Ply vs 2-Ply Decision Comparison")
    print(f"{'='*60}\n")

    agent.set_train_mode(False)
    game = LeducGame()
    num_test_games = 500

    total_decisions = 0
    changed_decisions = 0

    # Track by state features
    changed_by_round = defaultdict(lambda: {"total": 0, "changed": 0})
    changed_by_hand = defaultdict(lambda: {"total": 0, "changed": 0})
    changed_by_action = defaultdict(lambda: {"total": 0, "changed": 0})

    # Track value differences when 2-ply changes the decision
    value_diffs = []
    change_examples = []

    for _ in range(num_test_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            # Get both 1-ply and 2-ply evaluations
            evals_1ply = agent.get_action_evaluations(obs)
            evals_2ply = agent.get_action_evaluations_2ply(obs)

            if not evals_1ply or not evals_2ply:
                action = Action.CALL
                game.step(action)
                continue

            # Best action from each
            best_1ply = max(evals_1ply, key=lambda x: x["value"])
            best_2ply = max(evals_2ply, key=lambda x: x["value"])

            action_1ply = best_1ply["action"]
            action_2ply = best_2ply["action"]

            total_decisions += 1

            # Track by state features
            round_key = f"round_{obs.current_round}"
            hand_key = obs.player_hand
            changed_by_round[round_key]["total"] += 1
            changed_by_hand[hand_key]["total"] += 1

            if action_1ply != action_2ply:
                changed_decisions += 1
                changed_by_round[round_key]["changed"] += 1
                changed_by_hand[hand_key]["changed"] += 1

                # Record value difference
                val_diff = best_2ply["value"] - best_1ply["value"]
                value_diffs.append(val_diff)

                # Track which action changes are common
                change_key = f"{action_1ply.name}->{action_2ply.name}"
                changed_by_action[change_key]["changed"] += 1

                # Record example (up to 10)
                if len(change_examples) < 10:
                    change_examples.append({
                        "hand": obs.player_hand,
                        "board": obs.board,
                        "pot": obs.pot,
                        "round": obs.current_round,
                        "1ply_action": action_1ply.name,
                        "1ply_value": round(best_1ply["value"], 4),
                        "2ply_action": action_2ply.name,
                        "2ply_value": round(best_2ply["value"], 4),
                        "opp_probs": best_2ply.get("opp_probs", {}),
                    })

            # Use 2-ply action to continue the game
            game.step(action_2ply)

    # Compute statistics
    change_rate = changed_decisions / total_decisions if total_decisions > 0 else 0
    print(f"  Total decisions:     {total_decisions}")
    print(f"  Changed by 2-ply:    {changed_decisions} ({change_rate:.1%})")

    print(f"\n  By round:")
    for key in sorted(changed_by_round.keys()):
        data = changed_by_round[key]
        rate = data["changed"] / data["total"] if data["total"] > 0 else 0
        print(f"    {key}: {data['changed']}/{data['total']} ({rate:.1%})")

    print(f"\n  By hand:")
    for key in sorted(changed_by_hand.keys()):
        data = changed_by_hand[key]
        rate = data["changed"] / data["total"] if data["total"] > 0 else 0
        print(f"    {key}: {data['changed']}/{data['total']} ({rate:.1%})")

    print(f"\n  Action change types:")
    for key in sorted(changed_by_action.keys()):
        print(f"    {key}: {changed_by_action[key]['changed']}")

    if value_diffs:
        print(f"\n  Value improvement from 2-ply (when decision changes):")
        print(f"    Mean:   {np.mean(value_diffs):+.4f}")
        print(f"    Median: {np.median(value_diffs):+.4f}")
        print(f"    Std:    {np.std(value_diffs):.4f}")

    if change_examples:
        print(f"\n  Example changes (first {len(change_examples)}):")
        for ex in change_examples[:5]:
            print(f"    Hand={ex['hand']}, Board={ex['board']}, Pot={ex['pot']}, Rnd={ex['round']}")
            print(f"      1-ply: {ex['1ply_action']} (val={ex['1ply_value']})")
            print(f"      2-ply: {ex['2ply_action']} (val={ex['2ply_value']})")
            if ex['opp_probs']:
                print(f"      Opp probs: {ex['opp_probs']}")

    return {
        "total_decisions": total_decisions,
        "changed_decisions": changed_decisions,
        "change_rate": round(change_rate, 4),
        "by_round": {k: {
            "total": v["total"],
            "changed": v["changed"],
            "rate": round(v["changed"] / v["total"], 4) if v["total"] > 0 else 0
        } for k, v in changed_by_round.items()},
        "by_hand": {k: {
            "total": v["total"],
            "changed": v["changed"],
            "rate": round(v["changed"] / v["total"], 4) if v["total"] > 0 else 0
        } for k, v in changed_by_hand.items()},
        "action_changes": {k: v["changed"] for k, v in changed_by_action.items()},
        "value_improvement": {
            "mean": round(float(np.mean(value_diffs)), 4) if value_diffs else None,
            "median": round(float(np.median(value_diffs)), 4) if value_diffs else None,
            "std": round(float(np.std(value_diffs)), 4) if value_diffs else None,
        },
        "examples": change_examples[:5],
    }


def diagnose_model_predictions(agent):
    """
    Deep dive into the opponent model predictions across different game states.
    Examines how the model's predictions vary by hand strength, round, etc.
    """
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS 3: Opponent Model Prediction Analysis")
    print(f"{'='*60}\n")

    agent.set_train_mode(False)
    game = LeducGame()
    num_test_games = 300

    # Collect predictions by state category
    predictions_by_round = defaultdict(list)  # round -> list of prob vectors
    predictions_by_hand = defaultdict(list)   # hand card -> list of prob vectors
    predictions_by_board = defaultdict(list)  # board card -> list of prob vectors

    for _ in range(num_test_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            # Get opponent model predictions from both perspectives
            opp_id = 1 - cp
            opp_obs = game.get_observation(viewer_id=opp_id)
            opp_probs = agent._get_opponent_action_probs(opp_obs, opponent_id=opp_id)
            prob_list = opp_probs.tolist()

            predictions_by_round[obs.current_round].append(prob_list)
            predictions_by_hand[obs.player_hand].append(prob_list)
            board_key = obs.board if obs.board else "None"
            predictions_by_board[board_key].append(prob_list)

            action = agent.select_action(obs)
            game.step(action)

    # Analyze
    print("  Average predicted action probabilities:")
    print(f"  {'Category':>15s} | {'FOLD':>8s} {'CALL':>8s} {'RAISE':>8s} | {'Count':>6s}")
    print(f"  {'-'*15}-+-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*6}")

    analysis = {}

    for label, preds_dict in [("by_round", predictions_by_round),
                               ("by_hand", predictions_by_hand),
                               ("by_board", predictions_by_board)]:
        analysis[label] = {}
        for key in sorted(preds_dict.keys()):
            preds = preds_dict[key]
            if not preds:
                continue
            avg = np.mean(preds, axis=0)
            n = len(preds)
            cat_label = f"{label}/{key}"
            print(f"  {cat_label:>15s} | {avg[0]:8.3f} {avg[1]:8.3f} {avg[2]:8.3f} | {n:6d}")
            analysis[label][str(key)] = {
                "fold_prob": round(float(avg[0]), 4),
                "call_prob": round(float(avg[1]), 4),
                "raise_prob": round(float(avg[2]), 4),
                "count": n,
            }

    return analysis


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ROUND 4 — Direction 4: Opponent Response Planning Agent")
    print("  2-Ply Lookahead with Learned Opponent Model")
    print("=" * 60)

    all_results = {
        "experiment": "round4_opponent_model",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "eval_rounds": EVAL_ROUNDS,
        },
    }

    # Phase 1: Train
    agent, training_result = train_agent()
    all_results["training"] = training_result

    # Phase 2: Evaluate
    eval_results = evaluate_against_opponents(agent)
    all_results["evaluation"] = eval_results

    # Phase 3: Diagnostics
    print(f"\n{'#'*60}")
    print(f"  PHASE 3: DIAGNOSTICS")
    print(f"{'#'*60}")

    diag_accuracy = diagnose_opponent_model_accuracy(agent)
    all_results["diagnosis_accuracy"] = diag_accuracy

    diag_1ply_vs_2ply = diagnose_1ply_vs_2ply(agent)
    all_results["diagnosis_1ply_vs_2ply"] = diag_1ply_vs_2ply

    diag_predictions = diagnose_model_predictions(agent)
    all_results["diagnosis_predictions"] = diag_predictions

    # Phase 4: Save results
    os.makedirs(os.path.dirname(RESULTS_PATH) if os.path.dirname(RESULTS_PATH) else ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Training time: {training_result['training_time_s']}s")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"\n  Evaluation results:")
    for opp, score in eval_results.items():
        if score is not None:
            print(f"    vs {opp:20s}: {score:+.4f}")
    valid = [v for v in eval_results.values() if v is not None]
    if valid:
        print(f"\n    Average: {sum(valid)/len(valid):+.4f}")

    if diag_1ply_vs_2ply:
        print(f"\n  2-ply changed {diag_1ply_vs_2ply['change_rate']:.1%} of decisions")

    return all_results


if __name__ == "__main__":
    main()
