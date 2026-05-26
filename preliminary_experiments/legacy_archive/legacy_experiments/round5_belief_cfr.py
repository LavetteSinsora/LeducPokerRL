"""
Round 5 -- E1a: Belief-CFR Agent

Implements and evaluates an agent that maintains an explicit belief
distribution over the opponent's hand card, updated within each hand
using CFR Nash equilibrium strategy as the likelihood model
(instead of a learned MLP).

Protocol:
  1. Train for 40K episodes via self-play (value network only; CFR is frozen)
  2. Evaluate against 6 opponents (500 rounds each, both positions)
  3. Diagnose: Nash likelihood accuracy, belief shift, belief correctness,
     action distributions per hand
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
from src.agents.belief_cfr import BeliefCfrAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.belief_cfr_trainer import BeliefCfrTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TRAIN_EPISODES = 40000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EVAL_ROUNDS = 500
CFR_PATH = "models/cfr_agent.pt"
MODEL_PATH = "models/belief_cfr_agent.pt"
RESULTS_PATH = "experiments/round5_belief_cfr_results.json"

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

def train_belief_cfr_agent():
    """Train the Belief-CFR Agent via self-play (value network only)."""
    print("=" * 60)
    print("  TRAINING: Belief-CFR Agent")
    print(f"  Episodes: {TRAIN_EPISODES}, Batch: {BATCH_SIZE}")
    print(f"  Value LR: {LEARNING_RATE}")
    print(f"  CFR strategy: {CFR_PATH}")
    print(f"  (Likelihood model is frozen CFR Nash -- no learning)")
    print("=" * 60)

    agent = BeliefCfrAgent(cfr_path=CFR_PATH, temperature=1.0)
    trainer = BeliefCfrTrainer(agent, learning_rate=LEARNING_RATE)

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


def evaluate_against_opponents(belief_cfr_agent: BeliefCfrAgent):
    """Evaluate the belief-CFR agent against all opponents."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION ({EVAL_ROUNDS} rounds per matchup, both positions)")
    print("=" * 60)

    belief_cfr_agent.set_train_mode(False)
    results = {}

    for name, config in OPPONENTS.items():
        try:
            opponent = load_opponent(name, config)
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}")
            continue

        result = evaluate_agents(belief_cfr_agent, opponent, num_rounds=EVAL_ROUNDS)
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
# Diagnostics
# ──────────────────────────────────────────────

def diagnose_nash_likelihood_accuracy(agent: BeliefCfrAgent, num_games: int = 500):
    """
    Test how well Nash P(action | hand, state) matches actual opponent actions.

    Play games against a heuristic opponent, then check if the Nash strategy
    assigns high probability to the actions actually taken.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 1: Nash Likelihood Accuracy")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()

    total_log_likelihood = 0.0
    total_actions = 0
    correct_top1 = 0  # Nash top-1 prediction matches actual
    per_action_probs = {0: [], 1: [], 2: []}  # Nash prob assigned to actual action

    for _ in range(num_games):
        game.reset()
        action_history = []

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            # Let the heuristic choose
            action = heuristic.select_action(obs)

            # Check Nash likelihood for this (hand, state, action)
            hand = game.player_hands[cp]
            hand_idx = agent.CARD_MAP.get(hand)
            if hand_idx is not None:
                # Determine legal actions
                legal_actions = obs.legal_actions if obs.legal_actions else [
                    Action.FOLD, Action.CALL, Action.RAISE
                ]

                # Build infoset key for this player and look up Nash probs
                key = BeliefCfrAgent._build_opponent_infoset_key(
                    hand, obs.board, obs.current_round,
                    list(obs.action_history) if obs.action_history else []
                )
                strategy = agent.strategy_store.get_average_strategy(
                    key, legal_actions
                )

                actual_idx = int(action)
                nash_prob = max(strategy[actual_idx], 1e-10)

                total_log_likelihood += np.log(nash_prob)
                total_actions += 1
                per_action_probs[actual_idx].append(nash_prob)

                # Check top-1 prediction
                predicted = int(np.argmax(strategy))
                if predicted == actual_idx:
                    correct_top1 += 1

            game.step(action)

    avg_log_likelihood = total_log_likelihood / total_actions if total_actions > 0 else 0
    top1_accuracy = correct_top1 / total_actions if total_actions > 0 else 0

    print(f"  Total actions evaluated: {total_actions}")
    print(f"  Avg log-likelihood of actual action: {avg_log_likelihood:.4f}")
    print(f"  Top-1 prediction accuracy: {top1_accuracy:.3f}")

    action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}
    avg_probs = {}
    for a_idx in range(3):
        probs = per_action_probs[a_idx]
        if probs:
            avg_p = np.mean(probs)
            avg_probs[action_names[a_idx]] = round(float(avg_p), 4)
            print(f"  Avg Nash prob for {action_names[a_idx]:>5s}: "
                  f"{avg_p:.4f} (n={len(probs)})")
        else:
            avg_probs[action_names[a_idx]] = 0.0

    return {
        "avg_log_likelihood": round(avg_log_likelihood, 4),
        "top1_accuracy": round(top1_accuracy, 4),
        "total_actions": total_actions,
        "avg_nash_prob_per_action": avg_probs,
    }


def diagnose_belief_shift(agent: BeliefCfrAgent, num_games: int = 500):
    """
    Measure belief shift magnitude: |b_{t+1} - b_t| averaged over games.

    Tracks how much the belief changes after each opponent action.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 2: Belief Shift Magnitude")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()

    shifts = []  # L1 distance between consecutive beliefs
    per_round_shifts = {0: [], 1: []}  # Shifts by round

    for _ in range(num_games):
        game.reset()

        # Track beliefs from player 0's perspective
        my_hand = game.player_hands[0]
        board = game.board
        belief = agent.initialize_belief(my_hand, None)

        running_pot = [1, 1]
        running_round = 0
        running_raises = 0
        running_board = None
        running_action_history = []

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            # If opponent acted, track belief shift
            if cp == 1:  # Opponent is player 1
                old_belief = belief.copy()

                # Determine legal actions
                if running_raises >= 2:
                    legal_actions = [Action.FOLD, Action.CALL]
                else:
                    legal_actions = [Action.FOLD, Action.CALL, Action.RAISE]

                belief = agent.update_belief(
                    belief, action,
                    board=running_board,
                    current_round=running_round,
                    action_history=list(running_action_history),
                    legal_actions=legal_actions,
                )

                shift = np.sum(np.abs(belief - old_belief))
                shifts.append(shift)
                per_round_shifts[running_round].append(shift)

            # Update running state
            running_action_history.append((cp, action.name))

            if action == Action.FOLD:
                break
            elif action == Action.CALL:
                other_player = 1 - cp
                if running_pot[other_player] > running_pot[cp]:
                    running_pot[cp] = running_pot[other_player]
                    if running_round == 0:
                        running_round = 1
                        running_board = game.board
                        running_raises = 0
                        belief = agent.initialize_belief(my_hand, running_board)
                else:
                    if cp == 1:
                        if running_round == 0:
                            running_round = 1
                            running_board = game.board
                            running_raises = 0
                            belief = agent.initialize_belief(my_hand, running_board)
            elif action == Action.RAISE:
                bet = 2 if running_round == 0 else 4
                other_player = 1 - cp
                running_pot[cp] = running_pot[other_player] + bet
                running_raises += 1

            game.step(action)

    avg_shift = np.mean(shifts) if shifts else 0
    print(f"  Total belief updates: {len(shifts)}")
    print(f"  Average L1 shift: {avg_shift:.4f}")

    round_results = {}
    for rnd in [0, 1]:
        s = per_round_shifts[rnd]
        if s:
            avg = np.mean(s)
            round_results[f"round_{rnd}"] = {
                "avg_shift": round(float(avg), 4),
                "count": len(s),
            }
            print(f"  Round {rnd}: avg shift={avg:.4f} (n={len(s)})")

    return {
        "avg_l1_shift": round(float(avg_shift), 4),
        "total_updates": len(shifts),
        "per_round": round_results,
    }


def diagnose_belief_correctness(agent: BeliefCfrAgent, num_games: int = 500):
    """
    Measure belief correctness: compare belief(true_hand) to ground truth.

    For each opponent action, check how much probability mass the belief
    assigns to the opponent's actual hand.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 3: Belief Correctness")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()

    true_hand_probs = []  # Probability assigned to true opponent hand
    belief_entropy_list = []

    for _ in range(num_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                # Belief agent's turn -- check belief quality
                belief = agent.compute_belief_from_history(obs)

                # What's the true opponent hand?
                true_hand = game.player_hands[1]
                true_idx = agent.CARD_MAP[true_hand]
                true_hand_probs.append(belief[true_idx])

                # Belief entropy
                entropy = -np.sum(belief * np.log(np.maximum(belief, 1e-10)))
                belief_entropy_list.append(entropy)

                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            game.step(action)

    avg_true_prob = np.mean(true_hand_probs) if true_hand_probs else 0
    avg_entropy = np.mean(belief_entropy_list) if belief_entropy_list else 0

    # What would random belief give? Approximately 1/3 = 0.333
    # (actually depends on card removal, so ~0.4 avg)
    print(f"  Total decision points: {len(true_hand_probs)}")
    print(f"  Avg P(true opponent hand): {avg_true_prob:.4f}")
    print(f"  Avg belief entropy: {avg_entropy:.4f}")
    print(f"  (Random baseline: ~0.40 true prob, ~1.05 entropy)")

    # Breakdown by quintile of game progress
    n = len(true_hand_probs)
    if n >= 5:
        quintile_size = n // 5
        for q in range(5):
            start = q * quintile_size
            end = (q + 1) * quintile_size if q < 4 else n
            q_avg = np.mean(true_hand_probs[start:end])
            print(f"  Quintile {q+1}: avg P(true)={q_avg:.4f}")

    return {
        "avg_true_hand_prob": round(float(avg_true_prob), 4),
        "avg_belief_entropy": round(float(avg_entropy), 4),
        "total_decisions": len(true_hand_probs),
    }


def diagnose_action_distribution(agent: BeliefCfrAgent, num_games: int = 1000):
    """
    Analyze action distribution per hand card (J/Q/K -> FOLD/CALL/RAISE).

    Shows how the trained belief-CFR agent plays with each card.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 4: Action Distribution per Hand")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()

    # hand -> action -> count
    action_counts = {
        'J': {0: 0, 1: 0, 2: 0},
        'Q': {0: 0, 1: 0, 2: 0},
        'K': {0: 0, 1: 0, 2: 0},
    }

    # Also track by round
    round_action_counts = {
        0: {'J': {0: 0, 1: 0, 2: 0}, 'Q': {0: 0, 1: 0, 2: 0}, 'K': {0: 0, 1: 0, 2: 0}},
        1: {'J': {0: 0, 1: 0, 2: 0}, 'Q': {0: 0, 1: 0, 2: 0}, 'K': {0: 0, 1: 0, 2: 0}},
    }

    for _ in range(num_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                action = agent.select_action(obs)
                hand = game.player_hands[0]
                action_counts[hand][int(action)] += 1
                rnd = obs.current_round
                round_action_counts[rnd][hand][int(action)] += 1
            else:
                action = heuristic.select_action(obs)

            game.step(action)

    action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}
    results = {}

    print(f"\n  Overall action distribution:")
    for hand in ['J', 'Q', 'K']:
        total = sum(action_counts[hand].values())
        if total > 0:
            dist = {action_names[a]: round(action_counts[hand][a] / total, 3)
                    for a in range(3)}
        else:
            dist = {action_names[a]: 0.0 for a in range(3)}
        results[hand] = dist
        print(f"    {hand}: FOLD={dist['FOLD']:.3f}  CALL={dist['CALL']:.3f}  "
              f"RAISE={dist['RAISE']:.3f}  (n={total})")

    print(f"\n  Per-round breakdown:")
    round_results = {}
    for rnd in [0, 1]:
        rnd_name = "preflop" if rnd == 0 else "flop"
        round_results[rnd_name] = {}
        print(f"    Round {rnd} ({rnd_name}):")
        for hand in ['J', 'Q', 'K']:
            total = sum(round_action_counts[rnd][hand].values())
            if total > 0:
                dist = {action_names[a]: round(round_action_counts[rnd][hand][a] / total, 3)
                        for a in range(3)}
            else:
                dist = {action_names[a]: 0.0 for a in range(3)}
            round_results[rnd_name][hand] = dist
            print(f"      {hand}: FOLD={dist['FOLD']:.3f}  CALL={dist['CALL']:.3f}  "
                  f"RAISE={dist['RAISE']:.3f}  (n={total})")

    return {
        "overall": results,
        "per_round": round_results,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ROUND 5 -- E1a: Belief-CFR Agent")
    print("  (CFR Nash equilibrium as likelihood model)")
    print("=" * 60)

    all_results = {
        "agent": "belief_cfr",
        "experiment": "E1a",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "eval_rounds": EVAL_ROUNDS,
            "cfr_path": CFR_PATH,
            "description": "Belief agent with frozen CFR Nash likelihood (no learned likelihood model)",
        },
    }

    # Phase 1: Training
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1: TRAINING")
    print(f"{'#' * 60}")
    training_results = train_belief_cfr_agent()
    all_results["training"] = training_results

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")
    agent = BeliefCfrAgent(model_path=MODEL_PATH, cfr_path=CFR_PATH)
    agent.set_train_mode(False)
    eval_results, robustness = evaluate_against_opponents(agent)
    all_results["evaluation"] = eval_results
    all_results["robustness"] = robustness

    # Phase 3: Diagnostics
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: DIAGNOSTICS")
    print(f"{'#' * 60}")

    diag1 = diagnose_nash_likelihood_accuracy(agent)
    all_results["diagnostics"] = {"nash_likelihood_accuracy": diag1}

    diag2 = diagnose_belief_shift(agent)
    all_results["diagnostics"]["belief_shift"] = diag2

    diag3 = diagnose_belief_correctness(agent)
    all_results["diagnostics"]["belief_correctness"] = diag3

    diag4 = diagnose_action_distribution(agent)
    all_results["diagnostics"]["action_distribution"] = diag4

    # Phase 4: Save results
    os.makedirs(os.path.dirname(RESULTS_PATH) if os.path.dirname(RESULTS_PATH) else ".",
                exist_ok=True)
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
    print(f"\n  Nash likelihood top-1 accuracy: {diag1['top1_accuracy']:.3f}")
    print(f"  Avg belief shift (L1): {diag2['avg_l1_shift']:.4f}")
    print(f"  Avg P(true opponent hand): {diag3['avg_true_hand_prob']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
