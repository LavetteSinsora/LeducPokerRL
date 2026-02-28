"""
Quick E2d experiment: Train belief_stable agent (stable belief TD targets),
evaluate, and run diagnostics.
"""

import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.belief_stable import BeliefStableAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.belief_stable_trainer import BeliefStableTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics

TRAIN_EPISODES = 40000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LIKELIHOOD_LR = 5e-4
EVAL_ROUNDS = 500
MODEL_PATH = "models/belief_stable_agent.pt"
RESULTS_PATH = "experiments/round5_belief_stable_results.json"


def main():
    print("=" * 60)
    print("  E2d: Belief Stable Agent (Stable Belief TD Targets)")
    print(f"  Episodes: {TRAIN_EPISODES}")
    print("=" * 60)

    all_results = {
        "agent": "belief_stable",
        "experiment": "E2d",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "likelihood_lr": LIKELIHOOD_LR,
        },
    }

    # Phase 1: Training
    print(f"\n--- PHASE 1: TRAINING ---")
    agent = BeliefStableAgent(temperature=1.0)
    trainer = BeliefStableTrainer(
        agent,
        learning_rate=LEARNING_RATE,
        likelihood_lr=LIKELIHOOD_LR,
    )

    losses = []
    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])

    start = time.time()
    trainer.train(
        num_episodes=TRAIN_EPISODES,
        batch_size=BATCH_SIZE,
        save_path=MODEL_PATH,
        callback=callback,
    )
    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.1f}s, final loss: {losses[-1]:.4f}" if losses else "  No losses")

    all_results["training"] = {
        "training_time_s": round(elapsed, 1),
        "num_updates": len(losses),
        "final_loss": round(losses[-1], 6) if losses else None,
    }

    # Phase 2: Evaluation
    print(f"\n--- PHASE 2: EVALUATION ---")
    agent = BeliefStableAgent(model_path=MODEL_PATH)
    agent.set_train_mode(False)

    opponents = {}
    opponents["heuristic"] = HeuristicAgent()

    vb = ValueBasedAgent()
    if os.path.exists("models/value_based_agent.pt"):
        vb.load_model("models/value_based_agent.pt")
    vb.set_train_mode(False)
    opponents["value_based"] = vb

    try:
        from src.agents.adaptive_value import AdaptiveValueAgent
        av = AdaptiveValueAgent()
        if os.path.exists("models/adaptive_value_agent.pt"):
            av.load_model("models/adaptive_value_agent.pt")
        av.set_train_mode(False)
        opponents["adaptive_value"] = av
    except Exception as e:
        print(f"  Could not load adaptive_value: {e}")

    try:
        from src.agents.modulated_value import ModulatedValueAgent
        mv = ModulatedValueAgent()
        if os.path.exists("models/modulated_value_agent.pt"):
            mv.load_model("models/modulated_value_agent.pt")
        mv.set_train_mode(False)
        opponents["modulated_value"] = mv
    except Exception as e:
        print(f"  Could not load modulated_value: {e}")

    try:
        from src.agents.entropy_ac import EntropyACAgent
        ea = EntropyACAgent()
        if os.path.exists("models/entropy_ac_agent.pt"):
            ea.load_model("models/entropy_ac_agent.pt")
        ea.set_train_mode(False)
        opponents["entropy_ac"] = ea
    except Exception as e:
        print(f"  Could not load entropy_ac: {e}")

    try:
        from src.agents.cfr_agent import CFRAgent
        cfr = CFRAgent()
        if os.path.exists("models/cfr_agent.pt"):
            cfr.load_model("models/cfr_agent.pt")
        opponents["cfr"] = cfr
    except Exception as e:
        print(f"  Could not load cfr: {e}")

    eval_results = {}
    for name, opp in opponents.items():
        result = evaluate_agents(agent, opp, num_rounds=EVAL_ROUNDS)
        avg = result.agent_0_avg_chips
        eval_results[name] = round(avg, 4)
        print(f"  vs {name:20s}: {avg:+.4f}")

    robustness = compute_robustness_metrics(eval_results)
    print(f"\n  Avg: {robustness['avg']:+.4f}, Robustness: {robustness['robustness']:+.4f}")

    all_results["evaluation"] = eval_results
    all_results["robustness"] = robustness

    # Phase 3: Diagnostics
    print(f"\n--- PHASE 3: DIAGNOSTICS ---")

    # Diag 1: Belief shift & correctness
    game = LeducGame()
    belief_correct = 0
    total_hands = 0
    shifts = []
    belief_jumps_and_errors = []

    for _ in range(500):
        game.reset()
        init_b = None
        final_b = None
        step_beliefs = []

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                b = agent.compute_belief_from_history(obs)
                step_beliefs.append(b.copy())
                if init_b is None:
                    init_b = b.copy()
                final_b = b.copy()

            action = agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            game.step(action)

        if init_b is not None and final_b is not None:
            total_hands += 1
            shifts.append(np.sum(np.abs(final_b - init_b)))
            opp_hand = game.player_hands[1]
            opp_idx = agent.CARD_MAP.get(opp_hand)
            if opp_idx is not None and np.argmax(final_b) == opp_idx:
                belief_correct += 1

            # Track per-step jumps
            for i in range(1, len(step_beliefs)):
                jump = np.sum(np.abs(step_beliefs[i] - step_beliefs[i-1]))
                belief_jumps_and_errors.append(jump)

    belief_acc = belief_correct / total_hands if total_hands > 0 else 0
    avg_shift = np.mean(shifts) if shifts else 0
    avg_jump = np.mean(belief_jumps_and_errors) if belief_jumps_and_errors else 0
    print(f"  Belief correctness: {belief_acc:.3f}")
    print(f"  Avg total shift: {avg_shift:.4f}")
    print(f"  Avg per-step jump: {avg_jump:.4f}")

    # Diag 2: Action distribution
    print(f"\n  Action distribution per hand:")
    action_counts = {c: {0: 0, 1: 0, 2: 0} for c in ['J', 'Q', 'K']}
    for _ in range(500):
        game.reset()
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            if cp == 0:
                action = agent.select_action(obs)
                hand = obs.player_hand
                if hand in action_counts:
                    action_counts[hand][int(action)] += 1
            else:
                # Simple opponent
                action = HeuristicAgent().select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            game.step(action)

    action_dist = {}
    for card in ['J', 'Q', 'K']:
        total = sum(action_counts[card].values())
        if total > 0:
            dist = {
                "FOLD": round(action_counts[card][0] / total, 3),
                "CALL": round(action_counts[card][1] / total, 3),
                "RAISE": round(action_counts[card][2] / total, 3),
            }
        else:
            dist = {"FOLD": 0, "CALL": 0, "RAISE": 0}
        action_dist[card] = dist
        print(f"    {card}: F={dist['FOLD']:.3f} C={dist['CALL']:.3f} R={dist['RAISE']:.3f} (n={total})")

    all_results["diagnostics"] = {
        "belief_shift": {
            "belief_correctness": round(belief_acc, 4),
            "avg_total_shift_l1": round(float(avg_shift), 4),
            "avg_per_step_jump": round(float(avg_jump), 4),
        },
        "action_distribution": action_dist,
    }

    # Save
    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: E2d Belief Stable")
    print(f"  Avg: {robustness['avg']:+.4f}, Robustness: {robustness['robustness']:+.4f}")
    print(f"  Belief correctness: {belief_acc:.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
