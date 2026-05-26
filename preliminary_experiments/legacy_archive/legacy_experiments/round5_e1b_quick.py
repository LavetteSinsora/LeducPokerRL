"""
Quick E1b experiment: Train belief_modulated agent with reasonable session count,
then evaluate and run diagnostics.
"""

import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.poker_session import PokerSession
from src.agents.belief_modulated import BeliefModulatedAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.belief_modulated_trainer import BeliefModulatedTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics

# Configuration - 2000 sessions = 60K hands (comparable to E1a's 40K games)
TRAIN_SESSIONS = 2000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MODULATION_LR = 5e-4
HANDS_PER_SESSION = 30
ROTATE_EVERY = 100
EVAL_ROUNDS = 500
CFR_MODEL_PATH = "models/cfr_agent.pt"
MODEL_PATH = "models/belief_modulated_agent.pt"
RESULTS_PATH = "experiments/round5_belief_modulated_results.json"


def main():
    print("=" * 60)
    print("  E1b: Belief-Modulated Agent (Quick Training)")
    print(f"  Sessions: {TRAIN_SESSIONS}, Hands/session: {HANDS_PER_SESSION}")
    print("=" * 60)

    all_results = {
        "agent": "belief_modulated",
        "experiment": "E1b",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_sessions": TRAIN_SESSIONS,
            "total_hands": TRAIN_SESSIONS * HANDS_PER_SESSION,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "modulation_lr": MODULATION_LR,
        },
    }

    # Phase 1: Training
    print(f"\n--- PHASE 1: TRAINING ---")
    agent = BeliefModulatedAgent(
        cfr_model_path=CFR_MODEL_PATH if os.path.exists(CFR_MODEL_PATH) else None,
        temperature=1.0,
    )
    trainer = BeliefModulatedTrainer(
        agent,
        learning_rate=LEARNING_RATE,
        modulation_lr=MODULATION_LR,
        hands_per_session=HANDS_PER_SESSION,
        rotate_every=ROTATE_EVERY,
    )

    losses = []
    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])

    start = time.time()
    trainer.train(
        num_episodes=TRAIN_SESSIONS,
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
    agent = BeliefModulatedAgent(
        model_path=MODEL_PATH,
        cfr_model_path=CFR_MODEL_PATH if os.path.exists(CFR_MODEL_PATH) else None,
    )
    agent.set_train_mode(False)

    opponents = {}
    # Load opponents
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
        from src.preliminary_experiments.promoted_registry.modulated_value import ModulatedValueAgent
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

    # Diag 1: Likelihood accuracy
    print(f"\n  Likelihood accuracy (Nash vs Nash+modulation):")
    session = PokerSession()
    heuristic = HeuristicAgent()
    nash_correct = 0
    mod_correct = 0
    total = 0

    for g in range(300):
        if g % 30 == 0:
            session.reset()
        session.new_hand()
        while not session.is_finished:
            cp = session.current_player
            obs = session.get_observation(viewer_id=cp)
            if cp == 0:
                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)
                hand = session.game.player_hands[cp]
                actual = int(action)
                try:
                    nash_lp = agent.get_nash_log_probs(hand, obs)
                    if torch.argmax(nash_lp).item() == actual:
                        nash_correct += 1
                except Exception:
                    pass
                try:
                    opp_stats = agent._encode_opp_stats(obs)
                    with torch.no_grad():
                        mod_lp = agent.get_adjusted_log_probs(hand, obs, opp_stats)
                    if torch.argmax(mod_lp).item() == actual:
                        mod_correct += 1
                except Exception:
                    pass
                total += 1
            if isinstance(action, tuple):
                action = action[0]
            session.step(action)

    nash_acc = nash_correct / total if total > 0 else 0
    mod_acc = mod_correct / total if total > 0 else 0
    print(f"    Nash: {nash_acc:.3f}, Modulated: {mod_acc:.3f}, Improvement: {mod_acc - nash_acc:+.3f}")

    # Diag 2: Belief shift & correctness
    belief_correct = 0
    total_hands = 0
    shifts = []
    session = PokerSession()
    for g in range(300):
        if g % 30 == 0:
            session.reset()
        session.new_hand()
        init_b = None
        final_b = None
        while not session.is_finished:
            cp = session.current_player
            obs = session.get_observation(viewer_id=cp)
            if cp == 0:
                b = agent.compute_belief_from_history(obs)
                if init_b is None:
                    init_b = b.copy()
                final_b = b.copy()
                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            session.step(action)

        if init_b is not None and final_b is not None:
            total_hands += 1
            shifts.append(np.sum(np.abs(final_b - init_b)))
            opp_hand = session.game.player_hands[1]
            opp_idx = agent.CARD_MAP.get(opp_hand)
            if opp_idx is not None and np.argmax(final_b) == opp_idx:
                belief_correct += 1

    belief_acc = belief_correct / total_hands if total_hands > 0 else 0
    avg_shift = np.mean(shifts) if shifts else 0
    print(f"\n  Belief correctness: {belief_acc:.3f}, Avg shift: {avg_shift:.4f}")

    # Diag 3: Gate & delta analysis
    print(f"\n  Gate/Delta analysis:")
    profiles = {
        "passive": [0.6, 0.3, 0.1, 0.25],
        "calling": [0.1, 0.7, 0.2, 0.22],
        "aggressive": [0.1, 0.2, 0.7, 0.78],
        "balanced": [0.33, 0.33, 0.33, 0.5],
    }
    gate_delta_results = {}
    with torch.no_grad():
        for pname, stats_list in profiles.items():
            stats = torch.tensor(stats_list, dtype=torch.float32)
            gate = agent.gate_net(stats.unsqueeze(0)).item()
            delta = agent.delta_net(stats.unsqueeze(0)).squeeze().numpy()
            effective = gate * delta
            gate_delta_results[pname] = {
                "gate": round(float(gate), 4),
                "effective_delta": [round(float(d), 4) for d in effective],
            }
            print(f"    {pname:>12s}: gate={gate:.4f}, eff_delta=[{effective[0]:+.3f}, {effective[1]:+.3f}, {effective[2]:+.3f}]")

    # Diag 4: Action distribution
    print(f"\n  Action distribution per hand:")
    action_counts = {c: {0: 0, 1: 0, 2: 0} for c in ['J', 'Q', 'K']}
    session = PokerSession()
    for g in range(500):
        if g % 30 == 0:
            session.reset()
        session.new_hand()
        while not session.is_finished:
            cp = session.current_player
            obs = session.get_observation(viewer_id=cp)
            if cp == 0:
                action = agent.select_action(obs)
                hand = obs.player_hand
                if hand in action_counts:
                    action_counts[hand][int(action)] += 1
            else:
                action = heuristic.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            session.step(action)

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
        "likelihood_accuracy": {
            "nash_accuracy": round(nash_acc, 4),
            "modulated_accuracy": round(mod_acc, 4),
            "improvement": round(mod_acc - nash_acc, 4),
        },
        "belief_shift": {
            "belief_correctness": round(belief_acc, 4),
            "avg_belief_shift_l1": round(float(avg_shift), 4),
        },
        "gate_delta": gate_delta_results,
        "action_distribution": action_dist,
    }

    # Save
    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: E1b Belief-Modulated")
    print(f"  Avg: {robustness['avg']:+.4f}, Robustness: {robustness['robustness']:+.4f}")
    print(f"  Nash acc: {nash_acc:.3f}, Mod acc: {mod_acc:.3f}")
    print(f"  Belief correctness: {belief_acc:.3f}")
    print(f"{'=' * 60}")

    return all_results


if __name__ == "__main__":
    main()
