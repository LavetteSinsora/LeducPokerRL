"""
Round 5 -- Experiment E1b: Belief-Modulated Agent

Trains and evaluates a belief agent that uses CFR Nash equilibrium as the
base likelihood model, plus a learned gated modulation layer that adapts
based on opponent macro statistics.

Protocol:
  1. Train for 40K episodes (sessions x 30 hands) against rotating opponents
  2. Evaluate against all major opponents (500 rounds each, both positions)
  3. Run diagnostics:
     - Likelihood accuracy comparison: Nash-only vs Nash+modulation
     - Gate activation values across different opponents
     - Delta magnitudes -- how much does modulation shift the Nash base?
     - Belief shift magnitude and correctness
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
from src.engine.poker_session import PokerSession
from src.agents.belief_modulated import BeliefModulatedAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.belief_modulated_trainer import BeliefModulatedTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# -----------------------------------------------
# Configuration
# -----------------------------------------------

TRAIN_EPISODES = 40000  # sessions (each session = 30 hands)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MODULATION_LR = 5e-4
HANDS_PER_SESSION = 30
ROTATE_EVERY = 100  # rotate opponent every N sessions
EVAL_ROUNDS = 500
CFR_MODEL_PATH = "models/cfr_agent.pt"
MODEL_PATH = "models/belief_modulated_agent.pt"
RESULTS_PATH = "experiments/round5_belief_modulated_results.json"

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

def train_belief_modulated_agent():
    """Train the Belief-Modulated Agent against rotating opponents."""
    print("=" * 60)
    print("  TRAINING: Belief-Modulated Agent")
    print(f"  Sessions: {TRAIN_EPISODES}, Hands/session: {HANDS_PER_SESSION}")
    print(f"  Value LR: {LEARNING_RATE}, Modulation LR: {MODULATION_LR}")
    print(f"  Opponent rotation: every {ROTATE_EVERY} sessions")
    print("=" * 60)

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


def evaluate_against_opponents(agent: BeliefModulatedAgent):
    """Evaluate the belief-modulated agent against all opponents."""
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

def diagnose_likelihood_accuracy(agent: BeliefModulatedAgent, num_games: int = 500):
    """
    Compare likelihood accuracy: Nash-only vs Nash+modulation.

    For each opponent action, check whether the highest-probability action
    from each model matches the actual action taken.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 1: Likelihood Accuracy (Nash-only vs Nash+modulation)")
    print("=" * 60)

    session = PokerSession()
    heuristic = HeuristicAgent()

    nash_correct = 0
    modulated_correct = 0
    total = 0

    per_action_nash = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # [correct, total]
    per_action_mod = {0: [0, 0], 1: [0, 0], 2: [0, 0]}

    for game_idx in range(num_games):
        if game_idx % 30 == 0:
            session.reset()  # reset stats every 30 games

        session.new_hand()

        while not session.is_finished:
            cp = session.current_player
            obs = session.get_observation(viewer_id=cp)

            if cp == 0:
                # Belief-modulated agent
                action = agent.select_action(obs)
            else:
                # Heuristic opponent
                action = heuristic.select_action(obs)

                # Test likelihood accuracy for this opponent action
                hand = session.game.player_hands[cp]
                actual = int(action)

                # Nash-only prediction
                try:
                    nash_log_probs = agent.get_nash_log_probs(hand, obs)
                    nash_pred = torch.argmax(nash_log_probs).item()
                except Exception:
                    nash_pred = -1

                # Modulated prediction
                try:
                    opp_stats = agent._encode_opp_stats(obs)
                    with torch.no_grad():
                        mod_log_probs = agent.get_adjusted_log_probs(hand, obs, opp_stats)
                    mod_pred = torch.argmax(mod_log_probs).item()
                except Exception:
                    mod_pred = -1

                total += 1
                per_action_nash[actual][1] += 1
                per_action_mod[actual][1] += 1

                if nash_pred == actual:
                    nash_correct += 1
                    per_action_nash[actual][0] += 1
                if mod_pred == actual:
                    modulated_correct += 1
                    per_action_mod[actual][0] += 1

            if isinstance(action, tuple):
                action = action[0]
            session.step(action)

    nash_acc = nash_correct / total if total > 0 else 0
    mod_acc = modulated_correct / total if total > 0 else 0

    action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}

    print(f"  Total predictions: {total}")
    print(f"  Nash-only accuracy:     {nash_acc:.3f} ({nash_correct}/{total})")
    print(f"  Nash+modulation accuracy: {mod_acc:.3f} ({modulated_correct}/{total})")
    print(f"  Improvement: {mod_acc - nash_acc:+.3f}")

    print(f"\n  Per-action breakdown:")
    nash_per_action = {}
    mod_per_action = {}
    for a_idx in range(3):
        nc, nt = per_action_nash[a_idx]
        mc, mt = per_action_mod[a_idx]
        n_acc = nc / nt if nt > 0 else 0
        m_acc = mc / mt if mt > 0 else 0
        nash_per_action[action_names[a_idx]] = round(n_acc, 4)
        mod_per_action[action_names[a_idx]] = round(m_acc, 4)
        print(f"    {action_names[a_idx]:>5s}: Nash={n_acc:.3f}  Mod={m_acc:.3f}  ({nt} samples)")

    return {
        "nash_accuracy": round(nash_acc, 4),
        "modulated_accuracy": round(mod_acc, 4),
        "improvement": round(mod_acc - nash_acc, 4),
        "total_predictions": total,
        "nash_per_action": nash_per_action,
        "modulated_per_action": mod_per_action,
    }


def diagnose_gate_activations(agent: BeliefModulatedAgent, num_sessions: int = 20):
    """
    Measure gate activation values across different opponent types.

    The gate should output low values early (uninformative stats) and
    higher values as stats accumulate, with different magnitudes for
    different opponent types.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 2: Gate Activation Across Opponents")
    print("=" * 60)

    opponent_configs = {
        "heuristic": HeuristicAgent(),
    }

    try:
        vb = ValueBasedAgent()
        if os.path.exists("models/value_based_agent.pt"):
            vb.load_model("models/value_based_agent.pt")
        vb.set_train_mode(False)
        opponent_configs["value_based"] = vb
    except Exception:
        pass

    try:
        from src.agents.adaptive_value import AdaptiveValueAgent
        av = AdaptiveValueAgent()
        if os.path.exists("models/adaptive_value_agent.pt"):
            av.load_model("models/adaptive_value_agent.pt")
        av.set_train_mode(False)
        opponent_configs["adaptive_value"] = av
    except Exception:
        pass

    agent.set_train_mode(False)
    results = {}

    for opp_name, opponent in opponent_configs.items():
        gate_values = []
        gate_by_hand_count = {}  # hand_count -> [gate_values]

        for _ in range(num_sessions):
            session = PokerSession()

            for hand_num in range(30):
                session.new_hand()
                while not session.is_finished:
                    cp = session.current_player
                    obs = session.get_observation(viewer_id=cp)

                    if cp == 0:
                        # Measure gate activation
                        opp_stats = agent._encode_opp_stats(obs)
                        with torch.no_grad():
                            gate_val = agent.gate_net(opp_stats.unsqueeze(0)).item()
                        gate_values.append(gate_val)

                        bucket = hand_num // 5 * 5  # bucket by 5 hands
                        if bucket not in gate_by_hand_count:
                            gate_by_hand_count[bucket] = []
                        gate_by_hand_count[bucket].append(gate_val)

                        action = agent.select_action(obs)
                    else:
                        action = opponent.select_action(obs)

                    if isinstance(action, tuple):
                        action = action[0]
                    session.step(action)

        avg_gate = np.mean(gate_values) if gate_values else 0
        std_gate = np.std(gate_values) if gate_values else 0

        gate_evolution = {}
        for bucket in sorted(gate_by_hand_count.keys()):
            vals = gate_by_hand_count[bucket]
            gate_evolution[f"hands_{bucket}-{bucket+4}"] = round(np.mean(vals), 4)

        results[opp_name] = {
            "avg_gate": round(float(avg_gate), 4),
            "std_gate": round(float(std_gate), 4),
            "n_samples": len(gate_values),
            "gate_evolution": gate_evolution,
        }

        print(f"\n  vs {opp_name}:")
        print(f"    Avg gate: {avg_gate:.4f} +/- {std_gate:.4f}")
        for label, val in gate_evolution.items():
            print(f"    {label}: {val:.4f}")

    return results


def diagnose_delta_magnitudes(agent: BeliefModulatedAgent):
    """
    Measure how much the delta network shifts the Nash base probabilities
    for different synthetic opponent profiles.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 3: Delta Magnitudes")
    print("=" * 60)

    # Synthetic opponent profiles: [fold_rate, call_rate, raise_rate, aggression]
    profiles = {
        "passive_folder":    [0.6, 0.3, 0.1, 0.25],
        "calling_station":   [0.1, 0.7, 0.2, 0.22],
        "aggressive_raiser": [0.1, 0.2, 0.7, 0.78],
        "balanced_nash":     [0.33, 0.33, 0.33, 0.5],
        "uninformative":     [0.33, 0.33, 0.33, 0.0],
    }

    results = {}

    with torch.no_grad():
        for profile_name, stats_list in profiles.items():
            stats = torch.tensor(stats_list, dtype=torch.float32)
            gate = agent.gate_net(stats.unsqueeze(0)).item()
            delta = agent.delta_net(stats.unsqueeze(0)).squeeze().numpy()
            effective_delta = gate * delta

            results[profile_name] = {
                "gate": round(float(gate), 4),
                "raw_delta": [round(float(d), 4) for d in delta],
                "effective_delta": [round(float(d), 4) for d in effective_delta],
                "delta_l2_norm": round(float(np.linalg.norm(effective_delta)), 4),
            }

            print(f"\n  {profile_name}:")
            print(f"    Gate: {gate:.4f}")
            print(f"    Raw delta [F, C, R]: [{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}]")
            print(f"    Effective delta: [{effective_delta[0]:+.4f}, {effective_delta[1]:+.4f}, {effective_delta[2]:+.4f}]")
            print(f"    L2 norm: {np.linalg.norm(effective_delta):.4f}")

    return results


def diagnose_belief_shift(agent: BeliefModulatedAgent, num_games: int = 500):
    """
    Measure belief shift magnitude and correctness.

    For each hand, compute:
    - How much the belief shifts from prior to posterior
    - Whether the final belief assigns highest probability to the correct hand
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 4: Belief Shift Magnitude & Correctness")
    print("=" * 60)

    session = PokerSession()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    belief_correct = 0
    total_hands = 0
    shift_magnitudes = []

    for game_idx in range(num_games):
        if game_idx % 30 == 0:
            session.reset()

        session.new_hand()
        initial_belief = None
        final_belief = None

        while not session.is_finished:
            cp = session.current_player
            obs = session.get_observation(viewer_id=cp)

            if cp == 0:
                belief = agent.compute_belief_from_history(obs)
                if initial_belief is None:
                    initial_belief = belief.copy()
                final_belief = belief.copy()
                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            if isinstance(action, tuple):
                action = action[0]
            session.step(action)

        if initial_belief is not None and final_belief is not None:
            total_hands += 1

            # Belief shift magnitude (L1 norm)
            shift = np.sum(np.abs(final_belief - initial_belief))
            shift_magnitudes.append(shift)

            # Was the highest-probability hand correct?
            opp_hand = session.game.player_hands[1]
            opp_idx = agent.CARD_MAP.get(opp_hand)
            if opp_idx is not None and np.argmax(final_belief) == opp_idx:
                belief_correct += 1

    accuracy = belief_correct / total_hands if total_hands > 0 else 0
    avg_shift = np.mean(shift_magnitudes) if shift_magnitudes else 0
    std_shift = np.std(shift_magnitudes) if shift_magnitudes else 0

    print(f"  Total hands: {total_hands}")
    print(f"  Belief correctness (argmax = true hand): {accuracy:.3f} ({belief_correct}/{total_hands})")
    print(f"  Avg belief shift (L1): {avg_shift:.4f} +/- {std_shift:.4f}")

    # Breakdown by shift magnitude
    if shift_magnitudes:
        shifts = np.array(shift_magnitudes)
        print(f"  Shift distribution:")
        print(f"    < 0.1:  {np.mean(shifts < 0.1):.1%}")
        print(f"    0.1-0.5: {np.mean((shifts >= 0.1) & (shifts < 0.5)):.1%}")
        print(f"    > 0.5:  {np.mean(shifts >= 0.5):.1%}")

    return {
        "belief_correctness": round(accuracy, 4),
        "avg_belief_shift_l1": round(float(avg_shift), 4),
        "std_belief_shift_l1": round(float(std_shift), 4),
        "total_hands": total_hands,
    }


def diagnose_action_distribution(agent: BeliefModulatedAgent, num_games: int = 1000):
    """
    Analyze action distribution per hand card.

    Shows how the agent plays different hands -- useful for checking
    if belief information is actually influencing decisions.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 5: Action Distribution Per Hand")
    print("=" * 60)

    session = PokerSession()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    # action_counts[hand][round][action] = count
    action_counts = {}
    for card in ['J', 'Q', 'K']:
        action_counts[card] = {
            0: {0: 0, 1: 0, 2: 0},  # preflop
            1: {0: 0, 1: 0, 2: 0},  # flop
        }

    for game_idx in range(num_games):
        if game_idx % 30 == 0:
            session.reset()

        session.new_hand()
        while not session.is_finished:
            cp = session.current_player
            obs = session.get_observation(viewer_id=cp)

            if cp == 0:
                action = agent.select_action(obs)
                hand = obs.player_hand
                rnd = obs.current_round
                if hand in action_counts and rnd in action_counts[hand]:
                    action_counts[hand][rnd][int(action)] += 1
            else:
                action = heuristic.select_action(obs)

            if isinstance(action, tuple):
                action = action[0]
            session.step(action)

    action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}
    round_names = {0: "Preflop", 1: "Flop"}

    results = {}
    for card in ['J', 'Q', 'K']:
        results[card] = {}
        print(f"\n  Hand: {card}")
        for rnd in [0, 1]:
            counts = action_counts[card][rnd]
            total = sum(counts.values())
            if total == 0:
                continue
            dist = {action_names[a]: round(counts[a] / total, 3) for a in range(3)}
            results[card][round_names[rnd]] = dist
            dist_str = "  ".join(f"{n}:{p:.3f}" for n, p in dist.items())
            print(f"    {round_names[rnd]:>7s}: {dist_str}  (n={total})")

    return results


# -----------------------------------------------
# Main
# -----------------------------------------------

def main():
    print("=" * 60)
    print("  ROUND 5 -- E1b: Belief-Modulated Agent")
    print("=" * 60)

    all_results = {
        "agent": "belief_modulated",
        "experiment": "E1b",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "modulation_lr": MODULATION_LR,
            "hands_per_session": HANDS_PER_SESSION,
            "rotate_every": ROTATE_EVERY,
            "eval_rounds": EVAL_ROUNDS,
            "cfr_model_path": CFR_MODEL_PATH,
        },
    }

    # Phase 1: Training
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1: TRAINING")
    print(f"{'#' * 60}")
    training_results = train_belief_modulated_agent()
    all_results["training"] = training_results

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")
    agent = BeliefModulatedAgent(
        model_path=MODEL_PATH,
        cfr_model_path=CFR_MODEL_PATH if os.path.exists(CFR_MODEL_PATH) else None,
    )
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

    diag2 = diagnose_gate_activations(agent)
    all_results["diagnostics"]["gate_activations"] = diag2

    diag3 = diagnose_delta_magnitudes(agent)
    all_results["diagnostics"]["delta_magnitudes"] = diag3

    diag4 = diagnose_belief_shift(agent)
    all_results["diagnostics"]["belief_shift"] = diag4

    diag5 = diagnose_action_distribution(agent)
    all_results["diagnostics"]["action_distribution"] = diag5

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
    print(f"\n  Likelihood accuracy:")
    print(f"    Nash-only:     {diag1['nash_accuracy']:.3f}")
    print(f"    Nash+modulation: {diag1['modulated_accuracy']:.3f}")
    print(f"    Improvement:   {diag1['improvement']:+.3f}")
    print(f"\n  Belief correctness: {diag4['belief_correctness']:.3f}")
    print(f"  Avg belief shift: {diag4['avg_belief_shift_l1']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
