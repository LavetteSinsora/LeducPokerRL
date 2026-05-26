"""
Round 5 — Experiment E2b: Belief Oracle Agent

Implements and evaluates an agent that learns a perfect-information value
function V(s, my_hand, opp_hand) and uses belief-weighted action selection
at both training and evaluation time.

Protocol:
  1. Train MC model for 40K episodes
  2. Train TD(0) model for 40K episodes (comparison)
  3. Evaluate both against all major opponents (500 rounds each)
  4. Diagnostics:
     - State coverage
     - MC vs TD(0) value accuracy
     - Value ordering sanity checks
     - Action distribution per hand
     - Belief quality metrics
  5. Save to experiments/round5_belief_oracle_results.json
"""

import json
import os
import sys
import time
import random
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.belief_oracle import BeliefOracleAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.belief_oracle_trainer import BeliefOracleTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TRAIN_EPISODES = 40000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EVAL_ROUNDS = 500
MC_MODEL_PATH = "models/belief_oracle_mc_agent.pt"
TD_MODEL_PATH = "models/belief_oracle_td_agent.pt"
RESULTS_PATH = "experiments/round5_belief_oracle_results.json"

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

def train_belief_oracle(method: str = 'mc', save_path: str = MC_MODEL_PATH):
    """Train the Belief Oracle Agent via self-play."""
    method_name = "Monte Carlo" if method == 'mc' else "TD(0)"
    print("=" * 60)
    print(f"  TRAINING: Belief Oracle Agent ({method_name})")
    print(f"  Episodes: {TRAIN_EPISODES}, Batch: {BATCH_SIZE}")
    print(f"  LR: {LEARNING_RATE}")
    print("=" * 60)

    agent = BeliefOracleAgent(temperature=1.0, likelihood_source='cfr_nash')
    trainer = BeliefOracleTrainer(agent, learning_rate=LEARNING_RATE, method=method)

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
        save_path=save_path,
        callback=callback,
    )
    elapsed = time.time() - start_time

    print(f"\n  Training complete in {elapsed:.1f}s")
    if losses:
        print(f"  Final loss: {losses[-1]['loss']:.6f}")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]['avg_chips']:+.3f}")

    return {
        "method": method,
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


def evaluate_agent(agent: BeliefOracleAgent, label: str = ""):
    """Evaluate an agent against all opponents."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION: {label} ({EVAL_ROUNDS} rounds per matchup)")
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
    print(f"\n  Robustness: avg={robustness['avg']:+.4f}, "
          f"worst={robustness['worst_case']:+.4f}, "
          f"rob={robustness['robustness']:+.4f}")

    return results, robustness


# ──────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────

def diagnose_state_coverage(num_games: int = 5000):
    """
    Estimate what fraction of possible (state, my_hand, opp_hand) tuples
    were seen during training.

    In Leduc, the state space is small:
    - my_hand: 3 cards (J, Q, K)
    - opp_hand: 3 cards (but constrained by card removal)
    - board: 4 options (None, J, Q, K)
    - round: 2
    - raises: 0, 1, 2
    - pot configurations: varies

    We enumerate observed (hand, board, opp_hand, round, raises) tuples.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 1: State Coverage")
    print("=" * 60)

    game = LeducGame()
    seen_tuples = set()

    for _ in range(num_games):
        game.reset()
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            opp = game.player_hands[1 - cp]

            state_key = (
                obs.player_hand,
                obs.board,
                opp,
                obs.current_round,
                obs.raises_this_round,
                tuple(sorted(obs.pot)),
            )
            seen_tuples.add(state_key)

            # Random action for exploration
            action = random.choice(obs.legal_actions)
            game.step(action)

    # Estimate total possible tuples
    # Rough upper bound: 3 hands x 4 boards x 3 opp_hands x 2 rounds x 3 raises x ~10 pot configs
    # But many are impossible, so just report what we found
    print(f"  Unique (hand, board, opp, round, raises, pot) tuples seen: {len(seen_tuples)}")
    print(f"  Over {num_games} games with random play")

    return {
        "unique_tuples_seen": len(seen_tuples),
        "num_games": num_games,
    }


def diagnose_value_accuracy(mc_agent: BeliefOracleAgent,
                            td_agent: BeliefOracleAgent,
                            num_games: int = 2000):
    """
    Compare MC and TD(0) value predictions against actual game outcomes.

    For each game, record the agent's predicted V(s, my_hand, opp_hand)
    at the first decision point, then compare to the actual terminal reward.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 2: MC vs TD(0) Value Accuracy")
    print("=" * 60)

    game = LeducGame()
    mc_errors = []
    td_errors = []

    for _ in range(num_games):
        game.reset()

        # Record first prediction for player 0
        cp = game.current_player
        obs = game.get_observation(viewer_id=0)
        opp_hand = game.player_hands[1]
        opp_idx = mc_agent.CARD_MAP[opp_hand]

        with torch.no_grad():
            mc_enc = mc_agent.encode_state_with_opp(obs, viewer_id=0, opp_hand_idx=opp_idx)
            mc_pred = mc_agent.model(mc_enc).item()

            td_enc = td_agent.encode_state_with_opp(obs, viewer_id=0, opp_hand_idx=opp_idx)
            td_pred = td_agent.model(td_enc).item()

        # Play out the game randomly
        while not game.is_finished:
            obs = game.get_observation(viewer_id=game.current_player)
            action = random.choice(obs.legal_actions)
            game.step(action)

        actual = game.get_reward()[0]
        mc_errors.append((mc_pred - actual) ** 2)
        td_errors.append((td_pred - actual) ** 2)

    mc_mse = np.mean(mc_errors)
    td_mse = np.mean(td_errors)

    print(f"  MC value MSE:  {mc_mse:.4f}")
    print(f"  TD value MSE:  {td_mse:.4f}")
    print(f"  Advantage MC:  {td_mse - mc_mse:+.4f} (positive = MC better)")

    return {
        "mc_mse": round(float(mc_mse), 4),
        "td_mse": round(float(td_mse), 4),
        "advantage_mc": round(float(td_mse - mc_mse), 4),
        "num_games": num_games,
    }


def diagnose_value_ordering(agent: BeliefOracleAgent):
    """
    Sanity check: does the value network produce reasonable orderings?

    For example, V(state, my_hand=K, opp_hand=J) should generally be >
    V(state, my_hand=J, opp_hand=K).
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 3: Value Ordering Sanity Checks")
    print("=" * 60)

    # Test at a representative initial state (antes = [1,1], preflop)
    test_obs = Observation(
        player_hand='K',  # Will be overridden per test
        board=None,
        pot=[1, 1],
        current_player=0,
        current_round=0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        is_finished=False,
        raises_this_round=0,
    )

    cards = ['J', 'Q', 'K']
    card_map = {'J': 0, 'Q': 1, 'K': 2}
    results = {}
    correct_orderings = 0
    total_orderings = 0

    # Test all (my_hand, opp_hand) pairs at initial state
    print(f"\n  Values at initial state (pot=[1,1], preflop):")
    print(f"  {'my_hand':>8s} {'opp_hand':>8s} {'V(s)':>8s}")
    print(f"  {'':->8s} {'':->8s} {'':->8s}")

    values_grid = {}
    for my_hand in cards:
        for opp_hand in cards:
            obs = Observation(
                player_hand=my_hand,
                board=None,
                pot=[1, 1],
                current_player=0,
                current_round=0,
                legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                is_finished=False,
                raises_this_round=0,
            )
            opp_idx = card_map[opp_hand]
            with torch.no_grad():
                enc = agent.encode_state_with_opp(obs, viewer_id=0, opp_hand_idx=opp_idx)
                v = agent.model(enc).item()
            values_grid[(my_hand, opp_hand)] = v
            print(f"  {my_hand:>8s} {opp_hand:>8s} {v:>8.3f}")

    # Check orderings:
    # K vs J should be > J vs K
    # Q vs J should be > J vs Q
    # K vs Q should be > Q vs K
    ordering_checks = [
        ('K', 'J', 'J', 'K'),  # K vs J > J vs K
        ('Q', 'J', 'J', 'Q'),  # Q vs J > J vs Q
        ('K', 'Q', 'Q', 'K'),  # K vs Q > Q vs K
        ('K', 'J', 'Q', 'J'),  # K vs J > Q vs J (higher card better)
        ('K', 'Q', 'Q', 'Q'),  # K vs Q > Q vs Q (higher card better, unless pair)
    ]

    print(f"\n  Value ordering checks:")
    for my1, opp1, my2, opp2 in ordering_checks:
        v1 = values_grid[(my1, opp1)]
        v2 = values_grid[(my2, opp2)]
        passed = v1 > v2
        total_orderings += 1
        if passed:
            correct_orderings += 1
        status = "PASS" if passed else "FAIL"
        print(f"    V({my1} vs {opp1})={v1:.3f} > V({my2} vs {opp2})={v2:.3f}: {status}")

    # Also test on flop with board card
    print(f"\n  Values at flop state (pot=[3,3], board=Q):")
    print(f"  {'my_hand':>8s} {'opp_hand':>8s} {'V(s)':>8s}")
    print(f"  {'':->8s} {'':->8s} {'':->8s}")

    flop_values = {}
    for my_hand in cards:
        for opp_hand in cards:
            obs = Observation(
                player_hand=my_hand,
                board='Q',
                pot=[3, 3],
                current_player=0,
                current_round=1,
                legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                is_finished=False,
                raises_this_round=0,
            )
            opp_idx = card_map[opp_hand]
            with torch.no_grad():
                enc = agent.encode_state_with_opp(obs, viewer_id=0, opp_hand_idx=opp_idx)
                v = agent.model(enc).item()
            flop_values[(my_hand, opp_hand)] = v
            print(f"  {my_hand:>8s} {opp_hand:>8s} {v:>8.3f}")

    # Q (pair) vs K should be > K vs Q (no pair)
    flop_checks = [
        ('Q', 'K', 'K', 'J'),  # Q-pair vs K should be > K vs J
        ('Q', 'J', 'J', 'K'),  # Q-pair vs J should be > J vs K
    ]
    for my1, opp1, my2, opp2 in flop_checks:
        v1 = flop_values[(my1, opp1)]
        v2 = flop_values[(my2, opp2)]
        passed = v1 > v2
        total_orderings += 1
        if passed:
            correct_orderings += 1
        status = "PASS" if passed else "FAIL"
        print(f"    V({my1} vs {opp1}, board=Q)={v1:.3f} > "
              f"V({my2} vs {opp2}, board=Q)={v2:.3f}: {status}")

    pct = correct_orderings / total_orderings if total_orderings > 0 else 0
    print(f"\n  Ordering correctness: {correct_orderings}/{total_orderings} ({pct:.1%})")

    return {
        "preflop_values": {f"{k[0]}_vs_{k[1]}": round(v, 4) for k, v in values_grid.items()},
        "flop_values_board_Q": {f"{k[0]}_vs_{k[1]}": round(v, 4) for k, v in flop_values.items()},
        "ordering_correct": correct_orderings,
        "ordering_total": total_orderings,
        "ordering_pct": round(pct, 4),
    }


def diagnose_action_distribution(agent: BeliefOracleAgent, num_games: int = 2000):
    """
    Analyze the agent's action distribution per hand card.

    Shows whether the agent plays differently depending on hand strength
    (which it should, since it uses belief-weighted values).
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 4: Action Distribution Per Hand")
    print("=" * 60)

    game = LeducGame()
    # Track: action_counts[hand][round][action_name] = count
    action_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    agent.set_train_mode(False)

    for _ in range(num_games):
        game.reset()

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                # Agent's turn
                action = agent.select_action(obs)
                hand = obs.player_hand
                rnd = obs.current_round
                action_counts[hand][rnd][action.name] += 1
            else:
                # Opponent (heuristic)
                heuristic = HeuristicAgent()
                action = heuristic.select_action(obs)

            game.step(action)

    print(f"\n  Action distribution (agent as P0, {num_games} games):")
    for hand in ['J', 'Q', 'K']:
        for rnd in [0, 1]:
            rnd_name = "preflop" if rnd == 0 else "flop"
            counts = action_counts[hand][rnd]
            total = sum(counts.values())
            if total == 0:
                continue
            dist_str = ", ".join(
                f"{a}: {c/total:.3f}" for a, c in sorted(counts.items())
            )
            print(f"    {hand} ({rnd_name}): {dist_str}  (n={total})")

    # Convert to serializable format
    result = {}
    for hand in ['J', 'Q', 'K']:
        result[hand] = {}
        for rnd in [0, 1]:
            rnd_name = "preflop" if rnd == 0 else "flop"
            counts = action_counts[hand][rnd]
            total = sum(counts.values())
            if total > 0:
                result[hand][rnd_name] = {
                    a: round(c / total, 4) for a, c in sorted(counts.items())
                }
                result[hand][f"{rnd_name}_n"] = total

    return result


def diagnose_belief_quality(agent: BeliefOracleAgent, num_games: int = 1000):
    """
    Assess the quality of belief updates by comparing belief vectors
    to the true opponent hand.

    Metrics:
    - Average probability assigned to the TRUE opponent hand
    - Calibration: does belief[true_hand] trend toward 1 over time?
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 5: Belief Quality Metrics")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()

    # Track belief accuracy over decision points within a hand
    step_accuracies = defaultdict(list)  # step_idx -> list of P(true_hand)

    for _ in range(num_games):
        game.reset()
        step_idx = 0

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                # Belief agent
                belief = agent.compute_belief_from_history(obs)
                true_opp_hand = game.player_hands[1]
                true_idx = agent.CARD_MAP[true_opp_hand]
                prob_true = belief[true_idx]
                step_accuracies[step_idx].append(prob_true)
                step_idx += 1
                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            game.step(action)

    print(f"\n  Belief accuracy (P(true_opp_hand)) by decision step:")
    step_results = {}
    for step in sorted(step_accuracies.keys()):
        vals = step_accuracies[step]
        mean_p = np.mean(vals)
        n = len(vals)
        print(f"    Step {step}: mean P(true) = {mean_p:.3f}  (n={n})")
        step_results[f"step_{step}"] = {
            "mean_p_true": round(float(mean_p), 4),
            "n": n,
        }

    # Overall
    all_probs = []
    for vals in step_accuracies.values():
        all_probs.extend(vals)
    overall_mean = np.mean(all_probs) if all_probs else 0
    print(f"\n  Overall mean P(true_opp_hand): {overall_mean:.3f}")

    return {
        "per_step": step_results,
        "overall_mean_p_true": round(float(overall_mean), 4),
        "num_games": num_games,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ROUND 5 — E2b: Belief Oracle Agent")
    print("=" * 60)

    all_results = {
        "agent": "belief_oracle",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "eval_rounds": EVAL_ROUNDS,
        },
    }

    # Phase 1: Training MC
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1a: TRAINING (Monte Carlo)")
    print(f"{'#' * 60}")
    mc_training = train_belief_oracle(method='mc', save_path=MC_MODEL_PATH)
    all_results["training_mc"] = mc_training

    # Phase 1b: Training TD(0)
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1b: TRAINING (TD(0))")
    print(f"{'#' * 60}")
    td_training = train_belief_oracle(method='td', save_path=TD_MODEL_PATH)
    all_results["training_td"] = td_training

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")

    mc_agent = BeliefOracleAgent(model_path=MC_MODEL_PATH, likelihood_source='cfr_nash')
    mc_agent.set_train_mode(False)
    mc_eval, mc_robustness = evaluate_agent(mc_agent, label="MC Agent")
    all_results["evaluation_mc"] = mc_eval
    all_results["robustness_mc"] = mc_robustness

    td_agent = BeliefOracleAgent(model_path=TD_MODEL_PATH, likelihood_source='cfr_nash')
    td_agent.set_train_mode(False)
    td_eval, td_robustness = evaluate_agent(td_agent, label="TD(0) Agent")
    all_results["evaluation_td"] = td_eval
    all_results["robustness_td"] = td_robustness

    # Phase 3: Diagnostics
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: DIAGNOSTICS")
    print(f"{'#' * 60}")

    diagnostics = {}

    # Diagnostic 1: State coverage
    diagnostics["state_coverage"] = diagnose_state_coverage()

    # Diagnostic 2: MC vs TD(0) value accuracy
    diagnostics["value_accuracy"] = diagnose_value_accuracy(mc_agent, td_agent)

    # Diagnostic 3: Value ordering (use MC agent as primary)
    diagnostics["value_ordering"] = diagnose_value_ordering(mc_agent)

    # Diagnostic 4: Action distribution
    diagnostics["action_distribution"] = diagnose_action_distribution(mc_agent)

    # Diagnostic 5: Belief quality
    diagnostics["belief_quality"] = diagnose_belief_quality(mc_agent)

    all_results["diagnostics"] = diagnostics

    # Phase 4: Save results
    os.makedirs(os.path.dirname(RESULTS_PATH) if os.path.dirname(RESULTS_PATH) else ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  MC training time:  {mc_training['training_time_s']}s")
    print(f"  TD training time:  {td_training['training_time_s']}s")
    print(f"  MC final loss:     {mc_training.get('final_loss', 'N/A')}")
    print(f"  TD final loss:     {td_training.get('final_loss', 'N/A')}")
    print(f"\n  MC Evaluation (avg chips/round):")
    for name, score in mc_eval.items():
        print(f"    vs {name:20s}: {score:+.4f}")
    print(f"  MC Robustness: {mc_robustness['robustness']:+.4f}")
    print(f"\n  TD Evaluation (avg chips/round):")
    for name, score in td_eval.items():
        print(f"    vs {name:20s}: {score:+.4f}")
    print(f"  TD Robustness: {td_robustness['robustness']:+.4f}")

    if 'value_accuracy' in diagnostics:
        va = diagnostics['value_accuracy']
        print(f"\n  Value accuracy: MC MSE={va['mc_mse']:.4f}, TD MSE={va['td_mse']:.4f}")
    if 'value_ordering' in diagnostics:
        vo = diagnostics['value_ordering']
        print(f"  Value ordering: {vo['ordering_correct']}/{vo['ordering_total']} correct")
    if 'belief_quality' in diagnostics:
        bq = diagnostics['belief_quality']
        print(f"  Belief quality: mean P(true) = {bq['overall_mean_p_true']:.3f}")

    return all_results


if __name__ == "__main__":
    main()
