"""
Quick E2c experiment: Belief Confident Agent

Tests whether adding a 15th input dim (confidence = min(n_games, 30)/30) to the
belief value network allows the model to learn trust-calibrated strategies --
ignoring unreliable beliefs early in a session and trusting them later.

Training protocol:
  - 2000 sessions x 30 hands = 60K total hands
  - Population-based opponent rotation (heuristic, value_based, adaptive_value)
  - Session-based: confidence naturally increases 0 -> 1 within each session
  - TD(0) value learning + cross-entropy likelihood model training
  - Both value and likelihood update per batch

Evaluation: 500 rounds against 6 opponents (both positions).

Diagnostics:
  1. Confidence effect: action distribution at conf=0 vs conf=1 (TVD)
  2. Belief correctness: P(true opp hand) and entropy
  3. Ablation: performance at conf=0, 0.5, 1.0
  4. Action distribution per hand card (overall + per round)
"""

import json
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.poker_session import PokerSession
from src.engine.observation import Observation
from src.agents.belief_confident import BeliefConfidentAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.evaluation import evaluate_agents, compute_robustness_metrics

# --------------------- Configuration ---------------------

TRAIN_SESSIONS = 2000
HANDS_PER_SESSION = 30
BATCH_SIZE = 32           # hands per gradient update
LEARNING_RATE = 1e-4
LIKELIHOOD_LR = 5e-4
ROTATE_EVERY = 100        # rotate opponent every N sessions
EVAL_INTERVAL = 100       # evaluate every N sessions
EVAL_ROUNDS = 500
MODEL_PATH = "models/belief_confident_agent.pt"
RESULTS_PATH = "experiments/round5_belief_confident_results.json"


# --------------------- Opponent pool ---------------------

def build_opponent_pool():
    """Load pre-trained opponents for population-based training."""
    pool = []

    pool.append(("heuristic", HeuristicAgent()))

    vb = ValueBasedAgent()
    if os.path.exists("models/value_based_agent.pt"):
        vb.load_model("models/value_based_agent.pt")
    vb.set_train_mode(False)
    pool.append(("value_based", vb))

    try:
        from src.agents.adaptive_value import AdaptiveValueAgent
        av = AdaptiveValueAgent()
        if os.path.exists("models/adaptive_value_agent.pt"):
            av.load_model("models/adaptive_value_agent.pt")
        av.set_train_mode(False)
        pool.append(("adaptive_value", av))
    except Exception as e:
        print(f"  WARNING: Could not load adaptive_value: {e}")

    return pool


def load_eval_opponents():
    """Load all 6 evaluation opponents."""
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

    return opponents


# --------------------- Training ---------------------

def train_belief_confident_agent():
    """
    Population-based session training for BeliefConfidentAgent.

    Within each 30-hand session, confidence naturally increases from 0 to 1
    as games accumulate. The population rotates every ROTATE_EVERY sessions.
    """
    print("=" * 60)
    print("  TRAINING: Belief Confident Agent")
    print(f"  Sessions: {TRAIN_SESSIONS}, Hands/session: {HANDS_PER_SESSION}")
    print(f"  Total hands: {TRAIN_SESSIONS * HANDS_PER_SESSION}")
    print(f"  Value LR: {LEARNING_RATE}, Likelihood LR: {LIKELIHOOD_LR}")
    print("=" * 60)

    agent = BeliefConfidentAgent(temperature=1.0)
    agent.set_train_mode(True)

    value_optimizer = optim.Adam(agent.model.parameters(), lr=LEARNING_RATE)
    likelihood_optimizer = optim.Adam(
        agent.likelihood_model.parameters(), lr=LIKELIHOOD_LR
    )
    value_criterion = nn.MSELoss()
    likelihood_criterion = nn.NLLLoss()

    opponent_pool = build_opponent_pool()
    current_opp_idx = 0

    session = PokerSession()
    losses = []
    eval_scores = []

    # Batch accumulators
    batch_chains = []        # list of (chain, reward_for_player)
    batch_likelihood = []    # list of (hand_idx, obs, action)
    batch_hand_count = 0

    start_time = time.time()

    for sess_idx in range(TRAIN_SESSIONS):
        session.reset()
        opponent = opponent_pool[current_opp_idx][1]
        opp_name = opponent_pool[current_opp_idx][0]

        # Reset confidence at start of session
        agent.reset_session()

        for hand_idx_in_session in range(HANDS_PER_SESSION):
            session.new_hand()

            # Confidence for this hand
            confidence = agent.confidence

            chains_p0 = []  # post-action states for player 0 (our agent)

            while not session.is_finished:
                current_player = session.current_player
                obs = session.get_observation(viewer_id=current_player)

                if current_player == 0:
                    # Our agent
                    action = agent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    # Record post-action state for TD(0)
                    belief = agent.compute_belief_from_history(obs)
                    post_obs, _ = LeducGame.simulate_action(obs, action)
                    encoded = agent.encode_observation(
                        post_obs, viewer_id=0, belief=belief, confidence=confidence
                    )
                    chains_p0.append(encoded)
                else:
                    # Opponent
                    action = opponent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    # Record likelihood training data: opponent's hand + action
                    opp_hand = session.game.player_hands[current_player]
                    opp_hand_idx = agent.CARD_MAP.get(opp_hand)
                    if opp_hand_idx is not None:
                        batch_likelihood.append((opp_hand_idx, obs, action))

                session.step(action)

            rewards = session.game.get_reward()

            # Store TD chain for player 0
            if chains_p0:
                batch_chains.append((chains_p0, rewards[0]))

            batch_hand_count += 1

            # Increment game count after each completed hand
            agent.increment_game_count()

            # Update when batch is full
            if batch_hand_count >= BATCH_SIZE:
                # --- Value network update (TD(0)) ---
                value_optimizer.zero_grad()
                value_losses = []
                for chain, terminal_reward in batch_chains:
                    for t in range(len(chain)):
                        pred = agent.model(chain[t]).squeeze(0)
                        if t == len(chain) - 1:
                            target = torch.FloatTensor([terminal_reward])
                        else:
                            with torch.no_grad():
                                target = agent.model(chain[t + 1]).squeeze(0)
                        value_losses.append(value_criterion(pred, target))

                if value_losses:
                    v_loss = torch.stack(value_losses).mean()
                    v_loss.backward()
                    value_optimizer.step()
                    v_loss_val = v_loss.item()
                else:
                    v_loss_val = 0.0

                # --- Likelihood model update (NLL) ---
                likelihood_optimizer.zero_grad()
                ll_losses = []
                for opp_h_idx, ll_obs, ll_action in batch_likelihood:
                    inp = agent._encode_likelihood_input(opp_h_idx, ll_obs)
                    log_probs = agent.likelihood_model(inp)
                    action_target = torch.LongTensor([int(ll_action)])
                    ll_losses.append(likelihood_criterion(log_probs, action_target))

                if ll_losses:
                    ll_loss = torch.stack(ll_losses).mean()
                    ll_loss.backward()
                    likelihood_optimizer.step()
                    ll_loss_val = ll_loss.item()
                else:
                    ll_loss_val = 0.0

                combined_loss = v_loss_val + 0.1 * ll_loss_val
                losses.append(combined_loss)

                # Clear batch
                batch_chains = []
                batch_likelihood = []
                batch_hand_count = 0

        # Rotate opponent
        if (sess_idx + 1) % ROTATE_EVERY == 0:
            current_opp_idx = (current_opp_idx + 1) % len(opponent_pool)
            print(f"  Session {sess_idx + 1}: rotating opponent to "
                  f"{opponent_pool[current_opp_idx][0]}")

        # Periodic evaluation
        if (sess_idx + 1) % EVAL_INTERVAL == 0:
            agent.set_train_mode(False)
            agent.set_game_count(15)  # mid-session confidence for eval
            heuristic = HeuristicAgent()
            from src.training.evaluation import quick_evaluate
            avg_chips = quick_evaluate(agent, heuristic, num_rounds=100)
            eval_scores.append({
                "session": sess_idx + 1,
                "avg_chips": round(avg_chips, 4),
            })
            agent.set_train_mode(True)
            print(f"  Session {sess_idx + 1}/{TRAIN_SESSIONS}: "
                  f"loss={losses[-1]:.4f}, eval={avg_chips:+.3f}")

    elapsed = time.time() - start_time

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else ".", exist_ok=True)
    agent.save_model(MODEL_PATH)
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Model saved to {MODEL_PATH}")
    if losses:
        print(f"  Final loss: {losses[-1]:.4f}")

    return agent, {
        "training_time_s": round(elapsed, 1),
        "num_updates": len(losses),
        "final_loss": round(losses[-1], 6) if losses else None,
        "eval_history": eval_scores[-10:],
    }


# --------------------- Evaluation ---------------------

def evaluate_against_opponents(agent):
    """Evaluate against 6 opponents, 500 rounds each."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION ({EVAL_ROUNDS} rounds per matchup, both positions)")
    print("=" * 60)

    agent.set_train_mode(False)
    agent.set_game_count(15)  # mid-session confidence

    opponents = load_eval_opponents()
    eval_results = {}

    for name, opp in opponents.items():
        result = evaluate_agents(agent, opp, num_rounds=EVAL_ROUNDS)
        avg = result.agent_0_avg_chips
        eval_results[name] = round(avg, 4)
        print(f"  vs {name:20s}: {avg:+.4f}")

    robustness = compute_robustness_metrics(eval_results)
    print(f"\n  Avg: {robustness['avg']:+.4f}")
    print(f"  Worst: {robustness['worst_case']:+.4f}")
    print(f"  Best: {robustness['best_case']:+.4f}")
    print(f"  Std: {robustness['std']:.4f}")
    print(f"  Robustness: {robustness['robustness']:+.4f}")

    return eval_results, robustness


# --------------------- Diagnostics ---------------------

def diagnose_confidence_effect(agent, num_games=1000):
    """
    Diag 1: Compare action distributions at confidence=0 vs confidence=1.

    For the same game states, check if the agent behaves differently
    when it trusts vs doesn't trust its belief.
    """
    print("\n" + "=" * 60)
    print("  DIAG 1: Strategy Change with Confidence Level")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    action_counts_low = {"FOLD": 0, "CALL": 0, "RAISE": 0}
    action_counts_high = {"FOLD": 0, "CALL": 0, "RAISE": 0}

    for _ in range(num_games):
        game.reset()
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                # Evaluate at low confidence
                agent.set_game_count(0)
                evals_low = agent.get_action_evaluations(obs)
                best_low = max(evals_low, key=lambda x: x["value"])
                action_counts_low[best_low["action"].name] += 1

                # Evaluate at high confidence
                agent.set_game_count(30)
                evals_high = agent.get_action_evaluations(obs)
                best_high = max(evals_high, key=lambda x: x["value"])
                action_counts_high[best_high["action"].name] += 1

                action = best_high["action"]
            else:
                action = heuristic.select_action(obs)

            game.step(action)

    total_low = sum(action_counts_low.values())
    total_high = sum(action_counts_high.values())

    dist_low = {k: round(v / total_low, 4) if total_low > 0 else 0
                for k, v in action_counts_low.items()}
    dist_high = {k: round(v / total_high, 4) if total_high > 0 else 0
                 for k, v in action_counts_high.items()}

    print(f"  Confidence=0 (no trust):  F={dist_low['FOLD']:.4f}  "
          f"C={dist_low['CALL']:.4f}  R={dist_low['RAISE']:.4f}")
    print(f"  Confidence=1 (full trust): F={dist_high['FOLD']:.4f}  "
          f"C={dist_high['CALL']:.4f}  R={dist_high['RAISE']:.4f}")

    tvd = sum(abs(dist_low[a] - dist_high[a]) for a in dist_low) / 2
    print(f"\n  Total variation distance: {tvd:.4f}")
    print(f"  (0 = identical strategies, 1 = completely different)")

    return {
        "action_dist_confidence_0": dist_low,
        "action_dist_confidence_1": dist_high,
        "total_variation_distance": round(tvd, 4),
        "total_decisions": total_low,
    }


def diagnose_belief_correctness(agent, num_games=500):
    """
    Diag 2: How well does the belief track the true opponent hand?
    """
    print("\n" + "=" * 60)
    print("  DIAG 2: Belief Correctness")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)
    agent.set_game_count(15)

    true_hand_probs = []
    belief_entropies = []

    for _ in range(num_games):
        game.reset()
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            if cp == 0:
                belief = agent.compute_belief_from_history(obs)
                true_hand = game.player_hands[1]
                true_idx = agent.CARD_MAP[true_hand]
                true_hand_probs.append(belief[true_idx])

                entropy = -np.sum(belief * np.log(np.maximum(belief, 1e-10)))
                belief_entropies.append(entropy)

                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            game.step(action)

    avg_true_prob = np.mean(true_hand_probs) if true_hand_probs else 0
    avg_entropy = np.mean(belief_entropies) if belief_entropies else 0

    print(f"  Total decision points: {len(true_hand_probs)}")
    print(f"  Avg P(true opponent hand): {avg_true_prob:.4f}")
    print(f"  Avg belief entropy: {avg_entropy:.4f}")
    print(f"  (Random baseline: ~0.40 true prob, ~1.05 entropy)")

    return {
        "avg_true_hand_prob": round(float(avg_true_prob), 4),
        "avg_belief_entropy": round(float(avg_entropy), 4),
        "total_decisions": len(true_hand_probs),
    }


def diagnose_ablation_confidence_levels(agent, num_games=500):
    """
    Diag 3: Performance at confidence=0, 0.5, 1.0 vs heuristic.

    Tests whether the model has learned to USE the confidence signal:
    does performance improve when confidence is high (and belief is reliable)?
    """
    print("\n" + "=" * 60)
    print("  DIAG 3: Ablation -- Performance at Different Confidence Levels")
    print("=" * 60)

    agent.set_train_mode(False)
    heuristic = HeuristicAgent()

    results_by_conf = {}
    for conf_label, n_games_val in [("0.0", 0), ("0.5", 15), ("1.0", 30)]:
        agent.set_game_count(n_games_val)
        result = evaluate_agents(agent, heuristic, num_rounds=num_games)
        avg = result.agent_0_avg_chips
        results_by_conf[f"conf_{conf_label}"] = round(avg, 4)
        print(f"  Confidence={conf_label}: {avg:+.4f} chips/round")

    delta = results_by_conf["conf_1.0"] - results_by_conf["conf_0.0"]
    print(f"\n  Delta (high - low): {delta:+.4f}")

    results_by_conf["delta_high_minus_low"] = round(delta, 4)
    results_by_conf["num_games"] = num_games
    return results_by_conf


def diagnose_action_distribution(agent, num_games=1000):
    """
    Diag 4: Action distribution per hand card (J/Q/K) and per round.
    """
    print("\n" + "=" * 60)
    print("  DIAG 4: Action Distribution per Hand")
    print("=" * 60)

    game = LeducGame()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)
    agent.set_game_count(15)

    action_counts = {c: {0: 0, 1: 0, 2: 0} for c in ['J', 'Q', 'K']}
    round_counts = {
        0: {c: {0: 0, 1: 0, 2: 0} for c in ['J', 'Q', 'K']},
        1: {c: {0: 0, 1: 0, 2: 0} for c in ['J', 'Q', 'K']},
    }

    for _ in range(num_games):
        game.reset()
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            if cp == 0:
                action = agent.select_action(obs)
                hand = obs.player_hand
                if hand in action_counts:
                    action_counts[hand][int(action)] += 1
                    round_counts[obs.current_round][hand][int(action)] += 1
            else:
                action = heuristic.select_action(obs)
            game.step(action)

    action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}
    results = {}

    print(f"\n  Overall:")
    for hand in ['J', 'Q', 'K']:
        total = sum(action_counts[hand].values())
        if total > 0:
            dist = {action_names[a]: round(action_counts[hand][a] / total, 3)
                    for a in range(3)}
        else:
            dist = {action_names[a]: 0.0 for a in range(3)}
        results[hand] = dist
        print(f"    {hand}: F={dist['FOLD']:.3f}  C={dist['CALL']:.3f}  "
              f"R={dist['RAISE']:.3f}  (n={total})")

    round_results = {}
    for rnd, label in [(0, "preflop"), (1, "flop")]:
        round_results[label] = {}
        print(f"\n  {label.capitalize()}:")
        for hand in ['J', 'Q', 'K']:
            total = sum(round_counts[rnd][hand].values())
            if total > 0:
                dist = {action_names[a]: round(round_counts[rnd][hand][a] / total, 3)
                        for a in range(3)}
            else:
                dist = {action_names[a]: 0.0 for a in range(3)}
            round_results[label][hand] = dist
            print(f"    {hand}: F={dist['FOLD']:.3f}  C={dist['CALL']:.3f}  "
                  f"R={dist['RAISE']:.3f}  (n={total})")

    return {"overall": results, "per_round": round_results}


# --------------------- Main ---------------------

def main():
    print("=" * 60)
    print("  ROUND 5 -- E2c: Belief Confident Agent")
    print("  (Belief + confidence score, session-based training)")
    print("=" * 60)

    all_results = {
        "agent": "belief_confident",
        "experiment": "E2c",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_sessions": TRAIN_SESSIONS,
            "hands_per_session": HANDS_PER_SESSION,
            "total_hands": TRAIN_SESSIONS * HANDS_PER_SESSION,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "likelihood_lr": LIKELIHOOD_LR,
            "rotate_every": ROTATE_EVERY,
            "confidence_cap": BeliefConfidentAgent.CONFIDENCE_CAP,
            "eval_rounds": EVAL_ROUNDS,
        },
    }

    # Phase 1: Training
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1: TRAINING")
    print(f"{'#' * 60}")
    agent, training_results = train_belief_confident_agent()
    all_results["training"] = training_results

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")
    # Reload from saved model to be sure
    agent = BeliefConfidentAgent(model_path=MODEL_PATH)
    eval_results, robustness = evaluate_against_opponents(agent)
    all_results["evaluation"] = eval_results
    all_results["robustness"] = robustness

    # Phase 3: Diagnostics
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: DIAGNOSTICS")
    print(f"{'#' * 60}")

    diag1 = diagnose_confidence_effect(agent)
    diag2 = diagnose_belief_correctness(agent)
    diag3 = diagnose_ablation_confidence_levels(agent)
    diag4 = diagnose_action_distribution(agent)

    all_results["diagnostics"] = {
        "confidence_effect": diag1,
        "belief_correctness": diag2,
        "ablation_confidence_levels": diag3,
        "action_distribution": diag4,
    }

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: E2c Belief Confident")
    print(f"{'=' * 60}")
    print(f"  Training: {training_results['training_time_s']}s, "
          f"{training_results['num_updates']} updates")
    if training_results.get('final_loss') is not None:
        print(f"  Final loss: {training_results['final_loss']:.4f}")
    print(f"\n  Evaluation (avg chips/round):")
    for name, score in eval_results.items():
        print(f"    vs {name:20s}: {score:+.4f}")
    print(f"\n  Robustness: {robustness['robustness']:+.4f}")
    print(f"  Avg: {robustness['avg']:+.4f}, Worst: {robustness['worst_case']:+.4f}")
    print(f"\n  Confidence effect (TVD): {diag1['total_variation_distance']:.4f}")
    print(f"  Belief correctness: P(true)={diag2['avg_true_hand_prob']:.4f}")
    conf_delta = diag3.get('delta_high_minus_low', 0)
    print(f"  Ablation delta (high-low conf): {conf_delta:+.4f}")
    print(f"{'=' * 60}")

    return all_results


if __name__ == "__main__":
    main()
