"""
Round 4 -- Direction 5: Information-Hiding Agent

Implements and evaluates an actor-critic agent with an adversarial spy network
that learns to make the agent's hand unpredictable from its action history.

Protocol:
  1. Train for 40K episodes with lambda=0.1 via self-play
  2. Evaluate against 6 opponents (500 rounds each, both positions)
  3. Sweep lambda in [0.0, 0.05, 0.1, 0.2, 0.5] with 10K episodes each
  4. Diagnose: spy accuracy, action distributions by hand, comparison to baselines
  5. Save results + model
"""

import json
import os
import sys
import time
import copy
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.info_hiding import InfoHidingAgent, SpyNetwork
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.info_hiding_trainer import (
    InfoHidingTrainer, encode_action_sequence, CARD_IDX,
)
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TRAIN_EPISODES = 40000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SPY_LR = 1e-3
VALUE_COEFF = 0.5
INFO_HIDING_COEFF = 0.1          # lambda for main run
EVAL_ROUNDS = 500
MODEL_PATH = "models/info_hiding_agent.pt"
RESULTS_PATH = "experiments/round4_info_hiding_results.json"

LAMBDA_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.5]
SWEEP_EPISODES = 10000

OPPONENTS = {
    "heuristic": {"class": "HeuristicAgent", "model_path": None},
    "value_based": {"class": "ValueBasedAgent", "model_path": "models/value_based_agent.pt"},
    "adaptive_value": {"class": "AdaptiveValueAgent", "model_path": "models/adaptive_value_agent.pt"},
    "modulated_value": {"class": "ModulatedValueAgent", "model_path": "models/modulated_value_agent.pt"},
    "entropy_ac": {"class": "EntropyACAgent", "model_path": "models/entropy_ac_agent.pt"},
    "cfr": {"class": "CFRAgent", "model_path": "models/cfr_agent.pt"},
}


# ──────────────────────────────────────────────
# Helpers
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
        from src.agents.modulated_value import ModulatedValueAgent
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


def collect_play_data(agent, num_games=1000):
    """Play num_games with the agent (self-play) and collect action sequences
    plus hand cards.  Returns a list of dicts with keys:
      hand_card, action_seq, pot, round_reached, total_raises, player
    """
    game = LeducGame()
    data = []
    agent.set_train_mode(False)

    for _ in range(num_games):
        game.reset()
        action_seqs = [[], []]
        total_raises = 0

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = agent.select_action(obs)
            action_seqs[cp].append(action)
            if action == Action.RAISE:
                total_raises += 1
            game.step(action)

        for p in [0, 1]:
            if action_seqs[p]:
                data.append({
                    "hand_card": game.player_hands[p],
                    "action_seq": action_seqs[p],
                    "pot": list(game.pot),
                    "round_reached": game.current_round,
                    "total_raises": total_raises,
                    "player": p,
                })

    return data


def train_fresh_spy(play_data, epochs=20, lr=1e-3):
    """Train a fresh spy network on the given play data and return its accuracy."""
    spy = SpyNetwork(input_size=20, hidden_size=32)
    optimizer = torch.optim.Adam(spy.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Build tensors
    inputs = []
    labels = []
    for d in play_data:
        inp = encode_action_sequence(
            d["action_seq"], d["pot"], d["round_reached"], d["total_raises"]
        )
        inputs.append(inp)
        labels.append(CARD_IDX[d["hand_card"]])

    X = torch.stack(inputs)
    Y = torch.tensor(labels, dtype=torch.long)

    # Train/test split (80/20)
    n = len(X)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    spy.train()
    for epoch in range(epochs):
        # Mini-batch SGD
        batch_perm = torch.randperm(len(X_train))
        for start in range(0, len(X_train), 64):
            idx = batch_perm[start:start + 64]
            logits = spy(X_train[idx])
            loss = loss_fn(logits, Y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test accuracy
    spy.eval()
    with torch.no_grad():
        preds = spy(X_test).argmax(dim=-1)
        accuracy = (preds == Y_test).float().mean().item()

    return accuracy


# ──────────────────────────────────────────────
# Phase 1: Training
# ──────────────────────────────────────────────

def train_info_hiding_agent():
    """Train the Information-Hiding Agent via self-play."""
    print("=" * 60)
    print("  TRAINING: Information-Hiding Agent")
    print(f"  Episodes: {TRAIN_EPISODES}, Batch: {BATCH_SIZE}")
    print(f"  AC LR: {LEARNING_RATE}, Spy LR: {SPY_LR}")
    print(f"  value_coeff: {VALUE_COEFF}, info_hiding_coeff (lambda): {INFO_HIDING_COEFF}")
    print("=" * 60)

    agent = InfoHidingAgent()
    trainer = InfoHidingTrainer(
        agent,
        learning_rate=LEARNING_RATE,
        spy_lr=SPY_LR,
        value_coeff=VALUE_COEFF,
        info_hiding_coeff=INFO_HIDING_COEFF,
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
                "avg_chips": data["avg_chips_per_round"],
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
# Phase 2: Evaluation
# ──────────────────────────────────────────────

def evaluate_against_opponents(agent):
    """Evaluate the info-hiding agent against all opponents."""
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
    print(f"    Avg:        {robustness['avg']:+.4f}")
    print(f"    Worst:      {robustness['worst_case']:+.4f}")
    print(f"    Best:       {robustness['best_case']:+.4f}")
    print(f"    Std:        {robustness['std']:.4f}")
    print(f"    Robustness: {robustness['robustness']:+.4f}")

    return results, robustness


# ──────────────────────────────────────────────
# Phase 3: Lambda sweep
# ──────────────────────────────────────────────

def lambda_sweep():
    """Sweep info_hiding_coeff and evaluate each briefly."""
    print("\n" + "=" * 60)
    print(f"  LAMBDA SWEEP  ({SWEEP_EPISODES} episodes each)")
    print("=" * 60)

    sweep_results = {}

    for lam in LAMBDA_SWEEP:
        print(f"\n  --- lambda = {lam} ---")
        agent = InfoHidingAgent()
        trainer = InfoHidingTrainer(
            agent,
            learning_rate=LEARNING_RATE,
            spy_lr=SPY_LR,
            value_coeff=VALUE_COEFF,
            info_hiding_coeff=lam,
        )

        start_t = time.time()
        trainer.train(num_episodes=SWEEP_EPISODES, batch_size=BATCH_SIZE)
        elapsed = time.time() - start_t

        # Quick evaluation against heuristic and CFR
        agent.set_train_mode(False)
        heuristic = HeuristicAgent()
        h_result = evaluate_agents(agent, heuristic, num_rounds=300)
        vs_heuristic = round(h_result.agent_0_avg_chips, 4)

        try:
            from src.agents.cfr_agent import CFRAgent
            cfr = CFRAgent()
            if os.path.exists("models/cfr_agent.pt"):
                cfr.load_model("models/cfr_agent.pt")
            c_result = evaluate_agents(agent, cfr, num_rounds=300)
            vs_cfr = round(c_result.agent_0_avg_chips, 4)
        except Exception:
            vs_cfr = None

        # Measure spy accuracy with a fresh spy
        play_data = collect_play_data(agent, num_games=500)
        spy_acc = train_fresh_spy(play_data) if play_data else None

        sweep_results[str(lam)] = {
            "vs_heuristic": vs_heuristic,
            "vs_cfr": vs_cfr,
            "spy_accuracy": round(spy_acc, 4) if spy_acc is not None else None,
            "train_time_s": round(elapsed, 1),
        }

        print(f"    vs heuristic: {vs_heuristic:+.4f}")
        if vs_cfr is not None:
            print(f"    vs cfr:       {vs_cfr:+.4f}")
        if spy_acc is not None:
            print(f"    spy accuracy: {spy_acc:.4f}")

    return sweep_results


# ──────────────────────────────────────────────
# Phase 4: Diagnostics
# ──────────────────────────────────────────────

def diagnose_spy_accuracy(agent, label="info_hiding", num_games=1000):
    """Train a FRESH spy on the agent's games to measure how predictable
    the agent's play is.  Returns the test accuracy."""
    print(f"\n  Spy accuracy for {label}:")
    play_data = collect_play_data(agent, num_games=num_games)
    if not play_data:
        print(f"    No data collected")
        return None
    acc = train_fresh_spy(play_data, epochs=30)
    print(f"    Fresh spy accuracy: {acc:.4f}  (random baseline: 0.333)")
    return round(acc, 4)


def diagnose_predictability(ih_agent):
    """Compare spy accuracy across agents: info_hiding, value_based, entropy_ac, cfr."""
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 1: Hand Predictability Comparison")
    print("=" * 60)

    results = {}

    # Info-hiding agent
    results["info_hiding"] = diagnose_spy_accuracy(ih_agent, "info_hiding")

    # Value-based
    try:
        vb = ValueBasedAgent(model_path="models/value_based_agent.pt")
        vb.set_train_mode(False)
        results["value_based"] = diagnose_spy_accuracy(vb, "value_based")
    except Exception as e:
        print(f"    Could not test value_based: {e}")

    # Entropy AC
    try:
        from src.agents.entropy_ac import EntropyACAgent
        eac = EntropyACAgent(model_path="models/entropy_ac_agent.pt")
        eac.set_train_mode(False)
        results["entropy_ac"] = diagnose_spy_accuracy(eac, "entropy_ac")
    except Exception as e:
        print(f"    Could not test entropy_ac: {e}")

    # CFR
    try:
        from src.agents.cfr_agent import CFRAgent
        cfr = CFRAgent(model_path="models/cfr_agent.pt")
        results["cfr"] = diagnose_spy_accuracy(cfr, "cfr")
    except Exception as e:
        print(f"    Could not test cfr: {e}")

    print("\n  Summary:")
    for name, acc in results.items():
        marker = ""
        if acc is not None:
            if acc < 0.40:
                marker = " (very hard to predict)"
            elif acc < 0.50:
                marker = " (moderately hard)"
            else:
                marker = " (predictable)"
        print(f"    {name:20s}: {acc if acc is not None else 'N/A'}{marker}")

    return results


def diagnose_action_distributions(ih_agent, num_games=2000):
    """Measure raise / call / fold frequency per hand card (J, Q, K)
    for the info-hiding agent and compare to value_based.

    The key question: does J sometimes raise (bluff)?  Does K sometimes
    call or fold (slow-play / trap)?
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSIS 2: Action Distributions by Hand")
    print("=" * 60)

    def measure_action_dist(agent, label, num_games=2000):
        game = LeducGame()
        agent.set_train_mode(False)

        # counts[card][action] = count
        counts = {c: {a: 0 for a in ["FOLD", "CALL", "RAISE"]} for c in ["J", "Q", "K"]}
        totals = {c: 0 for c in ["J", "Q", "K"]}

        for _ in range(num_games):
            game.reset()
            while not game.is_finished:
                cp = game.current_player
                obs = game.get_observation(viewer_id=cp)
                action = agent.select_action(obs)

                card = game.player_hands[cp]
                if card in counts:
                    counts[card][action.name] += 1
                    totals[card] += 1

                game.step(action)

        # Convert to fractions
        dist = {}
        for card in ["J", "Q", "K"]:
            t = totals[card]
            if t > 0:
                dist[card] = {a: round(counts[card][a] / t, 4) for a in ["FOLD", "CALL", "RAISE"]}
            else:
                dist[card] = {a: 0.0 for a in ["FOLD", "CALL", "RAISE"]}

        print(f"\n  {label}:")
        for card in ["J", "Q", "K"]:
            f, c, r = dist[card]["FOLD"], dist[card]["CALL"], dist[card]["RAISE"]
            print(f"    {card}: FOLD={f:.3f}  CALL={c:.3f}  RAISE={r:.3f}  (n={totals[card]})")

        return dist

    # Info-hiding agent
    ih_dist = measure_action_dist(ih_agent, "InfoHidingAgent")

    # Value-based for comparison
    vb_dist = None
    try:
        vb = ValueBasedAgent(model_path="models/value_based_agent.pt")
        vb.set_train_mode(False)
        vb_dist = measure_action_dist(vb, "ValueBasedAgent")
    except Exception as e:
        print(f"    Could not load ValueBasedAgent: {e}")

    # Bluff / slow-play analysis
    print("\n  Deception analysis:")
    j_raise = ih_dist["J"]["RAISE"]
    k_call_fold = ih_dist["K"]["CALL"] + ih_dist["K"]["FOLD"]
    print(f"    J raise rate (bluff freq): {j_raise:.3f}")
    print(f"    K non-raise rate (slow-play/trap): {k_call_fold:.3f}")

    if vb_dist:
        vb_j_raise = vb_dist["J"]["RAISE"]
        vb_k_nonraise = vb_dist["K"]["CALL"] + vb_dist["K"]["FOLD"]
        print(f"    vs ValueBased J raise: {vb_j_raise:.3f}  (delta: {j_raise - vb_j_raise:+.3f})")
        print(f"    vs ValueBased K non-raise: {vb_k_nonraise:.3f}  (delta: {k_call_fold - vb_k_nonraise:+.3f})")

    # Information gap: difference in raise rate between K and J
    # Smaller gap = harder to read
    raise_gap = ih_dist["K"]["RAISE"] - ih_dist["J"]["RAISE"]
    print(f"\n    Raise-rate gap (K-raise - J-raise): {raise_gap:.3f}")
    print(f"    (Smaller = more deceptive. Zero means K and J raise equally.)")
    if vb_dist:
        vb_gap = vb_dist["K"]["RAISE"] - vb_dist["J"]["RAISE"]
        print(f"    ValueBased raise gap: {vb_gap:.3f}")

    results = {
        "info_hiding": ih_dist,
        "j_bluff_rate": j_raise,
        "k_slowplay_rate": k_call_fold,
        "raise_gap_K_minus_J": round(raise_gap, 4),
    }
    if vb_dist:
        results["value_based"] = vb_dist
        results["value_based_raise_gap"] = round(vb_gap, 4)

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ROUND 4 -- Direction 5: Information-Hiding Agent")
    print("=" * 60)

    all_results = {
        "agent": "info_hiding",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "spy_lr": SPY_LR,
            "value_coeff": VALUE_COEFF,
            "info_hiding_coeff": INFO_HIDING_COEFF,
            "eval_rounds": EVAL_ROUNDS,
            "sweep_episodes": SWEEP_EPISODES,
            "lambda_sweep": LAMBDA_SWEEP,
        },
    }

    # Phase 1: Training
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1: TRAINING (40K episodes)")
    print(f"{'#' * 60}")
    training_results = train_info_hiding_agent()
    all_results["training"] = training_results

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")
    agent = InfoHidingAgent(model_path=MODEL_PATH)
    agent.set_train_mode(False)
    eval_results, robustness = evaluate_against_opponents(agent)
    all_results["evaluation"] = eval_results
    all_results["robustness"] = robustness

    # Phase 3: Lambda sweep
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: LAMBDA SWEEP")
    print(f"{'#' * 60}")
    sweep_results = lambda_sweep()
    all_results["lambda_sweep"] = sweep_results

    # Phase 4: Diagnostics
    print(f"\n{'#' * 60}")
    print(f"  PHASE 4: DIAGNOSTICS")
    print(f"{'#' * 60}")

    # Reload the main trained agent for diagnostics
    agent = InfoHidingAgent(model_path=MODEL_PATH)
    agent.set_train_mode(False)

    diag1 = diagnose_predictability(agent)
    all_results["diagnostics"] = {"spy_accuracy_comparison": diag1}

    diag2 = diagnose_action_distributions(agent)
    all_results["diagnostics"]["action_distributions"] = diag2

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH) if os.path.dirname(RESULTS_PATH) else ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Summary ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Training time: {training_results['training_time_s']}s")
    print(f"  Final loss:    {training_results.get('final_loss', 'N/A')}")
    print(f"\n  Evaluation (avg chips/round):")
    for name, score in eval_results.items():
        print(f"    vs {name:20s}: {score:+.4f}")
    print(f"\n  Robustness: {robustness['robustness']:+.4f}")
    print(f"  Avg: {robustness['avg']:+.4f}, Worst: {robustness['worst_case']:+.4f}")

    print(f"\n  Spy accuracy comparison:")
    for name, acc in diag1.items():
        print(f"    {name:20s}: {acc if acc is not None else 'N/A'}")

    print(f"\n  Bluff / deception:")
    print(f"    J bluff rate:      {diag2['j_bluff_rate']:.3f}")
    print(f"    K slow-play rate:  {diag2['k_slowplay_rate']:.3f}")
    print(f"    Raise gap (K-J):   {diag2['raise_gap_K_minus_J']:.3f}")

    print(f"\n  Lambda sweep (best by heuristic score):")
    best_lam = max(sweep_results.items(), key=lambda x: x[1]["vs_heuristic"])
    print(f"    Best lambda: {best_lam[0]} -> vs_heuristic: {best_lam[1]['vs_heuristic']:+.4f}")

    return all_results


if __name__ == "__main__":
    main()
