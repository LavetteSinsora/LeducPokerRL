"""
Round 4 — Direction 3: Distributional Value Agent

This experiment trains a risk-sensitive distributional value agent that learns
the full distribution of returns (via quantile regression) and makes decisions
by optimising:

    score(a) = E[V(post(s,a))] - beta * Std[V(post(s,a))]

Key insight: the project's evaluation metric is robustness = avg - 1.5*std.
All other agents optimise expected value and hope variance is low. This agent
directly incorporates variance into its action selection, aligning its objective
with the metric we actually measure.

Protocol:
  1. Train for 30K episodes (self-play, quantile Huber loss)
  2. Evaluate against 6 opponents (500 rounds each, both positions)
  3. Sweep risk_beta in [0, 0.5, 1.0, 1.5, 2.0] to find optimal sensitivity
  4. Diagnose learned distributions for key states
  5. Save results to experiments/round4_distributional_results.json
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.agents.distributional_value import DistributionalValueAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.modulated_value import ModulatedValueAgent
from src.agents.entropy_ac import EntropyACAgent
from src.agents.cfr_agent import CFRAgent
from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.training.distributional_trainer import DistributionalTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TRAINING_EPISODES = 30000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MODEL_PATH = "models/distributional_agent.pt"
RESULTS_PATH = "experiments/round4_distributional_results.json"
EVAL_ROUNDS = 500
BETA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

OPPONENTS = {
    "heuristic": lambda: HeuristicAgent(),
    "value_based": lambda: _load_agent(ValueBasedAgent, "models/value_based_agent.pt"),
    "adaptive_value": lambda: _load_agent(AdaptiveValueAgent, "models/adaptive_value_agent.pt"),
    "modulated_value": lambda: _load_agent(ModulatedValueAgent, "models/modulated_value_agent.pt"),
    "entropy_ac": lambda: _load_agent(EntropyACAgent, "models/entropy_ac_agent.pt"),
    "cfr": lambda: _load_agent(CFRAgent, "models/cfr_agent.pt"),
}


def _load_agent(cls, path):
    """Load a trained agent from a model file."""
    if os.path.exists(path):
        agent = cls(model_path=path)
    else:
        print(f"  WARNING: {path} not found, using untrained agent")
        agent = cls()
    agent.set_train_mode(False)
    return agent


# ──────────────────────────────────────────────
# Phase 1: Training
# ──────────────────────────────────────────────

def train_distributional_agent():
    """Train the distributional value agent from scratch."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1: TRAINING DISTRIBUTIONAL VALUE AGENT")
    print(f"  Episodes: {TRAINING_EPISODES}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"  Risk beta (training): 0 (risk-neutral exploration, mean-only)")
    print(f"{'='*60}\n")

    # Train with risk_beta=0; select_action uses mean-only during training.
    # Risk sensitivity is only applied at evaluation time.
    agent = DistributionalValueAgent(risk_beta=0.0, temperature=1.0)
    trainer = DistributionalTrainer(agent, learning_rate=LEARNING_RATE)

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
        num_episodes=TRAINING_EPISODES,
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

    return {
        "training_time_s": round(elapsed, 1),
        "num_updates": len(losses),
        "final_loss": losses[-1]["loss"] if losses else None,
        "final_eval_vs_heuristic": eval_scores[-1]["avg_chips"] if eval_scores else None,
        "eval_history": eval_scores,
        "loss_samples": losses[::50] if losses else [],  # every 50th for brevity
    }


# ──────────────────────────────────────────────
# Phase 2: Evaluation against opponents
# ──────────────────────────────────────────────

def evaluate_against_opponents():
    """Evaluate the trained distributional agent against all opponents."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2: EVALUATION ({EVAL_ROUNDS} rounds per opponent)")
    print(f"{'='*60}\n")

    agent = DistributionalValueAgent(model_path=MODEL_PATH, risk_beta=0.5)
    agent.set_train_mode(False)

    scores = {}
    per_round_data = {}

    for opp_name, opp_factory in OPPONENTS.items():
        try:
            opponent = opp_factory()
            result = evaluate_agents(agent, opponent, num_rounds=EVAL_ROUNDS)
            avg = result.agent_0_avg_chips
            scores[opp_name] = round(avg, 4)

            # Compute per-matchup std
            rewards = [r[0] for r in result.round_results]
            matchup_std = (sum((r - avg)**2 for r in rewards) / len(rewards)) ** 0.5
            per_round_data[opp_name] = {
                "avg": round(avg, 4),
                "std": round(matchup_std, 4),
                "total_rounds": result.num_rounds,
            }
            print(f"  vs {opp_name:20s}: {avg:+.4f} chips/round (std={matchup_std:.4f})")
        except Exception as e:
            print(f"  vs {opp_name:20s}: ERROR - {e}")
            scores[opp_name] = 0.0
            per_round_data[opp_name] = {"avg": 0.0, "std": 0.0, "error": str(e)}

    robustness = compute_robustness_metrics(scores)
    print(f"\n  Robustness metrics:")
    print(f"    Avg:        {robustness['avg']:+.4f}")
    print(f"    Worst-case: {robustness['worst_case']:+.4f}")
    print(f"    Std:        {robustness['std']:.4f}")
    print(f"    Robustness: {robustness['robustness']:+.4f}")

    return {
        "scores": scores,
        "per_matchup": per_round_data,
        "robustness": robustness,
    }


# ──────────────────────────────────────────────
# Phase 3: Risk beta sweep
# ──────────────────────────────────────────────

def sweep_risk_beta():
    """Evaluate the same trained model with different risk_beta values."""
    print(f"\n{'='*60}")
    print(f"  PHASE 3: RISK BETA SWEEP")
    print(f"  Betas: {BETA_SWEEP}")
    print(f"{'='*60}\n")

    sweep_results = {}

    for beta in BETA_SWEEP:
        print(f"\n  --- beta = {beta} ---")
        agent = DistributionalValueAgent(model_path=MODEL_PATH, risk_beta=beta)
        agent.set_train_mode(False)

        scores = {}
        for opp_name, opp_factory in OPPONENTS.items():
            try:
                opponent = opp_factory()
                result = evaluate_agents(agent, opponent, num_rounds=EVAL_ROUNDS)
                avg = result.agent_0_avg_chips
                scores[opp_name] = round(avg, 4)
                print(f"    vs {opp_name:20s}: {avg:+.4f}")
            except Exception as e:
                print(f"    vs {opp_name:20s}: ERROR - {e}")
                scores[opp_name] = 0.0

        metrics = compute_robustness_metrics(scores)
        sweep_results[str(beta)] = {
            "scores": scores,
            "metrics": metrics,
        }
        print(f"    => avg={metrics['avg']:+.4f}, std={metrics['std']:.4f}, "
              f"robustness={metrics['robustness']:+.4f}")

    # Find optimal beta
    best_beta = max(sweep_results.keys(),
                    key=lambda b: sweep_results[b]["metrics"]["robustness"])
    print(f"\n  Optimal beta: {best_beta} "
          f"(robustness={sweep_results[best_beta]['metrics']['robustness']:+.4f})")

    return {
        "sweep": sweep_results,
        "optimal_beta": float(best_beta),
    }


# ──────────────────────────────────────────────
# Phase 4: Distribution diagnosis
# ──────────────────────────────────────────────

def diagnose_distributions():
    """Analyze learned value distributions for key game states."""
    print(f"\n{'='*60}")
    print(f"  PHASE 4: DISTRIBUTION DIAGNOSIS")
    print(f"{'='*60}\n")

    agent = DistributionalValueAgent(model_path=MODEL_PATH, risk_beta=0.5)
    agent.set_train_mode(False)

    diagnosis = {}

    # Key states to examine
    test_states = [
        # Preflop states (no board card)
        {"name": "J_preflop", "hand": "J", "board": None, "pot": [1, 1],
         "round": 0, "raises": 0},
        {"name": "Q_preflop", "hand": "Q", "board": None, "pot": [1, 1],
         "round": 0, "raises": 0},
        {"name": "K_preflop", "hand": "K", "board": None, "pot": [1, 1],
         "round": 0, "raises": 0},
        # Flop states: pair vs no-pair
        {"name": "K_pair_K", "hand": "K", "board": "K", "pot": [3, 3],
         "round": 1, "raises": 0},
        {"name": "K_nopair_J", "hand": "K", "board": "J", "pot": [3, 3],
         "round": 1, "raises": 0},
        {"name": "J_pair_J", "hand": "J", "board": "J", "pot": [3, 3],
         "round": 1, "raises": 0},
        {"name": "J_nopair_K", "hand": "J", "board": "K", "pot": [3, 3],
         "round": 1, "raises": 0},
        # High-pot state (after raises)
        {"name": "K_preflop_raised", "hand": "K", "board": None, "pot": [3, 3],
         "round": 0, "raises": 1},
        {"name": "J_preflop_raised", "hand": "J", "board": None, "pot": [3, 3],
         "round": 0, "raises": 1},
    ]

    for state in test_states:
        obs = Observation(
            player_hand=state["hand"],
            board=state["board"],
            pot=state["pot"],
            current_player=0,
            current_round=state["round"],
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE] if state["raises"] < 2 else [Action.FOLD, Action.CALL],
            is_finished=False,
            raises_this_round=state["raises"],
        )

        encoded = agent.encode_observation(obs, viewer_id=0)
        with torch.no_grad():
            mean_t, var_t = agent.model(encoded)
            mean = mean_t.item()
            std_val = var_t.item() ** 0.5

        # Also get per-action evaluations
        evals = agent.get_action_evaluations(obs)
        action_info = {}
        for e in evals:
            action_info[e["action"].name] = {
                "risk_score": round(e["value"], 4),
                "mean": round(e["mean"], 4),
                "std": round(e["std"], 4),
            }

        diagnosis[state["name"]] = {
            "state_mean": round(mean, 4),
            "state_std": round(std_val, 4),
            "actions": action_info,
            "selected_action": max(evals, key=lambda x: x["value"])["action"].name,
        }

        print(f"  {state['name']:20s}: mean={mean:+.3f}, std={std_val:.3f}, "
              f"selected={diagnosis[state['name']]['selected_action']}")
        for aname, ainfo in action_info.items():
            print(f"    {aname:6s}: risk_score={ainfo['risk_score']:+.3f}, "
                  f"mean={ainfo['mean']:+.3f}, std={ainfo['std']:.3f}")

    # Compare pair vs non-pair variance
    print(f"\n  PAIR vs NON-PAIR VARIANCE ANALYSIS:")
    pair_states = ["K_pair_K", "J_pair_J"]
    nopair_states = ["K_nopair_J", "J_nopair_K"]

    pair_stds = [diagnosis[s]["state_std"] for s in pair_states if s in diagnosis]
    nopair_stds = [diagnosis[s]["state_std"] for s in nopair_states if s in diagnosis]

    if pair_stds and nopair_stds:
        avg_pair_std = sum(pair_stds) / len(pair_stds)
        avg_nopair_std = sum(nopair_stds) / len(nopair_stds)
        print(f"    Avg pair std:    {avg_pair_std:.4f}")
        print(f"    Avg no-pair std: {avg_nopair_std:.4f}")
        print(f"    Ratio (no-pair/pair): {avg_nopair_std / avg_pair_std:.2f}x" if avg_pair_std > 0 else "")
        diagnosis["_pair_vs_nopair"] = {
            "avg_pair_std": round(avg_pair_std, 4),
            "avg_nopair_std": round(avg_nopair_std, 4),
            "ratio": round(avg_nopair_std / avg_pair_std, 4) if avg_pair_std > 0 else None,
        }

    # Compare preflop variance by hand strength
    print(f"\n  PREFLOP VARIANCE BY HAND STRENGTH:")
    for h in ["J_preflop", "Q_preflop", "K_preflop"]:
        if h in diagnosis:
            d = diagnosis[h]
            print(f"    {h:15s}: mean={d['state_mean']:+.3f}, std={d['state_std']:.3f}")

    return diagnosis


# ──────────────────────────────────────────────
# Phase 5: Comparison with scalar value agent
# ──────────────────────────────────────────────

def compare_with_scalar_agent():
    """Compare distributional agent decisions with scalar value agent."""
    print(f"\n{'='*60}")
    print(f"  PHASE 5: DISTRIBUTIONAL vs SCALAR AGENT COMPARISON")
    print(f"{'='*60}\n")

    dist_agent = DistributionalValueAgent(model_path=MODEL_PATH, risk_beta=0.5)
    dist_agent.set_train_mode(False)

    scalar_path = "models/value_based_agent.pt"
    if os.path.exists(scalar_path):
        scalar_agent = ValueBasedAgent(model_path=scalar_path)
    else:
        print("  Scalar agent model not found, skipping comparison")
        return {}

    scalar_agent.set_train_mode(False)

    # Play 200 games and compare decisions
    game = LeducGame()
    agreements = 0
    disagreements = 0
    disagreement_examples = []

    for _ in range(200):
        game.reset()
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            dist_action = dist_agent.select_action(obs)
            scalar_action = scalar_agent.select_action(obs)

            if dist_action == scalar_action:
                agreements += 1
            else:
                disagreements += 1
                if len(disagreement_examples) < 10:
                    dist_evals = dist_agent.get_action_evaluations(obs)
                    scalar_evals = scalar_agent.get_action_evaluations(obs)
                    disagreement_examples.append({
                        "hand": obs.player_hand,
                        "board": obs.board,
                        "pot": obs.pot,
                        "round": obs.current_round,
                        "raises": obs.raises_this_round,
                        "dist_action": dist_action.name,
                        "scalar_action": scalar_action.name,
                        "dist_evals": [{
                            "action": e["action"].name,
                            "risk_score": round(e["value"], 4),
                            "mean": round(e["mean"], 4),
                            "std": round(e["std"], 4),
                        } for e in dist_evals],
                        "scalar_evals": [{
                            "action": e["action"].name,
                            "value": round(e["value"], 4),
                        } for e in scalar_evals],
                    })

            game.step(dist_action)  # Use distributional agent's choice to advance

    total = agreements + disagreements
    agreement_rate = agreements / total if total > 0 else 0
    print(f"  Agreement rate: {agreements}/{total} = {agreement_rate:.1%}")
    print(f"  Disagreements: {disagreements}")

    if disagreement_examples:
        print(f"\n  Sample disagreements (up to 10):")
        for i, ex in enumerate(disagreement_examples[:5]):
            print(f"    {i+1}. hand={ex['hand']}, board={ex['board']}, "
                  f"pot={ex['pot']}, round={ex['round']}")
            print(f"       dist={ex['dist_action']}, scalar={ex['scalar_action']}")
            for de in ex["dist_evals"]:
                print(f"         dist  {de['action']:6s}: risk={de['risk_score']:+.3f} "
                      f"(mean={de['mean']:+.3f}, std={de['std']:.3f})")
            for se in ex["scalar_evals"]:
                print(f"         scalar {se['action']:6s}: value={se['value']:+.3f}")

    return {
        "agreements": agreements,
        "disagreements": disagreements,
        "total_decisions": total,
        "agreement_rate": round(agreement_rate, 4),
        "disagreement_examples": disagreement_examples[:10],
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ROUND 4 — DIRECTION 3: DISTRIBUTIONAL VALUE AGENT")
    print("=" * 60)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "training_episodes": TRAINING_EPISODES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "eval_rounds": EVAL_ROUNDS,
            "beta_sweep": BETA_SWEEP,
            "n_quantiles": 11,
        },
    }

    # Phase 1: Train
    all_results["training"] = train_distributional_agent()

    # Phase 2: Evaluate
    all_results["evaluation"] = evaluate_against_opponents()

    # Phase 3: Beta sweep
    all_results["beta_sweep"] = sweep_risk_beta()

    # Phase 4: Distribution diagnosis
    all_results["diagnosis"] = diagnose_distributions()

    # Phase 5: Comparison with scalar agent
    all_results["scalar_comparison"] = compare_with_scalar_agent()

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    eval_data = all_results.get("evaluation", {})
    rob = eval_data.get("robustness", {})
    print(f"  Training time:  {all_results['training']['training_time_s']:.0f}s")
    print(f"  Avg chips:      {rob.get('avg', 0):+.4f}")
    print(f"  Robustness:     {rob.get('robustness', 0):+.4f}")
    sweep = all_results.get("beta_sweep", {})
    print(f"  Optimal beta:   {sweep.get('optimal_beta', 'N/A')}")

    return all_results


if __name__ == "__main__":
    main()
