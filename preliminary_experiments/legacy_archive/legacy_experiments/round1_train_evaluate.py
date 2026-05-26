"""
Round 1 Experiment: Train and evaluate 3 new incremental agents.

Agents:
  1. actor_critic    — REINFORCE + learned value baseline (builds on policy_gradient)
  2. history_value   — Value agent + scalable action history encoding (builds on value_based)
  3. decay_adaptive  — Adaptive agent with EMA opponent stats (builds on adaptive_value)

Protocol:
  1. Train each new agent for 3000 episodes (comparable to existing agents)
  2. Evaluate every new agent against every existing agent (round-robin)
  3. Print a comprehensive results table
  4. Save results to JSON for further analysis
"""

import json
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.registry import registry
from src.training.evaluation import evaluate_agents, quick_evaluate


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

NEW_AGENTS = ["actor_critic", "history_value", "decay_adaptive"]

TRAINING_CONFIG = {
    "actor_critic":   {"episodes": 3000, "batch_size": 32, "lr": 1e-3},
    "history_value":  {"episodes": 3000, "batch_size": 32, "lr": 1e-4},
    "decay_adaptive": {"episodes": 100,  "batch_size": 32, "lr": 1e-4},  # 100 sessions × 30 hands = 3000 hands
}

# All agents to include in round-robin evaluation
ALL_AGENTS = ["heuristic", "value_based", "adaptive_value", "aux_value",
              "actor_critic", "history_value", "decay_adaptive"]

EVAL_ROUNDS = 500  # Rounds per matchup for statistical significance
RESULTS_PATH = "experiments/round1_results.json"


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_agent(agent_id: str, config: dict):
    """Train a single agent from scratch."""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {agent_id}")
    print(f"  Episodes: {config['episodes']}, Batch: {config['batch_size']}, LR: {config['lr']}")
    print(f"{'='*60}\n")

    model_path = f"models/{agent_id}_agent.pt"

    agent = registry.create(agent_id)
    metadata = registry.get_metadata(agent_id)
    trainer = metadata.trainer_class(agent, learning_rate=config["lr"])

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append({"episode": data["episode"], "loss": data["loss"]})
        elif data["type"] == "evaluation":
            eval_scores.append({
                "episode": data["episode"],
                "avg_chips": data["avg_chips_per_round"]
            })

    start_time = time.time()
    trainer.train(
        num_episodes=config["episodes"],
        batch_size=config["batch_size"],
        save_path=model_path,
        callback=callback,
    )
    elapsed = time.time() - start_time

    print(f"\n  Training complete in {elapsed:.1f}s")
    if eval_scores:
        final_score = eval_scores[-1]["avg_chips"]
        print(f"  Final eval vs heuristic: {final_score:+.3f} chips/round")

    return {
        "agent_id": agent_id,
        "training_time_s": round(elapsed, 1),
        "final_loss": losses[-1]["loss"] if losses else None,
        "final_eval_vs_heuristic": eval_scores[-1]["avg_chips"] if eval_scores else None,
        "loss_history": losses,
        "eval_history": eval_scores,
    }


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def load_agent(agent_id: str):
    """Load a trained agent (or untrained for heuristic)."""
    agent = registry.create(agent_id)
    model_path = f"models/{agent_id}_agent.pt"
    if os.path.exists(model_path):
        agent.load_model(model_path)
    agent.set_train_mode(False)
    return agent


def round_robin_evaluation(agent_ids: list, num_rounds: int = 500):
    """Evaluate all agent pairs and return a results matrix."""
    print(f"\n{'='*60}")
    print(f"  ROUND-ROBIN EVALUATION ({num_rounds} rounds per matchup)")
    print(f"{'='*60}\n")

    agents = {}
    for aid in agent_ids:
        try:
            agents[aid] = load_agent(aid)
        except Exception as e:
            print(f"  WARNING: Could not load {aid}: {e}")

    results = {}
    n = len(agents)
    agent_list = list(agents.keys())

    for i, a0_id in enumerate(agent_list):
        results[a0_id] = {}
        for j, a1_id in enumerate(agent_list):
            if a0_id == a1_id:
                results[a0_id][a1_id] = 0.0
                continue

            # Only compute upper triangle, mirror for lower
            if j > i:
                result = evaluate_agents(agents[a0_id], agents[a1_id], num_rounds=num_rounds)
                results[a0_id][a1_id] = round(result.agent_0_avg_chips, 4)
                # Results are zero-sum
                if a1_id not in results:
                    results[a1_id] = {}
                results[a1_id][a0_id] = round(result.agent_1_avg_chips, 4)
                print(f"  {a0_id:20s} vs {a1_id:20s}: {result.agent_0_avg_chips:+.4f}")

    return results


def print_results_table(results: dict, agent_ids: list):
    """Print a formatted results matrix."""
    print(f"\n{'='*60}")
    print(f"  RESULTS MATRIX (avg chips/round for row agent)")
    print(f"{'='*60}\n")

    # Header
    header = f"{'Agent':>20s} |"
    for aid in agent_ids:
        short = aid[:10]
        header += f" {short:>10s}"
    header += " |    AVG"
    print(header)
    print("-" * len(header))

    # Rows
    for a0 in agent_ids:
        row = f"{a0:>20s} |"
        scores = []
        for a1 in agent_ids:
            if a0 == a1:
                row += f" {'---':>10s}"
            else:
                val = results.get(a0, {}).get(a1, 0.0)
                row += f" {val:+10.4f}"
                scores.append(val)
        avg = sum(scores) / len(scores) if scores else 0
        row += f" | {avg:+.4f}"
        print(row)

    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ROUND 1 EXPERIMENT: Train & Evaluate New Agents")
    print("=" * 60)

    all_results = {
        "training": {},
        "evaluation": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Phase 1: Train new agents
    for agent_id in NEW_AGENTS:
        config = TRAINING_CONFIG[agent_id]
        training_result = train_agent(agent_id, config)
        all_results["training"][agent_id] = training_result

    # Phase 2: Round-robin evaluation
    eval_results = round_robin_evaluation(ALL_AGENTS, num_rounds=EVAL_ROUNDS)
    all_results["evaluation"] = eval_results

    # Phase 3: Print results
    print_results_table(eval_results, ALL_AGENTS)

    # Phase 4: Save results
    # Strip non-serializable data from training results
    serializable = {
        "timestamp": all_results["timestamp"],
        "training": {
            k: {kk: vv for kk, vv in v.items() if kk != "loss_history"}
            for k, v in all_results["training"].items()
        },
        "evaluation": all_results["evaluation"],
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return all_results


if __name__ == "__main__":
    main()
