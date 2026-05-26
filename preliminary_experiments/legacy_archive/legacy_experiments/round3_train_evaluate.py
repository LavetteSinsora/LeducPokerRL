"""
Round 3 Experiment: Train and evaluate 5 new agents with robustness metrics.

Agents:
  1. td_variant          — Systematic TD(0)/n-step/MC comparison (builds on value_based)
  2. pruned_history      — Pruned action history + stats (builds on adaptive_value)
  3. modulated_value     — Gated base + modulation architecture (builds on value_based)
  4. curriculum          — Block-scheduled population training (builds on adaptive_value)
  5. extended_adaptive   — Null hypothesis: same algorithm, 3x training (builds on adaptive_value)

Protocol:
  1. Train each new agent with agent-specific configs
  2. Evaluate all agents in round-robin (Round 1 + Round 2 + Round 3 + baselines)
  3. Compute robustness metrics: avg, worst-case, std, robustness score
  4. Print comprehensive results tables
  5. Save results to JSON

Changes from Round 2:
  - 1000 rounds per matchup (up from 500) for better statistical significance
  - Agent-specific training hyperparameters based on Round 2 diagnosis
  - Modulated value uses pretrained base network (frozen)
  - Curriculum trainer uses block scheduling with rehearsal
  - Extended adaptive gets 3x the standard training budget
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.registry import registry
from src.training.evaluation import evaluate_agents, quick_evaluate, compute_robustness_metrics


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

NEW_AGENTS = ["td_variant", "pruned_history", "modulated_value", "curriculum", "extended_adaptive"]

TRAINING_CONFIG = {
    # TD Variant: 20K episodes with n_steps=3 (best from Round 2 nstep diagnosis)
    # lr=5e-5 to compensate for n-step gradient magnitude (~1.87x larger than TD(0))
    "td_variant": {
        "episodes": 20000,
        "batch_size": 32,
        "lr": 5e-5,
        "trainer_kwargs": {"n_steps": 3},
    },
    # Pruned History: session-based, 667 sessions x 30 hands = ~20K hands
    # Same lr as adaptive_value baseline
    "pruned_history": {
        "episodes": 667,
        "batch_size": 32,
        "lr": 1e-4,
        "trainer_kwargs": {},
    },
    # Modulated Value: session-based, trains only mod+gate networks
    # Needs pretrained base from value_based_agent.pt
    "modulated_value": {
        "episodes": 667,
        "batch_size": 32,
        "lr": 1e-4,
        "trainer_kwargs": {},
    },
    # Curriculum: session-based with block scheduling
    # block_size=100 sessions per opponent, 20% rehearsal
    # 1000 sessions total (across 3 opponents)
    "curriculum": {
        "episodes": 1000,
        "batch_size": 32,
        "lr": 1e-4,
        "trainer_kwargs": {"block_size": 100, "rehearsal_ratio": 0.2},
    },
    # Extended Adaptive: 3x training budget = 2000 sessions x 30 hands = 60K hands
    "extended_adaptive": {
        "episodes": 2000,
        "batch_size": 32,
        "lr": 1e-4,
        "trainer_kwargs": {},
    },
}

# All agents to include in round-robin evaluation
ALL_AGENTS = [
    # Baselines
    "heuristic",
    # Round 0 (pre-existing)
    "value_based", "adaptive_value", "aux_value",
    # Round 1
    "actor_critic", "history_value", "decay_adaptive",
    # Round 2
    "nstep_value", "entropy_ac", "pop_adaptive", "adaptive_history", "target_value",
    # Round 3 (new)
    "td_variant", "pruned_history", "modulated_value", "curriculum", "extended_adaptive",
]

EVAL_ROUNDS = 1000  # Rounds per matchup for better statistical significance
RESULTS_PATH = "experiments/round3_results.json"


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_agent(agent_id: str, config: dict):
    """Train a single agent from scratch."""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {agent_id}")
    print(f"  Episodes: {config['episodes']}, Batch: {config['batch_size']}, LR: {config['lr']}")
    if config.get("trainer_kwargs"):
        print(f"  Extra kwargs: {config['trainer_kwargs']}")
    print(f"{'='*60}\n")

    model_path = f"models/{agent_id}_agent.pt"

    # Special handling for modulated_value: load pretrained base
    if agent_id == "modulated_value":
        base_path = "models/value_based_agent.pt"
        if os.path.exists(base_path):
            agent = registry.create(agent_id, base_model_path=base_path)
            print(f"  Loaded pretrained base from {base_path}")
        else:
            agent = registry.create(agent_id)
            print(f"  WARNING: No pretrained base found at {base_path}, using random init")
    else:
        agent = registry.create(agent_id)

    metadata = registry.get_metadata(agent_id)
    trainer_kwargs = config.get("trainer_kwargs", {})
    trainer = metadata.trainer_class(agent, learning_rate=config["lr"], **trainer_kwargs)

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append({"episode": data["episode"], "loss": data["loss"]})
            # Print progress every 100 updates
            if len(losses) % 100 == 0:
                print(f"    Update {len(losses)}: loss={data['loss']:.6f}")
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
    if losses:
        print(f"  Final loss: {losses[-1]['loss']:.6f}")
    if eval_scores:
        final_score = eval_scores[-1]["avg_chips"]
        print(f"  Final eval vs heuristic: {final_score:+.3f} chips/round")

    return {
        "agent_id": agent_id,
        "training_time_s": round(elapsed, 1),
        "num_updates": len(losses),
        "final_loss": losses[-1]["loss"] if losses else None,
        "final_eval_vs_heuristic": eval_scores[-1]["avg_chips"] if eval_scores else None,
        "eval_history": eval_scores,
        "config": {k: v for k, v in config.items() if k != "trainer_kwargs"},
        "trainer_kwargs": config.get("trainer_kwargs", {}),
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
    # Heuristic doesn't have set_train_mode
    if hasattr(agent, 'set_train_mode'):
        agent.set_train_mode(False)
    return agent


def round_robin_evaluation(agent_ids: list, num_rounds: int = 1000):
    """Evaluate all agent pairs and return a results matrix."""
    print(f"\n{'='*60}")
    print(f"  ROUND-ROBIN EVALUATION ({num_rounds} rounds per matchup)")
    print(f"  {len(agent_ids)} agents = {len(agent_ids) * (len(agent_ids)-1) // 2} matchups")
    print(f"{'='*60}\n")

    agents = {}
    for aid in agent_ids:
        try:
            agents[aid] = load_agent(aid)
        except Exception as e:
            print(f"  WARNING: Could not load {aid}: {e}")

    results = {}
    agent_list = list(agents.keys())

    # Initialize results dict for all agents
    for a in agent_list:
        results[a] = {}
        results[a][a] = 0.0

    total_matchups = len(agent_list) * (len(agent_list) - 1) // 2
    matchup_num = 0

    for i, a0_id in enumerate(agent_list):
        for j, a1_id in enumerate(agent_list):
            if j <= i:
                continue

            matchup_num += 1
            result = evaluate_agents(agents[a0_id], agents[a1_id], num_rounds=num_rounds)
            results[a0_id][a1_id] = round(result.agent_0_avg_chips, 4)
            results[a1_id][a0_id] = round(result.agent_1_avg_chips, 4)
            print(f"  [{matchup_num:3d}/{total_matchups}] {a0_id:20s} vs {a1_id:20s}: {result.agent_0_avg_chips:+.4f}")

    return results


def print_results_tables(results: dict, agent_ids: list):
    """Print formatted results matrix and robustness rankings."""
    available = [a for a in agent_ids if a in results]

    # ── Head-to-head matrix ──
    print(f"\n{'='*60}")
    print(f"  HEAD-TO-HEAD MATRIX (avg chips/round for row agent)")
    print(f"{'='*60}\n")

    header = f"{'Agent':>20s} |"
    for aid in available:
        short = aid[:10]
        header += f" {short:>10s}"
    header += " |    AVG"
    print(header)
    print("-" * len(header))

    for a0 in available:
        row = f"{a0:>20s} |"
        scores = []
        for a1 in available:
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

    # ── Robustness leaderboard ──
    print(f"\n{'='*60}")
    print(f"  ROBUSTNESS LEADERBOARD")
    print(f"{'='*60}\n")

    metrics = {}
    for agent_id in available:
        opponent_scores = {
            opp: results[agent_id].get(opp, 0.0)
            for opp in available if opp != agent_id
        }
        metrics[agent_id] = compute_robustness_metrics(opponent_scores)

    # Sort by robustness score (primary), then avg (secondary)
    sorted_agents = sorted(
        available,
        key=lambda a: (metrics[a]["robustness"], metrics[a]["avg"]),
        reverse=True,
    )

    header = f"{'Rank':>4s}  {'Agent':>20s}  {'Avg':>8s}  {'Worst':>8s}  {'Best':>8s}  {'Std':>8s}  {'Robustness':>10s}"
    print(header)
    print("-" * len(header))

    for rank, agent_id in enumerate(sorted_agents, 1):
        m = metrics[agent_id]
        # Mark Round 3 agents
        marker = " *" if agent_id in NEW_AGENTS else ""
        row = (f"{rank:>4d}  {agent_id:>20s}  {m['avg']:+8.4f}  "
               f"{m['worst_case']:+8.4f}  {m['best_case']:+8.4f}  "
               f"{m['std']:8.4f}  {m['robustness']:+10.4f}{marker}")
        print(row)

    print(f"\n  * = Round 3 agent")
    print()
    return metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ROUND 3 EXPERIMENT: Train & Evaluate 5 New Agents")
    print(f"  Training: agent-specific configs | Eval: {EVAL_ROUNDS} rounds/matchup")
    print("=" * 60)

    all_results = {
        "training": {},
        "evaluation": {},
        "robustness": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "eval_rounds": EVAL_ROUNDS,
            "training_configs": {k: {kk: vv for kk, vv in v.items()}
                                  for k, v in TRAINING_CONFIG.items()},
        },
    }

    # Phase 1: Train new agents
    print(f"\n{'#'*60}")
    print(f"  PHASE 1: TRAINING ({len(NEW_AGENTS)} agents)")
    print(f"{'#'*60}")

    for agent_id in NEW_AGENTS:
        config = TRAINING_CONFIG[agent_id]
        training_result = train_agent(agent_id, config)
        all_results["training"][agent_id] = training_result

    # Phase 2: Round-robin evaluation
    print(f"\n{'#'*60}")
    print(f"  PHASE 2: EVALUATION ({len(ALL_AGENTS)} agents, {EVAL_ROUNDS} rounds/matchup)")
    print(f"{'#'*60}")

    eval_results = round_robin_evaluation(ALL_AGENTS, num_rounds=EVAL_ROUNDS)
    all_results["evaluation"] = eval_results

    # Phase 3: Print results with robustness metrics
    robustness = print_results_tables(eval_results, ALL_AGENTS)
    all_results["robustness"] = robustness

    # Phase 4: Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return all_results


if __name__ == "__main__":
    main()
