"""
Shared comparison protocol for paper experiments.

Goals:
  - Evaluate all agents under consistent opponent pools and metrics.
  - Cover both seats when requested.
  - Reset opponent-stat trackers on the same schedule used in training.
  - Select checkpoints by pool-level metrics rather than a single matchup.
"""

from __future__ import annotations

import os
from typing import Callable

from agents.cfr.agent import CFRAgent
from agents.evaluation import compute_robustness_metrics, evaluate_agents
from agents.heuristic.agent import HeuristicAgent

from agents.rule_based.loose_aggressive import LooseAggressiveAgent
from agents.rule_based.loose_passive import LoosePassiveAgent
from agents.rule_based.maniac import ManiacAgent
from agents.rule_based.random_agent import RandomAgent
from agents.rule_based.tight_aggressive import TightAggressiveAgent
from agents.rule_based.tight_passive import TightPassiveAgent
from paper.evaluation.shared.stats_tracker import OpponentStatsTracker


STANDARD_OPPONENT_KEYS = [
    "heuristic",
    "cfr",
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
]

CHECKPOINT_METRICS = {"heuristic", "avg", "worst_case", "robustness"}


def build_standard_opponents(root: str) -> dict:
    cfr_path = os.path.join(root, "agents", "cfr", "checkpoint.pt")
    opponents = {
        "heuristic": HeuristicAgent(),
        "cfr": CFRAgent(model_path=cfr_path),
        "tight_passive": TightPassiveAgent(),
        "tight_aggressive": TightAggressiveAgent(),
        "loose_passive": LoosePassiveAgent(),
        "loose_aggressive": LooseAggressiveAgent(),
        "maniac": ManiacAgent(),
        "random": RandomAgent(),
    }
    for opponent in opponents.values():
        opponent.set_train_mode(False)
    return opponents


def compute_pool_summary(scores: dict) -> dict:
    summary = compute_robustness_metrics(scores)
    summary["metric_values"] = {
        "heuristic": scores.get("heuristic", float("-inf")),
        "avg": summary["avg"],
        "worst_case": summary["worst_case"],
        "robustness": summary["robustness"],
    }
    return summary


def checkpoint_metric_value(scores: dict, checkpoint_metric: str) -> float:
    if checkpoint_metric not in CHECKPOINT_METRICS:
        raise ValueError(f"Unsupported checkpoint metric: {checkpoint_metric}")
    summary = compute_pool_summary(scores)
    return summary["metric_values"][checkpoint_metric]


def format_pool_summary(summary: dict) -> str:
    return (
        f"avg:{summary['avg']:+.3f}  "
        f"worst:{summary['worst_case']:+.3f}  "
        f"rob:{summary['robustness']:+.3f}"
    )


def evaluate_standard_pool(
    agent,
    opponents: dict,
    num_rounds: int,
    opponent_keys: list[str] | None = None,
) -> dict:
    keys = opponent_keys or STANDARD_OPPONENT_KEYS
    scores = {}
    details = {}
    agent.set_train_mode(False)
    for key in keys:
        result = evaluate_agents(agent, opponents[key], num_rounds=num_rounds)
        avg = round(result.agent_0_avg_chips, 4)
        scores[key] = avg
        details[key] = {
            "avg_chips_per_round": avg,
            "total_chips": round(result.agent_0_total_chips, 2),
            "rounds": result.num_rounds,
        }
    return {"scores": scores, "details": details, "summary": compute_pool_summary(scores)}


def evaluate_stat_aware_matchup(
    agent,
    opponent,
    play_hand_fn: Callable,
    pool_means: dict,
    num_rounds: int,
    session_length: int,
    prior_strength: float,
    alternate_positions: bool = True,
) -> dict:
    if alternate_positions:
        seat_specs = [
            (0, (num_rounds + 1) // 2),
            (1, num_rounds // 2),
        ]
    else:
        seat_specs = [(0, num_rounds)]

    total = 0.0
    total_rounds = 0
    seat_details = {}
    for learner_id, rounds in seat_specs:
        tracker = OpponentStatsTracker(pool_means, prior_strength, session_length)
        subtotal = 0.0
        hands_in_session = 0
        for _ in range(rounds):
            if hands_in_session >= session_length:
                tracker.reset()
                hands_in_session = 0
            _, reward = play_hand_fn(agent, opponent, tracker, learner_id=learner_id)
            subtotal += reward
            total += reward
            total_rounds += 1
            hands_in_session += 1
        if rounds > 0:
            seat_details[f"learner_id_{learner_id}"] = {
                "avg_chips_per_round": round(subtotal / rounds, 4),
                "total_chips": round(subtotal, 2),
                "rounds": rounds,
            }

    avg = total / total_rounds if total_rounds else 0.0
    return {
        "avg_chips_per_round": round(avg, 4),
        "total_chips": round(total, 2),
        "rounds": total_rounds,
        "seat_details": seat_details,
    }


def evaluate_stat_aware_pool(
    agent,
    opponents: dict,
    play_hand_fn: Callable,
    pool_means: dict,
    num_rounds: int,
    session_length: int,
    prior_strength: float,
    opponent_keys: list[str] | None = None,
    alternate_positions: bool = True,
) -> dict:
    keys = opponent_keys or STANDARD_OPPONENT_KEYS
    scores = {}
    details = {}
    agent.set_train_mode(False)
    for key in keys:
        matchup = evaluate_stat_aware_matchup(
            agent=agent,
            opponent=opponents[key],
            play_hand_fn=play_hand_fn,
            pool_means=pool_means,
            num_rounds=num_rounds,
            session_length=session_length,
            prior_strength=prior_strength,
            alternate_positions=alternate_positions,
        )
        scores[key] = matchup["avg_chips_per_round"]
        details[key] = matchup
    return {"scores": scores, "details": details, "summary": compute_pool_summary(scores)}


def seed_best_metric(eval_history: list, checkpoint_metric: str) -> float:
    if not eval_history:
        return float("-inf")
    best = float("-inf")
    for record in eval_history:
        scores = {
            k: v for k, v in record.items()
            if k != "episode" and not k.startswith("_")
        }
        if not scores:
            continue
        best = max(best, checkpoint_metric_value(scores, checkpoint_metric))
    return best


def attach_summary(entry: dict) -> dict:
    scores = {
        k: v for k, v in entry.items()
        if k != "episode" and not k.startswith("_")
    }
    if not scores:
        return entry
    summary = compute_pool_summary(scores)
    return {**entry, "_summary": summary}
