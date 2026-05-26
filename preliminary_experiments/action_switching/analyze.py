"""
Oracle action-switching analysis for opponent-aware decision making.

The "oracle" here is one-step optimality under the existing ValueBasedAgent
continuation policy: for each legal action at the injected state, we force that
action once and then let the value agent play the rest of the hand. This aligns
with the decision problem solved by the current one-step lookahead architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from agents.value_based.agent import ValueBasedAgent
from engine.leduc_game import LeducGame
from preliminary_experiments.ev_variation_extras.code.collect import enumerate_states
from preliminary_experiments.ev_variation_extras.code.sim_engine import FixedStateSimulator
from preliminary_experiments.opp_stats_input_aug.agent import StatAugValueAgent
from agents.rule_based import ALL_AGENTS


ARCHETYPES = [
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
]
ACTION_ORDER = ["FOLD", "CALL", "RAISE"]
DEFAULT_BASELINE_CKPT = os.path.join(ROOT, "preliminary_experiments", "baseline_value_v1", "outputs", "checkpoint_best.pt")
DEFAULT_AUG_CKPT = os.path.join(
    ROOT, "preliminary_experiments", "opp_stats_input_aug", "outputs", "pool_random", "checkpoint_best.pt"
)
PROTO_PATH = os.path.join(ROOT, "paper", "evaluation", "shared", "data", "opponent_prototype_stats.json")
OUT_DIR = os.path.join(HERE, "outputs")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _inject_state(record):
    sim = FixedStateSimulator(
        hand0=record["hand0"],
        hand1=record["hand1"],
        pot=record["pot"],
        current_player=record["current_player"],
        rnd=record["round"],
        raises=record["raises"],
        board=record["board"],
    )
    game = LeducGame()
    sim._inject(game)
    return sim, game


def _oracle_action_values(record, value_agent, opponent, n_rollouts):
    sim = FixedStateSimulator(
        hand0=record["hand0"],
        hand1=record["hand1"],
        pot=record["pot"],
        current_player=record["current_player"],
        rnd=record["round"],
        raises=record["raises"],
        board=record["board"],
    )

    game = LeducGame()
    sim._inject(game)
    value_player = record["current_player"]
    obs = game.get_observation(viewer_id=value_player)
    legal_actions = list(obs.legal_actions)

    if record["round"] == 0:
        board_dist = sim._board_distribution()
    else:
        board_dist = {record["board"]: 1.0}

    action_values = {}
    for action in legal_actions:
        rewards = []
        for board_card, prob in board_dist.items():
            n_sub = max(1, round(n_rollouts * prob))
            for _ in range(n_sub):
                rollout_game = LeducGame()
                sim._inject(rollout_game, board_for_deck=board_card if record["round"] == 0 else None)
                rollout_game.step(action)
                while not rollout_game.is_finished:
                    acting = rollout_game.current_player
                    rollout_obs = rollout_game.get_observation(viewer_id=acting)
                    if acting == value_player:
                        next_action = value_agent.select_action(rollout_obs)
                    else:
                        next_action = opponent.select_action(rollout_obs)
                    rollout_game.step(next_action)
                rewards.append(float(rollout_game.get_reward()[value_player]))
        arr = np.array(rewards, dtype=np.float64)
        action_values[action.name] = {
            "ev": round(float(arr.mean()), 5),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 5),
            "n": int(len(arr)),
        }
    return action_values


def _best_action_from_values(action_values):
    best_name = None
    best_value = None
    for action_name in ACTION_ORDER:
        if action_name not in action_values:
            continue
        value = action_values[action_name]["ev"]
        if best_value is None or value > best_value:
            best_name = action_name
            best_value = value
    return best_name


def _baseline_best_action(agent, record):
    _, game = _inject_state(record)
    obs = game.get_observation(viewer_id=record["current_player"])
    evaluations = agent.get_action_evaluations(obs)
    best = max(evaluations, key=lambda item: item["value"])
    return best["action"].name


def _augmented_best_action(agent, record, opp_stats):
    _, game = _inject_state(record)
    obs = game.get_observation(viewer_id=record["current_player"])
    evaluations = agent.get_action_evaluations(obs, np.array(opp_stats, dtype=np.float32))
    best = max(evaluations, key=lambda item: item["value"])
    return best["action"].name


def _summarize_state_records(records):
    total = len(records)
    switch_states = [record for record in records if record["oracle_switches"]]
    baseline_matches = sum(record["baseline_match_count"] for record in records)
    aug_matches = sum(record["aug_match_count"] for record in records)
    total_labels = total * len(ARCHETYPES)
    switch_labels = len(switch_states) * len(ARCHETYPES)

    if switch_labels:
        baseline_switch_matches = sum(record["baseline_match_count"] for record in switch_states)
        aug_switch_matches = sum(record["aug_match_count"] for record in switch_states)
    else:
        baseline_switch_matches = 0
        aug_switch_matches = 0

    round_breakdown = {}
    for rnd in [0, 1]:
        subset = [record for record in records if record["round"] == rnd]
        if not subset:
            continue
        round_breakdown[str(rnd)] = {
            "n_states": len(subset),
            "oracle_switch_rate": round(sum(record["oracle_switches"] for record in subset) / len(subset), 4),
            "baseline_accuracy": round(
                sum(record["baseline_match_count"] for record in subset) / (len(subset) * len(ARCHETYPES)), 4
            ),
            "aug_accuracy": round(
                sum(record["aug_match_count"] for record in subset) / (len(subset) * len(ARCHETYPES)), 4
            ),
        }

    return {
        "n_states": total,
        "n_archetypes": len(ARCHETYPES),
        "oracle_switch_states": len(switch_states),
        "oracle_switch_rate": round(len(switch_states) / total, 4) if total else 0.0,
        "baseline_accuracy": round(baseline_matches / total_labels, 4) if total_labels else 0.0,
        "augmented_accuracy": round(aug_matches / total_labels, 4) if total_labels else 0.0,
        "baseline_accuracy_on_switch_states": round(baseline_switch_matches / switch_labels, 4) if switch_labels else None,
        "augmented_accuracy_on_switch_states": round(aug_switch_matches / switch_labels, 4) if switch_labels else None,
        "round_breakdown": round_breakdown,
    }


def run_analysis(rollouts, limit_states, baseline_ckpt, aug_ckpt):
    os.makedirs(OUT_DIR, exist_ok=True)

    states = enumerate_states()
    if limit_states is not None:
        states = states[:limit_states]

    opponents = {name: ALL_AGENTS[name]() for name in ARCHETYPES}
    for opponent in opponents.values():
        opponent.set_train_mode(False)

    baseline = ValueBasedAgent(model_path=baseline_ckpt)
    baseline.set_train_mode(False)
    aug_agent = StatAugValueAgent(model_path=aug_ckpt)
    aug_agent.set_train_mode(False)

    proto_raw = _load_json(PROTO_PATH)
    proto_stats = {name: proto_raw[name] for name in ARCHETYPES}

    state_records = []
    t0 = time.time()
    for idx, state in enumerate(states, start=1):
        oracle_best = {}
        oracle_values = {}
        baseline_best = _baseline_best_action(baseline, state)
        aug_best = {}

        for name in ARCHETYPES:
            values = _oracle_action_values(state, baseline, opponents[name], rollouts)
            oracle_values[name] = values
            oracle_best[name] = _best_action_from_values(values)
            aug_best[name] = _augmented_best_action(aug_agent, state, proto_stats[name])

        unique_oracle = sorted(set(oracle_best.values()))
        record = {
            "state_id": state["state_id"],
            "round": state["round"],
            "hand0": state["hand0"],
            "hand1": state["hand1"],
            "board": state["board"],
            "pot": state["pot"],
            "current_player": state["current_player"],
            "raises": state["raises"],
            "oracle_action_values": oracle_values,
            "oracle_best_actions": oracle_best,
            "oracle_switches": len(unique_oracle) > 1,
            "oracle_unique_actions": unique_oracle,
            "baseline_best_action": baseline_best,
            "baseline_match_count": sum(baseline_best == oracle_best[name] for name in ARCHETYPES),
            "augmented_best_actions": aug_best,
            "aug_match_count": sum(aug_best[name] == oracle_best[name] for name in ARCHETYPES),
        }
        state_records.append(record)

        if idx % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {idx}/{len(states)} states processed  ({elapsed:.0f}s elapsed)")

    summary = _summarize_state_records(state_records)
    top_switch = sorted(
        [record for record in state_records if record["oracle_switches"]],
        key=lambda record: max(
            vals["ev"] for per_opp in record["oracle_action_values"].values() for vals in per_opp.values()
        ) - min(vals["ev"] for per_opp in record["oracle_action_values"].values() for vals in per_opp.values()),
        reverse=True,
    )[:10]
    summary["top_switch_examples"] = [
        {
            "state_id": record["state_id"],
            "oracle_best_actions": record["oracle_best_actions"],
            "baseline_best_action": record["baseline_best_action"],
            "augmented_best_actions": record["augmented_best_actions"],
        }
        for record in top_switch
    ]

    output = {
        "metadata": {
            "rollouts_per_action": rollouts,
            "n_states": len(state_records),
            "baseline_checkpoint": baseline_ckpt,
            "augmented_checkpoint": aug_ckpt,
            "archetypes": ARCHETYPES,
        },
        "summary": summary,
        "records": state_records,
    }
    out_path = os.path.join(OUT_DIR, "action_switching_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=200, help="Monte Carlo rollouts per action/opponent")
    parser.add_argument("--limit-states", type=int, default=None, help="Optional cap for faster exploratory runs")
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--augmented-checkpoint", default=DEFAULT_AUG_CKPT)
    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print("  action_switching_analysis_v1")
    print(f"  states     : {'all' if args.limit_states is None else args.limit_states}")
    print(f"  rollouts   : {args.rollouts} per action/opponent")
    print(f"{'=' * 72}\n")

    out_path, summary = run_analysis(
        rollouts=args.rollouts,
        limit_states=args.limit_states,
        baseline_ckpt=args.baseline_checkpoint,
        aug_ckpt=args.augmented_checkpoint,
    )
    print("\nSummary")
    print(f"  Oracle switch rate                 : {summary['oracle_switch_rate']:.4f}")
    print(f"  Baseline oracle-label accuracy     : {summary['baseline_accuracy']:.4f}")
    print(f"  Augmented oracle-label accuracy    : {summary['augmented_accuracy']:.4f}")
    if summary["baseline_accuracy_on_switch_states"] is not None:
        print(f"  Baseline accuracy on switch states : {summary['baseline_accuracy_on_switch_states']:.4f}")
        print(f"  Augmented accuracy on switch states: {summary['augmented_accuracy_on_switch_states']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
