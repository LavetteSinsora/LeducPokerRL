"""
Round-Robin Tournament — opp_stats_modulation_v2
=================================================
Participants:
  v2a_ungated       UngatedModAgent   (stat-aware, opp_stats kwarg)
  v2b_state_gated   StateGatedModAgent (stat-aware, opp_stats kwarg)
  value_based       ValueBasedAgent   (standard)
  adaptive_value    AdaptiveValueAgent (standard + obs.opponent_stats)
  heuristic         HeuristicAgent    (standard)
  cfr               CFRAgent          (standard)

Protocol:
  - 5,000 rounds per ordered matchup (A vs B) with position alternation
  - Stats tracked for BOTH players in every hand (each tracks the other)
  - Results reported as chips/round from each agent's perspective
  - Full N×N matrix printed; summary rankings by avg score

Usage:
  python -m preliminary_experiments.opp_stats_modulation_v2.round_robin
  python -m preliminary_experiments.opp_stats_modulation_v2.round_robin --rounds 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from engine.leduc_game import LeducGame, Action
from paper.evaluation.shared.stats_tracker import OpponentStatsTracker
from paper.evaluation.shared.training_recipe import encode_game_state

# ── agent imports ────────────────────────────────────────────────────────────

from agents.value_based.agent import ValueBasedAgent
from agents.adaptive_value.agent import AdaptiveValueAgent
from agents.heuristic.agent import HeuristicAgent
from agents.cfr.agent import CFRAgent
from preliminary_experiments.opp_stats_modulation_v2.variant_a_ungated.agent import UngatedModAgent
from preliminary_experiments.opp_stats_modulation_v2.variant_b_state_gated.agent import StateGatedModAgent

# ── constants ────────────────────────────────────────────────────────────────

SESSION_LENGTH  = 100
PRIOR_STRENGTH  = 20.0
DEFAULT_ROUNDS  = 5_000


def _is_stat_aware(agent) -> bool:
    """Returns True if the agent uses the opp_stats= kwarg in select_action."""
    return isinstance(agent, (UngatedModAgent, StateGatedModAgent))


def _load_pool_means() -> dict:
    priors_path = os.path.join(HERE, "outputs", "variant_a_ungated", "pool_priors.json")
    if os.path.exists(priors_path):
        with open(priors_path) as f:
            return json.load(f)
    # Fallback: uniform uninformative means
    return {
        "preflop_fold_rate": 0.5,
        "preflop_raise_rate": 0.5,
        "flop_raise_rate": 0.5,
        "preflop_fold_to_raise": 0.5,
        "flop_fold_to_raise": 0.5,
        "raise_after_raise_rate": 0.5,
    }


def play_head_to_head(
    agent_a,
    agent_b,
    num_rounds: int,
    pool_means: dict,
    session_length: int = SESSION_LENGTH,
    prior_strength: float = PRIOR_STRENGTH,
) -> tuple[float, float]:
    """
    Play `num_rounds` hands between agent_a (seat 0) and agent_b (seat 1),
    alternating seats every hand for position fairness.

    Both agents get a stats tracker tracking the other player. Stat-aware
    agents (v2a/v2b) receive stats via select_action(obs, opp_stats=...).
    Standard agents receive select_action(obs) — no stats argument.
    AdaptiveValueAgent reads stats from obs.opponent_stats (injected here).

    Returns (avg_chips_a, avg_chips_b) over all rounds.
    """
    is_a_stat = _is_stat_aware(agent_a)
    is_b_stat = _is_stat_aware(agent_b)
    is_a_adaptive = isinstance(agent_a, AdaptiveValueAgent)
    is_b_adaptive = isinstance(agent_b, AdaptiveValueAgent)

    # Two trackers: tracker_for_a tracks b's actions (used by a to model b)
    #               tracker_for_b tracks a's actions (used by b to model a)
    tracker_for_a = OpponentStatsTracker(pool_means, prior_strength, session_length)
    tracker_for_b = OpponentStatsTracker(pool_means, prior_strength, session_length)

    hands_since_reset = 0
    total_a = 0.0
    total_b = 0.0

    for i in range(num_rounds):
        # Alternate seats every hand
        if i % 2 == 0:
            seat_a, seat_b = 0, 1
        else:
            seat_a, seat_b = 1, 0

        game = LeducGame()
        game.reset()

        prev_raise = False
        prev_round = -1

        while not game.is_finished:
            cp = game.current_player
            obs_cp = game.get_observation(viewer_id=cp)

            if obs_cp.current_round != prev_round:
                prev_raise = False
                prev_round = obs_cp.current_round

            if cp == seat_a:
                # agent_a acts; tracker_for_a has b's stats from a's POV
                if is_a_stat:
                    stats = tracker_for_a.get_features()
                    action = agent_a.select_action(obs_cp, opp_stats=stats)
                elif is_a_adaptive:
                    import dataclasses
                    stats_4 = tracker_for_a.get_features()[:4]
                    feats = stats_4.tolist() if hasattr(stats_4, "tolist") else list(stats_4)
                    class _StatsA:
                        def to_feature_vector(self):
                            return feats
                    obs_with_stats = dataclasses.replace(obs_cp, opponent_stats=_StatsA())
                    action = agent_a.select_action(obs_with_stats)
                else:
                    action = agent_a.select_action(obs_cp)
                # tracker_for_b records a's action (b is observing a)
                tracker_for_b.update_action(action, obs_cp.current_round,
                                            prev_raise, obs_cp.legal_actions)
            else:
                # agent_b acts; tracker_for_b has a's stats from b's POV
                if is_b_stat:
                    stats = tracker_for_b.get_features()
                    action = agent_b.select_action(obs_cp, opp_stats=stats)
                elif is_b_adaptive:
                    import dataclasses
                    stats_4 = tracker_for_b.get_features()[:4]
                    feats = stats_4.tolist() if hasattr(stats_4, "tolist") else list(stats_4)
                    class _StatsB:
                        def to_feature_vector(self):
                            return feats
                    obs_with_stats = dataclasses.replace(obs_cp, opponent_stats=_StatsB())
                    action = agent_b.select_action(obs_with_stats)
                else:
                    action = agent_b.select_action(obs_cp)
                # tracker_for_a records b's action (a is observing b)
                tracker_for_a.update_action(action, obs_cp.current_round,
                                            prev_raise, obs_cp.legal_actions)

            prev_raise = (action == Action.RAISE)
            game.step(action)

        tracker_for_a.update_hand_end()
        tracker_for_b.update_hand_end()

        rewards = game.get_reward()
        total_a += rewards[seat_a]
        total_b += rewards[seat_b]

        # Session reset
        hands_since_reset += 1
        if hands_since_reset >= session_length:
            tracker_for_a.reset()
            tracker_for_b.reset()
            hands_since_reset = 0

    return total_a / num_rounds, total_b / num_rounds


def build_agents(root: str) -> dict:
    agents = {}

    print("Loading agents...")

    vb_path = os.path.join(root, "agents", "value_based", "checkpoint.pt")
    agents["value_based"] = ValueBasedAgent(model_path=vb_path)
    agents["value_based"].set_train_mode(False)
    print("  value_based       ✓")

    av_path = os.path.join(root, "agents", "adaptive_value", "checkpoint.pt")
    agents["adaptive_value"] = AdaptiveValueAgent(model_path=av_path)
    agents["adaptive_value"].set_train_mode(False)
    print("  adaptive_value    ✓")

    agents["heuristic"] = HeuristicAgent()
    agents["heuristic"].set_train_mode(False)
    print("  heuristic         ✓")

    cfr_path = os.path.join(root, "agents", "cfr", "checkpoint.pt")
    agents["cfr"] = CFRAgent(model_path=cfr_path)
    agents["cfr"].set_train_mode(False)
    print("  cfr               ✓")

    v2a_path = os.path.join(HERE, "outputs", "variant_a_ungated", "checkpoint_best_robust.pt")
    agents["v2a_ungated"] = UngatedModAgent()
    agents["v2a_ungated"].load_model(v2a_path)
    agents["v2a_ungated"].set_train_mode(False)
    print(f"  v2a_ungated       ✓  ({v2a_path})")

    v2b_path = os.path.join(HERE, "outputs", "variant_b_state_gated", "checkpoint_best_robust.pt")
    agents["v2b_state_gated"] = StateGatedModAgent()
    agents["v2b_state_gated"].load_model(v2b_path)
    agents["v2b_state_gated"].set_train_mode(False)
    print(f"  v2b_state_gated   ✓  ({v2b_path})")

    return agents


def run_tournament(agents: dict, num_rounds: int, pool_means: dict) -> dict:
    names = list(agents.keys())
    n = len(names)
    # scores[i][j] = avg chips for agent i when playing against agent j
    scores = {a: {} for a in names}

    pairs = list(combinations(names, 2))
    total_pairs = len(pairs)

    print(f"\nRunning {total_pairs} matchups × {num_rounds} rounds each...")
    print(f"  Agents: {names}\n")

    for idx, (name_a, name_b) in enumerate(pairs, 1):
        t0 = time.time()
        avg_a, avg_b = play_head_to_head(
            agents[name_a], agents[name_b],
            num_rounds=num_rounds,
            pool_means=pool_means,
        )
        elapsed = time.time() - t0
        scores[name_a][name_b] = avg_a
        scores[name_b][name_a] = avg_b
        print(f"  [{idx:2d}/{total_pairs}] {name_a:20s} vs {name_b:20s} | "
              f"{name_a}: {avg_a:+.3f}  {name_b}: {avg_b:+.3f}  ({elapsed:.0f}s)")

    return scores


def print_matrix(scores: dict):
    names = list(scores.keys())
    col_w = 14

    # Header
    header = f"{'':20s}" + "".join(f"{n:>{col_w}s}" for n in names) + f"{'AVG':>{col_w}s}"
    print("\n" + "=" * len(header))
    print("RESULTS MATRIX  (row = hero, col = opponent, value = chips/round)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    avgs = {}
    for name_a in names:
        row = f"{name_a:20s}"
        vals = []
        for name_b in names:
            if name_a == name_b:
                row += f"{'—':>{col_w}s}"
            else:
                v = scores[name_a].get(name_b, float("nan"))
                row += f"{v:>+{col_w}.3f}"
                vals.append(v)
        avg = np.mean(vals) if vals else float("nan")
        avgs[name_a] = avg
        row += f"{avg:>+{col_w}.3f}"
        print(row)

    print("-" * len(header))

    # Rankings
    ranked = sorted(avgs.items(), key=lambda x: -x[1])
    print("\nRANKINGS by avg chips/round (excluding self-play):")
    for rank, (name, avg) in enumerate(ranked, 1):
        print(f"  #{rank}  {name:20s}  {avg:+.3f}")

    return avgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"Rounds per matchup (default {DEFAULT_ROUNDS})")
    parser.add_argument("--output", type=str,
                        default=os.path.join(HERE, "outputs", "analysis", "round_robin.json"),
                        help="Path to save JSON results")
    args = parser.parse_args()

    pool_means = _load_pool_means()
    agents = build_agents(ROOT)

    t_start = time.time()
    scores = run_tournament(agents, num_rounds=args.rounds, pool_means=pool_means)
    avgs = print_matrix(scores)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result = {
        "rounds_per_matchup": args.rounds,
        "agents": list(agents.keys()),
        "scores": scores,
        "averages": avgs,
        "rankings": sorted(avgs.items(), key=lambda x: -x[1]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
