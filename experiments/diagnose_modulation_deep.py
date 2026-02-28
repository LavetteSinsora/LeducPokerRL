"""
Deep modulation analysis of ModulatedValueAgent.

Architecture reminder:
  V(s, opp) = V_base(s) + gate(opp_stats) * delta(s, opp_stats)

  - V_base: frozen 15-dim value network (pretrained)
  - delta: ModulationNetwork(19->32->32->1) — takes [game_state, opp_stats]
  - gate: GateNetwork(4->16->1->sigmoid) — takes opp_stats ONLY

Questions investigated:
  1. Gate output distribution — across actual gameplay, when is it high vs low?
  2. What specific modulations does the agent learn? Show extreme deltas
     with full game-state context (hand, board, pot, opponent profile).
  3. State-conditioned modulation: does the SAME opponent profile produce
     different deltas in different game states? If so, the delta network
     is doing useful state-dependent work, but the gate can't distinguish.
  4. Sanity check: should the gate also receive game state? We test whether
     optimal gating varies by game state for the same opponent profile.
"""

import json
import os
import sys
import random
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.modulated_value import ModulatedValueAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.heuristic import HeuristicAgent
from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession, OpponentStats

# ─── Paths ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modulated_value_agent.pt")
BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "value_based_agent.pt")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "diagnose_modulation_deep_results.json")


# ─── Helpers ──────────────────────────────────────────────────────────

def describe_state(obs, viewer_id):
    """Human-readable description of a game state."""
    board_str = obs.board if obs.board else "—"
    has_pair = obs.board is not None and obs.player_hand == obs.board
    pot_mine = obs.pot[viewer_id]
    pot_opp = obs.pot[1 - viewer_id]
    return {
        "hand": obs.player_hand,
        "board": board_str,
        "has_pair": has_pair,
        "round": obs.current_round,
        "my_pot": pot_mine,
        "opp_pot": pot_opp,
        "total_pot": pot_mine + pot_opp,
        "raises": obs.raises_this_round,
        "is_my_turn": viewer_id == obs.current_player,
        "legal_actions": [a.name for a in obs.legal_actions],
    }


def decompose_value(agent, obs, viewer_id):
    """Decompose the modulated value into v_base, delta, gate, and final value."""
    base_enc = agent.encode_observation(obs, viewer_id=viewer_id)
    stats_vec = agent._encode_stats(obs)

    with torch.no_grad():
        v_base = agent.model(base_enc).item()
        mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)
        delta = agent.mod_net(mod_input).item()
        gate = agent.gate_net(stats_vec.unsqueeze(0)).item()
        modulated = v_base + gate * delta

    return {
        "v_base": round(v_base, 4),
        "delta": round(delta, 4),
        "gate": round(gate, 4),
        "effective_modulation": round(gate * delta, 4),
        "modulated_value": round(modulated, 4),
        "stats_vec": [round(x, 4) for x in stats_vec.tolist()],
    }


def make_stats(fold_rate=0.5, raise_rate=0.5, fold_to_raise=0.5, confidence=0.5):
    """Create OpponentStats with specific behavioral rates."""
    stats = OpponentStats()
    hands = int(confidence * 50)
    stats.hands_observed = hands
    if confidence > 0:
        total = max(hands * 2, 10)
        stats.total_actions = total
        stats.fold_count = int(round(fold_rate * total))
        stats.raise_count = int(round(raise_rate * total))
        stats.call_count = total - stats.fold_count - stats.raise_count
        if stats.call_count < 0:
            stats.call_count = 0
            stats.fold_count = total - stats.raise_count
        facing = max(int(total * 0.3), 1)
        stats.actions_facing_raise = facing
        stats.folds_facing_raise = int(round(fold_to_raise * facing))
    else:
        stats.total_actions = 0
    return stats


def collect_gameplay_data(agent, opponent, n_hands=200, seed=42):
    """Play actual hands and record decomposed values at every decision point."""
    random.seed(seed)
    torch.manual_seed(seed)

    game = LeducGame()
    # Track opponent stats manually
    opp_stats = OpponentStats()
    records = []

    for hand_idx in range(n_hands):
        game.reset()
        was_raise_pending = False

        while not game.is_finished:
            player = game.current_player
            obs = game.get_observation(viewer_id=player)

            if player == 0:  # Our agent
                # Attach accumulated stats
                from dataclasses import replace
                obs_with_stats = replace(obs, opponent_stats=opp_stats)
                decomp = decompose_value(agent, obs_with_stats, viewer_id=0)
                state_desc = describe_state(obs, 0)
                records.append({
                    "hand": hand_idx,
                    "state": state_desc,
                    **decomp,
                })
                action = agent.select_action(obs_with_stats)
            else:  # Opponent
                action = opponent.select_action(obs)
                # Record opponent action for stats
                opp_stats.record_action(action.name, was_raise_pending)

            was_raise_pending = (action == Action.RAISE)
            game.step(action)

        opp_stats.record_hand_complete()

    return records


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1: Gate Distribution in Actual Gameplay
# ═══════════════════════════════════════════════════════════════════════

def experiment_gate_distribution(all_records):
    """Analyze gate output distribution across actual gameplay decisions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Gate Distribution in Actual Gameplay")
    print("=" * 70)

    gates = [r["gate"] for r in all_records]
    results = {
        "n_decisions": len(gates),
        "mean": round(float(np.mean(gates)), 4),
        "std": round(float(np.std(gates)), 4),
        "min": round(float(np.min(gates)), 4),
        "max": round(float(np.max(gates)), 4),
        "percentiles": {
            "p5": round(float(np.percentile(gates, 5)), 4),
            "p25": round(float(np.percentile(gates, 25)), 4),
            "p50": round(float(np.percentile(gates, 50)), 4),
            "p75": round(float(np.percentile(gates, 75)), 4),
            "p95": round(float(np.percentile(gates, 95)), 4),
        },
    }

    # Gate by game round
    round_gates = defaultdict(list)
    for r in all_records:
        round_gates[r["state"]["round"]].append(r["gate"])
    results["by_round"] = {
        f"round_{k}": {"mean": round(float(np.mean(v)), 4), "n": len(v)}
        for k, v in sorted(round_gates.items())
    }

    # Gate by hand type (pair vs no pair)
    pair_gates = [r["gate"] for r in all_records if r["state"]["has_pair"]]
    no_pair_gates = [r["gate"] for r in all_records if not r["state"]["has_pair"]]
    results["by_pair"] = {
        "has_pair": {"mean": round(float(np.mean(pair_gates)), 4), "n": len(pair_gates)} if pair_gates else None,
        "no_pair": {"mean": round(float(np.mean(no_pair_gates)), 4), "n": len(no_pair_gates)} if no_pair_gates else None,
    }

    # Histogram (10 bins)
    hist, edges = np.histogram(gates, bins=10)
    results["histogram"] = {
        "bins": [round(float(e), 4) for e in edges],
        "counts": hist.tolist(),
    }

    print(f"  N decisions: {len(gates)}")
    print(f"  Gate range: [{results['min']}, {results['max']}]")
    print(f"  Mean: {results['mean']}, Std: {results['std']}")
    print(f"  Percentiles: {results['percentiles']}")
    print(f"  By round: {results['by_round']}")
    print(f"  By pair: {results['by_pair']}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Experiment 2: Extreme Modulations with Game State Context
# ═══════════════════════════════════════════════════════════════════════

def experiment_extreme_modulations(all_records, top_n=10):
    """Find the most extreme deltas and show their full game context."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Top {top_n} Most Extreme Modulations")
    print("=" * 70)

    # Sort by delta (most negative and most positive)
    sorted_by_delta = sorted(all_records, key=lambda r: r["delta"])
    most_negative = sorted_by_delta[:top_n]
    most_positive = sorted_by_delta[-top_n:][::-1]

    # Sort by effective modulation (gate * delta)
    sorted_by_eff = sorted(all_records, key=lambda r: r["effective_modulation"])
    most_neg_eff = sorted_by_eff[:top_n]
    most_pos_eff = sorted_by_eff[-top_n:][::-1]

    def format_record(r):
        s = r["state"]
        return {
            "hand": s["hand"],
            "board": s["board"],
            "has_pair": s["has_pair"],
            "round": s["round"],
            "pot": f"{s['my_pot']}:{s['opp_pot']}",
            "raises": s["raises"],
            "v_base": r["v_base"],
            "delta": r["delta"],
            "gate": r["gate"],
            "eff_mod": r["effective_modulation"],
            "final": r["modulated_value"],
            "stats": r["stats_vec"],
        }

    results = {
        "most_negative_delta": [format_record(r) for r in most_negative],
        "most_positive_delta": [format_record(r) for r in most_positive],
        "most_negative_effective": [format_record(r) for r in most_neg_eff],
        "most_positive_effective": [format_record(r) for r in most_pos_eff],
    }

    print("\n  --- Most Negative Deltas ---")
    for i, r in enumerate(most_negative[:5]):
        s = r["state"]
        print(f"  [{i+1}] hand={s['hand']} board={s['board']} pair={s['has_pair']} "
              f"pot={s['my_pot']}:{s['opp_pot']} raises={s['raises']} round={s['round']}")
        print(f"       v_base={r['v_base']:+.4f}  delta={r['delta']:+.4f}  "
              f"gate={r['gate']:.4f}  eff={r['effective_modulation']:+.4f}  "
              f"final={r['modulated_value']:+.4f}")
        print(f"       stats={r['stats_vec']}")

    print("\n  --- Most Positive Deltas ---")
    for i, r in enumerate(most_positive[:5]):
        s = r["state"]
        print(f"  [{i+1}] hand={s['hand']} board={s['board']} pair={s['has_pair']} "
              f"pot={s['my_pot']}:{s['opp_pot']} raises={s['raises']} round={s['round']}")
        print(f"       v_base={r['v_base']:+.4f}  delta={r['delta']:+.4f}  "
              f"gate={r['gate']:.4f}  eff={r['effective_modulation']:+.4f}  "
              f"final={r['modulated_value']:+.4f}")
        print(f"       stats={r['stats_vec']}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Experiment 3: State-Conditioned Modulation Variance
# ═══════════════════════════════════════════════════════════════════════

def experiment_state_conditioned_modulation(agent):
    """
    Test: does the SAME opponent profile produce different deltas in
    different game states?

    If delta varies significantly across states for a fixed opponent profile,
    then the modulation network IS doing state-dependent work. The gate
    (which only sees opponent stats) can't distinguish these states.

    This directly tests whether adding game state to the gate could help.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: State-Conditioned Modulation Variance")
    print("=" * 70)

    # Generate diverse game states
    game = LeducGame()
    random.seed(42)
    torch.manual_seed(42)

    states = []
    for _ in range(500):
        game.reset()
        while not game.is_finished:
            player = game.current_player
            obs = game.get_observation(viewer_id=player)
            if player == 0:
                states.append((obs, 0))
            action = random.choice(obs.legal_actions)
            game.step(action)

    # Categorize states
    state_categories = {
        "preflop_J": [],
        "preflop_Q": [],
        "preflop_K": [],
        "postflop_pair": [],
        "postflop_no_pair_high": [],  # K on non-K board
        "postflop_no_pair_low": [],   # J on non-J board
    }

    for obs, vid in states:
        if obs.current_round == 0:
            state_categories[f"preflop_{obs.player_hand}"].append((obs, vid))
        elif obs.board is not None:
            if obs.player_hand == obs.board:
                state_categories["postflop_pair"].append((obs, vid))
            elif obs.player_hand == "K":
                state_categories["postflop_no_pair_high"].append((obs, vid))
            elif obs.player_hand == "J":
                state_categories["postflop_no_pair_low"].append((obs, vid))

    # Test opponent profiles
    profiles = {
        "passive": make_stats(fold_rate=0.6, raise_rate=0.1, fold_to_raise=0.7, confidence=1.0),
        "aggressive": make_stats(fold_rate=0.1, raise_rate=0.6, fold_to_raise=0.2, confidence=1.0),
        "balanced": make_stats(fold_rate=0.33, raise_rate=0.33, fold_to_raise=0.5, confidence=1.0),
    }

    results = {}

    for profile_name, stats in profiles.items():
        stats_vec = torch.tensor(stats.to_feature_vector(), dtype=torch.float32)
        with torch.no_grad():
            gate = agent.gate_net(stats_vec.unsqueeze(0)).item()

        profile_data = {"gate": round(gate, 4), "stats_vec": [round(x, 4) for x in stats.to_feature_vector()]}
        category_deltas = {}

        for cat_name, cat_states in state_categories.items():
            if len(cat_states) < 5:
                continue

            deltas = []
            from dataclasses import replace
            for obs, vid in cat_states[:50]:  # cap at 50 per category
                obs_with_stats = replace(obs, opponent_stats=stats)
                base_enc = agent.encode_observation(obs_with_stats, viewer_id=vid)
                with torch.no_grad():
                    v_base = agent.model(base_enc).item()
                    mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)
                    delta = agent.mod_net(mod_input).item()
                deltas.append(delta)

            category_deltas[cat_name] = {
                "mean": round(float(np.mean(deltas)), 4),
                "std": round(float(np.std(deltas)), 4),
                "min": round(float(np.min(deltas)), 4),
                "max": round(float(np.max(deltas)), 4),
                "range": round(float(np.max(deltas) - np.min(deltas)), 4),
                "n": len(deltas),
            }

        profile_data["by_state_category"] = category_deltas

        # Compute cross-category variance (is delta different across states?)
        all_means = [v["mean"] for v in category_deltas.values()]
        if len(all_means) >= 2:
            profile_data["cross_category_std"] = round(float(np.std(all_means)), 4)
            profile_data["cross_category_range"] = round(float(max(all_means) - min(all_means)), 4)
        else:
            profile_data["cross_category_std"] = 0.0
            profile_data["cross_category_range"] = 0.0

        results[profile_name] = profile_data

        print(f"\n  --- Profile: {profile_name} (gate={gate:.4f}) ---")
        for cat, data in category_deltas.items():
            print(f"    {cat:>25s}: delta_mean={data['mean']:+.4f}  "
                  f"std={data['std']:.4f}  range=[{data['min']:+.4f}, {data['max']:+.4f}]")
        print(f"    Cross-category spread: std={profile_data['cross_category_std']:.4f}  "
              f"range={profile_data['cross_category_range']:.4f}")

    # Summary insight
    avg_cross_range = np.mean([v["cross_category_range"] for v in results.values()])
    avg_within_std = np.mean([
        np.mean([cat["std"] for cat in v["by_state_category"].values()])
        for v in results.values()
    ])
    results["summary"] = {
        "avg_cross_category_range": round(float(avg_cross_range), 4),
        "avg_within_category_std": round(float(avg_within_std), 4),
        "state_matters": avg_cross_range > 0.02,
        "interpretation": (
            "The delta network produces DIFFERENT modulations for different game states "
            "even with the SAME opponent profile. A gate that sees game state could "
            "selectively apply or suppress these state-dependent adjustments."
            if avg_cross_range > 0.02
            else "Delta varies little across states for a fixed opponent — "
            "the gate's state-blindness is not a limitation."
        ),
    }

    print(f"\n  SUMMARY:")
    print(f"    Avg cross-category range: {avg_cross_range:.4f}")
    print(f"    Avg within-category std:  {avg_within_std:.4f}")
    print(f"    State matters for modulation: {results['summary']['state_matters']}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Experiment 4: Gate Improvement Potential (State-Aware Gate Sanity Check)
# ═══════════════════════════════════════════════════════════════════════

def experiment_gate_improvement_potential(agent):
    """
    Sanity check: Would a state-aware gate be better?

    We compute the "oracle gate" for each state — the gate value that would
    minimize squared error between the modulated value and the actual outcome.

    If oracle gates vary significantly across states (for the same opponent
    profile), then a state-aware gate has room to improve.

    We also measure how much value is lost from using a single gate value
    for all states with a given opponent profile.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Gate Improvement Potential (State-Aware Gate)")
    print("=" * 70)

    # Generate states with known outcomes (play to completion)
    game = LeducGame()
    random.seed(123)
    torch.manual_seed(123)

    # Collect (state, outcome) pairs
    state_outcome_pairs = []
    base_agent = ValueBasedAgent(model_path=BASE_MODEL_PATH)

    for _ in range(300):
        game.reset()
        states_this_hand = []

        while not game.is_finished:
            player = game.current_player
            obs = game.get_observation(viewer_id=player)
            if player == 0:
                states_this_hand.append((obs, 0))
            action = base_agent.select_action(obs)
            game.step(action)

        # Get outcome for player 0
        if game.is_finished:
            rewards = game.get_reward()
            outcome = rewards[0]  # player 0's reward
            for obs, vid in states_this_hand:
                state_outcome_pairs.append((obs, vid, outcome))

    print(f"  Collected {len(state_outcome_pairs)} (state, outcome) pairs")

    # For each opponent profile, compute optimal per-state gate
    profiles = {
        "passive": make_stats(fold_rate=0.6, raise_rate=0.1, fold_to_raise=0.7, confidence=1.0),
        "aggressive": make_stats(fold_rate=0.1, raise_rate=0.6, fold_to_raise=0.2, confidence=1.0),
        "balanced": make_stats(fold_rate=0.33, raise_rate=0.33, fold_to_raise=0.5, confidence=1.0),
    }

    results = {}
    from dataclasses import replace

    for profile_name, stats in profiles.items():
        stats_vec = torch.tensor(stats.to_feature_vector(), dtype=torch.float32)
        with torch.no_grad():
            learned_gate = agent.gate_net(stats_vec.unsqueeze(0)).item()

        # For each state, compute v_base, delta
        data_points = []
        for obs, vid, outcome in state_outcome_pairs:
            obs_s = replace(obs, opponent_stats=stats)
            base_enc = agent.encode_observation(obs_s, viewer_id=vid)
            with torch.no_grad():
                v_base = agent.model(base_enc).item()
                mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)
                delta = agent.mod_net(mod_input).item()

            data_points.append({
                "v_base": v_base,
                "delta": delta,
                "outcome": outcome,
                "hand": obs.player_hand,
                "board": obs.board,
                "round": obs.current_round,
            })

        # Compute optimal per-state gate: g* = (outcome - v_base) / delta
        # Clamped to [0, 1] since gate is sigmoid
        oracle_gates = []
        for dp in data_points:
            if abs(dp["delta"]) > 1e-6:
                g_star = (dp["outcome"] - dp["v_base"]) / dp["delta"]
                g_star = max(0.0, min(1.0, g_star))
            else:
                g_star = 0.0
            oracle_gates.append(g_star)

        # MSE with learned gate vs oracle gates
        mse_learned = np.mean([
            (dp["v_base"] + learned_gate * dp["delta"] - dp["outcome"]) ** 2
            for dp in data_points
        ])
        mse_oracle = np.mean([
            (dp["v_base"] + g * dp["delta"] - dp["outcome"]) ** 2
            for dp, g in zip(data_points, oracle_gates)
        ])
        mse_base_only = np.mean([
            (dp["v_base"] - dp["outcome"]) ** 2
            for dp in data_points
        ])

        results[profile_name] = {
            "learned_gate": round(learned_gate, 4),
            "oracle_gate_mean": round(float(np.mean(oracle_gates)), 4),
            "oracle_gate_std": round(float(np.std(oracle_gates)), 4),
            "oracle_gate_range": [round(float(np.min(oracle_gates)), 4),
                                   round(float(np.max(oracle_gates)), 4)],
            "mse_base_only": round(float(mse_base_only), 4),
            "mse_learned_gate": round(float(mse_learned), 4),
            "mse_oracle_gate": round(float(mse_oracle), 4),
            "improvement_potential": round(float(mse_learned - mse_oracle), 4),
            "n_points": len(data_points),
        }

        print(f"\n  --- {profile_name} (learned_gate={learned_gate:.4f}) ---")
        print(f"    Oracle gate: mean={np.mean(oracle_gates):.4f}, std={np.std(oracle_gates):.4f}")
        print(f"    MSE base_only:    {mse_base_only:.4f}")
        print(f"    MSE learned_gate: {mse_learned:.4f}")
        print(f"    MSE oracle_gate:  {mse_oracle:.4f}")
        print(f"    Improvement potential: {mse_learned - mse_oracle:.4f}")

    # Overall summary
    avg_improvement = np.mean([v["improvement_potential"] for v in results.values()])
    avg_oracle_std = np.mean([v["oracle_gate_std"] for v in results.values()])
    results["summary"] = {
        "avg_improvement_potential": round(float(avg_improvement), 4),
        "avg_oracle_gate_variability": round(float(avg_oracle_std), 4),
        "state_aware_gate_would_help": avg_improvement > 0.01,
        "recommendation": (
            "A state-aware gate (taking game state + opponent stats) could meaningfully "
            "reduce prediction error. Future work should try GateNetwork(19->16->1->sigmoid) "
            "with the full 19-dim input."
            if avg_improvement > 0.01
            else "The improvement potential is small — the current stats-only gate is adequate. "
            "The delta network already handles state-dependent adjustments."
        ),
    }

    print(f"\n  SUMMARY:")
    print(f"    Average improvement potential: {avg_improvement:.4f}")
    print(f"    Oracle gate variability:       {avg_oracle_std:.4f}")
    print(f"    State-aware gate recommended:  {results['summary']['state_aware_gate_would_help']}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Experiment 5: Opponent-Specific Modulation Profiles
# ═══════════════════════════════════════════════════════════════════════

def experiment_opponent_profiles(agent):
    """
    Play against each actual opponent type and analyze how the modulation
    adapts as opponent stats accumulate over a session.

    Shows the evolution of gate/delta across a 50-hand session per opponent.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Opponent-Specific Modulation Profiles")
    print("=" * 70)

    opponents = {
        "heuristic": HeuristicAgent(),
        "value_based": ValueBasedAgent(model_path=BASE_MODEL_PATH),
    }

    # Try loading adaptive_value if available
    adap_path = os.path.join(os.path.dirname(__file__), "..", "models", "adaptive_value_agent.pt")
    if os.path.exists(adap_path):
        opponents["adaptive_value"] = AdaptiveValueAgent(model_path=adap_path)

    results = {}

    for opp_name, opponent in opponents.items():
        print(f"\n  --- vs {opp_name} ---")
        records = collect_gameplay_data(agent, opponent, n_hands=50, seed=42)

        # Gate evolution over hands
        hand_numbers = sorted(set(r["hand"] for r in records))
        gate_by_hand = defaultdict(list)
        delta_by_hand = defaultdict(list)
        for r in records:
            gate_by_hand[r["hand"]].append(r["gate"])
            delta_by_hand[r["hand"]].append(r["delta"])

        gate_evolution = []
        delta_evolution = []
        for h in hand_numbers:
            gate_evolution.append(round(float(np.mean(gate_by_hand[h])), 4))
            delta_evolution.append(round(float(np.mean(delta_by_hand[h])), 4))

        # Stats at end of session
        final_stats = records[-1]["stats_vec"] if records else [0.5, 0.5, 0.5, 0.0]

        # Aggregate metrics
        all_gates = [r["gate"] for r in records]
        all_deltas = [r["delta"] for r in records]
        all_eff = [r["effective_modulation"] for r in records]

        results[opp_name] = {
            "n_decisions": len(records),
            "final_stats": final_stats,
            "gate": {
                "start_avg": round(float(np.mean(all_gates[:5])), 4) if len(all_gates) >= 5 else None,
                "end_avg": round(float(np.mean(all_gates[-5:])), 4) if len(all_gates) >= 5 else None,
                "overall_mean": round(float(np.mean(all_gates)), 4),
            },
            "delta": {
                "mean": round(float(np.mean(all_deltas)), 4),
                "std": round(float(np.std(all_deltas)), 4),
            },
            "effective_modulation": {
                "mean": round(float(np.mean(all_eff)), 4),
                "std": round(float(np.std(all_eff)), 4),
            },
            "gate_evolution_by_hand": gate_evolution[:10] + ["..."] + gate_evolution[-5:] if len(gate_evolution) > 15 else gate_evolution,
        }

        print(f"    Decisions: {len(records)}")
        print(f"    Final stats: {final_stats}")
        print(f"    Gate: start={results[opp_name]['gate']['start_avg']}, "
              f"end={results[opp_name]['gate']['end_avg']}, "
              f"mean={results[opp_name]['gate']['overall_mean']}")
        print(f"    Delta: mean={results[opp_name]['delta']['mean']:+.4f}, "
              f"std={results[opp_name]['delta']['std']:.4f}")
        print(f"    Eff modulation: mean={results[opp_name]['effective_modulation']['mean']:+.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("DEEP MODULATION ANALYSIS — ModulatedValueAgent")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")

    # Load agent
    agent = ModulatedValueAgent(model_path=MODEL_PATH)
    agent.set_train_mode(False)

    # Collect gameplay data against heuristic (accumulates opponent stats)
    print("\nCollecting gameplay data (200 hands vs heuristic)...")
    heuristic = HeuristicAgent()
    gameplay_records = collect_gameplay_data(agent, heuristic, n_hands=200, seed=42)
    print(f"  Collected {len(gameplay_records)} decision points.")

    all_results = {}

    # Run experiments
    all_results["gate_distribution"] = experiment_gate_distribution(gameplay_records)
    all_results["extreme_modulations"] = experiment_extreme_modulations(gameplay_records)
    all_results["state_conditioned"] = experiment_state_conditioned_modulation(agent)
    all_results["gate_improvement"] = experiment_gate_improvement_potential(agent)
    all_results["opponent_profiles"] = experiment_opponent_profiles(agent)

    # ─── Final Synthesis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SYNTHESIS")
    print("=" * 70)

    state_matters = all_results["state_conditioned"]["summary"]["state_matters"]
    gate_helps = all_results["gate_improvement"]["summary"]["state_aware_gate_would_help"]

    print(f"\n  1. GATE DISTRIBUTION: The gate operates in a narrow band "
          f"[{all_results['gate_distribution']['min']}, {all_results['gate_distribution']['max']}] "
          f"with mean {all_results['gate_distribution']['mean']:.4f}.")

    print(f"\n  2. STATE-DEPENDENT MODULATION: {'YES' if state_matters else 'NO'} — "
          f"delta varies across game states (cross-category range: "
          f"{all_results['state_conditioned']['summary']['avg_cross_category_range']:.4f}).")

    print(f"\n  3. STATE-AWARE GATE POTENTIAL: {'RECOMMENDED' if gate_helps else 'NOT NEEDED'} — "
          f"improvement potential: "
          f"{all_results['gate_improvement']['summary']['avg_improvement_potential']:.4f}.")

    if state_matters and gate_helps:
        print("\n  RECOMMENDATION: The modulation network learns meaningfully different")
        print("  adjustments for different game states. A state-aware gate could selectively")
        print("  amplify helpful modulations and suppress harmful ones. Consider:")
        print("    - GateNetwork(19->16->1->sigmoid) taking full [game_state, opp_stats] input")
        print("    - Or GateNetwork(8->16->1->sigmoid) taking [hand_strength, round, pot, opp_stats]")
    elif state_matters and not gate_helps:
        print("\n  FINDING: While delta varies by state, the current gate adequately handles this.")
        print("  The delta network already adapts per-state; the gate's role is opponent-level gating.")
    else:
        print("\n  FINDING: Modulation is relatively uniform across states for a given opponent.")
        print("  The current stats-only gate design is well-matched to the modulation behavior.")

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
