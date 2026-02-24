"""
Experiment C: Feature Ablation — Oracle Stats vs Default Stats vs Zero Stats (H2)

HYPOTHESIS: The default stat values [0.5, 0.5, 0.5, 0.0] used during evaluation
(when no session context exists) are actively misleading the network. The value
0.5 for fold/raise rates is an arbitrary prior that may not reflect the true
heuristic opponent. Furthermore, cold-start training with these defaults trains
the network to treat 0.5-stat inputs as "uninformative", potentially suppressing
any conditioning on stats.

FALSIFICATION CONDITION: If oracle stats (true heuristic behavior) don't
improve performance over default stats, then H2 is false — the network
doesn't condition meaningfully on stats regardless of their accuracy.

TEST:
  1. Compute "oracle stats" from 10,000 games vs heuristic (true behavior profile)
  2. Evaluate adaptive agent with:
     a) Default stats [0.5, 0.5, 0.5, 0.0]
     b) Oracle stats (true heuristic fold/raise rates)
     c) Zeroed stats [0.0, 0.0, 0.0, 0.0]
     d) Inverted oracle stats (wrong model of opponent)
  3. Compare performance across conditions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
from dataclasses import replace

from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.heuristic import HeuristicAgent
from src.engine.leduc_game import LeducGame
from src.engine.poker_session import PokerSession, OpponentStats

ADAPTIVE_MODEL = "models/adaptive_value_agent.pt"
N_ORACLE_GAMES = 20_000
N_EVAL_GAMES = 10_000


def compute_oracle_stats(heuristic, n_games=20000):
    """
    Play n_games of heuristic vs heuristic and collect true action statistics.
    This gives us the "true" behavioral profile of the heuristic agent.
    """
    game = LeducGame()
    stats = OpponentStats()

    for _ in range(n_games):
        game.reset()
        while not game.is_finished:
            pid = game.current_player
            obs = game.get_observation(viewer_id=pid)
            action = heuristic.select_action(obs)
            opp_pot_before = obs.pot[1 - pid]
            my_pot_before = obs.pot[pid]
            was_facing_raise = opp_pot_before > my_pot_before
            stats.record_action(action.name, was_facing_raise)
            game.step(action)
        stats.record_hand_complete()

    return stats


def eval_with_fixed_stats(adaptive, heuristic, fixed_stats_vec, n_games=10000, label=""):
    """
    Evaluate adaptive agent where we override the stat features with a fixed vector.
    We monkey-patch the encode_observation to inject fixed stats.
    """
    game = LeducGame()
    total = 0.0
    fixed_stats_tensor = torch.tensor(fixed_stats_vec, dtype=torch.float32)

    original_encode = adaptive.encode_observation

    def patched_encode(obs, viewer_id=None):
        # Call parent (ValueBasedAgent) encode to get base 15 features
        from src.agents.value_based import ValueBasedAgent
        base = ValueBasedAgent.encode_observation(adaptive, obs, viewer_id)  # [1, 15]
        return torch.cat([base.squeeze(0), fixed_stats_tensor]).unsqueeze(0)  # [1, 19]

    # Also patch get_action_evaluations to use patched encode
    adaptive.encode_observation = patched_encode

    for _ in range(n_games):
        game.reset()
        swap = random.choice([True, False])
        adaptive_seat = 1 if swap else 0
        heuristic_seat = 1 - adaptive_seat
        agents = [None, None]
        agents[adaptive_seat] = adaptive
        agents[heuristic_seat] = heuristic

        while not game.is_finished:
            pid = game.current_player
            obs = game.get_observation(viewer_id=pid)
            action = agents[pid].select_action(obs)
            game.step(action)

        rewards = game.get_reward()
        total += rewards[adaptive_seat]

    adaptive.encode_observation = original_encode
    avg = total / n_games
    print(f"  {label}: {avg:+.4f} chips/round")
    return avg


def eval_patched_stats(adaptive, heuristic, stats_vec, n_games=10000):
    """
    Evaluate with a specific stat vector injected into every observation.
    Uses get_action_evaluations override to ensure stat vector is used consistently.
    """
    game = LeducGame()
    total = 0.0
    fixed = torch.tensor(stats_vec, dtype=torch.float32)

    def patched_encode(obs, viewer_id=None):
        from src.agents.value_based import ValueBasedAgent
        base = ValueBasedAgent.encode_observation(adaptive, obs, viewer_id)
        return torch.cat([base.squeeze(0), fixed]).unsqueeze(0)

    original_encode = adaptive.encode_observation
    adaptive.encode_observation = patched_encode

    for _ in range(n_games):
        game.reset()
        swap = random.choice([True, False])
        adaptive_seat = 1 if swap else 0
        agents = [None, None]
        agents[adaptive_seat] = adaptive
        agents[1 - adaptive_seat] = heuristic

        while not game.is_finished:
            pid = game.current_player
            obs = game.get_observation(viewer_id=pid)
            action = agents[pid].select_action(obs)
            game.step(action)

        total += game.get_reward()[adaptive_seat]

    adaptive.encode_observation = original_encode
    return total / n_games


def main():
    print("=" * 65)
    print("EXPERIMENT C: Oracle Stats vs Default Stats Ablation (H2)")
    print("=" * 65)

    adaptive = AdaptiveValueAgent(model_path=ADAPTIVE_MODEL)
    adaptive.model.eval()
    heuristic = HeuristicAgent()

    # ── Step 1: Compute oracle stats ──
    print(f"\n[1/4] Computing oracle stats from {N_ORACLE_GAMES} heuristic-vs-heuristic games...")
    oracle_stats = compute_oracle_stats(heuristic, N_ORACLE_GAMES)
    oracle_vec = oracle_stats.to_feature_vector()
    print(f"  Oracle fold_rate:           {oracle_vec[0]:.4f}")
    print(f"  Oracle raise_rate:          {oracle_vec[1]:.4f}")
    print(f"  Oracle fold_to_raise_rate:  {oracle_vec[2]:.4f}")
    print(f"  Oracle confidence:          {oracle_vec[3]:.4f} (clamped from {oracle_stats.hands_observed} hands)")

    default_vec = [0.5, 0.5, 0.5, 0.0]
    zero_vec    = [0.0, 0.0, 0.0, 0.0]
    # Inverted: wrong model (e.g. opponent never folds, always raises)
    inverted_vec = [1.0 - oracle_vec[0], 1.0 - oracle_vec[1], 1.0 - oracle_vec[2], 1.0]

    print(f"\n  Stat vectors to test:")
    print(f"    Default  [0.5,  0.5,  0.5,  0.0] — training eval default")
    print(f"    Oracle   {oracle_vec}    — true heuristic behavior")
    print(f"    Zero     [0.0,  0.0,  0.0,  0.0] — completely zeroed")
    print(f"    Inverted {[round(v,3) for v in inverted_vec]} — wrong model")

    # ── Step 2: Evaluate with each stat vector ──
    print(f"\n[2/4] Evaluating with each stat condition ({N_EVAL_GAMES} games each)...")
    results = {}
    results['default']  = eval_patched_stats(adaptive, heuristic, default_vec,  N_EVAL_GAMES)
    results['oracle']   = eval_patched_stats(adaptive, heuristic, oracle_vec,   N_EVAL_GAMES)
    results['zero']     = eval_patched_stats(adaptive, heuristic, zero_vec,     N_EVAL_GAMES)
    results['inverted'] = eval_patched_stats(adaptive, heuristic, inverted_vec, N_EVAL_GAMES)

    # ── Step 3: Oracle at hand 5 (partial confidence) ──
    partial_oracle = oracle_vec[:3] + [min(5 / 50.0, 1.0)]  # confidence after 5 hands
    print(f"\n[3/4] Oracle stats at different confidence levels (n hands seen)...")
    for n_hands in [0, 1, 5, 10, 20, 30, 50]:
        conf = min(n_hands / 50.0, 1.0)
        # When n_hands=0, fold/raise rates default to 0.5 regardless
        if n_hands == 0:
            vec = [0.5, 0.5, 0.5, 0.0]
        else:
            # Use true oracle rates but with partial confidence
            vec = oracle_vec[:3] + [conf]
        r = eval_patched_stats(adaptive, heuristic, vec, N_EVAL_GAMES // 5)
        print(f"    n_hands={n_hands:2d}, confidence={conf:.2f}: {r:+.4f} chips/round")

    # ── Step 4: Summary ──
    print(f"\n[4/4] RESULTS SUMMARY")
    print("=" * 65)
    for cond, val in results.items():
        print(f"  {cond:<12}: {val:+.4f}")

    oracle_minus_default = results['oracle'] - results['default']
    zero_minus_default   = results['zero']   - results['default']
    default_minus_inverted = results['default'] - results['inverted']

    print(f"\n  Oracle - Default:   {oracle_minus_default:+.4f}")
    print(f"  Zero   - Default:   {zero_minus_default:+.4f}")
    print(f"  Default - Inverted: {default_minus_inverted:+.4f}")

    max_gap = max(results.values()) - min(results.values())
    print(f"  Max performance gap across conditions: {max_gap:.4f}")

    print("\n  INTERPRETATION:")
    if max_gap < 0.05:
        print("  → Very small gap across all conditions.")
        print("    Network does NOT condition on stat features at all.")
        print("    H2 is FALSE (stats irrelevant), but H3 is STRONGLY SUPPORTED.")
    elif oracle_minus_default > 0.05:
        print("  → Oracle stats significantly outperform default.")
        print("    H2 is SUPPORTED: default stats are misleading.")
        print("    Agent CAN use stats, but wrong defaults hurt eval performance.")
    elif zero_minus_default < -0.05:
        print("  → Zero stats hurt vs default — 0.5 default is better than zero.")
        print("    Network was calibrated to 0.5 defaults during training.")
    else:
        print("  → Modest differences. Moderate sensitivity to stat values.")


if __name__ == "__main__":
    main()
