"""
Experiment A: Evaluation Distribution Mismatch (H1)

HYPOTHESIS: The adaptive agent is TRAINED with real accumulated session stats
but EVALUATED with default stats [0.5, 0.5, 0.5, 0.0] (because evaluation
uses LeducGame, not PokerSession). This train-test mismatch prevents the
agent from leveraging its learned stat-conditioning at evaluation time.

FALSIFICATION CONDITION: If performance is similar in both eval modes,
then H1 is false — the agent isn't using stats regardless.

TEST:
  1. Evaluate adaptive agent in single-hand mode (stats = default [0.5,0.5,0.5,0.0])
  2. Evaluate adaptive agent in session mode (stats accumulate across 30 hands)
  3. Track per-hand performance WITHIN a session to see if accumulating stats helps.

A large gap between (2) and (1), or a clear upward trend within sessions,
would confirm H1.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from collections import defaultdict

from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.heuristic import HeuristicAgent
from src.engine.leduc_game import LeducGame
from src.engine.poker_session import PokerSession
from src.training.evaluation import evaluate_agents

ADAPTIVE_MODEL = "models/adaptive_value_agent.pt"
N_SESSIONS = 500
HANDS_PER_SESSION = 30
N_SINGLE_HAND_GAMES = 10_000


def eval_single_hand_mode(adaptive, heuristic, n_games):
    """
    Standard single-hand evaluation: each game starts fresh with NO session context.
    Adaptive agent gets opponent_stats=None → encoded as [0.5, 0.5, 0.5, 0.0].
    This is what training's evaluate() function does.
    """
    game = LeducGame()
    total = 0.0
    for _ in range(n_games):
        game.reset()
        swap = random.choice([True, False])
        agents = [heuristic, adaptive] if swap else [adaptive, heuristic]
        adaptive_idx = 1 if swap else 0

        while not game.is_finished:
            pid = game.current_player
            obs = game.get_observation(viewer_id=pid)
            agents[pid].select_action(obs)
            game.step(agents[pid].select_action(obs))

        rewards = game.get_reward()
        total += rewards[adaptive_idx]
    return total / n_games


def eval_session_mode(adaptive, heuristic, n_sessions, hands_per_session):
    """
    Session-based evaluation: stats accumulate across hands within each session.
    Adaptive agent gets real opponent stats. Tracks per-hand position performance.
    """
    per_hand_rewards = defaultdict(list)   # hand_position -> list of rewards
    session_total = 0.0
    total_hands = 0

    for _ in range(n_sessions):
        session = PokerSession()
        session.reset()

        for hand_num in range(hands_per_session):
            session.new_hand()
            swap = random.choice([True, False])
            adaptive_seat = 1 if swap else 0
            heuristic_seat = 1 - adaptive_seat

            while not session.is_finished:
                pid = session.current_player
                obs = session.get_observation(viewer_id=pid)
                if pid == adaptive_seat:
                    action = adaptive.select_action(obs)
                else:
                    action = heuristic.select_action(obs)
                session.step(action)

            rewards = session.game.get_reward()
            adaptive_reward = rewards[adaptive_seat]
            per_hand_rewards[hand_num].append(adaptive_reward)
            session_total += adaptive_reward
            total_hands += 1

    per_hand_avg = {k: np.mean(v) for k, v in per_hand_rewards.items()}
    overall_avg = session_total / total_hands if total_hands > 0 else 0.0
    return per_hand_avg, overall_avg


def main():
    print("=" * 65)
    print("EXPERIMENT A: Evaluation Distribution Mismatch (H1)")
    print("=" * 65)

    adaptive = AdaptiveValueAgent(model_path=ADAPTIVE_MODEL)
    adaptive.model.eval()
    heuristic = HeuristicAgent()

    # ── Part 1: Standard single-hand eval (mimics what training uses) ──
    print(f"\n[1/3] Single-hand eval ({N_SINGLE_HAND_GAMES} games, stats=default)...")
    game = LeducGame()
    total_single = 0.0
    for _ in range(N_SINGLE_HAND_GAMES):
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
        total_single += rewards[adaptive_seat]

    avg_single = total_single / N_SINGLE_HAND_GAMES
    print(f"  Avg chips/round (single-hand): {avg_single:+.4f}")

    # ── Part 2: Session-mode eval (stats accumulate) ──
    print(f"\n[2/3] Session eval ({N_SESSIONS} sessions × {HANDS_PER_SESSION} hands)...")
    per_hand_avg, avg_session = eval_session_mode(adaptive, heuristic, N_SESSIONS, HANDS_PER_SESSION)
    print(f"  Avg chips/round (session mode, overall): {avg_session:+.4f}")

    # ── Part 3: Intra-session progression ──
    print(f"\n[3/3] Intra-session progression (avg chips by hand position):")
    early_hands = [per_hand_avg[k] for k in range(5) if k in per_hand_avg]
    mid_hands   = [per_hand_avg[k] for k in range(10, 20) if k in per_hand_avg]
    late_hands  = [per_hand_avg[k] for k in range(20, 30) if k in per_hand_avg]

    print(f"  Hands  1-5  (cold start, confidence≈0.0): {np.mean(early_hands):+.4f}")
    print(f"  Hands 11-20 (warming up, confidence≈0.3): {np.mean(mid_hands):+.4f}")
    print(f"  Hands 21-30 (warm,       confidence≈0.5): {np.mean(late_hands):+.4f}")

    print("\n  Per-hand breakdown:")
    for hand_pos in range(0, HANDS_PER_SESSION, 3):
        if hand_pos in per_hand_avg:
            conf = min(hand_pos / 50.0, 1.0)
            print(f"    Hand {hand_pos+1:2d} (confidence≈{conf:.2f}): {per_hand_avg[hand_pos]:+.4f}")

    # ── Summary ──
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Single-hand eval (training's eval method): {avg_single:+.4f}")
    print(f"  Session eval (stats accumulate):           {avg_session:+.4f}")
    gap = avg_session - avg_single
    print(f"  Gap (session - single):                    {gap:+.4f}")

    trend = np.mean(late_hands) - np.mean(early_hands) if early_hands and late_hands else 0
    print(f"  Intra-session trend (late - early):        {trend:+.4f}")

    print("\n  INTERPRETATION:")
    if abs(gap) < 0.05:
        print("  → Small gap: H1 is likely FALSE. Agent's performance doesn't")
        print("    meaningfully change with real vs default stats.")
    elif gap > 0.1:
        print("  → Large positive gap: H1 is SUPPORTED. Session stats help.")
        print("    The train-eval mismatch suppresses performance at eval time.")
    else:
        print("  → Moderate gap: H1 is PARTIALLY SUPPORTED.")

    if abs(trend) < 0.03:
        print("  → Flat intra-session trend: Agent doesn't improve within session.")
        print("    Stats may not be used (see H3) or are too noisy (H2).")
    elif trend > 0.05:
        print("  → Strong upward trend within session: Agent leverages accumulating stats.")
    else:
        print("  → Weak intra-session trend: Partial stat utilization.")


if __name__ == "__main__":
    main()
