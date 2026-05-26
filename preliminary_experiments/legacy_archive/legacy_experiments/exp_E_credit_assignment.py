"""
Experiment E: Cross-Hand Credit Assignment & Stat Calibration (H5 + H2)

HYPOTHESIS: The TD(0) update cannot attribute credit across hand boundaries.
When the adaptive agent uses stats accumulated from hands 1-20 to make
better decisions in hand 21, the reward signal from hand 21 doesn't
propagate back to strengthen the "use stats → win" association. This
broken credit assignment means the agent can't learn that stat-conditioned
decisions lead to better long-term outcomes within a session.

Additionally (H2), the cold-start period (hands 1-10 of 30) contributes
substantially to total training time, with noisy/default stats that teach
the network to ignore them.

FALSIFICATION CONDITION: If hand-position-conditioned rewards within a
session are uniform (no upward trend as confidence grows), then the agent
doesn't benefit from accumulated stats regardless — suggesting H5 is
irrelevant (the agent doesn't use stats at all, see H3).

TEST:
  1. Compute true heuristic oracle stats from many games.
  2. Track confidence growth within a typical 30-hand session.
  3. Run many sessions, record per-hand: stats snapshot, hand outcome, confidence.
  4. Correlate confidence at hand N with performance at hand N.
  5. Compare: early-session vs late-session performance for adaptive vs vanilla.
  6. Analyze what fraction of training time is spent in "cold start" (low confidence).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from collections import defaultdict

from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.heuristic import HeuristicAgent
from src.engine.leduc_game import LeducGame
from src.engine.poker_session import PokerSession

ADAPTIVE_MODEL = "models/adaptive_value_agent.pt"
VANILLA_MODEL  = "models/value_based_agent.pt"
N_SESSIONS     = 1000
HANDS_PER_SESSION = 30


def run_session_with_tracking(adaptive, heuristic, n_sessions, hands_per_session):
    """
    Run sessions and track per-hand: confidence, fold_rate, raise_rate, reward.
    Returns a list of hand records.
    """
    hand_records = []  # [{hand_pos, confidence, fold_rate, raise_rate, reward}]

    for session_idx in range(n_sessions):
        session = PokerSession()
        session.reset()

        for hand_pos in range(hands_per_session):
            session.new_hand()
            swap = random.choice([True, False])
            adaptive_seat = 1 if swap else 0

            while not session.is_finished:
                pid = session.current_player
                obs = session.get_observation(viewer_id=pid)
                if pid == adaptive_seat:
                    action = adaptive.select_action(obs)
                else:
                    action = heuristic.select_action(obs)
                session.step(action)

            reward = session.game.get_reward()[adaptive_seat]

            # Record stats at this point in the session
            opp_seat = 1 - adaptive_seat
            # adaptive's view of opponent = stats[adaptive_seat]
            stats_vec = session.stats[adaptive_seat].to_feature_vector()
            hand_records.append({
                'session': session_idx,
                'hand_pos': hand_pos,
                'confidence': stats_vec[3],
                'fold_rate': stats_vec[0],
                'raise_rate': stats_vec[1],
                'fold_to_raise_rate': stats_vec[2],
                'reward': reward,
            })

    return hand_records


def run_vanilla_session_tracking(vanilla, heuristic, n_sessions, hands_per_session):
    """
    Run vanilla agent in repeated single-hand games as a baseline.
    Vanilla agent has no session context, so we just run independent hands.
    Returns per-hand rewards grouped by 'hand position' (artificial, for comparison).
    """
    hand_records = []
    game = LeducGame()

    for session_idx in range(n_sessions):
        for hand_pos in range(hands_per_session):
            game.reset()
            swap = random.choice([True, False])
            vanilla_seat = 1 if swap else 0
            agents = [None, None]
            agents[vanilla_seat] = vanilla
            agents[1 - vanilla_seat] = heuristic

            while not game.is_finished:
                pid = game.current_player
                obs = game.get_observation(viewer_id=pid)
                action = agents[pid].select_action(obs)
                game.step(action)

            reward = game.get_reward()[vanilla_seat]
            hand_records.append({
                'session': session_idx,
                'hand_pos': hand_pos,
                'reward': reward,
            })

    return hand_records


def analyze_confidence_reward_correlation(records):
    """
    How does confidence at the time of each hand correlate with that hand's reward?
    """
    confidences = [r['confidence'] for r in records]
    rewards = [r['reward'] for r in records]

    corr = np.corrcoef(confidences, rewards)[0, 1]
    print(f"  Pearson correlation(confidence, reward): {corr:.4f}")

    # Bin by confidence level
    bins = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        bucket = [r['reward'] for r in records if lo <= r['confidence'] < hi]
        if bucket:
            n_hands = len(bucket)
            mean_reward = np.mean(bucket)
            n_obs = round(lo * 50)
            print(f"  Confidence [{lo:.1f}-{hi:.1f}) (≈{n_obs:2d}-{round(hi*50):2d} hands seen): "
                  f"n={n_hands:6d}, avg reward = {mean_reward:+.4f}")

    return corr


def analyze_hand_position_trends(adaptive_records, vanilla_records, hands_per_session):
    """
    Compare adaptive vs vanilla performance by hand position within 'session'.
    """
    adaptive_by_pos = defaultdict(list)
    vanilla_by_pos = defaultdict(list)

    for r in adaptive_records:
        adaptive_by_pos[r['hand_pos']].append(r['reward'])
    for r in vanilla_records:
        vanilla_by_pos[r['hand_pos']].append(r['reward'])

    print(f"\n  Per-hand-position performance (adaptive vs vanilla baseline):")
    print(f"  {'Hand':>5} | {'Adaptive':>10} | {'Vanilla':>10} | {'Delta':>8} | {'Conf.':>6}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")

    hand_positions = sorted(adaptive_by_pos.keys())
    adaptive_early, vanilla_early = [], []
    adaptive_late, vanilla_late   = [], []

    for pos in hand_positions:
        a_mean = np.mean(adaptive_by_pos[pos]) if adaptive_by_pos[pos] else float('nan')
        v_mean = np.mean(vanilla_by_pos[pos])  if vanilla_by_pos[pos]  else float('nan')
        delta  = a_mean - v_mean
        conf   = min(pos / 50.0, 1.0)

        if pos < 10:
            adaptive_early.append(a_mean)
            vanilla_early.append(v_mean)
        elif pos >= 20:
            adaptive_late.append(a_mean)
            vanilla_late.append(v_mean)

        # Print every few rows
        if pos % 5 == 0:
            print(f"  {pos+1:>5} | {a_mean:>+10.4f} | {v_mean:>+10.4f} | {delta:>+8.4f} | {conf:>6.2f}")

    if adaptive_early and vanilla_early:
        ae = np.mean(adaptive_early)
        ve = np.mean(vanilla_early)
        al = np.mean(adaptive_late)
        vl = np.mean(vanilla_late)

        print(f"\n  Summary:")
        print(f"    Adaptive early (hands 1-10):  {ae:+.4f}")
        print(f"    Adaptive late  (hands 21-30): {al:+.4f}")
        print(f"    Vanilla  early (hands 1-10):  {ve:+.4f}")
        print(f"    Vanilla  late  (hands 21-30): {vl:+.4f}")
        print(f"    Adaptive trend (late-early):  {al-ae:+.4f}")
        print(f"    Vanilla  trend (late-early):  {vl-ve:+.4f}  (should be ≈0)")

        if al - ae > 0.05:
            print("\n    → Adaptive improves within session: H5 PARTIALLY SUPPORTED.")
            print("      (Stats help, but credit assignment may limit how much learning occurs.)")
        else:
            print("\n    → No clear intra-session improvement. Stats don't help across hands.")


def analyze_cold_start_fraction(hands_per_session=30):
    """
    What fraction of training time is spent in 'cold start' (confidence < 0.5)?
    """
    confidences = [min(h / 50.0, 1.0) for h in range(hands_per_session)]
    below_50 = sum(1 for c in confidences if c < 0.5)
    below_20 = sum(1 for c in confidences if c < 0.2)

    print(f"\n  Cold-start analysis (within a {hands_per_session}-hand session):")
    for h in range(hands_per_session):
        conf = min(h / 50.0, 1.0)
        bar  = '█' * int(conf * 20)
        if h % 5 == 0:
            print(f"    Hand {h+1:2d}: confidence = {conf:.2f}  {bar}")

    print(f"\n  Hands with confidence < 0.2 (very unreliable): {below_20}/{hands_per_session} = {100*below_20/hands_per_session:.0f}%")
    print(f"  Hands with confidence < 0.5 (low reliability):  {below_50}/{hands_per_session} = {100*below_50/hands_per_session:.0f}%")
    print(f"  → {100*below_50/hands_per_session:.0f}% of each session's hands have low-confidence stats.")


def main():
    print("=" * 65)
    print("EXPERIMENT E: Credit Assignment & Cold-Start Analysis (H5+H2)")
    print("=" * 65)

    adaptive = AdaptiveValueAgent(model_path=ADAPTIVE_MODEL)
    adaptive.model.eval()
    vanilla = ValueBasedAgent(model_path=VANILLA_MODEL)
    vanilla.model.eval()
    heuristic = HeuristicAgent()

    # ── Part 1: Cold-start fraction ──
    print(f"\n[1/4] Cold-start fraction within {HANDS_PER_SESSION}-hand sessions:")
    analyze_cold_start_fraction(HANDS_PER_SESSION)

    # ── Part 2: Run sessions with tracking ──
    print(f"\n[2/4] Running {N_SESSIONS} sessions (adaptive vs heuristic)...")
    adaptive_records = run_session_with_tracking(adaptive, heuristic, N_SESSIONS, HANDS_PER_SESSION)
    print(f"  Collected {len(adaptive_records)} hand records.")

    print(f"\n  Running {N_SESSIONS} vanilla baseline sessions...")
    vanilla_records = run_vanilla_session_tracking(vanilla, heuristic, N_SESSIONS, HANDS_PER_SESSION)
    print(f"  Collected {len(vanilla_records)} hand records.")

    # ── Part 3: Confidence-reward correlation ──
    print(f"\n[3/4] Confidence-Reward Correlation Analysis:")
    corr = analyze_confidence_reward_correlation(adaptive_records)

    # ── Part 4: Hand position trends ──
    print(f"\n[4/4] Hand Position Trends (adaptive vs vanilla):")
    analyze_hand_position_trends(adaptive_records, vanilla_records, HANDS_PER_SESSION)

    # ── Final summary ──
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY (H5: Credit Assignment Broken?)")
    print("=" * 65)
    overall_adaptive = np.mean([r['reward'] for r in adaptive_records])
    overall_vanilla  = np.mean([r['reward'] for r in vanilla_records])
    print(f"  Overall adaptive avg reward: {overall_adaptive:+.4f}")
    print(f"  Overall vanilla  avg reward: {overall_vanilla:+.4f}")
    print(f"  Gap (adaptive - vanilla):    {overall_adaptive - overall_vanilla:+.4f}")

    print(f"\n  Confidence-reward correlation: {corr:.4f}")
    if abs(corr) < 0.02:
        print("  → Near-zero correlation: confidence doesn't predict reward.")
        print("    Stats are not being used effectively (see H3).")
    elif corr > 0.02:
        print("  → Positive correlation: higher confidence → better performance.")
        print("    H5 is relevant: credit assignment limits this from being learned.")
    else:
        print("  → Negative correlation: unexpected. May indicate overfit to defaults.")


if __name__ == "__main__":
    main()
