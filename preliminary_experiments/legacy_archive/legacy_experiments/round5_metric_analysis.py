"""
Round 5 Metric Analysis — Comprehensive evaluation of belief agents.

Analyses:
  1. CFR strategy table dump (all infosets)
  2. Correct random baselines (analytical)
  3. Standardized belief metrics (both avg P(true) and argmax accuracy for all agents)
  4. Belief distributions grouped by true opponent hand
  5. Likelihood evaluation (avg P(true action) and top-1 accuracy)
  6. Modulation gate analysis (gate values, delta magnitudes)

Output: all results printed to stdout for conversational discussion.
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession
from src.cfr.strategy import TabularStrategyStore
from src.agents.belief_cfr import BeliefCfrAgent
from src.agents.belief_modulated import BeliefModulatedAgent
from src.agents.belief_confident import BeliefConfidentAgent
from src.agents.belief_stable import BeliefStableAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.belief_common import initialize_belief, CARD_MAP, CARDS, CARD_COUNTS

CFR_MODEL_PATH = "models/cfr_agent.pt"

# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: CFR Strategy Table Dump
# ═══════════════════════════════════════════════════════════════════════

def dump_cfr_strategy():
    """Load and print the full CFR Nash strategy table."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 1: CFR NASH STRATEGY TABLE")
    print("=" * 80)

    store = TabularStrategyStore()
    store.load(CFR_MODEL_PATH)

    print(f"\n  Total infosets: {store.num_info_sets()}")

    # Organize by hand and round
    preflop_keys = {}  # hand -> list of (key, strategy)
    flop_keys = {}     # hand -> list of (key, strategy)

    all_actions = [Action.FOLD, Action.CALL, Action.RAISE]

    for key in sorted(store.data.keys()):
        info = store.data[key]
        strategy = info.get_average_strategy()

        # Parse the key to determine hand and round
        parts = key.split(":")
        hand = parts[0]
        if "/" in key:
            # Flop key: "hand:board:preflop/flop"
            round_type = "flop"
            if hand not in flop_keys:
                flop_keys[hand] = []
            flop_keys[hand].append((key, strategy))
        else:
            # Preflop key: "hand:actions"
            round_type = "preflop"
            if hand not in preflop_keys:
                preflop_keys[hand] = []
            preflop_keys[hand].append((key, strategy))

    # Print preflop strategies
    print(f"\n  ── PRE-FLOP STRATEGIES ──")
    print(f"  {'Key':<25} {'FOLD':>8} {'CALL':>8} {'RAISE':>8}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8}")
    for hand in ['J', 'Q', 'K']:
        if hand not in preflop_keys:
            continue
        for key, strat in sorted(preflop_keys[hand], key=lambda x: x[0]):
            print(f"  {key:<25} {strat[0]:>7.1%} {strat[1]:>7.1%} {strat[2]:>7.1%}")
        print()

    # Print flop strategies (grouped by hand, showing only a few board combos)
    print(f"\n  ── FLOP STRATEGIES (selected) ──")
    print(f"  {'Key':<35} {'FOLD':>8} {'CALL':>8} {'RAISE':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8}")
    for hand in ['J', 'Q', 'K']:
        if hand not in flop_keys:
            continue
        # Show first entries and key ones
        entries = sorted(flop_keys[hand], key=lambda x: x[0])
        shown = 0
        for key, strat in entries:
            if shown < 15:  # Show up to 15 per hand
                print(f"  {key:<35} {strat[0]:>7.1%} {strat[1]:>7.1%} {strat[2]:>7.1%}")
                shown += 1
        if shown < len(entries):
            print(f"  ... ({len(entries) - shown} more for {hand})")
        print()

    # Summary statistics
    all_strategies = [info.get_average_strategy() for info in store.data.values()]
    all_strategies = np.array(all_strategies)
    print(f"\n  ── STRATEGY SUMMARY ──")
    print(f"  Avg action dist:  FOLD={np.mean(all_strategies[:, 0]):.3f}  "
          f"CALL={np.mean(all_strategies[:, 1]):.3f}  "
          f"RAISE={np.mean(all_strategies[:, 2]):.3f}")

    # How uniform are strategies? Measure avg entropy
    entropies = []
    for s in all_strategies:
        s_safe = np.maximum(s, 1e-10)
        entropies.append(-np.sum(s_safe * np.log(s_safe)))
    print(f"  Avg strategy entropy: {np.mean(entropies):.4f} (max = {np.log(3):.4f})")
    print(f"  Min entropy: {np.min(entropies):.4f} (most deterministic)")
    print(f"  Max entropy: {np.max(entropies):.4f} (most mixed)")

    # How many strategies are nearly uniform (entropy > 0.95 * max)?
    max_entropy = np.log(3)
    nearly_uniform = sum(1 for e in entropies if e > 0.95 * max_entropy)
    print(f"  Nearly uniform (entropy > 95% max): {nearly_uniform}/{len(entropies)}")

    return store


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: Correct Random Baselines (Analytical)
# ═══════════════════════════════════════════════════════════════════════

def compute_random_baselines():
    """
    Compute the correct random baselines for avg P(true hand) and argmax accuracy.

    Enumerates all possible deals and computes expected metric values
    when beliefs are initialized from card removal but never updated.
    """
    print("\n" + "=" * 80)
    print("  ANALYSIS 2: RANDOM BASELINES (ANALYTICAL)")
    print("=" * 80)

    hands = ['J', 'Q', 'K']
    boards = ['J', 'Q', 'K']

    # ── Preflop baseline ──
    print(f"\n  ── PREFLOP (no board, 1 card removed) ──")
    preflop_ptrue_values = []
    preflop_argmax_correct = []

    for my_hand in hands:
        prior = initialize_belief(my_hand, board=None)
        my_idx = CARD_MAP[my_hand]

        # Opponent hand distribution (from the remaining cards)
        remaining = list(CARD_COUNTS)
        remaining[my_idx] -= 1
        total_remaining = sum(remaining)

        print(f"  Holding {my_hand}: prior = [{prior[0]:.3f}, {prior[1]:.3f}, {prior[2]:.3f}]")

        for opp_idx in range(3):
            if remaining[opp_idx] == 0:
                continue
            opp_prob = remaining[opp_idx] / total_remaining  # P(opponent has this hand)
            ptrue = prior[opp_idx]  # P(true) from uniform belief
            argmax = int(np.argmax(prior))
            is_correct = 1 if argmax == opp_idx else 0

            # Weight by probability of this opponent hand
            preflop_ptrue_values.append((ptrue, opp_prob))
            preflop_argmax_correct.append((is_correct, opp_prob))

    preflop_avg_ptrue = sum(v * w for v, w in preflop_ptrue_values) / sum(w for _, w in preflop_ptrue_values)
    preflop_avg_argmax = sum(v * w for v, w in preflop_argmax_correct) / sum(w for _, w in preflop_argmax_correct)
    print(f"\n  Preflop baselines:")
    print(f"    avg P(true hand):  {preflop_avg_ptrue:.4f}")
    print(f"    argmax accuracy:   {preflop_avg_argmax:.4f}")

    # ── Flop baseline (2 cards removed) ──
    print(f"\n  ── FLOP (board dealt, 2 cards removed) ──")
    flop_ptrue_values = []
    flop_argmax_correct = []

    for my_hand in hands:
        my_idx = CARD_MAP[my_hand]
        for board in boards:
            board_idx = CARD_MAP[board]
            # Check if this deal is possible
            remaining_after_mine = list(CARD_COUNTS)
            remaining_after_mine[my_idx] -= 1
            if remaining_after_mine[board_idx] <= 0:
                continue  # Can't deal this board
            remaining_after_mine[board_idx] -= 1

            prior = initialize_belief(my_hand, board)
            total_remaining = sum(remaining_after_mine)

            if total_remaining == 0:
                continue

            # Weight of this deal configuration
            # P(I get my_hand) * P(board = board | my hand removed)
            deal_weight = CARD_COUNTS[my_idx] * (CARD_COUNTS[board_idx] - (1 if board_idx == my_idx else 0))

            for opp_idx in range(3):
                if remaining_after_mine[opp_idx] == 0:
                    continue
                opp_prob = remaining_after_mine[opp_idx] / total_remaining
                ptrue = prior[opp_idx]
                argmax = int(np.argmax(prior))
                is_correct = 1 if argmax == opp_idx else 0

                flop_ptrue_values.append((ptrue, opp_prob * deal_weight))
                flop_argmax_correct.append((is_correct, opp_prob * deal_weight))

            print(f"  {my_hand} + board {board}: prior = [{prior[0]:.3f}, {prior[1]:.3f}, {prior[2]:.3f}]"
                  f"  (remaining: {remaining_after_mine})")

    flop_avg_ptrue = sum(v * w for v, w in flop_ptrue_values) / sum(w for _, w in flop_ptrue_values)
    flop_avg_argmax = sum(v * w for v, w in flop_argmax_correct) / sum(w for _, w in flop_argmax_correct)
    print(f"\n  Flop baselines:")
    print(f"    avg P(true hand):  {flop_avg_ptrue:.4f}")
    print(f"    argmax accuracy:   {flop_avg_argmax:.4f}")

    # Overall weighted average (assuming ~50/50 preflop/flop decisions)
    # Actually, compute exact ratio: preflop has 1 decision point per hand, flop has 0-2
    # For simplicity, report both
    overall_ptrue = (preflop_avg_ptrue + flop_avg_ptrue) / 2
    overall_argmax = (preflop_avg_argmax + flop_avg_argmax) / 2
    print(f"\n  ── SUMMARY (equal-weight avg of preflop + flop) ──")
    print(f"    avg P(true hand) baseline: {overall_ptrue:.4f}")
    print(f"    argmax accuracy baseline:  {overall_argmax:.4f}")

    return {
        "preflop_ptrue": preflop_avg_ptrue,
        "preflop_argmax": preflop_avg_argmax,
        "flop_ptrue": flop_avg_ptrue,
        "flop_argmax": flop_avg_argmax,
        "overall_ptrue": overall_ptrue,
        "overall_argmax": overall_argmax,
    }


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: Standardized Belief Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_belief_standardized(agent, agent_name, num_games=1000, use_session=False):
    """
    Evaluate belief quality using BOTH metrics:
      1. avg P(true hand) — average probability mass on correct hand
      2. argmax accuracy — does the most-probable hand match truth?

    Also groups results by true opponent hand and by round.
    """
    if use_session:
        session = PokerSession()
    else:
        game = LeducGame()

    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    # Collectors
    true_hand_probs = []        # P(true hand) at each decision point
    argmax_correct = []         # 1 if argmax matches truth, 0 otherwise
    belief_vectors_by_hand = {  # Grouped by TRUE opponent hand
        'J': [], 'Q': [], 'K': [],
    }
    results_by_round = {        # Grouped by game round
        0: {'ptrue': [], 'argmax': []},
        1: {'ptrue': [], 'argmax': []},
    }
    priors_by_round = {         # Card removal priors only
        0: {'ptrue': [], 'argmax': []},
        1: {'ptrue': [], 'argmax': []},
    }

    for game_idx in range(num_games):
        if use_session:
            if game_idx % 30 == 0:
                session.reset()
            session.new_hand()
            g = session.game
        else:
            game = LeducGame()
            game.reset()
            g = game

        while not g.is_finished:
            cp = g.current_player

            if use_session:
                obs = session.get_observation(viewer_id=cp)
            else:
                obs = g.get_observation(viewer_id=cp)

            if cp == 0:
                # Our belief agent's turn
                belief = agent.compute_belief_from_history(obs)
                true_hand = g.player_hands[1]
                true_idx = CARD_MAP[true_hand]

                ptrue = belief[true_idx]
                is_argmax = 1 if np.argmax(belief) == true_idx else 0

                true_hand_probs.append(ptrue)
                argmax_correct.append(is_argmax)
                belief_vectors_by_hand[true_hand].append(belief.copy())

                # By round
                rnd = obs.current_round
                results_by_round[rnd]['ptrue'].append(ptrue)
                results_by_round[rnd]['argmax'].append(is_argmax)

                # Also compute card removal prior for comparison
                prior = initialize_belief(obs.player_hand, obs.board)
                prior_ptrue = prior[true_idx]
                prior_argmax = 1 if np.argmax(prior) == true_idx else 0
                priors_by_round[rnd]['ptrue'].append(prior_ptrue)
                priors_by_round[rnd]['argmax'].append(prior_argmax)

                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            if use_session:
                session.step(action)
            else:
                g.step(action)

    # ── Results ──
    avg_ptrue = np.mean(true_hand_probs) if true_hand_probs else 0
    avg_argmax = np.mean(argmax_correct) if argmax_correct else 0
    n = len(true_hand_probs)

    result = {
        "agent": agent_name,
        "total_decisions": n,
        "avg_ptrue": round(float(avg_ptrue), 4),
        "argmax_accuracy": round(float(avg_argmax), 4),
    }

    # Per-round breakdown
    for rnd in [0, 1]:
        if results_by_round[rnd]['ptrue']:
            result[f"round_{rnd}_ptrue"] = round(float(np.mean(results_by_round[rnd]['ptrue'])), 4)
            result[f"round_{rnd}_argmax"] = round(float(np.mean(results_by_round[rnd]['argmax'])), 4)
            result[f"round_{rnd}_n"] = len(results_by_round[rnd]['ptrue'])
            result[f"round_{rnd}_prior_ptrue"] = round(float(np.mean(priors_by_round[rnd]['ptrue'])), 4)
            result[f"round_{rnd}_prior_argmax"] = round(float(np.mean(priors_by_round[rnd]['argmax'])), 4)

    # Belief distributions grouped by true opponent hand
    belief_by_hand = {}
    for hand_name in ['J', 'Q', 'K']:
        beliefs = belief_vectors_by_hand[hand_name]
        if beliefs:
            beliefs_arr = np.array(beliefs)
            hand_idx = CARD_MAP[hand_name]
            belief_by_hand[hand_name] = {
                "n": len(beliefs),
                "avg_belief": [round(float(x), 4) for x in np.mean(beliefs_arr, axis=0)],
                "avg_ptrue": round(float(np.mean(beliefs_arr[:, hand_idx])), 4),
                "std_ptrue": round(float(np.std(beliefs_arr[:, hand_idx])), 4),
                "min_ptrue": round(float(np.min(beliefs_arr[:, hand_idx])), 4),
                "max_ptrue": round(float(np.max(beliefs_arr[:, hand_idx])), 4),
                "argmax_rate": round(float(np.mean(np.argmax(beliefs_arr, axis=1) == hand_idx)), 4),
            }
    result["belief_by_true_hand"] = belief_by_hand

    return result


def run_standardized_evaluation():
    """Run standardized belief evaluation on all 5 agents."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 3: STANDARDIZED BELIEF EVALUATION")
    print("=" * 80)

    agents = []

    # E1a: BeliefCfrAgent (self-play trained)
    try:
        e1a = BeliefCfrAgent(model_path="models/belief_cfr_agent.pt", cfr_path=CFR_MODEL_PATH)
        agents.append(("E1a belief_cfr", e1a, False))
    except Exception as e:
        print(f"  Failed to load E1a: {e}")

    # Ablation: BeliefCfrAgent (pop-trained, no modulation)
    try:
        ablation = BeliefCfrAgent(model_path="models/belief_cfr_pop_ablation_agent.pt", cfr_path=CFR_MODEL_PATH)
        agents.append(("Ablation pop_no_mod", ablation, False))
    except Exception as e:
        print(f"  Failed to load Ablation: {e}")

    # E1b: BeliefModulatedAgent (pop-trained, with modulation)
    try:
        e1b = BeliefModulatedAgent(model_path="models/belief_modulated_agent.pt",
                                   cfr_model_path=CFR_MODEL_PATH)
        agents.append(("E1b belief_modulated", e1b, True))
    except Exception as e:
        print(f"  Failed to load E1b: {e}")

    # E2c: BeliefConfidentAgent
    try:
        e2c = BeliefConfidentAgent(model_path="models/belief_confident_agent.pt")
        agents.append(("E2c belief_confident", e2c, False))
    except Exception as e:
        print(f"  Failed to load E2c: {e}")

    # E2d: BeliefStableAgent
    try:
        e2d = BeliefStableAgent(model_path="models/belief_stable_agent.pt")
        agents.append(("E2d belief_stable", e2d, False))
    except Exception as e:
        print(f"  Failed to load E2d: {e}")

    results = []
    for name, agent, use_session in agents:
        print(f"\n  Evaluating {name}...")
        r = evaluate_belief_standardized(agent, name, num_games=1000, use_session=use_session)
        results.append(r)

    # ── Comparison table ──
    print(f"\n\n  ── STANDARDIZED COMPARISON TABLE ──")
    print(f"  {'Agent':<30} {'avg P(true)':>12} {'argmax acc':>12} {'N decisions':>12}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12}")
    for r in results:
        print(f"  {r['agent']:<30} {r['avg_ptrue']:>12.4f} {r['argmax_accuracy']:>12.4f} {r['total_decisions']:>12}")

    # ── Per-round breakdown ──
    print(f"\n  ── PER-ROUND BREAKDOWN ──")
    print(f"  {'Agent':<30} {'R0 P(true)':>10} {'R0 argmax':>10} {'R1 P(true)':>10} {'R1 argmax':>10}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in results:
        r0pt = r.get('round_0_ptrue', '-')
        r0am = r.get('round_0_argmax', '-')
        r1pt = r.get('round_1_ptrue', '-')
        r1am = r.get('round_1_argmax', '-')
        r0pt_s = f"{r0pt:.4f}" if isinstance(r0pt, float) else r0pt
        r0am_s = f"{r0am:.4f}" if isinstance(r0am, float) else r0am
        r1pt_s = f"{r1pt:.4f}" if isinstance(r1pt, float) else r1pt
        r1am_s = f"{r1am:.4f}" if isinstance(r1am, float) else r1am
        print(f"  {r['agent']:<30} {r0pt_s:>10} {r0am_s:>10} {r1pt_s:>10} {r1am_s:>10}")

    # ── Card removal prior baselines per round ──
    print(f"\n  ── CARD REMOVAL PRIOR BASELINES (measured from actual deals) ──")
    print(f"  {'Agent':<30} {'R0 prior':>10} {'R0 prior_am':>12} {'R1 prior':>10} {'R1 prior_am':>12}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 12}")
    for r in results:
        r0pp = r.get('round_0_prior_ptrue', '-')
        r0pa = r.get('round_0_prior_argmax', '-')
        r1pp = r.get('round_1_prior_ptrue', '-')
        r1pa = r.get('round_1_prior_argmax', '-')
        r0pp_s = f"{r0pp:.4f}" if isinstance(r0pp, float) else r0pp
        r0pa_s = f"{r0pa:.4f}" if isinstance(r0pa, float) else r0pa
        r1pp_s = f"{r1pp:.4f}" if isinstance(r1pp, float) else r1pp
        r1pa_s = f"{r1pa:.4f}" if isinstance(r1pa, float) else r1pa
        print(f"  {r['agent']:<30} {r0pp_s:>10} {r0pa_s:>12} {r1pp_s:>10} {r1pa_s:>12}")

    # ── Belief distributions by true hand ──
    print(f"\n  ── BELIEF DISTRIBUTIONS BY TRUE OPPONENT HAND ──")
    for r in results:
        print(f"\n  {r['agent']}:")
        if 'belief_by_true_hand' in r:
            print(f"    {'True Hand':<12} {'N':>6} {'Avg Belief':>25} {'P(true)':>10} {'argmax%':>10}")
            print(f"    {'-' * 12} {'-' * 6} {'-' * 25} {'-' * 10} {'-' * 10}")
            for hand in ['J', 'Q', 'K']:
                if hand in r['belief_by_true_hand']:
                    d = r['belief_by_true_hand'][hand]
                    b = d['avg_belief']
                    print(f"    {hand:<12} {d['n']:>6} [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}]"
                          f"   {d['avg_ptrue']:>10.4f} {d['argmax_rate']:>9.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 4: Likelihood Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_likelihood(num_games=1000):
    """
    Evaluate likelihood quality for CFR Nash and modulated Nash.

    Metrics:
      - avg P(true action): probability mass assigned to the action opponent actually chose
      - top-1 accuracy: does argmax of likelihood match the actual action?
      - Per-action breakdown
    """
    print("\n" + "=" * 80)
    print("  ANALYSIS 4: LIKELIHOOD EVALUATION")
    print("=" * 80)

    # Load agents
    e1a = BeliefCfrAgent(model_path="models/belief_cfr_agent.pt", cfr_path=CFR_MODEL_PATH)
    e1b = BeliefModulatedAgent(model_path="models/belief_modulated_agent.pt",
                               cfr_model_path=CFR_MODEL_PATH)
    e1a.set_train_mode(False)
    e1b.set_train_mode(False)

    game = LeducGame()
    heuristic = HeuristicAgent()

    # Collectors
    nash_ptrue_action = []
    nash_argmax_correct = []
    mod_ptrue_action = []
    mod_argmax_correct = []
    per_action_nash = {0: [], 1: [], 2: []}
    per_action_mod = {0: [], 1: [], 2: []}

    session = PokerSession()

    for game_idx in range(num_games):
        if game_idx % 30 == 0:
            session.reset()
        session.new_hand()
        g = session.game

        while not g.is_finished:
            cp = g.current_player
            obs = session.get_observation(viewer_id=cp)

            if cp == 1:
                # Opponent (heuristic) acts — this is what we're trying to predict
                actual_action = heuristic.select_action(obs)
                actual_idx = int(actual_action)
                true_hand = g.player_hands[1]

                # Get observation from player 0's perspective for likelihood query
                obs_p0 = session.get_observation(viewer_id=0)

                # Nash likelihood
                try:
                    all_actions = [Action.FOLD, Action.CALL, Action.RAISE]
                    key = e1a._build_cfr_key_for_opponent(true_hand, obs_p0)
                    nash_strategy = e1a.strategy_store.get_average_strategy(key, all_actions)
                    nash_p = nash_strategy[actual_idx]
                    nash_pred = int(np.argmax(nash_strategy))

                    nash_ptrue_action.append(nash_p)
                    nash_argmax_correct.append(1 if nash_pred == actual_idx else 0)
                    per_action_nash[actual_idx].append(nash_p)
                except Exception:
                    pass

                # Modulated likelihood
                try:
                    opp_stats = e1b._encode_opp_stats(obs_p0)
                    mod_log_probs = e1b.get_adjusted_log_probs(true_hand, obs_p0, opp_stats)
                    mod_probs = torch.exp(mod_log_probs).detach().numpy()
                    mod_p = mod_probs[actual_idx]
                    mod_pred = int(np.argmax(mod_probs))

                    mod_ptrue_action.append(float(mod_p))
                    mod_argmax_correct.append(1 if mod_pred == actual_idx else 0)
                    per_action_mod[actual_idx].append(float(mod_p))
                except Exception:
                    pass

                session.step(actual_action)
            else:
                action = e1a.select_action(obs)
                session.step(action)

    # Print results
    action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}

    print(f"\n  ── NASH LIKELIHOOD ──")
    print(f"  Total actions: {len(nash_ptrue_action)}")
    if nash_ptrue_action:
        print(f"  avg P(true action):  {np.mean(nash_ptrue_action):.4f}")
        print(f"  top-1 accuracy:      {np.mean(nash_argmax_correct):.4f}")
        print(f"\n  Per-action avg P(true action):")
        for a_idx in [0, 1, 2]:
            if per_action_nash[a_idx]:
                print(f"    {action_names[a_idx]:>6}: {np.mean(per_action_nash[a_idx]):.4f} (n={len(per_action_nash[a_idx])})")

    print(f"\n  ── MODULATED LIKELIHOOD ──")
    print(f"  Total actions: {len(mod_ptrue_action)}")
    if mod_ptrue_action:
        print(f"  avg P(true action):  {np.mean(mod_ptrue_action):.4f}")
        print(f"  top-1 accuracy:      {np.mean(mod_argmax_correct):.4f}")
        print(f"\n  Per-action avg P(true action):")
        for a_idx in [0, 1, 2]:
            if per_action_mod[a_idx]:
                print(f"    {action_names[a_idx]:>6}: {np.mean(per_action_mod[a_idx]):.4f} (n={len(per_action_mod[a_idx])})")

    if nash_ptrue_action and mod_ptrue_action:
        print(f"\n  ── COMPARISON ──")
        print(f"  avg P(true action):  Nash={np.mean(nash_ptrue_action):.4f} vs Mod={np.mean(mod_ptrue_action):.4f}"
              f"  (delta: {np.mean(mod_ptrue_action) - np.mean(nash_ptrue_action):+.4f})")
        print(f"  top-1 accuracy:      Nash={np.mean(nash_argmax_correct):.4f} vs Mod={np.mean(mod_argmax_correct):.4f}"
              f"  (delta: {np.mean(mod_argmax_correct) - np.mean(nash_argmax_correct):+.4f})")


def _build_cfr_key_for_opponent(self, opp_hand, obs_p0):
    """Build CFR key from the OPPONENT's perspective."""
    from src.agents.belief_common import build_cfr_infoset_key
    action_history = obs_p0.action_history if obs_p0.action_history else []
    return build_cfr_infoset_key(opp_hand, obs_p0.board or "", obs_p0.current_round, action_history)

# Monkey-patch for the analysis
BeliefCfrAgent._build_cfr_key_for_opponent = _build_cfr_key_for_opponent


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS 5: Modulation Gate Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_modulation_gates():
    """
    Analyze the modulation gate and delta networks.

    Shows:
      - Gate activation for various opponent stat profiles
      - Delta magnitudes for each action
      - How modulation changes with opponent behavior
    """
    print("\n" + "=" * 80)
    print("  ANALYSIS 5: MODULATION GATE ANALYSIS")
    print("=" * 80)

    e1b = BeliefModulatedAgent(model_path="models/belief_modulated_agent.pt",
                               cfr_model_path=CFR_MODEL_PATH)
    e1b.set_train_mode(False)

    # Test with various opponent profiles
    profiles = {
        "Uninformative (1/3, 1/3, 1/3, 0.0)": [1/3, 1/3, 1/3, 0.0],
        "Aggressive (0.1, 0.2, 0.7, 0.78)": [0.1, 0.2, 0.7, 0.78],
        "Passive (0.1, 0.7, 0.2, 0.22)": [0.1, 0.7, 0.2, 0.22],
        "Tight/foldy (0.6, 0.2, 0.2, 0.5)": [0.6, 0.2, 0.2, 0.5],
        "Calling station (0.0, 0.8, 0.2, 0.2)": [0.0, 0.8, 0.2, 0.2],
        "Balanced (0.3, 0.35, 0.35, 0.5)": [0.3, 0.35, 0.35, 0.5],
    }

    print(f"\n  ── GATE & DELTA FOR OPPONENT PROFILES ──")
    print(f"  {'Profile':<45} {'Gate':>6} {'dF':>8} {'dC':>8} {'dR':>8}")
    print(f"  {'-' * 45} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8}")

    with torch.no_grad():
        for name, stats in profiles.items():
            stats_t = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)
            gate = e1b.gate_net(stats_t).item()
            delta = e1b.delta_net(stats_t).squeeze().numpy()
            print(f"  {name:<45} {gate:>5.3f} {delta[0]:>+7.4f} {delta[1]:>+7.4f} {delta[2]:>+7.4f}")

    # Show effective adjustment (gate * delta)
    print(f"\n  ── EFFECTIVE ADJUSTMENT (gate × delta) ──")
    print(f"  {'Profile':<45} {'adj_F':>8} {'adj_C':>8} {'adj_R':>8}")
    print(f"  {'-' * 45} {'-' * 8} {'-' * 8} {'-' * 8}")

    with torch.no_grad():
        for name, stats in profiles.items():
            stats_t = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)
            gate = e1b.gate_net(stats_t).item()
            delta = e1b.delta_net(stats_t).squeeze().numpy()
            adj = gate * delta
            print(f"  {name:<45} {adj[0]:>+7.4f} {adj[1]:>+7.4f} {adj[2]:>+7.4f}")

    # Show how Nash probs change with modulation for key infosets
    print(f"\n  ── MODULATION EFFECT ON KEY INFOSETS ──")
    hands = ['J', 'Q', 'K']
    test_profile = [0.1, 0.2, 0.7, 0.78]  # aggressive opponent
    stats_t = torch.tensor(test_profile, dtype=torch.float32)

    # Create dummy observations for preflop
    game = LeducGame()

    print(f"\n  Opponent profile: Aggressive (0.1, 0.2, 0.7, 0.78)")
    print(f"  {'Infoset':<20} {'Nash':>25} {'Modulated':>25}")

    for hand in hands:
        game.reset()
        obs = game.get_observation(viewer_id=0)
        # Temporarily pretend the opponent has this hand
        nash_lp = e1b.get_nash_log_probs(hand, obs)
        nash_p = torch.exp(nash_lp).numpy()

        mod_lp = e1b.get_adjusted_log_probs(hand, obs, stats_t)
        mod_p = torch.exp(mod_lp).detach().numpy()

        print(f"  {hand + ' preflop':<20} "
              f"[F={nash_p[0]:.3f} C={nash_p[1]:.3f} R={nash_p[2]:.3f}]  "
              f"[F={mod_p[0]:.3f} C={mod_p[1]:.3f} R={mod_p[2]:.3f}]")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + "  ROUND 5 METRIC ANALYSIS — COMPREHENSIVE EVALUATION".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # 1. CFR strategy table
    store = dump_cfr_strategy()

    # 2. Random baselines
    baselines = compute_random_baselines()

    # 3. Standardized belief evaluation (both metrics for all agents)
    belief_results = run_standardized_evaluation()

    # 4. Likelihood evaluation
    try:
        evaluate_likelihood(num_games=1000)
    except Exception as e:
        print(f"\n  Likelihood evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Gate analysis
    try:
        analyze_modulation_gates()
    except Exception as e:
        print(f"\n  Gate analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)

    # Save results
    output = {
        "baselines": baselines,
        "belief_results": belief_results,
    }
    with open("experiments/round5_metric_analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to experiments/round5_metric_analysis_results.json")


if __name__ == "__main__":
    main()
