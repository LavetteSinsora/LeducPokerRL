"""
Training Target & Normalization Analysis Report
================================================
Investigates whether the ~40 MSE training loss is an irreducible variance floor
from Monte Carlo targets, and evaluates normalization/loss alternatives.

Run: python experiments/training_target_analysis.py
"""

import sys
import os
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.value_based import ValueBasedAgent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DecisionRecord:
    """One decision point from a self-play episode."""
    state_key: tuple          # (private_card, board, round, current_player, pot_tuple)
    encoded_state: torch.Tensor
    player_id: int
    reward: float             # terminal reward for this player
    episode_id: int
    step_index: int           # position within episode (0 = first decision)
    game_ending: str          # 'preflop_fold', 'flop_fold', 'showdown'
    # For TD(0) analysis
    next_encoded_state: Optional[torch.Tensor]
    next_player: Optional[int]
    is_terminal_step: bool


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def collect_data(agent: ValueBasedAgent, num_episodes: int = 50000) -> List[DecisionRecord]:
    """Play self-play episodes and record every decision point."""
    game = LeducGame()
    agent.set_train_mode(True)  # Boltzmann exploration for diversity
    records = []

    for ep in range(num_episodes):
        game.reset()
        episode_steps = []  # collect steps, then assign rewards after episode

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)

            # State key
            pot_tuple = tuple(obs.pot)
            state_key = (obs.player_hand, obs.board, obs.current_round, cp, pot_tuple)

            # Encoded state
            encoded = agent.encode_observation(obs)

            # Select action (train mode returns (action, encoded_state))
            action, _ = agent.select_action(obs)

            episode_steps.append({
                'state_key': state_key,
                'encoded_state': encoded,
                'player_id': cp,
                'step_index': len(episode_steps),
            })

            game.step(action)

        # Determine game ending type
        rewards = game.get_reward()
        last_action = game.history[-1][1] if game.history else 'CALL'
        if last_action == 'FOLD':
            if game.current_round == 0 or (game.board is None):
                game_ending = 'preflop_fold'
            else:
                game_ending = 'flop_fold'
        else:
            game_ending = 'showdown'

        # Build next-state links for TD(0) analysis
        for i, step in enumerate(episode_steps):
            is_terminal = (i == len(episode_steps) - 1)
            if not is_terminal:
                next_step = episode_steps[i + 1]
                next_encoded = next_step['encoded_state']
                next_player = next_step['player_id']
            else:
                next_encoded = None
                next_player = None

            records.append(DecisionRecord(
                state_key=step['state_key'],
                encoded_state=step['encoded_state'],
                player_id=step['player_id'],
                reward=rewards[step['player_id']],
                episode_id=ep,
                step_index=step['step_index'],
                game_ending=game_ending,
                next_encoded_state=next_encoded,
                next_player=next_player,
                is_terminal_step=is_terminal,
            ))

        if (ep + 1) % 10000 == 0:
            print(f"  Collected {ep + 1}/{num_episodes} episodes...")

    agent.set_train_mode(False)
    return records


# ---------------------------------------------------------------------------
# Section A: Reward Distribution Analysis
# ---------------------------------------------------------------------------

def section_a_reward_distribution(records: List[DecisionRecord]):
    print("\n" + "=" * 70)
    print("SECTION A: REWARD DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Get one reward per episode (use first decision point per episode)
    episode_rewards = {}
    episode_endings = {}
    for r in records:
        if r.episode_id not in episode_rewards:
            episode_rewards[r.episode_id] = r.reward
            episode_endings[r.episode_id] = r.game_ending

    all_rewards = list(episode_rewards.values())
    all_endings = list(episode_endings.values())

    print(f"\nTotal episodes: {len(all_rewards)}")
    print(f"Mean reward:    {np.mean(all_rewards):+.3f}")
    print(f"Std reward:     {np.std(all_rewards):.3f}")
    print(f"Min reward:     {np.min(all_rewards):+.0f}")
    print(f"Max reward:     {np.max(all_rewards):+.0f}")
    print(f"Variance:       {np.var(all_rewards):.3f}")

    # Histogram
    from collections import Counter
    reward_counts = Counter(int(r) for r in all_rewards)
    print("\nReward Histogram (from player-0 perspective of first decision):")
    print(f"{'Value':>6}  {'Count':>7}  {'Pct':>6}  Bar")
    for val in sorted(reward_counts.keys()):
        count = reward_counts[val]
        pct = 100.0 * count / len(all_rewards)
        bar = '#' * int(pct * 2)
        print(f"{val:>+6d}  {count:>7d}  {pct:>5.1f}%  {bar}")

    # Breakdown by game ending
    print("\nBreakdown by game-ending type:")
    ending_rewards = defaultdict(list)
    for eid in episode_rewards:
        ending_rewards[episode_endings[eid]].append(episode_rewards[eid])

    for ending in ['preflop_fold', 'flop_fold', 'showdown']:
        rews = ending_rewards.get(ending, [])
        if rews:
            print(f"  {ending:15s}: n={len(rews):>6d}, mean={np.mean(rews):+.2f}, "
                  f"std={np.std(rews):.2f}, range=[{np.min(rews):+.0f}, {np.max(rews):+.0f}]")
        else:
            print(f"  {ending:15s}: n=0")


# ---------------------------------------------------------------------------
# Section B: Per-State Variance Analysis
# ---------------------------------------------------------------------------

def section_b_per_state_variance(records: List[DecisionRecord]):
    print("\n" + "=" * 70)
    print("SECTION B: PER-STATE VARIANCE ANALYSIS (CORE EXPERIMENT)")
    print("=" * 70)

    # Group by state key
    state_groups = defaultdict(list)
    for r in records:
        state_groups[r.state_key].append(r.reward)

    # Compute per-state stats
    state_stats = []
    for key, rewards in state_groups.items():
        rewards_arr = np.array(rewards)
        state_stats.append({
            'key': key,
            'count': len(rewards),
            'mean': np.mean(rewards_arr),
            'std': np.std(rewards_arr),
            'var': np.var(rewards_arr),
            'min': np.min(rewards_arr),
            'max': np.max(rewards_arr),
            'private_card': key[0],
            'board': key[1],
            'round': key[2],
            'current_player': key[3],
            'pot': key[4],
        })

    # Weighted average variance
    total_count = sum(s['count'] for s in state_stats)
    weighted_var = sum(s['var'] * s['count'] for s in state_stats) / total_count

    print(f"\nTotal unique state keys: {len(state_stats)}")
    print(f"Total decision records:  {total_count}")
    print(f"\n*** WEIGHTED AVERAGE PER-STATE VARIANCE: {weighted_var:.2f} ***")
    print(f"*** Observed training loss:              ~40 ***")

    if abs(weighted_var - 40) < 15:
        print(f"*** CONCLUSION: The loss IS approximately the irreducible MC variance! ***")
    elif weighted_var < 25:
        print(f"*** CONCLUSION: Variance ({weighted_var:.1f}) is well below 40 — convergence issue likely. ***")
    else:
        print(f"*** CONCLUSION: Variance ({weighted_var:.1f}) — partial match, investigate further. ***")

    # Top-20 highest-variance states
    state_stats.sort(key=lambda s: s['var'], reverse=True)
    print(f"\nTop 20 highest-variance state keys:")
    print(f"{'Card':>5} {'Board':>5} {'Rnd':>3} {'CP':>3} {'Pot':>12} {'n':>6} "
          f"{'Mean':>7} {'Std':>6} {'Var':>8} {'Range':>12}")
    for s in state_stats[:20]:
        print(f"{s['private_card']:>5} {str(s['board']):>5} {s['round']:>3d} "
              f"{s['current_player']:>3d} {str(s['pot']):>12} {s['count']:>6d} "
              f"{s['mean']:>+7.2f} {s['std']:>6.2f} {s['var']:>8.2f} "
              f"[{s['min']:+.0f},{s['max']:+.0f}]")

    # Breakdown by round
    print(f"\nVariance breakdown by round:")
    for rnd in [0, 1]:
        rnd_stats = [s for s in state_stats if s['round'] == rnd]
        if rnd_stats:
            rnd_count = sum(s['count'] for s in rnd_stats)
            rnd_weighted_var = sum(s['var'] * s['count'] for s in rnd_stats) / rnd_count
            print(f"  Round {rnd}: {len(rnd_stats):>4d} unique states, "
                  f"weighted var = {rnd_weighted_var:.2f}, total samples = {rnd_count}")

    # Breakdown by private card at round 0
    print(f"\nVariance breakdown by private card (round 0 only):")
    for card in ['J', 'Q', 'K']:
        card_stats = [s for s in state_stats if s['private_card'] == card and s['round'] == 0]
        if card_stats:
            card_count = sum(s['count'] for s in card_stats)
            card_weighted_var = sum(s['var'] * s['count'] for s in card_stats) / card_count
            print(f"  {card}: {len(card_stats):>3d} unique states, "
                  f"weighted var = {card_weighted_var:.2f}, total samples = {card_count}")

    return weighted_var


# ---------------------------------------------------------------------------
# Section C: Value Network Output Analysis
# ---------------------------------------------------------------------------

def section_c_value_network_analysis(agent: ValueBasedAgent):
    print("\n" + "=" * 70)
    print("SECTION C: VALUE NETWORK OUTPUT ANALYSIS")
    print("=" * 70)

    cards = ['J', 'Q', 'K']
    boards = [None, 'J', 'Q', 'K']
    players = [0, 1]

    # Generate representative pot configurations
    pot_configs = [
        [1, 1],   # ante only
        [3, 3],   # one raise pre-flop, called
        [5, 5],   # two raises pre-flop, called
        [5, 9],   # flop with raise
        [9, 9],   # flop with raise called
    ]

    agent.set_train_mode(False)

    print("\n--- Hand Strength Ordering Check (Round 0, pot=[1,1]) ---")
    print("If network learned poker: V(K) > V(Q) > V(J) for same position\n")
    for cp in players:
        vals = {}
        for card in cards:
            obs = Observation(
                player_hand=card,
                board=None,
                pot=[1, 1],
                current_player=cp,
                current_round=0,
                legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                is_finished=False,
            )
            encoded = agent.encode_observation(obs)
            with torch.no_grad():
                v = agent.model(encoded).item()
            vals[card] = v
        ordering = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        ordering_str = " > ".join(f"{c}({v:+.2f})" for c, v in ordering)
        correct = vals['K'] > vals['Q'] > vals['J']
        print(f"  Player {cp}: {ordering_str}  {'CORRECT' if correct else 'WRONG'}")

    print("\n--- Pair vs No-Pair Check (Round 1) ---")
    print("If network learned poker: V(pair) > V(high card no pair)\n")
    for cp in players:
        for board in ['J', 'Q', 'K']:
            pair_card = board
            # Pick a non-pair card that's lower
            non_pair_cards = [c for c in cards if c != board]
            results = []
            for card in cards:
                obs = Observation(
                    player_hand=card,
                    board=board,
                    pot=[3, 3],
                    current_player=cp,
                    current_round=1,
                    legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                    is_finished=False,
                )
                encoded = agent.encode_observation(obs)
                with torch.no_grad():
                    v = agent.model(encoded).item()
                is_pair = (card == board)
                results.append((card, v, is_pair))
            results.sort(key=lambda x: x[1], reverse=True)
            fmt = ", ".join(f"{c}({'PAIR' if p else 'hi'})={v:+.2f}" for c, v, p in results)
            # Check: pair should be highest
            pair_val = [v for c, v, p in results if p]
            non_pair_vals = [v for c, v, p in results if not p]
            correct = all(pv > npv for pv in pair_val for npv in non_pair_vals) if pair_val else False
            print(f"  CP={cp}, Board={board}: {fmt}  {'CORRECT' if correct else 'WRONG'}")

    print("\n--- Player Symmetry Check ---")
    print("V(s, cp=0) should relate consistently to V(s, cp=1)\n")
    diffs = []
    for card in cards:
        for board in boards:
            rnd = 0 if board is None else 1
            for pot in [[1, 1], [3, 3]]:
                vals = {}
                for cp in [0, 1]:
                    obs = Observation(
                        player_hand=card,
                        board=board,
                        pot=pot,
                        current_player=cp,
                        current_round=rnd,
                        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                        is_finished=False,
                    )
                    encoded = agent.encode_observation(obs)
                    with torch.no_grad():
                        v = agent.model(encoded).item()
                    vals[cp] = v
                diff = vals[0] - vals[1]
                diffs.append(diff)
                print(f"  {card}, Board={board}, Pot={pot}: "
                      f"V(cp=0)={vals[0]:+.2f}, V(cp=1)={vals[1]:+.2f}, diff={diff:+.2f}")

    print(f"\n  Symmetry diff stats: mean={np.mean(diffs):+.3f}, std={np.std(diffs):.3f}")
    print(f"  (Ideal: near zero mean if game is symmetric)")

    # Full enumeration summary
    print("\n--- Full Prediction Range ---")
    all_vals = []
    for card in cards:
        for board in boards:
            rnd = 0 if board is None else 1
            for pot in pot_configs:
                for cp in players:
                    obs = Observation(
                        player_hand=card,
                        board=board,
                        pot=pot,
                        current_player=cp,
                        current_round=rnd,
                        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                        is_finished=False,
                    )
                    encoded = agent.encode_observation(obs)
                    with torch.no_grad():
                        v = agent.model(encoded).item()
                    all_vals.append(v)

    print(f"  Predictions across {len(all_vals)} state encodings:")
    print(f"  Min: {np.min(all_vals):+.3f}, Max: {np.max(all_vals):+.3f}, "
          f"Mean: {np.mean(all_vals):+.3f}, Std: {np.std(all_vals):.3f}")
    print(f"  (Reward range is [-13, +13]. Predictions should span a meaningful range.)")


# ---------------------------------------------------------------------------
# Section D: MC vs TD(0) Target Variance Comparison
# ---------------------------------------------------------------------------

def section_d_td_comparison(records: List[DecisionRecord], agent: ValueBasedAgent):
    print("\n" + "=" * 70)
    print("SECTION D: MC vs TD(0) TARGET VARIANCE COMPARISON")
    print("=" * 70)

    agent.set_train_mode(False)

    # Compute TD(0) targets
    td_targets = []
    mc_targets = []
    for r in records:
        mc_target = r.reward

        if r.is_terminal_step:
            # Terminal: TD target = actual reward (same as MC)
            td_target = mc_target
        else:
            # Non-terminal: TD target = V(s_{t+1}), negated if opponent's turn
            with torch.no_grad():
                v_next = agent.model(r.next_encoded_state).item()
            # If the next state's player is different, negate (zero-sum)
            if r.next_player != r.player_id:
                td_target = -v_next
            else:
                td_target = v_next

        td_targets.append(td_target)
        mc_targets.append(mc_target)

    # Group by state key for variance comparison
    state_mc = defaultdict(list)
    state_td = defaultdict(list)
    for i, r in enumerate(records):
        state_mc[r.state_key].append(mc_targets[i])
        state_td[r.state_key].append(td_targets[i])

    # Compute weighted average variance for each
    mc_vars = []
    td_vars = []
    counts = []
    for key in state_mc:
        mc_arr = np.array(state_mc[key])
        td_arr = np.array(state_td[key])
        n = len(mc_arr)
        mc_vars.append(np.var(mc_arr))
        td_vars.append(np.var(td_arr))
        counts.append(n)

    counts = np.array(counts)
    mc_vars = np.array(mc_vars)
    td_vars = np.array(td_vars)

    mc_weighted_var = np.sum(mc_vars * counts) / np.sum(counts)
    td_weighted_var = np.sum(td_vars * counts) / np.sum(counts)

    print(f"\nMC weighted avg per-state variance:    {mc_weighted_var:.2f}")
    print(f"TD(0) weighted avg per-state variance: {td_weighted_var:.2f}")
    if mc_weighted_var > 0:
        reduction = (1 - td_weighted_var / mc_weighted_var) * 100
        print(f"Variance reduction ratio:              {reduction:+.1f}%")
    print()

    # Breakdown by round
    print("Breakdown by round:")
    for rnd in [0, 1]:
        rnd_mc_vars = []
        rnd_td_vars = []
        rnd_counts = []
        for key in state_mc:
            if key[2] == rnd:  # key[2] is round
                mc_arr = np.array(state_mc[key])
                td_arr = np.array(state_td[key])
                rnd_mc_vars.append(np.var(mc_arr))
                rnd_td_vars.append(np.var(td_arr))
                rnd_counts.append(len(mc_arr))
        if rnd_counts:
            rnd_counts = np.array(rnd_counts)
            mc_wv = np.sum(np.array(rnd_mc_vars) * rnd_counts) / np.sum(rnd_counts)
            td_wv = np.sum(np.array(rnd_td_vars) * rnd_counts) / np.sum(rnd_counts)
            red = (1 - td_wv / mc_wv) * 100 if mc_wv > 0 else 0
            print(f"  Round {rnd}: MC var={mc_wv:.2f}, TD var={td_wv:.2f}, reduction={red:+.1f}%")

    # Breakdown by terminal vs non-terminal steps
    print("\nBreakdown by step type:")
    for is_term, label in [(True, 'Terminal'), (False, 'Non-terminal')]:
        sub_mc = defaultdict(list)
        sub_td = defaultdict(list)
        for i, r in enumerate(records):
            if r.is_terminal_step == is_term:
                sub_mc[r.state_key].append(mc_targets[i])
                sub_td[r.state_key].append(td_targets[i])
        if sub_mc:
            s_counts = []
            s_mc_vars = []
            s_td_vars = []
            for key in sub_mc:
                s_counts.append(len(sub_mc[key]))
                s_mc_vars.append(np.var(sub_mc[key]))
                s_td_vars.append(np.var(sub_td[key]))
            s_counts = np.array(s_counts)
            mc_wv = np.sum(np.array(s_mc_vars) * s_counts) / np.sum(s_counts)
            td_wv = np.sum(np.array(s_td_vars) * s_counts) / np.sum(s_counts)
            red = (1 - td_wv / mc_wv) * 100 if mc_wv > 0 else 0
            print(f"  {label:>12}: MC var={mc_wv:.2f}, TD var={td_wv:.2f}, reduction={red:+.1f}%")

    return mc_weighted_var, td_weighted_var


# ---------------------------------------------------------------------------
# Section E: Normalization Methods Analysis
# ---------------------------------------------------------------------------

def section_e_normalization(records: List[DecisionRecord], weighted_var: float):
    print("\n" + "=" * 70)
    print("SECTION E: NORMALIZATION METHODS ANALYSIS")
    print("=" * 70)

    all_rewards = np.array([r.reward for r in records])
    reward_mean = np.mean(all_rewards)
    reward_std = np.std(all_rewards)

    # Gather per-record pot totals
    pot_totals = np.array([sum(r.state_key[4]) for r in records])

    print(f"\nRaw reward stats: mean={reward_mean:+.3f}, std={reward_std:.3f}")
    print(f"Current MSE loss scale: ~{weighted_var:.1f}\n")

    # Method 1: Divide by max reward (r/13)
    max_reward = 13.0
    norm1 = all_rewards / max_reward
    norm1_var = np.var(norm1)
    # Per-state weighted variance under this normalization
    state_groups = defaultdict(list)
    for r in records:
        state_groups[r.state_key].append(r.reward / max_reward)
    n1_wvar = sum(np.var(v) * len(v) for v in state_groups.values()) / len(all_rewards)

    print(f"Method 1: Divide by max reward (r/{max_reward:.0f})")
    print(f"  Normalized range:         [{np.min(norm1):+.3f}, {np.max(norm1):+.3f}]")
    print(f"  Global variance:          {norm1_var:.4f}")
    print(f"  Weighted per-state var:   {n1_wvar:.4f}")
    print(f"  Expected MSE loss:        ~{n1_wvar:.2f}")
    print(f"  Scale factor:             1/{max_reward:.0f}^2 = {1/max_reward**2:.5f}")
    print(f"  Effect: Purely cosmetic — loss_new = loss_old * {1/max_reward**2:.5f}")
    print(f"          Network predictions would be in [-1, +1] range.")
    print()

    # Method 2: Per-pot normalization (r / player's pot contribution)
    state_groups2 = defaultdict(list)
    norm2_all = []
    for r in records:
        player_pot = r.state_key[4][r.player_id]
        if player_pot > 0:
            nr = r.reward / player_pot
        else:
            nr = r.reward
        norm2_all.append(nr)
        state_groups2[r.state_key].append(nr)
    norm2_all = np.array(norm2_all)
    n2_wvar = sum(np.var(v) * len(v) for v in state_groups2.values()) / len(norm2_all)

    print(f"Method 2: Per-pot normalization (r / total_pot)")
    # Also compute with total pot
    state_groups2b = defaultdict(list)
    norm2b_all = []
    for r in records:
        total_pot = sum(r.state_key[4])
        if total_pot > 0:
            nr = r.reward / total_pot
        else:
            nr = r.reward
        norm2b_all.append(nr)
        state_groups2b[r.state_key].append(nr)
    norm2b_all = np.array(norm2b_all)
    n2b_wvar = sum(np.var(v) * len(v) for v in state_groups2b.values()) / len(norm2b_all)

    print(f"  a) r / player_pot:  range=[{np.min(norm2_all):+.2f}, {np.max(norm2_all):+.2f}], "
          f"weighted var={n2_wvar:.4f}")
    print(f"  b) r / total_pot:   range=[{np.min(norm2b_all):+.2f}, {np.max(norm2b_all):+.2f}], "
          f"weighted var={n2b_wvar:.4f}")
    print(f"  Effect: Normalizes stakes so small-pot and big-pot games weigh equally.")
    print(f"          Within a state key, pot is constant, so this is per-state scaling.")
    print()

    # Method 3: Running mean/std normalization (PPO-style)
    norm3 = (all_rewards - reward_mean) / (reward_std + 1e-8)
    state_groups3 = defaultdict(list)
    for r in records:
        state_groups3[r.state_key].append((r.reward - reward_mean) / (reward_std + 1e-8))
    n3_wvar = sum(np.var(v) * len(v) for v in state_groups3.values()) / len(all_rewards)

    print(f"Method 3: Running mean/std normalization ((r - mu) / sigma)")
    print(f"  mu = {reward_mean:+.3f}, sigma = {reward_std:.3f}")
    print(f"  Normalized range:         [{np.min(norm3):+.3f}, {np.max(norm3):+.3f}]")
    print(f"  Global variance:          {np.var(norm3):.4f}")
    print(f"  Weighted per-state var:   {n3_wvar:.4f}")
    print(f"  Expected MSE loss:        ~{n3_wvar:.2f}")
    print(f"  Effect: Centers rewards at 0, unit variance. Standard in PPO.")
    print(f"          Requires tracking running stats. Loss becomes ~{n3_wvar:.2f}.")
    print()

    # Method 4: Huber loss analysis
    print(f"Method 4: Huber loss (delta=5) instead of MSE")
    # Simulate what Huber loss would look like
    # Huber(a) = 0.5*a^2 if |a|<=delta, delta*(|a|-0.5*delta) otherwise
    delta = 5.0
    # Compute what the 'loss' would be if prediction = per-state mean (optimal)
    huber_losses = []
    mse_losses = []
    for key, rewards in defaultdict(list, {r.state_key: [] for r in records}).items():
        pass
    # Recompute properly
    state_rewards = defaultdict(list)
    for r in records:
        state_rewards[r.state_key].append(r.reward)

    total_huber = 0.0
    total_mse = 0.0
    total_n = 0
    for key, rewards in state_rewards.items():
        mean_r = np.mean(rewards)
        for r in rewards:
            error = r - mean_r
            mse = error ** 2
            if abs(error) <= delta:
                huber = 0.5 * error ** 2
            else:
                huber = delta * (abs(error) - 0.5 * delta)
            total_huber += huber
            total_mse += mse
            total_n += 1

    avg_huber = total_huber / total_n
    avg_mse = total_mse / total_n

    print(f"  If predictions = per-state optimal (mean):")
    print(f"    MSE loss:   {avg_mse:.2f}")
    print(f"    Huber loss: {avg_huber:.2f}")
    print(f"    Reduction:  {(1 - avg_huber / avg_mse) * 100:.1f}%")
    print(f"  Effect: Clips gradient for |error| > {delta:.0f}.")
    print(f"          Reduces sensitivity to outlier episodes (big pots).")
    print(f"          Does NOT reduce variance, but reduces its impact on training.")

    # Distribution of errors from optimal
    errors = []
    for key, rewards in state_rewards.items():
        mean_r = np.mean(rewards)
        for r in rewards:
            errors.append(abs(r - mean_r))
    errors = np.array(errors)
    pct_above_delta = 100.0 * np.mean(errors > delta)
    print(f"  {pct_above_delta:.1f}% of samples have |error from state mean| > {delta:.0f}")
    print(f"  (These are the outliers Huber would clip.)")


# ---------------------------------------------------------------------------
# Summary & Recommendations
# ---------------------------------------------------------------------------

def print_summary(weighted_var: float, mc_var: float, td_var: float):
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print(f"""
Key Findings:
  1. Weighted per-state MC variance:  {weighted_var:.2f}
     Observed training loss:          ~40
     {'-> The loss IS the irreducible MC variance floor.' if abs(weighted_var - 40) < 15 else '-> The loss does NOT match — convergence issue likely.' if weighted_var < 25 else '-> Partial match — investigate further.'}

  2. TD(0) variance:                  {td_var:.2f}
     Variance reduction:              {(1 - td_var / mc_var) * 100:+.1f}%
     {'-> TD(0) would significantly reduce target variance.' if td_var < mc_var * 0.7 else '-> TD(0) provides moderate improvement.' if td_var < mc_var * 0.9 else '-> TD(0) provides minimal improvement.'}

Recommendations (in priority order):

  1. SWITCH TO TD(0) OR TD(lambda) TARGETS
     - Current MC targets assign the same episode-terminal reward to ALL
       decision points, even early ones with high outcome uncertainty.
     - TD(0) bootstraps from the value network, reducing variance at the
       cost of some bias (acceptable once the network is partially trained).
     - Expected loss reduction: ~{(1 - td_var / mc_var) * 100:.0f}%

  2. NORMALIZE REWARDS
     - Divide by 13 (max reward) for clean [-1, +1] targets.
     - This is cosmetic for MSE but helps with learning rate tuning and
       prevents exploding gradients.
     - New expected loss: ~{weighted_var / 169:.3f} (much easier to interpret)

  3. CONSIDER HUBER LOSS
     - Replace MSE with Huber(delta=5) to reduce outlier sensitivity.
     - Won't reduce the loss floor but stabilizes training.

  4. IF LOSS DOESN'T MATCH VARIANCE (convergence issue):
     - Check learning rate (may need warmup or decay)
     - Check for gradient issues (exploding/vanishing)
     - Add gradient clipping
     - Verify batch size is large enough for stable gradients
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TRAINING TARGET & NORMALIZATION ANALYSIS REPORT")
    print("=" * 70)
    print(f"Analyzing trained model: models/value_based_agent.pt")

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'value_based_agent.pt')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    agent = ValueBasedAgent(model_path=model_path)
    print(f"Model loaded. Input size: {agent.input_size}, Temperature: {agent.temperature}")

    # Data collection
    print(f"\n--- Data Collection Phase ---")
    print(f"Playing 50,000 self-play episodes with Boltzmann exploration...")
    records = collect_data(agent, num_episodes=50000)
    print(f"Collected {len(records)} decision records from 50,000 episodes.")

    # Analysis sections
    section_a_reward_distribution(records)
    weighted_var = section_b_per_state_variance(records)
    section_c_value_network_analysis(agent)
    mc_var, td_var = section_d_td_comparison(records, agent)
    section_e_normalization(records, weighted_var)
    print_summary(weighted_var, mc_var, td_var)

    print("\n" + "=" * 70)
    print("REPORT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
