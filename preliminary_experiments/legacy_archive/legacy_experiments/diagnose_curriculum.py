"""
Diagnostic experiment: Why did CurriculumAgent perform WORST in Round 3?
  - avg=-0.82, robustness=-1.976
  - Worse than pop_adaptive (avg=-0.557), which it was designed to fix
  - Far worse than adaptive_value (avg=+1.012)

CurriculumAgent was designed to fix PopAdaptive's failures by using:
  - Block scheduling (100 sessions per opponent before rotating)
  - Rehearsal buffer (20% old data mixed into each batch)
  - Both-player chain collection (train on both P0 and P1 data)
  - Opponent pool: heuristic -> value_based -> adaptive_value (weak to strong)
  - Training: 1000 sessions, lr=1e-4, block_size=100, rehearsal_ratio=0.2
  - Result: final_loss=16.89, final_eval_vs_heuristic=-1.16

Hypotheses:
  A) Both-player chain collection introduces conflicting value targets
     (P0 and P1 have opposite rewards, but the value net trains on both)
  B) Block scheduling doesn't help -- opponent diversity hurts regardless
  C) Rehearsal buffer mixes stale data that conflicts with current learning
  D) 1000 sessions is too few (adaptive_value uses 667 but only self-play)

Experiments:
  1. Control: fresh adaptive_value with 1000 sessions (fair budget comparison)
  2. Ablation -- no rehearsal: curriculum with rehearsal_ratio=0.0
  3. Ablation -- no blocking: curriculum with block_size=1 (rotate every session)
  4. Ablation -- single opponent: curriculum against heuristic only (no curriculum)
  5. Evaluate all variants against heuristic, value_based, adaptive_value
  6. Both-player chain analysis: compare loss magnitudes across player positions
"""

import os
import sys
import copy
import time
import json
import random
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.curriculum_agent import CurriculumAgent
from src.agents.pop_adaptive import PopAdaptiveAgent
from src.engine.poker_session import PokerSession
from src.engine.leduc_game import LeducGame, Action
from src.training.adaptive_trainer import AdaptiveTrainer
from src.training.curriculum_trainer import CurriculumTrainer
from src.training.evaluation import evaluate_agents, quick_evaluate, compute_robustness_metrics
from dataclasses import replace


def separator(title):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {title}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Experiment 0: Analyze the CRITICAL both-player chain problem
# ──────────────────────────────────────────────────────────────────────
def exp0_both_player_chain_analysis():
    """Investigate whether training on both P0 and P1 chains causes
    conflicting gradients. In pool training:
      - P0 is the curriculum agent, P1 is a frozen opponent
      - P0's reward = -P1's reward (zero-sum)
      - But update_model() trains on BOTH chains with respective rewards
      - This means the VALUE NETWORK sees:
        P0 states -> targets toward +reward
        P1 states -> targets toward -reward (opposite!)

    In self-play this is fine: both players ARE the same agent.
    In pool training, P1's states come from a DIFFERENT agent's policy,
    but are used to train the curriculum agent's value network.
    """
    separator("0 -- Both-player chain conflict analysis")

    agent = CurriculumAgent()
    trainer = CurriculumTrainer(agent, learning_rate=1e-4, hands_per_session=30,
                                block_size=100, rehearsal_ratio=0.0)
    agent.set_train_mode(True)

    # Collect some episodes and measure per-player loss magnitudes
    p0_losses = []
    p1_losses = []
    reward_signs = {"same": 0, "opposite": 0}

    for sess_idx in range(20):
        session_data = trainer.collect_episode()

        for chains, rewards in session_data:
            p0_chain = chains[0]
            p1_chain = chains[1]

            # Check reward structure
            if rewards[0] * rewards[1] < 0:
                reward_signs["opposite"] += 1
            elif rewards[0] == 0 and rewards[1] == 0:
                pass  # tie
            else:
                reward_signs["same"] += 1

            # Compute per-player TD losses (without backprop)
            for p_idx, chain in enumerate([p0_chain, p1_chain]):
                if not chain:
                    continue
                for t in range(len(chain)):
                    prediction = agent.model(chain[t]).squeeze(0)
                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = agent.model(chain[t + 1]).squeeze(0)
                    loss_val = (prediction - target).pow(2).item()
                    if p_idx == 0:
                        p0_losses.append(loss_val)
                    else:
                        p1_losses.append(loss_val)

    avg_p0_loss = sum(p0_losses) / len(p0_losses) if p0_losses else 0
    avg_p1_loss = sum(p1_losses) / len(p1_losses) if p1_losses else 0

    print(f"  Per-player loss analysis (20 sessions, untrained model):")
    print(f"    P0 (curriculum agent) chain entries: {len(p0_losses)}")
    print(f"    P1 (frozen opponent)  chain entries: {len(p1_losses)}")
    print(f"    P0 avg MSE loss: {avg_p0_loss:.4f}")
    print(f"    P1 avg MSE loss: {avg_p1_loss:.4f}")
    print(f"    Loss ratio P1/P0: {avg_p1_loss/avg_p0_loss:.2f}" if avg_p0_loss > 0 else "")
    print(f"\n  Reward signs:")
    print(f"    Opposite signs (zero-sum): {reward_signs['opposite']}")
    print(f"    Same signs:                {reward_signs['same']}")

    # The KEY problem: in pool training, P1 states are generated by the
    # opponent's POLICY but trained to predict the opponent's REWARD using
    # the curriculum agent's value network. The curriculum agent's value
    # network never controls P1's actions, so P1 states are off-policy
    # AND the targets push in the opposite direction of P0's targets.
    print(f"\n  CRITICAL ISSUE:")
    print(f"  In self-play, training on P1 chain is valid because both")
    print(f"  players ARE the same agent -- the value net predicts its own")
    print(f"  outcomes from its own states regardless of seat position.")
    print(f"  In pool training, P1's states come from a DIFFERENT agent's")
    print(f"  policy. The curriculum agent's value net is being trained on")
    print(f"  states it never visits (off-policy) with rewards that are")
    print(f"  OPPOSITE to what it would get from P0's perspective.")

    # Quantify: how different are P0 vs P1 encoded states?
    all_p0_states = []
    all_p1_states = []
    session_data = trainer.collect_episode()  # fresh session
    for chains, rewards in session_data:
        for s in chains[0]:
            all_p0_states.append(s.squeeze(0))
        for s in chains[1]:
            all_p1_states.append(s.squeeze(0))

    if all_p0_states and all_p1_states:
        p0_stack = torch.stack(all_p0_states)
        p1_stack = torch.stack(all_p1_states)
        p0_mean = p0_stack.mean(dim=0)
        p1_mean = p1_stack.mean(dim=0)
        dist = (p0_mean - p1_mean).pow(2).sum().sqrt().item()
        print(f"\n  State distribution analysis:")
        print(f"    P0 mean state norm: {p0_mean.pow(2).sum().sqrt().item():.4f}")
        print(f"    P1 mean state norm: {p1_mean.pow(2).sum().sqrt().item():.4f}")
        print(f"    L2 distance between P0 and P1 mean states: {dist:.4f}")
        print(f"    (Large distance = P0 and P1 visit very different states)")

    return {
        "p0_entries": len(p0_losses),
        "p1_entries": len(p1_losses),
        "p0_avg_loss": avg_p0_loss,
        "p1_avg_loss": avg_p1_loss,
        "reward_signs": reward_signs,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 1: Control — fresh adaptive_value with 1000 sessions
# ──────────────────────────────────────────────────────────────────────
def exp1_control_adaptive_value():
    """Train a fresh adaptive_value with same budget (1000 sessions)
    as curriculum agent, using pure self-play. This is the fair baseline."""
    separator("1 -- Control: fresh adaptive_value (1000 sessions, self-play)")

    agent = AdaptiveValueAgent()
    trainer = AdaptiveTrainer(agent, learning_rate=1e-4, hands_per_session=30)

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=1000, batch_size=32,
                  save_path="models/diag_curriculum_control.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {
        "losses": losses,
        "evals": eval_scores,
        "time": elapsed,
        "final_loss": losses[-1] if losses else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 2: Ablation — no rehearsal
# ──────────────────────────────────────────────────────────────────────
def exp2_no_rehearsal():
    """Train curriculum agent with rehearsal_ratio=0.0 to test whether
    the rehearsal buffer is helping or hurting."""
    separator("2 -- Ablation: no rehearsal (rehearsal_ratio=0.0)")

    agent = CurriculumAgent()
    trainer = CurriculumTrainer(agent, learning_rate=1e-4, hands_per_session=30,
                                block_size=100, rehearsal_ratio=0.0)

    losses = []
    eval_scores = []
    block_transitions = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=1000, batch_size=32,
                  save_path="models/diag_curriculum_no_rehearsal.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {
        "losses": losses,
        "evals": eval_scores,
        "time": elapsed,
        "final_loss": losses[-1] if losses else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 3: Ablation — no blocking (rotate every session)
# ──────────────────────────────────────────────────────────────────────
def exp3_no_blocking():
    """Train curriculum with block_size=1 — rotate opponent every session.
    This is similar to PopAdaptive's rotation but keeps both-player chains
    and rehearsal."""
    separator("3 -- Ablation: no blocking (block_size=1, rotate every session)")

    agent = CurriculumAgent()
    trainer = CurriculumTrainer(agent, learning_rate=1e-4, hands_per_session=30,
                                block_size=1, rehearsal_ratio=0.2)

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=1000, batch_size=32,
                  save_path="models/diag_curriculum_no_blocking.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {
        "losses": losses,
        "evals": eval_scores,
        "time": elapsed,
        "final_loss": losses[-1] if losses else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 4: Ablation — single opponent (heuristic only, no curriculum)
# ──────────────────────────────────────────────────────────────────────
def exp4_single_opponent():
    """Train curriculum agent against ONLY heuristic. This removes
    the curriculum entirely — the agent just plays heuristic with
    both-player chains and rehearsal."""
    separator("4 -- Ablation: single opponent (heuristic only)")

    agent = CurriculumAgent()
    trainer = CurriculumTrainer(agent, learning_rate=1e-4, hands_per_session=30,
                                block_size=1000, rehearsal_ratio=0.2)

    # Override opponent pool to only contain heuristic
    heuristic = HeuristicAgent()
    trainer.opponent_pool = [("heuristic", heuristic)]
    trainer.current_opponent_idx = 0

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=1000, batch_size=32,
                  save_path="models/diag_curriculum_heuristic_only.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {
        "losses": losses,
        "evals": eval_scores,
        "time": elapsed,
        "final_loss": losses[-1] if losses else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 5: Ablation — P0-only chains (fix the both-player issue)
# ──────────────────────────────────────────────────────────────────────
def exp5_p0_only_chains():
    """Train curriculum agent but override collect_episode to only record
    P0 chains (like PopAdaptiveTrainer does). If both-player training is
    the problem, this should perform much better."""
    separator("5 -- Ablation: P0-only chains (no both-player training)")

    agent = CurriculumAgent()
    trainer = CurriculumTrainer(agent, learning_rate=1e-4, hands_per_session=30,
                                block_size=100, rehearsal_ratio=0.2)

    # Monkey-patch collect_episode to only record P0 chains
    original_collect = trainer.collect_episode

    def p0_only_collect():
        """Modified collect_episode that only records P0 (training agent) chains."""
        trainer.session.reset()
        session_data = []
        opponent = trainer.opponent_pool[trainer.current_opponent_idx][1]

        for _ in range(trainer.hands_per_session):
            trainer.session.new_hand()
            chains = [[], []]  # chains[1] will stay empty

            while not trainer.session.is_finished:
                current_player = trainer.session.current_player
                obs = trainer.session.get_observation(viewer_id=current_player)

                if current_player == 0:
                    action = agent.select_action(obs)
                else:
                    action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]

                # Only record P0's chain (the training agent)
                if current_player == 0:
                    post_obs, _ = LeducGame.simulate_action(obs, action)
                    if obs.opponent_stats is not None:
                        post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)
                    encoded = agent.encode_observation(post_obs, viewer_id=current_player)
                    chains[current_player].append(encoded)

                trainer.session.step(action)

            rewards = trainer.session.game.get_reward()
            session_data.append((chains, rewards))

        trainer.sessions_in_current_block += 1

        # Store in rehearsal buffer
        for item in session_data:
            if len(trainer.rehearsal_buffer) < trainer.max_buffer_size:
                trainer.rehearsal_buffer.append(item)
            else:
                idx = random.randint(0, len(trainer.rehearsal_buffer) - 1)
                trainer.rehearsal_buffer[idx] = item

        # Block transition
        if trainer.sessions_in_current_block >= trainer.block_size:
            trainer._transition_to_next_opponent()

        return session_data

    trainer.collect_episode = p0_only_collect

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=1000, batch_size=32,
                  save_path="models/diag_curriculum_p0_only.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {
        "losses": losses,
        "evals": eval_scores,
        "time": elapsed,
        "final_loss": losses[-1] if losses else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 6: Loss trajectory comparison during training
# ──────────────────────────────────────────────────────────────────────
def exp6_loss_trajectory():
    """Compare loss trajectories of:
    - Full curriculum (both-player chains + rehearsal + blocking)
    - Adaptive self-play (baseline)
    Track per-player loss breakdown to isolate the conflict."""
    separator("6 -- Loss trajectory comparison + per-player breakdown")

    # Train full curriculum manually for 200 sessions, tracking per-player losses
    agent_curr = CurriculumAgent()
    trainer_curr = CurriculumTrainer(agent_curr, learning_rate=1e-4, hands_per_session=30,
                                     block_size=100, rehearsal_ratio=0.2)
    agent_curr.set_train_mode(True)

    losses_curr = []
    p0_loss_trace = []
    p1_loss_trace = []

    batch_data = []
    for sess in range(200):
        session_data = trainer_curr.collect_episode()
        batch_data.extend(session_data)

        if len(batch_data) >= 32:
            # Manually compute per-player losses before update
            p0_batch_loss = []
            p1_batch_loss = []
            for chains, rewards in batch_data:
                for p_idx in [0, 1]:
                    chain = chains[p_idx]
                    if not chain:
                        continue
                    for t in range(len(chain)):
                        with torch.no_grad():
                            pred = agent_curr.model(chain[t]).squeeze(0)
                            if t == len(chain) - 1:
                                target = torch.FloatTensor([rewards[p_idx]])
                            else:
                                target = agent_curr.model(chain[t + 1]).squeeze(0)
                            loss_val = (pred - target).pow(2).item()
                            if p_idx == 0:
                                p0_batch_loss.append(loss_val)
                            else:
                                p1_batch_loss.append(loss_val)

            if p0_batch_loss:
                p0_loss_trace.append(sum(p0_batch_loss) / len(p0_batch_loss))
            if p1_batch_loss:
                p1_loss_trace.append(sum(p1_batch_loss) / len(p1_batch_loss))

            # Actually update
            loss = trainer_curr.update_model(batch_data)
            losses_curr.append(loss)
            batch_data = []

    # Train adaptive self-play for 200 sessions
    agent_av = AdaptiveValueAgent()
    trainer_av = AdaptiveTrainer(agent_av, learning_rate=1e-4, hands_per_session=30)
    agent_av.set_train_mode(True)

    losses_av = []
    batch_data = []
    for sess in range(200):
        session_data = trainer_av.collect_episode()
        batch_data.extend(session_data)
        if len(batch_data) >= 32:
            loss = trainer_av.update_model(batch_data)
            losses_av.append(loss)
            batch_data = []

    # Print comparison
    def avg_window(lst, window=5):
        return [sum(lst[i:i+window])/window for i in range(0, len(lst)-window+1, window)]

    curr_smoothed = avg_window(losses_curr)
    av_smoothed = avg_window(losses_av)

    print(f"  Curriculum loss trajectory (windowed avg of 5):")
    for i, l in enumerate(curr_smoothed[:12]):
        print(f"    batch {i*5:>3d}-{i*5+4:>3d}: {l:.4f}")
    if curr_smoothed:
        print(f"    ... final: {curr_smoothed[-1]:.4f}")

    print(f"\n  AdaptiveValue self-play loss trajectory (windowed avg of 5):")
    for i, l in enumerate(av_smoothed[:12]):
        print(f"    batch {i*5:>3d}-{i*5+4:>3d}: {l:.4f}")
    if av_smoothed:
        print(f"    ... final: {av_smoothed[-1]:.4f}")

    # Per-player breakdown
    p0_smoothed = avg_window(p0_loss_trace)
    p1_smoothed = avg_window(p1_loss_trace)

    print(f"\n  Curriculum per-player loss breakdown:")
    print(f"  {'Batch':>10s}  {'P0 (agent)':>12s}  {'P1 (opponent)':>14s}  {'Ratio P1/P0':>12s}")
    for i in range(min(12, len(p0_smoothed), len(p1_smoothed))):
        ratio = p1_smoothed[i] / p0_smoothed[i] if p0_smoothed[i] > 0 else float('inf')
        print(f"  {i*5:>3d}-{i*5+4:>3d}     {p0_smoothed[i]:12.4f}  {p1_smoothed[i]:14.4f}  {ratio:12.2f}")

    if p0_smoothed and p1_smoothed:
        final_p0 = p0_smoothed[-1]
        final_p1 = p1_smoothed[-1]
        print(f"\n  Final P0 loss: {final_p0:.4f}")
        print(f"  Final P1 loss: {final_p1:.4f}")
        print(f"  Ratio: {final_p1/final_p0:.2f}" if final_p0 > 0 else "")

    return {
        "curr_losses": losses_curr,
        "av_losses": losses_av,
        "p0_losses": p0_loss_trace,
        "p1_losses": p1_loss_trace,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 7: Evaluate ALL variants head-to-head
# ──────────────────────────────────────────────────────────────────────
def exp7_full_evaluation():
    """Evaluate all trained variants against the three opponent types:
    heuristic, value_based, adaptive_value."""
    separator("7 -- Full evaluation of all variants")

    # Opponents for evaluation
    heuristic = HeuristicAgent()

    vb = ValueBasedAgent()
    if os.path.exists("models/value_based_agent.pt"):
        vb.load_model("models/value_based_agent.pt")
    vb.set_train_mode(False)

    av = AdaptiveValueAgent()
    if os.path.exists("models/adaptive_value_agent.pt"):
        av.load_model("models/adaptive_value_agent.pt")
    av.set_train_mode(False)

    opponents = {
        "heuristic": heuristic,
        "value_based": vb,
        "adaptive_value": av,
    }

    # Agents to evaluate
    variants = {}

    # Original curriculum agent
    if os.path.exists("models/curriculum_agent.pt"):
        a = CurriculumAgent()
        a.load_model("models/curriculum_agent.pt")
        a.set_train_mode(False)
        variants["curriculum_original"] = a

    # Control: adaptive_value 1000 sessions
    if os.path.exists("models/diag_curriculum_control.pt"):
        a = AdaptiveValueAgent()
        a.load_model("models/diag_curriculum_control.pt")
        a.set_train_mode(False)
        variants["control_av_1000"] = a

    # Ablation: no rehearsal
    if os.path.exists("models/diag_curriculum_no_rehearsal.pt"):
        a = CurriculumAgent()
        a.load_model("models/diag_curriculum_no_rehearsal.pt")
        a.set_train_mode(False)
        variants["no_rehearsal"] = a

    # Ablation: no blocking
    if os.path.exists("models/diag_curriculum_no_blocking.pt"):
        a = CurriculumAgent()
        a.load_model("models/diag_curriculum_no_blocking.pt")
        a.set_train_mode(False)
        variants["no_blocking"] = a

    # Ablation: single opponent
    if os.path.exists("models/diag_curriculum_heuristic_only.pt"):
        a = CurriculumAgent()
        a.load_model("models/diag_curriculum_heuristic_only.pt")
        a.set_train_mode(False)
        variants["heuristic_only"] = a

    # Ablation: P0-only chains
    if os.path.exists("models/diag_curriculum_p0_only.pt"):
        a = CurriculumAgent()
        a.load_model("models/diag_curriculum_p0_only.pt")
        a.set_train_mode(False)
        variants["p0_only_chains"] = a

    # Also add the original pretrained adaptive_value for reference
    if os.path.exists("models/adaptive_value_agent.pt"):
        a = AdaptiveValueAgent()
        a.load_model("models/adaptive_value_agent.pt")
        a.set_train_mode(False)
        variants["adaptive_value_pretrained"] = a

    # Also add pop_adaptive for reference
    if os.path.exists("models/pop_adaptive_agent.pt"):
        a = PopAdaptiveAgent()
        a.load_model("models/pop_adaptive_agent.pt")
        a.set_train_mode(False)
        variants["pop_adaptive_pretrained"] = a

    # Evaluate each variant against each opponent
    results = {}
    num_eval_rounds = 500

    print(f"  Evaluating {len(variants)} variants against {len(opponents)} opponents ({num_eval_rounds} rounds each)...")
    print()

    for var_name, var_agent in variants.items():
        results[var_name] = {}
        for opp_name, opp_agent in opponents.items():
            score = quick_evaluate(var_agent, opp_agent, num_rounds=num_eval_rounds)
            results[var_name][opp_name] = round(score, 4)
            print(f"    {var_name:>30s} vs {opp_name:<18s}: {score:+.4f}")
        print()

    # Print summary table
    print(f"\n  {'Variant':>30s} | {'vs heuristic':>14s} | {'vs value_based':>14s} | {'vs adaptive':>14s} | {'AVG':>8s} | {'Robustness':>10s}")
    print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}-+-{'-'*10}")

    for var_name in variants:
        scores = results[var_name]
        avg = sum(scores.values()) / len(scores)
        rob = compute_robustness_metrics(scores)
        print(f"  {var_name:>30s} | {scores.get('heuristic', 0):>+14.4f} | "
              f"{scores.get('value_based', 0):>+14.4f} | "
              f"{scores.get('adaptive_value', 0):>+14.4f} | "
              f"{avg:>+8.4f} | {rob['robustness']:>+10.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Experiment 8: Rehearsal buffer content analysis
# ──────────────────────────────────────────────────────────────────────
def exp8_rehearsal_analysis():
    """Analyze what's actually in the rehearsal buffer during training.
    Key question: does mixing old opponent data into new opponent batches
    create gradient conflicts?"""
    separator("8 -- Rehearsal buffer content analysis")

    agent = CurriculumAgent()
    trainer = CurriculumTrainer(agent, learning_rate=1e-4, hands_per_session=30,
                                block_size=100, rehearsal_ratio=0.2)
    agent.set_train_mode(True)

    # Train for 200 sessions and track rehearsal buffer growth
    buffer_sizes = []
    batch_losses_with_rehearsal = []
    batch_losses_without_rehearsal = []

    batch_data = []
    for sess in range(200):
        session_data = trainer.collect_episode()
        batch_data.extend(session_data)

        if len(batch_data) >= 32:
            buffer_sizes.append(len(trainer.rehearsal_buffer))

            # Measure loss WITHOUT rehearsal mixing
            loss_pure = trainer_compute_loss(agent, batch_data)
            batch_losses_without_rehearsal.append(loss_pure)

            # Compute what rehearsal would add
            if trainer.rehearsal_buffer and trainer.rehearsal_ratio > 0:
                n_rehearsal = max(1, int(len(batch_data) * trainer.rehearsal_ratio
                                        / (1 - trainer.rehearsal_ratio)))
                n_rehearsal = min(n_rehearsal, len(trainer.rehearsal_buffer))
                rehearsal_samples = random.sample(trainer.rehearsal_buffer, n_rehearsal)
                mixed_data = batch_data + rehearsal_samples
                loss_mixed = trainer_compute_loss(agent, mixed_data)
                batch_losses_with_rehearsal.append(loss_mixed)
            else:
                batch_losses_with_rehearsal.append(loss_pure)

            # Actual update (with rehearsal)
            trainer.update_model(batch_data)
            batch_data = []

    print(f"  Rehearsal buffer growth:")
    checkpoints = [0, 25, 50, 100, 150, 199]
    for i in checkpoints:
        if i < len(buffer_sizes):
            print(f"    After batch {i}: buffer size = {buffer_sizes[i]}")

    if batch_losses_without_rehearsal and batch_losses_with_rehearsal:
        print(f"\n  Loss comparison (first 10 batches):")
        print(f"  {'Batch':>6s}  {'Pure loss':>12s}  {'With rehearsal':>14s}  {'Difference':>12s}")
        for i in range(min(10, len(batch_losses_without_rehearsal))):
            diff = batch_losses_with_rehearsal[i] - batch_losses_without_rehearsal[i]
            print(f"  {i:>6d}  {batch_losses_without_rehearsal[i]:12.4f}  "
                  f"{batch_losses_with_rehearsal[i]:14.4f}  {diff:>+12.4f}")

        print(f"\n  Late training (last 10 batches):")
        start_idx = max(0, len(batch_losses_without_rehearsal) - 10)
        for i in range(start_idx, len(batch_losses_without_rehearsal)):
            diff = batch_losses_with_rehearsal[i] - batch_losses_without_rehearsal[i]
            print(f"  {i:>6d}  {batch_losses_without_rehearsal[i]:12.4f}  "
                  f"{batch_losses_with_rehearsal[i]:14.4f}  {diff:>+12.4f}")

    return {
        "final_buffer_size": buffer_sizes[-1] if buffer_sizes else 0,
        "n_batches": len(batch_losses_without_rehearsal),
    }


def trainer_compute_loss(agent, batch_data):
    """Compute MSE loss without backprop for analysis."""
    total_losses = []
    for chains, rewards in batch_data:
        for p_idx in [0, 1]:
            chain = chains[p_idx]
            if not chain:
                continue
            for t in range(len(chain)):
                with torch.no_grad():
                    prediction = agent.model(chain[t]).squeeze(0)
                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        target = agent.model(chain[t + 1]).squeeze(0)
                    loss = (prediction - target).pow(2).item()
                    total_losses.append(loss)
    return sum(total_losses) / len(total_losses) if total_losses else 0.0


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DIAGNOSIS: Why did CurriculumAgent perform WORST in Round 3?")
    print("  (avg=-0.82, robustness=-1.976)")
    print("=" * 70)

    all_results = {}

    # ── Phase 1: Quick analysis (no training) ──
    print("\n" + "=" * 70)
    print("  PHASE 1: Quick analysis (no training required)")
    print("=" * 70)

    all_results["exp0_chain_analysis"] = exp0_both_player_chain_analysis()

    # ── Phase 2: Loss trajectory comparison ──
    print("\n" + "=" * 70)
    print("  PHASE 2: Loss trajectory analysis (200 sessions each)")
    print("=" * 70)

    all_results["exp6_loss_trajectory"] = exp6_loss_trajectory()
    all_results["exp8_rehearsal"] = exp8_rehearsal_analysis()

    # ── Phase 3: Training experiments ──
    print("\n" + "=" * 70)
    print("  PHASE 3: Training experiments (1000 sessions each)")
    print("=" * 70)

    all_results["exp1_control"] = exp1_control_adaptive_value()
    all_results["exp2_no_rehearsal"] = exp2_no_rehearsal()
    all_results["exp3_no_blocking"] = exp3_no_blocking()
    all_results["exp4_single_opponent"] = exp4_single_opponent()
    all_results["exp5_p0_only"] = exp5_p0_only_chains()

    # ── Phase 4: Full evaluation ──
    print("\n" + "=" * 70)
    print("  PHASE 4: Full evaluation of all variants")
    print("=" * 70)

    all_results["exp7_evaluation"] = exp7_full_evaluation()

    # ── Save results ──
    results_path = "experiments/diagnose_curriculum_results.json"
    serializable = {}

    for k, v in all_results.items():
        if k == "exp0_chain_analysis":
            serializable[k] = v
        elif k in ("exp1_control", "exp2_no_rehearsal", "exp3_no_blocking",
                    "exp4_single_opponent", "exp5_p0_only"):
            serializable[k] = {
                "final_loss": v.get("final_loss"),
                "n_losses": len(v.get("losses", [])),
                "final_5_losses": v["losses"][-5:] if v.get("losses") else [],
                "evals": v.get("evals", []),
                "time": v.get("time"),
            }
        elif k == "exp6_loss_trajectory":
            serializable[k] = {
                "curr_n_batches": len(v.get("curr_losses", [])),
                "av_n_batches": len(v.get("av_losses", [])),
                "curr_final_loss": v["curr_losses"][-1] if v.get("curr_losses") else None,
                "av_final_loss": v["av_losses"][-1] if v.get("av_losses") else None,
                "p0_final_loss": v["p0_losses"][-1] if v.get("p0_losses") else None,
                "p1_final_loss": v["p1_losses"][-1] if v.get("p1_losses") else None,
            }
        elif k == "exp7_evaluation":
            serializable[k] = v
        elif k == "exp8_rehearsal":
            serializable[k] = v

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Summary ──
    separator("SUMMARY OF FINDINGS")

    # Hypothesis A: Both-player chain conflict
    chain = all_results.get("exp0_chain_analysis", {})
    print("  HYPOTHESIS A: Both-player chain introduces conflicting value targets")
    print(f"    P0 chain entries: {chain.get('p0_entries', '?')}")
    print(f"    P1 chain entries: {chain.get('p1_entries', '?')}")
    print(f"    P0 avg loss:     {chain.get('p0_avg_loss', '?'):.4f}" if isinstance(chain.get('p0_avg_loss'), float) else "")
    print(f"    P1 avg loss:     {chain.get('p1_avg_loss', '?'):.4f}" if isinstance(chain.get('p1_avg_loss'), float) else "")
    print(f"    Reward structure: {chain.get('reward_signs', {})}")
    print()

    # Compare ablation results
    eval_results = all_results.get("exp7_evaluation", {})
    if eval_results:
        print("  ABLATION COMPARISON (avg chips/round vs opponents):")
        print(f"  {'Variant':>30s}  {'vs heur':>10s}  {'vs vb':>10s}  {'vs av':>10s}  {'AVG':>10s}")
        print(f"  {'-'*80}")
        for var_name, scores in sorted(eval_results.items()):
            avg = sum(scores.values()) / len(scores) if scores else 0
            print(f"  {var_name:>30s}  {scores.get('heuristic', 0):>+10.4f}  "
                  f"{scores.get('value_based', 0):>+10.4f}  "
                  f"{scores.get('adaptive_value', 0):>+10.4f}  {avg:>+10.4f}")
        print()

    # Key conclusions
    print("  KEY CONCLUSIONS:")
    print()

    # Compare control vs curriculum
    ctrl = all_results.get("exp1_control", {})
    if ctrl.get("final_loss") is not None:
        print(f"  - Control (self-play, 1000 sessions) final loss: {ctrl['final_loss']:.4f}")

    for exp_name, label in [
        ("exp2_no_rehearsal", "No rehearsal"),
        ("exp3_no_blocking", "No blocking"),
        ("exp4_single_opponent", "Single opponent"),
        ("exp5_p0_only", "P0-only chains"),
    ]:
        exp_data = all_results.get(exp_name, {})
        if exp_data.get("final_loss") is not None:
            print(f"  - {label:25s} final loss: {exp_data['final_loss']:.4f}")

    print()
    print("  Compare evaluation scores to determine which ablation helped most.")
    print("  If P0-only >> full curriculum: both-player chains are the problem (H-A)")
    print("  If no-rehearsal >> full curriculum: rehearsal buffer hurts (H-C)")
    print("  If no-blocking >> full curriculum: block scheduling hurts (H-B)")
    print("  If single-opponent >> full curriculum: opponent diversity hurts (H-B/D)")
    print("  If control >> all curriculum variants: population training itself hurts")


if __name__ == "__main__":
    main()
