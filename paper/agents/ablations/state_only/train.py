"""
paper — state_only ablation training
================================================
Ablation A: StateOnlyAgent (no opponent statistics).

Architecture:  V(s) = V_base(s) [frozen] + head(s)
               head: 15 → 32 → 32 → 1, near-zero output init.

Residual TD targets identical to full_modulation but with only game_enc:
  Terminal    : target_residual = r − V_base(s_T)
  Non-terminal: target_residual = (V_base(s_{t+1}) + head(s_{t+1})) − V_base(s_t)
  Loss        = MSE(head(s_t), target_residual)

Tracker is maintained for session management (opponent switching) but stats
are NOT passed to the agent.

Usage:
  python -m paper.agents.ablations.state_only.train           # full run
  python -m paper.agents.ablations.state_only.train --smoke   # 500 episodes
  python -m paper.agents.ablations.state_only.train --seed 1
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from paper.agents.ablations.state_only.agent import StateOnlyAgent
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS,
    WeightedOpponentSampler,
    SessionManager,
)
from paper.evaluation.comparison_protocol import build_standard_opponents
from paper.evaluation.shared.stats_tracker import (
    compute_pool_means,
    OpponentStatsTracker,
)
from engine.leduc_game import LeducGame, Action
from paper.evaluation.shared.training_recipe import encode_game_state

# ── constants ──────────────────────────────────────────────────────────────────

SESSION_LENGTH      = 100
PRIOR_STRENGTH      = 20.0
CALIBRATION_HANDS   = 500
BATCH_SIZE          = 32
LR                  = 1e-4
NUM_EPISODES        = 300_000
CHECKPOINT_INTERVAL = 10_000
FINAL_EVAL_ROUNDS   = 5_000


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


# ── state-only game loop ──────────────────────────────────────────────────────

def play_hand_state_only(agent: StateOnlyAgent, opponent, tracker: OpponentStatsTracker,
                         learner_id: int = 0):
    """
    Play one hand for StateOnlyAgent. Tracker is updated (for session management)
    but stats are not passed to the agent.

    Returns:
        chain  : list of (game_enc_15,) tuples for each learner post-action state
        reward : terminal reward for the learner
    """
    game = LeducGame()
    game.reset()
    chain = []
    prev_raise = False
    prev_round = -1

    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)
        if obs.current_round != prev_round:
            prev_raise = False
            prev_round = obs.current_round

        if cp == learner_id:
            # Agent does not receive opp_stats
            action = agent.select_action(obs)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            game_enc = encode_game_state(post_obs, viewer_id=learner_id)
            chain.append((game_enc,))   # single-element tuple, no stats
        else:
            action = opponent.select_action(obs)
            tracker.update_action(action, obs.current_round,
                                  prev_raise, obs.legal_actions)

        prev_raise = (action == Action.RAISE)
        game.step(action)

    tracker.update_hand_end()
    rewards = game.get_reward()
    return chain, rewards[learner_id]


# ── TD(0) update ──────────────────────────────────────────────────────────────

def td_update(agent: StateOnlyAgent, optimizer, criterion, batch_data):
    """
    Residual TD(0) update using only game_enc (no stats).

    Terminal    : target_residual = r − V_base(s_T)
    Non-terminal: target_residual = (V_base(s_{t+1}) + head(s_{t+1})) − V_base(s_t)
    Loss        = MSE(head(s_t), target_residual)

    Returns (loss, grad_norm, delta_mean).
    """
    optimizer.zero_grad()
    losses = []
    deltas = []

    for chain, reward in batch_data:
        if not chain:
            continue
        for t, (game_enc,) in enumerate(chain):
            game_t = game_enc.unsqueeze(0)

            with torch.no_grad():
                v_base_t = agent.base(game_t).squeeze()

            delta_t = agent.head(game_t).squeeze()   # scalar, grad flows

            with torch.no_grad():
                deltas.append(delta_t.detach().abs().item())

            if t == len(chain) - 1:
                # Terminal
                target_residual = torch.tensor(
                    float(reward), dtype=torch.float32) - v_base_t
            else:
                # Non-terminal: bootstrap using game_enc only
                game_t1 = chain[t + 1][0].unsqueeze(0)
                with torch.no_grad():
                    v_base_t1 = agent.base(game_t1).squeeze()
                    delta_t1  = agent.head(game_t1).squeeze()
                    v_total_t1 = v_base_t1 + delta_t1
                target_residual = v_total_t1 - v_base_t

            losses.append(criterion(delta_t, target_residual.detach()))

    if not losses:
        return 0.0, 0.0, 0.0

    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()

    params = list(agent.head.parameters())
    grad_norm = sum(
        p.grad.norm() ** 2 for p in params if p.grad is not None
    ) ** 0.5
    grad_norm = float(grad_norm)

    optimizer.step()

    delta_mean = float(np.mean(deltas)) if deltas else 0.0
    return mean_loss.item(), grad_norm, delta_mean


# ── main training loop ────────────────────────────────────────────────────────

def train(out_dir: str, num_episodes: int, smoke: bool, seed: int):
    # ── seeding ────────────────────────────────────────────────────────────────
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── pool priors ────────────────────────────────────────────────────────────
    priors_path   = os.path.join(out_dir, "pool_priors.json")
    shared_priors = os.path.join(
        ROOT, "preliminary_experiments", "opp_stats_input_aug",
        "outputs", "pool_random", "pool_priors.json")

    opponents = build_standard_opponents(ROOT)

    if os.path.exists(priors_path):
        pool_means = _load_json(priors_path)
        print(f"Loaded pool priors from {priors_path}")
    elif os.path.exists(shared_priors):
        pool_means = _load_json(shared_priors)
        _write_json(priors_path, pool_means)
        print("Reused pool priors from opp_stats_input_augmentation_v1")
    else:
        print("Computing pool priors (calibrating stats tracker)...")
        pool_means = compute_pool_means(opponents, 50 if smoke else CALIBRATION_HANDS)
        _write_json(priors_path, pool_means)

    # ── agent + optimiser ──────────────────────────────────────────────────────
    agent     = StateOnlyAgent()
    optimizer = optim.Adam(agent.head.parameters(), lr=LR)
    criterion = nn.MSELoss()
    agent.set_train_mode(True)

    # ── session + sampler ──────────────────────────────────────────────────────
    sampler = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)
    session = SessionManager(SESSION_LENGTH, pool_means, PRIOR_STRENGTH, sampler)

    # ── training config ────────────────────────────────────────────────────────
    config = {
        "agent":               "StateOnlyAgent",
        "architecture":        "V_base(frozen, 15→64→64→1) + StateOnlyHead(15→32→32→1)",
        "ablation":            "A — no opponent statistics",
        "training_recipe":     "weighted_pool (cfr×3, heuristic×3, others×1)",
        "opponent_weights":    OPPONENT_WEIGHTS,
        "learning_rate":       LR,
        "batch_size":          BATCH_SIZE,
        "session_length":      SESSION_LENGTH,
        "prior_strength":      PRIOR_STRENGTH,
        "num_episodes":        num_episodes,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "seed":                seed,
        "smoke":               smoke,
    }
    _write_json(os.path.join(out_dir, "train_config.json"), config)

    freqs = sampler.expected_frequencies()
    print("Opponent sampling weights:")
    for k, p in freqs.items():
        print(f"  {k:<20s} {p:.1%}")

    print(f"\n{'='*65}")
    print(f"  paper — state_only (Ablation A)  seed={seed}")
    print(f"  {'SMOKE' if smoke else 'FULL'} | episodes={num_episodes:,} "
          f"batch={BATCH_SIZE} session={SESSION_LENGTH}")
    print(f"{'='*65}\n")

    batch_data = []
    log_path   = os.path.join(out_dir, "train_log.jsonl")
    log_file   = open(log_path, "w")

    last_loss       = None
    last_grad_norm  = None
    last_delta_mean = None

    t0 = time.time()
    for ep in range(1, num_episodes + 1):
        learner_id = ep % 2
        opp_name, opp = session.current_opponent()

        chain, reward = play_hand_state_only(
            agent, opp, session.tracker(learner_id), learner_id=learner_id)
        session.record_hand(learner_id)

        # confidence from tracker (for logging consistency), but not used by agent
        confidence = float(session.tracker(learner_id).get_features()[6])

        batch_data.append((chain, reward))

        ep_loss       = None
        ep_grad_norm  = None
        ep_delta_mean = None

        if len(batch_data) >= BATCH_SIZE:
            loss_val, grad_norm_val, delta_mean_val = td_update(
                agent, optimizer, criterion, batch_data)
            batch_data.clear()
            last_loss       = loss_val
            last_grad_norm  = grad_norm_val
            last_delta_mean = delta_mean_val
            ep_loss       = round(loss_val, 6)
            ep_grad_norm  = round(grad_norm_val, 6)
            ep_delta_mean = round(delta_mean_val, 6)

        log_entry = {
            "ep":         ep,
            "opp":        opp_name,
            "session":    session._counts[learner_id],
            "reward":     float(reward),
            "confidence": round(confidence, 6),
            "loss":       ep_loss,
            "grad_norm":  ep_grad_norm,
            "delta_mean": ep_delta_mean,
        }
        log_file.write(json.dumps(log_entry) + "\n")

        if ep % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_ep{ep:06d}.pt")
            agent.save_model(ckpt_path)

            l_str  = f"{last_loss:.4f}"       if last_loss       is not None else "n/a"
            gn_str = f"{last_grad_norm:.4f}"  if last_grad_norm  is not None else "n/a"
            dm_str = f"{last_delta_mean:.4f}" if last_delta_mean is not None else "n/a"
            print(f"[ep={ep:,}] loss={l_str} grad_norm={gn_str} "
                  f"delta_mean={dm_str} opp={opp_name} conf={confidence:.4f}")

    log_file.close()

    agent.save_model(os.path.join(out_dir, "checkpoint_final.pt"))

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Checkpoints saved to {out_dir}/")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train StateOnlyAgent — Ablation A (paper)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick pipeline check (500 episodes)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes")
    args = parser.parse_args()

    smoke    = args.smoke
    seed     = args.seed
    episodes = args.episodes or (500 if smoke else NUM_EPISODES)
    out_dir  = os.path.join(HERE, "outputs", f"seed_{seed}")

    train(out_dir=out_dir, num_episodes=episodes, smoke=smoke, seed=seed)
