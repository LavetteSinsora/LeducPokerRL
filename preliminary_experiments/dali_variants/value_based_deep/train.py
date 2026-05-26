"""
DALI_modulation — value_based_deep training
=============================================
Architecture:  15 → 64 → 64 → 64 → 1  (depth ablation)

Training:
  - Weighted opponent pool: CFR + heuristic 3×, others 1×
  - 200K episodes, lr=1e-4, batch=32, Adam
  - Standard TD(0) on full value (no modulation head)
  - Checkpoint every 10K episodes

Usage:
  python -m preliminary_experiments.dali_variants.value_based_deep.train --seed 0
  python -m preliminary_experiments.dali_variants.value_based_deep.train --seed 1 --smoke
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from preliminary_experiments.dali_variants.value_based_deep.agent import ValueDeepAgent
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS,
    WeightedOpponentSampler,
)
from paper.evaluation.comparison_protocol import build_standard_opponents
from paper.evaluation.shared.stats_tracker import compute_pool_means
from engine.leduc_game import LeducGame, Action

# ── constants ─────────────────────────────────────────────────────────────────

LR               = 1e-4
BATCH_SIZE       = 32
NUM_EPISODES     = 200_000
CKPT_INTERVAL    = 10_000
CAL_HANDS        = 500


# ── play one hand ─────────────────────────────────────────────────────────────

def play_hand_deep(agent: ValueDeepAgent, opponent, learner_id: int):
    """
    Play one hand; return (chain, reward) for the learner.
    chain = list of game_enc (15-dim) tensors at each of learner's post-action states.
    reward = terminal chip gain/loss for the learner.
    """
    game = LeducGame()
    game.reset()

    chain = []

    while not game.is_finished:
        cp  = game.current_player
        obs = game.get_observation(viewer_id=cp)

        if cp == learner_id:
            action = agent.select_action(obs)
            post_obs, done = LeducGame.simulate_action(obs, action)
            if not (done and action == Action.FOLD):
                enc = agent._encode_game(post_obs, learner_id)
                chain.append(enc)
        else:
            action = opponent.select_action(obs)

        game.step(action)

    rewards = game.get_reward()
    return chain, float(rewards[learner_id])


# ── TD(0) update ──────────────────────────────────────────────────────────────

def td_update(agent: ValueDeepAgent, optimizer, criterion, batch_data):
    """
    Standard TD(0) on the full value network.

    Terminal    : target = reward
    Non-terminal: target = V(s_{t+1})   [detached]
    Loss        = MSE(V(s_t), target)

    Returns (loss, grad_norm).
    """
    optimizer.zero_grad()
    losses = []

    for chain, reward in batch_data:
        if not chain:
            continue
        for t, game_enc in enumerate(chain):
            game_t = game_enc.unsqueeze(0)
            v_t    = agent.net(game_t).squeeze()    # scalar, grad flows

            if t == len(chain) - 1:
                target = torch.tensor(float(reward), dtype=torch.float32)
            else:
                game_t1 = chain[t + 1].unsqueeze(0)
                with torch.no_grad():
                    target = agent.net(game_t1).squeeze()

            losses.append(criterion(v_t, target.detach()))

    if not losses:
        return 0.0, 0.0

    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()

    grad_norm = float(
        sum(p.grad.norm() ** 2 for p in agent.net.parameters() if p.grad is not None) ** 0.5
    )
    optimizer.step()

    return mean_loss.item(), grad_norm


# ── training ──────────────────────────────────────────────────────────────────

def train(out_dir: str, num_episodes: int, smoke: bool, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    opponents = build_standard_opponents(ROOT)
    sampler   = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)

    agent     = ValueDeepAgent()
    optimizer = optim.Adam(agent.net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    agent.set_train_mode(True)

    config = {
        "agent":            "ValueDeepAgent",
        "architecture":     "15→64→64→64→1",
        "num_episodes":     num_episodes,
        "learning_rate":    LR,
        "batch_size":       BATCH_SIZE,
        "ckpt_interval":    CKPT_INTERVAL,
        "opponent_weights": OPPONENT_WEIGHTS,
        "seed":             seed,
        "smoke":            smoke,
    }
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    freqs = sampler.expected_frequencies()
    print(f"\n{'='*60}")
    print(f"  DALI — value_based_deep  seed={seed}")
    print(f"  {'SMOKE' if smoke else 'FULL'} | episodes={num_episodes:,} "
          f"batch={BATCH_SIZE} lr={LR}")
    print(f"{'='*60}")
    print("Opponent sampling weights:")
    for k, p in freqs.items():
        print(f"  {k:<20s} {p:.1%}")
    print()

    batch_data  = []
    log_path    = os.path.join(out_dir, "train_log.jsonl")
    last_loss   = None
    last_gnorm  = None
    ep_hand     = 0   # round-robin seat alternation

    t0 = time.time()
    with open(log_path, "w") as log_file:
        for ep in range(1, num_episodes + 1):
            learner_id = ep % 2
            opp_name, opp = sampler.sample()

            chain, reward = play_hand_deep(agent, opp, learner_id)
            batch_data.append((chain, reward))

            ep_loss = ep_gnorm = None
            if len(batch_data) >= BATCH_SIZE:
                loss_val, gnorm_val = td_update(agent, optimizer, criterion, batch_data)
                batch_data.clear()
                last_loss  = loss_val
                last_gnorm = gnorm_val
                ep_loss    = round(loss_val, 6)
                ep_gnorm   = round(gnorm_val, 6)

            log_entry = {
                "ep":      ep,
                "opp":     opp_name,
                "reward":  float(reward),
                "loss":    ep_loss,
                "gnorm":   ep_gnorm,
            }
            log_file.write(json.dumps(log_entry) + "\n")

            if ep % CKPT_INTERVAL == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_ep{ep:06d}.pt")
                agent.save_model(ckpt_path)

                l_str  = f"{last_loss:.4f}"  if last_loss  is not None else "n/a"
                gn_str = f"{last_gnorm:.4f}" if last_gnorm is not None else "n/a"
                elapsed = time.time() - t0
                print(f"[ep={ep:,}] loss={l_str} gnorm={gn_str} opp={opp_name} "
                      f"({elapsed:.0f}s)")

    agent.save_model(os.path.join(out_dir, "checkpoint_final.pt"))
    print(f"\nDone in {time.time() - t0:.0f}s  →  {out_dir}/")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int,  default=0)
    parser.add_argument("--episodes", type=int,  default=None)
    parser.add_argument("--smoke",    action="store_true")
    args = parser.parse_args()

    smoke    = args.smoke
    episodes = args.episodes or (500 if smoke else NUM_EPISODES)
    out_dir  = os.path.join(HERE, "outputs", f"seed_{args.seed}")

    train(out_dir=out_dir, num_episodes=episodes, smoke=smoke, seed=args.seed)
