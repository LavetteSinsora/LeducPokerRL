"""
REINFORCE v2 — pool training.

Replaces self-play with weighted opponent pool (same recipe as value_based_pool):
  CFR ×3, heuristic ×3, 6 rule-based ×1 each.

Fixes from v1:
  - Pool opponents instead of self-play
  - Return normalization guard: skip normalize when T==1 (avoids div-by-zero / zero-gradient)
  - 500K episodes

Usage:
    python -m paper.baselines.reinforce.train_v2 [--smoke] [--seed N]
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.leduc_game import Action, LeducGame
from paper.baselines.reinforce.agent import REINFORCEAgent
from paper.evaluation.comparison_protocol import build_standard_opponents
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS, WeightedOpponentSampler,
)

# ── hyperparameters ────────────────────────────────────────────────────────────
LR            = 1e-4
NUM_EPISODES  = 500_000
ENTROPY_COEF  = 0.01
PRINT_EVERY   = 20_000
CKPT_EVERY    = 50_000
SMOKE_EPS     = 500


def play_episode(agent: REINFORCEAgent, opponent, learner_id: int):
    """
    Play one hand vs pool opponent.
    Returns (trajectory, reward) for the learner.
    trajectory: list of (enc, action_idx, log_prob) at learner's turns.
    """
    game = LeducGame()
    game.reset()
    trajectory = []
    opp_id = 1 - learner_id

    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)

        if cp == learner_id:
            enc = agent.encode_observation(obs, viewer_id=cp)
            logits = agent.policy_net(enc).squeeze(0)
            mask = torch.full((3,), -1e9)
            for a in obs.legal_actions:
                mask[a.value] = 0.0
            probs = torch.softmax(logits + mask, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            trajectory.append((enc, action_idx, log_prob))
            action = Action(action_idx.item())
        else:
            action = opponent.select_action(obs)

        game.step(action)

    reward = game.get_reward()[learner_id]
    return trajectory, reward


def train(num_episodes: int, output_dir: str, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    opponents = build_standard_opponents(ROOT)
    sampler   = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)

    agent     = REINFORCEAgent()
    agent.set_train_mode(True)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=LR)

    log_path = os.path.join(output_dir, 'train_log.jsonl')
    log_file = open(log_path, 'w')

    running_reward = 0.0
    running_loss   = 0.0
    t0 = time.time()

    for ep in range(1, num_episodes + 1):
        learner_id        = ep % 2
        opp_name, opponent = sampler.sample()

        trajectory, reward = play_episode(agent, opponent, learner_id)

        loss_val = None
        if len(trajectory) > 0:
            T = len(trajectory)
            G = torch.tensor([float(reward)] * T)

            # Normalize only when T > 1 and std > 0; single-step episodes get raw G
            if T > 1:
                std = G.std()
                if std > 1e-8:
                    G = (G - G.mean()) / std

            log_probs = torch.stack([lp for (_, _, lp) in trajectory])

            # Entropy bonus (recompute probs)
            entropy_list = []
            for (enc, _, _) in trajectory:
                logits = agent.policy_net(enc).squeeze(0)
                probs  = torch.softmax(logits, dim=-1)
                entropy_list.append(-(probs * torch.log(probs + 1e-8)).sum())
            entropy = torch.stack(entropy_list).mean()

            pg_loss  = -(log_probs * G).mean()
            loss     = pg_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

        running_reward += reward
        running_loss   += (loss_val or 0.0)

        log_file.write(json.dumps({
            'ep': ep, 'opp': opp_name, 'reward': reward, 'loss': loss_val,
        }) + '\n')

        if ep % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            print(f"[REINFORCE-v2] ep={ep:,}/{num_episodes:,}  "
                  f"avg_r={running_reward/PRINT_EVERY:+.4f}  "
                  f"avg_loss={running_loss/PRINT_EVERY:.4f}  "
                  f"({elapsed:.0f}s)")
            running_reward = 0.0
            running_loss   = 0.0

        if ep % CKPT_EVERY == 0:
            agent.save_model(os.path.join(output_dir, 'checkpoints', f'ep{ep:06d}.pt'))

    agent.save_model(os.path.join(output_dir, 'checkpoint_final.pt'))
    log_file.close()
    print(f"[REINFORCE-v2] Done → {output_dir}  ({time.time()-t0:.0f}s total)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed',  type=int, default=0)
    args = parser.parse_args()

    n_ep    = SMOKE_EPS if args.smoke else NUM_EPISODES
    out_dir = os.path.join(HERE, 'outputs_v2', f'seed_{args.seed}')
    print(f"[REINFORCE-v2] seed={args.seed}  episodes={n_ep:,}  → {out_dir}")
    train(n_ep, out_dir, args.seed)
