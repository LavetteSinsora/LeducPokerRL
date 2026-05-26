"""
DQN v2 — pool training.

Replaces self-play with weighted opponent pool (same recipe as value_based_pool):
  CFR ×3, heuristic ×3, 6 rule-based ×1 each.

Algorithm unchanged: epsilon-greedy exploration, replay buffer, target network.
Episodes increased to 500K.

Usage:
    python -m paper.baselines.dqn.train_v2 [--smoke] [--seed N]
"""

import argparse
import json
import os
import random
import sys
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.leduc_game import Action, LeducGame
from paper.baselines.dqn.agent import DQNAgent
from paper.evaluation.comparison_protocol import build_standard_opponents
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS, WeightedOpponentSampler,
)

# ── hyperparameters ────────────────────────────────────────────────────────────
LR                 = 1e-4
NUM_EPISODES       = 500_000
GAMMA              = 0.99
BATCH_SIZE         = 32
BUFFER_SIZE        = 10_000
TARGET_UPDATE_FREQ = 500
EPSILON_START      = 1.0
EPSILON_END        = 0.05
EPSILON_DECAY      = 200_000   # anneal over first 200K episodes (2× v1)
PRINT_EVERY        = 20_000
CKPT_EVERY         = 50_000
SMOKE_EPS          = 500


def get_epsilon(ep: int) -> float:
    return max(EPSILON_END,
               EPSILON_START - (EPSILON_START - EPSILON_END) * ep / EPSILON_DECAY)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states, dim=0),
            torch.tensor(actions,  dtype=torch.long),
            torch.tensor(rewards,  dtype=torch.float32),
            torch.cat(next_states, dim=0),
            torch.tensor(dones,    dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def play_episode(agent: DQNAgent, replay_buffer: ReplayBuffer,
                 opponent, learner_id: int):
    game = LeducGame()
    game.reset()
    pending = None  # (enc, action_idx)

    while not game.is_finished:
        cp  = game.current_player
        obs = game.get_observation(viewer_id=cp)
        enc = agent.encode_observation(obs, viewer_id=cp)

        if cp == learner_id:
            action = agent.select_action(obs)  # epsilon-greedy via agent.epsilon
            action_idx = action.value

            if pending is not None:
                prev_enc, prev_idx = pending
                replay_buffer.push(prev_enc, prev_idx, 0.0, enc, False)

            pending = (enc, action_idx)
        else:
            action = opponent.select_action(obs)

        game.step(action)

    reward = game.get_reward()[learner_id]

    if pending is not None:
        prev_enc, prev_idx = pending
        zeros = torch.zeros_like(prev_enc)
        replay_buffer.push(prev_enc, prev_idx, float(reward), zeros, True)

    return reward


def update(agent: DQNAgent, optimizer, replay_buffer: ReplayBuffer) -> float:
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    q_taken = agent.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_max = agent.target_net(next_states).max(dim=1).values
        target = rewards + GAMMA * next_q_max * (1.0 - dones)

    loss = nn.functional.mse_loss(q_taken, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(num_episodes: int, output_dir: str, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    opponents     = build_standard_opponents(ROOT)
    sampler       = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    agent     = DQNAgent()
    agent.set_train_mode(True)
    optimizer = optim.Adam(agent.q_net.parameters(), lr=LR)

    log_file = open(os.path.join(output_dir, 'train_log.jsonl'), 'w')

    running_reward = 0.0
    running_loss   = 0.0
    loss_count     = 0
    t0 = time.time()

    for ep in range(1, num_episodes + 1):
        epsilon = get_epsilon(ep)
        agent.epsilon = epsilon
        learner_id         = ep % 2
        opp_name, opponent = sampler.sample()

        reward = play_episode(agent, replay_buffer, opponent, learner_id)

        loss_val = None
        if len(replay_buffer) >= BATCH_SIZE:
            loss_val = update(agent, optimizer, replay_buffer)
            running_loss += loss_val
            loss_count   += 1

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        running_reward += reward

        log_file.write(json.dumps({
            'ep': ep, 'opp': opp_name, 'reward': reward,
            'loss': loss_val, 'epsilon': round(epsilon, 4),
        }) + '\n')

        if ep % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            avg_l = running_loss / max(loss_count, 1)
            print(f"[DQN-v2] ep={ep:,}/{num_episodes:,}  "
                  f"avg_r={running_reward/PRINT_EVERY:+.4f}  "
                  f"avg_loss={avg_l:.4f}  eps={epsilon:.3f}  "
                  f"({elapsed:.0f}s)")
            running_reward = 0.0
            running_loss   = 0.0
            loss_count     = 0

        if ep % CKPT_EVERY == 0:
            agent.save_model(os.path.join(output_dir, 'checkpoints', f'ep{ep:06d}.pt'))

    agent.save_model(os.path.join(output_dir, 'checkpoint_final.pt'))
    log_file.close()
    print(f"[DQN-v2] Done → {output_dir}  ({time.time()-t0:.0f}s total)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed',  type=int, default=0)
    args = parser.parse_args()

    n_ep    = SMOKE_EPS if args.smoke else NUM_EPISODES
    out_dir = os.path.join(HERE, 'outputs_v2', f'seed_{args.seed}')
    print(f"[DQN-v2] seed={args.seed}  episodes={n_ep:,}  → {out_dir}")
    train(n_ep, out_dir, args.seed)
