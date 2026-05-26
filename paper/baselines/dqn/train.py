"""
DQN training script.

Self-play training: agent plays both seats (learner_id = ep % 2).
Off-policy TD updates from a replay buffer after each episode.

Usage:
    python -m paper.baselines.dqn.train [--smoke]
"""

import argparse
import json
import os
import random
import sys
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

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LR = 1e-4
NUM_EPISODES = 200_000
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 10_000
TARGET_UPDATE_FREQ = 500   # episodes between target network hard copies
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 100_000    # episodes over which epsilon anneals linearly
PRINT_EVERY = 10_000
CKPT_EVERY = 10_000

SMOKE_EPISODES = 500


def get_epsilon(ep: int) -> float:
    return max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * ep / EPSILON_DECAY)


class ReplayBuffer:
    """Stores (state_enc, action_idx, reward, next_state_enc, done) tuples."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states, dim=0),           # (B, 15)
            torch.tensor(actions, dtype=torch.long),   # (B,)
            torch.tensor(rewards, dtype=torch.float32),  # (B,)
            torch.cat(next_states, dim=0),       # (B, 15)
            torch.tensor(dones, dtype=torch.float32),    # (B,)
        )

    def __len__(self):
        return len(self.buffer)


def play_episode(agent: DQNAgent, replay_buffer: ReplayBuffer, learner_id: int):
    """
    Play one full episode in self-play, storing learner transitions.
    Returns reward.
    """
    game = LeducGame()
    game.reset()

    # Track learner's current state so we can store transitions
    learner_pending = None  # (obs_enc, action_idx)

    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)
        enc = agent.encode_observation(obs, viewer_id=cp)

        # Opponent uses greedy (no training needed for opponent side)
        if cp == learner_id:
            action = agent.select_action(obs)
            action_idx = action.value

            # Resolve previous pending transition (non-terminal, reward=0)
            if learner_pending is not None:
                prev_enc, prev_action_idx = learner_pending
                replay_buffer.push(prev_enc, prev_action_idx, 0.0, enc, False)

            learner_pending = (enc, action_idx)
        else:
            # Opponent: use greedy policy (eval-like, but same agent)
            legal = obs.legal_actions
            q_vals = agent.q_net(enc).squeeze(0)
            mask = torch.full((3,), -1e9)
            for a in legal:
                mask[a.value] = 0.0
            masked_q = q_vals + mask
            action_idx = masked_q.argmax().item()
            action = Action(action_idx)

        game.step(action)

    reward = game.get_reward()[learner_id]

    # Resolve final pending transition (terminal)
    if learner_pending is not None:
        prev_enc, prev_action_idx = learner_pending
        zeros = torch.zeros_like(prev_enc)
        replay_buffer.push(prev_enc, prev_action_idx, float(reward), zeros, True)

    return reward


def update(agent: DQNAgent, optimizer: optim.Optimizer,
           replay_buffer: ReplayBuffer) -> float:
    """Sample a batch and do one gradient update. Returns loss."""
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Current Q-values
    q_vals = agent.q_net(states)                    # (B, 3)
    q_vals_taken = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # Target Q-values
    with torch.no_grad():
        next_q = agent.target_net(next_states)      # (B, 3)
        next_q_max = next_q.max(dim=1).values       # (B,)
        target = rewards + GAMMA * next_q_max * (1.0 - dones)

    loss = nn.functional.mse_loss(q_vals_taken, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(num_episodes: int, output_dir: str):
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    agent = DQNAgent()
    agent.set_train_mode(True)

    optimizer = optim.Adam(agent.q_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    log_path = os.path.join(output_dir, 'train_log.jsonl')
    log_file = open(log_path, 'w')

    running_reward = 0.0
    running_loss = 0.0
    running_loss_count = 0

    for ep in range(1, num_episodes + 1):
        epsilon = get_epsilon(ep)
        agent.epsilon = epsilon
        learner_id = ep % 2

        reward = play_episode(agent, replay_buffer, learner_id)

        loss_val = None
        if len(replay_buffer) >= BATCH_SIZE:
            loss_val = update(agent, optimizer, replay_buffer)
            running_loss += loss_val
            running_loss_count += 1

        # Hard update target network
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        running_reward += reward

        log_entry = {
            'ep': ep,
            'reward': reward,
            'loss': loss_val,
            'epsilon': epsilon,
            'opp': 'self_play',
        }
        log_file.write(json.dumps(log_entry) + '\n')

        if ep % PRINT_EVERY == 0:
            avg_r = running_reward / PRINT_EVERY
            avg_l = running_loss / max(running_loss_count, 1)
            print(f"[DQN] ep={ep}/{num_episodes}  avg_reward={avg_r:.4f}  "
                  f"avg_loss={avg_l:.4f}  epsilon={epsilon:.4f}")
            running_reward = 0.0
            running_loss = 0.0
            running_loss_count = 0

        if ep % CKPT_EVERY == 0:
            ckpt_path = os.path.join(output_dir, 'checkpoints', f'checkpoint_ep{ep:06d}.pt')
            agent.save_model(ckpt_path)

    final_path = os.path.join(output_dir, 'checkpoint_final.pt')
    agent.save_model(final_path)
    print(f"[DQN] Training complete. Final checkpoint: {final_path}")

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Run 500 episodes for smoke test')
    args = parser.parse_args()

    n_ep = SMOKE_EPISODES if args.smoke else NUM_EPISODES
    out_dir = os.path.join(HERE, 'outputs')

    print(f"[DQN] Starting training: {n_ep} episodes → {out_dir}")
    train(num_episodes=n_ep, output_dir=out_dir)
