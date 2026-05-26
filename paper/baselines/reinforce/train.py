"""
REINFORCE training script.

Self-play training: agent plays both seats (learner_id = ep % 2).
Full-episode Monte Carlo returns, entropy bonus, one optimizer step per episode.

Usage:
    python -m paper.baselines.reinforce.train [--smoke]
"""

import argparse
import json
import os
import sys

import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.leduc_game import Action, LeducGame
from paper.baselines.reinforce.agent import REINFORCEAgent

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LR = 1e-4
NUM_EPISODES = 200_000
GAMMA = 1.0          # Short episodes — no discounting
ENTROPY_COEF = 0.01
PRINT_EVERY = 10_000
CKPT_EVERY = 10_000

SMOKE_EPISODES = 500


def play_episode(agent: REINFORCEAgent, learner_id: int):
    """
    Play one full episode in self-play.
    Returns (trajectory, reward) where trajectory is a list of
    (state_enc, action_idx, log_prob) tuples for the learner's turns.
    """
    game = LeducGame()
    game.reset()
    trajectory = []  # (state_enc, action_idx, log_prob)

    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)
        enc = agent.encode_observation(obs, viewer_id=cp)  # (1, 15)

        logits = agent.policy_net(enc).squeeze(0)  # (3,)
        mask = torch.full((3,), -1e9)
        for a in obs.legal_actions:
            mask[a.value] = 0.0
        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if cp == learner_id:
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            trajectory.append((enc, action_idx, log_prob))
        else:
            action_idx = dist.sample()

        action = Action(action_idx.item())
        game.step(action)

    reward = game.get_reward()[learner_id]
    return trajectory, reward


def compute_returns(rewards_scalar, T):
    """Compute discounted returns for a uniform reward at episode end."""
    # In Leduc each step doesn't give an intermediate reward, only terminal.
    # We assign the terminal reward to every step in the trajectory.
    return [rewards_scalar] * T


def train(num_episodes: int, output_dir: str):
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    agent = REINFORCEAgent()
    agent.set_train_mode(True)

    optimizer = optim.Adam(agent.policy_net.parameters(), lr=LR)

    log_path = os.path.join(output_dir, 'train_log.jsonl')
    log_file = open(log_path, 'w')

    running_reward = 0.0
    running_loss = 0.0

    for ep in range(1, num_episodes + 1):
        learner_id = ep % 2
        trajectory, reward = play_episode(agent, learner_id)

        loss_val = None

        if len(trajectory) > 0:
            T = len(trajectory)
            # All steps receive the terminal reward (sparse reward signal)
            returns = compute_returns(reward, T)
            G = torch.tensor(returns, dtype=torch.float32)

            # Normalize returns
            if T > 1:
                G = (G - G.mean()) / (G.std() + 1e-8)

            # Collect log_probs and compute entropy bonus
            log_probs = torch.stack([lp for (_, _, lp) in trajectory])

            # Compute entropy for each step (re-forward for entropy)
            entropy_list = []
            for (enc, _, _) in trajectory:
                logits = agent.policy_net(enc).squeeze(0)
                # We don't have per-step legal actions here easily,
                # so just compute entropy over full distribution
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * torch.log(probs + 1e-8)).sum()
                entropy_list.append(ent)
            entropy = torch.stack(entropy_list).mean()

            pg_loss = -(log_probs * G).mean()
            loss = pg_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

        running_reward += reward
        running_loss += (loss_val if loss_val is not None else 0.0)

        # Log
        log_entry = {'ep': ep, 'reward': reward, 'loss': loss_val, 'opp': 'self_play'}
        log_file.write(json.dumps(log_entry) + '\n')

        # Print progress
        if ep % PRINT_EVERY == 0:
            avg_r = running_reward / PRINT_EVERY
            avg_l = running_loss / PRINT_EVERY
            print(f"[REINFORCE] ep={ep}/{num_episodes}  avg_reward={avg_r:.4f}  avg_loss={avg_l:.4f}")
            running_reward = 0.0
            running_loss = 0.0

        # Checkpoint
        if ep % CKPT_EVERY == 0:
            ckpt_path = os.path.join(output_dir, 'checkpoints', f'checkpoint_ep{ep:06d}.pt')
            agent.save_model(ckpt_path)

    # Final checkpoint
    final_path = os.path.join(output_dir, 'checkpoint_final.pt')
    agent.save_model(final_path)
    print(f"[REINFORCE] Training complete. Final checkpoint: {final_path}")

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Run 500 episodes for smoke test')
    args = parser.parse_args()

    n_ep = SMOKE_EPISODES if args.smoke else NUM_EPISODES
    out_dir = os.path.join(HERE, 'outputs')

    print(f"[REINFORCE] Starting training: {n_ep} episodes → {out_dir}")
    train(num_episodes=n_ep, output_dir=out_dir)
