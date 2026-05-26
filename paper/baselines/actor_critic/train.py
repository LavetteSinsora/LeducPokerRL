"""
Actor-Critic (A2C) training script.

Self-play training: agent plays both seats (learner_id = ep % 2).
Online TD(0) updates at every learner action step.

Usage:
    python -m paper.baselines.actor_critic.train [--smoke]
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
from paper.baselines.actor_critic.agent import ActorCriticAgent

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LR = 1e-4
NUM_EPISODES = 200_000
GAMMA = 0.99
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
PRINT_EVERY = 10_000
CKPT_EVERY = 10_000

SMOKE_EPISODES = 500


def train_step(agent: ActorCriticAgent, optimizer: optim.Optimizer,
               obs_enc: torch.Tensor, action_idx: int,
               reward: float, next_obs_enc: torch.Tensor,
               done: bool, legal_actions, gamma: float = GAMMA):
    """
    Single TD(0) update step.
    Returns loss scalar.
    """
    # Forward current state
    logits, value = agent.forward(obs_enc)
    logits = logits.squeeze(0)
    value = value.squeeze()

    # Next state value (no grad)
    with torch.no_grad():
        if done:
            next_value = torch.tensor(0.0)
        else:
            _, nv = agent.forward(next_obs_enc)
            next_value = nv.squeeze()

    advantage = reward + gamma * next_value - value

    # Policy loss with legal action masking
    mask = torch.full((3,), -1e9)
    for a in legal_actions:
        mask[a.value] = 0.0
    masked_logits = logits + mask
    log_prob = torch.log_softmax(masked_logits, dim=-1)[action_idx]

    policy_loss = -log_prob * advantage.detach()
    value_loss = advantage.pow(2)

    # Entropy bonus
    probs = torch.softmax(masked_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum()

    loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def play_episode(agent: ActorCriticAgent, optimizer: optim.Optimizer, learner_id: int):
    """
    Play one full episode in self-play, calling train_step at each learner action.
    Returns (total_loss, reward, num_steps).
    """
    game = LeducGame()
    game.reset()
    total_loss = 0.0
    num_steps = 0

    # We need to track learner's last (obs_enc, action_idx, legal_actions)
    # to do online TD update
    pending = None  # (obs_enc, action_idx, legal_actions)

    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)
        enc = agent.encode_observation(obs, viewer_id=cp)

        logits, _ = agent.forward(enc)
        logits = logits.squeeze(0)
        mask = torch.full((3,), -1e9)
        for a in obs.legal_actions:
            mask[a.value] = 0.0
        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if cp == learner_id:
            action_idx = dist.sample().item()

            # If there's a pending transition, update it now (non-terminal)
            if pending is not None:
                prev_enc, prev_action, prev_legal = pending
                loss = train_step(
                    agent, optimizer,
                    obs_enc=prev_enc, action_idx=prev_action,
                    reward=0.0, next_obs_enc=enc,
                    done=False, legal_actions=prev_legal,
                )
                total_loss += loss
                num_steps += 1

            pending = (enc, action_idx, obs.legal_actions)
        else:
            action_idx = dist.sample().item()

        action = Action(action_idx)
        game.step(action)

    # Episode ended — resolve pending transition with terminal reward
    reward = game.get_reward()[learner_id]

    if pending is not None:
        prev_enc, prev_action, prev_legal = pending
        # Terminal: next_obs_enc is zeros, done=True
        zeros = torch.zeros_like(prev_enc)
        loss = train_step(
            agent, optimizer,
            obs_enc=prev_enc, action_idx=prev_action,
            reward=float(reward), next_obs_enc=zeros,
            done=True, legal_actions=prev_legal,
        )
        total_loss += loss
        num_steps += 1

    avg_loss = total_loss / max(num_steps, 1)
    return avg_loss, reward


def train(num_episodes: int, output_dir: str):
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    agent = ActorCriticAgent()
    agent.set_train_mode(True)

    optimizer = optim.Adam(agent.parameters(), lr=LR)

    log_path = os.path.join(output_dir, 'train_log.jsonl')
    log_file = open(log_path, 'w')

    running_reward = 0.0
    running_loss = 0.0

    for ep in range(1, num_episodes + 1):
        learner_id = ep % 2
        avg_loss, reward = play_episode(agent, optimizer, learner_id)

        running_reward += reward
        running_loss += avg_loss

        log_entry = {'ep': ep, 'reward': reward, 'loss': avg_loss, 'opp': 'self_play'}
        log_file.write(json.dumps(log_entry) + '\n')

        if ep % PRINT_EVERY == 0:
            avg_r = running_reward / PRINT_EVERY
            avg_l = running_loss / PRINT_EVERY
            print(f"[A2C] ep={ep}/{num_episodes}  avg_reward={avg_r:.4f}  avg_loss={avg_l:.4f}")
            running_reward = 0.0
            running_loss = 0.0

        if ep % CKPT_EVERY == 0:
            ckpt_path = os.path.join(output_dir, 'checkpoints', f'checkpoint_ep{ep:06d}.pt')
            agent.save_model(ckpt_path)

    final_path = os.path.join(output_dir, 'checkpoint_final.pt')
    agent.save_model(final_path)
    print(f"[A2C] Training complete. Final checkpoint: {final_path}")

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Run 500 episodes for smoke test')
    args = parser.parse_args()

    n_ep = SMOKE_EPISODES if args.smoke else NUM_EPISODES
    out_dir = os.path.join(HERE, 'outputs')

    print(f"[A2C] Starting training: {n_ep} episodes → {out_dir}")
    train(num_episodes=n_ep, output_dir=out_dir)
