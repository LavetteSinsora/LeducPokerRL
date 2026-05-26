"""
opponent_value_tables_v1 — Training Script
==========================================
Trains a separate ValueBasedAgent against each of the 6 rule-based opponents,
yielding per-opponent approximate ground-truth value functions V(s | opponent).

Each agent is trained with TD(0) where the opponent is fixed and non-learning.
The learning agent alternates positions (P0 / P1) across episodes for balance.

Usage:
    python train.py                              # train all 6 opponents sequentially
    python train.py --opponent tight_passive     # train one opponent
    python train.py --opponent maniac --resume   # resume from checkpoint
    python train.py --smoke                      # 500-episode pipeline validation
    python train.py --episodes 300000            # custom budget per opponent

Outputs per opponent in outputs/{opponent_name}/:
    checkpoint.pt         — final model weights
    checkpoint_best.pt    — best checkpoint by eval score vs training opponent
    train_config.json     — hyperparameters
    train_history.json    — [{episode, loss}, ...]
    eval_history.json     — [{episode, score}, ...]
    results.json          — STANDARDS §3 required keys
    training_curve.png    — EMA-smoothed TD loss
    eval_curve.png        — avg chips/round vs training opponent over training
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import torch

from agents.value_based.agent import ValueBasedAgent
from agents.value_based.trainer import SelfPlayTrainer
from agents.evaluation import quick_evaluate
from engine.leduc_game import LeducGame

from agents.rule_based import ALL_AGENTS

OUT_ROOT     = os.path.join(HERE, "outputs")
OPPONENT_KEYS = list(ALL_AGENTS.keys())

CONFIG_BASE = {
    "experiment_id":     "opponent_value_tables_v1",
    "architecture":      "ValueNetwork(input_size=15, hidden_size=64)",
    "input_size":        15,
    "hidden_size":       64,
    "learning_rate":     1e-4,
    "batch_size":        32,
    "temperature":       1.0,
    "training":          "TD(0) fixed-opponent play, MSELoss",
    "optimizer":         "Adam",
    "eval_interval":     500,
    "eval_rounds":       200,
    "final_eval_rounds": 5000,
}

FLUSH_EVERY = 500


# ── FixedOpponentTrainer ───────────────────────────────────────────────────────

class FixedOpponentTrainer(SelfPlayTrainer):
    """TD(0) trainer where the learning agent plays against a fixed, non-learning opponent.

    The agent alternates positions (P0 / P1) across episodes for position balance.
    Only the learning agent's post-action state chain enters the TD update;
    the opponent's actions merely advance the game state.

    update_model() is inherited unchanged from SelfPlayTrainer — it handles chains
    with one empty slot (the opponent's side) correctly.
    """

    def __init__(self, agent: ValueBasedAgent, opponent, learning_rate: float = 1e-4):
        super().__init__(agent, learning_rate=learning_rate)
        self.opponent = opponent
        self.episode_count = 0
        # Suppress BaseTrainer's built-in periodic evaluation; we handle it ourselves
        self.eval_interval = 10 ** 9

    def collect_episode(self):
        """Play one hand: learning agent vs fixed opponent, alternating positions."""
        self.episode_count += 1
        agent_pos = self.episode_count % 2   # 0 or 1

        self.game.reset()
        chain = []

        while not self.game.is_finished:
            cp  = self.game.current_player
            obs = self.game.get_observation(viewer_id=cp)

            if cp == agent_pos:
                action = self.agent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]
                post_obs, _ = LeducGame.simulate_action(obs, action)
                encoded = self.agent.encode_observation(post_obs, viewer_id=cp)
                chain.append(encoded)
            else:
                action = self.opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]

            self.game.step(action)

        rewards = self.game.get_reward()
        chains = [[], []]
        chains[agent_pos] = chain
        return chains, rewards


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _ema(values, alpha=0.95):
    s = values[0]
    out = []
    for v in values:
        s = alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def _convergence_check(losses):
    n = len(losses)
    if n < 10:
        return False, float("inf")
    seg = max(1, n // 5)
    prev = sum(losses[-2 * seg:-seg]) / seg
    last = sum(losses[-seg:]) / seg
    if prev == 0:
        return True, 0.0
    pct = abs(prev - last) / abs(prev) * 100
    return pct < 5.0, pct


def _find_saturation_episode(eval_records, window=10, threshold=0.02):
    scores   = [r.get("score")   for r in eval_records if r.get("score")   is not None]
    episodes = [r["episode"]     for r in eval_records if r.get("score")   is not None]
    if len(scores) < window:
        return None
    for i in range(window, len(scores)):
        w = scores[i - window:i]
        if max(w) - min(w) < threshold:
            return episodes[i - 1]
    return None


def generate_plots(out_dir, opp_key, train_history, eval_history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    if train_history:
        episodes = [r["episode"] for r in train_history]
        losses   = [r["loss"]    for r in train_history]
        smoothed = _ema(losses)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(episodes, losses,   color="lightblue", alpha=0.35, linewidth=0.6, label="raw loss")
        ax.plot(episodes, smoothed, color="steelblue",              linewidth=1.5, label="EMA (α=0.95)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("TD MSE Loss")
        ax.set_title(f"opponent_value_tables_v1 [{opp_key}] — Training Loss (TD MSE)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
        plt.close(fig)

    if eval_history:
        episodes = [r["episode"] for r in eval_history]
        scores   = [r["score"]   for r in eval_history]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(episodes, scores,         color="steelblue", alpha=0.25, linewidth=0.6, label="raw")
        ax.plot(episodes, _ema(scores),   color="steelblue", linewidth=1.8,
                label=f"vs {opp_key} (EMA)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Chips / Round")
        ax.set_title(f"opponent_value_tables_v1 [{opp_key}] — Eval vs Training Opponent")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "eval_curve.png"), dpi=150)
        plt.close(fig)

    print(f"  Plots saved.")


# ── per-opponent training ──────────────────────────────────────────────────────

def train_one(opp_key, num_episodes, resume=False, smoke=False):
    out_dir = os.path.join(OUT_ROOT, opp_key)
    os.makedirs(out_dir, exist_ok=True)

    opponent = ALL_AGENTS[opp_key]()
    opponent.set_train_mode(False)

    agent = ValueBasedAgent(temperature=CONFIG_BASE["temperature"])
    start_episode = 0

    if resume:
        ckpt = os.path.join(out_dir, "checkpoint.pt")
        if not os.path.exists(ckpt):
            print(f"  No checkpoint at {ckpt}. Run without --resume first.")
            return
        agent.load_model(ckpt)
        th = _load_json(os.path.join(out_dir, "train_history.json"), [])
        start_episode = th[-1]["episode"] if th else 0
        print(f"  Resuming from episode {start_episode:,}")

    trainer = FixedOpponentTrainer(agent, opponent,
                                   learning_rate=CONFIG_BASE["learning_rate"])

    config = {
        **CONFIG_BASE,
        "opponent":       opp_key,
        "num_episodes":   num_episodes,
        "start_episode":  start_episode,
        "total_episodes": start_episode + num_episodes,
        "resume":         resume,
    }
    _write_json(os.path.join(out_dir, "train_config.json"), config)

    train_history_path = os.path.join(out_dir, "train_history.json")
    eval_history_path  = os.path.join(out_dir, "eval_history.json")

    train_history = _load_json(train_history_path, []) if resume else []
    eval_history  = _load_json(eval_history_path,  []) if resume else []

    new_train = []
    new_eval  = []

    best_score       = max((r.get("score", float("-inf")) for r in eval_history),
                           default=float("-inf"))
    last_eval_bucket = start_episode // CONFIG_BASE["eval_interval"]

    def _flush():
        if new_train:
            train_history.extend(new_train)
            new_train.clear()
            _write_json(train_history_path, train_history)
        if new_eval:
            eval_history.extend(new_eval)
            new_eval.clear()
            _write_json(eval_history_path, eval_history)

    def callback(event):
        nonlocal best_score, last_eval_bucket

        if event["type"] != "batch_update":
            return

        new_train.append({"episode": event["episode"], "loss": round(event["loss"], 6)})
        if len(new_train) >= FLUSH_EVERY:
            _flush()

        bucket = event["episode"] // CONFIG_BASE["eval_interval"]
        if bucket > last_eval_bucket:
            last_eval_bucket = bucket
            ep = event["episode"]

            agent.set_train_mode(False)
            score = round(
                quick_evaluate(agent, opponent, num_rounds=CONFIG_BASE["eval_rounds"]), 4
            )
            agent.set_train_mode(True)

            new_eval.append({"episode": ep, "score": score})
            if len(new_eval) >= FLUSH_EVERY:
                _flush()

            if score > best_score:
                best_score = score
                torch.save(agent.model.state_dict(),
                           os.path.join(out_dir, "checkpoint_best.pt"))

            print(f"    [ep={ep:>7,}]  vs {opp_key}: {score:+.3f}  [best: {best_score:+.3f}]")

    t0 = time.time()
    trainer.train(
        num_episodes=num_episodes,
        batch_size=CONFIG_BASE["batch_size"],
        save_path=None,
        callback=callback,
        start_episode=start_episode,
    )
    elapsed = time.time() - t0
    _flush()

    torch.save(agent.model.state_dict(), os.path.join(out_dir, "checkpoint.pt"))

    # ── convergence + results ──────────────────────────────────────────────
    all_losses = [r["loss"] for r in train_history]
    converged, pct = _convergence_check(all_losses)
    loss_final = all_losses[-1] if all_losses else 0.0
    total_ep   = start_episode + num_episodes

    agent.set_train_mode(False)
    final_score = round(
        quick_evaluate(agent, opponent, CONFIG_BASE["eval_rounds"]), 4
    )
    peak_score = max((r.get("score", float("-inf")) for r in eval_history),
                     default=final_score)
    sat_ep = _find_saturation_episode(eval_history)

    results = {
        "experiment_id":      "opponent_value_tables_v1",
        "opponent":           opp_key,
        "training_episodes":  total_ep,
        "converged":          converged,
        "loss_plateau_pct":   round(pct, 2),
        "peak_eval_score":    round(peak_score, 4),
        "final_eval_score":   round(final_score, 4),
        "eval_opponent":      opp_key,
        "eval_rounds":        CONFIG_BASE["eval_rounds"],
        "loss_final":         round(loss_final, 6),
        "loss_components":    {"td_mse": round(loss_final, 6)},
        "saturation_episode": sat_ep,
        "elapsed_seconds":    round(elapsed, 1),
        "representation_metrics": {
            "effective_dim_80": None, "effective_dim_90": None,
            "reward_spearman_rho_pairwise": None,
            "hand_probe_accuracy": None, "hand_probe_chance": 0.333,
        },
        "notes": (
            f"Per-opponent ground-truth value network trained against "
            f"fixed {opp_key} opponent. Intended as approximate EV ground truth "
            f"for future analysis."
        ),
    }
    _write_json(os.path.join(out_dir, "results.json"), results)

    print(f"  [{opp_key}] {elapsed:.1f}s — "
          f"peak: {peak_score:+.3f}  final: {final_score:+.3f}  "
          f"converged: {converged}  sat_ep: {sat_ep}")

    if not smoke:
        generate_plots(out_dir, opp_key, train_history, eval_history)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=OPPONENT_KEYS, default=None,
                        help="Train one specific opponent (default: all 6)")
    parser.add_argument("--smoke",    action="store_true",
                        help="500-episode pipeline validation (all opponents)")
    parser.add_argument("--resume",   action="store_true",
                        help="Continue training from checkpoint")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodes per opponent (default: 200K)")
    args = parser.parse_args()

    if args.smoke:
        num_episodes = 500
    elif args.episodes is not None:
        num_episodes = args.episodes
    else:
        num_episodes = 200_000

    targets = [args.opponent] if args.opponent else OPPONENT_KEYS

    print(f"\n{'='*65}")
    mode = "SMOKE" if args.smoke else ("RESUME" if args.resume else "FRESH")
    print(f"  opponent_value_tables_v1  |  {mode}")
    print(f"  opponents : {targets}")
    print(f"  episodes  : {num_episodes:,} each")
    print(f"  eval every: {CONFIG_BASE['eval_interval']} episodes  "
          f"({CONFIG_BASE['eval_rounds']} rounds each)")
    print(f"{'='*65}\n")

    for opp_key in targets:
        print(f"\n── Training vs {opp_key} ──")
        train_one(opp_key, num_episodes, resume=args.resume, smoke=args.smoke)

    print("\nAll done. Run eval.py for the final 5000-round comprehensive evaluation.")


if __name__ == "__main__":
    main()
