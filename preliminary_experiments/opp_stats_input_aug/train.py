"""
opp_stats_input_augmentation_v1 — Training Script
================================
Trains StatAugValueAgent (22-dim input: 15 game + 7 opponent stats) using
TD(0) self-play or opponent-pool training.

Variants:
  self_play    — agent vs itself (300K episodes). Stats track the agent's own
                 behavior in the opposite seat. Baseline-comparable setup.
  pool_random  — each session (100 rounds), randomly sample one of 8 opponents.
                 Stats reset per session. 300K episodes total.
  pool_seq     — train K=500 sessions against each of 8 opponents sequentially.
                 Stats reset between opponents. ~400K total episodes.

Usage:
  python train.py --variant self_play
  python train.py --variant pool_random
  python train.py --variant pool_seq --k 500
  python train.py --variant self_play --smoke        # 500-ep pipeline check
"""

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from paper.evaluation.comparison_protocol import (
    STANDARD_OPPONENT_KEYS,
    build_standard_opponents,
    evaluate_stat_aware_pool,
    format_pool_summary,
    seed_best_metric,
)

from paper.evaluation.shared.stats_tracker import (
    OpponentStatsTracker, compute_pool_means, play_hand,
)
from preliminary_experiments.opp_stats_input_aug.agent import StatAugValueAgent

# ── constants ─────────────────────────────────────────────────────────────────
SESSION_LENGTH  = 100    # hands per session (resets stats)
PRIOR_STRENGTH  = 20.0
EVAL_INTERVAL   = 100    # episodes between quick evals
EVAL_ROUNDS     = 200    # rounds per quick eval
BATCH_SIZE      = 32
LR              = 1e-4
FLUSH_EVERY     = 500
CHECKPOINT_METRIC = "robustness"

OPPONENT_KEYS = list(STANDARD_OPPONENT_KEYS)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


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
    scores   = [r.get("heuristic") for r in eval_records if r.get("heuristic") is not None]
    episodes = [r["episode"]       for r in eval_records if r.get("heuristic") is not None]
    if len(scores) < window:
        return None
    for i in range(window, len(scores)):
        w = scores[i - window:i]
        if max(w) - min(w) < threshold:
            return episodes[i - 1]
    return None


# ── TD(0) update ──────────────────────────────────────────────────────────────

def update_model(agent, optimizer, criterion, batch_data):
    """TD(0) on per-learner post-action state chains."""
    optimizer.zero_grad()
    losses = []
    for chain, reward in batch_data:
        if not chain:
            continue
        for t in range(len(chain)):
            pred = agent.model(chain[t]).squeeze(0)
            if t == len(chain) - 1:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = agent.model(chain[t + 1]).squeeze(0)
            losses.append(criterion(pred, target))
    if not losses:
        return 0.0
    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()


# ── quick eval (uses neutral stats) ──────────────────────────────────────────

def quick_eval_stats(agent, opponents, eval_rounds, pool_means):
    """Evaluate with both seats and 100-hand session resets to match training."""
    return evaluate_stat_aware_pool(
        agent=agent,
        opponents=opponents,
        play_hand_fn=play_hand,
        pool_means=pool_means,
        num_rounds=eval_rounds,
        session_length=SESSION_LENGTH,
        prior_strength=PRIOR_STRENGTH,
        opponent_keys=OPPONENT_KEYS,
        alternate_positions=True,
    )


def generate_plots(train_history, eval_history, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    if train_history:
        ep = [r["episode"] for r in train_history]
        lo = [r["loss"]    for r in train_history]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ep, lo,         color="lightblue", alpha=0.35, linewidth=0.6, label="raw loss")
        ax.plot(ep, _ema(lo),   color="steelblue",              linewidth=1.5, label="EMA (α=0.95)")
        ax.set_xlabel("Episode"); ax.set_ylabel("TD MSE Loss")
        ax.set_title("opp_stats_input_augmentation_v1 — Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
        plt.close(fig)
        print("Saved training_curve.png")

    if eval_history:
        COLORS_TOP = {"heuristic": "firebrick", "cfr": "forestgreen"}
        COLORS_RB  = {
            "tight_passive": "#1f77b4", "tight_aggressive": "#ff7f0e",
            "loose_passive": "#9467bd", "loose_aggressive": "#e377c2",
            "maniac": "#8c564b",        "random": "#7f7f7f",
        }
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        for key, color in COLORS_TOP.items():
            pts = [(r["episode"], r[key]) for r in eval_history if key in r]
            if not pts: continue
            ep_k, sc_k = zip(*pts)
            ax1.plot(ep_k, sc_k,             color=color, alpha=0.25, linewidth=0.6)
            ax1.plot(ep_k, _ema(list(sc_k)), color=color, linewidth=1.8, label=key)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax1.set_ylabel("Avg Chips / Round")
        ax1.set_title("opp_stats_input_augmentation_v1 — Eval vs Heuristic & CFR"); ax1.legend(); ax1.grid(True, alpha=0.3)
        for key, color in COLORS_RB.items():
            pts = [(r["episode"], r[key]) for r in eval_history if key in r]
            if not pts: continue
            ep_k, sc_k = zip(*pts)
            ax2.plot(ep_k, sc_k,             color=color, alpha=0.25, linewidth=0.6)
            ax2.plot(ep_k, _ema(list(sc_k)), color=color, linewidth=1.8, label=key)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Avg Chips / Round")
        ax2.set_title("opp_stats_input_augmentation_v1 — Eval vs Rule-Based Agents")
        ax2.legend(loc="upper left", ncol=2); ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "eval_curve.png"), dpi=150)
        plt.close(fig)
        print("Saved eval_curve.png")


# ── main training loop ────────────────────────────────────────────────────────

def run_training(
    variant,
    out_dir,
    num_episodes,
    k_sessions,
    smoke,
    resume=False,
    eval_interval=EVAL_INTERVAL,
    eval_rounds=EVAL_ROUNDS,
    flush_every=FLUSH_EVERY,
):
    os.makedirs(out_dir, exist_ok=True)

    opponents = build_standard_opponents(ROOT)

    # ── calibrate pool priors ─────────────────────────────────────────────────
    priors_path = os.path.join(out_dir, "pool_priors.json")
    if os.path.exists(priors_path):
        pool_means = json.load(open(priors_path))
        print(f"Loaded pool priors from {priors_path}")
    else:
        print("Running calibration to compute pool priors...")
        calibration_hands = 100 if smoke else 500
        pool_means = compute_pool_means(opponents, calibration_hands)
        _write_json(priors_path, pool_means)

    # ── init agent + optimizer ────────────────────────────────────────────────
    agent     = StatAugValueAgent(temperature=1.0)
    optimizer = optim.Adam(agent.model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_hist_path = os.path.join(out_dir, "train_history.json")
    eval_hist_path  = os.path.join(out_dir, "eval_history.json")

    # ── resume: load checkpoint + existing histories ──────────────────────────
    start_episode = 0
    if resume:
        ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        agent.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Resumed from {ckpt_path}")
        train_history = _load_json(train_hist_path, [])
        eval_history  = _load_json(eval_hist_path,  [])
        if train_history:
            start_episode = train_history[-1]["episode"]
            print(f"Continuing from episode {start_episode:,} (+{num_episodes:,} more)")
    else:
        train_history = []
        eval_history  = []

    agent.set_train_mode(True)

    # ── build opponent schedule ───────────────────────────────────────────────
    ordered_opps = [opponents[k] for k in OPPONENT_KEYS]

    # ── save config ───────────────────────────────────────────────────────────
    config = {
        "experiment_id": f"opp_stats_input_augmentation_v1_{variant}",
        "variant":       variant,
        "architecture":  "StatAugValueAgent(input=22, hidden=64)",
        "input_size":    22, "hidden_size": 64,
        "learning_rate": LR, "batch_size": BATCH_SIZE,
        "temperature":   1.0, "prior_strength": PRIOR_STRENGTH,
        "session_length": SESSION_LENGTH,
        "num_episodes":  start_episode + num_episodes,
        "k_sessions":    k_sessions,
        "eval_interval": eval_interval, "eval_rounds": eval_rounds,
        "flush_every":   flush_every,
        "opponents":     OPPONENT_KEYS,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "seat_protocol": "balanced_alternating_training",
        "eval_protocol": "both_seats_with_100_hand_resets",
        "resumed_from":  start_episode if resume else None,
    }
    _write_json(os.path.join(out_dir, "train_config.json"), config)

    new_train = []; new_eval = []

    def flush():
        if new_train:
            train_history.extend(new_train); new_train.clear()
            _write_json(train_hist_path, train_history)
        if new_eval:
            eval_history.extend(new_eval); new_eval.clear()
            _write_json(eval_hist_path, eval_history)

    best_metric = seed_best_metric(eval_history, CHECKPOINT_METRIC)

    last_eval_bucket = start_episode // eval_interval
    batch_data       = []

    total_episodes = start_episode + num_episodes
    print(f"\n{'='*65}")
    print(f"  opp_stats_input_augmentation_v1 | {variant.upper()} | {'SMOKE' if smoke else 'RESUME' if resume else 'FULL RUN'}")
    print(f"  episodes: {start_episode:,} → {total_episodes:,}  (+{num_episodes:,})")
    print(f"  eval every {eval_interval} ep  session={SESSION_LENGTH} hands")
    print(f"{'='*65}\n")

    # ── opponent/tracker state ────────────────────────────────────────────────
    trackers = {
        0: OpponentStatsTracker(pool_means, PRIOR_STRENGTH, SESSION_LENGTH),
        1: OpponentStatsTracker(pool_means, PRIOR_STRENGTH, SESSION_LENGTH),
    }
    hands_in_session = {0: 0, 1: 0}
    current_opponent = None

    if variant == "self_play":
        current_opponent = agent      # plays against itself
    elif variant == "pool_random":
        current_opponent = random.choice(ordered_opps)
    elif variant == "pool_seq":
        opp_idx    = 0
        sess_count = 0
        current_opponent = ordered_opps[opp_idx]

    t0 = time.time()

    for ep in range(start_episode + 1, start_episode + num_episodes + 1):
        # Each position gets its own stream of 100-hand sessions.
        learner_id = ep % 2
        if hands_in_session[0] >= SESSION_LENGTH and hands_in_session[1] >= SESSION_LENGTH:
            for tracker in trackers.values():
                tracker.reset()
            hands_in_session = {0: 0, 1: 0}

            if variant == "pool_random":
                current_opponent = random.choice(ordered_opps)
            elif variant == "pool_seq":
                sess_count += 1
                if sess_count >= k_sessions:
                    sess_count  = 0
                    opp_idx     = (opp_idx + 1) % len(ordered_opps)
                    current_opponent = ordered_opps[opp_idx]
                    print(f"  [ep={ep:,}] Switching to opponent "
                          f"{OPPONENT_KEYS[opp_idx]}")

        # ── collect one hand ──────────────────────────────────────────────────
        chain, reward = play_hand(agent, current_opponent, trackers[learner_id], learner_id=learner_id)
        hands_in_session[learner_id] += 1
        batch_data.append((chain, reward))

        # ── batch update ──────────────────────────────────────────────────────
        if len(batch_data) >= BATCH_SIZE:
            loss = update_model(agent, optimizer, criterion, batch_data)
            batch_data.clear()
            entry = {"episode": ep, "loss": round(loss, 6)}
            new_train.append(entry)
            if len(new_train) >= flush_every: flush()
            if ep <= BATCH_SIZE or ep % 100 == 0:
                print(f"Episode {ep:,}, Batch Loss: {loss:.4f}")

        # ── periodic eval ─────────────────────────────────────────────────────
        bucket = ep // eval_interval
        if bucket > last_eval_bucket:
            last_eval_bucket = bucket
            pool_eval = quick_eval_stats(agent, opponents, eval_rounds, pool_means)
            scores = pool_eval["scores"]
            summary = pool_eval["summary"]
            entry  = {"episode": ep, **scores}
            new_eval.append(entry)
            if len(new_eval) >= flush_every: flush()

            metric_value = summary["metric_values"][CHECKPOINT_METRIC]
            if metric_value > best_metric:
                best_metric = metric_value
                torch.save(agent.model.state_dict(),
                           os.path.join(out_dir, "checkpoint_best.pt"))
            print(f"  [ep={ep:>7,}]  heuristic:{scores['heuristic']:+.3f}  cfr:{scores['cfr']:+.3f}"
                  f"  tp:{scores['tight_passive']:+.2f}  ta:{scores['tight_aggressive']:+.2f}"
                  f"  lp:{scores['loose_passive']:+.2f}  la:{scores['loose_aggressive']:+.2f}"
                  f"  mn:{scores['maniac']:+.2f}  rnd:{scores['random']:+.2f}"
                  f"  [{CHECKPOINT_METRIC}:{metric_value:+.3f}  best:{best_metric:+.3f}]"
                  f"  {format_pool_summary(summary)}")

    flush()
    elapsed = time.time() - t0

    # ── save final checkpoint ─────────────────────────────────────────────────
    torch.save(agent.model.state_dict(), os.path.join(out_dir, "checkpoint.pt"))
    print(f"\ncheckpoint.pt saved  ({elapsed:.1f}s elapsed)")

    # ── convergence check + results ───────────────────────────────────────────
    all_losses = [r["loss"] for r in train_history]
    converged, pct = _convergence_check(all_losses)
    loss_final = all_losses[-1] if all_losses else 0.0
    sat_ep = _find_saturation_episode(eval_history)

    agent.set_train_mode(False)
    final_eval = quick_eval_stats(agent, opponents, eval_rounds, pool_means)
    final_scores = final_eval["scores"]
    final_summary = final_eval["summary"]

    results = {
        "experiment_id":     f"opp_stats_input_augmentation_v1_{variant}",
        "variant":           variant,
        "training_episodes": start_episode + num_episodes,
        "converged":         converged,
        "loss_plateau_pct":  round(pct, 2),
        "peak_eval_score":   round(best_metric, 4),
        "final_eval_score":  round(final_scores["heuristic"], 4),
        "final_cfr_score":   round(final_scores["cfr"], 4),
        "eval_opponent":     "heuristic",
        "eval_rounds":       eval_rounds,
        "eval_interval":     eval_interval,
        "flush_every":       flush_every,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "best_checkpoint_metric_value": round(best_metric, 4),
        "overall_avg":       final_summary["avg"],
        "worst_case":        final_summary["worst_case"],
        "best_case":         final_summary["best_case"],
        "robustness":        final_summary["robustness"],
        "loss_final":        round(loss_final, 6),
        "loss_components":   {"td_mse": round(loss_final, 6)},
        "saturation_episode": sat_ep,
        "elapsed_seconds":   round(elapsed, 1),
        "representation_metrics": {
            "effective_dim_80": None, "effective_dim_90": None,
            "reward_spearman_rho_pairwise": None,
            "hand_probe_accuracy": None, "hand_probe_chance": 0.333,
        },
        "notes": (
            f"opp_stats_input_augmentation_v1 {variant}: 22-dim input "
            f"(15 game + 7 opponent stats, S={PRIOR_STRENGTH}), "
            "balanced-seat training and session-reset evaluation."
        ),
    }
    _write_json(os.path.join(out_dir, "results.json"), results)

    print(f"\nresults.json written")
    print(f"  best {CHECKPOINT_METRIC:>10}: {best_metric:+.3f}")
    print(f"  final vs heuristic: {final_scores['heuristic']:+.3f}")
    print(f"  final vs cfr      : {final_scores['cfr']:+.3f}")
    print(f"  {format_pool_summary(final_summary)}")
    print(f"  saturation episode: {sat_ep}")
    print(f"  converged         : {converged} ({pct:.1f}%)")

    if not smoke:
        generate_plots(train_history, eval_history, out_dir)

    print("\nDone. Run eval.py --variant <name> for final 5000-round evaluation.")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",  required=True,
                        choices=["self_play", "pool_random", "pool_seq"])
    parser.add_argument("--smoke",    action="store_true")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from checkpoint.pt, appending to existing histories")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Additional episodes when --resume, or total episodes otherwise")
    parser.add_argument("--k",        type=int, default=500,
                        help="Sessions per opponent for pool_seq (default 500)")
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL,
                        help="Episodes between in-training evaluations")
    parser.add_argument("--eval-rounds", type=int, default=EVAL_ROUNDS,
                        help="Rounds per opponent for in-training evaluations")
    parser.add_argument("--flush-every", type=int, default=FLUSH_EVERY,
                        help="Buffered history entries before writing JSON to disk")
    args = parser.parse_args()

    out_dir = os.path.join(HERE, "outputs", args.variant)

    if args.smoke:
        num_episodes = 500
    elif args.episodes is not None:
        num_episodes = args.episodes
    elif args.resume:
        num_episodes = 300_000   # default additional budget when resuming
    elif args.variant == "pool_seq":
        num_episodes = args.k * SESSION_LENGTH * 2 * len(OPPONENT_KEYS)
    else:
        num_episodes = 300_000

    run_training(
        variant=args.variant,
        out_dir=out_dir,
        num_episodes=num_episodes,
        k_sessions=args.k,
        smoke=args.smoke,
        resume=args.resume,
        eval_interval=args.eval_interval,
        eval_rounds=args.eval_rounds,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
