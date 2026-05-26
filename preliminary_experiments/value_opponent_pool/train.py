"""
value_opponent_pool — Training Script
======================================
Trains a ValueBasedAgent via TD(0) against a rotating pool of 8 rule-based
opponents (random, maniac, tight_passive, loose_passive, loose_aggressive,
tight_aggressive, heuristic, cfr). The opponent rotates every 1000 episodes
(one full cycle = 8000 episodes).

Paired baseline to baseline_value_v1 (self-play TD(0)).

Usage:
    python train.py                        # fresh 200K-episode run
    python train.py --episodes 400000      # custom episode count (fresh)
    python train.py --resume               # continue from checkpoint.pt (+200K)
    python train.py --resume --episodes N  # continue for N more episodes
    python train.py --smoke                # 500-episode pipeline validation

Outputs (in outputs/):
    train_config.json     — hyperparameters
    train_history.json    — [{episode, loss}, ...] one entry per batch update
    eval_history.json     — [{episode, heuristic, cfr, tight_passive, ...}, ...]
    checkpoint.pt         — final model weights
    checkpoint_best.pt    — best checkpoint by heuristic eval score
    results.json          — STANDARDS §3 required keys + saturation_episode
    training_curve.png    — EMA-smoothed TD loss vs episodes
    eval_curve.png        — avg chips/round vs all 8 opponents over training
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
from agents.heuristic.agent import HeuristicAgent
from agents.cfr.agent import CFRAgent
from agents.evaluation import quick_evaluate

from agents.rule_based.tight_passive    import TightPassiveAgent
from agents.rule_based.tight_aggressive import TightAggressiveAgent
from agents.rule_based.loose_passive    import LoosePassiveAgent
from agents.rule_based.loose_aggressive import LooseAggressiveAgent
from agents.rule_based.maniac           import ManiacAgent
from agents.rule_based.random_agent     import RandomAgent

from preliminary_experiments.value_opponent_pool.trainer import PoolTrainer

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

OPPONENT_KEYS = [
    "heuristic", "cfr",
    "tight_passive", "tight_aggressive",
    "loose_passive", "loose_aggressive",
    "maniac", "random",
]

ROTATE_EVERY = 1000   # episodes per opponent before rotating

CONFIG_BASE = {
    "experiment_id":   "value_opponent_pool",
    "architecture":    "ValueNetwork(input_size=15, hidden_size=64)",
    "input_size":      15,
    "hidden_size":     64,
    "learning_rate":   1e-4,
    "batch_size":      32,
    "temperature":     1.0,
    "training":        "TD(0) pool-based (single learner, alternating positions)",
    "optimizer":       "Adam",
    "rotate_every":    ROTATE_EVERY,
    "opponent_pool":   ["random", "maniac", "tight_passive", "loose_passive",
                        "loose_aggressive", "tight_aggressive", "heuristic", "cfr"],
    "eval_interval":   100,
    "eval_rounds":     200,
    "final_eval_rounds": 5000,
}

FLUSH_EVERY = 500   # write histories to disk every N new entries


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
    scores   = [r.get("heuristic") for r in eval_records if r.get("heuristic") is not None]
    episodes = [r["episode"]       for r in eval_records if r.get("heuristic") is not None]
    if len(scores) < window:
        return None
    for i in range(window, len(scores)):
        w = scores[i - window:i]
        if max(w) - min(w) < threshold:
            return episodes[i - 1]
    return None


def generate_plots(train_history, eval_history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    # ── training_curve.png ────────────────────────────────────────────────────
    if train_history:
        episodes = [r["episode"] for r in train_history]
        losses   = [r["loss"]    for r in train_history]
        smoothed = _ema(losses)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(episodes, losses,   color="lightblue", alpha=0.35, linewidth=0.6, label="raw loss")
        ax.plot(episodes, smoothed, color="steelblue",              linewidth=1.5, label="EMA (α=0.95)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("TD MSE Loss")
        ax.set_title("value_opponent_pool — Training Loss (TD MSE)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "training_curve.png"), dpi=150)
        plt.close(fig)
        print("Saved training_curve.png")

    # ── eval_curve.png ────────────────────────────────────────────────────────
    if eval_history:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        COLORS_TOP = {"heuristic": "firebrick", "cfr": "forestgreen"}
        COLORS_RB  = {
            "tight_passive":    "#1f77b4",
            "tight_aggressive": "#ff7f0e",
            "loose_passive":    "#9467bd",
            "loose_aggressive": "#e377c2",
            "maniac":           "#8c564b",
            "random":           "#7f7f7f",
        }

        for key, color in COLORS_TOP.items():
            pts = [(r["episode"], r[key]) for r in eval_history if key in r]
            if not pts:
                continue
            ep_k, sc_k = zip(*pts)
            ax1.plot(ep_k, sc_k,             color=color, alpha=0.25, linewidth=0.6)
            ax1.plot(ep_k, _ema(list(sc_k)), color=color, linewidth=1.8, label=key)

        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax1.set_ylabel("Avg Chips / Round")
        ax1.set_title("value_opponent_pool — Eval vs Heuristic & CFR")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        for key, color in COLORS_RB.items():
            pts = [(r["episode"], r[key]) for r in eval_history if key in r]
            if not pts:
                continue
            ep_k, sc_k = zip(*pts)
            ax2.plot(ep_k, sc_k,             color=color, alpha=0.25, linewidth=0.6)
            ax2.plot(ep_k, _ema(list(sc_k)), color=color, linewidth=1.8, label=key)

        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Avg Chips / Round")
        ax2.set_title("value_opponent_pool — Eval vs Rule-Based Agents")
        ax2.legend(loc="upper left", ncol=2)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "eval_curve.png"), dpi=150)
        plt.close(fig)
        print("Saved eval_curve.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",    action="store_true",
                        help="500-episode pipeline validation")
    parser.add_argument("--resume",   action="store_true",
                        help="Continue training from checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodes to run (fresh: default 200K; resume: default 200K)")
    args = parser.parse_args()

    if args.smoke:
        num_episodes = 500
    elif args.episodes is not None:
        num_episodes = args.episodes
    else:
        num_episodes = 200_000

    # ── load or create agent ──────────────────────────────────────────────────
    agent = ValueBasedAgent(temperature=CONFIG_BASE["temperature"])

    start_episode = 0
    if args.resume:
        ckpt_path = os.path.join(OUT, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint found at {ckpt_path}. Run without --resume first.")
            sys.exit(1)
        agent.load_model(ckpt_path)
        th = _load_json(os.path.join(OUT, "train_history.json"), [])
        start_episode = th[-1]["episode"] if th else 0
        print(f"Resuming from episode {start_episode:,}  ({ckpt_path})")

    trainer = PoolTrainer(agent,
                          learning_rate=CONFIG_BASE["learning_rate"],
                          rotate_every=ROTATE_EVERY)

    # ── evaluation opponents ───────────────────────────────────────────────────
    cfr_path = os.path.join(ROOT, "agents", "cfr", "checkpoint.pt")
    opponents = {
        "heuristic":        HeuristicAgent(),
        "cfr":              CFRAgent(model_path=cfr_path),
        "tight_passive":    TightPassiveAgent(),
        "tight_aggressive": TightAggressiveAgent(),
        "loose_passive":    LoosePassiveAgent(),
        "loose_aggressive": LooseAggressiveAgent(),
        "maniac":           ManiacAgent(),
        "random":           RandomAgent(),
    }
    for opp in opponents.values():
        opp.set_train_mode(False)

    # ── save config ───────────────────────────────────────────────────────────
    config = {**CONFIG_BASE,
              "num_episodes":   num_episodes,
              "start_episode":  start_episode,
              "total_episodes": start_episode + num_episodes,
              "resume":         args.resume,
              "eval_opponents": OPPONENT_KEYS}
    _write_json(os.path.join(OUT, "train_config.json"), config)

    print(f"\n{'='*65}")
    mode = "SMOKE TEST" if args.smoke else ("RESUME" if args.resume else "FRESH RUN")
    print(f"  value_opponent_pool  |  {mode}")
    print(f"  episodes this run : {num_episodes:,}  (ep {start_episode+1:,} → {start_episode+num_episodes:,})")
    print(f"  rotate every      : {ROTATE_EVERY:,} episodes  (1 cycle = {ROTATE_EVERY * 8:,} episodes)")
    print(f"  eval every        : {CONFIG_BASE['eval_interval']} episodes  ({CONFIG_BASE['eval_rounds']} rounds × 8 opponents)")
    print(f"{'='*65}\n")

    # ── load existing histories ───────────────────────────────────────────────
    train_history_path = os.path.join(OUT, "train_history.json")
    eval_history_path  = os.path.join(OUT, "eval_history.json")

    train_history = _load_json(train_history_path, []) if args.resume else []
    eval_history  = _load_json(eval_history_path,  []) if args.resume else []

    new_train_entries = []
    new_eval_entries  = []

    best_heuristic   = max((r.get("heuristic", float("-inf")) for r in eval_history),
                           default=float("-inf"))
    last_eval_bucket = start_episode // CONFIG_BASE["eval_interval"]

    def _flush():
        if new_train_entries:
            train_history.extend(new_train_entries)
            new_train_entries.clear()
            _write_json(train_history_path, train_history)
        if new_eval_entries:
            eval_history.extend(new_eval_entries)
            new_eval_entries.clear()
            _write_json(eval_history_path, eval_history)

    def callback(event):
        nonlocal best_heuristic, last_eval_bucket

        if event["type"] == "batch_update":
            entry = {"episode": event["episode"], "loss": round(event["loss"], 6)}
            new_train_entries.append(entry)
            if len(new_train_entries) >= FLUSH_EVERY:
                _flush()

        current_bucket = event["episode"] // CONFIG_BASE["eval_interval"]
        if event["type"] == "batch_update" and current_bucket > last_eval_bucket:
            last_eval_bucket = current_bucket
            ep = event["episode"]

            agent.set_train_mode(False)
            scores = {}
            for key, opp in opponents.items():
                scores[key] = round(quick_evaluate(agent, opp,
                                                   num_rounds=CONFIG_BASE["eval_rounds"]), 4)
            agent.set_train_mode(True)

            entry = {"episode": ep, **scores}
            new_eval_entries.append(entry)
            if len(new_eval_entries) >= FLUSH_EVERY:
                _flush()

            h = scores["heuristic"]
            if h > best_heuristic:
                best_heuristic = h
                torch.save(agent.model.state_dict(), os.path.join(OUT, "checkpoint_best.pt"))

            print(f"  [ep={ep:>7,}]  heuristic:{h:+.3f}  cfr:{scores['cfr']:+.3f}"
                  f"  tp:{scores['tight_passive']:+.2f}  ta:{scores['tight_aggressive']:+.2f}"
                  f"  lp:{scores['loose_passive']:+.2f}  la:{scores['loose_aggressive']:+.2f}"
                  f"  mn:{scores['maniac']:+.2f}  rnd:{scores['random']:+.2f}"
                  f"  [best_h:{best_heuristic:+.3f}]")

    # ── train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train(
        num_episodes=num_episodes,
        batch_size=CONFIG_BASE["batch_size"],
        save_path=None,
        callback=callback,
        start_episode=start_episode,
    )
    elapsed = time.time() - t0
    _flush()   # final flush

    # ── save checkpoints ──────────────────────────────────────────────────────
    torch.save(agent.model.state_dict(), os.path.join(OUT, "checkpoint.pt"))
    print(f"\ncheckpoint.pt saved  ({elapsed:.1f}s elapsed)")

    # ── convergence check ─────────────────────────────────────────────────────
    all_losses = [r["loss"] for r in train_history]
    converged, pct = _convergence_check(all_losses)
    loss_final = all_losses[-1] if all_losses else 0.0
    total_ep   = start_episode + num_episodes

    print(f"Convergence: {'PASSED' if converged else 'NOT YET'}  (last-20% Δloss = {pct:.1f}%)")

    # ── results ───────────────────────────────────────────────────────────────
    agent.set_train_mode(False)
    final_h = round(quick_evaluate(agent, opponents["heuristic"], CONFIG_BASE["eval_rounds"]), 4)
    final_c = round(quick_evaluate(agent, opponents["cfr"],       CONFIG_BASE["eval_rounds"]), 4)
    peak_h  = max((r.get("heuristic", float("-inf")) for r in eval_history), default=final_h)
    sat_ep  = _find_saturation_episode(eval_history)

    results = {
        "experiment_id":    "value_opponent_pool",
        "training_episodes": total_ep,
        "converged":        converged,
        "loss_plateau_pct": round(pct, 2),
        "peak_eval_score":  round(peak_h, 4),
        "final_eval_score": round(final_h, 4),
        "final_cfr_score":  round(final_c, 4),
        "eval_opponent":    "heuristic",
        "eval_rounds":      CONFIG_BASE["eval_rounds"],
        "loss_final":       round(loss_final, 6),
        "loss_components":  {"td_mse": round(loss_final, 6)},
        "saturation_episode": sat_ep,
        "elapsed_seconds":  round(elapsed, 1),
        "representation_metrics": {
            "effective_dim_80": None, "effective_dim_90": None,
            "reward_spearman_rho_pairwise": None,
            "hand_probe_accuracy": None, "hand_probe_chance": 0.333,
        },
        "notes": "Pool-based TD(0) baseline. 8 rule-based opponents, rotate every 1000 episodes.",
    }
    _write_json(os.path.join(OUT, "results.json"), results)

    print(f"\nresults.json written  (total episodes: {total_ep:,})")
    print(f"  peak vs heuristic : {peak_h:+.3f}")
    print(f"  final vs heuristic: {final_h:+.3f}")
    print(f"  final vs cfr      : {final_c:+.3f}")
    print(f"  saturation episode: {sat_ep}")

    if not args.smoke:
        generate_plots(train_history, eval_history)

    print("\nDone. Run eval.py for the final 5000-round comprehensive evaluation.")


if __name__ == "__main__":
    main()
