"""
Compare baseline and pool-trained value agents over the 180 unique infosets.

Outputs:
  - outputs/baseline_vs_pool_variants.png
  - outputs/baseline_vs_pool_variants_summary.json

The stat-augmented agents are evaluated at the mean prototype-stat vector of the
6 rule-based archetypes so they can be compared to non-stat-aware agents on a
single curve per infoset.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.value_based.agent import ValueBasedAgent
from engine.leduc_game import LeducGame
from preliminary_experiments.ev_variation_extras.code.sim_engine import FixedStateSimulator
from preliminary_experiments.opp_stats_input_aug.agent import StatAugValueAgent


EV_DATA_PATH = os.path.join(ROOT, "paper", "ev_analysis", "data.json")
PROTO_PATH = os.path.join(ROOT, "paper", "evaluation", "shared", "data", "opponent_prototype_stats.json")
OUT_DIR = os.path.join(HERE, "outputs")

CKPT_BASELINE = os.path.join(
    ROOT, "preliminary_experiments", "baseline_value_v1", "outputs", "checkpoint_best.pt"
)
CKPT_VALUE_POOL = os.path.join(
    ROOT, "preliminary_experiments", "value_opponent_pool", "outputs", "checkpoint_best.pt"
)
CKPT_POOL_RANDOM = os.path.join(
    ROOT,
    "preliminary_experiments",
    "opp_stats_input_aug",
    "outputs",
    "pool_random",
    "checkpoint_best.pt",
)
CKPT_POOL_SEQ = os.path.join(
    ROOT,
    "preliminary_experiments",
    "opp_stats_input_aug",
    "outputs",
    "pool_seq",
    "checkpoint_best.pt",
)

ARCHETYPES = [
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _infoset_label(key):
    hand, board, pot, rnd, raises, cp = key
    board_s = board if board is not None else "-"
    return f"{hand}|{board_s}|{pot}|r{rnd}|raises{raises}|cp{cp}"


def _infoset_key(record):
    cp = record["current_player"]
    hand = record["hand0"] if cp == 0 else record["hand1"]
    return (hand, record["board"], tuple(record["pot"]), record["round"], record["raises"], cp)


def _get_obs(record):
    game = LeducGame()
    sim = FixedStateSimulator(
        hand0=record["hand0"],
        hand1=record["hand1"],
        pot=record["pot"],
        current_player=record["current_player"],
        rnd=record["round"],
        raises=record["raises"],
        board=record["board"],
    )
    sim._inject(game)
    return game.get_observation(viewer_id=record["current_player"])


def _best_value(agent, obs, opp_stats=None):
    if opp_stats is None:
        evals = agent.get_action_evaluations(obs)
    else:
        evals = agent.get_action_evaluations(obs, opp_stats)
    best = max(evals, key=lambda e: e["value"])
    return best["action"].name, float(best["value"])


def build_rows():
    ev_data = _load_json(EV_DATA_PATH)
    proto = _load_json(PROTO_PATH)
    proto_mean = np.mean([np.array(proto[k], dtype=np.float32) for k in ARCHETYPES], axis=0)

    baseline = ValueBasedAgent(model_path=CKPT_BASELINE)
    value_pool = ValueBasedAgent(model_path=CKPT_VALUE_POOL)
    pool_random = StatAugValueAgent(model_path=CKPT_POOL_RANDOM)
    pool_seq = StatAugValueAgent(model_path=CKPT_POOL_SEQ)
    for agent in [baseline, value_pool, pool_random, pool_seq]:
        agent.set_train_mode(False)

    infosets = {}
    for record in ev_data["records"]:
        if record["opponent"] not in ARCHETYPES:
            continue
        infosets.setdefault(_infoset_key(record), record)

    rows = []
    for key, record in infosets.items():
        obs = _get_obs(record)
        base_action, base_value = _best_value(baseline, obs)
        vp_action, vp_value = _best_value(value_pool, obs)
        pr_action, pr_value = _best_value(pool_random, obs, proto_mean)
        ps_action, ps_value = _best_value(pool_seq, obs, proto_mean)
        rows.append(
            {
                "label": _infoset_label(key),
                "hand": key[0],
                "board": key[1],
                "pot": list(key[2]),
                "round": key[3],
                "raises": key[4],
                "current_player": key[5],
                "baseline_action": base_action,
                "baseline_value": base_value,
                "value_pool_action": vp_action,
                "value_pool_value": vp_value,
                "pool_random_action": pr_action,
                "pool_random_value": pr_value,
                "pool_seq_action": ps_action,
                "pool_seq_value": ps_value,
                "value_pool_delta": vp_value - base_value,
                "pool_random_delta": pr_value - base_value,
                "pool_seq_delta": ps_value - base_value,
            }
        )
    return rows


def save_summary(rows):
    def _top(key, n=12):
        ranked = sorted(rows, key=lambda r: abs(r[key]), reverse=True)
        return ranked[:n]

    summary = {
        "n_infosets": len(rows),
        "mean_values": {
            "baseline": round(float(np.mean([r["baseline_value"] for r in rows])), 4),
            "value_pool": round(float(np.mean([r["value_pool_value"] for r in rows])), 4),
            "pool_random": round(float(np.mean([r["pool_random_value"] for r in rows])), 4),
            "pool_seq": round(float(np.mean([r["pool_seq_value"] for r in rows])), 4),
        },
        "mean_deltas_vs_baseline": {
            "value_pool": round(float(np.mean([r["value_pool_delta"] for r in rows])), 4),
            "pool_random": round(float(np.mean([r["pool_random_delta"] for r in rows])), 4),
            "pool_seq": round(float(np.mean([r["pool_seq_delta"] for r in rows])), 4),
        },
        "action_disagreement_rates": {
            "value_pool": round(
                float(np.mean([r["value_pool_action"] != r["baseline_action"] for r in rows])), 4
            ),
            "pool_random": round(
                float(np.mean([r["pool_random_action"] != r["baseline_action"] for r in rows])), 4
            ),
            "pool_seq": round(
                float(np.mean([r["pool_seq_action"] != r["baseline_action"] for r in rows])), 4
            ),
        },
        "top_abs_delta_infosets": {
            "value_pool": _top("value_pool_delta"),
            "pool_random": _top("pool_random_delta"),
            "pool_seq": _top("pool_seq_delta"),
        },
    }
    out_path = os.path.join(OUT_DIR, "baseline_vs_pool_variants_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path


def plot_rows(rows):
    rows = sorted(rows, key=lambda r: r["baseline_value"])
    x = np.arange(len(rows))
    rounds = np.array([r["round"] for r in rows])

    baseline = np.array([r["baseline_value"] for r in rows])
    value_pool = np.array([r["value_pool_value"] for r in rows])
    pool_random = np.array([r["pool_random_value"] for r in rows])
    pool_seq = np.array([r["pool_seq_value"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.suptitle(
        "Baseline vs Pool-Trained Variants Across 180 Infosets\n"
        "Sorted by baseline best-action value",
        fontsize=12,
        fontweight="bold",
    )

    for i, rnd in enumerate(rounds):
        color = "#bbdefb" if rnd == 0 else "#e8f5e9"
        ax1.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.12, linewidth=0)
        ax2.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.12, linewidth=0)

    ax1.plot(x, baseline, color="black", linewidth=2.0, label="baseline_value_v1")
    ax1.plot(x, value_pool, color="#1565C0", linewidth=1.4, label="value_opponent_pool")
    ax1.plot(x, pool_random, color="#EF6C00", linewidth=1.4, label="opp_stats pool_random")
    ax1.plot(x, pool_seq, color="#C62828", linewidth=1.4, label="opp_stats pool_seq")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Best-action value (chips)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(x, value_pool - baseline, color="#1565C0", linewidth=1.4, label="value_pool - baseline")
    ax2.plot(
        x, pool_random - baseline, color="#EF6C00", linewidth=1.4, label="pool_random - baseline"
    )
    ax2.plot(x, pool_seq - baseline, color="#C62828", linewidth=1.4, label="pool_seq - baseline")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Delta vs baseline")
    ax2.set_xlabel(
        f"Infosets (n={len(rows)}; blue=pre-flop, green=flop)", fontsize=9
    )
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "baseline_vs_pool_variants.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = build_rows()
    summary_path = save_summary(rows)
    plot_path = plot_rows(rows)
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot:    {plot_path}")
    print(f"Infosets:      {len(rows)}")


if __name__ == "__main__":
    main()
