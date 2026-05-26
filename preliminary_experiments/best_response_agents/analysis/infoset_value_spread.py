"""
infoset_value_spread.py

Builds a full observation-level value comparison across the six
opponent-specific best-response value networks plus the self-play baseline.

For each reachable 15-feature observation class in the current Leduc repo:
  - evaluate the direct network output V(obs) for each fixed-opponent model
  - compute the spread across those six models
  - optionally compute an exact Nash reference for the same collapsed
    observation class using the CFR average strategy tables
  - sort infosets by spread and write a plot + CSV/JSON outputs

Run from project root:
    python -m preliminary_experiments.best_response_agents.analysis.infoset_value_spread
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from agents.cfr.solver import (
    BET_AMOUNTS,
    _generate_deals,
    _legal_actions,
    _make_key,
    _showdown,
)
from agents.cfr.strategy import TabularStrategyStore
from agents.value_based.agent import ValueBasedAgent
from engine.leduc_game import Action
from engine.observation import Observation
from preliminary_experiments.ev_variation_extras.code.collect import enumerate_states
from agents.rule_based import ALL_AGENTS


MODEL_ORDER = [
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
]

MODEL_LABELS = {
    "tight_passive": "Tight Passive",
    "tight_aggressive": "Tight Aggressive",
    "loose_passive": "Loose Passive",
    "loose_aggressive": "Loose Aggressive",
    "maniac": "Maniac",
    "random": "Random",
}

MODEL_COLORS = {
    "tight_passive": "#1f77b4",
    "tight_aggressive": "#d62728",
    "loose_passive": "#2ca02c",
    "loose_aggressive": "#ff7f0e",
    "maniac": "#9467bd",
    "random": "#8c564b",
}

ROUND_SHADE = {
    0: "#DDEAF7",
    1: "#DFF1E2",
}

BR_OUT_ROOT = os.path.join(ROOT, "preliminary_experiments", "best_response_agents", "outputs")
BASELINE_CKPT = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")
CFR_CKPT = os.path.join(ROOT, "agents", "cfr", "checkpoint.pt")
DEFAULT_OUTDIR = os.path.join(HERE, "outputs")


def pretty_round(rnd: int) -> str:
    return "Pre" if rnd == 0 else "Flop"


def pretty_board(board: str | None) -> str:
    return board if board is not None else "-"


def infoset_key_from_state(state: dict) -> Tuple[str, str | None, Tuple[int, int], int, int, int]:
    cp = state["current_player"]
    my_hand = state["hand0"] if cp == 0 else state["hand1"]
    return (
        my_hand,
        state["board"],
        tuple(state["pot"]),
        state["round"],
        state["raises"],
        cp,
    )


def infoset_label(key: Tuple[str, str | None, Tuple[int, int], int, int, int]) -> str:
    hand, board, pot, rnd, raises, cp = key
    return f"{pretty_round(rnd)} {hand}|{pretty_board(board)} [{pot[0]},{pot[1]}] cp{cp} r{raises}"


def infoset_id(key: Tuple[str, str | None, Tuple[int, int], int, int, int]) -> str:
    hand, board, pot, rnd, raises, cp = key
    board_token = board if board is not None else "none"
    round_token = "pf" if rnd == 0 else "fl"
    return f"{round_token}_{hand}_{board_token}_p{pot[0]}-{pot[1]}_cp{cp}_r{raises}"


def legal_actions_from_raises(raises: int) -> List[Action]:
    actions = [Action.FOLD, Action.CALL]
    if raises < 2:
        actions.append(Action.RAISE)
    return actions


def observation_from_key(
    key: Tuple[str, str | None, Tuple[int, int], int, int, int]
) -> Observation:
    hand, board, pot, rnd, raises, cp = key
    return Observation(
        player_hand=hand,
        board=board,
        pot=list(pot),
        current_player=cp,
        current_round=rnd,
        legal_actions=legal_actions_from_raises(raises),
        is_finished=False,
        raises_this_round=raises,
    )


def build_infosets() -> List[dict]:
    infosets: Dict[Tuple[str, str | None, Tuple[int, int], int, int, int], dict] = {}
    for state in enumerate_states():
        key = infoset_key_from_state(state)
        if key in infosets:
            continue
        hand, board, pot, rnd, raises, cp = key
        infosets[key] = {
            "key": key,
            "infoset_id": infoset_id(key),
            "label": infoset_label(key),
            "my_hand": hand,
            "board": board,
            "pot": list(pot),
            "round": rnd,
            "raises": raises,
            "current_player": cp,
            "obs": observation_from_key(key),
        }
    return sorted(
        infosets.values(),
        key=lambda item: (
            item["round"],
            item["my_hand"],
            pretty_board(item["board"]),
            item["pot"][0],
            item["pot"][1],
            item["raises"],
            item["current_player"],
        ),
    )


def checkpoint_for(opp_key: str, checkpoint_kind: str) -> Tuple[str, str]:
    best_ckpt = os.path.join(BR_OUT_ROOT, opp_key, "checkpoint_best.pt")
    final_ckpt = os.path.join(BR_OUT_ROOT, opp_key, "checkpoint.pt")

    if checkpoint_kind == "best":
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"Missing checkpoint_best.pt for {opp_key}")
        return best_ckpt, "best"
    if checkpoint_kind == "final":
        if not os.path.exists(final_ckpt):
            raise FileNotFoundError(f"Missing checkpoint.pt for {opp_key}")
        return final_ckpt, "final"

    if os.path.exists(best_ckpt):
        return best_ckpt, "best"
    if os.path.exists(final_ckpt):
        return final_ckpt, "final"
    raise FileNotFoundError(f"No checkpoint found for {opp_key}")


def load_value_agent(path: str) -> ValueBasedAgent:
    agent = ValueBasedAgent(model_path=path)
    agent.set_train_mode(False)
    return agent


def direct_network_value(agent: ValueBasedAgent, obs: Observation, viewer_id: int) -> float:
    with torch.no_grad():
        encoded = agent.encode_observation(obs, viewer_id=viewer_id)
        return float(agent.model(encoded).item())


class NashObservationValueExtractor:
    """
    Extract exact continuation values under the CFR average strategy, grouped by
    the current repo's collapsed observation class:

        (my_hand, board, pot, round, raises, current_player)

    Unlike true CFR infoset values, this grouping intentionally averages over
    hidden opponent hand and hidden action history whenever the 15-feature
    observation cannot distinguish them.
    """

    def __init__(self, strategy_store: TabularStrategyStore):
        self.store = strategy_store
        self.value_accum: Dict[Tuple[str, str | None, Tuple[int, int], int, int, int], float] = defaultdict(float)
        self.weight_accum: Dict[Tuple[str, str | None, Tuple[int, int], int, int, int], float] = defaultdict(float)

    def extract(self) -> Tuple[dict, dict]:
        self.value_accum.clear()
        self.weight_accum.clear()

        for p0_hand, p1_hand, board, chance_prob in _generate_deals():
            self._traverse(
                p0_hand=p0_hand,
                p1_hand=p1_hand,
                board=board,
                chance_prob=chance_prob,
                preflop="",
                flop="",
                rnd=0,
                player=0,
                pot0=1,
                pot1=1,
                raises=0,
                r0=1.0,
                r1=1.0,
            )

        values = {}
        reaches = {}
        for key, weight in self.weight_accum.items():
            if weight <= 0:
                continue
            values[key] = self.value_accum[key] / weight
            reaches[key] = weight
        return values, reaches

    def _obs_key(
        self,
        p0_hand: str,
        p1_hand: str,
        board: str,
        rnd: int,
        player: int,
        pot0: int,
        pot1: int,
        raises: int,
    ) -> Tuple[str, str | None, Tuple[int, int], int, int, int]:
        my_hand = p0_hand if player == 0 else p1_hand
        obs_board = None if rnd == 0 else board
        return (my_hand, obs_board, (pot0, pot1), rnd, raises, player)

    def _traverse(
        self,
        p0_hand: str,
        p1_hand: str,
        board: str,
        chance_prob: float,
        preflop: str,
        flop: str,
        rnd: int,
        player: int,
        pot0: int,
        pot1: int,
        raises: int,
        r0: float,
        r1: float,
    ) -> float:
        hand = p0_hand if player == 0 else p1_hand
        key = _make_key(hand, board, preflop, flop, rnd)
        legal = _legal_actions(raises)
        strategy = self.store.get_average_strategy(key, legal)

        node_val = 0.0
        for action in legal:
            action_prob = strategy[action.value]
            new_r0 = r0 * (action_prob if player == 0 else 1.0)
            new_r1 = r1 * (action_prob if player == 1 else 1.0)
            node_val += action_prob * self._apply(
                action=action,
                p0_hand=p0_hand,
                p1_hand=p1_hand,
                board=board,
                chance_prob=chance_prob,
                preflop=preflop,
                flop=flop,
                rnd=rnd,
                player=player,
                pot0=pot0,
                pot1=pot1,
                raises=raises,
                r0=new_r0,
                r1=new_r1,
            )

        acting_value = node_val if player == 0 else -node_val
        obs_key = self._obs_key(p0_hand, p1_hand, board, rnd, player, pot0, pot1, raises)
        reach = chance_prob * r0 * r1
        self.value_accum[obs_key] += reach * acting_value
        self.weight_accum[obs_key] += reach
        return node_val

    def _apply(
        self,
        action: Action,
        p0_hand: str,
        p1_hand: str,
        board: str,
        chance_prob: float,
        preflop: str,
        flop: str,
        rnd: int,
        player: int,
        pot0: int,
        pot1: int,
        raises: int,
        r0: float,
        r1: float,
    ) -> float:
        code = "fcr"[action.value]
        pf = preflop + code if rnd == 0 else preflop
        fl = flop + code if rnd == 1 else flop

        if action == Action.FOLD:
            return -pot0 if player == 0 else pot1

        other_pot = pot1 if player == 0 else pot0
        my_pot = pot0 if player == 0 else pot1
        new_pot0, new_pot1 = pot0, pot1

        if action == Action.RAISE:
            new_my = other_pot + BET_AMOUNTS[rnd]
            if player == 0:
                new_pot0 = new_my
            else:
                new_pot1 = new_my
            return self._traverse(
                p0_hand=p0_hand,
                p1_hand=p1_hand,
                board=board,
                chance_prob=chance_prob,
                preflop=pf,
                flop=fl,
                rnd=rnd,
                player=1 - player,
                pot0=new_pot0,
                pot1=new_pot1,
                raises=raises + 1,
                r0=r0,
                r1=r1,
            )

        round_ended = False
        if other_pot > my_pot:
            if player == 0:
                new_pot0 = other_pot
            else:
                new_pot1 = other_pot
            round_ended = True
        elif player == 1:
            round_ended = True

        if not round_ended:
            return self._traverse(
                p0_hand=p0_hand,
                p1_hand=p1_hand,
                board=board,
                chance_prob=chance_prob,
                preflop=pf,
                flop=fl,
                rnd=rnd,
                player=1 - player,
                pot0=new_pot0,
                pot1=new_pot1,
                raises=raises,
                r0=r0,
                r1=r1,
            )

        if rnd == 0:
            return self._traverse(
                p0_hand=p0_hand,
                p1_hand=p1_hand,
                board=board,
                chance_prob=chance_prob,
                preflop=pf,
                flop="",
                rnd=1,
                player=0,
                pot0=new_pot0,
                pot1=new_pot1,
                raises=0,
                r0=r0,
                r1=r1,
            )

        return _showdown(p0_hand, p1_hand, board, new_pot0, new_pot1)


def load_nash_observation_values() -> Tuple[dict, dict]:
    store = TabularStrategyStore()
    store.load(CFR_CKPT)
    extractor = NashObservationValueExtractor(store)
    return extractor.extract()


def build_records(checkpoint_kind: str, include_nash: bool) -> Tuple[List[dict], dict]:
    infosets = build_infosets()

    br_agents: Dict[str, ValueBasedAgent] = {}
    ckpt_meta = {}
    for opp_key in MODEL_ORDER:
        ckpt_path, source = checkpoint_for(opp_key, checkpoint_kind)
        br_agents[opp_key] = load_value_agent(ckpt_path)
        ckpt_meta[opp_key] = {"path": ckpt_path, "source": source}

    baseline_agent = load_value_agent(BASELINE_CKPT)

    nash_values = {}
    nash_reaches = {}
    if include_nash:
        nash_values, nash_reaches = load_nash_observation_values()

    records = []
    for info in infosets:
        obs = info["obs"]
        cp = info["current_player"]

        values = {}
        for opp_key, agent in br_agents.items():
            values[opp_key] = direct_network_value(agent, obs, viewer_id=cp)

        value_list = [values[k] for k in MODEL_ORDER]
        baseline_value = direct_network_value(baseline_agent, obs, viewer_id=cp)

        key = info["key"]
        nash_value = nash_values.get(key)
        nash_reach = nash_reaches.get(key)

        records.append(
            {
                **info,
                "values": values,
                "baseline_value": baseline_value,
                "nash_value": nash_value,
                "nash_reach": nash_reach,
                "br_min": float(min(value_list)),
                "br_max": float(max(value_list)),
                "br_mean": float(sum(value_list) / len(value_list)),
                "spread": float(max(value_list) - min(value_list)),
                "argmin_model": min(MODEL_ORDER, key=lambda key_: values[key_]),
                "argmax_model": max(MODEL_ORDER, key=lambda key_: values[key_]),
            }
        )

    records.sort(key=lambda row: (row["spread"], row["round"], row["label"]))
    for idx, row in enumerate(records):
        row["rank"] = idx
    metadata = {
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_meta": ckpt_meta,
        "baseline_checkpoint": BASELINE_CKPT,
        "cfr_checkpoint": CFR_CKPT if include_nash else None,
        "n_infosets": len(records),
        "nash_coverage": sum(1 for row in records if row["nash_value"] is not None),
    }
    return records, metadata


def write_csv(records: List[dict], path: str) -> None:
    fieldnames = [
        "rank",
        "infoset_id",
        "label",
        "my_hand",
        "board",
        "pot0",
        "pot1",
        "round",
        "raises",
        "current_player",
        "spread",
        "br_min",
        "br_max",
        "br_mean",
        "argmin_model",
        "argmax_model",
        "baseline_value",
        "nash_value",
        "nash_reach",
        *MODEL_ORDER,
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            out = {
                "rank": row["rank"],
                "infoset_id": row["infoset_id"],
                "label": row["label"],
                "my_hand": row["my_hand"],
                "board": row["board"],
                "pot0": row["pot"][0],
                "pot1": row["pot"][1],
                "round": row["round"],
                "raises": row["raises"],
                "current_player": row["current_player"],
                "spread": round(row["spread"], 6),
                "br_min": round(row["br_min"], 6),
                "br_max": round(row["br_max"], 6),
                "br_mean": round(row["br_mean"], 6),
                "argmin_model": row["argmin_model"],
                "argmax_model": row["argmax_model"],
                "baseline_value": round(row["baseline_value"], 6),
                "nash_value": None if row["nash_value"] is None else round(row["nash_value"], 6),
                "nash_reach": None if row["nash_reach"] is None else round(row["nash_reach"], 12),
            }
            for opp_key in MODEL_ORDER:
                out[opp_key] = round(row["values"][opp_key], 6)
            writer.writerow(out)


def write_summary(records: List[dict], metadata: dict, path: str) -> None:
    spreads = [row["spread"] for row in records]
    summary = {
        **metadata,
        "spread_mean": round(float(np.mean(spreads)), 6),
        "spread_median": round(float(np.median(spreads)), 6),
        "spread_max": round(float(np.max(spreads)), 6),
        "spread_min": round(float(np.min(spreads)), 6),
        "top_10_highest_spread": [
            {
                "rank": row["rank"],
                "infoset_id": row["infoset_id"],
                "label": row["label"],
                "spread": round(row["spread"], 6),
                "argmin_model": row["argmin_model"],
                "argmax_model": row["argmax_model"],
                "baseline_value": round(row["baseline_value"], 6),
                "nash_value": None if row["nash_value"] is None else round(row["nash_value"], 6),
            }
            for row in records[-10:]
        ],
        "top_10_lowest_spread": [
            {
                "rank": row["rank"],
                "infoset_id": row["infoset_id"],
                "label": row["label"],
                "spread": round(row["spread"], 6),
                "argmin_model": row["argmin_model"],
                "argmax_model": row["argmax_model"],
                "baseline_value": round(row["baseline_value"], 6),
                "nash_value": None if row["nash_value"] is None else round(row["nash_value"], 6),
            }
            for row in records[:10]
        ],
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def plot_records(records: List[dict], out_path: str) -> None:
    n = len(records)
    x = np.arange(n)
    spreads = np.array([row["spread"] for row in records], dtype=np.float64)
    br_min = np.array([row["br_min"] for row in records], dtype=np.float64)
    br_max = np.array([row["br_max"] for row in records], dtype=np.float64)
    baseline_vals = np.array([row["baseline_value"] for row in records], dtype=np.float64)
    nash_vals = np.array(
        [np.nan if row["nash_value"] is None else row["nash_value"] for row in records],
        dtype=np.float64,
    )
    round_ids = np.array([row["round"] for row in records], dtype=np.int64)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(21, 10),
        gridspec_kw={"height_ratios": [3.4, 1.2]},
    )

    fig.suptitle(
        "Observation-Level Value Spread Across Opponent-Specific Best-Response Nets\n"
        "180 reachable Leduc observation classes, sorted left to right by spread across the six fixed-opponent value networks",
        fontsize=13,
        fontweight="bold",
    )

    for i, rnd in enumerate(round_ids):
        shade = ROUND_SHADE[int(rnd)]
        ax1.axvspan(i - 0.5, i + 0.5, color=shade, alpha=0.24, linewidth=0)
        ax2.axvspan(i - 0.5, i + 0.5, color=shade, alpha=0.24, linewidth=0)

    ax1.fill_between(x, br_min, br_max, color="#9E9E9E", alpha=0.22, zorder=1)
    ax1.plot(x, br_min, color="#A9A9A9", linewidth=0.7, alpha=0.8, zorder=2)
    ax1.plot(x, br_max, color="#A9A9A9", linewidth=0.7, alpha=0.8, zorder=2)

    ax1.scatter(x, baseline_vals, s=16, color="black", zorder=6)

    if not np.isnan(nash_vals).all():
        ax1.scatter(
            x,
            nash_vals,
            s=22,
            marker="x",
            linewidths=0.9,
            color="#5F6368",
            zorder=7,
        )

    ax1.axhline(0.0, color="gray", linewidth=0.9, linestyle="--", alpha=0.65)
    ax1.set_ylabel("Direct value network output V(obs) [chips]")
    ax1.set_xticks([])
    ax1.grid(axis="y", alpha=0.22)

    top_rows = records[-8:]
    for row in top_rows:
        idx = row["rank"]
        ax1.text(
            idx,
            row["br_max"] + 0.12,
            row["infoset_id"],
            fontsize=7,
            rotation=45,
            ha="left",
            va="bottom",
            color="#444444",
        )

    spread_colors = ["#4E79A7" if rnd == 0 else "#59A14F" for rnd in round_ids]
    ax2.bar(x, spreads, width=0.9, color=spread_colors, alpha=0.82)
    ax2.set_ylabel("Spread")
    ax2.set_xlabel(
        "Infosets ordered by spread across fixed-opponent value networks"
    )
    ax2.grid(axis="y", alpha=0.22)
    ax2.set_xlim(-0.5, n - 0.5)

    legend_handles: List[object] = [
        Patch(facecolor="#9E9E9E", alpha=0.3, label="Range across 6 opponent-specific value nets"),
        Line2D(
            [0],
            [0],
            color="#A9A9A9",
            linewidth=1.0,
            label="Range boundary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor="black",
            markeredgewidth=0,
            label="Self-play value_based",
        ),
    ]
    if not np.isnan(nash_vals).all():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="none",
                markersize=6,
                color="#5F6368",
                label="Exact CFR/Nash value (same collapsed obs)",
            )
        )
    legend_handles.extend(
        [
            Patch(facecolor=ROUND_SHADE[0], alpha=0.8, label="Pre-flop"),
            Patch(facecolor=ROUND_SHADE[1], alpha=0.8, label="Flop"),
        ]
    )
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=8, ncol=2, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def print_summary(records: List[dict], metadata: dict) -> None:
    spreads = np.array([row["spread"] for row in records], dtype=np.float64)
    print()
    print("Value-spread summary")
    print("--------------------")
    print(f"Infosets         : {len(records)}")
    print(f"Mean spread      : {np.mean(spreads):.4f}")
    print(f"Median spread    : {np.median(spreads):.4f}")
    print(f"Min spread       : {np.min(spreads):.4f}")
    print(f"Max spread       : {np.max(spreads):.4f}")
    print(f"Nash coverage    : {metadata['nash_coverage']}/{metadata['n_infosets']}")

    print()
    print("Top 10 highest-spread infosets")
    print("------------------------------")
    for row in records[-10:]:
        print(
            f"{row['rank']:>3d}  {row['infoset_id']:<28s}  spread={row['spread']:.4f}  "
            f"min={row['argmin_model']:<18s}  max={row['argmax_model']:<18s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-kind",
        choices=["auto", "best", "final"],
        default="final",
        help="Which best-response checkpoint to use.",
    )
    parser.add_argument(
        "--no-nash",
        action="store_true",
        help="Skip exact CFR/Nash collapsed-observation values.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Directory for plot + CSV + JSON outputs.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    records, metadata = build_records(
        checkpoint_kind=args.checkpoint_kind,
        include_nash=not args.no_nash,
    )

    plot_path = os.path.join(args.outdir, "infoset_value_spread.png")
    csv_path = os.path.join(args.outdir, "infoset_value_spread.csv")
    json_path = os.path.join(args.outdir, "infoset_value_spread_summary.json")

    plot_records(records, plot_path)
    write_csv(records, csv_path)
    write_summary(records, metadata, json_path)
    print_summary(records, metadata)

    print()
    print(f"Plot written   : {plot_path}")
    print(f"CSV written    : {csv_path}")
    print(f"Summary written: {json_path}")


if __name__ == "__main__":
    main()
