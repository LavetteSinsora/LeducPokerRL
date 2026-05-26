#!/usr/bin/env python3
"""
Three Directions Experiment — improving the modulated value architecture.

Building on the 2x2 factorial findings:
  - Gate is protective (removing it hurts)
  - Population training creates larger deltas but they're inconsistent
  - Baseline (gate+self-play, avg +0.264) wins by keeping corrections tiny

Three approaches, each addressing a different failure mode:

  Dir-A  Warm-start fine-tuning: Take the strong baseline checkpoint,
         fine-tune with population training at low lr (1e-5). Preserves
         conservative behavior while adapting to specific opponents.

  Dir-B  Gate-scheduled curriculum: Phase 1 trains delta with gate=1
         (full gradient, self-play). Phase 2 unfreezes gate with population
         training. Separates "what to correct" from "when to correct."

  Dir-C  Tanh-bounded residual: Remove gate, hard-bound corrections
         to [-0.5, +0.5] via tanh. Avoids L2's gradient issues while
         architecturally constraining corrections.
"""

import os
import sys
import json
import time
import math
import copy
import torch
import torch.optim as optim
from typing import Dict, List
from dataclasses import replace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.preliminary_experiments.promoted_registry.modulated_value import ModulatedValueAgent
from src.agents.tanh_residual import TanhResidualAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.engine.poker_session import PokerSession
from src.engine.leduc_game import LeducGame
from src.training.evaluation import evaluate_agents
from src.training.modulated_pop_trainer import ModulatedPopTrainer
from src.training.residual_value_trainer import ResidualPopValueTrainer

RESULTS_PATH = os.path.join(ROOT, "experiments", "three_directions_results.json")
REPORT_PATH = os.path.join(ROOT, "experiments", "three_directions_report.html")

EVAL_ROUNDS = 2000


# ── Helpers ──────────────────────────────────────────────────────────────

def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def load_opponents():
    opponents = {}
    for name, cls in [("heuristic", HeuristicAgent),
                      ("value_based", ValueBasedAgent),
                      ("adaptive_value", AdaptiveValueAgent)]:
        opp = cls()
        path = os.path.join(ROOT, "models", f"{name}_agent.pt")
        if os.path.exists(path):
            opp.load_model(path)
        opp.set_train_mode(False)
        opponents[name] = opp
    return opponents


def diagnose_agent(agent, label: str) -> Dict:
    """Measure delta magnitude and variance across opponents."""
    agent.set_train_mode(False)
    opponents = load_opponents()
    has_gate = hasattr(agent, 'gate_net')
    results = {"label": label, "per_opponent": {}}

    for opp_name, opponent in opponents.items():
        deltas, gates = [], []
        session = PokerSession()

        for _ in range(200):
            session.new_hand()
            while not session.is_finished:
                current_player = session.current_player
                obs = session.get_observation(viewer_id=current_player)
                if current_player == 0:
                    base_enc = agent.encode_observation(obs, viewer_id=0)
                    stats_vec = agent._encode_stats(obs)
                    with torch.no_grad():
                        mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)
                        if has_gate:
                            delta = agent.mod_net(mod_input).item()
                            gate = agent.gate_net(stats_vec.unsqueeze(0)).item()
                            gates.append(gate)
                        else:
                            delta = agent.delta_net(mod_input).item()
                        deltas.append(delta)
                    action = agent.select_action(obs)
                else:
                    action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]
                session.step(action)

        opp_result = {
            "n_states": len(deltas),
            "delta_abs_mean": round(sum(abs(d) for d in deltas) / max(len(deltas), 1), 6),
            "delta_std": round(_std(deltas), 6),
            "delta_range": [round(min(deltas), 4), round(max(deltas), 4)] if deltas else [0, 0],
        }
        if gates:
            opp_result["gate_mean"] = round(sum(gates) / len(gates), 4)
            opp_result["gate_std"] = round(_std(gates), 4)
        results["per_opponent"][opp_name] = opp_result

    all_means = [r["delta_abs_mean"] for r in results["per_opponent"].values()]
    results["delta_grand_mean"] = round(sum(all_means) / len(all_means), 6)
    results["delta_cross_opponent_std"] = round(_std(all_means), 6)
    return results


def evaluate_agent(agent, label: str) -> Dict:
    agent.set_train_mode(False)
    opponents = load_opponents()
    scores = {}
    for opp_name, opponent in opponents.items():
        result = evaluate_agents(agent, opponent, num_rounds=EVAL_ROUNDS)
        scores[opp_name] = round(result.agent_0_avg_chips, 4)
        print(f"    {label} vs {opp_name}: {result.agent_0_avg_chips:+.4f}")

    values = list(scores.values())
    avg = sum(values) / len(values)
    worst = min(values)
    std = _std(values)
    return {
        "label": label,
        "per_opponent": scores,
        "avg": round(avg, 4),
        "worst_case": round(worst, 4),
        "robustness": round(avg - 1.5 * std, 4),
    }


# ── Direction A: Warm-Start Fine-Tuning ──────────────────────────────────

def train_direction_a() -> tuple:
    """Fine-tune existing modulated_value checkpoint with population at low lr."""
    print("\n" + "=" * 60)
    print("DIRECTION A: Warm-Start Population Fine-Tuning")
    print("=" * 60)

    agent = ModulatedValueAgent()
    base_ckpt = os.path.join(ROOT, "models", "modulated_value_agent.pt")
    if os.path.exists(base_ckpt):
        agent.load_model(base_ckpt)
        print(f"  Loaded warm-start checkpoint from {base_ckpt}")

    trainer = ModulatedPopTrainer(agent, learning_rate=1e-5)

    t_start = time.time()
    losses = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])

    trainer.train(num_episodes=3000, batch_size=32, callback=callback,
                  save_path=os.path.join(ROOT, "models", "dir_a_warmstart_agent.pt"))

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.1f}s, {len(losses)} updates")

    return agent, {
        "direction": "A", "name": "Warm-Start Fine-Tune",
        "num_sessions": 3000, "elapsed_s": round(elapsed, 1),
        "final_loss": round(losses[-1], 6) if losses else None,
    }


# ── Direction B: Gate-Scheduled Curriculum ────────────────────────────────

def train_direction_b() -> tuple:
    """Phase 1: gate=1 + self-play (delta learns). Phase 2: gate unfrozen + population."""
    print("\n" + "=" * 60)
    print("DIRECTION B: Gate-Scheduled Curriculum")
    print("=" * 60)

    agent = ModulatedValueAgent()
    base_path = os.path.join(ROOT, "models", "value_based_agent.pt")
    if os.path.exists(base_path):
        agent.model.load_state_dict(torch.load(base_path))
        for p in agent.model.parameters():
            p.requires_grad = False
        print(f"  Loaded frozen base from {base_path}")

    # ── Phase 1: gate=1, train only delta, self-play ──
    print("\n  Phase 1: gate=1.0, delta-only, self-play (2500 sessions)")

    # Freeze gate and force output to 1.0
    for p in agent.gate_net.parameters():
        p.requires_grad = False

    # Store original gate forward, replace with constant 1.0
    original_gate_forward = agent.gate_net.forward
    agent.gate_net.forward = lambda x: torch.ones(x.shape[0], 1)

    # Use adaptive trainer (self-play) with modulated value computation
    from src.training.modulated_value_trainer import ModulatedValueTrainer
    trainer_p1 = ModulatedValueTrainer(agent, learning_rate=1e-4)
    # Override optimizer to only train mod_net (gate is frozen)
    trainer_p1.optimizer = optim.Adam(agent.mod_net.parameters(), lr=1e-4)

    t_start = time.time()
    losses_p1 = []

    def cb1(data):
        if data["type"] == "batch_update":
            losses_p1.append(data["loss"])

    trainer_p1.train(num_episodes=2500, batch_size=32, callback=cb1)
    elapsed_p1 = time.time() - t_start
    print(f"  Phase 1 done: {elapsed_p1:.1f}s, {len(losses_p1)} updates")

    # ── Phase 2: unfreeze gate, population training ──
    print("\n  Phase 2: gate unfrozen, population training (2500 sessions)")

    # Restore gate
    agent.gate_net.forward = original_gate_forward
    for p in agent.gate_net.parameters():
        p.requires_grad = True

    trainer_p2 = ModulatedPopTrainer(agent, learning_rate=1e-4)

    losses_p2 = []

    def cb2(data):
        if data["type"] == "batch_update":
            losses_p2.append(data["loss"])

    t2_start = time.time()
    trainer_p2.train(num_episodes=2500, batch_size=32, callback=cb2,
                     save_path=os.path.join(ROOT, "models", "dir_b_curriculum_agent.pt"))
    elapsed_p2 = time.time() - t2_start
    total_elapsed = elapsed_p1 + elapsed_p2
    print(f"  Phase 2 done: {elapsed_p2:.1f}s, {len(losses_p2)} updates")

    return agent, {
        "direction": "B", "name": "Gate-Scheduled Curriculum",
        "num_sessions": 5000,
        "phase1_sessions": 2500, "phase1_elapsed_s": round(elapsed_p1, 1),
        "phase2_sessions": 2500, "phase2_elapsed_s": round(elapsed_p2, 1),
        "elapsed_s": round(total_elapsed, 1),
        "phase1_final_loss": round(losses_p1[-1], 6) if losses_p1 else None,
        "phase2_final_loss": round(losses_p2[-1], 6) if losses_p2 else None,
    }


# ── Direction C: Tanh-Bounded Residual ────────────────────────────────────

def train_direction_c() -> tuple:
    """Tanh-bounded delta, no gate, no L2, population training."""
    print("\n" + "=" * 60)
    print("DIRECTION C: Tanh-Bounded Residual")
    print("=" * 60)

    agent = TanhResidualAgent(max_correction=0.5)
    base_path = os.path.join(ROOT, "models", "value_based_agent.pt")
    if os.path.exists(base_path):
        agent.model.load_state_dict(torch.load(base_path))
        for p in agent.model.parameters():
            p.requires_grad = False
        print(f"  Loaded frozen base from {base_path}")

    # Custom trainer: population, no weight_decay
    trainer = ResidualPopValueTrainer(agent)
    # Override optimizer: no weight_decay for tanh-bounded
    trainer.optimizer = optim.Adam(agent.delta_net.parameters(), lr=1e-4, weight_decay=0.0)

    t_start = time.time()
    losses = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])

    trainer.train(num_episodes=5000, batch_size=32, callback=callback,
                  save_path=os.path.join(ROOT, "models", "dir_c_tanh_agent.pt"))

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.1f}s, {len(losses)} updates")

    return agent, {
        "direction": "C", "name": "Tanh-Bounded Residual",
        "num_sessions": 5000, "elapsed_s": round(elapsed, 1),
        "final_loss": round(losses[-1], 6) if losses else None,
        "max_correction": 0.5,
    }


# ── Baseline ──────────────────────────────────────────────────────────────

def load_baseline() -> tuple:
    """Load the original modulated_value checkpoint as baseline."""
    agent = ModulatedValueAgent()
    path = os.path.join(ROOT, "models", "modulated_value_agent.pt")
    if os.path.exists(path):
        agent.load_model(path)
    return agent, {"direction": "Baseline", "name": "Gate+SelfPlay (original)"}


# ── Report ────────────────────────────────────────────────────────────────

def generate_report(all_results: Dict):
    directions = ["Baseline", "A", "B", "C"]
    labels = {
        "Baseline": "Baseline (Gate+SelfPlay)",
        "A": "Dir-A: Warm-Start Fine-Tune",
        "B": "Dir-B: Gate-Scheduled Curriculum",
        "C": "Dir-C: Tanh-Bounded Residual",
    }

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Three Directions Experiment</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; line-height: 1.6; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }
h2 { color: #444; margin-top: 2em; }
h3 { color: #666; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
th { background: #f5f5f5; text-align: left; }
.best { background: #e8f5e9; font-weight: bold; }
.mechanism { background: #f3e5f5; padding: 1em; border-radius: 8px; margin: 1em 0; font-size: 0.95em; }
.finding { background: #e3f2fd; padding: 1em; border-radius: 8px; margin: 1em 0; }
.negative { color: #c62828; }
.positive { color: #2e7d32; }
pre { background: #f5f5f5; padding: 1em; border-radius: 4px; overflow-x: auto; font-size: 0.9em; }
</style></head><body>
<h1>Three Directions Experiment</h1>

<p><strong>Context:</strong> The 2&times;2 factorial showed the gate is protective (removing it hurts, &minus;0.128 avg)
and population training creates larger deltas that are inconsistent. The baseline (gate+self-play, avg +0.264)
wins by keeping corrections conservatively small.</p>

<h2>Directions Tested</h2>
<div class="mechanism">
<strong>A &mdash; Warm-Start Fine-Tune:</strong> Load baseline checkpoint, fine-tune with population at lr=1e-5.
Preserve conservative structure while adapting to known opponents.<br>
<strong>B &mdash; Gate-Scheduled Curriculum:</strong> Phase 1: gate=1.0, train delta only (full gradient, self-play).
Phase 2: unfreeze gate, population training. Separates "what" from "when."<br>
<strong>C &mdash; Tanh-Bounded Residual:</strong> No gate. Hard-bound corrections to [-0.5, +0.5] via tanh.
No L2 penalty. Population training.
</div>
"""

    # Performance table
    eval_data = all_results.get("evaluation", {})
    best_avg = max((eval_data[d]["avg"] for d in directions if d in eval_data), default=-99)

    html += "<h2>Performance (avg chips/round, 2000 rounds/matchup)</h2>\n<table>\n"
    html += "<tr><th>Direction</th><th>vs Heuristic</th><th>vs Value-Based</th><th>vs Adaptive</th>"
    html += "<th>Average</th><th>Worst Case</th><th>Robustness</th></tr>\n"

    for d in directions:
        if d not in eval_data:
            continue
        e = eval_data[d]
        cls = ' class="best"' if e["avg"] == best_avg else ""
        html += f'<tr{cls}><td>{labels[d]}</td>'
        for opp in ["heuristic", "value_based", "adaptive_value"]:
            v = e["per_opponent"].get(opp, 0)
            color = "positive" if v > 0 else "negative" if v < 0 else ""
            html += f'<td class="{color}">{v:+.4f}</td>'
        html += f'<td><strong>{e["avg"]:+.4f}</strong></td>'
        html += f'<td>{e["worst_case"]:+.4f}</td>'
        html += f'<td>{e["robustness"]:+.4f}</td></tr>\n'
    html += "</table>\n"

    # Delta diagnostics
    diag_data = all_results.get("diagnostics", {})
    html += "<h2>Delta Diagnostics</h2>\n"
    html += "<p>Core question: are corrections larger, more opponent-specific, and more helpful?</p>\n"
    html += "<table>\n<tr><th>Direction</th><th>Delta |mean|</th><th>Cross-opp std</th>"
    html += "<th>vs Heuristic |&delta;|</th><th>vs VB |&delta;|</th><th>vs Adaptive |&delta;|</th>"
    html += "<th>Gate mean</th></tr>\n"

    for d in directions:
        if d not in diag_data:
            continue
        diag = diag_data[d]
        html += f'<tr><td>{labels[d]}</td>'
        html += f'<td>{diag["delta_grand_mean"]:.4f}</td>'
        html += f'<td>{diag["delta_cross_opponent_std"]:.4f}</td>'
        for opp in ["heuristic", "value_based", "adaptive_value"]:
            v = diag["per_opponent"].get(opp, {}).get("delta_abs_mean", 0)
            html += f'<td>{v:.4f}</td>'
        gate = diag["per_opponent"].get("heuristic", {}).get("gate_mean")
        html += f'<td>{gate:.4f}</td>' if gate is not None else '<td>N/A</td>'
        html += '</tr>\n'
    html += "</table>\n"

    # Training info
    train_data = all_results.get("training", {})
    html += "<h2>Training Details</h2>\n<pre>"
    for d in directions:
        if d not in train_data:
            continue
        t = train_data[d]
        name = t.get("name", d)
        sessions = t.get("num_sessions", "N/A")
        elapsed = t.get("elapsed_s", "N/A")
        fl = t.get("final_loss") or t.get("phase2_final_loss", "N/A")
        html += f"{name}: {sessions} sessions, {elapsed}s, final_loss={fl}\n"
    html += "</pre>\n"

    # Analysis
    html += "<h2>Analysis</h2>\n"

    if all(d in eval_data for d in directions):
        base_avg = eval_data["Baseline"]["avg"]
        html += "<h3>Improvement over Baseline</h3>\n<table>\n"
        html += "<tr><th>Direction</th><th>&Delta; avg</th><th>&Delta; robustness</th><th>Verdict</th></tr>\n"
        for d in ["A", "B", "C"]:
            da = eval_data[d]["avg"] - base_avg
            dr = eval_data[d]["robustness"] - eval_data["Baseline"]["robustness"]
            verdict = "Improved" if da > 0.01 else "Comparable" if da > -0.01 else "Degraded"
            html += f'<tr><td>{labels[d]}</td><td>{da:+.4f}</td><td>{dr:+.4f}</td><td>{verdict}</td></tr>\n'
        html += "</table>\n"

    html += """
<h3>Key Findings</h3>
<div class="finding" id="findings"></div>

<h3>What Was Learned</h3>
<div class="finding" id="lessons"></div>
"""

    html += "</body></html>"

    with open(REPORT_PATH, "w") as f:
        f.write(html)
    print(f"\nReport saved to: {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("THREE DIRECTIONS EXPERIMENT")
    print("Improving the modulated value architecture")
    print("=" * 60)

    all_results = {"training": {}, "diagnostics": {}, "evaluation": {}}
    t_start = time.time()

    # Baseline
    baseline_agent, baseline_info = load_baseline()
    all_results["training"]["Baseline"] = baseline_info

    # Train all 3 directions
    agent_a, info_a = train_direction_a()
    all_results["training"]["A"] = info_a

    agent_b, info_b = train_direction_b()
    all_results["training"]["B"] = info_b

    agent_c, info_c = train_direction_c()
    all_results["training"]["C"] = info_c

    agents = {"Baseline": baseline_agent, "A": agent_a, "B": agent_b, "C": agent_c}

    # Diagnostics
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    for key, agent in agents.items():
        print(f"\n  {key}:")
        all_results["diagnostics"][key] = diagnose_agent(agent, key)
        d = all_results["diagnostics"][key]
        print(f"    delta |mean|={d['delta_grand_mean']:.4f}, cross-opp std={d['delta_cross_opponent_std']:.4f}")

    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    for key, agent in agents.items():
        print(f"\n  {key}:")
        all_results["evaluation"][key] = evaluate_agent(agent, key)

    total_elapsed = time.time() - t_start
    all_results["total_elapsed_s"] = round(total_elapsed, 1)

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_PATH}")

    generate_report(all_results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key in ["Baseline", "A", "B", "C"]:
        e = all_results["evaluation"][key]
        d = all_results["diagnostics"][key]
        print(f"  {key:10s}  avg={e['avg']:+.4f}  robust={e['robustness']:+.4f}  "
              f"delta={d['delta_grand_mean']:.4f}")

    base_avg = all_results["evaluation"]["Baseline"]["avg"]
    print(f"\n  Baseline avg: {base_avg:+.4f}")
    for key in ["A", "B", "C"]:
        delta = all_results["evaluation"][key]["avg"] - base_avg
        print(f"  {key} improvement: {delta:+.4f}")

    print(f"\nTotal time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
