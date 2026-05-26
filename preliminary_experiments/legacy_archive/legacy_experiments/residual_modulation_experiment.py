#!/usr/bin/env python3
"""
Residual Modulation Experiment — 2×2 factorial ablation.

Tests whether the modulated value architecture's failure is due to the gate
(gradient starvation) or the training method (self-play homogeneity), or both.

Design:
    Factor 1: Architecture — gate (V_base + gate*delta) vs no gate (V_base + delta)
    Factor 2: Training — self-play vs population (heuristic, value_based, adaptive_value)

Conditions:
    A: gate + self-play          = existing modulated_value (baseline, no retraining)
    B: gate + population         = modulated_pop_value (new training)
    C: no gate + self-play       = residual_value (new architecture)
    D: no gate + population      = residual_pop_value (new architecture + training)

Diagnostic measurements:
    1. Delta magnitude: mean |delta| across evaluation states
    2. Delta variance across opponents: is the correction opponent-specific?
    3. Delta variance across game states: is the correction state-specific?
    4. Gate distribution (conditions A, B): mean, std, range
    5. Head-to-head performance: avg chips/round against eval opponents

Mechanism prediction:
    Population training > gate removal (larger residuals = stronger gradient signal).
    D > B > C ≈ A (removing gate helps most when combined with population training).
"""

import os
import sys
import json
import time
import math
import torch
import random
from typing import Dict, List
from dataclasses import replace

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.agents import registry
from src.agents.value_based import ValueBasedAgent
from src.preliminary_experiments.promoted_registry.modulated_value import ModulatedValueAgent
from src.agents.residual_value import ResidualValueAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.engine.poker_session import PokerSession
from src.training.evaluation import evaluate_agents

RESULTS_PATH = os.path.join(ROOT, "experiments", "residual_modulation_results.json")
REPORT_PATH = os.path.join(ROOT, "experiments", "residual_modulation_report.html")

# Training config
NUM_SESSIONS = 5000
BATCH_SIZE = 32
EVAL_ROUNDS = 2000  # per matchup for head-to-head


# ── Training ─────────────────────────────────────────────────────────────

def train_condition(condition_id: str, agent_id: str, save_path: str) -> Dict:
    """Train one experimental condition and return training stats."""
    print(f"\n{'='*60}")
    print(f"Training condition {condition_id}: {agent_id}")
    print(f"{'='*60}")

    agent = registry.create(agent_id)

    # Load pretrained base weights
    base_path = os.path.join(ROOT, "models", "value_based_agent.pt")
    if os.path.exists(base_path):
        agent.model.load_state_dict(torch.load(base_path))
        for p in agent.model.parameters():
            p.requires_grad = False
        print(f"  Loaded frozen base from {base_path}")

    metadata = registry.get_metadata(agent_id)
    trainer = metadata.trainer_class(agent)

    t_start = time.time()
    losses = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])

    trainer.train(
        num_episodes=NUM_SESSIONS,
        batch_size=BATCH_SIZE,
        save_path=save_path,
        callback=callback,
    )

    elapsed = time.time() - t_start
    print(f"  Training complete in {elapsed:.1f}s, {len(losses)} updates")

    return {
        "condition": condition_id,
        "agent_id": agent_id,
        "num_sessions": NUM_SESSIONS,
        "elapsed_s": round(elapsed, 1),
        "final_loss": round(losses[-1], 6) if losses else None,
        "loss_trajectory": [round(l, 6) for l in losses[::max(1, len(losses)//50)]],
    }


# ── Diagnostics ──────────────────────────────────────────────────────────

def diagnose_delta(agent, agent_id: str, condition_id: str) -> Dict:
    """Measure delta magnitude and variance across opponents and states."""
    agent.set_train_mode(False)

    opponents = {
        "heuristic": HeuristicAgent(),
        "value_based": ValueBasedAgent(),
        "adaptive_value": AdaptiveValueAgent(),
    }

    # Load pretrained opponents
    for name, opp in opponents.items():
        opp_path = os.path.join(ROOT, "models", f"{name}_agent.pt")
        if os.path.exists(opp_path):
            opp.load_model(opp_path)
        opp.set_train_mode(False)

    has_gate = hasattr(agent, 'gate_net')
    results = {"condition": condition_id, "agent_id": agent_id, "per_opponent": {}}

    for opp_name, opponent in opponents.items():
        deltas = []
        gates = []
        session = PokerSession()

        # Play 200 hands to accumulate stats and collect delta values
        for hand_idx in range(200):
            session.new_hand()

            while not session.is_finished:
                current_player = session.current_player
                obs = session.get_observation(viewer_id=current_player)

                if current_player == 0:
                    # Collect delta (and gate) values for the training agent
                    base_enc = agent.encode_observation(obs, viewer_id=0)
                    stats_vec = agent._encode_stats(obs)

                    with torch.no_grad():
                        v_base = agent.model(base_enc)
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
            "delta_mean": round(sum(abs(d) for d in deltas) / len(deltas), 6),
            "delta_std": round(_std(deltas), 6),
            "delta_min": round(min(deltas), 6),
            "delta_max": round(max(deltas), 6),
            "delta_abs_mean": round(sum(abs(d) for d in deltas) / len(deltas), 6),
        }

        if gates:
            opp_result["gate_mean"] = round(sum(gates) / len(gates), 6)
            opp_result["gate_std"] = round(_std(gates), 6)
            opp_result["gate_min"] = round(min(gates), 6)
            opp_result["gate_max"] = round(max(gates), 6)

        results["per_opponent"][opp_name] = opp_result

    # Aggregate across opponents
    all_means = [r["delta_abs_mean"] for r in results["per_opponent"].values()]
    results["delta_grand_mean"] = round(sum(all_means) / len(all_means), 6)
    results["delta_cross_opponent_std"] = round(_std(all_means), 6)

    return results


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_condition(agent, condition_id: str) -> Dict:
    """Evaluate agent against standard opponent set."""
    agent.set_train_mode(False)

    opponents = {
        "heuristic": HeuristicAgent(),
        "value_based": ValueBasedAgent(),
        "adaptive_value": AdaptiveValueAgent(),
    }

    for name, opp in opponents.items():
        opp_path = os.path.join(ROOT, "models", f"{name}_agent.pt")
        if os.path.exists(opp_path):
            opp.load_model(opp_path)
        opp.set_train_mode(False)

    scores = {}
    for opp_name, opponent in opponents.items():
        print(f"    vs {opp_name}...", end=" ", flush=True)
        result = evaluate_agents(agent, opponent, num_rounds=EVAL_ROUNDS)
        scores[opp_name] = {
            "mean": round(result.agent_0_avg_chips, 4),
            "total": round(result.agent_0_total_chips, 1),
        }
        print(f"{result.agent_0_avg_chips:+.4f}")

    values = [s["mean"] for s in scores.values()]
    avg = sum(values) / len(values)
    worst = min(values)
    std = _std(values) if len(values) > 1 else 0.0

    return {
        "condition": condition_id,
        "per_opponent": scores,
        "avg": round(avg, 4),
        "worst_case": round(worst, 4),
        "robustness": round(avg - 1.5 * std, 4),
    }


# ── HTML Report ──────────────────────────────────────────────────────────

def generate_report(all_results: Dict):
    conditions = ["A", "B", "C", "D"]
    labels = {
        "A": "Gate + Self-play (baseline)",
        "B": "Gate + Population",
        "C": "No gate + Self-play",
        "D": "No gate + Population",
    }

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Residual Modulation Experiment</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; }
h1 { border-bottom: 2px solid #333; }
h2 { color: #555; margin-top: 2em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
th { background: #f5f5f5; text-align: left; }
.highlight { background: #e8f5e9; font-weight: bold; }
.mechanism { background: #f3e5f5; padding: 1em; border-radius: 8px; margin: 1em 0; }
pre { background: #f5f5f5; padding: 1em; border-radius: 4px; overflow-x: auto; }
</style></head><body>
<h1>Residual Modulation Experiment</h1>
<p><strong>Question:</strong> Is the modulated value agent's failure due to the gate (gradient starvation)
or the training method (self-play), or both?</p>

<div class="mechanism">
<strong>Mechanism hypothesis:</strong> The gate attenuates gradients for delta by ~0.4×,
and self-play produces small residuals (V_base is already good at self-play).
Together, these starve the modulation network of learning signal.
Removing the gate and using population training should restore gradient flow.
</div>
"""

    # Performance table
    html += "<h2>Performance (avg chips/round)</h2>\n<table>\n"
    html += "<tr><th>Condition</th><th>vs Heuristic</th><th>vs Value-Based</th><th>vs Adaptive</th><th>Average</th><th>Worst Case</th><th>Robustness</th></tr>\n"

    eval_data = all_results.get("evaluation", {})
    best_avg = max((eval_data[c]["avg"] for c in conditions if c in eval_data), default=0)

    for c in conditions:
        if c not in eval_data:
            continue
        e = eval_data[c]
        cls = ' class="highlight"' if e["avg"] == best_avg else ""
        html += f'<tr{cls}><td>{labels[c]}</td>'
        for opp in ["heuristic", "value_based", "adaptive_value"]:
            html += f'<td>{e["per_opponent"].get(opp, {}).get("mean", "N/A"):+.4f}</td>'
        html += f'<td><strong>{e["avg"]:+.4f}</strong></td>'
        html += f'<td>{e["worst_case"]:+.4f}</td>'
        html += f'<td>{e["robustness"]:+.4f}</td></tr>\n'
    html += "</table>\n"

    # Delta diagnostics table
    html += "<h2>Delta Diagnostics</h2>\n"
    html += "<p>Core test: if removing the gate and/or adding population training produce larger, more varied deltas, the gradient starvation hypothesis is confirmed.</p>\n"
    html += "<table>\n"
    html += "<tr><th>Condition</th><th>Delta |mean|</th><th>Delta cross-opp std</th>"
    html += "<th>vs Heuristic |δ|</th><th>vs Value-Based |δ|</th><th>vs Adaptive |δ|</th>"
    html += "<th>Gate mean</th><th>Gate std</th></tr>\n"

    diag_data = all_results.get("diagnostics", {})
    for c in conditions:
        if c not in diag_data:
            continue
        d = diag_data[c]
        html += f'<tr><td>{labels[c]}</td>'
        html += f'<td>{d["delta_grand_mean"]:.6f}</td>'
        html += f'<td>{d["delta_cross_opponent_std"]:.6f}</td>'
        for opp in ["heuristic", "value_based", "adaptive_value"]:
            html += f'<td>{d["per_opponent"].get(opp, {}).get("delta_abs_mean", 0):.6f}</td>'

        gate_mean = d["per_opponent"].get("heuristic", {}).get("gate_mean", None)
        gate_std = d["per_opponent"].get("heuristic", {}).get("gate_std", None)
        html += f'<td>{gate_mean:.4f}</td>' if gate_mean is not None else '<td>N/A</td>'
        html += f'<td>{gate_std:.4f}</td>' if gate_std is not None else '<td>N/A</td>'
        html += '</tr>\n'
    html += "</table>\n"

    # Factorial analysis
    html += "<h2>Factorial Decomposition</h2>\n"
    if all(c in eval_data for c in conditions):
        a, b, c_val, d_val = [eval_data[c]["avg"] for c in conditions]
        pop_effect = ((b + d_val) / 2) - ((a + c_val) / 2)
        gate_effect = ((c_val + d_val) / 2) - ((a + b) / 2)
        interaction = (d_val - c_val) - (b - a)

        html += "<table>\n"
        html += f'<tr><th>Main effect: Population training</th><td>{pop_effect:+.4f} chips/round</td></tr>\n'
        html += f'<tr><th>Main effect: Removing gate</th><td>{gate_effect:+.4f} chips/round</td></tr>\n'
        html += f'<tr><th>Interaction (gate removal × population)</th><td>{interaction:+.4f} chips/round</td></tr>\n'
        html += "</table>\n"

        html += "<p><strong>Interpretation:</strong> "
        if abs(pop_effect) > abs(gate_effect) * 2:
            html += "Population training is the dominant factor, confirming that gradient signal (larger residuals) matters more than architecture."
        elif abs(gate_effect) > abs(pop_effect) * 2:
            html += "Gate removal is the dominant factor, confirming gradient starvation as the primary bottleneck."
        else:
            html += "Both factors contribute meaningfully."

        if interaction > 0.02:
            html += " Positive interaction: removing the gate helps MORE with population training (synergistic)."
        elif interaction < -0.02:
            html += " Negative interaction: the benefits are partially redundant."
        html += "</p>\n"

    # Training curves
    html += "<h2>Training Loss</h2>\n<pre>"
    for c in conditions:
        train = all_results.get("training", {}).get(c)
        if train:
            fl = train.get('final_loss', 'N/A')
            el = train.get('elapsed_s', 'N/A')
            note = train.get('note', '')
            if note:
                html += f"{labels[c]}: {note}\n"
            else:
                html += f"{labels[c]}: final_loss={fl}, elapsed={el}s\n"
    html += "</pre>\n"

    html += "</body></html>"

    with open(REPORT_PATH, "w") as f:
        f.write(html)
    print(f"\nReport saved to: {REPORT_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RESIDUAL MODULATION EXPERIMENT")
    print("2×2 factorial: (gate vs no-gate) × (self-play vs population)")
    print("=" * 60)

    all_results = {"training": {}, "diagnostics": {}, "evaluation": {}}
    t_start = time.time()

    # Condition A: existing modulated_value (baseline — load, don't retrain)
    print("\n── Condition A: Gate + Self-play (existing baseline) ──")
    agent_a = ModulatedValueAgent()
    model_a_path = os.path.join(ROOT, "models", "modulated_value_agent.pt")
    if os.path.exists(model_a_path):
        agent_a.load_model(model_a_path)
        print(f"  Loaded existing model from {model_a_path}")
    else:
        print("  WARNING: No existing modulated_value model found!")

    all_results["training"]["A"] = {
        "condition": "A", "agent_id": "modulated_value",
        "note": "Pre-existing checkpoint, not retrained"
    }

    # Condition B: gate + population (new training)
    model_b_path = os.path.join(ROOT, "models", "modulated_pop_value_agent.pt")
    all_results["training"]["B"] = train_condition("B", "modulated_pop_value", model_b_path)
    agent_b = ModulatedValueAgent()
    if os.path.exists(model_b_path):
        agent_b.load_model(model_b_path)

    # Condition C: no gate + self-play
    model_c_path = os.path.join(ROOT, "models", "residual_value_agent.pt")
    all_results["training"]["C"] = train_condition("C", "residual_value", model_c_path)
    agent_c = ResidualValueAgent()
    if os.path.exists(model_c_path):
        agent_c.load_model(model_c_path)

    # Condition D: no gate + population
    model_d_path = os.path.join(ROOT, "models", "residual_pop_value_agent.pt")
    all_results["training"]["D"] = train_condition("D", "residual_pop_value", model_d_path)
    agent_d = ResidualValueAgent()
    if os.path.exists(model_d_path):
        agent_d.load_model(model_d_path)

    agents = {"A": agent_a, "B": agent_b, "C": agent_c, "D": agent_d}

    # Diagnostics
    print("\n" + "=" * 60)
    print("DIAGNOSTICS: Delta and Gate Analysis")
    print("=" * 60)
    for cond, agent in agents.items():
        print(f"\n── Condition {cond} ──")
        all_results["diagnostics"][cond] = diagnose_delta(
            agent, agent_id=f"condition_{cond}", condition_id=cond
        )

    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION: Head-to-Head Performance")
    print("=" * 60)
    for cond, agent in agents.items():
        print(f"\n── Condition {cond} ──")
        all_results["evaluation"][cond] = evaluate_condition(agent, cond)

    total_elapsed = time.time() - t_start
    all_results["total_elapsed_s"] = round(total_elapsed, 1)

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Generate report
    generate_report(all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    labels = {"A": "Gate+SelfPlay", "B": "Gate+Pop", "C": "NoGate+SelfPlay", "D": "NoGate+Pop"}
    for cond in ["A", "B", "C", "D"]:
        e = all_results["evaluation"].get(cond, {})
        d = all_results["diagnostics"].get(cond, {})
        print(f"  {labels[cond]:20s}  avg={e.get('avg', 'N/A'):+.4f}  "
              f"robust={e.get('robustness', 'N/A'):+.4f}  "
              f"delta_mag={d.get('delta_grand_mean', 'N/A'):.6f}")

    print(f"\nTotal time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
