#!/usr/bin/env python3
"""
Rigorous Ablation Investigation: ModulatedValueAgent gate=0 vs ValueBasedAgent

This script investigates why the original ablation study showed a performance
difference between:
  - "Base only (gate=0, no delta)" -> avg +0.163
  - "Plain pretrained base (original value_based)" -> avg +0.068

If gate=0, the modulated agent's value is just V_base(s). This should be
*identical* to the plain ValueBasedAgent if the weights are the same.

Investigation plan:
  1. WEIGHT IDENTITY CHECK: Compare base weights inside modulated_value_agent.pt
     vs value_based_agent.pt — are they byte-identical?
  2. DETERMINISTIC VALUE CHECK: Feed identical observations to both agents and
     compare V(s) outputs — do they produce identical values?
  3. DETERMINISTIC ACTION CHECK: Given identical game states, do both agents
     choose exactly the same actions?
  4. HEAD-TO-HEAD ABLATION: Run controlled experiments with many more rounds
     (10,000 instead of 1,000) and compute confidence intervals.
  5. GATE DISTRIBUTION: Collect gate values during actual gameplay and report
     distribution statistics + histogram data.

All results saved to JSON + HTML visualization.
"""

import json
import os
import sys
import time
import random
import math
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preliminary_experiments.promoted_registry.modulated_value import ModulatedValueAgent, ModulationNetwork, GateNetwork
from src.agents.value_based import ValueBasedAgent, ValueNetwork
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.heuristic import HeuristicAgent
from src.training.evaluation import evaluate_agents, quick_evaluate
from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession, OpponentStats

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "modulated_value_agent.pt")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "value_based_agent.pt")
ADAPTIVE_MODEL_PATH = os.path.join(BASE_DIR, "models", "adaptive_value_agent.pt")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ablation_investigation_results.json")
HTML_PATH = os.path.join(os.path.dirname(__file__), "ablation_investigation_report.html")

EVAL_ROUNDS = 10000  # High round count for statistical power


# ═════════════════════════════════════════════════════════════════════
# Ablated Agent
# ═════════════════════════════════════════════════════════════════════

class AblatedAgent(ModulatedValueAgent):
    """ModulatedValueAgent with explicit gate override and gate tracking."""

    def __init__(self, model_path, gate_override=None):
        super().__init__(model_path=model_path)
        self.gate_override = gate_override
        self.gate_log = []  # Track gate values during play

    def _get_value(self, obs, viewer_id):
        base_enc = self.encode_observation(obs, viewer_id=viewer_id)
        stats_vec = self._encode_stats(obs)

        with torch.no_grad():
            v_base = self.model(base_enc)

            if self.gate_override == 0.0:
                return v_base.item()

            mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)
            delta = self.mod_net(mod_input)

            if self.gate_override is not None:
                gate_val = self.gate_override
            else:
                gate_val = self.gate_net(stats_vec.unsqueeze(0)).item()

            self.gate_log.append(gate_val)
            value = v_base + gate_val * delta
            return value.item()


# ═════════════════════════════════════════════════════════════════════
# Experiment 1: Weight Identity Check
# ═════════════════════════════════════════════════════════════════════

def experiment_weight_check():
    """Compare base weights in modulated checkpoint vs standalone checkpoint."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Weight Identity Check")
    print("=" * 70)

    mod_data = torch.load(MODEL_PATH)
    mod_base_weights = mod_data["base"]
    vba_weights = torch.load(BASE_MODEL_PATH)

    results = {
        "mod_checkpoint_keys": list(mod_data.keys()) if isinstance(mod_data, dict) else "raw_state_dict",
        "mod_base_param_names": list(mod_base_weights.keys()),
        "vba_param_names": list(vba_weights.keys()),
    }

    keys_match = set(mod_base_weights.keys()) == set(vba_weights.keys())
    results["parameter_names_match"] = keys_match
    print(f"  Parameter names match: {keys_match}")

    if not keys_match:
        results["missing_in_mod"] = list(set(vba_weights.keys()) - set(mod_base_weights.keys()))
        results["extra_in_mod"] = list(set(mod_base_weights.keys()) - set(vba_weights.keys()))
        return results

    per_param = {}
    all_identical = True
    total_diff = 0.0
    total_params = 0

    for key in mod_base_weights:
        w_mod = mod_base_weights[key]
        w_vba = vba_weights[key]

        byte_identical = torch.equal(w_mod, w_vba)
        max_diff = (w_mod - w_vba).abs().max().item()
        mean_diff = (w_mod - w_vba).abs().mean().item()
        l2_diff = (w_mod - w_vba).norm().item()
        n_params = w_mod.numel()

        per_param[key] = {
            "shape": list(w_mod.shape),
            "byte_identical": byte_identical,
            "max_abs_diff": round(max_diff, 10),
            "mean_abs_diff": round(mean_diff, 10),
            "l2_diff": round(l2_diff, 10),
            "n_params": n_params,
        }

        if not byte_identical:
            all_identical = False
            total_diff += l2_diff
        total_params += n_params

        status = "IDENTICAL" if byte_identical else f"DIFFER (max_diff={max_diff:.2e})"
        print(f"  {key:30s} {str(list(w_mod.shape)):15s} {status}")

    results["per_parameter"] = per_param
    results["all_weights_identical"] = all_identical
    results["total_l2_diff"] = round(total_diff, 10)
    results["total_params"] = total_params

    print(f"\n  ALL WEIGHTS IDENTICAL: {all_identical}")
    if not all_identical:
        print(f"  Total L2 difference: {total_diff:.2e}")
    else:
        print(f"  The base network inside modulated_value_agent.pt is a bit-for-bit")
        print(f"  copy of value_based_agent.pt. No weight drift occurred during training.")

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 2: Deterministic Value + Action Check
# ═════════════════════════════════════════════════════════════════════

def experiment_deterministic_check():
    """Feed identical observations and check if outputs are identical."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Deterministic Value & Action Check")
    print("=" * 70)

    ablated_gate0 = AblatedAgent(model_path=MODEL_PATH, gate_override=0.0)
    plain_vba = ValueBasedAgent(model_path=BASE_MODEL_PATH)

    # Also build a VBA loaded from the modulated checkpoint's base weights
    vba_from_mod = ValueBasedAgent()
    mod_data = torch.load(MODEL_PATH)
    vba_from_mod.model.load_state_dict(mod_data["base"])
    vba_from_mod.model.eval()

    results = {
        "n_states": 0,
        "value_mismatches_gate0_vs_plain": 0,
        "value_mismatches_gate0_vs_mod_base": 0,
        "action_mismatches_gate0_vs_plain": 0,
        "action_mismatches_gate0_vs_mod_base": 0,
        "max_value_diff_gate0_vs_plain": 0.0,
        "max_value_diff_gate0_vs_mod_base": 0.0,
        "mismatched_examples": [],
    }

    random.seed(42)
    torch.manual_seed(42)
    session = PokerSession()

    n_checks = 0
    for game_idx in range(500):
        session.new_hand()
        while not session.is_finished:
            player = session.current_player
            obs = session.get_observation(viewer_id=player)

            evals_gate0 = ablated_gate0.get_action_evaluations(obs)
            evals_plain = plain_vba.get_action_evaluations(obs)
            evals_mod_base = vba_from_mod.get_action_evaluations(obs)

            for e0, ep, em in zip(evals_gate0, evals_plain, evals_mod_base):
                assert e0["action"] == ep["action"] == em["action"], "Action order mismatch"

                diff_plain = abs(e0["value"] - ep["value"])
                diff_mod_base = abs(e0["value"] - em["value"])

                if diff_plain > 1e-8:
                    results["value_mismatches_gate0_vs_plain"] += 1
                if diff_mod_base > 1e-8:
                    results["value_mismatches_gate0_vs_mod_base"] += 1

                results["max_value_diff_gate0_vs_plain"] = max(
                    results["max_value_diff_gate0_vs_plain"], diff_plain)
                results["max_value_diff_gate0_vs_mod_base"] = max(
                    results["max_value_diff_gate0_vs_mod_base"], diff_mod_base)

                if diff_plain > 1e-6 and len(results["mismatched_examples"]) < 5:
                    results["mismatched_examples"].append({
                        "game": game_idx,
                        "action": e0["action"].name,
                        "gate0_value": round(e0["value"], 8),
                        "plain_value": round(ep["value"], 8),
                        "mod_base_value": round(em["value"], 8),
                        "diff_plain": round(diff_plain, 10),
                        "diff_mod_base": round(diff_mod_base, 10),
                    })

            n_checks += len(evals_gate0)

            action_gate0 = max(evals_gate0, key=lambda x: x["value"])["action"]
            action_plain = max(evals_plain, key=lambda x: x["value"])["action"]
            action_mod_base = max(evals_mod_base, key=lambda x: x["value"])["action"]

            if action_gate0 != action_plain:
                results["action_mismatches_gate0_vs_plain"] += 1
            if action_gate0 != action_mod_base:
                results["action_mismatches_gate0_vs_mod_base"] += 1

            action = random.choice(obs.legal_actions)
            session.step(action)

    results["n_states"] = n_checks

    print(f"  Checked {n_checks} state-action evaluations across 500 games")
    print(f"\n  Gate=0 vs Plain VBA:")
    print(f"    Value mismatches (>1e-8): {results['value_mismatches_gate0_vs_plain']}")
    print(f"    Max value diff:           {results['max_value_diff_gate0_vs_plain']:.2e}")
    print(f"    Action mismatches:        {results['action_mismatches_gate0_vs_plain']}")
    print(f"\n  Gate=0 vs VBA-from-mod-checkpoint:")
    print(f"    Value mismatches (>1e-8): {results['value_mismatches_gate0_vs_mod_base']}")
    print(f"    Max value diff:           {results['max_value_diff_gate0_vs_mod_base']:.2e}")
    print(f"    Action mismatches:        {results['action_mismatches_gate0_vs_mod_base']}")

    if results["value_mismatches_gate0_vs_plain"] == 0:
        print(f"\n  CONFIRMED: gate=0 modulated agent produces IDENTICAL values to plain VBA")
    else:
        print(f"\n  DIFFERENCE FOUND between gate=0 and plain VBA!")
        if results["mismatched_examples"]:
            print(f"    Examples:")
            for ex in results["mismatched_examples"]:
                print(f"      Game {ex['game']}, {ex['action']}: "
                      f"gate0={ex['gate0_value']:.8f} vs plain={ex['plain_value']:.8f} "
                      f"(diff={ex['diff_plain']:.2e})")

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 3: Head-to-Head Ablation (high sample count)
# ═════════════════════════════════════════════════════════════════════

def _normal_cdf(x):
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def experiment_ablation_h2h(eval_rounds=EVAL_ROUNDS):
    """Run controlled ablation with high sample count and confidence intervals."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: Head-to-Head Ablation ({eval_rounds} rounds each)")
    print("=" * 70)

    print("  Loading agents...")
    full_model = AblatedAgent(model_path=MODEL_PATH, gate_override=None)
    gate_zero = AblatedAgent(model_path=MODEL_PATH, gate_override=0.0)
    gate_one = AblatedAgent(model_path=MODEL_PATH, gate_override=1.0)
    plain_vba = ValueBasedAgent(model_path=BASE_MODEL_PATH)

    vba_from_mod = ValueBasedAgent()
    mod_data = torch.load(MODEL_PATH)
    vba_from_mod.model.load_state_dict(mod_data["base"])
    vba_from_mod.model.eval()

    variants = {
        "full_model_learned_gate": full_model,
        "gate_1_no_gating": gate_one,
        "gate_0_base_only": gate_zero,
        "plain_vba_original_checkpoint": plain_vba,
        "vba_from_mod_base_weights": vba_from_mod,
    }

    opponents = {
        "heuristic": HeuristicAgent(),
        "value_based": ValueBasedAgent(model_path=BASE_MODEL_PATH),
        "adaptive_value": AdaptiveValueAgent(model_path=ADAPTIVE_MODEL_PATH),
    }

    results = {}

    for variant_name, variant_agent in variants.items():
        print(f"\n  --- {variant_name} ---")
        variant_results = {}

        for opp_name, opp_agent in opponents.items():
            t0 = time.time()
            result = evaluate_agents(variant_agent, opp_agent, num_rounds=eval_rounds)
            elapsed = time.time() - t0

            scores = [r[0] for r in result.round_results]
            n = len(scores)
            mean = sum(scores) / n
            std = math.sqrt(sum((s - mean)**2 for s in scores) / (n - 1))
            se = std / math.sqrt(n)
            ci95 = 1.96 * se

            variant_results[opp_name] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95": round(ci95, 4),
                "n_rounds": n,
                "elapsed_s": round(elapsed, 1),
            }
            print(f"    vs {opp_name:20s}: {mean:+.4f} +/- {ci95:.4f} (95% CI) "
                  f"[std={std:.3f}] ({elapsed:.1f}s)")

        means = [v["mean"] for v in variant_results.values()]
        avg = sum(means) / len(means)
        variant_results["average"] = round(avg, 4)
        print(f"    {'AVERAGE':>22s}: {avg:+.4f}")

        results[variant_name] = variant_results

    # Summary table
    print(f"\n  --- Summary ---")
    print(f"  {'Variant':<35s} {'Avg':>8s} {'vs heur':>8s} {'vs vb':>8s} {'vs av':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for vname, vdata in results.items():
        print(f"  {vname:<35s} {vdata['average']:+8.4f} "
              f"{vdata['heuristic']['mean']:+8.4f} "
              f"{vdata['value_based']['mean']:+8.4f} "
              f"{vdata['adaptive_value']['mean']:+8.4f}")

    g0_avg = results["gate_0_base_only"]["average"]
    pv_avg = results["plain_vba_original_checkpoint"]["average"]
    mb_avg = results["vba_from_mod_base_weights"]["average"]

    print(f"\n  KEY COMPARISONS:")
    print(f"    gate=0 average:                 {g0_avg:+.4f}")
    print(f"    plain VBA average:              {pv_avg:+.4f}")
    print(f"    VBA-from-mod-base average:      {mb_avg:+.4f}")
    print(f"    gate=0 - plain VBA:             {g0_avg - pv_avg:+.4f}")
    print(f"    gate=0 - VBA-from-mod-base:     {g0_avg - mb_avg:+.4f}")
    print(f"    plain VBA - VBA-from-mod-base:  {pv_avg - mb_avg:+.4f}")

    # Per-opponent statistical tests
    for opp_name in opponents:
        g0 = results["gate_0_base_only"][opp_name]
        pv = results["plain_vba_original_checkpoint"][opp_name]
        diff = g0["mean"] - pv["mean"]
        pooled_se = math.sqrt(g0["std"]**2/g0["n_rounds"] + pv["std"]**2/pv["n_rounds"])
        z = diff / pooled_se if pooled_se > 0 else 0
        p_approx = 2 * (1 - _normal_cdf(abs(z)))
        sig = "(significant)" if p_approx < 0.05 else "(NOT significant)"
        print(f"    vs {opp_name}: diff={diff:+.4f}, z={z:.2f}, p={p_approx:.4f} {sig}")

    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment 4: Gate Distribution During Gameplay
# ═════════════════════════════════════════════════════════════════════

def experiment_gate_distribution(n_rounds=5000):
    """Collect gate values during actual gameplay against various opponents."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 4: Gate Distribution During Gameplay ({n_rounds} rounds)")
    print("=" * 70)

    opponents = {
        "heuristic": HeuristicAgent(),
        "value_based": ValueBasedAgent(model_path=BASE_MODEL_PATH),
        "adaptive_value": AdaptiveValueAgent(model_path=ADAPTIVE_MODEL_PATH),
    }

    results = {}

    for opp_name, opp_agent in opponents.items():
        agent = AblatedAgent(model_path=MODEL_PATH, gate_override=None)
        agent.gate_log = []

        evaluate_agents(agent, opp_agent, num_rounds=n_rounds)

        gates = agent.gate_log
        if not gates:
            print(f"  WARNING: No gate values collected for {opp_name}")
            continue

        gates_arr = np.array(gates)
        percentiles = {
            "p5": float(np.percentile(gates_arr, 5)),
            "p25": float(np.percentile(gates_arr, 25)),
            "p50": float(np.percentile(gates_arr, 50)),
            "p75": float(np.percentile(gates_arr, 75)),
            "p95": float(np.percentile(gates_arr, 95)),
        }

        hist_counts, hist_edges = np.histogram(gates_arr, bins=50)
        histogram = {
            "counts": hist_counts.tolist(),
            "bin_edges": [round(e, 6) for e in hist_edges.tolist()],
        }

        opp_result = {
            "n_values": len(gates),
            "mean": round(float(gates_arr.mean()), 6),
            "std": round(float(gates_arr.std()), 6),
            "min": round(float(gates_arr.min()), 6),
            "max": round(float(gates_arr.max()), 6),
            "percentiles": {k: round(v, 6) for k, v in percentiles.items()},
            "histogram": histogram,
        }
        results[opp_name] = opp_result

        print(f"\n  vs {opp_name}: {len(gates)} gate values collected")
        print(f"    Mean={opp_result['mean']:.4f}  Std={opp_result['std']:.4f}  "
              f"Range=[{opp_result['min']:.4f}, {opp_result['max']:.4f}]")
        print(f"    Percentiles: p5={percentiles['p5']:.4f}  p25={percentiles['p25']:.4f}  "
              f"p50={percentiles['p50']:.4f}  p75={percentiles['p75']:.4f}  "
              f"p95={percentiles['p95']:.4f}")

    return results


# ═════════════════════════════════════════════════════════════════════
# HTML Report Generator
# ═════════════════════════════════════════════════════════════════════

def generate_html_report(all_results):
    """Generate a comprehensive HTML report with embedded visualizations."""

    weights_data = all_results.get("weight_check", {})
    det_data = all_results.get("deterministic_check", {})
    ablation_data = all_results.get("ablation_h2h", {})
    gate_data = all_results.get("gate_distribution", {})

    variant_display = {
        "full_model_learned_gate": "Full Model (learned gate)",
        "gate_1_no_gating": "Gate=1 (no gating)",
        "gate_0_base_only": "Gate=0 (base only)",
        "plain_vba_original_checkpoint": "Plain VBA (original ckpt)",
        "vba_from_mod_base_weights": "VBA from mod base",
    }
    vname_order = list(variant_display.keys())

    # Build ablation chart data
    chart_labels, chart_heur, chart_vb, chart_av, chart_avg = [], [], [], [], []
    for vname in vname_order:
        if vname not in ablation_data:
            continue
        vdata = ablation_data[vname]
        chart_labels.append(variant_display[vname])
        chart_heur.append(vdata.get("heuristic", {}).get("mean", 0))
        chart_vb.append(vdata.get("value_based", {}).get("mean", 0))
        chart_av.append(vdata.get("adaptive_value", {}).get("mean", 0))
        chart_avg.append(vdata.get("average", 0))

    # Build gate histogram data
    gate_hist_json = {}
    for opp_name, gdata in gate_data.items():
        if "histogram" in gdata:
            edges = gdata["histogram"]["bin_edges"]
            centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]
            gate_hist_json[opp_name] = {
                "centers": centers,
                "counts": gdata["histogram"]["counts"],
                "mean": gdata["mean"],
                "std": gdata["std"],
            }

    # ─── Build HTML table rows as strings ───
    weight_rows = ""
    for k, v in weights_data.get("per_parameter", {}).items():
        ident = "✓" if v["byte_identical"] else "✗"
        weight_rows += (f'<tr><td><code>{k}</code></td>'
                        f'<td class="num">{v["shape"]}</td>'
                        f'<td>{ident}</td>'
                        f'<td class="num">{v["max_abs_diff"]:.2e}</td></tr>\n')

    ablation_rows = ""
    for vname in vname_order:
        if vname not in ablation_data:
            continue
        vdata = ablation_data[vname]
        dn = variant_display[vname]
        h_m = vdata.get("heuristic", {}).get("mean", 0)
        h_c = vdata.get("heuristic", {}).get("ci95", 0)
        v_m = vdata.get("value_based", {}).get("mean", 0)
        v_c = vdata.get("value_based", {}).get("ci95", 0)
        a_m = vdata.get("adaptive_value", {}).get("mean", 0)
        a_c = vdata.get("adaptive_value", {}).get("ci95", 0)
        avg = vdata.get("average", 0)
        ablation_rows += (
            f'<tr><td>{dn}</td>'
            f'<td class="num {"positive" if h_m > 0 else "negative"}">{h_m:+.4f} +/- {h_c:.4f}</td>'
            f'<td class="num {"positive" if v_m > 0 else "negative"}">{v_m:+.4f} +/- {v_c:.4f}</td>'
            f'<td class="num {"positive" if a_m > 0 else "negative"}">{a_m:+.4f} +/- {a_c:.4f}</td>'
            f'<td class="num {"positive" if avg > 0 else "negative"}"><b>{avg:+.4f}</b></td></tr>\n'
        )

    gate_rows = ""
    for opp_name, gdata in gate_data.items():
        if "n_values" not in gdata:
            continue
        p = gdata["percentiles"]
        gate_rows += (
            f'<tr><td>{opp_name}</td>'
            f'<td class="num">{gdata["n_values"]}</td>'
            f'<td class="num">{gdata["mean"]:.4f}</td>'
            f'<td class="num">{gdata["std"]:.4f}</td>'
            f'<td class="num">{gdata["min"]:.4f}</td>'
            f'<td class="num">{p["p25"]:.4f}</td>'
            f'<td class="num">{p["p50"]:.4f}</td>'
            f'<td class="num">{p["p75"]:.4f}</td>'
            f'<td class="num">{gdata["max"]:.4f}</td></tr>\n'
        )

    # Verdict strings
    w_ok = weights_data.get("all_weights_identical", False)
    w_cls = "verdict-pass" if w_ok else "verdict-fail"
    w_txt = ("All weights are BIT-FOR-BIT IDENTICAL. No weight drift during training."
             if w_ok else "Weights DIFFER between the two checkpoints.")

    det_vm = det_data.get("value_mismatches_gate0_vs_plain", -1)
    det_am = det_data.get("action_mismatches_gate0_vs_plain", -1)
    det_n = det_data.get("n_states", "N/A")
    det_maxd = det_data.get("max_value_diff_gate0_vs_plain", 0.0)
    det_cls = "verdict-pass" if det_vm == 0 else "verdict-fail"
    det_txt = ("CONFIRMED: Gate=0 and plain VBA produce IDENTICAL values and actions."
               if det_vm == 0 else "Value differences detected! The agents are NOT identical.")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # ─── Assemble HTML ───
    # Use string concatenation for JS to avoid {{}} escaping hell
    css = """<style>
:root{--bg:#0f1117;--card:#1a1d28;--border:#2a2d3a;--text:#e2e4e9;--muted:#8b8fa3;
--accent:#6c5ce7;--green:#00b894;--red:#fd79a8;--yellow:#fdcb6e;--blue:#74b9ff}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
line-height:1.6;padding:2rem;max-width:1200px;margin:0 auto}
h1{font-size:1.8rem;font-weight:700;margin-bottom:.5rem;
background:linear-gradient(135deg,#6c5ce7,#74b9ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
h2{font-size:1.3rem;font-weight:600;margin:2rem 0 1rem;color:var(--accent);
border-bottom:1px solid var(--border);padding-bottom:.5rem}
.sub{color:var(--muted);font-size:.9rem;margin-bottom:2rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}
.v{padding:1rem 1.5rem;border-radius:8px;margin:1rem 0;font-weight:500}
.verdict-pass{background:rgba(0,184,148,.15);border-left:4px solid var(--green)}
.verdict-fail{background:rgba(253,121,168,.15);border-left:4px solid var(--red)}
.verdict-note{background:rgba(253,203,110,.15);border-left:4px solid var(--yellow)}
table{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.85rem}
th,td{padding:.6rem .8rem;text-align:right;border-bottom:1px solid var(--border)}
th{background:rgba(108,92,231,.1);font-weight:600;color:var(--accent)}
th:first-child,td:first-child{text-align:left}
td.num{font-family:'JetBrains Mono',monospace;font-size:.8rem}
.positive{color:var(--green)}.negative{color:var(--red)}
.cc{position:relative;margin:1rem 0;padding:1rem;background:rgba(0,0,0,.2);border-radius:8px}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin:1rem 0}
.sb{background:rgba(0,0,0,.2);border-radius:8px;padding:1rem;text-align:center}
.sv{font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace}
.sl{font-size:.8rem;color:var(--muted);margin-top:.3rem}
code{font-family:'JetBrains Mono',monospace;background:rgba(108,92,231,.15);
padding:.15rem .4rem;border-radius:4px;font-size:.8rem}
</style>"""

    det_maxd_str = f"{det_maxd:.2e}" if isinstance(det_maxd, float) else "N/A"
    vm_cls = "positive" if det_vm == 0 else "negative"
    am_cls = "positive" if det_am == 0 else "negative"

    html_body = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Ablation Investigation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
{css}</head><body>
<h1>Ablation Investigation Report</h1>
<p class="sub">ModulatedValueAgent: gate=0 vs plain ValueBasedAgent | {timestamp}</p>

<h2>1. Weight Identity Check</h2>
<div class="card">
<p>Base weights in <code>modulated_value_agent.pt</code> vs <code>value_based_agent.pt</code></p>
<div class="v {w_cls}">{w_txt}</div>
<table><tr><th>Parameter</th><th>Shape</th><th>Identical</th><th>Max Diff</th></tr>
{weight_rows}</table></div>

<h2>2. Deterministic Value &amp; Action Check</h2>
<div class="card">
<p>Given identical game states, do gate=0 and plain VBA produce identical V(s)?</p>
<div class="sg">
<div class="sb"><div class="sv">{det_n}</div><div class="sl">State-Actions Compared</div></div>
<div class="sb"><div class="sv {vm_cls}">{det_vm}</div><div class="sl">Value Mismatches</div></div>
<div class="sb"><div class="sv">{det_maxd_str}</div><div class="sl">Max Value Diff</div></div>
<div class="sb"><div class="sv {am_cls}">{det_am}</div><div class="sl">Action Mismatches</div></div>
</div>
<div class="v {det_cls}">{det_txt}</div></div>

<h2>3. Head-to-Head Ablation ({EVAL_ROUNDS} rounds/matchup)</h2>
<div class="card">
<table><tr><th>Variant</th><th>vs Heuristic</th><th>vs Value Based</th><th>vs Adaptive Value</th><th>Average</th></tr>
{ablation_rows}</table>
<div class="cc" style="height:400px"><canvas id="ablationChart"></canvas></div>
<div class="v verdict-note"><b>Interpretation:</b> If gate=0 and plain VBA are functionally identical
(Experiment 2), any performance difference is random variance. Check the 95% CIs.</div></div>

<h2>4. Gate Value Distribution During Gameplay</h2>
<div class="card">
<table><tr><th>Opponent</th><th>N</th><th>Mean</th><th>Std</th><th>Min</th><th>P25</th><th>Median</th><th>P75</th><th>Max</th></tr>
{gate_rows}</table>
<div class="cc" style="height:350px"><canvas id="gateHistChart"></canvas></div></div>
"""

    # JavaScript — use plain string concat to avoid f-string + JS brace conflicts
    js = ('<script>\n'
          'Chart.defaults.color="#8b8fa3";Chart.defaults.borderColor="#2a2d3a";\n'
          'new Chart(document.getElementById("ablationChart"),{type:"bar",data:{\n'
          '  labels:' + json.dumps(chart_labels) + ',\n'
          '  datasets:[\n'
          '    {label:"vs Heuristic",data:' + json.dumps(chart_heur)
          + ',backgroundColor:"rgba(0,184,148,0.7)",borderColor:"rgba(0,184,148,1)",borderWidth:1},\n'
          '    {label:"vs Value Based",data:' + json.dumps(chart_vb)
          + ',backgroundColor:"rgba(116,185,255,0.7)",borderColor:"rgba(116,185,255,1)",borderWidth:1},\n'
          '    {label:"vs Adaptive Value",data:' + json.dumps(chart_av)
          + ',backgroundColor:"rgba(108,92,231,0.7)",borderColor:"rgba(108,92,231,1)",borderWidth:1},\n'
          '    {label:"Average",data:' + json.dumps(chart_avg)
          + ',backgroundColor:"rgba(253,203,110,0.85)",borderColor:"rgba(253,203,110,1)",'
          'borderWidth:2,borderDash:[4,4],type:"line",order:0,pointRadius:6,pointHoverRadius:8},\n'
          ']},options:{responsive:true,maintainAspectRatio:false,\n'
          '  plugins:{title:{display:true,text:"Ablation: Avg Chips/Round",color:"#e2e4e9",'
          'font:{size:14,weight:600}},legend:{position:"bottom"}},\n'
          '  scales:{y:{title:{display:true,text:"Avg chips/round",color:"#8b8fa3"},'
          'grid:{color:"#2a2d3a"}},x:{grid:{display:false}}}}});\n'
          '\nvar gd=' + json.dumps(gate_hist_json) + ';\n'
          'var gc={heuristic:"#00b894",value_based:"#74b9ff",adaptive_value:"#6c5ce7"};\n'
          'var ds=[];\n'
          'for(var[o,d] of Object.entries(gd)){\n'
          '  var c=gc[o]||"#fd79a8";\n'
          '  ds.push({label:o+" (mean="+d.mean.toFixed(4)+")",\n'
          '    data:d.centers.map(function(x,i){return{x:x,y:d.counts[i]}}),\n'
          '    borderColor:c,backgroundColor:c+"40",fill:true,tension:0.3,pointRadius:0,borderWidth:2});\n'
          '}\n'
          'new Chart(document.getElementById("gateHistChart"),{type:"line",data:{datasets:ds},\n'
          '  options:{responsive:true,maintainAspectRatio:false,\n'
          '    plugins:{title:{display:true,text:"Gate Value Distribution",color:"#e2e4e9",'
          'font:{size:14,weight:600}},legend:{position:"bottom"}},\n'
          '    scales:{x:{type:"linear",title:{display:true,text:"Gate Value",color:"#8b8fa3"},'
          'grid:{color:"#2a2d3a"}},y:{title:{display:true,text:"Count",color:"#8b8fa3"},'
          'grid:{color:"#2a2d3a"}}}}});\n'
          '</script></body></html>')

    with open(HTML_PATH, "w") as f:
        f.write(html_body + js)
    print(f"\n  HTML report saved to: {HTML_PATH}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RIGOROUS ABLATION INVESTIGATION")
    print("ModulatedValueAgent gate=0 vs Plain ValueBasedAgent")
    print("=" * 70)
    t_start = time.time()

    all_results = {}

    all_results["weight_check"] = experiment_weight_check()
    all_results["deterministic_check"] = experiment_deterministic_check()
    all_results["ablation_h2h"] = experiment_ablation_h2h()
    all_results["gate_distribution"] = experiment_gate_distribution()

    elapsed = time.time() - t_start
    all_results["total_elapsed_s"] = round(elapsed, 1)

    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    weights_ok = all_results["weight_check"].get("all_weights_identical", False)
    values_ok = all_results["deterministic_check"].get("value_mismatches_gate0_vs_plain", 1) == 0

    if weights_ok and values_ok:
        print("\n  The base weights are IDENTICAL and the value outputs are IDENTICAL.")
        print("  Gate=0 modulated agent IS the same agent as plain ValueBasedAgent.")
        print("  Any performance difference in the original ablation study was")
        print("  purely due to RANDOM VARIANCE with only 1000 rounds per matchup.")
        print("\n  The original report's table was misleading — it presented random")
        print("  noise as if it were a systematic difference.")
    elif weights_ok and not values_ok:
        print("\n  Weights are identical but value outputs DIFFER!")
        print("  There may be a subtle code path difference (e.g., observation handling).")
    elif not weights_ok:
        print("\n  The base weights inside modulated_value_agent.pt DIFFER from")
        print("  value_based_agent.pt. This is unexpected — the base was supposed")
        print("  to be frozen during training. Weight drift is the root cause.")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    generate_html_report(all_results)
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
