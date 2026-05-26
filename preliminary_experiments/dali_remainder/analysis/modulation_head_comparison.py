"""
DALI_modulation — Modulation Head Comparison
=============================================
Compares the modulation head outputs of:
  full_modulation  (frozen base, head learns residual correction)
  finetuned_base   (unfrozen base, head + base train jointly)

For the same (state, opponent_stats) inputs, how differently do the
modulation heads Δ(s, stats) behave?

Questions answered:
  1. Magnitude: is |Δ| larger or smaller in finetuned_base vs full_modulation?
     (If the base absorbed the adjustment, the head should be smaller.)
  2. Cross-opponent spread: does Δ vary with opponent stats in either agent?
     (Low spread → head is not actually opponent-adaptive.)
  3. Correlation: are the two heads learning similar or different corrections?
  4. Base drift effect: how much has finetuned_base's V_base(s) shifted from
     the original value_based checkpoint, per state?

Method:
  - Sample 2000 unique game states from simulated play.
  - Evaluate 6 prototype opponent stat vectors (tight/loose × passive/aggressive,
    maniac, random) with confidence=0.9 (warm session).
  - Compute per-state Δ for all seeds × all stat vectors.
  - Print summary table + save results/modulation_comparison.json.

Usage:
  python -m DALI_modulation.analysis.modulation_head_comparison
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from engine.leduc_game import LeducGame, Action
from agents.value_based.agent import ValueNetwork
from paper.agents.full_modulation.agent import FullModulationAgent
from paper.agents.ablations.finetuned_base.agent import FinetunedBaseAgent

GAME_DIMS = 15
STAT_DIMS = 7

# ── prototype opponent stats (7-dim: pf_fold, pf_raise, fl_raise,
#    pf_fold2raise, fl_fold2raise, raise_after_raise, confidence=0.9) ─────────

PROTOTYPE_STATS = {
    "tight_passive":    [0.70, 0.10, 0.10, 0.70, 0.70, 0.05, 0.90],
    "tight_aggressive": [0.20, 0.65, 0.55, 0.30, 0.30, 0.55, 0.90],
    "loose_passive":    [0.15, 0.20, 0.15, 0.25, 0.20, 0.10, 0.90],
    "loose_aggressive": [0.10, 0.70, 0.65, 0.15, 0.15, 0.65, 0.90],
    "maniac":           [0.05, 0.90, 0.90, 0.10, 0.10, 0.80, 0.90],
    "random":           [0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.90],
}

# ── agent checkpoint paths ────────────────────────────────────────────────────

_DALI = os.path.join(ROOT, "preliminary_experiments", "dali_remainder")

FM_CKPTS = [
    os.path.join(ROOT, "paper", "agents", "full_modulation", "outputs", f"seed_{s}", "checkpoint_final.pt")
    for s in [0, 1, 2]
]
FB_CKPTS = [
    os.path.join(ROOT, "paper", "agents", "ablations", "finetuned_base", "outputs", f"seed_{s}", "checkpoint_final.pt")
    for s in [0, 1, 2]
]
ORIGINAL_BASE = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")

RESULTS_DIR = os.path.join(ROOT, "preliminary_experiments", "dali_remainder", "analysis", "results")


# ── encoding helper ───────────────────────────────────────────────────────────

_CARD_MAP  = {'J': 0, 'Q': 1, 'K': 2}
_MAX_CHIPS = 13


def _encode_game(obs, viewer_id: int) -> torch.Tensor:
    """15-dim encoding (same as FullModulationAgent._encode_game)."""
    hand_idx = _CARD_MAP.get(obs.player_hand)
    hand_vec = torch.zeros(3)
    if hand_idx is not None:
        hand_vec[hand_idx] = 1.0

    board_idx = _CARD_MAP.get(obs.board, 3)
    board_vec = torch.zeros(4)
    board_vec[board_idx] = 1.0

    p0, p1 = obs.pot
    pot_rel = [p0, p1] if viewer_id == 0 else [p1, p0]
    pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / _MAX_CHIPS

    feats = torch.tensor([
        1.0 if viewer_id == obs.current_player else 0.0,
        float(viewer_id),
        float(obs.current_round),
        1.0 if obs.is_finished else 0.0,
        1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
        obs.raises_this_round / 2.0,
    ])
    return torch.cat([hand_vec, board_vec, pot_vec, feats])   # (15,)


# ── state sampling ────────────────────────────────────────────────────────────

def sample_states(n_hands: int = 3000, seed: int = 42) -> list[torch.Tensor]:
    """
    Simulate n_hands hands and collect unique non-terminal game state encodings.
    Leduc Hold'em has ~500 unique states, so 3000 hands saturates coverage.
    Returns list of (15,) tensors.
    """
    from agents.heuristic.agent import HeuristicAgent
    from agents.cfr.agent import CFRAgent

    cfr_path = os.path.join(ROOT, "agents", "cfr", "checkpoint.pt")
    agents_pool = [HeuristicAgent(), CFRAgent(model_path=cfr_path)]
    for a in agents_pool:
        a.set_train_mode(False)

    seen   = set()
    states = []

    for i in range(n_hands):
        a0 = agents_pool[i % 2]
        a1 = agents_pool[(i + 1) % 2]

        game = LeducGame()
        game.reset()

        while not game.is_finished:
            cp  = game.current_player
            obs = game.get_observation(viewer_id=cp)
            enc = _encode_game(obs, cp)   # (15,)

            key = tuple(enc.tolist())
            if key not in seen:
                seen.add(key)
                states.append(enc)

            agent = a0 if cp == 0 else a1
            action = agent.select_action(obs)
            game.step(action)

    return states


# ── per-agent delta computation ───────────────────────────────────────────────

def compute_deltas_full_modulation(
    agent: FullModulationAgent,
    states: list[torch.Tensor],
    stats_vecs: list[np.ndarray],
) -> np.ndarray:
    """
    Returns array shape (n_states, n_stats) of Δ(s, stats) values.
    """
    agent.set_train_mode(False)
    deltas = np.zeros((len(states), len(stats_vecs)))

    for i, enc in enumerate(states):
        game_t = enc.unsqueeze(0)                      # (1, 15)
        for j, sv in enumerate(stats_vecs):
            stats_t  = torch.tensor(sv, dtype=torch.float32).unsqueeze(0)  # (1, 7)
            mod_inp  = torch.cat([game_t, stats_t], dim=1)                 # (1, 22)
            with torch.no_grad():
                delta = agent.mod(mod_inp).squeeze().item()
            deltas[i, j] = delta

    return deltas


def compute_deltas_finetuned_base(
    agent: FinetunedBaseAgent,
    states: list[torch.Tensor],
    stats_vecs: list[np.ndarray],
) -> np.ndarray:
    """
    Returns array shape (n_states, n_stats) of Δ(s, stats) values.
    """
    agent.set_train_mode(False)
    deltas = np.zeros((len(states), len(stats_vecs)))

    for i, enc in enumerate(states):
        game_t = enc.unsqueeze(0)
        for j, sv in enumerate(stats_vecs):
            stats_t  = torch.tensor(sv, dtype=torch.float32).unsqueeze(0)
            mod_inp  = torch.cat([game_t, stats_t], dim=1)
            with torch.no_grad():
                delta = agent.mod(mod_inp).squeeze().item()
            deltas[i, j] = delta

    return deltas


def compute_base_drift_per_state(
    agent: FinetunedBaseAgent,
    original_base: ValueNetwork,
    states: list[torch.Tensor],
) -> np.ndarray:
    """
    Returns array (n_states,) of |V_base_finetuned(s) - V_base_original(s)|.
    """
    agent.set_train_mode(False)
    drifts = np.zeros(len(states))

    for i, enc in enumerate(states):
        game_t = enc.unsqueeze(0)
        with torch.no_grad():
            v_new = agent.base(game_t).squeeze().item()
            v_old = original_base(game_t).squeeze().item()
        drifts[i] = abs(v_new - v_old)

    return drifts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats_names = list(PROTOTYPE_STATS.keys())
    stats_vecs  = [np.array(v, dtype=np.float32) for v in PROTOTYPE_STATS.values()]

    print("Sampling game states...")
    states = sample_states(n_hands=3000)
    print(f"  Collected {len(states)} unique states.\n")

    # Load original base for drift comparison
    original_base = ValueNetwork(GAME_DIMS, hidden_size=64)
    original_base.load_state_dict(
        torch.load(ORIGINAL_BASE, map_location="cpu", weights_only=True)
    )
    original_base.eval()

    # ── full_modulation seeds ─────────────────────────────────────────────────
    print("Computing full_modulation deltas...")
    fm_deltas = []   # list of (n_states, n_stats) arrays
    for seed, ckpt in enumerate(FM_CKPTS):
        if not os.path.isfile(ckpt):
            print(f"  [skip] seed {seed}: checkpoint not found")
            continue
        agent = FullModulationAgent()
        agent.load_model(ckpt)
        d = compute_deltas_full_modulation(agent, states, stats_vecs)
        fm_deltas.append(d)
        print(f"  seed {seed}: mean|Δ|={np.abs(d).mean():.4f}  "
              f"cross-opp spread={d.std(axis=1).mean():.4f}")

    # ── finetuned_base seeds ──────────────────────────────────────────────────
    print("\nComputing finetuned_base deltas + base drift...")
    fb_deltas = []
    fb_drifts = []
    for seed, ckpt in enumerate(FB_CKPTS):
        if not os.path.isfile(ckpt):
            print(f"  [skip] seed {seed}: checkpoint not found")
            continue
        agent = FinetunedBaseAgent()
        agent.load_model(ckpt)
        d = compute_deltas_finetuned_base(agent, states, stats_vecs)
        fb_deltas.append(d)
        drift = compute_base_drift_per_state(agent, original_base, states)
        fb_drifts.append(drift)
        print(f"  seed {seed}: mean|Δ|={np.abs(d).mean():.4f}  "
              f"cross-opp spread={d.std(axis=1).mean():.4f}  "
              f"mean_base_drift={drift.mean():.4f}")

    # ── aggregate statistics ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print("MODULATION HEAD COMPARISON")
    print("="*65)

    if fm_deltas:
        fm_stack = np.stack(fm_deltas)         # (n_seeds, n_states, n_stats)
        fm_mean_mag   = float(np.abs(fm_stack).mean())
        fm_opp_spread = float(fm_stack.std(axis=2).mean())   # spread across opponents
        fm_seed_std   = float(fm_stack.std(axis=0).mean())   # spread across seeds
        print(f"\nfull_modulation ({len(fm_deltas)} seeds):")
        print(f"  Mean |Δ|                 : {fm_mean_mag:.4f}")
        print(f"  Cross-opponent spread    : {fm_opp_spread:.4f}  "
              f"(avg std of Δ across 6 stat vectors per state)")
        print(f"  Cross-seed std           : {fm_seed_std:.4f}")
        print(f"  Per-stat-vector mean Δ:")
        for j, sn in enumerate(stats_names):
            mu = float(fm_stack[:, :, j].mean())
            print(f"    {sn:<22s} {mu:+.4f}")
    else:
        fm_mean_mag = fm_opp_spread = fm_seed_std = float("nan")

    if fb_deltas:
        fb_stack = np.stack(fb_deltas)
        fb_mean_mag   = float(np.abs(fb_stack).mean())
        fb_opp_spread = float(fb_stack.std(axis=2).mean())
        fb_seed_std   = float(fb_stack.std(axis=0).mean())
        drift_stack   = np.stack(fb_drifts)
        mean_drift    = float(drift_stack.mean())
        print(f"\nfinetuned_base ({len(fb_deltas)} seeds):")
        print(f"  Mean |Δ|                 : {fb_mean_mag:.4f}")
        print(f"  Cross-opponent spread    : {fb_opp_spread:.4f}")
        print(f"  Cross-seed std           : {fb_seed_std:.4f}")
        print(f"  Mean base drift per state: {mean_drift:.4f}")
        print(f"  Per-stat-vector mean Δ:")
        for j, sn in enumerate(stats_names):
            mu = float(fb_stack[:, :, j].mean())
            print(f"    {sn:<22s} {mu:+.4f}")
    else:
        fb_mean_mag = fb_opp_spread = fb_seed_std = mean_drift = float("nan")

    # ── correlation between the two agent types ───────────────────────────────
    if fm_deltas and fb_deltas:
        # Use seed-0 for direct comparison
        fm0 = fm_deltas[0].flatten()
        fb0 = fb_deltas[0].flatten()
        corr = float(np.corrcoef(fm0, fb0)[0, 1])
        mean_abs_diff = float(np.abs(fm0 - fb0).mean())
        print(f"\nSeed-0 comparison (full_modulation vs finetuned_base):")
        print(f"  Pearson correlation of Δ : {corr:+.4f}")
        print(f"  Mean |Δ_fm - Δ_fb|      : {mean_abs_diff:.4f}")
    else:
        corr = mean_abs_diff = float("nan")

    print("="*65)

    # ── save results ──────────────────────────────────────────────────────────
    result = {
        "n_states": len(states),
        "n_stat_vectors": len(stats_names),
        "stat_vector_names": stats_names,
        "prototype_stats": PROTOTYPE_STATS,
        "full_modulation": {
            "n_seeds": len(fm_deltas),
            "mean_abs_delta": fm_mean_mag,
            "cross_opp_spread": fm_opp_spread,
            "cross_seed_std": fm_seed_std,
            "per_stat_mean": {
                sn: float(np.stack(fm_deltas)[:, :, j].mean())
                for j, sn in enumerate(stats_names)
            } if fm_deltas else {},
        },
        "finetuned_base": {
            "n_seeds": len(fb_deltas),
            "mean_abs_delta": fb_mean_mag,
            "cross_opp_spread": fb_opp_spread,
            "cross_seed_std": fb_seed_std,
            "mean_base_drift": mean_drift,
            "per_stat_mean": {
                sn: float(np.stack(fb_deltas)[:, :, j].mean())
                for j, sn in enumerate(stats_names)
            } if fb_deltas else {},
        },
        "seed0_comparison": {
            "pearson_correlation": corr,
            "mean_abs_difference": mean_abs_diff,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "modulation_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
