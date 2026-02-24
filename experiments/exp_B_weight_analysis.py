"""
Experiment B: Feature Weight Analysis (H3)

HYPOTHESIS: The adaptive network has learned to IGNORE the stat features
(indices 15-18). Because stats start uninformative ([0.5, 0.5, 0.5, 0.0])
and only slowly become meaningful, the gradient signal for these features
is weak throughout training. The network ends up assigning near-zero
weights to the stat inputs — making the 19-feature agent behave like
a 15-feature agent, but with a harder optimization landscape.

FALSIFICATION CONDITION: If stat feature weights have similar magnitude to
game feature weights, H3 is false — the network IS using those features.

TEST:
  1. Extract first-layer weight matrix from both adaptive and vanilla models.
  2. Compare weight L2 norms for game features (cols 0-14) vs stat features (cols 15-18).
  3. Run a gradient sensitivity analysis: how much does zeroing each feature group change output?
  4. Compare L1-norm of stat weight columns vs game weight columns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.value_based import ValueBasedAgent

ADAPTIVE_MODEL = "models/adaptive_value_agent.pt"
VANILLA_MODEL = "models/value_based_agent.pt"

GAME_FEATURE_NAMES = [
    "hand_J", "hand_Q", "hand_K",
    "board_J", "board_Q", "board_K", "board_None",
    "my_pot", "opp_pot",
    "my_turn", "position", "round", "terminal", "has_pair", "raises"
]
STAT_FEATURE_NAMES = ["fold_rate", "raise_rate", "fold_to_raise_rate", "confidence"]


def analyze_weights(model, label, n_game_feats=15, n_stat_feats=4):
    """
    Analyze the first Linear layer's weight matrix.
    Weights are shape [hidden_size, input_size] = [64, 19] for adaptive, [64, 15] for vanilla.
    """
    first_layer = list(model.net.children())[0]  # nn.Linear(input_size, hidden_size)
    W = first_layer.weight.data  # [64, input_size]

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  Weight matrix shape: {W.shape}")

    # Per-feature column norms (how much each input feature is used)
    col_norms = W.norm(dim=0)  # [input_size]

    game_norms = col_norms[:n_game_feats]
    print(f"\n  Game features (cols 0-{n_game_feats-1}) L2 norms:")
    for i, (name, norm) in enumerate(zip(GAME_FEATURE_NAMES[:n_game_feats], game_norms)):
        bar = '█' * int(norm.item() * 10)
        print(f"    [{i:2d}] {name:<22}: {norm.item():.4f}  {bar}")

    print(f"\n  Game feature norm stats:")
    print(f"    Mean: {game_norms.mean().item():.4f}")
    print(f"    Std:  {game_norms.std().item():.4f}")
    print(f"    Min:  {game_norms.min().item():.4f}")
    print(f"    Max:  {game_norms.max().item():.4f}")

    if n_stat_feats > 0 and W.shape[1] >= n_game_feats + n_stat_feats:
        stat_norms = col_norms[n_game_feats:n_game_feats + n_stat_feats]
        print(f"\n  Stat features (cols {n_game_feats}-{n_game_feats+n_stat_feats-1}) L2 norms:")
        for i, (name, norm) in enumerate(zip(STAT_FEATURE_NAMES, stat_norms)):
            bar = '█' * int(norm.item() * 10)
            print(f"    [{n_game_feats+i:2d}] {name:<22}: {norm.item():.4f}  {bar}")

        print(f"\n  Stat feature norm stats:")
        print(f"    Mean: {stat_norms.mean().item():.4f}")
        print(f"    Std:  {stat_norms.std().item():.4f}")

        game_mean = game_norms.mean().item()
        stat_mean = stat_norms.mean().item()
        ratio = stat_mean / game_mean if game_mean > 0 else 0
        print(f"\n  Ratio (stat_mean / game_mean): {ratio:.4f}")
        print(f"  → {'Stat features are UNDERWEIGHTED' if ratio < 0.5 else 'Stat features have COMPARABLE weight'}")

    return col_norms


def gradient_sensitivity(model, label, n_game_feats=15, n_stat_feats=4):
    """
    Measure output sensitivity to each feature group via gradient of output w.r.t. input.
    Uses a typical input vector (mid-game state, default stats).
    """
    input_size = n_game_feats + n_stat_feats
    # Construct a representative "typical" input
    x = torch.zeros(1, input_size)
    # hand = Q (index 1)
    x[0, 1] = 1.0
    # board = None (index 6)
    x[0, 6] = 1.0
    # my_pot = 0.15, opp_pot = 0.15
    x[0, 7] = 0.15
    x[0, 8] = 0.15
    # my_turn = 1
    x[0, 9] = 1.0
    # round = 0
    x[0, 11] = 0.0
    # default stats (if present)
    if input_size > n_game_feats:
        x[0, n_game_feats] = 0.5    # fold_rate
        x[0, n_game_feats+1] = 0.5  # raise_rate
        x[0, n_game_feats+2] = 0.5  # fold_to_raise_rate
        x[0, n_game_feats+3] = 0.0  # confidence

    x.requires_grad_(True)
    output = model(x)
    output.backward()

    grads = x.grad.abs().squeeze(0)
    game_grad = grads[:n_game_feats]
    stat_grad = grads[n_game_feats:n_game_feats+n_stat_feats] if n_stat_feats > 0 else torch.zeros(0)

    print(f"\n  Gradient sensitivity ({label}, typical mid-game state):")
    print(f"    Game features avg |grad|:  {game_grad.mean().item():.6f}")
    if n_stat_feats > 0:
        print(f"    Stat features avg |grad|:  {stat_grad.mean().item():.6f}")
        ratio = stat_grad.mean().item() / (game_grad.mean().item() + 1e-10)
        print(f"    Ratio (stat/game):         {ratio:.4f}")
        print(f"    Per-stat gradients: {[f'{v:.6f}' for v in stat_grad.tolist()]}")


def output_delta_test(adaptive_model, n_game_feats=15, n_stat_feats=4):
    """
    Inject extreme stat values and measure output change.
    If stats are used, output should change significantly with extreme inputs.
    """
    input_size = n_game_feats + n_stat_feats
    base = torch.zeros(1, input_size)
    base[0, 1] = 1.0  # hand Q
    base[0, 6] = 1.0  # board None
    base[0, 9] = 1.0  # my_turn
    base[0, n_game_feats] = 0.5    # fold_rate = 0.5 (default)
    base[0, n_game_feats+1] = 0.5  # raise_rate = 0.5 (default)
    base[0, n_game_feats+2] = 0.5  # fold_to_raise_rate = 0.5 (default)
    base[0, n_game_feats+3] = 0.0  # confidence = 0 (default)

    with torch.no_grad():
        val_default = adaptive_model(base).item()

        # Extreme: opponent always folds, very aggressive raiser, high confidence
        extreme1 = base.clone()
        extreme1[0, n_game_feats] = 0.9    # fold_rate very high
        extreme1[0, n_game_feats+1] = 0.05 # raise_rate very low
        extreme1[0, n_game_feats+2] = 1.0  # always folds to raise
        extreme1[0, n_game_feats+3] = 1.0  # high confidence
        val_passive = adaptive_model(extreme1).item()

        # Extreme: opponent never folds, aggressive
        extreme2 = base.clone()
        extreme2[0, n_game_feats] = 0.05   # fold_rate very low
        extreme2[0, n_game_feats+1] = 0.9  # raise_rate very high
        extreme2[0, n_game_feats+2] = 0.0  # never folds to raise
        extreme2[0, n_game_feats+3] = 1.0  # high confidence
        val_aggressive = adaptive_model(extreme2).item()

    print(f"\n  Output delta test (same game state, different opponent profiles):")
    print(f"    Default stats [0.5,0.5,0.5,0.0]:      V = {val_default:+.4f}")
    print(f"    Passive opp   [0.9,0.05,1.0,1.0]:     V = {val_passive:+.4f}  (Δ={val_passive-val_default:+.4f})")
    print(f"    Aggressive opp[0.05,0.9,0.0,1.0]:     V = {val_aggressive:+.4f}  (Δ={val_aggressive-val_default:+.4f})")
    delta_range = abs(val_passive - val_aggressive)
    print(f"    Range across profiles: {delta_range:.4f}")
    if delta_range < 0.01:
        print("    → SMALL range: Network nearly IGNORES stat features! H3 SUPPORTED.")
    elif delta_range < 0.05:
        print("    → MODERATE range: Weak stat sensitivity. H3 PARTIALLY supported.")
    else:
        print("    → LARGE range: Network IS sensitive to stats. H3 NOT supported.")


def main():
    print("=" * 65)
    print("EXPERIMENT B: Feature Weight Analysis (H3)")
    print("=" * 65)

    adaptive = AdaptiveValueAgent(model_path=ADAPTIVE_MODEL)
    vanilla = ValueBasedAgent(model_path=VANILLA_MODEL)

    # ── Part 1: Weight magnitude analysis ──
    print("\n[1/3] FIRST-LAYER WEIGHT MAGNITUDES")
    vanilla_norms = analyze_weights(vanilla.model, "Vanilla Agent (15 features)", n_game_feats=15, n_stat_feats=0)
    adaptive_norms = analyze_weights(adaptive.model, "Adaptive Agent (19 features)", n_game_feats=15, n_stat_feats=4)

    # ── Part 2: Gradient sensitivity ──
    print("\n[2/3] GRADIENT SENSITIVITY ANALYSIS")
    gradient_sensitivity(vanilla.model, "Vanilla", n_game_feats=15, n_stat_feats=0)
    gradient_sensitivity(adaptive.model, "Adaptive", n_game_feats=15, n_stat_feats=4)

    # ── Part 3: Output delta test ──
    print("\n[3/3] OUTPUT DELTA TEST (Stat feature sensitivity)")
    output_delta_test(adaptive.model, n_game_feats=15, n_stat_feats=4)

    # ── Final summary ──
    game_norms_adaptive = adaptive_norms[:15]
    stat_norms_adaptive = adaptive_norms[15:19]

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY (H3: Network Ignores Stat Features?)")
    print("=" * 65)
    ratio = stat_norms_adaptive.mean().item() / (game_norms_adaptive.mean().item() + 1e-10)
    print(f"  Stat weight norm mean:  {stat_norms_adaptive.mean().item():.4f}")
    print(f"  Game weight norm mean:  {game_norms_adaptive.mean().item():.4f}")
    print(f"  Ratio:                  {ratio:.4f}")
    print(f"  Vanilla game weights:   {vanilla_norms.mean().item():.4f}")

    if ratio < 0.3:
        print("\n  CONCLUSION: H3 STRONGLY SUPPORTED — stat features are underweighted.")
        print("  The network has NOT learned to leverage opponent statistics.")
    elif ratio < 0.7:
        print("\n  CONCLUSION: H3 PARTIALLY SUPPORTED — stats used weakly.")
    else:
        print("\n  CONCLUSION: H3 NOT SUPPORTED — stat features have comparable weights.")


if __name__ == "__main__":
    main()
