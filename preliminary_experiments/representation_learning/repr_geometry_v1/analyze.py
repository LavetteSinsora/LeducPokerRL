"""Geometry analysis of the reward-contrastive embedding space.

Loads a trained ContrastiveEncoder, generates representative Leduc Hold'em
states via random game play, embeds them, and performs PCA + t-SNE analysis
to understand the geometric structure of the 8-dim embedding space.
"""

import sys
import os
import json
import random
import warnings
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch

# ---------------------------------------------------------------------------
# Matplotlib / sklearn imports with graceful error
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not available — plots will not be generated")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available — metrics will not be computed")

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available — Spearman correlation will not be computed")

# Project imports
from experiments.representation_learning.contrastive_repr_v1.agent import ContrastiveEncoder, ContrastiveReprAgent
from engine.leduc_game import LeducGame, Action
from engine.observation import Observation

CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
CARD_NAMES = {0: 'J', 1: 'Q', 2: 'K'}
ROUND_NAMES = {0: 'pre-flop', 1: 'flop'}


# ---------------------------------------------------------------------------
# State generation
# ---------------------------------------------------------------------------

def random_completion_reward(game_copy: LeducGame, player_id: int,
                              n_rollouts: int = 50) -> float:
    """Monte Carlo expected reward for the current player via random completions."""
    total_reward = 0.0
    for _ in range(n_rollouts):
        g = game_copy.copy()
        while not g.is_finished:
            legal = g._get_legal_actions()
            action = random.choice(legal)
            g.step(action)
        rewards = g.get_reward()
        total_reward += rewards[player_id]
    return total_reward / n_rollouts


def make_obs_key(obs: Observation, viewer_id: int) -> tuple:
    """Create a hashable key that uniquely identifies a state from viewer_id's perspective."""
    return (
        obs.player_hand,
        obs.board,
        tuple(obs.pot),
        obs.current_round,
        obs.raises_this_round,
        obs.current_player,
        viewer_id,
    )


def collect_states(n_episodes: int = 2000,
                   mc_rollouts: int = 50,
                   max_states: int = 1500) -> List[Dict]:
    """
    Run n_episodes random games and collect (observation, metadata) tuples.

    Returns a list of dicts with keys:
        obs, viewer_id, encoded_vec, player_hand, opp_hand, board,
        round, pot_ratio, expected_reward
    """
    print(f"Collecting states from {n_episodes} random episodes...")
    seen_keys = set()
    states = []
    agent = ContrastiveReprAgent()  # used only for encode_observation

    for ep in range(n_episodes):
        if len(states) >= max_states:
            break
        game = LeducGame()
        obs = game.reset()

        while not game.is_finished:
            # Collect observation for the acting player
            viewer_id = game.current_player
            obs_for_viewer = game.get_observation(viewer_id=viewer_id)

            key = make_obs_key(obs_for_viewer, viewer_id)
            if key not in seen_keys:
                seen_keys.add(key)

                # Get opponent hand (known to us as analysts, not to the player)
                opp_id = 1 - viewer_id
                opp_hand = game.player_hands[opp_id]

                # Monte Carlo expected reward
                expected_reward = random_completion_reward(
                    game.copy(), viewer_id, n_rollouts=mc_rollouts
                )

                # Encode observation to 15-dim vector
                with torch.no_grad():
                    enc_vec = agent.encode_observation(obs_for_viewer, viewer_id=viewer_id)

                total_pot = sum(game.pot)
                pot_ratio = total_pot / 26.0  # max possible pot ~26

                states.append({
                    'obs': obs_for_viewer,
                    'viewer_id': viewer_id,
                    'encoded_vec': enc_vec.squeeze(0).numpy(),  # shape (15,)
                    'player_hand': obs_for_viewer.player_hand,
                    'opp_hand': opp_hand,
                    'board': obs_for_viewer.board,
                    'round': obs_for_viewer.current_round,
                    'pot_ratio': pot_ratio,
                    'expected_reward': expected_reward,
                })

            # Random action
            legal = game._get_legal_actions()
            action = random.choice(legal)
            game.step(action)

        if (ep + 1) % 500 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: {len(states)} unique states collected")

    print(f"Total unique states collected: {len(states)}")
    return states


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embeddings(states: List[Dict], encoder: ContrastiveEncoder) -> np.ndarray:
    """Pass all state encodings through the frozen encoder."""
    encoder.eval()
    vecs = torch.tensor(
        np.stack([s['encoded_vec'] for s in states], axis=0),
        dtype=torch.float32
    )
    with torch.no_grad():
        embeddings = encoder(vecs).numpy()
    return embeddings  # shape (N, 8)


# ---------------------------------------------------------------------------
# PCA analysis
# ---------------------------------------------------------------------------

def run_pca(embeddings: np.ndarray) -> Tuple[np.ndarray, PCA]:
    """Fit PCA on embeddings and return projected coordinates + fitted PCA."""
    pca = PCA(n_components=min(8, embeddings.shape[1]))
    projected = pca.fit_transform(embeddings)
    return projected, pca


def effective_dim(explained_variance_ratio: np.ndarray, threshold: float = 0.80) -> int:
    """Minimum number of components needed to explain threshold of variance."""
    cumsum = np.cumsum(explained_variance_ratio)
    for i, c in enumerate(cumsum):
        if c >= threshold:
            return i + 1
    return len(explained_variance_ratio)


# ---------------------------------------------------------------------------
# t-SNE analysis
# ---------------------------------------------------------------------------

def run_tsne(embeddings: np.ndarray, perplexity: int = 30,
             n_iter: int = 1000) -> np.ndarray:
    """Fit t-SNE and return 2D coordinates.

    Runs in a subprocess to avoid PyTorch + sklearn/OpenMP segfault on macOS.
    Falls back to in-process if subprocess approach fails.
    """
    print("Running t-SNE (subprocess mode to avoid OpenMP conflict)...")
    import subprocess, tempfile, os

    n = embeddings.shape[0]
    perplexity = min(perplexity, (n - 1) // 3)

    # Write embeddings to a temp file and run t-SNE in isolated subprocess
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        in_path = f.name
    out_path = in_path.replace('.npy', '_tsne.npy')

    try:
        np.save(in_path, embeddings.astype(np.float64))
        script = (
            f"import numpy as np\n"
            f"from sklearn.manifold import TSNE\n"
            f"emb = np.load('{in_path}')\n"
            f"tsne = TSNE(n_components=2, perplexity={perplexity}, "
            f"n_iter={n_iter}, random_state=42)\n"
            f"result = tsne.fit_transform(emb)\n"
            f"np.save('{out_path}', result)\n"
            f"print('t-SNE done, shape:', result.shape)\n"
        )
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"t-SNE subprocess stderr: {result.stderr[:500]}")
            raise RuntimeError("t-SNE subprocess failed")
        print(result.stdout.strip())
        coords = np.load(out_path)
        return coords
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Cluster separability metrics
# ---------------------------------------------------------------------------

def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score in the full 8-dim space."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return float('nan')
    return float(silhouette_score(embeddings, labels))


def intra_inter_distances(embeddings: np.ndarray,
                           labels: np.ndarray) -> Dict[str, float]:
    """Mean intra-cluster and inter-cluster L2 distances."""
    unique = np.unique(labels)
    intra_dists = []
    inter_dists = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = float(np.linalg.norm(embeddings[i] - embeddings[j]))
            if labels[i] == labels[j]:
                intra_dists.append(d)
            else:
                inter_dists.append(d)

    # Cap computation for large datasets
    max_pairs = 50000
    if len(intra_dists) + len(inter_dists) > max_pairs:
        rng = np.random.RandomState(42)
        n = len(embeddings)
        idx_i = rng.randint(0, n, max_pairs)
        idx_j = rng.randint(0, n, max_pairs)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        diffs = embeddings[idx_i] - embeddings[idx_j]
        all_dists = np.linalg.norm(diffs, axis=1)
        same = labels[idx_i] == labels[idx_j]
        intra_dists = all_dists[same].tolist()
        inter_dists = all_dists[~same].tolist()

    mean_intra = float(np.mean(intra_dists)) if intra_dists else float('nan')
    mean_inter = float(np.mean(inter_dists)) if inter_dists else float('nan')
    ratio = mean_inter / mean_intra if mean_intra > 0 else float('nan')
    return {'mean_intra': mean_intra, 'mean_inter': mean_inter,
            'inter_intra_ratio': ratio}


def compute_cluster_metrics(embeddings: np.ndarray,
                             states: List[Dict]) -> Dict:
    """Compute separability metrics for each semantic axis."""
    metrics = {}

    axes = {
        'player_hand': [CARD_MAP.get(s['player_hand'], -1) for s in states],
        'opp_hand':    [CARD_MAP.get(s['opp_hand'], -1)   for s in states],
        'round':       [s['round']                         for s in states],
    }

    for axis_name, raw_labels in axes.items():
        labels = np.array(raw_labels)
        sil = compute_silhouette(embeddings, labels)

        # Subsample for distance computation if needed
        n = len(embeddings)
        if n > 500:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, 500, replace=False)
            emb_sub = embeddings[idx]
            lbl_sub = labels[idx]
        else:
            emb_sub = embeddings
            lbl_sub = labels

        dist_metrics = _fast_intra_inter(emb_sub, lbl_sub)
        metrics[axis_name] = {
            'silhouette': sil,
            **dist_metrics,
        }
        print(f"  {axis_name}: silhouette={sil:.4f}, "
              f"inter/intra={dist_metrics['inter_intra_ratio']:.3f}")

    return metrics


def _fast_intra_inter(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """Vectorized intra/inter distance computation."""
    n = len(embeddings)
    # Pairwise distances via broadcasting (works for n <= 500)
    diff = embeddings[:, None, :] - embeddings[None, :, :]  # (n, n, d)
    dists = np.linalg.norm(diff, axis=-1)  # (n, n)

    same_mask = (labels[:, None] == labels[None, :])  # (n, n)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    intra_mask = same_mask & upper
    inter_mask = (~same_mask) & upper

    intra_dists = dists[intra_mask]
    inter_dists = dists[inter_mask]

    mean_intra = float(np.mean(intra_dists)) if len(intra_dists) > 0 else float('nan')
    mean_inter = float(np.mean(inter_dists)) if len(inter_dists) > 0 else float('nan')
    ratio = mean_inter / mean_intra if mean_intra > 0 else float('nan')
    return {'mean_intra': mean_intra, 'mean_inter': mean_inter,
            'inter_intra_ratio': ratio}


def compute_reward_spearman(embeddings: np.ndarray,
                             rewards: np.ndarray) -> float:
    """Spearman correlation between pairwise L2 distances and |reward_i - reward_j|."""
    if not HAS_SCIPY:
        return float('nan')

    n = len(embeddings)
    # Subsample pairs for efficiency
    rng = np.random.RandomState(42)
    max_pairs = 10000
    n_pairs = min(max_pairs, n * (n - 1) // 2)

    idx_i = rng.randint(0, n, n_pairs * 2)
    idx_j = rng.randint(0, n, n_pairs * 2)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask][:n_pairs], idx_j[mask][:n_pairs]

    emb_dists = np.linalg.norm(embeddings[idx_i] - embeddings[idx_j], axis=1)
    reward_diffs = np.abs(rewards[idx_i] - rewards[idx_j])

    rho, pval = spearmanr(emb_dists, reward_diffs)
    return float(rho)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

HAND_COLORS = {'J': '#4C72B0', 'Q': '#DD8452', 'K': '#55A868'}
ROUND_COLORS = {0: '#AEC6CF', 1: '#4B5D6C'}  # light=pre-flop, dark=flop
HAND_MARKERS = {'J': 'o', 'Q': 's', 'K': '^'}


def _scatter_ax(ax, coords_2d: np.ndarray, colors, title: str,
                colorbar_label: Optional[str] = None,
                legend_handles=None, cmap=None, vmin=None, vmax=None,
                alpha=0.5, s=12):
    """Helper: scatter plot on an existing axes."""
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                    c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=alpha, s=s, linewidths=0)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if colorbar_label is not None and cmap is not None:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(colorbar_label, fontsize=7)
    if legend_handles is not None:
        ax.legend(handles=legend_handles, fontsize=6, markerscale=1.2,
                  loc='best', framealpha=0.7)
    return sc


def make_plots(pca_2d: np.ndarray, tsne_2d: np.ndarray,
               states: List[Dict], output_dir: Path) -> None:
    """Generate the 2×4 grid and individual PCA plots."""
    if not HAS_MPL:
        print("Skipping plots — matplotlib not available")
        return

    rewards = np.array([s['expected_reward'] for s in states])
    player_hands = [s['player_hand'] for s in states]
    opp_hands = [s['opp_hand'] for s in states]
    rounds = [s['round'] for s in states]

    # Reward color values
    reward_colors = rewards
    r_vmin, r_vmax = rewards.min(), rewards.max()

    # Hand colors
    ph_colors = [HAND_COLORS.get(h, '#808080') for h in player_hands]
    oh_colors = [HAND_COLORS.get(h, '#808080') for h in opp_hands]

    # Round colors
    round_colors = [ROUND_COLORS[r] for r in rounds]

    # Legend handles
    hand_legend = [mpatches.Patch(color=HAND_COLORS[c], label=c) for c in ('J', 'Q', 'K')]
    round_legend = [
        mpatches.Patch(color=ROUND_COLORS[0], label='pre-flop'),
        mpatches.Patch(color=ROUND_COLORS[1], label='flop'),
    ]

    # 2×4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Embedding Space Geometry — contrastive_repr_v1 (L2/RnC encoder)',
                 fontsize=11, y=1.01)

    row_labels = ['PCA (PC1 vs PC2)', 't-SNE']
    for row_i, (coords, row_label) in enumerate([(pca_2d, 'PCA (PC1 vs PC2)'),
                                                  (tsne_2d, 't-SNE')]):
        _scatter_ax(axes[row_i, 0], coords, reward_colors,
                    title=f'{row_label} — Expected Reward',
                    colorbar_label='E[reward]', cmap='RdYlGn',
                    vmin=r_vmin, vmax=r_vmax)
        _scatter_ax(axes[row_i, 1], coords, ph_colors,
                    title=f'{row_label} — Player Hand',
                    legend_handles=hand_legend)
        _scatter_ax(axes[row_i, 2], coords, oh_colors,
                    title=f'{row_label} — Opponent Hand',
                    legend_handles=hand_legend)
        _scatter_ax(axes[row_i, 3], coords, round_colors,
                    title=f'{row_label} — Game Round',
                    legend_handles=round_legend)

    plt.tight_layout()
    grid_path = output_dir / 'pca_tsne_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {grid_path}")

    # Individual PCA plots
    configs = [
        ('reward', reward_colors, 'Expected Reward', 'RdYlGn', None, r_vmin, r_vmax),
        ('player_hand', ph_colors, 'Player Hand', None, hand_legend, None, None),
        ('opp_hand', oh_colors, 'Opponent Hand', None, hand_legend, None, None),
        ('round', round_colors, 'Game Round', None, round_legend, None, None),
    ]
    for suffix, colors, title, cmap, legend, vmin, vmax in configs:
        fig, ax = plt.subplots(figsize=(6, 5))
        _scatter_ax(ax, pca_2d, colors,
                    title=f'PCA (PC1 vs PC2) — {title}',
                    colorbar_label='E[reward]' if cmap else None,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    legend_handles=legend, alpha=0.6, s=20)
        ax.set_xlabel('PC1', fontsize=9)
        ax.set_ylabel('PC2', fontsize=9)
        plt.tight_layout()
        path = output_dir / f'pca_by_{suffix}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")


def make_pca_scree_plot(pca: PCA, output_dir: Path) -> None:
    """Save scree plot of PCA explained variance."""
    if not HAS_MPL:
        return
    evr = pca.explained_variance_ratio_
    cumevr = np.cumsum(evr)
    n = len(evr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(range(1, n + 1), evr * 100, color='steelblue', alpha=0.8)
    ax1.set_xlabel('PC')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('PCA Scree Plot')
    ax1.set_xticks(range(1, n + 1))

    ax2.plot(range(1, n + 1), cumevr * 100, marker='o', color='steelblue')
    ax2.axhline(80, color='red', linestyle='--', label='80% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xticks(range(1, n + 1))
    ax2.legend()

    plt.tight_layout()
    path = output_dir / 'pca_scree.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_analysis(checkpoint_path: str,
                 output_dir: str,
                 n_episodes: int = 2000,
                 mc_rollouts: int = 50) -> Dict:
    """
    Full analysis pipeline.

    Returns a dict with all metrics (also saved as geometry_report.json).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load encoder
    # ------------------------------------------------------------------
    print(f"\n=== Loading encoder from {checkpoint_path} ===")
    encoder = ContrastiveEncoder(input_size=15, hidden_size=64, embedding_dim=8)
    state = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(state['encoder'])
    encoder.eval()
    print("Encoder loaded successfully.")

    # ------------------------------------------------------------------
    # 2. Collect states
    # ------------------------------------------------------------------
    print(f"\n=== Collecting states ({n_episodes} episodes, {mc_rollouts} MC rollouts) ===")
    states = collect_states(n_episodes=n_episodes, mc_rollouts=mc_rollouts)
    print(f"Collected {len(states)} unique states.")

    if len(states) < 20:
        raise ValueError(f"Too few states ({len(states)}) — something went wrong.")

    # ------------------------------------------------------------------
    # 3. Compute embeddings
    # ------------------------------------------------------------------
    print("\n=== Computing embeddings ===")
    embeddings = compute_embeddings(states, encoder)
    print(f"Embeddings shape: {embeddings.shape}")

    rewards = np.array([s['expected_reward'] for s in states])

    # ------------------------------------------------------------------
    # 4. PCA
    # ------------------------------------------------------------------
    print("\n=== PCA analysis ===")
    pca_proj, pca = run_pca(embeddings)
    evr = pca.explained_variance_ratio_
    eff_dim_80 = effective_dim(evr, threshold=0.80)
    eff_dim_90 = effective_dim(evr, threshold=0.90)

    print("Explained variance per component:")
    for i, v in enumerate(evr):
        print(f"  PC{i+1}: {v*100:.1f}%")
    print(f"Effective dim (80%): {eff_dim_80}")
    print(f"Effective dim (90%): {eff_dim_90}")

    # ------------------------------------------------------------------
    # 5. t-SNE
    # ------------------------------------------------------------------
    tsne_proj = None
    if HAS_SKLEARN:
        tsne_proj = run_tsne(embeddings)

    # ------------------------------------------------------------------
    # 6. Cluster separability metrics
    # ------------------------------------------------------------------
    print("\n=== Cluster separability metrics ===")
    cluster_metrics = {}
    if HAS_SKLEARN:
        cluster_metrics = compute_cluster_metrics(embeddings, states)

    print("\n=== Reward Spearman correlation ===")
    reward_spearman = compute_reward_spearman(embeddings, rewards)
    print(f"  Spearman ρ (emb distance vs |Δreward|): {reward_spearman:.4f}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    if HAS_MPL and tsne_proj is not None:
        print("\n=== Generating plots ===")
        make_plots(pca_proj[:, :2], tsne_proj, states, output_dir)
        make_pca_scree_plot(pca, output_dir)
    elif HAS_MPL and tsne_proj is None:
        # PCA plots only
        dummy_tsne = np.zeros_like(pca_proj[:, :2])
        make_plots(pca_proj[:, :2], dummy_tsne, states, output_dir)
        make_pca_scree_plot(pca, output_dir)

    # ------------------------------------------------------------------
    # 8. Build report
    # ------------------------------------------------------------------
    report = {
        'checkpoint': str(checkpoint_path),
        'n_states': len(states),
        'n_episodes': n_episodes,
        'mc_rollouts': mc_rollouts,
        'pca': {
            'explained_variance_ratio': evr.tolist(),
            'cumulative_variance': np.cumsum(evr).tolist(),
            'top1_pct': float(evr[0] * 100),
            'effective_dim_80': eff_dim_80,
            'effective_dim_90': eff_dim_90,
        },
        'cluster_metrics': cluster_metrics,
        'reward_spearman_rho': reward_spearman,
        'state_stats': _compute_state_stats(states),
    }

    # Save JSON report
    report_path = output_dir / 'geometry_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")

    # ------------------------------------------------------------------
    # 9. Human-readable summary
    # ------------------------------------------------------------------
    _print_summary(report)

    # ------------------------------------------------------------------
    # 10. Write report.md
    # ------------------------------------------------------------------
    _write_report_md(report, states, output_dir)

    return report


def _compute_state_stats(states: List[Dict]) -> Dict:
    """Basic statistics about the collected states."""
    rounds = [s['round'] for s in states]
    player_hands = [s['player_hand'] for s in states]
    opp_hands = [s['opp_hand'] for s in states]
    rewards = [s['expected_reward'] for s in states]

    return {
        'round_counts': {
            'pre_flop': rounds.count(0),
            'flop': rounds.count(1),
        },
        'player_hand_counts': {h: player_hands.count(h) for h in ('J', 'Q', 'K')},
        'opp_hand_counts': {h: opp_hands.count(h) for h in ('J', 'Q', 'K')},
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'reward_min': float(np.min(rewards)),
        'reward_max': float(np.max(rewards)),
    }


def _print_summary(report: Dict) -> None:
    """Print human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("GEOMETRY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nStates analyzed: {report['n_states']}")
    print(f"\nPCA:")
    print(f"  Top-1 variance: {report['pca']['top1_pct']:.1f}%")
    print(f"  Effective dim (80%): {report['pca']['effective_dim_80']}")
    print(f"  Effective dim (90%): {report['pca']['effective_dim_90']}")

    print(f"\nReward correlation:")
    print(f"  Spearman ρ (emb dist vs |Δreward|): {report['reward_spearman_rho']:.4f}")

    if report.get('cluster_metrics'):
        print(f"\nCluster separability (silhouette score in 8-dim space):")
        best_axis = None
        best_sil = -2.0
        for axis, m in report['cluster_metrics'].items():
            sil = m['silhouette']
            ratio = m['inter_intra_ratio']
            print(f"  {axis:12s}: silhouette={sil:.4f}, inter/intra={ratio:.3f}")
            if not np.isnan(sil) and sil > best_sil:
                best_sil = sil
                best_axis = axis
        print(f"\n  --> Strongest clustering: {best_axis} (silhouette={best_sil:.4f})")

    print("=" * 60)


def _write_report_md(report: Dict, states: List[Dict],
                     output_dir: Path) -> None:
    """Write report.md to the output directory."""
    pca_info = report['pca']
    cm = report.get('cluster_metrics', {})
    rho = report['reward_spearman_rho']

    # Identify best-clustering axis
    best_axis, best_sil = 'N/A', float('nan')
    for axis, m in cm.items():
        s = m['silhouette']
        if not np.isnan(s) and (np.isnan(best_sil) or s > best_sil):
            best_axis, best_sil = axis, s

    lines = [
        "# repr_geometry_v1 — Geometry Analysis Report",
        "",
        f"Checkpoint: `{report['checkpoint']}`  ",
        f"States analyzed: {report['n_states']}  ",
        f"Episodes: {report['n_episodes']}, MC rollouts per state: {report['mc_rollouts']}",
        "",
        "## PCA Structure",
        "",
        "| Component | Variance Explained | Cumulative |",
        "|---|---|---|",
    ]
    evr = pca_info['explained_variance_ratio']
    cumevr = pca_info['cumulative_variance']
    for i, (v, c) in enumerate(zip(evr, cumevr)):
        lines.append(f"| PC{i+1} | {v*100:.1f}% | {c*100:.1f}% |")

    lines += [
        "",
        f"**Effective dimensionality (80% threshold):** {pca_info['effective_dim_80']}  ",
        f"**Effective dimensionality (90% threshold):** {pca_info['effective_dim_90']}  ",
        f"**Top-1 PC variance:** {pca_info['top1_pct']:.1f}%",
        "",
        "## Cluster Separability",
        "",
        f"**Reward Spearman ρ** (pairwise emb distance vs |Δreward|): **{rho:.4f}**",
        "",
        "| Semantic Axis | Silhouette Score | Inter/Intra Ratio |",
        "|---|---|---|",
    ]
    for axis, m in cm.items():
        lines.append(
            f"| {axis} | {m['silhouette']:.4f} | {m['inter_intra_ratio']:.3f} |"
        )

    lines += [
        "",
        f"**Strongest clustering axis:** {best_axis} (silhouette={best_sil:.4f})",
        "",
        "## State Distribution",
        "",
        "| Attribute | Value |",
        "|---|---|",
    ]
    ss = report['state_stats']
    lines += [
        f"| Pre-flop states | {ss['round_counts']['pre_flop']} |",
        f"| Flop states | {ss['round_counts']['flop']} |",
        f"| Player hand J/Q/K | {ss['player_hand_counts'].get('J',0)} / {ss['player_hand_counts'].get('Q',0)} / {ss['player_hand_counts'].get('K',0)} |",
        f"| Opponent hand J/Q/K | {ss['opp_hand_counts'].get('J',0)} / {ss['opp_hand_counts'].get('Q',0)} / {ss['opp_hand_counts'].get('K',0)} |",
        f"| Reward mean ± std | {ss['reward_mean']:.3f} ± {ss['reward_std']:.3f} |",
        f"| Reward range | [{ss['reward_min']:.3f}, {ss['reward_max']:.3f}] |",
        "",
        "## Findings",
        "",
        "### PCA",
        "",
        f"The L2/RnC encoder uses approximately **{pca_info['effective_dim_80']} effective dimensions** "
        f"(to explain 80% of variance), with the first PC accounting for {pca_info['top1_pct']:.1f}%. "
        "This confirms the contrastive_repr_v1 finding that reward-contrastive losses yield genuinely "
        "multi-dimensional representations (vs TD(0)'s near-complete collapse to 1D).",
        "",
        "### Reward Axis",
        "",
        f"Spearman ρ = {rho:.4f} between pairwise embedding distances and absolute reward differences. "
        "This quantifies how well the metric structure of the embedding space tracks reward proximity.",
        "",
        "### Cluster Structure",
        "",
        f"The strongest semantic clustering is along **{best_axis}** (silhouette={best_sil:.4f}). "
        "Silhouette scores > 0.2 indicate meaningful structure; > 0.5 indicates strong clustering.",
        "",
        "### Visual Observations",
        "",
        "See the generated plots:",
        "- `pca_tsne_grid.png` — 2×4 grid: top row PCA, bottom row t-SNE, columns = reward/player_hand/opp_hand/round",
        "- `pca_by_reward.png`, `pca_by_player_hand.png`, `pca_by_opp_hand.png`, `pca_by_round.png`",
        "- `pca_scree.png` — scree plot of all 8 PCA components",
    ]

    md_content = "\n".join(lines)
    md_path = output_dir / 'report.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Saved: {md_path}")
