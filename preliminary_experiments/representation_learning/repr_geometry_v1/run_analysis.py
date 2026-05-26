"""CLI wrapper for repr_geometry_v1 analysis.

Usage:
    python -m experiments.representation_learning.repr_geometry_v1.run_analysis \\
        --checkpoint-path outputs/contrastive_repr_v1/encoder_l2.pt \\
        --output-dir outputs/repr_geometry_v1/
"""

import argparse
import sys
from pathlib import Path

from .analyze import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze geometry of the reward-contrastive embedding space."
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=str(Path(__file__).parent.parent / 'contrastive_repr_v1' / 'outputs' / 'encoder_l2.pt'),
        help='Path to the trained encoder checkpoint .pt file',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).parent / 'outputs'),
        help='Directory to save plots, reports, and metrics',
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=2000,
        help='Number of random game episodes to collect states from',
    )
    parser.add_argument(
        '--mc-rollouts',
        type=int,
        default=50,
        help='Number of Monte Carlo rollouts per state for expected reward estimation',
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    report = run_analysis(
        checkpoint_path=str(checkpoint_path),
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        mc_rollouts=args.mc_rollouts,
    )
    print("\nAnalysis complete.")
    return report


if __name__ == '__main__':
    main()
