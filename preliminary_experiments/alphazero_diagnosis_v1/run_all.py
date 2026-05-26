"""
Run all diagnosis scripts in priority order.

Usage:
  python run_all.py          # Run D4, D5, D1, D3, D6 (skip slow D2)
  python run_all.py --all    # Run all including D2 (PIMC search, ~30 min)
  python run_all.py --only d4 d5    # Run specific scripts
"""

import argparse
import time
import subprocess
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ('d4', 'd4_embeddings.py',  '~1 min',   False),  # (id, file, est_time, requires_search)
    ('d5', 'd5_strategy.py',    '~5 min',   False),
    ('d1', 'd1_belief.py',      '~15 min',  False),
    ('d3', 'd3_pubstate.py',    '~15 min',  False),
    ('d6', 'd6_portfolio.py',   '~15 min',  False),
    ('d2', 'd2_q_agreement.py', '~30 min',  True),   # slow: requires PIMC
]


def run_script(script_file, extra_args=None):
    path = os.path.join(_HERE, script_file)
    cmd = [sys.executable, path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {script_file}")
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(cmd, cwd=_HERE)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] {script_file} failed with exit code {result.returncode}")
        return False
    print(f"\n[DONE] {script_file} completed in {elapsed/60:.1f}m")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run AlphaZero diagnosis scripts')
    parser.add_argument('--all', action='store_true', help='Include D2 (requires PIMC, ~30 min)')
    parser.add_argument('--only', nargs='+', metavar='ID',
                        help='Run only specific scripts by ID (d1, d2, d3, d4, d5, d6)')
    parser.add_argument('--d2-games', type=int, default=150,
                        help='Games per checkpoint for D2 (default 150)')
    args = parser.parse_args()

    to_run = []
    if args.only:
        id_set = set(args.only)
        to_run = [s for s in SCRIPTS if s[0] in id_set]
        if not to_run:
            print(f"Unknown script IDs: {args.only}. Valid: {[s[0] for s in SCRIPTS]}")
            sys.exit(1)
    else:
        to_run = [s for s in SCRIPTS if not s[3] or args.all]

    print("AlphaZero Diagnosis — run plan:")
    for sid, fname, est, requires_search in to_run:
        tag = " [PIMC search]" if requires_search else ""
        print(f"  {sid}: {fname:<30}  estimated {est}{tag}")

    t_total = time.time()
    results = {}
    for sid, fname, _, requires_search in to_run:
        extra = [f'--games={args.d2_games}'] if sid == 'd2' else None
        ok = run_script(fname, extra)
        results[sid] = ok

    print(f"\n{'='*60}")
    print(f"All scripts finished in {(time.time()-t_total)/60:.1f}m total")
    print(f"{'='*60}")
    for sid, ok in results.items():
        status = 'OK' if ok else 'FAILED'
        print(f"  {sid}: {status}")

    print(f"\nPlots saved to: {os.path.join(_HERE, 'outputs')}/")


if __name__ == '__main__':
    main()
