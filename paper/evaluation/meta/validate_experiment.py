"""
Experiment Artifact Validator

Run this before writing any experiment report to check that all required
artifacts are present and that evaluation parameters match the standard
defined in EVAL_CONFIG.json.

Usage:
    python experiments/validate_experiment.py outputs/my_experiment/
    python experiments/validate_experiment.py outputs/my_experiment/ --strict
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_eval_config() -> dict:
    config_path = Path(__file__).parent / "EVAL_CONFIG.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def validate(output_dir: str, strict: bool = False) -> bool:
    path = Path(output_dir)
    config = load_eval_config()
    issues = []
    warnings = []
    ok_items = []

    def check(label: str, condition: bool, message: str, is_warning: bool = False):
        if condition:
            ok_items.append(label)
        else:
            (warnings if is_warning else issues).append(f"{label}: {message}")

    # --- Required artifacts ---
    check("checkpoint.pt",
          (path / "checkpoint.pt").exists(),
          "Final checkpoint missing")

    check("tournament_history.json",
          (path / "tournament_history.json").exists(),
          "Tournament history missing — was TournamentCheckpointer used during training?")

    check("checkpoint_best_avg.pt",
          (path / "checkpoint_best_avg.pt").exists(),
          "Best-avg checkpoint missing")

    check("checkpoint_best_robust.pt",
          (path / "checkpoint_best_robust.pt").exists(),
          "Best-robust checkpoint missing (primary candidate for promotion)")

    # At least one snapshot checkpoint
    snapshots = list(path.glob("checkpoint_ep*.pt"))
    check("snapshot checkpoints",
          len(snapshots) > 0,
          "No episode snapshot checkpoints found (checkpoint_ep*.pt)")

    # --- Tournament history parameter validation ---
    history_path = path / "tournament_history.json"
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)

            check("tournament_history non-empty",
                  len(history) > 0,
                  "tournament_history.json is empty")

            if history and config:
                latest = history[-1]
                # Check rounds_per_matchup via matchup round counts
                for opp_id, detail in latest.get("matchups", {}).items():
                    actual_rounds = detail.get("rounds_counted", 0)
                    expected_rounds = config.get("rounds_per_matchup", 2000)
                    check(f"rounds_per_matchup ({opp_id})",
                          actual_rounds == expected_rounds,
                          f"rounds_counted={actual_rounds}, expected {expected_rounds} "
                          f"(per EVAL_CONFIG.json)",
                          is_warning=True)
                    break  # Check just the first opponent; they should all match

                # Report best scores
                best_avg = max((r["tournament_avg_chips"] for r in history), default=None)
                best_robust = max((r["tournament_robustness"] for r in history), default=None)
                n_tournaments = len(history)
                ok_items.append(
                    f"tournament stats: {n_tournaments} tournaments, "
                    f"best_avg={best_avg:+.4f}, best_robust={best_robust:+.4f}"
                )

        except Exception as e:
            issues.append(f"tournament_history.json: could not parse ({e})")

    # --- Optional but recommended ---
    check("train_config.json",
          (path / "train_config.json").exists(),
          "Training config missing (recommended for reproducibility)",
          is_warning=True)

    # --- Print report ---
    print(f"\nValidating: {path}")
    print("=" * 60)

    for item in ok_items:
        print(f"  [OK]      {item}")

    for item in warnings:
        print(f"  [WARN]    {item}")

    for item in issues:
        print(f"  [MISSING] {item}")

    print("=" * 60)

    if not issues:
        print(f"  PASS — {len(ok_items)} checks passed, {len(warnings)} warnings")
    else:
        print(f"  FAIL — {len(issues)} missing required artifacts, "
              f"{len(warnings)} warnings")

    print()

    if strict:
        return len(issues) == 0 and len(warnings) == 0
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate experiment output directory before writing report"
    )
    parser.add_argument("output_dir", help="Path to experiment output directory")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on warnings as well as errors")
    args = parser.parse_args()

    passed = validate(args.output_dir, strict=args.strict)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
