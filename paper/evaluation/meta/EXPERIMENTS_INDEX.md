# Experiments

Each experiment lives in its own subdirectory. Discover experiments with `ls experiments/`.

## Folder Structure

Every experiment folder contains:

| File | Purpose |
|------|---------|
| `card.md` | Written BEFORE coding: hypothesis, success criteria, single changed axis |
| `train.py` | Training entrypoint |
| `eval.py` | Evaluation entrypoint |
| `report.md` | Findings, written after evaluation |
| `summary.json` | Machine-readable metrics and status |

Raw training outputs go to `<exp_id>/outputs/` inside the experiment folder (gitignored via `**/outputs/`).

## Standards and Protocol

See `CLAUDE.md` (root) for the experiment lifecycle and evaluation protocol.
See `STANDARDS.md` for detailed artifact requirements and report standards.
See `EVAL_CONFIG.json` for frozen evaluation and training parameters.

## Template

Copy `example_experiment/` to start a new experiment.
