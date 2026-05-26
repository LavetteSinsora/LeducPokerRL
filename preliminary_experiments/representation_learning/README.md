# Representation Learning Experiments

This subdirectory collects all experiments from the representation learning research
line. Each experiment follows the standard folder structure (`card.md`, `train.py`,
`report.md`, `summary.json`). Import paths all use the prefix
`experiments.representation_learning.<exp_id>.*`.

---

## Experiment Index

### Foundation

| Experiment | Status | One-line Description |
|------------|--------|----------------------|
| `contrastive_repr_v1` | complete | Baseline contrastive encoder (TD-control L0, distance-correlation L1, Rank-N-Contrast L2); L2 achieves eff_dim=5 and reward Spearman ρ=0.641 |

### Single-Objective Representation Probes

| Experiment | Status | One-line Description |
|------------|--------|----------------------|
| `hand_identity_repr_v1` | complete | Triplet + cross-entropy losses for opponent hand rank; 62.8% linear probe accuracy, ordinal collapse onto 1D hand-rank axis |
| `repr_geometry_v1` | complete | PCA/t-SNE geometry analysis of contrastive_repr_v1 L2 encoder; 3 eff dims at 80%, two dominant PCA axes, zero opponent hand footprint |
| `repr_policy_v1` | complete | REINFORCE policy on frozen/finetuned contrastive encoder; frozen L1 encoder +0.36 chips/round vs raw features baseline; finetuning hurts |

### Multi-Axis (Dual-Objective) Representation

| Experiment | Status | One-line Description |
|------------|--------|----------------------|
| `dual_axis_repr_v1` | complete | Joint InfoNCE for both axes; eff_dim=2, hand acc 53.8%, reward ρ=0.107; InfoNCE diverged but some structure learned |
| `dual_axis_repr_v2` | complete | Per-axis SupCon+VICReg; eff_dim=2, hand acc 63.3%, reward ρ=0.118; confirmed two dominant PCA axes |
| `dual_axis_repr_v3` | complete | Hybrid L1+SupCon; hand SupCon dominated ~99.96% of gradient; hand acc 67.8% but reward ρ collapsed to 0.083 |
| `dual_axis_repr_v4` | complete | EMA loss normalization to balance axes; overcorrected — reward ρ=0.672 (best ever) but hand acc collapsed to 38.3% |
| `dual_axis_repr_v5` | complete | Subspace partitioning (4 reward dims + 4 hand dims); FIRST to meet both targets simultaneously — reward ρ=0.536, hand acc=65.2% |

### Diagnostics & Analysis

| Experiment | Status | One-line Description |
|------------|--------|----------------------|
| `value_dim_search_v1` | complete | Grid search over value network widths; training instability (not capacity) is the bottleneck; 32×32 and 32×16 both regressed after brief peak |
| `value_based_repr_analysis` | complete | Analysis of TD(0) hidden layer geometry; implicit reward ρ=0.287, exceeding contrastive_repr_v1 baseline without explicit contrastive training |

---

## Cross-Experiment Dependencies

Two experiments import from `contrastive_repr_v1`:

- `repr_geometry_v1/analyze.py` — loads `ContrastiveEncoder` and `ContrastiveReprAgent` for geometry analysis
- `repr_policy_v1/agent.py` — loads `ContrastiveEncoder` as frozen backbone for policy networks

All other imports within this subdirectory are self-contained (importing only from the
same experiment's own modules).

## Research Narrative

The line started with `contrastive_repr_v1`, which established that contrastive losses
produce richer multi-dimensional representations than TD(0)'s 1D value collapse.
`repr_geometry_v1` and `repr_policy_v1` confirmed downstream utility (frozen encoder
improves REINFORCE). The dual_axis series then asked whether a single encoder could
serve two objectives simultaneously: reward-metric structure AND opponent hand
discriminability. After four failed balancing strategies (v1–v4), v5's subspace
partitioning was the first to meet both targets at once, establishing structural
isolation as the key design principle.
