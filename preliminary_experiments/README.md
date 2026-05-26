# `preliminary_experiments/` — Research Threads That Came Before

These are the directions we explored before converging on the opponent-modulated value network presented in the paper. **None of this is required to reproduce the paper.** It's preserved so reviewers, future students, and our future selves can see what was tried, what worked, and what didn't.

Each subdirectory typically has its own `card.md` (the hypothesis), `report.md` (the outcome), or a top-level docstring explaining intent.

---

## Index

### Alternative model families

| Directory | What we tried | Verdict |
|---|---|---|
| [`alphazero/`](alphazero/) | AlphaZero-style PUCT search + value network | Promising at small budgets, scaling issues with belief representation |
| [`alphazero_v2/`](alphazero_v2/) … [`alphazero_v5/`](alphazero_v5/) | Successive AlphaZero refinements | Diagnostics in [`alphazero_diagnosis_v1/`](alphazero_diagnosis_v1/) explain why we stopped |
| [`hand_conditioned/`](hand_conditioned/) | Hand-conditioned action likelihood model | Supervised signal too weak in 2-round Leduc |
| [`hand_posterior_belief/`](hand_posterior_belief/) | GRU posterior over opponent hand rank | Worked in toy setting, didn't beat the simpler stats-based approach |
| [`representation_learning/`](representation_learning/) | Contrastive / multi-axis embeddings as frozen feature extractor | `dual_axis_repr_v5` was the first to hit both targets but didn't transfer to gameplay improvement |

### Opponent modeling iterations (predecessors of `paper/agents/full_modulation/`)

| Directory | Notes |
|---|---|
| [`baseline_value_v1/`](baseline_value_v1/) | First TD(0) value net trained against rule-based pool |
| [`value_opponent_pool/`](value_opponent_pool/) | Studied pool composition and weighting |
| [`opp_stats_input_aug/`](opp_stats_input_aug/) | Concatenate opponent stats with state — informs `paper/evaluation/shared/stats_tracker.py` |
| [`opp_stats_modulation_v1/`](opp_stats_modulation_v1/) | First multiplicative-modulation prototype |
| [`opp_stats_modulation_v2/`](opp_stats_modulation_v2/) | Ungated vs state-gated variants; predecessor of the paper's gated head |
| [`opp_encoder_modulation/`](opp_encoder_modulation/) | Learned opponent encoder (v2, v3 variants); `opp_encoder_v1` was promoted as the paper's OOD opponent |
| [`best_response_agents/`](best_response_agents/) | Per-opponent best-response training for analysis |
| [`action_switching/`](action_switching/) | Analyzing where opponent statistics actually shift action selection |
| [`ev_variation_extras/`](ev_variation_extras/) | Full code for the EV-variation analysis (the data subset used by the paper is in `paper/ev_analysis/`) |

### Variants we trained but didn't include in the paper

| Directory | Notes |
|---|---|
| [`dali_variants/full_modulation_deep/`](dali_variants/full_modulation_deep/) | Deeper modulation head — no robustness gain |
| [`dali_variants/gated_modulation/`](dali_variants/gated_modulation/) | Predecessor of the final gating mechanism (single seed) |
| [`dali_variants/value_based_deep/`](dali_variants/value_based_deep/) | Deeper base net — Appendix capacity argument |
| [`dali_remainder/`](dali_remainder/) | Misc study report, launch script, and shared configs from the DALI modulation thread |

### Infrastructure / dashboard

| Directory | Notes |
|---|---|
| [`promoted_registry/`](promoted_registry/) | Earlier parallel implementation of the modulated agent + dashboard's training scaffolding (`base_trainer`, `training_manager`). Different code path from the paper's. |
| [`dashboard/`](dashboard/) | Flask web UI for visualising training history and watching agents play. `run.py` launches it. |
| [`example_experiment/`](example_experiment/) | Template scaffold for new experiments |
| [`capacity_study_v1/`](capacity_study_v1/) | Architecture sweep for the base value net |

### Misc

| Directory | Notes |
|---|---|
| [`alphazero_outputs/`](alphazero_outputs/) | Frozen artifacts from old AlphaZero/legacy runs |
| [`legacy_archive/`](legacy_archive/) | Pre-cleanup `src/` layout, old reports, deprecated tests, original wiki |

---

## Why keep this around?

A few reasons:

1. **Reviewer questions** — if a reader of the paper asks "did you try X?", this directory is the answer.
2. **Provenance** — `paper/evaluation/shared/stats_tracker.py` originated in `opp_stats_input_aug/`; `paper/baselines/opp_encoder_v1/` originated in `opp_encoder_modulation/opp_encoder_modulation_v1/`. The lineage is visible here.
3. **Future work** — several threads (hand-conditioned belief, learned opponent encoder) are listed as future directions in the paper. The starting points are here.
