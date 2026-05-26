# Experiment Card Template

Fill this in before writing any code. The card defines the research question and
success criteria. If you cannot fill it out clearly, the experiment is not ready
to run.

---

## Research Question
<!-- One sentence. Falsifiable. -->
Does [CHANGE] improve [METRIC] compared to [BASELINE]?

## Parent Agent / Prior Experiment
<!-- What this builds on. -->
- Parent: `value_based` (or other promoted agent)
- Prior experiment: (if incremental)

## Single Changed Axis
<!-- What is the ONE thing different from the parent? -->
- Changed: [e.g., "hidden width reduced from 64 to 32"]
- Control: everything else identical (training budget, lr, eval suite)

## Mechanism Story
<!-- Why should this work? Trace the causal chain step by step. -->
1. [Because X...]
2. [...the model should learn Y...]
3. [...which means metric Z should improve because...]

If you cannot write this story, reconsider whether the experiment is well-motivated.

## Hypothesis
<!-- Specific, numeric, falsifiable. -->
The experiment agent will achieve [METRIC] ≥ [VALUE] vs [BASELINE], compared to
the parent agent's [PARENT_VALUE].

## Success Criteria
<!-- What constitutes a definitive positive result? -->
- Primary: avg_chips_vs_heuristic ≥ [threshold]
- Secondary: [metric] ≥ [threshold]

## Required Artifacts (per STANDARDS.md)
<!-- Check these off as you produce them. -->
- [ ] `outputs/<exp_id>/checkpoint.pt`
- [ ] `outputs/<exp_id>/train_history.json`
- [ ] `outputs/<exp_id>/train_config.json`
- [ ] `outputs/<exp_id>/training_curve.png`
- [ ] `outputs/<exp_id>/eval_curve.png`
- [ ] `outputs/<exp_id>/results.json`
- [ ] (repr experiments) `repr_quality_curve.png`, `pca_snapshots/`, `effective_dim_curve.png`
- [ ] (policy/value) `entropy_curve.png`, `value_error_curve.png`, `eval_vs_baselines.png`

## Minimum Training Budget (per STANDARDS.md)
<!-- Fill in the applicable row. -->
- Value/policy network: **50,000 episodes**
- Representation learning: **30,000 episodes**
- Auxiliary head only: **20,000 episodes**

## Metrics to Report
<!-- List each metric with its METRICS_GLOSSARY.md name. -->
- `avg_chips_vs_heuristic` (1000 rounds, position-swapped)
- `reward_spearman_rho_pairwise` (if representation experiment)
- `hand_probe_accuracy` (if representation experiment)
- `effective_dim_80` (if representation experiment)

## Risks
<!-- What could go wrong? How would you detect it? -->
- Risk: [e.g., "training instability from self-play non-stationarity"]
- Detection: [e.g., "monitor loss_plateau_pct; check peak vs final score gap"]
