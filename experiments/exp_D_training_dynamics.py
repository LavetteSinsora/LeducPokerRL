"""
Experiment D: Training Dynamics Comparison (H4)

HYPOTHESIS: The adaptive agent's session-based self-play creates a highly
non-stationary training target. Both players accumulate stats about each
other WITHIN the same session, causing them to co-adapt during training.
This mutual adaptation makes the training signal noisier and harder to
converge on, compared to vanilla single-hand self-play where there is no
cross-hand state to create non-stationarity.

FALSIFICATION CONDITION: If training loss variance is similar for both
agents, H4 is false.

TEST:
  1. Load training history JSON files for both agents.
  2. Compare loss curves: mean, variance, convergence speed.
  3. Compute rolling variance of loss to detect instability.
  4. Compare final evaluation performance plateaus.
  5. Run short fresh training runs (1000 episodes each) and capture
     the raw loss trajectory to compare variability directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

VANILLA_HISTORY = "models/value_based_agent_history.json"
ADAPTIVE_HISTORY = "models/adaptive_value_agent_history.json"


def load_history(path):
    with open(path, 'r') as f:
        return json.load(f)


def extract_series(history, event_type):
    """Extract (episode, value) pairs for a given event type."""
    series = [(e['episode'], e.get('loss') if event_type == 'loss' else e.get('avg_chips_per_round'))
              for e in history if e.get('type') == event_type]
    # Filter None values
    series = [(ep, v) for ep, v in series if v is not None]
    return series


def analyze_loss_series(series, label, window=50):
    """Analyze a loss time series for convergence and variance properties."""
    if not series:
        print(f"  {label}: No data found.")
        return {}

    episodes = [s[0] for s in series]
    losses = [s[1] for s in series]

    losses_arr = np.array(losses)

    # Basic stats
    print(f"\n  {label}:")
    print(f"    Total loss updates:  {len(losses)}")
    print(f"    Episode range:       {episodes[0]} – {episodes[-1]}")
    print(f"    Loss mean:           {losses_arr.mean():.4f}")
    print(f"    Loss std:            {losses_arr.std():.4f}")
    print(f"    Loss min:            {losses_arr.min():.4f}")
    print(f"    Loss max:            {losses_arr.max():.4f}")
    print(f"    CV (std/mean):       {losses_arr.std()/losses_arr.mean():.4f}  ← higher = noisier")

    # Early vs late stability
    n = len(losses)
    early_loss = losses_arr[:n//5]
    late_loss  = losses_arr[4*n//5:]
    print(f"    Early loss mean (first 20%):  {early_loss.mean():.4f}  std={early_loss.std():.4f}")
    print(f"    Late  loss mean (last  20%):  {late_loss.mean():.4f}  std={late_loss.std():.4f}")
    print(f"    Improvement (early→late):     {early_loss.mean() - late_loss.mean():+.4f}")

    # Rolling variance
    if len(losses_arr) >= window:
        rolling_var = [losses_arr[i:i+window].var() for i in range(0, len(losses_arr)-window, window//2)]
        print(f"    Rolling variance (window={window}) — avg: {np.mean(rolling_var):.4f}, max: {np.max(rolling_var):.4f}")

    return {
        'mean': losses_arr.mean(),
        'std': losses_arr.std(),
        'cv': losses_arr.std() / losses_arr.mean(),
        'late_mean': late_loss.mean(),
        'late_std': late_loss.std(),
        'n': len(losses)
    }


def analyze_eval_series(series, label):
    """Analyze evaluation performance over training."""
    if not series:
        print(f"  {label}: No eval data.")
        return {}

    episodes = [s[0] for s in series]
    perfs = [s[1] for s in series]
    perfs_arr = np.array(perfs)

    n = len(perfs)
    early_perf = perfs_arr[:n//5]
    late_perf  = perfs_arr[4*n//5:]

    print(f"\n  {label} evaluation performance:")
    print(f"    Total eval points:   {n}")
    print(f"    Overall mean:        {perfs_arr.mean():+.4f}")
    print(f"    Overall std:         {perfs_arr.std():.4f}")
    print(f"    Early perf (avg):    {early_perf.mean():+.4f}")
    print(f"    Late  perf (avg):    {late_perf.mean():+.4f}")
    print(f"    Peak performance:    {perfs_arr.max():+.4f}")
    print(f"    Late-stage std:      {late_perf.std():.4f}  ← higher = more volatile")

    # Check for performance collapse (drop after peak)
    peak_idx = np.argmax(perfs_arr)
    if peak_idx < n - 1:
        post_peak = perfs_arr[peak_idx:]
        post_peak_drop = perfs_arr[peak_idx] - post_peak.min()
        print(f"    Peak at eval #{peak_idx+1}: {perfs_arr[peak_idx]:+.4f}")
        print(f"    Post-peak drop:      {post_peak_drop:.4f}  ← large = catastrophic forgetting")

    return {
        'late_mean': late_perf.mean(),
        'late_std': late_perf.std(),
        'peak': perfs_arr.max(),
    }


def compare_convergence_speed(vanilla_losses, adaptive_losses, threshold_pct=0.5):
    """
    How many episodes does it take each agent to reduce loss below
    threshold_pct of its initial loss?
    """
    def find_convergence(series, threshold_pct):
        if not series:
            return None
        losses = [s[1] for s in series]
        initial = np.mean(losses[:min(10, len(losses))])
        target = initial * threshold_pct
        for ep, loss in series:
            if loss < target:
                return ep
        return None

    v_conv = find_convergence(vanilla_losses, threshold_pct)
    a_conv = find_convergence(adaptive_losses, threshold_pct)

    print(f"\n  Convergence speed (time to reach {threshold_pct*100:.0f}% of initial loss):")
    print(f"    Vanilla:  {v_conv if v_conv else 'NEVER'}")
    print(f"    Adaptive: {a_conv if a_conv else 'NEVER'}")
    if v_conv and a_conv:
        ratio = a_conv / v_conv
        print(f"    Ratio (adaptive/vanilla): {ratio:.2f}x  ← >1 means slower")


def main():
    print("=" * 65)
    print("EXPERIMENT D: Training Dynamics Analysis (H4)")
    print("=" * 65)

    # Load histories
    print(f"\n[1/4] Loading training histories...")
    vanilla_history  = load_history(VANILLA_HISTORY)
    adaptive_history = load_history(ADAPTIVE_HISTORY)
    print(f"  Vanilla history:  {len(vanilla_history)} events")
    print(f"  Adaptive history: {len(adaptive_history)} events")

    # Print sample events to understand structure
    print(f"\n  Vanilla sample events: {vanilla_history[:3]}")
    print(f"  Adaptive sample events: {adaptive_history[:3]}")

    # Extract series
    vanilla_losses  = extract_series(vanilla_history,  'batch_update')
    adaptive_losses = extract_series(adaptive_history, 'batch_update')
    vanilla_evals   = extract_series(vanilla_history,  'evaluation')
    adaptive_evals  = extract_series(adaptive_history, 'evaluation')

    print(f"\n  Vanilla loss updates:   {len(vanilla_losses)}")
    print(f"  Adaptive loss updates:  {len(adaptive_losses)}")
    print(f"  Vanilla eval points:    {len(vanilla_evals)}")
    print(f"  Adaptive eval points:   {len(adaptive_evals)}")

    # ── Loss analysis ──
    print(f"\n[2/4] LOSS SERIES ANALYSIS")
    print("-" * 55)
    v_stats = analyze_loss_series(vanilla_losses,  "Vanilla")
    a_stats = analyze_loss_series(adaptive_losses, "Adaptive")

    # ── Eval analysis ──
    print(f"\n[3/4] EVALUATION PERFORMANCE ANALYSIS")
    print("-" * 55)
    v_eval = analyze_eval_series(vanilla_evals,  "Vanilla")
    a_eval = analyze_eval_series(adaptive_evals, "Adaptive")

    # ── Convergence speed ──
    print(f"\n[4/4] CONVERGENCE SPEED")
    print("-" * 55)
    compare_convergence_speed(vanilla_losses, adaptive_losses, threshold_pct=0.5)
    compare_convergence_speed(vanilla_losses, adaptive_losses, threshold_pct=0.25)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY (H4: Training Instability?)")
    print("=" * 65)

    if v_stats and a_stats:
        cv_ratio = a_stats['cv'] / (v_stats['cv'] + 1e-10)
        print(f"  Vanilla loss CV:  {v_stats['cv']:.4f}")
        print(f"  Adaptive loss CV: {a_stats['cv']:.4f}")
        print(f"  Ratio (adaptive/vanilla CV): {cv_ratio:.2f}x")

        if cv_ratio > 1.5:
            print("\n  → Adaptive training is significantly noisier. H4 SUPPORTED.")
        elif cv_ratio > 1.1:
            print("\n  → Adaptive training is slightly noisier. H4 WEAKLY supported.")
        else:
            print("\n  → Similar loss variance. H4 NOT strongly supported.")

    if v_eval and a_eval:
        print(f"\n  Vanilla late performance:  {v_eval['late_mean']:+.4f} ± {v_eval['late_std']:.4f}")
        print(f"  Adaptive late performance: {a_eval['late_mean']:+.4f} ± {a_eval['late_std']:.4f}")
        print(f"  Vanilla peak:              {v_eval['peak']:+.4f}")
        print(f"  Adaptive peak:             {a_eval['peak']:+.4f}")


if __name__ == "__main__":
    main()
