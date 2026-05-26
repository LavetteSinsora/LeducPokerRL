"""
D1: Belief Network Quality — Does the Agent Learn to Read Opponent Hands?

Tracks b_mine (player i's belief about opponent's hand) at each decision point
and evaluates it against the opponent's true hand.

Metrics:
  - Accuracy:    fraction of decisions where argmax(b_mine) == h_opp_true
  - Calibration: mean(b_mine[h_opp_idx]) — how much probability on the true hand
  - Entropy:     Shannon entropy of b_mine — lower = more confident

If belief network works:
  - Accuracy increases as more actions are observed within a game
  - Entropy decreases as the game progresses
  - Calibration exceeds the 1/3 random baseline

If belief network fails:
  - Accuracy stays near 33% throughout the game
  - Entropy stays near log(3) ≈ 1.10 regardless of actions seen
  - b_mine barely deviates from the informed prior

Plotted by decision_idx (0=first action in game, up to ~4-5).

Output:
  outputs/d1_accuracy.png     — belief accuracy per decision step
  outputs/d1_calibration.png  — mean P(true hand) per decision step
  outputs/d1_entropy.png      — entropy of b_mine per decision step
  outputs/d1_confusion.png    — confusion matrix (predicted vs true) at final step
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy

from diagnose import (
    CHECKPOINT_EPISODES, load_checkpoint, run_greedy_games,
    ensure_output_dir, OUTPUT_DIR, HAND_LABELS, COLORS, CARD_TO_IDX,
)

N_GAMES = 1500
MAX_DECISION_IDX = 5  # games have at most ~5 decisions (fold early = fewer)


def analyze_beliefs(records):
    """
    For each decision_idx (0..MAX), compute:
      accuracy, calibration (mean P(true)), entropy
    Returns dicts keyed by decision_idx.
    """
    # bucket by decision_idx
    acc_buckets    = defaultdict(list)  # True/False
    calib_buckets  = defaultdict(list)  # float (P(true hand))
    entropy_buckets = defaultdict(list) # float

    for rec in records:
        for dec in rec.decisions:
            idx = min(dec.decision_idx, MAX_DECISION_IDX)
            b = dec.b_mine       # [b_J, b_Q, b_K]
            h_true = dec.opp_hand
            h_idx = CARD_TO_IDX[h_true]

            predicted = int(np.argmax(b))
            is_correct = (predicted == h_idx)
            acc_buckets[idx].append(is_correct)
            calib_buckets[idx].append(b[h_idx])

            # Entropy
            probs = [max(p, 1e-8) for p in b]
            ent = float(scipy_entropy(probs))
            entropy_buckets[idx].append(ent)

    steps = sorted(set(list(acc_buckets.keys())))
    accuracy    = {s: float(np.mean(acc_buckets[s]))   for s in steps}
    calibration = {s: float(np.mean(calib_buckets[s])) for s in steps}
    ent_mean    = {s: float(np.mean(entropy_buckets[s])) for s in steps}
    ent_std     = {s: float(np.std(entropy_buckets[s]))  for s in steps}
    counts      = {s: len(acc_buckets[s]) for s in steps}

    return steps, accuracy, calibration, ent_mean, ent_std, counts


def confusion_matrix_at_step(records, target_step):
    """
    Build 3×3 confusion matrix: predicted_hand vs true_hand
    for decisions at decision_idx == target_step.
    Returns normalized (row-normalized) matrix.
    """
    matrix = np.zeros((3, 3))
    for rec in records:
        for dec in rec.decisions:
            if dec.decision_idx == target_step:
                pred = int(np.argmax(dec.b_mine))
                true = CARD_TO_IDX[dec.opp_hand]
                matrix[true][pred] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return matrix / row_sums


def main():
    ensure_output_dir()

    ep_results = {}

    for ep in CHECKPOINT_EPISODES:
        print(f"D1: ep {ep:,} — playing {N_GAMES} games...")
        config, state_enc, belief_net, q_net = load_checkpoint(ep)
        records = run_greedy_games(state_enc, belief_net, q_net, config, n_games=N_GAMES)
        steps, accuracy, calibration, ent_mean, ent_std, counts = analyze_beliefs(records)
        ep_results[ep] = {
            'steps': steps, 'accuracy': accuracy, 'calibration': calibration,
            'ent_mean': ent_mean, 'ent_std': ent_std, 'counts': counts,
            'records': records,
        }
        # Quick summary
        final_step = max(steps)
        print(f"  step 0 acc={accuracy.get(0,0):.3f}  "
              f"step {final_step} acc={accuracy.get(final_step,0):.3f}  "
              f"step 0 ent={ent_mean.get(0,0):.3f}  "
              f"step {final_step} ent={ent_mean.get(final_step,0):.3f}")

    ep_colors = dict(zip(CHECKPOINT_EPISODES, COLORS))

    # ── Plot 1: Belief Accuracy per decision step ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('D1: Belief Accuracy — argmax(b_mine) == true opponent hand\n'
                 '(baseline 1/3 = pure chance; increases → belief is learning)', fontsize=11)

    for ep in CHECKPOINT_EPISODES:
        r = ep_results[ep]
        steps = r['steps']
        accs  = [r['accuracy'][s] for s in steps]
        ax.plot(steps, accs, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')

    ax.axhline(1/3, color='gray', ls='--', lw=1.5, label='random (1/3)')
    ax.set_xlabel('Decision Index (0 = first action in game)', fontsize=10)
    ax.set_ylabel('Accuracy (fraction correct)', fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d1_accuracy.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Plot 2: Calibration (mean P on true hand) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('D1: Belief Calibration — mean probability assigned to true opponent hand\n'
                 '(baseline 1/3 = uninformative; higher = well-calibrated)', fontsize=11)

    for ep in CHECKPOINT_EPISODES:
        r = ep_results[ep]
        steps = r['steps']
        calibs = [r['calibration'][s] for s in steps]
        ax.plot(steps, calibs, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')

    ax.axhline(1/3, color='gray', ls='--', lw=1.5, label='uninformative (1/3)')
    ax.set_xlabel('Decision Index (0 = first action in game)', fontsize=10)
    ax.set_ylabel('Mean P(true hand)', fontsize=10)
    ax.set_ylim(0.2, 0.65)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d1_calibration.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Entropy per step ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('D1: Belief Entropy — Shannon entropy of b_mine\n'
                 '(baseline log(3)≈1.10 = uniform; decreasing → gaining confidence)', fontsize=11)

    for ep in CHECKPOINT_EPISODES:
        r = ep_results[ep]
        steps = r['steps']
        ents = [r['ent_mean'][s] for s in steps]
        ax.plot(steps, ents, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')

    ax.axhline(np.log(3), color='gray', ls='--', lw=1.5, label='uniform (log 3 ≈ 1.10)')
    ax.set_xlabel('Decision Index (0 = first action in game)', fontsize=10)
    ax.set_ylabel('Entropy (nats)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d1_entropy.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 4: Confusion matrices at decision step 2 (after opponent acts) ──
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle('D1: Belief Confusion Matrix at Decision Step 2\n'
                 '(row=true hand, col=predicted hand; diagonal = correct)', fontsize=12)

    for idx, ep in enumerate(CHECKPOINT_EPISODES):
        ax = axes[idx // 3][idx % 3]
        cm = confusion_matrix_at_step(ep_results[ep]['records'], target_step=2)

        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(HAND_LABELS, fontsize=10)
        ax.set_yticklabels(HAND_LABELS, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        ax.set_title(f'ep {ep:,}', fontsize=10)

        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                        fontsize=10, color='white' if cm[i, j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d1_confusion.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n── D1 Summary: accuracy at each decision step ──────────────────────")
    header = f"{'Episode':>10}" + "".join(f"  step{s}" for s in range(MAX_DECISION_IDX+1))
    print(header)
    for ep in CHECKPOINT_EPISODES:
        r = ep_results[ep]
        row = f"{ep:>10,}"
        for s in range(MAX_DECISION_IDX + 1):
            if s in r['accuracy']:
                row += f"  {r['accuracy'][s]:.3f}"
            else:
                row += "     —  "
        print(row)


if __name__ == '__main__':
    main()
