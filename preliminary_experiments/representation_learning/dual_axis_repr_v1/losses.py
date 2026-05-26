"""Loss functions for dual-axis contrastive representation learning.

Two components:
- DualAxisContrastiveLoss: InfoNCE-style loss with joint reward+hand pair construction
- VICRegVarianceLoss: anti-collapse regularizer (prevents 1D collapse)
"""

import torch
import torch.nn as nn


class DualAxisContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss with dual-axis positive/negative construction.

    Positive pair (i, j): |R_i - R_j| < reward_thresh AND hand_i == hand_j
    Negative pair (i, j): |R_i - R_j| > reward_margin OR hand_i != hand_j

    For each anchor i:
      - positives = {j: j is a positive pair with i}
      - negatives = {j: j is a negative pair with i}
      - loss = -log( sum_pos exp(sim(i,j)/τ) / sum_neg exp(sim(i,k)/τ) )
      - sim(i,j) = -||z_i - z_j||^2  (negative squared L2)

    Skips anchors with no positives or no negatives in the batch.

    Args:
        temperature: softmax temperature (default 0.5)
        reward_thresh: |ΔR| < this means reward-similar (default 0.5)
        reward_margin: |ΔR| > this means reward-dissimilar (default 1.5)
            Gap [reward_thresh, reward_margin] is the "ignore zone" for
            same-hand pairs — neither clearly positive nor negative.
    """

    def __init__(self, temperature: float = 0.5, reward_thresh: float = 0.5,
                 reward_margin: float = 1.5):
        super().__init__()
        self.temperature = temperature
        self.reward_thresh = reward_thresh
        self.reward_margin = reward_margin

    def forward(self, z: torch.Tensor, rewards: torch.Tensor,
                hand_labels: torch.Tensor) -> torch.Tensor:
        """Compute dual-axis contrastive loss.

        Args:
            z: embeddings (N, D)
            rewards: terminal rewards (N,)
            hand_labels: opponent hand labels (N,) — 0=J, 1=Q, 2=K

        Returns:
            scalar loss, or 0.0 if no valid anchors in batch
        """
        N = z.size(0)
        if N < 4:
            return torch.tensor(0.0, device=z.device)

        # --- Pair masks ---
        # Pairwise reward differences
        reward_diff = (rewards.unsqueeze(0) - rewards.unsqueeze(1)).abs()  # (N, N)

        # Pairwise hand equality
        same_hand = (hand_labels.unsqueeze(0) == hand_labels.unsqueeze(1))  # (N, N)

        # Positive mask: reward-similar AND same hand
        reward_similar = reward_diff < self.reward_thresh
        pos_mask = reward_similar & same_hand  # (N, N)

        # Negative mask: reward-dissimilar OR different hand
        reward_dissimilar = reward_diff > self.reward_margin
        diff_hand = ~same_hand
        neg_mask = reward_dissimilar | diff_hand  # (N, N)

        # Exclude self-pairs from both masks
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        pos_mask = pos_mask & ~eye
        neg_mask = neg_mask & ~eye

        # --- Similarities ---
        # sim(i, j) = -||z_i - z_j||^2 / temperature
        diffs = z.unsqueeze(0) - z.unsqueeze(1)  # (N, N, D)
        sim_matrix = -(diffs ** 2).sum(dim=2) / self.temperature  # (N, N)

        # --- Per-anchor loss ---
        total_loss = torch.tensor(0.0, device=z.device)
        num_valid = 0

        for i in range(N):
            pos_idx = pos_mask[i]   # (N,) bool
            neg_idx = neg_mask[i]   # (N,) bool

            if not pos_idx.any() or not neg_idx.any():
                # Skip anchors with no positives or no negatives
                continue

            # Numerator: log-sum-exp over positives
            pos_sims = sim_matrix[i][pos_idx]     # (n_pos,)
            log_pos = torch.logsumexp(pos_sims, dim=0)

            # Denominator: log-sum-exp over negatives
            neg_sims = sim_matrix[i][neg_idx]     # (n_neg,)
            log_neg = torch.logsumexp(neg_sims, dim=0)

            # Loss for this anchor: -log(sum_pos / sum_neg)
            anchor_loss = -(log_pos - log_neg)
            total_loss = total_loss + anchor_loss
            num_valid += 1

        if num_valid == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        return total_loss / num_valid


class VICRegVarianceLoss(nn.Module):
    """VICReg variance term: penalize dimensions with std below target.

    Prevents dimensional collapse (the 1D ordinal collapse seen in
    hand_identity_repr_v1) by encouraging all embedding dimensions to
    carry non-trivial variance.

    Loss = mean_over_dims( max(0, target_std - std(z_k))^2 )

    Args:
        target_std: minimum desired std per dimension (default 1.0)
    """

    def __init__(self, target_std: float = 1.0):
        super().__init__()
        self.target_std = target_std

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute variance regularization loss.

        Args:
            z: embeddings (N, D)

        Returns:
            scalar loss
        """
        std_per_dim = z.std(dim=0)  # (D,)
        return torch.relu(self.target_std - std_per_dim).pow(2).mean()
