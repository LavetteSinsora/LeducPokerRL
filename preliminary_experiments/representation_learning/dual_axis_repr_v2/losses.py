"""Loss functions for dual-axis SupCon representation learning.

Three components:
- SupConLoss: Supervised Contrastive Loss (Khosla et al. 2020, NeurIPS)
- get_reward_bin_labels: discretize continuous rewards into integer bins
- VICRegVarianceLoss: penalize low-variance dimensions to prevent collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    For each anchor i in the batch:
      positives = {j != i : label_j == label_i}
      L_i = -1/|P(i)| * sum_{j in P(i)} log(
              exp(sim(i,j)/tau) / sum_{k != i} exp(sim(i,k)/tau)
            )

    Total loss = mean over anchors with at least one positive.

    sim(i,j) = cosine similarity (embeddings L2-normalized internally).

    Args:
        temperature: softmax temperature tau (default 0.07, standard SupCon)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            embeddings: (N, D) raw embedding tensor (will be L2-normalized)
            labels: (N,) integer labels

        Returns:
            scalar loss (0.0 if no anchor has at least one positive)
        """
        N = embeddings.size(0)
        if N < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # L2-normalize embeddings
        z = F.normalize(embeddings, dim=1)  # (N, D)

        # Cosine similarity matrix scaled by temperature
        sim_matrix = z @ z.T / self.temperature  # (N, N)

        # Build label masks
        labels = labels.to(embeddings.device)
        labels_row = labels.unsqueeze(1)  # (N, 1)
        labels_col = labels.unsqueeze(0)  # (1, N)
        same_label = (labels_row == labels_col)  # (N, N)

        # Eye mask to exclude self
        eye = torch.eye(N, dtype=torch.bool, device=embeddings.device)

        # Positive mask: same label, not self
        pos_mask = same_label & ~eye  # (N, N)

        # For log-sum-exp denominator: all k != i
        # Set diagonal to -inf so it won't contribute to logsumexp
        sim_no_self = sim_matrix.clone()
        sim_no_self[eye] = float('-inf')

        # log of denominator: logsumexp over all k != i
        log_denom = torch.logsumexp(sim_no_self, dim=1)  # (N,)

        # For each anchor i, compute mean log-prob over its positives
        # log_prob[i, j] = sim(i,j)/tau - logsumexp(sim(i,k)/tau for k != i)
        log_prob = sim_matrix - log_denom.unsqueeze(1)  # (N, N)

        losses = []
        for i in range(N):
            pos_indices = pos_mask[i]
            if not pos_indices.any():
                continue
            # Mean log-prob over positives for anchor i
            loss_i = -log_prob[i][pos_indices].mean()
            losses.append(loss_i)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(losses).mean()


def get_reward_bin_labels(rewards: torch.Tensor, n_bins: int = 5) -> torch.Tensor:
    """Discretize continuous rewards into integer bin labels.

    Leduc rewards are typically in [-4, +4].
    Fixed thresholds: [-inf, -2.0, -0.5, 0.5, 2.0, +inf] -> 5 bins (0-4)

    Bin semantics:
      0: big loss (reward < -2.0)
      1: small loss (-2.0 <= reward < -0.5)
      2: near-zero (-0.5 <= reward < 0.5)
      3: small win (0.5 <= reward < 2.0)
      4: big win (reward >= 2.0)

    Args:
        rewards: (N,) float tensor of terminal rewards
        n_bins: number of bins (currently only 5 is supported with fixed thresholds)

    Returns:
        (N,) long tensor of bin labels in [0, n_bins-1]
    """
    thresholds = [-2.0, -0.5, 0.5, 2.0]
    labels = torch.zeros(len(rewards), dtype=torch.long, device=rewards.device)
    for i, t in enumerate(thresholds):
        labels += (rewards >= t).long()
    return labels  # values in {0, 1, 2, 3, 4}


class VICRegVarianceLoss(nn.Module):
    """VICReg variance term: penalize dimensions with std below target.

    From VICReg (Bardes et al. 2022). Prevents dimensional collapse by
    encouraging each embedding dimension to have sufficient variance.

    Args:
        target_std: minimum desired std per dimension (default 1.0)
    """

    def __init__(self, target_std: float = 1.0):
        super().__init__()
        self.target_std = target_std

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute variance regularization loss.

        Args:
            embeddings: (N, D) embedding tensor

        Returns:
            scalar loss (mean over dimensions of max(0, target_std - std(z_k)))
        """
        std_per_dim = embeddings.std(dim=0)
        return torch.relu(self.target_std - std_per_dim).mean()
