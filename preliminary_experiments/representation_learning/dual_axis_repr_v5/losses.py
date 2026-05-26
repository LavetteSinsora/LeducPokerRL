"""Loss functions for dual-axis subspace-partitioned representation learning (v5).

Combines:
- SoftDistanceL1Loss: L1 soft distance correlation from contrastive_repr_v1
  Preserves continuous metric ordering between reward values.
  Applied ONLY to dims 0:4 (reward subspace).
- SupConLoss: Supervised Contrastive Loss for opponent hand identity.
  Applied ONLY to dims 4:8 (hand subspace).
- VICRegVarianceLoss: Variance regularization to prevent dimensional collapse.
  Applied to full 8-dim embedding.

Total loss (after subspace split):
    L_total = L_L1_reward(z[:, 0:4]) + lambda_hand * L_SupCon(z[:, 4:8]) + lambda_var * L_VICReg(z)

Copied verbatim from dual_axis_repr_v3/losses.py — no changes needed.
The subspace slicing is done in the trainer, not here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDistanceL1Loss(nn.Module):
    """L1 soft distance correlation loss from contrastive_repr_v1.

    L = mean( (||z_i - z_j||_2 - beta * |R_i - R_j|)^2 )

    beta is auto-calibrated per batch:
        beta = mean_embed_dist / mean_reward_dist

    This preserves continuous metric ordering — states with more different
    rewards will have more different embeddings, proportionally.

    Args:
        eps: small value to prevent division by zero in beta calibration
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _calibrate_beta(self, z: torch.Tensor, rewards: torch.Tensor) -> float:
        """Auto-calibrate beta so target distances match embedding scale.

        beta = mean_embed_dist / mean_reward_dist
        Copied exactly from contrastive_repr_v1/losses.py::calibrate_beta
        """
        N = z.size(0)
        if N < 2:
            return 1.0

        with torch.no_grad():
            diffs = z.unsqueeze(0) - z.unsqueeze(1)
            d_embed = diffs.norm(dim=2)
            d_reward = (rewards.unsqueeze(0) - rewards.unsqueeze(1)).abs()

            mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
            mean_embed = d_embed[mask].mean().item()
            mean_reward = d_reward[mask].mean().item()

            if mean_reward < self.eps:
                return 1.0
            return mean_embed / mean_reward

    def forward(self, z: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Compute soft distance correlation loss with auto-calibrated beta.

        Args:
            z: embeddings (N, D)
            rewards: terminal rewards (N,)

        Returns:
            scalar loss
        """
        N = z.size(0)
        if N < 2:
            return torch.tensor(0.0, device=z.device)

        # Auto-calibrate beta each batch
        beta = self._calibrate_beta(z, rewards)

        # Pairwise embedding distances
        diffs = z.unsqueeze(0) - z.unsqueeze(1)  # (N, N, D)
        d_embed = diffs.norm(dim=2)  # (N, N)

        # Pairwise reward distances
        d_reward = (rewards.unsqueeze(0) - rewards.unsqueeze(1)).abs()  # (N, N)

        # Mask diagonal (self-pairs)
        mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

        # Loss: squared difference between predicted and target distances
        target = beta * d_reward
        loss = ((d_embed - target) ** 2)[mask].mean()

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Copied from dual_axis_repr_v2/losses.py.

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


class VICRegVarianceLoss(nn.Module):
    """VICReg variance term: penalize dimensions with std below target.

    Copied from dual_axis_repr_v2/losses.py.

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
