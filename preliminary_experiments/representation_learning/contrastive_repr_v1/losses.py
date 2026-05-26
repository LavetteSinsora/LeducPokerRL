"""Loss functions for contrastive representation learning.

Three formulations:
- L0: TD(0) value prediction (control baseline)
- L1: Soft distance correlation (spring system)
- L2: Rank-N-Contrast (ranking-based contrastive)
Plus VICReg variance regularization for anti-collapse.
"""

import torch
import torch.nn as nn


def vicreg_variance_loss(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """VICReg variance term: penalize dimensions with std below target.

    Args:
        z: embeddings (N, D)
        target_std: minimum desired std per dimension

    Returns:
        scalar loss (mean over dimensions of max(0, target_std - std(z_k)))
    """
    std_per_dim = z.std(dim=0)
    return torch.relu(target_std - std_per_dim).mean()


def soft_distance_correlation_loss(z: torch.Tensor, rewards: torch.Tensor,
                                   beta: float = 1.0) -> torch.Tensor:
    """L1: Soft distance correlation.

    loss = mean( (||z_i - z_j|| - beta * |R_i - R_j|)^2 )

    Args:
        z: embeddings (N, D)
        rewards: terminal rewards (N,)
        beta: exchange rate between embedding and reward distances

    Returns:
        scalar loss
    """
    N = z.size(0)
    if N < 2:
        return torch.tensor(0.0)

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


def calibrate_beta(z: torch.Tensor, rewards: torch.Tensor) -> float:
    """Auto-calibrate beta so target distances match embedding scale.

    beta = mean_embed_dist / mean_reward_dist
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

        if mean_reward < 1e-8:
            return 1.0
        return mean_embed / mean_reward


def rank_n_contrast_loss(z: torch.Tensor, rewards: torch.Tensor,
                         temperature: float = 0.5) -> torch.Tensor:
    """L2: Rank-N-Contrast loss for continuous labels.

    For each anchor i and candidate j, the negative set S_{i,j} contains
    all samples k where |R_i - R_k| > |R_i - R_j|.

    loss = (1/2N) sum_i sum_{j!=i} -log[ exp(sim(i,j)/tau) / sum_{k in S_{i,j}} exp(sim(i,k)/tau) ]

    Based on: Rank-N-Contrast (NeurIPS 2023), arxiv.org/abs/2210.01189

    Args:
        z: embeddings (N, D)
        rewards: terminal rewards (N,)
        temperature: controls ranking sharpness

    Returns:
        scalar loss
    """
    N = z.size(0)
    if N < 3:
        return torch.tensor(0.0)

    # Pairwise similarities: sim(i,j) = -||z_i - z_j||^2
    diffs = z.unsqueeze(0) - z.unsqueeze(1)  # (N, N, D)
    sim_matrix = -(diffs ** 2).sum(dim=2)  # (N, N)

    # Pairwise reward distances
    reward_dists = (rewards.unsqueeze(0) - rewards.unsqueeze(1)).abs()  # (N, N)

    # Scale similarities by temperature
    sim_scaled = sim_matrix / temperature

    total_loss = torch.tensor(0.0, device=z.device)
    num_valid = 0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # S_{i,j} = {k : |R_i - R_k| > |R_i - R_j|, k != i}
            d_ij = reward_dists[i, j]
            neg_mask = (reward_dists[i] > d_ij)
            neg_mask[i] = False  # exclude self

            if neg_mask.sum() == 0:
                continue

            # log[ exp(sim(i,j)/tau) / sum_{k in S} exp(sim(i,k)/tau) ]
            numerator = sim_scaled[i, j]
            neg_sims = sim_scaled[i][neg_mask]
            # Include j in denominator for numerical stability
            log_sum_exp = torch.logsumexp(
                torch.cat([numerator.unsqueeze(0), neg_sims]), dim=0
            )
            total_loss -= (numerator - log_sum_exp)
            num_valid += 1

    if num_valid == 0:
        return torch.tensor(0.0, device=z.device)

    return total_loss / num_valid


def rank_n_contrast_loss_vectorized(z: torch.Tensor, rewards: torch.Tensor,
                                    temperature: float = 0.5) -> torch.Tensor:
    """Vectorized version of Rank-N-Contrast for larger batches.

    Same semantics as rank_n_contrast_loss but avoids Python loops.
    """
    N = z.size(0)
    if N < 3:
        return torch.tensor(0.0)

    # Pairwise similarities: sim(i,j) = -||z_i - z_j||^2
    diffs = z.unsqueeze(0) - z.unsqueeze(1)  # (N, N, D)
    sim_matrix = -(diffs ** 2).sum(dim=2) / temperature  # (N, N)

    # Pairwise reward distances
    reward_dists = (rewards.unsqueeze(0) - rewards.unsqueeze(1)).abs()  # (N, N)

    # For each (i,j), we need: S_{i,j} = {k : reward_dists[i,k] > reward_dists[i,j]}
    # neg_mask[i,j,k] = True if k is a valid negative for the (i,j) pair
    # Shape: (N, N, N)
    d_ij = reward_dists.unsqueeze(2)  # (N, N, 1)
    d_ik = reward_dists.unsqueeze(1).expand(N, N, N)  # (N, 1, N) -> (N, N, N)
    neg_mask = d_ik > d_ij  # (N, N, N)

    # Exclude self (k == i)
    eye_mask = torch.eye(N, dtype=torch.bool, device=z.device)
    neg_mask &= ~eye_mask.unsqueeze(1).expand(N, N, N)

    # Exclude diagonal pairs (i == j)
    pair_mask = ~eye_mask  # (N, N)

    # Check which (i,j) pairs have at least one negative
    has_negatives = neg_mask.any(dim=2) & pair_mask  # (N, N)

    if not has_negatives.any():
        return torch.tensor(0.0, device=z.device)

    # For valid (i,j) pairs, compute the loss
    # We need logsumexp over {j} union S_{i,j} for each (i,j)
    # Use masking: set invalid positions to -inf before logsumexp

    # Expand sim_matrix for the k dimension
    sim_expanded = sim_matrix.unsqueeze(1).expand(N, N, N)  # sim[i,k] for each (i,j,k)

    # Include j itself in the denominator: mask is neg_mask OR (k == j)
    j_indices = torch.arange(N, device=z.device).unsqueeze(0).unsqueeze(2).expand(N, N, N)
    k_indices = torch.arange(N, device=z.device).unsqueeze(0).unsqueeze(0).expand(N, N, N)
    include_j = (k_indices == j_indices)  # (N, N, N)

    denom_mask = neg_mask | include_j  # (N, N, N)
    denom_mask &= ~eye_mask.unsqueeze(1).expand(N, N, N)  # exclude k==i

    # Set masked-out positions to -inf
    denom_sims = sim_expanded.clone()
    denom_sims[~denom_mask] = float('-inf')

    # logsumexp over k dimension
    log_denom = torch.logsumexp(denom_sims, dim=2)  # (N, N)

    # Numerator is just sim[i, j]
    log_num = sim_matrix  # (N, N)

    # Loss for each valid (i, j)
    per_pair_loss = -(log_num - log_denom)

    # Mask and average
    valid_loss = per_pair_loss[has_negatives]

    return valid_loss.mean()
