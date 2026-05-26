"""Loss functions for hand-identity representation learning.

Two formulations:
- TripletLoss: online triplet mining with hard negatives, supervised by hand label
- CrossEntropyHandLoss: direct classification of opponent hand (J/Q/K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Online triplet loss with hard negative mining, supervised by hand label.

    For each anchor in the batch:
      - Positive: a different sample with the same hand label
      - Hard negative: the sample with a different hand label whose embedding
        is closest to the anchor (hardest negative)

    Loss: mean(max(0, d(a, p) - d(a, n) + margin))

    Uses L2 distance between embeddings.

    Args:
        margin: triplet margin (default 1.0)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss.

        Args:
            embeddings: (N, D) embedding tensor
            labels: (N,) integer hand labels (0=J, 1=Q, 2=K)

        Returns:
            scalar triplet loss
        """
        N = embeddings.size(0)
        if N < 3:
            return torch.tensor(0.0, requires_grad=True)

        # Pairwise L2 distances: (N, N)
        diffs = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, D)
        dist_matrix = diffs.norm(dim=2)  # (N, N)

        # Label masks
        labels_row = labels.unsqueeze(1)  # (N, 1)
        labels_col = labels.unsqueeze(0)  # (1, N)
        same_label = (labels_row == labels_col)     # (N, N)
        diff_label = ~same_label                    # (N, N)

        # Exclude self from same_label
        eye = torch.eye(N, dtype=torch.bool, device=embeddings.device)
        same_label_no_self = same_label & ~eye      # (N, N)

        losses = []
        for i in range(N):
            # Find a positive (same label, not self)
            pos_mask = same_label_no_self[i]
            if not pos_mask.any():
                continue

            # Use mean distance to all positives (more stable than single hardest)
            d_pos = dist_matrix[i][pos_mask].mean()

            # Find the hardest negative (different label, closest embedding)
            neg_mask = diff_label[i]
            if not neg_mask.any():
                continue

            d_neg = dist_matrix[i][neg_mask].min()

            triplet_loss = F.relu(d_pos - d_neg + self.margin)
            losses.append(triplet_loss)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(losses).mean()


class CrossEntropyHandLoss(nn.Module):
    """Cross-entropy classification loss for opponent hand prediction.

    Takes embeddings and a linear classification head (8 -> 3) and computes
    the cross-entropy loss between predicted hand logits and true hand labels.

    Args:
        embedding_dim: dimensionality of encoder output (default 8)
        num_classes: number of hand classes (default 3: J, Q, K)
    """

    def __init__(self, embedding_dim: int = 8, num_classes: int = 3):
        super().__init__()
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            embeddings: (N, D) embedding tensor
            labels: (N,) integer hand labels (0=J, 1=Q, 2=K)

        Returns:
            scalar cross-entropy loss
        """
        logits = self.head(embeddings)  # (N, num_classes)
        return F.cross_entropy(logits, labels)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        with torch.no_grad():
            logits = self.head(embeddings)
            return logits.argmax(dim=1)
