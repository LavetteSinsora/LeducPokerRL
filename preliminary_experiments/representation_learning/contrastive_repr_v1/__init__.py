"""Contrastive State Representation Learning v1.

Phase 1: representation learning only (no downstream action selection).
Three formulations: L0 (TD control), L1 (distance correlation), L2 (Rank-N-Contrast).
"""

from experiments.representation_learning.contrastive_repr_v1.agent import ContrastiveReprAgent
