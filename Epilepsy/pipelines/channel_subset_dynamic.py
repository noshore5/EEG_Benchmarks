"""Per-window channel subset selection via cheap affinity (cosine)."""
from __future__ import annotations

import numpy as np
import torch


def absolute_cosine_affinity(x: torch.Tensor) -> torch.Tensor:
    """
    x: (C, T) or (B, C, T) float tensor.
    Returns abs cosine similarity matrix (C, C) or (B, C, C).
    """
    # L2-normalize along time
    x = x - x.mean(dim=-1, keepdim=True)
    norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    x = x / norms
    if x.dim() == 2:
        sim = x @ x.T
    else:
        sim = torch.matmul(x, x.transpose(-1, -2))
    return sim.abs()


def top_k_channels_from_affinity(
    affinity: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    affinity: (C, C) or (B, C, C)
    Returns channel indices (k,) or (B, k), sorted ascending for stable indexing.
    Ranking score = mean affinity to others (exclude self).
    """
    if affinity.dim() == 2:
        C = affinity.shape[0]
        if k >= C:
            return torch.arange(C, device=affinity.device)
        # zero diagonal so self-similarity doesn't dominate
        score = affinity.sum(dim=-1) - affinity.diag()
        score = score / max(C - 1, 1)
        top = torch.topk(score, k=k, largest=True).indices
        return top.sort().values

    B, C, _ = affinity.shape
    if k >= C:
        return torch.arange(C, device=affinity.device).expand(B, -1)
    diag = torch.diagonal(affinity, dim1=-2, dim2=-1)
    score = (affinity.sum(dim=-1) - diag) / max(C - 1, 1)
    top = torch.topk(score, k=k, largest=True, dim=-1).indices
    return top.sort(dim=-1).values


def select_channel_subset(
    x_window: torch.Tensor,
    k: int,
    metric: str = "abs_cosine",
) -> torch.Tensor:
    """
    x_window: (C, T) or (B, C, T)
    Returns indices into the channel axis.
    """
    if metric != "abs_cosine":
        raise ValueError(f"Unsupported metric: {metric!r}")
    aff = absolute_cosine_affinity(x_window)
    return top_k_channels_from_affinity(aff, k)