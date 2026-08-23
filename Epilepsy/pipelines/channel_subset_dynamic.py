"""Per-window channel subset selection via cheap affinity (cosine).

Selection v1 picks k channels; live edges are the undirected clique among
those k (m = k*(k-1)/2). Pair order matches SparseEvidenceGNNCore's
registered src_idx/dst_idx (upper_pair_indices: i<j nested loop), NOT
ordered_pair_indices (directed i!=j). A mismatch would silently scramble
which full-E slot a live feature is written into.
"""
from __future__ import annotations

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


def canonical_undirected_pairs(n_channels: int) -> list[tuple[int, int]]:
    """(i, j) with i<j, same nested-loop order as upper_pair_indices and
    SparseEvidenceGNNCore's src_idx/dst_idx (cwt_gnn_classifiers.py)."""
    return [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]


def pair_to_edge_index(n_channels: int) -> dict[tuple[int, int], int]:
    """Map (i, j) with i<j to edge index e in [0, E), E = C*(C-1)/2."""
    return {(i, j): e for e, (i, j) in enumerate(canonical_undirected_pairs(n_channels))}


def live_edges_from_channel_subset(
    selected_channels: torch.Tensor,
    pair_index: dict[tuple[int, int], int],
) -> torch.Tensor:
    """Undirected clique on `selected_channels` as indices into the full-E
    layout defined by pair_index.

    selected_channels: (k,) int tensor of channel ids.
    Returns a LongTensor of edge indices, length m = k*(k-1)/2 (empty if k<2).
    """
    if selected_channels.dim() != 1:
        raise ValueError(
            "selected_channels must be 1D (k,), "
            f"got shape {tuple(selected_channels.shape)}"
        )
    ch = [int(c) for c in selected_channels.tolist()]
    live: list[int] = []
    for a in range(len(ch)):
        for b in range(a + 1, len(ch)):
            i, j = ch[a], ch[b]
            if i > j:
                i, j = j, i
            live.append(pair_index[(i, j)])
    if not live:
        return torch.empty(0, dtype=torch.long, device=selected_channels.device)
    return torch.tensor(live, dtype=torch.long, device=selected_channels.device)
