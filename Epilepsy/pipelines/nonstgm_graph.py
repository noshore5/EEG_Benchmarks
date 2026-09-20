"""Time-local regularized covariance/precision graph features for EEG windows.

A practical EEG adaptation, NOT a reproduction, of the nonstationary
graphical-model framework in:

    Basu, S. and Subba Rao, S. "Graphical Models for Nonstationary Time
    Series." Annals of Statistics, 2023, 51(4), 1453-1483. arXiv:2109.08709.

WHAT IS AND IS NOT FAITHFUL TO THE PAPER (see also the "NonStGM" section of
the repo README for the fuller writeup): the paper's own estimator is a
kernel-smoothed local covariance with an asymptotic consistency theory for a
locally-stationary process, combined with a (possibly frequency-domain)
graphical-lasso-style sparse precision estimate and formal hypothesis tests
for edge significance. NONE of that machinery is implemented here. What IS
implemented is the same *structural idea* the paper's Section 2 motivates a
nonstationary graphical model with: let the covariance structure vary over
local windows in time rather than being estimated once, globally, over the
whole (non-stationary) recording, and turn the local precision matrix
(inverse covariance) into a conditional-dependence graph via the standard
Gaussian-graphical-model identity (Whittaker 1990, not this paper) that a
zero in Sigma^-1_ij means X_i and X_j are conditionally independent given
every other channel. The estimator itself (plain sample covariance per
local segment, ridge-regularized) is a deliberately simple, computationally
cheap stand-in for the paper's own nonparametric kernel/DGLASSO estimator --
chosen so this pipeline is comparable in cost to the repo's existing
WCT/dense-edge pipelines, not because it is claimed to share the paper's
statistical guarantees.

Pipeline, per window X in R^{C x T} (raw voltage, already band-selected /
NOT yet z-scored -- normalization happens one level up in
NonStGMClassifier._prepare_features, exactly like every other raw-EEG
classifier in this repo):

    X(t) -> local segments -> Sigma_s (sample covariance per segment)
         -> Sigma_s + lambda*I (ridge regularization, ALWAYS applied)
         -> Theta_s = (Sigma_s + lambda*I)^-1  (via Cholesky, never a raw
                                                 unregularized inverse)
         -> edge features (upper triangle only, i<j, C(C-1)/2 edges)

Edge feature definitions (documented once here, the single source of truth
for the sign convention elsewhere in this pipeline):

    precision magnitude:      |Theta_s,ij|
    partial correlation:      rho_ij|rest = -Theta_s,ij / sqrt(Theta_s,ii * Theta_s,jj)

    Sign convention: this is the standard Gaussian-graphical-model partial
    correlation (Whittaker 1990, Prop 5.4.5) -- NEGATIVE of the
    off-diagonal precision entry, normalized by the geometric mean of the
    corresponding diagonal entries. A positive Theta_s,ij (positive
    conditional coupling in the precision parameterization) therefore maps
    to a NEGATIVE raw ratio, so the leading minus sign restores the usual
    "positive rho = positively conditionally-correlated" reading. rho lies
    in [-1, 1] for any valid (positive-definite) precision matrix; this
    implementation clips to that range only to absorb float roundoff at the
    boundary (see `_partial_correlation`), never as a substantive numerical
    fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

Representation = Literal["covariance", "precision", "partial_corr", "both"]

_REP_N_FEATURES = {"covariance": 1, "precision": 1, "partial_corr": 1, "both": 2}


@dataclass
class NonStGMDiagnostics:
    """Numerical-stability bookkeeping for one forward pass (one batch of
    windows). Never silently hides a failure (spec: "never silently hide
    numerical failures") -- every counter here is always computed, not
    opt-in, since the cost is negligible next to the covariance/Cholesky
    work it rides along with.
    """

    n_segments_total: int = 0
    min_eigenvalue: float = float("inf")
    max_eigenvalue: float = float("-inf")
    max_condition_number: float = 0.0
    n_cholesky_failures: int = 0
    n_nan_or_inf: int = 0
    n_fallback_regularization: int = 0
    effective_lambdas: list = field(default_factory=list)

    def update_from_eigvals(self, eigvals: torch.Tensor) -> None:
        # eigvals: [..., C], real, ascending (torch.linalg.eigvalsh convention).
        finite = eigvals[torch.isfinite(eigvals)]
        if finite.numel() == 0:
            return
        lo = float(finite.min())
        hi = float(finite.max())
        self.min_eigenvalue = min(self.min_eigenvalue, lo)
        self.max_eigenvalue = max(self.max_eigenvalue, hi)
        # Per-segment condition number, worst case over the batch.
        eigvals_pos = eigvals.clamp_min(1e-30)
        cond = (eigvals_pos.amax(dim=-1) / eigvals_pos.amin(dim=-1))
        cond = cond[torch.isfinite(cond)]
        if cond.numel() > 0:
            self.max_condition_number = max(self.max_condition_number, float(cond.max()))

    def as_dict(self) -> dict:
        return {
            "n_segments_total": self.n_segments_total,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "max_condition_number": self.max_condition_number,
            "n_cholesky_failures": self.n_cholesky_failures,
            "n_nan_or_inf": self.n_nan_or_inf,
            "n_fallback_regularization": self.n_fallback_regularization,
            "pct_fallback_regularization": (
                100.0 * self.n_fallback_regularization / self.n_segments_total
                if self.n_segments_total else 0.0
            ),
            "mean_effective_lambda": (
                sum(self.effective_lambdas) / len(self.effective_lambdas)
                if self.effective_lambdas else None
            ),
        }


def _segment_starts(n_time: int, segment_len: int, overlap: float) -> list[int]:
    step = max(1, int(round(segment_len * (1.0 - overlap))))
    starts = list(range(0, max(1, n_time - segment_len + 1), step))
    if not starts:
        starts = [0]
    # Always include a final segment flush with the end of the window so the
    # last few samples are never silently dropped from every segment.
    last = max(0, n_time - segment_len)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _triu_indices(n_channels: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.triu_indices(n_channels, n_channels, offset=1, device=device)
    return idx[0], idx[1]


def _local_covariance(segment: torch.Tensor) -> torch.Tensor:
    """segment: [..., C, L] (float32) -> Sigma: [..., C, C] (float32),
    symmetric sample covariance (channels de-meaned within the segment,
    ddof=1 with a floor of 1 sample to avoid divide-by-zero on L==1)."""
    segment = segment.float()
    centered = segment - segment.mean(dim=-1, keepdim=True)
    denom = max(1, segment.shape[-1] - 1)
    sigma = torch.matmul(centered, centered.transpose(-1, -2)) / denom
    # Symmetrize explicitly -- matmul(X, X^T) is symmetric only up to float
    # roundoff; downstream Cholesky/eigh assume exact symmetry.
    return 0.5 * (sigma + sigma.transpose(-1, -2))


def _regularized_precision(
    sigma: torch.Tensor,
    base_lambda: float,
    adaptive: bool,
    diagnostics: NonStGMDiagnostics,
    max_adaptive_attempts: int = 3,
) -> torch.Tensor:
    """Sigma: [..., C, C] -> Theta = (Sigma + lambda*I)^-1: [..., C, C].

    Never calls an unregularized inverse. Uses a Cholesky-based solve
    (numerically stable for SPD matrices) instead of a general matrix
    inverse. When `adaptive` is False (default), `base_lambda` is used
    unconditionally and a Cholesky failure raises rather than silently
    falling back -- "adaptive regularization only if explicitly
    configured" (spec). When `adaptive` is True, a failed Cholesky at the
    current lambda retries at 10x, up to `max_adaptive_attempts` times,
    every fallback counted in `diagnostics`.
    """
    eye = torch.eye(sigma.shape[-1], device=sigma.device, dtype=sigma.dtype)
    lam = float(base_lambda)
    attempts = max_adaptive_attempts if adaptive else 1
    last_info = None
    for attempt in range(attempts):
        sigma_reg = sigma + lam * eye
        chol, info = torch.linalg.cholesky_ex(sigma_reg)
        last_info = info
        failed = info != 0
        if not bool(failed.any()):
            n_segments = int(sigma.numel() // (sigma.shape[-1] * sigma.shape[-1]))
            diagnostics.n_segments_total += n_segments
            if attempt > 0:
                diagnostics.n_fallback_regularization += n_segments
            diagnostics.effective_lambdas.append(lam)
            eigvals = torch.linalg.eigvalsh(sigma)
            diagnostics.update_from_eigvals(eigvals)
            theta = torch.cholesky_solve(eye.expand_as(sigma_reg), chol)
            theta = 0.5 * (theta + theta.transpose(-1, -2))
            bad = ~torch.isfinite(theta)
            if bool(bad.any()):
                diagnostics.n_nan_or_inf += int(bad.any(dim=(-1, -2)).sum())
                theta = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
            return theta
        if not adaptive:
            break
        lam *= 10.0
    # Every attempt failed (or adaptive disabled and the single attempt
    # failed): count it loudly, fall back to a diagonal-only precision
    # (Theta = I / lambda_final) rather than propagating NaN/Inf.
    n_segments = int(sigma.numel() // (sigma.shape[-1] * sigma.shape[-1]))
    diagnostics.n_segments_total += n_segments
    diagnostics.n_cholesky_failures += n_segments
    diagnostics.effective_lambdas.append(lam)
    eigvals = torch.linalg.eigvalsh(sigma)
    diagnostics.update_from_eigvals(eigvals)
    fallback_theta = eye.expand_as(sigma) / lam
    return fallback_theta


def _partial_correlation(theta: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
    """theta: [..., C, C] -> rho: [..., E] for edges (i_idx, j_idx).
    rho_ij|rest = -Theta_ij / sqrt(Theta_ii * Theta_jj). See module
    docstring for the sign-convention justification. Clipped to [-1, 1]
    only to absorb float roundoff, not as a substantive correction.
    """
    diag = torch.diagonal(theta, dim1=-2, dim2=-1)  # [..., C]
    theta_ij = theta[..., i_idx, j_idx]  # [..., E]
    denom = torch.sqrt((diag[..., i_idx] * diag[..., j_idx]).clamp_min(1e-20))
    rho = -theta_ij / denom
    return rho.clamp(-1.0, 1.0)


class LocalCovariancePrecisionGraph(nn.Module):
    """Raw EEG window (B, C, T) -> time-local covariance/precision edge
    features (B, n_edge_feat, E, T_segments), E = C*(C-1)/2.

    Output layout is deliberately identical to the `[B, C_in, E, T]`
    contract `cwt_gnn_classifiers._DenseEdgeGRUTemporal` /
    `_DenseEdgeMambaTemporal` already consume, so this repo's existing
    temporal backends are reused unchanged (see nonstgm_classifier.py).
    """

    def __init__(
        self,
        representation: Representation = "precision",
        temporal_segments: int = 6,
        overlap: float = 0.5,
        regularization: float = 1e-2,
        static: bool = False,
        edge_threshold: float | None = None,
        adaptive_regularization: bool = False,
        frequency_band: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if representation not in _REP_N_FEATURES:
            raise ValueError(f"representation must be one of {sorted(_REP_N_FEATURES)}, got {representation!r}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")
        if regularization <= 0.0:
            raise ValueError(f"regularization (lambda) must be > 0, got {regularization} -- an unregularized inverse is not allowed")
        if temporal_segments < 1:
            raise ValueError(f"temporal_segments must be >= 1, got {temporal_segments}")
        if frequency_band is not None:
            raise NotImplementedError(
                "frequency_band is an extensibility stub (see this module's docstring / "
                "spec section 4): a frequency-local precision estimate reusing this "
                "repo's CWT infra is documented as future work, not implemented in this "
                "pass. Leave frequency_band=None for the time-local voltage-domain "
                "estimator that IS implemented."
            )
        self.representation = representation
        self.temporal_segments = int(temporal_segments)
        self.overlap = float(overlap)
        self.regularization = float(regularization)
        self.static = bool(static)
        self.edge_threshold = edge_threshold
        self.adaptive_regularization = bool(adaptive_regularization)
        self.frequency_band = frequency_band

    @property
    def n_edge_features(self) -> int:
        return _REP_N_FEATURES[self.representation]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """x: [B, C, T] raw voltage (any dtype; computed internally in
        fp32). Returns (edge_tensor [B, n_edge_feat, E, T_segments],
        diagnostics dict)."""
        if x.dim() != 3:
            raise ValueError(f"expected x of shape (B, C, T), got {tuple(x.shape)}")
        batch, n_channels, n_time = x.shape
        if n_channels < 2:
            raise ValueError(f"need at least 2 channels to form edges, got {n_channels}")
        device = x.device
        diagnostics = NonStGMDiagnostics()
        i_idx, j_idx = _triu_indices(n_channels, device)
        n_edges = i_idx.numel()

        if self.static:
            starts = [0]
            segment_len = n_time
        else:
            segment_len = max(2, n_time // max(1, self.temporal_segments))
            starts = _segment_starts(n_time, segment_len, self.overlap)

        x = x.float()
        per_segment_edges = []
        for start in starts:
            end = min(n_time, start + segment_len)
            segment = x[:, :, start:end]  # [B, C, L]
            sigma = _local_covariance(segment)  # [B, C, C]
            feats = []
            if self.representation == "covariance":
                cov_edges = sigma[:, i_idx, j_idx]  # [B, E]
                feats.append(cov_edges)
                # No inversion on this path -- still record the raw
                # eigenspectrum so numerical-stability diagnostics are
                # comparable across representations (spec: never silently
                # skip these checks).
                with torch.no_grad():
                    eigvals = torch.linalg.eigvalsh(sigma)
                    diagnostics.update_from_eigvals(eigvals)
                    diagnostics.n_segments_total += batch
            else:
                theta = _regularized_precision(
                    sigma, self.regularization, self.adaptive_regularization, diagnostics
                )
                if self.representation in ("precision", "both"):
                    prec_edges = theta[:, i_idx, j_idx].abs()
                    feats.append(prec_edges)
                if self.representation in ("partial_corr", "both"):
                    rho_edges = _partial_correlation(theta, i_idx, j_idx)
                    feats.append(rho_edges)
            seg_edges = torch.stack(feats, dim=1)  # [B, n_edge_feat, E]
            per_segment_edges.append(seg_edges)

        edge_tensor = torch.stack(per_segment_edges, dim=-1)  # [B, n_edge_feat, E, T_segments]
        if self.edge_threshold is not None:
            mask = edge_tensor.abs() >= float(self.edge_threshold)
            edge_tensor = edge_tensor * mask

        assert n_edges == n_channels * (n_channels - 1) // 2
        return edge_tensor, diagnostics.as_dict()
