"""Unit tests for the NonStGM nonstationary covariance/precision graph
pipeline (Epilepsy/pipelines/nonstgm_graph.py, nonstgm_classifier.py).

Synthetic data only, no CHB-MIT download and no dataset/AWS dependency --
same convention as test_dense_edge_mamba.py / test_tf_node_encoder.py.
Covers the math/leakage/numerical-stability checks from the implementation
spec (symmetry, positive-definiteness after regularization, exact edge
count C(C-1)/2, no NaN/Inf on well-conditioned input, loud diagnostics on a
deliberately singular input, train-only normalization stats, and both
temporal backends' shape contracts).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Epilepsy.pipelines.nonstgm_graph import (  # noqa: E402
    LocalCovariancePrecisionGraph,
    _local_covariance,
    _regularized_precision,
    NonStGMDiagnostics,
)
from Epilepsy.pipelines.nonstgm_classifier import NonStGMClassifier  # noqa: E402

N_CHANNELS = 6
N_TIME = 256
BATCH = 4


def _synthetic_window(seed: int = 0, n_channels: int = N_CHANNELS, n_time: int = N_TIME) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, n_channels, n_time, generator=g)


def test_covariance_symmetry():
    x = _synthetic_window()
    sigma = _local_covariance(x)
    assert torch.allclose(sigma, sigma.transpose(-1, -2), atol=1e-5)


def test_regularized_covariance_is_positive_definite():
    x = _synthetic_window()
    sigma = _local_covariance(x)
    diagnostics = NonStGMDiagnostics()
    lam = 1e-2
    sigma_reg = sigma + lam * torch.eye(sigma.shape[-1])
    chol, info = torch.linalg.cholesky_ex(sigma_reg)
    assert bool((info == 0).all()), "regularized covariance must be positive definite"
    eigvals = torch.linalg.eigvalsh(sigma_reg)
    assert bool((eigvals > 0).all())


def test_precision_symmetry():
    x = _synthetic_window()
    sigma = _local_covariance(x)
    diagnostics = NonStGMDiagnostics()
    theta = _regularized_precision(sigma, base_lambda=1e-2, adaptive=False, diagnostics=diagnostics)
    assert torch.allclose(theta, theta.transpose(-1, -2), atol=1e-4)


@pytest.mark.parametrize("representation", ["covariance", "precision", "partial_corr", "both"])
def test_edge_count_matches_c_choose_2(representation):
    graph = LocalCovariancePrecisionGraph(representation=representation, temporal_segments=3, overlap=0.5)
    x = _synthetic_window()
    edges, diagnostics = graph(x)
    expected_edges = N_CHANNELS * (N_CHANNELS - 1) // 2
    assert edges.shape[2] == expected_edges
    assert edges.shape[0] == BATCH
    assert edges.shape[1] == graph.n_edge_features


def test_partial_correlation_range():
    graph = LocalCovariancePrecisionGraph(representation="partial_corr", temporal_segments=2, overlap=0.0)
    x = _synthetic_window()
    edges, _ = graph(x)
    assert torch.all(edges >= -1.0 - 1e-6)
    assert torch.all(edges <= 1.0 + 1e-6)


def test_diagonal_never_emitted_as_edge():
    graph = LocalCovariancePrecisionGraph(representation="precision", temporal_segments=1, static=True)
    x = _synthetic_window()
    edges, _ = graph(x)
    # E = C(C-1)/2 strictly excludes the C diagonal entries -- checked
    # directly against the channel count rather than just re-deriving the
    # same formula the implementation uses.
    assert edges.shape[2] == 15  # C(C-1)/2 for C=6


def test_no_nan_or_inf_on_well_conditioned_input():
    graph = LocalCovariancePrecisionGraph(representation="both", temporal_segments=6, overlap=0.5, regularization=1e-2)
    x = _synthetic_window(seed=1)
    edges, diagnostics = graph(x)
    assert torch.isfinite(edges).all()
    assert diagnostics["n_nan_or_inf"] == 0
    assert diagnostics["n_cholesky_failures"] == 0


def test_singular_input_is_caught_not_silently_nan():
    # Duplicate one channel exactly -> covariance is rank-deficient
    # (singular) before regularization. With adaptive_regularization off
    # and a tiny fixed lambda, this should still produce finite output
    # (ridge-regularized inverse always applied) and the diagnostics must
    # reflect a degenerate eigenspectrum, not silently look healthy.
    x = _synthetic_window(seed=2)
    x = x.clone()
    x[:, 1, :] = x[:, 0, :]  # channel 1 is an exact copy of channel 0
    graph = LocalCovariancePrecisionGraph(representation="precision", temporal_segments=1, static=True, regularization=1e-3)
    edges, diagnostics = graph(x)
    assert torch.isfinite(edges).all(), "regularized inverse must never propagate NaN/Inf"
    assert diagnostics["min_eigenvalue"] < 1e-4, "diagnostics must show the near-zero eigenvalue, not hide it"


def test_adaptive_regularization_records_effective_lambda():
    x = _synthetic_window(seed=3)
    x = x.clone()
    x[:, 1, :] = x[:, 0, :]
    graph = LocalCovariancePrecisionGraph(
        representation="precision", temporal_segments=1, static=True,
        regularization=1e-8, adaptive_regularization=True,
    )
    edges, diagnostics = graph(x)
    assert torch.isfinite(edges).all()
    assert diagnostics["mean_effective_lambda"] is not None


def test_static_collapses_to_one_segment():
    graph = LocalCovariancePrecisionGraph(representation="precision", static=True, temporal_segments=6)
    x = _synthetic_window()
    edges, _ = graph(x)
    assert edges.shape[-1] == 1


def test_time_varying_produces_multiple_segments():
    graph = LocalCovariancePrecisionGraph(representation="precision", static=False, temporal_segments=6, overlap=0.5)
    x = _synthetic_window()
    edges, _ = graph(x)
    assert edges.shape[-1] > 1


def test_frequency_band_not_implemented():
    with pytest.raises(NotImplementedError):
        LocalCovariancePrecisionGraph(frequency_band=(1.0, 4.0))


def test_unregularized_lambda_rejected():
    with pytest.raises(ValueError):
        LocalCovariancePrecisionGraph(regularization=0.0)


def test_classifier_normalization_is_train_only_no_leakage():
    n = 40
    X = np.random.RandomState(0).randn(n, N_CHANNELS, 64).astype(np.float32)
    y = np.array([0, 1] * (n // 2))
    train_idx = np.arange(0, 30)
    clf = NonStGMClassifier(
        representation="precision", temporal_segments=2, epochs=1, batch_size=8,
        validation_split=0.0, device="cpu", temporal_backend="gru",
    )
    clf._prepare_features(X, fit=True, train_idx=train_idx)
    expected_mean = float(X[train_idx].mean())
    expected_std = float(X[train_idx].std())
    assert clf.X_mean_ == pytest.approx(expected_mean)
    assert clf.X_std_ == pytest.approx(expected_std, rel=1e-6) or clf.X_std_ == pytest.approx(1.0)

    # Mutating the held-out rows must not change the already-fit stats.
    X_mutated = X.copy()
    X_mutated[30:] = 1e6
    clf2 = NonStGMClassifier(temporal_segments=2, epochs=1, device="cpu")
    clf2._prepare_features(X, fit=True, train_idx=train_idx)
    clf3 = NonStGMClassifier(temporal_segments=2, epochs=1, device="cpu")
    clf3._prepare_features(X_mutated, fit=True, train_idx=train_idx)
    assert clf2.X_mean_ == pytest.approx(clf3.X_mean_)
    assert clf2.X_std_ == pytest.approx(clf3.X_std_)


def test_classifier_fit_predict_proba_gru_smoke():
    n = 24
    X = np.random.RandomState(1).randn(n, N_CHANNELS, 64).astype(np.float32)
    y = np.array([0, 1] * (n // 2))
    clf = NonStGMClassifier(
        representation="precision", temporal_segments=2, overlap=0.0,
        epochs=1, batch_size=8, validation_split=0.0, device="cpu",
        temporal_backend="gru", temporal_hidden=4, head_hidden=4,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.all(np.isfinite(proba))
    assert clf.diagnostics_  # numerical diagnostics surfaced after predict_proba


def test_classifier_fit_predict_proba_mamba_smoke():
    pytest.importorskip("mambapy")
    n = 24
    X = np.random.RandomState(2).randn(n, N_CHANNELS, 64).astype(np.float32)
    y = np.array([0, 1] * (n // 2))
    clf = NonStGMClassifier(
        representation="both", temporal_segments=2, overlap=0.0,
        epochs=1, batch_size=8, validation_split=0.0, device="cpu",
        temporal_backend="mamba", temporal_hidden=4, head_hidden=4,
        mamba_d_model=4, mamba_d_state=4,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.all(np.isfinite(proba))
