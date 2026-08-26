"""Unit tests for CGMambaNetClassifier (cg_mambanet_classifier.py).

Synthetic, no CHB-MIT download -- same convention as
test_slimseiz_channel_select.py / test_dense_edge_mamba.py. Checks:
`_build_model_from_features` wires the model to the right channel/patch
counts; a `patch_size` that doesn't evenly divide the window length raises
(same convention as DBConformer's own patch_size check); the fixed-channel
slice (mirroring SlimSeiz's `channel_select_fixed_indices`) actually reduces
the GCN's node count, not just `X`'s shape; and `predict_proba` accepts the
FULL, unsliced montage at inference time (the LOSO loop always passes the
untouched fold test array -- slicing must happen inside the classifier).
Tiny hyperparameters throughout (mamba_n_layers=2, not the paper's 12;
patch_size=8) purely for test speed -- not representative of a real run's
config (see CG_MAMBANET_PARAMS in run_pipelines.py for that).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("mambapy")

from Epilepsy.pipelines.cg_mambanet_classifier import CGMambaNetClassifier  # noqa: E402


def _synthetic_data(seed: int = 0, n_samples: int = 24, n_channels: int = 16, n_timepoints: int = 32):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    y = rng.randint(0, 2, size=n_samples).astype(np.int64)
    return X, y


_TINY_KWARGS = dict(
    patch_size=8,
    mamba_n_layers=2,
    mamba_d_state=8,
    epochs=1,
    batch_size=8,
    validation_split=0.0,
    device="cpu",
    verbose=0,
)


def test_cg_mambanet_classifier_wires_full_montage():
    X, y = _synthetic_data(seed=1, n_channels=16, n_timepoints=32)
    clf = CGMambaNetClassifier(**_TINY_KWARGS)
    clf.fit(X, y)

    assert clf.model_.n_channels == 16
    assert clf.model_.patch_size == 8
    assert clf.model_.n_patches == 4  # 32 / 8
    assert clf.model_.gcn.n_nodes == 16

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(proba))


def test_cg_mambanet_classifier_wires_fixed_channel_slice():
    X, y = _synthetic_data(seed=2, n_channels=20, n_timepoints=32)
    fixed_indices = [1, 3, 5, 7, 9, 11]
    clf = CGMambaNetClassifier(channel_select_fixed_indices=fixed_indices, **_TINY_KWARGS)
    clf.fit(X, y)

    # The GCN's node count must reflect the SLICED channel count, not the
    # full montage -- this is the wiring _prepare_features has to get right
    # (fixed_indices slices X before _build_model_from_features ever sees it).
    assert clf.model_.n_channels == len(fixed_indices)
    assert clf.model_.gcn.n_nodes == len(fixed_indices)

    # predict_proba must accept the FULL montage (n_channels=20), exactly
    # what the LOSO loop passes -- slicing happens inside the classifier.
    X_test = np.random.RandomState(3).randn(10, 20, X.shape[2]).astype(np.float32)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (10, 2)
    assert np.all(np.isfinite(proba))


def test_cg_mambanet_patch_size_must_divide_window_length():
    X, y = _synthetic_data(seed=4, n_channels=16, n_timepoints=30)  # 30 not divisible by 8
    clf = CGMambaNetClassifier(**_TINY_KWARGS)
    with pytest.raises(ValueError, match="patch_size"):
        clf.fit(X, y)
