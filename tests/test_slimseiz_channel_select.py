"""Unit tests for SlimSeiz stage 1 (adaptive channel selection).

Synthetic, no CHB-MIT download -- same convention as test_tf_node_encoder.py
/ test_dense_edge_mamba.py. Checks: `select_slimseiz_channels` actually
recovers channels with real signal over decoy noise channels;
`SlimSeizClassifier.fit()` wires the selected subset into the stage-2
network's `input_channels` (verified via `conv1[0].in_channels`, not just
`selected_channel_idx_`, so a wiring regression in `_prepare_features`
would be caught even if selection itself still returned the right
indices); `predict_proba` accepts full-montage input at inference time
(the LOSO loop always passes the untouched fold test array -- slicing
must happen inside the classifier, not the caller); and
`select_channels=False` reproduces the pre-2026-08-25 full-montage
behavior unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("imblearn")

from Epilepsy.pipelines.slimseiz_channel_select import select_slimseiz_channels  # noqa: E402
from Epilepsy.pipelines.slimseiz_classifier import SlimSeizClassifier  # noqa: E402


def _synthetic_data(seed: int = 0, n_samples: int = 240, n_channels: int = 10, n_timepoints: int = 64):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    pos_idx = rng.choice(n_samples, size=n_samples // 8, replace=False)
    y[pos_idx] = 1
    # Only channels 2 and 6 actually separate the classes -- every other
    # channel is pure noise, so a correct selector must prefer these two.
    X[pos_idx, 2, :] += 4.0
    X[pos_idx, 6, :] += 4.0
    return X, y


def test_select_slimseiz_channels_recovers_informative_channels():
    X, y = _synthetic_data(seed=1)
    selected = select_slimseiz_channels(
        X, y, n_select=4, n_iterations=6, seed=42, verbose=0
    )
    assert selected.shape == (4,)
    assert list(selected) == sorted(selected.tolist())  # returned ascending
    assert 2 in selected and 6 in selected


def test_select_slimseiz_channels_noop_when_n_select_ge_n_channels():
    rng = np.random.RandomState(2)
    X = rng.randn(60, 4, 32).astype(np.float32)
    y = rng.randint(0, 2, size=60).astype(np.int64)
    selected = select_slimseiz_channels(X, y, n_select=8, n_iterations=2, seed=0)
    np.testing.assert_array_equal(selected, np.arange(4))


def test_slimseiz_classifier_wires_selected_channels_into_model():
    X, y = _synthetic_data(seed=3, n_channels=8)
    clf = SlimSeizClassifier(
        epochs=1,
        batch_size=16,
        validation_split=0.0,
        select_channels=True,
        n_select_channels=3,
        channel_select_iterations=2,
        device="cpu",
        verbose=0,
    )
    clf.fit(X, y)

    assert clf.selected_channel_idx_ is not None
    assert len(clf.selected_channel_idx_) == 3
    # The network must actually be built for the SELECTED channel count,
    # not the full montage -- this is the wiring _prepare_features has to
    # get right (fit() computes the subset, _prepare_features slices X
    # before _build_model_from_features ever sees it).
    assert clf.model_.conv1[0].in_channels == 3

    # predict_proba must accept the FULL montage (n_channels=8), exactly
    # what the LOSO loop passes -- slicing happens inside the classifier.
    X_test = np.random.RandomState(4).randn(20, 8, X.shape[2]).astype(np.float32)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (20, 2)
    assert np.all(np.isfinite(proba))


def test_slimseiz_classifier_select_channels_false_keeps_full_montage():
    X, y = _synthetic_data(seed=5, n_channels=7)
    clf = SlimSeizClassifier(
        epochs=1,
        batch_size=16,
        validation_split=0.0,
        select_channels=False,
        device="cpu",
        verbose=0,
    )
    clf.fit(X, y)
    assert clf.selected_channel_idx_ is None
    assert clf.model_.conv1[0].in_channels == 7
