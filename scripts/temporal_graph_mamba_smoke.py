"""End-to-end smoke test for temporal_graph_mode="mamba" -- the
"aggregate-then-Mamba" swap for event_mode="temporal_graph"'s existing
per-node GRU (_temporal_graph_node_states). Construct, fit on tiny
synthetic data, predict_proba, for both "gru" (baseline, must still work
unchanged) and "mamba" (new) -- CPU-only, before touching run_pipelines.py.

Run: python scripts/temporal_graph_mamba_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Epilepsy.pipelines.cwt_gnn_classifiers import SparseEvidenceGNNClassifier  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

N_CHANNELS = 6
SAMPLING_RATE = 32
N_TIME = SAMPLING_RATE * 4  # 4s window
NFREQS = 4
N_TRIALS = 12

rng = np.random.default_rng(0)
X = (rng.standard_normal((N_TRIALS, N_CHANNELS, N_TIME)).astype(np.float32)) * 20.0
y = rng.integers(0, 2, size=N_TRIALS)


def _run(temporal_graph_mode: str) -> None:
    print(f"\n=== temporal_graph_mode={temporal_graph_mode!r} ===")
    clf = SparseEvidenceGNNClassifier(
        sampling_rate=SAMPLING_RATE,
        lowest=2.0,
        highest=14.0,
        nfreqs=NFREQS,
        event_mode="temporal_graph",
        event_aggregation="mean",
        coherence_threshold_mode="fixed",
        coherence_threshold=0.5,
        hidden_dim=8,
        temporal_graph_edge_dim=8,
        temporal_graph_mode=temporal_graph_mode,
        temporal_graph_mamba_d_state=8,
        temporal_graph_mamba_expand=2,
        temporal_graph_mamba_chunk_size=64,
        dense_edge_time_downsample=4,
        epochs=2,
        batch_size=4,
        validation_split=0.0,
        early_stopping_patience=None,
        cwt_backend="torch",
        device="cpu",
        verbose=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    print(f"fit+predict_proba OK, proba.shape={proba.shape}, "
          f"row sums ~1: {np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)}")
    n_params = sum(p.numel() for p in clf.model_.parameters())
    n_mamba = (
        sum(p.numel() for p in clf.model_.temporal_node_mamba.parameters())
        if clf.model_.temporal_node_mamba is not None
        else 0
    )
    n_gru = (
        sum(p.numel() for p in clf.model_.temporal_node_gru.parameters())
        if clf.model_.temporal_node_gru is not None
        else 0
    )
    print(f"total_params={n_params} temporal_node_mamba_params={n_mamba} "
          f"temporal_node_gru_params={n_gru}")


_run("gru")
_run("mamba")

# Reject the invalid combination explicitly, per __init__'s own validation.
print("\n=== rejecting temporal_graph_mode='mamba' with event_mode='dense' ===")
try:
    SparseEvidenceGNNClassifier(
        sampling_rate=SAMPLING_RATE, lowest=2.0, highest=14.0, nfreqs=NFREQS,
        event_mode="dense", temporal_graph_mode="mamba",
    )._build_model(n_channels=N_CHANNELS, n_classes=2)
    raise AssertionError("expected ValueError, none raised")
except ValueError as exc:
    print(f"raised ValueError as expected: {exc}")

print("\nAll smoke checks passed.")
