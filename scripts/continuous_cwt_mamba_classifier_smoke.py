"""End-to-end smoke test for ContinuousCWTMambaClassifier: construct, fit
on a few tiny synthetic recordings, predict_proba -- the "does the whole
wired-together pipeline actually run" check, CPU-only, before touching
run_pipelines.py at all (same discipline as this paradigm's other pieces).

Run: python scripts/continuous_cwt_mamba_classifier_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Epilepsy.pipelines.continuous_cwt_mamba_classifier import ContinuousCWTMambaClassifier  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

N_CHANNELS = 4
SAMPLING_RATE = 32
NFREQS = 6


def _make_recording(n_total: int, n_windows: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    raw_x = (rng.standard_normal((N_CHANNELS, n_total)).astype(np.float32)) * 20.0
    window_len_samples = 64
    step = max(1, (n_total - window_len_samples) // max(1, n_windows - 1)) if n_windows > 1 else 0
    windows = []
    for i in range(n_windows):
        start = min(i * step, n_total - window_len_samples)
        windows.append(
            {
                "start_sample": start,
                "end_sample": start + window_len_samples,
                "window_start": start / SAMPLING_RATE,
                "window_end": (start + window_len_samples) / SAMPLING_RATE,
                "label": int(rng.integers(0, 2)),
            }
        )
    return {
        "raw_x": raw_x,
        "sampling_rate": float(SAMPLING_RATE),
        "subject": 1,
        "session": "0",
        "run": f"r{seed}",
        "run_start_time": None,
        "windows": windows,
    }


train_recordings = [_make_recording(n_total=SAMPLING_RATE * 30, n_windows=6, seed=i) for i in range(4)]
test_recordings = [_make_recording(n_total=SAMPLING_RATE * 30, n_windows=6, seed=100)]

clf = ContinuousCWTMambaClassifier(
    sampling_rate=SAMPLING_RATE,
    lowest=2.0,
    highest=14.0,
    nfreqs=NFREQS,
    coherence_threshold_mode="fixed",
    coherence_threshold=0.5,
    dense_edge_time_downsample=1,
    time_averaged_graph=False,
    cwt_backend="torch",
    device="cpu",
    epochs=2,
    learning_rate=1e-3,
    validation_split=0.25,
    early_stopping_patience=None,
    t_chunk=16,
    mamba_d_model=8,
    mamba_d_state=8,
    mamba_expand=2,
    mamba_n_layers=1,
    verbose=1,
)

print("Fitting...")
clf.fit(train_recordings)
print("Predicting...")
probs = clf.predict_proba(test_recordings)
assert len(probs) == 1
p = probs[0]
assert p.shape == (6, 2), p.shape
assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4), "softmax rows must sum to 1"
print(f"predict_proba shape={p.shape}, row sums~1: OK")
print("\nSMOKE TEST PASSED -- ContinuousCWTMambaClassifier fits and predicts end to end.")
