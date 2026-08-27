"""End-to-end smoke test for --pipeline hermitian_ssm, CPU-only, synthetic
recordings -- before touching run_pipelines.py.

Builds a handful of fake ``get_continuous_data()``-shaped recording dicts,
runs the deterministic spectral precompute (no disk cache), fits
``HermitianSSMClassifier`` for a couple of epochs, and predicts. Verifies
shapes and that the whole encoder -> Mamba -> head chain runs forward and
backward.

Run:  python scripts/hermitian_ssm_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Epilepsy.pipelines.hermitian_ssm_cache import HermitianSpectralConfig  # noqa: E402
from Epilepsy.pipelines.hermitian_ssm_classifier import HermitianSSMClassifier  # noqa: E402

SR = 256.0
WIN_S = 8.0
STEP_S = 8.0


def _make_recording(subject: int, run: str, n_seconds: int, seizure: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(n_seconds * SR)
    c = 23
    t = np.arange(n) / SR
    x = rng.standard_normal((c, n)).astype(np.float32) * 10.0
    # inject a bit of cross-channel structure that differs by class
    carrier = np.sin(2 * np.pi * (25 if seizure else 12) * t).astype(np.float32)
    x[0] += 8 * carrier
    x[3] += 8 * np.roll(carrier, 5)
    if seizure:
        x[7] += 6 * carrier

    ns_win = int(round(WIN_S * SR))
    step = int(round(STEP_S * SR))
    windows = []
    n_wins = (n - ns_win) // step + 1
    for i in range(n_wins):
        s0 = i * step
        # last third of a seizure recording is "preictal"
        label = 1 if (seizure and i >= 2 * n_wins // 3) else 0
        wm = {
            "start_sample": s0,
            "end_sample": s0 + ns_win,
            "window_start": s0 / SR,
            "window_end": (s0 + ns_win) / SR,
            "label": label,
        }
        wm["seizure_id"] = f"{subject}_{run}_0" if label == 1 else None
        wm["seizure_onset"] = float(n_seconds - 10) if label == 1 else None
        wm["seizure_offset"] = float(n_seconds) if label == 1 else None
        windows.append(wm)
    return {
        "raw_x": x,
        "sampling_rate": SR,
        "subject": subject,
        "session": "0",
        "run": run,
        "run_start_time": None,
        "windows": windows,
    }


def main() -> int:
    train = [
        _make_recording(1, "01", 90, seizure=True, seed=1),
        _make_recording(1, "02", 90, seizure=True, seed=2),
        _make_recording(1, "03", 120, seizure=False, seed=3),
    ]
    test = [_make_recording(1, "04", 90, seizure=True, seed=4)]

    cfg = HermitianSpectralConfig(nfreqs=24, freq_downsample=2, highest=110.0)
    clf = HermitianSSMClassifier(
        epochs=3,
        batch_size=16,
        validation_split=0.25,
        early_stopping_patience=None,
        spectral_config=cfg,
        cache_root=None,  # recompute, no disk
        d_model=64,
        d_mode=16,
        d_freq=24,
        mamba_chunk_size=64,
        head_hidden=32,
        device="cpu",
        verbose=1,
    )
    clf.fit(train)
    probs = clf.predict_proba(test)
    assert len(probs) == 1
    assert probs[0].shape == (len(test[0]["windows"]), 2), probs[0].shape
    assert np.isfinite(probs[0]).all()
    row_sums = probs[0].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)
    print(f"\nOK: predict_proba -> {probs[0].shape}, "
          f"mean p(preictal)={probs[0][:, 1].mean():.3f}, classes_={clf.classes_.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
