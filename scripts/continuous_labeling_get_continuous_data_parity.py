"""Correctness check for `ContinuousLabelingParadigm.get_continuous_data`
(paradigms/continuous_labeling.py): does it agree with the already-tested
`get_data()` on the SAME synthetic recording -- same window count, same
labels/seizure metadata, same raw samples at each window's bounds. Both
methods share the exact same `_window_starts`/`_label_windows[_prediction]`
internals by construction; this confirms the NEW method's own bookkeeping
(window bounds, chronological ordering, recording-level fields) doesn't
diverge from the already-tested one, for both label_mode values.

Synthetic MNE Raw (real mne.io.RawArray, not a stub -- exercises the real
`raw.copy()/.load_data()/.filter()/.resample()/.annotations` calls both
methods make) + a minimal fake "dataset" object exposing just
`get_data(subjects=...)` and `unit_factor`, matching
`datasets.epilepsy.CHBMIT`'s own shape without needing a real download.

Run: python scripts/continuous_labeling_get_continuous_data_parity.py
"""
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paradigms.continuous_labeling import ContinuousLabelingParadigm  # noqa: E402

SFREQ = 32.0
N_CHANNELS = 3
DURATION_S = 600.0  # 10 min synthetic recording
N_SAMPLES = int(DURATION_S * SFREQ)


class _FakeDataset:
    """Minimal stand-in for datasets.epilepsy.CHBMIT -- just enough surface
    (get_data, unit_factor) for ContinuousLabelingParadigm to run against."""

    unit_factor = 1e6

    def __init__(self, raw):
        self._raw = raw

    def get_data(self, subjects=None):
        return {1: {"0": {"0": self._raw}}}


def _make_raw() -> mne.io.RawArray:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float64) * 1e-5  # volts-scale
    info = mne.create_info([f"ch{i}" for i in range(N_CHANNELS)], SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    # Two synthetic seizures, well-separated, matching the "seizure" label_event.
    onsets = [120.0, 400.0]
    durations = [30.0, 25.0]
    raw.set_annotations(
        mne.Annotations(onset=onsets, duration=durations, description=["seizure"] * 2)
    )
    return raw


def check(label_mode: str, **paradigm_kwargs) -> None:
    print(f"\n=== label_mode={label_mode!r} kwargs={paradigm_kwargs} ===")
    raw = _make_raw()
    dataset = _FakeDataset(raw)
    paradigm = ContinuousLabelingParadigm(
        window_length=4.0, step_size=2.0, label_mode=label_mode, **paradigm_kwargs
    )

    X, y, metadata = paradigm.get_data(dataset)
    recordings = paradigm.get_continuous_data(dataset)

    assert len(recordings) == 1, f"expected 1 recording, got {len(recordings)}"
    rec = recordings[0]
    windows = rec["windows"]
    print(f"  get_data windows={len(metadata)}  get_continuous_data windows={len(windows)}")
    assert len(windows) == len(metadata) == X.shape[0] == y.shape[0], "window count mismatch"

    assert rec["raw_x"].shape == (N_CHANNELS, N_SAMPLES)
    assert rec["sampling_rate"] == SFREQ
    assert rec["subject"] == 1 and rec["run"] == "0"

    # windows are already chronological in both -- get_data's metadata list
    # is built in the same starts-order get_continuous_data sorts by.
    for i, (w, meta) in enumerate(zip(windows, metadata)):
        assert w["window_start"] == meta["window_start"], f"window {i} start mismatch"
        assert w["window_end"] == meta["window_end"]
        assert w["label"] == int(y[i])
        if label_mode == "prediction":
            assert w["seizure_id"] == meta["seizure_id"], f"window {i} seizure_id mismatch"
        # Raw content check: slicing rec["raw_x"] at this window's bounds
        # must reproduce get_data()'s own X row for the same window exactly.
        sliced = rec["raw_x"][:, w["start_sample"] : w["end_sample"]]
        np.testing.assert_allclose(sliced, X[i], rtol=0, atol=0)
    print(f"  PASS -- {len(windows)} windows match get_data() exactly (bounds, labels, raw content).")


if __name__ == "__main__":
    check("detection")
    check("prediction", sph=30.0, sop=60.0, postictal_buffer=30.0)
    print("\nAll get_continuous_data parity checks passed.")
