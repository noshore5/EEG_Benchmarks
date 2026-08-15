"""Sanity check: CHB-MIT subject 1 loads end-to-end through the
continuous-labeling paradigm, and the resulting ictal windows land near the
7 documented seizures in chb01-summary.txt.

    CMPX/bin/python -m pytest tests/test_chb_mit_continuous_labeling.py -v -s

Downloads ~300MB (the 7 recordings in chb01 that contain a seizure; the
other ~35 seizure-free recordings for chb01 are skipped) into MNE's usual
cache dir (``~/mne_data`` by default) the first time it's run; subsequent
runs reuse the cache. Marked ``slow`` so it can be excluded from routine
runs with ``-m "not slow"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.epilepsy import CHBMIT  # noqa: E402
from paradigms.continuous_labeling import ContinuousLabelingParadigm  # noqa: E402


SUBJECT = 1
KNOWN_SEIZURE_COUNT = 7  # per chb01-summary.txt

# Window/step for this test: coarser than the paradigm's own defaults
# (window_length=4.0, step_size=1.0) purely to bound memory/runtime when
# windowing 7 full one-hour recordings here -- every documented seizure
# lasts >= 30s, so a 15s step still lands multiple windows inside each one.
# A real run would pick step_size to match whatever transform (e.g. a
# wavelet-coherence hop) consumes the windows downstream.
WINDOW_LENGTH = 4.0
STEP_SIZE = 15.0
TOLERANCE = WINDOW_LENGTH + STEP_SIZE  # seconds of slack vs. the documented onset/offset


def ictal_clusters(y: np.ndarray, metadata: list[dict]) -> list[dict]:
    """Group consecutive ictal-labeled windows (same run) into clusters.

    Returns one dict per cluster: run, first/last ictal window bounds.
    """
    clusters = []
    current = None
    for label, meta in zip(y, metadata):
        if label == 1:
            if current is not None and current["run"] == meta["run"]:
                current["end"] = meta["window_end"]
            else:
                current = {
                    "run": meta["run"],
                    "start": meta["window_start"],
                    "end": meta["window_end"],
                }
                clusters.append(current)
        else:
            current = None
    return clusters


@pytest.fixture(scope="module")
def seizure_records():
    """The chb01 recordings documented to contain a seizure (cheap: summary only)."""
    records = CHBMIT().list_seizure_records(SUBJECT)
    assert len(records) == KNOWN_SEIZURE_COUNT, (
        f"Expected {KNOWN_SEIZURE_COUNT} seizure recordings for chb{SUBJECT:02d} "
        f"per the summary file, found {len(records)}: {records}"
    )
    for r in records:
        assert len(r["seizures"]) == 1, f"Unexpected multi-seizure file: {r}"
    return records


@pytest.fixture(scope="module")
def windowed_data(seizure_records):
    """Load + window just those 7 recordings (skips chb01's ~35 seizure-free ones)."""
    filenames = [r["filename"] for r in seizure_records]
    dataset = CHBMIT(records={SUBJECT: filenames})
    paradigm = ContinuousLabelingParadigm(
        window_length=WINDOW_LENGTH, step_size=STEP_SIZE, label_event="seizure"
    )
    return paradigm.get_data(dataset, subjects=[SUBJECT])


@pytest.mark.slow
def test_chb01_seizure_count_matches_summary(seizure_records):
    assert len(seizure_records) == KNOWN_SEIZURE_COUNT


@pytest.mark.slow
def test_windowing_shape(windowed_data):
    X, y, metadata = windowed_data
    assert X.ndim == 3, f"Expected X to be (n_windows, n_channels, n_samples), got {X.shape}"
    assert X.shape[0] == len(y) == len(metadata), "X/y/metadata length mismatch"
    n_windows, n_channels, n_samples = X.shape
    assert n_channels == 23, f"Expected 23 channels, got {n_channels}"
    assert n_samples == int(round(WINDOW_LENGTH * 256)), "Unexpected samples-per-window"
    assert int(y.sum()) > 0, "No ictal windows found at all"


@pytest.mark.slow
def test_ictal_windows_land_near_documented_seizures(seizure_records, windowed_data):
    X, y, metadata = windowed_data

    # runs are keyed "01".."07" in the same chronological order as
    # seizure_records (both filtered from the same parsed, ordered summary).
    run_to_known = {
        f"{i + 1:02d}": record["seizures"][0]
        for i, record in enumerate(seizure_records)
    }

    clusters = ictal_clusters(y, metadata)
    print(f"\n{len(clusters)} contiguous ictal window cluster(s) found:")
    for c in clusters:
        print(f"  run {c['run']}: windows span [{c['start']:.0f}s, {c['end']:.0f}s)")

    assert len(clusters) == KNOWN_SEIZURE_COUNT, (
        f"Expected exactly {KNOWN_SEIZURE_COUNT} ictal clusters (one per seizure "
        f"recording), found {len(clusters)}"
    )

    clusters_by_run = {c["run"]: c for c in clusters}
    for run, (known_onset, known_offset) in sorted(run_to_known.items()):
        assert run in clusters_by_run, (
            f"No ictal windows found for run {run} (seizure at {known_onset}s)"
        )
        cluster = clusters_by_run[run]
        onset_err = abs(cluster["start"] - known_onset)
        offset_err = abs(cluster["end"] - known_offset)
        print(
            f"  run {run}: documented [{known_onset:.0f}s, {known_offset:.0f}s) vs. "
            f"detected [{cluster['start']:.0f}s, {cluster['end']:.0f}s)  "
            f"(onset err={onset_err:.0f}s, offset err={offset_err:.0f}s)"
        )
        assert onset_err <= TOLERANCE, (
            f"run {run}: detected ictal window start {cluster['start']}s is more than "
            f"{TOLERANCE}s from the documented onset {known_onset}s"
        )
        assert offset_err <= TOLERANCE, (
            f"run {run}: detected ictal window end {cluster['end']}s is more than "
            f"{TOLERANCE}s from the documented offset {known_offset}s"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
