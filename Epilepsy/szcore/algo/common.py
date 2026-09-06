"""Shared EEG-loading + windowing + event post-processing for the SzCORE
container and the training script -- kept here so training and inference
build windows identically.
"""

from __future__ import annotations

import numpy as np

FS = 256  # SzCORE standard input sampling rate
BIPOLAR_DBANANA = (
    "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp1-F7", "F7-T3", "T3-T5", "T5-O1",
    "Fz-Cz", "Cz-Pz", "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fp2-F8", "F8-T4",
    "T4-T6", "T6-O2",
)
N_CHANNELS = len(BIPOLAR_DBANANA)

# Windowing (seconds). WINDOW_S must match the checkpoint's n_time / FS.
WINDOW_S = 4.0
TRAIN_STEP_S = 2.0
INFER_STEP_S = 1.0

# Event post-processing (seconds).
MERGE_GAP_S = 30.0   # events closer than this are fused
MIN_EVENT_S = 10.0   # events shorter than this are dropped


def load_bipolar_eeg(edf_path: str) -> tuple[np.ndarray, float]:
    """EDF -> (data [N_CHANNELS, n_samples] float32, fs). Accepts either the
    SzCORE unipolar common-average montage (re-referenced to double-banana
    bipolar here) or an already-bipolar file (e.g. raw CHB-MIT)."""
    from epilepsy2bids.eeg import Eeg

    try:
        eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_path)
        if eeg.montage is Eeg.Montage.UNIPOLAR:
            eeg.reReferenceToBipolar()
    except (ValueError, KeyError):
        # Auto-detect trips on CHB-MIT's uppercase "FP1-F7" labels; ask for
        # the bipolar pairs explicitly (Eeg._findChannelIndex handles the
        # FP1/Fp1 and T7/T3 synonyms).
        eeg = Eeg.loadEdf(
            edf_path, montage=Eeg.Montage.BIPOLAR, electrodes=list(BIPOLAR_DBANANA)
        )

    data = np.asarray(eeg.data, dtype=np.float32)
    fs = float(eeg.fs)
    if abs(fs - FS) > 1e-6:
        data = _resample(data, fs, FS)
        fs = float(FS)
    if data.shape[0] != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} bipolar channels, got {data.shape[0]}")
    return data, fs


def _resample(data: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(int(round(fs_in)), int(fs_out))
    up, down = fs_out // g, int(round(fs_in)) // g
    return resample_poly(data, up, down, axis=1).astype(np.float32)


def window_starts(n_samples: int, window: int, step: int) -> np.ndarray:
    if n_samples < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_samples - window + 1, step, dtype=np.int64)


def make_windows(data: np.ndarray, window: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """(data [C, N]) -> (windows [W, C, window], starts [W])."""
    starts = window_starts(data.shape[1], window, step)
    if starts.size == 0:
        return np.empty((0, data.shape[0], window), dtype=np.float32), starts
    windows = np.stack([data[:, s : s + window] for s in starts]).astype(np.float32)
    return windows, starts


def probs_to_mask(
    probs: np.ndarray,
    starts: np.ndarray,
    n_samples: int,
    fs: float,
    threshold: float,
    step: int,
) -> np.ndarray:
    """Per-window seizure probabilities -> sample-level boolean hypothesis
    mask, with gap-merging and a minimum-duration filter."""
    mask = np.zeros(n_samples, dtype=bool)
    fired = probs >= threshold
    for s, hit in zip(starts, fired):
        if hit:
            mask[s : s + step] = True
    if not mask.any():
        return mask

    edges = np.diff(mask.astype(np.int8))
    on = list(np.flatnonzero(edges == 1) + 1)
    off = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        on.insert(0, 0)
    if mask[-1]:
        off.append(n_samples)

    merge_gap = int(round(MERGE_GAP_S * fs))
    min_len = int(round(MIN_EVENT_S * fs))
    merged: list[list[int]] = []
    for a, b in zip(on, off):
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    out = np.zeros(n_samples, dtype=bool)
    for a, b in merged:
        if b - a >= min_len:
            out[a:b] = True
    return out
