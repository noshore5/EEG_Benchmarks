"""In-memory cache for per-window CWT coefficients, shared across separate
SparseEvidenceGNNClassifier instances/fit() calls that see the same
physical window under the same CWT config but different (fold-specific)
z-score normalization stats.

Why caching pre-normalization CWT and rescaling on retrieval is EXACT, not
an approximation: global z-score normalization here is
X_norm = (X_raw - mean) / std, one scalar mean/std per fit() call (see
common.fit_global_zscore_stats/apply_global_zscore). The wavelets fcwt uses
(Morlet-family, admissible/zero-mean) have ~zero response to a constant
offset, so CWT((X_raw - mean) / std) == CWT(X_raw) / std to within float32
noise. Measured empirically (not assumed): max relative error 2.3e-7
between CWT(X_raw)/std and CWT((X_raw-mean)/std) for two different
(mean, std) pairs applied to the same synthetic EEG-scale window, including
at the trial edges (not just the COI-safe center) -- see the commit that
added this file. So caching CWT(X_raw) once and dividing by whichever
fold's `std` is in effect on retrieval reproduces the uncached result to
float32 precision, regardless of which fold computed that std.

This matters because leave-one-seizure-out (or any other grouped CV) reuses
most of the same physical windows across folds -- each fold just excludes a
different group -- so without this, the expensive part (the actual fcwt
transform, not the normalization) gets recomputed from scratch, from
scratch, every fold, for windows that were already transformed in a
previous fold.

Only the CWT step is cached. `raw_x` (the time-domain signal
ChannelSignalEncoder consumes) is NOT going through a zero-DC-response
filter -- mean subtraction matters there -- so it's recomputed fresh from
(X_raw, mean, std) every call. That's cheap (elementwise arithmetic, no
wavelet transform), so there's nothing worth caching in it.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch
from scipy.signal import resample
from tqdm.auto import tqdm

try:
    from Epilepsy.pipelines.common import cwt_progress_context, prepare_cwt_tf
except ModuleNotFoundError:
    from pipelines.common import cwt_progress_context, prepare_cwt_tf


def _window_cache_key(
    raw_channel: np.ndarray,
    *,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: Optional[int],
) -> str:
    digest = hashlib.sha256(
        np.ascontiguousarray(raw_channel, dtype=np.float32).tobytes()
    ).hexdigest()
    return f"{digest}|{sampling_rate}|{highest}|{lowest}|{nfreqs}|{cwt_resample_n_time}"


def compute_cwt_real_imag_tensors_cached(
    X_raw: np.ndarray,
    *,
    mean: float,
    std: float,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: Optional[int],
    transform_fn,
    verbose: int,
    cache: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in equivalent of common.compute_cwt_real_imag_tensors that takes
    the RAW (pre-normalization, post-channel-subset) window plus this fit
    call's (mean, std) instead of an already-normalized array, and caches
    CWT(raw window) keyed by content + CWT config so repeat calls on the
    same physical window (any fold, any mean/std) skip re-running the
    actual wavelet transform. See this module's docstring for why the
    `/ std` rescale on retrieval is exact.

    `cache` is a plain dict the caller owns -- pass the SAME dict across
    multiple classifier instances (e.g. one per CV fold) to get reuse
    across them; pass a fresh `{}` for no sharing.
    """
    n_samples, n_channels, n_time_orig = X_raw.shape
    n_time = n_time_orig if cwt_resample_n_time is None else int(cwt_resample_n_time)
    if n_time <= 0:
        raise ValueError("cwt_resample_n_time must be a positive integer or None.")

    w_real = np.zeros((n_samples, n_channels, n_time, nfreqs), dtype=np.float32)
    w_imag = np.zeros((n_samples, n_channels, n_time, nfreqs), dtype=np.float32)
    scale = 1.0 / (std + 1e-8)

    with cwt_progress_context(
        "shared-tensors-cached",
        verbose=verbose,
        samples=n_samples,
        channels=n_channels,
        transforms=n_samples * n_channels,
        input_time=n_time_orig,
        output_time=n_time,
        nfreqs=nfreqs,
    ) as show_progress:
        _, freqs = transform_fn(X_raw[0, 0, :], sampling_rate, highest, lowest, nfreqs=nfreqs)
        freqs = torch.from_numpy(freqs).float().expand(n_samples, nfreqs)

        n_hits = 0
        with tqdm(
            total=n_samples * n_channels, desc="CWT(cached)", disable=not show_progress, leave=False
        ) as pbar:
            for sample_idx in range(n_samples):
                for ch_idx in range(n_channels):
                    raw_channel = X_raw[sample_idx, ch_idx, :]
                    key = _window_cache_key(
                        raw_channel,
                        sampling_rate=sampling_rate,
                        highest=highest,
                        lowest=lowest,
                        nfreqs=nfreqs,
                        cwt_resample_n_time=cwt_resample_n_time,
                    )
                    cached = cache.get(key)
                    if cached is None:
                        coeffs, _ = transform_fn(raw_channel, sampling_rate, highest, lowest, nfreqs=nfreqs)
                        coeffs_tf = prepare_cwt_tf(coeffs, nfreqs=nfreqs, n_time=n_time)
                        cached = (
                            np.real(coeffs_tf).astype(np.float32),
                            np.imag(coeffs_tf).astype(np.float32),
                        )
                        cache[key] = cached
                    else:
                        n_hits += 1
                    real_raw, imag_raw = cached
                    w_real[sample_idx, ch_idx] = real_raw * scale
                    w_imag[sample_idx, ch_idx] = imag_raw * scale
                    pbar.update(1)

    if verbose >= 1 and n_samples * n_channels > 0:
        print(
            f"[CWT cache] {n_hits}/{n_samples * n_channels} windows*channels "
            f"reused from cache ({100 * n_hits / (n_samples * n_channels):.1f}%)"
        )

    raw_x = (X_raw - mean) / (std + 1e-8)
    raw_x = resample(raw_x, n_time, axis=2) if n_time != n_time_orig else raw_x
    raw_x = np.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return (
        torch.from_numpy(raw_x).float(),
        torch.from_numpy(w_real).float(),
        torch.from_numpy(w_imag).float(),
        freqs,
    )
