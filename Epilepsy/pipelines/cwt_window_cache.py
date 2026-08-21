"""Per-window CWT computation for the Epilepsy SparseEvidenceGNNClassifier
pipelines.

2026-08-21: this module used to wrap the raw CWT computation in caching --
first a plain in-memory dict, then a disk-backed `DiskCWTCache` (persisting
across separate process runs), then briefly a bounded in-memory LRU on top
of that, then a `DISABLE_CWT_CACHE` sentinel to opt individual calls out of
all of it. All of it is gone now, by explicit user decision (not a
per-backend toggle): every call below always recomputes, nothing is hashed,
nothing touches disk. Why: the caching existed to avoid re-running the
actual wavelet transform when that transform was the expensive part (fcwt's
per-item CPU loop) or when the cache's own I/O was cheap relative to it.
Neither holds anymore -- the torch-native (`torch.fft`, GPU-batched) CWT
backend this pipeline uses by default is fast enough (measured
0.16-0.23ms/call on real CUDA hardware at the current 30s-window config,
see Epilepsy/Session_notes/2026_08_20/
torch_native_cwt_module_and_parity_validation.md Part 8) that the cache's
SHA256 hashing and disk reads/writes had become the dominant real cost
(measured ~70% of epoch time on a real Runpod run -- see
Epilepsy/Session_notes/2026_08_19/
truong_stft_cnn_prediction_run_and_dense_edge_gru_cache_bottleneck.md) --
and the caching layer had independently produced two real correctness
bugs along the way (a missing `cwt_backend` key component silently serving
stale fcwt-computed values back to a torch-backend caller; an earlier
in-memory dict that grew unbounded across CV folds and pushed a 16GB
machine into swap). Simpler, and no longer slower, to just not cache.

Why caching pre-normalization CWT and rescaling on retrieval WAS exact
(kept here because the compute-on-raw-then-rescale approach below still
uses the same math, independent of caching): global z-score normalization
here is X_norm = (X_raw - mean) / std, one scalar mean/std per fit() call
(see common.fit_global_zscore_stats/apply_global_zscore). The wavelets
fcwt/torch_cwt use (Morlet-family, admissible/zero-mean) have ~zero
response to a constant offset, so CWT((X_raw - mean) / std) == CWT(X_raw)
/ std to within float32 noise. Measured empirically (not assumed): max
relative error 2.3e-7 between CWT(X_raw)/std and CWT((X_raw-mean)/std) for
two different (mean, std) pairs applied to the same synthetic EEG-scale
window, including at the trial edges (not just the COI-safe center). This
lets the function below take the still-raw window plus this fit call's
(mean, std) and do one division on the way out, instead of z-scoring the
window before transforming it -- a minor convenience kept for its own
sake now, not because of caching.

Only the CWT step goes through this module. `raw_x` (the time-domain
signal ChannelSignalEncoder consumes) is NOT going through a zero-DC-
response filter -- mean subtraction matters there -- so it's computed
fresh from (X_raw, mean, std) every call regardless.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.signal import resample
from tqdm.auto import tqdm

try:
    from Epilepsy.pipelines.common import cwt_progress_context, prepare_cwt_tf, prepare_cwt_tf_batch
except ModuleNotFoundError:
    from pipelines.common import cwt_progress_context, prepare_cwt_tf, prepare_cwt_tf_batch


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
    batch_transform_fn=None,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes CWT(X_raw) fresh every call -- no caching, see module
    docstring -- and rescales by 1/std on the way out, which is exact (not
    approximate) per the module docstring's zero-DC argument. `mean` is
    applied to `raw_x` only (the CWT step never sees it -- it's already
    ~invariant to a constant offset).

    `batch_transform_fn`, if given (e.g. utils.torch_cwt.transform_batch
    bound to a GPU device), computes up to `batch_size` (sample, channel)
    signals per call instead of one `transform_fn` call per signal -- one
    host<->device transfer and one batched kernel launch per chunk instead
    of one of each per signal (see common.compute_cwt_real_imag_tensors's
    docstring for the non-cached sibling this mirrors). `None` (default,
    e.g. the fcwt backend, which has no batched interface to exploit)
    falls back to the original one-call-per-item loop.
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

        all_indices = [(s, c) for s in range(n_samples) for c in range(n_channels)]

        if batch_transform_fn is not None and all_indices:
            step = max(1, int(batch_size))
            with tqdm(
                total=len(all_indices), desc="CWT(batched)", disable=not show_progress, leave=False
            ) as pbar:
                for start in range(0, len(all_indices), step):
                    chunk = all_indices[start : start + step]
                    flat = np.stack([X_raw[s, c, :] for (s, c) in chunk], axis=0)
                    coeffs_b, _ = batch_transform_fn(flat, sampling_rate, highest, lowest, nfreqs=nfreqs)
                    coeffs_tf_b = prepare_cwt_tf_batch(coeffs_b, nfreqs=nfreqs, n_time=n_time)
                    real_b = np.real(coeffs_tf_b).astype(np.float32)
                    imag_b = np.imag(coeffs_tf_b).astype(np.float32)
                    for i, (sample_idx, ch_idx) in enumerate(chunk):
                        w_real[sample_idx, ch_idx] = real_b[i] * scale
                        w_imag[sample_idx, ch_idx] = imag_b[i] * scale
                    pbar.update(len(chunk))
        elif all_indices:
            with tqdm(
                total=len(all_indices), desc="CWT", disable=not show_progress, leave=False
            ) as pbar:
                for sample_idx, ch_idx in all_indices:
                    raw_channel = X_raw[sample_idx, ch_idx, :]
                    coeffs, _ = transform_fn(raw_channel, sampling_rate, highest, lowest, nfreqs=nfreqs)
                    coeffs_tf = prepare_cwt_tf(coeffs, nfreqs=nfreqs, n_time=n_time)
                    w_real[sample_idx, ch_idx] = np.real(coeffs_tf).astype(np.float32) * scale
                    w_imag[sample_idx, ch_idx] = np.imag(coeffs_tf).astype(np.float32) * scale
                    pbar.update(1)

    raw_x = (X_raw - mean) / (std + 1e-8)
    raw_x = resample(raw_x, n_time, axis=2) if n_time != n_time_orig else raw_x
    raw_x = np.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return (
        torch.from_numpy(raw_x).float(),
        torch.from_numpy(w_real).float(),
        torch.from_numpy(w_imag).float(),
        freqs,
    )
