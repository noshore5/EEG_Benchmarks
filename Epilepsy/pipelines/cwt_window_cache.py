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
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.signal import resample
from tqdm.auto import tqdm

try:
    from Epilepsy.pipelines.common import cwt_progress_context, prepare_cwt_tf, prepare_cwt_tf_batch
except ModuleNotFoundError:
    from pipelines.common import cwt_progress_context, prepare_cwt_tf, prepare_cwt_tf_batch


def default_cwt_cache_root() -> Path:
    """Mirrors cwt_gnn_classifiers.default_surrogate_cache_root's
    own convention (same env-var precedence, same "just a subfolder under
    mne_data" default) so this cache lands next to the other on-disk
    caches this pipeline already writes, without adding a new env var."""
    configured = (
        os.environ.get("MNE_DATASETS_BNCI_PATH")
        or os.environ.get("MNE_DATA")
        or str(Path.home() / "mne_data")
    )
    return Path(configured).expanduser() / "cwt_window_cache"


class DiskCWTCache:
    """Disk-backed drop-in for the plain `{}` compute_cwt_real_imag_tensors_cached
    expects (only `.get(key)` / `cache[key] = value` are used -- see that
    function's `cache` param). Persists across separate process runs (e.g.
    two `python run_pipelines.py --smoke` invocations), not just across
    folds within one `leave_one_seizure_out()` call the way a plain dict
    shared across fold instances already did.

    One `.npz` file per key under `cache_dir`; atomic write (temp file +
    os.replace, same convention as cwt_gnn_classifiers.py's
    save_surrogate_null_cache) so a process killed mid-write never leaves a
    half-written entry for a later run to load.

    2026-08-16: NO in-memory front-cache -- every `.get()` is a real disk
    read, every hit costs that read again on a later fold. An earlier
    version kept a `dict` in front of disk so repeat lookups within one
    process were free; on this 16GB machine, that dict grew unbounded
    across a leave-one-seizure-out run's folds (they mostly overlap, so by
    fold 2-3 it held close to the whole dataset's decompressed CWT tensors
    at once -- ~4.4GB for the real 2,991-window/23-channel config) and
    pushed the machine into swap partway through a real run (epoch time
    crept 6s -> 20s+ as swapping got worse, not from more compute). Trading
    that back for a real (cheap, ~64KB/entry) disk read per hit.
    """

    def __init__(self, cache_dir: Path | str | None = None):
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_cwt_cache_root()

    def get(self, key: str):
        path = self._cache_dir / f"{key}.npz"
        if not path.is_file():
            return None
        try:
            with np.load(path) as data:
                return (data["real"], data["imag"])
        except Exception:
            return None  # corrupt/partial file -- treat as a miss, recompute+overwrite

    def __setitem__(self, key: str, value: tuple[np.ndarray, np.ndarray]) -> None:
        real, imag = value
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = self._cache_dir / f"{key}.npz"
        tmp_path = self._cache_dir / f".{key}.{os.getpid()}.tmp.npz"
        # Compressed (2026-08-16): measured ~54% smaller on the matching
        # dense-edge cache's tensors (see dense_edge_cache.py's save_dense_edge);
        # CWT coefficients are less sparse (no COI-zeroing at this stage) so
        # the ratio here is expected to be worse, but any reduction helps the
        # same disk budget both caches share.
        np.savez_compressed(tmp_path, real=real, imag=imag)
        os.replace(tmp_path, final_path)

    def __len__(self) -> int:
        # Counts .npz files on disk -- unlike the old in-memory dict's
        # count, this reflects everything ever cached under this cache_dir
        # (including from prior process runs), not just this process's own
        # touched set. Only called once, at the end of a run, for the
        # summary print -- a directory listing there is cheap enough.
        return sum(1 for _ in self._cache_dir.glob("*.npz")) if self._cache_dir.is_dir() else 0


def _window_cache_key(
    raw_channel: np.ndarray,
    *,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: Optional[int],
    cwt_backend: str = "fcwt",
) -> str:
    """`cwt_backend` (added 2026-08-20, Step 6 of the torch-native-cwt
    swap): fcwt and torch_cwt compute numerically DIFFERENT (if
    near-identical -- see utils/torch_cwt.py's parity validation)
    coefficients for the same raw signal + config, but this cache
    (in-memory `dict` OR the on-disk DiskCWTCache below, which persists
    ACROSS separate process runs) is otherwise keyed only on signal
    content + CWT config, not on which transform computed the cached
    entry. Without this, switching cwt_backend on a machine/pod that
    already has a populated disk cache (e.g. from a prior fcwt run)
    would silently serve stale fcwt-computed values back to a
    cwt_backend="torch" caller instead of recomputing -- confirmed
    happening in exactly this way while validating Step 6 (both backends
    read 100% cache hits from an existing on-disk cache and produced
    bit-identical eval scores). Defaults to "fcwt" so every disk cache
    entry written before this param existed keys identically to before
    -- no silent invalidation of existing fcwt caches.
    """
    digest = hashlib.sha256(
        np.ascontiguousarray(raw_channel, dtype=np.float32).tobytes()
    ).hexdigest()
    return f"{digest}|{sampling_rate}|{highest}|{lowest}|{nfreqs}|{cwt_resample_n_time}|{cwt_backend}"


def precompute_window_cache_keys(
    X_raw: np.ndarray,
    *,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: Optional[int],
    cwt_backend: str = "fcwt",
) -> np.ndarray:
    """Runs _window_cache_key over every (sample, channel) in X_raw ONCE,
    returning a [n_samples, n_channels] object array of key strings, for
    callers that will look the same X_raw up in the cache repeatedly
    within one fit() call (see StreamingSparseEvidenceGNNClassifier's
    _LazyFeatureBatchDataset, cwt_gnn_classifiers.py).

    2026-08-19: that class recomputes CWT tensors from cache every
    training batch by design (its whole point is bounding memory, not
    materializing the whole training set -- see its docstring), but
    compute_cwt_real_imag_tensors_cached's per-(sample, channel) cache
    lookup was re-running _window_cache_key's SHA256 hash of the raw
    ~4KB channel on EVERY call, hit or miss -- so a 100%-cache-hit batch
    still paid full hashing cost. Measured on a Runpod smoke run: ~380-
    400 hashes/s, 736 (sample, channel) pairs/batch (32 trials x 23
    channels) -> ~1.9s/batch of pure hashing+Python-loop overhead, x ~24
    batches/epoch, while the actual CWT/dense-edge compute those hashes
    gate was a one-time ~2.3s per fold (proved by comparing epoch_time
    across a batch with real dense-edge compute vs. one with none: both
    ~22.6-22.8s, i.e. unaffected) -- meaning this hashing, not the
    wavelet/coherence math, was the dominant real cost.

    A window's content (and therefore its key) never changes within one
    fit() call, so hashing it once here -- at _LazyFeatureBatchDataset
    construction, before any batches are drawn -- and passing the result
    into compute_cwt_real_imag_tensors_cached's `window_keys` makes every
    per-batch access a pure dict lookup, no rehash, without changing what
    gets cached or its cross-fold/cross-classifier reuse semantics (the
    key computed here is byte-for-byte identical to what
    _window_cache_key would have computed inline).
    """
    n_samples, n_channels = int(X_raw.shape[0]), int(X_raw.shape[1])
    keys = np.empty((n_samples, n_channels), dtype=object)
    for sample_idx in range(n_samples):
        for ch_idx in range(n_channels):
            keys[sample_idx, ch_idx] = _window_cache_key(
                X_raw[sample_idx, ch_idx, :],
                sampling_rate=sampling_rate, highest=highest, lowest=lowest,
                nfreqs=nfreqs, cwt_resample_n_time=cwt_resample_n_time,
                cwt_backend=cwt_backend,
            )
    return keys


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
    window_keys: Optional[np.ndarray] = None,
    batch_transform_fn=None,
    batch_size: int = 256,
    cwt_backend: str = "fcwt",
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

    `window_keys`, if given, is a [n_samples, n_channels] array of
    already-computed _window_cache_key strings (see
    precompute_window_cache_keys) aligned to X_raw's rows/channels --
    skips re-hashing each raw channel on this call, for callers that
    already hashed this exact X_raw once (e.g.
    StreamingSparseEvidenceGNNClassifier's _LazyFeatureBatchDataset,
    which would otherwise rehash every batch -- see that function's
    docstring for the measured cost). None (the default) reproduces the
    original always-hash-inline behavior, unchanged for every other
    caller.

    `batch_transform_fn`, if given, replaces the fallback per-item
    `transform_fn` call for cache MISSES only (hits never touch either
    transform) with chunked calls covering up to `batch_size` misses at
    once -- see common.compute_cwt_real_imag_tensors's docstring for why.
    Cache keys/lookups/writes are unaffected either way; this only changes
    how a miss's coefficients get computed.

    `cwt_backend` must match whichever backend `transform_fn`/
    `batch_transform_fn` actually are (see _window_cache_key's docstring)
    -- MUST be passed explicitly (not left at the "fcwt" default) whenever
    the caller is running cwt_backend="torch", or this call will silently
    read/write fcwt's cache entries under a torch-computed caller (or vice
    versa on a later fcwt run). Only matters when `window_keys` is None
    (this function computes its own keys); if `window_keys` was
    precomputed by the caller, it already baked in whatever cwt_backend
    that computation used.
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

        # Pass 1: resolve every (sample, channel)'s cache key and serve hits
        # immediately -- cheap dict lookups, no transform involved. Misses
        # are deferred (not computed inline) so they can be run as one
        # batch below instead of the original per-item loop.
        keys = np.empty((n_samples, n_channels), dtype=object)
        miss_indices: list[tuple[int, int]] = []
        n_hits = 0
        for sample_idx in range(n_samples):
            for ch_idx in range(n_channels):
                if window_keys is not None:
                    key = window_keys[sample_idx, ch_idx]
                else:
                    key = _window_cache_key(
                        X_raw[sample_idx, ch_idx, :],
                        sampling_rate=sampling_rate,
                        highest=highest,
                        lowest=lowest,
                        nfreqs=nfreqs,
                        cwt_resample_n_time=cwt_resample_n_time,
                        cwt_backend=cwt_backend,
                    )
                keys[sample_idx, ch_idx] = key
                cached = cache.get(key)
                if cached is None:
                    miss_indices.append((sample_idx, ch_idx))
                else:
                    n_hits += 1
                    real_raw, imag_raw = cached
                    w_real[sample_idx, ch_idx] = real_raw * scale
                    w_imag[sample_idx, ch_idx] = imag_raw * scale

        # Pass 2: fill in misses, either one batched call per chunk (torch
        # backend) or the original one-call-per-item loop (fcwt backend,
        # which has no batched interface to exploit).
        if batch_transform_fn is not None and miss_indices:
            step = max(1, int(batch_size))
            with tqdm(
                total=len(miss_indices), desc="CWT(cached,batched)", disable=not show_progress, leave=False
            ) as pbar:
                for start in range(0, len(miss_indices), step):
                    chunk = miss_indices[start : start + step]
                    flat = np.stack([X_raw[s, c, :] for (s, c) in chunk], axis=0)
                    coeffs_b, _ = batch_transform_fn(flat, sampling_rate, highest, lowest, nfreqs=nfreqs)
                    coeffs_tf_b = prepare_cwt_tf_batch(coeffs_b, nfreqs=nfreqs, n_time=n_time)
                    real_b = np.real(coeffs_tf_b).astype(np.float32)
                    imag_b = np.imag(coeffs_tf_b).astype(np.float32)
                    for i, (sample_idx, ch_idx) in enumerate(chunk):
                        real_raw, imag_raw = real_b[i], imag_b[i]
                        cache[keys[sample_idx, ch_idx]] = (real_raw, imag_raw)
                        w_real[sample_idx, ch_idx] = real_raw * scale
                        w_imag[sample_idx, ch_idx] = imag_raw * scale
                    pbar.update(len(chunk))
        elif miss_indices:
            with tqdm(
                total=len(miss_indices), desc="CWT(cached)", disable=not show_progress, leave=False
            ) as pbar:
                for sample_idx, ch_idx in miss_indices:
                    raw_channel = X_raw[sample_idx, ch_idx, :]
                    coeffs, _ = transform_fn(raw_channel, sampling_rate, highest, lowest, nfreqs=nfreqs)
                    coeffs_tf = prepare_cwt_tf(coeffs, nfreqs=nfreqs, n_time=n_time)
                    real_raw = np.real(coeffs_tf).astype(np.float32)
                    imag_raw = np.imag(coeffs_tf).astype(np.float32)
                    cache[keys[sample_idx, ch_idx]] = (real_raw, imag_raw)
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
