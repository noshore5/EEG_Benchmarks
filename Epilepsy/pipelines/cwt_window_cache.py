"""Per-window CWT computation for the Epilepsy SparseEvidenceGNNClassifier
pipelines.

2026-08-21: caching (a plain in-memory dict, then a disk-backed
`DiskCWTCache` persisting across separate process runs, then briefly a
bounded in-memory LRU, then a `DISABLE_CWT_CACHE` sentinel) was removed
entirely by explicit user decision, measured on a Linux/Runpod GPU pod:
with the torch-native GPU-batched CWT backend fast enough (0.16-0.23ms/
call), the cache's own SHA256 hashing + disk I/O had become the dominant
real cost there (~70% of epoch time -- see Epilepsy/Session_notes/
2026_08_19/truong_stft_cnn_prediction_run_and_dense_edge_gru_cache_bottleneck.md),
and removing it measured 34% faster end-to-end (see Epilepsy/
Session_notes/2026_08_21/pod_image_dataset_bake_and_cwt_dense_edge_cache_removal.md).

2026-08-22: restored (`DiskCWTCache`/`_window_cache_key`/
`precompute_window_cache_keys`/`DISABLE_CWT_CACHE` below), at the user's
explicit request, to try again on a DIFFERENT machine (this session's
Windows/WDDM CUDA box, not the Linux pod the removal was measured on) --
disk I/O, CPU thread count, and recompute-every-epoch cost may trade off
differently here, and this box has ~293GB free disk (vs. the ~55GB
constraint that drove the 2026-08-16 nfreqs/compression cuts), so the
disk-budget side of the original tradeoff is much less pressing. Both of
the caching layer's two prior real correctness bugs are already fixed in
what's restored here, not reintroduced: `_window_cache_key` includes
`cwt_backend` (the "silently serving stale fcwt values to a torch caller"
bug), and `DiskCWTCache` has no in-memory front-cache (the "unbounded dict
pushed a 16GB machine into swap" bug) -- see `DiskCWTCache`'s own
docstring. Caching is opt-in (`cwt_cache=None`/absent still means "no
caching" at the classifier-constructor level, same default as before this
restoration) -- nothing changes for a caller that doesn't ask for it.

STATUS (2026-08-22, end of session) -- where things actually stand:

- Validated end-to-end on this machine, real run (--pipeline dense_edge_gru
  --label-mode detection --device cuda, subject 1, one fold, one epoch
  forced): a cold-cache run and a warm-cache rerun of the exact same
  command both completed precompute successfully. Cache correctness
  confirmed directly from the logs: cold run printed
  "[CWT cache] 0/58443 windows*channels reused... (0.0%)" and
  "[dense-edge cache] 0/2541 trials reused... (0.0%)"; the warm rerun
  printed 58443/58443 (100.0%) and 2541/2541 (100.0%) respectively -- hits
  register correctly, no correctness regression observed.
- Real disk footprint measured directly (not estimated): 55,902 CWT files
  totaling 3.7GB, 2,541 dense-edge files totaling 4.9GB, for this one
  fold's training set (4s windows, 23 channels, 253 edges) -- matches the
  2026-08-16 session note's per-entry math almost exactly (~66KB/CWT
  entry, ~1.93MB/dense-edge trial, uncompressed).
- Real measured speedup: clf.fit()-start-to-OOM-crash (see below for why
  it OOMs) went from 4m13.9s (cold) to 2m16.7s (warm) for this one fold --
  a real ~46% drop, entirely from the CWT-transform and dense-edge-compute
  phases collapsing to ~0 once every lookup is a hit. Extrapolating (NOT
  independently re-measured as a full run) to a real 7-fold CV at this
  same config: ~2:53/fold with no caching at all (pure compute, current
  pre-2026-08-22 baseline) vs. ~4:14 fold 1 (cache-cold, hash+write
  overhead makes the FIRST fold slower, not faster) + ~2:17 x 6 remaining
  folds (cache-warm) = ~17:56 total, vs. ~20:11 with no caching -- a real
  but modest ~11% win on THIS machine, well short of the 34% measured on
  the Linux/Runpod pod that motivated the original 2026-08-21 removal.
- A genuine Windows portability bug was found and fixed while validating
  this restoration: `_window_cache_key` builds a `|`-joined string, which
  `DiskCWTCache` used directly as a filename -- valid on macOS/Linux
  (where this cache was originally written and tested) but not on
  Windows/NTFS (`OSError: [Errno 22] Invalid argument`, hit directly).
  Fixed by hashing the full key internally for the filename (see
  `DiskCWTCache._filename`) while keeping the original `|`-joined key for
  the dict-like `.get`/`__setitem__` interface -- nothing about the key
  FORMAT changed, just how `DiskCWTCache` turns one into a path.
  DISK-CACHE ENTRIES WRITTEN ON MACOS/LINUX BEFORE THIS FIX WILL NOT BE
  FOUND BY THIS FILENAME SCHEME (different filename derivation) -- not
  actually a problem in practice, since no such entries exist (the cache
  was removed 2026-08-21 before ever running as this restored version
  elsewhere), but worth knowing if this ever needs to interoperate with a
  cache directory populated by a different platform/commit.
- SCOPE LIMITATION, not yet resolved: `StreamingSparseEvidenceGNNClassifier`
  (`_keep_features_on_device=True`, used by `label_mode="prediction"`,
  most of this session's earlier work) bypasses this cache ENTIRELY
  whenever its keep_on_device conditions hold (the common real config:
  cwt_backend="torch", device="cuda", cwt_resample_n_time=None) -- see
  `_prepare_features`'s keep_on_device comment, cwt_gnn_classifiers.py.
  GPU-residency (this session, earlier) and disk-caching (this session,
  later) were never asked to compose; this restoration only actually
  helps detection-mode (eager `SparseEvidenceGNNClassifier`) runs.
- ORTHOGONAL, PRE-EXISTING finding surfaced during validation, NOT caused
  by or fixed as part of this restoration: the same validation run OOM'd
  during TRAINING (GRU forward pass, not precompute) at detection mode's
  default batch_size=736 on this 12GB card -- `torch.OutOfMemoryError:
  ...Tried to allocate 2.10 GiB...`. Never tested on this machine before
  (all of today's earlier chunk-size/OOM tuning was in prediction mode).
  Flagged here, not investigated or fixed.
- `--disable-disk-cache` (run_pipelines.py and pipeline_debug.py) opts
  back out to the pre-2026-08-22 no-cache behavior if a future machine/
  config (e.g. a fast Linux GPU pod again) measures recompute as faster,
  per the original 2026-08-21 finding -- don't assume this session's
  re-enable holds everywhere.

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


class _DisableCWTCacheSentinel:
    """Identity marker for `cwt_cache=DISABLE_CWT_CACHE` at the classifier
    constructor level (cwt_gnn_classifiers.py's _init_cwt_gnn_classifier).

    Distinct from that constructor's own `cwt_cache=None` default, which
    means "give me a private, per-instance, per-fit-call dict" -- this
    means "never cache at all, not even within one fit() call."
    _prepare_features translates this sentinel into the real `cache=None`
    that compute_cwt_real_imag_tensors_cached (this module) treats
    specially: skip the key-hashing/lookup loop outright, not just
    guarantee a miss.
    """


DISABLE_CWT_CACHE = _DisableCWTCacheSentinel()


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

    NO in-memory front-cache -- every `.get()` is a real disk read, every
    hit costs that read again on a later fold. An earlier version kept a
    `dict` in front of disk so repeat lookups within one process were
    free; on a 16GB machine, that dict grew unbounded across a
    leave-one-seizure-out run's folds (they mostly overlap, so by fold 2-3
    it held close to the whole dataset's decompressed CWT tensors at once
    -- ~4.4GB for the real 2,991-window/23-channel config) and pushed the
    machine into swap partway through a real run (epoch time crept
    6s -> 20s+ as swapping got worse, not from more compute). Trading that
    back for a real (cheap, sub-MB/entry) disk read per hit.
    """

    def __init__(self, cache_dir: Path | str | None = None):
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_cwt_cache_root()

    @staticmethod
    def _filename(key: str) -> str:
        """`key` (from `_window_cache_key`) is a `|`-joined string, not a
        bare hash -- fine as an in-memory dict key, but `|` is an invalid
        filename character on Windows (`OSError: [Errno 22] Invalid
        argument`, confirmed directly restoring this on a Windows/NTFS
        machine -- this cache was originally written/tested on macOS,
        where `|` is valid). Re-hashing the whole key here (not changing
        `_window_cache_key`'s own format, which callers/tests may already
        depend on) keeps every filename a plain hex string, portable
        everywhere, while `get`/`__setitem__`'s dict-like interface still
        takes the original `|`-joined key.
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, key: str):
        path = self._cache_dir / f"{self._filename(key)}.npz"
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
        filename = self._filename(key)
        final_path = self._cache_dir / f"{filename}.npz"
        tmp_path = self._cache_dir / f".{filename}.{os.getpid()}.tmp.npz"
        # Uncompressed: np.savez_compressed's DEFLATE pass measurably
        # dominates write time for a modest size win on this pipeline's
        # actual tensors (see dense_edge_cache.py's save_dense_edge
        # docstring for the equivalent measurement one stage downstream).
        # np.load reads both formats transparently.
        np.savez(tmp_path, real=real, imag=imag)
        os.replace(tmp_path, final_path)

    def __len__(self) -> int:
        # Counts .npz files on disk -- unlike an in-memory dict's count,
        # this reflects everything ever cached under this cache_dir
        # (including from prior process runs), not just this process's own
        # touched set. Only called for the occasional summary print -- a
        # directory listing there is cheap enough.
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
    """`cwt_backend`: fcwt and torch_cwt compute numerically DIFFERENT (if
    near-identical -- see utils/torch_cwt.py's parity validation)
    coefficients for the same raw signal + config, but this cache
    (in-memory `dict` OR the on-disk DiskCWTCache below, which persists
    ACROSS separate process runs) is otherwise keyed only on signal
    content + CWT config, not on which transform computed the cached
    entry. Without this, switching cwt_backend on a machine/pod that
    already has a populated disk cache (e.g. from a prior fcwt run)
    would silently serve stale fcwt-computed values back to a
    cwt_backend="torch" caller instead of recomputing -- this is the
    exact bug this pipeline hit once already (see this module's top-level
    docstring). Defaults to "fcwt" so every disk cache entry written
    before this param existed keys identically to before -- no silent
    invalidation of existing fcwt caches.
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
    _LazyFeatureBatchDataset, cwt_gnn_classifiers.py) -- avoids re-hashing
    every raw channel on every training batch (measured ~380-400 hashes/s,
    736 (sample, channel) pairs/batch -> ~1.9s/batch of pure hashing
    overhead when done inline instead).

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
    cache: Optional[dict] = None,
    window_keys: Optional[np.ndarray] = None,
    batch_transform_fn=None,
    batch_size: int = 256,
    cwt_backend: str = "fcwt",
    keep_on_device: bool = False,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rescales by 1/std on the way out, which is exact (not approximate)
    per the module docstring's zero-DC argument. `mean` is applied to
    `raw_x` only (the CWT step never sees it -- it's already ~invariant to
    a constant offset).

    `cache` is a plain dict the caller owns (or a `DiskCWTCache` -- same
    `.get`/`__setitem__` interface) -- pass the SAME cache across multiple
    classifier instances (e.g. one per CV fold) to reuse CWT work for
    windows they share instead of recomputing per fold. `None` (default)
    disables caching entirely: every (sample, channel) is treated as a
    miss and nothing is written back, not even for reuse within this one
    call -- restores this function's pre-2026-08-22 (no-cache) behavior
    exactly when the caller doesn't opt in. Ignored when `keep_on_device`
    is True (see that param).

    `window_keys`, if given, is a [n_samples, n_channels] array of
    already-computed _window_cache_key strings (see
    precompute_window_cache_keys) aligned to X_raw's rows/channels --
    skips re-hashing each raw channel on this call, for callers that
    already hashed this exact X_raw once (e.g.
    StreamingSparseEvidenceGNNClassifier's _LazyFeatureBatchDataset, which
    would otherwise rehash every batch). `None` (default) hashes inline.

    `cwt_backend` must match whichever backend `transform_fn`/
    `batch_transform_fn` actually are (see _window_cache_key's docstring)
    -- MUST be passed explicitly (not left at the "fcwt" default) whenever
    the caller is running cwt_backend="torch", or this call will silently
    read/write fcwt's cache entries under a torch-computed caller (or vice
    versa on a later fcwt run). Only matters when `window_keys` is None
    (this function computes its own keys); if `window_keys` was
    precomputed by the caller, it already baked in whatever cwt_backend
    that computation used.

    `batch_transform_fn`, if given (e.g. utils.torch_cwt.transform_batch
    bound to a GPU device), computes up to `batch_size` (sample, channel)
    signals per call instead of one `transform_fn` call per signal -- one
    host<->device transfer and one batched kernel launch per chunk instead
    of one of each per signal (see common.compute_cwt_real_imag_tensors's
    docstring for the non-cached sibling this mirrors). `None` (default,
    e.g. the fcwt backend, which has no batched interface to exploit)
    falls back to the original one-call-per-item loop. Cache hits never
    touch either transform.

    `keep_on_device=True` (`device` required in that case) routes to
    `_compute_cwt_real_imag_tensors_device_resident` below instead of the
    CPU-numpy path -- see that function's docstring. Only
    `StreamingSparseEvidenceGNNClassifier` sets this (one training batch
    at a time; see cwt_gnn_classifiers.py's `_keep_features_on_device`) --
    the eager classifiers precompute their WHOLE training set in one call
    here and must stay CPU-resident to avoid holding tens of GB in VRAM,
    so this is opt-in, not a new default. `cache`/`window_keys` are NOT
    threaded into that path -- disk caching requires CPU-resident numpy
    arrays to hash/persist, which is exactly what keep_on_device exists to
    avoid round-tripping through; the two optimizations target different
    classifiers (eager whole-dataset precompute vs. per-batch streaming)
    and were never asked to compose.
    """
    if keep_on_device:
        return _compute_cwt_real_imag_tensors_device_resident(
            X_raw,
            mean=mean,
            std=std,
            sampling_rate=sampling_rate,
            highest=highest,
            lowest=lowest,
            nfreqs=nfreqs,
            cwt_resample_n_time=cwt_resample_n_time,
            transform_fn=transform_fn,
            verbose=verbose,
            batch_transform_fn=batch_transform_fn,
            batch_size=batch_size,
            device=device,
        )

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
        # immediately -- cheap dict/disk lookups, no transform involved.
        # Misses are deferred (not computed inline) so they can be run as
        # one batch below instead of a per-item loop.
        #
        # cache=None: skips this loop's hashing/lookups entirely instead of
        # just guaranteeing every lookup misses -- every (sample, channel)
        # goes straight into miss_indices below, no _window_cache_key/
        # SHA256 call at all. Reproduces the pre-restoration (no-cache)
        # behavior exactly.
        keys = np.empty((n_samples, n_channels), dtype=object)
        n_hits = 0
        if cache is None:
            miss_indices = [(s, c) for s in range(n_samples) for c in range(n_channels)]
        else:
            miss_indices = []
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
                        if cache is not None:
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
                    if cache is not None:
                        cache[keys[sample_idx, ch_idx]] = (real_raw, imag_raw)
                    w_real[sample_idx, ch_idx] = real_raw * scale
                    w_imag[sample_idx, ch_idx] = imag_raw * scale
                    pbar.update(1)

    if verbose >= 1 and n_samples * n_channels > 0:
        if cache is None:
            print(f"[CWT cache] disabled -- recomputed all {n_samples * n_channels} windows*channels")
        else:
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


def _compute_cwt_real_imag_tensors_device_resident(
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
    batch_transform_fn,
    batch_size: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`keep_on_device=True` sibling of `compute_cwt_real_imag_tensors_cached`
    above: same math (CWT, 1/std rescale on the way out, mean-centered
    raw_x), but every intermediate stays a tensor on `device` -- no
    `np.real`/`np.imag`/`.cpu()`/`torch.from_numpy` round trip through host
    memory anywhere in this function.

    Written for StreamingSparseEvidenceGNNClassifier's per-batch precompute
    path specifically (2026-08-22 session: measured GPU utilization
    bursting to 90-100% then dropping to ~0% between chunks, ~86% of a
    32-trial batch's wall time going to CWT+dense-edge recompute even
    though the actual GPU compute inside each stage is fast -- the
    CPU<->GPU bounce between CWT, dense-edge, and the train step was the
    real cost, not raw device throughput). A single training batch (tens
    of trials) is trivial for a GPU's VRAM budget, unlike the eager
    classifiers' whole-training-set precompute, which does NOT use this
    path and is completely unaffected by it.

    Requires `batch_transform_fn` (torch backend only -- fcwt has no
    tensor-in/tensor-out interface to keep on device) and
    `cwt_resample_n_time=None` (real pipeline config's own default --
    see run_pipelines.py's `_SHARED_ARCH_PARAMS` -- meaning the CWT's own
    native time axis already equals the input's; the CPU path's resample
    step, above, exists for the general case but has no torch-side
    equivalent implemented here). Both are asserted by the caller
    (cwt_gnn_classifiers.py's `_prepare_features`), which falls back to
    `keep_on_device=False` whenever either doesn't hold, so this function
    itself just double-checks and raises rather than silently degrading.
    """
    if batch_transform_fn is None:
        raise ValueError(
            "keep_on_device=True requires a torch batch_transform_fn "
            "(cwt_backend='torch') -- the fcwt backend has no tensor-out "
            "interface to keep on device."
        )
    n_samples, n_channels, n_time_orig = X_raw.shape
    if cwt_resample_n_time is not None and int(cwt_resample_n_time) != n_time_orig:
        raise NotImplementedError(
            "keep_on_device=True doesn't support cwt_resample_n_time "
            f"(got {cwt_resample_n_time}, input n_time={n_time_orig}) -- "
            "the real pipeline config never sets this (see this function's "
            "docstring); pass keep_on_device=False for that case."
        )
    if device is None:
        raise ValueError("keep_on_device=True requires an explicit `device`.")

    n_time = n_time_orig
    scale = 1.0 / (std + 1e-8)

    _, freqs_np = transform_fn(X_raw[0, 0, :], sampling_rate, highest, lowest, nfreqs=nfreqs)
    freqs = torch.from_numpy(freqs_np).float().to(device).expand(n_samples, nfreqs)

    w_real = torch.zeros((n_samples, n_channels, n_time, nfreqs), dtype=torch.float32, device=device)
    w_imag = torch.zeros((n_samples, n_channels, n_time, nfreqs), dtype=torch.float32, device=device)
    flat_X = X_raw.reshape(n_samples * n_channels, n_time_orig)
    w_real_flat = w_real.view(n_samples * n_channels, n_time, nfreqs)
    w_imag_flat = w_imag.view(n_samples * n_channels, n_time, nfreqs)
    total = flat_X.shape[0]
    step = max(1, int(batch_size))

    show_progress = verbose >= 1
    with tqdm(
        total=total, desc="CWT(batched,gpu-resident)", disable=not show_progress, leave=False
    ) as pbar:
        for start in range(0, total, step):
            end = min(start + step, total)
            coeffs_t, _ = batch_transform_fn(
                flat_X[start:end], sampling_rate, highest, lowest, nfreqs=nfreqs,
                return_tensor=True,
            )
            # Mirrors prepare_cwt_tf_batch's transpose convention (coeffs_t
            # is [B, nfreqs, T] unless already [B, T, nfreqs]).
            coeffs_tf = coeffs_t.permute(0, 2, 1) if coeffs_t.shape[1] == nfreqs else coeffs_t
            if coeffs_tf.shape[1] != n_time:
                raise NotImplementedError(
                    f"CWT output time axis ({coeffs_tf.shape[1]}) != n_time "
                    f"({n_time}) under keep_on_device=True -- this would need "
                    "a resample step, which isn't implemented on this path "
                    "(see this function's docstring). Shouldn't be reachable "
                    "given the cwt_resample_n_time check above unless "
                    "cwt_torch's own output length ever stops matching its "
                    "input length."
                )
            coeffs_tf = torch.nan_to_num(coeffs_tf, nan=0.0, posinf=0.0, neginf=0.0)
            w_real_flat[start:end] = coeffs_tf.real.float() * scale
            w_imag_flat[start:end] = coeffs_tf.imag.float() * scale
            pbar.update(end - start)

    raw_x = torch.from_numpy(np.ascontiguousarray(X_raw, dtype=np.float32)).to(device)
    raw_x = (raw_x - mean) / (std + 1e-8)
    raw_x = torch.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0)

    return raw_x, w_real, w_imag, freqs
