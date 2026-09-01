"""On-disk cache for per-trial dense-edge-input tensors -- the [4, E, T, F]
(coherence, sin phase, cos phase, significance) arrays
SparseEvidenceGNNClassifier._precompute_dense_edge_inputs (cwt_gnn_classifiers.py)
produces from CWT
coefficients via compute_dense_edge_input. Same idea as cwt_window_cache.py
(content-hash key, persists across separate process runs, not just within
one leave_one_seizure_out() call), applied one stage further downstream.

RESTRICTED TO coherence_threshold_mode="fixed". Why that mode specifically:
_build_dense_edge_input (cwt_gnn_classifiers.py) runs on nothing
but w_real/w_imag/freqs when coherence_threshold_mode="fixed" -- no
surrogate/cluster calibration branch, no dependence on raw_x_native beyond
that. coherence = |S12|/sqrt(S1*S2) and phase = angle(S12) are both exactly
invariant to the global per-fold scalar `scale = 1/(std+eps)` applied to
w_real/w_imag (see cwt_window_cache.py's docstring for the same argument
one stage earlier) -- so the [4, E, T, F] output for a given physical
window is IDENTICAL regardless of which fold's mean/std produced the
w_real/w_imag that fed it. That's what makes caching by RAW window content
(not by the fold-normalized CWT tensors) exact rather than approximate.

NOT valid for coherence_threshold_mode in {"surrogate", "surrogate_cluster"}
-- those branches consume raw_x_native directly to calibrate a per-trial
null distribution, so their dense-edge output is NOT just a function of
(raw window, config); callers must not use this cache under those modes
(SparseEvidenceGNNClassifier._precompute_dense_edge_inputs enforces this by
only ever building cache keys when mode_label == "fixed").

On-disk layout (2026-08-24): files are uncompressed npz. Full-mesh entries
store `dense` as `[4, E, T, F]`. Live-clique entries (`channel_subset_k`)
store only the nonzero edge slots (`dense` `[4, m, T, F]`, `edge_idx`,
`n_edges`) and expand back to full E on load -- otherwise k=4 was writing
15MB/trial of which 247/253 edges were the scatter zeros, and a 100%
hit-rate cache was disk-bound (~7.6GB/epoch at batch_size=32).
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def default_dense_edge_cache_root() -> Path:
    """Same env-var precedence as cwt_window_cache.default_cwt_cache_root
    and cwt_gnn_classifiers.default_surrogate_cache_root -- a
    sibling subfolder under the same root, not a new env var."""
    configured = (
        os.environ.get("MNE_DATASETS_BNCI_PATH")
        or os.environ.get("MNE_DATA")
        or str(Path.home() / "mne_data")
    )
    return Path(configured).expanduser() / "dense_edge_cache"


def dense_edge_cache_key(
    raw_trial: np.ndarray,
    *,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: Optional[int],
    coherence_threshold: float,
    smooth_kernel_size,
    smooth_kernel_sigma,
    coi_enabled: bool,
    dense_edge_time_downsample: int,
    time_averaged_graph: bool,
    scale_adaptive_smoothing: bool,
    scale_adaptive_cycles,
    scale_adaptive_max_kernel,
    cwt_backend: str = "fcwt",
    channel_subset_k: int | None = None,
    channel_subset_metric: str = "abs_cosine",
    dense_edge_source: str = "disk_cache",
    dense_edge_ch3: str = "significance",
) -> str:
    """`raw_trial` is one trial's FULL [n_channels, n_time] raw (pre-
    normalization, post-channel-subset) window -- the whole trial in one
    key, not one channel at a time the way cwt_window_cache hashes
    per-channel, since dense-edge output mixes channel pairs (edges) and
    has no per-channel decomposition to key on separately.

    Every argument that changes what compute_dense_edge_input actually
    computes is included (mirrors the same completeness discipline as
    cwt_gnn_classifiers.surrogate_null_cache_key) -- anything
    left out here would risk silently serving a stale-config entry.

    `cwt_backend` (added 2026-08-20, Step 6 of the torch-native-cwt swap):
    this cache sits downstream of the CWT step -- see
    cwt_window_cache.py's `_window_cache_key` docstring for why an entry
    must be keyed on which transform produced it, not just on (raw
    trial, config). Defaults to "fcwt" so existing on-disk entries key
    identically to before this param existed.

    `channel_subset_k`/`channel_subset_metric` (dynamic per-window live-edge
    subset): when channel_subset_k is set, WCT runs only for the cosine-top-k
    clique and the result is scattered into a full-E zeros tensor. A
    full-mesh entry and a subset entry for the SAME raw_trial bytes must not
    collide. None/"abs_cosine" defaults key identically to before this pair
    of params existed (full mesh, unchanged)."""
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(raw_trial, dtype=np.float32).tobytes())
    config_tuple = (
        int(sampling_rate), float(highest), float(lowest), int(nfreqs),
        None if cwt_resample_n_time is None else int(cwt_resample_n_time),
        float(coherence_threshold),
        tuple(smooth_kernel_size), tuple(smooth_kernel_sigma),
        bool(coi_enabled), int(dense_edge_time_downsample), bool(time_averaged_graph),
        bool(scale_adaptive_smoothing), float(scale_adaptive_cycles),
        int(scale_adaptive_max_kernel), str(cwt_backend),
        None if channel_subset_k is None else int(channel_subset_k),
        str(channel_subset_metric),
    )
    # Only extend the tuple for non-default provenance, so existing
    # "disk_cache" on-disk entries keep hashing identically to before this
    # param existed (a running job depends on that cache).
    if str(dense_edge_source) != "disk_cache":
        config_tuple = config_tuple + (str(dense_edge_source),)
    # Same non-default-only discipline: "coi_mask" changes stack channel 3,
    # so it must not collide with a "significance" entry for the same bytes.
    if str(dense_edge_ch3) != "significance":
        config_tuple = config_tuple + (f"ch3={dense_edge_ch3}",)
    hasher.update(repr(config_tuple).encode("utf-8"))
    return hasher.hexdigest()


def precompute_dense_edge_cache_keys(
    raw_trials: np.ndarray,
    **key_kwargs,
) -> np.ndarray:
    """One `dense_edge_cache_key` per trial, hashed once per fit() rather
    than again on every streaming batch (same role as
    cwt_window_cache.precompute_window_cache_keys)."""
    n = int(raw_trials.shape[0])
    keys = np.empty(n, dtype=object)
    for i in range(n):
        keys[i] = dense_edge_cache_key(raw_trials[i], **key_kwargs)
    return keys


def _npz_has_edge_idx(path: Path) -> bool:
    """True when the npz was written in the compact live-edge layout.
    Reads only the zip directory, not the 15MB dense array."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return "edge_idx.npy" in zf.namelist()
    except Exception:
        return False


def _expand_compact_dense(array: np.ndarray, edge_idx: np.ndarray, n_edges: int) -> np.ndarray:
    """Scatter live-edge `[4, m, T, F]` into a full-E zeros tensor."""
    full = np.zeros(
        (array.shape[0], int(n_edges), array.shape[2], array.shape[3]),
        dtype=array.dtype,
    )
    if edge_idx.size:
        full[:, np.asarray(edge_idx, dtype=np.int64)] = array
    return full


def _compact_dense_payload(array: np.ndarray) -> dict[str, np.ndarray]:
    """Drop exact-zero edge slots. k=4 live clique is 6/253 edges -- the
    full `[4, 253, T, F]` fp32 tensor is 15MB of which ~97% is the zeros
    `_scatter_live_dense_edge` wrote. Streaming training reloads this
    every batch every epoch, so the zeros dominated wall time (~26ms/file,
    ~7.6GB/epoch at batch_size=32) and made a 100% hit-rate cache slower
    than MPS page-cache of the same payload. Compact files are ~0.37MB."""
    if array.ndim != 4:
        return {"dense": array}
    n_edges = int(array.shape[1])
    edge_max = np.abs(array).reshape(array.shape[0], n_edges, -1).max(axis=(0, 2))
    live = np.flatnonzero(edge_max > 0).astype(np.int32)
    if live.size >= n_edges:
        return {"dense": array}
    return {
        "dense": np.ascontiguousarray(array[:, live]),
        "edge_idx": live,
        "n_edges": np.int32(n_edges),
    }


def load_dense_edge(
    cache_dir: Path,
    key: str,
    *,
    require_compact: bool = False,
    migrate_compact: bool = False,
) -> Optional[torch.Tensor]:
    """None on any miss -- absent file, unreadable, or partially-written by
    a killed process -- never an exception; a bad entry just triggers a
    recompute+overwrite, same convention as cwt_gnn_classifiers'
    load_surrogate_null_cache.

    `require_compact=True` (CUDA + channel_subset_k): a pre-compact full-E
    file is treated as a miss. Recomputing the live clique is cheaper than
    reading 15MB of zeros, and the subsequent save writes the compact
    layout under the same key.

    `migrate_compact=True` (MPS/CPU + channel_subset_k): load the fat file
    (page cache makes that cheap there), then rewrite it compact so later
    epochs do not keep paying 15MB reads. Not used on CUDA -- that path
    refuses the fat file instead.
    """
    path = cache_dir / f"{key}.npz"
    if not path.is_file():
        return None
    is_compact = _npz_has_edge_idx(path)
    if require_compact and not is_compact:
        return None
    try:
        with np.load(path) as data:
            array = data["dense"]
            if "edge_idx" in data.files:
                array = _expand_compact_dense(
                    array,
                    data["edge_idx"],
                    int(np.asarray(data["n_edges"]).item()),
                )
    except Exception:
        return None
    if migrate_compact and not is_compact:
        try:
            save_dense_edge(cache_dir, key, torch.from_numpy(array))
        except Exception:
            pass
    return torch.from_numpy(array)


def save_dense_edge(cache_dir: Path, key: str, tensor: torch.Tensor) -> None:
    """Atomic write (temp file + os.replace) so a concurrent reader (a
    different fold's process, if ever run in parallel) never observes a
    half-written file.

    Uncompressed (2026-08-20, reverting the 2026-08-16 compressed write):
    this call sits inside _precompute_dense_edge_inputs's per-chunk GPU
    loop, so its cost is directly added to real training wall time, not
    paid off-line. Measured on real chb01 data/GPU while investigating why
    dense-edge precompute was slow despite compute_dense_edge_input itself
    being genuine vectorized torch (batched cross-spectrum + separable-conv
    smoothing, confirmed no hidden loops): np.savez_compressed's DEFLATE
    pass cost 499ms/trial for only an 11% size reduction on this pipeline's
    actual prediction-mode tensors (13.73MB -> 15.51MB uncompressed) --
    nowhere near the 2026-08-16 note's 54%-smaller figure, which came from
    a different (dense-edge-GRU, larger T) config's tensors, not these.
    That 499ms dwarfed the GPU compute it sat next to (117ms for an entire
    4-trial chunk) -- 95% of total precompute wall time was this DEFLATE
    pass, not the tensor math. np.savez (no compression) measured 6.6ms for
    the same tensor -- disk space cost is real but modest (this cache is
    fully regenerable, not source data) and utterly dominated by the
    write-time-CPU cost on the hot path. See the 2026-08-20 session note's
    "dense-edge, not CWT" section for the full before/after numbers.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / f"{key}.npz"
    tmp_path = cache_dir / f".{key}.{os.getpid()}.tmp.npz"
    # .float() before numpy: dense_edge_amp_bf16 leaves bf16 on the GPU,
    # and numpy has no bfloat16 dtype.
    payload = _compact_dense_payload(tensor.detach().cpu().float().numpy())
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, final_path)
