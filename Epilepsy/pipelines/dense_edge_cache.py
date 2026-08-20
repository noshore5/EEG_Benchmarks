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
"""

from __future__ import annotations

import hashlib
import os
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
    identically to before this param existed."""
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
    )
    hasher.update(repr(config_tuple).encode("utf-8"))
    return hasher.hexdigest()


def load_dense_edge(cache_dir: Path, key: str) -> Optional[torch.Tensor]:
    """None on any miss -- absent file, unreadable, or partially-written by
    a killed process -- never an exception; a bad entry just triggers a
    recompute+overwrite, same convention as cwt_gnn_classifiers'
    load_surrogate_null_cache."""
    path = cache_dir / f"{key}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            array = data["dense"]
    except Exception:
        return None
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
    np.savez(tmp_path, dense=tensor.detach().cpu().numpy())
    os.replace(tmp_path, final_path)
