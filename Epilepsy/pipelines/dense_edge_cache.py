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
) -> str:
    """`raw_trial` is one trial's FULL [n_channels, n_time] raw (pre-
    normalization, post-channel-subset) window -- the whole trial in one
    key, not one channel at a time the way cwt_window_cache hashes
    per-channel, since dense-edge output mixes channel pairs (edges) and
    has no per-channel decomposition to key on separately.

    Every argument that changes what compute_dense_edge_input actually
    computes is included (mirrors the same completeness discipline as
    cwt_gnn_classifiers.surrogate_null_cache_key) -- anything
    left out here would risk silently serving a stale-config entry."""
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(raw_trial, dtype=np.float32).tobytes())
    config_tuple = (
        int(sampling_rate), float(highest), float(lowest), int(nfreqs),
        None if cwt_resample_n_time is None else int(cwt_resample_n_time),
        float(coherence_threshold),
        tuple(smooth_kernel_size), tuple(smooth_kernel_sigma),
        bool(coi_enabled), int(dense_edge_time_downsample), bool(time_averaged_graph),
        bool(scale_adaptive_smoothing), float(scale_adaptive_cycles),
        int(scale_adaptive_max_kernel),
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

    Compressed (2026-08-16): this tensor is ~40% exact zero (COI-masked --
    see this module's docstring / the 2026-08-15 session note's measured
    fraction), so np.savez_compressed's DEFLATE pass finds real structure to
    exploit -- measured 8.23MB -> 3.76MB (~54% smaller) on an actual cached
    tensor, not a guess. Costs write-time CPU; on a machine where disk space
    is the binding constraint (not CPU), that trade is the whole point.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / f"{key}.npz"
    tmp_path = cache_dir / f".{key}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp_path, dense=tensor.detach().cpu().numpy())
    os.replace(tmp_path, final_path)
