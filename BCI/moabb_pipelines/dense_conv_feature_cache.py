"""On-disk cache for FROZEN dense_edge_conv output features (2026-08-10).

Companion to the surrogate_null_cache machinery in
sparse_evidence_gnn_classifier.py, but caching a different, more expensive
thing: not the (config-only, trial-only) surrogate null distribution, but
one SPECIFIC trained dense_edge_conv's per-edge output
[E, dense_conv_out_channels] for every trial of a subject. Unlike the
surrogate cache (valid for any conv, since it precedes dense_edge_conv
entirely), this cache is only valid for the exact trained weights it was
built from -- dense_edge_conv is a real trainable Conv2d stack, and two
different training runs (even with identical hyperparameters) land on
different weights.

Motivation: this pipeline's downstream-architecture experimentation
(flat-vs-graph capacity sweep, see run_dense_edge_frozen_sweep.py) wants to
try many cheap classifier-head variants on top of the SAME trained conv's
features without re-running that conv's forward pass (a real Conv2d over
[E=36, T~1001, F=16] per trial) on every training step of every variant.
Freezing the conv and caching its output once turns each downstream variant
into a fast fit over a small [N_trials, E, dense_conv_out_channels] array.

Cache key covers three independent things that must ALL match for a cached
entry to be reused safely:
  1. Every config value that affects `dense_edge_raw`, the conv's INPUT
     (mirrors surrogate_null_cache_key's own config tuple).
  2. The conv's architecture hyperparameters (kernel_size, pool_size,
     intermediate_channels, out_channels, ...) -- these affect what
     `dense_edge_conv` even IS, independent of its trained weight values.
  3. A hash of the conv's actual trained weight VALUES
     (common.torch_parameter_hashes' value_model_hash) -- this is the part
     that makes a cache entry specific to ONE trained checkpoint, not just
     one architecture. Two runs with identical hyperparameters but
     different training (different seed, different data split, or even
     just a re-run) get different weight hashes and therefore different
     cache keys -- there is no way for a different conv to silently reuse
     stale features under this scheme, matching surrogate_null_cache's own
     "a change invalidates the key, not the content" design.

Values cached: `features` ([N_trials, E, dense_conv_out_channels],
float32, one row per trial in a FIXED, caller-defined order) plus
`trial_hashes` (an [N_trials]-length array of the same per-trial sha256
digest surrogate_null_cache_key uses, in that same order) so a caller can
verify alignment against its own trial ordering before trusting the cache,
the same defensive posture surrogate_null_cache_key's own docstring
describes for its own key.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from BCI.moabb_pipelines.common import torch_parameter_hashes


def default_dense_conv_feature_cache_root() -> Path:
    configured = (
        os.environ.get("MNE_DATASETS_BNCI_PATH")
        or os.environ.get("MNE_DATA")
        or str(Path.home() / "mne_data")
    )
    return Path(configured).expanduser() / "dense_conv_feature_cache"


def trial_content_hash(raw_trial: np.ndarray) -> str:
    """Same convention surrogate_null_cache_key uses for its own raw_trial
    hashing -- sha256 of the trial's raw (channel-subset-applied,
    pre-normalization) float32 bytes. Used both to build a per-trial
    `trial_hashes` array for this cache and, by callers, to verify a loaded
    cache's trial order actually matches the trial order they're about to
    index into it with.
    """
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(raw_trial, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def conv_weight_hash(dense_edge_conv: nn.Module) -> str:
    """The trained-weight-identity component of the cache key -- see module
    docstring. Delegates to common.torch_parameter_hashes so this uses the
    same tolerance-aware, reproducible hashing the rest of the codebase
    already relies on for model fingerprinting (print_torch_parameter_hashes).
    """
    _, value_model_hash, _ = torch_parameter_hashes(dense_edge_conv)
    return value_model_hash


def dense_conv_feature_cache_key(
    *,
    # --- config affecting dense_edge_raw (mirrors surrogate_null_cache_key) ---
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: int | None,
    smooth_kernel_size: tuple,
    smooth_kernel_sigma: tuple,
    coi_enabled: bool,
    coherence_threshold_mode: str,
    surrogate_count: int,
    surrogate_seed: int,
    surrogate_percentile: float,
    scale_adaptive_smoothing: bool = False,
    scale_adaptive_cycles: float = 1.5,
    scale_adaptive_max_kernel: int = 101,
    channel_subset: tuple[int, ...] = (),
    # --- dense_edge_conv architecture ---
    dense_conv_kernel_size: int,
    dense_conv_pool_size: int,
    dense_conv_intermediate_channels: int,
    dense_conv_intermediate_channels_reduced: int | None,
    dense_conv_out_channels: int,
    # --- trained-weight identity ---
    checkpoint_id: str,
    weight_hash: str,
) -> str:
    """Deterministic cache key. See module docstring for what each group of
    fields protects against. `checkpoint_id` is a human-readable label (e.g.
    "flat_control_seed42_holdout=1test") kept in the key purely so two
    checkpoints that happen to hash-collide in weight_hash (astronomically
    unlikely, not otherwise guarded against) would still be distinguished --
    weight_hash is the real safety property; checkpoint_id is a legibility
    aid that also happens to strengthen the key.
    """
    hasher = hashlib.sha256()
    config_tuple = (
        int(sampling_rate), float(highest), float(lowest), int(nfreqs),
        None if cwt_resample_n_time is None else int(cwt_resample_n_time),
        tuple(smooth_kernel_size), tuple(smooth_kernel_sigma),
        bool(coi_enabled), str(coherence_threshold_mode),
        int(surrogate_count), int(surrogate_seed), float(surrogate_percentile),
        bool(scale_adaptive_smoothing), float(scale_adaptive_cycles),
        int(scale_adaptive_max_kernel), tuple(int(c) for c in channel_subset),
        int(dense_conv_kernel_size), int(dense_conv_pool_size),
        int(dense_conv_intermediate_channels),
        None if dense_conv_intermediate_channels_reduced is None
        else int(dense_conv_intermediate_channels_reduced),
        int(dense_conv_out_channels),
        str(checkpoint_id), str(weight_hash),
    )
    hasher.update(repr(config_tuple).encode("utf-8"))
    return hasher.hexdigest()


def save_dense_conv_feature_cache(
    cache_dir: Path,
    key: str,
    features: np.ndarray,
    trial_hashes: np.ndarray,
) -> None:
    """Atomic write (temp file + rename), matching save_surrogate_null_cache's
    own crash-safety convention -- a concurrent reader never sees a
    half-written cache file (see [[run-wct-gnn-concurrent-write-race]] in
    project memory for why that convention matters on this codebase)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / f"{key}.npz"
    tmp_path = cache_dir / f".{key}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        features=np.ascontiguousarray(features, dtype=np.float32),
        trial_hashes=np.asarray(trial_hashes, dtype="<U64"),
    )
    os.replace(tmp_path, final_path)


def load_dense_conv_feature_cache(cache_dir: Path, key: str) -> dict | None:
    """Returns {'features': ndarray[N,E,C], 'trial_hashes': ndarray[N]} or
    None on a cache miss (file absent, unreadable, or corrupt -- all a plain
    miss, not an error, matching load_surrogate_null_cache's own
    permissiveness)."""
    path = cache_dir / f"{key}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return {
                "features": data["features"],
                "trial_hashes": data["trial_hashes"],
            }
    except Exception:
        return None


@torch.no_grad()
def compute_dense_conv_features(
    dense_edge_conv: nn.Module,
    dense_edge_raw: torch.Tensor,
    *,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Runs a (presumably frozen/eval-mode) dense_edge_conv over every trial
    in `dense_edge_raw` ([N, 4, E, T, F]) under torch.no_grad(), chunked to
    bound peak memory the same way _precompute_dense_edge_inputs chunks its
    own, much larger, [4, E, T, F]-per-trial precompute. Identical reshape
    to SparseEvidenceGNNCore._dense_edge_features/DenseEdgeFlatCore.forward
    -- see either docstring for why frequency folds into the conv's input
    channels while E stays the untouched, weight-shared spatial axis.

    Returns [N, E, dense_conv_out_channels], float32, on CPU -- this is the
    thing that gets cached.
    """
    was_training = dense_edge_conv.training
    dense_edge_conv.eval()
    outputs = []
    n = dense_edge_raw.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = dense_edge_raw[start:end]
        batch_size, c_in, num_edges, n_time, nfreqs = chunk.shape
        conv_in = chunk.permute(0, 1, 4, 2, 3).reshape(
            batch_size, c_in * nfreqs, num_edges, n_time
        )
        conv_out = dense_edge_conv(conv_in)  # [b, out_channels, E, 1]
        feat = conv_out.squeeze(-1).permute(0, 2, 1)  # [b, E, out_channels]
        outputs.append(feat.cpu())
    if was_training:
        dense_edge_conv.train()
    return torch.cat(outputs, dim=0).float()
