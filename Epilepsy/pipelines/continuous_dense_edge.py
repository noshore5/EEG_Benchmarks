"""Whole-recording chunked CWT + dense-edge feature extraction for the
continuous-cwt-mamba paradigm.

CONTEXT.md's "Open threads" item 2 ("Design + build the continuous CHB-MIT
loading path: whole-recording CWT ... TBPTT chunk boundaries ...") -- this
module is that piece. Everything downstream of raw CWT coefficients
(coherence/phase/significance, smoothing, COI masking, time-downsampling)
is the EXISTING, already-tuned `SparseEvidenceGNNCore.compute_dense_edge_input`
pipeline in `cwt_gnn_classifiers.py`, reused completely unmodified -- this
module's only job is chunking a whole recording in TIME (not just rows,
unlike `_precompute_dense_edge_inputs`'s row-chunking) into successive
`conv_in`-shaped pieces that pipeline can be called on one at a time, while
staying numerically indistinguishable from one giant unchunked call.

Design: pad-and-trim, not new math. Each chunk is grown by a left/right
context buffer pulled from the recording's own adjacent samples (zero
buffer past a genuine recording boundary -- same convention `_coi_valid_
mask` already applies at an ordinary window's true edges), run through the
UNMODIFIED per-window CWT + `compute_dense_edge_input` call as if the
padded chunk were one big window, then the buffer-affected fringe is
trimmed back off before the chunk is handed to the caller. This is the same
"carry real context across the boundary, discard it after" shape
`_DenseEdgeMambaContinuous._mixer_chunk_step`'s conv1d left-context cache
already uses for the SSM step -- applied here to the CWT/smoothing step
instead.

Two independent local operations eat into a window between raw samples and
this pipeline's final [4, E, T_out, F] output, and the pad must cover both:

1. CWT's own wavelet support / cone of influence. `utils.torch_cwt.
   _boundary_pad(sampling_rate, f0, fb=MORLET_FB)` is EXACTLY this width in
   raw samples (`scale=fs/f0, support=ceil(fb*scale*3.0)`) -- and per that
   function's own docstring, it is deliberately kept in lockstep with
   `cwt_gnn_classifiers._coi_valid_mask`'s identical formula, so reusing it
   here can't silently drift out of alignment with what COI masking
   actually assumes.
2. The smoothing conv2d's own VALID-convolution shrink
   (`SparseEvidenceGNNCore._time_offset_samples()`), applied downstream of
   CWT/COI, in the SAME raw-sample-indexed T axis (this pipeline runs with
   cwt_resample_n_time=None throughout, which is what keeps "raw-sample-
   indexed" true for both COI and smoothing -- see `_coi_valid_mask`'s own
   docstring on why that assumption matters).

Correctness argument for the trim amount (verified empirically, not just
on paper -- see scripts/continuous_cwt_chunk_parity.py): a smoothed/COI-
space output index j's smoothing kernel reads raw window
[j, j + 2*time_offset] (VALID convolution, kernel half-width time_offset
each side, `_coi_valid_mask`'s own `t_idx = arange(T_out) + time_offset`
convention is exactly this window's CENTER). For that window to be free of
any left-buffer contribution, j must be >= pad_left (in smoothed-space,
which shares the same buffer SAMPLE COUNT as raw-space, even though the
index origin itself is shifted by time_offset -- the shift cancels out of
a plain count). Symmetric on the right. So the smoothed-space trim is
exactly `pad_left` / `pad_right` samples -- no separate time_offset
adjustment needed once expressed as a count rather than an absolute index.
Converting that count into POST-downsample (`dense_edge_time_downsample`)
output steps rounds UP (`ceil`), not down: an under-trim would leave a
pooled cell partly built from buffer-affected smoothed-space samples
(silently wrong), while an over-trim by a few samples only shrinks a
chunk's usable core slightly (harmless).

`time_averaged_graph=True` collapses a window's whole T axis to 1 -- not
supported here (defeats the point of a continuous per-timestep state);
`iter_continuous_dense_edge_chunks` raises if the classifier is configured
that way.

Second, smaller source of numerical divergence from the one-shot reference
(measured directly, `scripts/continuous_cwt_chunk_parity.py`): each
chunk's own CWT runs as an INDEPENDENT FFT over that chunk's own (much
shorter) padded length, while the reference runs one FFT over the whole
recording -- `utils.torch_cwt._build_filter_bank`'s frequency-domain
Gaussian filter is evaluated at `rfftfreq(n_padded, ...)`, whose bin
spacing depends on `n_padded`, so a short chunk's filter and the
reference's whole-recording filter are two genuinely different (both
individually valid) discretizations of the same continuous-frequency
Gaussian -- not literally identical numbers even in a region neither
one's COI/boundary padding excludes. Measured magnitude: up to ~1-2%
absolute on values in a [-1, 1]/[0, 1] range, shrinking toward true
float32 noise (~1e-6) near the middle of a chunk and growing smoothly
(not a sharp jump -- ruled out as an indexing bug directly) toward its
edges. Well within the same order of magnitude as the bf16 precision this
pipeline already trains dense-edge features in by default
(`--dense-edge-amp-bf16`) -- not a training-relevant source of error, just
worth knowing this module's output is not bit-identical to a one-shot
computation the way `dense_edge_mamba_continuous_parity.py`'s pure-tensor
chunking checks are.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import torch

from Epilepsy.pipelines.cwt_window_cache import compute_cwt_real_imag_tensors_cached
from utils.torch_cwt import MORLET_FB, _boundary_pad


def continuous_chunk_pad_samples(classifier, core) -> int:
    """Raw-sample half-width to pad each chunk with on each side -- see
    module docstring. Split across two objects, mirroring exactly how
    `_precompute_dense_edge_inputs` itself is called: `classifier` (a
    `_BaseCWTGNNClassifier` instance, e.g. `SparseEvidenceGNNClassifier`/
    `StreamingSparseEvidenceGNNClassifier`, fit or at least device_-
    resolved) owns the CWT-transform config (sampling_rate/lowest); `core`
    (a `SparseEvidenceGNNCore` instance -- the classifier's own `model_`,
    or a throwaway "helper" instance built the same way
    `_precompute_dense_edge_inputs` builds one, since this math has no
    trainable parameters) owns `_time_offset_samples()`."""
    pad_cwt = _boundary_pad(classifier.sampling_rate, classifier.lowest, fb=MORLET_FB)
    pad_smooth = core._time_offset_samples()
    return int(pad_cwt) + int(pad_smooth)


def iter_continuous_dense_edge_chunks(
    classifier,
    core,
    raw_x: np.ndarray,
    chunk_size: int,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yields `(start, end, conv_in)`, one per `chunk_size`-sized
    (POST-downsample output steps) span of `raw_x`'s time axis, covering
    the whole recording in chronological order -- ready to feed
    `conv_in` (shaped `[1, 4*nfreqs, E, T_chunk]`, dtype/device matching
    `classifier.device_`) straight into
    `_DenseEdgeMambaContinuous.forward(chunk, cache)` one at a time, with
    `cache` threaded across successive calls.

    `start`/`end` are this chunk's NOMINAL raw-sample range `[start, end)`
    (the same range the caller's own chunking loop chose, before padding)
    -- the kept/trimmed output actually corresponds to raw positions within
    ~`_time_offset_samples()` samples of that range (see module docstring's
    correctness argument), a gap negligible next to a `dense_edge_time_
    downsample`-sized output bucket. Callers needing to map a raw sample
    position to this chunk's own output-index space (e.g. converting a
    classification window's bounds into `pool_continuous_edge_stream_to_
    windows`' index space) can treat `[start, end)` as that mapping's
    domain directly.

    raw_x: (n_channels, n_total_samples) -- ONE whole recording's raw
    signal, already channel-subset the same way `_apply_channel_subset`
    would (this function does not apply it). Still in RAW (not z-scored)
    units -- `mean`/`std` are applied internally by
    `compute_cwt_real_imag_tensors_cached` the same way a normal per-window
    call already does (see that function's docstring: CWT(raw)/std ==
    CWT((raw-mean)/std) to float32 noise, so raw_x is passed through
    unmodified and only rescaled on the way out).

    chunk_size: target number of OUTPUT (post-`dense_edge_time_downsample`)
    time steps per yielded chunk. The actual number of raw samples consumed
    per chunk is chunk_size * dense_edge_time_downsample; the last chunk of
    a recording may be shorter if the recording length doesn't divide
    evenly (dropped remainder, same convention `_downsample_dense_edge_
    time`'s own avg_pool2d already uses for a window's trailing remainder).

    Every yielded chunk's edge-index axis (dim 2, E) and channel axis
    (dim 1, 4*nfreqs = C_in) match `_precompute_dense_edge_inputs`'s
    canonical full-mesh layout -- this function does not support
    channel_subset_k's live-edge-subset scatter path; pass the classifier's
    canonical (n_channels, n_channels) full mesh only.
    """
    if bool(getattr(core, "time_averaged_graph", False)):
        raise ValueError(
            "iter_continuous_dense_edge_chunks: core.time_averaged_graph=True "
            "collapses a window's whole T axis to 1, which defeats the point of a "
            "continuous per-timestep SSM state -- not supported here."
        )
    n_channels, n_total = raw_x.shape
    downsample = max(1, int(core.dense_edge_time_downsample))
    pad = continuous_chunk_pad_samples(classifier, core)
    # Round UP to a multiple of `downsample` (extra margin, never less than
    # the analytically-required minimum -- always safe). This is not just
    # rounding hygiene: `_downsample_dense_edge_time`'s avg_pool2d tiles
    # each call's own OUTPUT in non-overlapping `downsample`-wide windows
    # anchored at THAT call's own local position 0 -- i.e. at
    # `start - pad_left` in absolute raw terms. For an interior chunk,
    # `start` is always a multiple of `downsample` (raw_chunk_size =
    # chunk_size*downsample), so this local grid coincides with the SAME
    # global grid the one-shot reference's own pooling uses (anchored at
    # absolute raw position 0) exactly when `pad_left` is ALSO a multiple
    # of downsample -- otherwise every interior chunk's pool grid is
    # phase-shifted from the reference's by a constant, non-numerical-noise
    # amount (measured directly: pushes chunk-vs-reference error from
    # ~1e-4-3e-2 to ~0.19, an order of magnitude jump that's a real
    # misalignment, not FFT/Gaussian-filter imprecision -- see
    # scripts/continuous_cwt_chunk_parity.py's development history).
    # min(pad, start)/min(pad, n_total-end) below preserve this multiple-of-
    # downsample property under clamping too, since `start` (and `end` for
    # every non-final chunk) already are one.
    if downsample > 1 and pad % downsample:
        pad = ((pad // downsample) + 1) * downsample
    raw_chunk_size = int(chunk_size) * downsample
    if raw_chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size!r}.")

    start = 0
    while start < n_total:
        end = min(start + raw_chunk_size, n_total)
        pad_left = min(pad, start)
        pad_right = min(pad, n_total - end)
        slice_raw = raw_x[:, start - pad_left : end + pad_right]
        X_batch = slice_raw[np.newaxis, :, :].astype(np.float32)  # [1, C, T_padded]

        _raw_x_t, w_real, w_imag, freqs = compute_cwt_real_imag_tensors_cached(
            X_batch,
            mean=mean,
            std=std,
            sampling_rate=classifier.sampling_rate,
            highest=classifier.highest,
            lowest=classifier.lowest,
            nfreqs=classifier.nfreqs,
            cwt_resample_n_time=None,
            transform_fn=classifier.transform_,
            verbose=0,
            cache=None,
            batch_transform_fn=classifier.batch_transform_,
            cwt_backend=classifier.cwt_backend,
            keep_on_device=False,
            device=classifier.device_,
        )
        w_real = w_real.to(classifier.device_)
        w_imag = w_imag.to(classifier.device_)
        freqs = freqs.to(classifier.device_)

        dense = core.compute_dense_edge_input(w_real, w_imag, freqs)
        # dense: [1, 4, E, T_out_padded, F] -- see compute_dense_edge_input's
        # docstring. T_out_padded already reflects the smoothing valid-conv
        # shrink and (if >1) the dense_edge_time_downsample avg-pool.
        trim_left = math.ceil(pad_left / downsample)
        trim_right = math.ceil(pad_right / downsample)
        t_out = dense.shape[3]
        trimmed = dense[:, :, :, trim_left : t_out - trim_right, :]
        if trimmed.shape[3] <= 0:
            raise RuntimeError(
                f"iter_continuous_dense_edge_chunks: chunk [{start}, {end}) trimmed to "
                f"<= 0 output steps (t_out={t_out}, trim_left={trim_left}, "
                f"trim_right={trim_right}) -- chunk_size is too small relative to this "
                f"config's pad ({pad} raw samples / {downsample} downsample)."
            )
        b, four, num_edges, t_core, nfreqs = trimmed.shape
        # Fold [1, 4, E, T, F] -> conv_in's [1, C_in=4*F, E, T] layout --
        # SAME reshape SparseEvidenceGNNCore.forward's dense branch (and
        # _downsample_dense_edge_time) already use: permute F next to the
        # 4-stack channel axis, fold them together.
        conv_in = trimmed.permute(0, 1, 4, 2, 3).reshape(b, four * nfreqs, num_edges, t_core)
        yield start, end, conv_in
        start = end
