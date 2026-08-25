"""Continuous (whole-recording) CWT computation, and window-sized slices of
it -- the additive first step of the continuous-CWT plan (see
Session_notes/2026_08_25/continuous_cwt_plan_and_mamba_state_design.md for
the full writeup this supports).

NOT wired into the real training pipeline yet. `cwt_window_cache.py`'s
`compute_cwt_real_imag_tensors_cached` (the function every real classifier
actually calls) is completely untouched by this module -- this is a
standalone building block plus a parity script
(scripts/continuous_cwt_parity.py) to validate it against that existing
per-window path BEFORE anything downstream is asked to consume it.

Why this exists: `ContinuousLabelingParadigm.get_data()` (paradigms/
continuous_labeling.py) slices each recording into independent
fixed-length windows and discards the un-windowed continuous signal.
Every window then gets its OWN CWT computed from scratch
(cwt_window_cache.py), which means every window pays its own
cone-of-influence edge loss at BOTH its start and end -- even though, for
all but the true first/last window of a recording, that "edge" is really
just an arbitrary window boundary the paradigm chose, not a real edge of
the underlying continuous signal. Computing the CWT ONCE over the whole
recording and slicing window-sized pieces out of it means only the
recording's TRUE start/end pay that cost; every interior window boundary
becomes fully COI-valid. See `continuous_coi_valid_mask` below for the
mask fix this requires (it needs the window's ABSOLUTE offset within the
recording and the recording's FULL sample count, not the window's own).

`utils.torch_cwt.transform_batch` already handles arbitrary-length
signals in one batched call (see Session_notes/2026_08_20/
torch_native_cwt_module_and_parity_validation.md Part 8 -- benchmarked up
to 3hr continuous on real CUDA hardware, 0.79ms for 1hr/nfreqs=8) -- so
computing a continuous CWT is not new transform math, just calling that
existing function with the whole recording (one row per channel) instead
of one row per window.
"""

from __future__ import annotations

import numpy as np


def compute_continuous_cwt(
    raw_channels: np.ndarray,
    *,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    transform_batch_fn,
    device=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One CWT call over an ENTIRE continuous recording (every channel at
    once), instead of one independent call per fixed-length window.

    Parameters
    ----------
    raw_channels : np.ndarray, shape [n_channels, n_time_full]
        A whole recording's raw signal (e.g. ``raw.get_data()`` before any
        windowing) -- NOT mean/std-normalized here; that rescale, like the
        existing per-window path, belongs to the caller (see
        cwt_window_cache.py's module docstring on why CWT(raw)/std ==
        CWT(raw/std) exactly, for the zero-DC-response Morlet family).
    transform_batch_fn : callable
        e.g. ``utils.torch_cwt.transform_batch``, bound to a device via
        functools.partial if needed -- same convention
        cwt_gnn_classifiers.py's ``_resolve_transform_fns`` already uses
        for the per-window path. Passed in rather than imported directly
        so this module has no hard dependency on torch_cwt specifically.

    Returns
    -------
    real_full, imag_full : np.ndarray, shape [n_channels, n_time_full, nfreqs]
        Same axis convention (channel, time, frequency) the per-window
        path's ``w_real``/``w_imag`` already use -- ``slice_continuous_cwt_window``
        below extracts a window-sized piece of these directly, no
        transpose needed at the call site.
    freqs : np.ndarray, shape [nfreqs]
    """
    if raw_channels.ndim != 2:
        raise ValueError(
            f"raw_channels must be [n_channels, n_time_full] (2-D), got shape {raw_channels.shape}."
        )
    coeffs, freqs = transform_batch_fn(
        raw_channels, sampling_rate, highest, lowest, nfreqs=nfreqs, device=device,
    )
    # coeffs: [n_channels, nfreqs, n_time_full] complex64 (transform_batch's
    # own documented return shape) -> move to [n_channels, n_time_full,
    # nfreqs], matching w_real/w_imag's convention in cwt_window_cache.py.
    coeffs = np.moveaxis(np.asarray(coeffs), 1, -1)
    real_full = np.real(coeffs).astype(np.float32)
    imag_full = np.imag(coeffs).astype(np.float32)
    return real_full, imag_full, np.asarray(freqs)


def slice_continuous_cwt_window(
    real_full: np.ndarray,
    imag_full: np.ndarray,
    *,
    start_sample: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one window's [n_channels, n_samples, nfreqs] CWT slice from
    a continuous recording's full-length CWT tensors (see
    ``compute_continuous_cwt``) -- the counterpart to computing CWT fresh
    for that window. ``start_sample``/``n_samples`` are in raw samples
    (not seconds) -- matches
    ``ContinuousLabelingParadigm``'s own ``window_start`` (seconds) times
    the recording's sampling rate; the paradigm's metadata already carries
    everything needed to compute this (subject/run identify the recording,
    window_start identifies the offset) without any paradigm change.

    Pure slicing, no copy beyond what numpy's basic indexing already does
    -- deliberately cheap, since this is meant to run once per window
    every time a batch is drawn, not just once per recording.
    """
    end_sample = start_sample + n_samples
    if start_sample < 0 or end_sample > real_full.shape[1]:
        raise ValueError(
            f"window [{start_sample}, {end_sample}) out of bounds for a "
            f"recording with {real_full.shape[1]} samples."
        )
    return (
        real_full[:, start_sample:end_sample, :],
        imag_full[:, start_sample:end_sample, :],
    )


def continuous_coi_valid_mask(
    freqs: np.ndarray,
    *,
    sampling_rate: float,
    n_time_full: int,
    start_sample: int,
    n_samples: int,
    fb: float = 2.0,
) -> np.ndarray:
    """Absolute-offset counterpart to
    ``SparseEvidenceGNNCore._coi_valid_mask`` (cwt_gnn_classifiers.py),
    which assumes ``time_offset=0`` and validates against the WINDOW's own
    length -- correct only when each window's CWT was computed in
    isolation. Once CWT is computed continuously (this module) and a
    window is just a slice of it, validity has to be checked against the
    window's ABSOLUTE position within the full recording (``start_sample``
    .. ``start_sample + n_samples``) and the recording's FULL sample count
    (``n_time_full``), not the window's own start/length -- an interior
    window (comfortably clear of both ``0`` and ``n_time_full`` by more
    than the widest wavelet's support) comes back fully valid (this
    function's whole point: no more per-window edge loss for interior
    windows).

    Returns a [n_samples, nfreqs] bool array, same per-(time, frequency)
    shape convention ``_coi_valid_mask`` already uses (minus its batch/
    edge broadcast dims -- this is the single-recording building block a
    caller can then batch/broadcast as needed).

    NOT YET WIRED into ``SparseEvidenceGNNCore`` -- ``_coi_valid_mask``
    there still uses window-relative offsets. Swapping it for this is a
    separate, deliberately deferred step (see this module's top docstring)
    once the continuous-CWT path itself is validated end to end.
    """
    scale = sampling_rate / freqs  # [F]
    support = np.floor(fb * scale * 3.0)  # [F]
    t_idx = np.arange(start_sample, start_sample + n_samples)[:, None]  # [n_samples, 1]
    return (t_idx >= support[None, :]) & (t_idx < (n_time_full - support[None, :]))
