"""Deterministic spectral precompute + per-recording disk cache for the
``--pipeline hermitian_ssm`` paradigm (see ``Epilepsy/hermitian_ssm.md`` --
"Graph Spectral Mamba-3 -- Design Reference").

The "hard boundary" from that doc's Section 1: everything in this module is
deterministic and computed once per recording --

    raw signal
      -> 57-63 / 117-123 Hz mains notch          (US 60 Hz + 2nd harmonic)
      -> complex Morlet CWT per channel           (utils.torch_cwt, log grid)
      -> mean-pool time by ``time_downsample``     (primary smoothing S, anti-alias)
      -> wavelet cross-spectrum  X_ij = W_i conj(W_j)
      -> short Gaussian time smoothing            (rest of S)
      -> wavelet coherence  magnitude + relative phase
      -> complex Hermitian channel graph A(f, t)  (real |W_i|^2 diagonal)
      -> Hermitian eigendecomposition  A = U Lambda U^H  (torch.linalg.eigh)
      -> reorder by |lambda| descending, keep top-k
      -> canonicalise each eigenvector's phase gauge
      -> cache  (eigenvalues, eigenvectors, coi_valid, metadata)

Everything to the RIGHT of the cache (spectral encoder -> Mamba -> head)
lives in ``hermitian_ssm_classifier.py`` and needs forward/backward passes.

Cache design (answers the "different windows over one cache" requirement):
the eigendecomposition depends only on ``(frequency, downsampled timestep)``
-- both properties of the *recording*, not of any windowing. So the cache is
keyed per recording on the continuous downsampled-time grid; a window is
just a contiguous slice ``[t0:t1]`` of it. Changing ``window_length`` /
``step_size`` re-slices the same cache. Changing anything in
``HermitianSpectralConfig`` (band, ``nfreqs``, ``k``, ``time_downsample``,
notch, diagonal, sort, smoothing) changes ``config.cache_key()`` and the
cache rebuilds.

Deliberate deviation from the windowed WCT pipeline
(``cwt_gnn_classifiers.py``): the coherence smoothing operator ``S`` here is
**time-only** (a 1-D Gaussian over the downsampled time axis), not that
pipeline's 2-D ``(5, 3)`` time x frequency kernel. Reason: the design doc
(Section 16) treats each frequency's graph as independent and fuses
frequency later in the learned encoder; cross-frequency smoothing would
pre-mix what the encoder is meant to combine. Documented here so a future
reader does not "fix" it to match the other pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

# torch-native Morlet CWT (utils/torch_cwt.py) -- same backend the rest of
# the repo standardised on (cwt_backend="torch"). Resolved the same way
# cwt_gnn_classifiers._resolve_torch_cwt does: repo_root/utils on sys.path.
try:
    from utils.torch_cwt import (
        MORLET_AMPLITUDE_SCALE,
        MORLET_FB,
        _boundary_pad,
        _next_pow2,
    )
except ModuleNotFoundError:  # pragma: no cover -- depends on how the process was launched
    import sys as _sys

    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    from utils.torch_cwt import (
        MORLET_AMPLITUDE_SCALE,
        MORLET_FB,
        _boundary_pad,
        _next_pow2,
    )

import math as _math


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HermitianSpectralConfig:
    """Every knob that changes the cached tensors. ``cache_key()`` hashes
    this whole object, so two runs share a cache iff every field matches."""

    sampling_rate: float = 256.0

    # CWT frequency grid. utils.torch_cwt always uses a *log* (geometric)
    # grid between ``lowest`` and ``highest``; ``freqs`` is returned
    # highest-first (index 0 -> highest). 8-124 Hz avoids the unusable
    # 125-128 Hz Nyquist edge (see CONTEXT.md's band open-thread / the
    # IOPscience covariance-eigenvalue precedent).
    lowest: float = 8.0
    highest: float = 124.0
    nfreqs: int = 60

    # Graph timestep = ``time_downsample`` raw samples, mean-pooled. At
    # 256 Hz, 16 -> 62.5 ms/timestep (design doc Section 5).
    time_downsample: int = 16

    # Frequency decimation applied to the smoothed cross-/auto-spectra
    # AFTER the Gaussian smoothing but BEFORE coherence is formed (the
    # frequency analogue of the time mean-pool -- Welch-style averaging of
    # adjacent log-spaced bins). ``nfreqs`` must be divisible by it. The
    # cached frequency axis has ``nfreqs // freq_downsample`` bins, each
    # labelled by the geometric mean of the bins it pooled. 1 disables it.
    freq_downsample: int = 2

    # Extra Gaussian smoothing (in *downsampled* timesteps) applied to the
    # cross-spectrum / auto-spectra before forming coherence. The mean-pool
    # above is the bulk of the smoothing operator S; this stabilises it.
    smooth_time_steps: int = 5

    # Mains notch (US CHB-MIT data: 60 Hz + 120 Hz). halfwidth 3 -> 57-63
    # and 117-123 Hz stop-bands, matching truong_stft_cnn_classifier.py.
    mains_notch: bool = True
    mains_notch_freqs: tuple[float, ...] = (60.0, 120.0)
    mains_notch_halfwidth_hz: float = 3.0

    # Hermitian graph.
    diagonal: str = "power"          # "power" -> A_ii = |W_i|^2 ; "zero" -> 0
    eigenvalue_sort: str = "abs"     # "abs" -> |lambda| desc ; "value" -> lambda desc
    k: int = 2                       # top-k eigenpairs kept, 1 <= k <= n_channels

    # Cone-of-influence: mark (freq, timestep) invalid when the wavelet at
    # that frequency would reach past a recording edge. Constant is the
    # e-folding count; 2.0 is the common convention.
    coi_cycles: float = 2.0

    # Numerical.
    eigenvector_dtype: str = "complex64"   # compute dtype; never reduce below this pre-validation (doc Section 15)
    # On-disk / in-dataset storage for the eigenvectors ONLY (the eigh itself
    # always runs at eigenvector_dtype). "complex64" = 8 B/component;
    # "float16" = real/imag split as float16, 4 B/component -> halves the
    # cache and the training mmap so a full fold's eigenvectors fit in RAM.
    # Eigenvector components are unit-norm (|u| = 1), so float16's ~1e-3
    # relative error there is far below the ~1e-6 eigh accuracy and the
    # coherence smoothing / freq decimation already applied. Affects
    # cache_key() -> switching triggers a one-time recompute.
    eigenvector_storage: str = "complex64"

    version: int = 1  # bump to invalidate every cache when the math changes

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.diagonal not in ("power", "zero"):
            raise ValueError(f"diagonal must be 'power' or 'zero', got {self.diagonal!r}")
        if self.eigenvalue_sort not in ("abs", "value"):
            raise ValueError(f"eigenvalue_sort must be 'abs' or 'value', got {self.eigenvalue_sort!r}")
        if self.eigenvector_storage not in ("complex64", "float16"):
            raise ValueError(
                f"eigenvector_storage must be 'complex64' or 'float16', got {self.eigenvector_storage!r}"
            )
        if self.lowest <= 0 or self.highest <= self.lowest:
            raise ValueError(f"require 0 < lowest < highest, got {self.lowest}, {self.highest}")
        if self.freq_downsample < 1 or self.nfreqs % self.freq_downsample != 0:
            raise ValueError(
                f"nfreqs={self.nfreqs} must be divisible by freq_downsample={self.freq_downsample}"
            )
        if self.highest >= self.sampling_rate / 2:
            raise ValueError(
                f"highest={self.highest} Hz is at/above Nyquist "
                f"({self.sampling_rate / 2} Hz) -- CWT would alias."
            )

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Deterministic per-recording precompute
# --------------------------------------------------------------------------

def _iirnotch_sos(freq: float, halfwidth_hz: float, fs: float) -> np.ndarray:
    from scipy.signal import iirnotch, tf2sos

    q = float(freq) / (2.0 * float(halfwidth_hz))  # bw = 2*halfwidth -> Q = f0/bw
    b, a = iirnotch(w0=freq, Q=q, fs=fs)
    return tf2sos(b, a)


def _apply_mains_notch(raw_x: np.ndarray, cfg: HermitianSpectralConfig) -> np.ndarray:
    """Zero-phase IIR notch at each mains frequency below Nyquist. raw_x:
    [n_channels, n_samples]. Returns a filtered copy (float64)."""
    if not cfg.mains_notch:
        return raw_x
    from scipy.signal import sosfiltfilt

    nyq = cfg.sampling_rate / 2.0
    out = np.asarray(raw_x, dtype=np.float64)
    for f0 in cfg.mains_notch_freqs:
        if f0 >= nyq:
            continue
        sos = _iirnotch_sos(f0, cfg.mains_notch_halfwidth_hz, cfg.sampling_rate)
        out = sosfiltfilt(sos, out, axis=-1)
    return np.ascontiguousarray(out)


def _gaussian_kernel1d(width_steps: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalised 1-D Gaussian, odd length >= 1, sigma = (width-1)/2 (same
    convention as common.make_gaussian_weight2d)."""
    w = max(1, int(width_steps))
    if w % 2 == 0:
        w += 1
    if w == 1:
        return torch.ones(1, device=device, dtype=dtype)
    sigma = (w - 1) / 2.0
    x = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2.0
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_time(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve the last axis of x [..., T] with a 1-D kernel, 'same' length,
    reflect padding. Runs the conv on a flattened [N, 1, T] view."""
    if kernel.numel() == 1:
        return x
    lead = x.shape[:-1]
    t = x.shape[-1]
    xf = x.reshape(-1, 1, t)
    pad = kernel.numel() // 2
    xf = torch.nn.functional.pad(xf, (pad, pad), mode="reflect")
    out = torch.nn.functional.conv1d(xf, kernel.view(1, 1, -1))
    return out.reshape(*lead, t)


def _mean_pool_time(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Non-overlapping mean pool over the last axis; trailing remainder
    dropped. Works for real or complex x."""
    if factor <= 1:
        return x
    t = x.shape[-1]
    t_ds = t // factor
    x = x[..., : t_ds * factor]
    return x.reshape(*x.shape[:-1], t_ds, factor).mean(dim=-1)


class _MorletCWT:
    """FFT-based Morlet CWT that transforms an explicit *subset* of the full
    log frequency grid at a time, so peak memory scales with ``freq_chunk``
    rather than ``nfreqs``. Padding / ``n_padded`` are fixed by the grid's
    lowest frequency (``cfg.lowest``), so every chunk's coefficients align
    exactly with what a single full ``cwt_torch`` call would return.

    Reuses ``utils.torch_cwt``'s Morlet constants and boundary-pad rule --
    the only thing done differently is building the frequency-domain filter
    bank for an arbitrary freqs tensor instead of a ``(f0, f1, fn)`` triple.
    """

    def __init__(self, signal: torch.Tensor, cfg: "HermitianSpectralConfig") -> None:
        self.dev = signal.device
        self.sr = float(cfg.sampling_rate)
        self.n_time = int(signal.shape[-1])
        self.pad = _boundary_pad(self.sr, cfg.lowest, fb=MORLET_FB)
        self.n_padded = _next_pow2(self.n_time + 2 * self.pad)
        x_padded = torch.nn.functional.pad(
            signal.to(torch.float32), (self.pad, self.n_padded - self.n_time - self.pad)
        )
        self.spectrum = torch.fft.rfft(x_padded, n=self.n_padded, dim=-1)  # [C, n_fft_bins]
        self.bin_freqs = torch.fft.rfftfreq(
            self.n_padded, d=1.0 / self.sr, device="cpu", dtype=torch.float64
        )

    def transform(self, freqs_subset: torch.Tensor) -> torch.Tensor:
        """freqs_subset: [fc] float. Returns coeffs [C, fc, n_time] complex64."""
        fsub = freqs_subset.detach().to("cpu", torch.float64)
        sigma_sec = MORLET_FB / fsub                                   # [fc]
        delta = self.bin_freqs[None, :] - fsub[:, None]                # [fc, n_fft_bins]
        filters = MORLET_AMPLITUDE_SCALE * torch.exp(
            -2.0 * (_math.pi ** 2) * (sigma_sec[:, None] ** 2) * (delta ** 2)
        )
        filters = filters.to(torch.complex64).to(self.dev)            # [fc, n_fft_bins]
        product = self.spectrum.unsqueeze(-2) * filters               # [C, fc, n_fft_bins]
        n_fft_bins = self.n_padded // 2 + 1
        full = torch.nn.functional.pad(product, (0, self.n_padded - n_fft_bins))
        coeffs = torch.fft.ifft(full, n=self.n_padded, dim=-1)        # [C, fc, n_padded]
        return coeffs[..., self.pad : self.pad + self.n_time].contiguous().to(torch.complex64)


def _coi_valid_mask(
    freqs: torch.Tensor, n_samples: int, factor: int, sampling_rate: float, cycles: float
) -> torch.Tensor:
    """[T_ds, F] bool. False where a wavelet centred at that downsampled
    timestep, at that frequency, reaches past a recording edge (within
    ``cycles`` e-foldings)."""
    t_ds = n_samples // factor
    centres = (torch.arange(t_ds, device=freqs.device, dtype=torch.float64) + 0.5) * factor
    half = cycles * sampling_rate / freqs.double()  # [F], samples
    lo = centres[:, None] - half[None, :]
    hi = centres[:, None] + half[None, :]
    return (lo >= 0) & (hi <= n_samples)


def compute_recording_spectral(
    raw_x: np.ndarray,
    cfg: HermitianSpectralConfig,
    *,
    device: str | torch.device = "cpu",
    freq_chunk: int = 8,
    verbose: bool = False,
) -> dict:
    """Full deterministic pipeline for ONE recording.

    Parameters
    ----------
    raw_x : [n_channels, n_samples] float -- the recording's whole signal,
        already scaled the way ``get_continuous_data`` returns it.
    cfg : HermitianSpectralConfig
    device : where eigh / the tensor math run. CWT filter banks scale with
        ``freq_chunk`` so peak memory is bounded regardless of recording
        length; "cpu" is the safe default (precompute is one-time).
    freq_chunk : CWT is computed ``freq_chunk`` frequencies at a time.

    Returns
    -------
    dict with:
      eigenvalues  : np.ndarray [T_ds, F_out, k]        float32
      eigenvectors : np.ndarray [T_ds, F_out, k, C]     complex64
      coi_valid    : np.ndarray [T_ds, F_out]           bool
      freqs        : np.ndarray [F_out]                 float32  (highest-first)
      time_downsample : int
      n_channels   : int
      n_samples    : int

    ``F_out = nfreqs // freq_downsample`` -- the CWT and coherence smoothing
    run at the full ``nfreqs`` grid, then adjacent log-freq bins are pooled.
    """
    dev = torch.device(device)
    fd = int(cfg.freq_downsample)
    if freq_chunk % fd != 0:
        freq_chunk = max(fd, (freq_chunk // fd) * fd)
    x_np = _apply_mains_notch(raw_x, cfg)
    n_channels, n_samples = x_np.shape
    if cfg.k > n_channels:
        raise ValueError(f"k={cfg.k} > n_channels={n_channels}")

    signal = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32)).to(dev)

    # Full log frequency grid (highest-first, matching utils.torch_cwt).
    ratio = cfg.lowest / cfg.highest
    exps = torch.linspace(0.0, 1.0, cfg.nfreqs, dtype=torch.float64)
    freqs_full = (cfg.highest * ratio ** exps).to(torch.float32)  # [nfreqs]
    # Pooled grid: geometric mean of each block of `fd` adjacent bins.
    f_out_total = cfg.nfreqs // fd
    freqs_out = torch.exp(
        torch.log(freqs_full.double()).reshape(f_out_total, fd).mean(dim=1)
    ).to(torch.float32)  # [F_out]

    t_ds = n_samples // cfg.time_downsample
    if t_ds < 1:
        raise ValueError(
            f"recording too short: {n_samples} samples < time_downsample={cfg.time_downsample}"
        )

    smooth_kernel = _gaussian_kernel1d(cfg.smooth_time_steps, dev)

    eigvals = np.empty((t_ds, f_out_total, cfg.k), dtype=np.float32)
    eigvecs = np.empty((t_ds, f_out_total, cfg.k, n_channels), dtype=np.complex64)

    iu, ju = torch.triu_indices(n_channels, n_channels, offset=1, device=dev)  # unique pairs
    cwt = _MorletCWT(signal, cfg)

    def _pool_freq(t: torch.Tensor) -> torch.Tensor:
        """Mean-pool axis 1 (frequency) in blocks of `fd`. t: [*, fc, T_ds]."""
        if fd == 1:
            return t
        lead, fc_, tail = t.shape[0], t.shape[1], t.shape[2]
        return t.reshape(lead, fc_ // fd, fd, tail).mean(dim=2)

    for f_start in range(0, cfg.nfreqs, freq_chunk):
        f_end = min(f_start + freq_chunk, cfg.nfreqs)
        fc = f_end - f_start
        fc_out = fc // fd
        fo_start = f_start // fd
        coeffs = cwt.transform(freqs_full[f_start:f_end])  # [C, fc, N] complex64

        # Primary smoothing S: mean-pool complex coefficients over time.
        w_ds = _mean_pool_time(coeffs, cfg.time_downsample)  # [C, fc, T_ds] complex
        del coeffs

        auto = (w_ds.real ** 2 + w_ds.imag ** 2)                      # [C, fc, T_ds]
        auto = _smooth_time(auto, smooth_kernel)

        w_i = w_ds[iu]   # [P, fc, T_ds]
        w_j = w_ds[ju]
        xwt = w_i * torch.conj(w_j)                                   # [P, fc, T_ds] complex
        xr = _smooth_time(xwt.real, smooth_kernel)
        xi = _smooth_time(xwt.imag, smooth_kernel)
        del w_i, w_j, xwt, w_ds

        # Frequency decimation: pool the smoothed cross-/auto-spectra over
        # adjacent log-freq bins BEFORE forming coherence (Welch-style).
        xr = _pool_freq(xr)
        xi = _pool_freq(xi)
        auto = _pool_freq(auto)                                       # [C, fc_out, T_ds]

        denom = torch.sqrt(auto[iu] * auto[ju] + 1e-12)
        coh = torch.sqrt(xr ** 2 + xi ** 2 + 1e-20) / denom          # [P, fc_out, T_ds]
        coh = coh.clamp(0.0, 1.0)
        phase = torch.atan2(xi, xr)
        del xr, xi, denom

        # Assemble Hermitian A [fc_out, T_ds, C, C] and eigendecompose.
        a = torch.zeros((fc_out, t_ds, n_channels, n_channels), dtype=torch.complex64, device=dev)
        off = coh * torch.exp(1j * phase)                             # [P, fc_out, T_ds] complex
        off = off.permute(1, 2, 0)                                    # [fc_out, T_ds, P]
        a[:, :, iu, ju] = off
        a[:, :, ju, iu] = torch.conj(off)
        if cfg.diagonal == "power":
            diag = auto.permute(1, 2, 0).to(torch.complex64)          # [fc_out, T_ds, C]
            eye = torch.arange(n_channels, device=dev)
            a[:, :, eye, eye] = diag
        del off, coh, phase, auto

        # torch.linalg.eigh: ascending real eigenvalues, orthonormal
        # (generally complex) eigenvectors. Batched over [fc_out, T_ds].
        #
        # Real CHB-MIT recordings contain flatlined / dropped-electrode
        # segments whose cross-spectral matrix is ~0 or has (near-)repeated
        # eigenvalues; LAPACK's Hermitian divide-and-conquer (syevd) then
        # fails to converge (LinAlgError, error code 22) and takes the whole
        # batch down. Two guards:
        #   1. nan_to_num -- a NaN/Inf anywhere in the batch fails the solve;
        #      zero it (degenerate slice, masked / noise downstream).
        #   2. On LinAlgError, fall back to the general (geev) solver
        #      torch.linalg.eig for the whole chunk: a different LAPACK path
        #      with no syevd convergence mode. Hermitian input => real
        #      eigenvalues (drop the ~1e-7 imaginary residue) and unit-norm,
        #      near-orthonormal eigenvectors -- good enough, since the code
        #      below re-sorts by |lambda| and phase-gauges anyway.
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            evals, evecs = torch.linalg.eigh(a)                       # [fc_out,T_ds,C], [...,C,C]
        except torch._C._LinAlgError:
            ce, cv = torch.linalg.eig(a.to(torch.complex64))
            evals = ce.real.to(torch.float32)
            evecs = cv
        del a

        # Reorder: design doc Section 9 -- do not trust the solver order.
        if cfg.eigenvalue_sort == "abs":
            order = torch.argsort(evals.abs(), dim=-1, descending=True)
        else:
            order = torch.argsort(evals, dim=-1, descending=True)
        order_k = order[..., : cfg.k]                                 # [fc, T_ds, k]
        sel_vals = torch.gather(evals, -1, order_k)                   # [fc, T_ds, k]
        sel_vecs = torch.gather(
            evecs, -1, order_k.unsqueeze(-2).expand(-1, -1, n_channels, -1)
        )                                                            # [fc, T_ds, C, k]
        sel_vecs = sel_vecs.permute(0, 1, 3, 2)                       # [fc, T_ds, k, C]

        # Phase gauge (design doc Section 11): rotate so the largest-|.|
        # component is real, non-negative.
        amax = torch.argmax(sel_vecs.abs(), dim=-1, keepdim=True)     # [fc, T_ds, k, 1]
        anchor = torch.gather(sel_vecs, -1, amax)                     # [fc, T_ds, k, 1]
        theta = torch.angle(anchor)
        sel_vecs = sel_vecs * torch.exp(-1j * theta)

        eigvals[:, fo_start : fo_start + fc_out, :] = sel_vals.permute(1, 0, 2).cpu().numpy()
        eigvecs[:, fo_start : fo_start + fc_out, :, :] = sel_vecs.permute(1, 0, 2, 3).cpu().numpy()
        if verbose:
            print(f"    freqs [{f_start}:{f_end}] -> out [{fo_start}:{fo_start + fc_out}] done")

    coi_valid = _coi_valid_mask(
        freqs_out.to(dev), n_samples, cfg.time_downsample, cfg.sampling_rate, cfg.coi_cycles
    ).cpu().numpy()

    return {
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "coi_valid": coi_valid,
        "freqs": freqs_out.numpy(),
        "time_downsample": int(cfg.time_downsample),
        "n_channels": int(n_channels),
        "n_samples": int(n_samples),
    }


# --------------------------------------------------------------------------
# Per-recording disk cache
# --------------------------------------------------------------------------

def default_hermitian_ssm_cache_root() -> Path:
    """Sits next to the other on-disk caches (``<mne_data>/hermitian_ssm_cache``)
    -- same resolution order as dense_edge_cache / cwt_window_cache."""
    base = (
        os.environ.get("MNE_DATASETS_BNCI_PATH")
        or os.environ.get("MNE_DATA")
        or str(Path.home() / "mne_data")
    )
    return Path(base) / "hermitian_ssm_cache"


_ARRAY_FILES = ("eigenvalues", "eigenvectors", "coi_valid", "freqs")


def _pack_eigenvectors(evecs: np.ndarray, storage: str) -> np.ndarray:
    """complex64 [T, F, k, C]  ->  the on-disk representation for `storage`.
    "complex64": unchanged. "float16": real/imag split to float16,
    [T, F, k, C, 2] (last axis = [Re, Im])."""
    if storage == "complex64":
        return evecs
    return np.stack([evecs.real, evecs.imag], axis=-1).astype(np.float16)


def _is_packed_float16(evecs: np.ndarray) -> bool:
    """True for the float16 [..., C, 2] layout, False for complex64 [..., C]."""
    return evecs.dtype == np.float16 and evecs.ndim == 5 and evecs.shape[-1] == 2


class HermitianSpectralCache:
    """Disk-backed per-recording spectral cache.

    Layout::

        <root>/<config.cache_key()>/
            config.json
            <subject>_<run>/eigenvalues.npy    [T_ds, F_out, k]     float32
            <subject>_<run>/eigenvectors.npy   [T_ds, F_out, k, C]  complex64
                                  OR [T_ds, F_out, k, C, 2] float16  (eigenvector_storage="float16")
            <subject>_<run>/coi_valid.npy      [T_ds, F_out]        bool
            <subject>_<run>/freqs.npy          [F_out]              float32
            <subject>_<run>/meta.json          {time_downsample, n_channels, n_samples}

    One ``.npy`` per array (not a single ``.npz``) so ``get(..., mmap=True)``
    can memory-map ``eigenvectors`` -- a full 6-fold training set is far
    larger than RAM, and the classifier only ever touches small window
    slices of it. ``root=None`` disables persistence (recompute + hold in
    memory) for smoke tests.
    """

    def __init__(
        self,
        cfg: HermitianSpectralConfig,
        root: Path | None = None,
        *,
        device: str | torch.device = "cpu",
        verbose: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.verbose = verbose
        self.root: Path | None
        if root is None:
            self.root = None
        else:
            self.root = Path(root) / cfg.cache_key()
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "config.json").write_text(
                json.dumps(asdict(cfg), sort_keys=True, indent=2, default=list)
            )
        self._mem: dict[tuple, dict] = {}   # only used when root is None

    def _dir(self, subject, run) -> Path | None:
        return None if self.root is None else self.root / f"{subject}_{run}"

    def is_cached(self, subject, run) -> bool:
        d = self._dir(subject, run)
        return d is not None and all((d / f"{n}.npy").exists() for n in _ARRAY_FILES)

    def ensure(self, subject, run, raw_x: np.ndarray) -> None:
        """Compute + persist this recording's spectral arrays if missing."""
        if self.root is None or self.is_cached(subject, run):
            return
        if self.verbose:
            print(f"  [hermitian_ssm cache] computing spectral for subject={subject} run={run} ...")
        out = compute_recording_spectral(raw_x, self.cfg, device=self.device, verbose=self.verbose)
        d = self._dir(subject, run)
        tmp = d.parent / f".{d.name}.tmp"
        if tmp.exists():
            import shutil

            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        out = dict(out)
        out["eigenvectors"] = _pack_eigenvectors(out["eigenvectors"], self.cfg.eigenvector_storage)
        for n in _ARRAY_FILES:
            np.save(tmp / f"{n}.npy", out[n])
        (tmp / "meta.json").write_text(
            json.dumps(
                {
                    "time_downsample": int(out["time_downsample"]),
                    "n_channels": int(out["n_channels"]),
                    "n_samples": int(out["n_samples"]),
                }
            )
        )
        os.replace(tmp, d)
        if self.verbose:
            mb = sum((d / f"{n}.npy").stat().st_size for n in _ARRAY_FILES) / 1e6
            print(f"  [hermitian_ssm cache] wrote {d.name}/ ({mb:.0f} MB)")

    def get(self, subject, run, raw_x: np.ndarray, *, mmap: bool = False) -> dict:
        key = (subject, run)
        if self.root is None:
            if key not in self._mem:
                if self.verbose:
                    print(f"  [hermitian_ssm cache] computing (in-memory) subject={subject} run={run} ...")
                mem = compute_recording_spectral(
                    raw_x, self.cfg, device=self.device, verbose=self.verbose
                )
                mem["eigenvectors"] = _pack_eigenvectors(
                    mem["eigenvectors"], self.cfg.eigenvector_storage
                )
                self._mem[key] = mem
            return self._mem[key]

        self.ensure(subject, run, raw_x)
        d = self._dir(subject, run)
        mm = "r" if mmap else None
        out = {n: np.load(d / f"{n}.npy", mmap_mode=mm) for n in _ARRAY_FILES}
        out.update(json.loads((d / "meta.json").read_text()))
        return out

    def window_features(
        self, subject, run, raw_x: np.ndarray, start_sample: int, end_sample: int
    ) -> dict:
        """Slice one window [start_sample, end_sample) out of the recording
        cache. Returns eigenvalues [Tw, F, k], eigenvectors [Tw, F, k, C],
        coi_valid [Tw, F] (materialised, not mmap views)."""
        rec = self.get(subject, run, raw_x)
        f = rec["time_downsample"]
        t0 = start_sample // f
        t1 = max(t0 + 1, end_sample // f)
        t1 = min(t1, rec["eigenvalues"].shape[0])
        evecs = rec["eigenvectors"][t0:t1]
        if _is_packed_float16(rec["eigenvectors"]):
            evecs = (evecs[..., 0] + 1j * evecs[..., 1]).astype(np.complex64)
        return {
            "eigenvalues": np.ascontiguousarray(rec["eigenvalues"][t0:t1]),
            "eigenvectors": np.ascontiguousarray(evecs),
            "coi_valid": np.ascontiguousarray(rec["coi_valid"][t0:t1]),
        }


def estimate_cache_bytes_per_recording(cfg: HermitianSpectralConfig, n_channels: int, duration_s: float) -> int:
    t_ds = int(duration_s * cfg.sampling_rate / cfg.time_downsample)
    f_out = cfg.nfreqs // cfg.freq_downsample
    vals = t_ds * f_out * cfg.k * 4
    bytes_per_component = 4 if cfg.eigenvector_storage == "float16" else 8  # float16 re/im vs complex64
    vecs = t_ds * f_out * cfg.k * n_channels * bytes_per_component
    coi = t_ds * f_out  # bool
    return vals + vecs + coi
