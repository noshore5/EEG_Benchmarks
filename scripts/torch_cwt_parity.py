"""Standalone parity check: utils.coherence_utils.transform (fcwt/FFTW) vs.
utils.torch_cwt.transform (torch.fft), on real CHB-MIT trials.

Not a pytest test -- a one-off validation script (see the Step 3 gate in
the torch-native-CWT task this supports: don't wire torch_cwt in anywhere
until this shows near-exact parity). Run it directly:

    python scripts/torch_cwt_parity.py [--edf PATH] [--n-trials 8]

Needs an environment with BOTH `fcwt` and `torch` importable -- fcwt
doesn't build from the PyPI sdist on Apple Silicon (see this repo's
setup.sh, which builds it from source on Linux instead). On this dev
machine that means running this script with a Python that already has a
working fcwt install, e.g.:

    /Users/noahshore/Documents/CoherIQs/CMPX_EEG/CMPX/bin/python3 \\
        scripts/torch_cwt_parity.py

Reports, per frequency bin AND pooled over all bins:
  - magnitude Pearson correlation (fcwt |coeffs| vs torch_cwt |coeffs|)
  - phase alignment: mean cos(phase_old - phase_new) (1.0 = identical
    phase) and a circular correlation coefficient
  - max absolute error, separately for the real and imaginary parts
  - the same four numbers restricted to the COI-valid region only (see
    cwt_gnn_classifiers.py's _coi_valid_mask -- the region actually used
    downstream), to separate genuine edge/boundary drift from anything
    that would actually reach the model.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EDF = (
    Path.home() / "mne_data" / "MNE-chbmit-data" / "chbmit" / "1.0.0" / "chb01" / "chb01_01.edf"
)

# Canonical config -- Epilepsy/run_pipelines.py's _SHARED_ARCH_PARAMS.
SAMPLING_RATE = 256
LOWEST = 8.0
HIGHEST = 40.0
NFREQS = 8
WINDOW_SECONDS = 4.0  # detection-mode window_length default
MORLET_FB = 2.0  # must match utils/torch_cwt.py's MORLET_FB


def _coi_valid_mask(freqs: np.ndarray, n_time: int) -> np.ndarray:
    """Same formula as SparseEvidenceGNNCore._coi_valid_mask
    (cwt_gnn_classifiers.py), with time_offset=0 since we're validating raw
    (un-smoothed) CWT coefficients here, not the post-smoothing feature
    stack. Returns a [n_time, nfreqs] bool array."""
    scale = SAMPLING_RATE / freqs  # [F]
    support = np.floor(MORLET_FB * scale * 3.0)  # [F]
    t_idx = np.arange(n_time)[:, None]  # [T, 1]
    return (t_idx >= support[None, :]) & (t_idx < (n_time - support[None, :]))


def _load_real_trials(edf_path: Path, n_trials: int) -> tuple[np.ndarray, list[str]]:
    """Real (non-synthetic) CHB-MIT segments: n_trials evenly-spaced,
    non-overlapping WINDOW_SECONDS-long single-channel windows, drawn from
    a handful of distinct channels so trial content actually differs (not
    n_trials copies of channel 0)."""
    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - SAMPLING_RATE) > 1e-6:
        raise ValueError(f"{edf_path} is sampled at {raw.info['sfreq']} Hz, expected {SAMPLING_RATE}.")

    data = raw.get_data()  # [n_channels, n_samples], volts
    n_channels, n_samples = data.shape
    win = int(round(WINDOW_SECONDS * SAMPLING_RATE))
    max_start = n_samples - win
    starts = np.linspace(0, max_start, n_trials, dtype=int)

    trials = np.empty((n_trials, win), dtype=np.float64)
    labels = []
    for i, start in enumerate(starts):
        ch = i % n_channels
        # Scale volts -> microvolts-ish magnitude, matching what a real
        # pipeline run would feed in (raw EDF values here are ~1e-4 scale,
        # which is fine for the wavelet transform itself -- coherence_utils
        # doesn't care about physical units -- but keeping a realistic
        # amplitude avoids exercising float32 underflow corners real runs
        # never hit).
        trials[i] = data[ch, start : start + win] * 1e6
        labels.append(f"ch={raw.ch_names[ch]} t={start / SAMPLING_RATE:.1f}s")
    return trials, labels


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _circular_corr(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    """Circular correlation coefficient (Jammalamadaka & SenGupta), robust
    to phase wraparound -- a plain Pearson correlation on raw angles would
    be corrupted by the +-pi wrap."""
    a = phase_a.ravel()
    b = phase_b.ravel()
    sa = np.sin(a - np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    sb = np.sin(b - np.arctan2(np.mean(np.sin(b)), np.mean(np.cos(b))))
    num = np.sum(sa * sb)
    den = math.sqrt(np.sum(sa**2) * np.sum(sb**2))
    return float(num / den) if den > 1e-12 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, default=DEFAULT_EDF)
    parser.add_argument("--n-trials", type=int, default=8)
    args = parser.parse_args()

    try:
        import fcwt  # noqa: F401  -- presence check; utils.coherence_utils imports it too
    except ImportError as exc:
        raise SystemExit(
            "fcwt is not importable in this interpreter -- see this script's docstring "
            "for a working interpreter to run it with."
        ) from exc

    from utils import coherence_utils, torch_cwt

    if not args.edf.is_file():
        raise SystemExit(f"EDF file not found: {args.edf}")

    print(f"Loading {args.n_trials} real trials from {args.edf} ...")
    trials, labels = _load_real_trials(args.edf, args.n_trials)
    n_time = trials.shape[1]

    all_mag_old, all_mag_new = [], []
    all_phase_old, all_phase_new = [], []
    all_re_err, all_im_err = [], []
    coi_mag_old, coi_mag_new = [], []
    coi_phase_old, coi_phase_new = [], []
    coi_re_err, coi_im_err = [], []
    per_freq_mag_corr = None
    per_freq_max_abs_err = None
    freqs_ref = None

    for trial, label in zip(trials, labels):
        coeffs_old_raw, freqs_old = coherence_utils.transform(
            trial, SAMPLING_RATE, HIGHEST, LOWEST, nfreqs=NFREQS
        )
        coeffs_new_raw, freqs_new = torch_cwt.transform(
            trial, SAMPLING_RATE, HIGHEST, LOWEST, nfreqs=NFREQS
        )

        # Both come back freq-major (nfreqs, n_time) -- transpose to
        # (n_time, nfreqs) for the COI mask's convention.
        coeffs_old = coeffs_old_raw.T if coeffs_old_raw.shape[0] == NFREQS else coeffs_old_raw
        coeffs_new = coeffs_new_raw.T if coeffs_new_raw.shape[0] == NFREQS else coeffs_new_raw

        if freqs_ref is None:
            freqs_ref = freqs_old
            print(f"fcwt freqs:      {np.round(freqs_old, 3)}")
            print(f"torch_cwt freqs: {np.round(freqs_new, 3)}")
            freq_err = np.max(np.abs(freqs_old.astype(np.float64) - freqs_new.astype(np.float64)))
            print(f"max |freq diff|: {freq_err:.6g}\n")

        mask = _coi_valid_mask(freqs_old.astype(np.float64), n_time)

        mag_old, mag_new = np.abs(coeffs_old), np.abs(coeffs_new)
        phase_old, phase_new = np.angle(coeffs_old), np.angle(coeffs_new)
        re_err = np.abs(coeffs_old.real - coeffs_new.real)
        im_err = np.abs(coeffs_old.imag - coeffs_new.imag)

        if per_freq_mag_corr is None:
            per_freq_mag_corr = [[] for _ in range(NFREQS)]
            per_freq_max_abs_err = [[] for _ in range(NFREQS)]
        for f in range(NFREQS):
            per_freq_mag_corr[f].append(_pearson(mag_old[:, f], mag_new[:, f]))
            per_freq_max_abs_err[f].append(
                float(np.max(np.abs(coeffs_old[:, f] - coeffs_new[:, f])))
            )

        all_mag_old.append(mag_old.ravel())
        all_mag_new.append(mag_new.ravel())
        all_phase_old.append(phase_old.ravel())
        all_phase_new.append(phase_new.ravel())
        all_re_err.append(re_err.ravel())
        all_im_err.append(im_err.ravel())

        coi_mag_old.append(mag_old[mask])
        coi_mag_new.append(mag_new[mask])
        coi_phase_old.append(phase_old[mask])
        coi_phase_new.append(phase_new[mask])
        coi_re_err.append(re_err[mask])
        coi_im_err.append(im_err[mask])

        print(
            f"  {label}: full max|Δre|={np.max(re_err):.4g} max|Δim|={np.max(im_err):.4g} "
            f"| COI-valid max|Δre|={np.max(re_err[mask]) if mask.any() else float('nan'):.4g}"
        )

    def _concat(chunks):
        return np.concatenate(chunks) if chunks else np.array([])

    print("\n=== Pooled over all trials, ALL samples (incl. boundary) ===")
    print(f"magnitude Pearson r:     {_pearson(_concat(all_mag_old), _concat(all_mag_new)):.6f}")
    print(
        f"phase mean cos(Δphase):  "
        f"{float(np.mean(np.cos(_concat(all_phase_old) - _concat(all_phase_new)))):.6f}"
    )
    print(
        f"phase circular corr:     "
        f"{_circular_corr(_concat(all_phase_old), _concat(all_phase_new)):.6f}"
    )
    print(f"max |Δreal|:             {float(np.max(_concat(all_re_err))):.6g}")
    print(f"max |Δimag|:             {float(np.max(_concat(all_im_err))):.6g}")

    print("\n=== Pooled over all trials, COI-VALID samples only ===")
    print(f"magnitude Pearson r:     {_pearson(_concat(coi_mag_old), _concat(coi_mag_new)):.6f}")
    print(
        f"phase mean cos(Δphase):  "
        f"{float(np.mean(np.cos(_concat(coi_phase_old) - _concat(coi_phase_new)))):.6f}"
    )
    print(
        f"phase circular corr:     "
        f"{_circular_corr(_concat(coi_phase_old), _concat(coi_phase_new)):.6f}"
    )
    print(f"max |Δreal|:             {float(np.max(_concat(coi_re_err))):.6g}")
    print(f"max |Δimag|:             {float(np.max(_concat(coi_im_err))):.6g}")

    print("\n=== Per-frequency (pooled over trials) ===")
    print(f"{'freq (Hz)':>10} {'mag corr':>10} {'max|Δcoeff|':>12}")
    for f in range(NFREQS):
        corr = float(np.nanmean(per_freq_mag_corr[f]))
        max_err = float(np.max(per_freq_max_abs_err[f]))
        print(f"{freqs_ref[f]:>10.3f} {corr:>10.6f} {max_err:>12.6g}")


if __name__ == "__main__":
    main()
