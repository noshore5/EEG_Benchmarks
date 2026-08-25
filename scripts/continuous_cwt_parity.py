"""Standalone parity check: continuous (whole-recording) CWT + window
slicing (Epilepsy/pipelines/continuous_cwt.py) vs. today's real pipeline
path -- independent per-window CWT (cwt_window_cache.py, cache=None, same
as a real run's cwt_backend="torch" path). Real CHB-MIT data, not
synthetic.

Not a pytest test -- a one-off validation script, same convention as
scripts/torch_cwt_parity.py (which this mirrors: pooled + COI-valid-only
correlation/error numbers). Run it directly:

    python scripts/continuous_cwt_parity.py [--edf PATH] [--n-windows 5]

What this is checking, specifically:
  - Interior windows (comfortably clear of the recording's true start/end)
    should be NEAR-IDENTICAL between the two paths pooled over ALL
    samples, including each window's own edges -- because those "edges"
    are only real signal boundaries for windows actually touching the
    recording's start/end; everywhere else they're arbitrary window
    cuts the continuous path doesn't pay for.
  - The two paths should therefore diverge MOST at each window's own
    edge samples specifically (that's the COI loss the continuous path is
    fixing, not a bug) -- reported separately from the window's interior.
  - Whichever window in the sample genuinely touches the recording's true
    start or end should show the same edge behavior in BOTH paths (there's
    no free lunch at the actual recording boundary).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EDF = (
    Path.home() / "mne_data" / "MNE-chbmit-data" / "chbmit" / "1.0.0" / "chb01" / "chb01_03.edf"
)

# Canonical prediction-mode config -- Epilepsy/run_pipelines.py's _SHARED_ARCH_PARAMS.
SAMPLING_RATE = 256
LOWEST = 8.0
HIGHEST = 40.0
NFREQS = 8
WINDOW_SECONDS = 30.0
MORLET_FB = 2.0  # must match utils/torch_cwt.py's MORLET_FB


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, default=DEFAULT_EDF)
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import mne

    from Epilepsy.pipelines.continuous_cwt import (
        compute_continuous_cwt,
        slice_continuous_cwt_window,
    )
    from utils import torch_cwt

    if not args.edf.is_file():
        raise SystemExit(f"EDF file not found: {args.edf}")

    print(f"Loading {args.edf} ...")
    raw = mne.io.read_raw_edf(str(args.edf), preload=True, verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - SAMPLING_RATE) > 1e-6:
        raise SystemExit(f"{args.edf} is sampled at {raw.info['sfreq']} Hz, expected {SAMPLING_RATE}.")

    raw_data = raw.get_data() * 1e6  # volts -> microvolts-ish, matches paradigm's dataset.unit_factor convention
    n_channels, n_time_full = raw_data.shape
    duration_s = n_time_full / SAMPLING_RATE
    print(f"n_channels={n_channels} n_time_full={n_time_full} duration={duration_s:.1f}s")

    n_samples = int(round(WINDOW_SECONDS * SAMPLING_RATE))
    max_start = n_time_full - n_samples
    if max_start <= 0:
        raise SystemExit(f"Recording shorter than one {WINDOW_SECONDS}s window.")
    starts = np.linspace(0, max_start, args.n_windows, dtype=int)

    print(f"\nComputing continuous CWT over the whole {duration_s:.1f}s recording (one call, "
          f"all {n_channels} channels) ...")
    t0 = __import__("time").perf_counter()
    real_full, imag_full, freqs = compute_continuous_cwt(
        raw_data, sampling_rate=SAMPLING_RATE, highest=HIGHEST, lowest=LOWEST, nfreqs=NFREQS,
        transform_batch_fn=torch_cwt.transform_batch, device=args.device,
    )
    t_continuous = __import__("time").perf_counter() - t0
    print(f"continuous CWT: {t_continuous * 1000:.2f}ms  freqs={np.round(freqs, 3)}")

    support = np.floor(MORLET_FB * (SAMPLING_RATE / freqs) * 3.0)  # [nfreqs], samples

    print(f"\nComparing {args.n_windows} independent {WINDOW_SECONDS}s windows "
          f"(per-window CWT, today's real path) vs. slices of the continuous CWT:\n")

    pooled_full_re, pooled_full_im = [], []
    pooled_full_re_c, pooled_full_im_c = [], []
    pooled_edge_re, pooled_edge_im = [], []
    pooled_interior_re, pooled_interior_im = [], []

    t_perwindow_total = 0.0
    for start in starts:
        t0 = __import__("time").perf_counter()
        window = raw_data[:, start : start + n_samples]
        coeffs_pw, freqs_pw = torch_cwt.transform_batch(
            window, SAMPLING_RATE, HIGHEST, LOWEST, nfreqs=NFREQS, device=args.device,
        )
        t_perwindow_total += __import__("time").perf_counter() - t0
        coeffs_pw = np.moveaxis(np.asarray(coeffs_pw), 1, -1)  # [n_channels, n_samples, nfreqs]
        real_pw, imag_pw = np.real(coeffs_pw).astype(np.float32), np.imag(coeffs_pw).astype(np.float32)

        real_sl, imag_sl = slice_continuous_cwt_window(
            real_full, imag_full, start_sample=int(start), n_samples=n_samples
        )

        re_err = np.abs(real_pw - real_sl)
        im_err = np.abs(imag_pw - imag_sl)

        # Widest-frequency support (lowest freq, index depends on freqs'
        # ordering -- take the max over freqs to be safe) marks each
        # window's own edge region, in window-local samples.
        edge_w = int(support.max())
        edge_mask = np.zeros(n_samples, dtype=bool)
        edge_mask[:edge_w] = True
        edge_mask[n_samples - edge_w :] = True
        interior_mask = ~edge_mask

        touches_start = start == 0
        touches_end = (start + n_samples) == n_time_full
        edge_note = " (touches recording start)" if touches_start else (
            " (touches recording end)" if touches_end else ""
        )
        print(
            f"  window [{start / SAMPLING_RATE:6.1f}s, {(start + n_samples) / SAMPLING_RATE:6.1f}s){edge_note}: "
            f"full max|dre|={re_err.max():.4g} max|dim|={im_err.max():.4g}  |  "
            f"edge-region max|dre|={re_err[:, edge_mask, :].max():.4g}  "
            f"interior max|dre|={re_err[:, interior_mask, :].max():.4g}"
        )

        pooled_full_re.append(real_pw.ravel()); pooled_full_re_c.append(real_sl.ravel())
        pooled_full_im.append(imag_pw.ravel()); pooled_full_im_c.append(imag_sl.ravel())
        pooled_edge_re.append(re_err[:, edge_mask, :].ravel())
        pooled_edge_im.append(im_err[:, edge_mask, :].ravel())
        pooled_interior_re.append(re_err[:, interior_mask, :].ravel())
        pooled_interior_im.append(im_err[:, interior_mask, :].ravel())

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.array([])

    print("\n=== Pooled over all windows ===")
    print(f"real  Pearson r (per-window vs continuous-slice): {_pearson(_cat(pooled_full_re), _cat(pooled_full_re_c)):.6f}")
    print(f"imag  Pearson r (per-window vs continuous-slice): {_pearson(_cat(pooled_full_im), _cat(pooled_full_im_c)):.6f}")
    print(f"edge-region     max|dre|={_cat(pooled_edge_re).max():.6g}  max|dim|={_cat(pooled_edge_im).max():.6g}"
          f"  (EXPECTED to be the largest divergence -- this is the COI loss the continuous path avoids)")
    print(f"interior        max|dre|={_cat(pooled_interior_re).max():.6g}  max|dim|={_cat(pooled_interior_im).max():.6g}"
          f"  (EXPECTED near float32 noise -- same math, same input, away from any edge)")

    print(f"\n=== Timing ===")
    print(f"continuous CWT (whole recording, one call): {t_continuous * 1000:.2f}ms")
    print(f"per-window CWT ({args.n_windows} independent calls, sum):   {t_perwindow_total * 1000:.2f}ms")


if __name__ == "__main__":
    main()
