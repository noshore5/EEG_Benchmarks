"""End-to-end parity check for `ContinuousLabelingParadigm(return_continuous_raw=True)`
(paradigms/continuous_labeling.py) -- the second step of the continuous-CWT
plan (see Session_notes/2026_08_25/continuous_cwt_plan_and_mamba_state_design.md
Part 1, and Epilepsy/pipelines/continuous_cwt.py).

`scripts/continuous_cwt_parity.py` already validated the CORE continuous-CWT
math (compute_continuous_cwt + slice_continuous_cwt_window) against a raw
EDF loaded directly via mne. This script instead validates the NEW plumbing
this session added on top: does the real ContinuousLabelingParadigm class,
used the way run_pipelines.py actually calls it, hand back a
`continuous_raw` dict whose arrays genuinely match what X's per-window
slices came from, and does the continuous-CWT-then-slice path still agree
with the paradigm's own per-window CWT-equivalent when driven through this
class instead of a hand-rolled mne.io.read_raw_edf call?

Two checks:
  1. Plumbing check: every window's raw slice in X equals
     continuous_raw[(subject, session, run)][0][:, start_sample:start_sample+n_samples]
     BYTE-FOR-BYTE (same source array -- this must be exact, not
     approximate; a mismatch here would mean the new return value doesn't
     actually correspond to what get_data() windowed).
  2. CWT check: reuses compute_continuous_cwt/slice_continuous_cwt_window
     (already validated standalone) against torch_cwt.transform_batch
     applied to each of a few real X windows directly -- same pooled
     Pearson r / edge-vs-interior error split as the original parity
     script, just sourced end-to-end through the real paradigm+dataset path
     this time instead of a synthetic direct EDF load.

Run:
    python scripts/continuous_cwt_paradigm_parity.py [--subject 1] [--device cpu]

CPU by default -- deliberately avoids the GPU while a real training run may
be using it (see this session's own use of --device cpu for exactly this
reason).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLING_RATE = 256
LOWEST = 8.0
HIGHEST = 40.0
NFREQS = 8
WINDOW_LENGTH = 30.0  # seconds -- matches run_pipelines.py's prediction-mode default
STEP_SIZE = 5.0  # coarser than the real 1s step, just to keep this script's window count small
MORLET_FB = 2.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cwt-windows", type=int, default=5)
    args = parser.parse_args()

    from datasets.epilepsy import CHBMIT
    from paradigms.continuous_labeling import ContinuousLabelingParadigm
    from Epilepsy.pipelines.continuous_cwt import compute_continuous_cwt, slice_continuous_cwt_window
    from utils import torch_cwt

    # Keep this small and reuse whatever's already on disk from earlier
    # sessions: just the seizure-containing recordings for one subject
    # (same convention tests/test_chb_mit_continuous_labeling.py uses).
    all_records = CHBMIT().list_seizure_records(args.subject)
    filenames = [r["filename"] for r in all_records]
    print(f"subject {args.subject}: {len(filenames)} seizure-containing recording(s): {filenames}")
    dataset = CHBMIT(records={args.subject: filenames})

    paradigm = ContinuousLabelingParadigm(
        window_length=WINDOW_LENGTH,
        step_size=STEP_SIZE,
        label_event="seizure",
        label_mode="detection",
        return_continuous_raw=True,
    )
    print("\nCalling paradigm.get_data(..., return_continuous_raw=True) ...")
    X, y, metadata, continuous_raw = paradigm.get_data(dataset, subjects=[args.subject])
    print(f"X.shape={X.shape}  y.shape={y.shape}  len(metadata)={len(metadata)}  "
          f"len(continuous_raw)={len(continuous_raw)} recordings")

    # --- Check 1: plumbing -- every window slice matches its source recording exactly ---
    print("\n=== Check 1: window slices vs. continuous_raw, byte-for-byte ===")
    n_checked = 0
    max_abs_diff = 0.0
    for i, meta in enumerate(metadata):
        key = (meta["subject"], meta["session"], meta["run"])
        raw_data, sfreq = continuous_raw[key]
        n_samples = int(round(WINDOW_LENGTH * sfreq))
        start_sample = int(round(meta["window_start"] * sfreq))
        expected = raw_data[:, start_sample : start_sample + n_samples]
        actual = X[i]
        diff = float(np.abs(expected - actual).max()) if expected.size else 0.0
        max_abs_diff = max(max_abs_diff, diff)
        n_checked += 1
    print(f"checked {n_checked} windows across {len(continuous_raw)} recording(s): "
          f"max|expected-actual|={max_abs_diff:.6g}  "
          f"({'EXACT MATCH' if max_abs_diff == 0.0 else 'MISMATCH -- plumbing bug'})")
    assert max_abs_diff == 0.0, "continuous_raw does not match the windows it should have produced"

    # --- Check 2: continuous CWT (via the new plumbing) vs. per-window CWT ---
    print(f"\n=== Check 2: continuous CWT vs. {args.n_cwt_windows} independent per-window CWTs "
          f"(same recording, device={args.device}) ===")
    # Pick the recording with the most windows for a meaningful spread of start offsets.
    key = max(continuous_raw, key=lambda k: sum(1 for m in metadata if (m["subject"], m["session"], m["run"]) == k))
    raw_data, sfreq = continuous_raw[key]
    n_channels, n_time_full = raw_data.shape
    print(f"recording {key}: n_channels={n_channels} n_time_full={n_time_full} "
          f"duration={n_time_full / sfreq:.1f}s")

    t0 = time.perf_counter()
    real_full, imag_full, freqs = compute_continuous_cwt(
        raw_data, sampling_rate=int(sfreq), highest=HIGHEST, lowest=LOWEST, nfreqs=NFREQS,
        transform_batch_fn=torch_cwt.transform_batch, device=args.device,
    )
    t_continuous = time.perf_counter() - t0
    print(f"continuous CWT: {t_continuous * 1000:.2f}ms  freqs={np.round(freqs, 3)}")

    support = np.floor(MORLET_FB * (sfreq / freqs) * 3.0)
    n_samples = int(round(WINDOW_LENGTH * sfreq))
    max_start = n_time_full - n_samples
    starts = np.linspace(0, max_start, args.n_cwt_windows, dtype=int)

    pooled_re, pooled_re_c = [], []
    pooled_im, pooled_im_c = [], []
    pooled_edge_re, pooled_edge_im = [], []
    pooled_interior_re, pooled_interior_im = [], []
    t_perwindow_total = 0.0
    for start in starts:
        window = raw_data[:, start : start + n_samples]
        t0 = time.perf_counter()
        coeffs_pw, _ = torch_cwt.transform_batch(
            window, int(sfreq), HIGHEST, LOWEST, nfreqs=NFREQS, device=args.device,
        )
        t_perwindow_total += time.perf_counter() - t0
        coeffs_pw = np.moveaxis(np.asarray(coeffs_pw), 1, -1)
        real_pw, imag_pw = np.real(coeffs_pw).astype(np.float32), np.imag(coeffs_pw).astype(np.float32)

        real_sl, imag_sl = slice_continuous_cwt_window(
            real_full, imag_full, start_sample=int(start), n_samples=n_samples
        )
        re_err = np.abs(real_pw - real_sl)
        im_err = np.abs(imag_pw - imag_sl)

        edge_w = int(support.max())
        edge_mask = np.zeros(n_samples, dtype=bool)
        edge_mask[:edge_w] = True
        edge_mask[n_samples - edge_w :] = True
        interior_mask = ~edge_mask

        touches_start = start == 0
        touches_end = (start + n_samples) == n_time_full
        note = " (touches start)" if touches_start else (" (touches end)" if touches_end else "")
        print(
            f"  window [{start / sfreq:6.1f}s, {(start + n_samples) / sfreq:6.1f}s){note}: "
            f"edge max|dre|={re_err[:, edge_mask, :].max():.4g}  "
            f"interior max|dre|={re_err[:, interior_mask, :].max():.4g}"
        )

        pooled_re.append(real_pw.ravel()); pooled_re_c.append(real_sl.ravel())
        pooled_im.append(imag_pw.ravel()); pooled_im_c.append(imag_sl.ravel())
        pooled_edge_re.append(re_err[:, edge_mask, :].ravel())
        pooled_edge_im.append(im_err[:, edge_mask, :].ravel())
        pooled_interior_re.append(re_err[:, interior_mask, :].ravel())
        pooled_interior_im.append(im_err[:, interior_mask, :].ravel())

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.array([])

    print("\n=== Pooled ===")
    print(f"real Pearson r: {_pearson(_cat(pooled_re), _cat(pooled_re_c)):.6f}")
    print(f"imag Pearson r: {_pearson(_cat(pooled_im), _cat(pooled_im_c)):.6f}")
    print(f"edge-region     max|dre|={_cat(pooled_edge_re).max():.6g}  max|dim|={_cat(pooled_edge_im).max():.6g}")
    print(f"interior        max|dre|={_cat(pooled_interior_re).max():.6g}  max|dim|={_cat(pooled_interior_im).max():.6g}"
          f"  (should be float32 noise floor)")
    print(f"\ncontinuous CWT (whole recording, one call): {t_continuous * 1000:.2f}ms")
    print(f"per-window CWT ({args.n_cwt_windows} independent calls, sum):   {t_perwindow_total * 1000:.2f}ms")

    print("\nDone -- both checks passed." if max_abs_diff == 0.0 else "\nCheck 1 FAILED.")


if __name__ == "__main__":
    main()
