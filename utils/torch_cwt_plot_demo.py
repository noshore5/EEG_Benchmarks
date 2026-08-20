"""Visual sanity check: compute the CWT of 2 real trials via both
utils.coherence_utils.transform (fcwt/FFTW) and utils.torch_cwt.transform
(torch.fft), and plot the magnitude scalograms side by side.

Companion to scripts/torch_cwt_parity.py (which reports the numeric
correlation/error this is meant to make visually inspectable) -- not a
pytest test, a one-off diagnostic. Lives in utils/ (not scripts/) so it's
a plain sibling import of coherence_utils/torch_cwt, no path juggling.
Run it directly:

    python utils/torch_cwt_plot_demo.py [--edf PATH] [--out PATH.png]

Needs an environment with BOTH `fcwt` and `torch` importable -- see
torch_cwt_parity.py's docstring for why that means a specific interpreter
on this dev machine, e.g.: (the OMP_NUM_THREADS/KMP_DUPLICATE_LIB_OK
workaround for fcwt/torch's colliding bundled OpenMP runtimes is set
in-process below, no need to export it yourself)

    /Users/noahshore/Documents/CoherIQs/CMPX_EEG/CMPX/bin/python3 \\
        utils/torch_cwt_plot_demo.py

Produces one PNG, one row per trial, 4 columns: raw signal, fcwt
scalogram, torch_cwt scalogram, |difference| scalogram -- with the
cone-of-influence (see cwt_gnn_classifiers.py's _coi_valid_mask) shaded
out on both scalograms so it's visually obvious which region parity
actually needs to hold over.
"""

from __future__ import annotations

import os

# Must happen before numpy/torch/fcwt are imported (each can pull in its
# own bundled OpenMP runtime -- fcwt's and torch's collide on macOS,
# crashing with "OMP: Error #179: Function pthread_mutex_init failed" /
# segfault otherwise). Only sets what isn't already set, so an explicit
# env var from the caller's shell still wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EDF = (
    Path.home() / "mne_data" / "MNE-chbmit-data" / "chbmit" / "1.0.0" / "chb01" / "chb01_01.edf"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "torch_cwt_demo.png"

# Canonical config -- Epilepsy/run_pipelines.py's _SHARED_ARCH_PARAMS.
SAMPLING_RATE = 256
LOWEST = 8.0
HIGHEST = 40.0
NFREQS = 300
WINDOW_SECONDS = 4.0
MORLET_FB = 2.0  # must match utils/torch_cwt.py's MORLET_FB

# Two trials picked for visual contrast, not randomly: one quiet stretch,
# one with a visibly larger-amplitude transient, both from real recording
# time (not synthetic), each a different channel.
TRIAL_SPECS = [
    {"channel": 0, "start_sec": 100.0, "label": "ch=0, t=100s"},
    {"channel": 4, "start_sec": 958.9, "label": "ch=4, t=958.9s"},
]


def _coi_valid_mask(freqs: np.ndarray, n_time: int) -> np.ndarray:
    """Same formula as SparseEvidenceGNNCore._coi_valid_mask
    (cwt_gnn_classifiers.py), time_offset=0 since we're plotting raw
    (un-smoothed) CWT coefficients, not the post-smoothing feature stack.
    Returns a [nfreqs, n_time] bool array (True == COI-valid)."""
    scale = SAMPLING_RATE / freqs  # [F]
    support = np.floor(MORLET_FB * scale * 3.0)  # [F]
    t_idx = np.arange(n_time)[None, :]  # [1, T]
    return (t_idx >= support[:, None]) & (t_idx < (n_time - support[:, None]))


def _load_trial(edf_path: Path, channel: int, start_sec: float) -> tuple[np.ndarray, str]:
    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - SAMPLING_RATE) > 1e-6:
        raise ValueError(f"{edf_path} is sampled at {raw.info['sfreq']} Hz, expected {SAMPLING_RATE}.")
    data = raw.get_data()  # [n_channels, n_samples], volts
    win = int(round(WINDOW_SECONDS * SAMPLING_RATE))
    start = int(round(start_sec * SAMPLING_RATE))
    trial = data[channel, start : start + win] * 1e6  # -> microvolt-scale, matches parity script
    return trial, raw.ch_names[channel]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, default=DEFAULT_EDF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        import fcwt  # noqa: F401 -- presence check; utils.coherence_utils imports it too
    except ImportError as exc:
        raise SystemExit(
            "fcwt is not importable in this interpreter -- see this script's docstring "
            "for a working interpreter to run it with."
        ) from exc

    import matplotlib.pyplot as plt

    from utils import coherence_utils, torch_cwt

    if not args.edf.is_file():
        raise SystemExit(f"EDF file not found: {args.edf}")

    n_trials = len(TRIAL_SPECS)
    fig, axes = plt.subplots(n_trials, 4, figsize=(18, 4.2 * n_trials), squeeze=False)

    for row, spec in enumerate(TRIAL_SPECS):
        trial, ch_name = _load_trial(args.edf, spec["channel"], spec["start_sec"])
        n_time = trial.shape[0]
        t_axis = np.arange(n_time) / SAMPLING_RATE

        coeffs_old, freqs_old = coherence_utils.transform(
            trial, SAMPLING_RATE, HIGHEST, LOWEST, nfreqs=NFREQS
        )
        coeffs_new, freqs_new = torch_cwt.transform(trial, SAMPLING_RATE, HIGHEST, LOWEST, nfreqs=NFREQS)
        assert coeffs_old.shape == coeffs_new.shape == (NFREQS, n_time)

        mag_old, mag_new = np.abs(coeffs_old), np.abs(coeffs_new)
        diff = np.abs(coeffs_old - coeffs_new)
        mask = _coi_valid_mask(freqs_old.astype(np.float64), n_time)  # [F, T], True == valid
        vmax = max(mag_old.max(), mag_new.max())

        ax_sig, ax_old, ax_new, ax_diff = axes[row]

        ax_sig.plot(t_axis, trial, color="black", linewidth=0.7)
        ax_sig.set_title(f"raw signal\n{spec['label']} ({ch_name})")
        ax_sig.set_xlabel("time (s)")
        ax_sig.set_ylabel(chr(956) + "V")
        ax_sig.set_xlim(t_axis[0], t_axis[-1])

        def _scalogram(ax, mag, title, vmax_=vmax):
            im = ax.imshow(
                mag,
                aspect="auto",
                origin="upper",
                extent=[t_axis[0], t_axis[-1], NFREQS - 0.5, -0.5],
                cmap="magma",
                vmin=0.0,
                vmax=vmax_,
            )
            # Shade the COI-invalid region: overlay a semi-transparent mask
            # where mask==False, so the boundary region parity doesn't need
            # to (and isn't expected to) hold over is visually obvious.
            invalid = np.where(mask, np.nan, 1.0)  # NaN over valid cells -> transparent there
            ax.imshow(
                invalid,
                aspect="auto",
                origin="upper",
                extent=[t_axis[0], t_axis[-1], NFREQS - 0.5, -0.5],
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                alpha=0.55,
            )
            ax.set_yticks(range(NFREQS))
            ax.set_yticklabels([f"{f:.1f}" for f in freqs_old])
            ax.set_title(title)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("freq (Hz)")
            return im

        im_old = _scalogram(ax_old, mag_old, "fcwt (FFTW) |coeffs|")
        _scalogram(ax_new, mag_new, "torch_cwt |coeffs|")
        im_diff = _scalogram(ax_diff, diff, "|fcwt - torch_cwt|", vmax_=diff.max())

        fig.colorbar(im_old, ax=[ax_old, ax_new], fraction=0.025, pad=0.01, label=chr(956) + "V")
        fig.colorbar(im_diff, ax=ax_diff, fraction=0.05, pad=0.02, label=chr(956) + "V")

    fig.suptitle(
        "fcwt (FFTW) vs. torch_cwt (torch.fft) -- 2 real CHB-MIT trials\n"
        "(shaded region = outside cone of influence, not compared downstream)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
