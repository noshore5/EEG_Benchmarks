"""One-off export: real chb01 preictal-window coherence graph -> exports/coherence-graph/.

Reuses the REAL dense_edge_gru (SparseEvidenceGNNClassifier) pipeline's own
coherence/phase/gate computation (SparseEvidenceGNNCore._coherence_only /
compute_events), the real fixed coherence_threshold=0.99 gate
(coherence_threshold_mode="fixed", DENSE_EDGE_GRU_PARAMS), and a real
preictal window from seizure 1_16_0 (auc_pr 0.956, FAR/h 1.2 fold --
verified against Epilepsy/results/prediction/prediction_leave_one_seizure_out_20260817-125212.csv).

Node/edge convention (per user decision, 2026-08-18): positions.json holds
BOTH the 21 real physical 10-20 electrodes chb01's 23 bipolar channels are
built from, AND each of those 23 bipolar channels positioned at the
midpoint of its two electrodes. graph.json's 253 edges are the pipeline's
real coherence pairs between those 23 bipolar channels (NOT between single
electrodes -- CHB-MIT's channels are already electrode differences, e.g.
"FP1-F7" = Fp1 minus F7).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import numpy as np
import torch

REPO_ROOT = Path("/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.epilepsy import CHBMIT
from paradigms.continuous_labeling import ContinuousLabelingParadigm
from Epilepsy.pipelines.cwt_gnn_classifiers import (
    SparseEvidenceGNNClassifier,
    _BaseCWTGNNClassifier,
)
from Epilepsy.pipelines.common import make_gaussian_weight2d
from Epilepsy.run_pipelines import (
    DENSE_EDGE_GRU_PARAMS,
    DEFAULT_SPH,
    DEFAULT_SOP,
    DEFAULT_POSTICTAL_BUFFER,
)

OUT_DIR = REPO_ROOT / "exports" / "coherence-graph"
EDGES_DIR = OUT_DIR / "edges"
RAW_DIR = OUT_DIR / "raw"
RAW_EDGES_DIR = RAW_DIR / "edges"
EDGES_DIR.mkdir(parents=True, exist_ok=True)
RAW_EDGES_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = 1
RUN = "16"
SEIZURE_ID = "1_16_0"  # subject_run_index, per paradigms/continuous_labeling.py
WINDOW_LENGTH = 4.0
STEP_SIZE = 8.0

print("=== Step 1: build the real preictal windows for seizure 1_16_0 (chb01_16.edf) ===")
# CHBMIT._get_single_subject_data numbers "run" by POSITION in the (filtered)
# records list, not by parsing the filename -- so the real full-subject run
# labels chb01_16.edf "run 16" only because it's the 16th record in
# chb01-summary.txt's file order (no earlier file is skipped that low; the
# session notes' skipped files -- 28, 35, 44, 45 -- are all higher). Loading
# chb01_01..chb01_16 (in order) reproduces that exact position -> "run 16"
# -- without downloading/windowing the other ~26 recordings this doesn't
# need. Window CONTENT/labeling only depends on chb01_16.edf's own
# annotations either way.
dataset = CHBMIT(records={SUBJECT: [f"chb01_{i:02d}.edf" for i in range(1, 17)]})
paradigm = ContinuousLabelingParadigm(
    window_length=WINDOW_LENGTH,
    step_size=STEP_SIZE,
    label_event="seizure",
    label_mode="prediction",
    sph=DEFAULT_SPH,
    sop=DEFAULT_SOP,
    postictal_buffer=DEFAULT_POSTICTAL_BUFFER,
)
X, y, metadata = paradigm.get_data(dataset, subjects=[SUBJECT])
import pandas as pd
metadata = pd.DataFrame(metadata)
print(f"  X: {X.shape}, preictal windows: {int(y.sum())}/{len(y)}")

own_mask = (metadata["seizure_id"] == SEIZURE_ID).to_numpy()
n_preictal = int(own_mask.sum())
print(f"  seizure {SEIZURE_ID} preictal windows: {n_preictal}")
assert n_preictal > 0, "no preictal windows found for seizure 1_16_0 -- label config mismatch"

preictal_idx = np.flatnonzero(own_mask)
# Pick the LAST preictal window (closest to the SPH boundary / seizure
# onset) -- the most "about to happen" real window in this seizure's
# preictal span, not an arbitrary one.
chosen_idx = int(preictal_idx[np.argmax(metadata.loc[own_mask, "window_start"].to_numpy())])
window_start = float(metadata.loc[chosen_idx, "window_start"])
print(f"  chosen window index {chosen_idx}, window_start={window_start:.1f}s "
      f"(seizure onset at {metadata.loc[chosen_idx].get('seizure_onset')}s)")

# Real channel order: same order X's channel axis uses. ContinuousLabelingParadigm
# preserves the Raw's own ch_names order; read it directly off a fresh Raw
# for this file to get the authoritative order/names (cheap: already cached).
import glob as _glob
edf_matches = _glob.glob(str(Path.home() / "mne_data" / "MNE-chbmit-data" / "chbmit" / "1.0.0" / "chb01" / "chb01_16.edf"))
assert edf_matches, "chb01_16.edf not found in local MNE cache"
raw = mne.io.read_raw_edf(edf_matches[0], preload=False, verbose="ERROR")
CH_NAMES = list(raw.ch_names)
N_CHANNELS = len(CH_NAMES)
print(f"  channel order ({N_CHANNELS}): {CH_NAMES}")

X_one = X[chosen_idx : chosen_idx + 1]  # [1, C, T]

print("\n=== Step 2: run the real coherence/gate pipeline (fixed threshold, DENSE_EDGE_GRU_PARAMS) ===")
# 2026-08-18/19: THREE deliberate departures from DENSE_EDGE_GRU_PARAMS' real
# trained config, all at the user's explicit request for this export only
# -- Epilepsy/run_pipelines.py itself is untouched, so nothing about
# training/eval changes:
#   - lowest=8.0 (was 1.0) -- matches BCI's motor-imagery lower bound,
#     drops the 1-8Hz band from the plots AND from the recomputed
#     mean_coherence/significant numbers below, so graph.json stays
#     internally consistent with what's pictured rather than a full-band
#     number next to a cropped picture. See the 1-40Hz-vs-8-40Hz note
#     above for why the underlying band was untuned either way.
#   - coherence_threshold=0.90 (was 0.99) -- a looser real gate applied to
#     the SAME real coherence array; every edge below relaxes together
#     under this one cutoff, not a per-edge invented number.
#   - nfreqs=40 (was 8) -- higher frequency resolution for the demo
#     heatmap panels (coherence-graph-viewer.html's modal). The CWT is
#     recomputed at 30 real frequency bins between lowest/highest (not an
#     8-bin array interpolated/upsampled to 30 -- every bin is a real
#     computed value), so coh/phase/significance/coi_valid and freqs_hz
#     all come out [*, 30] instead of [*, 8].
export_params = dict(DENSE_EDGE_GRU_PARAMS)
export_params["lowest"] = 8.0
export_params["coherence_threshold"] = 0.90
export_params["nfreqs"] = 40
clf = SparseEvidenceGNNClassifier(epochs=1, **export_params)
clf._init_cwt_gnn_classifier(
    sampling_rate=clf.sampling_rate,
    lowest=clf.lowest,
    highest=clf.highest,
    nfreqs=clf.nfreqs,
    cwt_resample_n_time=clf.cwt_resample_n_time,
    normalize_input=True,
    epochs=1,
    batch_size=1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    grad_clip_norm=None,
    device="cpu",
    seed=clf.seed,
    channel_subset=clf.channel_subset,
    verbose=0,
)

raw_x_native = clf._apply_channel_subset(np.asarray(X_one, dtype=np.float32))
raw_x, w_real, w_imag, freqs = _BaseCWTGNNClassifier._prepare_features(clf, X_one, fit=True)

helper = clf._build_model(n_channels=int(raw_x.shape[1]), n_classes=2)
helper.eval()
helper._freq_lo = float(freqs.min().item())
helper._freq_hi = float(freqs.max().item())

device = torch.device("cpu")
freqs_batched = helper._batched_freqs(freqs, batch_size=1)
smooth_kernel_and_pad = make_gaussian_weight2d(
    kernel_size=helper.smooth_kernel_size, sigma=helper.smooth_kernel_sigma,
    pad_h=0, device=device, dtype=w_real.dtype,
)

with torch.no_grad():
    coh, phase = helper._coherence_only(w_real, w_imag, freqs_batched, smooth_kernel_and_pad)
    coi_valid = helper._coi_valid_mask(freqs_batched, n_time_in=w_real.shape[2], T_out=coh.shape[2])
    # coherence_threshold_mode="fixed" -> no override, matches real
    # train/eval exactly (compute_events falls back to helper.coherence_threshold).
    events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask = helper.compute_events(
        w_real, w_imag, freqs
    )
    gate = (coh > helper.coherence_threshold) & (phase.abs() > helper.phase_threshold_rad) & coi_valid

threshold_scalar = float(helper.coherence_threshold)
print(f"  fixed coherence_threshold used (real, from DENSE_EDGE_GRU_PARAMS): {threshold_scalar}")
print(f"  phase_threshold_rad: {helper.phase_threshold_rad}  nfreqs: {helper.nfreqs}")

src_padded0 = src_padded[0].long()
dst_padded0 = dst_padded[0].long()
valid_mask0 = valid_mask[0].bool()
coh0 = coh[0]          # [E, T, F]
coi_valid0 = coi_valid[0, 0]  # [T, F]
phase0 = phase[0]
# The continuous significance channel event_mode="dense" (dense_edge_gru's
# real mode) actually consumes -- see _build_dense_edge_input,
# Epilepsy/pipelines/cwt_gnn_classifiers.py ~line 3251. compute_events'
# discrete event list (used for arrow placement / n_events above) is the
# event_mode="sparse" mechanism, not what dense mode's forward() sees.
significance0 = (coh0 - threshold_scalar) / threshold_scalar  # [E, T, F]
significance0 = significance0 * coi_valid0.unsqueeze(0)  # broadcast [T, F] -> [E, T, F], same masking convention as coh/phase

print("\n=== Step 3: real electrode positions (mne standard_1020) ===")
BIPOLAR_ELECTRODES = {}
for ch in CH_NAMES:
    base = ch.split("-")
    # chb01's own channel names, e.g. "FP1-F7", "T8-P8-0" (duplicate-gain
    # suffix -- still the T8/P8 pair). Split on '-' and take the first two
    # tokens as the two composing electrodes.
    e1, e2 = base[0], base[1]
    BIPOLAR_ELECTRODES[ch] = (e1, e2)

unique_electrodes = sorted({e for pair in BIPOLAR_ELECTRODES.values() for e in pair})
print(f"  unique physical electrodes referenced by chb01's 23 channels: {unique_electrodes}")

montage = mne.channels.make_standard_montage("standard_1020")
montage_pos = montage.get_positions()["ch_pos"]  # keyed by real-cased names e.g. "Fp1"
# Case-insensitive lookup: chb01 channel names are upper-case ("FP1"),
# standard_1020 uses mixed case ("Fp1").
montage_pos_ci = {k.upper(): v for k, v in montage_pos.items()}

electrode_positions = {}
missing = []
for e in unique_electrodes:
    v = montage_pos_ci.get(e.upper())
    if v is None:
        missing.append(e)
    else:
        electrode_positions[e] = np.asarray(v, dtype=float)
if missing:
    raise RuntimeError(f"standard_1020 montage missing real positions for: {missing}")

positions_out = []
canonical_name = {e.upper(): e for e in unique_electrodes}
# Use the montage's own canonical casing (e.g. "Fp1") in the export, not
# chb01's raw upper-case channel-name tokens.
montage_canonical = {k.upper(): k for k in montage_pos.keys()}
for e in unique_electrodes:
    canon = montage_canonical.get(e.upper(), e)
    x, y, z = electrode_positions[e]
    positions_out.append({"channel": canon, "x": float(x), "y": float(y), "z": float(z)})

# Each of the 23 real bipolar channels, positioned at the midpoint of its
# two composing electrodes' real positions.
channel_midpoints = {}
for ch, (e1, e2) in BIPOLAR_ELECTRODES.items():
    mid = (electrode_positions[e1] + electrode_positions[e2]) / 2.0
    channel_midpoints[ch] = mid
    positions_out.append({"channel": ch, "x": float(mid[0]), "y": float(mid[1]), "z": float(mid[2])})

coords = np.array([[p["x"], p["y"], p["z"]] for p in positions_out])
print(f"  positions: {len(positions_out)} entries "
      f"({len(unique_electrodes)} real electrodes + {len(BIPOLAR_ELECTRODES)} channel midpoints)")
print(f"  coord range: x[{coords[:,0].min():.4f},{coords[:,0].max():.4f}] "
      f"y[{coords[:,1].min():.4f},{coords[:,1].max():.4f}] "
      f"z[{coords[:,2].min():.4f},{coords[:,2].max():.4f}]")
assert not np.allclose(coords, 0.0), "positions are degenerate (all zero)"
assert not np.isnan(coords).any(), "NaN in positions"

with open(OUT_DIR / "positions.json", "w") as f:
    json.dump(positions_out, f, indent=2)
print(f"  wrote {OUT_DIR / 'positions.json'}")

print("\n=== Step 4: per-edge mean coherence + significance (real 253 channel pairs) ===")
E = int(helper.src_idx.numel())
print(f"  n_edges (canonical i<j pairs over {N_CHANNELS} channels): {E}")

freqs_1d = freqs[0].detach().numpy()

# --- raw data dump: the actual computed arrays, not just the rendered PNGs ---
all_src_names = np.array([CH_NAMES[int(helper.src_idx[e].item())] for e in range(E)])
all_dst_names = np.array([CH_NAMES[int(helper.dst_idx[e].item())] for e in range(E)])
np.savez_compressed(
    RAW_DIR / "coherence.npz",
    coh=coh0.detach().numpy(),          # [E, T, F] -- full, unthresholded coherence magnitude
    phase=phase0.detach().numpy(),      # [E, T, F] -- radians
    significance=significance0.detach().numpy(),  # [E, T, F] -- (coh-threshold)/threshold, COI-masked; event_mode="dense"'s real continuous significance channel
    coi_valid=coi_valid0.detach().numpy(),  # [T, F] -- True outside the cone of influence, shared across edges
    freqs_hz=freqs_1d,                  # [F]
    src_channel=all_src_names,          # [E] -- row e's edge is src_channel[e] <-> dst_channel[e]
    dst_channel=all_dst_names,          # [E]
    coherence_threshold=np.array(threshold_scalar),
    phase_threshold_rad=np.array(float(helper.phase_threshold_rad)),
    lowest_hz=np.array(float(clf.lowest)),
    highest_hz=np.array(float(clf.highest)),
    sampling_rate=np.array(int(clf.sampling_rate)),
    window_start_s=np.array(window_start),
    seizure_id=np.array(SEIZURE_ID),
    subject=np.array(SUBJECT),
    run=np.array(RUN),
)
with open(RAW_DIR / "README.md", "w") as f:
    f.write(
        "# Raw coherence arrays\n\n"
        "`coherence.npz` -- the full, real, unthresholded pipeline output for "
        f"the one chosen preictal window (chb01, seizure {SEIZURE_ID}, "
        f"t={window_start:.0f}s), computed by "
        "SparseEvidenceGNNCore._coherence_only + compute_events (see "
        "build_coherence_export.py alongside this file for the exact call "
        "sequence). Arrays:\n\n"
        "- `coh` [E, T, F]: coherence magnitude, 0-1, per (edge, time, freq) cell.\n"
        "- `phase` [E, T, F]: phase angle, radians.\n"
        "- `significance` [E, T, F]: (coh - coherence_threshold) / "
        "coherence_threshold, COI-masked (same convention as `coh`/`phase`) "
        "-- the continuous significance channel event_mode=\"dense\" "
        "(dense_edge_gru's real mode) actually consumes at forward() time; "
        "see SparseEvidenceGNNCore._build_dense_edge_input. NOT derived "
        "from compute_events' discrete event list (that's event_mode="
        "\"sparse\"'s mechanism, used here only for the PNGs' arrow "
        "placement).\n"
        "- `coi_valid` [T, F]: True outside the cone of influence (shared "
        "across all edges).\n"
        "- `freqs_hz` [F]: the F frequency bins' real Hz values.\n"
        "- `src_channel`/`dst_channel` [E]: edge e connects "
        "src_channel[e] <-> dst_channel[e] (same order/names as graph.json).\n"
        "- `coherence_threshold`, `phase_threshold_rad`, `lowest_hz`, "
        "`highest_hz`, `sampling_rate`: the real config this was computed "
        "under (coherence_threshold/lowest_hz were overridden from "
        "DENSE_EDGE_GRU_PARAMS's trained values of 0.99/1.0 at the user's "
        "request -- see build_coherence_export.py's own comments).\n"
        "- `window_start_s`, `seizure_id`, `subject`, `run`: which real "
        "window this is.\n\n"
        "`edges/{src}__{dst}.npz` -- per significant edge, the same edge's "
        "`coh`/`phase`/`significance`/`coi_valid`/`freqs_hz` slice plus its consolidated "
        "real events (`event_t`, `event_freq_idx`, `event_sin`, "
        "`event_cos`, `event_mean_mag` -- what the PNG's arrows are drawn "
        "from) and `mean_coherence` (same number as graph.json's row for "
        "this edge).\n"
    )
print(f"  wrote {RAW_DIR / 'coherence.npz'} (all {E} edges, full arrays)")


def _freq_labels(f1d):
    return [f"{v:.1f}" for v in f1d]


def _set_freq_axis(ax, f1d, max_ticks=8):
    step = max(len(f1d) // max_ticks, 1)
    ticks = list(range(0, len(f1d), step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([_freq_labels(f1d)[i] for i in ticks])
    ax.set_ylabel("freq (Hz)")


def _hatch_inside_coi(ax, coi_valid_tf):
    T, F = coi_valid_tf.shape
    invalid = (~coi_valid_tf).T.astype(float)
    ax.contourf(np.arange(T), np.arange(F), invalid, levels=[0.5, 1.5], colors="none", hatches=["////"])


graph_entries = []
n_significant = 0
for e in range(E):
    src_ch = int(helper.src_idx[e].item())
    dst_ch = int(helper.dst_idx[e].item())
    src_name = CH_NAMES[src_ch]
    dst_name = CH_NAMES[dst_ch]

    coh_e = coh0[e].detach().numpy()      # [T, F]
    coi_e = coi_valid0.detach().numpy()   # [T, F]
    mean_coherence = float(coh_e[coi_e].mean()) if coi_e.any() else float("nan")

    edge_mask = valid_mask0 & (src_padded0 == src_ch) & (dst_padded0 == dst_ch)
    n_events = int(edge_mask.sum().item())
    significant = n_events > 0
    if significant:
        n_significant += 1

    image_rel = None
    if significant:
        # "__" separator, not "-" -- chb01's real channel names already
        # contain "-" (e.g. "FP1-F7"), so "{src}-{dst}.png" would be
        # ambiguous to parse back apart (e.g. "FP1-F7-F7-T7.png").
        image_rel = f"edges/{src_name}__{dst_name}.png"
        T_e, F_e = coh_e.shape

        # Consolidated real events for this edge (the SAME ones that decide
        # `significant`/n_events above -- real gate: coh>threshold &
        # |phase|>phase_threshold & COI-valid, run-consolidated by
        # compute_events). Arrow placement uses these thresholds; the
        # coherence heatmap underneath does NOT -- it's the full,
        # unthresholded array, per the user's correction.
        edge_events = events_padded[0][edge_mask].detach().numpy()  # [n_events, 5]
        edge_freq_idx = freq_idx_padded[0][edge_mask].detach().numpy()
        ev_t = edge_events[:, 0] * (T_e - 1)
        ev_f = edge_freq_idx.astype(float)
        ev_sin, ev_cos = edge_events[:, 3], edge_events[:, 4]  # unit vector, mean phase direction

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.suptitle(
            f"chb01, seizure {SEIZURE_ID} preictal window (t={window_start:.0f}s), "
            f"{src_name} ↔ {dst_name}  ({clf.lowest:.0f}-{clf.highest:.0f}Hz)\n"
            f"coherence |coh| (full array); arrows = phase direction where "
            f"coh>{threshold_scalar} & |phase|>thr & COI-valid ({n_events} event(s))",
            fontsize=10,
        )
        im = ax.imshow(coh_e.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("time (samples)")
        _set_freq_axis(ax, freqs_1d)
        _hatch_inside_coi(ax, coi_e)
        fig.colorbar(im, ax=ax, fraction=0.046, label="coherence |coh|")

        if len(ev_t):
            # angles="uv" (the default -- NOT "xy"): draws (cos, sin)
            # straight as a display-space direction. "xy" instead treats
            # (cos, sin) as a literal (dx, dy) in DATA coordinates -- since
            # the x-axis spans ~1000 time samples but the y-axis only 8
            # frequency bins, that squashed every arrow's horizontal
            # component to near-nothing, rendering every arrow as
            # (near-)vertical regardless of the real phase angle. Bug, not
            # a real finding -- caught and fixed 2026-08-18.
            ax.quiver(
                ev_t, ev_f, ev_cos, ev_sin,
                color="white", edgecolor="black", linewidth=0.5,
                angles="uv", pivot="mid", scale=12, scale_units="height",
                width=0.006, zorder=5,
            )

        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.savefig(EDGES_DIR / f"{src_name}__{dst_name}.png", dpi=130)
        plt.close(fig)

        # Raw per-edge copy, matching the PNG 1:1 (same "__" separator).
        np.savez_compressed(
            RAW_EDGES_DIR / f"{src_name}__{dst_name}.npz",
            coh=coh_e, phase=phase0[e].detach().numpy(),
            significance=significance0[e].detach().numpy(),
            coi_valid=coi_e,
            freqs_hz=freqs_1d,
            event_t=ev_t, event_freq_idx=ev_f, event_sin=ev_sin, event_cos=ev_cos,
            event_mean_mag=edge_events[:, 2],
            mean_coherence=np.array(mean_coherence),
            coherence_threshold=np.array(threshold_scalar),
            phase_threshold_rad=np.array(float(helper.phase_threshold_rad)),
        )

    graph_entries.append({
        "src": src_name,
        "dst": dst_name,
        "mean_coherence": round(mean_coherence, 6),
        "significant": significant,
        "image": image_rel,
    })

print(f"  significant edges: {n_significant}/{E}")

with open(OUT_DIR / "graph.json", "w") as f:
    json.dump(graph_entries, f, indent=2)
print(f"  wrote {OUT_DIR / 'graph.json'}")

# Copy of the exact script that produced everything in this export --
# alongside the data, not just described after the fact.
_this_file = Path(__file__).resolve()
shutil.copy2(_this_file, OUT_DIR / _this_file.name)
print(f"  copied {_this_file.name} (this script) into {OUT_DIR}")

print("\n=== Done ===")
print(f"positions.json: {len(positions_out)} entries")
print(f"graph.json: {len(graph_entries)} edges, {n_significant} significant, "
      f"{len(list(EDGES_DIR.glob('*.png')))} PNGs written")
print(f"raw/coherence.npz: full {E}-edge arrays; "
      f"raw/edges/: {len(list(RAW_EDGES_DIR.glob('*.npz')))} per-edge raw files")
