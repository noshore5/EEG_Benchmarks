"""
Debug / visualization harness for SparseEvidenceGNNCore under the CURRENT
canonical config -- coherence_threshold_mode="surrogate" -- i.e. the
per-(edge, frequency) phase-randomization-surrogate-calibrated significance
threshold that replaced the fixed coherence_threshold magnitude cutoff (see
SparseEvidenceGNNClassifier's coherence_threshold_mode docstring and
run_pipelines.py's EVOLVING_GRAPH_PARAMS, the pipeline this debug script's
config is aligned to).

This is the surrogate-mode successor to the earlier fixed-threshold
debug_plots/edge0_to_messages_native_coi.png-style figure: same one-edge,
coherence -> gate -> consolidated-events -> GNN-message story, but with the
gate's coherence half now a per-(edge, freq) surrogate threshold instead of
one flat number everywhere, plus a panel showing that threshold curve itself.

Every number plotted is read off the REAL pipeline objects (SparseEvidenceGNNCore
/ SparseEvidenceGNNClassifier methods, called directly, not reimplemented) --
_coherence_only, _coi_valid_mask, _surrogate_coherence_threshold, and
compute_events() are the same methods _precompute_sparse_events calls during
real training/eval.

USAGE
-----
    python -m coheriqs_contributions.moabb_pipelines.debug_sparse_evidence_gnn \\
        --trial 0 --out coheriqs_contributions/debug_plots/surrogate_pipeline/

(or `python coheriqs_contributions/moabb_pipelines/debug_sparse_evidence_gnn.py ...`
from the repo root -- the sys.path shim below makes both work.)

The first run may take a while if this exact trial/config combination isn't
already in the surrogate null cache (~/mne_data/surrogate_null_cache by
default) -- surrogate_count=100 phase-randomized surrogates have to be run
through the full CWT -> cross-spectrum -> smoothing pipeline once. Subsequent
runs (or runs against a trial any prior canonical-sparse run already touched)
are a cache hit and return almost instantly.

2026-08-09: edges are now SparseEvidenceGNNCore's canonical (i<j) undirected
pairing (36 for the 9-channel motor subset, was 72 directed i->j/j->i edges)
-- see that module's docstring. `--edge` now indexes 0..35; there is no
longer a separate "reverse" edge to plot for a given channel pair (panel 5's
gate is |phase|>threshold now, and panel 6's per-event arrows already show
each event's own direction via sign(mean_angle), so both directions of a
pair's activity show up on the one figure for that edge).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

_PKG_ROOT = Path(__file__).resolve().parents[2]  # .../moabb
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

try:
    from coheriqs_contributions.moabb_pipelines.common import make_gaussian_weight2d
    from coheriqs_contributions.moabb_pipelines.sparse_evidence_gnn_classifier import (
        SparseEvidenceGNNClassifier,
    )
    from coheriqs_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
        _BaseCWTGNNClassifier,
    )
    from coheriqs_contributions.run_pipelines import EVOLVING_GRAPH_PARAMS
except ModuleNotFoundError:
    from moabb_pipelines.common import make_gaussian_weight2d
    from moabb_pipelines.sparse_evidence_gnn_classifier import SparseEvidenceGNNClassifier
    from moabb_pipelines.xwt_phase_gnn_classifier import _BaseCWTGNNClassifier
    from run_pipelines import EVOLVING_GRAPH_PARAMS

# Built from run_pipelines.py's EVOLVING_GRAPH_PARAMS (the surrogate-mode
# pipeline this debug script's config is aligned to) rather than a second
# hand-maintained copy. Read back through the classifier's own get_params()
# so keys EVOLVING_GRAPH_PARAMS leaves at their class default (e.g.
# smooth_kernel_sigma) still resolve correctly. Only a training-only subset
# of keys is relevant here (this script builds a throwaway model directly,
# not a full SparseEvidenceGNNClassifier.fit(), so epochs/batch_size/
# optimizer kwargs would be accepted but ignored -- left out for clarity).
_DEBUG_RELEVANT_KEYS = (
    "sampling_rate", "lowest", "highest", "nfreqs", "cwt_resample_n_time",
    "coherence_threshold", "phase_threshold_deg", "channel_encoder_dilation",
    "hidden_dim", "channel_embed_dim", "smooth_kernel_size",
    "smooth_kernel_sigma", "coi_enabled", "channel_subset",
    "coherence_threshold_mode", "surrogate_count", "surrogate_percentile",
    "surrogate_seed", "seed",
)
_full_params = SparseEvidenceGNNClassifier(**EVOLVING_GRAPH_PARAMS).get_params()
CANONICAL_SPARSE_KWARGS = {key: _full_params[key] for key in _DEBUG_RELEVANT_KEYS}
# This is an interactive/debugging tool where a silent ~10s (or longer, on a
# true cache miss) hang is the wrong default -- verbose=1 turns on
# SparseEvidenceGNNClassifier's existing progress instrumentation
# (_surrogate_null_percentile_grid's per-chunk tqdm bar), so a cache miss
# shows live progress instead of nothing until it's done. Overrides
# SPARSE_EVIDENCE_GNN_PARAMS's verbose=2 (this script doesn't want the full
# training-loop console output that setting implies elsewhere, since it
# never trains a model).
CANONICAL_SPARSE_KWARGS["verbose"] = 1


def _freq_labels(freqs_1d: np.ndarray) -> list[str]:
    return [f"{v:.1f}" for v in freqs_1d]


def _set_freq_axis(ax, freqs_1d: np.ndarray, max_ticks: int = 8):
    step = max(len(freqs_1d) // max_ticks, 1)
    ticks = list(range(0, len(freqs_1d), step))
    ax.set_yticks(ticks)
    ax.set_yticklabels(_freq_labels(freqs_1d)[i] for i in ticks)
    ax.set_ylabel("freq (Hz)")


def _unique_path(path: Path) -> Path:
    """Returns `path` unchanged if nothing exists there yet, otherwise the
    first `<stem>_2<suffix>`, `<stem>_3<suffix>`, ... that doesn't -- so
    re-running this script (e.g. after a code change, to compare against a
    previous figure for the same subject/trial/edge/phase) never clobbers an
    existing PNG."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _hatch_inside_coi(ax, coi_valid_tf: np.ndarray):
    """Hatches the region INSIDE the cone of influence (coi_valid == False),
    matching the earlier fixed-threshold figure's "hatched=inside COI"
    convention. `coi_valid_tf` is [T, F], True where outside COI (valid)."""
    T, F = coi_valid_tf.shape
    invalid = (~coi_valid_tf).T.astype(float)  # [F, T]
    ax.contourf(
        np.arange(T), np.arange(F), invalid,
        levels=[0.5, 1.5], colors="none", hatches=["////"],
    )


# ---------------------------------------------------------------------------
# 1. Load real data + run it through the REAL current-canonical pipeline
# ---------------------------------------------------------------------------
def load_pipeline_and_trial(trial_idx: int = 0, subject: int = 2, phase_threshold_deg: float | None = None):
    """Loads real BNCI2014-001 data for one subject and runs it through the
    REAL SparseEvidenceGNNClassifier's own `_prepare_features` (no separate
    CWT/normalization reimplementation here), at the current canonical
    surrogate-mode config. Returns everything needed to inspect one trial's
    coherence -> gate -> event pipeline.

    `phase_threshold_deg`, if given, overrides CANONICAL_SPARSE_KWARGS's own
    value (see run_pipelines.py's EVOLVING_GRAPH_PARAMS for the current
    canonical setting) -- e.g. to compare against a value being swept
    elsewhere.
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=35)
    # return_epochs=True so we get the real 10-20 electrode names
    # (epochs.info['ch_names']) in the exact same channel order the
    # returned array's channel axis uses -- no separate/guessable montage
    # lookup, so channel_subset indices always resolve to the right name.
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset, subjects=[subject], return_epochs=True
    )
    ch_names_all = list(epochs.info["ch_names"])
    # return_epochs=True skips moabb's own array pipeline, which (for the
    # return_epochs=False path used everywhere else in this pipeline) always
    # multiplies epochs.get_data() by dataset.unit_factor (1e6 for
    # BNCI2014-001, Volts -> microvolts) -- see
    # BaseParadigm._get_array_pipeline in moabb/paradigms/base.py. Skipping
    # it silently shrank the raw signal ~1e6x, which produced a degenerate
    # (all-zero) surrogate threshold. Applying it by hand here keeps this
    # script's numbers on the same scale as every other run.
    X = epochs.get_data() * dataset.unit_factor

    kwargs = dict(CANONICAL_SPARSE_KWARGS)
    if phase_threshold_deg is not None:
        kwargs["phase_threshold_deg"] = float(phase_threshold_deg)
    # verbose comes from CANONICAL_SPARSE_KWARGS now (not hardcoded here) so
    # its progress-bar setting stays a single source of truth -- passing it
    # both ways would be a TypeError (duplicate keyword via **kwargs).
    clf = SparseEvidenceGNNClassifier(epochs=1, batch_size=1, **kwargs)
    # `_prepare_features` / `_build_model` need attrs normally set up by
    # `_init_torch_classifier` (self.verbose, self.transform_, self.seed,
    # ...) -- calling `_init_cwt_gnn_classifier` once ensures those exist
    # without duplicating its logic (same pattern debug_wct_evidence_gnn.py
    # uses for the windowed pipeline).
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
        # _init_torch_classifier (called from within _init_cwt_gnn_classifier)
        # does `self.verbose = verbose`, unconditionally overwriting whatever
        # the constructor above set -- hardcoding 0 here silently undid
        # CANONICAL_SPARSE_KWARGS's verbose=1 and killed the progress bar.
        # Carry the constructed value through instead.
        verbose=clf.verbose,
    )

    X_one = X[trial_idx : trial_idx + 1]  # keep batch dim: [1, C, T]
    # Channel-subset-applied but NOT z-score-normalized -- this is what the
    # surrogate cache actually hashes on (see
    # SparseEvidenceGNNClassifier._prepare_features's raw_x_native comment).
    raw_x_native = clf._apply_channel_subset(np.asarray(X_one, dtype=np.float32))
    # Deliberately the BASE (_BaseCWTGNNClassifier) _prepare_features, not
    # SparseEvidenceGNNClassifier's own override -- the subclass override
    # additionally runs the full event-precompute and returns
    # (raw_x, events_padded, src_padded, dst_padded, valid_mask) instead of
    # (raw_x, w_real, w_imag, freqs). We want the latter here so this script
    # can drive the coherence/gate/event stages itself and capture
    # intermediates for one edge.
    raw_x, w_real, w_imag, freqs = _BaseCWTGNNClassifier._prepare_features(
        clf, X_one, fit=True
    )

    n_channels = int(raw_x.shape[1])
    helper = clf._build_model(n_channels=n_channels, n_classes=2)
    helper.eval()
    helper._freq_lo = float(freqs.min().item())
    helper._freq_hi = float(freqs.max().item())

    return clf, helper, raw_x_native, w_real, w_imag, freqs, str(labels[trial_idx]), ch_names_all


# ---------------------------------------------------------------------------
# 2. Run the real (non-trainable) coherence -> gate -> event pipeline
# ---------------------------------------------------------------------------
def compute_pipeline_stage(clf, helper, raw_x_native, w_real, w_imag, freqs):
    """Calls the pieces of the REAL pipeline directly (same convention the
    earlier fixed-threshold debug_plots/edge0_*.png scripts used) -- nothing
    here reimplements the coherence/gate/threshold math, it just captures
    the intermediates SparseEvidenceGNNCore.compute_events() already
    computes internally."""
    device = torch.device("cpu")
    freqs_batched = helper._batched_freqs(freqs, batch_size=1)
    smooth_kernel_and_pad = make_gaussian_weight2d(
        kernel_size=helper.smooth_kernel_size,
        sigma=helper.smooth_kernel_sigma,
        pad_h=0,
        device=device,
        dtype=w_real.dtype,
    )

    # Real trial's coherence + phase -- identical math _build_sparse_events
    # itself uses (see SparseEvidenceGNNCore._coherence_only's docstring).
    coh, phase = helper._coherence_only(w_real, w_imag, freqs_batched, smooth_kernel_and_pad)
    coi_valid = helper._coi_valid_mask(
        freqs_batched, n_time_in=w_real.shape[2], T_out=coh.shape[2]
    )

    # The REAL surrogate-calibration call (coherence_threshold_mode=
    # "surrogate") -- not a reimplementation. Cache-hits near-instantly if
    # this trial/config combo was already calibrated by an earlier run
    # against ~/mne_data/surrogate_null_cache (e.g. a prior canonical-sparse
    # run touching this same subject/trial).
    rng = np.random.default_rng(int(clf.surrogate_seed))
    threshold = clf._surrogate_coherence_threshold(
        helper, raw_x_native[0], smooth_kernel_and_pad, rng, device
    )  # [1, E, 1, F]

    # The REAL event-building call, with that same per-(edge,freq) threshold
    # -- this is exactly what _precompute_sparse_events does per trial.
    events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask = helper.compute_events(
        w_real, w_imag, freqs, coherence_threshold_override=threshold,
    )

    # phase.abs() (two-sided), not phase> alone -- matches the REAL gate in
    # SparseEvidenceGNNCore._build_sparse_events. Edges are the canonical
    # (i<j) undirected pairing as of 2026-08-09 (see that module's docstring),
    # so a single edge can legitimately carry both directions' activity at
    # different (t, freq) cells -- direction lives in sign(mean_angle), not
    # in which of two duplicate edges fired.
    gate = (coh > threshold) & (phase.abs() > helper.phase_threshold_rad) & coi_valid
    # Raw per-CHANNEL wavelet magnitude (not per-edge coherence) -- w_real/
    # w_imag are [B, C, T, F] single-channel CWT coefficients (see
    # xwt_phase_gnn_classifier.py's src_r = w_real[:, self.src_idx, t, :]
    # indexing), so sqrt(real^2+imag^2) is each channel's own CWT power,
    # independent of any other channel. Kept at native (unsmoothed) T/F
    # resolution -- this answers "is there mu-band activity on THIS channel
    # at all", which coherence (a cross-channel, smoothed quantity) can't:
    # a channel can have strong mu-band power yet show low coherence with
    # its partner if the two aren't phase-locked.
    cwt_mag = torch.sqrt(w_real**2 + w_imag**2)
    return {
        "coh": coh, "phase": phase, "coi_valid": coi_valid, "threshold": threshold,
        "gate": gate, "events_padded": events_padded, "src_padded": src_padded,
        "dst_padded": dst_padded, "freq_idx_padded": freq_idx_padded,
        "valid_mask": valid_mask, "cwt_mag": cwt_mag,
    }


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------
def plot_pipeline_figure(
    clf, helper, stage, freqs, subject: int, trial_idx: int, trial_label: str, edge_idx: int, out_path: Path,
    ch_names_all: list[str] | None = None,
):
    freqs_1d = freqs[0].detach().numpy()
    coh = stage["coh"][0, edge_idx].detach().numpy()       # [T, F]
    phase = stage["phase"][0, edge_idx].detach().numpy()   # [T, F]
    coi_valid = stage["coi_valid"][0, 0].detach().numpy()  # [T, F], broadcasts over edges
    threshold_edge = stage["threshold"][0, edge_idx, 0].detach().numpy()  # [F]
    gate = stage["gate"][0, edge_idx].detach().numpy()     # [T, F]
    T, F = coh.shape

    src_ch = int(helper.src_idx[edge_idx].item())
    dst_ch = int(helper.dst_idx[edge_idx].item())
    subset = clf.channel_subset
    # Real 10-20 electrode names (e.g. "FC3"), not raw channel_subset
    # indices -- subset[src_ch]/subset[dst_ch] are indices into the FULL
    # (pre-subset) channel list, which ch_names_all is keyed on.
    # NOTE (2026-08-09): edges are the canonical (i<j) UNDIRECTED pairing --
    # src_ch/dst_ch here is a fixed display convention, not a physical
    # direction. This edge carries BOTH directions' real activity, at
    # different (t, freq) cells, distinguished by sign(mean_angle) (see
    # panel 6's arrows) -- there is no separate "reverse" edge to plot.
    if ch_names_all is not None and subset is not None:
        src_label = ch_names_all[subset[src_ch]]
        dst_label = ch_names_all[subset[dst_ch]]
    else:
        src_label = subset[src_ch] if subset is not None else src_ch
        dst_label = subset[dst_ch] if subset is not None else dst_ch

    events_padded = stage["events_padded"][0]  # [max_count, 5]
    src_padded = stage["src_padded"][0].long()
    dst_padded = stage["dst_padded"][0].long()
    valid_mask = stage["valid_mask"][0].bool()
    edge_mask = valid_mask & (src_padded == src_ch) & (dst_padded == dst_ch)
    edge_events = events_padded[edge_mask].detach().numpy()  # [n_events, 5]
    n_events = edge_events.shape[0]
    # event_features columns: [timestamp01, freq01, mean_mag, sin(angle), cos(angle)]
    ev_t = edge_events[:, 0] * (T - 1) if n_events else np.array([])
    ev_freq_val = edge_events[:, 1] * (helper._freq_hi - helper._freq_lo) + helper._freq_lo
    ev_f = (
        np.array([np.argmin(np.abs(freqs_1d - v)) for v in ev_freq_val]) if n_events else np.array([])
    )
    ev_mag = edge_events[:, 2] if n_events else np.array([])
    ev_sin, ev_cos = (edge_events[:, 3], edge_events[:, 4]) if n_events else (np.array([]), np.array([]))
    ev_angle = np.arctan2(ev_sin, ev_cos) if n_events else np.array([])

    # Raw per-channel CWT magnitude for the two channels THIS edge connects
    # -- unlike coh (a cross-channel, smoothed quantity), this shows whether
    # each channel has mu-band power on its own at all.
    mag_src = stage["cwt_mag"][0, src_ch].detach().numpy()  # [T_raw, F]
    mag_dst = stage["cwt_mag"][0, dst_ch].detach().numpy()
    mag_vmax = max(float(mag_src.max()), float(mag_dst.max()), 1e-12)
    mu_band_hi_hz = 13.0  # canonical mu/alpha upper bound; CANONICAL_SPARSE_KWARGS's lowest=8.0 is already the mu-band floor
    # Fractional bin index where freqs_1d crosses mu_band_hi_hz, for an
    # axhline on the same integer-bin-index y-axis _set_freq_axis uses.
    # freqs_1d comes back DESCENDING from the CWT transform (index 0 ~=
    # highest freq e.g. 35Hz, index F-1 ~= lowest e.g. 8Hz -- verified
    # empirically, not just assumed) -- np.interp requires its xp array
    # increasing or it silently returns nonsense (no error), so sort first
    # rather than assuming either direction.
    _order = np.argsort(freqs_1d)
    mu_band_hi_bin = float(np.interp(mu_band_hi_hz, freqs_1d[_order], np.arange(F)[_order]))
    in_mu_band = freqs_1d <= mu_band_hi_hz
    # coi_valid is at coh/phase's (T, F) resolution, NOT cwt_mag's native
    # (T_raw, F) one -- _smooth's time-domain convolution is VALID
    # (unpadded), so coh/phase's time axis is shorter than w_real/w_imag's
    # by 2*helper._time_offset_samples() (that many raw samples trimmed off
    # each end; see that method's docstring). A prior version of this script
    # assumed the two matched whenever cwt_resample_n_time=None and silently
    # fell back to NO COI masking (all-True) the rest of the time -- which
    # is exactly the canonical config actually used here (verified: T_raw
    # was 1001, coi_valid's T was 997, a silent 4-sample mismatch from
    # smooth_kernel_size[0]=5). Rebuild a raw-sample-space COI mask instead:
    # place coi_valid's real values at the offset where they apply, and
    # mark the trimmed-off leading/trailing edge (no smoothed coherence
    # computed there at all) invalid too, since that's the correct read on
    # "not COI-valid" for a native-resolution sample the pipeline never
    # evaluates.
    time_offset = helper._time_offset_samples()
    T_raw = mag_src.shape[0]
    coi_for_mag = np.zeros((T_raw, F), dtype=bool)
    coi_for_mag[time_offset : time_offset + coi_valid.shape[0], :] = coi_valid
    def _band_means(mag):
        valid_mu = coi_for_mag & in_mu_band[np.newaxis, :]
        valid_hi = coi_for_mag & ~in_mu_band[np.newaxis, :]
        mu_mean = float(mag[valid_mu].mean()) if valid_mu.any() else float("nan")
        hi_mean = float(mag[valid_hi].mean()) if valid_hi.any() else float("nan")
        return mu_mean, hi_mean
    mu_mean_src, hi_mean_src = _band_means(mag_src)
    mu_mean_dst, hi_mean_dst = _band_means(mag_dst)
    print(
        f"[mu-band <= {mu_band_hi_hz:.0f}Hz, COI-valid] "
        f"{src_label}: mu={mu_mean_src:.4g} vs. above-mu={hi_mean_src:.4g} "
        f"(ratio={mu_mean_src / hi_mean_src:.2f}); "
        f"{dst_label}: mu={mu_mean_dst:.4g} vs. above-mu={hi_mean_dst:.4g} "
        f"(ratio={mu_mean_dst / hi_mean_dst:.2f})"
    )

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.55)
    fig.suptitle(
        f"Subject {subject}, trial {trial_idx} ({trial_label}), {src_label}<->{dst_label}: "
        f"surrogate-calibrated coherence -> events -> GNN messages",
        fontsize=13,
    )

    # 1. Coherence array, plain -- hatched = inside COI only, no threshold
    #    overlay, for comparison against panel 2's significance contour.
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(coh.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    _hatch_inside_coi(ax1, coi_valid)
    fig.colorbar(im1, ax=ax1, label="coherence")
    ax1.set_title(f"1. Coherence array ({src_label}<->{dst_label})\nhatch=inside COI", fontsize=9.5, pad=6)
    ax1.set_xlabel("time (samples)")
    _set_freq_axis(ax1, freqs_1d)

    # 2. Same coherence array, contoured = coh above the surrogate-calibrated
    #    per-(edge,freq) threshold, at clf.surrogate_percentile (coherence
    #    half of the gate only -- phase not applied yet here, so this traces
    #    a superset of what panel 5's full gate keeps).
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(coh.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    _hatch_inside_coi(ax2, coi_valid)
    significant = coh > threshold_edge[np.newaxis, :]  # [T, F], broadcast over time
    ax2.contour(
        np.arange(T), np.arange(F), significant.T.astype(float),
        levels=[0.5], colors="red", linewidths=1.3,
    )
    fig.colorbar(im2, ax=ax2, label="coherence")
    ax2.set_title(
        f"2. Same array, red=coh>surrogate {clf.surrogate_percentile:.0f}th pct",
        fontsize=9.5, pad=6,
    )
    ax2.set_xlabel("time (samples)")
    _set_freq_axis(ax2, freqs_1d)

    # 3. Phase
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(phase.T, aspect="auto", origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im3, ax=ax3, label="radians")
    ax3.set_title("3. Phase (angle of smoothed cross-spectrum)", fontsize=9.5, pad=6)
    ax3.set_xlabel("time (samples)")
    _set_freq_axis(ax3, freqs_1d)

    # 4. Surrogate-calibrated threshold vs. frequency (edge_idx), on the
    #    SAME freq axis as panels 1-3/5 so rows line up visually.
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(threshold_edge, np.arange(F), marker="o", color="tab:red", label="surrogate threshold")
    mean_coh_per_freq = np.nanmean(np.where(coi_valid, coh, np.nan), axis=0)
    ax4.plot(mean_coh_per_freq, np.arange(F), marker=".", color="tab:blue", alpha=0.6,
              label="mean real coh (COI-valid)")
    ax4.set_xlim(0, 1)
    ax4.set_xlabel("coherence")
    ax4.set_title(
        f"4. Surrogate null threshold vs. freq\n"
        f"({clf.surrogate_percentile:.0f}th pct of {clf.surrogate_count} surrogates)",
        fontsize=9.5, pad=6,
    )
    _set_freq_axis(ax4, freqs_1d)
    ax4.legend(fontsize=7, loc="lower right")

    # 5. Gate: coh > surrogate_threshold(edge,freq) & phase>threshold & outside COI
    ax5 = fig.add_subplot(gs[1, 0])
    gate_masked = np.ma.masked_where(~coi_valid.T, gate.T.astype(float))
    cmap_gate = plt.get_cmap("gray").copy()
    cmap_gate.set_bad("lightgray")
    im5 = ax5.imshow(gate_masked, aspect="auto", origin="lower", cmap=cmap_gate, vmin=0, vmax=1)
    fig.colorbar(im5, ax=ax5)
    ax5.set_title(
        f"5. Gate: coh>surrogate_thresh(edge,freq)\n"
        f"& |phase|>{clf.phase_threshold_deg:.0f}deg & outside COI",
        fontsize=9.5, pad=6,
    )
    ax5.set_xlabel("time (samples)")
    _set_freq_axis(ax5, freqs_1d)

    # 6. Events as points with phase-angle arrows
    ax6 = fig.add_subplot(gs[1, 1])
    _hatch_inside_coi(ax6, coi_valid)
    if n_events:
        mag_range = max(float(np.ptp(ev_mag)), 1e-6)
        sc6 = ax6.scatter(
            ev_t, ev_f, s=40 + 400 * (ev_mag - ev_mag.min()) / mag_range,
            c=ev_mag, cmap="plasma", edgecolors="black", linewidths=0.6, zorder=3,
        )
        # Arrows encode phase angle as a direction on an arbitrary unit
        # circle (not a spatial/physical direction) -- purely a compact way
        # to show mean_angle per event, same convention the earlier
        # fixed-threshold figure used.
        #
        # (ev_cos, ev_sin) is a true unit vector, but this axes' x-range
        # (~T samples, e.g. ~1000) and y-range (~F freq bins, e.g. 16) are
        # wildly different in DATA units. Feeding (cos, sin) straight into
        # quiver's default "xy"/"xy" data-space scaling made every arrow's
        # horizontal component ~60x smaller in on-screen size than its
        # vertical component -- every arrow rendered as visually ~90 degrees
        # regardless of the real angle (verified: actual mean_angle values
        # for this edge range widely, e.g. 32-130 degrees, not clustered
        # near 90). Fix: scale dx/dy by each axis's own data range so equal
        # cos/sin components map to equal on-screen length again.
        x_range = max(T - 1, 1)
        y_range = max(F - 1, 1)
        arrow_frac = 0.07  # arrow length as a fraction of each axis's range
        dx = ev_cos * x_range * arrow_frac
        dy = ev_sin * y_range * arrow_frac
        ax6.quiver(
            ev_t, ev_f, dx, dy, angles="xy", scale_units="xy", scale=1.0,
            width=0.004, color="black", zorder=4,
        )
        fig.colorbar(sc6, ax=ax6, label="mean coherence")
    ax6.set_title(
        f"6. Consolidated events (n={n_events}): (t, freq)\nsize/color=mag, arrow=phase",
        fontsize=9.5, pad=6,
    )
    ax6.set_xlabel("time (samples)")
    ax6.set_xlim(0, T - 1)
    ax6.set_ylim(-0.5, F - 0.5)
    _set_freq_axis(ax6, freqs_1d)

    # 7. Event -> message -> destination node architecture (static diagram --
    #    the trainable submodule shapes, unchanged by surrogate vs. fixed mode).
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.axis("off")
    ax7.set_title("7. Event -> message -> destination node", fontsize=9.5, pad=6)
    _draw_message_diagram(ax7, clf.channel_embed_dim, clf.hidden_dim, helper.n_channels)

    # 8. Config / summary info panel
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")
    # valid_mask.mean() would be trivially 1.0 at batch_size=1 (no padding to
    # average against) -- use the same n_runs/(E*F) "bursts per row" metric
    # SparseEvidenceGNNCore._build_sparse_events itself reports instead.
    n_edges = int(helper.src_idx.numel())
    n_total_events = int(valid_mask.sum().item())
    density = n_total_events / max(n_edges * helper.nfreqs, 1)
    thr_lo, thr_hi = float(threshold_edge.min()), float(threshold_edge.max())
    lines = [
        f"coherence_threshold_mode = \"surrogate\"",
        f"surrogate_count = {clf.surrogate_count}",
        f"surrogate_percentile = {clf.surrogate_percentile:.1f}",
        f"phase_threshold_deg = {clf.phase_threshold_deg:.1f}",
        f"smooth_kernel_size = {clf.smooth_kernel_size}",
        f"coi_enabled = {clf.coi_enabled}",
        f"channel_subset = {clf.channel_subset}",
        "",
        f"edge {edge_idx}: {src_label} <-> {dst_label}",
        f"threshold range (this edge) = [{thr_lo:.3f}, {thr_hi:.3f}]",
        f"events on this edge = {n_events}",
        f"total events this trial = {n_total_events} ({n_edges} edges x {helper.nfreqs} freqs)",
        f"bursts-per-(edge,freq)-row = {density:.4f}",
        "",
        f"terminology: \"edge\" = 1 of {n_edges} fixed, UNDIRECTED",
        "channel-pair connections in the GNN graph (this figure",
        f"shows edge {edge_idx} only; src/dst above is a fixed display",
        "convention, i<j -- an edge carries BOTH directions' activity,",
        "distinguished by sign(mean_angle) per event, not by which of",
        f"two edges fired). \"event\" = 1 consolidated (t,freq) burst",
        f"riding on that edge -- many events can share one edge (all",
        f"n={n_events} above ride this same edge).",
    ]
    ax8.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9.5, family="monospace")
    ax8.set_title("8. Config / this-edge summary", fontsize=10)

    # 9/10. Raw per-channel CWT magnitude for this edge's two channels --
    # answers "does this channel have mu-band power at all" directly,
    # independent of whether the two channels are coherent with each other
    # (panels 1/2/5 can look empty in the mu band either because there's no
    # power there on either channel, or because there IS power on both but
    # it's not phase-locked -- these two panels tell those cases apart).
    # Dashed line = mu_band_hi_hz boundary; text box = COI-valid mean
    # magnitude inside vs. outside the mu band, on the same freq axis as
    # every other panel so rows line up visually.
    ax9 = fig.add_subplot(gs[2, 0:2])
    im9 = ax9.imshow(mag_src.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=mag_vmax)
    if coi_for_mag.shape == mag_src.shape:
        _hatch_inside_coi(ax9, coi_for_mag)
    ax9.axhline(mu_band_hi_bin, color="cyan", linestyle="--", linewidth=1.2)
    fig.colorbar(im9, ax=ax9, label="|CWT|")
    ax9.set_title(
        f"9. Raw CWT magnitude -- {src_label} (src)\n"
        f"cyan dash=mu-band<={mu_band_hi_hz:.0f}Hz top | "
        f"mean mu={mu_mean_src:.3g} vs. above-mu={hi_mean_src:.3g}",
        fontsize=9.5, pad=6,
    )
    ax9.set_xlabel("time (samples)")
    _set_freq_axis(ax9, freqs_1d)

    ax10 = fig.add_subplot(gs[2, 2:4])
    im10 = ax10.imshow(mag_dst.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=mag_vmax)
    if coi_for_mag.shape == mag_dst.shape:
        _hatch_inside_coi(ax10, coi_for_mag)
    ax10.axhline(mu_band_hi_bin, color="cyan", linestyle="--", linewidth=1.2)
    fig.colorbar(im10, ax=ax10, label="|CWT|")
    ax10.set_title(
        f"10. Raw CWT magnitude -- {dst_label} (dst)\n"
        f"cyan dash=mu-band<={mu_band_hi_hz:.0f}Hz top | "
        f"mean mu={mu_mean_dst:.3g} vs. above-mu={hi_mean_dst:.3g}",
        fontsize=9.5, pad=6,
    )
    ax10.set_xlabel("time (samples)")
    _set_freq_axis(ax10, freqs_1d)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def _draw_message_diagram(ax, channel_embed_dim: int, hidden_dim: int, n_channels: int):
    """Static box-and-arrow diagram of SparseEvidenceGNNCore.forward()'s
    trainable path (channel_encoder -> concat -> sparse_message_mlp ->
    scatter_add -> classifier). Dimensions are read off the real model, not
    hardcoded, so this stays accurate if channel_embed_dim/hidden_dim change."""
    message_in = 5 + 2 * channel_embed_dim
    boxes = [
        (0.05, 0.82, 0.4, 0.13, f"event features (5):\n[t, freq, mag, sinφ, cosφ]\nper consolidated burst", "#dbeafe"),
        (0.55, 0.82, 0.4, 0.13, f"src/dst ch embed ({channel_embed_dim}):\nChannelSignalEncoder\n(dilated conv1d over raw signal)", "#fde2e2"),
        (0.15, 0.62, 0.7, 0.11, f"concat -> [{message_in}]\n({5} event + {channel_embed_dim} src + {channel_embed_dim} dst)", "#fef3c7"),
        (0.15, 0.44, 0.7, 0.11, f"sparse_message_mlp:\nLinear({message_in},{hidden_dim}) -> GELU -> Linear({hidden_dim},{hidden_dim})", "#fef3c7"),
        (0.25, 0.28, 0.5, 0.1, f"message ({hidden_dim}) -- one per event", "#d1fae5"),
        (0.05, 0.05, 0.9, 0.15,
         f"scatter_add into dst node, /active_count -> node evidence [{n_channels},{hidden_dim}]\n-> flatten -> classifier -> logits",
         "#d1fae5"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.01", linewidth=1,
            edgecolor="black", facecolor=color, transform=ax.transAxes,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5,
                 transform=ax.transAxes)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes,
                     textcoords=ax.transAxes,
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    arrow(0.25, 0.82, 0.42, 0.73)
    arrow(0.75, 0.82, 0.55, 0.73)
    arrow(0.5, 0.62, 0.5, 0.55)
    arrow(0.5, 0.44, 0.5, 0.38)
    arrow(0.5, 0.28, 0.5, 0.20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Debug plot: current (surrogate-calibrated) SparseEvidenceGNN pipeline"
    )
    parser.add_argument("--trial", type=int, default=0, help="Trial index to visualize")
    parser.add_argument("--subject", type=int, default=1, help="MOABB subject id")
    parser.add_argument("--edge", type=int, default=0, help="Edge index to visualize")
    parser.add_argument(
        "--phase-threshold-deg", type=float, default=None,
        help="Override CANONICAL_SPARSE_KWARGS's phase_threshold_deg (default 30.0)",
    )
    parser.add_argument(
        "--out", type=str, default="coheriqs_contributions/debug_plots/surrogate_pipeline",
        help="Output directory for the PNG (a subfolder named with today's date is created under it)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Pop up the plot window after saving (off by default; the PNG is always saved either way)",
    )
    args = parser.parse_args()

    # Nest today's runs under a same-day subfolder so a week of debugging
    # sessions doesn't pile every PNG into one flat, hard-to-scan directory.
    out_dir = Path(args.out) / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    clf, helper, raw_x_native, w_real, w_imag, freqs, trial_label, ch_names_all = load_pipeline_and_trial(
        trial_idx=args.trial, subject=args.subject, phase_threshold_deg=args.phase_threshold_deg,
    )
    print(f"Loaded trial {args.trial} (label={trial_label}), n_channels={helper.n_channels}")
    print(f"phase_threshold_deg = {clf.phase_threshold_deg}")
    print("Computing/loading surrogate calibration + real events (may take a while on cache miss)...")
    stage = compute_pipeline_stage(clf, helper, raw_x_native, w_real, w_imag, freqs)

    phase_suffix = f"_phase{clf.phase_threshold_deg:.0f}deg"
    out_path = _unique_path(
        out_dir
        / f"subj{args.subject}_edge{args.edge}_surrogate_pipeline_trial{args.trial}{phase_suffix}.png"
    )
    fig = plot_pipeline_figure(
        clf, helper, stage, freqs, args.subject, args.trial, trial_label, args.edge, out_path,
        ch_names_all=ch_names_all,
    )
    print(f"Wrote {out_path.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
