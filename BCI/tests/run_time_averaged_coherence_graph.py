"""Ablation: replace time-resolved event extraction with a single
time-averaged coherence graph per trial (2026-08-11).

Question: does the canonical dense/concat Sparse-Evidence-GNN pipeline (see
session_notes/run_logs/2026-08-10_sparse-evidence-gnn_dense-concat-canonical-
4subject.md, cohort mean 0.7897) actually need within-trial TIME resolution
at all, or does a static, per-(edge, frequency) functional-connectivity-style
summary -- mean coherence/phase/significance over the whole trial, one value
per edge per frequency, no time axis left -- carry the same signal? This is
the most literal version of "how much does temporal dynamics matter" this
pipeline can ask: every other architectural piece (channel_encoder,
sparse_message_mlp, _aggregate_events="concat", sparse_classifier) is
IDENTICAL to the canonical run; only event BUILDING changes.

Uses SparseEvidenceGNNClassifier's new time_averaged_graph=True (see
SparseEvidenceGNNCore's constructor docstring in
sparse_evidence_gnn_classifier.py) -- _build_dense_edge_input collapses its
[B, 4, E, T, F] coherence/sinφ/cosφ/significance stack down to [B, 4, E, 1,
F] via a COI-valid-weighted average over T, once per trial at (non-trainable)
precompute time, same discipline compute_events/compute_dense_edge_input
already use. dense_edge_conv still runs (now over a width-1 time axis, so
dense_conv_kernel_size=dense_conv_pool_size=1 is REQUIRED -- see that param's
docstring for why a k>1 kernel has nothing left to convolve over), everything
downstream is bit-identical to event_mode="dense" today.

CANONICAL_KWARGS below is copied from the exact config documented in
session_notes/run_logs/2026-08-10_sparse-evidence-gnn_dense-concat-canonical-
4subject.md (NOT read from run_sparse_evidence_gnn.py, which is under
concurrent live edits in this session -- same precedent
run_dense_edge_flat_control.py's own CANONICAL_HELPER_KWARGS set), with three
deltas: dense_conv_kernel_size 5->1, dense_conv_pool_size 4->1 (required, see
above), dense_edge_time_downsample 8->1 (mutually exclusive with
time_averaged_graph -- __init__ raises otherwise), plus time_averaged_graph
itself turned on.

USAGE
-----
    python BCI/tests/run_time_averaged_coherence_graph.py smoke
    python BCI/tests/run_time_averaged_coherence_graph.py run [subject ...]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.pipeline import make_pipeline

from BCI.moabb_pipelines.sparse_evidence_gnn_classifier import (
    SparseEvidenceGNNClassifier,
)

CANONICAL_KWARGS: dict[str, object] = dict(
    sampling_rate=250,
    lowest=8.0,
    highest=30.0,
    nfreqs=16,
    cwt_resample_n_time=None,
    coherence_threshold=0.99,
    phase_threshold_deg=10.0,
    channel_encoder_dilation=5,
    hidden_dim=8,
    channel_embed_dim=8,
    epochs=60,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-4,
    grad_clip_norm=0.1,
    normalize_input=True,
    noise_augmentation_enabled=False,
    validation_split=0.0,
    validation_group_column=None,
    early_stopping_patience=None,
    device="auto",
    surrogate_seed=42,
    smooth_kernel_size=(5, 3),
    smooth_kernel_sigma=(None, None),
    coi_enabled=True,
    scale_adaptive_smoothing=False,
    scale_adaptive_cycles=1.5,
    scale_adaptive_max_kernel=101,
    raw_x_resample_n_time=None,
    feature_ablation="zero_channel_embed",
    channel_subset=[1, 5, 7, 8, 9, 10, 11, 13, 17],
    coherence_threshold_mode="fixed",
    surrogate_count=100,
    surrogate_percentile=99.999,
    mu_band_surrogate_percentile=None,
    mu_band_range_hz=(8.0, 13.0),
    cluster_significance_percentile=95.0,
    surrogate_device="auto",
    surrogate_cache_dir=None,
    surrogate_cache_enabled=True,
    verbose=0,
    n_hops=1,
    freq_aware_hops=False,
    event_mode="dense",
    event_aggregation="concat",
    dense_conv_intermediate_channels=32,
    dense_conv_out_channels=8,
    # --- the three deltas from the canonical dense/concat config, all
    # required by time_averaged_graph=True (see module docstring) ---
    dense_conv_kernel_size=1,
    dense_conv_pool_size=1,
    dense_edge_time_downsample=1,
    time_averaged_graph=True,
)


def _build_classifier(seed: int) -> SparseEvidenceGNNClassifier:
    kwargs = dict(CANONICAL_KWARGS)
    kwargs["seed"] = seed
    return SparseEvidenceGNNClassifier(**kwargs)


def smoke_test() -> None:
    """Tiny synthetic-data fit/predict round-trip -- catches shape errors in
    the new time_averaged_graph path (_time_average_dense_edge_input's
    broadcast, dense_edge_conv's kernel_size=1 construction, the downstream
    "concat" aggregation's edge-axis bookkeeping) fast, without waiting on a
    real CrossSessionEvaluation run.
    """
    print("=== time_averaged_graph smoke test ===")
    rng = np.random.default_rng(0)
    # 22 channels to match channel_subset's indices (up to 17) into the full
    # BNCI2014_001 montage -- the classifier subsets down to 9 internally.
    n_trials, n_channels, n_time = 12, 22, 500
    X = rng.standard_normal((n_trials, n_channels, n_time)).astype(np.float32)
    y = np.array(["left", "right"] * (n_trials // 2))

    clf = _build_classifier(seed=42)
    # Trim epochs so the smoke test is fast -- architecture/shape correctness
    # doesn't depend on how many epochs run.
    clf.epochs = 2
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    preds = clf.predict(X)
    assert proba.shape == (n_trials, 2), f"unexpected proba shape {proba.shape}"
    assert len(preds) == n_trials

    core = clf.model_
    assert core.time_averaged_graph is True
    # dense_edge_conv's kernel_size=1/pool_size=1 construction should leave
    # its temporal axis untouched at width 1 all the way through -- confirm
    # directly against a fresh precompute rather than trusting fit() alone.
    _raw_x, dense_edge_raw = clf._prepare_features(X, fit=False)
    assert dense_edge_raw.shape[3] == 1, (
        f"expected time axis collapsed to 1, got shape {dense_edge_raw.shape}"
    )
    print(f"dense_edge_raw shape: {tuple(dense_edge_raw.shape)} (B, 4, E, T=1, F)")
    print("PASS: fit/predict round-trip OK, time axis collapsed to 1 as expected.\n")
    print("ALL SMOKE TESTS PASSED")


def run(subjects: list[int], seeds: list[int]) -> None:
    from moabb.datasets import BNCI2014_001
    from moabb.evaluations import CrossSessionEvaluation
    from moabb.paradigms import LeftRightImagery

    warnings.filterwarnings("ignore")
    dataset = BNCI2014_001()
    dataset.subject_list = subjects
    paradigm = LeftRightImagery(fmin=8, fmax=35)
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        n_jobs=1,
        random_state=42,
        progress_bar=False,
        suffix="time-averaged-coherence-graph",
    )

    rows = []
    for seed in seeds:
        label = f"time-avg-graph-seed{seed}"
        pipelines = {label: make_pipeline(_build_classifier(seed=seed))}
        print(f"\n=== {label} (subjects={subjects}) ===", flush=True)
        results = evaluation.process(pipelines)
        print(results[["subject", "session", "score"]].to_string(index=False), flush=True)
        by_subject = results.groupby("subject")["score"].mean().to_dict()
        row = {"seed": seed, **{f"subj{k}": v for k, v in by_subject.items()}}
        row["cohort_mean"] = float(results["score"].mean())
        rows.append(row)
        print(row, flush=True)

    print("\n=== Summary: time_averaged_graph=True (dense/concat, no time resolution) ===")
    for r in rows:
        print(r)
    means = [r["cohort_mean"] for r in rows]
    print(
        f"cohort_mean={np.mean(means):.4f} "
        f"std={(np.std(means, ddof=1) if len(means) > 1 else 0.0):.4f}"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke_test()
    else:
        subj_args = sys.argv[2:]
        subjects = [int(s) for s in subj_args] if subj_args else [1, 2, 3, 4]
        run(subjects, seeds=[45])
