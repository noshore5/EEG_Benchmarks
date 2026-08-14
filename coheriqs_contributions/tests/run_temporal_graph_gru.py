"""event_mode="temporal_graph" -- a genuine "evolving graph" ablation
(2026-08-11).

Question: neither of this pipeline's existing "richer than sparse events"
architectures actually walks forward through time. event_mode="dense" pools
the whole T' axis away in one shot (dense_edge_conv's Conv2d stack +
AdaptiveAvgPool2d) before the graph ever sees more than one vector per edge.
n_hops>1 adds DEPTH -- extra rounds of message passing ACROSS THE GRAPH
within one already-pooled snapshot -- never touching time at all. Neither
result rules in or out whether a mechanism that can actually integrate
evidence node-by-node, timestep-by-timestep, would help or hurt.

event_mode="temporal_graph" (see SparseEvidenceGNNCore's own event_mode
docstring in sparse_evidence_gnn_classifier.py, and
_temporal_graph_node_states) reuses the exact same precomputed, non-trainable
[B, 4, E, T, F] stack "dense" mode consumes (same dense_edge_time_downsample,
COI mask, smoothing), but processes it step-by-step: a small per-timestep
Linear+GELU (temporal_edge_proj, NOT the deep Conv2d stack) folds frequency
into a per-edge embedding at every surviving timestep, sparse_message_mlp
(the SAME shared weights every other event_mode uses) turns those into
messages, the existing "mean" aggregation (event_mode="temporal_graph"
requires event_aggregation="mean" -- deliberately NOT "concat", to keep this
experiment isolated to the temporal question rather than compounding it with
the aggregation question) scatters them to per-node evidence AT EVERY
TIMESTEP, and a single nn.GRU (weight-shared across nodes) walks that
sequence with a persistent hidden state. The GRU's final hidden state per
node IS `evidence`, in the exact [B, n_channels, hidden_dim] shape every
other event_mode's aggregation already produces -- so n_hops>1 propagation
(if requested; orthogonal, across-the-graph depth on top of this mode's own
across-time propagation) and sparse_classifier are shared code, not a
duplicated path.

CANONICAL_HELPER_KWARGS below is copied from
run_dense_edge_flat_control.py's own CANONICAL_HELPER_KWARGS (same
precedent every one-off script in this line of investigation uses --
hardcoded, NOT read from run_sparse_evidence_gnn.py, which is under
concurrent live edits in this session) -- the exact config that produced
the two numbers this ablation is measured against:
  - "flat" non-graph control on dense features: 0.9291 (seed 42 only)
    session_notes/run_logs/2026-08-10_sparse-evidence-gnn_dense-edge-flat-
    no-graph-control.md
  - event_mode="dense" mean-pool-then-graph, event pathway alone
    (feature_ablation="zero_channel_embed"): 0.8886 (mean, seeds 42-45)
    session_notes/run_logs/2026-08-10_sparse-evidence-gnn_dense-event-mode-
    sparse-vs-dense.md
dense_edge_time_downsample=8 is added on top of that shared config (not
present in either source run, both predating that param) -- the task's own
guidance to keep sequence length tractable for a GRU that can't parallelize
across time the way Conv2d can; 8 matches the value already validated as
helpful for the dense/concat conv path (see run_sparse_evidence_gnn.py's own
DENSE_EDGE_TIME_DOWNSAMPLE history).

USAGE
-----
    python coheriqs_contributions/tests/run_temporal_graph_gru.py smoke
    python coheriqs_contributions/tests/run_temporal_graph_gru.py run [subject ...]

`smoke` fits/predicts the "temporal_graph" variant (plus a "dense" regression
check, same event pathway, mean aggregation, no time-stepping) on tiny
synthetic data for a few epochs -- confirms shapes/gradients are correct and
both variants actually train (loss decreases). It does NOT compare accuracy
across variants -- that's `run`'s job.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.pipeline import make_pipeline

from coheriqs_contributions.moabb_pipelines.sparse_evidence_gnn_classifier import (
    SparseEvidenceGNNClassifier,
    SparseEvidenceGNNCore,
)

# Copied verbatim from run_dense_edge_flat_control.py's CANONICAL_HELPER_
# KWARGS -- the exact config behind this ablation's two comparison numbers
# (flat 0.9291, dense mean-pool 0.8886). See module docstring.
CANONICAL_KWARGS: dict[str, object] = dict(
    sampling_rate=250,
    lowest=8.0,
    highest=30.0,
    nfreqs=16,
    cwt_resample_n_time=None,
    coherence_threshold=0.0,
    phase_threshold_deg=10.0,
    channel_encoder_dilation=5,
    hidden_dim=8,
    channel_embed_dim=8,
    epochs=75,
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
    coherence_threshold_mode="surrogate",
    surrogate_count=100,
    surrogate_percentile=99.0,
    mu_band_surrogate_percentile=None,
    mu_band_range_hz=(8.0, 13.0),
    cluster_significance_percentile=95.0,
    surrogate_device="auto",
    surrogate_cache_dir=None,
    surrogate_cache_enabled=True,
    verbose=0,
    n_hops=1,
    freq_aware_hops=False,
    event_aggregation="mean",
    # --- this ablation's own axis -- overridden per variant below ---
    event_mode="dense",
    dense_conv_kernel_size=5,
    dense_conv_pool_size=4,
    dense_conv_intermediate_channels=32,
    dense_conv_out_channels=8,
    dense_edge_time_downsample=1,
)

VARIANTS: dict[str, dict[str, object]] = {
    # Regression check: identical event pathway/aggregation to the recorded
    # 0.8886 number, just re-run through this script's own driver.
    "dense": {},
    # This ablation's own new mode. dense_edge_time_downsample=8 per this
    # script's own module docstring -- keeps GRU sequence length tractable;
    # not present in the "dense" variant above (native resolution, matching
    # the 0.8886 number's own original config exactly).
    "temporal_graph": {"event_mode": "temporal_graph", "dense_edge_time_downsample": 8},
}


def _build_classifier(variant: str, seed: int) -> SparseEvidenceGNNClassifier:
    kwargs = dict(CANONICAL_KWARGS)
    kwargs.update(VARIANTS[variant])
    kwargs["seed"] = seed
    return SparseEvidenceGNNClassifier(**kwargs)


def _gradient_check() -> None:
    """Direct forward/backward gradient check on SparseEvidenceGNNCore's
    temporal_edge_proj/temporal_node_gru submodules, bypassing the
    classifier wrapper -- catches shape/wiring bugs in the new
    _temporal_graph_node_states path fastest, before any fit()/predict()
    plumbing is involved."""
    print("=== temporal_graph direct gradient check ===")
    torch.manual_seed(0)
    n_channels, nfreqs, n_classes = 9, 16, 2
    batch_size, n_edges, n_time = 4, 36, 40  # small T -- this is a shape/gradient check, not a speed test

    core = SparseEvidenceGNNCore(
        n_channels=n_channels, nfreqs=nfreqs, n_classes=n_classes,
        hidden_dim=8, channel_embed_dim=8, event_mode="temporal_graph",
        event_aggregation="mean", temporal_graph_edge_dim=8,
    )
    assert core.temporal_edge_proj is not None
    assert core.temporal_node_gru is not None
    assert core.dense_edge_conv is None

    raw_x = torch.randn(batch_size, n_channels, 300, requires_grad=False)
    dense_edge_raw = torch.randn(batch_size, 4, n_edges, n_time, nfreqs)

    logits, event_density = core(raw_x, dense_edge_raw)
    assert logits.shape == (batch_size, n_classes), f"unexpected logits shape {logits.shape}"
    assert np.isclose(event_density, 1.0 / nfreqs), f"unexpected event_density {event_density}"

    target = torch.tensor([0, 1, 0, 1])
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    for name, param in [
        ("temporal_edge_proj", core.temporal_edge_proj[0].weight),
        ("temporal_node_gru", next(core.temporal_node_gru.parameters())),
        ("sparse_message_mlp", core.sparse_message_mlp[0].weight),
        ("sparse_classifier", core.sparse_classifier.weight),
        ("channel_encoder", core.channel_encoder.net[0].weight),
    ]:
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.any(param.grad != 0), f"{name} gradient is all-zero"
    print("PASS: temporal_edge_proj/temporal_node_gru/sparse_message_mlp/"
          "sparse_classifier/channel_encoder all received nonzero gradients.")

    # n_hops=3 on top of temporal_graph: evidence (the GRU's final hidden
    # state) must flow into _propagate_hops unmodified -- hop_message_mlp/
    # hop_update should also get gradient.
    core_hops = SparseEvidenceGNNCore(
        n_channels=n_channels, nfreqs=nfreqs, n_classes=n_classes,
        hidden_dim=8, channel_embed_dim=8, event_mode="temporal_graph",
        event_aggregation="mean", temporal_graph_edge_dim=8, n_hops=3,
    )
    logits_hops, _ = core_hops(raw_x, dense_edge_raw)
    loss_hops = torch.nn.functional.cross_entropy(logits_hops, target)
    loss_hops.backward()
    assert core_hops.hop_update.weight_ih.grad is not None
    assert torch.any(core_hops.hop_update.weight_ih.grad != 0)
    print("PASS: temporal_graph + n_hops=3 -- hop_update received nonzero gradient too.")

    # Constructor validation: event_aggregation != "mean" must raise.
    try:
        SparseEvidenceGNNCore(
            n_channels=n_channels, nfreqs=nfreqs, n_classes=n_classes,
            event_mode="temporal_graph", event_aggregation="concat",
        )
        raise AssertionError("expected ValueError for temporal_graph + concat")
    except ValueError as exc:
        assert "temporal_graph" in str(exc)
        print(f"PASS: event_mode='temporal_graph' + event_aggregation='concat' rejected: {exc}")

    try:
        SparseEvidenceGNNCore(
            n_channels=n_channels, nfreqs=nfreqs, n_classes=n_classes,
            event_mode="temporal_graph", freq_aware_hops=True, n_hops=2,
        )
        raise AssertionError("expected ValueError for temporal_graph + freq_aware_hops")
    except ValueError as exc:
        print(f"PASS: event_mode='temporal_graph' + freq_aware_hops=True rejected: {exc}")

    print("\nALL DIRECT GRADIENT/VALIDATION CHECKS PASSED")


def smoke_test() -> None:
    """Tiny synthetic-data fit/predict round-trip for both variants -- see
    module docstring."""
    _gradient_check()

    print("\n=== fit/predict smoke test ===")
    rng = np.random.default_rng(0)
    n_trials, n_channels, n_time = 12, 22, 500
    X = rng.standard_normal((n_trials, n_channels, n_time)).astype(np.float32)
    y = np.array(["left", "right"] * (n_trials // 2))

    # temporal_graph needs MANY more optimizer steps than dense to show a
    # clear decrease on this tiny (12-trial, 2-steps/epoch) synthetic set --
    # a direct core-level check at this classifier's own lr=1e-3 showed loss
    # sitting flat near ln(2)=0.693 through ~100 steps, then dropping
    # sharply from ~0.69 to <0.01 between steps ~125-200. This is a real,
    # reproducible slow-warm-up property of this GRU-based path (plausibly
    # standard recurrent-net "vanishing early gradient" dynamics, compounded
    # by grad_clip_norm=0.1 being tight relative to the small gradients this
    # path produces before the GRU's hidden state finds useful dynamics) --
    # NOT a stalled/broken gradient path (the direct gradient check above
    # already confirms every submodule receives nonzero gradient from step
    # 1). 150 epochs (300 steps at batch_size=8/n_trials=12) comfortably
    # clears the warm-up window observed above.
    smoke_epochs = {"dense": 8, "temporal_graph": 150}

    for variant in ("dense", "temporal_graph"):
        print(f"\n--- variant={variant} ---")
        clf = _build_classifier(variant, seed=42)
        clf.epochs = smoke_epochs[variant]
        clf.surrogate_count = 3  # trivially fast surrogate calibration for the smoke test
        t0 = time.time()
        clf.fit(X, y)
        fit_s = time.time() - t0
        proba = clf.predict_proba(X)
        preds = clf.predict(X)
        assert proba.shape == (n_trials, 2), f"unexpected proba shape {proba.shape}"
        assert len(preds) == n_trials

        core = clf.model_
        assert core.event_mode == variant

        losses = clf.train_loss_history_
        print(f"fit wall-clock: {fit_s:.1f}s, loss history: {[round(l, 4) for l in losses]}")
        assert len(losses) == clf.epochs, f"expected {clf.epochs} loss entries, got {len(losses)}"
        assert losses[-1] < losses[0], (
            f"variant={variant}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"
        )
        print(f"PASS: variant={variant} fit/predict round-trip OK, loss decreased "
              f"({losses[0]:.4f} -> {losses[-1]:.4f}).")

    print("\nALL SMOKE TESTS PASSED")


def run(subjects: list[int], seeds: list[int], variants: tuple[str, ...] = ("temporal_graph",)) -> None:
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
        suffix="temporal-graph-gru",
    )

    rows = []
    for variant in variants:
        for seed in seeds:
            label = f"{variant}-seed{seed}"
            pipelines = {label: make_pipeline(_build_classifier(variant, seed=seed))}
            print(f"\n=== {label} (subjects={subjects}) ===", flush=True)
            t0 = time.time()
            results = evaluation.process(pipelines)
            wall_s = time.time() - t0
            print(results[["subject", "session", "score"]].to_string(index=False), flush=True)
            by_subject = results.groupby("subject")["score"].mean().to_dict()
            row = {
                "variant": variant,
                "seed": seed,
                "wall_clock_s": round(wall_s, 1),
                **{f"subj{k}": v for k, v in by_subject.items()},
            }
            row["cohort_mean"] = float(results["score"].mean())
            rows.append(row)
            print(row, flush=True)

    print("\n=== Summary: temporal_graph vs. recorded comparison points ===")
    print("flat (non-graph control on dense features): 0.9291 (seed 42, subj1)")
    print("dense mean-pool, event pathway alone: 0.8886 mean (seeds 42-45, subj1)")
    print("dense concat, event pathway alone, end-to-end: 0.8973 (seed 42, subj1)")
    for r in rows:
        print(r)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke_test()
    else:
        rest = sys.argv[2:]
        variant_args = [a.split("=", 1)[1] for a in rest if a.startswith("--variant=")]
        subj_args = [a for a in rest if not a.startswith("--variant=")]
        subjects = [int(s) for s in subj_args] if subj_args else [1]
        variants = tuple(variant_args) if variant_args else ("temporal_graph",)
        for v in variants:
            if v not in VARIANTS:
                raise ValueError(f"--variant must be one of {tuple(VARIANTS)}, got {v!r}.")
        run(subjects, seeds=[42], variants=variants)
