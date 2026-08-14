"""Ablation: swap dense_edge_conv's Conv2d temporal step for a GRU that
integrates the FULL T' sequence with memory, plus a shuffled-time-order
negative control (2026-08-11).

Question: the time_averaged_graph ablation (see
session_notes/run_logs/2026-08-11_sparse-evidence-gnn_time-averaged-
coherence-graph-ablation.md, [[sparse-evidence-gnn-time-averaged-graph-
feature]]) found collapsing dense_edge_conv's input to a single COI-weighted
time average doesn't hurt accuracy -- but Conv2d's own receptive field
(dense_conv_kernel_size=5, applied twice) is small and local, so that result
never ruled out the pipeline benefiting from temporal structure a Conv2d
literally can't see. This ablation gives dense_edge_conv a mechanism that CAN
integrate the whole T' sequence with memory (a GRU,
dense_edge_temporal_mode="rnn" -- see _DenseEdgeGRUTemporal's docstring in
sparse_evidence_gnn_classifier.py) before drawing any conclusion about
whether time matters, plus shuffle_time_order=True as the matched negative
control (same architecture/parameter count, T' order scrambled per (edge,
frequency) before the GRU sees it).

Everything else (channel_encoder, sparse_message_mlp,
event_aggregation="concat", sparse_classifier, dense_edge_time_downsample=8)
is IDENTICAL to the canonical dense/concat config (see
session_notes/run_logs/2026-08-10_sparse-evidence-gnn_dense-concat-canonical-
4subject.md, cohort mean 0.7897) -- only dense_edge_conv's temporal mechanism
changes between the three variants below.

CANONICAL_KWARGS is copied from that same run log (NOT read from
run_sparse_evidence_gnn.py, which is under concurrent live edits in this
session -- same precedent run_dense_edge_flat_control.py /
run_time_averaged_coherence_graph.py's own CANONICAL_KWARGS sets already
use), with dense_edge_temporal_mode/shuffle_time_order added per variant.

USAGE
-----
    python coheriqs_contributions/tests/run_dense_edge_gru_temporal.py smoke
    python coheriqs_contributions/tests/run_dense_edge_gru_temporal.py run [subject ...]

`smoke` fits/predicts all three variants (conv, rnn, rnn+shuffle) on tiny
synthetic data for a few epochs -- confirms conv is unchanged, rnn/rnn+
shuffle run end-to-end and train (loss decreases). It does NOT compare
accuracy across variants -- that's `run`'s job, a separate/later ablation.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.pipeline import make_pipeline

from coheriqs_contributions.moabb_pipelines.sparse_evidence_gnn_classifier import (
    SparseEvidenceGNNClassifier,
)

# Copied verbatim from
# session_notes/run_logs/2026-08-10_sparse-evidence-gnn_dense-concat-canonical-
# 4subject.md's documented config -- the canonical dense/concat run this
# ablation's "conv" variant should reproduce.
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
    dense_conv_kernel_size=5,
    dense_conv_pool_size=4,
    dense_edge_time_downsample=8,
    time_averaged_graph=False,
    # --- this ablation's own axis -- overridden per variant below ---
    dense_edge_temporal_mode="conv",
    shuffle_time_order=False,
)

VARIANTS: dict[str, dict[str, object]] = {
    "conv": {},
    "rnn": {"dense_edge_temporal_mode": "rnn"},
    "rnn-shuffled": {"dense_edge_temporal_mode": "rnn", "shuffle_time_order": True},
}


def _build_classifier(variant: str, seed: int) -> SparseEvidenceGNNClassifier:
    kwargs = dict(CANONICAL_KWARGS)
    kwargs.update(VARIANTS[variant])
    kwargs["seed"] = seed
    return SparseEvidenceGNNClassifier(**kwargs)


def smoke_test() -> None:
    """Tiny synthetic-data fit/predict round-trip for all three variants --
    catches shape errors in the new dense_edge_temporal_mode="rnn"/
    shuffle_time_order paths fast, without waiting on a real
    CrossSessionEvaluation run, and confirms the "rnn" path actually trains
    (loss decreases over epochs).
    """
    print("=== dense_edge_temporal_mode smoke test ===")
    rng = np.random.default_rng(0)
    # 22 channels to match channel_subset's indices (up to 17) into the full
    # BNCI2014_001 montage -- the classifier subsets down to 9 internally.
    n_trials, n_channels, n_time = 12, 22, 500
    X = rng.standard_normal((n_trials, n_channels, n_time)).astype(np.float32)
    y = np.array(["left", "right"] * (n_trials // 2))

    for variant in ("conv", "rnn", "rnn-shuffled"):
        print(f"\n--- variant={variant} ---")
        clf = _build_classifier(variant, seed=42)
        # Trim epochs so the smoke test is fast -- architecture/shape
        # correctness and "loss decreases" don't need the full 60.
        clf.epochs = 8
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        preds = clf.predict(X)
        assert proba.shape == (n_trials, 2), f"unexpected proba shape {proba.shape}"
        assert len(preds) == n_trials

        core = clf.model_
        assert core.dense_edge_temporal_mode == VARIANTS[variant].get(
            "dense_edge_temporal_mode", "conv"
        )
        assert core.shuffle_time_order == VARIANTS[variant].get("shuffle_time_order", False)

        losses = clf.train_loss_history_
        print(f"loss history: {[round(l, 4) for l in losses]}")
        assert len(losses) == clf.epochs, f"expected {clf.epochs} loss entries, got {len(losses)}"
        assert losses[-1] < losses[0], (
            f"variant={variant}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"
        )
        print(f"PASS: variant={variant} fit/predict round-trip OK, loss decreased "
              f"({losses[0]:.4f} -> {losses[-1]:.4f}).")

    print("\nALL SMOKE TESTS PASSED")


def run(subjects: list[int], seeds: list[int], variants: tuple[str, ...] = ("conv", "rnn", "rnn-shuffled")) -> None:
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
        suffix="dense-edge-gru-temporal",
    )

    rows = []
    for variant in variants:
        for seed in seeds:
            label = f"{variant}-seed{seed}"
            pipelines = {label: make_pipeline(_build_classifier(variant, seed=seed))}
            print(f"\n=== {label} (subjects={subjects}) ===", flush=True)
            results = evaluation.process(pipelines)
            print(results[["subject", "session", "score"]].to_string(index=False), flush=True)
            by_subject = results.groupby("subject")["score"].mean().to_dict()
            row = {
                "variant": variant,
                "seed": seed,
                **{f"subj{k}": v for k, v in by_subject.items()},
            }
            row["cohort_mean"] = float(results["score"].mean())
            rows.append(row)
            print(row, flush=True)

    print("\n=== Summary: dense_edge_temporal_mode ablation (conv vs rnn vs rnn-shuffled) ===")
    for r in rows:
        print(r)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke_test()
    else:
        rest = sys.argv[2:]
        # Optional --variant=conv|rnn|rnn-shuffled (repeatable) filters which
        # of the three variants run() actually runs -- default (no flag) is
        # all three, same as before this option existed. Lets you run just
        # the "rnn" variant on its own instead of the full 3-variant sweep.
        variant_args = [a.split("=", 1)[1] for a in rest if a.startswith("--variant=")]
        subj_args = [a for a in rest if not a.startswith("--variant=")]
        subjects = [int(s) for s in subj_args] if subj_args else [1, 2, 3, 4]
        variants = tuple(variant_args) if variant_args else ("conv", "rnn", "rnn-shuffled")
        for v in variants:
            if v not in VARIANTS:
                raise ValueError(f"--variant must be one of {tuple(VARIANTS)}, got {v!r}.")
        run(subjects, seeds=[45], variants=variants)
