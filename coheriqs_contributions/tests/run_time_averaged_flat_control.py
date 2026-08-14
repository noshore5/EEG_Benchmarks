"""Ablation: time-averaged coherence graph (2026-08-11's
time_averaged_graph=True -- see run_time_averaged_coherence_graph.py) fed
into the no-graph flat control (2026-08-10's DenseEdgeFlatCore -- see
run_dense_edge_flat_control.py) instead of the full graph/message-passing
readout.

Question: run_dense_edge_flat_control.py found that skipping graph structure
entirely (flatten dense_edge_conv's per-edge output straight to one Linear)
beats message-passing on top of TIME-RESOLVED dense features
(0.9291 vs 0.8886, single seed, subject 1 -- see
session_notes/run_logs/2026-08-10_sparse-evidence-gnn_dense-edge-flat-no-
graph-control.md). run_time_averaged_coherence_graph.py separately found that
collapsing TIME resolution to a single per-trial average, while KEEPING the
full graph/message-passing readout, doesn't cost accuracy either (0.7999 vs
0.7897 cohort mean, 4 subjects -- see
session_notes/run_logs/2026-08-11_sparse-evidence-gnn_time-averaged-
coherence-graph-ablation.md). Neither result says what happens with BOTH
simplifications stacked: does a flat (no-graph) readout on top of a static
(no-time) coherence summary still hold up, or does removing time resolution
finally start costing something once the graph's own redundant structure
isn't there to compensate?

Mechanically: reuses DenseEdgeFlatClassifier/DenseEdgeFlatCore UNCHANGED
(2026-08-11 added a `time_averaged_graph` pass-through constructor arg to
DenseEdgeFlatClassifier specifically for this combination -- see that
class's own docstring), with dense_conv_kernel_size=dense_conv_pool_size=1
(required once the precomputed dense_edge_raw's T axis is 1 -- a k>1 kernel
has nothing left to convolve over, same requirement
SparseEvidenceGNNCore.time_averaged_graph's docstring describes) and
time_averaged_graph=True. With kernel/pool collapsed to 1x1,
DenseEdgeFlatCore's own dense_edge_conv becomes two stacked 1x1-kernel
Conv2d+GELU layers -- a per-edge MLP over the folded (4*nfreqs) input
channels, no temporal or cross-edge mixing at all -- flattened straight to
the single Linear classifier head, same as the original flat control.

USAGE
-----
    python coheriqs_contributions/tests/run_time_averaged_flat_control.py smoke
    python coheriqs_contributions/tests/run_time_averaged_flat_control.py run [subject ...]
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

from coheriqs_contributions.tests.run_dense_edge_flat_control import (
    DenseEdgeFlatClassifier,
    DenseEdgeFlatCore,
)


def _build_classifier(seed: int) -> DenseEdgeFlatClassifier:
    return DenseEdgeFlatClassifier(
        seed=seed,
        epochs=75,  # matches run_dense_edge_flat_control.py's own default
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=0.1,
        dense_conv_kernel_size=1,
        dense_conv_pool_size=1,
        dense_conv_intermediate_channels=32,
        dense_conv_out_channels=8,
        time_averaged_graph=True,
    )


def smoke_test() -> None:
    """Same shape/gradient-flow check run_dense_edge_flat_control.py's own
    smoke_test does, but against a T=1 dense_edge_raw (the shape
    time_averaged_graph=True actually produces) instead of a synthetic
    T=300 array -- confirms DenseEdgeFlatCore's kernel_size=1/pool_size=1
    construction tolerates a collapsed time axis (MaxPool2d(kernel_size=1)
    is a no-op, not an error, but worth checking directly rather than
    assuming)."""
    print("=== time_averaged_graph + flat control smoke test ===")
    torch.manual_seed(0)
    B, E, T, F, n_classes = 4, 36, 1, 16, 2
    model = DenseEdgeFlatCore(
        n_edges=E, nfreqs=F, n_classes=n_classes,
        dense_conv_kernel_size=1, dense_conv_pool_size=1,
        dense_conv_intermediate_channels=16, dense_conv_out_channels=8,
    )
    dense_edge_raw = torch.randn(B, 4, E, T, F)
    y = torch.randint(0, n_classes, (B,))

    logits = model(dense_edge_raw)
    print("logits shape:", logits.shape)
    assert logits.shape == (B, n_classes)

    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    any_grad = any(p.grad is not None and p.grad.norm().item() > 0 for p in model.parameters())
    assert any_grad, "no gradient reached model parameters!"
    print("PASS: T=1 forward/backward OK, gradients flow.\n")

    # Now the real end-to-end path: fit/predict through DenseEdgeFlatClassifier
    # itself (helper precompute -> time-average -> DenseEdgeFlatCore), on
    # synthetic 22-channel data (channel_subset indices go up to 17).
    rng = np.random.default_rng(0)
    n_trials, n_channels, n_time = 12, 22, 500
    X = rng.standard_normal((n_trials, n_channels, n_time)).astype(np.float32)
    y_np = np.array(["left", "right"] * (n_trials // 2))

    clf = _build_classifier(seed=42)
    clf.epochs = 2
    clf.fit(X, y_np)
    proba = clf.predict_proba(X)
    assert proba.shape == (n_trials, 2), f"unexpected proba shape {proba.shape}"
    print("PASS: end-to-end fit/predict round-trip OK.\n")
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
        suffix="time-averaged-flat-control",
    )

    rows = []
    for seed in seeds:
        label = f"time-avg-flat-seed{seed}"
        pipelines = {label: make_pipeline(_build_classifier(seed=seed))}
        print(f"\n=== {label} (subjects={subjects}) ===", flush=True)
        results = evaluation.process(pipelines)
        print(results[["subject", "session", "score"]].to_string(index=False), flush=True)
        by_subject = results.groupby("subject")["score"].mean().to_dict()
        row = {"seed": seed, **{f"subj{k}": v for k, v in by_subject.items()}}
        row["cohort_mean"] = float(results["score"].mean())
        rows.append(row)
        print(row, flush=True)

    print("\n=== Summary: time_averaged_graph=True + flat (no graph) control ===")
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
