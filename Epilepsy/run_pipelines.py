"""Entry point for the Epilepsy dense-edge-GRU pipeline.

    python Epilepsy/run_pipelines.py --subjects 1
    python Epilepsy/run_pipelines.py --smoke

This is the epilepsy counterpart to BCI/run_pipelines.py's "dense_edge_gru"
pipeline (SparseEvidenceGNNClassifier, event_mode="dense",
dense_edge_temporal_mode="rnn") -- but NOT a copy of that script. Two things
there don't carry over:

  - BCI/run_pipelines.py evaluates via moabb.evaluations.CrossSessionEvaluation
    /CrossSubjectEvaluation, which call paradigm.get_data(dataset, subjects)
    expecting a real moabb.paradigms.base.BaseParadigm. ContinuousLabelingParadigm
    deliberately isn't one (see paradigms/continuous_labeling.py's docstring),
    and CHBMIT only has one session per subject anyway, so there's nothing for
    CrossSessionEvaluation to cross-validate across. Evaluation here is a
    small custom leave-one-seizure-out loop instead (below).
  - The classifier itself is a standalone copy under Epilepsy/pipelines/, not
    an import of BCI's (see that folder's files for what changed and why:
    use_class_weights now actually does something, sampling_rate/frequency
    band defaults match CHB-MIT instead of BNCI2014_001 motor imagery).

WINDOWING DEFAULTS ARE NOT TUNED. window_length/step_size below are a
starting point, not a validated choice -- see ContinuousLabelingParadigm's
docstring for how they interact with whatever downstream feature step size
you actually want to match.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.epilepsy import CHBMIT
from paradigms.continuous_labeling import ContinuousLabelingParadigm
from Epilepsy.pipelines.sparse_evidence_gnn_classifier import (
    SparseEvidenceGNNClassifier,
)


DEFAULT_SUBJECTS = [1]

# Mirrors BCI/run_pipelines.py's DENSE_EDGE_GRU_PARAMS (event_mode="dense",
# dense_edge_temporal_mode="rnn") with the necessary epilepsy-side changes:
#   - sampling_rate/lowest/highest: CHB-MIT's native rate + a broad,
#     NOT-dataset-tuned band (see sparse_evidence_gnn_classifier.py's
#     __init__ docstring) instead of BNCI2014_001's 250Hz mu/beta.
#   - channel_subset: None (use all 23) instead of BNCI's 9-channel
#     motor-cortex index list, which is meaningless for CHB-MIT's bipolar
#     10-20 montage.
#   - use_class_weights: True (this fork's whole reason to exist -- ictal
#     windows are a small fraction of the data; see run_pipelines.py's own
#     module docstring... er, this one).
#   - validation_split: 0.0 -- each leave-one-seizure-out fold's training
#     set is already small; not carving further off it for early stopping.
DENSE_EDGE_GRU_PARAMS: dict[str, object] = dict(
    seed=42,
    sampling_rate=256,
    lowest=1.0,
    highest=40.0,
    nfreqs=16,
    cwt_resample_n_time=None,
    coherence_threshold=0.99,
    phase_threshold_deg=10.0,
    channel_encoder_dilation=5,
    hidden_dim=8,
    channel_embed_dim=8,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-4,
    grad_clip_norm=0.1,
    normalize_input=True,
    validation_split=0.0,
    device="auto",
    surrogate_seed=42,
    smooth_kernel_size=(5, 3),
    coi_enabled=True,
    feature_ablation="none",
    channel_subset=None,
    coherence_threshold_mode="fixed",
    surrogate_count=100,
    surrogate_percentile=99.0,
    use_class_weights=True,
    event_mode="dense",
    event_aggregation="concat",
    dense_edge_time_downsample=8,
    dense_edge_temporal_mode="rnn",
    dense_conv_kernel_size=5,
    dense_conv_pool_size=4,
    time_averaged_graph=False,
    verbose=1,
)


def _build_windowed_dataset(subjects: list[int], window_length: float, step_size: float):
    """CHBMIT + ContinuousLabelingParadigm -> (X, y, metadata_df) for `subjects`.

    Restricts each subject to just its seizure-containing recordings (see
    CHBMIT.list_seizure_records) -- same reasoning as
    tests/test_chb_mit_continuous_labeling.py: avoids downloading/windowing
    every seizure-free recording for a subject when only the seizure ones
    are needed for training/evaluation here.
    """
    discovery = CHBMIT()
    records = {subject: [r["filename"] for r in discovery.list_seizure_records(subject)] for subject in subjects}
    dataset = CHBMIT(records=records)
    paradigm = ContinuousLabelingParadigm(
        window_length=window_length, step_size=step_size, label_event="seizure"
    )
    X, y, metadata = paradigm.get_data(dataset, subjects=subjects)
    return X, y, pd.DataFrame(metadata)


def leave_one_seizure_out(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    clf_params: dict,
    epochs: int,
) -> pd.DataFrame:
    """Leave-one-seizure-out CV: hold out one recording's windows at a time.

    Each fold's group is `(subject, run)` -- one CHB-MIT recording, which
    (given the seizure-only `records` filter above) contains exactly one
    seizure's worth of ictal windows plus that recording's interictal
    windows. Trains on every other recording, tests on the held-out one.

    Folds share a single CWT cache (see cwt_window_cache.py): each fold
    excludes only one recording, so most windows appear in most folds'
    training sets -- without sharing, every fold would recompute the same
    wavelet transform for the same physical windows from scratch.
    """
    groups = list(zip(metadata["subject"], metadata["run"]))
    unique_groups = sorted(set(groups), key=lambda g: (g[0], g[1]))
    shared_cwt_cache: dict = {}

    rows = []
    for group in unique_groups:
        # Plain tuple `==` (not a numpy array of tuples -- np.array() on a
        # list of same-length tuples builds a 2D array, silently turning
        # this into an elementwise-per-column mask instead of one bool per
        # window) so each comparison is a single True/False per window.
        test_mask = np.array([g == group for g in groups])
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        clf = SparseEvidenceGNNClassifier(epochs=epochs, cwt_cache=shared_cwt_cache, **clf_params)
        clf.fit(X_train, y_train)
        # predict() and predict_proba() each independently recompute CWT
        # features for X_test (common.py's TorchEEGClassifier caches
        # _prepare_features's output across epochs *within* one fit()/
        # predict() call, but not *between* separate calls) -- CWT is the
        # dominant per-fold cost, so calling both back-to-back would CWT the
        # test set twice for no reason. Call predict_proba once and derive
        # the predicted class the same way predict() does internally
        # (argmax over self.classes_) instead.
        proba = clf.predict_proba(X_test)
        y_score = proba[:, 1]
        y_pred = clf.classes_[np.argmax(proba, axis=1)]

        row = {
            "subject": group[0],
            "run": group[1],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_test_ictal": int(y_test.sum()),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "average_precision": average_precision_score(y_test, y_score),
        }
        try:
            row["roc_auc"] = roc_auc_score(y_test, y_score)
        except ValueError:
            row["roc_auc"] = float("nan")  # only one class present in this fold
        rows.append(row)
        print(
            f"  fold subject={group[0]} run={group[1]}: "
            f"n_test={row['n_test']} (ictal={row['n_test_ictal']})  "
            f"precision={row['precision']:.3f} recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} auc_pr={row['average_precision']:.3f}"
        )

    print(f"  CWT cache: {len(shared_cwt_cache)} unique (window, channel) transforms cached across all folds")
    return pd.DataFrame(rows)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--window-length", type=float, default=4.0, help="Window length in seconds (default 4.0, untuned)."
    )
    parser.add_argument(
        "--step-size", type=float, default=4.0, help="Hop between window starts, in seconds (default 4.0, untuned)."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Tiny/fast run to check the wiring (small window count via a coarse "
            "step_size, epochs=2). Does NOT test whether the model can learn "
            "anything real -- see run_dense_edge_gru_temporal.py's own smoke mode "
            "in BCI/tests for the same convention."
        ),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    window_length = args.window_length
    step_size = args.step_size
    epochs = args.epochs
    subjects = args.subjects
    if args.smoke:
        print("=== SMOKE MODE: verifying wiring only, not model quality ===")
        window_length, step_size, epochs = 4.0, 30.0, 2
        subjects = subjects[:1]

    print(f"Building windowed dataset for subjects={subjects} "
          f"(window_length={window_length}s, step_size={step_size}s)...")
    X, y, metadata = _build_windowed_dataset(subjects, window_length, step_size)
    print(f"  X: {X.shape}  ictal windows: {int(y.sum())}/{len(y)} "
          f"({100 * y.mean():.1f}%)")

    clf_params = dict(DENSE_EDGE_GRU_PARAMS)
    clf_params["seed"] = args.seed
    clf_params["device"] = args.device

    print(f"Running leave-one-seizure-out (epochs={epochs})...")
    results = leave_one_seizure_out(X, y, metadata, clf_params, epochs)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"leave_one_seizure_out_{run_id}.csv"
    results.to_csv(results_path, index=False)
    print(f"\nWrote per-fold results to {results_path}")

    numeric_cols = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
    means = results[numeric_cols].mean(numeric_only=True)
    print("\n=== Mean across folds ===")
    print(means.to_string())


if __name__ == "__main__":
    argument_parser = _build_argument_parser()
    main(argument_parser.parse_args())
