"""
One-epoch, high-detail debug runner for Epilepsy/run_pipelines.py.

Place this file next to:
    Epilepsy/run_pipelines.py

Run, for example:
    python Epilepsy/debug_one_epoch.py

Or:
    python Epilepsy/debug_one_epoch.py --pipeline dense_edge_gru
    python Epilepsy/debug_one_epoch.py --pipeline dense_edge
    python Epilepsy/debug_one_epoch.py --pipeline truong_stft_cnn

The purpose is NOT model evaluation. It is to make one complete fold
transparent and identify where time/memory/errors are occurring.

It:
  - builds the same dataset as run_pipelines.py
  - uses subject 1 by default
  - uses prediction mode by default
  - runs exactly ONE epoch
  - runs only ONE leave-one-seizure-out fold
  - prints detailed timing and array information
  - prints model parameters
  - prints training/test label distributions
  - prints prediction statistics
  - prints memory usage where available
  - writes a timestamped text log next to this file

No changes to run_pipelines.py are required.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Import the actual pipeline module rather than duplicating its configuration.
import Epilepsy.run_pipelines as rp


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Tee:
    """Write output both to stdout and to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


LOG_FILE = None


def log(message: str = ""):
    """Timestamped diagnostic output."""
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] {message}")


def section(title: str):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def timed(label: str):
    """Simple context manager for timing."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            log(f"START: {label}")
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.perf_counter() - self.start
            if exc is None:
                log(f"END:   {label} -- {elapsed:.3f} sec")
            else:
                log(f"FAIL:  {label} -- {elapsed:.3f} sec")
            return False

    return Timer()


# ---------------------------------------------------------------------------
# Memory diagnostics
# ---------------------------------------------------------------------------

def memory_info():
    """
    Best-effort memory reporting.

    On macOS this uses resource.getrusage for process RSS.
    On Linux it does the same, with the usual platform-specific units.
    """
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if sys.platform == "darwin":
            rss_mb = rss / (1024 ** 2)
        else:
            rss_mb = rss / 1024

        log(f"PROCESS MAX RSS: {rss_mb:.1f} MB")
    except Exception as exc:
        log(f"Could not read process memory: {exc}")

    try:
        import torch

        if torch.backends.mps.is_available():
            log("TORCH MPS: available")

            # These APIs vary between torch versions, so keep this optional.
            try:
                allocated = torch.mps.current_allocated_memory()
                log(f"MPS allocated: {allocated / (1024 ** 2):.1f} MB")
            except Exception:
                pass

            try:
                driver = torch.mps.driver_allocated_memory()
                log(f"MPS driver allocated: {driver / (1024 ** 2):.1f} MB")
            except Exception:
                pass

        elif torch.cuda.is_available():
            log("TORCH CUDA: available")
            try:
                log(
                    f"CUDA allocated: "
                    f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB"
                )
                log(
                    f"CUDA reserved: "
                    f"{torch.cuda.memory_reserved() / (1024 ** 2):.1f} MB"
                )
            except Exception:
                pass
        else:
            log("TORCH GPU: no CUDA/MPS backend available")

    except Exception as exc:
        log(f"Could not inspect torch memory: {exc}")


# ---------------------------------------------------------------------------
# Array / dataframe diagnostics
# ---------------------------------------------------------------------------

def describe_array(name: str, arr):
    arr = np.asarray(arr)

    log(
        f"{name}: "
        f"shape={arr.shape}, "
        f"dtype={arr.dtype}, "
        f"ndim={arr.ndim}, "
        f"n={arr.size}"
    )

    if arr.size:
        try:
            log(
                f"{name}: "
                f"min={np.nanmin(arr):.6g}, "
                f"max={np.nanmax(arr):.6g}, "
                f"mean={np.nanmean(arr):.6g}, "
                f"std={np.nanstd(arr):.6g}"
            )
        except Exception:
            pass

        log(
            f"{name}: "
            f"nan={np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.number) else 'n/a'}"
        )

        try:
            log(f"{name}: contiguous={arr.flags['C_CONTIGUOUS']}")
        except Exception:
            pass

        try:
            log(f"{name}: approx memory={arr.nbytes / (1024 ** 2):.2f} MB")
        except Exception:
            pass


def describe_labels(y, name="y"):
    y = np.asarray(y)

    log(f"{name}: shape={y.shape}, dtype={y.dtype}")

    unique, counts = np.unique(y, return_counts=True)

    for value, count in zip(unique, counts):
        log(
            f"  label={value!r}: "
            f"{int(count)} windows "
            f"({100.0 * count / len(y):.2f}%)"
        )


def describe_metadata(metadata: pd.DataFrame):
    log(f"metadata shape={metadata.shape}")
    log(f"metadata columns={list(metadata.columns)}")

    for col in metadata.columns:
        try:
            n_unique = metadata[col].nunique(dropna=False)
            log(f"  {col}: dtype={metadata[col].dtype}, unique={n_unique}")
        except Exception:
            pass

    if "subject" in metadata:
        log(f"subjects: {sorted(metadata['subject'].dropna().unique().tolist())}")

    if "run" in metadata:
        runs = metadata["run"].dropna().unique().tolist()
        log(f"runs: {len(runs)} unique")

    if "seizure_id" in metadata:
        seizure_ids = metadata["seizure_id"].dropna().unique().tolist()
        log(f"seizure IDs: {seizure_ids}")

    if "window_start" in metadata:
        try:
            log(
                "window_start range: "
                f"{metadata['window_start'].min()} -> "
                f"{metadata['window_start'].max()}"
            )
        except Exception:
            pass

    print()
    print("First metadata rows:")
    print(metadata.head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Torch diagnostics
# ---------------------------------------------------------------------------

def describe_torch():
    section("TORCH ENVIRONMENT")

    try:
        import torch

        log(f"torch version: {torch.__version__}")
        log(f"python: {platform.python_version()}")
        log(f"platform: {platform.platform()}")
        log(f"CPU count: {os.cpu_count()}")

        log(f"MPS built: {torch.backends.mps.is_built()}")
        log(f"MPS available: {torch.backends.mps.is_available()}")

        log(f"CUDA built: {torch.backends.cuda.is_built()}")
        log(f"CUDA available: {torch.cuda.is_available()}")

        try:
            log(f"torch threads: {torch.get_num_threads()}")
        except Exception:
            pass

        try:
            log(f"torch interop threads: {torch.get_num_interop_threads()}")
        except Exception:
            pass

    except Exception as exc:
        log(f"Could not import torch: {exc}")


# ---------------------------------------------------------------------------
# Main debug procedure
# ---------------------------------------------------------------------------

def run_debug(args):
    section("ONE-EPOCH DEBUG RUN")

    log(f"Started: {datetime.now().isoformat()}")
    log(f"Repository root: {REPO_ROOT}")
    log(f"Script directory: {HERE}")
    log(f"Python executable: {sys.executable}")

    describe_torch()
    memory_info()

    # -----------------------------------------------------------------------
    # Resolve exactly the same defaults as run_pipelines.py
    # -----------------------------------------------------------------------

    pipeline = args.pipeline
    label_mode = args.label_mode

    if pipeline == "truong_stft_cnn":
        label_mode = "prediction"

    if label_mode == "prediction":
        window_length = (
            args.window_length
            if args.window_length is not None
            else rp.DEFAULT_TRUONG_WINDOW_LENGTH
        )
        step_size = (
            args.step_size
            if args.step_size is not None
            else rp.DEFAULT_TRUONG_STEP_SIZE
        )
    else:
        window_length = (
            args.window_length
            if args.window_length is not None
            else 4.0
        )
        step_size = (
            args.step_size
            if args.step_size is not None
            else 8.0
        )

    # IMPORTANT:
    # This debug script deliberately forces one epoch regardless of the
    # value in run_pipelines.py.
    epochs = 1

    subjects = args.subjects

    if args.max_interictal_recordings is not None:
        max_interictal_recordings = args.max_interictal_recordings
    elif label_mode == "prediction":
        max_interictal_recordings = rp.SMOKE_MAX_INTERICTAL_RECORDINGS
    else:
        max_interictal_recordings = None

    if label_mode == "prediction":
        negative_to_positive_ratio = args.negative_to_positive_ratio
        if negative_to_positive_ratio is None:
            negative_to_positive_ratio = (
                rp.DEFAULT_TRUONG_NEGATIVE_TO_POSITIVE_RATIO
                if pipeline == "truong_stft_cnn"
                else rp.DEFAULT_NEGATIVE_TO_POSITIVE_RATIO
            )
    else:
        negative_to_positive_ratio = None

    section("RESOLVED CONFIGURATION")

    log(f"pipeline                = {pipeline}")
    log(f"label_mode              = {label_mode}")
    log(f"subjects                = {subjects}")
    log(f"epochs                  = {epochs}  <-- FORCED")
    log(f"window_length           = {window_length}")
    log(f"step_size               = {step_size}")
    log(f"sph                     = {args.sph}")
    log(f"sop                     = {args.sop}")
    log(f"postictal_buffer        = {args.postictal_buffer}")
    log(f"max_interictal_records  = {max_interictal_recordings}")
    log(f"negative:positive       = {negative_to_positive_ratio}")
    log(f"device                  = {args.device}")
    log(f"seed                    = {args.seed}")

    # -----------------------------------------------------------------------
    # Dataset construction
    # -----------------------------------------------------------------------

    section("DATASET CONSTRUCTION")

    dataset_start = time.perf_counter()

    with timed("BUILD WINDOWED DATASET"):
        X, y, metadata = rp._build_windowed_dataset(
            subjects,
            window_length,
            step_size,
            label_mode=label_mode,
            sph=args.sph,
            sop=args.sop,
            postictal_buffer=args.postictal_buffer,
            interictal_buffer=args.interictal_buffer,
            max_interictal_recordings=max_interictal_recordings,
        )

    dataset_elapsed = time.perf_counter() - dataset_start

    log(f"Dataset construction total: {dataset_elapsed:.3f} sec")

    describe_array("X", X)
    describe_labels(y)
    describe_metadata(metadata)

    memory_info()

    # -----------------------------------------------------------------------
    # Identify folds
    # -----------------------------------------------------------------------

    section("FOLD DISCOVERY")

    if label_mode == "prediction":
        positive_meta = metadata[metadata["seizure_id"].notna()]

        unique_seizures = (
            positive_meta[
                [
                    "subject",
                    "run",
                    "seizure_id",
                    "seizure_onset",
                    "seizure_offset",
                ]
            ]
            .drop_duplicates(subset="seizure_id")
            .sort_values(["subject", "run", "seizure_onset"])
            .to_dict("records")
        )

        log(f"Found {len(unique_seizures)} seizures/folds")

        for i, seizure in enumerate(unique_seizures):
            log(
                f"fold {i}: "
                f"subject={seizure['subject']} "
                f"run={seizure['run']} "
                f"seizure_id={seizure['seizure_id']} "
                f"onset={seizure['seizure_onset']} "
                f"offset={seizure['seizure_offset']}"
            )

        if not unique_seizures:
            raise RuntimeError("No prediction folds were found.")

        # Debug exactly ONE fold.
        fold_i = args.fold

        if fold_i >= len(unique_seizures):
            raise ValueError(
                f"--fold {fold_i} requested, but only "
                f"{len(unique_seizures)} folds exist."
            )

        seizure = unique_seizures[fold_i]

        log(f"DEBUGGING ONLY FOLD {fold_i}")
        log(f"held-out seizure = {seizure}")

        subject_arr = metadata["subject"].to_numpy()
        run_arr = metadata["run"].to_numpy()
        seizure_id_arr = metadata["seizure_id"].to_numpy()

        all_run_pairs = sorted(
            set(zip(subject_arr.tolist(), run_arr.tolist()))
        )

        seizure_run_pairs = set(
            zip(
                positive_meta["subject"],
                positive_meta["run"],
            )
        )

        interictal_only_runs = [
            rp
            for rp in all_run_pairs
            if rp not in seizure_run_pairs
        ]

        n_folds = len(unique_seizures)

        fold_interictal_runs = [
            [
                rp
                for j, rp in enumerate(interictal_only_runs)
                if j % n_folds == i
            ]
            for i in range(n_folds)
        ]

        subject = seizure["subject"]
        run = seizure["run"]
        seizure_id = seizure["seizure_id"]

        test_run_pairs = {
            (subject, run),
            *fold_interictal_runs[fold_i],
        }

        test_mask = np.array(
            [
                (s, r) in test_run_pairs
                for s, r in zip(
                    subject_arr.tolist(),
                    run_arr.tolist(),
                )
            ]
        )

        train_mask = (
            (~test_mask)
            & (seizure_id_arr != seizure_id)
        )

    else:
        groups = list(
            zip(
                metadata["subject"],
                metadata["run"],
            )
        )

        unique_groups = sorted(
            set(groups),
            key=lambda g: (g[0], g[1]),
        )

        log(f"Found {len(unique_groups)} recording folds")

        for i, group in enumerate(unique_groups):
            log(
                f"fold {i}: "
                f"subject={group[0]} run={group[1]}"
            )

        fold_i = args.fold

        if fold_i >= len(unique_groups):
            raise ValueError(
                f"--fold {fold_i} requested, but only "
                f"{len(unique_groups)} folds exist."
            )

        group = unique_groups[fold_i]

        log(f"DEBUGGING ONLY FOLD {fold_i}")
        log(f"held-out recording = {group}")

        test_mask = np.array(
            [g == group for g in groups]
        )
        train_mask = ~test_mask

    # -----------------------------------------------------------------------
    # Split
    # -----------------------------------------------------------------------

    section("TRAIN / TEST SPLIT")

    log(f"train windows BEFORE subsampling: {train_mask.sum()}")
    log(f"test windows: {test_mask.sum()}")

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    meta_train = metadata[train_mask].reset_index(drop=True)
    meta_test = metadata[test_mask].reset_index(drop=True)

    describe_array("X_train", X_train)
    describe_labels(y_train, "y_train")

    describe_array("X_test", X_test)
    describe_labels(y_test, "y_test")

    log(f"metadata train rows: {len(meta_train)}")
    log(f"metadata test rows:  {len(meta_test)}")

    # -----------------------------------------------------------------------
    # Training-side negative subsampling
    # -----------------------------------------------------------------------

    if (
        label_mode == "prediction"
        and negative_to_positive_ratio is not None
    ):
        section("TRAINING NEGATIVE SUBSAMPLING")

        before = len(y_train)

        with timed("SUBSAMPLE TRAINING NEGATIVES"):
            X_train, y_train, meta_train = rp._subsample_negative_windows(
                X_train,
                y_train,
                meta_train,
                negative_to_positive_ratio,
                args.seed,
            )

        after = len(y_train)

        log(
            f"Training windows: {before} -> {after}"
        )

        describe_labels(y_train, "y_train after subsampling")
        describe_array("X_train after subsampling", X_train)

    memory_info()

    # -----------------------------------------------------------------------
    # Classifier construction
    # -----------------------------------------------------------------------

    section("CLASSIFIER CONFIGURATION")

    if pipeline == "truong_stft_cnn":
        clf_params = dict(rp.TRUONG_STFT_CNN_PARAMS)
        clf_params["seed"] = args.seed
        clf_params["device"] = args.device
        if args.verbose is not None:
            clf_params["verbose"] = args.verbose

        clf_class = rp.TruongSTFTCNNClassifier

    elif pipeline in ("dense_edge", "dense_edge_gru"):
        clf_params = dict(rp._dense_family_params(pipeline, label_mode))
        clf_params["seed"] = args.seed
        clf_params["device"] = args.device
        clf_params["precompute_chunk_size"] = args.precompute_chunk_size
        clf_params["compile_dense_edge_helper"] = args.compile_dense_edge_helper
        clf_params["dense_edge_amp_bf16"] = args.dense_edge_amp_bf16
        if not args.disable_disk_cache:
            clf_params["cwt_cache"] = rp.DiskCWTCache(rp.default_cwt_cache_root())
            clf_params["dense_edge_cache_dir"] = rp.default_dense_edge_cache_root()
        if args.verbose is not None:
            clf_params["verbose"] = args.verbose

        clf_class = (
            rp.StreamingSparseEvidenceGNNClassifier
            if label_mode == "prediction"
            else rp.SparseEvidenceGNNClassifier
        )

    else:
        raise ValueError(f"unknown --pipeline {pipeline!r}")

    log(f"classifier class: {clf_class.__name__}")

    for key in sorted(clf_params):
        value = clf_params[key]
        log(f"  {key} = {value!r}")

    log(f"epochs = {epochs}")

    # -----------------------------------------------------------------------
    # Instantiate
    # -----------------------------------------------------------------------

    with timed("CLASSIFIER INITIALIZATION"):
        clf = clf_class(
            epochs=epochs,
            **clf_params,
        )

    log(f"classifier object: {clf!r}")

    # -----------------------------------------------------------------------
    # Fit -- exactly ONE epoch
    # -----------------------------------------------------------------------

    section("TRAINING -- EXACTLY ONE EPOCH")

    memory_info()

    fit_start = time.perf_counter()

    try:
        with timed("clf.fit(X_train, y_train)"):
            clf.fit(X_train, y_train)

    except Exception:
        section("TRAINING FAILED")

        elapsed = time.perf_counter() - fit_start

        log(f"Training failed after {elapsed:.3f} sec")
        log("")
        log("FULL TRACEBACK:")
        traceback.print_exc()

        memory_info()

        raise

    fit_elapsed = time.perf_counter() - fit_start

    log(f"TOTAL FIT TIME: {fit_elapsed:.3f} sec")

    memory_info()

    # -----------------------------------------------------------------------
    # Inspect fitted classifier
    # -----------------------------------------------------------------------

    section("POST-FIT CLASSIFIER STATE")

    for attr in [
        "classes_",
        "n_features_in_",
        "history_",
        "model_",
        "device",
        "epochs",
        "batch_size",
    ]:
        if hasattr(clf, attr):
            try:
                value = getattr(clf, attr)
                log(f"{attr}: {value!r}")
            except Exception as exc:
                log(f"{attr}: <error reading: {exc}>")

    # Try to show loss history if available.
    if hasattr(clf, "history_"):
        try:
            history = clf.history_

            log("Training history:")

            if isinstance(history, dict):
                for key, value in history.items():
                    log(f"  {key}: {value}")
            else:
                log(f"  {history!r}")

        except Exception as exc:
            log(f"Could not inspect history_: {exc}")

    # -----------------------------------------------------------------------
    # Test prediction
    # -----------------------------------------------------------------------

    section("TEST PREDICTION")

    log("Calling predict_proba() ONCE.")
    log("This is intentionally kept separate from predict() so CWT/features")
    log("are not recomputed unnecessarily.")

    prediction_start = time.perf_counter()

    try:
        with timed("clf.predict_proba(X_test)"):
            proba = clf.predict_proba(X_test)

    except Exception:
        section("PREDICTION FAILED")

        elapsed = time.perf_counter() - prediction_start

        log(f"Prediction failed after {elapsed:.3f} sec")
        traceback.print_exc()

        memory_info()

        raise

    prediction_elapsed = time.perf_counter() - prediction_start

    describe_array("proba", proba)

    if hasattr(clf, "classes_"):
        classes = clf.classes_
    else:
        classes = np.array([0, 1])

    y_score = proba[:, 1]

    y_pred = classes[
        np.argmax(proba, axis=1)
    ]

    describe_labels(y_pred, "y_pred")

    log(
        f"probability class-1 range: "
        f"{y_score.min():.6f} -> {y_score.max():.6f}"
    )

    log(
        f"probability class-1 mean: "
        f"{y_score.mean():.6f}"
    )

    log(
        f"probability class-1 std: "
        f"{y_score.std():.6f}"
    )

    log(
        f"predicted positive count: "
        f"{int((y_pred == 1).sum())}/{len(y_pred)}"
    )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    section("ONE-FOLD METRICS")

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    log(f"accuracy:          {accuracy_score(y_test, y_pred):.6f}")
    log(f"precision:         {precision_score(y_test, y_pred, zero_division=0):.6f}")
    log(f"recall:            {recall_score(y_test, y_pred, zero_division=0):.6f}")
    log(f"f1:                {f1_score(y_test, y_pred, zero_division=0):.6f}")
    log(
        f"average_precision: "
        f"{average_precision_score(y_test, y_score):.6f}"
    )

    try:
        log(
            f"roc_auc:           "
            f"{roc_auc_score(y_test, y_score):.6f}"
        )
    except ValueError as exc:
        log(f"roc_auc:           undefined ({exc})")

    log("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -----------------------------------------------------------------------
    # Timing summary
    # -----------------------------------------------------------------------

    section("TIMING SUMMARY")

    log(f"dataset construction: {dataset_elapsed:.3f} sec")
    log(f"training / fit:       {fit_elapsed:.3f} sec")
    log(f"test prediction:      {prediction_elapsed:.3f} sec")
    log(
        f"dataset + fit + pred: "
        f"{dataset_elapsed + fit_elapsed + prediction_elapsed:.3f} sec"
    )

    if len(y_train):
        log(
            f"training windows/sec: "
            f"{len(y_train) / fit_elapsed:.3f}"
        )

    if len(y_test):
        log(
            f"test windows/sec: "
            f"{len(y_test) / prediction_elapsed:.3f}"
        )

    memory_info()

    # -----------------------------------------------------------------------
    # Save a tiny machine-readable debug summary
    # -----------------------------------------------------------------------

    section("DEBUG SUMMARY")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": pipeline,
        "label_mode": label_mode,
        "subjects": str(subjects),
        "fold": fold_i,
        "epochs": 1,
        "window_length": window_length,
        "step_size": step_size,
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_positive": int(y_train.sum()),
        "n_test_positive": int(y_test.sum()),
        "fit_seconds": fit_elapsed,
        "prediction_seconds": prediction_elapsed,
        "dataset_seconds": dataset_elapsed,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_test, y_score)
        ),
    }

    try:
        summary["roc_auc"] = float(
            roc_auc_score(y_test, y_score)
        )
    except ValueError:
        summary["roc_auc"] = float("nan")

    for key, value in summary.items():
        log(f"{key}: {value}")

    log("")
    log("DEBUG RUN FINISHED SUCCESSFULLY.")

    gc.collect()
    memory_info()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one fold and exactly one epoch of "
            "Epilepsy/run_pipelines.py with detailed diagnostics."
        )
    )

    parser.add_argument(
        "--pipeline",
        choices=["dense_edge_gru", "dense_edge", "truong_stft_cnn"],
        default="dense_edge_gru",
    )

    parser.add_argument(
        "--label-mode",
        choices=["detection", "prediction"],
        default="prediction",
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=[1],
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Only this fold is executed. Default: 0.",
    )

    parser.add_argument(
        "--window-length",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--sph",
        type=float,
        default=rp.DEFAULT_SPH,
    )

    parser.add_argument(
        "--sop",
        type=float,
        default=rp.DEFAULT_SOP,
    )

    parser.add_argument(
        "--postictal-buffer",
        type=float,
        default=rp.DEFAULT_POSTICTAL_BUFFER,
    )

    parser.add_argument(
        "--interictal-buffer",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--max-interictal-recordings",
        type=int,
        default=None,
        help=(
            "Default for this debug script is the smoke value "
            f"({rp.SMOKE_MAX_INTERICTAL_RECORDINGS}) for prediction mode."
        ),
    )

    parser.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="mps",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--precompute-chunk-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--compile-dense-edge-helper",
        action="store_true",
        help=(
            "dense_edge_gru only: torch.compile(backend='cudagraphs') the dense-edge "
            "helper's compute_dense_edge_input -- see that constructor param's 2026-08-22 "
            "docstring in cwt_gnn_classifiers.py. CUDA only; no-op elsewhere. Measured "
            "2026-08-22: no net win on this Windows/no-Triton box (steady-state per-chunk "
            "compute unchanged, plus one-time capture overhead) -- kept as an opt-in for a "
            "Linux/Triton box where the default Inductor path (real kernel fusion) is a "
            "different, untested code path."
        ),
    )

    parser.add_argument(
        "--dense-edge-amp-bf16",
        action="store_true",
        help=(
            "dense_edge_gru only: torch.autocast(dtype=torch.bfloat16) around the "
            "non-trainable dense-edge helper's compute_dense_edge_input call -- see that "
            "constructor param's 2026-08-22 docstring in cwt_gnn_classifiers.py. CUDA only; "
            "no-op elsewhere."
        ),
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=None,
        help=(
            "Override clf_params['verbose']. 2 turns on "
            "_precompute_dense_edge_inputs's per-chunk phase-timing breakdown "
            "(dense_edge_gru/prediction pipelines only) -- see run_pipelines.py's "
            "--verbose help for details."
        ),
    )

    parser.add_argument(
        "--disable-disk-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Skip the disk-backed CWT/dense-edge cache. Unset: disabled on "
            "CUDA, enabled on MPS/CPU (see run_pipelines.resolve_disable_disk_cache). "
            "Force with --disable-disk-cache / --no-disable-disk-cache."
        ),
    )

    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path for the detailed log.",
    )

    return parser.parse_args()


def main():
    global LOG_FILE

    args = parse_args()
    args.disable_disk_cache = rp.resolve_disable_disk_cache(
        args.device, args.disable_disk_cache,
    )

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = HERE / f"debug_one_epoch_{timestamp}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as fh:
        LOG_FILE = fh

        # Tee stdout so you see the same output live and get a complete log.
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = Tee(original_stdout, fh)
        sys.stderr = Tee(original_stderr, fh)

        try:
            log(f"Detailed log file: {log_path}")
            run_debug(args)

        except KeyboardInterrupt:
            section("INTERRUPTED")
            log("Run interrupted by user.")
            traceback.print_exc()
            raise

        except Exception:
            section("UNHANDLED FAILURE")
            traceback.print_exc()
            memory_info()
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print()
    print(f"Detailed log written to: {log_path}")


if __name__ == "__main__":
    main()