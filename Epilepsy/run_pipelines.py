"""Entry point for the Epilepsy dense-edge-GRU pipeline.

    python Epilepsy/run_pipelines.py --subjects 1
    python Epilepsy/run_pipelines.py --smoke
    python Epilepsy/run_pipelines.py --label-mode prediction --smoke

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

--label-mode {detection, prediction}: two DELIBERATELY separate testing
paradigms sharing this file's plumbing (dataset loading, model, caches) but
kept independent everywhere it matters -- separate label rule (see
paradigms/continuous_labeling.py), separate leave-one-seizure-out fold
function (below), separate hyperparameter block (PREDICTION_GRU_PARAMS vs.
DENSE_EDGE_GRU_PARAMS -- see the comment above _SHARED_ARCH_PARAMS for why),
separate output directory (results/ vs. results/prediction/), and separate
evaluation (prediction mode adds event-level hit/miss + false-alarms/hour on
top of the window-level metrics both modes report). See results/README.md
for why detection and prediction scores are not directly comparable.
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
from Epilepsy.pipelines.cwt_gnn_classifiers import (
    SparseEvidenceGNNClassifier,
)
from Epilepsy.pipelines.cwt_window_cache import DiskCWTCache, default_cwt_cache_root
from Epilepsy.pipelines.dense_edge_cache import default_dense_edge_cache_root


DEFAULT_SUBJECTS = [1]

# label_mode="prediction" defaults (task: add SPH/SOP as configurable
# parameters). Starting points, not tuned -- see
# paradigms.continuous_labeling.ContinuousLabelingParadigm's docstring for
# the exact interval rules these drive.
DEFAULT_SPH = 300.0  # seizure prediction horizon, seconds
DEFAULT_SOP = 1800.0  # seizure occurrence period, seconds
DEFAULT_POSTICTAL_BUFFER = 1800.0  # seconds after offset excluded from interictal

# Parameters shared by BOTH label modes: pure model architecture (channel
# encoder, GNN/GRU shape, coherence/dense-edge settings, device, caching) and
# training-dynamics knobs that were NOT specifically tuned against
# detection's behavior. epochs/batch_size/learning_rate are excluded from
# this block on purpose -- those WERE empirically tuned against detection's
# easy, fast-converging signal (see the 2026-08-16 session notes under
# Epilepsy/Session_notes/), and prediction is a harder, noisier task that may
# need different values entirely. Tuning one mode's epochs/batch_size/
# learning_rate must not silently move the other's -- see
# DENSE_EDGE_GRU_PARAMS / PREDICTION_GRU_PARAMS below.
_SHARED_ARCH_PARAMS: dict[str, object] = dict(
    seed=42,
    sampling_rate=256,
    lowest=1.0,
    highest=40.0,
    # 2026-08-16: halved from 16 -- both the CWT and dense-edge disk caches
    # scale linearly in nfreqs, and a real (non-smoke) 7-fold run's cache
    # didn't fit in available disk space at 16 (see the 2026-08-15/16
    # session note's disk-budget investigation). Coarser frequency
    # resolution across the whole model, not just the caches -- a real
    # tradeoff, not a free win the way compression is.
    nfreqs=8,
    cwt_resample_n_time=None,
    coherence_threshold=0.99,
    phase_threshold_deg=10.0,
    channel_encoder_dilation=5,
    hidden_dim=8,
    channel_embed_dim=8,
    weight_decay=1e-4,
    grad_clip_norm=0.1,
    normalize_input=True,
    validation_split=0.0,
    device="mps",
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
    # 2026-08-16: doubled from 8 -- halves the dense-edge cache's T axis
    # (only the dense-edge cache, not CWT, since this downsample happens
    # after CWT). Same disk-budget motivation as nfreqs above; coarser
    # temporal resolution feeding dense_edge_conv/the GRU.
    dense_edge_time_downsample=16,
    dense_edge_temporal_mode="rnn",
    dense_conv_kernel_size=5,
    dense_conv_pool_size=4,
    time_averaged_graph=False,
    verbose=1,
)

# label_mode="detection" training dynamics (device × batch-size × epochs
# measured, see the 2026-08-16 session notes).
DENSE_EDGE_GRU_PARAMS: dict[str, object] = dict(
    _SHARED_ARCH_PARAMS,
    batch_size=32,
    learning_rate=2e-3,
)

# label_mode="prediction" training dynamics -- its OWN block (not a copy
# reused by reference) so future tuning of one mode can't silently move the
# other. Currently starts as the same numbers as DENSE_EDGE_GRU_PARAMS --
# a reasonable starting point, nothing more; prediction's positive rate and
# task difficulty are meaningfully different (see
# leave_one_seizure_out_prediction below), so these are expected to need
# their own tuning pass, not validated yet.
PREDICTION_GRU_PARAMS: dict[str, object] = dict(
    _SHARED_ARCH_PARAMS,
    batch_size=32,
    learning_rate=2e-3,
)

DEFAULT_DETECTION_EPOCHS = 20
# Starting point, same reasoning as PREDICTION_GRU_PARAMS above -- kept as
# its own constant (not a read of DEFAULT_DETECTION_EPOCHS) so it can move
# independently once prediction mode has its own evidence of where its
# training-loss flatline lands.
DEFAULT_PREDICTION_EPOCHS = 20

# 2026-08-16: --smoke's prediction-mode file cap. Loading the full subject
# (needed for real runs -- see _build_windowed_dataset's docstring) measured
# ~180KB/s from PhysioNet regardless of local connection speed (confirmed:
# the same link hit ~3.4MB/s against a CDN, so this is PhysioNet-side
# throttling, not fixable from here) -- ~33 extra files would make --smoke
# cost 1-3+ hours instead of the "verify the wiring, fast" it's supposed to
# be. 5 extra (evenly spaced, not just the first 5) still exercises the
# real "seizure-free recordings supply negatives" path and gives every fold
# at least one, just with a much smaller download (~200MB, ~20min at the
# measured rate).
SMOKE_MAX_INTERICTAL_RECORDINGS = 5


def _build_windowed_dataset(
    subjects: list[int],
    window_length: float,
    step_size: float,
    label_mode: str = "detection",
    sph: float = DEFAULT_SPH,
    sop: float = DEFAULT_SOP,
    postictal_buffer: float = DEFAULT_POSTICTAL_BUFFER,
    interictal_buffer: float | None = None,
    max_interictal_recordings: int | None = None,
):
    """CHBMIT + ContinuousLabelingParadigm -> (X, y, metadata_df) for `subjects`.

    label_mode="detection": restricts each subject to just its seizure-
    containing recordings (see CHBMIT.list_seizure_records) -- same
    reasoning as tests/test_chb_mit_continuous_labeling.py: avoids
    downloading/windowing every seizure-free recording for a subject when
    only the seizure ones are needed (interictal windows come from the
    non-ictal parts of those same recordings).

    label_mode="prediction": ALSO loads seizure-free recordings, not just
    seizure-containing ones. With the sph/sop/postictal_buffer defaults
    above, the span around a single seizure that can never count as
    negative (>= sph + sop + postictal_buffer, ~3900s+) exceeds one
    CHB-MIT recording's own length (~3600s) -- confirmed against chb01's
    actual documented seizure timings, not assumed -- so a seizure-
    containing recording can never supply its own negative/interictal
    windows under these defaults. Genuinely seizure-free recordings are the
    only source of them.

    max_interictal_recordings : int | None (default None)
        None loads every seizure-free recording for the subject (chb01:
        ~33 extra files on top of the 7 seizure ones, ~1.7GB total -- the
        real/full-run default). An int caps how many seizure-free
        recordings are loaded, chosen evenly spaced through the subject's
        recording order (not just the first N, so the sample isn't all
        clustered at the start of the monitoring session) -- used by
        --smoke to keep the download small (see SMOKE_MAX_INTERICTAL_RECORDINGS)
        while still exercising the real "seizure-free recordings supply
        negatives" code path, just with fewer of them.
    """
    discovery = CHBMIT()
    if label_mode == "prediction":
        if max_interictal_recordings is None:
            dataset = CHBMIT()  # no records filter -> every recording for `subjects` (see docstring)
        else:
            records = {}
            for subject in subjects:
                all_records = discovery.list_records(subject)
                seizure_files = [r["filename"] for r in all_records if r["seizures"]]
                interictal_files = [r["filename"] for r in all_records if not r["seizures"]]
                if len(interictal_files) > max_interictal_recordings:
                    idx = sorted(
                        {int(round(i)) for i in np.linspace(0, len(interictal_files) - 1, max_interictal_recordings)}
                    )
                    interictal_files = [interictal_files[i] for i in idx]
                records[subject] = seizure_files + interictal_files
            dataset = CHBMIT(records=records)
    else:
        records = {subject: [r["filename"] for r in discovery.list_seizure_records(subject)] for subject in subjects}
        dataset = CHBMIT(records=records)
    paradigm = ContinuousLabelingParadigm(
        window_length=window_length,
        step_size=step_size,
        label_event="seizure",
        label_mode=label_mode,
        sph=sph,
        sop=sop,
        postictal_buffer=postictal_buffer,
        interictal_buffer=interictal_buffer,
    )
    X, y, metadata = paradigm.get_data(dataset, subjects=subjects)
    return X, y, pd.DataFrame(metadata)


def leave_one_seizure_out_detection(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    clf_params: dict,
    epochs: int,
) -> pd.DataFrame:
    """Leave-one-seizure-out CV for label_mode="detection": hold out one
    recording's windows at a time.

    Each fold's group is `(subject, run)` -- one CHB-MIT recording, which
    (given the seizure-only `records` filter above) contains exactly one
    seizure's worth of ictal windows plus that recording's interictal
    windows. Trains on every other recording, tests on the held-out one.

    Folds share a single CWT cache AND a single dense-edge-input cache (see
    cwt_window_cache.py / dense_edge_cache.py): each fold excludes only one
    recording, so most windows appear in most folds' training sets --
    without sharing, every fold would recompute the same wavelet transform
    (and, for coherence_threshold_mode="fixed", the same coherence/phase
    dense-edge stack) for the same physical windows from scratch. Both
    caches are disk-backed (default root: <mne_data>/cwt_window_cache and
    <mne_data>/dense_edge_cache) so they also persist across separate
    invocations of this script, not just across folds within one run.
    """
    groups = list(zip(metadata["subject"], metadata["run"]))
    unique_groups = sorted(set(groups), key=lambda g: (g[0], g[1]))
    shared_cwt_cache = DiskCWTCache(default_cwt_cache_root())
    shared_dense_edge_cache_dir = default_dense_edge_cache_root()

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

        clf = SparseEvidenceGNNClassifier(
            epochs=epochs,
            cwt_cache=shared_cwt_cache,
            dense_edge_cache_dir=shared_dense_edge_cache_dir,
            **clf_params,
        )
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

    print(f"  CWT cache: {len(shared_cwt_cache)} unique (window, channel) transforms on disk (no in-memory cache)")
    return pd.DataFrame(rows)


def leave_one_seizure_out_prediction(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    clf_params: dict,
    epochs: int,
    window_length: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-seizure-out CV for label_mode="prediction".

    Deliberately a SEPARATE implementation from
    leave_one_seizure_out_detection above, not that function reused with an
    if/else -- see this file's module docstring for why. The fold unit here
    is a SEIZURE (metadata["seizure_id"], one value per preictal window;
    None for interictal windows), not a recording. Today's CHB-MIT usage
    (seizure-only recordings, one documented seizure each) means this
    coincides with grouping by (subject, run) the way detection's folds do,
    but the explicit `seizure_id`-based train-set exclusion below makes "no
    part of the held-out seizure's preictal window leaks into training" an
    enforced invariant rather than an accident of today's one-seizure-per-
    recording data -- it stays correct if that ever stops holding (e.g. a
    subject with two seizures documented in one recording).

    Reports window-level metrics (precision/recall/f1/average_precision/
    roc_auc) for continuity with detection's output shape, PLUS event-level
    metrics: per-seizure hit/miss (did any alarm fire among that seizure's
    own preictal windows) and false-alarms-per-hour (false positives among
    this fold's interictal windows, per hour of interictal time monitored).
    Window-level metrics alone don't answer the question a seizure
    prediction system is actually judged on, and aren't comparable to
    field-standard prediction papers without the event-level numbers -- see
    results/README.md.

    Also returns a second DataFrame, one row per held-out seizure, logging
    enough about each (duration, recording, hit/miss, preictal window
    counts/scores) that misses can later be checked for clustering by
    seizure characteristics instead of assumed to be uniform modeling
    failure.
    """
    if "seizure_id" not in metadata.columns:
        raise ValueError(
            "metadata has no seizure_id column -- was this dataset built with "
            "label_mode='prediction'?"
        )

    positive_meta = metadata[metadata["seizure_id"].notna()]
    unique_seizures = (
        positive_meta[["subject", "run", "seizure_id", "seizure_onset", "seizure_offset"]]
        .drop_duplicates(subset="seizure_id")
        .sort_values(["subject", "run", "seizure_onset"])
        .to_dict("records")
    )
    if not unique_seizures:
        raise ValueError(
            "No positive (preictal) windows survived label_mode='prediction' "
            "labeling -- sph+sop may exceed the available lead time before "
            "every documented seizure in this dataset. Nothing to "
            "leave-one-seizure-out over."
        )

    shared_cwt_cache = DiskCWTCache(default_cwt_cache_root())
    shared_dense_edge_cache_dir = default_dense_edge_cache_root()

    subject_arr = metadata["subject"].to_numpy()
    run_arr = metadata["run"].to_numpy()
    seizure_id_arr = metadata["seizure_id"].to_numpy()

    # Which (subject, run) recordings ever contribute a positive/preictal
    # window (contain a seizure) vs. never do (genuinely seizure-free).
    # Necessary because of a real constraint found while building this: with
    # sph+sop+postictal_buffer's defaults (>= 3900s combined) exceeding one
    # CHB-MIT recording's own length (~3600s), a seizure-containing
    # recording can NEVER supply its own negative/interictal windows --
    # its entire span always falls inside that seizure's positive/excluded
    # zone. Confirmed against chb01's real documented seizure timings, not
    # assumed. So every negative window in this dataset comes from a
    # genuinely seizure-free recording, and if the held-out test set were
    # just "the seizure's own recording" (mirroring detection's fold
    # boundary), every fold would have zero interictal test windows and
    # false_alarms_per_hour would be undefined everywhere. Each fold
    # instead gets an explicit, disjoint slice of the seizure-free
    # recordings for its own interictal test data.
    all_run_pairs = sorted(set(zip(subject_arr.tolist(), run_arr.tolist())))
    seizure_run_pairs = set(zip(positive_meta["subject"], positive_meta["run"]))
    interictal_only_runs = [rp for rp in all_run_pairs if rp not in seizure_run_pairs]
    n_folds = len(unique_seizures)
    # Round-robin, not all-in-one-fold, so the (limited) interictal pool is
    # spread evenly across folds rather than e.g. handed entirely to fold 0.
    fold_interictal_runs = [
        [rp for j, rp in enumerate(interictal_only_runs) if j % n_folds == i] for i in range(n_folds)
    ]
    if not interictal_only_runs:
        print(
            "  WARNING: no genuinely seizure-free recordings in this dataset -- "
            "every fold's false_alarms_per_hour will be undefined (zero "
            "interictal test windows). Was this built with label_mode="
            "'prediction' loading the full subject, not just "
            "list_seizure_records?"
        )

    fold_rows = []
    per_seizure_rows = []
    for fold_i, seizure in enumerate(unique_seizures):
        subject, run, seizure_id = seizure["subject"], seizure["run"], seizure["seizure_id"]
        onset, offset = seizure["seizure_onset"], seizure["seizure_offset"]

        test_run_pairs = {(subject, run), *fold_interictal_runs[fold_i]}
        test_mask = np.array(
            [(s, r) in test_run_pairs for s, r in zip(subject_arr.tolist(), run_arr.tolist())]
        )
        # Explicit seizure_id exclusion, on top of the recording-level
        # test_mask above -- see this function's docstring for why this is
        # the line that actually enforces the "no leakage of the held-out
        # seizure's preictal lead-up" requirement, not an accident of
        # today's one-seizure-per-recording grouping.
        train_mask = (~test_mask) & (seizure_id_arr != seizure_id)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        meta_test = metadata[test_mask].reset_index(drop=True)

        clf = SparseEvidenceGNNClassifier(
            epochs=epochs,
            cwt_cache=shared_cwt_cache,
            dense_edge_cache_dir=shared_dense_edge_cache_dir,
            **clf_params,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        y_score = proba[:, 1]
        y_pred = clf.classes_[np.argmax(proba, axis=1)]

        row = {
            "subject": subject,
            "run": run,
            "seizure_id": seizure_id,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_test_preictal": int(y_test.sum()),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "average_precision": average_precision_score(y_test, y_score),
        }
        try:
            row["roc_auc"] = roc_auc_score(y_test, y_score)
        except ValueError:
            row["roc_auc"] = float("nan")

        # --- event-level evaluation (task: window-level metrics alone
        # don't answer the question that matters for prediction) ---
        own_seizure_mask = (meta_test["seizure_id"] == seizure_id).to_numpy()
        n_preictal = int(own_seizure_mask.sum())
        n_preictal_hit = int((y_pred[own_seizure_mask] == 1).sum()) if n_preictal else 0
        hit = n_preictal_hit > 0

        interictal_mask = (y_test == 0)
        n_interictal = int(interictal_mask.sum())
        n_false_alarms = int((y_pred[interictal_mask] == 1).sum())
        # Monitored interictal hours, approximated as n_windows *
        # window_length. Exact only when step_size >= window_length
        # (non-overlapping windows); with overlap this overcounts monitored
        # time (each second of signal shared by multiple windows), so
        # treat false_alarms_per_hour as an upper bound on true FAR/h, not
        # an exact field-comparable figure, until windows are non-overlapping.
        interictal_hours = (n_interictal * window_length) / 3600.0
        far_per_hour = (n_false_alarms / interictal_hours) if interictal_hours > 0 else float("nan")

        row["hit"] = hit
        row["n_preictal_windows"] = n_preictal
        row["n_preictal_windows_predicted_positive"] = n_preictal_hit
        row["n_interictal_windows"] = n_interictal
        row["n_false_alarms"] = n_false_alarms
        row["interictal_hours_monitored"] = interictal_hours
        row["false_alarms_per_hour"] = far_per_hour
        fold_rows.append(row)

        # --- per-seizure outcome log (task: cheap now, expensive to
        # reconstruct later -- lets misses be checked for clustering by
        # seizure characteristics rather than assumed uniform failure) ---
        mean_preictal_score = float(y_score[own_seizure_mask].mean()) if n_preictal else float("nan")
        run_start_time = meta_test["run_start_time"].iloc[0] if len(meta_test) else None
        per_seizure_rows.append(
            {
                "subject": subject,
                "run": run,
                "seizure_id": seizure_id,
                "seizure_onset_s": onset,
                "seizure_offset_s": offset,
                "seizure_duration_s": offset - onset,
                "run_start_time": run_start_time,  # best-effort clock time; often None (see paradigm docstring)
                "hit": hit,
                "n_preictal_windows": n_preictal,
                "n_preictal_windows_predicted_positive": n_preictal_hit,
                "mean_preictal_score": mean_preictal_score,
                "n_interictal_windows": n_interictal,
                "n_false_alarms": n_false_alarms,
                "false_alarms_per_hour": far_per_hour,
            }
        )

        print(
            f"  seizure {seizure_id}: n_test={row['n_test']} preictal={n_preictal}  "
            f"hit={hit}  FAR/h={far_per_hour:.3f}  "
            f"precision={row['precision']:.3f} recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} auc_pr={row['average_precision']:.3f}"
        )

    # Cluster/timing context for the per-seizure log: gap (seconds) to the
    # nearest OTHER seizure in the same recording, directly computable from
    # in-recording onset times without needing real clock time. Cross-
    # recording gaps would need run_start_time on every seizure involved --
    # left as None when that's not available (see the paradigm's docstring
    # on why CHB-MIT/PhysioNet files don't always carry it).
    per_seizure_df = pd.DataFrame(per_seizure_rows)
    gaps = []
    for _, r in per_seizure_df.iterrows():
        same_run = per_seizure_df[
            (per_seizure_df["subject"] == r["subject"])
            & (per_seizure_df["run"] == r["run"])
            & (per_seizure_df["seizure_id"] != r["seizure_id"])
        ]
        if same_run.empty:
            gaps.append(float("nan"))
        else:
            gaps.append(float((same_run["seizure_onset_s"] - r["seizure_onset_s"]).abs().min()))
    per_seizure_df["gap_to_nearest_seizure_same_recording_s"] = gaps

    print(f"  CWT cache: {len(shared_cwt_cache)} unique (window, channel) transforms on disk (no in-memory cache)")
    return pd.DataFrame(fold_rows), per_seizure_df


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--label-mode",
        choices=["detection", "prediction"],
        default="detection",
        help=(
            "'detection': binary ictal/interictal (original behavior). "
            "'prediction': SPH/SOP-style preictal/interictal, a harder and "
            "separately-evaluated task -- see this file's module docstring."
        ),
    )
    parser.add_argument(
        "--sph", type=float, default=DEFAULT_SPH,
        help=f"label_mode=prediction only: seizure prediction horizon, seconds (default {DEFAULT_SPH}, untuned).",
    )
    parser.add_argument(
        "--sop", type=float, default=DEFAULT_SOP,
        help=f"label_mode=prediction only: seizure occurrence period, seconds (default {DEFAULT_SOP}, untuned).",
    )
    parser.add_argument(
        "--postictal-buffer", type=float, default=DEFAULT_POSTICTAL_BUFFER,
        help=f"label_mode=prediction only: seconds after offset excluded from interictal (default {DEFAULT_POSTICTAL_BUFFER}).",
    )
    parser.add_argument(
        "--interictal-buffer", type=float, default=None,
        help="label_mode=prediction only: extra padding (seconds) around each seizure required to count a window "
             "negative. Defaults to --postictal-buffer if not set.",
    )
    parser.add_argument(
        "--max-interictal-recordings", type=int, default=None,
        help=(
            "label_mode=prediction only: cap on how many seizure-free recordings to load (evenly spaced through "
            f"the subject's recording order) as a source of negative windows. Unset: real runs load every "
            f"seizure-free recording for the subject (chb01: ~33 extra files, ~1.7GB). --smoke overrides this to "
            f"{SMOKE_MAX_INTERICTAL_RECORDINGS} unless you pass this flag explicitly."
        ),
    )
    parser.add_argument(
        "--window-length", type=float, default=4.0, help="Window length in seconds (default 4.0, untuned)."
    )
    parser.add_argument(
        # 2026-08-16: default raised from 4.0 -- halves window count (and
        # thus both disk caches and per-fold training-set size) at the cost
        # of coarser window overlap. Still untuned -- a deliberate size/speed
        # tradeoff, not a validated choice of temporal resolution.
        "--step-size", type=float, default=8.0, help="Hop between window starts, in seconds (default 8.0, untuned)."
    )
    parser.add_argument(
        # 2026-08-16: default lowered from 30 -- the real (non-smoke) fold 1
        # run hit exactly zero training loss by epoch 24 and stayed there
        # through epoch 30 (see the 2026-08-16 session note / that run's own
        # log), so epochs 24-30 did nothing for that fold. 20 leaves margin
        # before that observed flatline rather than cutting at it -- not
        # validated against every fold, and this is a per-fold TRAINING-loss
        # observation, not a held-out-performance one (validation_split=0.0
        # here, so there's no direct evidence of where held-out performance
        # itself peaks). A train-loss-based early-stopping criterion would
        # replace this guess with a measured per-fold cutoff; not built yet.
        # default=None: resolved per --label-mode in main() (see
        # DEFAULT_DETECTION_EPOCHS / DEFAULT_PREDICTION_EPOCHS) unless
        # explicitly overridden here.
        "--epochs", type=int, default=None,
        help="Override epoch count for either mode. Unset: defaults depend on --label-mode "
             f"(detection={DEFAULT_DETECTION_EPOCHS}, prediction={DEFAULT_PREDICTION_EPOCHS}, independently tunable).",
    )
    # 2026-08-16: default changed from "auto" -- main() below unconditionally
    # overwrites clf_params["device"] with this value, which was silently
    # clobbering DENSE_EDGE_GRU_PARAMS's device="mps" on every invocation
    # that didn't pass --device explicitly. "auto" also only ever checks
    # CUDA (resolve_torch_device, common.py), so it fell back to CPU on this
    # Apple Silicon machine regardless -- measured ~4x faster per epoch on
    # mps+batch_size=32 vs the cpu+batch_size=8 this was silently running
    # under (see the 2026-08-16 session note).
    parser.add_argument("--device", default="mps")
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
    label_mode = args.label_mode
    subjects = args.subjects
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = DEFAULT_PREDICTION_EPOCHS if label_mode == "prediction" else DEFAULT_DETECTION_EPOCHS
    max_interictal_recordings = args.max_interictal_recordings
    if args.smoke:
        print(f"=== SMOKE MODE ({label_mode}): verifying wiring only, not model quality ===")
        window_length, step_size, epochs = 4.0, 30.0, 2
        subjects = subjects[:1]
        if label_mode == "prediction" and max_interictal_recordings is None:
            max_interictal_recordings = SMOKE_MAX_INTERICTAL_RECORDINGS

    print(f"Building windowed dataset for subjects={subjects} label_mode={label_mode} "
          f"(window_length={window_length}s, step_size={step_size}s)...")
    X, y, metadata = _build_windowed_dataset(
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
    label_word = "preictal" if label_mode == "prediction" else "ictal"
    print(f"  X: {X.shape}  {label_word} windows: {int(y.sum())}/{len(y)} "
          f"({100 * y.mean():.1f}%)")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    _write_results_readme(output_dir)

    if label_mode == "prediction":
        clf_params = dict(PREDICTION_GRU_PARAMS)
        clf_params["seed"] = args.seed
        clf_params["device"] = args.device

        print(f"Running leave-one-seizure-out prediction (epochs={epochs}, "
              f"sph={args.sph}s, sop={args.sop}s)...")
        results, per_seizure = leave_one_seizure_out_prediction(
            X, y, metadata, clf_params, epochs, window_length
        )

        # Separate output path (task 6, bullet 1): never pooled with
        # detection's results/leave_one_seizure_out_*.csv by a downstream
        # script or summary table that isn't expecting a different label
        # rule and different achievable ceiling -- see results/README.md.
        prediction_dir = output_dir / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        results_path = prediction_dir / f"prediction_leave_one_seizure_out_{run_id}.csv"
        per_seizure_path = prediction_dir / f"prediction_per_seizure_{run_id}.csv"
        results.to_csv(results_path, index=False)
        per_seizure.to_csv(per_seizure_path, index=False)
        print(f"\nWrote per-fold results to {results_path}")
        print(f"Wrote per-seizure log to {per_seizure_path}")

        numeric_cols = [
            "accuracy", "precision", "recall", "f1", "average_precision", "roc_auc",
            "false_alarms_per_hour",
        ]
        means = results[numeric_cols].mean(numeric_only=True)
        n_hits = int(results["hit"].sum())
        n_seizures = len(results)
        print("\n=== Mean across folds (label_mode=prediction; NOT comparable to detection's numbers) ===")
        print(means.to_string())
        print(f"event-level hit rate: {n_hits}/{n_seizures} ({100 * n_hits / n_seizures:.1f}%)")
    else:
        clf_params = dict(DENSE_EDGE_GRU_PARAMS)
        clf_params["seed"] = args.seed
        clf_params["device"] = args.device

        print(f"Running leave-one-seizure-out detection (epochs={epochs})...")
        results = leave_one_seizure_out_detection(X, y, metadata, clf_params, epochs)

        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"leave_one_seizure_out_{run_id}.csv"
        results.to_csv(results_path, index=False)
        print(f"\nWrote per-fold results to {results_path}")

        numeric_cols = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
        means = results[numeric_cols].mean(numeric_only=True)
        print("\n=== Mean across folds ===")
        print(means.to_string())


def _write_results_readme(output_dir: Path) -> None:
    """Task 6, bullet 5: a short, hard-to-miss note that detection and
    prediction scores are not directly comparable. Idempotent -- overwrites
    with the same fixed content every run, cheap enough not to bother
    checking existence first."""
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# Results: detection vs. prediction are NOT comparable\n\n"
        "This directory holds output from two deliberately separate testing "
        "paradigms (see Epilepsy/run_pipelines.py's module docstring):\n\n"
        "- `leave_one_seizure_out_*.csv` (this directory): `--label-mode "
        "detection` -- binary ictal/interictal, recognizing a seizure that's "
        "already happening.\n"
        "- `prediction/prediction_leave_one_seizure_out_*.csv` + "
        "`prediction/prediction_per_seizure_*.csv`: `--label-mode prediction` "
        "-- SPH/SOP-style preictal/interictal, predicting a seizure before it "
        "starts.\n\n"
        "**A lower prediction score is not necessarily a worse model.** These "
        "are different tasks with different achievable ceilings -- "
        "detection's positive windows sit inside the seizure's own abnormal "
        "signal, while prediction's positive windows are, by definition, "
        "signal recorded before anything overtly abnormal has happened yet. "
        "Do not average, rank, or otherwise pool rows from `detection` and "
        "`prediction` files together in any downstream script or summary "
        "table.\n\n"
        "Prediction results also carry event-level metrics (`hit`, "
        "`false_alarms_per_hour` per held-out seizure) that detection's "
        "output doesn't have and that window-level precision/recall/F1/"
        "average_precision/roc_auc can't substitute for -- see "
        "`leave_one_seizure_out_prediction`'s docstring in run_pipelines.py.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    argument_parser = _build_argument_parser()
    main(argument_parser.parse_args())
