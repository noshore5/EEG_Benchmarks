"""Entry point for the Epilepsy dense-edge pipelines (and, via --pipeline,
the Truong et al. 2018 STFT+CNN replica).

    python Epilepsy/run_pipelines.py --subjects 1
    python Epilepsy/run_pipelines.py --smoke
    python Epilepsy/run_pipelines.py --label-mode prediction --smoke
    python Epilepsy/run_pipelines.py --pipeline dense_edge --smoke
    python Epilepsy/run_pipelines.py --pipeline truong_stft_cnn --smoke

--pipeline {dense_edge_gru, dense_edge, truong_stft_cnn}: which CLASSIFIER
runs, on top of --label-mode's which TASK it's trained for. "dense_edge_gru"
(default) is this file's own SparseEvidenceGNNClassifier with
dense_edge_temporal_mode="rnn" (per-edge GRU) and respects --label-mode
normally. "dense_edge" is the same classifier with
dense_edge_temporal_mode="conv" (Conv2d + pool over time) -- same
event_mode="dense" graph, same leave-one-seizure-out loops, same
--label-mode. Results go under results/dense_edge/ so they are never pooled
with the GRU CSVs under results/ and results/prediction/. "truong_stft_cnn"
is TruongSTFTCNNClassifier (see
Epilepsy/pipelines/truong_stft_cnn_classifier.py's module docstring for
exactly what it replicates from the paper) -- a prediction-only classifier
by construction, so it forces label_mode="prediction" regardless of
--label-mode, uses its OWN hyperparameter block (TRUONG_STFT_CNN_PARAMS),
its own leave-one-seizure-out loop (leave_one_seizure_out_truong, below --
deliberately not a branch inside leave_one_seizure_out_prediction, same
"kept independent everywhere it matters" reasoning as --label-mode gets),
its own window_length/step_size defaults (30s/30s, the paper's own segment
length, vs. dense_edge_gru's 4s/8s), and its own results/truong_stft_cnn/
output directory. Dataset loading (_build_windowed_dataset,
_subsample_negative_windows below) IS shared across --pipeline values --
that part is generic to any label_mode="prediction" run, not GNN-specific.

This is the epilepsy counterpart to BCI/run_pipelines.py's "dense_edge" /
"dense_edge_gru" pipelines (SparseEvidenceGNNClassifier, event_mode="dense",
dense_edge_temporal_mode="conv" / "rnn") -- but NOT a copy of that script.
Two things there don't carry over:

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
function (below), separate hyperparameter block (the four DENSE_EDGE_* /
PREDICTION_* dicts -- see the comment above _SHARED_ARCH_PARAMS for why),
separate output directory (results/ vs. results/prediction/ for GRU,
results/dense_edge/ for conv), and separate evaluation (prediction mode
adds event-level hit/miss + false-alarms/hour on top of the window-level
metrics both modes report). See results/README.md for why detection and
prediction scores are not directly comparable.
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
    StreamingSparseEvidenceGNNClassifier,
)
from Epilepsy.pipelines.cwt_window_cache import DISABLE_CWT_CACHE, DiskCWTCache, default_cwt_cache_root
from Epilepsy.pipelines.dense_edge_cache import default_dense_edge_cache_root
from Epilepsy.pipelines.truong_stft_cnn_classifier import TruongSTFTCNNClassifier, k_of_n_alarm


def resolve_disable_disk_cache(device, explicit: bool | None) -> bool:
    """Unset (`explicit is None`): disable the CWT/dense-edge disk cache on
    CUDA, leave it on for MPS/CPU.

    Windows/CUDA (this box, 30s prediction, 15MB/trial full-E tensors):
    `np.load` of a cache hit is slower than bf16 recompute -- measured
    26ms/file vs 14-17ms/trial GPU, and a warm cache turned yesterday's
    10.5s epochs into 26-30s. Same direction as the 2026-08-21 pod
    (removing both caches cut epoch time ~34%). MPS page-cache made those
    files cheap (~11-14s), so it keeps the cache unless asked not to.
    `--disable-disk-cache` / `--no-disable-disk-cache` override.
    """
    if explicit is not None:
        return bool(explicit)
    kind = str(device).split(":", 1)[0].lower()
    return kind == "cuda"


DEFAULT_SUBJECTS = [1]

# label_mode="prediction" defaults (task: add SPH/SOP as configurable
# parameters). Starting points, not tuned -- see
# paradigms.continuous_labeling.ContinuousLabelingParadigm's docstring for
# the exact interval rules these drive.
DEFAULT_SPH = 300.0  # seizure prediction horizon, seconds (5min)
# 2026-08-16: lowered from 1800s (30min) -- checked against chb01's real
# seizure onset times (seconds into each recording): 327, 1015, 1467, 1720,
# 1732, 1862, 2996. At sph+sop=2100s (35min) required lead time, 6/7
# seizures had their preictal window at least partly truncated by their
# own recording's start (only chb01_03's 2996s onset had room for the
# full window) -- a data-fit problem, not a model problem. At sph+sop=1200s
# (20min), 5/7 get a full, uncensored preictal window. Narrows the "how far
# in advance" claim in exchange for a healthier dataset to actually test
# whether the model learns anything real; SPH left unchanged.
DEFAULT_SOP = 900.0  # seizure occurrence period, seconds (15min)
DEFAULT_POSTICTAL_BUFFER = 1800.0  # seconds after offset excluded from interictal

# Parameters shared by BOTH dense-family pipelines and BOTH label modes:
# pure model architecture (channel encoder, GNN / dense-edge temporal block,
# coherence/dense-edge settings, device, caching) and training-dynamics knobs
# that were NOT specifically tuned against detection's behavior.
# epochs/batch_size/learning_rate are excluded from this block on purpose --
# those WERE empirically tuned against detection's easy, fast-converging
# signal (see the 2026-08-16 session notes under Epilepsy/Session_notes/),
# and prediction is a harder, noisier task that may need different values
# entirely. Tuning one mode's epochs/batch_size/learning_rate must not
# silently move the other's -- see the four DENSE_EDGE_* / PREDICTION_*
# dicts below. dense_edge_temporal_mode lives in those dicts too (conv vs
# rnn) rather than here, so flipping the shared default cannot silently
# turn a GRU run into a Conv2d run.
_SHARED_ARCH_PARAMS: dict[str, object] = dict(
    channel_subset_k=None,          # None = full mesh
    channel_subset_metric="abs_cosine",
    seed=42,
    sampling_rate=256,
    lowest=8.0,
    highest=40.0,
    # 2026-08-16: halved from 16 -- both the CWT and dense-edge disk caches
    # scale linearly in nfreqs, and a real (non-smoke) 7-fold run's cache
    # didn't fit in available disk space at 16 (see the 2026-08-15/16
    # session note's disk-budget investigation). Coarser frequency
    # resolution across the whole model, not just the caches -- a real
    # tradeoff, not a free win the way compression is.
    nfreqs=8,
    cwt_resample_n_time=None,
    coherence_threshold=0.90,
    phase_threshold_deg=10.0,
    channel_encoder_dilation=5,
    hidden_dim=8,
    channel_embed_dim=8,
    weight_decay=1e-4,
    grad_clip_norm=0.1,
    normalize_input=True,
    validation_split=0.0,
    # Only fires when validation_split > 0 (common.py's _train_loop needs a
    # val_loader to score val_loss). Consecutive epochs without a val_loss
    # improvement before the loop breaks; the best-val checkpoint is still
    # restored either way. None = run every epoch.
    early_stopping_patience=5,
    device="mps",
    surrogate_seed=42,
    smooth_kernel_size=(5, 3),
    coi_enabled=True,
    feature_ablation="zero_channel_embed",
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
    dense_conv_kernel_size=5,
    dense_conv_pool_size=4,
    time_averaged_graph=False,
    verbose=1,
    # 2026-08-20: flipped from the classifier's own "fcwt" default now that
    # torch-native-cwt (utils/torch_cwt.py, batched via
    # compute_cwt_real_imag_tensors_cached's batch_transform_fn) is stress-
    # tested on-pod. fcwt runs one signal at a time on CPU; this is the
    # actual fix for hours-long pod runs, not just a dependency cleanup --
    # see cwt_gnn_classifiers.py's cwt_backend param for the revert switch
    # if this ever needs to go back ("fcwt", no code change required).
    cwt_backend="torch",
)

# label_mode="detection" training dynamics (device × batch-size × epochs
# measured, see the 2026-08-16 session notes). Four independent copies
# (pipeline × label-mode) so tuning one cell cannot silently move another.
# dense_edge starts as the same numbers as dense_edge_gru plus
# dense_edge_temporal_mode="conv" -- a reasonable starting point, not a
# conv-specific tuning pass.
DENSE_EDGE_PARAMS: dict[str, object] = dict(
    _SHARED_ARCH_PARAMS,
    batch_size=736,
    learning_rate=2e-3,
    dense_edge_temporal_mode="conv",
)

DENSE_EDGE_GRU_PARAMS: dict[str, object] = dict(
    _SHARED_ARCH_PARAMS,
    batch_size=736,
    learning_rate=2e-3,
    dense_edge_temporal_mode="rnn",
)

# label_mode="prediction" training dynamics -- its OWN block (not a copy
# reused by reference) so future tuning of one mode can't silently move the
# other. Currently starts as the same numbers as the detection dicts --
# a reasonable starting point, nothing more; prediction's positive rate and
# task difficulty are meaningfully different (see
# leave_one_seizure_out_prediction below), so these are expected to need
# their own tuning pass, not validated yet.
PREDICTION_DENSE_EDGE_PARAMS: dict[str, object] = dict(
    _SHARED_ARCH_PARAMS,
    batch_size=32,
    learning_rate=2e-3,
    dense_edge_temporal_mode="conv",
    # Same validation_split=0.0 note as PREDICTION_GRU_PARAMS below --
    # copied rather than inherited so a later GRU-only val-split fix cannot
    # silently move the conv pipeline.
    validation_split=0.0,
)

PREDICTION_GRU_PARAMS: dict[str, object] = dict(
    _SHARED_ARCH_PARAMS,
    batch_size=32,
    learning_rate=2e-3,
    dense_edge_temporal_mode="rnn",
    # 2026-08-19: REVERTED to _SHARED_ARCH_PARAMS's validation_split=0.0 --
    # the apples-to-apples validation_split=0.2 fix (see git history) crashes
    # every prediction-mode dense_edge_gru run: leave_one_seizure_out_prediction
    # (above) trains via StreamingSparseEvidenceGNNClassifier, which explicitly
    # raises NotImplementedError for validation_split > 0 (see that class's fit(),
    # cwt_gnn_classifiers.py) -- lazy val_loader batching for the streaming
    # classifier was never built. Until that's implemented, dense_edge_gru
    # prediction mode trains on 100% of each fold's training windows while
    # truong_stft_cnn (unaffected -- separate classifier, TRUONG_STFT_CNN_PARAMS
    # below) trains on 80% -- a real, currently-unavoidable confound in comparing
    # the two. Detection mode is untouched (DENSE_EDGE_GRU_PARAMS still inherits
    # 0.0; SparseEvidenceGNNClassifier there is the non-streaming variant and was
    # never affected by this).
    validation_split=0.0,
)


def _dense_family_params(pipeline: str, label_mode: str) -> dict:
    """Param dict for --pipeline dense_edge / dense_edge_gru × --label-mode.

    Returns the module-level dict itself (caller must `dict(...)` copy
    before overlaying CLI overrides). Raises on anything that is not one
    of the two dense-family pipelines -- truong_stft_cnn has its own
    TRUONG_STFT_CNN_PARAMS path in main().
    """
    if pipeline == "dense_edge":
        return PREDICTION_DENSE_EDGE_PARAMS if label_mode == "prediction" else DENSE_EDGE_PARAMS
    if pipeline == "dense_edge_gru":
        return PREDICTION_GRU_PARAMS if label_mode == "prediction" else DENSE_EDGE_GRU_PARAMS
    raise ValueError(f"not a dense-family pipeline: {pipeline!r}")


def _dense_family_result_dir(
    output_dir: Path,
    pipeline: str,
    label_mode: str,
    shuffle_labels: bool,
) -> Path:
    """Where dense-family CSVs land. dense_edge_gru keeps the historical
    layout (results/leave_one_seizure_out_*.csv and results/prediction/)
    so existing GRU numbers stay GRU numbers. dense_edge goes under
    results/dense_edge/ so conv vs GRU cannot be pooled by a glob.
    """
    if pipeline == "dense_edge":
        root = output_dir / "dense_edge"
        if label_mode == "prediction":
            return root / ("prediction_shuffled_control" if shuffle_labels else "prediction")
        return root / "shuffled_control" if shuffle_labels else root
    if pipeline == "dense_edge_gru":
        if label_mode == "prediction":
            return output_dir / ("prediction_shuffled_control" if shuffle_labels else "prediction")
        return output_dir / "shuffled_control" if shuffle_labels else output_dir
    raise ValueError(f"not a dense-family pipeline: {pipeline!r}")


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

# 2026-08-17: real (non-smoke) prediction runs default to this, applied in
# main() -- the untrimmed negative pool (~23:1 at current sph/sop/
# step_size) makes a fold's training-set CWT+dense-edge tensors too big to
# fit in memory (measured OOM kill: ~52GB needed against a 16GB+15GB-swap
# machine). 5.0 targets a training set in the same ballpark as detection
# mode's proven-safe scale (~3-4k windows/fold) -- see
# _subsample_negative_windows's docstring. Starting point, not tuned.
DEFAULT_NEGATIVE_TO_POSITIVE_RATIO = 5.0

# --pipeline="truong_stft_cnn" -- Truong et al. 2018 STFT+CNN architecture
# replica (see Epilepsy/pipelines/truong_stft_cnn_classifier.py's module
# docstring for exactly what's replicated vs. deliberately not). Prediction-
# mode ONLY: it's a preictal/interictal classifier by construction, same as
# the paper -- main() below forces label_mode="prediction" for this
# pipeline regardless of --label-mode. Own constants block, not a reuse of
# DEFAULT_PREDICTION_EPOCHS/DEFAULT_NEGATIVE_TO_POSITIVE_RATIO/etc. above --
# same "independently tunable, not silently coupled" reasoning this file
# already applies to DENSE_EDGE_GRU_PARAMS vs. PREDICTION_GRU_PARAMS.
DEFAULT_TRUONG_EPOCHS = 20  # not specified by the paper -- see that module's docstring.
DEFAULT_TRUONG_WINDOW_LENGTH = 30.0  # seconds -- the paper's own preictal/interictal segment length.
DEFAULT_TRUONG_STEP_SIZE = 30.0  # seconds -- non-overlapping windows, so false_alarms_per_hour is exact,
# not an upper bound (see leave_one_seizure_out_prediction's docstring on why overlap breaks that).
DEFAULT_TRUONG_K_OF_N_K = 8  # paper Section II.D defaults
DEFAULT_TRUONG_K_OF_N_N = 10
DEFAULT_TRUONG_NEGATIVE_TO_POSITIVE_RATIO = 5.0

TRUONG_STFT_CNN_PARAMS: dict[str, object] = dict(
    seed=42,
    sampling_rate=256,
    stft_window_sec=1.0,
    stft_hop_sec=0.5,
    powerline_freq=60.0,  # CHB-MIT was recorded in the US
    notch_halfwidth_hz=3.0,
    fc1_dim=256,
    dropout=0.5,
    weight_decay=0.0,
    grad_clip_norm=None,
    validation_split=0.2,
    early_stopping_patience=5,
    device="mps",
    use_class_weights=True,
    batch_size=32,
    learning_rate=1e-3,
    verbose=1,
)


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

    Returns the FULL windowed dataset (no negative-window subsampling here
    -- that used to happen in this function, applied once before any fold
    split, which meant every fold's TEST interictal windows were also
    silently thinned to the training-side ratio: false_alarms_per_hour and
    the window-level metrics ended up computed against a sparse ~1/5 random
    slice of each recording instead of continuous monitoring. Subsampling
    now happens per-fold, train-side only, inside
    leave_one_seizure_out_prediction -- see that function's
    negative_to_positive_ratio docstring for the (still very real) OOM
    reason it exists. See the 2026-08-17 session note.
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
    metadata = pd.DataFrame(metadata)
    return X, y, metadata


def _subsample_negative_windows(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Randomly drop negative/interictal windows down to `ratio * n_positive`
    total, stratified per (subject, run) recording so every recording's
    relative share of the negative pool is preserved -- keeping the SAME
    downsample fraction within every recording rather than pooling
    globally means no recording's negative windows get zeroed out as a
    side effect of random chance, which would otherwise silently starve
    leave_one_seizure_out_prediction's round-robin interictal-recording
    fold assignment. All positive windows are always kept.
    """
    rng = np.random.default_rng(seed)
    positive_mask = y == 1
    n_positive = int(positive_mask.sum())
    negative_idx = np.flatnonzero(~positive_mask)
    target_negative = int(round(ratio * n_positive))
    if target_negative >= len(negative_idx):
        return X, y, metadata  # already under the target, nothing to trim

    keep_frac = target_negative / len(negative_idx)
    neg_meta = metadata.iloc[negative_idx]
    keep_neg_idx: list[int] = []
    for _, group in neg_meta.groupby(["subject", "run"], sort=False):
        n_keep = max(1, int(round(len(group) * keep_frac)))
        n_keep = min(n_keep, len(group))
        chosen = rng.choice(group.index.to_numpy(), size=n_keep, replace=False)
        keep_neg_idx.extend(chosen.tolist())

    keep_idx = np.sort(np.concatenate([np.flatnonzero(positive_mask), np.array(keep_neg_idx, dtype=np.int64)]))
    print(
        f"  Subsampled negative windows: {len(negative_idx)} -> {len(keep_neg_idx)} "
        f"(target ratio {ratio}:1, {n_positive} positive kept in full)"
    )
    return X[keep_idx], y[keep_idx], metadata.iloc[keep_idx].reset_index(drop=True)


def leave_one_seizure_out_detection(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    clf_params: dict,
    epochs: int,
    disable_disk_cache: bool = False,
    max_folds: int | None = None,
    skip_folds: set[int] | None = None,
) -> pd.DataFrame:
    """Leave-one-seizure-out CV for label_mode="detection": hold out one
    recording's windows at a time.

    Each fold's group is `(subject, run)` -- one CHB-MIT recording, which
    (given the seizure-only `records` filter above) contains exactly one
    seizure's worth of ictal windows plus that recording's interictal
    windows. Trains on every other recording, tests on the held-out one.

    Folds share a single dense-edge-input cache (see dense_edge_cache.py):
    each fold excludes only one recording, so most windows appear in most
    folds' training sets -- without sharing, every fold would recompute the
    same coherence/phase dense-edge stack (coherence_threshold_mode="fixed")
    for the same physical windows from scratch. Disk-backed (default root:
    <mne_data>/dense_edge_cache) so it also persists across separate
    invocations of this script, not just across folds within one run.

    2026-08-22: both caches restored and enabled unconditionally (including
    cwt_backend="torch", unlike the pre-2026-08-21 wiring, which disabled
    CWT caching specifically for the torch backend via DISABLE_CWT_CACHE --
    see cwt_window_cache.py's module docstring). This is a deliberate
    re-test on a different machine (this session's Windows/WDDM CUDA box)
    than the Linux/Runpod pod the 2026-08-21 removal was measured on; not
    yet re-validated as a net win here. If this ever runs again on a
    machine/config where recompute wins (e.g. a fast Linux GPU pod), revert
    to not passing cwt_cache/dense_edge_cache_dir here rather than assuming
    this default still holds.
    """
    groups = list(zip(metadata["subject"], metadata["run"]))
    unique_groups = sorted(set(groups), key=lambda g: (g[0], g[1]))
    if max_folds is not None:
        # Diagnostic only (same convention as --max-channels): run just the
        # first N folds, e.g. for a quick cache-vs-no-cache or timing
        # comparison without paying for every recording. Does not change
        # any fold's own train/test split -- only how many folds run.
        unique_groups = unique_groups[: max(1, int(max_folds))]
    shared_cwt_cache = DISABLE_CWT_CACHE if disable_disk_cache else DiskCWTCache(default_cwt_cache_root())
    shared_dense_edge_cache_dir = None if disable_disk_cache else default_dense_edge_cache_root()

    rows = []
    for fold_i, group in enumerate(unique_groups):
        if skip_folds and fold_i in skip_folds:
            # Diagnostic only (same convention as --max-folds): fold_i here
            # is this group's position in unique_groups (0-based, stable
            # across --max-folds truncation since that only trims the
            # tail) -- lets a crashed/partial run resume by skipping the
            # folds it already completed instead of redoing them.
            print(f"  fold {fold_i} subject={group[0]} run={group[1]}: SKIPPED (--skip-folds)")
            continue
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

    if shared_cwt_cache is DISABLE_CWT_CACHE:
        print("  CWT cache: disabled (--disable-disk-cache)")
    else:
        print(f"  CWT cache: {len(shared_cwt_cache)} unique (window, channel) transforms on disk")
    return pd.DataFrame(rows)


def leave_one_seizure_out_prediction(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    clf_params: dict,
    epochs: int,
    window_length: float,
    negative_to_positive_ratio: float | None = None,
    subsample_seed: int = 42,
    disable_disk_cache: bool = False,
    # 2026-08-23: added so this pipeline's event-level numbers are directly
    # comparable to leave_one_seizure_out_truong's, not just its window-level
    # ones -- same k_of_n_alarm (Truong et al. 2018 Section II.D), same
    # default k/n, applied the same way (per held-out (subject, run)
    # recording, in chronological window order, never across recordings).
    # Reuses truong's own constants/CLI flags (--k-of-n-k/--k-of-n-n) rather
    # than defining a separate pair -- there's no reason this pipeline's
    # smoothing window would want to differ from the baseline it's being
    # compared against.
    k_of_n_k: int = DEFAULT_TRUONG_K_OF_N_K,
    k_of_n_n: int = DEFAULT_TRUONG_K_OF_N_N,
    max_folds: int | None = None,
    skip_folds: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-seizure-out CV for label_mode="prediction".

    negative_to_positive_ratio : float | None (default None)
        If set, randomly drops each fold's TRAINING-set negative/interictal
        windows (stratified per (subject, run) recording -- see
        _subsample_negative_windows) down to at most
        `negative_to_positive_ratio * n_positive`. TEST windows are never
        subsampled -- they stay at full density so false_alarms_per_hour
        and the window-level metrics reflect continuous monitoring of the
        held-out recording(s), not a random sparse slice of them. Exists
        because the full negative pool makes a fold's training-set feature
        tensors too large to fit in memory (measured: an OOM kill training
        on the untrimmed set -- ~3.55MB/window combined CWT+dense-edge
        times ~14,700 training windows/fold is ~52GB against a
        16GB+15GB-swap machine). Subsampling used to happen once on the
        whole dataset before the fold split, which silently thinned TEST
        interictal windows too (see _build_windowed_dataset's docstring)
        -- see the 2026-08-17 session note.

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
    # CWT/dense-edge caching -- see leave_one_seizure_out_detection's
    # docstring for the full rationale and the 2026-08-22 restoration note.
    shared_cwt_cache = DISABLE_CWT_CACHE if disable_disk_cache else DiskCWTCache(default_cwt_cache_root())
    shared_dense_edge_cache_dir = None if disable_disk_cache else default_dense_edge_cache_root()

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
    # n_folds/fold_interictal_runs are sized off the FULL unique_seizures --
    # deliberately BEFORE max_folds truncates below -- since this round-robin
    # split is what gives each fold its own disjoint 1/n_folds share of the
    # (limited) interictal pool (see the comment above). Truncating first
    # would shrink n_folds too, so with e.g. max_folds=1 a single remaining
    # fold would round-robin to n_folds=1 and claim EVERY interictal-only
    # recording for its own test set -- leaving zero for training and
    # starving that fold down to one class (confirmed: this is exactly what
    # happened before this ordering fix, IndexError on predict_proba's
    # single-column output). max_folds only trims which of these real folds
    # actually run, never how big each one's own split is.
    n_folds = len(unique_seizures)
    # Round-robin, not all-in-one-fold, so the (limited) interictal pool is
    # spread evenly across folds rather than e.g. handed entirely to fold 0.
    fold_interictal_runs = [
        [rp for j, rp in enumerate(interictal_only_runs) if j % n_folds == i] for i in range(n_folds)
    ]
    if max_folds is not None:
        # Diagnostic only (same convention as --max-channels): run just the
        # first N seizures' folds, e.g. for a quick cache-vs-no-cache or
        # timing comparison without paying for every fold. fold_interictal_
        # runs above is already indexed 0..n_folds-1 against the FULL fold
        # count, so slicing unique_seizures here still lines up correctly --
        # fold_i for the folds we keep is unchanged (still 0, 1, ..., not
        # renumbered), only the trailing folds are skipped.
        unique_seizures = unique_seizures[: max(1, int(max_folds))]
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

        if skip_folds and fold_i in skip_folds:
            # Diagnostic only (same convention as --max-folds): fold_i here
            # is this seizure's position in unique_seizures, indexed against
            # the FULL (pre-max_folds) fold count -- see the max_folds
            # comment above on why that numbering is stable. Lets a
            # crashed/partial multi-fold run resume by skipping folds
            # already completed instead of redoing them.
            print(f"  fold {fold_i} seizure {seizure_id} (subject={subject} run={run}): SKIPPED (--skip-folds)")
            continue

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
        if negative_to_positive_ratio is not None:
            meta_train = metadata[train_mask].reset_index(drop=True)
            X_train, y_train, _ = _subsample_negative_windows(
                X_train, y_train, meta_train, negative_to_positive_ratio, subsample_seed
            )
        X_test, y_test = X[test_mask], y[test_mask]
        meta_test = metadata[test_mask].reset_index(drop=True)

        # StreamingSparseEvidenceGNNClassifier, NOT the detection-mode
        # SparseEvidenceGNNClassifier above -- prediction mode's training
        # sets are much larger (full-subject scope); the streaming variant
        # computes features per-batch instead of materializing the whole
        # training set's feature tensors at once, avoiding the OOM this
        # mode hit on a real run (see cwt_gnn_classifiers.py's comment on
        # StreamingSparseEvidenceGNNClassifier for the full rationale).
        clf = StreamingSparseEvidenceGNNClassifier(
            epochs=epochs,
            cwt_cache=shared_cwt_cache,
            dense_edge_cache_dir=shared_dense_edge_cache_dir,
            **clf_params,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        y_score = proba[:, 1]
        y_pred = clf.classes_[np.argmax(proba, axis=1)]

        # k-of-n smoothing (Truong et al. 2018 Section II.D), same as
        # leave_one_seizure_out_truong -- per (subject, run) recording in
        # this fold, in chronological window order (meta_test's
        # "window_start" column from ContinuousLabelingParadigm), never
        # across recordings.
        y_pred_smoothed = np.array(y_pred, dtype=np.int64, copy=True)
        time_col = "window_start" if "window_start" in meta_test.columns else None
        for (_s, _r), group in meta_test.groupby(["subject", "run"], sort=False):
            order = group.sort_values(time_col).index if time_col else group.index
            order = order.to_numpy()
            raw = (y_pred[order] == 1).astype(np.int64)
            y_pred_smoothed[order] = k_of_n_alarm(raw, k=k_of_n_k, n=k_of_n_n)

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
        interictal_mask = (y_test == 0)
        n_interictal = int(interictal_mask.sum())
        # Monitored interictal hours, approximated as n_windows *
        # window_length. Exact only when step_size >= window_length
        # (non-overlapping windows); with overlap this overcounts monitored
        # time (each second of signal shared by multiple windows), so
        # treat false_alarms_per_hour as an upper bound on true FAR/h, not
        # an exact field-comparable figure, until windows are non-overlapping.
        interictal_hours = (n_interictal * window_length) / 3600.0

        def _event_metrics(preds: np.ndarray) -> dict[str, object]:
            n_preictal = int(own_seizure_mask.sum())
            n_hit = int((preds[own_seizure_mask] == 1).sum()) if n_preictal else 0
            n_false_alarms = int((preds[interictal_mask] == 1).sum())
            far = (n_false_alarms / interictal_hours) if interictal_hours > 0 else float("nan")
            return {
                "hit": n_hit > 0,
                "n_preictal_windows_predicted_positive": n_hit,
                "n_false_alarms": n_false_alarms,
                "false_alarms_per_hour": far,
            }

        raw_events = _event_metrics(np.asarray(y_pred))
        smoothed_events = _event_metrics(y_pred_smoothed)
        n_preictal = int(own_seizure_mask.sum())

        row["hit"] = raw_events["hit"]
        row["n_preictal_windows"] = n_preictal
        row["n_preictal_windows_predicted_positive"] = raw_events["n_preictal_windows_predicted_positive"]
        row["n_interictal_windows"] = n_interictal
        row["n_false_alarms"] = raw_events["n_false_alarms"]
        row["interictal_hours_monitored"] = interictal_hours
        row["false_alarms_per_hour"] = raw_events["false_alarms_per_hour"]
        row["hit_smoothed"] = smoothed_events["hit"]
        row["n_false_alarms_smoothed"] = smoothed_events["n_false_alarms"]
        row["false_alarms_per_hour_smoothed"] = smoothed_events["false_alarms_per_hour"]
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
                "hit": raw_events["hit"],
                "hit_smoothed": smoothed_events["hit"],
                "n_preictal_windows": n_preictal,
                "n_preictal_windows_predicted_positive": raw_events["n_preictal_windows_predicted_positive"],
                "mean_preictal_score": mean_preictal_score,
                "n_interictal_windows": n_interictal,
                "n_false_alarms": raw_events["n_false_alarms"],
                "false_alarms_per_hour": raw_events["false_alarms_per_hour"],
                "false_alarms_per_hour_smoothed": smoothed_events["false_alarms_per_hour"],
            }
        )

        print(
            f"  seizure {seizure_id}: n_test={row['n_test']} preictal={n_preictal}  "
            f"hit={raw_events['hit']} (smoothed={smoothed_events['hit']})  "
            f"FAR/h={raw_events['false_alarms_per_hour']:.3f} "
            f"(smoothed={smoothed_events['false_alarms_per_hour']:.3f})  "
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

    if shared_cwt_cache is DISABLE_CWT_CACHE:
        print("  CWT cache: disabled (--disable-disk-cache)")
    else:
        print(f"  CWT cache: {len(shared_cwt_cache)} unique (window, channel) transforms on disk")
    return pd.DataFrame(fold_rows), per_seizure_df


def leave_one_seizure_out_truong(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    clf_params: dict,
    epochs: int,
    window_length: float,
    negative_to_positive_ratio: float | None = None,
    subsample_seed: int = 42,
    k_of_n_k: int = DEFAULT_TRUONG_K_OF_N_K,
    k_of_n_n: int = DEFAULT_TRUONG_K_OF_N_N,
    skip_folds: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-seizure-out CV for TruongSTFTCNNClassifier.

    Same fold-construction rules as leave_one_seizure_out_prediction above
    (one held-out seizure's recording plus a round-robin share of the
    seizure-free recordings per fold; explicit seizure_id exclusion so no
    part of the held-out seizure's preictal lead-up leaks into training) --
    deliberately a SEPARATE implementation, not that function reused with an
    if/else, for the same "kept independent everywhere it matters" reasoning
    this file's module docstring gives for detection vs. prediction. No CWT/
    dense-edge disk cache here -- STFT features are computed fresh per fold
    (they're far cheaper than CWT+dense-edge, and TruongSTFTCNNClassifier
    has no cache plumbing to share one with anyway).

    Reports window-level metrics on the RAW per-window predictions, plus
    event-level hit/false_alarms_per_hour computed BOTH on raw predictions
    and on k-of-n-smoothed ones (Truong et al. 2018 Section II.D) --
    smoothing is applied per held-out recording, in chronological window
    order, never across recordings.
    """
    if "seizure_id" not in metadata.columns:
        raise ValueError("metadata has no seizure_id column -- was this dataset built with label_mode='prediction'?")

    positive_meta = metadata[metadata["seizure_id"].notna()]
    unique_seizures = (
        positive_meta[["subject", "run", "seizure_id", "seizure_onset", "seizure_offset"]]
        .drop_duplicates(subset="seizure_id")
        .sort_values(["subject", "run", "seizure_onset"])
        .to_dict("records")
    )
    if not unique_seizures:
        raise ValueError(
            "No positive (preictal) windows survived label_mode='prediction' labeling -- sph+sop may exceed the "
            "available lead time before every documented seizure in this dataset."
        )

    subject_arr = metadata["subject"].to_numpy()
    run_arr = metadata["run"].to_numpy()
    seizure_id_arr = metadata["seizure_id"].to_numpy()

    all_run_pairs = sorted(set(zip(subject_arr.tolist(), run_arr.tolist())))
    seizure_run_pairs = set(zip(positive_meta["subject"], positive_meta["run"]))
    interictal_only_runs = [rp for rp in all_run_pairs if rp not in seizure_run_pairs]
    n_folds = len(unique_seizures)
    fold_interictal_runs = [
        [rp for j, rp in enumerate(interictal_only_runs) if j % n_folds == i] for i in range(n_folds)
    ]
    if not interictal_only_runs:
        print(
            "  WARNING: no genuinely seizure-free recordings in this dataset -- every fold's "
            "false_alarms_per_hour will be undefined."
        )

    fold_rows = []
    per_seizure_rows = []
    for fold_i, seizure in enumerate(unique_seizures):
        subject, run, seizure_id = seizure["subject"], seizure["run"], seizure["seizure_id"]
        onset, offset = seizure["seizure_onset"], seizure["seizure_offset"]

        if skip_folds and fold_i in skip_folds:
            # See leave_one_seizure_out_prediction's identical check --
            # same fold_i numbering, same "resume a partial run" purpose.
            print(f"  fold {fold_i} seizure {seizure_id} (subject={subject} run={run}): SKIPPED (--skip-folds)")
            continue

        test_run_pairs = {(subject, run), *fold_interictal_runs[fold_i]}
        test_mask = np.array([(s, r) in test_run_pairs for s, r in zip(subject_arr.tolist(), run_arr.tolist())])
        train_mask = (~test_mask) & (seizure_id_arr != seizure_id)

        X_train, y_train = X[train_mask], y[train_mask]
        if negative_to_positive_ratio is not None:
            meta_train = metadata[train_mask].reset_index(drop=True)
            X_train, y_train, _ = _subsample_negative_windows(
                X_train, y_train, meta_train, negative_to_positive_ratio, subsample_seed
            )
        X_test, y_test = X[test_mask], y[test_mask]
        meta_test = metadata[test_mask].reset_index(drop=True)

        clf = TruongSTFTCNNClassifier(epochs=epochs, **clf_params)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        y_score = proba[:, 1]
        y_pred = clf.classes_[np.argmax(proba, axis=1)]

        # k-of-n smoothing, per (subject, run) recording in this fold, in
        # chronological order -- meta_test carries a "window_start" column
        # from ContinuousLabelingParadigm; sort by it per-recording rather
        # than trusting row order.
        y_pred_smoothed = np.array(y_pred, dtype=np.int64, copy=True)
        time_col = "window_start" if "window_start" in meta_test.columns else None
        for (s, r), group in meta_test.groupby(["subject", "run"], sort=False):
            order = group.sort_values(time_col).index if time_col else group.index
            order = order.to_numpy()
            raw = (y_pred[order] == 1).astype(np.int64)
            y_pred_smoothed[order] = k_of_n_alarm(raw, k=k_of_n_k, n=k_of_n_n)

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

        own_seizure_mask = (meta_test["seizure_id"] == seizure_id).to_numpy()
        interictal_mask = y_test == 0
        interictal_hours = (int(interictal_mask.sum()) * window_length) / 3600.0

        def _event_metrics(preds: np.ndarray) -> dict[str, object]:
            n_preictal = int(own_seizure_mask.sum())
            n_hit = int((preds[own_seizure_mask] == 1).sum()) if n_preictal else 0
            n_false_alarms = int((preds[interictal_mask] == 1).sum())
            far = (n_false_alarms / interictal_hours) if interictal_hours > 0 else float("nan")
            return {
                "hit": n_hit > 0,
                "n_preictal_windows_predicted_positive": n_hit,
                "n_false_alarms": n_false_alarms,
                "false_alarms_per_hour": far,
            }

        raw_events = _event_metrics(np.asarray(y_pred))
        smoothed_events = _event_metrics(y_pred_smoothed)
        row["n_preictal_windows"] = int(own_seizure_mask.sum())
        row["n_interictal_windows"] = int(interictal_mask.sum())
        row["interictal_hours_monitored"] = interictal_hours
        row["hit"] = raw_events["hit"]
        row["n_false_alarms"] = raw_events["n_false_alarms"]
        row["false_alarms_per_hour"] = raw_events["false_alarms_per_hour"]
        row["hit_smoothed"] = smoothed_events["hit"]
        row["n_false_alarms_smoothed"] = smoothed_events["n_false_alarms"]
        row["false_alarms_per_hour_smoothed"] = smoothed_events["false_alarms_per_hour"]
        fold_rows.append(row)

        mean_preictal_score = float(y_score[own_seizure_mask].mean()) if row["n_preictal_windows"] else float("nan")
        per_seizure_rows.append(
            {
                "subject": subject,
                "run": run,
                "seizure_id": seizure_id,
                "seizure_onset_s": onset,
                "seizure_offset_s": offset,
                "seizure_duration_s": offset - onset,
                "hit": raw_events["hit"],
                "hit_smoothed": smoothed_events["hit"],
                "n_preictal_windows": row["n_preictal_windows"],
                "mean_preictal_score": mean_preictal_score,
                "false_alarms_per_hour": raw_events["false_alarms_per_hour"],
                "false_alarms_per_hour_smoothed": smoothed_events["false_alarms_per_hour"],
            }
        )

        print(
            f"  seizure {seizure_id}: n_test={row['n_test']} preictal={row['n_preictal_windows']}  "
            f"hit={raw_events['hit']} (smoothed={smoothed_events['hit']})  "
            f"FAR/h={raw_events['false_alarms_per_hour']:.3f} "
            f"(smoothed={smoothed_events['false_alarms_per_hour']:.3f})  "
            f"precision={row['precision']:.3f} recall={row['recall']:.3f} f1={row['f1']:.3f}"
        )

    return pd.DataFrame(fold_rows), pd.DataFrame(per_seizure_rows)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--pipeline",
        choices=["dense_edge_gru", "dense_edge", "truong_stft_cnn"],
        default="dense_edge_gru",
        help=(
            "'dense_edge_gru' (default): SparseEvidenceGNNClassifier with "
            "dense_edge_temporal_mode='rnn' (per-edge GRU) -- respects --label-mode. "
            "'dense_edge': the same classifier with dense_edge_temporal_mode='conv' "
            "(Conv2d + pool over time); results go under results/dense_edge/ so they "
            "are never pooled with the GRU CSVs. "
            "'truong_stft_cnn': Truong et al. 2018's STFT+CNN architecture replica "
            "(see Epilepsy/pipelines/truong_stft_cnn_classifier.py's module docstring) "
            "-- prediction-mode ONLY, forces --label-mode=prediction regardless of "
            "what's passed. Uses its own hyperparameters (TRUONG_STFT_CNN_PARAMS, "
            "DEFAULT_TRUONG_* constants above), --window-length/--step-size default "
            "to 30s/30s instead of 4s/8s to match the paper, and results are written "
            "to their own results/truong_stft_cnn/ subdirectory."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=["detection", "prediction"],
        default="prediction",
        help=(
            "'detection': binary ictal/interictal (original behavior). "
            "'prediction': SPH/SOP-style preictal/interictal, a harder and "
            "separately-evaluated task -- see this file's module docstring. "
            "Ignored (forced to 'prediction') when --pipeline=truong_stft_cnn."
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
        "--max-channels", type=int, default=None,
        help=(
            "Diagnostic only (2026-08-23): truncate X to the first N channels right after "
            "_build_windowed_dataset, before any classifier sees it -- e.g. to A/B time the "
            "dense-edge precompute stage against edge count (dense edges scale as C*(C-1)/2, "
            "so chb01's 23 channels -> 253 edges vs. 5 channels -> 10 edges). Cheap slice, no "
            "re-download/re-window needed. Not a real channel-selection feature -- picks "
            "whatever channels sort first in the EDF, not a principled electrode subset."
        ),
    )
    parser.add_argument(
        "--max-folds", type=int, default=None,
        help=(
            "Diagnostic only: run just the first N leave-one-seizure-out folds "
            "(label_mode=prediction) / leave-one-recording-out folds "
            "(label_mode=detection) instead of every one -- e.g. for a quick "
            "cache-vs-no-cache or timing A/B without paying for the whole dataset. "
            "Does not change any fold's own train/test split. Unset (default): "
            "every fold runs, current behavior."
        ),
    )
    parser.add_argument(
        "--skip-folds", type=int, nargs="+", default=None, metavar="FOLD_I",
        help=(
            "Skip these fold indices (0-based, in the same order printed as "
            "'fold N ...' -- (subject,run) order for label_mode=detection, "
            "unique_seizures order for prediction/truong) instead of running "
            "every fold. Combines with --max-folds (skip_folds is checked "
            "against the same fold_i numbering, unaffected by max_folds' own "
            "truncation). Useful to resume a partial multi-fold run without "
            "redoing folds that already finished, e.g. --skip-folds 0 1 2. "
            "Unset (default): no folds skipped."
        ),
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
        "--negative-to-positive-ratio", type=float, default=None,
        help=(
            "label_mode=prediction only: cap negative/interictal windows at this multiple of the positive/"
            "preictal count (stratified per recording), instead of keeping every negative window. Unset: keeps "
            "everything (today's full dataset: ~23:1). Exists so a fold's training-set feature tensors fit in "
            f"memory -- see DEFAULT_NEGATIVE_TO_POSITIVE_RATIO ({DEFAULT_NEGATIVE_TO_POSITIVE_RATIO}), applied "
            "automatically for non-smoke prediction runs unless overridden here."
        ),
    )
    parser.add_argument(
        "--window-length", type=float, default=None,
        help=(
            "Window length in seconds. Unset: resolved per --label-mode in main() -- "
            "DEFAULT_TRUONG_WINDOW_LENGTH (30.0, the paper's own segment length) for "
            "label_mode=prediction (both pipelines, so dense_edge_gru stays apples-to-apples "
            "comparable to truong_stft_cnn), 4.0 for label_mode=detection (untuned)."
        ),
    )
    parser.add_argument(
        # 2026-08-16: default raised from 4.0 -- halves window count (and
        # thus both disk caches and per-fold training-set size) at the cost
        # of coarser window overlap. Still untuned -- a deliberate size/speed
        # tradeoff, not a validated choice of temporal resolution. (Only
        # applies to label_mode=detection now -- see 2026-08-19 comment in
        # main() for why prediction mode no longer uses this default.)
        "--step-size", type=float, default=None,
        help=(
            "Hop between window starts, in seconds. Unset: resolved per --label-mode in "
            "main() -- DEFAULT_TRUONG_STEP_SIZE (30.0, non-overlapping) for "
            "label_mode=prediction (both pipelines), 8.0 for label_mode=detection (untuned)."
        ),
    )
    parser.add_argument(
        "--k-of-n-k", type=int, default=DEFAULT_TRUONG_K_OF_N_K,
        help=f"--pipeline=truong_stft_cnn only: alarm-smoothing k (default {DEFAULT_TRUONG_K_OF_N_K}, paper value).",
    )
    parser.add_argument(
        "--k-of-n-n", type=int, default=DEFAULT_TRUONG_K_OF_N_N,
        help=f"--pipeline=truong_stft_cnn only: alarm-smoothing n (default {DEFAULT_TRUONG_K_OF_N_N}, paper value).",
    )
    parser.add_argument(
        # 2026-08-16: default lowered from 30 -- the real (non-smoke) fold 1
        # run hit exactly zero training loss by epoch 24 and stayed there
        # through epoch 30 (see the 2026-08-16 session note / that run's own
        # log), so epochs 24-30 did nothing for that fold. 20 leaves margin
        # before that observed flatline rather than cutting at it -- not
        # validated against every fold, and this is a per-fold TRAINING-loss
        # observation, not a held-out-performance one when
        # validation_split=0.0). With validation_split > 0, common.py's
        # val-loss early_stopping_patience (see _SHARED_ARCH_PARAMS /
        # TRUONG_STFT_CNN_PARAMS) cuts the fold once val_loss stops
        # improving and restores the best checkpoint; this epoch cap is
        # then just an upper bound. default=None: resolved per
        # --label-mode in main() (see DEFAULT_DETECTION_EPOCHS /
        # DEFAULT_PREDICTION_EPOCHS) unless explicitly overridden here.
        "--epochs", type=int, default=None,
        help="Override epoch count for any pipeline/mode. Unset: defaults depend on --pipeline/--label-mode "
             f"(detection={DEFAULT_DETECTION_EPOCHS}, prediction={DEFAULT_PREDICTION_EPOCHS}, "
             f"truong_stft_cnn={DEFAULT_TRUONG_EPOCHS}, independently tunable).",
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
    parser.add_argument(
        "--validation-split",
        "--validation_split",
        type=float,
        default=None,
        dest="validation_split",
        help=(
            "Fraction of each fold's training windows held out for val_loss "
            "(0.2 in smoke_test.py PARAMS). Unset: uses the pipeline param "
            "dict (0.0 for dense_edge / dense_edge_gru, 0.2 for truong_stft_cnn). "
            "Required for --early-stopping-patience to fire."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "Consecutive epochs without val_loss improvement before training "
            "stops (common.py _train_loop). Requires validation_split > 0 or "
            "it is a no-op. Unset: uses the pipeline param dict "
            "(early_stopping_patience=5). Pass 0 to stop on the first "
            "non-improving epoch."
        ),
    )
    parser.add_argument(
        "--precompute-chunk-size", type=int, default=None,
        help=(
            "--pipeline=dense_edge / dense_edge_gru only: trials-per-torch-call for the CWT/dense-edge "
            "precompute stage (see SparseEvidenceGNNClassifier's precompute_chunk_size docstring, "
            "cwt_gnn_classifiers.py). Unset: original min(batch_size, 4) cap, tuned for a "
            "~16-17GB-RAM machine. On a high-RAM/many-core machine (e.g. a Runpod pod), raising "
            "this (up to batch_size) lets torch's own CPU intra-op threading actually use the "
            "extra cores -- measured ~1.3/32 cores utilized at the chunk=4 default on a 32-core "
            "pod. Pure throughput/memory tradeoff, does not change model output."
        ),
    )
    parser.add_argument(
        "--compile-dense-edge-helper",
        action="store_true",
        help=(
            "--pipeline=dense_edge / dense_edge_gru only: torch.compile(backend='cudagraphs') the "
            "dense-edge helper's compute_dense_edge_input -- see that constructor param's "
            "2026-08-22 docstring in cwt_gnn_classifiers.py. CUDA only; no-op elsewhere. "
            "Measured 2026-08-22: no net win on a Windows/no-Triton box."
        ),
    )
    parser.add_argument(
        "--channel-subset-k",
        type=int,
        default=None,
        help=(
            "--pipeline=dense_edge / dense_edge_gru only: per-window absolute-cosine top-k "
            "channels form a clique of live edges; only those pairs get "
            "WCT/coherence features, scattered into the full E=C*(C-1)/2 "
            "tensor (inactive slots = 0). The GNN always sees n_channels=C "
            "and full E. None (default) is the current full-mesh behaviour."
        ),
    )
    parser.add_argument(
        "--channel-subset-metric",
        default="abs_cosine",
        choices=["abs_cosine"],
        help="Cheap affinity used to rank channels when --channel-subset-k is set.",
    )
    parser.add_argument(
        "--dense-edge-amp-bf16",
        action="store_true",
        help=(
        "--pipeline=dense_edge / dense_edge_gru only: torch.autocast(dtype=torch.bfloat16) around "
        "the non-trainable dense-edge helper's compute_dense_edge_input call -- see "
        "that constructor param's 2026-08-22 docstring in cwt_gnn_classifiers.py. "
        "CUDA only; no-op elsewhere."
        ),
    )
    parser.add_argument(
        "--train-amp-bf16",
        action="store_true",
        help=(
            "--pipeline=dense_edge / dense_edge_gru only: torch.autocast(dtype=torch.bfloat16) around "
            "the TRAINABLE forward pass (channel_encoder/dense_edge_conv/GRU/classifier), "
            "not the dense-edge precompute stage --dense-edge-amp-bf16 already covers -- "
            "see that constructor param's 2026-08-23 docstring in cwt_gnn_classifiers.py "
            "(TorchEEGClassifier.train_amp_bf16 in common.py). CUDA only; no-op elsewhere. "
            "Independent of --dense-edge-amp-bf16 -- either, both, or neither."
        ),
    )
    parser.add_argument(
        "--verbose", type=int, default=None,
        help=(
            "Override clf_params['verbose'] (baked in at 1). 2 turns on "
            "SparseEvidenceGNNClassifier._precompute_dense_edge_inputs's per-chunk "
            "phase-timing breakdown (threshold/transfer/compute/copy_back, synced "
            "so CUDA/MPS numbers are real -- see that method's 2026-08-22 comment) "
            "-- use this to see where a slow --pipeline=dense_edge / dense_edge_gru run's time "
            "actually goes before changing --precompute-chunk-size/--device blind. "
            "Unset: leaves clf_params's own default untouched."
        ),
    )
    parser.add_argument(
        "--disable-disk-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "--pipeline=dense_edge / dense_edge_gru only: skip the disk-backed "
            "CWT/dense-edge cache. Unset: disabled on CUDA (recompute measured "
            "faster for 30s prediction tensors on this Windows box and the "
            "2026-08-21 pod), enabled on MPS/CPU. Pass --disable-disk-cache or "
            "--no-disable-disk-cache to force either way."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help=(
            "Permutation-test null-hypothesis control: after building the dataset, randomly "
            "shuffle y (class balance preserved, positive/negative assignment becomes pure "
            "noise) before running the exact same leave-one-seizure-out pipeline. Compare this "
            "run's mean roc_auc/average_precision against a real run's -- if the real run isn't "
            "well above this, its apparent signal isn't distinguishable from noise. Writes to a "
            "separate *_shuffled_control/ output directory so it can never be mistaken for a "
            "real result. See the 2026-08-17 session note."
        ),
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=123,
        help="Seed for --shuffle-labels' permutation (default 123, deliberately distinct from --seed).",
    )
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
    pipeline = args.pipeline
    cache_flag_explicit = args.disable_disk_cache
    args.disable_disk_cache = resolve_disable_disk_cache(
        args.device, args.disable_disk_cache,
    )
    if cache_flag_explicit is None and pipeline in ("dense_edge", "dense_edge_gru"):
        print(
            f"  disk cache: {'disabled' if args.disable_disk_cache else 'enabled'} "
            f"(default for device={args.device!r}; "
            f"{'--no-disable-disk-cache' if args.disable_disk_cache else '--disable-disk-cache'} to override)"
        )
    label_mode = args.label_mode
    if pipeline == "truong_stft_cnn" and label_mode != "prediction":
        print(
            f"  note: --pipeline=truong_stft_cnn only supports label_mode=prediction -- "
            f"overriding --label-mode={label_mode}."
        )
        label_mode = "prediction"

    # 2026-08-19: keyed off label_mode (not pipeline) -- dense_edge_gru's
    # prediction-mode runs need to be apples-to-apples comparable against
    # truong_stft_cnn on the SAME task, which means the same window/step
    # (previously dense_edge_gru defaulted to 4.0s/8.0s regardless of
    # label_mode: a non-overlapping-with-gaps window that only classifies
    # ~1/3 of the interictal signal, vs. truong_stft_cnn's contiguous
    # 30.0s/30.0s -- false_alarms_per_hour and event-level recall from the
    # two pipelines weren't measuring the same amount of monitored time).
    # detection mode (dense_edge_gru only -- truong_stft_cnn forces
    # label_mode="prediction" above) keeps its own untouched 4.0s/8.0s.
    window_length = args.window_length
    if window_length is None:
        window_length = DEFAULT_TRUONG_WINDOW_LENGTH if label_mode == "prediction" else 4.0
    step_size = args.step_size
    if step_size is None:
        step_size = DEFAULT_TRUONG_STEP_SIZE if label_mode == "prediction" else 8.0
    subjects = args.subjects
    if args.epochs is not None:
        epochs = args.epochs
    elif pipeline == "truong_stft_cnn":
        epochs = DEFAULT_TRUONG_EPOCHS
    else:
        epochs = DEFAULT_PREDICTION_EPOCHS if label_mode == "prediction" else DEFAULT_DETECTION_EPOCHS
    max_interictal_recordings = args.max_interictal_recordings
    negative_to_positive_ratio = args.negative_to_positive_ratio
    if args.smoke:
        print(f"=== SMOKE MODE ({pipeline}/{label_mode}): verifying wiring only, not model quality ===")
        epochs = 2
        subjects = subjects[:1]
        if pipeline == "truong_stft_cnn":
            # NOT shrinking window_length the way dense_edge_gru's smoke mode
            # does below -- TruongCNNCore's first conv block needs at least a
            # 5x5 STFT spatial input (see that class's docstring), so the
            # paper's 30s window can't be shrunk the way an arbitrary window
            # length can.
            pass
        else:
            window_length, step_size = 4.0, 30.0
        if label_mode == "prediction" and max_interictal_recordings is None:
            max_interictal_recordings = SMOKE_MAX_INTERICTAL_RECORDINGS
        # NOT applying DEFAULT_NEGATIVE_TO_POSITIVE_RATIO/DEFAULT_TRUONG_
        # NEGATIVE_TO_POSITIVE_RATIO here -- smoke's dataset is already small
        # (SMOKE_MAX_INTERICTAL_RECORDINGS caps the file count), so it never
        # hit the OOM this subsampling exists for; leaving it untrimmed keeps
        # smoke's numbers comparable to the smoke results already on record.
    elif label_mode == "prediction" and negative_to_positive_ratio is None:
        negative_to_positive_ratio = (
            DEFAULT_TRUONG_NEGATIVE_TO_POSITIVE_RATIO if pipeline == "truong_stft_cnn" else DEFAULT_NEGATIVE_TO_POSITIVE_RATIO
        )

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
    if args.max_channels is not None and X.shape[1] > args.max_channels:
        print(f"  --max-channels={args.max_channels}: truncating X from {X.shape[1]} "
              f"to {args.max_channels} channels (diagnostic only, see that flag's help)")
        X = X[:, :args.max_channels, :]

    label_word = "preictal" if label_mode == "prediction" else "ictal"
    print(f"  X: {X.shape}  {label_word} windows: {int(y.sum())}/{len(y)} "
          f"({100 * y.mean():.1f}%)")

    if args.shuffle_labels:
        # Permutation-test null control (task: is the real run's apparent
        # signal distinguishable from noise?) -- shuffles WHICH windows are
        # labeled positive while preserving the exact class balance, then
        # runs the identical downstream pipeline (same fold construction,
        # same subsampling, same classifier). metadata['seizure_id'] is
        # deliberately left untouched, so leave_one_seizure_out_prediction's
        # event-level hit/FAR bookkeeping still refers to the TRUE preictal
        # windows -- expected to degrade toward a random classifier's
        # behavior, not a clean signal, and is secondary here; the
        # window-level roc_auc/average_precision means are the real
        # comparison against a non-shuffled run. See the 2026-08-17 session
        # note.
        shuffle_rng = np.random.default_rng(args.shuffle_seed)
        y = shuffle_rng.permutation(y)
        print(
            f"  *** LABEL PERMUTATION CONTROL (seed={args.shuffle_seed}): y randomly shuffled -- "
            f"class balance preserved ({int(y.sum())}/{len(y)} positive), but positive/negative "
            f"assignment is now pure noise. This run tests the null hypothesis; its results are "
            f"NOT a real model evaluation. ***"
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    _write_results_readme(output_dir)

    if pipeline == "truong_stft_cnn":
        clf_params = dict(TRUONG_STFT_CNN_PARAMS)
        clf_params["seed"] = args.seed
        clf_params["device"] = args.device
        if args.verbose is not None:
            clf_params["verbose"] = args.verbose
        if args.validation_split is not None:
            clf_params["validation_split"] = args.validation_split
        if args.early_stopping_patience is not None:
            clf_params["early_stopping_patience"] = args.early_stopping_patience

        print(f"Running leave-one-seizure-out (Truong STFT+CNN, epochs={epochs}, sph={args.sph}s, sop={args.sop}s)...")
        results, per_seizure = leave_one_seizure_out_truong(
            X, y, metadata, clf_params, epochs, window_length,
            negative_to_positive_ratio=negative_to_positive_ratio,
            k_of_n_k=args.k_of_n_k, k_of_n_n=args.k_of_n_n,
            skip_folds=set(args.skip_folds) if args.skip_folds else None,
        )

        # Own subdirectory -- same "never pooled with a different label rule/
        # architecture's numbers" reasoning as prediction_dir below.
        truong_dir = output_dir / ("truong_stft_cnn_shuffled_control" if args.shuffle_labels else "truong_stft_cnn")
        truong_dir.mkdir(parents=True, exist_ok=True)
        results_path = truong_dir / f"truong_stft_cnn_leave_one_seizure_out_{run_id}.csv"
        per_seizure_path = truong_dir / f"truong_stft_cnn_per_seizure_{run_id}.csv"
        results.to_csv(results_path, index=False)
        per_seizure.to_csv(per_seizure_path, index=False)
        print(f"\nWrote per-fold results to {results_path}")
        print(f"Wrote per-seizure log to {per_seizure_path}")

        numeric_cols = [
            "accuracy", "precision", "recall", "f1", "average_precision", "roc_auc",
            "false_alarms_per_hour", "false_alarms_per_hour_smoothed",
        ]
        means = results[numeric_cols].mean(numeric_only=True)
        n_hits = int(results["hit"].sum())
        n_hits_smoothed = int(results["hit_smoothed"].sum())
        n_seizures = len(results)
        control_note = " -- LABEL-SHUFFLED NULL CONTROL, NOT a real result" if args.shuffle_labels else ""
        print(f"\n=== Mean across folds (pipeline=truong_stft_cnn){control_note} ===")
        print(means.to_string())
        print(f"event-level hit rate (raw):    {n_hits}/{n_seizures} ({100 * n_hits / n_seizures:.1f}%)")
        print(f"event-level hit rate (k-of-n): {n_hits_smoothed}/{n_seizures} ({100 * n_hits_smoothed / n_seizures:.1f}%)")
    elif label_mode == "prediction":
        clf_params = dict(_dense_family_params(pipeline, "prediction"))
        clf_params["seed"] = args.seed
        clf_params["device"] = args.device
        clf_params["precompute_chunk_size"] = args.precompute_chunk_size
        clf_params["compile_dense_edge_helper"] = args.compile_dense_edge_helper
        clf_params["dense_edge_amp_bf16"] = args.dense_edge_amp_bf16
        clf_params["train_amp_bf16"] = args.train_amp_bf16
        clf_params["channel_subset_k"] = args.channel_subset_k
        clf_params["channel_subset_metric"] = args.channel_subset_metric
        if args.verbose is not None:
            clf_params["verbose"] = args.verbose
        if args.validation_split is not None:
            clf_params["validation_split"] = args.validation_split
        if args.early_stopping_patience is not None:
            clf_params["early_stopping_patience"] = args.early_stopping_patience

        print(
            f"Running leave-one-seizure-out prediction (pipeline={pipeline}, "
            f"dense_edge_temporal_mode={clf_params['dense_edge_temporal_mode']}, "
            f"epochs={epochs}, sph={args.sph}s, sop={args.sop}s)..."
        )
        results, per_seizure = leave_one_seizure_out_prediction(
            X, y, metadata, clf_params, epochs, window_length,
            negative_to_positive_ratio=negative_to_positive_ratio,
            disable_disk_cache=args.disable_disk_cache,
            k_of_n_k=args.k_of_n_k, k_of_n_n=args.k_of_n_n,
            max_folds=args.max_folds,
            skip_folds=set(args.skip_folds) if args.skip_folds else None,
        )

        # Separate output path (task 6, bullet 1): never pooled with
        # detection's results/leave_one_seizure_out_*.csv by a downstream
        # script or summary table that isn't expecting a different label
        # rule and different achievable ceiling -- see results/README.md.
        # --shuffle-labels gets its OWN separate subdirectory for the same
        # reason -- a null-control run must never be readable as a real one.
        # dense_edge vs dense_edge_gru: conv CSVs never land next to GRU CSVs
        # (results/dense_edge/prediction/ vs results/prediction/).
        prediction_dir = _dense_family_result_dir(
            output_dir, pipeline, "prediction", args.shuffle_labels,
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        results_path = prediction_dir / f"prediction_leave_one_seizure_out_{run_id}.csv"
        per_seizure_path = prediction_dir / f"prediction_per_seizure_{run_id}.csv"
        results.to_csv(results_path, index=False)
        per_seizure.to_csv(per_seizure_path, index=False)
        print(f"\nWrote per-fold results to {results_path}")
        print(f"Wrote per-seizure log to {per_seizure_path}")

        numeric_cols = [
            "accuracy", "precision", "recall", "f1", "average_precision", "roc_auc",
            "false_alarms_per_hour", "false_alarms_per_hour_smoothed",
        ]
        means = results[numeric_cols].mean(numeric_only=True)
        n_hits = int(results["hit"].sum())
        n_hits_smoothed = int(results["hit_smoothed"].sum())
        n_seizures = len(results)
        control_note = " -- LABEL-SHUFFLED NULL CONTROL, NOT a real result" if args.shuffle_labels else ""
        print(
            f"\n=== Mean across folds (pipeline={pipeline}, label_mode=prediction; "
            f"NOT comparable to detection's numbers){control_note} ==="
        )
        print(means.to_string())
        print(f"event-level hit rate (raw):    {n_hits}/{n_seizures} ({100 * n_hits / n_seizures:.1f}%)")
        print(f"event-level hit rate (k-of-n): {n_hits_smoothed}/{n_seizures} ({100 * n_hits_smoothed / n_seizures:.1f}%)")
    else:
        clf_params = dict(_dense_family_params(pipeline, "detection"))
        clf_params["seed"] = args.seed
        clf_params["device"] = args.device
        clf_params["precompute_chunk_size"] = args.precompute_chunk_size
        clf_params["compile_dense_edge_helper"] = args.compile_dense_edge_helper
        clf_params["dense_edge_amp_bf16"] = args.dense_edge_amp_bf16
        clf_params["train_amp_bf16"] = args.train_amp_bf16
        clf_params["channel_subset_k"] = args.channel_subset_k
        clf_params["channel_subset_metric"] = args.channel_subset_metric
        if args.verbose is not None:
            clf_params["verbose"] = args.verbose
        if args.validation_split is not None:
            clf_params["validation_split"] = args.validation_split
        if args.early_stopping_patience is not None:
            clf_params["early_stopping_patience"] = args.early_stopping_patience

        print(
            f"Running leave-one-seizure-out detection (pipeline={pipeline}, "
            f"dense_edge_temporal_mode={clf_params['dense_edge_temporal_mode']}, "
            f"epochs={epochs})..."
        )
        results = leave_one_seizure_out_detection(
            X, y, metadata, clf_params, epochs,
            disable_disk_cache=args.disable_disk_cache,
            max_folds=args.max_folds,
            skip_folds=set(args.skip_folds) if args.skip_folds else None,
        )

        # --shuffle-labels: same separate-subdirectory reasoning as the
        # prediction branch above -- a null-control run must never land in
        # the same place as (or be mistaken for) a real detection result.
        detection_dir = _dense_family_result_dir(
            output_dir, pipeline, "detection", args.shuffle_labels,
        )
        detection_dir.mkdir(parents=True, exist_ok=True)
        results_path = detection_dir / f"leave_one_seizure_out_{run_id}.csv"
        results.to_csv(results_path, index=False)
        print(f"\nWrote per-fold results to {results_path}")

        numeric_cols = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
        means = results[numeric_cols].mean(numeric_only=True)
        control_note = " -- LABEL-SHUFFLED NULL CONTROL, NOT a real result" if args.shuffle_labels else ""
        print(f"\n=== Mean across folds (pipeline={pipeline}){control_note} ===")
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
        "`leave_one_seizure_out_prediction`'s docstring in run_pipelines.py.\n\n"
        "- `truong_stft_cnn/truong_stft_cnn_leave_one_seizure_out_*.csv` + "
        "`truong_stft_cnn/truong_stft_cnn_per_seizure_*.csv`: `--pipeline "
        "truong_stft_cnn` -- a DIFFERENT ARCHITECTURE (STFT+CNN, replicating "
        "Truong et al. 2018) on the same prediction task, not just a "
        "different label rule. Also not comparable to `detection/`'s numbers, "
        "and not directly comparable to `prediction/`'s either (different "
        "model, different window length by default) even though both solve "
        "the same task -- see Epilepsy/pipelines/truong_stft_cnn_classifier.py's "
        "module docstring.\n\n"
        "- `dense_edge/leave_one_seizure_out_*.csv` and "
        "`dense_edge/prediction/`: `--pipeline dense_edge` -- same "
        "SparseEvidenceGNNClassifier and leave-one-seizure-out loops as "
        "`dense_edge_gru`, but `dense_edge_temporal_mode=\"conv\"` (Conv2d "
        "over time) instead of `\"rnn\"` (per-edge GRU). Do not pool these "
        "with `leave_one_seizure_out_*.csv` / `prediction/` -- those are "
        "the GRU pipeline's historical layout.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    argument_parser = _build_argument_parser()
    main(argument_parser.parse_args())



#VQ  CH  EG  WT  PS  FA G
#HK  XR   IL   B

#LJ XC TC RH 
#IE IP HP