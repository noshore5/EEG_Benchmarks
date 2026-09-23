"""First real benchmark run on the Kuhlmann NeuroVista (NV) contest data,
Pat1 only, per explicit scope from the user 2026-09-22.

NOT leave-one-seizure-out, despite that being this repo's usual protocol
for CHB-MIT (leave_one_seizure_out_raw_classifier in Epilepsy/
run_pipelines.py) -- and that's a deliberate, forced substitution, not an
oversight. See datasets/epilepsy/kuhlmann_nv.py's module docstring: the
released NV files carry no seizure/event id, hour-block, or timing
metadata at all (only `data`, checked empirically) -- the six preictal
segments the contest generator cut from one seizure's lead-up are not
identifiable as a group from anything shipped. Grouping by "run" (like
CHB-MIT's one-recording-per-file structure) doesn't apply either -- every
NV file already IS one independent 10-minute segment, not a chunk of a
longer continuous recording.

What's used instead: StratifiedGroupKFold over sub-windows, grouped by
individual SEGMENT (default 6 folds, chosen to land in the same ballpark
as the ~6-8 fold counts CHB-MIT LOSO runs produce, purely for rough
comparability of fold count -- NOT because it recovers any real seizure
grouping). Grouping by segment keeps every sub-window of one 10-minute
segment on the same side of the split (no within-segment leakage), but a
fold can still contain segments from the same underlying seizure's
lead-up as the training set (the contest cuts ~6 consecutive 10-minute
segments per one-hour pre-seizure block, and that grouping isn't
recoverable from anything released -- see kuhlmann_nv.py). So this is the
honest substitute, not a renamed LOSO: these numbers are more optimistic
than true LOSO and should be reported as such, never pooled into the
same table as CHB-MIT's leave-one-seizure-out results without that
caveat attached.

2026-09-22: tried reconstructing the true block grouping from boundary
signal continuity. First attempt (raw amplitude L2 distance) FAILED its
own validation check on Pat1 (best-match/median-distance ratio: smooth
0.41-1.0 continuum, no separation between genuine and coincidental
matches) and was abandoned. A second attempt -- per-channel z-scoring
each segment before comparing boundaries (undoes whatever per-segment
amplitude normalization was masking real continuity under attempt 1),
plus cosine similarity instead of raw distance -- DID pass validation:
a clear cluster near 0.95-0.97 similarity-gap distinct from a ~0.4 bulk,
recovering ~12 plausible chains (sizes 2-5) covering 34/253 preictal
segments. `--grouping block` uses this (kuhlmann_nv.reconstruct_
preictal_blocks); `--grouping segment` (default) is the plain per-segment
fallback. Since only ~14% of preictal segments get grouped, `block`
strictly weakly-dominates `segment` for fold-construction purposes (it
can only correct segment-level's blind spot, never introduce a new one:
every ungrouped segment behaves exactly as it would under `segment`).

Reuses GodoyTMCClassifier and its GODOY_TMC_PARAMS default hyperparameters
verbatim from Epilepsy/run_pipelines.py (same architecture/params as the
CHB-MIT godoy_tmc pipeline) so any future comparison is apples-to-apples
on the model side -- only the fold construction differs, and only because
it has to.

NV segments are 240,000 samples (400Hz x 600s) vs. CHB-MIT's 7,680
(256Hz x 30s) windows GodoyTMC is tuned around. Fed whole, GodoyTMC's
channel-major conv tokenizer blows the Transformer's token count up to
n_channels * (T/360) -- confirmed empirically 2026-09-22: OOM'd a single
13.5GB attention buffer on a 4-sample batch. So each 10-minute segment is
sliced into non-overlapping --window-length (default 30s, matching this
repo's own CHB-MIT prediction-mode window size) sub-windows inheriting
the segment's label, via datasets/epilepsy/kuhlmann_nv.window_segments --
see that function's docstring. Folds are then split with
StratifiedGroupKFold, grouped by ORIGINATING SEGMENT (or reconstructed
block, see --grouping above), so no segment's sub-windows ever land on
both sides of a fold -- the "well-documented leakage risk"
godoy_tmc_classifier.py's own docstring calls out for adjacent-window
random splits.

METRICS ARE COMPUTED PER SEGMENT, NOT PER SUB-WINDOW (fixed 2026-09-22,
per explicit user pushback on an earlier version of this script that
scored sub-windows directly): windowing to 30s is purely an architecture
accommodation for GodoyTMC's token budget -- the actual task, and the
only thing Test's structure (or any real deployment) would ever ask
about, is one label per whole 10-minute segment. A model could look good
scored per-sub-window while telling you nothing about per-segment
performance, so each fold's sub-window probabilities are mean-pooled back
to one score per originating segment_id before precision/recall/F1/
ROC-AUC are computed. No invented finer-grained ground truth is used
anywhere in this file -- a segment's label is still the one real label
the contest shipped for it; only the SCORING unit changed, not what's
being predicted.

Usage:
    python scripts/run_nv_pat1.py
    python scripts/run_nv_pat1.py --limit 40 --folds 3   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
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
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.epilepsy.kuhlmann_nv import (  # noqa: E402
    load_patient_train,
    reconstruct_preictal_blocks,
    window_segments,
)
from Epilepsy.pipelines.godoy_tmc_classifier import GodoyTMCClassifier  # noqa: E402
from Epilepsy.run_pipelines import GODOY_TMC_PARAMS  # noqa: E402


def run(
    patient: int,
    n_folds: int,
    epochs: int,
    batch_size: int,
    device: str,
    limit: int | None,
    seed: int,
    window_length: float,
    decimate: int,
    grouping: str,
    output_dir: Path,
) -> pd.DataFrame:
    native_fs = 400.0
    fs = native_fs / decimate
    print(
        f"[nv_pat{patient}] loading Train segments (decimate={decimate} -> {fs:.0f}Hz, "
        "see kuhlmann_nv.load_patient_train's docstring re: laptop memory)...",
        flush=True,
    )
    t0 = time.time()
    X_seg, y_seg, segment_ids = load_patient_train(patient, limit=limit, decimate=decimate)
    print(
        f"[nv_pat{patient}] loaded {len(y_seg)} segments "
        f"({int(y_seg.sum())} preictal / {int((y_seg == 0).sum())} interictal) "
        f"in {time.time() - t0:.0f}s, X.shape={X_seg.shape} "
        f"({X_seg.nbytes / 1e9:.1f}GB resident)",
        flush=True,
    )
    if grouping == "block":
        group_key = reconstruct_preictal_blocks(X_seg, y_seg, segment_ids)
    elif grouping == "segment":
        group_key = segment_ids
    else:
        raise ValueError(f"grouping must be 'segment' or 'block', got {grouping!r}")

    # Window on the TRUE segment_ids (not group_key) -- window_win_seg_ids
    # is what lets us aggregate a fold's predictions back to one
    # prediction per real 10-minute segment before scoring (see this
    # script's module docstring: the actual task, and Test's actual
    # structure, is one label per segment -- scoring at the sub-window
    # level answers a question nothing downstream ever asks). group_key
    # (segment id or reconstructed block id) is used ONLY for the fold
    # split, via a per-segment -> per-window broadcast below, since
    # window_segments already computes a fixed windows-per-segment count.
    X, y, window_seg_ids = window_segments(X_seg, y_seg, segment_ids, window_length=window_length, fs=fs)
    n_windows_per_segment = len(y) // len(segment_ids)
    groups = np.repeat(group_key, n_windows_per_segment)
    print(
        f"[nv_pat{patient}] windowed into {len(y)} {window_length}s sub-windows "
        f"({int(y.sum())} preictal / {int((y == 0).sum())} interictal), X.shape={X.shape}",
        flush=True,
    )

    clf_params = dict(GODOY_TMC_PARAMS)
    clf_params["batch_size"] = batch_size
    clf_params["device"] = device
    clf_params["seed"] = seed

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test_win = X[test_idx], y[test_idx]

        clf = GodoyTMCClassifier(epochs=epochs, **clf_params)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        y_score_win = proba[:, 1]

        # Aggregate sub-window predictions back to ONE score per real
        # 10-minute segment before scoring -- see this script's module
        # docstring: sub-window-level metrics answer a question nothing
        # downstream (Test, or any real deployment) ever asks. Every
        # sub-window of one segment shares that segment's single label by
        # construction (window_segments), so grouping by segment id and
        # taking the label's own value is safe -- .mean() on a
        # column-of-one-repeated-value is just that value.
        test_seg_ids = window_seg_ids[test_idx]
        agg = pd.DataFrame({"segment_id": test_seg_ids, "y": y_test_win, "score": y_score_win})
        agg = agg.groupby("segment_id", sort=False).mean()
        y_test = agg["y"].to_numpy().astype(np.int64)
        y_score = agg["score"].to_numpy()
        y_pred = (y_score >= 0.5).astype(np.int64)

        row = {
            "fold": fold_i,
            "n_train_windows": int(len(train_idx)),
            "n_test_windows": int(len(test_idx)),
            "n_test_segments": int(len(agg)),
            "n_test_segments_preictal": int(y_test.sum()),
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
        rows.append(row)
        print(
            f"[nv_pat{patient}] fold {fold_i}: n_test_segments={row['n_test_segments']} "
            f"(preictal={row['n_test_segments_preictal']})  precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1']:.3f} "
            f"auc_pr={row['average_precision']:.3f} roc_auc={row['roc_auc']:.3f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_path = output_dir / f"nv_pat{patient}_stratified_kfold_{grouping}grouped_{run_id}.csv"
    results.to_csv(results_path, index=False)
    print(f"\nWrote per-fold results to {results_path}")

    numeric_cols = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
    means = results[numeric_cols].mean(numeric_only=True)
    print(
        f"\n=== Mean across folds (pipeline=godoy_tmc, dataset=nv_pat{patient}, grouping={grouping}, "
        f"NOT leave-one-seizure-out -- see this script's module docstring) ==="
    )
    print(means.to_string())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient", type=int, default=1)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None, help="cap segments loaded (smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window-length", type=float, default=30.0,
        help="sub-window length in seconds each 10-min segment is sliced into (see module docstring)",
    )
    parser.add_argument(
        "--decimate", type=int, default=4,
        help="integer downsample factor applied at load (400Hz/decimate) -- default 4 (100Hz) keeps "
        "Pat1Train's resident footprint around ~3GB instead of ~13GB; see kuhlmann_nv.load_patient_train's "
        "docstring. Pass 1 for native 400Hz only if you have the headroom.",
    )
    parser.add_argument(
        "--grouping", choices=["segment", "block"], default="segment",
        help="'segment' (default): plain per-segment StratifiedGroupKFold, the honest baseline. "
        "'block': group by kuhlmann_nv.reconstruct_preictal_blocks's reconstructed preictal blocks "
        "instead -- a genuine (validated 2026-09-22) but partial improvement; see that function's "
        "docstring before trusting it further.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "Epilepsy" / "results" / "godoy_tmc" / "nv"),
    )
    args = parser.parse_args()

    run(
        patient=args.patient,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        seed=args.seed,
        window_length=args.window_length,
        decimate=args.decimate,
        grouping=args.grouping,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
