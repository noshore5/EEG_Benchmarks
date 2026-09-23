"""temporal_graph_mamba (dense-edge CWT -> Mamba temporal backend) applied
to the Kuhlmann NeuroVista (NV) Pat1 data.

Same dataset, same CV/scoring caveats as scripts/run_nv_pat1.py -- see
that script's module docstring for the full explanation (StratifiedGroupKFold
over segments/reconstructed blocks, NOT leave-one-seizure-out; segment-level
scoring via mean-pooling sub-window probabilities before metrics). This
script only swaps the CLASSIFIER (StreamingSparseEvidenceGNNClassifier,
event_mode="temporal_graph", temporal_graph_mode="mamba" -- the same class
Epilepsy/run_pipelines.py's --pipeline temporal_graph_mamba uses on CHB-MIT)
in place of GodoyTMCClassifier; the fold-construction/aggregation logic is
identical, deliberately not re-derived.

Key difference from CHB-MIT's usage: PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS's
sampling_rate=256 default is CHB-MIT's native rate (see _SHARED_ARCH_PARAMS
in run_pipelines.py) -- overridden here to match NV's own decimated rate
(400Hz/--decimate, default 100Hz), or the CWT/coherence band (8-40Hz) would
be computed against the wrong sample rate entirely.

Does NOT reuse run_pipelines.py's leave_one_seizure_out_prediction's
seizure_id-based fold *exclusion* logic -- that's built around CHB-MIT's
metadata contract (real seizure_id/subject/run columns) and doesn't map
onto NV's segment/reconstructed-block grouping.

**2026-09-23 correction -- caching regime rewritten twice in one day:**

*Attempt 1* (this script's very first version) passed no cache at all
(cwt_cache/dense_edge_cache_dir/dense_edge_mem_cache all None), on the
stated assumption that NV's "much smaller scale (hundreds, not tens of
thousands, of windows)" made per-fold recompute an acceptable tradeoff.
Wrong: full-scale NV Pat1 windows to ~16,180 sub-windows, not hundreds,
and with no cache every batch of every epoch recomputed its own
dense-edge/CWT tensors from scratch (no reuse even *within* a single
fit() call). `nv-pat1-tgm-block-6fold` ran 30+ minutes still stuck on
fold 0's first epoch before being killed.

*Attempt 2* (this version) does NOT copy CHB-MIT's exact regime either --
NV's working set is too large for that, not too small. leave_one_seizure_
out_prediction relies mostly on a GPU-resident dense_edge_mem_cache
(DenseEdgeMemCache, byte-budgeted LRU, ~15-23GB typical budget). Full-mesh
dense-edge entries run ~15MB/trial (dense_edge_cache.py's own docstring);
at 16,180 unique windows that's ~243GB of unique entries -- 10-15x any
single GPU's VRAM budget, so a GPU-only cache would constantly evict and
effectively still recompute most trials, same failure mode as no cache at
all, just with extra bookkeeping overhead on top. CHB-MIT's LOSO runs
don't hit this because their per-subject/per-seizure working sets are
smaller and mostly fit the VRAM budget already.

The fix here is two-tier, matching what leave_one_seizure_out_prediction
*also* does by default (disk cache always on) but relying on the disk
tier as the real floor instead of treating it as a fallback:
  1. **Disk cache is the floor** (DiskCWTCache for raw CWT tensors,
     dense_edge_cache_dir for dense-edge tensors) -- content-addressed,
     survives across the whole run and across process restarts, and its
     capacity is bounded by EBS/NVMe, not VRAM, so it comfortably holds
     NV's full ~243GB working set. First-touch cost per window is paid
     once; every later hit (same window, any batch/epoch/fold) is a disk
     read instead of a full CWT+coherence recompute. On by default here;
     `--disable-disk-cache` turns it off for a from-scratch A/B timing
     comparison, not for routine use.
  2. **GPU-resident dense_edge_mem_cache is an opt-in accelerator on top**
     (`--dense-edge-gpu-cache`), same auto-sizing as run_pipelines.py's
     flag of the same name -- speeds up whichever subset of windows is
     currently hot (the active fold's training windows) without needing
     to hold the whole dataset resident. Cleared at each fold's train->
     eval boundary (train and test windows are disjoint within a fold,
     so training's cache entries are dead weight by the time eval runs --
     same reasoning run_pipelines.py's 2026-09-19 fix documents), but NOT
     cleared between folds -- NV's folds share most of their windows
     across each other's training sets too, same as CHB-MIT's LOSO folds.

No fold-level checkpoint/resume (leave_one_seizure_out_prediction's
--checkpoint-dir doesn't apply here) -- a spot reclaim mid-run loses all
progress. Acceptable for now given NV Pat1's smaller scale; worth adding
if this becomes a routinely-rerun pipeline.

Per-fold teardown (del + gc.collect() + torch.cuda.empty_cache()) is
included from the start, not added reactively -- see FAILURE_LOG.md #17:
the SAME dataset (NV Pat1, full scale) already OOM'd godoy_tmc's raw
classifier on g5.xlarge's 16GB host RAM once today, and dense-edge/CWT
feature tensors are typically MORE memory-hungry per window than a raw
classifier's. Launch on a bigger-RAM instance type (g5.2xlarge/g6.2xlarge)
until there's real evidence 16GB is enough for this pipeline on this data.
"""

from __future__ import annotations

import argparse
import gc
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
from Epilepsy.pipelines.cwt_gnn_classifiers import (  # noqa: E402
    StreamingSparseEvidenceGNNClassifier,
)
from Epilepsy.pipelines.cwt_window_cache import (  # noqa: E402
    DiskCWTCache,
    default_cwt_cache_root,
)
from Epilepsy.pipelines.dense_edge_cache import (  # noqa: E402
    DenseEdgeMemCache,
    default_dense_edge_cache_root,
)
from Epilepsy.run_pipelines import (  # noqa: E402
    PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS,
    _resolve_dense_edge_cache_gb,
)


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
    nfreqs: int,
    precompute_chunk_size: int | None,
    disable_disk_cache: bool,
    dense_edge_gpu_cache: bool,
    dense_edge_gpu_cache_gb: float | None,
    dense_edge_gpu_cache_headroom_gb: float,
    output_dir: Path,
) -> pd.DataFrame:
    native_fs = 400.0
    fs = native_fs / decimate
    print(
        f"[nv_pat{patient}_tgm] loading Train segments (decimate={decimate} -> {fs:.0f}Hz, "
        "see kuhlmann_nv.load_patient_train's docstring re: laptop memory)...",
        flush=True,
    )
    t0 = time.time()
    X_seg, y_seg, segment_ids = load_patient_train(patient, limit=limit, decimate=decimate)
    print(
        f"[nv_pat{patient}_tgm] loaded {len(y_seg)} segments "
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

    # Window on the TRUE segment_ids (not group_key) -- same reasoning as
    # run_nv_pat1.py: window_seg_ids is what lets fold predictions be
    # aggregated back to one score per real 10-minute segment before
    # scoring; group_key (segment id or reconstructed block id) is used
    # ONLY for the fold split.
    X, y, window_seg_ids = window_segments(X_seg, y_seg, segment_ids, window_length=window_length, fs=fs)
    n_windows_per_segment = len(y) // len(segment_ids)
    groups = np.repeat(group_key, n_windows_per_segment)
    print(
        f"[nv_pat{patient}_tgm] windowed into {len(y)} {window_length}s sub-windows "
        f"({int(y.sum())} preictal / {int((y == 0).sum())} interictal), X.shape={X.shape}",
        flush=True,
    )

    clf_params = dict(PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS)
    clf_params["batch_size"] = batch_size
    clf_params["device"] = device
    clf_params["seed"] = seed
    # Override CHB-MIT's native-rate default (256) -- see module docstring.
    clf_params["sampling_rate"] = int(round(fs))
    clf_params["nfreqs"] = nfreqs
    # Same VRAM-safety flags LAUNCH_CHECKLIST.md's verified-good CHB-MIT
    # command uses (--temporal-graph-edge-complex-native,
    # --precompute-chunk-size 2) -- no reason to assume NV needs less care.
    clf_params["temporal_graph_edge_complex_native"] = True
    clf_params["precompute_chunk_size"] = precompute_chunk_size

    # Two-tier cache shared across the WHOLE run (all folds) -- see module
    # docstring's "caching regime rewritten twice in one day" section for
    # why disk is the floor here and GPU residency is only an accelerator.
    shared_cwt_cache = None if disable_disk_cache else DiskCWTCache(default_cwt_cache_root())
    shared_dense_edge_cache_dir = None if disable_disk_cache else default_dense_edge_cache_root()
    shared_dense_edge_mem_cache = (
        DenseEdgeMemCache(max_bytes=int(
            _resolve_dense_edge_cache_gb(dense_edge_gpu_cache_gb, dense_edge_gpu_cache_headroom_gb)
            * (1024 ** 3)
        ))
        if dense_edge_gpu_cache else None
    )

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test_win = X[test_idx], y[test_idx]

        clf = StreamingSparseEvidenceGNNClassifier(
            epochs=epochs,
            cwt_cache=shared_cwt_cache,
            dense_edge_cache_dir=shared_dense_edge_cache_dir,
            dense_edge_mem_cache=shared_dense_edge_mem_cache,
            **clf_params,
        )
        clf.fit(X_train, y_train)
        # Train/test windows are disjoint within a fold, so training's mem-
        # cache entries are guaranteed dead weight for eval -- clear before
        # predict_proba (same reasoning as run_pipelines.py's 2026-09-19
        # eval-boundary fix). NOT cleared *between* folds -- see docstring.
        if shared_dense_edge_mem_cache is not None:
            shared_dense_edge_mem_cache.clear()
        proba = clf.predict_proba(X_test)
        y_score_win = proba[:, 1]

        # Aggregate sub-window predictions back to ONE score per real
        # 10-minute segment before scoring -- see run_nv_pat1.py's
        # docstring for why sub-window-level metrics answer a question
        # nothing downstream ever asks.
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
            f"[nv_pat{patient}_tgm] fold {fold_i}: n_test_segments={row['n_test_segments']} "
            f"(preictal={row['n_test_segments_preictal']})  precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1']:.3f} "
            f"auc_pr={row['average_precision']:.3f} roc_auc={row['roc_auc']:.3f}",
            flush=True,
        )

        # Per-fold teardown -- see module docstring / FAILURE_LOG.md #17.
        del clf, proba, y_score_win, X_train, y_train, X_test, y_test_win
        gc.collect()
        try:
            import resource

            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"[fold {fold_i} teardown] host RSS (peak so far)={rss_gb:.2f}GB", flush=True)
        except Exception:
            pass
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_path = output_dir / f"nv_pat{patient}_tgm_stratified_kfold_{grouping}grouped_{run_id}.csv"
    results.to_csv(results_path, index=False)
    print(f"\nWrote per-fold results to {results_path}")

    numeric_cols = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
    means = results[numeric_cols].mean(numeric_only=True)
    print(
        f"\n=== Mean across folds (pipeline=temporal_graph_mamba, dataset=nv_pat{patient}, "
        f"grouping={grouping}, NOT leave-one-seizure-out -- see run_nv_pat1.py's docstring) ==="
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
        help="integer downsample factor applied at load (400Hz/decimate) -- default 4 (100Hz), "
        "same as run_nv_pat1.py's default; sampling_rate is overridden to match automatically.",
    )
    parser.add_argument(
        "--grouping", choices=["segment", "block"], default="segment",
        help="same as run_nv_pat1.py's --grouping -- see that script's docstring.",
    )
    parser.add_argument("--nfreqs", type=int, default=8)
    parser.add_argument(
        "--precompute-chunk-size", type=int, default=2,
        help="dense-edge precompute chunk size (VRAM safety valve) -- see LAUNCH_CHECKLIST.md.",
    )
    parser.add_argument(
        "--disable-disk-cache", action="store_true",
        help="skip the on-disk CWT/dense-edge cache (see module docstring's caching-regime "
        "section) -- for a from-scratch timing A/B only, not routine use.",
    )
    parser.add_argument(
        "--dense-edge-gpu-cache", action="store_true",
        help="also keep a byte-budgeted GPU-resident dense-edge cache on top of the disk "
        "cache (see module docstring) -- speeds up the active fold's hot windows.",
    )
    parser.add_argument(
        "--dense-edge-gpu-cache-gb", type=float, default=None,
        help="override the GPU cache's byte budget (GiB); default auto-sizes from the card's "
        "total VRAM minus --dense-edge-gpu-cache-headroom-gb.",
    )
    parser.add_argument(
        "--dense-edge-gpu-cache-headroom-gb", type=float, default=8.0,
        help="VRAM to leave unclaimed by the GPU cache when auto-sizing it (ignored if "
        "--dense-edge-gpu-cache-gb is also passed).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "Epilepsy" / "results" / "temporal_graph_mamba" / "nv"),
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
        nfreqs=args.nfreqs,
        precompute_chunk_size=args.precompute_chunk_size,
        disable_disk_cache=args.disable_disk_cache,
        dense_edge_gpu_cache=args.dense_edge_gpu_cache,
        dense_edge_gpu_cache_gb=args.dense_edge_gpu_cache_gb,
        dense_edge_gpu_cache_headroom_gb=args.dense_edge_gpu_cache_headroom_gb,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
