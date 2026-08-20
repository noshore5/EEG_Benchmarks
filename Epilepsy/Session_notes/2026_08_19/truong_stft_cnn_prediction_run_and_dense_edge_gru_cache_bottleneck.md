# Session notes — truong_stft_cnn real prediction run, dense_edge_gru cache-bottleneck diagnosis (2026-08-19)

Branch: `fix/prediction-validation-split-and-chunk-tuning`. Repo:
`/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`.

Follow-on to
[2026-08-17's note](../2026_08_17/oom_fix_streaming_classifier_and_corrected_prediction_results.md),
which left `dense_edge_gru` prediction mode with a completed, non-confounded
real run. This session made `dense_edge_gru` and `truong_stft_cnn` runs
apples-to-apples (`d53b8a4`), ran `truong_stft_cnn`'s full prediction pipeline
for real, then hit and fixed a genuine performance bug in
`dense_edge_gru`'s GPU path on a Runpod pod.

---

## Part 1 — Apples-to-apples fix (`d53b8a4`)

Two mismatches were making the two pipelines' prediction runs not
comparable:

- `window_length`/`step_size` were keyed off `--pipeline`, so
  `dense_edge_gru` defaulted to its own untuned 4s/8s window instead of
  `truong_stft_cnn`'s contiguous 30s/30s — only ~1/3 of the interictal
  signal was being classified, so `false_alarms_per_hour` and event-level
  recall weren't measuring the same monitored time across pipelines. Now
  keyed off `--label-mode`: any `label_mode=prediction` run uses 30s/30s
  regardless of pipeline. Detection mode untouched.
- `PREDICTION_GRU_PARAMS` was silently inheriting `validation_split=0.0`
  from `_SHARED_ARCH_PARAMS`, while `TRUONG_STFT_CNN_PARAMS` used 0.2 —
  `dense_edge_gru` was training on 100% of each fold's windows,
  `truong_stft_cnn` on 80%. Set explicitly to 0.2 to match. (This later
  turned out to crash `dense_edge_gru`'s prediction path — see Part 3.)

---

## Part 2 — `truong_stft_cnn` real prediction run (CHB-MIT subject 1)

Full (non-smoke) `--label-mode prediction` run, leave-one-seizure-out CV
across the subject's 6 seizures (runs 03, 04, 15, 16, 18, 26). Completed,
exit 0. Results:
[`truong_stft_cnn_leave_one_seizure_out_20260819-103235.csv`](../../results/truong_stft_cnn/truong_stft_cnn_leave_one_seizure_out_20260819-103235.csv) /
[`truong_stft_cnn_per_seizure_20260819-103235.csv`](../../results/truong_stft_cnn/truong_stft_cnn_per_seizure_20260819-103235.csv)
— both currently **untracked** in git.

| seizure | n_test | preictal | hit (smoothed) | precision | recall | f1 | roc_auc | avg_precision | FAR/hr (smoothed) |
|---|---|---|---|---|---|---|---|---|---|
| 03 | 750 | 30 | ✅ | 0.206 | 0.867 | 0.333 | 0.887 | 0.147 | 13.0 |
| 04 | 650 | 30 | ✅ | 0.169 | 0.733 | 0.275 | 0.871 | 0.150 | 14.3 |
| 15 | 718 | 30 | ❌ | 0.000 | 0.000 | 0.000 | 0.967 | 0.369 | 1.2 |
| 16 | 743 | 23 | ✅ | 0.920 | 1.000 | 0.958 | 1.000 | 1.000 | 0.0 |
| 18 | 750 | 30 | ❌ | 0.000 | 0.000 | 0.000 | 0.979 | 0.504 | 0.0 |
| 26 | 630 | 30 | ✅ | 0.628 | 0.900 | 0.740 | 0.990 | 0.833 | 1.8 |

**Event-level hit rate: 4/6 (66.7%)**, both raw and smoothed thresholds.
Seizure 16 is the standout — precision 0.92, recall 1.0, F1 0.958,
roc_auc/avg_precision both 1.0, zero false alarms. Seizures 15 and 18
missed at the operating threshold (precision/recall/F1 all 0) but still
rank well (roc_auc 0.967/0.979, avg_precision 0.369/0.504) — same pattern
as 2026-08-17's `dense_edge_gru` note: threshold miscalibration on those
folds, not an absence of signal. Mean smoothed FAR/hr across all six
≈5.1/hour.

Not yet done for this run: the label-permutation null control that
2026-08-17's note ran for `dense_edge_gru` (Part 6 there). Numbers above
haven't been checked against a chance baseline.

---

## Part 3 — `dense_edge_gru` prediction crash, then cache-bottleneck diagnosis (Runpod pod)

Part 1's `validation_split=0.2` change crashed `dense_edge_gru` prediction
(`2cd2712` reverted it back to `0.0` for that pipeline specifically,
keeping `truong_stft_cnn` at 0.2 — not fully apples-to-apples on this one
axis, but avoids the crash). `0e8054f` added `precompute_chunk_size` for
tuning CWT/dense-edge precompute batching.

Deployed to a Runpod GPU pod to run `dense_edge_gru` prediction for real.
It was far slower than expected. Diagnosed and fixed two genuine per-batch
overheads in `StreamingSparseEvidenceGNNClassifier`
([cwt_gnn_classifiers.py](../../pipelines/cwt_gnn_classifiers.py)),
committed together as `b635a89`:

1. **Dense-edge helper model rebuilt every call.** Fixed-mode
   `_precompute_dense_edge_inputs` was constructing and transferring a
   throwaway helper model to GPU on *every* call instead of once. Now
   cached on `self._dense_edge_helper_cache`, keyed by
   `(n_channels, device)`. Measured: cache-cold 32-trial dense-edge GPU
   compute ≈2.3s (vs. an estimated ~77s on CPU at ~2.4s/trial). This
   confirmed the GPU-offload path genuinely works and is fast — it was
   never the bottleneck.
2. **CWT cache keys re-hashed every batch.** `_LazyFeatureBatchDataset`
   re-computed SHA256 cache keys for every raw (sample, channel) pair on
   every training batch, not once per `fit()` call — ≈380-400 hashes/s,
   736 pairs/batch → ≈1.9s/batch of pure hashing overhead ×20-24
   batches/epoch. Added `precompute_window_cache_keys()` to
   [cwt_window_cache.py](../../pipelines/cwt_window_cache.py), called once
   in `_LazyFeatureBatchDataset.__init__`, sliced per-batch instead of
   recomputed. Measured: epoch_time on a comparable 32-trial/24-step batch
   dropped from 22.56-22.80s → 13.06-13.11s (~40%), validated locally with
   a clean smoke-test run (7/7 event-level hit rate maintained, no crash).

**Remaining bottleneck, not fixed — by design.** Even after both fixes,
per-step profiling (temporary `verbose=3` on `PREDICTION_GRU_PARAMS`,
reverted after capture) showed ~70% of epoch time still goes to
`_prepare_features`'s disk cache lookups. Both `DiskCWTCache`
(cwt_window_cache.py) and `dense_edge_cache.py` are **deliberately
disk-only**, with no in-memory front-cache — per a documented prior
OOM/swap incident (an earlier in-memory dict cache grew unbounded across
CV folds to ~4.4GB and pushed a 16GB machine into swap, epoch time crept
6s→20s+; see `DiskCWTCache`'s own docstring). Real `np.load()` +
DEFLATE decompression on every batch/trial is the dominant remaining cost.
Presented this memory-vs-speed tradeoff directly; decision was **accept
current speed, don't build an in-memory LRU cache to chase it further**.

Idle Runpod pod was terminated (`runpodctl remove pod`) once no longer
actively in use.

---

## Current state

- `truong_stft_cnn` prediction: completed real run, 4/6 event hit rate,
  results untracked (see table above and file links).
- `dense_edge_gru` prediction: `validation_split=0.0` (not apples-to-apples
  with `truong_stft_cnn`'s 0.2 — crashes otherwise), two real per-batch
  performance fixes committed (`b635a89`), ~40% epoch-time improvement
  validated locally. Not yet re-run for real, end-to-end, on the pod with
  both fixes in place.
- `Epilepsy/results/prediction/prediction_leave_one_seizure_out_20260819-163844.csv`
  / `..._per_seizure_20260819-163844.csv` exist locally from the post-fix
  smoke-test validation (7/7 hit rate) — smoke-scale, not a real run,
  untracked.

## Open items

- `dense_edge_gru` hasn't had a real (non-smoke), full prediction run
  completed with both `b635a89` fixes in place — the pod was terminated
  before that was done.
- No label-permutation null control run yet for this session's
  `truong_stft_cnn` result (2026-08-17's note has one for `dense_edge_gru`
  only).
- Untracked result CSVs (`truong_stft_cnn` real run, `dense_edge_gru`
  smoke validation) not yet committed or otherwise preserved — will be
  lost on cleanup if not addressed.
- In-memory LRU front-cache for `DiskCWTCache`/`dense_edge_cache.py`
  deliberately not built — explicit user decision to accept current
  disk-bound speed. Don't re-attempt without the user reopening this.
