# Session notes — `truong_stft_cnn` CUDA verbose trace and real 6-fold run (2026-08-22)

Branch: `main`. Repo: `C:\Users\User\Documents\noshore5\EEG_Benchmarks`
(Windows/CUDA machine — RTX 3060, 12GB, WDDM — this session's dev box,
distinct from the Mac/MPS and Runpod/CUDA machines referenced in earlier
notes).

Same-day session also covered an extensive `dense_edge_gru` CUDA
performance investigation (GPU-residency end-to-end, a real chunk-size
ceiling at this GPU's 12GB, a `torch.compile`/cudagraphs trial that came
back a net loss, and a `bf16` autocast attempt that hit a hard
`torch.complex()` dtype crash, now flagged in code comments rather than
fixed) — not written up here; this note covers only the two
`truong_stft_cnn` runs logged today.

Two distinct `truong_stft_cnn` prediction-mode runs happened today, at
different scales and for different purposes. They are **not** the same
run — noted explicitly because both are named "truong" and it's easy to
conflate them:

| | single-fold verbose trace | real 6-fold run |
|---|---|---|
| driver | `Epilepsy/pipeline_debug.py` | `Epilepsy/run_pipelines.py` |
| folds | 1 (fold 0 only, hardcoded debug) | 6 (real leave-one-seizure-out) |
| epochs | 1 (forced, debug-only) | full (`DEFAULT_PREDICTION_EPOCHS=20`) |
| purpose | verbose (`verbose=2`) instrumented trace | real result numbers |
| logged to | `cuda_truong.log` (this session's tmp dir) | result CSVs only, no stdout log captured |
| timestamp | 14:19–14:20 | 11:21 |

---

## Part 1 — Single-fold verbose debug trace (`cuda_truong.log`)

Ran via `pipeline_debug.py --pipeline truong_stft_cnn --label-mode
prediction --device cuda --verbose 2`, subject 1, window/step 30s/30s,
SPH/SOP/postictal-buffer at defaults (300/900/1800s). This harness always
forces exactly one fold (fold 0: held-out seizure `1_02_0`, onset
2996.0s–3036.0s) and exactly one epoch — it exists to get a fast,
fully-instrumented look at one training pass, not a real result.

Environment printed by the harness: torch 2.8.0+cu128, Python 3.11.9,
Windows-10-10.0.26200-SP0, CUDA available, 12 CPU cores, 6 torch
threads/6 interop threads.

Dataset: 773 total windows (600 interictal / 173 preictal), fold-0 split
623 train / 150 test (143/30 preictal). `TruongSTFTCNNClassifier`:
197,060 total params, `batch_size=32`, `validation_split=0.2` (125/623
held out), class weights `[0.458, 1.542]`.

One epoch, 16 optimizer steps: `loss=0.3347 acc=0.8173 roc_auc=0.9300`,
val `val_loss=0.1265 val_acc=0.9600 val_roc_auc=0.9993`, `epoch_time=1.13s`.
Best-model restore triggered (val_loss 0.1265 at epoch 1, the only epoch).

Fold-0 test-set metrics (150 windows, 30 preictal), single epoch only —
**not comparable to the real run's fold-0 numbers in Part 2**, which
trained the full schedule:

| metric | value |
|---|---|
| accuracy | 0.960 |
| precision | 0.833 |
| recall | 1.000 |
| f1 | 0.909 |
| average_precision | 0.996 |
| roc_auc | 0.999 |
| confusion matrix | `[[114, 6], [0, 30]]` |

Timing: dataset construction 3.62s, fit 6.12s (includes one-time model
build + parameter-hash logging), test prediction 0.79s. Training
throughput 101.8 windows/sec, test throughput 188.8 windows/sec. CUDA
memory after fit: 17.0 MB allocated / 214.0 MB reserved — confirms this
pipeline's actual working set is tiny; `truong_stft_cnn`'s speed advantage
over `dense_edge_gru` is architectural (single 2D/3D CNN over a compact
STFT tensor, not a 253-edge dense coherence graph), not a memory-budget
artifact.

---

## Part 2 — Real 6-fold leave-one-seizure-out run (11:21, CSVs only)

Full `run_pipelines.py` run, `truong_stft_cnn`, `label_mode=prediction`,
subject 1, real epoch schedule, leave-one-seizure-out CV across the
subject's 6 seizures (runs 03, 04, 15, 16, 18, 26). Completed successfully.
Results:
[`truong_stft_cnn_leave_one_seizure_out_20260822-112156.csv`](../../results/truong_stft_cnn/truong_stft_cnn_leave_one_seizure_out_20260822-112156.csv) /
[`truong_stft_cnn_per_seizure_20260822-112156.csv`](../../results/truong_stft_cnn/truong_stft_cnn_per_seizure_20260822-112156.csv)
— both currently **untracked** in git. No verbose stdout/log was captured
for this run (only the two result CSVs exist) — a gap if this run needs
to be reproduced or debugged later.

| seizure | n_train | n_test | preictal | hit | hit (smoothed) | precision | recall | f1 | roc_auc | avg_precision | FAR/hr (smoothed) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 03 | 3491 | 750 | 30 | ✅ | ✅ | 0.181 | 0.700 | 0.288 | 0.875 | 0.136 | 12.3 |
| 04 | 3591 | 650 | 30 | ✅ | ✅ | 0.201 | 0.933 | 0.331 | 0.919 | 0.227 | 13.9 |
| 15 | 3523 | 718 | 30 | ❌ | ❌ | 0.000 | 0.000 | 0.000 | 0.971 | 0.400 | 1.6 |
| 16 | 3498 | 743 | 23 | ✅ | ✅ | 0.259 | 0.957 | 0.407 | 0.980 | 0.616 | 3.2 |
| 18 | 3491 | 750 | 30 | ✅ | ❌ | 0.857 | 0.200 | 0.324 | 0.995 | 0.852 | 0.0 |
| 26 | 3611 | 630 | 30 | ✅ | ✅ | 0.558 | 0.967 | 0.707 | 0.983 | 0.619 | 2.8 |

**Event-level hit rate: raw 5/6 (83.3%), smoothed 4/6 (66.7%)** — same
smoothed hit/miss pattern (miss on 15 and 18) as
[2026-08-19's `truong_stft_cnn` run](../2026_08_19/truong_stft_cnn_prediction_run_and_dense_edge_gru_cache_bottleneck.md#part-2--truong_stft_cnn-real-prediction-run-chb-mit-subject-1)
on the same subject/folds. Per-fold precision/recall/f1/roc_auc numbers
are close to but not identical to that run (e.g. seizure 03: precision
0.181 vs. 0.206 then) — expected, not investigated further here, since
the pipeline changed materially between the two runs (torch-native CWT
merged 2026-08-20, CWT/dense-edge disk cache removed 2026-08-21); this
note doesn't attribute the shift to a specific cause. The overall
qualitative pattern (16 and 26 strong, 15 and 18 miss at threshold despite
good ranking, 03/04 borderline) is stable across both runs.

No label-permutation null-control run for either this run or 2026-08-19's
— still an open item from that note, unaddressed here.

---

## Current state

- `truong_stft_cnn` prediction, subject 1: two runs logged today, a
  single-fold/single-epoch verbose debug trace (Part 1, `cuda_truong.log`)
  and a real 6-fold leave-one-seizure-out run (Part 2, CSVs only, no log).
  Real run's event-level hit rate (4/6 smoothed) matches 2026-08-19's
  result on the same folds.
- Result CSVs from Part 2 are untracked in git — will be lost on cleanup
  if not committed or otherwise preserved (same standing issue noted
  2026-08-19).

## Open items

- Part 2's real 6-fold run has no accompanying verbose/stdout log — if it
  needs to be reproduced or debugged, it'll need to be re-run.
- No label-permutation null control yet for any `truong_stft_cnn`
  prediction run (2026-08-19 or this one).
- The precision/recall/f1 drift vs. 2026-08-19's run (same folds, same
  hit pattern, different numbers) hasn't been attributed to a specific
  intervening change — flagged, not investigated.
- Untracked result CSVs not yet committed or otherwise preserved.
