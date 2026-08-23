# Session notes — real 6-fold subject-1 run: full results (2026-08-23)

Branch: `main` (`b97f06c` at session start, changes still uncommitted).
Repo: `C:\Users\User\Documents\noshore5\EEG_Benchmarks`. Machine: local
Windows box, RTX 3070 Ti (8GB VRAM). Companion to
[windows_cuda_bf16_speedup_predict_oom_fix_and_kofn.md](windows_cuda_bf16_speedup_predict_oom_fix_and_kofn.md)
(the fixes exercised by this run: `dense_edge_amp_bf16`, `train_amp_bf16`,
the `_predict_logits` OOM fix, and k-of-n smoothing) — this note covers only
the run itself and its numbers.

User's explicit request: "yes, kick off the full 6 fold... also include the
k of n post-process to match truong." First attempt (uncapped, 20
epochs/fold) crashed with a CUDA OOM in `predict_proba` — see the companion
note's Part 5. This note is the successful relaunch, after that fix.

---

## Run configuration

```
python Epilepsy/run_pipelines.py --device cuda --subjects 1 --verbose 2 \
    --dense-edge-amp-bf16 --train-amp-bf16
```

- `--pipeline dense_edge_gru`, `--label-mode prediction` (defaults) —
  `StreamingSparseEvidenceGNNClassifier`, leave-one-seizure-out CV, subject
  chb01, 6 seizures → 6 folds.
- No `--epochs` override → real default of **20 epochs/fold** (120 epochs
  total across the run).
- No `--max-channels`/`--max-interictal-recordings` caps — full 23-channel,
  uncapped-interictal real data, not a diagnostic/smoke configuration.
- `negative_to_positive_ratio=5.0` (prediction-mode default) — subsamples
  only TRAINING negatives; TEST windows are never subsampled, so each
  fold's test set is the full, real recording density (650–750 windows
  here).
- k-of-n alarm smoothing on by default (`DEFAULT_TRUONG_K_OF_N_K=8`,
  `DEFAULT_TRUONG_K_OF_N_N=10`), applied per held-out (subject, run)
  recording in chronological order.

## Outcome

**Completed successfully — no OOM, no errors, exit code 0.** This is the
first time this exact run configuration (uncapped test sets, both bf16
flags) has completed end to end on this machine; the previous attempt
crashed at exactly this point (fold 1's 750-window `predict_proba` call).

**Total wall time: ~34 minutes** (13:26:01 → 14:00:05), matching the
~30–40min estimate given to the user before launch. Per-epoch training
time held steady at ~13.8–15.4s/epoch (both-bf16 rate — higher than the
smoke-scale ~11s measured earlier this session, since real scale has more
trials per batch).

## Per-fold results (all 6 seizures)

| seizure | n_test | preictal | hit (raw→smoothed) | FAR/h (raw→smoothed) | precision | recall | f1 | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| 1_03_0 | 750 | 30 | True → False | 8.333 → 2.167 | 0.038 | 0.067 | 0.049 | 0.108 |
| 1_04_0 | 650 | 30 | True → True | 22.258 → 9.290 | 0.190 | 0.900 | 0.314 | 0.458 |
| 1_15_0 | 718 | 30 | True → True | 6.453 → 1.221 | 0.413 | 0.867 | 0.559 | 0.536 |
| 1_16_0 | 743 | 23 | True → True | 1.667 → 0.000 | 0.667 | 0.870 | 0.755 | 0.803 |
| 1_18_0 | 750 | 30 | True → False | 1.000 → 0.000 | 0.500 | 0.200 | 0.286 | 0.447 |
| 1_26_0 | 630 | 30 | True → False | 3.400 → 0.000 | 0.056 | 0.033 | 0.042 | 0.123 |

## Mean across folds

| metric | value |
|---|---|
| accuracy | 0.921141 |
| precision | 0.310587 |
| recall | 0.489372 |
| f1 | 0.333995 |
| average_precision | 0.412697 |
| roc_auc | 0.916192 |
| false_alarms_per_hour (raw) | 7.185259 |
| false_alarms_per_hour (k-of-n) | 2.112987 |
| **event-level hit rate (raw)** | **6/6 (100.0%)** |
| **event-level hit rate (k-of-n)** | **3/6 (50.0%)** |

Results written to:
`Epilepsy/results/prediction/prediction_leave_one_seizure_out_20260823-132639.csv`,
`Epilepsy/results/prediction/prediction_per_seizure_20260823-132639.csv`.

## The k-of-n tradeoff this run surfaced

k-of-n smoothing erased three real detections (1_03_0, 1_18_0, 1_26_0)
along with the false positives it was meant to suppress — those seizures'
preictal windows were flagged by the raw per-window classifier, but the
positive predictions never sustained k=8-of-n=10 consecutive-window density
required to register as a smoothed alarm. Net effect: FAR/h dropped ~70%
(7.19 → 2.11) but event-level hit rate dropped from 6/6 to 3/6 — a genuine
recall cost on this subject, not a free win.

This did **not** show up in this session's earlier small-scale sanity check
(`verify_predict_fix.log`: 6/6 both raw and smoothed) or in the diagnostic
comparison run of the Truong baseline itself (`truong_baseline.log`: 6/6
both raw and smoothed at k=8/n=10) — both of those were smaller/easier
diagnostic configurations, not the same real, full-density test sets this
run used. **Not yet resolved**: whether k=8/n=10 (inherited as-is from
Truong et al. 2018's own paper defaults) is actually the right smoothing
window for this pipeline's window/step-size configuration, or whether it
needs independent tuning rather than reusing Truong's value unexamined.
Worth a deliberate decision before treating k-of-n as a default-on
improvement for this pipeline.

## Open items

- k-of-n tuning (above) — genuinely open, needs a decision, not just a
  bigger run.
- This is subject 1 (chb01) only — no claim here about how this
  generalizes to other subjects.
- Neither this run's code (the companion note's fixes) nor its results
  are committed yet.
