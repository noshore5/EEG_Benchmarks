# Session notes — Mamba temporal backend, k=23 (full mesh), full-length run (2026-08-24)

Branch: `mamba-temporal-edge-model`. Repo: `C:\Users\User\Documents\noshore5\EEG_Benchmarks`.
Machine: local Windows 11, NVIDIA GeForce RTX 3070 Ti (8GB, driver 591.86,
CUDA 13.1), `torch 2.8.0+cu128`. Subject chb01, seizure `1_02_0` held out
(single leave-one-seizure-out fold, `--max-folds 1`).

Follow-up to [`dense_edge_mamba_temporal_backend.md`](dense_edge_mamba_temporal_backend.md)
(same session, same day) — that note covers the Mamba backend's
implementation, the k=4 smoke-scale GRU-vs-Mamba comparison, and the
`mamba_chunk_size` OOM fix. This note is purely an additional timing/
convergence data point at **`channel_subset_k=23`** (i.e. `None`-equivalent
— all 23 chb01 channels live, full 253-edge mesh active, no synthetic
edge-sparsification), run to actual completion (real epoch budget +
early stopping) rather than the 2-epoch smoke-test default. No code
changes were made for this note — same commit as the mamba backend
(`8e0a6bd`).

## Command

```
python Epilepsy\smoke_test.py --pipeline dense_edge_mamba --channel-subset-k 23 --device cuda --epochs 20
```

`smoke_test.py`'s other `PARAMS` left at their defaults: `label_mode=
"prediction"`, `subjects=[1]`, `window_length=30.0`, `step_size=30.0`,
`max_folds=1`, `max_interictal_recordings=5`, `channel_subset_metric=
"abs_cosine"`, `dense_edge_amp_bf16=True`, `train_amp_bf16=True`,
`disable_disk_cache=True` (forced on `cuda`), `validation_split=0.2`,
`early_stopping_patience=5`, `seed=42`. `--epochs 20` overrides the
smoke-test's own 2-epoch default with `run_pipelines.DEFAULT_PREDICTION_EPOCHS`
(the real pipeline's default), so this is what an actual `run_pipelines.py
--pipeline dense_edge_mamba` fold would do, not an artificially truncated
smoke check.

Dataset: `X: (773, 23, 7680)`, 173/773 (22.4%) preictal windows. 623
train windows (125 held out for validation), 150 test windows (30
preictal) on the held-out seizure.

## Epoch-by-epoch

| epoch | loss | val_loss | acc | val_acc | roc_auc | val_roc_auc | epoch_time |
|---|---|---|---|---|---|---|---|
| 1/20 | 0.868968 | 0.694965 | 0.490 | 0.232 | 0.4756 | 0.7751 | 50.54s |
| 2/20 | 0.718727 | 0.674759 | 0.506 | 0.768 | 0.5186 | 0.9646 | 50.91s |
| 3/20 | 0.615772 | 0.456209 | 0.590 | 0.696 | 0.7627 | 0.9731 | 51.93s |
| 4/20 | 0.252272 | 0.534150 | 0.902 | 0.864 | 0.9645 | 0.9968 | 50.32s |
| 5/20 | 0.262160 | 0.017397 | 0.936 | 1.000 | 0.9803 | 1.0000 | 50.78s |
| 6/20 | 0.025012 | 0.003343 | 0.988 | 1.000 | 0.9997 | 1.0000 | 50.91s |
| 7/20 | 0.006888 | 0.000318 | 0.998 | 1.000 | 0.9999 | 1.0000 | 50.59s |
| 8/20 | 0.000000 | 0.000000 | 1.000 | 1.000 | 1.0000 | 1.0000 | 50.78s |
| 9/20 | 0.000000 | 0.000000 | 1.000 | 1.000 | 1.0000 | 1.0000 | 50.35s |
| 10/20 | 0.000000 | 0.000000 | 1.000 | 1.000 | 1.0000 | 1.0000 | 50.25s ← best (restored) |
| 11/20 | 0.000000 | 0.000000 | 1.000 | 1.000 | 1.0000 | 1.0000 | 50.55s |
| 12/20 | 0.000000 | 0.000000 | 1.000 | 1.000 | 1.0000 | 1.0000 | 51.08s |
| 13/20 | 0.000000 | 0.000000 | 1.000 | 1.000 | 1.0000 | 1.0000 | 50.75s |
| 14/20 | 0.000000 | 0.000000 | 1.000 | 1.000 | 1.0000 | 1.0000 | 50.74s |
| 15/20 | 0.000000 | 0.000000 | 1.000 | 1.000 | 1.0000 | 1.0000 | 50.69s |

`mean epoch_time=50.74s  min=50.25s  max=51.93s  n=15`

Training stopped at epoch 15/20 — `early_stopping_patience=5` counting
from the best epoch (10, `val_loss=0.000000`), matching `5+5=15` less one
off-by-fencepost (patience counts epochs *without* improvement after the
best one; epochs 11–15 are the 5 non-improving epochs). Best model
weights (epoch 10) were restored before evaluation:
`[Train] restored best model from epoch 10 (val_loss=0.000000)`.

Total wall time for the fold (dataset build + all 15 epochs + eval,
`disable_disk_cache=True` so every dense-edge tensor was recomputed from
scratch every epoch, no caching benefit): **769.34s (~12.8 min)**.

## Held-out seizure (`1_02_0`) eval, at the restored best-epoch weights

```
n_test=150  preictal=30  hit=True (smoothed=True)
FAR/h=27.000 (smoothed=0.000)
precision=0.526  recall=1.000  f1=0.690  auc_pr=0.682  roc_auc=0.9417
```

Read with the same caution as any single-fold, single-subject result:
recall=1.0 / roc_auc=0.94 look strong, but raw FAR/h=27 (i.e. the
per-window classifier alarms on far more than just the true preictal
windows — smoothing brings it down to 0.0 false alarms/hour on this
fold, which is the metric that matters for a usable alarm, but that's a
property of the post-hoc smoothing step, not of the raw classifier). This
is one seizure from one subject — no claim of generalization is being
made here; it's a convergence/timing data point for the Mamba backend at
full channel count, not a benchmark result.

## Timing vs. the k=4 smoke run

| run | channel_subset_k | epochs run | mean epoch_time |
|---|---|---|---|
| smoke (prior note) | 4 | 2 | ~42.78s |
| this run | 23 (full mesh) | 15 | 50.74s |

+18.6% mean epoch_time going from k=4 to k=23 (full mesh), for a fold
that also happened to run 15 real training epochs rather than 2. As
documented in the prior note, `channel_subset_k` only zeros out
*non-live* edges in the dense-edge tensor — it does not shrink `E=253`,
the fixed edge-tensor shape `dense_edge_conv` (GRU, Conv, or Mamba)
actually processes. So the modest increase here (not a multiplicative
blowup) is the expected result: `_DenseEdgeMambaTemporal`'s chunked scan
pays a cost set mainly by the fixed `B*E` row count, not by how many of
those edges are "live" under the current `channel_subset_k`. The k=4 vs
k=23 gap that does exist is presumably downstream compute (dense-edge
coherence/phase/significance precompute over more live channel pairs)
rather than the Mamba scan itself.

## Reproduce

```powershell
python Epilepsy\smoke_test.py --pipeline dense_edge_mamba --channel-subset-k 23 --device cuda --epochs 20
```

No code changes accompany this note — timing/convergence data only, on
the `mamba-temporal-edge-model` branch at commit `8e0a6bd`.
