# Session notes — dense_edge (conv) vs dense_edge_gru smoke 6-fold (2026-08-24)

Branch: `dynmaic_subset` (`b51c504`; this session's changes uncommitted).
Repo: `/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`. Machine: local
macOS, Apple Silicon, MPS, `torch 2.8.0`. Subject chb01.

Follow-on to
[yesterday's fixed-graph k=4 smoke runs](../2026_08_23/fixed_graph_zero_masked_subset_and_smoke_runs.md)
(those were all `--pipeline dense_edge_gru`). This session wired
`--pipeline dense_edge` (`dense_edge_temporal_mode="conv"`) into
`Epilepsy/run_pipelines.py`, turned val-loss early stopping on
(`early_stopping_patience=5`, needs `validation_split>0`), then ran the
same smoke-scale 6-fold with the conv temporal block.

**Not comparable** to the uncapped 6-fold on `main`
([full_6fold_subject1_run_results.md](../2026_08_23/full_6fold_subject1_run_results.md)):
that used `max_interictal_recordings=None` (~650–750 test windows/fold).
Both numbers here use `smoke_test.py`'s `max_interictal_recordings=5`, so
FAR/h is against ~1 hour of interictal per fold.

---

## What changed this session (wiring, not the 6-fold itself)

`dense_edge` vs `dense_edge_gru` is one switch on the same classifier:
`dense_edge_temporal_mode="conv"` (Conv2d + pool over time) vs `"rnn"`
(per-edge GRU). Same `event_mode="dense"` graph, same leave-one-seizure-out
loops, same k=4 scatter into full E=253 zeros.

- `_SHARED_ARCH_PARAMS` no longer hardcodes `"rnn"`. Four independent
  param dicts (pipeline × label-mode). GRU dicts set `"rnn"`; conv dicts
  set `"conv"`.
- `--pipeline {dense_edge_gru, dense_edge, truong_stft_cnn}`. Default
  remains `dense_edge_gru`.
- Conv CSVs would go under `results/dense_edge/` so they cannot be globbed
  with the historical GRU `results/prediction/` files. This 6-fold was
  launched via `smoke_test.py`, which prints the table and does not write
  those CSVs.
- `smoke_test.py` / `pipeline_debug.py` pick params through
  `_dense_family_params`. Disk cache key does **not** include
  `dense_edge_temporal_mode` (features are the `[4, E, T, F]` stack, not
  the trainable temporal block) — conv reused the GRU k=4 cache at 100%.

Early stopping: `common.py`'s `_train_loop` already restored the best
val checkpoint; the break was gated on `early_stopping_patience is not
None`. Set to 5 in `_SHARED_ARCH_PARAMS`, `TRUONG_STFT_CNN_PARAMS`, and
`smoke_test.py` PARAMS. Only fires when `validation_split > 0`.
`run_pipelines.py` GRU/conv prediction still has `validation_split=0.0`,
so that entry point still runs every epoch unless overridden.

---

## Shared smoke config

```
PYTHONUNBUFFERED=1 /usr/bin/python3 Epilepsy/smoke_test.py \
    --pipeline dense_edge --max-folds 6 --epochs 20
```

(`validation_split=0.2` and `early_stopping_patience=5` from PARAMS.)

| | GRU run 3 (yesterday) | conv 6-fold (this note) |
|---|---|---|
| pipeline | `dense_edge_gru` (`temporal_mode=rnn`) | `dense_edge` (`temporal_mode=conv`) |
| label_mode | prediction | same |
| subject / windows | chb01, 30s / 30s | same |
| dataset | `X=(773, 23, 7680)`, 173/773 preictal, `max_interictal=5` | same |
| k | 4 → 6 live edges into E=253 | same; printed `n_channels=23 edges=253` |
| seed | 42 | same |
| device | MPS | same |
| epochs cap | 10, no val, no early stop | 20, `validation_split=0.2`, patience=5 |
| train/val | 100% of fold train | 498/125 of 623 (fold 1; later folds similar) |
| k-of-n | 8/10 | same |
| cache | disk CWT + dense-edge | 100% dense-edge hits (shared with GRU) |

The training protocol is **not** matched. GRU run 3 is 10 epochs on 100%
of each fold's training windows. Conv is a 20-epoch cap with a 0.2 val
split and early stopping, so most folds restored an earlier checkpoint
and trained on 80% of the fold. Read the per-fold table as conv-vs-GRU
under those two protocols, not as an isolated temporal-block ablation.

GRU 1-fold run 4 (same val split, 20 epochs, **no** early stop, restored
epoch 12) is the closer protocol match for seizure `1_02_0` only.

---

## Conv 6-fold outcome

**Completed, exit 0. Wall 1509.46s (~25.2 min).** Mean epoch_time 17.30s
(min 11.80, max 45.76, n=84). Printed config:
`dense_edge_temporal_mode=conv`, `n_channels=23 edges=253`,
`total_params=16763`. First fold (`1_02_0`) bit-matched the earlier
1-fold conv smoke (same seed; `max_folds` is a prefix, so fold 1 reran).

### Early stopping per fold

| fold | seizure | stopped at | best epoch | best val_loss |
|---|---|---|---|---|
| 1 | `1_02_0` | 14 / 20 | 9 | 0.365 |
| 2 | `1_03_0` | 9 / 20 | 4 | 0.447 |
| 3 | `1_05_0` | no (ran 20) | 20 | 0.161 |
| 4 | `1_06_0` | 12 / 20 | 7 | 0.318 |
| 5 | `1_07_0` | 9 / 20 | 4 | 0.507 |
| 6 | `1_10_0` | no (ran 20) | 19 | 0.050 |

`1_05_0` was still improving val_loss at epoch 20. `1_10_0` val_loss hit
0.050 at epoch 19 then jumped to 0.180 at 20; restore used 19. That val
set is a split of the **training** pool, not the held-out 30/30-preictal
test set.

Epoch time: folds 1–3 settle ~12s (16 optimizer steps). Fold 6 is 19
steps/epoch and noisier (16.9–45.8s). First epoch of each fold is the
slow one.

---

## Per-fold: conv (this run) vs GRU run 3

GRU numbers from yesterday's run 3 (10 epochs, no val). Same six
seizures, same k=4 / 30s / 5-recording interictal cap / seed 42.

| seizure | n_test (pre) | GRU F1 / AP / AUC | conv F1 / AP / AUC | GRU hit raw→k-of-n | conv hit raw→k-of-n | GRU FAR raw→sm | conv FAR raw→sm |
|---|---|---|---|---|---|---|---|
| `1_02_0` | 150 (30) | 0.732 / 0.672 / 0.936 | **0.754 / 0.687 / 0.941** | T → T | T → T | 15 → 0 | 13 → 0 |
| `1_03_0` | 150 (30) | 0.656 / 0.606 / 0.929 | **0.733 / 0.651 / 0.939** | T → T | T → T | 11 → 0 | 8 → 0 |
| `1_05_0` | 150 (30) | 0.761 / 0.694 / 0.942 | **0.779 / 0.948 / 0.985** | T → T | T → T | 14 → 0 | 17 → 0 |
| `1_06_0` | 143 (23) | 0.282 / 0.150 / 0.470 | 0.292 / 0.205 / **0.637** | T → T | T → T | 105 → 101 | 100 → 94 |
| `1_07_0` | 150 (30) | 0.281 / 0.370 / 0.799 | **0.522 / 0.409 / 0.813** | T → **F** | T → **T** | 25 → 0 | 21 → 0 |
| `1_10_0` | 30 (30) | **0.776 / 1.000 / —** | 0.378 / 1.000 / — | T → **T** | T → **F** | — | — |

`1_02_0` under the closer protocol (GRU run 4: val 0.2, 20 epochs, restore
epoch 12, no early stop): F1 0.697 / AP 0.647 / AUC 0.930 / FAR 13 → 0.
Conv's early-stopped epoch-9 checkpoint on the same fold: F1 0.754 / AP
0.687 / AUC 0.941 / FAR 13 → 0.

---

## Means

All 6 folds (AUC over the 5 that have interictal test windows):

| metric | GRU run 3 | conv 6-fold |
|---|---|---|
| accuracy | **0.700** | 0.661 |
| precision | 0.562 | **0.612** |
| recall | 0.713 | **0.724** |
| F1 | **0.581** | 0.576 |
| AP | 0.582 | **0.650** |
| ROC-AUC (5 folds) | 0.815 | **0.863** |
| hit raw | 6/6 | 6/6 |
| hit k-of-n | 5/6 (`1_07_0` miss) | 5/6 (`1_10_0` miss) |

GRU's 6-fold accuracy/F1 are pulled up by `1_10_0` (recall 0.633 vs conv
0.233 on a 30/30-preictal test set). The four folds that are not the
known smoke-scale artifacts (`1_02_0`, `1_03_0`, `1_05_0`, `1_07_0`):

| metric | GRU run 3 | conv 6-fold |
|---|---|---|
| accuracy | 0.828 | **0.862** |
| F1 | 0.608 | **0.697** |
| AP | 0.586 | **0.674** |
| ROC-AUC | 0.902 | **0.919** |
| k-of-n hits | 3/4 | **4/4** |

---

## What this does and doesn't show

Do:

- `--pipeline dense_edge` trains. Conv2d temporal block, full C/E, k=4
  scatter, same cache as GRU.
- Early stopping fired on 4/6 folds (patience 5). `1_05_0` and `1_10_0`
  used the epoch cap.
- On the four non-degenerate folds, conv is ahead of yesterday's GRU 10-
  epoch/no-val run on F1, AP, AUC, and k-of-n (picks up `1_07_0`).
- `1_06_0` is still the FAR disaster under both temporal blocks (raw
  ~100/h, k-of-n does not save it). Conv's test AUC there is better
  (0.637 vs 0.470) but not usable as a predictor.

Don't:

- Not an isolated conv-vs-GRU ablation. Epoch cap, val split, and early
  stopping all moved at the same time as `temporal_mode`.
- Not comparable to the uncapped `main` 6-fold (different interictal
  pool, different test density, full mesh vs k=4).
- `1_10_0` remains a degenerate test set from the 5-recording cap +
  round-robin assignment. Conv's low recall there is a 7/30 hit on an
  all-positive 30-window test, not evidence that conv "can't detect
  preictal."
- MPS, so neither bf16 flag ran.

To make the temporal-block comparison clean: same 6-fold, same k=4, same
`validation_split=0.2` + patience=5, only `--pipeline dense_edge` vs
`dense_edge_gru`.
