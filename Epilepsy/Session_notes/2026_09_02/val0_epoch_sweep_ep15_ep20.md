# val=0 sweep -- ep20 full, ep15 seed42, epoch-count is the real knob

2026-09-02 (Mac, offline for most of the run)

## What ran

Continuation of the `--validation-split 0` sweep from 2026-09-01
(`_to_delete/run_pre_val0.py <seed> <epochs>`, `temporal_graph_mamba`
"pre" prediction, chb01 LOSO 6-fold, MPS). Per-fold leak fix held the
whole time -- flat ~50-60 s/epoch except while a concurrent shell was
thrashing RAM (epochs ballooned to ~130 s, recovered when the user paused
that shell). Dense-edge cache 100% disk hits throughout; fully offline,
commits queued locally and pushed when the network returned.

## Results

### ep12 (seeds 42-46, from 2026-09-01 + 09-02)

| seed | 6-fold mean AP |
|------|----------------|
| 42   | 0.627 |
| 43   | 0.569 |
| 44   | 0.540 |
| 45   | 0.571 |
| 46   | 0.412 |
| mean | ~0.544 |

### ep20 (seeds 42-46, all complete)

| seed | 6-fold mean AP | delta vs ep12 |
|------|----------------|---------------|
| 42   | 0.554 | -0.073 |
| 43   | 0.471 | -0.098 |
| 44   | 0.529 | -0.011 |
| 45   | 0.514 | -0.057 |
| 46   | 0.362 | -0.050 |
| mean | ~0.486 | **every seed dropped** |

### ep15 (seed 42 only -- seeds 43-46 deliberately skipped)

`prediction_leave_one_seizure_out_20260902-131908.csv`:

| seizure | AP    | hit_smoothed | FAR/h smoothed |
|---------|-------|--------------|----------------|
| 1_03_0  | 0.795 | True         | 16.00 |
| 1_04_0  | 0.448 | True         | 25.35 |
| 1_15_0  | 0.385 | **False**    | 1.22  |
| 1_16_0  | 0.994 | True         | 0.00  |
| 1_18_0  | 0.882 | True         | 0.00  |
| 1_26_0  | 0.369 | **False**    | 1.60  |
| **mean**| **0.6455** | | |

ep15 seed42 (0.6455) > ep12 seed42 (0.627) > ep20 seed42 (0.554).

## Read

**Epoch count, not the val split, is the dominant knob at val=0.** With no
early stopping / no best-ckpt restore, 20 epochs overfits the training
folds uniformly (all 5 seeds lost 0.01-0.10 AP vs ep12). 12-15 epochs is
the sweet spot; seed42 at ep15 lands right on the val-split=0.2 anchor
(~0.64-0.65).

So the 0.2 validation split is **net-positive and not the noise source**:
its best-ckpt restore is effectively doing an implicit early-stop that
lands the model in the same ~0.64 place ep15 reaches, and the seed-to-
seed spread is the same ~0.2 range with or without it (val=0 ep12/ep20
spreads: 0.21 / 0.19). Dropping the split buys nothing and costs the
safety net. **Keep `--validation-split 0.2` in the headline config.**

The chronic weak folds are unchanged: 1_26 (never above ~0.37 at any
epoch count), and 1_15/1_04 wobble. 1_16 and 1_18 are reliably strong.
1_03 is epoch-sensitive (0.567 at ep12 -> 0.795 at ep15).

## Still wanted (deferred -- do NOT run now)

- **ep15 seeds 43, 44, 45, 46** to get a full 5-seed ep15 row and confirm
  the ep15 mean sits above ep12 (~0.544) and near the val-split anchor.
  Launch: clone `_to_delete/val0_ep20.sh` -> ep15 with `for seed in 43 44
  45 46` (an `_to_delete/val0_ep15.sh` from this session already exists;
  edit its seed list). Self-serializes on the
  `run_pre_(fp32_emptycache|val0)` guard.

## State at session end

- ep15 sweep wrapper (`_to_delete/val0_ep15.sh`, pid 71937) KILLED after
  seed42 finished; seeds 43-46 skipped per user.
- No val0 compute running.
- Next per CONTEXT.md: AWS S3 upload of `~/mne_data/dense_edge_cache`
  (~36 GB), then fp32 6-fold tracking run.
