# Per-fold memory leak in LOSO prediction + first val_split=0 row

2026-09-01 (afternoon/evening session, Mac)

## The bug (found + fixed)

`leave_one_seizure_out_prediction` (`Epilepsy/run_pipelines.py`, the
`for fold_i, seizure in enumerate(unique_seizures)` loop) never released a
fold's `StreamingSparseEvidenceGNNClassifier` (model weights, optimizer
state, per-batch feature tensors) before starting the next fold. Reference
cycles kept the previous fold's objects reachable until the cyclic GC
happened to run, so the process footprint climbed **monotonically across
the 6-fold LOSO loop**.

Observed on the fp32 seed-43 run (pid 34495):

| fold  | epoch_time |
|-------|-----------|
| 1_03  | 49 s      |
| 1_04  | 81 s      |
| 1_15  | 87 s      |
| 1_16  | 4126 s    |
| 1_18  | 2432 -> 2235 s |

`footprint 34495` at fold 5: **phys_footprint 25 GB, peak 28 GB**, with
RSS only 289 MB -- i.e. ~25 GB of anonymous memory, almost all
compressed/swapped, on a 16 GB machine. Swap pegged at 20 GB, whole
machine (all terminals) thrashing. Per-fold *work* is roughly constant;
the slowdown is purely the machine crossing into swap around fold 4.

This is the real cause of every "page-cache thrash on folds 2-6" note in
CONTEXT.md item 2b/3 -- it was NOT just dense-edge cache size. The
`_sync_and_release` monkeypatch in the run wrappers only calls
`torch.mps.empty_cache()` (frees the MPS pool, does nothing about
CPU-side retained Python objects).

The two disk caches (`DiskCWTCache`, `dense_edge_cache`) were checked and
are clean -- pure disk, no in-RAM dict. `DiskCWTCache` had this exact
class of bug once before (see its docstring) and was already fixed.

### Fix

Commit `51cf7a3` on `main`: explicit teardown at the end of each fold --

```python
del clf, proba, y_score, y_pred, y_pred_smoothed
del X_train, y_train, X_test, y_test, meta_test
_gc.collect()
try:
    import torch as _torch
    if _torch.backends.mps.is_available():
        _torch.mps.empty_cache()
except Exception:
    pass
```

Verified on val=0 ep12 seed42: **~52-56 s/epoch flat through all 6
folds**, swap never pegged, no thrash. Fold 5 (where the old run hit
2400 s) ran at 55 s.

## Cleanup done this session

- Killed fp32 seed-43 job (pid 34495, mid-fold-5) + its old val0 driver
  (pid 41150). fp32 seed-43 partial **discarded** -- fp16-vs-fp32 was
  already a null result (split-noise, not a dtype effect), so no loss.
- Phase B fp32 sweep is DONE at seeds 42/43/44/45. The infinite
  `phaseB_fp32_sweep.sh` driver stays dead. Do not run more fp32 seeds.

## val_split=0 sweep -- first row

`_to_delete/val0_sweep.sh` (driver) + `_to_delete/run_pre_val0.py`
(wrapper, stock fp32 dense-edge cache, no fp16 patch). Sweep design:
epochs {12,20} x seeds {42,43,44,45,46}, `--validation-split 0` (train on
100% of each fold, no early stopping, no best-ckpt restore). One MPS job
at a time, auto git add/commit/push per completed job.

With val=0 the seed no longer controls a train/val partition. What it
still controls: weight init, DataLoader shuffle, SGD stochasticity. The
negative subsample is pinned (`subsample_seed` defaults to 42 and the
prediction path never overrides it) and the test set is LOSO-fixed. So
there is **no split-luck channel** -- any seed-to-seed spread in this
sweep is pure optimization noise, which is the error bar worth reporting.

### ep12 seed42 (commit `88aff75`)

| seizure | auc_pr (AP) | hit_smoothed | FAR/h smoothed |
|---------|-------------|--------------|----------------|
| 1_03_0  | 0.567       | True         | 11.17          |
| 1_04_0  | 0.576       | True         | 19.16          |
| 1_15_0  | 0.372       | True         | 2.27           |
| 1_16_0  | 1.000       | True         | 0.00           |
| 1_18_0  | 0.889       | **False**    | 0.00           |
| 1_26_0  | 0.355       | True         | 6.80           |
| **mean**| **0.627**   |              |                |

vs the val-split=0.2 references (per-fold auc_pr):

- fp16 seed43 ep20: 1_03=0.234 1_04=0.376 1_15=0.463 1_16=0.939
  1_18=0.714 1_26=0.216  -> mean ~0.490
- fp32 seed43 (partial): 1_03=0.240 1_04=0.324 1_15=0.460 1_16=0.940

Read: the two chronic "noise folds" 1_03/1_04 jump hard (+0.33 / +0.22)
once the model trains on 100% of the fold instead of holding 20% out --
and this is ep12 vs the reference's ep20. 1_15 gives back ~0.09. 1_18 AP
rises (0.71 -> 0.89) but the k-of-n smoothing suppressed its alarm
(hit_smoothed=False) -- high preictal scores, just not k consecutive.
6-fold mean +0.14 over the val-split runs.

**One seed, one epoch count -- not a conclusion.** The sweep's real
question is whether the 6-fold-mean spread across seeds 42-46 is tighter
than the val-split sweep's ~0.05-0.13. Need >=3 complete seeds to judge.
ep20 rows are the more comparable ones (reference is ep20).

## State at session end

- **All compute stopped** (laptop being closed = sleep). Sweep driver
  killed; only ep12 seed42 completed and is pushed.
- **To resume:** relaunch `nohup zsh _to_delete/val0_sweep.sh <dead-pid>`
  from the repo root. It restarts at ep12 seed42 (no skip-folds wiring in
  the wrapper), so either let it redo seed42 or edit the loop to start at
  seed43.
- `AWS_INFRA.md` + `CONTEXT.md` left with **staged uncommitted changes
  from a concurrent AWS-box shell** (merged clean, no conflict markers) --
  that shell should commit them.
