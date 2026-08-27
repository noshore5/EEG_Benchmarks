# `temporal_graph_mamba` real 6-fold LOSO run, plus an overnight regularization-tuning attempt that didn't pan out

## What this is

The first real, full-epoch-budget, 6-fold leave-one-seizure-out
`--pipeline temporal_graph_mamba --label-mode prediction` run on chb01
(branch `graph-state-mamba` -- see `temporal_graph_mamba_aggregate_
then_mamba.md`, 2026-08-26, for what this pipeline is and why it exists).
Three separate process runs, overnight 2026-08-26 -> 2026-08-27, are
stitched together into one complete 6-fold result below.

## Run 1: the real 6-fold run, killed externally mid-fold-6

`/tmp/tg_mamba_full2.log`, `--device cpu --output-dir /tmp/tg_mamba_full2`,
default (untuned) `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS`. This was a
*restart* of an earlier attempt that crashed from disk exhaustion (both
`~/mne_data/dense_edge_cache` and `~/mne_data/cwt_window_cache` were
wiped and rebuilt cold going into this run -- see the disk-usage
discussion below).

Completed folds 1-5 cleanly, then the process was **killed by something
external** partway into fold 6 (`1_26_0`, 1/20 epochs in) -- not a disk
crash (disk was at a stable ~20-27GB free at the time), not anything this
session did. Root cause not identified; flagged here in case it recurs.

Results, folds 1-5 (all hit=True raw and smoothed):

| Fold | FAR/h raw→sm | precision | recall | f1 | AP (auc_pr) |
|---|---|---|---|---|---|
| `1_03_0` | 18.500→17.167 | 0.213 | 1.000 | 0.351 | **0.792** |
| `1_04_0` | 35.613→26.323 | 0.140 | 1.000 | 0.246 | **0.830** |
| `1_15_0` | 5.058→1.570 | 0.396 | 0.633 | 0.487 | 0.331 |
| `1_16_0` | 1.000→0.000 | 0.793 | 1.000 | 0.885 | **0.996** |
| `1_18_0` | 0.167→0.000 | 0.909 | 0.333 | 0.488 | 0.802 |

Notable pattern: AP is consistently strong-to-excellent, and the two
folds with the *worst* FAR/h (1, 2) are also the two with the *highest*
AP -- reinforcing the "good ranking, bad calibration at the fixed 0.5
threshold" hypothesis already parked in `CONTEXT.md`'s open threads
(decision-threshold calibration).

## Disk usage note (why this run needed ~70GB of cache)

This was the first full 6-fold *real* run ever executed **locally on CPU
with disk caching enabled**. `dense_edge_cache`/`cwt_window_cache` default
ON for `device="cpu"`, OFF for CUDA -- every prior full-scale run either
used GPU (cache off) or was capped by `--smoke`/`--max-folds`. Combined
with the cold rebuild after the pre-run wipe, this made disk usage look
alarming (down to ~16-20GB free at the tightest points) relative to any
run seen before, even though nothing was actually wrong. Caching is
keyed by `(window, channel-pair)` features, not by a fold's train/test
role, so growth per fold is NOT purely additive -- most of a new fold's
training set reuses windows already cached by earlier folds; only each
newly-held-out seizure's test-window cache is genuinely new per fold.

## Run 2: overnight regularization-tuning attempt (all 6 folds, but NOT adopted)

After Run 1 was killed, launched a second full 6-fold attempt with two
constants tuned in `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS`
(`Epilepsy/run_pipelines.py`) as a best-effort, unvalidated attempt at
improving FAR/h at the fixed 0.5 decision threshold (no threshold-
calibration code exists yet -- see `CONTEXT.md`):

```python
weight_decay=3e-4,          # was 1e-4 (_SHARED_ARCH_PARAMS default)
temporal_graph_mamba_dropout=0.15,  # was 0.0
```

`/tmp/tg_mamba_tuned_full.log`, `--output-dir /tmp/tg_mamba_tuned_full`,
wrapped in a small launcher script (`/tmp/tg_mamba_tuned_launcher.sh`)
that auto-retries and clears both disk caches on ENOSPC/low-disk (<8GB
free), up to 3 attempts, run via `nohup ... & disown` so it survives the
session ending. Finished cleanly overnight (all 6 folds, no retries
needed), `=== finished successfully ===` at 2026-08-27 02:03am.

**Verdict: net negative, not adopted.**

| Fold | Untuned AP | Tuned AP | Untuned FAR sm | Tuned FAR sm | Notes |
|---|---|---|---|---|---|
| `1_03_0` | 0.792 | 0.232 | 17.167 | 11.667 | AP collapsed |
| `1_04_0` | 0.830 | **0.074** | 26.323 | 0.000 | **failed to train** -- restored epoch 1 (never improved past init), hit=False, everything zeroed |
| `1_15_0` | 0.331 | 0.604 | 1.570 | 3.837 | only clear win |
| `1_16_0` | 0.996 | 0.976 | 0.000 | 0.000 | wash |
| `1_18_0` | 0.802 | 0.657 | 0.000 | 0.000 | slightly worse |
| `1_26_0` | 0.293 | 0.314 | 5.600 | 5.800 | roughly a wash |

Fold 2 (`1_04_0`) outright failing to train (early stopping restoring
epoch 1's checkpoint) is the clearest signal: `weight_decay=3e-4` +
`dropout=0.15` together were too strong a regularization pull for this
fold size, not a targeted fix for the FAR/calibration gap. The code edit
was left in place uncommitted in the working tree (visible via `git
diff`) for the user to inspect, but the untuned Run 1/Run 3 numbers
remain the pipeline's real baseline -- **do not treat the tuned numbers
as the pipeline's real performance.**

## Run 3: fold-6-only rerun to complete the untuned baseline

Since Run 1 never produced an untuned `1_26_0` result, reran just that
one fold under the original (untuned) params using the existing
`--skip-folds` CLI flag (`leave_one_seizure_out_prediction` already
supports this -- diagnostic-only, lets a crashed/partial multi-fold run
resume without redoing completed folds):

```
python3 Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba \
  --label-mode prediction --device cpu --skip-folds 0 1 2 3 4 \
  --output-dir /tmp/tg_mamba_fold6_only
```

To avoid running this under the (uncommitted, not-yet-reverted) tuned
params, temporarily `git stash push -- Epilepsy/run_pipelines.py` before
launching, then `git stash pop` immediately after the process had
started (safe -- Python doesn't re-read source after import, so the
already-launched process kept its untuned params while the working tree
was restored to the tuned edit for anyone inspecting it afterward).

Result: `seizure 1_26_0: n_test=630 preictal=30  hit=True (smoothed=True)
FAR/h=8.200 (smoothed=5.600)  precision=0.339  recall=0.700  f1=0.457
auc_pr=0.293`.

## Complete untuned 6-fold `temporal_graph_mamba` baseline

| Fold | FAR/h raw→sm | precision | recall | f1 | AP (auc_pr) | hit raw→sm |
|---|---|---|---|---|---|---|
| `1_03_0` | 18.500→17.167 | 0.213 | 1.000 | 0.351 | 0.792 | True/True |
| `1_04_0` | 35.613→26.323 | 0.140 | 1.000 | 0.246 | 0.830 | True/True |
| `1_15_0` | 5.058→1.570 | 0.396 | 0.633 | 0.487 | 0.331 | True/True |
| `1_16_0` | 1.000→0.000 | 0.793 | 1.000 | 0.885 | 0.996 | True/True |
| `1_18_0` | 0.167→0.000 | 0.909 | 0.333 | 0.488 | 0.802 | **True/False** |
| `1_26_0` | 8.200→5.600 | 0.339 | 0.700 | 0.457 | 0.293 | True/True |

**CORRECTED 2026-08-27 (second finding-independent confirmation, see
below):** this table originally claimed a 6/6 smoothed hit rate here, but
`1_18_0`'s own printed result line always said `hit=True (smoothed=
False)` -- the table's `1_18_0` row was simply transcribed wrong when
this table was first compiled (stitching together numbers across three
separate partial runs). **Actual hit rate: 6/6 raw, 5/6 smoothed
(83.3%)** -- matching GRU and DBConformer's own smoothed hit rate, not
uniquely better than them on this axis. `temporal_graph_mamba` still
leads the group clearly on mean AP/precision/f1 (see the mean-across-
folds comparison table above) -- that part of the finding stands. Re-
verified 2026-08-27 via a second, fully independent, non-`--skip-folds`
6-fold run (`/tmp/tg_mamba_retrain_full.log`, `--dump-window-scores`,
same untuned params, ran clean to completion end-to-end with no external
kill and no stashing needed) -- all 6 folds' numbers matched this table
exactly, confirming the run is deterministic and reproducible, just
correcting the one mistranscribed cell.

Fold-by-fold, this pipeline shows a real pattern worth treating as a
finding, not noise: AP is consistently high (0.29-0.996), and the folds
where FAR/h is worst (`1_03_0`, `1_04_0`) are the SAME folds where AP is
highest. Consistent with a genuinely well-ranked model being scored at a
poorly-chosen fixed 0.5 decision threshold, rather than a genuinely bad
or leaky model -- see the code-level leakage check in
`temporal_graph_mamba_aggregate_then_mamba.md`'s sibling discussion this
session (verified `event_mode="dense"` vs `"temporal_graph"` consume
identical precomputed input, no information asymmetry) and the
decision-threshold-calibration open thread already in `CONTEXT.md`.

## What's NOT done yet

- **Not committed.** `Epilepsy/run_pipelines.py`'s tuned-params edit
  (`weight_decay=3e-4`, `temporal_graph_mamba_dropout=0.15` on
  `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS`) sits uncommitted in the
  working tree. Given the tuning attempt's negative verdict above, this
  should probably be reverted rather than committed -- left for the user
  to decide.
- **No comparison against `temporal_graph_gru`.** That comparison run was
  explicitly deferred by the user ("don't kick it off yet, I'll do it
  before bed") and was never started this session.
- **Decision-threshold calibration still unimplemented** -- the open
  thread in `CONTEXT.md` (calibrate against the validation split's own PR
  curve instead of hardcoding 0.5) remains the most promising untried
  lever for the FAR/h problem this fold-by-fold pattern keeps pointing at.
- **The external kill of Run 1 mid-fold-6 is unexplained.** Not a disk
  issue (confirmed via `df -h` at the time), not caused by anything this
  session ran. Worth watching for recurrence on future long unattended
  runs.

## Next steps, in order

1. Decide whether to revert the tuned-params edit in `run_pipelines.py`
   (recommended, given the negative verdict) or keep it as a documented
   dead end.
2. Implement decision-threshold calibration against the validation
   split's own PR curve (the actual promising lever, still unbuilt).
3. `temporal_graph_gru` real 6-fold run, when the user is ready to kick
   it off, for the GRU-vs-Mamba comparison this branch exists to answer.
