# NV Pat1 first benchmark (`godoy_tmc`) — segment-grouped, sub-window-scored

**Date:** 2026-09-22
**Scope:** Kuhlmann NeuroVista (NV) contest data, Pat1 only, per explicit
user request ("only worry about patient 1"). First real number off this
dataset since the DUA-gated download landed this session.

## What this run is

`scripts/run_nv_pat1.py` (no `--grouping` flag, i.e. the `segment`
default) -- `GodoyTMCClassifier`/`GODOY_TMC_PARAMS` reused verbatim from
`Epilepsy/run_pipelines.py`, `decimate=4` (100Hz), 30s sub-windows,
`StratifiedGroupKFold(n_splits=6)` grouped by individual segment id.

Full log: `/tmp/nv_pat1_godoy_run.log` (not durable -- copy out if this
needs to survive a reboot). CSV:
`Epilepsy/results/godoy_tmc/nv/nv_pat1_stratified_kfold_20260922-134903.csv`.

## Results (6-fold mean, SUB-WINDOW-level scoring)

| metric | value |
|---|---|
| accuracy | 0.755 |
| precision | 0.603 |
| recall | 0.736 |
| f1 | 0.654 |
| average_precision (AUC-PR) | 0.752 |
| roc_auc | 0.854 |

Per-fold roc_auc: 0.757, 0.875, 0.875, 0.855, 0.905, 0.858 -- fairly
tight (0.76-0.91), no single catastrophic fold the way CHB-MIT's
`godoy_tmc` seed-sweep saw (`1_03`/`1_26` carrying most of that run's
variance).

## Why this number needs real caveats before it's used anywhere

1. **NOT leave-one-seizure-out.** No seizure/event-grouping metadata
   survives in the released NV files (only a `data` key per .mat) -- see
   `datasets/epilepsy/kuhlmann_nv.py`'s module docstring. Segment-level
   `StratifiedGroupKFold` stops one segment's own sub-windows leaking
   across a fold, but different segments from the same underlying
   one-hour preictal block can still land on opposite sides of a fold.
   That's a real, not-fully-closed leakage risk -- never pool this number
   into the same table as CHB-MIT's true LOSO results without this
   caveat attached.

2. **Scored at the wrong granularity.** This run's precision/recall/F1/
   ROC-AUC are computed per SUB-WINDOW (~2,700 per fold), not per
   10-minute segment (~135 per fold) -- caught via user pushback mid-
   session, already fixed in `run_nv_pat1.py` (mean-pool sub-window
   probabilities back to one score per segment before scoring) but NOT
   YET RE-RUN at the time of this note. The numbers above are the
   sub-window-scored ones; treat them as a first look only, not the
   number to cite going forward. See "Next" below.

3. **Label granularity mismatch vs. CHB-MIT.** Every 30s sub-window here
   inherits its parent 10-minute segment's single label -- unlike
   CHB-MIT's `label_mode="prediction"`, which labels each window
   individually via precise SPH/SOP timing against a real annotated
   seizure onset (`paradigms/continuous_labeling.py::_label_windows_
   prediction`). A sub-window from the start of a preictal segment (up to
   an hour before the seizure, if it's segment 1 of ~6 in a block) gets
   the same label as one from the last 30 seconds before onset. Real
   label noise, likely the single biggest reason this doesn't compare
   cleanly to CHB-MIT. NOT fixable by inventing finer labels -- the real
   contest task (and Test's actual structure) is genuinely one label per
   whole segment, so finer labels would target something Test never
   asks about. Left as an open, load-bearing caveat, not a bug to fix.

## vs. CHB-MIT `godoy_tmc` (true LOSO, `prediction_leave_one_seizure_out_*.csv`, 2026-08-31 seed sweep, chb01)

| metric | CHB-MIT (seed-sweep mean) | NV Pat1 (this run) |
|---|---|---|
| roc_auc | **0.950** | 0.854 |
| average_precision | 0.552 | 0.752 |
| precision | 0.443 | 0.603 |
| recall | 0.698 | 0.736 |
| f1 | 0.485 | 0.654 |

Not apples-to-apples (see caveats above) -- AUC-PR favors NV mechanically
since its Train set is ~31% preictal vs. CHB-MIT's continuous-recording
imbalance; ROC-AUC is the fairer comparison of the two and CHB-MIT wins
on it. NV's number is also probably still somewhat leakage-inflated
relative to CHB-MIT's stricter protocol (see caveat 1), so the true gap
is likely larger than this table shows.

## Also this session: reconstruct_preictal_blocks (`datasets/epilepsy/kuhlmann_nv.py`)

Tried to recover the contest's true ~6-segment preictal block grouping
from boundary signal continuity, to fix caveat 1 above.

- **v1 (raw amplitude L2 distance):** FAILED its own validation check --
  after excluding 51/253 preictal segments with flat/clipped boundaries
  (real recording-dropout artifact), the best-match/median-distance ratio
  was a smooth 0.41-1.0 continuum, no separation between genuine and
  coincidental matches. Abandoned.
- **v2 (per-channel z-score each segment, then cosine similarity):**
  PASSED validation -- clear cluster near 0.95-0.97 similarity-gap,
  distinct from a ~0.4 bulk. Recovered 12 chains (sizes 2-5) covering
  34/253 preictal segments. Wired in as `--grouping block`.
- A separate full run with `--grouping block` was in flight in parallel
  with this one at the time of writing (see companion note once it
  finishes) -- one fold in so far (roc_auc 0.873 vs. segment-grouped's
  0.757 on fold 0), too early to say whether block-grouping moves the
  6-fold mean meaningfully.

## Next

- Finish and note the `--grouping block` comparison run.
- Re-run BOTH groupings with the segment-level-aggregation scoring fix
  (already in `run_nv_pat1.py` as of this session, not yet exercised on
  a full 6-fold run) -- that's the number that should actually get cited
  going forward, not the sub-window-scored one in the table above.
- Pat2Train / Pat3Train / any Test folder: still not downloaded, out of
  scope per current instructions.
