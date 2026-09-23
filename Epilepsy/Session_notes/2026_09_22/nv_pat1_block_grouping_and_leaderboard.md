# NV Pat1 block-grouped comparison + real contest leaderboard found

**Date:** 2026-09-22
**Companion to:** `nv_pat1_first_benchmark.md` (same session, read that one first
for the base setup/caveats -- this note only covers what's new).

## Block-grouped vs segment-grouped, final 6-fold means

Both `scripts/run_nv_pat1.py` runs, same params, only `--grouping` differs.
Numbers below are still SUB-WINDOW-scored (the segment-level aggregation
fix landed in the script this session but has not been exercised on a
full run yet -- see "Still open" below).

| metric | segment-grouped | block-grouped | delta |
|---|---|---|---|
| roc_auc | 0.854 | 0.882 | +0.027 |
| average_precision (AP) | 0.752 | 0.805 | +0.053 |
| accuracy | 0.755 | 0.767 | +0.012 |
| precision | 0.603 | 0.600 | -0.003 |
| recall | 0.736 | 0.792 | +0.056 |
| f1 | 0.654 | 0.680 | +0.026 |

CSVs: `nv_pat1_stratified_kfold_20260922-134903.csv` (segment),
`nv_pat1_stratified_kfold_blockgrouped_20260922-141154.csv` (block), both
under `Epilepsy/results/godoy_tmc/nv/`.

**This is the opposite direction from what was predicted going in.** The
hypothesis was that fixing segment-only grouping's blind spot (same-block
segments landing on both sides of a fold) would REDUCE the apparent
score, since it removes an inflation source. Instead block-grouping came
back higher on every metric except precision (roughly flat).

Read on this, with appropriate uncertainty: most likely fold-to-fold
noise dominates a change this size. Only 34/253 preictal segments (~13%)
were affected by the regrouping -- a correction that small, against folds
that already swing 0.757-0.905 on their own, isn't guaranteed to move the
mean in either predictable direction. This result should NOT be read as
"block-grouping doesn't work" or "leakage wasn't real" -- it demonstrates
block-grouping does something different from segment-grouping (chain
members no longer split across folds), but with this few segments
affected, isolating a clean leakage-correction signal from ordinary
variance isn't possible with 6 folds. A cleaner test would need either
many more reconstructed chains (would need Pat2/Pat3 too, or a larger
patient) or repeated-seed runs to average out fold variance -- neither
done here, out of scope for "Pat1 only, this session."

## Real contest leaderboard found: `datasets/epilepsy/NV_Contest_results.csv`

Found in-repo (moved from `Epilepsy/` to `datasets/epilepsy/` per user
request this session) -- **a real leaderboard scored against the actual
withheld NV Test answer key**, which we do not have and cannot reproduce.
68 submissions across 7 teams (MatthiasEb, gorjant, dvarnai, michaln,
hlya23dd, notsorandomanymore, sfwon17), ranked on average rank across
3 metrics: Overall AUC (pooled across all patients' files), Average AUC
across patients, Minimum AUC across patients.

- Best single submission: `MatthiasEb/trainval_test.csv`, Overall AUC
  **0.867**
- Official contest winner (best avg rank, not best single metric):
  `notsorandomanymore/winning_solution.csv`, Overall AUC **0.807**
- Full-field Overall AUC: mean 0.711, median 0.728, min 0.537, max 0.867,
  std 0.103 (n=68)

User corrected an initial (wrong) assumption made this session that this
leaderboard was the original 2016 Kaggle contest, which would have
predated `GodoyTMCClassifier`'s source paper (arXiv:2209.11172, 2022) and
ruled out any submission resembling it on timing alone -- user says this
is actually a 2024 private-contest leaderboard (some filenames are dated
2019, suggesting either a long-running evaluation server or a mix of
historical + newer submissions; not fully resolved). No method
descriptions, code, or write-ups for any team came with this file --
only team/filename/score. **Cannot identify what architecture any
submission used**, and specifically cannot confirm or rule out whether
anything resembling GodoyTMC was ever submitted.

## Can we estimate how `godoy_tmc` would rank on this leaderboard? No.

Asked directly this session, answered no on four independent grounds
(any one alone would be disqualifying):

1. Our numbers are still sub-window-scored, not segment-scored -- wrong
   unit vs. how real submissions were evaluated.
2. Our numbers are Pat1-only; leaderboard metrics pool/average across all
   3 patients. No way to know if Pat1 is representative.
3. Our numbers are internal Train-CV; leaderboard numbers are real
   held-out Test. CV-vs-Test gap is well-documented and unmeasured here.
4. Our own fold-to-fold noise (0.757-0.915 across both grouping runs) is
   MORE THAN DOUBLE the entire competitive margin on this leaderboard
   (winner 0.807 vs. top score 0.867, a 0.06 gap; full field spread is
   0.537-0.867, a 0.33 range comparable to our own single-run fold
   spread). A point estimate from our data couldn't resolve "mid-pack"
   from "near the top" even in principle.

## AP vs. ROC-AUC and base-rate normalization (context from earlier this
   session, holds for both grouping runs)

NV Pat1's Train preictal base rate (~31%) is far higher than CHB-MIT's
(~4%), so raw AP isn't comparable across the two datasets -- AP's
no-skill baseline equals the base rate. Lift-over-baseline (AP / base
rate) told the more honest story: CHB-MIT ~13.6x, NV segment-grouped
~2.4x, NV block-grouped ~2.6x -- i.e. by this normalization NV Pat1 is
the LESS-well-modeled problem, not the easier one, despite higher raw AP.
Best current guess for why: every 30s sub-window inherits its whole
10-minute segment's single label (no per-window SPH/SOP-style timing
like CHB-MIT has), so a meaningful fraction of "preictal" sub-windows are
probably far enough from the seizure to look interictal -- real label
noise baked into the sub-window-level scoring this note's numbers still
use.

## Still open / next

- **Segment-level aggregation rerun, both groupings.** The scoring fix
  (mean-pool sub-window probabilities to one score per real segment
  before computing metrics) is in `run_nv_pat1.py` as of this session but
  has not been run to completion yet. This is the number that should
  actually get cited going forward -- everything in this note and
  `nv_pat1_first_benchmark.md` is sub-window-scored and should be treated
  as a first look, not a result.
- No way to check any of this against real Test performance without
  either (a) a real submission process to the actual contest (not being
  pursued -- see this session's earlier discussion of why probing a
  scoring oracle would be leaderboard-probing, declined), or (b) getting
  Levin to share the actual withheld Test labels for Pat1 specifically
  under the existing DUA, if that's even something he's able to share.
- Pat2Train / Pat3Train / any Test folder: still not downloaded, out of
  scope per current instructions ("only worry about patient 1").
