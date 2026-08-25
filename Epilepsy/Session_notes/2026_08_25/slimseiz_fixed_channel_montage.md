# SlimSeiz fixed-channel montage vs. per-fold adaptive selection (chb01)

## Motivation

User asked whether SlimSeiz publishes any pretrained subsets/channel
lists, to get a comparison point against this repo's own per-fold
adaptive channel selection (`select_slimseiz_channels`,
`Epilepsy/pipelines/slimseiz_channel_select.py`).

`github.com/guoruilu/SlimSeiz` ships no pretrained weights/checkpoints
(bare repo, 5 notebooks + a one-line README pointing at a SharePoint data
dump). But `Common_channesl.ipynb` (fetched via `gh api`, not vendored
into this repo as a file — not one of the two notebooks already
ported/cited) publishes the *result* of running the paper's own
per-patient selection loop across all 24 CHB-MIT patients, then tallying
which channels land in each patient's own top-8 most often:

```
P8-O2: 18   C3-P3: 18   P3-O1: 18   FZ-CZ: 17
C4-P4: 17   P4-O2: 17   CZ-PZ: 15   F3-C3: 14
```
(counts out of 24 patients)

The notebook's own final step takes the top 8 of that tally —
`{P3-O1, P8-O2, C3-P3, C4-P4, FZ-CZ, P4-O2, CZ-PZ, F3-C3}` — as a single
fixed montage, not a genuinely per-patient one. This reads like the
condition the paper's 94.8%/95.5%/94.0% headline numbers are actually
reported under. For chb01 specifically, its own top-8 overlaps this fixed
set in 7/8 (missing only `F3-C3`).

**Methodological note, also flagged to the user:** if this fixed,
cohort-wide-tallied set is what's used to evaluate every patient
(including the ones that voted into the tally), that's a real — if likely
mild — selection-bias leak: each patient's own channel-selection output
partially informs the channel set used to then evaluate that same
patient. Classic "feature selection on the full dataset before
per-subject CV" pitfall. Not confirmed against the actual published
methods section, only inferred from this notebook's code.

## Implementation

Added `--slimseiz-fixed-channels` to `run_pipelines.py`:
- `CHB01_CHANNEL_NAMES` — chb01's 23-channel order, read directly off
  `chb01_01.edf` via `mne.io.read_raw_edf(preload=False)` (confirmed
  `datasets/epilepsy/chb_mit.py` does no picking/reordering, so this is
  the real order every fold's `X` channel axis uses).
- `SLIMSEIZ_PAPER_FIXED_CHANNELS` — the 8-channel list above.
- `SlimSeizClassifier.channel_select_fixed_indices` (new param) — when
  set, `fit()` skips `select_slimseiz_channels` (stage 1) entirely, no
  PCA/SMOTE/DecisionTree call at all, just slices `X` to the given
  indices before stage 2. Strictly safer than the default adaptive path
  memory-wise, since it removes the stage implicated in the earlier crash
  investigation (see `CONTEXT.md`'s "Known gotchas").
- CLI: `--slimseiz-fixed-channels` (no args = default list above, or pass
  explicit channel names). chb01-only today — errors if `--subjects` isn't
  `[1]`.

Smoke-tested (`--smoke --slimseiz-fixed-channels --max-folds 1`): resolved
indices `[7,15,6,10,16,11,17,5]`, hand-verified against the EDF header,
peak RSS ~1GB.

## Real 6-fold run

Launched after a queued watcher confirmed a concurrent `--pipeline
dbconformer` job had exited (see `CONTEXT.md` "Right now" for the
handoff). RSS-capped at 12GB via the crash investigation's watchdog
wrapper script.

**Operational hiccup (not a crash):** first launch reused that wrapper's
hardcoded ~700s wall-clock kill, sized for testing a single fold during
the crash investigation. A real 6-fold pass takes longer than that; the
watchdog killed the process cleanly mid-fold-5 (`1_18_0`, epoch 6/20).
Memory was never a problem — RSS stayed ~6.6GB the whole time, nowhere
near the 12GB cap. Fixed by making the wrapper's timeout a `TIMEOUT_S` env
var and rerunning just the 2 remaining folds via `--skip-folds 0 1 2 3`
(fold order is `(subject, run, seizure_onset)`-sorted: `1_03_0, 1_04_0,
1_15_0, 1_16_0, 1_18_0, 1_26_0` → indices 0-5). Second launch completed
cleanly, peak RSS 8.3GB.

## Results: fixed montage vs. adaptive per-fold selection

Adaptive baseline: `prediction_leave_one_seizure_out_20260825-171651.csv`
(real 6-fold run, completed before the original crash attempt at
18:23:06). Fixed-montage run: folds 0-3 from
`full6fold_slimseiz_fixedch_20260825-204111.log` (CSV lost to the timeout
kill, values pulled from the per-fold console summary lines instead),
folds 4-5 from `prediction_leave_one_seizure_out_20260825-205851.csv`.

| seizure | adaptive f1 | fixed f1 | adaptive recall | fixed recall | fixed hit (raw/smoothed) |
|---|---|---|---|---|---|
| 1_03_0 | 0.324 | 0.246 | 0.967 | 0.700 | True / True |
| 1_04_0 | 0.331 | 0.342 | 0.867 | 0.867 | True / True |
| 1_15_0 | 0.109 | 0.103 | 0.100 | 0.100 | True / False |
| 1_16_0 | 0.979 | 0.979 | 1.000 | 1.000 | True / True |
| 1_18_0 | 0.000 | 0.000 | 0.000 | 0.000 | False / False |
| 1_26_0 | 0.296 | 0.436 | 0.400 | 0.567 | True / True |

Mean across 6 folds:

| metric | adaptive (per-fold) | fixed (paper's 8) |
|---|---|---|
| precision | 0.286 | 0.297 |
| recall | 0.556 | 0.539 |
| f1 | 0.340 | 0.351 |
| FAR/h raw / smoothed | 8.56 / 6.31 | 8.22 / 6.08 |
| hit rate, raw | 5/6 | 5/6 |
| hit rate, smoothed | 4/6 | 4/6 |

**Conclusion:** essentially a tie on aggregate. Both configurations miss
the exact same seizure on hit rate (`1_18_0` — mean preictal score
~0.0004 under both, a genuinely hard fold rather than a channel
selection artifact) and the same fold on the smoothed metric (`1_15_0`).
Per-fold, `1_03_0` gets worse under the fixed set (f1 0.324→0.246) while
`1_26_0` gets better (f1 0.296→0.436) — these roughly cancel out. For
chb01, the crash-implicated stage-1 selection isn't buying anything
measurable over the paper's own fixed montage; `--slimseiz-fixed-channels`
reproduces the same result more cheaply (no PCA/SMOTE/DecisionTree at
all) and more safely. Whether this generalizes past chb01 is untested —
`--subjects` is chb01-only in this repo today.

## Related memory files

- `slimseiz-channel-select-crashed-mac.md` — the original crash
  investigation this run builds on.
- `slimseiz-fixed-8channel-montage.md` — the `Common_channesl.ipynb`
  finding and the CLI flag, written before this run's results came in.
