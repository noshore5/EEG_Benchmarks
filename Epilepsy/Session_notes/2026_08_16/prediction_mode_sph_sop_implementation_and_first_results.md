# Session notes — prediction mode (SPH/SOP), S3 mirror discovery, first real-ish results, disk cleanup (2026-08-16)

Branch: `main`. Repo: `/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`.

Follow-on to [today's earlier session](mps_batch_tuning_memory_fix_and_second_real_result.md),
which left detection mode tuned and working (mean F1 0.864, ROC-AUC 0.999,
7-fold leave-one-seizure-out, chb01 only). This session added a second,
deliberately separate testing paradigm — seizure *prediction*
(preictal/interictal, SPH/SOP-style) — alongside detection, per an explicit
6-point spec: configurable SPH/SOP, a genuinely different per-window label
rule (not a label swap on shared infrastructure), updated fold logic, fresh
class weights, and kept fully separate from detection everywhere it
matters (fold logic, hyperparameters, output paths, evaluation).

---

## Part 1 — SPH/SOP labeling, kept as a separate paradigm

[paradigms/continuous_labeling.py](../../../paradigms/continuous_labeling.py)
gained `label_mode="detection"|"prediction"` plus `sph` (default 300s),
`sop` (1800s), `postictal_buffer` (1800s), `interictal_buffer` (defaults to
`postictal_buffer`). `_label_windows_prediction` implements a 3-way rule
per window, combined across every seizure in a recording (excluded wins
over positive, e.g. for clustered seizures):

- **positive**: overlaps `[onset - sph - sop, onset - sph)`
- **excluded** (dropped, not counted either class): overlaps the SPH warn
  zone `[onset - sph, onset)`, the ictal span, or the postictal buffer
- **negative**: everything else, provided it also clears
  `interictal_buffer` padding around every seizure's full
  preictal-through-postictal span

Verified against a synthetic seizure at every interval boundary (preictal
start/end, warn zone, ictal, postictal, buffer zone) before trusting it on
real data.

[Epilepsy/run_pipelines.py](../../run_pipelines.py) got a parallel,
**separate** `leave_one_seizure_out_prediction` (not the detection loop
reused with an if/else) — folds are keyed by seizure (`seizure_id` in
metadata), with an explicit train-set exclusion by `seizure_id` on top of
the recording-level split, so "no part of the held-out seizure's preictal
lead-up leaks into training" is an enforced invariant rather than an
accident of today's one-seizure-per-recording data. Adds event-level
metrics (per-seizure hit/miss, false-alarms-per-hour) and a per-seizure
outcome log (duration, hit/miss, scores, same-recording seizure-clustering
gap) on top of the window-level metrics both modes share.
`PREDICTION_GRU_PARAMS`/`DEFAULT_PREDICTION_EPOCHS` are their own
hyperparameter block (currently a straight copy of detection's tuned
numbers as a starting point, but structurally decoupled — tuning one can't
silently move the other). Output goes to `results/prediction/`, separate
from detection's `results/*.csv`, with a new `results/README.md` warning
against pooling the two.

Class weights needed no code change — `common.py`'s `_criterion` already
recomputes them fresh from whatever `y` is passed to `fit()` every call;
verified this was already true rather than assuming it.

---

## Part 2 — Two real design problems found before anything could run

**Problem 1: under these exact SPH/SOP/postictal defaults, a
seizure-containing recording can never supply its own negative window.**
The minimum always-excluded span around one seizure
(`sph + sop + postictal_buffer` ≈ 3900s+) exceeds a CHB-MIT recording's own
length (~3600s) — confirmed against chb01's real documented seizure
timings, not assumed. `_build_windowed_dataset` previously loaded only
`list_seizure_records`'s 7 seizure-containing files (detection mode's
scope); for prediction mode this meant **zero negative windows in the
entire dataset**. User chose (of three options) to load the full subject
(~40 recordings) for prediction mode instead of shrinking SPH/SOP or
bounding to adjacent files.

**Problem 2 (found immediately after fixing #1): even with full-subject
data, the fold boundary itself needed to change.** Detection-style
"test = the held-out seizure's own recording" gives **zero interictal test
windows every fold** under the same constraint above — a seizure recording
still can't supply its own negatives, even as a test set. Fix: each
held-out seizure's test set now also gets an explicit, round-robin,
disjoint slice of the genuinely seizure-free recordings, so every fold has
real interictal test data for `false_alarms_per_hour` to be computed
against.

---

## Part 3 — PhysioNet is throttled; found and switched to their S3 mirror

Downloading the ~33 extra recordings for prediction mode measured
~180KB/s from `physionet.org` — confirmed **not** a local network issue
(same connection hit 3.4MB/s against a CDN, and the user's 5G hotspot
measured the same ~170KB/s against physionet.org specifically). PhysioNet
publishes this exact dataset on a public, no-auth S3 bucket
(`s3://physionet-open/chbmit/1.0.0/`, linked as an official bulk-download
option on their own dataset page) — verified byte-identical (sha256)
against an already-downloaded file before switching.
[datasets/epilepsy/chb_mit.py](../../../datasets/epilepsy/chb_mit.py)'s
`BASE_URL` now points at the S3 mirror: **~3.9MB/s, ~20x faster**, no
credentials needed.

Also added (before finding the mirror, kept afterward as a real feature):
`--max-interictal-recordings` on
[run_pipelines.py](../../run_pipelines.py), capping how many seizure-free
recordings prediction mode loads (evenly spaced through the subject's
recording order, not just the first N). `--smoke` defaults to
`SMOKE_MAX_INTERICTAL_RECORDINGS=5` unless overridden. Not load-bearing
now that the mirror is fast, but keeps `--smoke` bounded regardless of
network conditions going forward.

---

## Part 4 — Results so far (all still `--smoke`, epochs=2 — wiring/shape checks, not real training)

**Bounded run** (12 files: 7 seizure + 5 evenly-spaced interictal,
step_size=30s):

| | detection | prediction |
|---|---|---|
| roc_auc | 0.989 | 0.905 |
| average_precision | 0.946 | 0.870 |
| f1 | 0.316 | 0.767 |
| event-level hit rate | n/a | 7/7 |

**Full-subject run** (all 42 files, step_size=30s — still smoke's coarse
step, real full-subject scope): mean roc_auc 0.925, average_precision
0.440, false_alarms_per_hour 66.6 (mean), hit rate 7/7. Per-fold, every
seizure's `average_precision` beat its own chance baseline (chance =
positive-class prevalence in that fold, NOT 0.5 — this metric's chance
level moves with class balance) by 2x–23x:

| seizure | prevalence | chance AP | actual AP | × chance | FAR/h |
|---|---|---|---|---|---|
| 1_03_0 | 9.5% | 0.095 | 0.689 | 7.2x | 19.0 |
| 1_04_0 | 6.1% | 0.061 | 0.525 | 8.6x | 9.0 |
| 1_15_0 | 7.4% | 0.074 | 0.549 | 7.4x | 27.0 |
| 1_16_0 | 3.8% | 0.038 | 0.878 | 22.8x | 3.0 |
| 1_18_0 | 7.4% | 0.074 | 0.153 | 2.1x | 73.5 |
| 1_21_0 | 0.2% | 0.002 | 0.030 | (n=1, ignore) | 262.8 |
| 1_26_0 | 8.0% | 0.080 | 0.252 | 3.2x | 72.0 |

Read with real caution, not as a validated result:
- Still 2 epochs — training-loss trajectory within just epoch 1→2 already
  showed large jumps (e.g. one fold's ROC-AUC 0.51→0.88 in a single epoch),
  consistent with a model still climbing steeply, not one that's
  plateaued. Direction of smoke's bias on the ranking metrics is more
  plausibly "deflating" than "inflating," but unconfirmed.
- `false_alarms_per_hour` (mean 66.6/hour) looks like a threshold-
  calibration problem more than a ranking-signal problem —
  `use_class_weights=True` plausibly biases early training toward
  over-predicting the rare positive class before real discrimination sets
  in; the gap between "ranking is consistently above chance" (threshold-
  independent) and "default-0.5-threshold FAR is terrible"
  (threshold-dependent) is the same distinction that shows up as low F1 /
  high average_precision in detection mode's own numbers.
- Single subject, patient-specific by construction (every fold trains on
  this same child's other seizures) — real risk the model is picking up
  recording/session-specific correlates rather than genuine preictal
  physiology; not ruled out yet. No chance-baseline statistical
  significance test built (field-standard practice would compare hit rate
  against a random alarm generator at matched FAR/h, not read hit rate or
  AP in isolation).
- `1_21_0`'s near-empty preictal window (only 1 test window survived) is a
  structural artifact of that seizure's very early onset (327s into its
  recording) leaving almost no room for the full SPH+SOP lead time —
  expected given the real data, not a bug.

A **true real run** (full subject, step_size=8 default, real epoch count)
was estimated at ~3-3.5 hours and ~35GB of fresh cache (step_size=8's
windows barely overlap step_size=30's cached ones — different start times,
different cache keys) — not run yet this session; see Open items.

---

## Part 5 — Disk cleanup

Freed **~9GB** from `~/mne_data` (33GB → 24GB); system free space went
41GiB → ~49GiB ahead of that ~35GB real run:
- Deleted `~/mne_data/surrogate_null_cache` (3.5GB) — BCI pipeline's
  surrogate-calibration cache, unrelated to this task.
- Deleted ~198 other unrelated items from `~/mne_data`: `MNE-sample-data`
  (2.7GB), `MNE-bnci-data` (1.5GB), `MNE-fsaverage-data` (761MB),
  `MNE-eegbci-data` (83MB), and a large pile of small named BCI
  experiment-log directories (`canonical-*`, `seedsweep-*`, `manual-*`,
  etc.) — all confirmed disposable by the user first. Kept exactly 7 items
  under `~/mne_data`: `cwt_window_cache`, `dense_edge_cache`,
  `MNE-chbmit-data`, `dense_conv_feature_cache`, `results`,
  `run_ledger.csv`, `run_ledger.csv.lock`.
- **Near-miss worth flagging for next time**: the first deletion attempt
  used `for k in $KEEP` with an unquoted multi-word shell variable to
  check list membership. zsh (this environment's shell) doesn't
  word-split unquoted scalar variables the way bash does, so the
  membership check silently never matched anything — had the accompanying
  `rm -rf` not separately failed (on an unrelated empty-argument bug), it
  would have deleted the entire `~/mne_data` directory, including the
  caches and results meant to be kept. Caught by inspecting the "to
  delete" list before running `rm`, not by the logic being correct.
  Redone with a portable `case` statement instead of list-membership
  looping.

---

## Current state

- Both label modes wired, [paradigms/continuous_labeling.py](../../../paradigms/continuous_labeling.py)
  and [Epilepsy/run_pipelines.py](../../run_pipelines.py). Detection mode
  re-verified unchanged (`--smoke --label-mode detection` matches prior
  behavior).
- `datasets/epilepsy/chb_mit.py`'s `BASE_URL` now points at PhysioNet's S3
  mirror (~20x faster than their HTTPS server).
- `results/README.md` exists, states detection/prediction scores aren't
  comparable.
- Only `--smoke` (2 epochs) results exist for prediction mode so far — no
  real (full epoch count, step_size=8) run yet.
- `~/mne_data` cleaned to ~24GB (was ~33GB), ~49GiB free on disk.

## Open items

- **The real prediction run** (step_size=8, real epoch count — 10 was
  discussed, `DEFAULT_PREDICTION_EPOCHS=20` is the current unvalidated
  default) hasn't been run. Estimated ~3-3.5 hours, ~35GB fresh cache;
  disk now has room (~49GiB free) but this is still an estimate, not
  measured.
- `PREDICTION_GRU_PARAMS`/`DEFAULT_PREDICTION_EPOCHS` are still a straight
  copy of detection's tuned numbers — real tuning for prediction mode not
  started.
- No chance-baseline statistical significance test built (e.g. hit rate
  vs. a random alarm generator matched to the same FAR/h) — needed before
  any prediction result can be called validated rather than suggestive.
- False-alarm rate at the default 0.5 threshold looks bad across smoke
  folds; threshold calibration (not necessarily retraining) flagged as a
  plausible cheap lever, not yet tried.
- Single-subject risk (recording/session-specific confounds vs. genuine
  preictal signal) not ruled out — would need cross-subject testing to
  address, out of scope so far (same deliberate scope limit as detection
  mode: chb01/subject 1 only).
