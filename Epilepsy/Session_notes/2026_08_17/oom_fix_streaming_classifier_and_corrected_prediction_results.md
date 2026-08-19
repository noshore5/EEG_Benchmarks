# Session notes — OOM fix, streaming classifier, and the first trustworthy prediction-mode results (2026-08-17)

Branch: `main`. Repo: `/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`.

Follow-on to [2026-08-16's session](../2026_08_16/prediction_mode_sph_sop_implementation_and_first_results.md),
which left prediction mode wired end-to-end but only smoke-tested (2 epochs,
never a real run). Also lowered `DEFAULT_SOP` from 1800s to 900s (15min)
before this session started, based on chb01's real seizure timings
truncating 6/7 preictal windows at 30min SOP vs. 2/7 at 15min. This session:
ran the first real (non-smoke) prediction run, hit an OOM crash, fixed it
two independent ways, got a real run to complete, then found — via a user
question that didn't accept the numbers at face value — that the fix itself
had silently corrupted the evaluation, fixed that too, and reran for the
numbers that actually stand.

---

## Part 1 — First real run: OOM kill (exit 137)

`--label-mode prediction --subjects 1 --epochs 10` (no `--smoke`, real
`step_size=8`) crashed. Diagnosed via `vm_stat`/`sysctl vm.swapusage`
(12.6GB swap in use on a 16GB+15GB-swap machine) and by measuring actual
cache-entry sizes directly: `dense_edge_cache` ≈2.04MB/window uncompressed,
`cwt_window_cache` ≈1.51MB/window (23 channels combined) — ≈3.55MB/window
combined. `common.py`'s `fit()` calls `_prepare_features(X, fit=True,
train_idx=train_idx)` **once for the whole training set** before the epoch
loop starts (confirmed by reading `common.py:1499-1500`). At prediction
mode's scale (~14,700 training windows/fold, since a seizure-containing
recording can never supply its own negatives — see 2026-08-16's note —
so the negative pool is the *entire rest of the subject*), that's
≈52GB materialized in memory before training even begins. Not a leak,
not a bug in the model — a real architectural mismatch between
"materialize everything up front" (fine at detection's ~700-window scale)
and prediction's real dataset size.

User's instruction: fix both root causes at once — **shrink the training
set** and **stop materializing it all at once** — with the second one built
as a genuinely separate class so it can't touch detection's already-tuned
path.

---

## Part 2 — Two independent, separately-verified fixes

**Fix 1: stratified negative-window subsampling.**
[Epilepsy/run_pipelines.py](../../run_pipelines.py)'s new
`_subsample_negative_windows` randomly drops negative windows down to
`DEFAULT_NEGATIVE_TO_POSITIVE_RATIO=5.0` × positive count, stratified per
`(subject, run)` so every recording keeps a proportional share — avoids
accidentally zeroing out a recording that
`leave_one_seizure_out_prediction`'s round-robin interictal-fold assignment
depends on. All positive (preictal) windows always kept. Not applied under
`--smoke` (already small via file-count capping).

**Fix 2: `StreamingSparseEvidenceGNNClassifier`.**
Added to [Epilepsy/pipelines/cwt_gnn_classifiers.py](../../pipelines/cwt_gnn_classifiers.py)
(same file as `SparseEvidenceGNNClassifier`, not a new module — user
explicitly asked for this after an initial separate-file version).
Subclasses `SparseEvidenceGNNClassifier`, overrides only `fit()`: computes
`_prepare_features` **per batch** via a `_LazyFeatureBatchDataset` +
`_BatchIndexSampler`, instead of once for the whole training set — O(batch
size) memory instead of O(dataset size). Reuses the shared, unmodified
`_train_loop`. `leave_one_seizure_out_detection` still instantiates the
original `SparseEvidenceGNNClassifier`, untouched.

**Verification** (script in scratchpad, not committed): bit-exact
reproduction of the original class's training trajectory turned out not to
be achievable — `DataLoader(shuffle=True, generator=g)` draws an extra
`_base_seed` from the shared generator before the `RandomSampler` uses it
(traced via `inspect.getsource` to `_BaseDataLoaderIter.__init__`), shifting
its subsequent output vs. a manually-constructed sampler on the same
generator. Reframed to what actually matters instead: (1) every training
index visited exactly once per epoch (completeness, verified exhaustively),
(2) per-batch `_prepare_features` output is numerically **identical** to
whole-set `_prepare_features` output for the same windows (0.0 abs diff,
verified), (3) end-to-end fit/predict sanity (loss decreases, valid
probabilities). Documented in the class's header comment so the "why isn't
this bit-exact" question doesn't need re-investigating later.

Both fixes verified together at `--smoke` scale before the real rerun.

---

## Part 3 — First completed real run (results later found to be confounded — see Part 4)

`--label-mode prediction --subjects 1 --epochs 10`, ~1h32m, exit 0, all 7
folds, no crash. Mean roc_auc 0.868, average_precision 0.552,
false_alarms_per_hour 69.0, hit rate 6/7. Looked strong enough (auc_pr 0.83
on the best fold) to prompt the user to push back rather than accept it:
*"I'm finding these numbers suspicious in how well it learns — this is like
my first trial run on a task that existing SOTA models don't do well on."*

That pushback led to re-reading `leave_one_seizure_out_prediction` and
`_build_windowed_dataset` end-to-end rather than trusting the earlier
smoke-scale verification, which surfaced a real bug:

**`_subsample_negative_windows` (Fix 1 above) was being applied once,
globally, before the fold split** — `main()` called it inside
`_build_windowed_dataset`, and the already-subsampled `X, y, metadata` were
what `leave_one_seizure_out_prediction` sliced into train/test. Every
fold's **test** interictal windows were therefore also thinned to the same
~1/5 ratio as train — `false_alarms_per_hour` and the window-level metrics
were being computed against a random ~21% slice of each recording, not
continuous monitoring. No train/test leakage (fold construction itself —
recording exclusion, explicit `seizure_id` exclusion, z-score stats fit on
train only — checked out clean), but a real reliability problem: small,
single-seed, non-representative sample sizes for the metrics that matter
most for a prediction system's real-world usefulness.

Direction wasn't obvious in advance and turned out to not be systematic —
FAR/h moved in **both** directions per-fold after the fix (down for seizure
03, up for seizure 04) — consistent with subsampling being unbiased noise,
not a directional inflation.

---

## Part 4 — The fix: subsample train only, per fold

Moved the subsampling call out of `_build_windowed_dataset` (which now
always returns the full, unsampled dataset) and into
`leave_one_seizure_out_prediction`, applied to each fold's `train_mask`
slice only, right after computing `train_mask`/`test_mask` and before
`clf.fit()`. Test windows (`X_test`/`y_test`) are now never touched by
subsampling, at any ratio.

Verified two ways at `--smoke` scale before rerunning for real:
- `--negative-to-positive-ratio 5.0` (today's real default): print
  statement never fired — smoke's dataset is already small enough that
  `target_negative >= len(negative_idx)` per fold, so nothing to trim.
  Expected, not a bug.
- `--negative-to-positive-ratio 0.5` (forced to actually engage): print
  fired **once per fold** (`480 -> 72`, `480 -> 76`, `600 -> 85`, etc. —
  different per-fold counts, confirming it's operating on each fold's
  train-only pool, not a shared global one), while every fold's `n_test`
  (150, 150, 150, 144, 150, 1, 30) matched exactly what the unsubsampled
  baseline produced — confirming test windows are untouched regardless of
  ratio.

---

## Part 5 — Second real run: the numbers that actually stand

Same command, ~1h53m (longer than the first — expected, since the
previously-dropped ~80% of each fold's interictal windows needed fresh
CWT/dense-edge cache entries the first time; train-side cost was unchanged
since train-side subsampling scope/ratio didn't change). Exit 0, all 7
folds, no crash.

| seizure | preictal | n_test (full) | hit | recall | precision | auc_pr | FAR/hr |
|---|---|---|---|---|---|---|---|
| 03 | 112 | 2245 | ✅ | 0.634 | 0.504 | 0.520 | 29.5 |
| 04 | 113 | 2363 | ✅ | 0.850 | 0.282 | 0.295 | 97.6 |
| 15 | 112 | 2362 | ✅ | 0.714 | 0.584 | 0.677 | 22.8 |
| 16 | 89 | 2339 | ✅ | 0.742 | 0.957 | **0.956** | **1.2** |
| 18 | 113 | 2363 | ✅ | 0.469 | 0.186 | 0.187 | 92.8 |
| 21 | 3 | 1878 | ❌ | 0.000 | 0.000 | 0.003 | 122.9 |
| 26 | 112 | 2362 | ✅ | 0.179 | 0.147 | 0.124 | 46.4 |

Mean: accuracy 0.920, precision 0.380, recall 0.512, f1 0.413,
**average_precision 0.395**, **roc_auc 0.882**, false_alarms_per_hour 59.0,
**hit rate 6/7 (85.7%)**.

vs. the confounded run: roc_auc held up (0.868 → 0.882) — the ranking
signal was real, not an artifact of the eval bug. average_precision dropped
substantially (0.552 → 0.395) now that it's scored against the true ~20x
larger interictal population per fold. Seizure 16 is the clear standout
both before and after the fix (auc_pr 0.956, FAR/h 1.2 — the one fold in
genuinely clinically-interesting territory); seizures 04, 18, 26 are
weak-to-near-chance (auc_pr 0.12–0.30). The one miss (`1_21_0`) is the same
structural artifact flagged on 2026-08-16 — only 3 preictal windows survive
labeling because the seizure's own early onset (327s into its recording)
leaves almost no room for the full SPH+SOP lead time before the recording
starts; not a modeling failure.

Discussed with the user why FAR/h being high (up to 97.6/hour, ≈1 false
alarm every 37s) doesn't by itself mean there's no signal:
`false_alarms_per_hour`/precision/recall are read off one fixed, untuned
decision threshold (argmax at the `use_class_weights`-skewed 0.5 boundary),
while `roc_auc`/`average_precision` are threshold-independent rank
statistics. For seizure 04's actual class balance (113/2363 ≈ 4.8%
positive), a zero-skill model would score auc_pr≈0.048; the observed 0.295
is ~6x that floor — real signal, just poorly exploited by the current
uncalibrated threshold, not absent.

---

## Part 6 — Label-permutation null control (addresses "is this real?")

After the results in Part 5, the user pushed back again -- reasonably, per
a specific external critique relayed mid-session -- with two concrete
leakage-risk claims (fold-boundary exclusion, SPH warn-zone boundary).
Both were verified directly against the real code and real data (see the
companion note,
[fold_boundary_and_sph_boundary_verification.md](fold_boundary_and_sph_boundary_verification.md))
and found clean. That rules out two specific failure modes but doesn't by
itself answer "is there real signal" -- for that, added a proper
label-permutation null control: `--shuffle-labels` (new CLI flag) shuffles
`y` globally (class balance preserved exactly, positive/negative
assignment becomes pure noise) right after dataset construction, then runs
the *identical* downstream pipeline -- same fold construction, same
per-fold train-only subsampling, same classifier, same everything except
which windows are labeled positive. Writes to a separate
`results/prediction_shuffled_control/` directory (and
`results/shuffled_control/` for detection mode) so a null-control run can
never be mistaken for a real one. `metadata['seizure_id']` is deliberately
left untouched by the shuffle, so event-level hit/FAR bookkeeping still
refers to the true seizure windows -- expected to be noisy/uninformative
under shuffled labels, not a clean signal; the window-level
`roc_auc`/`average_precision` means are the real comparison.

Verified the flag's wiring at `--smoke` scale first (roc_auc collapsed to
0.531, right at chance) before running it for real, matched exactly to
Part 5's real-run config (`--label-mode prediction --subjects 1 --epochs
10`). Ran fast (~1h vs. Part 5's ~1h53m) since the CWT/dense-edge caches
are keyed by raw window content, independent of labels -- fully cache-hot.

| seizure | real auc_pr | null auc_pr | ratio |
|---|---|---|---|
| 03 | 0.520 | 0.044 | 11.8x |
| 04 | 0.295 | 0.042 | 7.0x |
| 15 | 0.677 | 0.046 | 14.7x |
| 16 | 0.956 | 0.052 | 18.4x |
| 18 | 0.187 | 0.037 | 5.1x |
| 21 | 0.003 | 0.043 | n/a -- see below |
| 26 | 0.124 | 0.038 | 3.3x |

Mean roc_auc: real **0.882** vs. null **0.491** (textbook chance). Mean
average_precision: real **0.395** vs. null **0.043** (~9x).

Seizure 21's row isn't a real counter-example -- `n_preictal=3` printed
identically in both runs because that count is read from
`metadata['seizure_id']` (untouched by the shuffle), but the *auc_pr*
itself is computed against however many shuffled-positive windows actually
landed in that fold's test set, which is **77**, not 3 (checked directly:
a global permutation of 654 positives across 15,912 windows puts ~77 into
any ~1,878-window test slice by base rate). So "real (n=3, high-variance)
auc_pr=0.003" and "null (n=77, stable) auc_pr=0.043" aren't a fair
comparison in either direction -- both numbers are legitimate, they're just
answering different-precision questions. Every other fold's `n_preictal`
(89-113) is large enough that this issue doesn't apply.

Net: six of seven folds show a clean, consistent real-vs-null gap (3x-18x),
and the aggregate roc_auc sits almost exactly on the 0.5 chance line for
the null while the real run holds at 0.882 -- solid evidence the real
run's signal isn't a pipeline artifact, on top of (not instead of) the two
leakage mechanisms already ruled out in Part 6's companion note.

---

## Current state

- Prediction mode has a completed, non-confounded, 7-fold real run:
  `results/prediction/prediction_leave_one_seizure_out_20260817-125212.csv`
  / `prediction_per_seizure_20260817-125212.csv`.
- `StreamingSparseEvidenceGNNClassifier` lives in
  [cwt_gnn_classifiers.py](../../pipelines/cwt_gnn_classifiers.py), used
  only by `leave_one_seizure_out_prediction`; detection's
  `SparseEvidenceGNNClassifier`/`leave_one_seizure_out_detection` path is
  byte-for-byte unmodified.
- `_subsample_negative_windows` (ratio 5.0 default) now runs per-fold,
  train-only, inside `leave_one_seizure_out_prediction`;
  `_build_windowed_dataset` always returns the full unsampled dataset.
- Earlier same-day result files
  (`*_20260817-110945.csv`/`*_20260817-110236.csv` and similarly-timed
  `per_seizure` files) are the confounded/superseded runs and smoke-scale
  verification runs from this session — not deleted, but should not be
  read as trustworthy numbers going forward.
- Label-permutation null control now exists (`--shuffle-labels`,
  `--shuffle-seed`) and has a completed real run:
  `results/prediction_shuffled_control/prediction_leave_one_seizure_out_20260817-182000.csv`
  / `prediction_per_seizure_20260817-182000.csv`. Mean roc_auc 0.491
  (chance) vs. the real run's 0.882 — see Part 6.
- Two specific leakage-risk claims (fold-boundary exclusion, SPH
  warn-zone boundary) raised via external critique were checked directly
  against real code/data and found clean — see the companion note
  [fold_boundary_and_sph_boundary_verification.md](fold_boundary_and_sph_boundary_verification.md).

## Open items

- **Decision threshold is untuned.** Every FAR/h number in Part 5 is read
  off the default class-weight-skewed 0.5 cutoff. Sweeping the threshold
  (trading recall for FAR/h) is flagged as a plausible cheap lever, same as
  2026-08-16's note flagged and still not tried.
- **Single-subject scope.** Everything so far is chb01-only,
  patient-specific leave-one-seizure-out. This is the *easy* end of seizure
  prediction — the harder, SOTA-relevant problem is cross-patient
  generalization, untested here. Strong single-subject numbers (esp.
  seizure 16) aren't evidence of anything beyond "this pipeline can model
  this one subject's preictal state to some degree." The label-permutation
  control (Part 6) rules out "no real signal at all," not "this
  generalizes beyond chb01."
- ~~No chance-baseline statistical significance test built yet~~ — done in
  Part 6 via label permutation (real roc_auc 0.882 vs. null 0.491, real
  average_precision 0.395 vs. null 0.043). A stricter complementary check
  — a circular time-shift control (same signal, seizure-onset alignment
  randomized instead of labels) — was discussed but not built; would rule
  out a narrower failure mode (model keying on *some* arbitrarily-timed
  window shape in a recording rather than genuine seizure-onset proximity)
  that label-shuffling alone doesn't fully address.
- `PREDICTION_GRU_PARAMS`/`DEFAULT_PREDICTION_EPOCHS` are still a straight
  copy of detection's tuned numbers — no prediction-specific tuning done
  yet.
- The raw windowing/labeling step (`paradigm.get_data()` inside
  `_build_windowed_dataset`) is not disk-cached — every process launch
  re-loads and re-windows all ~40 of a subject's recordings from scratch
  (cheap relative to CWT/dense-edge computation, but a real gap; only the
  downstream feature caches persist across runs).
