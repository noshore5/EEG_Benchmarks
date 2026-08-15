# Session notes — CHB-MIT dataset/paradigm + Epilepsy dense-edge-GRU pipeline (2026-08-15)

Branch: `main`. Repo: `/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`.

Goal for the session: get one epilepsy dataset (CHB-MIT) loading end-to-end
through a dataset/paradigm interface adapted from MOABB's pattern, then (a
follow-on ask mid-session) stand up a dedicated `Epilepsy/` copy of the
`dense_edge_gru` pipeline (`BCI/run_pipelines.py`'s
`SparseEvidenceGNNClassifier`, `event_mode="dense"`,
`dense_edge_temporal_mode="rnn"`) against it. Both done; see "Open items"
for what's left before this is a real, tuned result rather than a wiring
proof.

---

## Part 1 — `datasets/epilepsy/chb_mit.py` + `paradigms/continuous_labeling.py`

**Environment correction first:** initially assumed `moabb` wasn't
installed (only the system `python3` was checked) and built a fully
standalone `BaseDataset`. Once corrected — the project's actual interpreter
is `/Users/noahshore/Documents/CoherIQs/CMPX_EEG/CMPX/bin/python` (set in
`.vscode/settings.json`), which has `moabb` 1.4.3 editable-installed from
the local fork at `/Users/noahshore/Documents/CoherIQs/moabb` — reworked
`CHBMIT` to subclass the real `moabb.datasets.base.BaseDataset`. All
subsequent work (including everything in Part 2) runs under that CMPX
interpreter, not system `python3`.

**Real bug found along the way:** `BaseDataset`'s default
`_create_process_pipeline()` runs `SetRawAnnotations`, which rewrites
*every* annotation's duration to one fixed `interval[1] - interval[0]`
span — correct for MOABB's fixed-length trials, but it would have silently
overwritten each seizure's real (variable, 27-95s) duration with a
constant if left in place.
[chb_mit.py](../../../datasets/epilepsy/chb_mit.py)'s `CHBMIT` overrides
`_create_process_pipeline()` to a no-op `FixedPipeline([])` so
`_get_single_subject_data`'s real `mne.Annotations` pass through
untouched. This is the reason `paradigms/continuous_labeling.py`'s
`ContinuousLabelingParadigm` does **not** subclass
`moabb.paradigms.base.BaseProcessing` either — that pipeline
(`FixedPipeline`/`StepType`/`RawToEvents`/`SetRawAnnotations`) is built
entirely around a stim channel firing discrete, fixed-length trials, which
CHB-MIT's sparse, variable-length seizure spans don't fit without the same
distortion.

**What `CHBMIT` does** ([chb_mit.py](../../../datasets/epilepsy/chb_mit.py)):
downloads `chbXX-summary.txt` + requested `.edf` files via
`moabb.datasets.download`, parses the summary with a regex-based
`parse_summary()` (handles both the single-seizure `Seizure Start Time:`
format and the numbered `Seizure N Start Time:` format subjects with
multiple seizures/file use — verified against real chb01 and chb15 files),
and attaches seizure spans as `mne.Annotations` in
`_get_single_subject_data`. Constructor takes an optional
`records={subject: [filenames]}` filter — used throughout to restrict to
just the seizure-containing recordings via `list_seizure_records()`,
avoiding downloading all ~40 recordings/subject when only a handful are
needed.

**What `ContinuousLabelingParadigm` does**
([continuous_labeling.py](../../../paradigms/continuous_labeling.py)):
slides a fixed `window_length`/`step_size` window across each `Raw`,
labels each window ictal (`1`) if it overlaps a `label_event` annotation
(default `"seizure"`) by more than `min_overlap` seconds, else interictal
(`0`). Returns plain `(X, y, metadata)` arrays — `X` shape `(n_windows,
n_channels, n_samples)`, scaled by `dataset.unit_factor` (µV, matching
MOABB paradigm convention) — not `mne.Epochs`, since there's no trial to
epoch around. `metadata` carries `subject`/`session`/`run`/`window_start`/
`window_end` per window, which Part 2's leave-one-seizure-out split uses
directly (`run` = one CHB-MIT recording = one group).

**Test** ([test_chb_mit_continuous_labeling.py](../../../tests/test_chb_mit_continuous_labeling.py),
pytest, marked `slow`): downloads chb01's 7 seizure recordings (~300MB,
cached in `~/mne_data` after first run), windows at a deliberately coarse
`step_size=15s` (memory/runtime bound for a 7×1-hour-file sanity check, not
a modeling choice), and asserts exactly 7 contiguous ictal window clusters
land within `window_length + step_size` (19s) of each documented
onset/offset. **All 7 passed** — onset errors 2-8s, offset errors 2-11s (pure
quantization from the 15s step grid, not detection error — the labeling
rule is a deterministic overlap check against the same ground truth it's
being validated against, not a prediction).

---

## Part 2 — `Epilepsy/` dense-edge-GRU pipeline

Asked to duplicate `BCI/run_pipelines.py`'s `dense_edge_gru` pipeline into
`Epilepsy/`. Two things that made this not a straight copy-and-rename:

1. **Evaluation engine.** `dense_edge_gru` normally runs through
   `moabb.evaluations.CrossSessionEvaluation`/`CrossSubjectEvaluation`,
   which call `paradigm.get_data(dataset, subjects)` expecting a real
   `BaseParadigm`. `ContinuousLabelingParadigm` deliberately isn't one (see
   Part 1), and `CHBMIT` only has one session/subject anyway, so there's
   nothing for `CrossSessionEvaluation` to cross-validate across. Built a
   small custom `leave_one_seizure_out()` in
   [run_pipelines.py](../../run_pipelines.py) instead: one fold per
   `(subject, run)` group, train on the rest, score
   precision/recall/F1/average-precision/ROC-AUC (not plain accuracy —
   windows are ~2% ictal).
2. **Standalone classifier fork.** `sparse_evidence_gnn_classifier.py`
   (4232 lines) is under active concurrent development in `BCI/` per its
   own docstrings — a real copy starts diverging the moment BCI's version
   next changes, but was explicitly asked for over an import-based reuse.
   Copied verbatim: `Epilepsy/pipelines/{common,xwt_phase_gnn_classifier,
   sparse_evidence_gnn_classifier}.py` (7607 lines) +
   `Epilepsy/experiment_logging.py` (322 lines, so the fork has zero import
   dependency on `BCI/`). Internal `from BCI.moabb_pipelines...` imports
   repointed to `Epilepsy.pipelines...` (kept the existing
   try/`ModuleNotFoundError`-fallback-to-bare-module-name pattern each file
   already used).

### Real changes made to the fork (not just a rename)

- **`use_class_weights`** was hardcoded `False` in
  `_BaseCWTGNNClassifier._init_cwt_gnn_classifier`'s call to
  `_init_torch_classifier` (fine for motor imagery's balanced trials,
  wrong for ~2% ictal). `TorchEEGClassifier._criterion` (common.py:1439)
  already implements inverse-class-frequency weighting — just needed to
  stop being discarded. Exposed as a real constructor param, default
  `True`, threaded through `SparseEvidenceGNNClassifier.__init__` →
  `_init_cwt_gnn_classifier`. Confirmed working in the first real run:
  `[Train] class weights: [0.044, 1.956]`.
- **Defaults**: `sampling_rate=250, lowest=8.0, highest=35.0` (BNCI2014_001
  native rate + mu/beta motor-imagery band) → `sampling_rate=256, lowest=1.0,
  highest=40.0` (CHB-MIT's native rate; 1-40Hz is an explicitly
  **not-tuned** broad placeholder, flagged as such in the docstring —
  picking a real epilepsy-appropriate band is unstarted work).
- `channel_subset` defaults to `None` (all 23 channels) in
  `DENSE_EDGE_GRU_PARAMS` — BNCI's 9-channel motor-cortex index list
  (`[1, 5, 7, 8, 9, 10, 11, 13, 17]`) is meaningless for CHB-MIT's bipolar
  10-20 montage.

### Bugs caught by actually running it, not just reading it

- **Fold mask silently wrong.** First `leave_one_seizure_out` built
  `groups_arr = np.array(list_of_(subject,run)_tuples, dtype=object)` —
  `np.array()` on same-length tuples builds a genuine 2D array, not an
  array of tuple objects, so `g == group` compared per-column instead of
  per-row and `test_mask` came out shaped `(n_windows, 2)` instead of
  `(n_windows,)`. Caught by an `IndexError` on the very first smoke run,
  not by inspection. Fixed by comparing the plain Python list of tuples
  directly (tuple `==` is exact), not a numpy array of them.
- **Redundant CWT pass.** `leave_one_seizure_out` originally called both
  `clf.predict(X_test)` and `clf.predict_proba(X_test)` — each
  independently calls `_predict_logits()` → `_prepare_features()`, so the
  test set's CWT got computed twice per fold for no reason (`predict`'s
  result is fully recoverable from `predict_proba`'s via argmax over
  `clf.classes_`). Fixed to call `predict_proba` once. Measured: 22:00 →
  20:39 for the 7-fold `--smoke` run (798 windows, 2 epochs/fold) — modest,
  as expected from removing one of three redundant CWT passes.

### CWT caching — built, verified exact, kept despite measuring net-negative

Leave-one-seizure-out's folds mostly overlap (each excludes only one of 7
recordings), so investigated caching CWT output across fold-local
classifier instances to avoid recomputing the wavelet transform for
windows already transformed in an earlier fold.

**The catch**: each fold's `_prepare_features` z-score-normalizes with
that fold's *own* global scalar `(mean, std)` (`common.py`'s
`fit_global_zscore_stats`, computed fresh per `fit()` call from that fold's
training data) before running CWT — so the literal bytes fed to
`transform_fn` differ slightly fold-to-fold even for the identical
physical window, defeating naive content-addressed caching.

**Verified empirically before trusting it** (matching this codebase's own
"validated via debug_plots before being wired in" precedent): fcwt's
wavelets are admissible/zero-mean, so `CWT((x - mean) / std) == CWT(x) /
std` to within float32 noise (~2.3e-7 max relative error, measured against
two different synthetic `(mean, std)` pairs, including at the trial edges
— not just the COI-safe center). This makes caching `CWT(raw window)` once
and rescaling by `1/std` on retrieval **exact**, not approximate, for the
mean-subtraction discards. Built
[cwt_window_cache.py](../../pipelines/cwt_window_cache.py) on that basis:
content-addressed (SHA256 of the raw window's bytes + CWT config), stores
raw `(w_real, w_imag)`, rescales by `1/(std+1e-8)` on retrieval. Wired into
`_BaseCWTGNNClassifier._prepare_features` via a new `cwt_cache: dict | None`
constructor param (`None` default = private per-instance cache, i.e. no
behavior change unless the caller shares one) —
`leave_one_seizure_out` builds one shared dict and passes it to every
fold's classifier.

Unit-verified against the original uncached `compute_cwt_real_imag_tensors`
with two different `(mean, std)` pairs applied to the same synthetic
window: max relative diff ~2.4e-7 (float32 noise floor) on `w_real`/
`w_imag`, exact on `raw_x`/`freqs`. Re-running with a pre-warmed cache
reproduced a fresh reference computation to the same tolerance.

**Measured on the real `--smoke` run: made it slower, not faster** — 20:39
(no cache) → 25:59 (with cache), despite a genuine 100% cache-hit rate from
fold 2 onward (`CWT cache: 17556 unique (window, channel) transforms
cached`). Root cause, found by comparing epoch times (rose from ~11-14s to
~17-18s) against a 16GB-RAM machine already under load: the ~2.4-2.8GB the
cache holds in memory is enough to cause pressure (`ps` showed the process
in uninterruptible-sleep/I/O-wait state near the end of the run), and CWT
was never the dominant cost to begin with.

**The actual dominant cost, found while investigating why caching CWT
didn't help**: `SparseEvidenceGNNClassifier._precompute_dense_edge_inputs`
(coherence/phase/significance construction from the CWT tensors, called
once per `fit()`/`predict()` for `event_mode="dense"`) has **no progress
bar in `coherence_threshold_mode="fixed"`** (`show_progress = (surrogate_mode
or cluster_mode) and self.verbose >= 1` — always `False` for our config),
so it was invisible while watching CWT's own (cached, now-fast) bars. Its
sparse-mode analog is documented elsewhere in this same file as "94.8% of
forward()'s time" before a prior optimization — strongly suggesting this,
not CWT, is the real per-fold-repeated bottleneck.

Traced whether this is cacheable too:
`compute_dense_edge_input`/`_build_dense_edge_input`
([sparse_evidence_gnn_classifier.py:2246](../../pipelines/sparse_evidence_gnn_classifier.py#L2246))
returns `[coherence, sin(phase), cos(phase), significance]` — coherence is
an exact ratio (`|S12|/√(S1·S2)`, the `(1/std)²` scale factors cancel
algebraically, not just approximately), phase is an angle (invariant to
scaling by any positive real), and significance is `(coh - threshold) /
threshold` with `threshold` a **fixed config constant** in
`coherence_threshold_mode="fixed"` — so unlike CWT, this stage is exactly
scale-invariant and wouldn't even need the `1/std` rescale trick. **Not
implemented**: with `channel_subset=None` (23 channels → `C(23,2)=253`
edges, vs. the 36-edge/9-channel case the code's own "~9.2MB/trial"
estimate assumes) and `dense_edge_time_downsample=8`, a full cache would be
~7.9MB/window — **~6.3GB for 798 windows**, on top of the CWT cache's
~2.4GB, on a 16GB machine that already showed memory-pressure symptoms from
the smaller cache alone. Flagged this tradeoff and asked rather than guess;
decision was to keep the (measured net-negative, but correct) CWT cache and
not extend further, rather than revert or build a size-bounded/LRU version.

---

## Current state

- [datasets/epilepsy/chb_mit.py](../../../datasets/epilepsy/chb_mit.py),
  [paradigms/continuous_labeling.py](../../../paradigms/continuous_labeling.py) —
  working, tested against real chb01 data.
- [Epilepsy/pipelines/](../../pipelines/) — standalone fork, `use_class_weights`
  fixed, CHB-MIT-native defaults, CWT cache in place (opt-in, currently
  net-neutral-to-slightly-negative at smoke scale on this machine).
- [Epilepsy/run_pipelines.py](../../run_pipelines.py) — `--smoke` (798
  windows, step_size=30s, epochs=2) runs end-to-end in ~21-26 min depending
  on caching config; a real run (`--epochs 30`, finer `--step-size`,
  default is 4.0s) has **not** been executed and will take considerably
  longer.

## Open items

- Pick a real `window_length`/`step_size` (currently 4.0s/4.0s defaults,
  explicitly untuned) and frequency band (currently 1-40Hz, explicitly
  untuned) instead of placeholders.
- `_precompute_dense_edge_inputs` is the actual bottleneck; not cached.
  A size-bounded (LRU, capped memory) cache was proposed but not built.
- Only subject 1 exercised. Looping all 24 subjects, and TUSZ/Siena/Bonn,
  are deliberately out of scope per the original task ("stop there... once
  this works for one subject, one dataset, we'll templatize it").
- No real (non-`--smoke`) leave-one-seizure-out run has been scored yet —
  everything reported here is wiring verification, not a model-quality
  result.
