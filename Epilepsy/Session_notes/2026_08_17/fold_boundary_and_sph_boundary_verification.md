# Session notes — empirical verification of two leakage-risk claims (2026-08-17)

Branch: `main`. Repo: `/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`.

Companion to [today's earlier note](oom_fix_streaming_classifier_and_corrected_prediction_results.md)
(OOM fix, streaming classifier, corrected real-run results) and to the
label-permutation null-control run added as a follow-on check on those
results (`--shuffle-labels`, see that note's Open Items). While the
permutation control was running, the user relayed a specific external
critique raising two concrete leakage-risk claims about the prediction
pipeline, worth checking directly against the real code and real data
rather than by re-reading source and trusting it:

1. **Does the fold-boundary exclusion actually exclude a held-out
   seizure's full preictal lead-up from every other fold's training set**
   — not just its ictal period?
2. **Is the SPH warn-zone boundary `[onset-sph, onset)` airtight** — does
   the labeling rule prevent any near-onset window from slipping into the
   positive/preictal class, even one that straddles the preictal/warn-zone
   boundary?

Both were suggested as the kind of subtle bug that would produce exactly
the pattern seen so far (surprisingly strong early results) for a fake
reason rather than a real one.

---

## Method

`verify_fold_boundary.py` (scratchpad, not committed to the repo, not
linked here since scratchpad paths are session-ephemeral) builds the real
subject-1 prediction dataset
(same `sph=300s`, `sop=900s` as the real run) and checks both claims
directly against it:

- **Claim 2** is checked by independently recomputing, from the built
  dataset's own per-window metadata (`window_start`/`window_end`/
  `seizure_onset`), whether any positive-labeled window overlaps its
  seizure's `[onset-sph, onset)` range — not by re-reading
  `_label_windows_prediction`'s source, a genuinely independent
  recomputation.
- **Claim 1** is checked by re-running `leave_one_seizure_out_prediction`'s
  own fold-construction expressions (`test_run_pairs`/`test_mask`/
  `train_mask`, copied verbatim from that function) for all 7 folds, then
  for each fold checking (a) zero training windows share the held-out
  seizure's `(subject, run)` recording, and (b) zero training windows from
  that same recording overlap the seizure's preictal time range.

**First attempt at Claim 1 was itself buggy** — an initial "belt-and-
suspenders" version compared `window_start`/`window_end` against `onset`
across the *entire* dataset regardless of recording, which produced
thousands of apparent "leaks" per fold. Root cause: `window_start`/
`window_end`/`seizure_onset` are all timestamps **relative to their own
recording's start** (see `get_data`'s docstring), not a shared/global
clock — two unrelated recordings both having windows with, say,
recording-relative `start=2000s` means nothing. Fixed by restricting the
time-overlap check to windows from the *same* recording as the held-out
seizure (which is what the check was actually meant to test); re-run
below is the corrected version. Worth remembering next time a
cross-recording time comparison seems tempting in this codebase.

---

## Results

```
=== Claim 2: SPH warn-zone boundary ===
  PASS: 0/654 positive windows overlap their seizure's SPH warn zone --
  boundary is airtight across all 7 seizures.

=== Claim 1: fold-boundary exclusion ===
  fold 0 (seizure 1_03_0, subject=1 run=03): PASS -- same-recording-in-train=0  time-overlap-leak-in-train=0
  fold 1 (seizure 1_04_0, subject=1 run=04): PASS -- same-recording-in-train=0  time-overlap-leak-in-train=0
  fold 2 (seizure 1_15_0, subject=1 run=15): PASS -- same-recording-in-train=0  time-overlap-leak-in-train=0
  fold 3 (seizure 1_16_0, subject=1 run=16): PASS -- same-recording-in-train=0  time-overlap-leak-in-train=0
  fold 4 (seizure 1_18_0, subject=1 run=18): PASS -- same-recording-in-train=0  time-overlap-leak-in-train=0
  fold 5 (seizure 1_21_0, subject=1 run=21): PASS -- same-recording-in-train=0  time-overlap-leak-in-train=0
  fold 6 (seizure 1_26_0, subject=1 run=26): PASS -- same-recording-in-train=0  time-overlap-leak-in-train=0

ALL FOLDS PASS
```

**Claim 1, confirmed true** — and the mechanism is stronger than the
minimum required. `test_mask` in `leave_one_seizure_out_prediction` is
computed at *whole-recording* granularity (`(subject, run)` set
membership), not by targeting the preictal window range specifically. That
means the entire held-out recording — preictal windows, ictal windows,
postictal windows, everything physically in that file — is excluded from
every other fold's training set, a strict superset of "just exclude the
preictal lead-up." The additional `seizure_id_arr != seizure_id` filter in
`train_mask` is redundant under today's one-seizure-per-recording CHB-MIT
data (as `leave_one_seizure_out_prediction`'s own docstring already notes)
but is there as a robustness invariant for a hypothetical future
multi-seizure-per-recording case.

**Claim 2, confirmed true.** `_label_windows_prediction`'s
`excluded |= overlap_warn | overlap_ictal | overlap_postictal` is computed
independently of `positive`, and the final assignment
(`labels = np.where(excluded, -1, np.where(positive, 1, 0))`) gives
exclusion strict precedence. A window straddling the preictal/warn-zone
boundary — overlapping both intervals at once — is therefore always
dropped, never labeled positive, regardless of how much of it falls in the
genuinely-preictal side.

---

## Current state

- Neither leakage mechanism raised by the critique is present in the
  current code. This doesn't by itself validate the prediction results —
  it rules out two specific, plausible failure modes, which is what was
  asked for.
- The label-permutation null control (`--shuffle-labels`, see the
  companion note's Part 6) is a separate, complementary check — it tests
  "is there learnable structure at all," independent of whether that
  structure could be explained by either of these two leakage paths. Both
  checks now complete: no leakage found here, and the real run cleared the
  null control (mean roc_auc 0.882 vs. null 0.491) — together a
  meaningfully stronger claim than either alone.
- `verify_fold_boundary.py` is scratchpad-only, not committed to the repo.
  If this kind of check is wanted on an ongoing basis (e.g. re-run whenever
  SPH/SOP change), it should move into `tests/` instead of being
  re-derived from scratch next time.
