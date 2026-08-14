# Session notes — cross-subject eval, surrogate significance calibration, phase-gate direction bug (2026-08-07)

Branch: `channel_subset`. Repo: `/Users/noahshore/Documents/CoherIQs/moabb`.
Session ID: `f0981f26-56f2-4106-8c91-15120f3d82b4`.
Continues from [2026-08-06_full_session_summary.md](2026-08-06_full_session_summary.md)
(Arcs 1-6). See that file's "Where things stand" for context predating this one.

**Last updated: 2026-08-07**. Update this file in place for further 2026-08-07
work in this session; start a new dated file for a different day.

---

## Arc 1 — Cross-subject evaluation, surrogate significance calibration

1. **`--cross-subject` CLI flag** added to `run_wct_gnn.py`, wiring MOABB's
   `CrossSubjectEvaluation` (LOSO) in alongside the existing
   `CrossSessionEvaluation`. Validated end-to-end on subjects 1-3, epochs=100:
   within-subject (eval=cross) mean 0.768 → LOSO (eval=subject) mean 0.699 —
   a real but noisy (3-subject, single-seed) ~7-point drop, driven mostly by
   2 of the 3 subjects.
2. **Surrogate-based coherence significance calibration** implemented as an
   opt-in mode (`coherence_threshold_mode="surrogate"` on
   `SparseEvidenceGNNClassifier`, default stays `"fixed"` — fully backward
   compatible). Phase-randomizes each channel's FFT spectrum (`common.py`'s
   `phase_randomize_surrogates`, preserves magnitude/power, destroys real
   cross-channel phase coupling), runs `surrogate_count` (e.g. 100) surrogates
   through the identical CWT→cross-spectrum→smoothing pipeline per real
   trial, and uses the resulting null coherence distribution's
   `surrogate_percentile` (e.g. 95th) as a per-(edge, frequency) threshold
   instead of one fixed global cutoff.
   - Bottleneck profiling: `fcwt.cwt()` itself is fast (~0.44ms/call) —
     the real cost is the cross-spectrum + smoothing step
     (`_full_edge_wct_maps`+`_smooth_wct_maps`), which is memory-bandwidth-
     bound, not dispatch-bound (hand-rolled shift-and-add was only ~9%
     faster than `conv2d`).
   - Added `resolve_best_available_device()` (cuda→mps→cpu) in `common.py`,
     used **only** for this surrogate-calibration path (not the main
     training loop, which stays CPU-only pending separate validation).
     Measured ~10x speedup for the smoothing step on Apple M4 MPS
     (22ms vs 232ms warmed-up), ~3.7x real-world end-to-end
     (5m25s → 1m28s at `surrogate_count=10`).
   - Surrogate-calibrated threshold distribution (4 real subject-1 trials,
     `surrogate_percentile=95`, `phase_threshold_deg=30`): mean 0.9205,
     median 0.9147, range [0.876, 0.988] — lands almost exactly where the
     user's independently, manually-tuned `coherence_threshold≈0.99` had
     already converged. Read as mutual validation: the small `(5,3)`
     smoothing kernel gives few "degrees of freedom," so raw coherence is
     inherently biased toward 1.0 under the null even with zero true
     coupling, and the surrogate method is correctly detecting that rather
     than being miscalibrated.
   - Not yet built: any caching of the surrogate calibration. Every
     `_precompute_sparse_events` call recomputes all `surrogate_count`
     surrogates from scratch (per trial, per `fit()`/`score()` call) — a
     disk cache keyed by (trial signal hash + CWT/kernel/COI config),
     independent of `surrogate_percentile` itself, was proposed but not
     built (would let percentile sweeps reuse one cached null distribution
     instead of re-running surrogates).

## Arc 2 — The phase-gate direction bug: reported, "fixed," then reverted

This is the most consequential thread of 2026-08-07 and the one most likely
to be revisited, so documented in full.

**Original code** (present all along, in `_build_sparse_events`,
`sparse_evidence_gnn_classifier.py`):

```python
gate = (coh > threshold) & (phase > self.phase_threshold_rad)
```

**The user's report**: "it should be + or - the threshold passes. the sign
of the angle determines the source vs destination" — read (correctly, in
spirit) as: a one-sided test looks like it only catches lead/lag in one
rotational direction and silently drops the other. Implemented literally as
a symmetric/two-sided gate:

```python
gate = (coh > threshold) & (phase.abs() > self.phase_threshold_rad)
```

plus a companion fix in the run-consolidation logic (a same-sign-continuation
check, so a consolidated "event" couldn't silently average a positive-lag
sample with a negative-lag sample into a meaningless near-zero mean angle).

**Effect measured**: event count roughly doubled (8 real subject-1 trials,
`coherence_threshold=0.5`, `phase_threshold_deg=30`: one-sided mean ~2473
events/trial → two-sided mean ~4892, ratio ~1.98x). A controlled
fixed-vs-surrogate comparison at the time (mean 0.836 fixed / 0.813
surrogate) looked consistent with historical baselines, so the two-sided
change was initially believed to be a real bug fix with no regression.

**Then the user's own live run got slower and scored worse** than before the
change, at their actively-tuned settings (`coherence_threshold=0.9-0.95`,
`phase_threshold_deg=10-15°`). Re-deriving the cross-spectrum math (not just
re-running numbers) found the actual mechanism:

- `ordered_pair_indices` (`common.py`) instantiates **both** directed copies
  of every channel pair — a separate edge object for i→j and for j→i (72
  directed edges for the 9-channel subset used here: 36 unordered pairs × 2).
- `_full_edge_wct_maps`'s cross-spectrum is `W_src · conj(W_dst)`:
  `xwt_real = src_r*dst_r + src_i*dst_i` (symmetric under src↔dst swap),
  `xwt_imag = src_i*dst_r - src_r*dst_i` (antisymmetric under the swap).
  Swapping which channel is src and which is dst — i.e. going from edge i→j
  to edge j→i — therefore gives `xwt_(j→i) = conj(xwt_(i→j))` **exactly**.
- Smoothing is a real-linear operator applied identically to both directed
  copies, so the conjugate relationship survives smoothing intact:
  `phase_(j→i) = -phase_(i→j)` and `coh_(i→j) = coh_(j→i)`, always, for
  every pair, every (time, freq) cell.
- Consequence: the **original one-sided** gate (`phase > +threshold`) is
  already directionally correct by construction. For a given pair, i→j can
  fire (`phase_ij > threshold`) or j→i can fire (`phase_ji = -phase_ij >
  threshold`, i.e. `phase_ij < -threshold`) — these are mutually exclusive
  by sign, so exactly one of the two directed edges ever fires for a given
  phase relationship, and which one is chosen **is** "the sign of the angle
  determines the source vs destination." The two-sided (`.abs()`) gate broke
  this: `|phase| > threshold` is true for **both** i→j and j→i together
  whenever the pair has a strong phase relationship in either direction —
  so instead of picking one direction, it fires both simultaneously,
  roughly doubling event volume with duplicate, non-directional signal
  flowing into the message-passing GNN. That inflation is the direct cause
  of both the slowdown (more events processed per trial) and the accuracy
  drop (noisier, contradictory-direction edges replacing what used to be a
  clean directional selection).

**Resolution**: reverted the gate to the original one-sided form
(`phase > self.phase_threshold_rad`), with a comment in the code explaining
why it's correct rather than a bug — see
`sparse_evidence_gnn_classifier.py::_build_sparse_events` (~line 264). The
same-sign-continuation run-consolidation logic added alongside the `.abs()`
change is now dead code (a no-op under a one-sided gate, since every sample
within a passing run is already guaranteed same-signed by construction) —
left in place since it's harmless and self-documents why, but worth removing
if the file gets a cleanup pass.

**Verification, two forms**:
1. *Re-run at real settings*: `run_canonical_setup.py` (`CANONICAL_VARIANT=
   "sparse"`, subject 1, current grid: `coherence_threshold=0.95`,
   `phase_threshold_deg=10.0`, `batch_size=8`, `epochs=100`,
   `validation_split=0.0`) with the reverted gate scored **0train=0.890,
   1test=0.818, mean=0.854** — the best subject-1 number logged this session
   (vs. 0.801 in the 4-subject canonical baseline docstring, vs. 0.836 for
   the earlier fixed-mode comparison at `coherence_threshold=0.5`). Single
   seed/subject, not a fully controlled before/after at these exact
   thresholds, but consistent with the diagnosed mechanism.
2. *Concrete numerical trace on real data* (script preserved at
   `/private/tmp/.../scratchpad/trace_direction_gate.py` during the session;
   not committed to the repo — rerun against a real subject-1 trial to
   reproduce): picked one real channel pair (local channels 0, 1) from one
   real subject-1 trial. At the (time, freq) cell where their coherence
   peaked: `phase_(i→j) = +0.241094 rad`, `phase_(j→i) = -0.241094 rad`
   (sum = 0.0 exactly), `coh_(i→j) = coh_(j→i) = 0.999934` (diff = 0.0
   exactly) — confirming the conjugate-symmetry math is not just algebra but
   holds bit-exactly on real data. The one-sided gate fired on i→j and
   correctly did not fire on j→i at that cell. Scanned across **every**
   (time, freq) cell for that same pair over the whole trial: i→j fired on
   1089 cells, j→i fired on 534 cells, **0 cells fired on both
   simultaneously** — full mutual exclusivity confirmed empirically across
   the whole trial, not just the one cell picked for illustration.

**Open items from this arc**:
- No before/after re-run yet at the *exact* current settings
  (`coherence_threshold=0.95`, `phase_threshold_deg=10°`) under the old
  `.abs()` gate, to get a clean paired delta at today's thresholds (the
  ~2x-event / ~2.3-point-mean numbers above are from the earlier
  `coherence_threshold=0.5`/`phase_threshold_deg=30°` settings).
- The identical one-sided-gate pattern exists in sibling files
  (`msc_evidence_gnn.py`, `wct_evidence_gnn_classifier.py`,
  `wct_june18.py`, `wct_phase_gnn_classifier.py`) — not touched, since this
  investigation was scoped to `sparse_evidence_gnn_classifier.py` only.

## Where things stand / open threads (this file)

- **Not yet built**: surrogate-null-distribution caching (Arc 1), so
  `surrogate_percentile` sweeps still pay the full recompute cost each time.
- **Not yet run**: a controlled before/after of the `.abs()` vs one-sided
  phase gate at *today's* thresholds (`coherence_threshold=0.95`,
  `phase_threshold_deg=10°`) — Arc 2's ~2x-event-count numbers are from the
  earlier `0.5`/`30°` settings.
- **Note**: while this investigation was in progress, a concurrent Claude
  Code session independently edited the same
  `sparse_evidence_gnn_classifier.py`/`common.py` files (added the
  surrogate-null-cache feature — `load_surrogate_null_cache`/
  `save_surrogate_null_cache`/`surrogate_null_cache_key`/
  `default_surrogate_cache_root` in `common.py`). Its edits didn't touch the
  phase-gate line itself, so no conflict was hit, but two sessions were live
  on the same files concurrently on 2026-08-07 — worth checking before
  trusting either file's state blindly.
- **Canonical config as of 2026-08-07**: epochs=100, batch_size=8,
  `channel_encoder_dilation=5`, `coi_enabled=True`, native-resolution CWT/
  coherence (no resampling), `smooth_kernel_size=(5,3)`,
  `coherence_threshold=0.95`, `phase_threshold_deg=10.0`, `highest=35.0`,
  `lowest=8.0`, one-sided phase gate (`phase > threshold`, see Arc 2),
  `coherence_threshold_mode="fixed"` (surrogate mode is opt-in, Arc 1).
  Subject 1 canonical-sparse mean = 0.854 (0train=0.890, 1test=0.818),
  the best single-subject-1 number logged across both this file and
  2026-08-06's.
