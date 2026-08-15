# Session notes — mu-band/scale-adaptive smoothing experiments + ChannelSignalEncoder pooling regression (2026-08-09)

Branch: `channel_subset`. Repo: `/Users/noahshore/Documents/CoherIQs/moabb`.
Continues from [2026-08-08_sparse_evidence_gnn_surrogate_debug_plots.md](2026-08-08_sparse_evidence_gnn_surrogate_debug_plots.md)
-- picks up its subject-2 event-starvation/mu-band findings (Arcs 2-5 there)
and tries several concrete fixes for them. See
[[sparse-evidence-gnn-seed-variance]] for the standing caveat that every
single-seed number quoted below (this file and that one) should be treated
with caution.

---

## Arc 1 -- Four single-variable subject-2 experiments, all near chance

Motivated by a debug-plot screenshot (`debug_sparse_evidence_gnn.py`
output, `subj2_edge0_..._trial0`) showing large stretches of "confident-
looking" phase signal in regions of near-zero coherence -- explained as an
artifact: `torch.angle()` is defined even on near-zero-magnitude noise, and
the `(5,3)` smoothing kernel's overlapping windows correlate that noise
into visually "structured" streaks. Not a bug, just a reason to never trust
the raw phase panel without the coherence-gated overlay.

Ran four single-variable changes from canonical (`phase_threshold_deg=30`,
`surrogate_percentile=99`, `coherence_threshold_mode="surrogate"`,
`surrogate_count=100`), subject 2 only, `--subjects 2`:

| lever | mean | bursts/row |
| --- | --- | --- |
| `phase_threshold_deg=10` | 0.517 | 2.29 |
| `surrogate_percentile=90` | 0.553 | 3.99 |
| mu-band `pct=80` in 8-13Hz (new feature, see Arc 2) | 0.482 | 1.5-1.8 |
| `smooth_kernel_size=(25,3)` | 0.501 | 1.1-1.4 |

**None of these moved subject 2 off chance**, even though event density
recovered substantially for two of them. Turned out later (Arc 6) that
these results were also confounded by an unrelated regression active for
this whole session -- numbers here should not be taken as a clean verdict
on any of these levers; see Arc 6.

## Arc 2 -- New feature: `mu_band_surrogate_percentile` / `mu_band_range_hz`

Added targeted per-frequency-band percentile relaxation to
`sparse_evidence_gnn_classifier.py`, motivated by 2026-08-08's Arc 5
finding (genuinely phase-consistent mu-band cells never clear a flat high
percentile in either subject 1 or 2). Implementation:

- `_interp_percentile_grid` (module-level) now accepts either a scalar
  percentile (original behavior, unchanged) or a 1-D `[F]` array giving a
  different percentile per frequency bin, via `np.take_along_axis`.
- `SparseEvidenceGNNClassifier.__init__` gained
  `mu_band_surrogate_percentile: float | None = None` and
  `mu_band_range_hz: tuple[float, float] = (8.0, 13.0)`. When set, bins
  inside that range use the looser percentile; every other bin is
  unaffected. `None` (default) preserves the original flat-percentile
  behavior exactly.
- New `_percentile_vector(freqs_1d)` helper resolves which percentile
  (scalar or array) to hand `_interp_percentile_grid`.
- Threaded `freqs_1d` through `_surrogate_coherence_threshold` and
  `_surrogate_cluster_thresholds` (both now take an optional `freqs_1d`
  kwarg); `_precompute_sparse_events` passes `freqs[0]` (every row of
  `freqs` is identical per trial-batch).
- Deliberately did NOT touch `_surrogate_null_percentile_grid`'s own
  internal `forming_threshold_np` (used only to build/cache `cluster_null`)
  -- that stays on the plain scalar `self.surrogate_percentile`, since its
  cache entry is keyed on `cluster_null_forming_percentile=
  self.surrogate_percentile` and feeding it a per-frequency array would
  silently break that cache-key/content correspondence for
  `coherence_threshold_mode="surrogate_cluster"` (not used by any
  experiment in this file, but kept correct for later).

Sanity-checked directly (`_interp_percentile_grid` with a per-freq array
matches per-sub-range scalar calls; `_percentile_vector` returns the right
values) before spending any training-run compute on it. Result: mean=0.482
on subject 2 (Arc 1 table) -- no better than the flat-percentile levers.
`bursts_per_row` barely moved (1.5-1.8 vs canonical's ~1.3) because only
2-3 of 16 log-spaced bins (`lowest=8, highest=30` at the time) actually
fall inside 8-13Hz, so relaxing just those bins doesn't move the
*overall* average much even though it should be adding real events
specifically in that band.

## Arc 3 -- 4-subject sweep, `surrogate_percentile=90`: mild universal win, not a subject-2 fix

`--subjects 1 2 3 4`, only `surrogate_percentile` changed from canonical:

| subject | pct=99 (docstring-canonical, now known stale -- see Arc 5) | pct=90 | delta |
| --- | --- | --- | --- |
| 1 | 0.801 | 0.808 | +0.007 |
| 2 | 0.557 | 0.560 | +0.003 |
| 3 | 0.947 | 0.963 | +0.016 |
| 4 | 0.538 | 0.567 | +0.029 |
| **mean** | **0.711** | **0.725** | **+0.014** |

Nobody got worse -- worth keeping as a generically slightly-better setting
-- but subject 2 barely moved (+0.003, noise-level) even though everyone
else moved more. Confirms Arc 1's single-subject reading: this lever isn't
the subject-2 fix, just a mild broad improvement. (Also confounded by
Arc 6's regression, like everything else in this file predating Arc 6 --
not retested after the revert.)

## Arc 4 -- New feature: `scale_adaptive_smoothing` (period-proportional time kernel)

User's own observation, mechanistically confirmed: mu-band events are rare
regardless of threshold because the wavelet's own oscillation period is
long relative to the flat `(5,3)` smoothing kernel's raw-sample width. At
`sampling_rate=250`: one cycle is ~31 samples at 8Hz vs ~8 samples at 30Hz,
so a flat 5-sample kernel is ~16-24% of one mu-band cycle but ~60% of one
30Hz cycle -- far less real temporal averaging (fewer independent looks at
the phase relationship) exactly where it's most needed. This also
retroactively explains why the historical `(25,3)` kernel test (module
docstring, predates this file) was a wash: 25 samples is *still* under one
full cycle at 8Hz (~0.8 cycles) while being ~3 cycles (over-smoothed) at
30Hz -- a flat kernel size structurally can't get both ends of the
spectrum right at once.

Implemented the standard "scale-adaptive" wavelet-coherence smoothing fix
(Torrence & Webster 1999: time-smoothing width proportional to each
frequency's own period, not one flat width everywhere), scoped entirely to
`SparseEvidenceGNNCore` -- does NOT touch `WCTEvidenceGNNCore.
_smooth_wct_maps` (shared with the windowed WCT-Evidence-GNN pipeline):

- `SparseEvidenceGNNCore`/`SparseEvidenceGNNClassifier` gained
  `scale_adaptive_smoothing: bool = False`, `scale_adaptive_cycles: float
  = 1.5`, `scale_adaptive_max_kernel: int = 101`.
- `_scale_adaptive_time_kernel`: builds one Gaussian kernel per frequency
  bin, width = `round(scale_adaptive_cycles * sampling_rate / freq)`
  (clipped to `[3, scale_adaptive_max_kernel]`, rounded odd), each
  centered and zero-padded out to a common `max_k` width so a single
  grouped `F.conv1d(groups=nfreqs)` call applies all of them at once.
- `_smooth_wct_maps_scale_adaptive`: per-frequency time-conv (valid/
  unpadded, matching the flat path's `pad_h=0` convention -- the uniform
  `max_k-1` shrink applies regardless of each bin's real kernel width,
  since every row is zero-padded to `max_k`) followed by the SAME
  unchanged frequency-axis smoothing the flat path uses.
- `_smooth(...)` dispatcher added as the one choke point both
  `_coherence_only` and `_build_sparse_events` now call, instead of either
  calling `_smooth_wct_maps` directly -- avoids the two call sites ever
  drifting onto different paths.
- `_coi_valid_mask`'s `time_offset` (needed to map T_out-space indices
  back to raw-sample space) now goes through a new
  `_time_offset_samples()` helper instead of inlining
  `(smooth_kernel_size[0]-1)//2` -- that formula is wrong once the actual
  time-shrink is `scale_adaptive_max_kernel-1` instead.
- `surrogate_null_cache_key` (common.py) gained
  `scale_adaptive_smoothing`/`scale_adaptive_cycles`/
  `scale_adaptive_max_kernel` kwargs (default-preserving, same pattern as
  the existing `edge_topology` arg) -- these change the actual computed
  coherence values, not just how they're read, so a cache entry computed
  under one setting must never be loaded under another.

**Real bug found and fixed before any training run**: `_scale_adaptive_time_kernel`
did `freqs_1d.detach().to("cpu", dtype=torch.float64)` -- a fused
device+dtype `.to()` call. MPS doesn't support float64 at all, and
casting-before-moving in the fused call raises even though the tensor is
about to leave MPS anyway. Fixed by splitting into
`.detach().cpu().to(dtype=torch.float64)`. Caught by a real smoke test
(`compute_events` on synthetic MPS tensors) before spending training-run
time on it.

Kernel-width sanity check (subject-independent, `sampling_rate=250`,
`scale_adaptive_cycles=1.5`): landed within ~0.5-13% of the exact target
at every tested frequency (8-30Hz), e.g. 47 samples at 8Hz, 13 samples at
30Hz -- confirms the period-proportional math is doing what it's supposed
to.

Training result (subject 2, `scale_adaptive_cycles=1.5`, everything else
canonical): mean=0.487 -- still chance, fifth single-variable lever with a
chance-level result. Also confounded by Arc 6 (see below); not yet
retested clean.

## Arc 5 -- "Canonical" reference numbers were stale, and drifted further from each other

User pushback ("we usually get around 85% on sub1") didn't match either
number this file had been comparing against:

- The `sparse_evidence_gnn_classifier.py` module docstring / `run_wct_gnn.
  py`'s `_make_sparse_evidence_gnn()` comment (subj1=0.801, mean=0.711)
  is dated **2026-08-06** -- before the 2026-08-07 switch to
  surrogate-calibrated thresholds. Stale.
- `run_canonical_setup.py`'s own comment (2026-08-07) documents a better,
  later number much closer to "~85%": subject 1 mean=0.8395 at
  **`surrogate_percentile=95`**. But `CANONICAL_CONFIG["sparse"]["args"]`
  in that SAME file currently sets `surrogate_percentile=99`, not 95 --
  the comment was never updated when the percentile was later tightened.
  A comment/code drift bug in its own right.
- Scanned every `scores_*.csv` under
  `~/mne_data/results/LeftRightImagery/CrossSessionEvaluation/` for
  Sparse-Evidence-GNN runs covering all of subjects {1,2,3,4}, ranked by
  mean score. Highest on record:
  `scores_sub1best-4subj-phase10-surr99-2026-08-08` -- **mean=0.7408**
  (subj1=0.856, subj2=0.491, subj3=0.969, subj4=0.648), config
  `highest=35.0, phase_threshold_deg=10.0, surrogate_percentile=99.0,
  batch_size=8, channel_encoder_dilation=5, coi_enabled=True`. This is
  where the user's "~85% on sub1" comes from.
- Notably, `highest=35.0` there vs **`highest=30.0`** in today's
  (uncommitted) `_make_sparse_evidence_gnn()` default -- confirmed via
  `git diff` this was changed today, before this session started editing
  the file. Every experiment in Arcs 1-4 above ran at `highest=30`,
  un-matched to the 2026-08-08 best-known config. Turned out NOT to be the
  actual explanation for the day's bad numbers (see Arc 6's `highest=30`
  vs `35` re-test) -- but was a real, live confound worth ruling out
  explicitly rather than assuming away.

## Arc 6 -- Root cause of the day's near-chance results: ChannelSignalEncoder pooling regression (not mine)

A **different, concurrent Claude Code session** (evidenced by "modified on
disk since you last read it" edit-tool warnings received several times
while this file's session was mid-edit) had changed `ChannelSignalEncoder`
sometime during today's session, before Arc 1's experiments ran:

- Removed `nn.AdaptiveAvgPool1d(1)` from `ChannelSignalEncoder.net`, so
  `forward()` now returns `[B, C, embed_dim, n_time]` (per-raw-sample) 
  instead of `[B, C, embed_dim]` (pooled over the whole trial).
- `SparseEvidenceGNNCore.forward()` was updated to gather each event's
  embedding at its own approximate raw-sample time index
  (`events_padded[...,0]` -> `time_idx`) instead of using one
  trial-pooled vector broadcast onto every event.
- Rationale (from its own docstring, legitimate): the old pooled version
  gave two events on the same channel at different trial times the
  IDENTICAL embedding -- zero timing information -- which is consistent
  with `feature_ablation="zero_event_features"` barely moving accuracy
  (see [[sparse-evidence-gnn-channel-encoder-dominates]]): under pooling,
  the channel-embed block was the only one of the two message-MLP inputs
  that could vary meaningfully trial-to-trial.
- Its own docstring flagged it as **"Not yet re-validated against the
  4-subject canonical suite"** -- i.e. landed untested.

**Verified the indexing itself was mechanically correct** (`channel_feat
[batch_idx, src_padded, :, time_idx]` matched a slow-loop reference
exactly) -- not a shape/gather bug. Suspected mechanism instead: gathering
one timestep per event means `channel_encoder`'s conv layers only receive
gradient at the handful of raw-sample positions actual events land on per
training step, out of ~1000 total timesteps -- vs. every position
contributing via the old pooled average. Much sparser/noisier gradient
signal, plausible root cause for training collapsing toward chance
(observed pattern: train accuracy hitting ~100% while held-out stayed
~0.46-0.60, i.e. severe overfitting/poor generalization, not a crash).

**Reverted**, surgically (this file also carries Arcs 1-5's real, wanted
changes, so no blanket `git checkout`): `ChannelSignalEncoder` restored to
the pooled `AdaptiveAvgPool1d(1)` version, `forward()` restored to the
simple `channel_emb[batch_idx, src_padded]` gather. Full snapshot of the
per-timestep version kept at
[snapshots/2026-08-09_channel_encoder_per_timestep.md](snapshots/2026-08-09_channel_encoder_per_timestep.md)
for easy reinstatement if revisited -- likely next step there: a LOCAL
window pool around each event's time index (not whole-trial, not a single
raw sample) to get event-relative timing back without cutting gradient
coverage down to single points.

**Verified the revert with two real training runs**, subject 1,
`phase_threshold_deg=10, surrogate_percentile=99, batch_size=8` (matching
the 2026-08-08 best-known config):

| config | 0train | 1test | mean |
| --- | --- | --- | --- |
| `highest=35.0` (exact 2026-08-08 match) | 0.879 | 0.826 | **0.852** |
| `highest=30.0` (today's default) | 0.880 | 0.825 | **0.852** |

Both essentially identical to the historical 0.856, and to each other --
confirms (a) the revert actually fixes the regression, restoring subject 1
to its expected ~85% range, and (b) `highest=30` vs `35` makes no real
difference once the actual bug is fixed, resolving Arc 5's open question:
`highest` was never the cause of anything, safe to keep at 30.

## Where things stand / open threads (this file)

- **`sparse_evidence_gnn_classifier.py` is back to a known-good state**
  (`ChannelSignalEncoder` pooled, subject 1 verified at ~0.852 under the
  2026-08-08-matching config). `mu_band_surrogate_percentile`/
  `mu_band_range_hz` and `scale_adaptive_smoothing`/`scale_adaptive_cycles`/
  `scale_adaptive_max_kernel` are real, working, additive features now
  live in the classifier (default off / original behavior) -- not yet
  re-tested against subject 2 under the FIXED encoder, since every result
  quoted for them in Arcs 1-4 predates the Arc 6 revert and is confounded
  by it.
- **Most actionable next step**: re-run Arc 1's four single-variable
  subject-2 experiments (and Arc 3's 4-subject `pct=90` sweep) now that
  the encoder regression is fixed -- none of their near-chance verdicts
  should be trusted as-is.
- `surrogate_percentile=90` (Arc 3) is a plausible new default regardless
  of subject 2 (mild universal win, pre-Arc-6 numbers) -- worth
  reconfirming post-revert before adopting.
- `run_canonical_setup.py`'s comment claiming 0.8395 at
  `surrogate_percentile=95` vs. its actual current `99.0` param value
  (Arc 5) is still an open, unresolved drift -- not yet re-tested at 95
  under the fixed encoder either.
- Coordination risk made concrete today: this file and at least one other
  concurrent Claude Code session both edited
  `sparse_evidence_gnn_classifier.py` today with no lock/branch
  separation between them, and the other session's change landed
  untested and broke training broadly before anyone noticed via accuracy
  numbers alone. Worth a deliberate branch-per-session or explicit
  hand-off convention if concurrent sessions against this file continue.
