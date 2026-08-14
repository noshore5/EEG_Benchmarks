# concat_h4 cross-subject validation, productionizing "concat", and a new dense-mode time-downsample option — 2026-08-10

## Why

Follow-up to the same day's [dense frozen-feature capacity/aggregation
sweep](2026-08-10_sparse-evidence-gnn_dense-frozen-feature-capacity-aggregation-sweep.md),
which left three things open:

1. `concat_h4`'s end-to-end validation (0.8973) was **single-subject
   (subject 1 only)** by design ("run it once") — subject 1 is historically
   easy for this pipeline, so a real cross-subject read was still missing.
2. Whether `concat_h4`'s regression relative to flat (0.8973 vs 0.9237-0.9291)
   traces to the aggregation/readout step or to `dense_edge_conv`'s own
   learned representations degrading under joint training with concat's
   gradient — flagged as the natural next check.
3. `event_aggregation="concat"` only existed in the throwaway experimental
   script (`run_dense_edge_frozen_sweep.py`), not in the actual production
   classifier (`sparse_evidence_gnn_classifier.py`) or its canonical driver
   (`run_sparse_evidence_gnn.py`) — nothing tested there could be used for a
   real evaluation run.

This session closes all three: a 4-subject cross-subject validation, a
"does swapping the readout recover flat's score" check (answer: no), and
productionizing `concat` as a real, validated `SparseEvidenceGNNClassifier`
option — plus a new, separate, not-yet-benchmarked speed option
(`dense_edge_time_downsample`) motivated by the productionized concat
config's own training cost.

## Part A — Part 3b: flat head on the concat-trained conv's features (subject 1)

Before touching other subjects, re-ran Part 3 (`concat_h4`, seed 42,
subject 1) to confirm reproducibility and to generate a fresh, always-cached
conv checkpoint to test against (see "Always-on caching" below).
**Bit-identical to the original run**: 0train=0.8559, 1test=0.9387,
mean=0.8973.

Took that trained conv's frozen features and re-ran them through the
ORIGINAL flat head (`flat_direct_288`, no hidden layer) instead of
`concat_h4`'s own head — same conv, readout swapped only:

| readout on the concat_h4-trained conv's features | 0train | 1test | mean |
| --- | --- | --- | --- |
| concat_h4's own head (end-to-end, reference) | 0.8559 | 0.9387 | 0.8973 |
| flat head, same frozen conv | 0.8573 | 0.9338 | **0.8955** |
| flat-on-flat-features (original flat control, reference) | 0.8949 | 0.9633 | 0.9291 |

**Swapping the readout recovers essentially nothing** (0.8955 vs 0.8973's
own end-to-end score — within 0.002 of each other, both ~0.033 below
flat-on-flat's 0.9291). This answers open question #2: the regression is
**not** the aggregation/readout step. `dense_edge_conv`'s own learned
representations degrade when trained jointly with concat's gradient — the
readout is not the specific problem. Consistent with (now confirming,
not just speculating) the earlier write-up's hypothesis that concat's
gradient — routed through per-node grouping into a wide, largely-redundant
classifier input — is a harder optimization target than flat's direct,
uniform gradient to every edge's own slot.

## Part B — always-on conv-feature caching + cross-subject infrastructure

`run_dense_edge_frozen_sweep.py` changes (all in the experimental script,
not production):

- **Always-on caching hook** (`cache_all_trial_features`): any code path in
  the script that trains a `dense_edge_conv` — Part 1's flat-control
  retrain AND Part 3's end-to-end graph/concat training — now caches that
  conv's frozen features immediately after `fit()`, the same way
  `surrogate_null_cache` is written as an automatic side effect rather than
  a separate opt-in step. Previously only Part 1 cached; Part 3 required a
  full retrain to look at its own conv's features again (which is what Part
  A above needed and is now built on).
- **`subject` parameterized** through `run_part3_validate`/
  `run_flat_on_frozen` (previously hardcoded to subject 1).
- **`part3_manifest.json` made concurrency-safe**: a `flock`-protected
  read-modify-write (`_upsert_part3_manifest_run`), plus atomic
  temp-file-then-`os.replace` writes (matching `save_dense_conv_feature_cache`'s
  own convention — see [[run-wct-gnn-concurrent-write-race]]) — needed
  because subjects 2-4 were run as three concurrent processes against the
  same shared manifest file. Verified with a 4-thread × 3-write stress test
  (12/12 entries survived, no lost updates) before trusting the real run.

## Part C — cross-subject validation: concat_h4 end-to-end, subjects 1-4, seed 42

Ran subjects 2, 3, 4 as three parallel background processes (subject 1's
result reused from Part A) — all three finished cleanly, manifest intact
(verified all 8 new cache files present and every entry resolvable).

| subject | 0train | 1test | mean |
| --- | --- | --- | --- |
| 1 | 0.8559 | 0.9387 | 0.8973 |
| 2 | 0.5592 | 0.6172 | **0.5882** |
| 3 | 0.9316 | 0.9150 | 0.9233 |
| 4 | 0.6809 | 0.7118 | 0.6964 |
| **cohort mean (n=4)** | | | **0.7763** |

**Huge across-subject spread** — subject 2 is barely above chance (0.588),
subject 4 is weak (0.696), subjects 1 and 3 are strong (0.897, 0.923).
Subject 1 (the only one tested until now) happened to land on the good end;
a single-subject screen was masking this spread entirely. For loose
reference, [[sparse-evidence-gnn-9subj-first-full-cohort-run]] recorded a
9-subject mean of 0.7474 for the production pipeline's *normal*
(non-concat) configuration — `concat_h4`'s 4-subject 0.7763 is in a similar
range, though not a controlled comparison (different subjects, n=4 vs n=9,
different architecture entirely).

Caveat: single seed (42) per subject, by design (matching this
investigation's "run it once, then decide" scope throughout) — not a
seed-variance-controlled number for any individual subject.

## Part D — productionizing `event_aggregation="concat"`

Ported the validated `concat` aggregation from the experimental
`ConcatGraphHead` into the real pipeline: `sparse_evidence_gnn_classifier.py`'s
`SparseEvidenceGNNCore` and `SparseEvidenceGNNClassifier` both now accept
`event_aggregation="concat"` as a third option alongside `"mean"`/
`"gated_softmax"`.

- **Validated at construction** (both classes, matching the existing
  `freq_aware_hops`/`event_mode` incompatibility precedent): requires
  `event_mode="dense"` (concat's fixed per-node slot mapping assumes dense
  mode's every-edge-always-fires, fixed-order event list — sparse mode's
  variable-length per-trial event list has no such guarantee) and
  `n_hops=1` (multi-hop propagation assumes one hidden_dim vector per node;
  concat's own representation keeps every incident edge distinct instead).
  Both raise `ValueError` otherwise.
- New `concat_slot_idx` buffer (built unconditionally in `__init__`, no RNG
  cost) maps each destination channel's k-th incident edge to its position
  in the canonical edge order, with a dedicated pad row for channels with
  fewer than `max_degree=n_channels-1` incident edges.
- `_aggregate_events` gained a `"concat"` branch (gather via
  `concat_slot_idx` instead of scatter_add+mean or scatter-softmax);
  `sparse_classifier`'s input width is sized correctly for it
  (`n_channels * max_degree * hidden_dim` vs `mean`/`gated_softmax`'s
  `n_channels * hidden_dim`); `forward()`'s readout reshape branches on
  aggregation mode.
- **Verified**: both error paths raise correctly; a synthetic end-to-end
  fit/predict with `concat` runs clean (12-trial smoke test); `mean` and
  `gated_softmax` are completely unaffected (regression-checked against the
  same synthetic data) — only the new mode's code path is new.

`run_sparse_evidence_gnn.py` gained `USE_CONCAT_DENSE` (opt-in toggle,
`ZERO_CHANNEL_EMBED`-style): flipping it swaps `event_mode`/
`event_aggregation`/`n_hops`/`freq_aware_hops` to the concat-on-dense
configuration, on top of everything else `SPARSE_EVIDENCE_GNN_PARAMS`
already sets (channel_subset, thresholds, epochs, seed). Left `False`
initially since the file was mid-experiment on an unrelated sparse-mode
config (`gated_softmax`, `n_hops=3`, `freq_aware_hops=True`); the user has
since flipped it to `True` with `SUBJECTS=[1,2,3,4]` for a live canonical
run using the Part C numbers above as its expected ballpark.

## Part E — new option: `dense_edge_time_downsample`

Motivated directly by the user observing ~3s/epoch under the newly-enabled
`USE_CONCAT_DENSE=True` config. Root cause (not concat-specific): in
`event_mode="dense"`, `dense_edge_conv` runs a real `Conv2d` over
`[E=36, T~1001, F=16]` every forward pass, every epoch — unlike sparse
mode's events, which are precomputed once per trial and cached.

Key structural fact: `dense_edge_conv` already ends in
`nn.AdaptiveAvgPool2d((None, 1))` — every surviving timestep gets
average-pooled to exactly one value per (edge, out_channel) regardless of
input length, and its two internal `MaxPool2d(pool_size=4)` stages already
collapse `T~1001` to ~61 internally before that final pool. The network is
already discarding ~94% of the time axis's resolution every forward call;
it's just doing so expensively, inside two Conv2d layers that have to
process the full native `T` first.

**Implementation**: new param `dense_edge_time_downsample` (default `1`,
off, bit-identical to before) on both classes. When `>1`, average-pools the
`[B, 4, E, T, F]` coherence/phase/significance stack's time axis by that
factor, applied inside `_build_dense_edge_input` **after** smoothing (a
real low-pass filter) and **after** COI-masking, at native resolution —
once per trial at precompute time, not every epoch. Validated: must be
`>=1`; raises if set `!=1` while `event_mode="sparse"` (inert there,
rejected explicitly rather than silently ignored, matching every other
dense-mode-only param's precedent in this file).

Deliberately a **different, safer operation** than the existing
`cwt_resample_n_time` (which resamples raw complex CWT coefficients
*before* coherence is computed, destroying real high-frequency signal and
breaking the COI mask's own timing assumptions — see that param's
docstring): downsampling an already-smoothed, already-COI-masked signal is
the textbook-correct order (filter, then downsample), not the reverse. One
approximation it does introduce: a pooling window can blend a few native
COI-valid/invalid timesteps near the mask's edge — but `dense_edge_conv`'s
own first-layer kernel already blends across that same boundary at native
resolution today (unmasked convolution), so this isn't a new category of
error, just a coarser instance of one the architecture already tolerates.

**Verification**:
- Exact shape/math check: `[0,1,...,7]` pooled by 4 → `[1.5, 5.5]` (exact
  circular-mean-safe average, since phase is stored as separate sin/cos
  channels rather than a raw angle — averaging those two directly is
  already the mathematically correct way to downsample a phase signal).
- All three `event_aggregation` modes (`mean`, `gated_softmax`, `concat`)
  work correctly combined with downsampling; default sparse-mode path
  completely unaffected (regression-checked).
- Speed, at closer-to-production scale (native T=1001, nfreqs=16, real conv
  sizes, 12 synthetic trials): `downsample=4` cut fit time 2.31s → 0.73s
  (~3.2x) — consistent with the "roughly linear in the factor" expectation
  from the conv's own FLOPs scaling with input width.

Wired into `run_sparse_evidence_gnn.py` (initially off by default). Comment
flags to keep any chosen factor modest relative to `smooth_kernel_size[0]=5`
(the real anti-aliasing filter width) and to treat a run using it as its
own experiment, not a free drop-in speedup.

**First real-data result (same day, after this note was first drafted)**:
the user ran the actual canonical `run_sparse_evidence_gnn.py` twice back to
back, identical config otherwise (subject 1, seed 45, concat +
zero_channel_embed, epochs=100, run-id `freq-aware-hops-subj1-test-h2`),
differing only in `dense_edge_time_downsample` — confirmed via the run
ledger (`~/mne_data/run_ledger.csv`), not just the console output:

| dense_edge_time_downsample | 0train | 1test | mean |
| --- | --- | --- | --- |
| 1 (off) | 0.8622 | 0.9269 | 0.8945 |
| 4 | 0.8696 | 0.9406 | **0.9051** |

`k=4` scored higher on **both** folds independently (+0.0074, +0.0137), not
just a mean that could be hiding one fold worse. A genuinely encouraging
first signal — but still single-seed, single-subject; not yet a real
benchmark. See [[sparse-evidence-gnn-concat-productionized-cross-subject]]'s
own 2026-08-10 update for the same numbers.

## Caveats

1. Part C (cross-subject) is single-seed (42) per subject — a real
   multi-seed cross-subject sweep is the obvious next step before treating
   any individual subject's number (especially subject 2's 0.588) as
   settled rather than partly seed noise. See [[sparse-evidence-gnn-seed-
   variance]] for this pipeline's general seed-sensitivity.
2. `dense_edge_time_downsample`'s first real-data result (Part E, subject 1,
   seed 45, k=4) is positive on both folds — but still just one seed/one
   subject. Treat `k>1` as a promising, not yet validated, experimental
   knob; a multi-seed/multi-subject check is the natural next step before
   trusting the direction.
3. Every other Part 2 graph-family number from the prior sweep
   (`graph_h8/16/32`, `concat_h1`) still uses a flat-trained frozen conv,
   not a graph-trained one — unaffected by, and not revisited in, this
   session's work.
4. Productionized `concat` inherits every caveat the experimental version
   already carried (subject 1 historically easy, untuned `dense_conv_*`
   hyperparameters) — Part C's 4-subject numbers are the first real dent in
   the "subject 1 only" caveat specifically.

## Concurrent-edit check

`sparse_evidence_gnn_classifier.py`: only `SparseEvidenceGNNCore`/
`SparseEvidenceGNNClassifier`'s `event_aggregation`/
`dense_edge_time_downsample`-related code was touched (new params, new
buffer, new `_aggregate_events`/`_build_dense_edge_input` branches);
`_build_dense_feature_conv`, sparse-mode's event pipeline, and every other
aggregation mode's existing code paths are unchanged and regression-tested.
`run_sparse_evidence_gnn.py`: only additive (`USE_CONCAT_DENSE`,
`DENSE_EDGE_TIME_DOWNSAMPLE` toggles + their `.update()` application);
`SPARSE_EVIDENCE_GNN_PARAMS`'s existing keys and `RUN_ID`/`CONSOLE_ARGS`
untouched. `run_dense_edge_frozen_sweep.py`: caching/manifest changes are
additive and backward-compatible (existing Part 1/2 cache files and
`manifest.json` untouched, verified still readable after this session's
changes).

See [[sparse-evidence-gnn-capacity-confound-refuted]] and
[[sparse-evidence-gnn-dense-flat-control-beats-graph]] for the two prior
write-ups this one directly follows up on.
