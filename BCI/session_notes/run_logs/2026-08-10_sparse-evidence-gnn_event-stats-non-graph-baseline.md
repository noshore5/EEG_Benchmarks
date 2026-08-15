# Sparse-Evidence-GNN non-graph sanity-check baseline (event summary stats vs. message passing) — 2026-08-10

## Why

[[sparse-evidence-gnn-channel-encoder-dominates]] and the same-day
[zero_channel_embed re-check](2026-08-10_sparse-evidence-gnn_zero-channel-embed-4seed-threshold-recheck.md)
isolate the event pathway from raw-signal channel embeddings *inside* the
GNN (feature_ablation="zero_channel_embed": event content + graph topology
only, no ChannelSignalEncoder input) and get mean=0.628 (seeds 42-45,
std=0.0153). That still leaves channel_encoder/sparse_message_mlp/
_aggregate_events/graph structure in the loop. This run asks the
complementary question: how much of that 0.628 comes from the
graph/message-passing machinery itself, vs. from event statistics a much
simpler, non-graph model could read directly off the same events?

## Method

New standalone script:
[`run_sparse_event_stats_baseline.py`](../../tests/run_sparse_event_stats_baseline.py)
(does not modify any existing file).

- Reuses `SparseEvidenceGNNClassifier._prepare_features` (via a throwaway
  helper instance, exactly the technique `debug_sparse_evidence_gnn.py`
  already uses to call this pipeline's real methods directly) to get real
  `compute_events()` output — the same already-fixed, surrogate-calibrated,
  native-resolution, COI-masked event pipeline every other run uses. No
  SparseEvidenceGNNCore model is ever constructed or trained.
- Per trial, per canonical edge (36 for this 9-channel subset): `[count,
  t_mean, t_std, f_mean, f_std, mag_mean, phase_mean]` (7 stats, not the
  originally-sketched ~5-6 — kept both raw mean AND std for time/frequency
  since both were cheap and potentially informative; `phase_mean` is the
  CIRCULAR mean, `atan2(Σsinφ, Σcosφ)`, not a naive mean of an angle).
  Zero-event edges get an all-zero stat row (not omitted), so every trial's
  feature vector is a fixed 36×7=252-length vector.
- Plain sklearn classifier on those vectors: `StandardScaler` +
  `LogisticRegression` (first choice), also `MLPClassifier`
  (`hidden_layer_sizes=(32,)`) as a bonus nonlinear check since the LR
  result didn't call for it (see Results). No message passing, no
  scatter-add, no learned channel embeddings anywhere in this pathway.
- Evaluated with `moabb.evaluations.CrossSessionEvaluation` directly
  (`BNCI2014_001`, `LeftRightImagery(fmin=8, fmax=35)`, `random_state=42`)
  — the same evaluation class every other pipeline run in this repo goes
  through, not a hand-rolled split. Scored by moabb's default (ROC-AUC for
  this binary paradigm), matching how the GNN's own 0.628/0.8527 numbers
  were scored.
- Subject 1 only, seeds 42-45 (matching the zero_channel_embed re-check's
  own methodology). Event-stat feature extraction is DETERMINISTIC given
  fixed X — `surrogate_seed` is pinned at 42 independent of this script's
  own `seed` sweep, and z-score normalization doesn't affect coherence
  (see sparse_evidence_gnn_classifier.py's module docstring) — so the
  expensive `compute_events()` pass runs once per (train-session,
  test-session) fold and is process-locally cached across all 4 seeds x 2
  model types, not recomputed 8+ times; only the cheap sklearn refit varies
  per seed.

### Event-extraction config

Copied verbatim from the "Full effective parameters" block in the
[zero_channel_embed re-check note](2026-08-10_sparse-evidence-gnn_zero-channel-embed-4seed-threshold-recheck.md)
(`phase_threshold_deg=10.0`, `surrogate_percentile=99.0`,
`coherence_threshold_mode="surrogate"`, `surrogate_count=100`,
`channel_subset=[1,5,7,8,9,10,11,13,17]`, `smooth_kernel_size=(5,3)`,
`coi_enabled=True`, `sampling_rate=250`, `lowest=8.0`, `highest=30.0`,
`nfreqs=16`, `cwt_resample_n_time=None`) — **not** from
`run_sparse_evidence_gnn.py`'s current on-disk state. See "Concurrent-edit
check" below for why that distinction matters here.

## Results

### Logistic regression

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.6514 | 0.7259 | 0.6887 |
| 43 | 0.6514 | 0.7259 | 0.6887 |
| 44 | 0.6514 | 0.7259 | 0.6887 |
| 45 | 0.6514 | 0.7259 | 0.6887 |

mean=0.6887, std=0.0000 (sample). Completely seed-invariant, as expected —
`LogisticRegression`'s default `lbfgs` solver on a convex loss doesn't
meaningfully depend on `random_state`; feature extraction itself is also
seed-invariant (see Method). Not a bug, not a sign anything is broken —
just means "multi-seed" only adds real information here via the MLP check
below.

### MLP (bonus check — not strictly required, since LR already didn't underperform)

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.6381 | 0.7481 | 0.6931 |
| 43 | 0.6379 | 0.7114 | 0.6747 |
| 44 | 0.6657 | 0.7091 | 0.6874 |
| 45 | 0.6971 | 0.7614 | 0.7293 |

mean=0.6961, std=0.0234 (sample), min=0.6747, max=0.7293. Real seed
variance this time (genuine weight-init/SGD stochasticity), still
comfortably above both LR's 0.6887 and the GNN comparison point below.

### Comparison to `feature_ablation="zero_channel_embed"` GNN result

Reusing the [same-day re-check](2026-08-10_sparse-evidence-gnn_zero-channel-embed-4seed-threshold-recheck.md)'s
numbers (identical subject/config/seeds):

| model | mean | std |
| --- | --- | --- |
| GNN, `zero_channel_embed` (message passing, event+topology only) | 0.628 | 0.0153 |
| This baseline, LogisticRegression (event stats only, no graph) | **0.6887** | 0.0000 |
| This baseline, MLP (event stats only, no graph) | **0.6961** | 0.0234 |

Both non-graph baselines score **above** the GNN's own event-only ablation
— LR by ~0.061, MLP by ~0.068 (mean-to-mean; note the GNN score is a
scored-metric mean, same ROC-AUC convention as this baseline's `mean`
column). Neither result is "close to 0.628 from below" — the a-priori
framing this run was set up to test — both clear it.

For context, the full GNN with channel embeddings (`feature_ablation="none"`,
same subject/config/seeds) scores mean=0.8527 (8-seed sweep note) — so this
baseline and the `zero_channel_embed` ablation both sit well below the full
model; ChannelSignalEncoder's raw-signal embeddings are still doing most of
the work overall. This run is specifically about the *event-only* slice of
that picture.

## Finding

**Plain read, per this run's own pre-registered framing:** the baseline
does not land "close to" 0.628 (which would have meant the GNN architecture
roughly matches a raw-stats ceiling) or "well below, near chance" (which
would have meant message passing is doing real, hard-to-replace work). It
lands **above** the GNN's own event-only number, on the exact same events.
Under the letter of the original framing ("if this baseline gets close to
0.628, the GNN architecture isn't adding value... priority should shift to
event-detection quality"), this is the same conclusion but stronger: the
graph/message-passing machinery isn't just failing to add value over raw
event statistics here, it's doing *worse* than a linear model on a fixed
per-edge stat vector, given the identical event input. Priority should
shift toward event-detection quality (what the sparse pipeline extracts),
not further architecture work on how detected events get aggregated.

Two caveats worth keeping in view before treating this as final:

1. **Feature engineering isn't neutral.** `[count, t_mean, t_std, f_mean,
   f_std, mag_mean, phase_mean]` per edge is a specific, human-chosen
   summary — it happens to expose exactly the kind of channel-blind
   aggregate statistics (event counts/timing/frequency distribution per
   edge) that the zero_channel_embed re-check's own "best-guess mechanism"
   section already flagged as the likely driver of that ablation's
   above-chance result. The GNN's `sparse_message_mlp` + mean/gated_softmax
   aggregation has to discover something equivalent from raw per-event
   [t, freq, mag, sinφ, cosφ] tuples; it may simply not be finding as good
   a summarization as a hand-designed count/mean/std feature under this
   pipeline's current hyperparameters (hidden_dim=8, epochs=75-100) rather
   than message passing being fundamentally unable to.
2. **This isn't a fully controlled comparison.** The two "seed" axes mean
   different things: GNN seed variance reflects full model training
   stochasticity (weight init, batch order, dropout-like effects across a
   much larger parameter count); this baseline's LR seed axis is inert and
   its MLP axis reflects a much smaller model's init/SGD variance. The
   mean-vs-mean comparison is still fair (same events, same split, same
   scoring), but "variance" isn't directly comparable across rows.

Either way, this sharpens (doesn't just confirm) the standing finding in
[[sparse-evidence-gnn-channel-encoder-dominates]]: not only does the event
pathway contribute far less than channel embeddings to the full model, but
the specific graph/message-passing mechanism wrapped around events doesn't
even clear a simple stats-on-events baseline. Combined with
[[sparse-evidence-gnn-frequency-fragmentation-bias]] and the
already-completed `event_aggregation="gated_softmax"` experiment (training
found nothing worth differentiating between events under flat mean-pooling
either), this is now three independent angles pointing the same direction:
the ceiling on this pathway looks like it's set well before message
passing ever runs.

## Follow-up: does more hidden_dim capacity close the gap?

Caveat 1 above flagged that `sparse_message_mlp`/aggregation might simply
lack the capacity (hidden_dim=8) to extract as good a summary as the
hand-designed stats vector, rather than message passing being
fundamentally unable to. Tested directly: re-ran
`feature_ablation="zero_channel_embed"`, same subject/config/seeds, at
`hidden_dim=16` (double; `channel_embed_dim` held fixed at 8 -- single
variable changed). Driver: throwaway scratchpad script (not committed,
same convention as the original re-check), calling
`moabb.evaluations.CrossSessionEvaluation` directly with
`SparseEvidenceGNNClassifier(**kwargs)`. Sanity-checked first at
`hidden_dim=8, seed=42` against this same driver to confirm methodology
parity with the original run_wct_gnn.main()-based note before trusting the
sweep: got mean=0.6230 vs. the original 0.623 -- matches closely (0.6244
vs 0.624, 0.6215 vs 0.622).

| hidden_dim | seed 42 | seed 43 | seed 44 | seed 45 | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| 8 (original) | 0.623 | 0.610 | 0.646 | 0.633 | 0.628 | 0.0153 |
| 16 (bumped) | 0.6295 | 0.6284 | 0.6464 | 0.6291 | 0.6334 | 0.0087 |

Doubling hidden_dim moves the mean by ~0.005 -- well inside seed noise
(std at both settings), and nowhere close to either non-graph baseline
(0.6887 LR / 0.6961 MLP). **Capacity is not the bottleneck.** This closes
off caveat 1's main alternative explanation: the gap isn't "the MLP is too
small to represent an equivalent summary," it's something more structural
about how `_aggregate_events`' mean/gated_softmax pooling combines
per-event messages (already separately supported by the
`event_aggregation="gated_softmax"` result finding nothing worth
differentiating between events under flat mean-pooling). Caveat 2 (feature
engineering isn't neutral -- the stats vector may just expose the right
sufficient statistics more directly than raw per-event tuples force the
MLP to discover) remains the more likely explanation and is not addressed
by this follow-up.

## Concurrent-edit check

`sparse_evidence_gnn_classifier.py` (the file every layer of this
experiment ultimately calls into via `_prepare_features`/`compute_events`)
carries **uncommitted, in-progress changes from a separate, concurrent
task in this same session** (an `event_mode="sparse"|"dense"` feature,
added earlier today, default `event_mode="sparse"` preserving all existing
behavior exactly — this baseline never touches `event_mode` and always
gets the pre-existing sparse path). That work finished at 11:19; this
baseline script and its runs happened afterward and do not overlap with it
functionally.

Separately, `run_sparse_evidence_gnn.py` (**not** used by this script —
see "Event-extraction config" above) was found mid-edit for an unrelated
`freq_aware_hops`/subject-2 experiment at the start of the dense-mode task
earlier today (`phase_threshold_deg=0.0`, `surrogate_percentile=99.999`,
`epochs=100`, `subjects=[2]` on disk at ~11:01) and was modified **again**
after that check, up to at least 12:06 — over two hours before this
baseline's runs (~14:20+) — indicating another concurrent session actively
iterating on it throughout this window. This baseline deliberately hardcodes
its own `CANONICAL_EVENT_KWARGS` (sourced from the zero_channel_embed
re-check note, not from `run_sparse_evidence_gnn.py`'s live, moving state)
specifically to avoid picking up whatever that other session's in-progress
edits happen to leave on disk at any given moment.

## Implementation notes

- Base-class order bug found and fixed while wiring this up:
  `SparseEventStatsBaseline` must subclass `(ClassifierMixin, BaseEstimator)`
  in that order, not `(BaseEstimator, ClassifierMixin)`. This sklearn
  version (1.9.0) resolves `is_classifier()`/`get_tags()` via cooperative
  `__sklearn_tags__()` calls up the MRO; `BaseEstimator`-first silently
  yields `estimator_type=None` (`is_classifier()` → `False`) with no error.
  That broke moabb's ROC-AUC scorer specifically:
  `sklearn.utils._response._get_response_values` skips its binary
  positive-class column selection when `is_classifier()` is `False` and
  hands `roc_auc_score` the raw `(n_samples, 2)` `predict_proba` matrix,
  which fails deep inside `roc_curve` with "y should be a 1d array, got an
  array of shape (N, 2) instead." Confirmed by isolated repro against a
  minimal reproduction class before fixing. Worth remembering for any
  future custom sklearn-compatible estimator in this codebase.
- `LogisticRegression.fit` on the full 252-feature vector emits
  `RuntimeWarning: overflow encountered in matmul` a few times during
  optimization (StandardScaler-normalized inputs, default L2
  regularization, ~144 training trials vs. 252 features — real but modest
  overparameterization, likely compounded by near-zero-variance columns on
  edges with very few events across trials). Scores came out finite and
  reproducible (bit-identical across all 4 seeds), so this looks like
  solver-path noise, not a corrupted result — flagged here rather than
  silently ignored in case it recurs somewhere the output ISN'T this
  clean.

See [[sparse-evidence-gnn-channel-encoder-dominates]],
[[sparse-evidence-gnn-frequency-fragmentation-bias]], and the
[zero_channel_embed re-check](2026-08-10_sparse-evidence-gnn_zero-channel-embed-4seed-threshold-recheck.md)
for related findings this run builds on.
