# Freeze + cache dense_edge_conv, then sweep downstream capacity/aggregation — 2026-08-10

## Why

Two open questions carried over from today's earlier dense-mode work:

1. The [flat-control write-up](2026-08-10_sparse-evidence-gnn_dense-edge-flat-no-graph-control.md)
   found a no-graph control (flatten `dense_edge_conv`'s output straight to a
   single Linear) beats the graph pathway's own `zero_channel_embed` ablation
   (0.9291 vs 0.8886, seed 42/seeds 42-45) and flagged a capacity confound:
   the flat control's classifier sees 288 dims (`E * dense_conv_out_channels`,
   36 edges x 8), the graph pathway's `sparse_classifier` only ever sees 72
   (`n_channels * hidden_dim`, 9 x 8) after `_aggregate_events`' scatter-mean
   collapses ~4 edges/channel into one vector. Untested: does widening
   `hidden_dim` to match 288 dims close the gap?
2. Re-running `dense_edge_conv`'s forward pass (a real Conv2d over
   `[E=36, T~1001, F=16]` per trial) on every training step of every
   downstream-architecture variant is the dominant cost in this whole line of
   investigation (~450-530s per seed) — prohibitively slow for a real sweep.

This run addresses both: freeze one trained `dense_edge_conv`, cache its
per-edge output once, then sweep downstream heads on the cached features for
free, followed by an honest end-to-end check of whichever design wins.

## Method

New files:
[`dense_conv_feature_cache.py`](../../moabb_pipelines/dense_conv_feature_cache.py)
(cache key/load/save utilities) and
[`run_dense_edge_frozen_sweep.py`](../../tests/run_dense_edge_frozen_sweep.py)
(driver, four CLI subcommands: `cache`, `sweep`, `validate`,
`flat-on-frozen`). Neither touches `sparse_evidence_gnn_classifier.py`;
`run_dense_edge_flat_control.py` is imported unchanged
(`CANONICAL_HELPER_KWARGS`, `DenseEdgeFlatClassifier`), not duplicated.

**Follow-up added same day**: caching was originally Part 1-only (a manual
step run before Part 2's sweep). It's now an **always-on hook**
(`cache_all_trial_features`, mirroring how `surrogate_null_cache` gets
written as an automatic side effect of the pipeline's own forward pass
rather than a separate opt-in step) — every code path in the script that
trains a `dense_edge_conv`, Part 1's flat-control retrain *and* Part 3's
end-to-end graph/concat training, now caches that conv's frozen features
immediately after training, keyed and verified the same way. Part 3's runs
are recorded in a new `part3_manifest.json` (mirrors Part 1's own
`manifest.json`) so a later step can reload any past Part 3 run's cached
features by `cache_key` without retraining anything. This is what Part 3b
below is built on.

### Part 1 — freeze + cache + verify

The 2026-08-10 flat-control run never saved a checkpoint, so this **retrains**
(same seed 42, same code path, same hyperparameters — `DenseEdgeFlatClassifier`
imported directly, not reimplemented) rather than reloading. Subject 1 has
exactly two sessions (`0train`/`1test`); `CrossSessionEvaluation` trains one
model per held-out session (`LeaveOneGroupOut`-equivalent, replicated by hand
here to get the trained model object back — `moabb`'s own `evaluate()` only
yields scores, not fitted estimators). For each fold's trained model:

- Score on its own held-out session, compare to the recorded 2026-08-10
  numbers (0.8949/0.9633).
- Run the trained `dense_edge_conv`, `torch.no_grad()`, once over **all 288
  subject-1 trials** (not just that fold's own split) via
  `compute_dense_conv_features` — chunked (16 trials/batch) the same way
  `_precompute_dense_edge_inputs` chunks its own, larger, precompute.
- Cache the result (`[288, 36, 8]` float32, ~310-320KB/fold — vs. the
  ~2.6GB `[288, 4, 36, ~1001, 16]` raw input each trial's precompute
  produces) keyed on every config value that affects `dense_edge_raw`
  (mirrors `surrogate_null_cache_key`) **plus** the conv's architecture
  hyperparameters **plus** a `blake2b` hash of its actual trained weight
  values (`common.torch_parameter_hashes`) — so a differently-trained conv
  can never silently reuse stale features under the same key.
- **Verify**: flatten the cached features for that fold's held-out trials
  and run them through the model's own trained `classifier` `nn.Linear`
  directly, bypassing `dense_edge_conv` entirely. Compare to the model's
  live `predict_proba` on the same trials.

Smoke tests (random-tensor forward/backward + gradient checks on every new
module, plus a 12-trial synthetic `fit`/`predict_proba` integration check
with `surrogate_count=3`) run before the real data, per this investigation's
established practice.

### Part 2 — fast sweep on cached features

With cached features loaded from disk (no conv in the loop), fit small heads
directly, multi-seed (42-45), same training regime as the flat control
(`AdamW`, `CrossEntropyLoss`, grad-norm clip 0.1, `epochs=75`, `batch_size=8`,
a shuffle generator tied to `seed` — see
[[torch-eeg-classifier-dataloader-shuffle-seed-fix]]). Each fold's head
trains ONLY on that fold's own cached features (extracted by that fold's own
trained conv) and is scored on that fold's held-out session — same 2-fold
cross-session structure the flat control itself used, just with the
expensive conv held fixed.

Two families:

- **Flat** (`FlatHead`): flatten → optional hidden layer → 2 classes.
  `hidden_dim=None` reproduces the original flat control exactly (single
  Linear, no hidden layer). Tested `None`, `72` (bottlenecked to match the
  graph pathway's own readout width), `288` (matched to flat's own full
  width, but now with an actual hidden layer + GELU instead of one linear
  projection) — isolating whether a 72-vs-288 bottleneck matters for the
  flat topology specifically.
- **Graph, mean aggregation** (`FrozenGraphHead`): ports
  `SparseEvidenceGNNCore`'s exact `sparse_message_mlp` + `_aggregate_events`
  "mean" formula (copy-pasted, not reinvented) plus `sparse_classifier`,
  using the same `upper_pair_indices` topology import both use.
  `feature_ablation="zero_channel_embed"` is reproduced *structurally* —
  literal zero tensors concatenated in place of channel embeddings — rather
  than by zeroing a computed `channel_encoder` output, which is
  mathematically identical (that ablation's `torch.zeros_like` already
  detaches `channel_encoder` from the graph — see
  `SparseEvidenceGNNCore.forward`'s docstring) and lets this sweep skip
  running `channel_encoder` at all. `hidden_dim` swept at 8 (current
  default, 72-dim readout), 16, 32 (288-dim readout, matched to flat).
- **Graph, concat aggregation** (`ConcatGraphHead`, added mid-run — see
  Results): same message MLP, but concatenates each node's incident-edge
  messages in a fixed canonical order instead of mean-pooling them, so no
  information is discarded at the aggregation step itself. Dense mode's
  topology is a fixed complete graph (every edge always "fires"), so each
  node's incident-edge set is static across trials — a dedicated all-zero
  pad row handles nodes with fewer than `max_degree=n_channels-1=8`
  incoming edges (node *j*'s in-degree is exactly *j* under the canonical
  `i<j` dst-only scatter, so node 0 always has in-degree 0 — an existing
  pipeline property this control reproduces faithfully, not a new bug).
  Readout width = `n_channels * max_degree * hidden_dim` = `9*8*hidden_dim`;
  swept at `hidden_dim=1` (72-dim, matched to `graph_h8`) and `hidden_dim=4`
  (288-dim, matched to flat).

### Part 3 — end-to-end validation of the winner

`DenseEdgeGraphCore`/`DenseEdgeGraphClassifier` (parameterized by
`head_kind` ∈ {`mean`, `concat`}) wire a trainable `dense_edge_conv`
(identical construction to `DenseEdgeFlatCore`) directly into either graph
head, trained fully jointly from a fresh init — conv **not** frozen, unlike
Parts 1-2. Only the winning Part 2 design (`concat_h4`, see Results) was run
this way, **seed 42 only** ("run it once", matching the original flat
control's own single-seed-first scope) — not the full 4-seed sweep, since
this is a real ~500s/seed joint-training run, not a frozen-feature screen.

### Part 3b — flat head on the concat-trained conv's frozen features

Part 3's 0.8973 is worse than every flat-family number, but that alone
doesn't say *why*: is `concat_h4`'s own aggregation/readout the specific
problem on top of an otherwise-fine conv, or did training jointly with
concat's gradient shape *worse conv representations* than flat's own
gradient does? Using the caching hook above, Part 3's own
`concat_h4`-trained `dense_edge_conv` (both folds, seed 42) has its frozen
features re-run through the **original flat head** (`flat_direct_288`, no
hidden layer, identical training regime) instead of its own concat head —
same conv, different readout only. If flat-on-concat-features recovers
toward flat's own ~0.92+, the conv's representations were fine and the
aggregation step was the problem; if it stays down near concat's own
~0.89-0.90, the conv's representations themselves are the problem.

## Results

### Part 1: cache verification

| test_session | retrained | recorded (2026-08-10) | delta | cache-reconstructed | max\|Δproba\| |
| --- | --- | --- | --- | --- | --- |
| 0train | 0.8949 | 0.8949 | +0.0000 | 0.8949 | 0.00e+00 |
| 1test | 0.9633 | 0.9633 | +0.0000 | 0.9633 | 0.00e+00 |

**Bit-exact.** Retraining reproduced the recorded numbers to 4 decimal
places on both folds (PyTorch determinism on the same seed/code path/device,
as expected — see the driver's own "Reproduces, not reuses" docstring note),
and the cache-reconstructed score matches the live forward pass with
**zero** floating-point difference. Mean = 0.9291, exactly the recorded
value. The cache is trustworthy; Part 2 proceeds on it directly.

Cache size: ~310-320KB/fold (`[288, 36, 8]` float32) vs. the ~2.6GB raw
`dense_edge_raw` input each trial's precompute produces — the whole point of
freezing past the conv rather than caching further upstream.

### Part 2: frozen-feature screening (seeds 42-45)

| variant | mean | std | readout dims |
| --- | --- | --- | --- |
| flat_direct_288 (original flat control) | **0.9273** | 0.0012 | 288 |
| flat_hidden72 | 0.9239 | 0.0002 | 288→72→2 |
| flat_hidden288 | 0.9237 | 0.0014 | 288→288→2 |
| graph_h8 (current default, mean-pool) | 0.9104 | 0.0036 | 72 |
| graph_h16 (mean-pool) | 0.9051 | 0.0009 | 144 |
| graph_h32 (mean-pool, capacity-matched) | 0.9004 | 0.0024 | 288 |
| concat_h1 (concat) | 0.9086 | 0.0147 | 72 |
| concat_h4 (concat, capacity-matched) | 0.9184 | 0.0017 | 288 |

Reference points (from prior runs, joint-trained conv, not this frozen
conv): graph `zero_channel_embed` (hidden_dim=8) = 0.8886±0.0067; graph full
pipeline (channel embeds + events) = 0.9048±0.0031; flat control single-seed
= 0.9291.

**Capacity-matching does NOT close the gap for mean-pool aggregation — it
widens it.** Going from `hidden_dim` 8→16→32 (72→144→288 readout dims)
monotonically **decreases** score (0.9104→0.9051→0.9004), the opposite of
what the capacity-confound hypothesis predicted. Flat, by contrast, is
essentially flat across the same width range (0.9273→0.9239→0.9237, a mild
~0.003-0.004 decline from adding a hidden layer at all, nowhere near
graph's decline) — and `flat_hidden72` ≈ `flat_hidden288` (0.9239 vs 0.9237,
within noise), meaning the 72-vs-288 bottleneck width itself barely matters
once there's a hidden layer, for the flat topology. Put together: **it's
the mean-pool aggregation step itself, not readout dimensionality, driving
the gap.**

Following that up, **concat aggregation (still capacity-matched at 288
dims) meaningfully narrows the gap**: `concat_h4` (0.9184) beats `graph_h32`
(0.9004) by 0.018, closing roughly two-thirds of the remaining gap to
`flat_hidden288` (0.9237) at that same capacity (gap 0.0233 → 0.0053). It
does not fully close it. `concat_h1` (0.9086, 72-dim, matched to `graph_h8`)
is noisier (std 0.0147 vs `graph_h8`'s 0.0036) but in the same range.

**Caveat surfaced mid-analysis, important**: every `graph_*`/`concat_*`
number in this table uses `dense_edge_conv` **frozen from the flat
control's own training** — a conv that never saw a graph-aggregation
gradient, only a flat linear readout's. `graph_h8`'s 0.9104 here is
therefore *not* directly comparable to the previously-recorded
`zero_channel_embed` joint-trained 0.8886 (+0.022 higher) — it's a
different experiment ("how well does graph aggregation read out
flat-shaped features" vs. "how well does graph aggregation do when the conv
adapts to it"), and Part 3 exists specifically to resolve which of those
numbers is closer to the graph pathway's real ceiling.

### Part 3: end-to-end validation (seed 42 only)

| design | frozen-feature screening (seed 42) | end-to-end (conv unfrozen, seed 42) | Δ |
| --- | --- | --- | --- |
| concat_h4 | 0.9196 (0.8840/0.9552) | **0.8973** (0.8559/0.9387) | **−0.022** |

**The frozen-feature screening number was optimistic, not conservative.**
Letting `dense_edge_conv` co-adapt to `concat_h4`'s own aggregation, from a
fresh init, trained jointly, scores *worse* than reusing the flat-trained
conv's frozen features through the same head (0.8973 vs 0.9196) — and worse
than every flat-family reference (0.9237-0.9291). It lands close to (very
slightly below) the original mean-pool `zero_channel_embed`'s 0.8886. This
mirrors the same direction as the Part 2 caveat above: whatever gradient
signal the graph-aggregation pathway (mean *or* concat) provides back into
`dense_edge_conv` during joint training appears to shape *worse* conv
features than the flat pathway's uniform, un-pooled gradient does — a
plausible, not yet confirmed, mechanism (concat's own gradient is still
routed through per-node grouping and a much wider, mostly-redundant
classifier input, which may be a harder optimization target than a flat
288-dim readout with a direct gradient to every edge's own slot).

### Part 3b: flat head on the concat-trained conv's features (seed 42)

Reran Part 3 first to confirm reproducibility before building on it —
**bit-identical to the original run**: 0train=0.8559, 1test=0.9387,
mean=0.8973, exactly matching the first pass. Both folds' conv features
cached (`3792e819...`, `6393ffc1...`) via the always-on hook.

| readout on the `concat_h4`-trained conv's features | 0train | 1test | mean |
| --- | --- | --- | --- |
| concat_h4's own head (end-to-end, reference) | 0.8559 | 0.9387 | 0.8973 |
| flat head (`flat_direct_288`, same frozen conv) | 0.8573 | 0.9338 | **0.8955** |
| flat-on-flat-features (original flat control, reference) | 0.8949 | 0.9633 | 0.9291 |

**Swapping the readout does not recover flat's score.** Flat-on-concat-
features (0.8955) lands within 0.002 of `concat_h4`'s own end-to-end score
(0.8973) — per-fold, the two readouts are within 0.001-0.005 of each other
on both `0train` and `1test` — and both are ~0.033-0.034 below
flat-on-flat-features (0.9291). Changing *only* the classifier head on top
of an identical, already-trained conv barely moved the number at all.

This answers Part 3's open "why" directly: **it is not the aggregation step
at readout time.** The regression traces to what `dense_edge_conv` itself
learned when trained jointly with `concat_h4`'s gradient, not to how that
conv's (perfectly fine) features get pooled afterward. This is the "more
fundamental problem" branch flagged as a live possibility in Part 3's own
Results caveat above (concat's gradient, routed through per-node grouping
into a wide, largely-redundant classifier input, may simply be a harder
optimization target for the conv than flat's own direct, uniform gradient
to every edge's own slot) — now confirmed rather than merely speculated.

## Interpretation — capacity-confound verdict

**The last two write-ups' standing open question is answered: capacity is
NOT the bottleneck, and widening the graph pathway does not close the gap
to the flat control — at every tested point, matching or exceeding the flat
control's own dimensionality made the graph pathway worse, not better.**

The follow-up hypothesis (mean-pool's forced averaging, not dimensionality,
is the lossy step) found real support under frozen-feature screening
(concat closes ~2/3 of the remaining capacity-matched gap), but did **not**
survive honest end-to-end validation — the best graph-family design tested
across this whole two-day investigation (`concat_h4`) still lands at 0.8973
end-to-end, ~0.03-0.04 below every flat-family number on record
(0.9237-0.9291). Combined with
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]] (non-graph beats
graph on sparse events too) and the original flat-control finding (non-graph
beats graph on dense events, both ablations), this is now three independent
architecture families and two independent aggregation schemes all pointing
the same direction: **this pipeline's message-passing/aggregation layer is
a net cost relative to a flat readout of the same underlying features,
regardless of capacity, and regardless of whether aggregation is mean-pool
or concat.**

**Part 3b sharpens this further: the cost is upstream of the readout, in
what `dense_edge_conv` learns, not in how its output gets pooled.** Holding
`concat_h4`'s trained conv fixed and swapping in flat's own simple readout
recovered essentially nothing (0.8955 vs the conv's own concat readout at
0.8973 — both ~0.03 below flat-on-flat's 0.9291). If the aggregation step
itself were the bottleneck, this swap should have closed most of the gap,
the way concat's screening result initially suggested; it didn't. The
practical implication: further readout/aggregation-scheme experiments on
top of a graph-trained conv are unlikely to help much on their own — any
future graph-family attempt on this pipeline should treat "does joint
training with this head degrade the conv's representations" as the thing to
test first, not the readout design.

## Caveats

1. **Part 3 is single-seed (42), single-subject (1), by design** ("run it
   once", matching the flat control's own original scope) — a 4-seed
   end-to-end sweep of `concat_h4` is the obvious next step before treating
   0.8973 as more than a strong single-point signal, though it agrees in
   direction with the Part 2 mean-pool capacity trend and the standing
   "non-graph beats graph" pattern across three independent experiments
   now, which makes a reversal on more seeds less likely than usual.
2. **Every Part 2 `graph_*`/`concat_*` number uses a flat-trained frozen
   conv**, not a graph-trained one — see the Results caveat above. Part 2's
   numbers answer "does this downstream head read out flat-shaped features
   well," not "what is this downstream head's true joint-trained ceiling."
   Part 3 is the correction for the one design where it mattered most; the
   other five graph-family Part 2 numbers (graph_h8/16/32, concat_h1) were
   NOT re-validated end-to-end and could show a similar or different
   direction of drift.
3. Single subject (1, historically easy for this pipeline), untuned
   `dense_conv_*` hyperparameters, `coherence_threshold_mode="surrogate"`
   only — every caveat from the flat-control and dense-mode write-ups this
   follows up on still applies and compounds.
4. `ConcatGraphHead`'s node-0-always-zero property (in-degree 0 under the
   canonical dst-only scatter) is inherited from the production topology,
   not introduced here — flagged for visibility, not treated as a bug to
   fix in this run.
5. **Part 3b is also single-seed (42), same conv/subject/fold pair as Part
   3** — it isolates *where* the regression lives (conv representations,
   not readout) for this one trained checkpoint, not across seeds. A
   different seed's `concat_h4`-trained conv could in principle degrade
   less; nothing here rules that out. It also doesn't explain *why* concat's
   gradient degrades the conv (harder optimization target vs. a genuinely
   worse learning signal) — only that the readout swap rules out "it's just
   pooling."

## Concurrent-edit check

`sparse_evidence_gnn_classifier.py` and `run_dense_edge_flat_control.py`
were only ever READ from (`_build_dense_feature_conv`,
`SparseEvidenceGNNClassifier`, `DenseEdgeFlatClassifier`,
`CANONICAL_HELPER_KWARGS` imported unchanged), never modified, by either new
file this run added.

See [[sparse-evidence-gnn-dense-flat-control-beats-graph]],
[[sparse-evidence-gnn-dense-mode-beats-sparse]], and
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]] for the three prior
write-ups this one directly follows up on.
