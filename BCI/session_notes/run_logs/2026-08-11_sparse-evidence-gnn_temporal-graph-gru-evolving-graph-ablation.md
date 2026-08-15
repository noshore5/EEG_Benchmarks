# Sparse-Evidence-GNN `event_mode="temporal_graph"` — a genuine evolving-graph test — 2026-08-11

## Why

Two existing mechanisms in this pipeline sound like they test "does time/depth
matter" but neither actually walks forward through time:

- `event_mode="dense"` pools the whole T' axis away in one shot
  (`dense_edge_conv`'s Conv2d stack + `AdaptiveAvgPool2d`) before the graph
  ever sees more than one vector per edge.
- `n_hops>1` adds DEPTH — extra rounds of message passing ACROSS THE GRAPH
  within one already-pooled snapshot — never touching time at all.

`event_mode="temporal_graph"` (new, this run) is the missing third axis: a
mechanism that genuinely updates each node's state as new evidence arrives,
timestep by timestep, distinct from both of the above.

## Implementation

New `event_mode: Literal["sparse", "dense", "temporal_graph"]` value on
`SparseEvidenceGNNCore`/`SparseEvidenceGNNClassifier`
(`sparse_evidence_gnn_classifier.py`). Default `"sparse"` unaffected; `"dense"`
bit-identical to before. `"temporal_graph"`:

- Reuses `_build_dense_edge_input`'s exact precomputed, non-trainable
  `[B, 4, E, T, F]` stack `"dense"` mode consumes (same
  `dense_edge_time_downsample`, COI mask, smoothing, surrogate/fixed
  threshold math — unchanged, shared infrastructure).
- `temporal_edge_proj`: a single small `Linear(4*nfreqs, temporal_graph_edge_dim)`
  + GELU (NOT the deep two-block Conv2d stack `dense_edge_conv` uses) folds
  frequency into a per-edge embedding at EVERY surviving timestep, vectorized
  over `(edge, time)` in one call.
- Those per-timestep embeddings feed `sparse_message_mlp` (the SAME shared
  weights every other `event_mode` uses) and are aggregated to per-node
  evidence via the EXISTING "mean" aggregation, generalized with an extra `T`
  axis — applied at every timestep. `event_mode="temporal_graph"` requires
  `event_aggregation="mean"` (raises otherwise) — deliberately NOT "concat",
  to keep this experiment isolated to the temporal question rather than
  compounding it with the (separately unresolved) aggregation question.
- The resulting per-node sequence feeds a single `nn.GRU`, weight-shared
  across nodes (nodes folded into the batch dim) — a genuinely persistent
  hidden state updated timestep by timestep, unlike `n_hops>1`'s `GRUCell`
  rounds (which never see more than one already-pooled snapshot).
- The GRU's final hidden state per node IS `evidence`, in the EXACT
  `[B, n_channels, hidden_dim]` shape `_aggregate_events`' "mean" branch
  already returns — so `n_hops>1` propagation (orthogonal: across-the-graph
  depth on top of this mode's own across-time propagation) and
  `sparse_classifier` are shared code, not a duplicated path. See
  `_temporal_graph_node_states` in `sparse_evidence_gnn_classifier.py`.
- `freq_aware_hops=True` rejected together with this mode, same reasoning as
  `"dense"`: `temporal_edge_proj` folds the whole frequency axis into its
  input, so there's no discrete per-event frequency bin left to index.

New tracked script:
[`run_temporal_graph_gru.py`](../../tests/run_temporal_graph_gru.py).

## Smoke test

Direct `SparseEvidenceGNNCore` forward/backward on random `[B=4, 4, E=36,
T=40, F=16]` input:
- `temporal_edge_proj`, `temporal_node_gru`, `sparse_message_mlp`,
  `sparse_classifier`, `channel_encoder` all received nonzero gradients.
- `temporal_graph` + `n_hops=3`: `hop_update` also received nonzero gradient
  (across-time and across-graph propagation compose correctly).
- Constructor validation: `event_aggregation="concat"` and
  `freq_aware_hops=True` both correctly rejected together with
  `event_mode="temporal_graph"`.

Full `fit`/`predict` round-trip on tiny synthetic data (12 trials, 22
channels) for both `"dense"` (regression check) and `"temporal_graph"`:
both trained (loss decreased). `"temporal_graph"` needed far more optimizer
steps than `"dense"` to show it on this tiny set — a direct core-level check
at this classifier's own `lr=1e-3` found loss sitting flat near `ln(2)=0.693`
through roughly the first 100 steps, then dropping sharply from ~0.69 to
<0.01 by step ~150-275. This is a real, reproducible slow "warm-up" property
of this GRU-based path (plausibly standard recurrent-net dynamics, compounded
by `grad_clip_norm=0.1` being tight relative to the small gradients this path
produces before the GRU's hidden state finds useful dynamics) — NOT a
stalled/broken gradient path (confirmed by the direct gradient check, which
shows nonzero gradient from step 1). The smoke test's `temporal_graph` epoch
count was raised to 150 (from `"dense"`'s 8) to clear this window.

## Method (real-data run)

`SparseEvidenceGNNClassifier(event_mode="temporal_graph", event_aggregation="mean",
dense_edge_time_downsample=8, ...)`, otherwise the EXACT config behind the
`"flat"` (0.9291) and `"dense"` mean-pool (0.8886) comparison numbers below —
copied verbatim from `run_dense_edge_flat_control.py`'s own
`CANONICAL_HELPER_KWARGS` (`phase_threshold_deg=10`, `surrogate_percentile=99`,
`coherence_threshold_mode="surrogate"`, `channel_subset=[1,5,7,8,9,10,11,13,17]`,
`epochs=75`, `batch_size=8`, `learning_rate=1e-3`, `weight_decay=1e-4`,
`grad_clip_norm=0.1`, `hidden_dim=8`, `channel_embed_dim=8`,
`channel_encoder_dilation=5`, `feature_ablation="zero_channel_embed"` — the
event pathway alone, matching the cleanest existing comparison point).
`dense_edge_time_downsample=8` is the one addition on top of that shared
config (neither source run had this param yet) — this task's own guidance to
keep GRU sequence length tractable; 8 matches the value already validated as
helpful for the dense/concat Conv2d path elsewhere in this pipeline.

Subject 1 only, `BNCI2014_001`, `LeftRightImagery(fmin=8, fmax=35)`,
`CrossSessionEvaluation`, seed 42 only (matches the flat control's own single
seed, and is one of the four seeds averaged into the dense mean-pool 0.8886
number) — per this run's own "run it once, then decide" scope.

**Expected wall-clock, flagged before running**: guessed 400-900s, reasoning
that GRU processing can't parallelize across time the way Conv2d can, even
with the 8x downsample. Actual: **65.6s** — dramatically faster than guessed,
and much faster than `"dense"` mode's own recorded ~450-530s/seed at native
resolution. See Caveats — this is NOT a clean apples-to-apples wall-clock
comparison (confounded by `dense_edge_time_downsample`, see below), so it
should not be read as "temporal_graph is inherently faster than dense."

## Results

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.8358 | 0.8048 | **0.8203** |

Wall-clock: 65.6s (both `CrossSessionEvaluation` folds together).

## Comparison table (subject 1, seed 42 where available)

| configuration | seed 42 | mean (seeds 42-45) | event-building mechanism |
| --- | --- | --- | --- |
| Flat (non-graph control on dense features) | **0.9291** | *(1 seed)* | Conv2d, T pooled, no graph |
| Dense GNN, mean-pool, event-only | 0.8893 | 0.8886 | Conv2d, T pooled, graph (mean) |
| Dense GNN, concat, event-only, end-to-end | 0.8973 | *(1 seed)* | Conv2d, T pooled, graph (concat) |
| **temporal_graph, event-only (this run)** | **0.8203** | *(1 seed)* | GRU, T walked step-by-step, graph (mean) |

`temporal_graph` scores **below every other configuration on record** for
this ablation series — ~0.07 below dense mean-pool (its closest, most
apples-to-apples comparison: same graph backend, same "mean" aggregation,
same `zero_channel_embed` ablation, only the event-building/temporal
mechanism differs), and ~0.11 below the flat control.

## Interpretation

This is the sharper of the two possible outcomes this ablation was set up to
distinguish: it does not merely fail to show evolving-graph propagation
helps, it finds evidence the mechanism costs something relative to simply
pooling time away (`"dense"`) or discarding the graph entirely (`"flat"`).
Consistent with the broader throughline this whole ablation series has been
finding
([[sparse-evidence-gnn-channel-encoder-dominates]],
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]],
[[sparse-evidence-gnn-dense-flat-control-beats-graph]],
[[sparse-evidence-gnn-time-averaged-graph-feature]]): this pipeline's real
discriminative signal in the dense event pathway looks like it's well
summarized by a single scalar-ish per-edge/per-frequency quantity, and
architecture that spends more compute trying to extract additional structure
from time or topology — whether a deep conv stack, multi-hop graph
propagation, or now a genuinely time-stepping GRU — has consistently NOT
paid for itself on this subject/config. A plausible (not confirmed)
mechanism specific to this result: 300 real optimizer steps (75 epochs x ~4
steps/epoch on this fold's train split) is well inside the "slow warm-up"
window the smoke test found needs ~150-300 steps just to start dropping
sharply on a *trivial* 12-trial synthetic task — on the real, much harder
9-way classification problem, the GRU pathway may simply not have gotten far
enough into its own useful-dynamics regime within `epochs=75` to compete with
`"dense"`'s faster-converging Conv2d.

Per the task's own "run it once, then decide" framework: single-seed
0.8203 is clearly BELOW every comparison point, not "promising enough to
justify" a multi-seed follow-up. **Not extending to multi-seed on this
result.**

## Caveats

1. **Single seed, single subject.** Same standing caveat as every other
   entry in this series — see [[sparse-evidence-gnn-seed-variance]].
2. **Wall-clock comparison is confounded by `dense_edge_time_downsample`.**
   The 65.6s number ran at `dense_edge_time_downsample=8` (T'~125); the
   "dense mean-pool 0.8886" baseline it's informally compared against ran at
   `dense_edge_time_downsample=1` (native T'~1001, that param didn't exist
   yet when that number was produced). A fair same-downsample wall-clock
   comparison (`event_mode="dense"` also at `dense_edge_time_downsample=8`)
   was not run in this pass — the 65.6s figure should be read as "this
   config trains fast," not as "GRU processing is inherently cheaper than
   Conv2d processing."
3. **The "slow warm-up" property is untested for whether more epochs would
   close the accuracy gap.** The smoke test showed the GRU pathway needs
   many more optimizer steps than Conv2d to leave its near-`ln(2)` plateau on
   a trivial synthetic task; whether `epochs=75` on the real task landed
   before or after that pathway's own equivalent plateau was not directly
   checked (e.g. by inspecting `train_loss_history_`). An `epochs` sweep
   would be the natural next step before concluding the architecture itself
   (rather than under-training) explains the accuracy gap — but per the
   "run it once, then decide" framework, this is flagged rather than
   pursued here.
4. **`temporal_graph_edge_dim=8` (default, untuned)** — matches
   `dense_conv_out_channels`'s own default purely for comparable sizing, not
   because either has been tuned against the other.
5. Same-seed comparison to "flat" (0.9291) and "dense mean-pool" (0.8893 at
   seed 42) is clean (identical `CANONICAL_HELPER_KWARGS` config, same seed);
   the "dense concat, end-to-end" 0.8973 comparator is a DIFFERENT
   aggregation mode (`"concat"`, not `"mean"`) and a different exact config
   lineage (`sparse-evidence-gnn-concat-productionized-cross-subject`), shown
   for reference only, not a clean isolation of the temporal question.

## Concurrent-edit check

`sparse_evidence_gnn_classifier.py` was actively edited by this same session
to add `event_mode="temporal_graph"` — not a concurrent-edit risk here since
this IS that edit. `run_sparse_evidence_gnn.py` was not touched or read for
config (per this run's own `CANONICAL_HELPER_KWARGS`, hardcoded from
`run_dense_edge_flat_control.py`, same precedent every prior script in this
line of investigation uses).

See [[sparse-evidence-gnn-dense-mode-beats-sparse]],
[[sparse-evidence-gnn-dense-flat-control-beats-graph]],
[[sparse-evidence-gnn-concat-productionized-cross-subject]], and
[[sparse-evidence-gnn-dense-edge-gru-temporal-mode]] (the other GRU-based
variant in this pipeline, per-edge rather than per-node, also not yet a
clear win) for the comparison points this run is measured against.
