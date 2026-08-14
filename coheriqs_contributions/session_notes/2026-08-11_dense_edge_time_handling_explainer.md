# Session notes — how `dense_edge` handles time (2026-08-11)

Branch: `channel_subset`. Repo: `/Users/noahshore/Documents/CoherIQs/moabb`.
Explainer, not a new run — asked how `--pipeline dense_edge`
([run_pipelines.py](../run_pipelines.py)'s `DENSE_EDGE_PARAMS`, currently
`event_mode="dense"`, `event_aggregation="concat"`,
`dense_edge_time_downsample=8`, `dense_edge_temporal_mode="conv"`) treats
the trial's time axis. No code changed; this just traces the existing
mechanism through
[sparse_evidence_gnn_classifier.py](../moabb_pipelines/sparse_evidence_gnn_classifier.py).

---

## Short answer

`dense_edge` collapses the entire trial's time axis down to **one static
feature vector per edge, per forward pass** — there is no timestep-level
structure left by the time evidence reaches `sparse_message_mlp`. It gets
there in two stages: a cheap, non-trainable, fixed-factor downsample done
once per trial at precompute time, then the trainable `dense_edge_conv`
itself finishes the collapse to 1 every forward call. This is a deliberate
design point on a spectrum this file has explicitly built out and ablated
(`dense_edge_time_downsample`, `time_averaged_graph`,
`dense_edge_temporal_mode="rnn"`, `event_mode="temporal_graph"` all vary
*how much* and *how* time gets discapsulated) — see
[[sparse-evidence-gnn-time-averaged-graph-feature]] and
[[sparse-evidence-gnn-time-averaged-flat-control-result]] for the finding
that motivated pushing on this axis at all: collapsing time to 1 explicitly
(`time_averaged_graph=True`) ties or beats the time-resolved baseline.

## Stage 1 — non-trainable precompute (`_build_dense_edge_input`, once per trial)

Runs the *same* cross-spectrum + smoothing math sparse mode's
`_coherence_only` uses (native resolution, `smooth_kernel_size=(5, 3)`),
so the surrogate null distribution stays valid for dense mode too. Instead
of thresholding into a discrete event list, it keeps the full
`[coherence, sin(phase), cos(phase), significance]` stack at shape
`[B, 4, E, T, F]`, post-COI-mask (zeroed outside the cone of influence,
values not a discrete drop —
[sparse_evidence_gnn_classifier.py:2272-2276](../moabb_pipelines/sparse_evidence_gnn_classifier.py#L2272-L2276)).

With `dense_edge_time_downsample=8` (current `DENSE_EDGE_PARAMS`), `T` is
then average-pooled by that factor via `avg_pool2d(kernel=(1,8),
stride=(1,8))` — **after** smoothing and **after** the COI mask, at native
resolution, not before (contrast `cwt_resample_n_time`, which resamples
raw complex CWT coefficients pre-coherence and breaks the COI mask's
timing assumptions entirely — see
[sparse_evidence_gnn_classifier.py:909-927](../moabb_pipelines/sparse_evidence_gnn_classifier.py#L909-L927)).
Trailing remainder timesteps that don't fill a full window are dropped.
This whole stage runs once per trial and is cached
(`SparseEvidenceGNNClassifier._precompute_dense_edge_inputs`) — it costs
nothing extra per epoch, unlike the conv below.

At native `T~1001` (250Hz), `dense_edge_time_downsample=8` leaves
`T' ~ 125` going into the conv.

## Stage 2 — trainable `dense_edge_conv` (`_dense_edge_features`, every forward() call)

Frequency is folded into the conv's input channels
(`[B, 4, E, T', F] -> [B, 4*F, E, T']`) before `dense_edge_conv` ever sees
it, so the conv is free to learn cross-frequency combinations; the edge
axis (`E`) stays `kernel=stride=1` throughout, meaning **all edges are
convolved independently with shared weights** — same pattern
`ChannelSignalEncoder` uses across channels.

Current config (`dense_edge_temporal_mode="conv"`,
`_build_dense_feature_conv`) is two `Conv2d(kernel=(1,5)) -> GELU ->
MaxPool2d(kernel=(1,4))` blocks back to back, ending in
`AdaptiveAvgPool2d((None, 1))` — an unconditional full-average-pool of
whatever timesteps survive down to exactly 1, regardless of how many are
left. Two `MaxPool2d(pool_size=4)` stages alone already take `T'~125` down
by ~16x before that final pool runs — so most of the real compute
(processing full `T'` through two real `Conv2d` layers) is spent on an axis
that's about to be discarded anyway. That's *why*
`dense_edge_time_downsample` exists: moving as much of that reduction as
possible upstream, non-trainable and once-per-trial, instead of paying for
it inside the conv every epoch
([sparse_evidence_gnn_classifier.py:885-895](../moabb_pipelines/sparse_evidence_gnn_classifier.py#L885-L895)).

The conv's output (`[B, dense_conv_out_channels=8, E, 1]`) is squeezed and
repackaged into the same `(events_padded, src_padded, dst_padded, ...)`
shape sparse mode's real event lists use — every one of the `E` canonical
edges is "valid" every trial (no notion of an edge not firing, unlike a
sparse event list that can be empty on some edges), so `max_count == E`
and the downstream graph machinery (`sparse_message_mlp`,
`_aggregate_events` with `event_aggregation="concat"`, hops,
`sparse_classifier`) is entirely shared/unmodified code — only event
*building* differs, per `event_mode`'s own `__init__` docstring.

## What this pipeline does NOT do (siblings that handle time differently)

- **`dense_edge_gru`** (`dense_edge_temporal_mode="rnn"`,
  `_DenseEdgeGRUTemporal`): same Stage 1, but Stage 2 replaces the
  Conv2d+pool stack with a per-edge `nn.GRU` (weight-shared across edges,
  edges folded into the batch dim) that consumes the *full* `T'` sequence
  with memory instead of Conv2d's small fixed local kernel, using the GRU's
  final hidden state as the "T pooled to one value" summary. Still
  collapses to one vector per edge in the end — just via sequential memory
  instead of local conv+pool. See
  [[sparse-evidence-gnn-dense-edge-gru-temporal-mode]] — implemented +
  smoke-tested, not yet run for real accuracy.
- **`evolving_graph`** (`event_mode="temporal_graph"`): the only genuinely
  different mechanism. Never pools per-edge in isolation before the graph
  sees it — a small `temporal_edge_proj` produces a per-edge embedding at
  *every* surviving timestep, those feed `sparse_message_mlp` +
  `event_aggregation="mean"` (required) at each timestep to get one
  aggregated per-node vector per timestep, and a single node-shared
  `nn.GRU` walks that node-state sequence forward with a persistent hidden
  state — evidence genuinely evolves timestep-by-timestep across the graph,
  not just across time within one static edge feature. Scored 0.8203
  subj1/seed42, below flat/dense-mean-pool/dense-concat — see
  [[sparse-evidence-gnn-temporal-graph-gru-underperforms]].
- **`time_averaged_graph=True`** (not currently wired into any
  `run_pipelines.py` pipeline key, available as a `--param-names` override):
  the extreme end of `dense_edge_time_downsample` — collapses `T` to
  exactly 1 at Stage 1 itself, via a COI-valid-weighted average (not a
  plain mean, which would bias low frequencies toward zero given their
  wider cones) rather than a fixed pooling factor. Requires
  `dense_conv_kernel_size == dense_conv_pool_size == 1` since there's
  nothing left for a `k>1` conv to convolve over. See
  [[sparse-evidence-gnn-time-averaged-graph-feature]].

## Open thread

`dense_edge`'s current `dense_edge_time_downsample=8` was chosen for speed
(cutting the conv's dominant per-epoch cost roughly linearly in the
factor), not validated against `1` (native) or other factors for accuracy
specifically on the `concat`-aggregation config `DENSE_EDGE_PARAMS` now
uses — the sibling `time_averaged_graph` ablation numbers
([[sparse-evidence-gnn-time-averaged-graph-feature]]) were run on a
different aggregation/threshold config, so they're suggestive (time
resolution may not matter much here) but not a direct A/B for this exact
pipeline.
