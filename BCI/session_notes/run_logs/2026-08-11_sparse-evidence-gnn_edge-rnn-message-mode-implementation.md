# Sparse-Evidence-GNN edge-level RNN message mode — implementation — 2026-08-11

## REVERTED 2026-08-11

All code described below was removed the same day, before any real-data
result existed. `dense_recurrent_gnn.py` was deleted; every
`event_message_mode`/`shuffle_time_order`/`edge_idx_padded`/`edge_rnn`
change to `sparse_evidence_gnn_classifier.py` (and its debug-script/
run-script call sites) was reverted back to the pre-RNN state. Reason: the
comparison this was built to support kept getting confounded by leftover
config state (`n_hops`, `channel_subset`, stale toggle values) in the
shared `run_sparse_evidence_gnn.py` config, producing a string of
confusing, hard-to-interpret single-subject results (0.48, 0.57, 0.61 mean)
that took multiple rounds of ledger forensics to explain -- by the time a
clean, fully-explicit comparison script was finally built
(`run_edge_rnn_vs_mlp.py`, now also deleted), the user judged the whole
effort too messy to be worth continuing and asked for a full revert rather
than a fourth attempt at a clean run. No real-data evidence for or against
"does event order help" was ever obtained. If this direction is revisited,
start from a single fully-explicit script (no toggles, no shared mutable
config) from the very first run, not after three confounded ones.

## Why (as originally written, before the revert)

`dense_recurrent_gnn.py` (an earlier design sketch, not previously wired
into the real pipeline) proposed replacing `event_mode="sparse"`'s per-event
message step with a per-edge one: instead of `sparse_message_mlp` treating
every event on an edge independently and `_aggregate_events` summing/
averaging/softmaxing them (order-agnostic — a channel's evidence is
identical whether its events arrived in one order or the reverse), each
edge's own events, sorted by time, get run through a GRU
(`EdgeEventRNNAggregator`) whose final hidden state becomes that edge's one
message. This gives the model genuine capability to use event ORDER and
TEMPORAL SPACING on an edge, which the existing MLP path structurally
cannot represent (it only ever sees one event's own scalar timestamp, never
a neighboring event's).

The module's own design notes flagged the risk up front: any ordered-vs-
baseline win could just be "a GRU has more parameters than concat+MLP," not
"sequence order carries signal" — so a mandatory shuffled-order negative
control (identical architecture, scrambled per-edge event order) was
required from the start, not added after a promising result.

## What was found before implementing

The sketch's own `build_padded_edge_sequences` helper assumed events arrive
pre-bucketed per edge (`[B, E, L, F]` with a per-`(batch, edge)` length).
What `_build_sparse_events` actually produces is flat PER-TRIAL padding
instead — `events_padded [B, max_count, 5]`, `dst_padded`/`src_padded`
`[B, max_count]`, `valid_mask` — one ragged list per trial, not grouped by
edge, and critically no edge-index field was kept (the edge index is
computed transiently while building `dst_node`/`src_node`, then discarded).
Demonstrated concretely with a small hand-built example before writing any
production code (5 events on a 4-channel/6-edge graph) — see this session's
transcript for the before/after tensors.

## What was built

1. **`edge_idx_padded`**: new padded field alongside `freq_idx_padded` in
   `_build_sparse_events`/`compute_events`/`_precompute_sparse_events`/
   `_prepare_features`/`forward()` (and a trivial `arange(E)` counterpart in
   dense mode's `_dense_edge_features`, for call-site parity even though
   `event_message_mode="rnn"` is rejected in dense mode). Threaded through
   every touchpoint the pre-existing `freq_idx_padded` already passes
   through — `_prepare_features`'s generic tuple-based
   `to_float_tensors`/`TensorDataset` machinery in `common.py` needed no
   changes, it already handles an arbitrary-length feature tuple.

2. **`_build_edge_sequences_from_events`** (new `SparseEvidenceGNNCore`
   method): re-buckets the flat per-trial event list into the per-edge,
   time-sorted `[B, E, L, 5]` + `[B, E]` shape the RNN needs. Fully
   vectorized (no per-event Python loop) via one sort + one groupby-cumcount
   trick: sort every trial's events by a single `edge_idx + timestamp`
   composite key (timestamp already normalized to `[0, 1)`, clamped to
   `[0, 0.999]` to avoid boundary collision with the next edge), then derive
   each event's rank-within-its-edge via a cummax-based groupby-cumcount,
   then scatter directly into the padded output via advanced-index
   assignment (same pattern `_build_sparse_events` itself already uses).
   Unit-tested against a hand-computed 5-event/6-edge example — exact match,
   including that unsorted-within-edge events (edge 0's `0.55` then `0.12`)
   come out correctly time-ordered.

3. **`event_message_mode: Literal["mlp", "rnn"] = "mlp"`** and
   **`shuffle_time_order: bool = False`**, added to both
   `SparseEvidenceGNNCore.__init__` and `SparseEvidenceGNNClassifier.__init__`
   (classifier duplicates the cross-checks at construction time, matching
   every other option's own precedent). `"mlp"` (default) is the pre-existing
   behavior, bit-identical — `forward()`'s original per-event code path is
   preserved verbatim, now under an `else` branch. `"rnn"` builds the
   per-edge sequences, runs `edge_rnn` (with `shuffle_time_order` forwarded
   straight into `EdgeEventRNNAggregator.forward`, exactly as that module's
   own docstring specifies), concatenates the result with src/dst channel
   embeddings, runs `edge_message_mlp`, then feeds the existing
   `_aggregate_events` unchanged — reusing dense mode's own "always-present,
   canonical edge axis" trick (`dst_padded` = `self.dst_idx` broadcast,
   `valid_mask` all-True) since `edge_rnn`'s learned `no_event_embedding`
   fallback means every edge always has *some* message, real or not, same
   guarantee dense mode already relies on.

4. **Validation** (both classes): `event_message_mode="rnn"` requires
   `event_mode="sparse"` (dense mode has already collapsed each edge to one
   summary vector, no per-event list to sequence-model); currently only
   supports `event_aggregation="mean"` (`"gated_softmax"` would need its own
   differently-shaped `event_gate`; `"concat"` plausibly composes — RNN
   output is canonical-edge-axis just like dense mode's — but is untested,
   rejected explicitly rather than silently allowed); rejects
   `freq_aware_hops=True` (RNN output has no single-frequency identity left,
   same rationale as the existing dense-mode rejection); `shuffle_time_order`
   is a no-op (rejected) unless `event_message_mode="rnn"`.

5. `edge_rnn`/`edge_message_mlp` are built LAST in `__init__` (after
   `dense_edge_conv`'s own block), and only when `event_message_mode="rnn"`
   — same "constructed after every pre-existing submodule, only when
   actually used" precedent as `dense_edge_conv`/`event_gate`/
   `hop_message_mlp_freq`, specifically to avoid the RNG-draw-order init-
   shift bug documented in
   [[sparse-evidence-gnn-event-gate-init-shift-bug]]. At the default
   `event_message_mode="mlp"`, neither submodule is constructed, so every
   other submodule's random init is bit-identical to before this feature
   existed.

## Testing performed

All synthetic/smoke-level (no real EEG data run yet):

- Error paths: `rnn`+`dense`, `rnn`+`gated_softmax`, `rnn`+`concat`,
  `rnn`+`freq_aware_hops`, `shuffle_time_order=True` without `rnn`, invalid
  `event_message_mode` string — all raise `ValueError` at both the Core and
  Classifier construction level.
- `_build_edge_sequences_from_events` unit-tested against a hand-computed
  5-event/6-edge example (exact match) and an all-invalid-batch edge case
  (no crash, `L=1`, all lengths 0).
- End-to-end `fit()`/`predict_proba()` smoke tests on small synthetic data
  (12-16 trials, 4 channels): default `"mlp"` mode, `"rnn"` mode with
  `shuffle_time_order` both `False` and `True` — all clean, no NaNs.
- Regression check on pre-existing, untouched code paths after this change:
  `gated_softmax` (mlp mode), `dense`+`concat`, and `rnn`+`n_hops=2` (hops
  are NOT rejected for `rnn`, only the `freq_aware_hops` variant is) all
  still fit/predict cleanly.
- Confirmed the shuffle control actually does something: with a fixed seed,
  `shuffle_time_order=False` is deterministic run-to-run; `shuffle_time_order
  =True` gives different output for different shuffle seeds
  (max-abs-diff ~0.44 on a synthetic 3-event busy edge) and differs from the
  ordered run (~0.066) — the control is mechanically capable of showing a
  real effect if the GRU is using order, not silently a no-op.

## Status / Caveats

1. **Completely untested on real EEG data.** Every check above is
   synthetic/shape/math-level. No accuracy number exists yet for
   `event_message_mode="rnn"` in either direction.
2. Per the module's own design note and this option's docstring: any real
   result MUST be reported alongside its `shuffle_time_order=True` control,
   not on its own — an ordered-vs-`"mlp"` delta alone can't distinguish
   "order carries signal" from "a GRU just has more parameters than
   concat+MLP" (`"mlp"` mode's `sparse_message_mlp` is a 2-layer perceptron
   applied per event; `edge_rnn` is a GRU with its own weight matrices, a
   materially different parameter budget).
3. `event_aggregation="concat"` + `event_message_mode="rnn"` is explicitly
   rejected, not because it's known invalid, but because it isn't
   implemented/validated by this landing — see the docstring's own note
   that the shapes plausibly compose (both produce an always-present,
   canonical-edge-axis message).
4. Not wired into `run_sparse_evidence_gnn.py` — left as a classifier-level
   option only, consistent with this session's pattern of landing new
   options on the classifier first and letting the user opt them into the
   canonical driver script when ready to actually run them.

## Concurrent-edit check

`run_sparse_evidence_gnn.py` was not touched by this change and continues
to be live-edited in the IDE independently of this work.

See [[sparse-evidence-gnn-edge-rnn-message-mode]] in project memory for the
condensed pointer, and `dense_recurrent_gnn.py`'s own module docstring for
the original design rationale this implements.
