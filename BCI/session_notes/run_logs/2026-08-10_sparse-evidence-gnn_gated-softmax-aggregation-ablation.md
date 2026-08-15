# Sparse-Evidence-GNN gated-softmax event aggregation vs. mean-pooling — 2026-08-10

- Run IDs: `agg-{mean,gated_softmax}-{none,zero_event_features}-s{42,43,44}`
  (12 runs) + two untracked ad-hoc fits for the entropy and message-content
  diagnostics below (no run-id, not a CV evaluation -- just `.fit()` on the
  `0train` fold).
- Dataset: BNCI2014-001, subject 1 only, `LeftRightImagery` paradigm,
  `CrossSessionEvaluation`.
- Base config: canonical `SPARSE_EVIDENCE_GNN_PARAMS` (epochs=75, see
  [[sparse-evidence-gnn-reorg-2026-08-09]]), with `seed`,
  `event_aggregation`, and `feature_ablation` varied.

## Correction (2026-08-10, after this sweep ran): silent init-shift bug

`event_gate` was originally constructed unconditionally in `__init__`,
regardless of `event_aggregation`, positioned BEFORE `sparse_classifier`.
Since `nn.Linear` draws its random initial weights from the global RNG at
construction time, this silently shifted `sparse_classifier`'s own initial
weights for every `"mean"`-mode model too -- confirmed directly by replaying
the same construction order with/without that line under an identical seed
and diffing `sparse_classifier`'s weights (they differed). Fixed by
constructing `event_gate` only when `event_aggregation="gated_softmax"`
(verified this restores `"mean"` mode's initialization to true
pre-`event_aggregation`-change behavior, bit-for-bit).

**This does NOT invalidate the mean-vs-gated_softmax comparison below**:
every run in this sweep used the (buggy-but-internally-consistent)
unconditional-construction code, so `"mean"` and `"gated_softmax"` runs
here were compared on equal footing -- the shift applied identically to
both. It DOES mean the absolute numbers below aren't directly comparable to
`"mean"`-only baselines computed under the truly-original code (e.g. the
seed=42/epochs=100 baseline in [[sparse-evidence-gnn-channel-encoder-dominates]],
predates `event_aggregation` entirely). More importantly: this bug was live
in the canonical pipeline (`event_aggregation` defaults to `"mean"`) for
however long it took to notice, meaning it's a real, structural candidate
explanation for any accuracy drift observed on canonical runs during that
window, on top of/instead of any deliberate parameter change (e.g. the
epochs=75 switch) made around the same time. See
[[sparse-evidence-gnn-event-gate-init-shift-bug]].

## Motivation

[[sparse-evidence-gnn-channel-encoder-dominates]] found accuracy comes
almost entirely from `ChannelSignalEncoder`; zeroing event features costs
essentially nothing. One candidate explanation: the event pathway's
aggregation was a flat, unweighted mean over every event landing on a
destination channel (`scatter_add` then divide by count) -- there was no
mechanism for one event to matter more than another, only for
`sparse_message_mlp`'s output magnitude to be diluted by however many other
events happened to land on the same channel. Implemented `event_aggregation`
("mean" (original) vs. "gated_softmax", a new `event_gate: Linear(message_in,
1)` head, softmax-normalized per (trial, destination-channel) group via a
manual scatter-softmax) to test whether that structural limitation was
actually suppressing the event pathway's usefulness.

## Correctness (unit-level, before the real-data run)

Verified directly on synthetic events, not just end-to-end forward passes:
- Weights sum to exactly 1 per group (checked via an all-ones-message probe
  -- 2-event and 1-event groups both land on exactly 1.0).
- Empty groups (0 valid events) stay at exact zero.
- An invalid (padding) event with an extreme message value (999) is fully
  excluded, not just down-weighted.
- No NaNs, including a trial with zero valid events anywhere.
- The result is a genuine input-dependent convex combination, not a
  disguised average (probe messages [10, 0, 5] combined to 6.86, not 5.0).
- Gradients flow through `event_gate`, `sparse_message_mlp`, and
  `channel_encoder` correctly; `"mean"` mode is bit-for-bit unaffected by the
  refactor (regression-checked against the pre-change forward pass).

## Results: accuracy, mean vs. gated_softmax x none vs. zero_event_features

| aggregation | ablation | mean | std | n |
| --- | --- | --- | --- | --- |
| mean | none | 0.8405 | 0.0143 | 3 |
| mean | zero_event_features | 0.8345 | 0.0233 | 3 |
| gated_softmax | none | 0.8400 | 0.0139 | 3 |
| gated_softmax | zero_event_features | 0.8330 | 0.0224 | 3 |

Per-seed detail:

| aggregation | ablation | seed42 | seed43 | seed44 |
| --- | --- | --- | --- | --- |
| mean | none | 0.8240 | 0.8495 | 0.8480 |
| mean | zero_event_features | 0.8091 | 0.8548 | 0.8394 |
| gated_softmax | none | 0.8239 | 0.8472 | 0.8488 |
| gated_softmax | zero_event_features | 0.8093 | 0.8540 | 0.8356 |

`gated_softmax/none` vs `mean/none`: statistically indistinguishable
(0.8400 vs 0.8405; per-seed differences all <=0.002). The
`zero_event_features` cost is ~0.006 under mean-pooling and ~0.007 under
gated softmax -- both well inside the ~0.014-0.023 seed-to-seed std, i.e.
no real difference. Only 3 seeds/cell -- enough to rule out a large effect,
not a small one, but there is no large effect here to find.

## Why: event_gate converged to near-uniform weights

Trained one `event_aggregation="gated_softmax"` model on the real `0train`
fold (seed 42, otherwise canonical config) and recomputed the exact
scatter-softmax math externally against its trained `event_gate` and real
precomputed events (1296 (trial, destination-channel) groups across 144
trials x 9 channels; 1152 of them had >=2 valid events).

- entropy(actual weights) / entropy(uniform over that group's event count):
  mean=0.9932, std=0.0034, min=0.9746, max=0.9981 across all 1152 groups
  (1.0 = indistinguishable from uniform).
- Raw weight std (all valid events, pooled): 0.00543, against a uniform
  baseline of ~0.0055 for the median group size (181 events).

Despite the mechanism being verified capable of concentrating weight
sharply on a subset of events (see the correctness section above), training
converged to weights that are within ~1% of perfectly uniform almost
everywhere. Given the freedom to differentiate events and every incentive
(gradient signal) to do so if it helped, the model chose not to.

## Follow-up: is it the gate's weakness, or genuinely nothing to weight?

`event_gate` was a bare `nn.Linear(message_in, 1)` -- no hidden layer, no
nonlinearity, softmax-ing over a median of 181 competing events per group.
Its convergence to near-uniform weights doesn't by itself rule out a real
but *nonlinear* importance signal existing that a weak one-layer probe
couldn't express, or gradient dilution from softmax-ing over that many
competing items. Ran a gate-independent check instead: does
`sparse_message_mlp` itself (the real 2-layer, GELU MLP every event's
message goes through, present regardless of aggregation mode) actually
depend on an event's own content? Trained one `event_aggregation="mean"`
model (the canonical/deployed default) on real subject-1 `0train` data,
then measured, over all 187,754 valid events:

- `||msg(real) - msg(event content zeroed)|| / ||msg(real)||` = **0.0731**
  -- zeroing an event's own `[t, freq, mag, sinphi, cosphi]` barely moves
  its message.
- `||msg(real) - msg(channel embeddings zeroed)|| / ||msg(real)||` = **1.0265**
  -- zeroing `src_emb`/`dst_emb` changes the message completely.
- Within-(trial, destination-channel)-group variance of messages (different
  events landing on the same channel in the same trial) is **5.7%** of
  total variance; the other 94.3% is between-group (i.e. explained by which
  channel/trial, not which event).

This is stronger and more direct evidence than the gate-entropy check,
since it doesn't depend on any gate's capacity: `sparse_message_mlp` itself
has learned to route its output almost entirely through channel identity
and largely ignore each event's own content, independent of aggregation.
No weighting mechanism -- however expressive -- can route around messages
that already don't encode much about the individual event they came from.

## Conclusion

The event pathway's low contribution to accuracy is **not** explained by an
inability to express "this event matters more than that one" -- it can, and
doing so doesn't help. The message-content check narrows this further,
independent of any gate's own capacity: `sparse_message_mlp` has learned to
largely ignore an event's own `[t, freq, mag, sinphi, cosphi]` and encode
almost entirely which channel it's on. Events post-surrogate-gating do
reach the model with real per-event content -- the ablation numbers
elsewhere in this repo confirm the gate/threshold machinery is doing real,
selective work -- but by the time that content passes through
`sparse_message_mlp`, the network has learned not to use most of it. This
points investigation at `sparse_message_mlp`'s own training dynamics (why
it converged this way -- e.g. whether the event-content block is
underpowered relative to the channel-embedding block, or whether the
classification loss simply never rewarded using it given `channel_encoder`
alone already explains most of the label) as much as it points upstream at
*what* events get built/gated -- see
[[sparse-evidence-gnn-frequency-fragmentation-bias]] for that upstream
angle. `event_aggregation` is left in the codebase as a real, working,
tested option (default stays `"mean"`,
zero behavior change for the canonical pipeline) in case a future change
upstream (different event content/density) makes weighting relevant again.

See also [[sparse-evidence-gnn-channel-encoder-dominates]] (the original
ablation finding this follows up on) and
[[sparse-evidence-gnn-reorg-2026-08-09]] (`SPARSE_EVIDENCE_GNN_PARAMS`, the
base config this sweep varied).
