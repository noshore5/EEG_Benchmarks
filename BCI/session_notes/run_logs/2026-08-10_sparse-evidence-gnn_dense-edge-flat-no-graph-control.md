# Dense-edge non-graph control — does message-passing add anything on top of dense features? — 2026-08-10

## Why

The [dense-mode write-up](2026-08-10_sparse-evidence-gnn_dense-event-mode-sparse-vs-dense.md)
found `event_mode="dense"` scores 0.8886±0.0067 event-pathway-alone
(`zero_channel_embed`, seeds 42-45) — dramatically above sparse mode's
0.628 and the sparse-mode non-graph baseline's 0.689-0.696 — but flagged an
open question in its "Caveats" section: dense mode's richer per-edge
features still route through the SAME graph/message-passing back-end
([[sparse-evidence-gnn-event-stats-baseline-beats-graph]] already showed
that back-end underperforms a simple stats baseline on sparse events), so
it wasn't clear whether message-passing was adding value on top of dense
features specifically, or whether the whole 0.8886 was coming from
`dense_edge_conv` alone. This run isolates that directly.

## Method

New tracked script:
[`run_dense_edge_flat_control.py`](../../tests/run_dense_edge_flat_control.py).

Reuses `dense_edge_conv` and its exact `[B, 4, E, T, F]` input
(`_build_dense_edge_input`/`compute_dense_edge_input`, via a throwaway
`SparseEvidenceGNNClassifier(event_mode="dense", ...)` helper's
`_prepare_features` — same technique every driver today has used) UNCHANGED
— same architecture, same defaults (`dense_conv_kernel_size=5`,
`dense_conv_pool_size=4`, `dense_conv_intermediate_channels=32`,
`dense_conv_out_channels=8`). After `dense_edge_conv` produces
`[B, E, dense_conv_out_channels]`, this control SKIPS `sparse_message_mlp`,
`_aggregate_events` (scatter-to-node), and `_propagate_hops` entirely:
flattens directly to `[B, E * dense_conv_out_channels]` = `[B, 288]` and
feeds a single `nn.Linear(288, n_classes)` — no hidden layer, deliberately
matching `sparse_classifier`'s own "one Linear, no hidden layer" simplicity
so a score difference can't be chalked up to a bigger classifier head. No
scatter-to-node, no src/dst channel identity, no edge topology used beyond
"which flattened slot this edge's features land in" (fixed, consistent,
not a message-passing structure). Also, deliberately, **no
ChannelSignalEncoder anywhere** — this isolates dense event features from
BOTH graph structure and raw-signal channel embeddings simultaneously,
unlike `feature_ablation="zero_channel_embed"` (which keeps the graph,
only drops channel embeddings).

Training: hand-rolled loop matching the canonical pipeline's own training
regime as closely as a from-scratch loop reasonably can (`AdamW`,
`CrossEntropyLoss`, gradient-norm clipping at 0.1, a shuffle-order
generator tied to `seed` independent of model-construction RNG draws — see
[[torch-eeg-classifier-dataloader-shuffle-seed-fix]]) — same
`epochs=75, batch_size=8, learning_rate=1e-3, weight_decay=1e-4` as every
other run in this line of investigation today.

Subject 1, `BNCI2014_001`, `LeftRightImagery(fmin=8, fmax=35)`,
`CrossSessionEvaluation`, canonical config identical to the dense-mode
write-up (`phase_threshold_deg=10`, `surrogate_percentile=99`,
`coherence_threshold_mode="surrogate"`,
`channel_subset=[1,5,7,8,9,10,11,13,17]`, etc. — hardcoded, not read from
`run_sparse_evidence_gnn.py`). **Seed 42 only**, per this run's own scope
(not a 4-seed sweep) — see Caveats.

Practical cost flagged before starting: expected similar to full dense
mode (`dense_edge_conv` is the dominant cost either way — a real Conv2d
forward+backward over the full native-resolution array every training
step; removing `sparse_message_mlp`/`_aggregate_events`/`_propagate_hops`
was expected to save comparatively little). Confirmed: this run took a
comparable few hundred seconds, same ballpark as the ~430-530s/seed
dense-mode runs.

### Smoke test

Random `[B=4, 4, E=36, T=300, F=16]` input through `DenseEdgeFlatCore` →
`CrossEntropyLoss` → backward: both `dense_edge_conv` and the `Linear`
classifier head received nonzero gradients. Also ran a small synthetic
`fit`/`predict`/`predict_proba` integration check (12 trials,
`surrogate_count=3`, 2 epochs) before the real run to catch shape/plumbing
bugs cheaply.

## Results

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.8949 | 0.9633 | **0.9291** |

| configuration | mean | std | seeds |
| --- | --- | --- | --- |
| Sparse GNN, event-only (`zero_channel_embed`) | 0.628 | 0.0153 | 42-45 |
| Non-graph baseline on SPARSE events (LogReg/MLP) | 0.689-0.696 | 0.000-0.023 | 42-45 |
| Dense GNN, event-only (`zero_channel_embed`) | 0.8886 | 0.0067 | 42-45 |
| Dense GNN, full pipeline (`none`) | 0.9048 | 0.0031 | 42-45 |
| **Non-graph control on DENSE features (this run)** | **0.9291** | n/a (1 seed) | 42 |

Same-seed comparison (seed 42 only, apples-to-apples): dense
`zero_channel_embed` scored 0.8571/0.9215 (mean 0.8893); this control
scored 0.8949/0.9633 (mean 0.9291) — **both folds improved** (+0.0378 on
`0train`, +0.0418 on `1test`), not a one-fold fluke.

This control — no graph, no channel embeddings, dense features alone —
**beats every other configuration on record for this pipeline**, including
dense mode's own full pipeline (0.9048, which still has channel
embeddings) and dense mode's graph-based event-only ablation (0.8886).

## Interpretation

Per this run's own pre-registered framework: the control clearly **beats**
0.8886 (not "close to," not "meaningfully below"). Per that framework's own
terms: **message-passing is not adding value on top of dense features —
on this subject/seed, removing it entirely and just flattening + a single
Linear layer does *better* than routing through the graph.** This is the
sharper of the two possible outcomes this run was set up to distinguish;
it does not merely fail to find evidence the graph helps, it finds
evidence the graph structure costs something here.

A plausible (not confirmed) mechanism: `_aggregate_events`' scatter-to-node
step mean-pools every edge landing on a destination channel into ONE
`hidden_dim=8` vector per channel before the final classifier ever sees
it — with 36 edges routing onto 9 channels (~4 edges/channel on average),
that's a real compression step, and the final `sparse_classifier` only
ever sees `n_channels * hidden_dim = 72` values. This control's flatten
keeps every edge's own distinct `dense_conv_out_channels=8`-wide feature
intact all the way to the classifier — `E * dense_conv_out_channels = 288`
values, ~4x more than the graph pathway's own readout width, with zero
forced averaging across edges. That's a simple, testable capacity/
lossiness explanation that doesn't require topology itself to be
"wrong" — worth checking directly (e.g., does raising `hidden_dim`
enough to match 288 total dims close the gap?) before concluding topology
per se is the problem rather than this particular aggregation's
information loss.

## Caveats

1. **Single seed, single subject.** This run was deliberately scoped to
   seed 42 only, per its own instructions. [[sparse-evidence-gnn-seed-
   variance]] already documents real seed-to-seed variance in this
   pipeline generally; a 4-seed sweep (matching every other comparison in
   this line of investigation) is the obvious next step before treating
   "beats 0.8886" as more than a strong single-point signal. That said,
   both folds moved the same direction by a comparable margin, which is at
   least mildly reassuring against "one lucky fold."
2. **Everything from the dense-mode write-up's own caveats still applies**
   here too, and compounds: single subject (1, historically an easy one
   for this pipeline), untuned `dense_conv_*` hyperparameters, no
   independent leakage audit beyond the reasoning already done for dense
   mode (this control reuses the identical precompute path, so that
   reasoning transfers, but wasn't independently re-verified here).
3. **This doesn't test whether SOME graph structure would help** — only
   that THIS pipeline's specific aggregation (mean/gated_softmax scatter
   to destination channel, `hidden_dim=8` bottleneck) doesn't. A wider
   `hidden_dim` or a different aggregation might behave differently; see
   "Interpretation" above.

## Concurrent-edit check

`sparse_evidence_gnn_classifier.py` mtime unchanged (11:19, same as every
prior check today) — confirmed via `git status`/`ls -la` immediately
before writing this note. This control only ever READS from that file
(`_build_dense_feature_conv`, `SparseEvidenceGNNClassifier`'s
`_prepare_features`/`_precompute_dense_edge_inputs`), never modifies it, so
even if it had changed, this run's own new code (`run_dense_edge_flat_
control.py`) would be unaffected beyond whatever those reused pieces
changed to.

`run_sparse_evidence_gnn.py` (still not used by this or any driver today)
remains at its 15:00 mtime from the last check — no further change since.

See [[sparse-evidence-gnn-dense-mode-beats-sparse]],
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]], and the
[dense-mode write-up](2026-08-10_sparse-evidence-gnn_dense-event-mode-sparse-vs-dense.md)
this run directly follows up on.
