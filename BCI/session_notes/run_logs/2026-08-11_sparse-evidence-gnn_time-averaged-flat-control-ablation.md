# Sparse-Evidence-GNN time-averaged coherence + flat (no-graph) control, subjects 1-4 — 2026-08-11

## Why

Two independent simplifications of the dense/concat canonical pipeline had
each, on their own, matched or beaten it:
[flattening the graph away](2026-08-10_sparse-evidence-gnn_dense-edge-flat-no-graph-control.md)
(no message-passing, subject 1 only: 0.9291 vs. 0.8886) and
[time-averaging the coherence input](2026-08-11_sparse-evidence-gnn_time-averaged-coherence-graph-ablation.md)
(no within-trial time resolution, 4 subjects: 0.7999 vs. 0.7897, graph kept).
Neither result says what happens with BOTH stacked: does a flat readout on
top of a STATIC coherence summary still hold up, or does removing time
resolution finally cost something once the graph's own redundant structure
isn't there to compensate for it?

## Method

`DenseEdgeFlatClassifier` ([[sparse-evidence-gnn-dense-flat-control-beats-graph]])
gained a `time_averaged_graph: bool = False` pass-through constructor arg
(2026-08-11, additive, default preserves prior behavior exactly) that threads
into its helper `SparseEvidenceGNNClassifier`'s own
[[sparse-evidence-gnn-time-averaged-graph-feature]]. New script
[run_time_averaged_flat_control.py](../../tests/run_time_averaged_flat_control.py)
sets `time_averaged_graph=True` plus the required
`dense_conv_kernel_size=dense_conv_pool_size=1` (once the precomputed
`dense_edge_raw`'s T axis is 1, a k>1 kernel has nothing to convolve over —
same requirement `time_averaged_graph`'s own docstring describes). With
kernel/pool collapsed to 1x1, `DenseEdgeFlatCore`'s `dense_edge_conv` becomes
two stacked 1x1-kernel Conv2d+GELU layers over the folded (4*nfreqs) input
channels — a per-edge MLP, no temporal or cross-edge mixing at all — flattened
straight to the single `Linear` classifier head, same architecture the
original flat control uses.

Training hyperparameters: `epochs=75`, `batch_size=8`, `learning_rate=1e-3`,
`weight_decay=1e-4`, `grad_clip_norm=0.1`, `dense_conv_intermediate_channels=32`,
`dense_conv_out_channels=8` — identical to `run_dense_edge_flat_control.py`'s
own defaults, so the delta from that baseline is attributable to
`time_averaged_graph` alone. Dense-edge precompute config
(`coherence_threshold_mode="surrogate"`, `surrogate_percentile=99.0`, etc.)
inherited unchanged from `CANONICAL_HELPER_KWARGS` — NOT the same
`coherence_threshold_mode="fixed"` config the dense/concat canonical run and
[[sparse-evidence-gnn-time-averaged-graph-feature]]'s own graph-based
comparison use (see Caveats).

Smoke-tested first (T=1 synthetic forward/backward + full 22-channel
fit/predict round-trip) before the real run. Same evaluation as the other
ablations: `moabb.evaluations.CrossSessionEvaluation` (`BNCI2014_001`,
`LeftRightImagery(fmin=8, fmax=35)`, `random_state=42`), subjects 1-4, 2
folds/subject, single seed (45).

## Results

| subject | 0train | 1test | mean |
| --- | --- | --- | --- |
| 1 | 0.9026 | 0.9705 | 0.9365 |
| 2 | 0.6024 | 0.5980 | 0.6002 |
| 3 | 0.9489 | 0.9392 | 0.9441 |
| 4 | 0.7485 | 0.7062 | 0.7273 |
| **cohort mean** | | | **0.8020** |

## Comparison across all four combinations tested so far

Cohort-mean (4 subjects) where available; subject-1-only where the flat +
time-resolved control was never run cohort-wide:

| config | graph? | time-resolved? | subj1 | cohort mean (4 subj) |
| --- | --- | --- | --- | --- |
| [dense/concat canonical](2026-08-10_sparse-evidence-gnn_dense-concat-canonical-4subject.md) | yes | yes | 0.9253 | 0.7897 |
| [time-averaged graph](2026-08-11_sparse-evidence-gnn_time-averaged-coherence-graph-ablation.md) | yes | no | 0.9196 | 0.7999 |
| [flat control](2026-08-10_sparse-evidence-gnn_dense-edge-flat-no-graph-control.md) | no | yes | 0.9291 | *(not run cohort-wide)* |
| **flat + time-averaged (this run)** | no | no | **0.9365** | **0.8020** |

## Finding

Stacking both simplifications did **not** compound into a loss — it produced
the best cohort mean (0.8020) and the best subject-1 score (0.9365) of every
combination tested to date, on this single-seed comparison. Removing graph
structure and removing within-trial time resolution appear to be
independent, non-interacting simplifications for this pipeline/dataset:
neither one's benefit depends on the other still being in place.

## Interpretation

Extends the same throughline as
[[sparse-evidence-gnn-channel-encoder-dominates]],
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]],
[[sparse-evidence-gnn-dense-flat-control-beats-graph]], and
[[sparse-evidence-gnn-time-averaged-graph-feature]]'s own result: this
pipeline's real discriminative signal in the dense event pathway looks
consistent with a single scalar-ish per-edge/per-frequency quantity (roughly,
"how coupled were these two channels on average, at this frequency, this
trial") that graph message-passing and per-timestep temporal convolution both
spend real compute processing without adding anything the data supports.
`DenseEdgeFlatCore` at `kernel_size=pool_size=1` is now, architecturally,
almost exactly that — a per-edge/per-frequency linear-ish summary flattened
straight into a linear classifier — and it's the best-performing
configuration found in this whole line of ablations so far.

## Caveats

1. Single seed (45) — same caveat as every other entry in this ablation
   series; see [[sparse-evidence-gnn-seed-variance]]. A 0.8020 vs. 0.7999 vs.
   0.7897 spread across three 4-subject cohort means is plausibly within
   seed noise for all three, not a confirmed ranking.
2. **Not a clean apples-to-apples control on coherence_threshold_mode**: this
   run's helper config (`CANONICAL_HELPER_KWARGS`, inherited unchanged from
   `run_dense_edge_flat_control.py`) uses
   `coherence_threshold_mode="surrogate"`/`surrogate_percentile=99.0`, while
   the dense/concat canonical baseline and the graph-based time-averaging
   comparison both use `coherence_threshold_mode="fixed"`/
   `coherence_threshold=0.99`. The significance channel
   ((coh - threshold) / threshold) that feeds into the averaged coherence
   graph is therefore computed against a different per-trial threshold in
   this run than in the other rows of the comparison table above — the
   `graph? no / time-resolved? yes` row (flat control) shares this same
   surrogate-threshold config, so THAT comparison (flat + time-resolved vs.
   flat + time-averaged) is clean; the cross-row comparisons against the
   two `coherence_threshold_mode="fixed"` rows are not as clean as the table
   presentation suggests.
3. Subject-1-only for the flat + time-resolved control (never run
   cohort-wide) — its cohort-mean cell is genuinely missing, not omitted.
4. No wall-clock measurement here either (same gap flagged in the
   time-averaged graph run log) — the practical throughput case for
   `time_averaged_graph` (dense_edge_conv processing T=1 instead of
   native T~1001) is still unmeasured across every ablation in this series.

## Concurrent-edit check

`run_sparse_evidence_gnn.py` continues to be live-edited in the IDE
throughout this session. This ablation is unaffected -- its config comes from
`CANONICAL_HELPER_KWARGS` (`run_dense_edge_flat_control.py`), not from that
file.

See [[sparse-evidence-gnn-time-averaged-graph-feature]] and
[[sparse-evidence-gnn-dense-flat-control-beats-graph]] for the two
individual ablations this combines, and
[[sparse-evidence-gnn-seed-variance]] for this series' shared single-seed
caveat.
