# Sparse-Evidence-GNN time-averaged coherence graph ablation, subjects 1-4 — 2026-08-11

## Why

Question: does the canonical dense/concat pipeline
([dense/concat canonical run](2026-08-10_sparse-evidence-gnn_dense-concat-canonical-4subject.md),
cohort mean 0.7897) actually need within-trial TIME resolution at all, or
does a static, per-(edge, frequency) functional-connectivity-style summary —
mean coherence/phase/significance over the whole trial, one value per edge
per frequency, no time axis left — carry the same signal? This directly
extends the standing pattern that this pipeline's signal keeps turning out to
live somewhere other than where it was designed to
([[sparse-evidence-gnn-channel-encoder-dominates]],
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]],
[[sparse-evidence-gnn-dense-flat-control-beats-graph]]) by asking the same
question of the time axis specifically.

## Method

New `time_averaged_graph: bool = False` flag added to `SparseEvidenceGNNCore`/
`SparseEvidenceGNNClassifier`
([[sparse-evidence-gnn-time-averaged-graph-feature]] has the full
implementation writeup). When `True`, `_build_dense_edge_input`'s `[B, 4, E,
T, F]` coherence/sinφ/cosφ/significance stack is collapsed to `[B, 4, E, 1,
F]` via a COI-valid-weighted average over T (weighted by count of COI-valid
timesteps per frequency, not a plain mean — avoids biasing low frequencies,
which have the widest excluded cone-of-influence regions, toward zero) at
non-trainable precompute time. Everything downstream (`dense_edge_conv`,
`sparse_message_mlp`, `_aggregate_events="concat"`, `sparse_classifier`) is
architecturally identical to `event_mode="dense"` today — only event BUILDING
changes, same relationship `event_mode="dense"` itself has to
`_build_sparse_events`.

Ran via new standalone script
[run_time_averaged_coherence_graph.py](../../tests/run_time_averaged_coherence_graph.py)
(same one-off-ablation convention as `run_dense_edge_flat_control.py`: a
frozen `CANONICAL_KWARGS` snapshot copied from the documented dense/concat
canonical run, NOT read live from `run_sparse_evidence_gnn.py`, which stays
under concurrent edit in this session).

Config: identical to the dense/concat canonical run
(`event_mode="dense"`, `event_aggregation="concat"`, `n_hops=1`,
`feature_ablation="zero_channel_embed"`, `epochs=60`, `batch_size=8`,
`learning_rate=1e-3`, `weight_decay=1e-4`, `grad_clip_norm=0.1`, `seed=45`,
`surrogate_seed=42`, `validation_split=0.0`, `coherence_threshold_mode="fixed"`,
`coherence_threshold=0.99`, `phase_threshold_deg=10.0`,
`channel_subset=[1,5,7,8,9,10,11,13,17]`, `nfreqs=16`, `lowest=8.0`,
`highest=30.0`, `sampling_rate=250`, `channel_encoder_dilation=5`,
`hidden_dim=8`, `channel_embed_dim=8`, `smooth_kernel_size=(5,3)`,
`coi_enabled=True`, `dense_conv_intermediate_channels=32`,
`dense_conv_out_channels=8`), with exactly three deltas required by
`time_averaged_graph=True`: `dense_conv_kernel_size` 5→1,
`dense_conv_pool_size` 4→1 (no temporal kernel left to convolve once T=1),
`dense_edge_time_downsample` 8→1 (mutually exclusive with
`time_averaged_graph`), plus `time_averaged_graph=True` itself.

Same evaluation as the baseline: `moabb.evaluations.CrossSessionEvaluation`
(`BNCI2014_001`, `LeftRightImagery(fmin=8, fmax=35)`, `random_state=42`),
subjects 1-4, 2 folds/subject (`0train`/`1test`), single seed (45, matching
the baseline run's own seed exactly for a like-for-like comparison).

## Results

| subject | 0train | 1test | mean |
| --- | --- | --- | --- |
| 1 | 0.8709 | 0.9682 | 0.9196 |
| 2 | 0.6316 | 0.5637 | 0.5976 |
| 3 | 0.9440 | 0.9416 | 0.9428 |
| 4 | 0.7704 | 0.7089 | 0.7397 |
| **cohort mean** | | | **0.7999** |

## Comparison to the time-resolved dense/concat baseline (same day-adjacent, same config otherwise)

| subject | dense/concat (time-resolved) | time_averaged_graph | Δ |
| --- | --- | --- | --- |
| 1 | 0.9253 | 0.9196 | -0.0057 |
| 2 | 0.5743 | 0.5976 | +0.0233 |
| 3 | 0.9297 | 0.9428 | +0.0131 |
| 4 | 0.7296 | 0.7397 | +0.0101 |
| **cohort mean** | **0.7897** | **0.7999** | **+0.0102** |

## Finding

Collapsing the entire dense event pathway's time axis to a single COI-
weighted average per trial did **not** cost accuracy — it came out slightly
*ahead* of the time-resolved baseline on 3 of 4 subjects (only subject 1
dipped, by a small 0.0057) for a +0.0102 cohort-mean edge, on a single
matched seed. A static, per-edge/per-frequency functional-connectivity-style
graph carries at least as much signal here as the time-resolved
coherence/phase array `dense_edge_conv` was built to exploit.

## Interpretation

This lands squarely on the same axis as the standing findings that this
pipeline's discriminative signal keeps turning out to be shallower than its
architecture assumes: [[sparse-evidence-gnn-channel-encoder-dominates]]
(accuracy comes almost entirely from `ChannelSignalEncoder`, not the event
pathway), [[sparse-evidence-gnn-event-stats-baseline-beats-graph]] (simple
per-edge stats beat the graph), and
[[sparse-evidence-gnn-dense-flat-control-beats-graph]] (no-graph flat control
beats message-passing on top of dense features). This result adds: whatever
the event pathway (dense/concat) IS contributing doesn't depend on WHEN
within the trial coherence happens, only on its trial-long average value per
edge/frequency. Consistent with, but not proof of, the mu-band-starvation/
frequency-fragmentation hypothesis in
[[sparse-evidence-gnn-frequency-fragmentation-bias]] — if real discriminative
coupling in this data is closer to a sustained trait (motor-imagery-driven
average coherence shift) than a bursty state, this is exactly the result
you'd expect, and `dense_edge_conv`'s real Conv2d-over-native-T cost
(the dominant per-epoch cost in `event_mode="dense"`, per
`dense_edge_time_downsample`'s own docstring) buys nothing worth its
compute here.

## Caveats

1. Single seed (45), single matched comparison — no variance estimate either
   direction; see [[sparse-evidence-gnn-seed-variance]] for how large that
   variance has run on this pipeline historically. A +0.0102 cohort-mean
   delta is well within plausible seed noise for this dataset/pipeline; this
   result is "time-averaging doesn't clearly hurt," not "time-averaging is
   proven better."
2. Practical upside not yet measured: with T=1, `dense_edge_conv` becomes two
   1x1-kernel Conv2d layers over a single timestep instead of a real Conv2d
   over native T~1001 samples — expected to be substantially cheaper per
   epoch (the exact cost `dense_edge_time_downsample`'s own docstring
   identifies as event_mode="dense"'s dominant per-epoch cost), but
   wall-clock wasn't measured in this run.
3. Only tested against the dense/concat config's own hyperparameters
   (epochs=60, etc.) — not re-tuned for the now much-lower-capacity
   `dense_edge_conv` (two 1x1 convs vs. two real kernel-5 convs +
   pooling), so this may understate time_averaged_graph's real ceiling.
4. `event_aggregation="mean"`/`"gated_softmax"` combinations, and n_hops>1,
   were not tested with `time_averaged_graph=True` — only the concat
   configuration this repo's canonical config currently uses.

## Concurrent-edit check

`run_sparse_evidence_gnn.py` continues to be live-edited in the IDE
throughout this session (see prior run logs' own notes on this). This
ablation's script and config were built independently of that file, from the
already-documented dense/concat canonical run's own committed write-up, so
it isn't affected by those concurrent edits.

See [[sparse-evidence-gnn-time-averaged-graph-feature]] for the
implementation itself, [[sparse-evidence-gnn-concat-productionized-cross-subject]]
for the baseline config's own validation history, and the
[dense/concat canonical run](2026-08-10_sparse-evidence-gnn_dense-concat-canonical-4subject.md)
this compares against.
