# Sparse-Evidence-GNN `event_mode="dense"` — learned conv over coherence arrays vs. hard-threshold sparse events — 2026-08-10

## Why

Two same-day findings motivated this:
[[sparse-evidence-gnn-channel-encoder-dominates]]/[the zero_channel_embed
re-check](2026-08-10_sparse-evidence-gnn_zero-channel-embed-4seed-threshold-recheck.md)
showed the event pathway contributes little inside the full GNN (0.628 vs.
0.8527 with channel embeddings), and
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]] then showed a
*non-graph* model reading simple stats off the same discrete events
(0.689-0.696) beats the GNN's own event-only pathway (0.628) — pointing at
event-*detection* quality (hard threshold + consolidate discarding
information) as the more likely bottleneck than the graph/message-passing
mechanism itself. This run tests that directly: replace hard thresholding
with a learned, continuous, differentiable front-end over the exact same
already-fixed coherence arrays, keep the graph/message-passing back-end
identical, and see what happens.

**`event_mode="dense"` is still a GNN** — it only replaces event
*building* (`_build_sparse_events`'s threshold-and-consolidate step).
Everything downstream is bit-identical machinery to sparse mode: same
36-canonical-edge topology, same `sparse_message_mlp`, same
`_aggregate_events` scatter-to-node aggregation, same optional `n_hops`
propagation. See implementation section below and
[[sparse-evidence-gnn-dense-event-mode-untested]] (now superseded by this
note — dense mode has been smoke-tested and run on real data).

## Implementation

New `event_mode: Literal["sparse", "dense"] = "sparse"` on
`SparseEvidenceGNNCore`/`SparseEvidenceGNNClassifier`
(`sparse_evidence_gnn_classifier.py`). Default `"sparse"` is bit-identical
to pre-existing behavior (`dense_edge_conv` is only constructed, and
`message_in` only changes, when `event_mode="dense"`; built LAST in
`__init__` so it can't shift any pre-existing submodule's RNG-drawn init —
same precedent as the existing `event_gate`/`hop_message_mlp_freq` guards,
see [[sparse-evidence-gnn-event-gate-init-shift-bug]]).

`"dense"` mode:
- `_build_dense_edge_input`/`compute_dense_edge_input` (non-trainable,
  precomputed once per trial by
  `SparseEvidenceGNNClassifier._precompute_dense_edge_inputs`, mirroring
  `_precompute_sparse_events`'s own "non-trainable work runs once, not
  every forward()" optimization) run the SAME cross-spectrum/smoothing math
  `_coherence_only`/`_build_sparse_events` use, but skip the hard
  threshold+consolidate step entirely. Output: `[B, 4, E, T, F]` stack of
  `[coherence, sin(phase), cos(phase), significance]`, post-COI-mask
  (zeroed outside the cone of influence). `significance = (coh -
  surrogate_threshold) / surrogate_threshold` — the SAME per-(edge,
  frequency) surrogate-calibrated threshold the hard gate uses (identical
  null-distribution cache, so a warm cache from prior sparse-mode runs is
  reused), fed continuously instead of as a cutoff.
- A new TRAINABLE `dense_edge_conv` (`_build_dense_feature_conv`, modeled
  on but not copied from `WCTEvidenceGNNCore._build_feature_conv` — new
  `dense_conv_kernel_size`/`dense_conv_pool_size`/
  `dense_conv_intermediate_channels`/`dense_conv_out_channels` constructor
  args, same two-block conv+pool pattern) runs on this stack EVERY
  `forward()` call — unlike sparse event-building, this can't be
  precomputed once since it has learnable weights (frequency folded into
  conv input channels so the first layer can learn cross-frequency
  combinations; edge axis stays untouched/weight-shared across edges,
  matching how `WCTEvidenceGNNCore.feature_conv` treats its channel axis).
  Produces one feature vector per edge (`dense_conv_out_channels`-wide,
  replacing sparse events' fixed 5-wide `[t, freq, mag, sinφ, cosφ]`).
- `_dense_edge_features` packages that into an "every edge always valid"
  `events_padded`/`src_padded`/`dst_padded`/`valid_mask`/`batch_idx`
  matching sparse mode's shape exactly, so `feature_ablation`,
  `sparse_message_mlp`, `_aggregate_events`, and `_propagate_hops` are
  IDENTICAL code paths between modes — only event-building differs.
- `freq_aware_hops=True` is rejected together with `event_mode="dense"`
  (dense features have no discrete per-event frequency bin to index by).

## Smoke test

Random data (`B=4, C=9, T=300, F=16`), synthetic `[B, 4, E, T, F]` dense
input, forward → `CrossEntropyLoss` → backward. All passed:
- Dense mode: `dense_edge_conv` and `channel_encoder` both receive nonzero
  gradients (`sparse_message_mlp`/`sparse_classifier` too).
- Dense mode + `n_hops=3`: `dense_edge_conv` and `hop_message_mlp` both
  receive nonzero gradients.
- `event_mode="sparse"` (default): unaffected regression check, still
  works exactly as before with synthetic sparse event tensors.
- Constructor validation: bad `event_mode` value rejected; `event_mode="dense"`
  + `freq_aware_hops=True` rejected with the intended error message.

## Method (real-data runs)

Same driver pattern as the earlier baseline/hidden_dim-sweep scripts today
(throwaway, not committed): `moabb.evaluations.CrossSessionEvaluation`
called directly with `SparseEvidenceGNNClassifier(**kwargs)`, subject 1,
`BNCI2014_001`, `LeftRightImagery(fmin=8, fmax=35)`, `random_state=42`.
Canonical config identical to the zero_channel_embed re-check's "Full
effective parameters" block (`phase_threshold_deg=10`,
`surrogate_percentile=99`, `coherence_threshold_mode="surrogate"`,
`epochs=75`, `batch_size=8`, `channel_subset=[1,5,7,8,9,10,11,13,17]`,
etc.), plus dense-only knobs left at their new defaults
(`dense_conv_kernel_size=5`, `dense_conv_pool_size=4`,
`dense_conv_intermediate_channels=32`, `dense_conv_out_channels=8` — not
tuned). Seeds 42-45, both `feature_ablation="none"` (full pipeline) and
`feature_ablation="zero_channel_embed"` (event pathway alone). Driver
sanity-checked earlier today at `hidden_dim=8` against this exact
methodology and reproduced the on-record sparse `zero_channel_embed`
number almost exactly (0.6230 vs. 0.623) — same driver used here.

`event_mode="sparse"` numbers are **not** re-run here — reused directly
from the same-day, same-config, same-seed
[zero_channel_embed re-check](2026-08-10_sparse-evidence-gnn_zero-channel-embed-4seed-threshold-recheck.md)
(0.628) and the
[8-seed epochs sweep](2026-08-09_sparse-evidence-gnn_epochs50-75-100_8seed-sweep.md)
(0.8527, "none"), both already cross-validated against each other and, per
the sanity check above, against this driver.

## Results

**Read this section's `zero_channel_embed` comparison first** — it's the
real, apples-to-apples test of whether dense event processing beats sparse
event processing at the thing this whole detour is about. The `none`
comparison further down is confounded by channel embeddings (which already
dominate accuracy on their own) and is presented second, with that caveat.

### Primary comparison: event pathway alone (`feature_ablation="zero_channel_embed"`)

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.8571 | 0.9215 | 0.8893 |
| 43 | 0.8499 | 0.9119 | 0.8809 |
| 44 | 0.8447 | 0.9290 | 0.8869 |
| 45 | 0.8497 | 0.9446 | 0.8972 |

mean=**0.8886**, std=0.0067 (sample), min=0.8809, max=0.8972.

| pathway | mean | std |
| --- | --- | --- |
| Sparse GNN, event-only (`zero_channel_embed`) | 0.628 | 0.0153 |
| Non-graph baseline, LogReg (event stats only) | 0.6887 | 0.0000 |
| Non-graph baseline, MLP (event stats only) | 0.6961 | 0.0234 |
| **Dense GNN, event-only (`zero_channel_embed`)** | **0.8886** | **0.0067** |
| *(for reference)* Sparse GNN, full pipeline (`none`, channel embeds + events) | 0.8527 | n/a (8-seed sweep) |

Dense mode's event-pathway-alone score (0.8886) is **~0.26 above the
sparse GNN's own event-only score, ~0.19-0.20 above the non-graph
baseline, and even ~0.036 above sparse mode's FULL pipeline** (channel
embeddings included). This is not "closes the gap" — it's a different
regime entirely.

### Secondary comparison: full pipeline (`feature_ablation="none"`)

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.8608 | 0.9405 | 0.9007 |
| 43 | 0.8617 | 0.9498 | 0.9058 |
| 44 | 0.8646 | 0.9452 | 0.9049 |
| 45 | 0.8576 | 0.9582 | 0.9079 |

mean=0.9048, std=0.0031, min=0.9007, max=0.9079. Above sparse mode's
0.8527, but this comparison mixes channel-embedding contribution with
event-pathway contribution and is NOT a clean isolation of the dense
mechanism — presented for completeness, not as the headline number.

**Notable structural observation**: in dense mode, `zero_channel_embed`
(0.8886) and `none` (0.9048) are only ~0.016 apart — channel embeddings
add almost nothing once the dense event pathway is already this strong.
This is close to a reversal of
[[sparse-evidence-gnn-channel-encoder-dominates]]'s standing finding, but
ONLY for the dense event representation — sparse events still show the
large ~0.22 gap that finding was built on. Channel embeddings aren't
"broken"; the dense event pathway has simply become informative enough
that it stops needing them as much.

## Practical cost

Dense mode is dramatically more expensive: ~430-530s/seed (both
CrossSessionEvaluation folds) vs. sparse mode's ~15-30s/seed (almost
entirely CWT/surrogate precompute there, since sparse event-building is
non-trainable and cached once per trial). Root cause: `dense_edge_conv`
has learnable weights and therefore CANNOT be precomputed once per trial
the way sparse event-building can — it runs a real Conv2d forward+backward
over the full native-resolution `[edge=36, time≈1001, freq=16]` array on
every training step (75 epochs × ~18 batches/epoch × 2 evaluation folds ≈
2700 forward+backward passes/seed), on CPU (`device="auto"` resolves to
CPU on this machine, kept at the canonical setting for a fair comparison
rather than switched for speed). This 8-run sweep took ~65 minutes total
wall-clock.

## Caveats — read before treating this as settled

1. **Single subject, 4 seeds.** Subject 1 is one of this pipeline's
   stronger subjects historically (subj2/subj4 sit near chance in earlier
   canonical 4-subject runs — see the classifier module docstring). A
   result this dramatic on one subject does not establish it holds across
   the cohort; a 4-subject (or full 9-subject) dense-mode run is the
   obvious next step before treating 0.89 as a real pipeline number rather
   than a promising single-subject signal.
2. **Magnitude prompted a leakage/bug check** — this score is well above
   everything else on record for this pipeline (including the full model
   with channel embeddings), which is exactly the kind of result that
   deserves scrutiny before celebrating. Checked and found no evidence of
   an issue: (a) `_build_dense_feature_conv` has no BatchNorm/running-stats
   layers that could leak train/test information; (b) the conv's edge axis
   is kernel-height-1 throughout, so no cross-edge mixing; (c)
   normalization stats are fit on train-fold-only data as usual and don't
   affect coherence anyway (unchanged from sparse mode, already
   established); (d) the surrogate significance threshold is computed
   per-trial from that trial's own phase-randomized surrogates, no label
   information involved; (e) "0train"/"1test" are BNCI2014-001's own two
   session NAMES (both are genuine held-out generalization scores, not a
   train-vs-test-set comparison — same convention every other note in this
   log uses). No smoking gun found, but this is a single check, not an
   independent audit — worth keeping in mind especially given (1) above.
3. **Not hyperparameter-tuned.** `dense_conv_*` params are untouched
   defaults, unlike every other knob in this pipeline (which has been
   tuned extensively per the module docstring/session notes). The real
   ceiling (or floor) for dense mode is unknown.
4. **`event_mode="dense"` is a strictly more expensive front-end into the
   SAME graph/message-passing back-end** [[sparse-evidence-gnn-event-stats-baseline-beats-graph]]
   already showed underperforms a simple stats vector, in sparse mode. That
   dense mode's much richer per-edge features still route through that
   same (previously found weak) aggregation mechanism and still score this
   high suggests event-detection quality really was the larger bottleneck
   — but a non-graph baseline built on TOP of dense mode's own edge
   features (analogous to the sparse-mode stats baseline) has not been
   run, and would isolate whether the graph/message-passing layer is
   adding anything at all on top of dense features specifically.

## Finding

Under the letter of the original framing this line of investigation was
set up to test ("if 'dense' mode... captures more signal than the current
sparse event pathway alone... that's the actual question"): yes,
decisively, on this subject. Dense mode's event-only score (0.8886) is not
a marginal improvement over sparse mode's event-only score (0.628) — it's
a different tier of performance, on par with or exceeding the full model.
Combined with [[sparse-evidence-gnn-event-stats-baseline-beats-graph]]'s
finding that sparse events' bottleneck was upstream of message passing,
this strongly points to hard thresholding as the actual costly step in the
original pipeline — discarding continuous coherence/phase/significance
information down to a handful of discrete events was destroying far more
signal than the graph architecture around those events ever could recover.

Recommended next steps, roughly in priority order: (a) multi-subject dense
run (at minimum the canonical 4-subject set) before trusting this
generalizes past subject 1; (b) a non-graph baseline on dense features
(mirroring [[sparse-evidence-gnn-event-stats-baseline-beats-graph]]'s
methodology) to isolate whether message-passing is adding value on top of
dense features specifically; (c) only then, dense_conv_* hyperparameter
tuning and a cost/accuracy tradeoff pass (dense mode's ~15-20x wall-clock
cost over sparse is real and will matter at cohort scale).

## Concurrent-edit check

`sparse_evidence_gnn_classifier.py` mtime unchanged (11:19) from when this
same session's dense-mode implementation finished — confirmed via `git
status`/`ls -la` immediately before writing this note. No concurrent edits
to this file since; every result above was produced against the exact code
described in "Implementation."

`run_sparse_evidence_gnn.py` (not used by any driver in this note — see
"Method") continued changing throughout this run: mtime moved from 12:06
(noted in the [event-stats baseline note](2026-08-10_sparse-evidence-gnn_event-stats-non-graph-baseline.md))
to 15:00, confirming another concurrent session kept iterating on it
through the ~65-minute sweep window. All canonical config used here was
hardcoded from the zero_channel_embed re-check note, not read from that
file, for exactly this reason.

See [[sparse-evidence-gnn-dense-event-mode-untested]] (memory, now
superseded — dense mode is tested as of this note),
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]], and
[[sparse-evidence-gnn-channel-encoder-dominates]].
