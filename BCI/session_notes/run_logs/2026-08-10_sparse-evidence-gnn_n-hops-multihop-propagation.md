# Sparse-Evidence-GNN multi-hop (`n_hops`) message passing — 2026-08-10

- Run IDs: `nhops-subj1-baseline-h1` (n_hops=1), `nhops-subj1-sweep-h2`
  (n_hops=2) — single seed each (canonical `seed=44`), no seed sweep yet.
- Dataset: BNCI2014-001, subject 1 only, `LeftRightImagery` paradigm,
  `CrossSessionEvaluation` (`0train`/`1test` folds).
- Base config: canonical `SPARSE_EVIDENCE_GNN_PARAMS` (epochs=75, see
  [[sparse-evidence-gnn-reorg-2026-08-09]]) unmodified except `n_hops`.
  `surrogate_seed` is pinned in the canonical config, so both runs hit the
  **same** surrogate-null cache entries and therefore build **identical**
  sparse events — the only thing that differs between the two runs is
  `n_hops` itself, not event content/count.

## State of the file this was built on (concurrent-edit check)

Per the task's own instruction to flag this: `sparse_evidence_gnn_classifier.py`
had substantial **uncommitted** changes already in the working tree before
this session started (`git diff --stat` against HEAD `efe04ee` showed ~785
insertions/98 deletions pre-existing) — the 2026-08-09 single-file
reorganization ([[sparse-evidence-gnn-reorg-2026-08-09]]) plus the
same-day 2026-08-10 `event_aggregation="gated_softmax"` addition
([[sparse-evidence-gnn-channel-encoder-dominates]]'s follow-up). `n_hops`
was implemented **on top of** that state, not against a clean checkout.
Checked the file's mtime immediately after finishing the `n_hops` edits
(08:31:33) and again after both training runs completed (~08:37) — mtime
unchanged, confirming no other concurrent session edited this file while
these results were being collected. The accuracy numbers below therefore
reflect `n_hops` alone, on top of (not instead of) the reorg +
gated_softmax state, not some earlier/cleaner version of the file.

## What was added

`SparseEvidenceGNNCore.__init__` gained `n_hops: int = 1` (validated `>=1`,
raises otherwise), following the same "gated behind a flag, default
preserves current behavior exactly" pattern as `scale_adaptive_smoothing`
and `event_aggregation`:

- `n_hops=1` (default): forward() never calls the new code path at all —
  bit-for-bit unchanged from before this change.
- `n_hops=K>1`: after the existing single-hop `_aggregate_events` builds
  per-channel evidence (unchanged), `_propagate_hops` runs `K-1` additional
  rounds of message passing over the same 36-canonical-edge topology real
  events use (`src_idx`/`dst_idx`), but bidirectionally (both `i->j` and
  `j->i`, via new `hop_src_idx`/`hop_dst_idx` buffers — the base topology is
  undirected). Each round: `hop_message_mlp` (2-layer, `2*hidden_dim ->
  hidden_dim`) forms one message per directed edge from `[h_dst, h_src]`,
  messages are `scatter_add`ed per destination node, and a `GRUCell`
  (`hop_update`) folds the incoming neighbor evidence into each node's
  prior-hop state (Gilmer et al. 2017 MPNN-style gated update).

`hop_message_mlp`/`hop_update` are constructed unconditionally (like
`event_gate` before it) but **after** every pre-existing submodule
(`channel_encoder`, `sparse_message_mlp`, `event_gate`, `sparse_classifier`)
so their construction cannot shift the RNG draws those layers consume —
verified directly (see below), not just asserted from reading the order.

`n_hops` was threaded through `SparseEvidenceGNNClassifier.__init__` ->
`_build_model` the same way `event_aggregation` was. `WCTEvidenceGNNCore`
was not touched, per the task's scope.

## Correctness (smoke test, before any real-data run)

Random CWT-shaped tensors through `compute_events -> forward -> .backward()`
(script: not committed, see `n_hops` smoke test in this session's tool
transcript for the exact checks) verified:

- `n_hops=1`: `hop_message_mlp`/`hop_update` receive **zero** gradient (dead
  code path, as intended) while every pre-existing submodule gets real
  gradient — confirms `n_hops=1` is a true no-op addition.
- `n_hops=2` and `n_hops=3`: finite logits, and gradient flows into
  `hop_message_mlp`/`hop_update` as well as every pre-existing submodule.
- Zero-event edge case (`coherence_threshold=1e6`, gate never fires): ran
  without crashing at `n_hops` in `{1, 2, 3}` — `_propagate_hops` on an
  all-zero initial node state doesn't NaN or error (message-MLP biases
  produce some nonzero message even with zero input, which is expected
  learned behavior, not a bug).
- RNG-parity check: built two models under the identical seed, one with
  `n_hops=1` and one with `n_hops=2`, and confirmed every pre-existing
  submodule's parameters (`channel_encoder`, `sparse_message_mlp`,
  `event_gate`, `sparse_classifier`) are **bit-identical** between the two —
  directly verifies the "constructed after everything else" ordering claim
  above, not just the doc comment asserting it.

All smoke tests passed before the real-data run below.

## Results: subject 1, n_hops=1 (baseline) vs n_hops=2

| n_hops | 0train | 1test | mean |
| --- | --- | --- | --- |
| 1 (baseline) | 0.8733 | 0.8187 | **0.8460** |
| 2 | 0.8281 | 0.7961 | **0.8121** |

n_hops=2 is **worse** by ~0.034 (both folds individually worse, not just on
average) at this single seed. Single-seed numbers on this pipeline carry a
known catastrophic-failure risk on the `0train` fold specifically (see
[[sparse-evidence-gnn-seed-variance]]), so this is not yet a confirmed
regression across seeds — but it gives no reason to expect n_hops=2 helps,
and a mild reason to expect it hurts.

## Discussion

Consistent with two prior findings on this pipeline, not contradicting them:

- [[sparse-evidence-gnn-channel-encoder-dominates]]: accuracy comes almost
  entirely from `ChannelSignalEncoder`, not the event pathway.
- The 2026-08-10 `event_aggregation="gated_softmax"` result (same file,
  earlier today): giving the event pathway a *within-hop* weighting
  mechanism changed nothing, and `sparse_message_mlp` was independently
  shown to mostly ignore each event's own content and encode channel
  identity instead (see that session's `msg(real) vs msg(ablated)` check).

`_propagate_hops` operates on exactly the same per-channel evidence vectors
that `sparse_message_mlp` already produces — if the network has learned to
route most of its useful signal through `channel_encoder`'s embeddings
(present on every event regardless of gate-worthiness) rather than event
content, then propagating THAT evidence one hop further mostly moves
"which channel you are, restated" around the graph rather than adding new
discriminative information, while adding two more trainable modules
(`hop_message_mlp`, `hop_update`) and 5 more scatter/GRU operations to fit
per hop — plausible extra-capacity overfitting on a 144-trial single-subject
training set, matching the observed direction of the effect.

## Not yet done

- No multi-seed comparison (only `seed=44`) — given
  [[sparse-evidence-gnn-seed-variance]]'s single-seed catastrophic-failure
  risk, this single-seed drop should not be treated as conclusive on its
  own.
- No 4-subject canonical sweep — explicitly deferred by the task until the
  single-subject n_hops=1-vs-2 comparison justified it, and it didn't (a
  regression, not an improvement, on the one seed tested).
- `n_hops=3+` untested on real data (smoke-tested only).

Given the single-seed result points the wrong direction and the two prior
findings above already suggest the event/graph pathway is not where this
pipeline's accuracy lives, a multi-seed re-check of `n_hops=2` (rather than
jumping straight to a 4-subject sweep) is the natural next step if this is
revisited, not a full sweep at the current evidence level.

See also [[sparse-evidence-gnn-reorg-2026-08-09]] (`SPARSE_EVIDENCE_GNN_PARAMS`,
the base config this varied) and [[sparse-evidence-gnn-channel-encoder-dominates]].
