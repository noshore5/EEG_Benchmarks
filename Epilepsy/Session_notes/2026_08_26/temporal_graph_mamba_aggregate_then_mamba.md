# temporal_graph_mode="mamba": aggregate-then-Mamba, branch `graph-state-mamba`

## What this is

A per-node Mamba block for `event_mode="temporal_graph"`, as an alternative
to the existing `temporal_node_gru` (`nn.GRU`). This is the "aggregate-then-
temporal" ordering: mean-aggregate every edge's per-timestep message to its
destination node *first*, producing one graph-state sequence per node, then
run a single shared temporal model across that sequence. The existing
`dense_edge_mamba` pipeline does the mirror image -- Mamba per edge first,
aggregate the (already time-pooled) summaries once at the end.

`event_mode="temporal_graph"` itself already existed (2026-08-11, this
codebase) as exactly this "aggregate-then-GRU" architecture -- it was never
wired to a `--pipeline` CLI flag, but is a real, working classifier
capability (`SparseEvidenceGNNClassifier(event_mode="temporal_graph")`).
This session added a second temporal backend for it, mirroring the existing
`dense_edge_temporal_mode` "rnn" vs "mamba" choice for the PER-EDGE case.

## What changed

`Epilepsy/pipelines/cwt_gnn_classifiers.py`:

- New constructor param `temporal_graph_mode: Literal["gru", "mamba"] = "gru"`
  on both `SparseEvidenceGNNCore` and `SparseEvidenceGNNClassifier`, plus
  `temporal_graph_mamba_{d_state,d_conv,expand,n_layers,dropout,
  chunk_size,use_cuda_kernel}` mirroring `_DenseEdgeMambaTemporal`'s own
  knobs.
- Validation: `temporal_graph_mode` must be "gru"/"mamba"; "mamba" is
  rejected (`ValueError`) unless `event_mode="temporal_graph"` -- same
  "explicit no-op rejection" precedent as `event_aggregation`'s existing
  `temporal_graph` requirement.
- When `temporal_graph_mode="mamba"`, `self.temporal_node_mamba` is built by
  reusing `_DenseEdgeMambaTemporal` UNCHANGED, with the node axis (`n_channels`,
  e.g. 23) in the slot that class's usual caller (`dense_edge_conv`) puts the
  edge axis (`E`, 253) in -- its `[B, C_in, X, T] -> [B, out_channels, X, 1]`
  contract does not care what `X` indexes. `in_channels=out_channels=
  d_model=hidden_dim` for a like-for-like comparison against
  `temporal_node_gru`'s own hidden_dim->hidden_dim shape (no extra width
  knob added).
- `_temporal_graph_node_states`: everything through building `node_seq`
  (`[B, n_channels, T, hidden_dim]`, the per-timestep mean-aggregated node
  sequence) is untouched. New branch at the end: `temporal_graph_mode=
  "mamba"` permutes to `[B, hidden_dim, n_channels, T]`, calls
  `self.temporal_node_mamba`, and un-permutes the `[B, hidden_dim,
  n_channels, 1]` result back to `evidence`'s usual `[B, n_channels,
  hidden_dim]` shape. `"gru"` path is bit-identical to before.

`scripts/temporal_graph_mamba_smoke.py` (new): constructs, fits (2 epochs),
predicts on tiny synthetic data for both `"gru"` and `"mamba"`, checks
`predict_proba` rows sum to 1, prints trainable-param counts (confirms the
right submodule gets built, the other stays `None`), and checks the
`ValueError` guard fires for the invalid `event_mode="dense"` +
`temporal_graph_mode="mamba"` combination. **Ran successfully, CPU, this
session** -- both backends fit/predict cleanly; guard raises as expected.

## What's NOT done yet

- **No `--pipeline` CLI wiring.** `event_mode="temporal_graph"` (with either
  temporal backend) is usable today only by constructing
  `SparseEvidenceGNNClassifier` directly, the way this smoke test does --
  same state it was already in before this session, now just with a second
  temporal-backend choice available once wired. `run_pipelines.py`'s
  dense-family helpers (`_dense_family_param_dict`,
  `_dense_family_result_dir`, the pipeline-name validation list around line
  2138, the cache-flag check around line 2632) all gate on a fixed set of
  pipeline-name strings (`"dense_edge"`/`"dense_edge_gru"`/
  `"dense_edge_mamba"`) -- adding a `temporal_graph`/`temporal_graph_mamba`
  pipeline needs a new branch in each of those, not just a new params dict,
  and touches several dispatch points in a ~2600-line file. Deliberately
  NOT attempted this session to avoid touching working pipelines'
  dispatch logic without room to verify each site carefully.
- **No real-scale run.** Only the tiny synthetic smoke test above. No
  chb01 LOSO run, no comparison against `temporal_graph_mode="gru"` or
  against `dense_edge_mamba`'s numbers.
- **No memory/speed characterization.** `temporal_graph_mamba_chunk_size`
  defaults to 128 (copied from `_DenseEdgeMambaTemporal`'s own default),
  but the node axis here (`n_channels`, ~23) is far smaller than the edge
  axis (`E`, 253) that default was tuned for -- chunking is almost
  certainly unnecessary at this scale, untested at real batch sizes.

## Next steps, in order

1. CLI wiring: add `--pipeline temporal_graph_gru`/`temporal_graph_mamba`
   (or a single `--pipeline temporal_graph` plus a `--temporal-graph-mode`
   flag, following whichever naming convention reads more consistently
   with the existing `dense_edge`/`dense_edge_gru`/`dense_edge_mamba`
   trio) to `run_pipelines.py`, touching every dispatch site listed above.
2. Smoke run via that CLI path (`--smoke --max-folds 1`) before any real
   run, same discipline every other pipeline addition in this repo follows.
3. Real chb01 LOSO run, `temporal_graph_mode="gru"` vs `"mamba"`, to see
   whether the aggregate-then-Mamba ordering changes anything relative to
   its GRU counterpart -- and separately, how "aggregate-then-X" (either
   backend) compares to `dense_edge_mamba`'s "X-then-aggregate" ordering,
   the actual open question this branch exists to answer.
