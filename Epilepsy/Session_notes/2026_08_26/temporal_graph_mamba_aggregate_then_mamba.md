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

## CLI wiring (completed, same session, after the above)

Added `--pipeline temporal_graph_gru` / `temporal_graph_mamba` to
`run_pipelines.py`:

- New param dicts `TEMPORAL_GRAPH_GRU_PARAMS`/`TEMPORAL_GRAPH_MAMBA_PARAMS`
  (detection) and `PREDICTION_TEMPORAL_GRAPH_GRU_PARAMS`/
  `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS` (prediction) -- exact copies of
  `DENSE_EDGE_GRU_PARAMS`'s numbers with `event_mode="temporal_graph"`,
  `event_aggregation="mean"` (required), `temporal_graph_mode="gru"`/
  `"mamba"`, same "isolated ablation, not a separately-tuned config"
  reasoning `DENSE_EDGE_MAMBA_PARAMS` already uses.
- `_dense_family_params`/`_dense_family_result_dir` extended with branches
  for both new pipeline names -- despite the name, this was the right place
  to add them rather than a parallel dispatch path: both still build a
  `SparseEvidenceGNNClassifier`/`StreamingSparseEvidenceGNNClassifier` from
  a plain `**clf_params` dict via the SAME `leave_one_seizure_out_
  prediction`/`leave_one_seizure_out_detection` loops every dense-family
  pipeline already uses, so reusing that dispatch avoided duplicating the
  whole call chain for a config that only differs by which dict comes back.
- Results land under `results/temporal_graph_gru/` and
  `results/temporal_graph_mamba/` -- own subdirectories, never pooled with
  dense-family CSVs (different `event_mode` entirely).
- Added both new names to `--pipeline`'s argparse `choices` + help text,
  and to the `cache_flag_explicit` print-message pipeline tuple (cosmetic
  only -- the dense-edge disk cache itself is generic/config-hash-keyed and
  needed no changes; `event_mode="temporal_graph"` reuses the exact same
  non-trainable `_build_dense_edge_input` precompute `event_mode="dense"`
  does, confirmed by the smoke run below: `[dense-edge cache] N/N trials
  reused from disk`).

**One real bug found and fixed via this session's own `--smoke` run**: the
first `temporal_graph_*` params dicts used `dense_edge_temporal_mode="rnn"`
as an "inert, never-read" placeholder (needed only because
`leave_one_seizure_out_prediction`/`detection`'s print statements read
`clf_params['dense_edge_temporal_mode']` unconditionally, regardless of
`event_mode`) -- but a PRE-EXISTING, unrelated validation in
`SparseEvidenceGNNClassifier.__init__` rejects `dense_edge_temporal_mode`
in `("rnn", "mamba")` outright for any `event_mode != "dense"`. Fixed by
using `"conv"` as the placeholder instead (the one value that check does
not restrict). Caught immediately by actually running `--smoke` rather
than trusting the code-reading pass alone -- exactly the reason this
repo's convention is "smoke-test before trusting a new pipeline wire-up."

**Verified, CPU, this session** (`--label-mode prediction --smoke
--max-folds 1`, real chb01 windows, disk-cache-reused CWT/dense-edge
features):

- `--pipeline temporal_graph_gru`: exit 0, wrote
  `results/temporal_graph_gru/prediction/*.csv`.
- `--pipeline temporal_graph_mamba`: exit 0, wrote
  `results/temporal_graph_mamba/prediction/*.csv`,
  `_DenseEdgeMambaTemporal use_cuda_kernel=False (mambapy pure-PyTorch
  pscan)` printed as expected (no CUDA on this machine).

Both smoke runs' 0/1 hit rate and precision/recall=0 are expected --
`--smoke` caps epochs at 2 and one fold, verifying wiring only, not model
quality (same as every other pipeline's `--smoke` mode).

## What's NOT done yet

- **No real-scale run.** Only the tiny synthetic smoke test and the
  1-fold/2-epoch CLI `--smoke` runs above. No real chb01 LOSO run (all
  seizures, real epoch budget), no comparison against `dense_edge_mamba`'s
  numbers or between `temporal_graph_mode="gru"` vs `"mamba"` at real scale.
- **No memory/speed characterization at scale.** `temporal_graph_mamba_
  chunk_size` defaults to 128 (copied from `_DenseEdgeMambaTemporal`'s own
  default), but the node axis here (`n_channels`, ~23) is far smaller than
  the edge axis (`E`, 253) that default was tuned for -- chunking is
  almost certainly unnecessary at this scale, untested at real batch sizes
  or on GPU.
- **Not pushed.** Committed locally on branch `graph-state-mamba` only.

## Next steps, in order

1. Real chb01 LOSO run (`--pipeline temporal_graph_gru` then
   `temporal_graph_mamba`, `--label-mode prediction`, full epoch budget,
   no `--smoke`/`--max-folds`) to see whether the aggregate-then-Mamba
   ordering changes anything relative to its GRU counterpart -- and
   separately, how "aggregate-then-X" (either backend) compares to
   `dense_edge_mamba`'s "X-then-aggregate" numbers, the actual open
   question this branch exists to answer.
2. If that shows anything interesting, revisit `temporal_graph_mamba_
   chunk_size`/GPU behavior at real batch sizes before trusting wall-clock
   comparisons against `dense_edge_mamba`.
