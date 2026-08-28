# temporal_graph_mamba: wide-band experiment (negative), aggregate pre/post knob, graph-state-mamba merged (2026-08-28)

Same session as `hermitian_ssm_first_6fold_and_eigh_fix.md`. After the
hermitian_ssm float16 6-fold landed (AP 0.253, still the weakest
pipeline), turned to `temporal_graph_mamba` -- the current prediction
leader (AP 0.674) -- with two questions:

1. Does its 8-40 Hz / nfreqs=8 band (a 2026-08-16 disk-budget choice,
   never swept for accuracy) leave signal on the table?
2. It collapses the 253-edge coherence graph to 23 nodes *before* the
   Mamba. Does retaining the edge graph through the temporal model help?

## Q1: wide-band experiment -- ABANDONED, clean negative

`PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS` widened toward hermitian_ssm's
8-124 Hz regime.

| attempt | config | fold 1_03 AP | fold 1_04 AP | verdict |
|---|---|---|---|---|
| baseline | 8-40, nfreqs=8, tds=16 | 0.792 | 0.830 | -- |
| 1 | 8-124, nfreqs=15, tds=32 | 0.307 | 0.315 | killed after 2 folds |
| 2 | 8-124, nfreqs=10, tds=16 | 0.236 | -- | killed after fold 1 |

Attempt 1 cut `dense_edge_time_downsample` 16->32 (T 480->240 into the
Mamba) to keep the `cwt_window_cache`/`dense_edge_cache` disk footprint
from ~doubling (both scale ~linearly in nfreqs; ~70 GB at nfreqs=8).
Attempt 2 isolated the band from that tds cut -- and came in **worse**
(0.236 < 0.307), so the tds halving was not the cause.

**The wide band itself hurts.** 40-124 Hz scalp-EEG coherence is
dominated by EMG / line-noise harmonics / broadband muscle, and
`temporal_edge_proj` (`Linear(4*nfreqs -> temporal_graph_edge_dim=8)`)
linearly mixes all frequency bins into the 8-wide neck -- so adding junk
bands dilutes the 8-40 Hz signal that the baseline relies on. Reverted to
`_SHARED_ARCH_PARAMS`; kept the finding as a comment block in the params
dict.

Corollary finding (not acted on): the whole `temporal_graph_mamba`
pathway funnels the full precomputed `[4, E, T, F]` edge stack
(coherence, sinphi, cosphi, significance -- full detail) through
`temporal_graph_edge_dim=8` then `hidden_dim=8` (= the Mamba's d_model).
Both widths are 2026-08-16 disk-budget picks, never swept. Widening the
neck (bump both to ~32 in the `TEMPORAL_GRAPH_*_PARAMS` dicts, NOT
`_SHARED_ARCH_PARAMS`) or replacing the learned `4F->8` squeeze with a
fixed band-pool (theta/alpha/beta/gamma) at precompute time are the
candidate levers. Filed under Open threads in CONTEXT.md.

## Q2: temporal_graph_aggregate pre/post knob -- BUILT

New param `temporal_graph_aggregate: "pre" | "post"` on
`SparseEvidenceGNNCore` + `SparseEvidenceGNNClassifier` (Streaming
inherits). `temporal_graph_mode="mamba"` only.

- **"pre"** (default, bit-identical to before): scatter-mean the
  `[B, E, T, H]` per-edge messages to `[B, C, T, H]` node sequences
  FIRST, then one Mamba per node over T. This is the AP 0.674 run.
- **"post"**: run `_DenseEdgeMambaTemporal` over each of the ~253 EDGE
  sequences' own T FIRST (its original per-edge role -- the
  `[B, C_in, X, T] -> [B, out, X, 1]` contract is X-agnostic), THEN
  scatter-mean the per-edge summaries to C nodes. Same `dst_idx` /
  `temporal_node_in_degree` divisor as "pre". The temporal model keeps
  the full edge graph instead of a per-timestep 23-node average.

Cost: ~E/C (~11x) more Mamba rows. But `n_rows = B*E` (~8000) far
exceeds `temporal_graph_mamba_chunk_size=128`, so `_DenseEdgeMambaTemporal`
gradient-checkpoints (which it barely does in "pre" mode, B*C ~= 736) and
peak memory stays bounded. **Cache `[B, 4, E, T, F]` is unchanged** --
this is a pure forward-path reordering.

Implementation: ~20 lines in `_temporal_graph_node_states` (early-return
branch before the node aggregation) + param threading + a guard
(`"post"` requires `temporal_graph_mode="mamba"`, else ValueError).
Unit-tested: both paths forward+backward finite, gradients flow, guard
raises, `--pipeline temporal_graph_mamba --smoke` exits 0 for both.

Committed `2e48dd2` (knob + wide-band revert). `run_pipelines.py`'s
committed default is `"pre"`; a 6-fold `"post"` A/B (only that knob
changed, baseline band/widths) is running as of this writeup.

## graph-state-mamba merged to main

`graph-state-mamba` was 9 commits ahead of `main`, 0 behind ->
fast-forward. Carried: `temporal_graph_gru`/`temporal_graph_mamba`
`--pipeline` wiring, the entire `hermitian_ssm` pipeline + its 3 6-fold
result sets, and the `temporal_graph_aggregate` knob. Merged (FF), pushed,
branch deleted local + `origin`. `main` is the working branch again.

## Open / next

- The `"post"` A/B result (pending). If it beats 0.674, flip the default.
- Seed-repeat the `temporal_graph_mamba` "pre" baseline -- the 0.674 mean
  has per-fold AP 0.29-1.00 (1 subject, 6 seizures, ~30 preictal
  windows/fold) and is fragile (a mild reg change collapsed fold 1_03
  0.79->0.23 on 2026-08-27). Confirm 0.674 is real before building on it.
- `temporal_graph_gru` (per-node GRU) 6-fold, for the gru-vs-mamba
  comparison at the aggregate-first structure.
- Neck-widening (`hidden_dim`/`temporal_graph_edge_dim` 8->32) and/or
  precompute-time band-pooling -- see Q1 corollary.
- hermitian_ssm parked at AP 0.253; the eigen-feature diagnosis (rank-1
  effective, power stripped) and its candidate fixes are in
  `hermitian_ssm_first_6fold_and_eigh_fix.md` + CONTEXT.md.
