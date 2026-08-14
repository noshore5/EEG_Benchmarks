# Dense-Edge: time-averaged vs. time-resolved, 4-seed sweep, subject 1 (2026-08-11)

Branch: `channel_subset`. Repo: `/Users/noahshore/Documents/CoherIQs/moabb`.
Two 4-seed sweeps of `--pipeline dense_edge`, run in parallel, differing in
exactly one axis: whether `dense_edge` discards within-trial time resolution
(`time_averaged_graph=True`) or keeps its usual fixed-factor downsample
(`time_averaged_graph=False`, `dense_edge_time_downsample=8`). See
[[sparse-evidence-gnn-dense-edge-time-handling]] for the mechanism and
`run_pipelines.py`'s `DENSE_EDGE_PARAMS` comment for why
`dense_conv_kernel_size`/`dense_conv_pool_size` had to be listed explicitly
there first (they weren't overridable via `--param-names` otherwise).

## Method

Both sweeps: `BNCI2014_001` subject 1 only, `CrossSessionEvaluation`
(`0train`/`1test` folds), seeds 42-45, current `DENSE_EDGE_PARAMS` (
`event_mode="dense"`, `event_aggregation="concat"`, `epochs=60`,
`batch_size=8`, `learning_rate=1e-3`, `weight_decay=1e-4`,
`grad_clip_norm=0.1`, `hidden_dim=8`, `channel_embed_dim=8`,
`channel_encoder_dilation=5`, `phase_threshold_deg=10.0`,
`coherence_threshold_mode="fixed"`, `coherence_threshold=0.99`,
`surrogate_percentile=99.0`, `channel_subset=[1,5,7,8,9,10,11,13,17]`,
`feature_ablation="none"` -- full pipeline, channel embeddings included,
**not** the `zero_channel_embed` event-pathway-alone ablation the earlier
4-subject time-averaged result used) as the shared base.

- **OFF**: `time_averaged_graph=False` (default) -- `dense_edge_time_downsample=8`,
  `dense_conv_kernel_size=5`, `dense_conv_pool_size=4` (all `DENSE_EDGE_PARAMS`
  defaults, untouched).
- **ON**: `--param-names time_averaged_graph dense_edge_time_downsample
  dense_conv_kernel_size dense_conv_pool_size --param-values true 1 1 1`.

Run-ids: `dense-edge-off-s{42,43,44,45}-4seedsweep-20260811`,
`dense-edge-timeavg-s{42,43,44,45}-4seedsweep-20260811`. Both sweeps'
4 seeds ran concurrently with each other (8 fits total sharing the machine),
so per-epoch wall-clock during the run (~3s/epoch on OFF) was inflated by
contention -- not representative of either condition's isolated cost (see
[[sparse-evidence-gnn-dense-edge-time-handling]] for isolated numbers:
~0.6-0.7s/epoch for `dense_edge_time_downsample=8`, ~3.2-3.9s/epoch native).

## Results

| condition | seed 42 | seed 43 | seed 44 | seed 45 | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| Time-resolved (OFF) | 0.9251 | 0.9282 | 0.9173 | 0.9329 | **0.9259** | 0.0057 |
| Time-averaged (ON) | 0.9198 | 0.9241 | 0.9187 | 0.9212 | **0.9210** | 0.0020 |

Per-fold detail:

| condition | seed | 0train | 1test |
| --- | --- | --- | --- |
| OFF | 42 | 0.9007 | 0.9495 |
| OFF | 43 | 0.9020 | 0.9544 |
| OFF | 44 | 0.8883 | 0.9464 |
| OFF | 45 | 0.9066 | 0.9591 |
| ON | 42 | 0.8995 | 0.9402 |
| ON | 43 | 0.9019 | 0.9464 |
| ON | 44 | 0.8963 | 0.9412 |
| ON | 45 | 0.8960 | 0.9464 |

## Interpretation

Time-resolved edges out time-averaged by ~0.005 (0.9259 vs 0.9210) --
noise-level given time-resolved's own 0.0057 seed std, and the direction
flips on one of the four seeds (44: time-averaged wins by 0.0014). This
reads as a genuine tie on subject 1 under the current `concat`/`feature_
ablation="none"` config, consistent with the earlier 4-subject finding
under a different config (`zero_channel_embed`,
[[sparse-evidence-gnn-time-averaged-graph-feature]]: 0.7999 vs 0.7897,
also a tie/slight edge for time-averaged there) -- direction of the small
edge isn't consistent across the two config points, reinforcing that the
effect size is genuinely near zero rather than a real, sign-consistent
advantage either way.

The more interesting difference here is **seed variance**, not mean
accuracy: time-averaged is ~3x more consistent across seeds (std 0.0020 vs
0.0057). Plausible mechanism -- collapsing `T` to 1 removes essentially all
of `dense_edge_conv`'s own capacity to overfit to a particular random init's
early trajectory through the time axis (no more Conv2d+MaxPool2d stack to
learn arbitrary local-window emphasis, just a fixed COI-weighted average
feeding a `1x1` conv), so a lower-capacity representation is a priori less
seed-sensitive. Not confirmed against other capacity-reduction axes (e.g. a
smaller `hidden_dim` under the time-resolved config) -- worth a follow-up
if seed-variance reduction becomes a goal in itself, independent of whether
it also affects mean accuracy.

## Caveats

1. **Single subject.** Subject 1 is historically one of this pipeline's
   easier subjects (see [[sparse-evidence-gnn-seed-variance]] and every
   other single-subject entry in this ablation series) -- not a cohort
   result. The prior 4-subject time-averaged comparison used a different
   `feature_ablation` setting, so it's suggestive corroboration, not a
   direct replication.
2. **`feature_ablation="none"` here, `"zero_channel_embed"` in the earlier
   4-subject test.** Both include channel-embedding contributions in this
   run, unlike the earlier "event pathway alone" isolation -- means the two
   results aren't strictly apples-to-apples, only directionally consistent.
3. **Wall-clock numbers from this run are not representative** of either
   condition's real per-epoch cost -- both sweeps ran concurrently (8 fits
   contending for the same cores). Use
   [[sparse-evidence-gnn-dense-edge-time-handling]]'s isolated numbers for
   actual cost comparisons.
