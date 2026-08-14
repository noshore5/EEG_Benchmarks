# Sparse-Evidence-GNN `zero_channel_embed` ablation, re-checked under current thresholds + isolated phase_threshold_deg sweep, 4 seeds each — 2026-08-10

- Run ID: `zero-channel-embed-4seed` (reused sequentially across all 4 seeds
  in one process, not concurrent invocations -- safe against the documented
  CSV/summary-writer race, see [[run-wct-gnn-concurrent-write-race]]; only
  the last seed's own summary/CSV artifact survives on disk, per-seed history
  below and in `~/mne_data/run_ledger.csv`).
- Dataset: BNCI2014-001, **subject 1 only**, `LeftRightImagery` paradigm,
  `CrossSessionEvaluation`.
- `feature_ablation="zero_channel_embed"` -- zeros the `ChannelSignalEncoder`
  src/dst embedding block right before `sparse_message_mlp`, keeping only
  each event's own `[t, freq, mag, sinφ, cosφ]`. All other params at the
  current canonical config (`n_hops=1`, `freq_aware_hops=False` forced
  explicitly by the driver, independent of whatever
  `run_sparse_evidence_gnn.py`'s tracked file happened to have at the time).
- Executed via a throwaway scratchpad driver script (not committed) that
  imports `SPARSE_EVIDENCE_GNN_PARAMS`/`SUBJECTS` from
  `run_sparse_evidence_gnn.py`, overrides `feature_ablation`/`n_hops`/
  `freq_aware_hops`/`seed` per iteration, and calls `run_wct_gnn.main()`
  directly, seeds 42-45 in sequence -- same methodology as the [8-seed
  epochs sweep](2026-08-09_sparse-evidence-gnn_epochs50-75-100_8seed-sweep.md).

## Why

[[sparse-evidence-gnn-channel-encoder-dominates]] reported a single-seed
(seed=42) `zero_channel_embed` result of 0.569 mean -- "collapses to near
chance" -- but that measurement was taken under an **older** gating config
(`phase_threshold_deg=30`, `surrogate_percentile=95`, stricter → fewer
events). The canonical config has since moved to `phase_threshold_deg=10`/
`surrogate_percentile=99` (looser → far more events). This run re-checks
whether the "near chance" collapse still holds under current thresholds, and
adds 3 more seeds since the original claim was single-seed.

## Results

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.624 | 0.622 | 0.623 |
| 43 | 0.615 | 0.605 | 0.610 |
| 44 | 0.656 | 0.636 | 0.646 |
| 45 | 0.638 | 0.629 | 0.633 |

mean=0.628, std=0.0153 (sample), min=0.610, max=0.646. Chance = 0.5.

### Comparison to same-seed `feature_ablation="none"` baseline

Baseline numbers reused from the [8-seed epochs=75
sweep](2026-08-09_sparse-evidence-gnn_epochs50-75-100_8seed-sweep.md) (same
current thresholds, same subject, seeds 42/43/44/45):

| seed | none (baseline) | zero_channel_embed | gap |
| --- | --- | --- | --- |
| 42 | 0.8557 | 0.623 | 0.233 |
| 43 | 0.8590 | 0.610 | 0.249 |
| 44 | 0.8518 | 0.646 | 0.206 |
| 45 | 0.8443 | 0.633 | 0.211 |

mean baseline=0.8527, mean zero_channel_embed=0.628, mean gap=0.225.

## Finding

The qualitative claim in [[sparse-evidence-gnn-channel-encoder-dominates]]
needs revising: under today's canonical (looser) gating thresholds,
`zero_channel_embed` does **not** collapse to near chance -- it lands clearly
above chance (0.628 vs. 0.5) and is stable across seeds (std=0.0153, tight
relative to this pipeline's usual seed variance). The gap to the full-feature
baseline is still large (~22.5 points) and channel embeddings still clearly
dominate accuracy, but the event-only pathway is no longer negligible the way
the single-seed 0.569 number suggested.

Best-guess mechanism: `phase_threshold_deg=10`/`surrogate_percentile=99`
produces far more events per trial than the stricter config the original
0.569 was measured under. Even with channel identity zeroed, a much denser
event set gives the model channel-blind aggregate statistics to key on
(event counts/timing/frequency distribution per channel-pair), where a
sparser event set under the old thresholds apparently didn't leave enough
signal for that.

### Follow-up: isolated `phase_threshold_deg` sweep (same day, same seeds)

Confirmed directly with a second 4-seed run, `phase_threshold_deg=30` as the
**only** variable changed (`surrogate_percentile` held at the current
canonical 99, not reverted to the original 95 -- a genuinely isolated
single-variable test, run-id `zero-channel-embed-4seed-phase30`):

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.549 | 0.557 | 0.553 |
| 43 | 0.557 | 0.541 | 0.549 |
| 44 | 0.522 | 0.549 | 0.536 |
| 45 | 0.538 | 0.556 | 0.547 |

mean=0.546, std=0.0075 (sample) -- right at chance, matching the original
single-seed 0.569 finding and clearly below the phase_threshold_deg=10 result
above (0.628, std=0.0153) at the same seeds/subject with everything else
held fixed. This confirms `phase_threshold_deg` (gating strictness / event
density) is the driver of whether the event-only pathway collapses to chance
or carries real signal -- not some other config drift between the two
original measurements.

## Full effective parameters (shared across all 4 runs; only `seed` varies)

```
batch_size: 8
channel_embed_dim: 8
channel_encoder_dilation: 5
channel_subset: [1, 5, 7, 8, 9, 10, 11, 13, 17]
cluster_significance_percentile: 95.0
coherence_threshold: 0.0
coherence_threshold_mode: 'surrogate'
coi_enabled: True
cwt_resample_n_time: None
device: 'auto'  (resolved to cpu)
early_stopping_patience: None
epochs: 75
feature_ablation: 'zero_channel_embed'
freq_aware_hops: False
grad_clip_norm: 0.1
hidden_dim: 8
highest: 30.0
last_batch_min_ratio: 0.0
learning_rate: 0.001
lowest: 8.0
mu_band_range_hz: (8.0, 13.0)
mu_band_surrogate_percentile: None
n_hops: 1
nfreqs: 16
noise_apply_prob: 0.0
noise_augmentation_enabled: False
noise_bank_seed: None
noise_bank_size: 128
noise_strength: 0.0
normalize_input: True
optimizer_step_batch_mode: 'credit'
optimizer_step_batch_size: None
optimizer_step_remainder_policy: 'flush'
phase_threshold_deg: 10.0
raw_x_resample_n_time: None
sampling_rate: 250
scale_adaptive_cycles: 1.5
scale_adaptive_max_kernel: 101
scale_adaptive_smoothing: False
selector_alpha_val_update_rate: 1.0
smooth_kernel_sigma: (None, None)
smooth_kernel_size: (5, 3)
surrogate_cache_dir: None
surrogate_cache_enabled: True
surrogate_count: 100
surrogate_device: 'auto'
surrogate_percentile: 99.0
surrogate_seed: 42
validation_group_column: None
validation_split: 0.0
verbose: 2
weight_decay: 0.0001
```

See also [[sparse-evidence-gnn-channel-encoder-dominates]] for the original
finding this re-checks, [[sparse-evidence-gnn-seed-variance]] for this
pipeline's standing seed-variance risk, and
[2026-08-09_sparse-evidence-gnn_epochs50-75-100_8seed-sweep.md](2026-08-09_sparse-evidence-gnn_epochs50-75-100_8seed-sweep.md)
for the same-threshold `none`-ablation baseline used above.
