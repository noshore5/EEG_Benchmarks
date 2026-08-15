# Sparse-Evidence-GNN 9-subject sweep, `phase_threshold_deg=10`, post MPS/channel-encoder fixes — 2026-08-09 17:08–17:57

- Run ID: `sweep-9subj-post-fixes`
- Source artifacts: `results_sweep-9subj-post-fixes__cross__inner-none.hdf5`,
  `scores_sweep-9subj-post-fixes__cross__inner-none.csv`,
  `summary_sweep-9subj-post-fixes__cross__inner-none.md` (under
  `/Users/noahshore/mne_data/results/LeftRightImagery/CrossSessionEvaluation/`)
- Experiment log: `/Users/noahshore/mne_data/sweep-9subj-post-fixes/experiment_20260809-170823-551812.log`
- Dataset: BNCI2014-001, **all 9 subjects**, `LeftRightImagery` paradigm,
  `CrossSessionEvaluation`
- Executed single-process, serial, via `run_wct_gnn.py --subjects 1 2 3 4 5
  6 7 8 9 --run-id sweep-9subj-post-fixes --param-names
  coherence_threshold_mode surrogate_count surrogate_percentile
  phase_threshold_deg highest batch_size --param-values surrogate 100 99.0
  10 30.0 8`; ~49 minutes wall clock (17:08:23 → 17:57:26), CPU (`device:
  auto` resolved to `cpu`).
- Off-canonical overrides vs. the `run_canonical_setup.py` "sparse" config:
  `phase_threshold_deg=10` (canonical is 30) and `highest=30.0` (canonical
  is 35.0); everything else at canonical, including `surrogate_percentile=
  99.0`.

First full 9-subject run after three fixes landed the same day/prior day:
the [[sparse-evidence-gnn-mps-nonzero-bug]] fix (`_max_cluster_statistic` /
`_build_sparse_events` forced off MPS), the 72→36-edge canonical topology
change, and the revert of the `ChannelSignalEncoder` per-timestep
regression back to `AdaptiveAvgPool1d` pooling
([[sparse-evidence-gnn-channel-encoder-event-locality-fix]]).
`phase_threshold_deg=10` was carried over from the single-variable subject-2
experiment in the same day's mu-band/scale-adaptive session notes (Arc 1 of
`2026-08-09_mu_band_relaxation_scale_adaptive_kernel_and_channel_encoder_regression.md`)
that showed a mild recovery for subject 2 alone (0.517, vs. worse for the
other 3 single-variable levers tried there) — not picked from a full sweep,
just the least-bad single lever found for subject 2 that day.

## Results

| subject | 0train | 1test | mean |
| --- | --- | --- | --- |
| 1 | 0.8798 | 0.8291 | 0.8545 |
| 2 | 0.5062 | 0.5274 | 0.5168 |
| 3 | 0.9633 | 0.9774 | 0.9704 |
| 4 | 0.7016 | 0.6397 | 0.6706 |
| 5 | 0.4892 | 0.5332 | 0.5112 |
| 6 | 0.7209 | 0.6944 | 0.7077 |
| 7 | 0.6481 | 0.6723 | 0.6602 |
| 8 | 0.9516 | 0.9628 | 0.9572 |
| 9 | 0.8229 | 0.9342 | 0.8786 |
| **mean** | — | — | **0.7474** |

**Findings**:
- Subject 2 (0.517) reproduces almost exactly the single-subject Arc-1
  number (0.517) from the same day's mu-band session notes — consistent,
  not a fluke of that isolated test.
- Subjects 1–4 vs. the earlier post-MPS-fix 4-subject validation in
  [[sparse-evidence-gnn-native-resolution-fix]] (0.847/0.553/0.969/0.666,
  `phase_threshold_deg=30`, no `channel_subset`/other differences noted
  there): subject 1 up (+0.008), subject 2 down (-0.036), subject 3 flat
  (+0.001), subject 4 up (+0.005). Small, mixed deltas — not a clean win
  from `phase_threshold_deg=10` at the whole-cohort level, despite the
  subject-2-only Arc-1 result looking like a recovery in isolation.
  **Single seed** — see [[sparse-evidence-gnn-seed-variance]] before reading
  much into any per-subject delta here.
- New subjects 5–9 span a wide range (0.511–0.957), with subjects 5, 7 near
  subject-2/4-tier difficulty and subjects 8, 9 near subject-3-tier ease.
  No prior canonical baseline exists for subjects 5–9 to compare against —
  this is the first time they've been run under this pipeline.
- Overall 9-subject mean **0.7474** is the first whole-cohort accuracy
  number for this pipeline; prior reported means (0.711, 0.725, 0.759,
  0.838, etc. — see [[sparse-evidence-gnn-native-resolution-fix]] and the
  `bs8-vs-bs10` sweep) were all subjects-1–4-only or subject-1-only.

## Full effective parameters

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
epochs: 100
feature_ablation: 'none'
grad_clip_norm: 0.1
hidden_dim: 8
highest: 30.0
last_batch_min_ratio: 0.0
learning_rate: 0.001
lowest: 8.0
mu_band_range_hz: (8.0, 13.0)
mu_band_surrogate_percentile: None
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
seed: 42
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

See also [[sparse-evidence-gnn-seed-variance]] (single-seed caveat),
[[sparse-evidence-gnn-mps-nonzero-bug]] and
[[sparse-evidence-gnn-channel-encoder-event-locality-fix]] (the two fixes
landed just before this run), and [[sparse-evidence-gnn-native-resolution-fix]]
(prior 4-subject baseline this partially compares against).
