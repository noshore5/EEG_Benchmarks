# Sparse-Evidence-GNN batch_size=8 vs. batch_size=10, 8-seed sweep — 2026-08-06 22:43

- Run IDs: `sweep-bs8-s{42..49}`, `sweep-bs10-s{42..49}` (16 runs total)
- Source artifacts: `results_sweep-bs{8,10}-s{seed}__cross__inner-none.{hdf5,csv}`,
  `summary_sweep-bs{8,10}-s{seed}__cross__inner-none.md` (under
  `/Users/noahshore/mne_data/results/LeftRightImagery/CrossSessionEvaluation/`)
- Experiment logs: `/Users/noahshore/mne_data/sweep-bs{8,10}-s{seed}/experiment_latest.log`
- Dataset: BNCI2014-001, **subject 1 only**, `LeftRightImagery` paradigm,
  `CrossSessionEvaluation`
- All other params fixed at the current canonical config (epochs=100,
  see full parameter block below); only `batch_size` and `seed` varied.
- Executed 3-way parallel (`xargs -P3`) via
  `coheriqs_contributions/run_wct_gnn.py --param-names batch_size seed
  --param-values <bs> <seed>`, ~14.5 minutes wall clock for all 16 runs.

Follow-up to the earlier 4-seed/4-batch-size sweep
([[sparse-evidence-gnn-seed-variance]]), prompted by a single-seed (seed=42)
bs=8 vs. bs=10 comparison that looked like a wash (0.8383 vs 0.8304 mean) —
this sweep checks whether that's a stable difference or noise, and whether
bs=10 reproduces the seed-43 `0train` failure mode seen earlier at bs=4/16.

## Results

| bs | seed | 0train | 1test | mean |
| --- | --- | --- | --- | --- |
| 8 | 42 | 0.8792 | 0.7975 | 0.8383 |
| 8 | 43 | 0.8709 | 0.7984 | 0.8347 |
| 8 | 44 | 0.8764 | 0.7600 | 0.8182 |
| 8 | 45 | 0.8698 | 0.7712 | 0.8205 |
| 8 | 46 | 0.8438 | 0.8385 | 0.8411 |
| 8 | 47 | 0.8663 | 0.8274 | 0.8468 |
| 8 | 48 | 0.8517 | 0.7961 | 0.8239 |
| 8 | 49 | 0.8970 | 0.8019 | 0.8494 |
| 10 | 42 | 0.8681 | 0.7928 | 0.8304 |
| **10** | **43** | **0.5482** | 0.8040 | 0.6761 |
| 10 | 44 | 0.8688 | 0.7542 | 0.8115 |
| 10 | 45 | 0.8862 | 0.7973 | 0.8417 |
| 10 | 46 | 0.8463 | 0.8522 | 0.8492 |
| 10 | 47 | 0.8586 | 0.8368 | 0.8477 |
| 10 | 48 | 0.8376 | 0.7897 | 0.8137 |
| 10 | 49 | 0.8810 | 0.8135 | 0.8472 |

| bs | 0train mean (std) | 1test mean (std) | overall mean (std) |
| --- | --- | --- | --- |
| 8  | 0.8694 (0.0154) | 0.7989 (0.0242) | 0.8341 (0.0112) |
| 10 | 0.8243 (0.1055) | 0.8051 (0.0281) | 0.8147 (0.0542) |

**Finding**: seed 43 collapses on `0train` at bs=10 (0.548), essentially
replicating the same seed's collapse at bs=16 and bs=4 (~0.57) from the
earlier sweep, with `1test` staying normal every time (0.80 here). Excluding
seed 43, bs=10's distribution is indistinguishable from bs=8's. bs=8 remains
the only batch size (of 4/8/10/16/32 tested across two sweeps) that has not
reproduced this failure for seed 43. That's stronger evidence than the
original 4-seed sweep gave, but the fact that seed 43 fails across most other
batch sizes suggests the bad-seed effect mostly lives in seed 43's
init/validation-split draw interacting with the `0train` session's data, not
primarily in batch size -- bs=8 most likely has training dynamics that happen
to route around it rather than being structurally immune.

## Epoch timing

Aggregated across all 8 seeds per batch size (1600 epoch-records each: 8
seeds x 2 sessions x 100 epochs). Run 3-way parallel, so these are **not**
comparable to isolated single-run timings (~0.31-0.40s/epoch measured
earlier the same day) -- CPU contention from 3 concurrent training
processes inflates every number here.

| bs | mean epoch_time | min | max |
| --- | --- | --- | --- |
| 8  | 0.5323s | 0.4300s | 0.7400s |
| 10 | 0.5582s | 0.3100s | 1.0000s |

## Full effective parameters (shared across all 16 runs; only batch_size/seed vary)

```
channel_embed_dim: 8
channel_encoder_dilation: 5
channel_subset: [1, 5, 7, 8, 9, 10, 11, 13, 17]
coherence_threshold: 0.5
coi_enabled: True
cwt_resample_n_time: None
device: 'auto'  (resolved to cpu)
early_stopping_patience: None
epochs: 100
grad_clip_norm: 0.1
hidden_dim: 8
highest: 35.0
last_batch_min_ratio: 0.0
learning_rate: 0.001
lowest: 8.0
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
phase_threshold_deg: 30.0
raw_x_resample_n_time: None
sampling_rate: 250
selector_alpha_val_update_rate: 1.0
smooth_kernel_sigma: (None, None)
smooth_kernel_size: (5, 3)
validation_group_column: None
validation_split: 0.2
verbose: 2
weight_decay: 0.0001
```

See also [[sparse-evidence-gnn-seed-variance]] for the original 4-seed/
4-batch-size sweep this replicates and extends, and
[[sparse-evidence-gnn-native-resolution-fix]] for the epochs=100/batch_size=8
canonical run this was checking a candidate alternative to.
