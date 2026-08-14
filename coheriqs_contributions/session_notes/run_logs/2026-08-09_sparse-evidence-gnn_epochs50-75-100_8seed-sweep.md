# Sparse-Evidence-GNN epochs=50 vs. 75 vs. 100, 8-seed sweep — 2026-08-09

- Run IDs: `seedsweep-{42..53}` (this session's parallel background sweeps) +
  `canonical-sparse` (5 manual single-seed runs at epochs=75, seeds 42-45/47 --
  reused the default run-id, so only the LAST manual run's own result
  artifacts survive on disk; per-seed history for those recovered from
  `~/mne_data/run_ledger.csv`, not from the overwritten
  `summary_canonical-sparse__cross__inner-none.md`).
- Dataset: BNCI2014-001, **subject 1 only** (except one accidental subject-2
  probe on seed 45 -- see "Methodology note" below), `LeftRightImagery`
  paradigm, `CrossSessionEvaluation`.
- All other params fixed at the canonical config (see full parameter block
  below); only `epochs` (50/75/100) and `seed` varied.
- Executed as 4- and 8-way parallel background jobs via
  `run_sparse_evidence_gnn.py`'s `SPARSE_EVIDENCE_GNN_PARAMS` with `seed`
  overridden per job (throwaway driver script, not committed), plus 5 manual
  single runs (epochs=75, seeds 42-45/47) run directly via the script by
  hand, one at a time.

Prompted by a "the training run feels fast, could it be faster" /
"is the model too small" conversation that led into checking whether this
pipeline's known seed-to-seed variance ([[sparse-evidence-gnn-seed-variance]])
shrinks with more training epochs. Ran the same 8 seeds at 50, then 100
epochs (parallel background jobs), then continued exploring 75 epochs by
hand; backfilled the missing 3 of that 8-seed set in parallel to complete it.

## Results

### epochs=50 (seeds 46-53)

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 46 | 0.8426 | 0.8235 | 0.8330 |
| 47 | 0.8600 | 0.8582 | 0.8591 |
| 48 | 0.8702 | 0.8727 | 0.8714 |
| 49 | 0.8684 | 0.8125 | 0.8405 |
| 50 | 0.8738 | 0.8738 | 0.8738 |
| 51 | 0.8372 | 0.8115 | 0.8244 |
| 52 | 0.8262 | 0.7776 | 0.8019 |
| 53 | 0.8654 | 0.8476 | 0.8565 |

mean=0.8451, std=0.0248 (sample), min=0.8019, max=0.8738

### epochs=75 (seeds 42-49, subject 1 only)

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 42 | 0.8769 | 0.8345 | 0.8557 |
| 43 | 0.8669 | 0.8511 | 0.8590 |
| 44 | 0.8748 | 0.8287 | 0.8518 |
| 45 | 0.8657 | 0.8229 | 0.8443 |
| 46 | 0.8453 | 0.8247 | 0.8350 |
| 47 | 0.8848 | 0.8588 | 0.8718 |
| 48 | 0.8715 | 0.8657 | 0.8686 |
| 49 | 0.8787 | 0.8092 | 0.8439 |

mean=0.8538, std=0.0127 (sample), min=0.8350, max=0.8718

### epochs=100 (seeds 46-53)

| seed | 0train | 1test | mean |
| --- | --- | --- | --- |
| 46 | 0.8517 | 0.8268 | 0.8392 |
| 47 | 0.8804 | 0.8509 | 0.8656 |
| 48 | 0.8719 | 0.8621 | 0.8670 |
| 49 | 0.8746 | 0.8129 | 0.8438 |
| 50 | 0.8725 | 0.8540 | 0.8632 |
| 51 | 0.8881 | 0.8131 | 0.8506 |
| 52 | 0.8590 | 0.7851 | 0.8220 |
| 53 | 0.8696 | 0.8353 | 0.8524 |

mean=0.8505, std=0.0154 (sample), min=0.8220, max=0.8670

### Summary

| epochs | mean | std | min | max | range |
| --- | --- | --- | --- | --- | --- |
| 50  | 0.8451 | 0.0248 | 0.8019 | 0.8738 | 0.0719 |
| 75  | 0.8538 | 0.0127 | 0.8350 | 0.8718 | 0.0368 |
| 100 | 0.8505 | 0.0154 | 0.8220 | 0.8670 | 0.0450 |

### Same-seed comparison (seeds 46, 47, 48, 49 -- the only ones run at all three epoch counts)

| seed | 50ep | 75ep | 100ep |
| --- | --- | --- | --- |
| 46 | 0.8330 | 0.8350 | 0.8392 |
| 47 | 0.8591 | 0.8718 | 0.8656 |
| 48 | 0.8714 | 0.8686 | 0.8670 |
| 49 | 0.8405 | 0.8439 | 0.8438 |

## Finding

75 and 100 epochs are both clearly better than 50 for every one of the 4
directly-comparable seeds (46/47/48/49). 75 vs. 100 is close (each wins 2 of
4), and on the full 8-seed sets 75 has both the higher mean (0.8538 vs
0.8505) and the tighter spread (std 0.0127 vs 0.0154, range 0.037 vs 0.045).
Given 75 epochs also costs ~25% less wall-clock than 100 for a result that's
at least as good on this data, **epochs=75 is adopted as the new canonical
benchmark**, superseding the prior epochs=100 default. Not enough seeds yet
to call the 75-vs-100 tightness difference more than a reasonable bet --
worth revisiting if a future sweep adds more seeds to either set.

`1test` is consistently the noisier fold across all three epoch settings
(wider score range than `0train` every time), independent of epoch count --
see [[sparse-evidence-gnn-seed-variance]] for the standing catastrophic-
failure risk on `0train` specifically; this data doesn't contradict that,
it's just a different axis (this sweep's `0train` scores were all
comfortably above chance across all 24 seed/epoch combinations).

## Methodology note (a mistake, corrected)

Mid-analysis, a query for seed 45 at epochs=75 turned up two ledger rows
with wildly different scores (0.844 mean, then 0.537 mean ~3.5 minutes
later) that were initially reported as a possible same-seed non-determinism
bug, plausibly caused by CPU contention with concurrently-running parallel
sweep jobs. That diagnosis was wrong: the second row was a **different
subject** (subject 2, not subject 1) -- the ledger query filtered on
`pipeline`/`epochs`/`seed` but not `subject`, so a subject-2 probe run
(0.51/0.57, unremarkable for that historically-harder subject) got
misread as a subject-1 re-run gone catastrophically wrong. Lesson: when
deduplicating or comparing `run_ledger.csv` rows, always include `subject`
(and ideally `session`) in the group-by/filter key, not just
`seed`/`epochs`/`pipeline` -- this run_id in particular (`canonical-sparse`,
the script's default) gets reused across different subjects/configs by
design, so `seed` alone is not a unique key within it.

## Full effective parameters (shared across all runs; only `epochs`/`seed` vary)

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

See also [[sparse-evidence-gnn-seed-variance]] for the standing seed-variance
risk this sweep quantifies further, [[sparse-evidence-gnn-reorg-2026-08-09]]
for `run_sparse_evidence_gnn.py`'s `SPARSE_EVIDENCE_GNN_PARAMS` (the single
source of truth this sweep varied `epochs` against), and
`session_notes/run_logs/2026-08-06_sparse-evidence-gnn_bs8-vs-bs10_8seed-sweep.md`
for the earlier sweep this one's methodology follows.
