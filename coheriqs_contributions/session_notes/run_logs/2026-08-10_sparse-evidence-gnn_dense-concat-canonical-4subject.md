# Sparse-Evidence-GNN dense/concat canonical run, subjects 1-4 — 2026-08-10

## Why

Recording the canonical `run_sparse_evidence_gnn.py` config's most recent
real 4-subject run for the ledger, and as the "dense GNN" side of a same-day
comparison against an [EEGNet canonical run](2026-08-10_eegnet_canonical-4subject.md)
on the identical `BNCI2014_001`/`LeftRightImagery` cross-session benchmark.

## Method

`python coheriqs_contributions/run_sparse_evidence_gnn.py`, on-disk config at
run time (`run_id="freq-aware-hops-subj1-test-h2"`, `SUBJECTS=[1,2,3,4]`):

- `event_mode="dense"`, `event_aggregation="concat"`, `n_hops=1`,
  `freq_aware_hops=False` (the `USE_CONCAT_DENSE=True` override — see
  [[sparse-evidence-gnn-concat-productionized-cross-subject]]), with
  `dense_conv_kernel_size=5`, `dense_conv_pool_size=4`,
  `dense_conv_intermediate_channels=32`, `dense_conv_out_channels=8`,
  `dense_edge_time_downsample=8`.
- `feature_ablation="zero_channel_embed"` (`ZERO_CHANNEL_EMBED=True`).
- `epochs=60`, `batch_size=8`, `learning_rate=1e-3`, `weight_decay=1e-4`,
  `grad_clip_norm=0.1`, `seed=45`, `surrogate_seed=42`.
- `validation_split=0.0`, `early_stopping_patience=None` — deliberately off;
  see [[sparse-evidence-gnn-validation-split-hurts-cross-session]] (same-day
  A/B test on this exact config found `validation_split=0.2` costs ~0.048
  mean accuracy across all 8 folds vs. `0.0`).
- `coherence_threshold_mode="fixed"`, `coherence_threshold=0.99`,
  `phase_threshold_deg=10.0`.
- `channel_subset=[1, 5, 7, 8, 9, 10, 11, 13, 17]` (9-channel motor subset),
  `nfreqs=16`, `lowest=8.0`, `highest=30.0`, `sampling_rate=250`,
  `channel_encoder_dilation=5`, `hidden_dim=8`, `channel_embed_dim=8`,
  `smooth_kernel_size=(5, 3)`, `coi_enabled=True`.

Evaluated via `moabb.evaluations.CrossSessionEvaluation`
(`BNCI2014_001`, `LeftRightImagery(fmin=8, fmax=35)`, `random_state=42`) —
2 folds per subject (`0train`: train on session 0, test on session 1;
`1test`: reverse).

## Results

| subject | 0train | 1test | mean |
| --- | --- | --- | --- |
| 1 | 0.9029 | 0.9477 | 0.9253 |
| 2 | 0.6004 | 0.5482 | 0.5743 |
| 3 | 0.9406 | 0.9188 | 0.9297 |
| 4 | 0.7665 | 0.6927 | 0.7296 |
| **cohort mean** | | | **0.7897** |

Every fold's training log hit `acc=1.0000`/`roc_auc=1.0000`/`loss<0.00002` by
epoch 60 (all 8 folds, checked directly in the run's experiment log) — full
training-set memorization on this small `batch_size=8` config, same pattern
across every subject/fold, not isolated to the hard subjects. With
`validation_split=0.0` this doesn't affect which weights get kept (no
best-val-loss reversion happens), it's just the expected shape of the loss
curve for this architecture/data-size combination — flagged here as context,
not as a problem with this specific run.

Subject 2 is the clear outlier (0.5743, barely above chance) — consistent
with its status as a historically hard subject for this pipeline (see the
subject 1-4 canonical numbers cited in `sparse_evidence_gnn_classifier.py`'s
module docstring: subj2=0.557 on an earlier, differently-configured run).

## Comparison to EEGNet (same day, same benchmark)

| subject | Sparse-Evidence-GNN (dense/concat) | EEGNet | Δ (EEGNet − GNN) |
| --- | --- | --- | --- |
| 1 | 0.9253 | 0.9654 | +0.040 |
| 2 | 0.5743 | 0.5772 | +0.003 |
| 3 | 0.9297 | 0.9942 | +0.065 |
| 4 | 0.7296 | 0.7887 | +0.059 |
| **cohort mean** | **0.7897** | **0.8314** | **+0.042** |

EEGNet — a standard, off-the-shelf compact CNN with no custom wavelet/
coherence/graph machinery — outperforms this canonical dense/concat config on
3 of 4 subjects and ties on the hard one (subject 2), for a +0.042 cohort
mean. This extends the standing "non-graph/simpler beats graph" pattern from
[[sparse-evidence-gnn-dense-flat-control-beats-graph]] and
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]] one level further
out: it's not just that flat/non-graph readouts on this pipeline's own
extracted features beat the graph pathway — a completely independent,
much simpler architecture with no bespoke feature engineering at all beats
the whole pipeline, on this same 4-subject cross-session split.

## Caveats

1. Single-seed (45) — no seed-sweep variance estimate for this specific
   config; see [[sparse-evidence-gnn-seed-variance]] for how large that
   variance has run in past sweeps on this pipeline.
2. The EEGNet comparison uses each pipeline's own current canonical
   hyperparameters, not a hyperparameter-matched comparison (different
   epoch counts, batch sizes, optimizers-in-effect, etc.) — it answers "how
   does the current best-effort config of each compare," not "is the
   architecture family inherently worse holding everything else fixed."
3. `run_id="freq-aware-hops-subj1-test-h2"` is heavily reused across many
   same-day experiments in `~/mne_data/run_ledger.csv` (config drifted
   throughout the day) — this write-up documents specifically the last
   batch logged under that run-id (`2026-08-10T17:08:12Z`,
   `experiment_20260810-210217-239287.log`), not the run-id as a whole.

## Concurrent-edit check

`run_sparse_evidence_gnn.py` is open and actively edited in the IDE
throughout this session (`SUBJECTS` and `validation_split` were both changed
live during the same-day `validation_split` A/B test). This write-up
documents the config as captured directly from that run's own experiment
log / `run_ledger.csv` row (not re-read from the file after the fact), so it
reflects exactly what actually ran regardless of later edits.

See [[sparse-evidence-gnn-concat-productionized-cross-subject]],
[[sparse-evidence-gnn-validation-split-hurts-cross-session]], and the
[EEGNet canonical run](2026-08-10_eegnet_canonical-4subject.md) this
compares against.
