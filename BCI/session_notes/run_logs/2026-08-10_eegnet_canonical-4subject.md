# EEGNet canonical run, subjects 1-4 — 2026-08-10

## Why

Baseline comparison point for the same-day
[Sparse-Evidence-GNN dense/concat canonical run](2026-08-10_sparse-evidence-gnn_dense-concat-canonical-4subject.md)
— a standard, non-custom architecture on the identical `BNCI2014_001`/
`LeftRightImagery` cross-session benchmark.

## Method

`run_id="canonical-eegnet"`, `EEGNetClassifier` via
`run_wct_gnn.py`'s `_make_eegnet()` builder — either
`run_canonical_setup.py` with `CANONICAL_VARIANT="eegnet"` or
`run_wct_gnn.py --pipeline EEGNet` directly (see `_make_eegnet()`,
`BCI/run_wct_gnn.py`).

Effective config (from the run's own `run_ledger.csv` row):
`epochs=100`, `batch_size=32`, `learning_rate=0.001`, `dropout_rate=0.5`,
`weight_decay=0.0`, `grad_clip_norm=None`, `device="cpu"`, `seed=42`,
`validation_split=0.0`, `early_stopping_patience=None`. `channel_subset=None`
(all 22 `BNCI2014_001` EEG channels — no motor-channel subsetting, unlike
Sparse-Evidence-GNN's 9-channel comparison run).

Same evaluation as the dense-GNN comparison run:
`moabb.evaluations.CrossSessionEvaluation` (`BNCI2014_001`,
`LeftRightImagery(fmin=8, fmax=35)`, `random_state=42`), subjects 1-4,
2 folds/subject (`0train`/`1test`).

## Results

| subject | 0train | 1test | mean |
| --- | --- | --- | --- |
| 1 | 0.9574 | 0.9734 | 0.9654 |
| 2 | 0.5687 | 0.5856 | 0.5772 |
| 3 | 0.9965 | 0.9919 | 0.9942 |
| 4 | 0.8268 | 0.7506 | 0.7887 |
| **cohort mean** | | | **0.8314** |

Training curve (checked directly in the run's experiment log) does **not**
show the same full-memorization pattern as the dense-GNN comparison run —
final-epoch train metrics still show real epoch-to-epoch noise (e.g. one
fold's epoch 100: `acc=0.8750`, swinging ±0.03-0.08 over the preceding ~10
epochs rather than sitting flat at 1.0). Consistent with EEGNet's much
smaller parameter count and larger `batch_size` (32 vs. 8) giving it less
capacity/opportunity to fully memorize 100+ trials of raw multi-channel EEG
in 100 epochs, unlike the GNN's higher-capacity graph pathway on far fewer
effective per-batch samples.

## Comparison to Sparse-Evidence-GNN (same day, same benchmark)

See the [dense-GNN run's own write-up](2026-08-10_sparse-evidence-gnn_dense-concat-canonical-4subject.md#comparison-to-eegnet-same-day-same-benchmark)
for the full side-by-side table. Summary: EEGNet wins on 3 of 4 subjects and
ties on the hard one (subject 2), for a +0.042 cohort-mean edge over the
canonical dense/concat Sparse-Evidence-GNN config — consistent with the
standing pattern that simpler/non-custom architectures keep beating this
repo's bespoke coherence-graph pipelines
([[sparse-evidence-gnn-dense-flat-control-beats-graph]],
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]]).

## Caveats

1. Not a hyperparameter-matched comparison — see caveat 2 on the dense-GNN
   write-up.
2. `channel_subset=None` here vs. the GNN run's 9-channel motor subset —
   EEGNet had access to 22 channels, the GNN only 9. `_make_eegnet()`'s own
   comment flags this exact asymmetry and how to close it
   (`channel_subset=[1, 5, 7, 8, 9, 10, 11, 13, 17]`) for a channel-matched
   rerun, not yet done.
3. Single-seed (42) — no variance estimate for this config.
4. `device="cpu"` is hardcoded in `_make_eegnet()`, unlike the GNN
   pipelines' `device="auto"` — not expected to change the score, only
   wall-clock time.

See [[sparse-evidence-gnn-dense-flat-control-beats-graph]] and
[[sparse-evidence-gnn-event-stats-baseline-beats-graph]] for the prior
same-direction findings this result extends.
