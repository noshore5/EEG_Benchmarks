# SzCORE / EpilepsyBench submission scaffold

**Date:** 2026-09-06 · Mac shell · branch `main` (uncommitted)

## What was built

`Epilepsy/szcore/` -- packages the `godoy_tmc` TMC-T architecture as a
SzCORE seizure-**detection** container.

- `algo/model.py` -- verbatim vendored `_ConvTokenizer` + `TMCTransformer`
  (keeps the container free of the `Epilepsy` import chain).
- `algo/common.py` -- `load_bipolar_eeg` (epilepsy2bids: unipolar
  common-average -> double-banana 18-ch bipolar; auto-detect falls back to
  explicit `Montage.BIPOLAR` for CHB-MIT's uppercase `FP1-F7` labels),
  4 s / 1 s windowing, `probs_to_mask` (threshold -> 30 s gap merge ->
  10 s min-duration).
- `algo/infer.py` -- `python -m algo $INPUT $OUTPUT`; softmax[:,1] per
  window -> mask -> `Annotations.loadMask(mask, fs).saveTsv(out)`.
- `train_detector.py` -- CHB-MIT via repo `CHBMIT`, ictal window =
  >=50 % overlap with a summary-file seizure, negative subsample (default
  12:1), fit `GodoyTMCClassifier`, auto-pick threshold at train sample-F1
  optimum, dump `algo/model_weights.pt`.
- `Dockerfile` (SzCORE template, CPU torch), `coheriq_tmct.yaml`,
  `test_local.py`, `README.md`.

`epilepsy2bids` + `timescoring` installed into `.venv` (needs py>=3.10;
the CommandLineTools `python3` is 3.9 and cannot).

## Verified

- 3-epoch and full (early-stopped epoch 7, best epoch 2) chb01 training
  runs complete on MPS; ~1.9 s/epoch, 2938 windows after subsample.
- Inference: chb01_03 -> hit at 2998 s (truth 2996-3036) + 1 FP;
  chb01_16 -> hit at 1000 s (truth 1015) + FPs; chb01_01 (seizure-free)
  -> `bckg` header row only. `python -m algo` entrypoint + `$INPUT/$OUTPUT`
  env vars work.

## State / next

- Checkpoint is **chb01-only** and in-sample-FP-heavy -- CI-validation
  quality, not a real submission. Threshold auto-picked at 0.92.
- NEXT: `train_detector.py --subjects 1 2 3 5 ...` on AWS (each new
  subject is a ~1 GB PhysioNet pull; only chb01 cached). Then
  `docker build Epilepsy/szcore`, push public to
  `ghcr.io/noshore5/eeg_benchmarks-szcore:0.1.0`, PR the yaml to a fork of
  `esl-epfl/szcore`.
- Post-proc constants (`MERGE_GAP_S`, `MIN_EVENT_S`, threshold) untuned.
