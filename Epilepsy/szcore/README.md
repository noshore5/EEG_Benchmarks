# SzCORE seizure-detection submission

Wraps this repo's TMC-T model (`godoy_tmc` architecture) as a
[SzCORE](https://epilepsybenchmarks.com) / EpilepsyBench container:
one frozen checkpoint, EDF in -> seizure-annotation TSV out, scored on
SzCORE's private held-out dataset (event F1, sensitivity, FP/24h).

**Task note:** SzCORE scores seizure *detection* (ictal onset/offset),
not the preictal *prediction* task the repo's headline rows target. This
checkpoint is trained on ictal-vs-nonictal windows and its numbers are
**not** comparable to `results/*/prediction/`.

## Layout

| path | what |
|---|---|
| `algo/model.py` | vendored copy of `TMCTransformer` (no `Epilepsy` import) |
| `algo/common.py` | EDF load (double-banana bipolar), windowing, event post-proc |
| `algo/infer.py` | `python -m algo $INPUT $OUTPUT` -- the container entrypoint |
| `algo/model_weights.pt` | trained checkpoint (produced by `train_detector.py`, git-ignored) |
| `train_detector.py` | build CHB-MIT windows -> fit `GodoyTMCClassifier` -> write checkpoint |
| `test_local.py` | run `algo/infer.py` on a local EDF, no Docker |
| `Dockerfile` | SzCORE template, CPU torch |
| `coheriq_tmct.yaml` | submission descriptor (goes in a fork of `esl-epfl/szcore`) |

## 1. Train the checkpoint

```bash
.venv/bin/python Epilepsy/szcore/train_detector.py --subjects 1 --epochs 20 --device mps
```

`--subjects 1 2 3 5 ...` pools more CHB-MIT patients (each extra subject is
a ~1 GB PhysioNet download the first time; only chb01 is pre-cached).
`--max-records-per-subject N` caps seizure-free recordings for a memory-
safe smoke. The checkpoint stores the auto-picked probability threshold
(train sample-F1 optimum); override with `--threshold`.

More patients = better generalisation to SzCORE's private set. A
single-subject checkpoint is fine to validate the CI but will score
poorly cross-patient -- train the real one on AWS with many subjects.

## 2. Test inference locally

```bash
.venv/bin/python Epilepsy/szcore/test_local.py \
    ~/mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/chb01_03.edf
```

## 3. Build + push the image

```bash
docker build -t ghcr.io/noshore5/eeg_benchmarks-szcore:0.1.0 Epilepsy/szcore
# smoke it:
mkdir -p in out && cp <some>.edf in/sample.edf
docker run --rm -v "$PWD/in:/data" -v "$PWD/out:/output" \
    -e INPUT=sample.edf -e OUTPUT=sample.tsv \
    ghcr.io/noshore5/eeg_benchmarks-szcore:0.1.0
docker push ghcr.io/noshore5/eeg_benchmarks-szcore:0.1.0   # must be PUBLIC
```

## 4. Submit

1. Fork `github.com/esl-epfl/szcore`.
2. Copy `coheriq_tmct.yaml` to `algorithms/coheriq_tmct.yaml` in the fork
   (update `image:` / `version:` if changed).
3. Open a PR. CI validates the YAML + runs the image on sample data.
4. On merge, EpilepsyBench runs it on every dataset and updates the
   leaderboard.

The private test set is never released, so training on all public data
(CHB-MIT, TUH, Siena, ...) is expected and leakage-free.

## Contract (from the challenge spec)

- input EDF: 256 Hz, 19-ch unipolar common-average (`Fp1-Avg ... T6-Avg`);
  `algo/common.py` re-references to the 18-ch double-banana bipolar montage.
- output TSV columns: `onset duration eventType confidence channels
  dateTime recordingDuration`; seizure rows have `eventType=sz`. Written by
  `epilepsy2bids.Annotations.saveTsv` (empty mask -> header only).
- offline: all deps install at build time; no network at inference.
