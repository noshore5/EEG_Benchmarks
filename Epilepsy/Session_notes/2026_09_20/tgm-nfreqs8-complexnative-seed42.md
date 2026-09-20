# tgm-nfreqs8-complexnative-seed42 — nfreqs ablation: 8 beats 16

Same command/infra as `chunk2-seed42` (complex-native 2ch `[re,im]` edge
representation, `--precompute-chunk-size 2`, `--dense-edge-gpu-cache`),
but `--nfreqs 8` instead of 16 — isolating the frequency-resolution axis
from the significance-channel-drop axis (see `CONTEXT.md`'s "why is
performance so much worse now" discussion).

Instance: g5.xlarge, `i-09d14560c6e016f31`.

## Results (mean across 6 folds)

| metric | nfreqs=16 (`ssmverify`) | nfreqs=8 (this run) |
|---|---|---|
| accuracy | 0.8827 | 0.9096 |
| precision | 0.3873 | 0.4674 |
| recall | 0.7778 | 0.7800 |
| f1 | 0.4088 | 0.5028 |
| average_precision (AP) | 0.4836 | **0.5831** |
| roc_auc | 0.9431 | **0.9610** |
| epoch_time | 3.46s | 2.81s |
| event-level hit rate (raw) | 6/6 | 6/6 |
| event-level hit rate (smoothed) | 5/6 | **6/6** |

**nfreqs=8 clearly beats nfreqs=16 on every metric** — the opposite of
what "more frequency resolution should help" would predict. Both configs
hit 100% dense-edge cache reuse during training, so this isn't a cache-
starvation artifact of the smaller nfreqs; it's a real accuracy
difference between resolutions.

## Working hypothesis: nfreqs=16 overfits

Same fold (`1_26`), final epoch, both runs:

| | nfreqs=16 | nfreqs=8 |
|---|---|---|
| train loss | 0.1195 | 0.1539 |
| val loss | 0.2349 (~2x train) | 0.0768 (< train) |
| train roc_auc | 0.9905 | 0.9844 |
| val roc_auc | 0.9838 (< train) | 0.9857 (> train) |

nfreqs=16's val loss nearly doubling train loss with val roc_auc trailing
train is a classic overfitting signature; nfreqs=8's val metrics matching
or beating train indicate a healthy fit. The dense-edge tensor is
`[channels, E, T, F]` with `F=nfreqs`, feeding directly into
`temporal_edge_proj`/the complex projection -- doubling `F` doubles that
layer's input width and parameter count against the same small LOSO
training set (tens of positive preictal windows per fold), with no new
real information (same 8-40Hz band, just sliced finer, and finer bins are
individually noisier). Matches this repo's general pattern
(`NEGATIVES.md`): added capacity on this dataset size tends to wash or
hurt rather than help.

**Caveat:** single-fold comparison, not the full 6-fold aggregate train/val
curves -- worth confirming the same gap pattern holds across all 6 folds
before treating this as settled, though the aggregate AP already points
the same direction.

## Auto-push: also failed (4th occurrence)

Same `could not read deploy key from SSM` failure as `ssmverify` above --
see that run's note and `FAILURE_LOG.md` #13. Results recovered manually
via `eeg-tail.yml`'s new results-dump section; CSVs committed alongside
this note.
