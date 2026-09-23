# nv-pat1-smoke3

_Autonomous `eeg-run`; templated by `promote_results.sh` (no LLM key / call failed)._

Smoke retry #2. nv-pat1-smoke2 (with readable logs after the first fix) showed the real bug: fetch_kuhlmann_nv_s3.sh extracted the tarball assuming a Pat1Train/ top-level folder that isn't actually there -- FileNotFoundError immediately. Fixed in f663844 (extract to scratch dir, find *.mat files wherever they land, move into place). This run should get past data loading.

## Run

| | |
|---|---|
| command | `python scripts/run_nv_pat1.py --patient 1 --folds 3 --device cuda --grouping segment --limit 60 --epochs 3` |
| repo | `f663844` |
| started | 2026-09-23T09:30:23Z |
| ended | 2026-09-23T09:35:50Z |
| wall | 5 min |
| host | 4 vCPU, 15 GB RAM, GPU Tesla T4 |
| exit | rc=0 |

## Result files

### `Epilepsy/results/godoy_tmc/nv/nv_pat1_stratified_kfold_segmentgrouped_20260923-093547.csv`

```
(could not render)
```

## run.log (tail)

```
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
[fetch_kuhlmann_nv_s3] done -- 1652 files in /root/repo/datasets/epilepsy/kuhlmann_nv/Pat1Train
[kuhlmann_nv] dropped 1/60 Pat1Train segments with off-canonical length (recording dropout, not a bug -- see load_patient_train's docstring)
[nv_pat1] loaded 59 segments (32 preictal / 27 interictal) in 52s, X.shape=(59, 16, 60000) (0.2GB resident)
[nv_pat1] windowed into 1180 30.0s sub-windows (640 preictal / 540 interictal), X.shape=(1180, 16, 3000)
[Train] validation split: 160/800 samples.
[Train] class weights: [1.1 0.9]
[Train] start epochs=3 batches/epoch=20 batch_size=32 optimizer_step_batch_size=32 optimizer_step_batch_mode=credit device=cuda
[Train][Epoch 1/3] loss=0.597494 (improve +0.000000) acc=0.7078 (delta +0.0000) roc_auc=0.7898 (delta n/a) optimizer_steps=20 epoch_time=14.47s val_loss=0.418118 val_acc=0.8688 val_roc_auc=0.9216
[Train][Epoch 2/3] loss=0.451917 (improve +0.145578) acc=0.8031 (delta +0.0953) roc_auc=0.8684 (delta 0.0786) optimizer_steps=20 epoch_time=0.48s val_loss=0.335900 val_acc=0.8688 val_roc_auc=0.9432
[Train][Epoch 3/3] loss=0.375579 (improve +0.076338) acc=0.8219 (delta +0.0188) roc_auc=0.9052 (delta 0.0368) optimizer_steps=20 epoch_time=0.48s val_loss=0.430381 val_acc=0.8562 val_roc_auc=0.9568
[Train] restored best model from epoch 2 (val_loss=0.335900)
[nv_pat1] fold 0: n_test_segments=11 (preictal=2)  precision=0.000 recall=0.000 f1=0.000 auc_pr=0.156 roc_auc=0.111
[Train] validation split: 156/780 samples.
[Train] class weights: [1.0769231 0.9230769]
[Train] start epochs=3 batches/epoch=20 batch_size=32 optimizer_step_batch_size=32 optimizer_step_batch_mode=credit device=cuda
[Train][Epoch 1/3] loss=0.527575 (improve +0.000000) acc=0.7660 (delta +0.0000) roc_auc=0.8597 (delta n/a) optimizer_steps=20 epoch_time=0.96s val_loss=0.321096 val_acc=0.8782 val_roc_auc=0.9567
[Train][Epoch 2/3] loss=0.334464 (improve +0.193111) acc=0.8750 (delta +0.1090) roc_auc=0.9289 (delta 0.0692) optimizer_steps=20 epoch_time=0.48s val_loss=0.262966 val_acc=0.9103 val_roc_auc=0.9706
[Train][Epoch 3/3] loss=0.324107 (improve +0.010357) acc=0.8718 (delta -0.0032) roc_auc=0.9359 (delta 0.0070) optimizer_steps=20 epoch_time=0.47s val_loss=0.255743 val_acc=0.8846 val_roc_auc=0.9759
[Train] restored best model from epoch 3 (val_loss=0.255743)
[nv_pat1] fold 1: n_test_segments=11 (preictal=2)  precision=0.000 recall=0.000 f1=0.000 auc_pr=0.141 roc_auc=0.000
[Train] validation split: 156/780 samples.
[Train] class weights: [1.0769231 0.9230769]
[Train] start epochs=3 batches/epoch=20 batch_size=32 optimizer_step_batch_size=32 optimizer_step_batch_mode=credit device=cuda
[Train][Epoch 1/3] loss=0.561853 (improve +0.000000) acc=0.7452 (delta +0.0000) roc_auc=0.8256 (delta n/a) optimizer_steps=20 epoch_time=0.47s val_loss=0.436510 val_acc=0.8462 val_roc_auc=0.8924
[Train][Epoch 2/3] loss=0.424578 (improve +0.137274) acc=0.8381 (delta +0.0929) roc_auc=0.8835 (delta 0.0578) optimizer_steps=20 epoch_time=0.47s val_loss=0.398841 val_acc=0.8333 val_roc_auc=0.8867
[Train][Epoch 3/3] loss=0.347060 (improve +0.077518) acc=0.8702 (delta +0.0321) roc_auc=0.9302 (delta 0.0468) optimizer_steps=20 epoch_time=0.48s val_loss=0.308659 val_acc=0.8462 val_roc_auc=0.9550
[Train] restored best model from epoch 3 (val_loss=0.308659)
[nv_pat1] fold 2: n_test_segments=11 (preictal=2)  precision=0.000 recall=0.000 f1=0.000 auc_pr=0.156 roc_auc=0.111

Wrote per-fold results to /root/repo/Epilepsy/results/godoy_tmc/nv/nv_pat1_stratified_kfold_segmentgrouped_20260923-093547.csv

=== Mean across folds (pipeline=godoy_tmc, dataset=nv_pat1, grouping=segment, NOT leave-one-seizure-out -- see this script's module docstring) ===
accuracy             0.363636
precision            0.000000
recall               0.000000
f1                   0.000000
average_precision    0.150673
roc_auc              0.074074
```
