# nonstgm-mamba-smoke-v3

_Autonomous `eeg-run`; templated by `promote_results.sh` (no LLM key / call failed)._

nonstgm_mamba GPU sanity re-run on a 32GB-RAM box (g5.2xlarge/g6.2xlarge) after v1 and v2 both SIGKILL'd/rc=137 entering fold 1 on 16GB-RAM boxes. v2 added a per-fold teardown fix (insufficient alone) and an RSS diagnostic that showed fold 0's own peak already at 14.24GB on the 16GB box -- pointing to a single fold's working set being intrinsically close to the ceiling, not a fold-to-fold leak. See FAILURE_LOG.md #14.

## Run

| | |
|---|---|
| command | `python Epilepsy/run_pipelines.py --pipeline nonstgm_mamba --label-mode prediction --subjects 1 --device cuda` |
| repo | `cf7a892` |
| started | 2026-09-21T08:48:45Z |
| ended | 2026-09-21T08:58:48Z |
| wall | 10 min |
| host | 8 vCPU, 30 GB RAM, GPU NVIDIA L4 |
| exit | rc=0 |

## Result files

### `Epilepsy/results/nonstgm_mamba/`


## run.log (tail)

```
[Train][Epoch 6/20] loss=0.695868 (improve +0.001431) acc=0.3975 (delta -0.1994) roc_auc=0.5057 (delta 0.0197) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.63s val_loss=0.692198 val_acc=0.8343 val_roc_auc=0.5536
[Train][Epoch 7/20] loss=0.693188 (improve +0.002681) acc=0.7175 (delta +0.3199) roc_auc=0.5011 (delta -0.0046) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.63s val_loss=0.681724 val_acc=0.8343 val_roc_auc=0.5695
[Train][Epoch 8/20] loss=0.698749 (improve -0.005561) acc=0.6676 (delta -0.0499) roc_auc=0.4550 (delta -0.0461) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.60s val_loss=0.687558 val_acc=0.8343 val_roc_auc=0.5656
[Train][Epoch 9/20] loss=0.695798 (improve +0.002951) acc=0.5263 (delta -0.1413) roc_auc=0.4633 (delta 0.0084) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.62s val_loss=0.688434 val_acc=0.8343 val_roc_auc=0.5567
[Train][Epoch 10/20] loss=0.696435 (improve -0.000636) acc=0.7784 (delta +0.2521) roc_auc=0.4781 (delta 0.0148) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.62s val_loss=0.666733 val_acc=0.8343 val_roc_auc=0.5698
[Train][Epoch 11/20] loss=0.694843 (improve +0.001592) acc=0.5416 (delta -0.2368) roc_auc=0.4882 (delta 0.0101) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.58s val_loss=0.687436 val_acc=0.8343 val_roc_auc=0.5595
[Train][Epoch 12/20] loss=0.693633 (improve +0.001210) acc=0.7715 (delta +0.2299) roc_auc=0.4876 (delta -0.0005) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.58s val_loss=0.685106 val_acc=0.8343 val_roc_auc=0.5651
[Train][Epoch 13/20] loss=0.694983 (improve -0.001349) acc=0.6925 (delta -0.0789) roc_auc=0.4935 (delta 0.0058) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.60s val_loss=0.685082 val_acc=0.8343 val_roc_auc=0.4210
[Train][Epoch 14/20] loss=0.695999 (improve -0.001016) acc=0.7147 (delta +0.0222) roc_auc=0.4637 (delta -0.0297) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.59s val_loss=0.675227 val_acc=0.8343 val_roc_auc=0.4630
[Train][Epoch 15/20] loss=0.692244 (improve +0.003754) acc=0.7285 (delta +0.0139) roc_auc=0.5156 (delta 0.0519) optimizer_steps=23 fallback_regularization_fraction=0.000000 epoch_time=5.61s val_loss=0.677461 val_acc=0.8343 val_roc_auc=0.3934
[Train] early stopping at epoch 15; best epoch=10 best_val_loss=0.666733
[Train] restored best model from epoch 10 (val_loss=0.666733)
  seizure 1_16_0: n_test=743 preictal=23  hit=False (smoothed=False)  FAR/h=0.000 (smoothed=0.000)  precision=0.000 recall=0.000 f1=0.000
[fold 3 teardown] host RSS (peak so far)=15.26GB
  Subsampled negative windows: 3348 -> 725 (target ratio 5.0:1, 143 positive kept in full)
[Train] validation split: 174/868 samples.
_DenseEdgeMambaTemporal use_cuda_kernel=True (mamba-ssm fused scan)
[Train] class weights: [0.32853025 1.6714697 ]
[Train] start epochs=20 batches/epoch=22 batch_size=32 optimizer_step_batch_size=32 optimizer_step_batch_mode=credit device=cuda
[Train][Epoch 1/20] loss=0.704035 (improve +0.000000) acc=0.4467 (delta +0.0000) roc_auc=0.4985 (delta n/a) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.36s val_loss=0.658139 val_acc=0.8333 val_roc_auc=0.6326
[Train][Epoch 2/20] loss=0.702480 (improve +0.001555) acc=0.5504 (delta +0.1037) roc_auc=0.4983 (delta -0.0003) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.40s val_loss=0.696603 val_acc=0.1667 val_roc_auc=0.6259
[Train][Epoch 3/20] loss=0.687491 (improve +0.014989) acc=0.6571 (delta +0.1066) roc_auc=0.5511 (delta 0.0529) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.15s val_loss=0.669110 val_acc=0.8333 val_roc_auc=0.6314
[Train][Epoch 4/20] loss=0.702491 (improve -0.015000) acc=0.5231 (delta -0.1340) roc_auc=0.4632 (delta -0.0879) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.35s val_loss=0.700896 val_acc=0.1667 val_roc_auc=0.6245
[Train][Epoch 5/20] loss=0.695149 (improve +0.007341) acc=0.6657 (delta +0.1427) roc_auc=0.4878 (delta 0.0245) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.39s val_loss=0.689274 val_acc=0.8276 val_roc_auc=0.6254
[Train][Epoch 6/20] loss=0.696847 (improve -0.001698) acc=0.6153 (delta -0.0504) roc_auc=0.4716 (delta -0.0162) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.41s val_loss=0.697301 val_acc=0.1667 val_roc_auc=0.6193
[Train] early stopping at epoch 6; best epoch=1 best_val_loss=0.658139
[Train] restored best model from epoch 1 (val_loss=0.658139)
  seizure 1_18_0: n_test=750 preictal=30  hit=False (smoothed=False)  FAR/h=0.000 (smoothed=0.000)  precision=0.000 recall=0.000 f1=0.000
[fold 4 teardown] host RSS (peak so far)=15.26GB
  Subsampled negative windows: 3468 -> 722 (target ratio 5.0:1, 143 positive kept in full)
[Train] validation split: 173/865 samples.
_DenseEdgeMambaTemporal use_cuda_kernel=True (mamba-ssm fused scan)
[Train] class weights: [0.32947975 1.6705202 ]
[Train] start epochs=20 batches/epoch=22 batch_size=32 optimizer_step_batch_size=32 optimizer_step_batch_mode=credit device=cuda
[Train][Epoch 1/20] loss=0.714249 (improve +0.000000) acc=0.5231 (delta +0.0000) roc_auc=0.4527 (delta n/a) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.33s val_loss=0.681733 val_acc=0.8324 val_roc_auc=0.5745
[Train][Epoch 2/20] loss=0.695085 (improve +0.019164) acc=0.6445 (delta +0.1214) roc_auc=0.5276 (delta 0.0749) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.18s val_loss=0.670338 val_acc=0.8324 val_roc_auc=0.5738
[Train][Epoch 3/20] loss=0.692980 (improve +0.002105) acc=0.6720 (delta +0.0275) roc_auc=0.5083 (delta -0.0193) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.41s val_loss=0.674878 val_acc=0.8324 val_roc_auc=0.5721
[Train][Epoch 4/20] loss=0.704350 (improve -0.011370) acc=0.5853 (delta -0.0867) roc_auc=0.4314 (delta -0.0769) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.37s val_loss=0.687717 val_acc=0.8324 val_roc_auc=0.5685
[Train][Epoch 5/20] loss=0.694425 (improve +0.009925) acc=0.4552 (delta -0.1301) roc_auc=0.5196 (delta 0.0882) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.39s val_loss=0.684830 val_acc=0.8324 val_roc_auc=0.5678
[Train][Epoch 6/20] loss=0.697303 (improve -0.002878) acc=0.7572 (delta +0.3020) roc_auc=0.5021 (delta -0.0175) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.37s val_loss=0.676471 val_acc=0.8324 val_roc_auc=0.3755
[Train][Epoch 7/20] loss=0.693670 (improve +0.003634) acc=0.5578 (delta -0.1994) roc_auc=0.5002 (delta -0.0019) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=5.41s val_loss=0.698341 val_acc=0.1676 val_roc_auc=0.4128
[Train] early stopping at epoch 7; best epoch=2 best_val_loss=0.670338
[Train] restored best model from epoch 2 (val_loss=0.670338)
  seizure 1_26_0: n_test=630 preictal=30  hit=False (smoothed=False)  FAR/h=0.000 (smoothed=0.000)  precision=0.000 recall=0.000 f1=0.000
[fold 5 teardown] host RSS (peak so far)=15.28GB

Wrote per-fold results to /root/repo/Epilepsy/results/nonstgm_mamba/prediction/prediction_leave_one_seizure_out_20260921-085300.csv
Wrote per-seizure log to /root/repo/Epilepsy/results/nonstgm_mamba/prediction/prediction_per_seizure_20260921-085300.csv

=== Mean across folds (pipeline=nonstgm_mamba, label_mode=prediction; NOT comparable to detection's numbers) ===
accuracy                          0.958683
precision                         0.000000
recall                            0.000000
f1                                0.000000
average_precision                 0.096047
roc_auc                           0.533698
false_alarms_per_hour             0.029070
false_alarms_per_hour_smoothed    0.000000
event-level hit rate (raw):    0/6 (0.0%)
event-level hit rate (k-of-n): 0/6 (0.0%)
```
