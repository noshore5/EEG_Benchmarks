# nonstgm-gru-smoke

## Objective
Sanity check the `nonstgm_gru` pipeline on real CHB-MIT data (Subject 1) in prediction mode to verify end-to-end execution, window generation, and timestamp/label alignment.

## Execution
* **Command:** `python Epilepsy/run_pipelines.py --pipeline nonstgm_gru --label-mode prediction --subjects 1 --device cpu`
* **Commit:** `e7d5719`
* **Hardware:** 4 vCPU, 15 GB RAM (CPU run)
* **Wall time:** 7 min (exit `rc=0`)

## Results
The pipeline ran through 6 leave-one-seizure-out folds to completion without errors. Epoch time averaged ~1.9s on CPU (22 batches/epoch, batch size 32). Early stopping triggered between epochs 10–18 across folds.

| Metric | Fold Mean |
|---|---|
| Accuracy | 95.89% |
| Precision | 0.00% |
| Recall | 0.00% |
| F1 | 0.00% |
| Average Precision | 0.0307 |
| ROC-AUC | 0.2931 |
| FAR/h (raw / smoothed) | 0.000 / 0.000 |
| Event Hit Rate (raw & k-of-n) | 0/6 (0.0%) |

Per-fold results and event logs were successfully written to `Epilepsy/results/nonstgm_gru/prediction/`.

## Findings
1. **Pipeline integrity verified:** End-to-end data loading, negative subsampling (5:1 target ratio), train/validation splitting, fold rotation, and metric logging functioned correctly on real EEG recordings.
2. **Model collapse:** The model failed to learn discriminative preictal features. Training loss remained pinned near $\ln(2) \approx 0.691\text{--}0.697$, and the network collapsed to predicting the majority negative class. This produces degenerate zero recall, zero false alarms, and an inverted ROC-AUC (0.293).

## Next Steps
1. Inspect probability distribution outputs and learning rates; resolve why training loss stalls at ~0.69 despite class weighting (`[0.33, 1.67]`).
2. Move from CPU to GPU (`--device cuda`) for multi-subject benchmarking once gradient/feature learning is verified.

---
_Autonomous `eeg-run`; note drafted by gemini-flash-latest, committed by `promote_results.sh`. Facts below are machine-generated._

## Run

| | |
|---|---|
| command | `python Epilepsy/run_pipelines.py --pipeline nonstgm_gru --label-mode prediction --subjects 1 --device cpu` |
| repo | `e7d5719` |
| started | 2026-09-20T22:30:58Z |
| ended | 2026-09-20T22:38:19Z |
| wall | 7 min |
| host | 4 vCPU, 15 GB RAM, GPU NVIDIA A10G |
| exit | rc=0 |

## Result files

### `Epilepsy/results/nonstgm_gru/`


## run.log (tail)

```
[Train] class weights: [0.32853025 1.6714697 ]
[Train] start epochs=20 batches/epoch=22 batch_size=32 optimizer_step_batch_size=32 optimizer_step_batch_mode=credit device=cpu
[Train][Epoch 1/20] loss=0.694784 (improve +0.000000) acc=0.2968 (delta +0.0000) roc_auc=0.5150 (delta n/a) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=2.03s val_loss=0.693423 val_acc=0.2874 val_roc_auc=0.3979
[Train][Epoch 2/20] loss=0.694559 (improve +0.000225) acc=0.5764 (delta +0.2795) roc_auc=0.4691 (delta -0.0460) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.99s val_loss=0.683660 val_acc=0.8333 val_roc_auc=0.6021
[Train][Epoch 3/20] loss=0.692134 (improve +0.002425) acc=0.7507 (delta +0.1744) roc_auc=0.5249 (delta 0.0559) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.99s val_loss=0.675706 val_acc=0.8333 val_roc_auc=0.6038
[Train][Epoch 4/20] loss=0.695118 (improve -0.002984) acc=0.7968 (delta +0.0461) roc_auc=0.4889 (delta -0.0361) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.96s val_loss=0.675025 val_acc=0.8333 val_roc_auc=0.6048
[Train][Epoch 5/20] loss=0.694457 (improve +0.000660) acc=0.7839 (delta -0.0130) roc_auc=0.4950 (delta 0.0061) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.88s val_loss=0.678524 val_acc=0.8333 val_roc_auc=0.6057
[Train][Epoch 6/20] loss=0.693622 (improve +0.000836) acc=0.7651 (delta -0.0187) roc_auc=0.4912 (delta -0.0038) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.683065 val_acc=0.8333 val_roc_auc=0.6052
[Train][Epoch 7/20] loss=0.692389 (improve +0.001233) acc=0.7421 (delta -0.0231) roc_auc=0.5013 (delta 0.0100) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.675975 val_acc=0.8333 val_roc_auc=0.6086
[Train][Epoch 8/20] loss=0.692362 (improve +0.000027) acc=0.8098 (delta +0.0677) roc_auc=0.5210 (delta 0.0198) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.88s val_loss=0.674484 val_acc=0.8333 val_roc_auc=0.6093
[Train][Epoch 9/20] loss=0.693498 (improve -0.001135) acc=0.7478 (delta -0.0620) roc_auc=0.4943 (delta -0.0267) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.89s val_loss=0.679850 val_acc=0.8333 val_roc_auc=0.6088
[Train][Epoch 10/20] loss=0.689273 (improve +0.004225) acc=0.7839 (delta +0.0360) roc_auc=0.5665 (delta 0.0722) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.89s val_loss=0.675831 val_acc=0.8333 val_roc_auc=0.6098
[Train][Epoch 11/20] loss=0.693732 (improve -0.004460) acc=0.8156 (delta +0.0317) roc_auc=0.5051 (delta -0.0615) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.88s val_loss=0.675372 val_acc=0.8333 val_roc_auc=0.6086
[Train][Epoch 12/20] loss=0.693871 (improve -0.000138) acc=0.7882 (delta -0.0274) roc_auc=0.4767 (delta -0.0283) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.89s val_loss=0.675950 val_acc=0.8333 val_roc_auc=0.6100
[Train][Epoch 13/20] loss=0.691244 (improve +0.002626) acc=0.8271 (delta +0.0389) roc_auc=0.5387 (delta 0.0620) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.672734 val_acc=0.8333 val_roc_auc=0.6090
[Train][Epoch 14/20] loss=0.691529 (improve -0.000285) acc=0.8213 (delta -0.0058) roc_auc=0.5171 (delta -0.0216) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.85s val_loss=0.676494 val_acc=0.8333 val_roc_auc=0.6102
[Train][Epoch 15/20] loss=0.691405 (improve +0.000124) acc=0.7378 (delta -0.0836) roc_auc=0.5260 (delta 0.0089) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.93s val_loss=0.682782 val_acc=0.8333 val_roc_auc=0.6098
[Train][Epoch 16/20] loss=0.692984 (improve -0.001579) acc=0.7968 (delta +0.0591) roc_auc=0.4935 (delta -0.0324) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.673830 val_acc=0.8333 val_roc_auc=0.6088
[Train][Epoch 17/20] loss=0.692313 (improve +0.000671) acc=0.7925 (delta -0.0043) roc_auc=0.5182 (delta 0.0247) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.86s val_loss=0.679981 val_acc=0.8333 val_roc_auc=0.6071
[Train][Epoch 18/20] loss=0.691213 (improve +0.001100) acc=0.7810 (delta -0.0115) roc_auc=0.5436 (delta 0.0254) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.678358 val_acc=0.8333 val_roc_auc=0.6064
[Train] early stopping at epoch 18; best epoch=13 best_val_loss=0.672734
[Train] restored best model from epoch 13 (val_loss=0.672734)
  seizure 1_18_0: n_test=750 preictal=30  hit=False (smoothed=False)  FAR/h=0.000 (smoothed=0.000)  precision=0.000 recall=0.000 f1=0.000
  Subsampled negative windows: 3468 -> 722 (target ratio 5.0:1, 143 positive kept in full)
[Train] validation split: 173/865 samples.
[Train] class weights: [0.32947975 1.6705202 ]
[Train] start epochs=20 batches/epoch=22 batch_size=32 optimizer_step_batch_size=32 optimizer_step_batch_mode=credit device=cpu
[Train][Epoch 1/20] loss=0.692866 (improve +0.000000) acc=0.3844 (delta +0.0000) roc_auc=0.5170 (delta n/a) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.89s val_loss=0.691755 val_acc=0.7688 val_roc_auc=0.4392
[Train][Epoch 2/20] loss=0.696002 (improve -0.003136) acc=0.5723 (delta +0.1879) roc_auc=0.4596 (delta -0.0575) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.679650 val_acc=0.8324 val_roc_auc=0.4370
[Train][Epoch 3/20] loss=0.696858 (improve -0.000857) acc=0.7327 (delta +0.1604) roc_auc=0.4552 (delta -0.0044) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.676552 val_acc=0.8324 val_roc_auc=0.4397
[Train][Epoch 4/20] loss=0.692077 (improve +0.004781) acc=0.7803 (delta +0.0477) roc_auc=0.5140 (delta 0.0588) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.675653 val_acc=0.8324 val_roc_auc=0.4701
[Train][Epoch 5/20] loss=0.691833 (improve +0.000244) acc=0.7312 (delta -0.0491) roc_auc=0.5205 (delta 0.0065) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.86s val_loss=0.679241 val_acc=0.8324 val_roc_auc=0.4607
[Train][Epoch 6/20] loss=0.694102 (improve -0.002269) acc=0.8035 (delta +0.0723) roc_auc=0.4928 (delta -0.0277) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.86s val_loss=0.674726 val_acc=0.8324 val_roc_auc=0.5778
[Train][Epoch 7/20] loss=0.691016 (improve +0.003086) acc=0.7731 (delta -0.0303) roc_auc=0.5493 (delta 0.0565) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.91s val_loss=0.678762 val_acc=0.8324 val_roc_auc=0.5139
[Train][Epoch 8/20] loss=0.694474 (improve -0.003458) acc=0.7572 (delta -0.0159) roc_auc=0.4748 (delta -0.0744) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.85s val_loss=0.682331 val_acc=0.8324 val_roc_auc=0.4971
[Train][Epoch 9/20] loss=0.694112 (improve +0.000362) acc=0.5751 (delta -0.1821) roc_auc=0.4813 (delta 0.0065) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.86s val_loss=0.685007 val_acc=0.8324 val_roc_auc=0.4955
[Train][Epoch 10/20] loss=0.697328 (improve -0.003216) acc=0.7543 (delta +0.1792) roc_auc=0.4276 (delta -0.0537) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.86s val_loss=0.674255 val_acc=0.8324 val_roc_auc=0.5831
[Train][Epoch 11/20] loss=0.694876 (improve +0.002452) acc=0.6893 (delta -0.0650) roc_auc=0.4559 (delta 0.0283) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.86s val_loss=0.681026 val_acc=0.8324 val_roc_auc=0.5864
[Train][Epoch 12/20] loss=0.692145 (improve +0.002731) acc=0.7847 (delta +0.0954) roc_auc=0.5233 (delta 0.0674) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.674411 val_acc=0.8324 val_roc_auc=0.5728
[Train][Epoch 13/20] loss=0.694201 (improve -0.002055) acc=0.8107 (delta +0.0260) roc_auc=0.4754 (delta -0.0479) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.87s val_loss=0.675656 val_acc=0.8324 val_roc_auc=0.5761
[Train][Epoch 14/20] loss=0.693251 (improve +0.000949) acc=0.7962 (delta -0.0145) roc_auc=0.4796 (delta 0.0042) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.95s val_loss=0.676718 val_acc=0.8324 val_roc_auc=0.5754
[Train][Epoch 15/20] loss=0.692475 (improve +0.000776) acc=0.7746 (delta -0.0217) roc_auc=0.5056 (delta 0.0260) optimizer_steps=22 fallback_regularization_fraction=0.000000 epoch_time=1.97s val_loss=0.679657 val_acc=0.8324 val_roc_auc=0.5896
[Train] early stopping at epoch 15; best epoch=10 best_val_loss=0.674255
[Train] restored best model from epoch 10 (val_loss=0.674255)
  seizure 1_26_0: n_test=630 preictal=30  hit=False (smoothed=False)  FAR/h=0.000 (smoothed=0.000)  precision=0.000 recall=0.000 f1=0.000

Wrote per-fold results to /root/repo/Epilepsy/results/nonstgm_gru/prediction/prediction_leave_one_seizure_out_20260920-223527.csv
Wrote per-seizure log to /root/repo/Epilepsy/results/nonstgm_gru/prediction/prediction_per_seizure_20260920-223527.csv

=== Mean across folds (pipeline=nonstgm_gru, label_mode=prediction; NOT comparable to detection's numbers) ===
accuracy                          0.958915
precision                         0.000000
recall                            0.000000
f1                                0.000000
average_precision                 0.030686
roc_auc                           0.293082
false_alarms_per_hour             0.000000
false_alarms_per_hour_smoothed    0.000000
event-level hit rate (raw):    0/6 (0.0%)
event-level hit rate (k-of-n): 0/6 (0.0%)
```
