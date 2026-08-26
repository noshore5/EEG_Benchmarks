# Dense-Edge Mamba: mambapy vs fused mamba-ssm comparison

## Purpose

Follow-up to the 2026-08-25 dense-edge pipeline comparison. The documented Mamba baseline showed an unusually poor result on seizure `1_18_0`, where recall collapsed to 0.100 and F1 to 0.171. A follow-up run was performed using the same dense-edge Mamba pipeline but with the `mamba-ssm` fused CUDA implementation and bf16 AMP.

The purpose of this run was to determine whether the `1_18_0` failure was characteristic of the dense-edge Mamba architecture or specific to the original Mamba implementation/configuration.

## Experimental configurations

### Documented Mamba baseline

Run: `20260825-113049`

* Pipeline: `dense_edge_mamba`
* Temporal model: Mamba via **mambapy**
* No fused Mamba CUDA kernel
* Prediction task
* Subject: CHB-MIT subject 1
* LOSO folds:

  * `1_03_0`
  * `1_04_0`
  * `1_15_0`
  * `1_16_0`
  * `1_18_0`
  * `1_26_0`
* Channel subset: `k=23`
* SPH: 300 s
* SOP: 900 s
* 20 training epochs
* Validation split: 0.2

### Fused-kernel follow-up

Command:

```bash
python Epilepsy/run_pipelines.py \
  --pipeline dense_edge_mamba \
  --device cuda \
  --disable-disk-cache \
  --channel-subset-k 23 \
  --epochs 20 \
  --dense-edge-amp-bf16 \
  --train-amp-bf16 \
  --skip-folds 0 1 2 3 4
```

Configuration:

* Pipeline: `dense_edge_mamba`
* Temporal model: dense-edge Mamba
* **`mamba-ssm` fused CUDA scan enabled**
* **bf16 AMP enabled for dense-edge processing**
* **bf16 AMP enabled during training**
* Disk CWT cache disabled
* Channel subset: `k=23`
* Same six-fold LOSO set
* Same prediction task, SPH/SOP, and 20-epoch budget

The run explicitly reports `use_cuda_kernel=True (mamba-ssm fused scan)`.

## Fold-by-fold comparison

| Seizure      | MambaPy AP |  Fused AP |       Δ AP | MambaPy Recall | Fused Recall |   Δ Recall | MambaPy F1 |  Fused F1 |       Δ F1 |
| ------------ | ---------: | --------: | ---------: | -------------: | -----------: | ---------: | ---------: | --------: | ---------: |
| `1_03_0`     |      0.156 |     0.273 |     +0.117 |          0.800 |        0.967 |     +0.167 |      0.310 |     0.341 | **+0.031** |
| `1_04_0`     |      0.480 |     0.598 |     +0.118 |          0.967 |        1.000 |     +0.033 |      0.283 |     0.244 | **−0.039** |
| `1_15_0`     |      0.629 |     0.573 |     −0.056 |          0.900 |        0.967 |     +0.067 |      0.551 |     0.496 | **−0.055** |
| `1_16_0`     |      0.883 |     0.977 |     +0.094 |          1.000 |        1.000 |      0.000 |      0.657 |     0.622 | **−0.035** |
| **`1_18_0`** |  **0.618** | **0.778** | **+0.160** |      **0.100** |    **0.633** | **+0.533** |  **0.171** | **0.704** | **+0.533** |
| `1_26_0`     |      0.229 |     0.231 |     +0.002 |          0.733 |        0.800 |     +0.067 |      0.364 |     0.387 | **+0.023** |

### Key observation

The F1 difference is extremely small on every fold except `1_18_0`:

* `1_03_0`: +0.031
* `1_04_0`: −0.039
* `1_15_0`: −0.055
* `1_16_0`: −0.035
* **`1_18_0`: +0.533**
* `1_26_0`: +0.023

Thus, the two Mamba configurations produce broadly similar fold-level behavior on **five of six seizures**. The aggregate difference is dominated by a single catastrophic MambaPy result on `1_18_0`.

## The `1_18_0` anomaly

The original MambaPy run essentially failed to detect the preictal interval on `1_18_0`:

* Recall: **0.100**
* F1: **0.171**
* AUC-PR: **0.618**

The fused-kernel/bf16 configuration instead produced:

* Recall: **0.633**
* F1: **0.704**
* AUC-PR: **0.778**
* Precision: **0.792**
* FAR/h: **0.833 → 0.000** after smoothing
* Preictal hit: **yes**

This is a very large behavioral difference on exactly one seizure. The fused run does not merely improve the score slightly; it changes `1_18_0` from the principal failure case into the strongest F1 result in the six-fold experiment.

## Full fused-kernel results

|  # | Seizure  | Hit | FAR/h raw → smoothed | Precision | Recall |    F1 | AUC-PR |
| -: | -------- | :-: | -------------------: | --------: | -----: | ----: | -----: |
|  1 | `1_03_0` |  ✓  |      18.500 → 14.000 |     0.207 |  0.967 | 0.341 |  0.273 |
|  2 | `1_04_0` |  ✓  |      36.000 → 25.161 |     0.139 |  1.000 | 0.244 |  0.598 |
|  3 | `1_15_0` |  ✓  |       10.116 → 4.186 |     0.333 |  0.967 | 0.496 |  0.573 |
|  4 | `1_16_0` |  ✓  |        4.667 → 0.000 |     0.451 |  1.000 | 0.622 |  0.977 |
|  5 | `1_18_0` |  ✓  |        0.833 → 0.000 |     0.792 |  0.633 | 0.704 |  0.778 |
|  6 | `1_26_0` |  ✓  |       14.000 → 6.600 |     0.255 |  0.800 | 0.387 |  0.231 |

The `1_26_0` run completed successfully with F1=0.387, recall=0.800, and AUC-PR=0.231.

The resulting fused-kernel six-fold means are:

* AP: **0.572**
* Recall: **0.895**
* F1: **0.466**

The run also achieved a 6/6 event-level hit rate.

For comparison, the documented MambaPy six-fold means were:

* AP: **0.499**
* Recall: **0.750**
* F1: **0.389**

## Interpretation

The important result is not simply that the fused-kernel configuration has higher six-fold means. The fold-level pattern is more informative.

The fused and MambaPy configurations remain within roughly ±0.055 F1 on **five of the six seizures**. The only major divergence is `1_18_0`, where F1 increases from 0.171 to 0.704 and recall increases from 0.100 to 0.633.

Furthermore, `1_26_0` is almost unchanged between configurations:

* AP: 0.229 → 0.231
* F1: 0.364 → 0.387
* Recall: 0.733 → 0.800

This makes the `1_18_0` result particularly notable. The fused configuration does not appear to be producing a wholesale change in model behavior across all seizures. Instead, it appears to eliminate a specific failure observed in the MambaPy run.

### Current conclusion

The original `1_18_0` Mamba result should therefore **not be interpreted as evidence that dense-edge Mamba intrinsically fails on this seizure**.

The more defensible conclusion is:

> The MambaPy run exhibited a severe, apparently configuration-sensitive failure on `1_18_0`. Re-running the same dense-edge Mamba pipeline with the `mamba-ssm` fused CUDA kernel and bf16 AMP reproduced broadly similar performance on five of six folds while eliminating the `1_18_0` recall/F1 collapse.

There is still a confound: the follow-up changes both the Mamba implementation and numerical precision (fused `mamba-ssm` + bf16), so the experiment does **not yet establish which change caused the rescue**. A controlled FP32 fused-kernel or bf16 MambaPy run would be needed to isolate that effect.

Nevertheless, the fold-by-fold result strongly suggests that the dramatic `1_18_0` failure in the original MambaPy experiment was an **outlier associated with that particular implementation/configuration**, rather than a stable architectural characteristic of dense-edge Mamba.
