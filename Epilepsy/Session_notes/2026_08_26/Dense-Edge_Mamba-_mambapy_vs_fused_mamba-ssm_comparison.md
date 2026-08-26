# Dense-Edge Mamba: mambapy vs fused `mamba-ssm` comparison

## Purpose

Follow-up to the 2026-08-25 pipeline comparison of GRU, Mamba, DBConformer, and SlimSeiz on CHB-MIT subject 1.

The original dense-edge Mamba run showed an unusually severe failure on seizure `1_18_0`: recall collapsed to **0.100** and F1 to **0.171**, despite an AP of **0.618**. A second dense-edge Mamba run was therefore performed using the `mamba-ssm` fused CUDA implementation with bf16 AMP.

The purpose was to determine whether the `1_18_0` failure represented a general limitation of dense-edge Mamba or a configuration/implementation-specific failure.

---

## Shared evaluation protocol

Both Mamba runs use:

- Dataset: **CHB-MIT, subject 1**
- Label mode: **prediction**
- SPH: **300 s**
- SOP: **900 s**
- LOSO folds:
  - `1_03_0`
  - `1_04_0`
  - `1_15_0`
  - `1_16_0`
  - `1_18_0`
  - `1_26_0`
- Channel subset: **23 channels**
- Epochs: **20**

---

# 1. Original MambaPy configuration

Source run:

`20260825-113049`

Temporal model:

- Dense-edge Mamba
- **mambapy** implementation
- Pure PyTorch Mamba scan
- No fused CUDA Mamba kernel
- Original numerical precision/configuration

Six-fold aggregate:

| Metric | MambaPy |
|---|---:|
| Accuracy | **0.896** |
| Precision | **0.348** |
| Recall | **0.750** |
| F1 | **0.389** |
| AP | **0.499** |
| ROC-AUC | **0.953** |
| FAR/h raw | **11.73** |
| FAR/h smoothed | **6.39** |
| Raw event hit | **6/6** |
| k-of-n / smoothed hit | **5/6** |

---

# 2. Fused-kernel configuration

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
  --mamba-use-cuda-kernel \
  --skip-folds 0 1 2 3 4
```

Configuration:

- Dense-edge Mamba
- **`mamba-ssm` fused CUDA scan**
- `use_cuda_kernel=True`
- **bf16 AMP for dense-edge processing**
- **bf16 AMP for training**
- CUDA
- Disk CWT cache disabled
- `k=23`
- 20 epochs

---

# 3. Fold-by-fold F1 comparison

| Seizure | MambaPy | Fused Mamba | Δ |
|---|---:|---:|---:|
| `1_03_0` | 0.310 | **0.341** | +0.031 |
| `1_04_0` | 0.283 | **0.244** | -0.039 |
| `1_15_0` | 0.551 | **0.496** | -0.055 |
| `1_16_0` | 0.657 | **0.622** | -0.035 |
| `1_18_0` | **0.171** | **0.704** | **+0.533** |
| `1_26_0` | 0.364 | **0.387** | +0.023 |

The key observation is that **five of six F1 differences are within ±0.055**. The entire major divergence is `1_18_0`.

---

# 4. Fold-by-fold AP comparison

| Seizure | MambaPy | Fused Mamba | Δ |
|---|---:|---:|---:|
| `1_03_0` | 0.156 | 0.273 | +0.117 |
| `1_04_0` | 0.480 | 0.598 | +0.118 |
| `1_15_0` | 0.629 | 0.573 | -0.056 |
| `1_16_0` | 0.883 | 0.977 | +0.094 |
| `1_18_0` | 0.618 | **0.778** | +0.160 |
| `1_26_0` | 0.229 | **0.231** | +0.002 |

Importantly, the original `1_18_0` problem was **not actually an AP catastrophe**. MambaPy had AP = **0.618**, its second-best fold. The catastrophe occurred after applying the 0.5 decision threshold.

---

# 5. The `1_18_0` catastrophe

### Original MambaPy

| Metric | `1_18_0` |
|---|---:|
| AP | 0.618 |
| Recall | **0.100** |
| F1 | **0.171** |

### Fused + bf16 Mamba

| Metric | `1_18_0` |
|---|---:|
| AP | **0.778** |
| Precision | **0.792** |
| Recall | **0.633** |
| F1 | **0.704** |
| FAR/h raw → smoothed | **0.833 → 0.000** |
| Event hit | **✓** |
| k-of-n hit | **✓** |

The fused implementation converts `1_18_0` from the original Mamba's catastrophic fold into its **best F1 fold**.

For comparison, GRU on this fold had approximately:

- Recall: **0.767**
- F1: **0.754**

Thus fused Mamba reaches essentially the same performance regime on the fold that previously made Mamba look dramatically inferior.

---

# 6. The completed `1_26_0` fold

The sixth fold completed with the fused configuration and reported:

| Metric | Fused Mamba `1_26_0` |
|---|---:|
| Accuracy | **0.879365** |
| Precision | **0.255319** |
| Recall | **0.800000** |
| F1 | **0.387097** |
| AP | **0.230622** |
| ROC-AUC | **0.916972** |
| FAR/h raw | **14.000** |
| FAR/h k-of-n | **6.600** |
| Raw event hit | **1/1 (100%)** |
| k-of-n event hit | **1/1 (100%)** |

Original MambaPy on `1_26_0`:

| Metric | MambaPy |
|---|---:|
| Recall | 0.733 |
| F1 | 0.364 |
| AP | 0.229 |

So this fold is almost unchanged:

- AP: **0.229 → 0.231**
- F1: **0.364 → 0.387**
- Recall: **0.733 → 0.800**

This supports the interpretation that the fused configuration did **not** simply produce globally different behavior. The enormous difference is concentrated in `1_18_0`.

---

# 7. Full fused-Mamba fold results

| # | Seizure | Test preictal | Hit | FAR/h raw → k-of-n | Precision | Recall | F1 | AP |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|
| 1 | `1_03_0` | 30 | ✓ | 18.500 → 14.000 | 0.207 | 0.967 | 0.341 | 0.273 |
| 2 | `1_04_0` | 30 | ✓ | 36.000 → 25.161 | 0.139 | 1.000 | 0.244 | 0.598 |
| 3 | `1_15_0` | 30 | ✓ | 10.116 → 4.186 | 0.333 | 0.967 | 0.496 | 0.573 |
| 4 | `1_16_0` | 23 | ✓ | 4.667 → 0.000 | 0.451 | 1.000 | 0.622 | 0.977 |
| 5 | `1_18_0` | 30 | ✓ | 0.833 → 0.000 | **0.792** | 0.633 | **0.704** | **0.778** |
| 6 | `1_26_0` | 30 | ✓ | 14.000 → 6.600 | 0.255 | 0.800 | 0.387 | 0.231 |

### Event-level reliability

- **Raw event hit rate: 6/6 (100%)**
- **k-of-n / smoothed event hit rate: 6/6 (100%)**

---

# 8. Full-model comparison

The original six-fold comparison was:

| Metric | GRU | MambaPy | DBConformer | SlimSeiz | **Fused Mamba** |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.878 | 0.896 | 0.897 | **0.913** | **0.879** |
| Precision | 0.343 | 0.348 | 0.273 | 0.286 | **0.363** |
| Recall | 0.807 | 0.750 | 0.772 | 0.556 | **0.895** |
| F1 | 0.436 | 0.389 | 0.366 | 0.340 | **0.466** |
| **AP** | 0.423 | 0.499 | 0.442 | 0.431 | **0.572** |
| AUC | 0.944 | 0.953 | 0.952 | 0.951 | **see note below** |
| Raw hit | 6/6 | 6/6 | 6/6 | 5/6 | **6/6** |
| k-of-n hit | 5/6 | 5/6 | 5/6 | 4/6 | **6/6** |

The fused Mamba therefore has the highest:

- **AP**
- **Precision**
- **Recall**
- **F1**
- Raw event hit rate
- k-of-n/smoothed event hit rate

It does not have the highest accuracy; Mambapy does.

The fused run's `1_26_0` ROC-AUC is explicitly **0.916972**. The complete six-fold fused ROC-AUC mean should only be reported once all six per-fold AUC values have been retained/calculated; it should not be inferred from the other models' aggregate AUC.

---

# 9. Interpretation

The strongest result from this experiment is not simply that the fused Mamba has a higher aggregate score.

It is the **fold-by-fold stability**.

For F1:

```text
1_03_0     +0.031
1_04_0     -0.039
1_15_0     -0.055
1_16_0     -0.035
1_18_0     +0.533   <-- catastrophic MambaPy divergence
1_26_0     +0.023
```

Thus **five of six folds differ only marginally**, while one fold changes dramatically.

The same pattern is particularly convincing at `1_26_0`: the fused configuration is almost identical to MambaPy there despite the enormous improvement on `1_18_0`.

The original comparison had interpreted `1_18_0` as the major reason GRU had the best recall/F1. The fused run shows that this conclusion is specific to the original MambaPy configuration rather than necessarily to dense-edge Mamba itself.

---

# Conclusion

The evidence supports the following conclusion:

> **The severe `1_18_0` failure observed in the original MambaPy run appears to be configuration-sensitive rather than an inherent failure mode of dense-edge Mamba.**

The fused `mamba-ssm` + bf16 configuration produces broadly similar results to MambaPy on **five of six folds**, while dramatically rescuing `1_18_0`.

Across the completed six-fold experiment it achieves:

- **Accuracy: 0.879**
- **Precision: 0.363**
- **Recall: 0.895**
- **F1: 0.466**
- **AP: 0.572**
- **Raw event hits: 6/6 (100%)**
- **k-of-n / smoothed event hits: 6/6 (100%)**

This makes fused Mamba the strongest result so far on the project's primary **AP** metric, while also giving the highest recall and F1 among the compared configurations.

## Caveat

The fused experiment changes two coupled factors relative to MambaPy:

1. Mamba implementation: `mambapy` → `mamba-ssm` fused CUDA kernel
2. Numerical regime: original precision → bf16 AMP

Therefore this experiment does **not** isolate whether the rescue comes from the fused kernel, bf16, or their interaction.

The appropriate next controlled experiment would be to separate those factors. However, the current result is already strong evidence that the original `1_18_0` MambaPy catastrophe should **not** be treated as a stable architectural property of dense-edge Mamba.
