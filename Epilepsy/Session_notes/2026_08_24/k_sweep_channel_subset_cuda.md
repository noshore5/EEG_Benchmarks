# Session notes — channel-subset k sweep on Windows/CUDA (2026-08-24)

Branch: `main` (`c1c7278` plus uncommitted cache/CLI wiring). Repo:
`C:\Users\User\Documents\noshore5\EEG_Benchmarks`. Machine: local Windows
box, RTX 3070 Ti (8GB VRAM). Subject chb01.

Follow-on to
[today's compact-cache note](windows_cuda_compact_dense_edge_cache_and_smoke_times.md)
(CUDA defaults the disk cache off) and
[yesterday's documented 23-channel 6-fold](../2026_08_23/full_6fold_subject1_run_results.md)
(`20260823-132639`, AP 0.413, no val split, all 20 epochs, ~34 min).

User asked for a sequential sweep `k = 4, 8, 12, 16, 20` after concluding
`--disable-disk-cache` wins on this box, and pushed back that 34 min was
the 23-channel ceiling — smaller k plus early stopping should be faster.
Both were true. All five runs exit 0. Total wall **54 min 11 s**
(09:04:12 → 09:58:23 +04).

---

## Setup (shared across k)

```
python Epilepsy/run_pipelines.py --device cuda --channel-subset-k <k> \
    --validation-split 0.2 --dense-edge-amp-bf16 --train-amp-bf16
```

Sequential PowerShell loop; logs under
`Epilepsy/results/prediction/k_sweep_20260824-090412/`.

| | |
|---|---|
| pipeline | `dense_edge_gru` (prediction) |
| subject / CV | chb01 leave-one-seizure-out, 6 seizures (`03/04/15/16/18/26`) |
| windows | 30s / 30s, SPH 300s, SOP 900s |
| dataset | uncapped interictal, `X=4241` windows (173 preictal); train negatives 5:1 |
| n_test | 630–750 / fold (~4% preictal; chance AP ≈ 0.04) |
| graph | always `n_channels=23`, `E=253`; live clique scattered into zeros |
| live edges | `k(k-1)/2` → 6 / 28 / 66 / 120 / 190 (full mesh is 253) |
| CWT | still all 23 channels |
| epochs | cap 20, `validation_split=0.2`, `early_stopping_patience=5` |
| amp | both bf16 |
| disk cache | off (CUDA default; see companion note) |
| k-of-n | 8/10 |

`--validation-split` (underscore alias `--validation_split`) was wired
this session so the flag actually reaches `clf_params`. GRU prediction
still defaults `validation_split=0.0` if you omit it — yesterday's 23ch
run did. Early stopping is a no-op without a val split.

**Not a clean k-vs-23 ablation.** Yesterday's 23ch run trained every
epoch on 100% of fold train. This sweep holds out 20% and restores the
best val checkpoint.

AP is sklearn `average_precision_score` (area under the PR curve,
printed as `auc_pr`). Precision/recall/F1 are one 0.5-threshold point.

---

## Mean across 6 folds

| k | wall | ~s/epoch | acc | prec | rec | f1 | **AP** | AUC | FAR/h | FAR/h k-of-n | hit raw | hit k-of-n |
|---|------|----------|-----|------|-----|-----|--------|-----|-------|--------------|---------|------------|
| 4 | 7.4 min | ~3.9 | 0.789 | 0.112 | 0.574 | 0.183 | 0.167 | 0.809 | 24.18 | 9.41 | 6/6 | 5/6 |
| 8 | 8.4 min | ~4.4 | 0.846 | 0.188 | 0.627 | 0.279 | 0.186 | 0.854 | 17.42 | 5.61 | 6/6 | 4/6 |
| 12 | 11.2 min | ~6.2 | 0.885 | 0.305 | 0.784 | 0.418 | 0.457 | 0.932 | 13.35 | 6.23 | 6/6 | **6/6** |
| 16 | 12.5 min | ~9.2 | 0.887 | 0.356 | 0.817 | 0.453 | 0.534 | 0.950 | 13.12 | 5.91 | 6/6 | 5/6 |
| **20** | **14.7 min** | **~12.1** | **0.897** | **0.420** | **0.845** | **0.513** | **0.567** | **0.953** | **12.02** | **5.64** | 6/6 | 5/6 |

AP, AUC, precision, recall, F1, and raw FAR all improve with k. Smoothed
FAR is the one non-monotonic number (best at k=8). k=12 is the only
setting with 6/6 k-of-n hits; k=16 and k=20 drop `1_26_0`.

Live WCT scales with `k(k-1)/2`. Epoch time 3.9s → 12.1s is in the same
direction as yesterday's 23ch ~13.8–15.4s (253 edges, no val).

## Early stop (patience 5; “20” = hit the cap)

| k | stop epoch / fold (`03 04 15 16 18 26`) | best epoch |
|---|---|---|
| 4 | 19 / 15 / 9 / 9 / 20 / 20 | 14 / 10 / 4 / 4 / — / — |
| 8 | 16 / 11 / 20 / 20 / 9 / 18 | 11 / 6 / — / 15 / 4 / 13 |
| 12 | 17 / 15 / 18 / 14 / 12 / 14 | 12 / 10 / 13 / 9 / 7 / 9 |
| 16 | 10 / 12 / 12 / 10 / 13 / 12 | 5 / 7 / 7 / 5 / 8 / 7 |
| 20 | 10 / 12 / 11 / 9 / 9 / 10 | 5 / 7 / 6 / 4 / 4 / 5 |

Larger k stopped earlier. That is why k=20 was only 2× k=4 wall time
despite ~3× epoch time.

## Per-fold AP

| seizure | k=4 | k=8 | k=12 | k=16 | k=20 |
|---|---|---|---|---|---|
| `1_03_0` | 0.116 | 0.224 | 0.424 | **0.588** | 0.361 |
| `1_04_0` | 0.113 | 0.138 | 0.255 | 0.409 | **0.437** |
| `1_15_0` | 0.445 | 0.256 | 0.560 | **0.627** | 0.544 |
| `1_16_0` | 0.086 | 0.239 | 0.503 | 0.477 | **0.921** |
| `1_18_0` | 0.175 | 0.152 | 0.838 | 0.943 | **0.950** |
| `1_26_0` | 0.065 | 0.105 | 0.160 | 0.162 | **0.187** |

`1_18_0` is the fold that unlocks at k≥12. `1_26_0` stays weak at every
k (k-of-n miss except at k=12). k=20's mean win is mostly `1_16_0`
(0.477 → 0.921); `1_03_0` actually dropped vs k=16.

## k=20 per fold (best mean AP in this sweep)

| seizure | hit raw→k-of-n | FAR/h | prec | rec | f1 | AP |
|---|---|---|---|---|---|---|
| `1_03_0` | T→T | 20.8→15.3 | 0.194 | 1.000 | 0.324 | 0.361 |
| `1_04_0` | T→T | 31.4→16.3 | 0.152 | 0.967 | 0.262 | 0.437 |
| `1_15_0` | T→T | 7.2→0.9 | 0.397 | 0.900 | 0.551 | 0.544 |
| `1_16_0` | T→T | 1.2→0.0 | 0.741 | 0.870 | 0.800 | 0.921 |
| `1_18_0` | T→T | 0.8→0.0 | 0.844 | 0.900 | 0.871 | 0.950 |
| `1_26_0` | T→**F** | 10.8→1.4 | 0.194 | 0.433 | 0.268 | 0.187 |

## Vs other real 6-fold runs (same seizures, 30s/30s, uncapped test)

| setup | AP | AUC | hit k-of-n | protocol |
|---|---|---|---|---|
| **k=20 GRU (this sweep)** | **0.567** | 0.953 | 5/6 | val 0.2, early stop, both bf16 |
| Truong STFT-CNN `20260823-114957` | 0.538 | 0.949 | 4/6 | different architecture |
| k=16 GRU | 0.534 | 0.950 | 5/6 | same as k=20 |
| k=12 GRU | 0.457 | 0.932 | 6/6 | same as k=20 |
| 23ch GRU `20260823-132639` | 0.413 | 0.916 | 3/6 | no val, all 20 epochs, both bf16 |

k=20 beats the documented full-mesh GRU on AP/AUC/k-of-n and is tied
with Truong on AP. Do not read that as “k=20 > 23 channels” until 23ch
is rerun with val 0.2 + early stopping.

Smoke / `max_interictal=5` numbers (AP ~0.85–0.97, n_test≈150) are a
different test distribution and are not in this table.

---

## CSVs

| k | leave-one-seizure-out | per-seizure |
|---|---|---|
| 4 | `prediction_leave_one_seizure_out_20260824-090435.csv` | `prediction_per_seizure_20260824-090435.csv` |
| 8 | `…-091201.csv` | `…-091201.csv` |
| 12 | `…-092027.csv` | `…-092027.csv` |
| 16 | `…-093137.csv` | `…-093137.csv` |
| 20 | `…-094407.csv` | `…-094407.csv` |

All under `Epilepsy/results/prediction/`. Sweep stdout:
`k_sweep_20260824-090412/{sweep.log,k4.log,k8.log,k12.log,k16.log,k20.log}`.

## Open

- Rerun 23ch with `--validation-split 0.2` so k=20 vs full mesh is a real
  ablation (CWT is already 23ch; only WCT/live edges change).
- `1_26_0` is the persistent miss — not solved by adding edges.
- k-of-n hit peaked at k=12 (6/6), not at the best-AP k.
