# `godoy_tmc` (TMC-T) 5-seed sweep — chb01 prediction LOSO

**Date:** 2026-08-31
**Driver:** `_to_delete/run_godoy_seed_sweep.py` (pid 38814), subprocess
loop, seeds 42–46, each a full independent
`run_pipelines.py --pipeline godoy_tmc --label-mode prediction --device mps --seed <s>`.
Logs: session `18ef5218-…/scratchpad/godoy_seed{42..46}_*.log`.

## Why

The single seed-42 run (2026-08-31, `*_20260831-080715.csv`) landed
6-fold mean AP **0.619** — close enough to the `temporal_graph_mamba`
"pre" leader (0.644 same-machine / 0.674 historical) to be read as a
near-tie #2. One run, one seed, and the per-fold spread was huge
(1_04 0.291 vs 1_18 1.000). This sweep puts an error bar on 0.619
before it is treated as real.

## What varies across seeds

`common.set_seed` seeds python / numpy / torch-CPU (model init, batch
shuffle) but **not MPS**, and the LOSO loop's negative-subsample seed is
hard-fixed at 42 regardless of `--seed`. So the sweep measures
model-init + batch-order variance with the training negative set held
fixed. Note: the seed-42 sweep run (`*_20260831-124010.csv`) came back
**bit-identical** to the original `*_20260831-080715.csv` — godoy_tmc on
MPS is deterministic given the seed here, so seeds 43–46 are genuine
init/shuffle draws, not MPS noise.

## Results — per-fold average precision

| fold  | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | fold mean | min–max |
|-------|---------|---------|---------|---------|---------|-----------|---------|
| 1_03_0 | 0.549  | 0.158  | 0.155  | 0.137  | 0.112  | 0.222 | 0.11–0.55 |
| 1_04_0 | 0.291  | 0.162  | 0.359  | 0.281  | 0.367  | 0.292 | 0.16–0.37 |
| 1_15_0 | 0.311  | 0.352  | 0.199  | 0.335  | 0.173  | 0.274 | 0.17–0.35 |
| 1_16_0 | 1.000  | 1.000  | 1.000  | 1.000  | 1.000  | 1.000 | — |
| 1_18_0 | 1.000  | 1.000  | 1.000  | 1.000  | 0.797  | 0.959 | 0.80–1.00 |
| 1_26_0 | 0.564  | 0.616  | 0.830  | 0.550  | 0.369  | 0.586 | 0.37–0.83 |
| **6-fold mean** | **0.619** | **0.548** | **0.590** | **0.550** | **0.470** | | |

**Sweep headline: mean AP 0.556, sample std 0.056** (pop std 0.050).
Range 0.470–0.619.

CSVs (`Epilepsy/results/godoy_tmc/prediction/`):
seed 42 `prediction_leave_one_seizure_out_20260831-124010.csv` (≡ `-080715`),
43 `-125636`, 44 `-131322`, 45 `-133242`, 46 `-135037`.

## Reading

- **0.619 was the top of the range, not the centre.** Seed 42 drew a
  favourable `1_03` (AP 0.549); the other four seeds all land `1_03` at
  0.11–0.16. Strip that one lucky fold and seed 42 looks like the rest.
- **All the seed variance is in two folds.** `1_03` (0.11–0.55) and
  `1_26` (0.37–0.83) account for essentially the entire spread of the
  6-fold mean. `1_16` is perfect every seed; `1_18` perfect on 4/5.
  This is the *same* "two hard folds carry the variance" structure as
  "pre" (whose 0.644↔0.674 gap is entirely `1_03`).
- **`1_15` is a recurring miss** — 0 preictal windows predicted on
  seeds 42, 45, 46 (hit=False), AP 0.17–0.35. Calibration failure on
  that fold, consistent across seeds.
- **Effective error bars, same machine / week / protocol:**
  "pre" 0.644–0.674, "post" ~0.639, **godoy 0.47–0.62 (mean 0.556)**.
  godoy is a clear third, ~0.08–0.09 AP behind the coherence-graph
  models — not a near-tie.

## Bottom line

`godoy_tmc`'s honest headline is **0.556 ± 0.056**. It is a useful
*representative raw-signal Transformer baseline* — cite it with the
mean±std — but not "the SOTA we beat": it is a reconstruction (not the
authors' code), untuned (paper's stated TMC-T hyperparameters), used
off-label (TMC-T is a detection architecture, run here in prediction
mode), and high-variance. Its real contribution to the comparison is
the ablation point that raw-signal deep models plateau ~0.1 AP below
the coherence representation on this data budget (~30 preictal
windows/fold) — the same "common thread" every capacity lever in
`NEGATIVES.md` keeps hitting.

## Follow-up

d_model capacity mini-sweep launched after this sweep to check whether
*less* / *more* capacity generalises better on the tiny positive class.
Per user call it was cut to **seed 42 only** at d16 and d64 (d16 seed 42
came back clearly below d32, so seeds 43/44 were gated off; d64 was
scoped to one seed from the start). n_heads stays 8 throughout
(16/8 = 2 dims/head, 64/8 = 8 dims/head — both non-degenerate).
Driver: `_to_delete/_godoy_run_one.py <d_model> <n_heads> <seed>`
(monkeypatches `gt._ConvTokenizer.__init__` → `mid_channels=(16,32,d)`).

### d_model capacity sweep — per-fold AP (seed 42)

| fold  | d16 | **d32** | d64 |
|-------|-----|---------|-----|
| 1_03_0 | 0.130 | **0.549** | 0.134 |
| 1_04_0 | 0.169 | **0.291** | 0.362 |
| 1_15_0 | 0.342 | **0.311** | 0.168 |
| 1_16_0 | 0.994 | **1.000** | 0.945 |
| 1_18_0 | 1.000 | **1.000** | 1.000 |
| 1_26_0 | 0.478 | **0.564** | 0.382 |
| **6-fold mean** | **0.519** | **0.619** | **0.498** |

d16 CSV `*_20260831-140959.csv`, d64 CSV `*_20260831-142719.csv`,
d32 (seed 42) `*_20260831-080715.csv` (≡ `-124010`).

### Verdict — {16, 32, 64} is an inverted-U with the peak at 32

d_model=32 (the paper's stated TMC-T width) is the best of the three on
seed 42: **0.619 vs 0.519 (d16) vs 0.498 (d64)**. Both directions lose
~0.10 AP. Cutting capacity (d16) sags `1_03`/`1_04`/`1_26`; adding it
(d64) sags the same folds *plus* `1_15` and even `1_16` — d64 fits its
val split fine (val_roc_auc peaks ~0.998) and loses the held-out
seizure, the exact `NEGATIVES.md` "common thread" failure. Note the
comparison here is single-seed d16/d64 vs single-seed d32 (0.619); the
honest d32 mean is 0.556 ± 0.056, and one d16/d64 draw carries the same
~0.05 fold-variance, so read this as "no width beats 32," not a
precise curve. Either way: **godoy_tmc is already at its best
config — the width sweep does not rescue it toward "pre" (0.644).**
No d_model=8 run (gate not met). Capacity axis closed for godoy_tmc.
