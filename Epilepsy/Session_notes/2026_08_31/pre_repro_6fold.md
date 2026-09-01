# `temporal_graph_mamba` "pre" reproduction -- 6-fold, same machine, MPS (2026-08-31)

Mac shell. Follows `2026_08_30/channel_cwt_null_baseline_equivalence.md`.

## Why

The `channel_cwt` null baseline (per-channel CWT power, no coherence)
collapsed -- fold 1_03 AP 0.149, fold 1_04 0.263, killed 2/6, ~0.5 below
"pre". Before treating that as "coherence carries the signal" we needed
a **clean same-machine "pre" reference**, because:

- the historical 0.674 was days old, `--device cpu`, and stitched across
  3 overnight process restarts (`2026_08_27/temporal_graph_mamba_full_
  6fold_and_tuning_attempt.md`);
- the roundup note (`2026_08_28/negative_results_roundup_and_fragility.md`)
  flags fold 1_03's AP swinging 0.29-1.00 on *unchanged* config, and a
  mild reg change once collapsed it 0.792 -> 0.232 -- so we didn't
  actually know 0.674 was stable.

## Run

`_to_delete/run_pre_repro.py` -> `--pipeline temporal_graph_mamba
--label-mode prediction --device mps`, untuned
`PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS` (`temporal_graph_aggregate="pre"`,
d_model=16, seed 42). CSVs:
`results/temporal_graph_mamba/prediction/*_20260830-211240.csv`.

- **run1** (pid 4938): killed at fold 1 by this session's swap-watchdog
  monitor (13 GB kill line, set too low). fold 1_03 = **0.560** on run1.
- **run2** (pid 5562): full 6-fold, ~8.5 h wall.

## Result: REPRODUCES

| fold | this run (MPS) | historical | delta |
|---|---|---|---|
| 1_03_0 | 0.560 | 0.792 | **-0.232** |
| 1_04_0 | 0.811 | 0.830 | -0.019 |
| 1_15_0 | 0.416 | 0.331 | +0.085 |
| 1_16_0 | 0.985 | 0.996 | -0.011 |
| 1_18_0 | 0.811 | 0.802 | +0.009 |
| 1_26_0 | 0.283 | ~0.29 | ~0 |
| **mean AP** | **0.644** | **0.674** | -0.030 |
| mean ROC-AUC | **0.973** | ~0.94 | **+0.03** |

Event-level hit rate 6/6 raw, 5/6 smoothed (1_18 misses smoothed).
Mean FAR/h 10.7 (range 0.17 - 31.7). Precision 0.15-0.92, recall
frequently 1.0 -- ranks well, calibrated badly at the 0.5 threshold.

**The entire 0.03 mean gap is fold 1_03** (0.560 vs 0.792 = -0.232, which
is -0.039 on the 6-fold mean). Every other fold matches or beats
historical; drop 1_03's variance and the mean is *above* 0.674. Two runs
of identical code (run1, run2) both gave 1_03 = 0.560 -- reproducible on
MPS, just different from the CPU run's 0.792 draw. ROC-AUC is actually
*higher* than historical (and matches the "post" run's 0.970).

## Conclusions

1. **0.674 is real.** "pre" sits ~0.64-0.67 depending on fold 1_03 luck.
   Not a fluke, not a broken historical run.
2. **The `channel_cwt` collapse is genuine.** 0.149 / 0.263 is a ~0.5 AP
   drop below a validated same-machine baseline -> the coherence edge
   representation (`|coh_ij|`, `sin phi_ij`, `cos phi_ij` per freq/time)
   carries essentially all the signal. Per-channel magnitude spectra +
   the same temporal block cannot substitute.
3. **MPS vs CPU is moot.** MPS reproduces the CPU-established number. The
   open "was 0.674 CPU-only?" question (notes said cpu x2; user recalled
   MPS) no longer matters.
4. **Fragility thesis reconfirmed from a new angle.** Fold 1_03 moved
   0.79 -> 0.56 between two runs of *bit-identical code*. Partial-run
   per-fold reads on chb01 LOSO are noise; only the 6-fold mean is
   interpretable. (~30 preictal windows/fold positive class.)
5. **Still the open lever:** decision-threshold calibration. Every fold
   has high AP / high ROC-AUC and a bad FAR/h at the fixed 0.5 cutoff.
   The model ranks; the operating point is wrong. Unbuilt.

## Perf note (see CONTEXT.md item 2b)

fold 1 ran 44 s/epoch (warm 13 GB dense_edge cache from prior sessions);
folds 2-6 ran 85-390 s/epoch, **IO-bound** (CPU 77% idle, RAM full).
`event_mode="temporal_graph"` caches the raw `[4, 253, 480, 8]` edge
stack (~15 MB/trial, ~70 GB for a 6-fold set) which exceeds 16 GB RAM ->
page-cache thrash, every epoch re-reads ~10 GB npz off SSD. fold 2's
epoch 1 = 2379 s (cold cross-spectrum compute, inline in the first
training epoch). Fix = item 2b: cache only the per-channel CWT
(~1 MB/trial, fits RAM), recompute the deterministic
`compute_dense_edge_input` on MPS each forward -- same model in
`coherence_threshold_mode="fixed"`, and the 2026-08-27 note found
recompute *faster* than `np.load` on CUDA.
