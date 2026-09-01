# All-model comparison — chb01 seizure-prediction LOSO (2026-08-31)

Supersedes `2026_08_25/pipeline_comparison_gru_mamba_dbconformer_slimseiz.md`
as the single place every prediction pipeline's number lives. That doc
(+ its CG-MambaNet addendum) is still the detailed write-up for the four
it covers; this one is the full board, refreshed with the pipelines built
since (`temporal_graph_mamba` "pre"/"post", `godoy_tmc`) and the
recompute/significance ablations.

## Protocol (identical across every row unless noted)

chb01, `label_mode=prediction`, `sph=300 s` / `sop=900 s`, the same 6
leave-one-seizure-out folds (`1_03_0, 1_04_0, 1_15_0, 1_16_0, 1_18_0,
1_26_0`), same `run_pipelines.py` shared training loop:
`validation_split=0.2`, `early_stopping_patience=5`,
`epochs=20`, `use_class_weights=True`, negative windows subsampled 5:1.

**What is and isn't comparable:**

- **Metrics are comparable; wall-clock is not.** GRU and enc-free Mamba
  ran on a Windows/CUDA RTX 3070 Ti box (the `dense_edge_mamba`
  worktree); CG-MambaNet on a RunPod RTX 4090; everything else on this
  Mac (`--device mps`). Different hardware, different seeds/environments.
- **Single untuned run per pipeline** (each at its own best-known config,
  not a symmetric hyperparameter search). No error bars anywhere. Per-fold
  AP spread on the *unchanged* "pre" baseline is ~0.28–0.99, so treat
  gaps under ~0.05 mean AP as noise.
- **Two input representations.** `temporal_graph_mamba` "pre"/"post",
  enc-free GRU/Mamba, and `hermitian_ssm` all consume this repo's CWT +
  coherence-graph feature stack. `godoy_tmc`, DBConformer, SlimSeiz,
  CG-MambaNet classify raw `(23, T)` windows directly. That's the axis
  being compared, not a confound.
- **"pre" has a same-machine number.** `run_pre_repro.py` (2026-08-31,
  MPS) reproduced the historical 0.674 leader at **mean AP 0.644** — the
  0.03 is entirely fold `1_03` seed variance. The "pre" vs `godoy_tmc`
  pair is the cleanest in the table: same Mac, same week, same protocol.

---

## Primary board — candidate pipelines, by mean AP

| pipeline | input | AP | AUC | acc | prec | rec | f1 | FAR/h | FAR/h sm | hit raw | hit k-of-n | source run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`temporal_graph_mamba` "pre"** (leader) | CWT graph | **0.644** | **0.973** | 0.907 | 0.478 | 0.833 | 0.521 | 10.73 | 7.62 | **6/6** | 5/6 | `20260830-211240` (pre_repro); hist 0.674 |
| `temporal_graph_mamba` "post" | CWT graph | 0.639 | 0.970 | 0.903 | 0.511 | 0.883 | 0.582 | 11.51 | 8.90 | **6/6** | **6/6** | `20260828-161354` |
| `godoy_tmc` (TMC-T) | raw EEG | **0.556 ± 0.056** | 0.968 | **0.923** | 0.457 | 0.811 | 0.532 | 8.66 | 6.31 | 5/6 | 5/6 | 5-seed sweep 42–46; single-run 0.619 = `20260831-080715` |
| enc-free **Mamba** (dense-edge) | CWT graph | 0.499 | 0.953 | 0.896 | 0.348 | 0.750 | 0.389 | 11.73 | 6.39 | **6/6** | 5/6 | `20260825-113049` (CUDA) |
| **DBConformer** (depth 5, weighted) | raw EEG | 0.442 | 0.952 | 0.897 | 0.273 | 0.772 | 0.366 | 11.73 | 7.95 | **6/6** | 5/6 | `20260825-175207` |
| `hermitian_ssm` (eigenvector encoder) | CWT graph | 0.436 | 0.947 | 0.888 | 0.346 | 0.824 | 0.411 | 13.16 | 8.02 | **6/6** | **6/6** | `20260829-111904` |
| **SlimSeiz** (adaptive channel select) | raw EEG | 0.430 | 0.951 | 0.913 | 0.285 | 0.556 | 0.340 | **8.56** | **6.31** | 5/6 | 4/6 | `20260825-171651` |
| enc-free **GRU** (dense-edge) | CWT graph | 0.423 | 0.944 | 0.878 | 0.343 | 0.807 | 0.436 | 14.26 | 9.21 | **6/6** | 5/6 | `20260825-105659` (CUDA) |
| **CG-MambaNet** | raw EEG | 0.127 | 0.797 | 0.837 | 0.124 | 0.516 | 0.197 | 17.90 | 1.50 | **6/6** | 3/6 | `20260829-114933` (CUDA) |

### Per-fold AP

| seizure | "pre" | "post" | godoy | Mamba | DBConf | herm-eig | SlimSeiz | GRU | CG-MN |
|---|---|---|---|---|---|---|---|---|---|
| `1_03_0` | 0.560 | 0.407 | 0.549 | 0.156 | 0.279 | 0.254 | 0.261 | 0.141 | 0.159 |
| `1_04_0` | 0.811 | 0.837 | 0.291 | 0.480 | 0.405 | 0.324 | 0.355 | 0.339 | 0.204 |
| `1_15_0` | 0.416 | 0.487 | 0.311 | 0.629 | 0.518 | 0.414 | 0.330 | 0.472 | 0.101 |
| `1_16_0` | 0.985 | 0.994 | 1.000 | 0.883 | 0.883 | 0.279 | 1.000 | 0.619 | 0.100 |
| `1_18_0` | 0.811 | 0.854 | 1.000 | 0.618 | 0.362 | 0.788 | 0.425 | 0.776 | 0.091 |
| `1_26_0` | 0.283 | 0.257 | 0.564 | 0.229 | 0.203 | 0.556 | 0.212 | 0.193 | 0.108 |

---

## Ablations & variants (not candidate pipelines — diagnostic runs)

| run | what it tests | AP | verdict |
|---|---|---|---|
| `channel_cwt` null baseline | per-channel CWT power, **no coherence**, "pre"'s temporal block | ~**0.15–0.26** (killed 2/6: `1_03` 0.149, `1_04` 0.263) | NEGATIVE — coherence carries ~all the signal |
| `channel_cwt` + `mamba_n_hops=2` | + 1 round learned complete-graph message passing, still no coherence | **0.173** (`.150/.190/.180/.185/.189/.147` — dead flat) | NEGATIVE — it's the cross-spectrum `S_ij` / `arg S_ij` specifically, not cross-channel mixing in general |
| `dense_edge_source="recompute"` (item 2b) | cache CWT pre-pooled ×16, rebuild edge stack in-forward at T=480 (RAM-fit, not bit-identical) | **0.469** (`.396/.404/.279/.682/.842/.208`) | NEGATIVE (accuracy) — 2–6× faster epochs but −0.175 AP from coarse-grid smooth/xspec |
| `hermitian_ssm` encoder variants | de-engineered / complex / matrix / projector / graph encoders on the eigenpair stack | 0.27–0.31 or killed; `mamba3` 0.408; k=6→12 0.390 | all NEGATIVE vs the 0.436 eigenvector encoder — ceiling ~0.44, rank-6 truncation is the bind |

(`continuous_cwt_mamba` — the streaming paradigm — has a 1-fold CPU smoke
only, never run 6-fold per user call. Not in either table.)

---

## Reading the board

**"pre" is the leader on the metrics this repo optimizes** — highest AP
(0.644) and AUC (0.973), 6/6 raw hits. "post" (0.639) is a statistical
tie that keeps the full 253-edge graph through the temporal model
(~11× compute); it's the fallback "graph-native" deliverable. The gap
from "pre" to the best raw-EEG model is small.

**`godoy_tmc` — honest number is 0.556 ± 0.056, not 0.619.** The
single-run 0.619 (seed 42) was the *top* of a 5-seed sweep (42–46:
0.619, 0.548, 0.590, 0.550, 0.470); mean 0.556, sample std 0.056. That
puts the raw-signal 1-layer Transformer ~0.09 AP below the same-machine
"pre" (0.644) and ~0.08 below "post" (0.639) — a clear third, not a
near-tie. The whole seed spread lives in `1_03` (AP 0.11–0.55 across
seeds) and `1_26` (0.37–0.83); seed 42 simply drew a good `1_03`
(0.549 vs ~0.15 for the other four). `1_16` stays perfect every seed
(AP 1.000) and `1_18` nearly so (0.80–1.00), both on ≤30 test windows.
`1_15` is missed (0 preictal windows predicted) on 3 of 5 seeds. Same
"two hard folds carry all the variance" fragility as "pre" — but where
"pre"'s error bar is 0.644–0.674, godoy's is 0.47–0.62. Off-label too:
TMC-T is a *detection* architecture and this is a reconstruction, not
the authors' code. Cite it as "a representative raw-signal Transformer
baseline, 0.556 ± 0.056," not as a settled #2. Sweep detail:
`Session_notes/2026_08_31/godoy_tmc_seed_sweep.md`.

**The enc-free GRU/Mamba rows are stale references, not current
contenders.** They predate "pre"/"post" and ran on different hardware.
enc-free Mamba (0.499) was the old leader; "pre" is the same feature
family walked through time with a per-node Mamba instead of Conv2d-pooled
edge features, and it's worth +0.14 AP.

**DBConformer / SlimSeiz / hermitian-eigenvector cluster at ~0.43–0.44**
— none purpose-built for this protocol. SlimSeiz is a distinct operating
point (lowest FAR/h, lowest recall, most conservative), not uniformly
worse. `hermitian_ssm` topped out here: the encoder-variant investigation
is closed, ceiling ~0.44, "pre" stays ahead.

**CG-MambaNet (0.127) is last by a wide margin** — a cross-patient-scale
architecture reconstructed from a paper, run on single-patient folds;
overfits fast, and a capacity cut didn't fix it (see the 2026_08_25 doc's
addendum). Treat as "run once as reconstructed," not the architecture's
ceiling.

### Fold-level structure worth knowing

- **`1_18_0`** — hard for every raw-EEG model at the 0.5 threshold
  (recall 0.00–0.10) *except* `godoy_tmc` (perfect). "pre" ranks it fine
  (AP 0.811) but its thresholded recall drops to 0.40. A
  decision-threshold effect, not a ranking one — flagged in four notes
  now, still uninvestigated.
- **`1_26_0`** — second-hardest across the board; every pipeline's
  second- or third-lowest AP. `godoy_tmc` (0.564) and hermitian-eig
  (0.556) handle it best; "pre" (0.283) and the enc-free models (~0.2)
  worst. It's "pre"'s persistent k-of-n miss.
- **`1_16_0`** — easiest fold for everyone; `godoy_tmc`, SlimSeiz score
  perfect AP, "pre"/"post" ~0.99.
- **`1_03_0`** — the fold that carries all of "pre"'s run-to-run
  variance (0.56 this run, 0.79 historical).

---

## Caveats

- No error bars. One run per pipeline, each its own seed/environment,
  three different machines. Gaps under ~0.05 mean AP are not
  distinguishable from fold variance.
- "Best-known config" ≠ "architecture ceiling." GRU/Mamba and "pre"
  weren't put through diagnostics analogous to DBConformer's
  depth/class-weight sweep.
- The raw-EEG pipelines (godoy, DBConformer, SlimSeiz, CG-MambaNet) get
  no benefit from this repo's CWT/coherence feature engineering; the
  CWT-graph pipelines get no benefit from end-to-end learning on the raw
  signal. Comparing them is comparing (architecture + representation) as
  a package.
- SlimSeiz's row is the adaptive-selection run (complete data); its
  fixed-montage variant tied it on f1/precision/FAR-h and is
  operationally safer — see the 2026_08_25 doc.
