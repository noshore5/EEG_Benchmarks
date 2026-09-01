# Tried & shelved — experiment ledger

Running list of levers we've **tried** on the CHB-MIT seizure-prediction
task and stopped (early-killed or run to a negative/wash). Not "dead" —
just tried, so a new shell doesn't re-derive them. Add a row when you
kill or conclude an experiment. Newest first within each section.

**"pre" reproduced 2026-08-31 on this Mac (MPS): 6-fold mean AP 0.644,
ROC-AUC 0.973** (per-fold .560/.811/.416/.985/.811/.283). The 0.03 gap to
0.674 is entirely fold 1_03 variance. So 0.674 is real; "pre" sits
~0.64-0.67. channel_cwt (.149/.263) collapsed ~0.5 below this.

Baselines to beat: **`temporal_graph_mamba` "pre" mean AP 0.674**
(prediction leader), **"post" 0.639** (fallback deliverable, graph-native),
**`hermitian_ssm` eigenvector encoder ~0.44** (hermitian ceiling).

Per-fold AP spread on the *unchanged* "pre" baseline is ~0.29–1.00, so
**partial-run reads are misleading, not just noisy** — a killed-at-2-folds
result that's ~0.6 below "pre" on both is real; one that's ~0.1 below is
not distinguishable from fold variance without paying for the full 6-fold.

---

## `hermitian_ssm` (Hermitian coherence graph → eigenpairs → encoder → temporal SSM)

| date | lever | expectation | result | verdict | note |
|---|---|---|---|---|---|
| 2026-08-30 | **`channel_cwt`** null baseline — per-channel CWT power, NO coherence, same `_DenseEdgeMambaTemporal` block as "pre" | if it matches "pre", the graph buys nothing | killed 2/6: 1_03 **0.149** vs 0.792, 1_04 **0.263** vs 0.830 | **NEGATIVE** — clean 1-var ablation; coherence representation carries ~all the signal | `Session_notes/2026_08_30/channel_cwt_null_baseline_equivalence.md` |
| 2026-08-31 | **ablation 1: `channel_cwt` + `mamba_n_hops=2`** — per-channel CWT power + 1 round learned complete-graph message passing (`MLP([h_j,h_i])` + GRUCell over all 23·22 pairs), still NO coherence | does cross-channel *mixing* recover the gap, or is it coherence specifically? | full 6-fold mean AP **0.173** (.150/.190/.180/.185/.189/.147), ROC-AUC 0.878 — dead flat every fold, ≤ no-hops | **NEGATIVE** — learned interaction of per-channel summaries can't substitute for the cross-spectrum; it's `arg S_ij` specifically. `results/hermitian_ssm/prediction/*_20260831-054118.csv` | same note |
| 2026-08-29 | `mamba_backend="mamba3"` (complex-diagonal SSM) on eigenvector encoder | complex state better fits complex eigen-features | 6-fold **0.408 vs 0.436**, lost most folds | NEGATIVE | `Session_notes/2026_08_29/mamba3_and_perfreq.md` |
| 2026-08-29 | k axis: top-6 → top-12 eigenpairs | more modes = more coupling structure | 6-fold **0.390 vs 0.436**, lost 5/6 | NEGATIVE — k=6 is the right truncation; modes 7+ are estimation noise on ~30 windows | k=12 cache `293ac41c6e53675f` deleted |
| 2026-08-29 | `encoder_mode="matrix"` (`_ComplexMatrixEncoder`) — complex channel-mix `M=WAW^H`, no C×C flatten | tests if `graph`'s 0.169 was the flatten hack or the rank-k neck | killed 3/6: 0.213/0.290/0.349 vs 0.254/0.324/0.414 | NEGATIVE — beats `graph` but ~0.06 under eigenvector; rank-6 truncation is the binding constraint | `Session_notes/2026_08_29/hermitian_ssm_bandmatch_6fold.md` |
| 2026-08-29 | `encoder_mode="complex"` (`_ComplexSpectralEncoder`) — de-engineered eigenvector encoder, 3.35× fewer params | matches baseline with far less hand-engineering | killed 3/6: 0.291/0.318/0.312 vs 0.254/0.324/0.414 | wash-to-worse; possible *methods* story, not a number win | as above |
| 2026-08-29 | `encoder_mode="projector"` (`_ProjectorEncoder`) — gauge-invariant node summaries of `P=Σλ_r u_r u_r^H` | remove eigenvector gauge noise | 6-fold **0.273**, ROC-AUC 0.898, 5/6 folds down | NEGATIVE — gauge noise wasn't the bottleneck; C×C→C-vector collapse loses more than it saves | `Session_notes/2026_08_29/` |
| 2026-08-29 | `encoder_mode="graph"` (upper triangle of `P`, lossless) | keep the whole graph as an object | fold 1 = **0.169**, killed | NEGATIVE | as above |
| 2026-08-29 | `encoder_mode="evolution"` (complex k×k `M(t)=U(t)^H U(t-1)` + λ, pure mode space, zero channel identity) | temporal mode-mixing is the signal | folds 1–2 ~chance (.062, .091), val_auc stuck <0.77, killed | NEGATIVE | as above |
| 2026-08-28 | `hermitian_ssm` first real 6-fold (float16, d_model=64, 8–124 Hz) | richer than per-node Mamba → beat 0.674 | mean AP **0.253**, weakest pipeline in repo | NEGATIVE (band later matched → ~0.44 ceiling, still < 0.674) | `Session_notes/2026_08_28/hermitian_ssm_first_6fold_and_eigh_fix.md` |

**Encoder-variant investigation CLOSED 2026-08-29:** result degrades
monotonically with how much channel identity the encoder drops. What
hermitian needs is *which channels couple*, in raw eigenvector
coordinates. Ceiling ~0.44; "pre" (0.674) stays the leader.

---

## `temporal_graph_mamba` (the prediction leader — attempts to improve it)

| date | lever | expectation | result | verdict | note |
|---|---|---|---|---|---|
| 2026-08-28 | `temporal_graph_aggregate="post"` (per-edge Mamba, keep 253-edge graph through the temporal model) | pre-aggregation discards which pair carried a transient | full 6-fold **0.639 vs 0.674** (gap all fold 1_03), ROC-AUC **0.970 vs 0.94** | WASH — ~11× compute for a tie; default stays "pre", knob parked. Now the **fallback deliverable** (graph-native) | `Session_notes/2026_08_28/temporal_graph_aggregate_post_6fold.md` |
| 2026-08-31 | `dense_edge_source="recompute"` (item 2b): cache CWT pre-pooled ×16 `[2,23,480,8]` ~0.7 MB/trial, rebuild the `[4,253,480,8]` edge stack in-forward at T=480 instead of disk-caching the 15 MB/trial stack | the disk-cache 6-fold set > 16 GB RAM → page-cache thrash (folds 2-6 at 85-370 s/epoch); recompute fits RAM | full 6-fold mean AP **0.469 vs 0.644** (pre_repro) / 0.674 (hist) — down **0.175**, worse on 5/6 folds; **epoch time ~42-57 s every fold** (2-6× faster) | NEGATIVE (accuracy) — non-bit-identical coarse-grid recompute (`smooth@480 ≠ smooth@7680-then-pool`, `xspec(downsample) ≠ downsample(xspec)`) is a pure test-set generalisation loss (train curves fine). Keep `disk_cache` as production; param+code parked (guarded, default off) | CONTEXT.md item 2b; CSV `results/temporal_graph_mamba/prediction/*_20260831-091141.csv` |
| 2026-08-31 | `temporal_graph_edge_drop_significance` (item 3): drop stack component 3 (significance) from `temporal_edge_proj` — `Linear(3·nfreqs→edge_dim)` on `[coh, sinφ, cosφ]` only, in-forward slice, disk cache untouched | in fixed mode significance is an affine copy of coherence — should be free to drop | full 6-fold mean AP **0.542 vs 0.644** (pre_repro) — down 0.10, but MIXED: worse on 1_03/1_04/1_16, **better** on 1_15 (.568 vs .416) and 1_18 (.872 vs .811) | MILD NEGATIVE / noisy — deficit ~2× fold noise, concentrated in 1_03. Motivates item 3b (`dense_edge_ch3="coi_mask"`): significance's only non-redundant content in fixed mode is `−coi_valid` after masking, so try a clean COI-mask channel instead | CONTEXT.md item 3; CSV `results/temporal_graph_mamba/prediction/*_20260831-110945.csv` |
| 2026-08-28 | wide band 8–40 → 8–124 Hz (nfreqs 15 then 10) | 8–40 was a disk-budget pick, never swept | fold 1_03 **0.792 → 0.31 → 0.24**, killed 1–2 folds | NEGATIVE — wide band dilutes the 8–40 signal through `temporal_edge_proj`'s linear mix | comment in `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS` |
| 2026-08-31 | `channel_subset_k=12` (item 5d, RAM-resident mission): keep only the top-12 channels by `abs_cosine` → 66/253-edge clique, ~3.8× dense-edge cache cut | a 66-edge clique holds the top coherence structure; cache fits in RAM at the current band | folds 1_03/1_04/1_15 AP **0.296 / 0.528 / 0.460** vs pre_repro **0.560 / 0.811 / 0.416** — 3-fold mean **0.428 vs 0.596**, down ~0.17; **killed after fold 3** per run-discipline gate | NEGATIVE — dropping 187/253 edges (74% of the graph) is amputation, not compression; the top-12 cosine clique loses too much of the pairwise coherence signal. K=4 (queued behind) cancelled with it. Pivot to fp16 cache (2×, zero edges dropped) as the RAM lever instead | CONTEXT.md item 5d; log `_to_delete/pre_csk12_real_20260831-145254.log` (no CSV — stopped mid-run) |
| 2026-08-27 | reg tuning: `weight_decay` 1e-4→3e-4, dropout 0→0.15 | cut FAR/h at the fixed 0.5 threshold | fold 1_03 **0.792 → 0.23**; fold 1_04 failed to train | NEGATIVE — reverted to untuned defaults | `Session_notes/2026_08_27/temporal_graph_mamba_full_6fold_and_tuning_attempt.md` |

---

## Other pipelines (cross-patient CHB-MIT)

| date | pipeline / lever | result | verdict | note |
|---|---|---|---|---|
| 2026-08-30 | `cg_mambanet` capacity cut (match per-subject LOSO data scale) | re-run did NOT fix the overfitting-driven gap, ~wash | NEGATIVE | commit `37eedf4` |
| 2026-08-2x | SlimSeiz per-patient channel-select | headline numbers likely use one fixed 8-ch montage, not per-patient; `--slimseiz-fixed-channels` added | (see memory `slimseiz-*`) | |
| 2026-08-25 | `dbconformer` real 6-fold + 2 diagnostics | negative diagnostics | NEGATIVE | CONTEXT.md ~line 629 |

---

## The common thread (why these keep coming back negative-to-wash)

The bottleneck is **not representational capacity** — "pre" funnels the
full `[4, E=253, T', F=8]` edge stack through an 8-wide neck + 8-dim Mamba
and still hits 0.674. The binding constraint is the **data budget**:
chb01-only LOSO, 6 seizures, ~30 preictal windows in the positive class
per fold. Anything that adds effective capacity (wider band, more
eigenmodes, per-edge modeling, bigger encoder, complex state) trades
against that ~30-sample signal — the model fits its own val split
(val_roc_auc ~0.97 every time) and generalizes *worse* to the held-out
seizure.

Levers considered more promising than "add richness":
1. **Decision-threshold calibration** against the val PR curve — every
   "pre" fold has high AP but terrible FAR/h (precision ~0.2, recall 1.0);
   it ranks well and is scored at a bad operating point. Still unbuilt.
2. **Seed-repeat the 0.674 baseline** (2–3 seeds, full 6-fold) for an
   error bar, so future A/Bs are interpretable. `run_pre_repro.py` is a
   start.
3. **The anomaly-detection reframe** — train on interictal only, flag
   preictal as departure from normal; never learns preictal features so
   the ~30-window budget stops being the cap. Best early signal the
   direction has produced (smoke ROC-AUC 0.80). See
   `Session_notes/2026_08_30/less_engineered_encoders_and_anomaly_reframe.md`.
