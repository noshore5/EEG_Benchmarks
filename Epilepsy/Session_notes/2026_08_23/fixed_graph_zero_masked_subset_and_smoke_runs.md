# Session notes — fixed-graph zero-masked dense-edge subset + smoke runs (2026-08-23)

Branch: `dynmaic_subset` (`b51c504` at session start; this session's
changes uncommitted). Repo:
`/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`. Machine: local macOS,
Apple Silicon, MPS, `torch 2.8.0`. Subject chb01 throughout.

Follow-on to the earlier shrinking-subset work on this branch. The user
asked for a **fixed-shape** path: the GNN always sees full C and full
E=C*(C-1)/2, and only a selected clique of edges gets WCT/coherence
features. Non-live edge slots are zeros. Then three real (smoke-scale)
runs to check that it trains.

These runs are **not** comparable to
[the real uncapped 6-fold on `main`](full_6fold_subject1_run_results.md)
(`max_interictal_recordings=None`, ~650–750 test windows/fold). Everything
below uses `smoke_test.py`'s `max_interictal_recordings=5` cap, so test
FAR/h is computed against ~1 hour of interictal per fold, not a full
recording.

---

## Part 1 — Design: live clique, scatter into full-E zeros

Previous code on this branch **gathered** cosine-top-k channels and rebuilt
the GNN at `n_channels=k` (k-node graph, E=k*(k-1)/2). That is out of
scope. Replaced with:

1. `n_channels` stays full C (23 on CHB-MIT). Channel embeddings / node
   identity unchanged.
2. Edge layout stays full E=253, same `src_idx`/`dst_idx` order as
   `upper_pair_indices` (i<j nested loop) — **not** `ordered_pair_indices`
   (directed i≠j). Mismatch would silently scramble which full-E slot a
   live feature is written into.
3. Per window: absolute-cosine top-k channels → live edges = the undirected
   clique among those k (`m = k*(k-1)/2`). For k=4, m=6.
4. `_full_edge_wct_maps` / `compute_dense_edge_input` /
   `_build_dense_edge_input` take optional `src_idx`/`dst_idx` overrides
   and run WCT only for those pairs.
5. Live features scatter into a zeros tensor `[B, 4, E, T, F]`. Comment at
   the scatter site: fixed E layout, live edges only, zeros = not computed.
6. `channel_subset_k=None` or `<=0` or `>=C`: existing full-mesh path,
   unchanged.
7. Different trials in a batch can pick different cliques — per-trial loop;
   trial-0's subset is never applied to the rest.

Files:

- `Epilepsy/pipelines/channel_subset_dynamic.py` — kept
  `absolute_cosine_affinity` / `top_k_channels_from_affinity` /
  `select_channel_subset`; added `canonical_undirected_pairs`,
  `pair_to_edge_index`, `live_edges_from_channel_subset`.
- `Epilepsy/pipelines/cwt_gnn_classifiers.py` — constructor params
  `channel_subset_k` / `channel_subset_metric`; removed the gather that
  shrank C; live WCT + scatter in `_precompute_dense_edge_inputs`.
- `Epilepsy/pipelines/dense_edge_cache.py` — cache key hashes
  `channel_subset_k` and `channel_subset_metric` (defaults `None` /
  `"abs_cosine"` key identically to omitted args).
- `Epilepsy/run_pipelines.py` — `--channel-subset-k` /
  `--channel-subset-metric`, wired next to `dense_edge_amp_bf16` in both
  prediction and detection `clf_params`. `_SHARED_ARCH_PARAMS` defaults
  `channel_subset_k=None`.
- `datasets/epilepsy/chb_mit.py` — local `moabb 1.2.0` has no
  `FixedPipeline`; identity sklearn pipeline fallback so
  `smoke_test.py` can import on this machine.
- Later in the session: streaming classifier gained a lazy val loader
  (`_SequentialBatchSampler` + `_LazyFeatureBatchDataset`) so
  `validation_split=0.2` no longer raises `NotImplementedError`.

### Offline checks (synthetic, before any CHB-MIT run)

Pair order matches `upper_pair_indices` and `SparseEvidenceGNNCore`'s
registered `src_idx`/`dst_idx`. Live-override WCT slots match the
corresponding full-mesh slots (atol 1e-5); inactive slots are exact zeros.
`k=C` matches `k=None`. `k=4` keeps E full with 6 nonzero edges per trial.
Different trials in a batch picked different cliques. Cache keys differ
when k changes. `_build_model_from_features` still reports
`n_channels=C`, `E=C*(C-1)/2`. No `AttributeError` on
`self.channel_subset_k`.

---

## Shared smoke config (all three CHB-MIT runs)

```
python3 -u Epilepsy/smoke_test.py \
    --channel-subset-k 4 --window-length 30.0 \
    # plus per-run: --epochs, --max-folds, --validation-split
```

| | |
|---|---|
| pipeline | `dense_edge_gru` / `StreamingSparseEvidenceGNNClassifier` |
| label_mode | prediction |
| subject | chb01 |
| windows | 30s / 30s hop |
| dataset | `X=(773, 23, 7680)`, 173/773 preictal (22.4%), `max_interictal_recordings=5` |
| k | 4 → clique of 6 live edges, scattered into E=253 zeros |
| GNN | **n_channels=23, edges=253** (`sparse_classifier` is 2×4048 = 23×22×8 — full concat, not a 4-node model) |
| device | MPS. `dense_edge_amp_bf16` / `train_amp_bf16` are CUDA-only; no-ops here |
| batch_size | 32 (prediction default). 16 steps/epoch when val_split=0.2 (498 train), 20 when val_split=0 |
| nfreqs / downsample | 8 / 16 |
| k-of-n | 8-of-10 (pipeline default) |
| seed | 42 |
| cache | disk CWT + dense-edge on; 100% dense-edge hits after epoch 1 of the first run |

---

## Run 1 — 1 fold, 10 epochs, no val split

```
python3 -u Epilepsy/smoke_test.py --epochs 10 --channel-subset-k 4 --window-length 30.0
```

(`max_folds=1` is `smoke_test.py`'s PARAMS default.)

**Completed, exit 0. Wall 220.86s.** First confirmation the fixed-graph
path trains: printed config was `n_channels=23 edges=253`, not 4/6.

Live-edge WCT (coherence_threshold_mode="fixed", MPS): first 32-trial
chunk ~39 ms/trial cold; later chunks ~7–11 ms/trial. After epoch 1,
`[dense-edge cache] 32/32 trials reused from disk (100.0%)`.

Train (whole training set, no val):

| epoch | loss | acc | roc_auc | epoch_time |
|---|---|---|---|---|
| 1 | 0.773 | 0.417 | 0.535 | 61.16s (cold CWT/WCT) |
| 2 | 0.726 | 0.527 | 0.577 | 25.90s |
| 3 | 0.568 | 0.735 | 0.828 | 16.09s |
| 5 | 0.397 | 0.793 | 0.888 | 14.70s |
| 10 | 0.358 | 0.782 | 0.904 | 14.23s |

Held-out `1_02_0`: n_test=150 (30 preictal). acc 0.873, prec 0.634, rec
0.867, F1 0.732, AP 0.672, ROC-AUC **0.936**. Hit True (k-of-n True).
FAR/h 15.0 → 0.0 smoothed.

---

## Run 2 — aborted 7-fold (`--max-folds none`)

User asked for "all 7 folds", then corrected to the usual chb01 prediction
count of 6. Killed after three completed folds (`1_02_0`, `1_03_0`,
`1_05_0`) and relaunched as run 3. Numbers from those three folds match
run 3 (same seed, same data, warm cache) and are not tabulated separately.

---

## Run 3 — 6 folds, 10 epochs, no val split

```
python3 -u Epilepsy/smoke_test.py --epochs 10 --channel-subset-k 4 \
    --window-length 30.0 --max-folds 6
```

**Completed, exit 0. Wall 1212.12s (~20.2 min).** Epoch time mean 19.45s
(min 14.74, max 34.13, n=60).

| fold | seizure | n_test (preictal) | acc | prec | rec | F1 | AP | ROC-AUC | hit raw→k-of-n | FAR/h raw→smoothed |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `1_02_0` | 150 (30) | 0.873 | 0.634 | 0.867 | 0.732 | 0.672 | 0.936 | True → True | 15 → 0 |
| 2 | `1_03_0` | 150 (30) | 0.860 | 0.645 | 0.667 | 0.656 | 0.606 | 0.929 | True → True | 11 → 0 |
| 3 | `1_05_0` | 150 (30) | 0.887 | 0.659 | 0.900 | 0.761 | 0.694 | 0.942 | True → True | 14 → 0 |
| 4 | `1_06_0` | 143 (23) | 0.252 | 0.167 | 0.913 | 0.282 | 0.150 | 0.470 | True → True | 105 → 101 |
| 5 | `1_07_0` | 150 (30) | 0.693 | 0.265 | 0.300 | 0.281 | 0.370 | 0.799 | True → False | 25 → 0 |
| 6 | `1_10_0` | 30 (30) | 0.633 | 1.000 | 0.633 | 0.776 | 1.000 | — | True → True | — |

Means: acc 0.700, precision 0.562, recall 0.713, F1 0.581, AP 0.582,
ROC-AUC 0.815 (5 folds; `1_10_0` has no interictal test windows so AUC/FAR
are undefined). Event-level hit **6/6 raw, 5/6 k-of-n**.

Two caveats, both data-shape not model-crash:

- **`1_06_0` is the outlier** — train ROC ~0.95, test 0.47, FAR 105. k-of-n
  did not save it (smoothed FAR still 101).
- **`1_10_0` is a degenerate test set** (30/30 preictal) from the
  5-recording interictal cap + round-robin assignment, not a 4-node-GNN
  failure. The GNN was still 23/253 on this fold.

---

## Run 4 — 1 fold, 20 epochs, validation_split=0.2

Streaming `fit()` previously raised `NotImplementedError` for
`validation_split > 0`. Added a lazy val loader (same
`_LazyFeatureBatchDataset` as training, sequential batches, no shuffle)
so a 0.2 split does not materialize the val set all at once.

```
python3 -u Epilepsy/smoke_test.py --epochs 20 --channel-subset-k 4 \
    --window-length 30.0 --max-folds 1 --validation-split 0.2
```

**Completed, exit 0. Wall 367.87s.** Val split: **125/623** samples (fold's
training pool, not the held-out seizure). 16 optimizer steps/epoch.
Best checkpoint restored from **epoch 12** (`val_loss=0.452222`).

| epoch | train loss | train acc | train AUC | val loss | val acc | val AUC | epoch_s |
|---|---|---|---|---|---|---|---|
| 1 | 0.716 | 0.540 | 0.542 | 0.685 | 0.600 | 0.809 | 16.49 |
| 2 | 0.677 | 0.693 | 0.623 | 0.658 | 0.608 | 0.814 | 16.60 |
| 3 | 0.558 | 0.757 | 0.803 | 0.783 | 0.576 | 0.812 | 16.05 |
| 4 | 0.503 | 0.725 | 0.827 | 0.545 | 0.632 | 0.821 | 16.04 |
| 5 | 0.405 | 0.763 | 0.892 | 0.480 | 0.768 | 0.840 | 18.05 |
| 6 | 0.389 | 0.775 | 0.894 | 0.502 | 0.712 | 0.851 | 17.36 |
| 7 | 0.368 | 0.783 | 0.902 | 0.465 | 0.752 | 0.848 | 16.81 |
| 8 | 0.367 | 0.795 | 0.904 | 0.460 | 0.720 | 0.854 | 17.17 |
| 9 | 0.351 | 0.795 | 0.907 | 0.474 | 0.712 | 0.838 | 23.93 |
| 10 | 0.344 | 0.801 | 0.913 | 0.455 | 0.712 | 0.856 | 20.25 |
| 11 | 0.340 | 0.785 | 0.913 | 0.453 | 0.760 | 0.846 | 22.78 |
| **12** | **0.327** | 0.777 | 0.915 | **0.452** | 0.712 | 0.839 | 15.40 |
| 13 | 0.327 | 0.815 | 0.920 | 0.475 | 0.728 | 0.854 | 15.81 |
| 14 | 0.336 | 0.819 | 0.920 | 0.478 | 0.720 | 0.852 | 17.34 |
| 15 | 0.330 | 0.799 | 0.920 | 0.462 | 0.752 | 0.848 | 20.92 |
| 16 | 0.330 | 0.803 | 0.919 | 0.469 | 0.760 | 0.848 | 18.75 |
| 17 | 0.318 | 0.783 | 0.930 | 0.465 | 0.704 | 0.857 | 19.54 |
| 18 | 0.323 | 0.795 | 0.928 | 0.487 | 0.712 | 0.859 | 17.46 |
| 19 | 0.312 | 0.817 | 0.928 | 0.459 | 0.760 | 0.857 | 16.66 |
| 20 | 0.303 | 0.813 | 0.931 | 0.461 | 0.808 | 0.862 | 17.65 |

Train loss keeps falling after epoch 12; val loss bottoms there and
wobbles in 0.45–0.49. Mild overfitting, not a collapse. Mean epoch_time
18.05s (min 15.40, max 23.93).

Held-out `1_02_0` **using epoch-12 weights**: n_test=150 (30 preictal).
acc 0.867, prec 0.639, rec 0.767, F1 0.697, AP 0.647, ROC-AUC **0.930**.
Hit True (k-of-n True). FAR/h 13.0 → 0.0 smoothed. Slightly worse recall
than run 1's epoch-10/no-val checkpoint on the same fold (0.767 vs 0.867),
as expected from training on 80% of the fold and restoring an earlier
epoch.

---

## What these runs do and don't show

Do:

- The GNN stays at C=23 / E=253 with `--channel-subset-k 4`. Confirmed
  from the live printed config and from `sparse_classifier` shape, not
  from the code path alone.
- Live-edge precompute is cheap on MPS once the disk cache is warm
  (~15–20s/epoch for this smoke-scale fold).
- Train loss decreases; a 0.2 val split is now actually computed in
  prediction mode and tracks a best-epoch restore.

Don't:

- Not an apples-to-apples comparison with the uncapped 6-fold on `main`
  (different interictal pool, 10 vs 20 epochs, k=4 vs full mesh).
- `1_06_0` / `1_10_0` in run 3 are smoke-scale fold artifacts; don't read
  them as "k=4 is broken" without a full-interictal rerun.
- MPS, so neither bf16 flag was exercised.
