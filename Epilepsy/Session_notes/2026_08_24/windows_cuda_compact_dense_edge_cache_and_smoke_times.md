# Session notes — compact dense-edge cache, skip-CWT on hits, CUDA cache-off default (2026-08-24)

Branch: `main` (`c1c7278` at session start; this session's cache changes
are uncommitted in the working tree —
`dense_edge_cache.py`, `cwt_gnn_classifiers.py`, `run_pipelines.py`,
`smoke_test.py`, `pipeline_debug.py`). Repo:
`C:\Users\User\Documents\noshore5\EEG_Benchmarks`. Machine: local Windows
box, RTX 3070 Ti (8GB VRAM). Subject chb01.

Follow-on to
[yesterday's Windows/CUDA bf16 note](../2026_08_23/windows_cuda_bf16_speedup_predict_oom_fix_and_kofn.md)
(23-channel smoke 10.51–15.36s/epoch, both bf16 flags) and the
[MPS k=4 smoke on `dynmaic_subset`](../2026_08_23/fixed_graph_zero_masked_subset_and_smoke_runs.md)
(~11–14s/epoch with 100% disk hits). Same live-clique scatter into a
fixed 23-node / 253-edge graph; this note is only the cache path and the
smoke epoch times that forced the rewrite.

User's starting observation, after a `git pull` and a k=4 CUDA smoke:
100% dense-edge cache hits, epoch times **25.82–29.91s**, "it looks like
the cache is not helping us." They also changed `save_dense_edge` to
`.float().numpy()` so bf16 tensors would write (numpy has no bfloat16).
That change is correct and is kept; it was not the bottleneck.

---

## Why a 100% hit-rate cache was slower than yesterday's 23-channel recompute

`--channel-subset-k 4` only runs WCT for the cosine-top-k clique
(`m = k(k-1)/2 = 6` live edges) and scatters into a full-E zeros tensor
`[4, 253, T, F]`. The GNN still sees E=253. The disk cache stored that
scattered tensor as-is:

- 30s windows, `downsample=16` → `[4, 253, 479, 8]` float32 ≈ **15.5 MB/trial**
- 247/253 edges were the scatter zeros (~97%)
- Streaming `fit=False` reloads every batch every epoch
- Measured **26 ms/file** `np.load` on this box → ~0.8s/batch of I/O,
  **~7.6 GB/epoch** at `batch_size=32` / 16 train batches
- CWT still ran on every batch: `keep_on_device` already bypasses the CWT
  disk cache, and `w_real`/`w_imag` are only an input to
  `compute_dense_edge_input`. On a dense-edge hit they were thrown away.

Yesterday's 23-channel both-bf16 recompute was **14–17 ms/trial** for the
full mesh. k=4 live WCT (after this session's compact rewrite, measured
on the same GPU) is **~2.3–2.5 ms/trial**. Reading 15.5 MB of zeros was
slower than either.

MPS looked like ~11s for the same k=4 setup because unified-memory page
cache made the 15 MB files cheap after a couple of epochs. CWT was still
running there too (fp32 CWT + fp32 train). The Windows box stayed
disk-bound.

`.float()` on save: `dense_edge_amp_bf16` leaves bf16 on the GPU;
`tensor.cpu().numpy()` without `.float()` raises. Kept in
`save_dense_edge`. Not the 26s.

---

## What changed

### 1. Compact live-edge npz (`dense_edge_cache.py`)

Full-mesh entries still store `dense` as `[4, E, T, F]`. Live-clique
entries (`channel_subset_k` set) now write only the nonzero edge slots:

```
dense     [4, m, T, F]
edge_idx  int32[m]
n_edges   int32 (full E, so load can scatter back)
```

`_compact_dense_payload` drops exact-zero edge slots (max-abs over
`(4, T, F)`). `_expand_compact_dense` scatters back to full E on load.
`_npz_has_edge_idx` reads only the zip directory so a fat 15 MB file is
detectable without loading the array.

Roundtrip on this box (synthetic 6-of-253, same T/F as prediction):

| | |
|---|---|
| compact size | **0.369 MB** (was 15.5 MB) |
| save | 10.6 ms |
| load + expand | 1.7 ms |
| 773 chb01 windows on disk | **0.28 GB** (was ~12 GB) |

### 2. Skip CWT on a complete dense-edge hit (`cwt_gnn_classifiers.py`)

Streaming batches call `_prepare_features(..., fit=False)`. If every
trial in the batch is a dense-edge disk hit, CWT is unused — skip it
and only z-score the raw windows (`_raw_x_tensor_from_windows`).
All-or-nothing: a mixed batch still goes through CWT so the misses can
be computed. `_LazyFeatureBatchDataset` now precomputes
`dense_edge_cache_key`s once per `fit()`
(`precompute_dense_edge_cache_keys`), same role as the CWT window keys.

### 3. Fat-file policy is device-dependent

Same 15 MB k-subset file is a win on MPS and a loss on CUDA.

| | CUDA (`require_compact=True`) | MPS/CPU (`migrate_compact=True`) |
|---|---|---|
| fat k-subset file | treat as **miss** | **load** (page cache is cheap) |
| then | recompute live clique, write compact | rewrite compact under the same key |
| why | live WCT ~2.5 ms/trial ≪ 26 ms `np.load` | treating as miss would force CWT+WCT recompute (~61s cold on that machine) |

Full-mesh entries (`channel_subset_k is None`) never take this path —
no zeros to drop, so `require_compact`/`migrate_compact` stay False.

`_dense_edge_fat_cache_kwargs(n_channels)` is the single switch.

### 4. CUDA defaults the disk cache **off**

User: "for this machine, it looks like `--disable-disk-cache` wins every
time." Same direction as the
[2026-08-21 pod](../2026_08_21/pod_image_dataset_bake_and_cwt_dense_edge_cache_removal.md)
(removing both caches cut epoch time ~34%). Full 23-channel tensors are
still 15.5 MB with no zeros to drop; a warm cache there can still lose
to yesterday's GPU compute.

`resolve_disable_disk_cache(device, explicit)` in `run_pipelines.py`:

- unset + CUDA → cache off (recompute every trial)
- unset + MPS/CPU → cache on
- `--disable-disk-cache` / `--no-disable-disk-cache` force either way

`smoke_test.py` / `pipeline_debug.py` go through the same helper.
`PARAMS["disable_disk_cache"] = None` (was hardcoded `False`).

Compact + skip-CWT still matter for MPS (and for anyone forcing
`--no-disable-disk-cache` on CUDA). They are not the default on this box.

---

## Smoke config (every row below)

```
python Epilepsy/smoke_test.py --device cuda --channel-subset-k 4
```

`dense_edge_gru`, prediction, chb01, 30s/30s, `epochs=2`, `max_folds=1`,
`max_interictal_recordings=5`, `validation_split=0.2`, both bf16 flags,
k=4 → 6 live edges into E=253. Train fold `1_02_0`: 623 windows, 16
optimizer steps/epoch, batch_size=32. Test: 150 windows, 30 preictal.

Loss was identical across the cache-on runs (`0.716010` / `0.677136`) —
the rewrite did not change features, only I/O.

## Measured epoch times (this box, this smoke)

| | epoch 1 | epoch 2 | mean | wall | what the cache was doing |
|---|---|---|---|---|---|
| User's run (session start) | 29.91s | 25.82s | **27.87s** | 64.53s | 100% hits of 15.5 MB zeros; CWT still ran |
| After compact (cold rewrite) | 13.79s | 8.34s | 11.06s | 28.78s | fat treated as miss; live WCT ~2.3 ms/trial; compact files written |
| After compact + skip-CWT (warm) | **9.46s** | **7.75s** | **8.61s** | **21.66s** | 100% compact hits, CWT skipped |

Cold-rewrite dense-edge phase timing (8 chunks of ≤4 trials, device=cuda):
`compute=0.067–0.070s` → **2.26–2.34 ms/trial**. Yesterday's full-mesh
bf16 figure was 14–17 ms/trial; k=4 is ~6× fewer live edges, numbers
line up.

## How this sits next to yesterday / MPS

| | machine | k | cache | epoch_time |
|---|---|---|---|---|
| 2026-08-23 smoke, both bf16 | this CUDA box | 23 (full mesh) | recompute | 10.51–15.36s (best **10.51s**) |
| 2026-08-23 real 6-fold, both bf16 | this CUDA box | 23 | recompute | ~13.8–15.4s, ~34 min wall |
| 2026-08-23 MPS k=4 smoke | Apple Silicon | 4 | 100% fat hits, CWT still ran | ~11–14s |
| This session, fat hits | this CUDA box | 4 | 100% 15.5 MB hits | 25.82–29.91s |
| This session, compact warm | this CUDA box | 4 | 100% 0.37 MB hits, CWT skipped | **7.75–9.46s** |

Compact+skip-CWT recovered the k=4 smoke from ~28s back under yesterday's
23-channel floor, which is the right direction: live WCT is cheaper than
the full mesh, and 0.37 MB reads should not dominate. MPS should get the
skip-CWT win in steady state without paying CUDA's fat-miss rewrite; that
was not re-measured on this box.

## CUDA default after this: cache off

The user then set policy: on this machine disable the disk cache. The
uncapped k-sweep launched later this session
(`run_pipelines.py --device cuda --channel-subset-k {4,8,12,16,20}
--validation-split 0.2 --dense-edge-amp-bf16 --train-amp-bf16`) therefore
runs with `resolve_disable_disk_cache("cuda") == True`. k=4 there was
~3.8–4.0s/epoch (uncapped 4241 windows, not the smoke's 623) — recompute
of 6 live edges plus skip-the-disk, not a cache hit. That sweep is a
separate result, not this note.

## Files

- [`Epilepsy/pipelines/dense_edge_cache.py`](../../pipelines/dense_edge_cache.py)
  — compact payload, `require_compact` / `migrate_compact`, `.float()` on save,
  `precompute_dense_edge_cache_keys`.
- [`Epilepsy/pipelines/cwt_gnn_classifiers.py`](../../pipelines/cwt_gnn_classifiers.py)
  — `_try_load_complete_dense_edge_batch`, skip CWT in `_prepare_features`
  when `fit=False`, `_dense_edge_fat_cache_kwargs`, precomputed keys on
  `_LazyFeatureBatchDataset`.
- [`Epilepsy/run_pipelines.py`](../../run_pipelines.py) —
  `resolve_disable_disk_cache`, `--disable-disk-cache` /
  `--no-disable-disk-cache`.
- [`Epilepsy/smoke_test.py`](../../smoke_test.py) /
  [`pipeline_debug.py`](../../pipeline_debug.py) — same default.
