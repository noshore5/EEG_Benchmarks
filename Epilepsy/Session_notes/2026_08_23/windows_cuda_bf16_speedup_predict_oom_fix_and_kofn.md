# Session notes — Windows/CUDA bf16 speedup, `_predict_logits` OOM fix, k-of-n for prediction mode (2026-08-23)

Branch: `main` (`b97f06c` at session start; this session's changes are
uncommitted in the working tree — `common.py`, `cwt_gnn_classifiers.py`,
`run_pipelines.py`, `requirements.txt`). Repo:
`C:\Users\User\Documents\noshore5\EEG_Benchmarks`. Machine: local Windows
box, RTX 3070 Ti (8GB VRAM). Follow-on to
[2026-08-22's cache-restoration note](../2026_08_22/) (`c3ed741` — first
real Windows/CUDA validation of this pipeline; flagged `dense_edge_amp_bf16`
as a documented non-win, blocked on `torch.complex()` rejecting BFloat16).

Driven throughout by the user pushing back hard on unverified/mislabeled
timing claims — every number below is measured from a log, not estimated,
and one of my own mislabeled figures (a per-8-chunk total quoted as
per-chunk) was caught and corrected mid-session.

---

## Part 1 — `requirements.txt`: plain PyPI silently gives a CPU-only torch

Root cause of "why doesn't CUDA work on this machine yet": plain PyPI only
hosts CPU-only `torch` wheels for Windows/Linux. `pip install torch==2.8.0`
with no index override installs a build that reports
`torch.cuda.is_available() == False` even with a working GPU (confirmed:
driver 591.86, CUDA 13.1, `nvidia-smi` fine, `torch.cuda.is_available()`
False until reinstalled). The RunPod pod image never hit this — its base
image already had a CUDA torch baked in, and PEP 440 local-version matching
means an installed `2.8.0+cu128` already satisfies a bare `torch==2.8.0`
constraint, so `pip install -r requirements.txt` never touched it there.

Fixed: added `--extra-index-url https://download.pytorch.org/whl/cu128` and
pinned the exact `torch==2.8.0+cu128` local version, closing the gap for
both a baked image and a fresh environment. (Also dropped `uvloop==0.22.1`
— Unix-only, was breaking the Windows install.)

## Part 2 — `dense_edge_amp_bf16`: from "documented non-win" to a real ~10x fix

2026-08-22 had left this flag in place but broken: `torch.complex()` only
accepts Half/Float/Double, so wrapping `compute_dense_edge_input` in
`torch.autocast(dtype=torch.bfloat16)` crashed the moment `_smooth_wct_maps`
tried to build a complex tensor from bf16 conv2d output.

**Two real bugs found and fixed, not just the crash:**

1. **`_full_edge_wct_maps`'s elementwise ops were never autocast-eligible in
   the first place.** Confirmed directly: `torch.autocast` only downcasts
   conv2d/matmul-type ops, not plain `index_select`/multiply/subtract — so
   this stage's `w_real`/`w_imag`/`freqs` tensors stayed fp32 regardless of
   the outer autocast context. This is exactly what OOM'd at 23
   channels/T=7680 (`xwt_imag = src_i * dst_r - src_r * dst_i` tried to
   allocate 1.85GiB with nothing free). Fixed with an explicit
   `.to(amp_dtype)` cast gated on `torch.is_autocast_enabled()`.
2. **Fixing (1) alone still OOM'd at the same size** — `_smooth_wct_maps`
   forced its two conv2d output slices back to float32 just to satisfy
   `torch.complex()`, and complex64 is *always* 8 bytes/element regardless
   of the real dtype fed in, silently re-inflating the exact memory the
   change exists to shrink. Fixed by dropping `torch.complex()` entirely:
   `torch.angle(complex(r,i)) == atan2(i,r)` and
   `complex(r,i).abs()**2 == r*r + i*i` algebraically, so
   `_smooth_wct_maps`/`_smooth_wct_maps_scale_adaptive` now return `phase`
   (via `atan2`) directly instead of a complex tensor, and every caller
   (`_coherence_only`, `_build_sparse_events`, `_build_dense_edge_input`)
   takes it directly instead of calling `torch.angle()` itself. `coh`/`phase`
   now stay in bf16 the whole way through under autocast, matching fp32
   output bit-for-bit when the flag is off.

**Verified numerically before trusting it** (fp32-vs-bf16 dense-edge output
on a synthetic test-tone-like signal, not just timed): max abs diff
~0.007–0.008 on coh/significance (range [0,1]) — noise-level by this
pipeline's own existing bar (matches the (5,3)-vs-(25,3) smoothing-kernel
comparison precedent).

**Measured speedup**, dense-edge phase timing at 23 channels/253 edges
(`gnn_30s_23ch.log` vs `gnn_30s_23ch_bf16.log`):

| | compute/8-chunk call | per-trial |
|---|---|---|
| fp32 (before) | 2.9–5.9s | 91–183ms |
| bf16 (after) | 0.45–0.56s | 14–17ms |

~11x faster on this stage alone. Also added a `--max-channels` diagnostic
flag (truncate `X` right after windowing, cheap slice, no re-download) to
confirm dense-edge cost scales with edge count (`C*(C-1)/2`): a 5-channel
run's dense-edge compute was ~4x faster per-chunk than 23-channel at
comparable trial counts, consistent with 10 edges vs 253 edges.

## Part 3 — `train_amp_bf16`: the other ~60% of epoch time

`dense_edge_amp_bf16` only covers the *precompute* stage (no autograd
graph, `torch.no_grad()` throughout) — it left the trainable forward pass
(`channel_encoder`/`dense_edge_conv`/GRU/classifier) at full fp32. Added a
new, independent flag: `train_amp_bf16` on `SparseEvidenceGNNClassifier`
(constructor param) → `TorchEEGClassifier.train_amp_bf16` (`common.py`,
shared base class, opt-in/False-by-default so every other pipeline is
unaffected) → wraps `_model_forward`'s `self.model_(*batch_inputs)` call in
the same `torch.autocast(dtype=torch.bfloat16)` context.

bf16 (not fp16) for both flags for the same reason: full fp32 exponent
range means no `GradScaler`/loss-scaling is needed — optimizer state and
master weights stay fp32 regardless. One subtlety handled explicitly:
`criterion(...)` (CrossEntropyLoss → log_softmax/nll_loss) runs at call
sites *outside* `_model_forward`, outside the autocast context, so it
wouldn't get autocast's own built-in fp32-upcast safety net for those ops.
Fixed by always casting `_model_forward`'s returned `logits` to fp32 before
returning, regardless of the flag — a no-op (same tensor) when
`train_amp_bf16=False`.

**Measured, real GPU, 23 channels** (`gnn_30s_23ch_bothbf16.log`), per-epoch
wall time:

| config | epoch_time |
|---|---|
| fp32 baseline | 119.73s |
| `--dense-edge-amp-bf16` only | 23.95–33.49s |
| both flags | 10.51–15.36s (one 417.32s outlier, not reproduced) |

**~8–11x faster overall** vs. the fp32 baseline, at this smoke scale.
Metrics matched prior fp32 runs closely throughout — no divergence beyond
expected bf16 noise.

## Part 4 — k-of-n alarm smoothing added to `leave_one_seizure_out_prediction`

User's explicit ask: make the GNN prediction pipeline's event-level metrics
directly comparable to `leave_one_seizure_out_truong`'s own k-of-n-smoothed
numbers, not just window-level accuracy. Added the identical
`k_of_n_alarm` post-process (Truong et al. 2018 §II.D; same
`DEFAULT_TRUONG_K_OF_N_K=8`/`DEFAULT_TRUONG_K_OF_N_N=10` defaults, same CLI
flags `--k-of-n-k`/`--k-of-n-n`, no new pair invented) applied per held-out
(subject, run) recording in chronological `window_start` order, never
across recordings. Refactored event-metric computation into a shared
`_event_metrics(preds)` helper (mirroring truong's own), producing both
`raw_events` and `smoothed_events`; added `hit_smoothed`,
`n_false_alarms_smoothed`, `false_alarms_per_hour_smoothed` to both the
per-fold and per-seizure output, and both raw/smoothed lines to the final
"Mean across folds" summary print.

Verified on a diagnostic small-scale run
(`verify_predict_fix.log`): `false_alarms_per_hour` 26.6 → smoothed 15.0,
hit rate unchanged at 6/6 — smoothing cuts false alarms substantially
without costing detections at this scale.

## Part 5 — `_predict_logits` CUDA OOM: root-caused and fixed

**The bug**: `TorchEEGClassifier._predict_logits` (`common.py`) calls
`self._prepare_features(X, fit=False)` on the *entire* test set in one
un-chunked call before any batching happens — fine for eager classifiers
with small test sets, wrong for `StreamingSparseEvidenceGNNClassifier`'s
real use case. `negative_to_positive_ratio` deliberately never subsamples
TEST windows (only training negatives), so a real, uncapped
leave-one-seizure-out test fold can be as large as a training fold —
confirmed 750 windows for chb01 fold 1. First real 6-fold run (uncapped,
20 epochs/fold) crashed on exactly this call, right after fold 1's 20
training epochs completed cleanly:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 594.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Of the
allocated memory 11.90 GiB is allocated by PyTorch, ...
```
Traceback: `predict_proba` → `_predict_logits` → `_prepare_features` →
`_precompute_dense_edge_inputs` → `compute_dense_edge_input` →
`_build_dense_edge_input` → `_smooth` → `_smooth_wct_maps` → `F.pad`.

**Fix**: overrode `_predict_logits` on `StreamingSparseEvidenceGNNClassifier`
to reuse `_LazyFeatureBatchDataset` (already built for training) so only
one inference batch's CWT/dense-edge features are ever materialized at a
time, sequential order (not `_BatchIndexSampler`'s shuffle — prediction
order must match the caller's `X` row order), no labels needed.

**Verified before relaunching**: a quick small-scale sanity run
post-fix (`verify_predict_fix.log`) produced metrics matching the last
known-good pre-fix run almost exactly (accuracy 0.739549, roc_auc
0.977565), confirming no regression from either this fix or Part 4's
k-of-n addition, before trusting it at real scale.

The real 6-fold subject-1 run exercising all of the above (relaunched
after Part 5's fix) is written up separately:
[full_6fold_subject1_run_results.md](full_6fold_subject1_run_results.md).

## Open items

- None of this session's code changes are committed yet — `common.py`,
  `cwt_gnn_classifiers.py`, `run_pipelines.py`, `requirements.txt` all
  still uncommitted in the working tree at write time.
- The one-off 417.32s epoch-time outlier in `gnn_30s_23ch_bothbf16.log`
  was not investigated (not reproduced on the next run) — likely a one-time
  driver/OS stall (Windows/WDDM), not a code issue, but flagged rather than
  silently dropped.
- `dense_edge_amp_bf16`'s real/imag-instead-of-complex64 rewrite
  (`_smooth_wct_maps`, `_smooth_wct_maps_scale_adaptive`) changes the return
  type of `_smooth` for *every* caller, not just the bf16 path — verified
  correct when the flag is off (falls back to fp32 `atan2`/`r*r+i*i`,
  algebraically identical to the old `torch.complex`-based path) but worth
  a second pair of eyes given how many call sites it touched.
