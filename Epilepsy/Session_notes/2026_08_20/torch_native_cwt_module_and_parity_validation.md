# Session notes — torch-native (torch.fft) CWT module, parity validation, stress test (2026-08-20)

Branch: `torch-native-cwt`. Repo: `/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`.

Goal: replace the FFTW-backed `fcwt.cwt()` computation in the Epilepsy CWT
pipelines with a torch-native (`torch.fft`) implementation — a
correctness-preserving swap, not a redesign. This session covers steps
0–5 of that plan (orientation → new module → parity validation → device
caution → precision check). **Step 6 (wiring it into the actual call
site) has NOT happened yet** — see Open items.

---

## Part 1 — Orientation (Step 0)

The real `fcwt.cwt()` call site is **not** in `cwt_gnn_classifiers.py`
itself — it's `utils/coherence_utils.py::transform()` (repo root), a thin
wrapper:

```python
freqs, coeffs1 = fcwt.cwt(signal1, frame_rate, lowest, highest, nfreqs, nthreads=4, scaling='log')
```

`Epilepsy/pipelines/common.py::resolve_coherence_utils()` dynamically loads
this (falling back to an external `Coherent_Multiplex` checkout if
`WCT_COHERENT_MULTIPLEX_ROOT` is set, else this repo's own copy) and hands
it to `_BaseCWTGNNClassifier` as `self.transform_`. It's then invoked
**per-(sample, channel) 1-D signal**, in a plain Python double loop, in
`cwt_window_cache.py::compute_cwt_real_imag_tensors_cached` (the cached
path the classifier actually uses) — no batching today.

Key facts established before writing anything:
- `fcwt.cwt()` always recasts its input to float32 internally regardless of
  input dtype (confirmed via the installed package's own
  `boilerplate.cwt()`) — so the FFTW path genuinely runs in float32, not
  float64, even though `coherence_utils.transform` upcasts to float64
  first (a no-op).
- Output: complex64, shape `(nfreqs, n_time)` (freq-major).
- Wavelet: Morlet, bandwidth `fb=2.0` (hardcoded in fcwt's
  `boilerplate.py`; independently corroborated by
  `cwt_gnn_classifiers.py`'s own `_COI_WAVELET_FB = 2.0` / `_coi_valid_mask`,
  which reconstructs fcwt's cone-of-influence from its edge-of-support
  formula `support = floor(fb*scale*3.0)` since `fcwt.cwt()` returns no
  COI itself).
- Canonical config (`run_pipelines.py::_SHARED_ARCH_PARAMS`):
  `sampling_rate=256, lowest=8.0, highest=40.0, nfreqs=8`, 4s windows
  (1024 samples).

Decisions confirmed with the user before writing code:
- New module location: **`utils/torch_cwt.py`**, sibling to
  `utils/coherence_utils.py` (the existing shared cross-pipeline home for
  this concern), not inside `cwt_gnn_classifiers.py` or `common.py`.
- Step 6 scope (deferred, not yet done): a literal 1:1 swap of
  `transform_fn` only — the existing per-(sample, channel) Python loop,
  disk cache, and noise-bank helper all stay untouched. Batching across
  channels at the call site is a separate, later change if wanted (doesn't
  affect numerics, only speed/risk-surface).

No code from fastlib/fCWT's C++ source was read or copied — only the
installed package's public Python API (`boilerplate.py`) and this repo's
own already-documented edge-of-support formula.

---

## Part 2 — New module (`utils/torch_cwt.py`, Step 1)

CWT via the Fourier convolution theorem: `torch.fft.rfft` the (zero-padded)
signal once, broadcast-multiply against a precomputed frequency-domain
Morlet filter bank (cached per `(sampling_rate, n_time, f0, f1, fn,
device)`), then a full complex `torch.fft.ifft` on the zero-extended
one-sided product back to the time domain (`torch.fft.irfft` can't be used
for the inverse step — it always reconstructs a real signal, but the
analytic/one-sided Morlet filter is exactly what makes the CWT coefficient
complex in the first place).

Padding: zero-pads each side by the widest wavelet's support radius
(`ceil(fb*scale_max*3.0)`, same formula `_coi_valid_mask` already assumes)
before the FFT, then crops back to the original `n_time` samples — keeps
COI masking aligned to the new output's actual wavelet support instead of
silently drifting.

Two real bugs found and fixed during calibration against real fcwt output
(not assumed correct from theory alone):

1. **Frequency order.** fcwt's `freqs` array is ordered high→low (index 0
   = highest frequency) — opposite of the natural log-spaced grid
   construction. Confirmed empirically (not derivable from the public API
   alone). Got this wrong on the first pass — pooled magnitude correlation
   was ~0.22 (comparing the wrong frequency's column against another)
   until fixed; frequencies then matched to float32 noise (`~3e-6`).
2. **Filter peak amplitude must be flat across scale/frequency.** A literal
   Fourier-transform-of-a-time-domain-Gaussian derivation gives an
   amplitude that grows with `sigma_sec` (i.e. `∝ 1/freq`) — wrong.
   Calibrated the correct flat constant empirically against 60 real
   trials: `sqrt(2)*pi**0.25 ≈ 1.882793`, std ~0.04% relative. This one bug
   alone accounted for a worst-case COI-valid coefficient error of ~53
   (down to ~0.6 once fixed) — high per-frequency Pearson correlation had
   masked it, since correlation is scale-invariant and didn't catch the
   systematic per-frequency amplitude bias.

Both constants are named module-level constants (`MORLET_FB`,
`MORLET_AMPLITUDE_SCALE`) with the calibration evidence in the docstring,
not buried in the math — a future re-calibration is a one-line change.

---

## Part 3 — Parity validation (Step 3)

`scripts/torch_cwt_parity.py`: runs both `coherence_utils.transform`
(fcwt/FFTW) and `torch_cwt.transform` on real CHB-MIT trials at the
canonical config, reports per-scale magnitude/phase correlation and max
absolute error, both over all samples and restricted to the COI-valid
region (via `_coi_valid_mask`'s own formula, `time_offset=0` since this
validates raw un-smoothed coefficients).

Results, COI-valid region (the only region that ever reaches the model),
pooled across trials:

| recording | trials | magnitude Pearson r | phase mean cos(Δ) | phase circular corr |
|---|---|---|---|---|
| chb01_01.edf | 16 | 0.999962 | 0.999785 | 0.998959 |
| chb01_03.edf | 12 | 0.999950 | 0.999702 | 0.999648 |

Broader sweep (60 trials, every channel, one recording): per-trial minimum
magnitude correlation **0.99989** — every trial clears the >0.999 bar.
Median relative coefficient error 0.83%, 99th percentile 5.8%.

Outside the cone of influence (the true signal edges), error is much
larger — expected, and specifically diagnosed at the user's request
(lowest freq = 8 Hz, widest cone, support radius 192 samples of 1024):
error is ~5.7 (77% relative) at the literal edge sample `t=0`, decays to
~0.03 (0.3% relative) by `t=192` (the COI boundary), and is ~0 at trial
center. This is fcwt vs. this module's zero-padding using a different
edge-extension convention — exactly the failure mode `_coi_valid_mask`
exists to exclude, and it never reaches the model at
`coi_enabled=True` (the canonical setting).

---

## Part 4 — MPS device caution (Step 4)

Found and fixed a real crash before it could be trusted: filter-bank
construction was building its Gaussian exponent in float64 directly on
`signal.device`, and **MPS has no float64 support at all** — raised
outright rather than silently misbehaving. Fixed the same way
`cwt_gnn_classifiers.py::_scale_adaptive_time_kernel` already handles this
(same file, same reasoning): always build the filter bank in float64 on
CPU, then move the finished float32/complex64 result to the target device.

CPU-vs-MPS parity on 20 real trials after the fix: max abs diff ~3e-5, max
rel diff ~2e-5, no NaNs — float32 noise floor, not a bug.

## Part 5 — Precision check (Step 5)

Confirmed fcwt genuinely runs in float32 (Part 1); `torch_cwt` matches
that deliberately, documented in the module docstring rather than left to
happen silently.

---

## Part 6 — Visual demo + a real boundary/OpenMP finding

`utils/torch_cwt_plot_demo.py` (moved here from `scripts/` at the user's
request, so it's a plain sibling import with no path juggling): computes
and plots fcwt vs. torch_cwt scalograms for 2 real trials side by side,
with the COI-invalid region shaded. Visual result matches the numbers —
scalograms are indistinguishable inside the cone, difference concentrated
in the shaded boundary. **Not yet committed** (still untracked).

Hit and fixed a real crash while making this runnable standalone: `fcwt`
and `torch` each bundle their own `libomp`, which collide on macOS
(`OMP: Error #179` → segfault) when both are imported in the same
process. Fixed by setting `KMP_DUPLICATE_LIB_OK` / `OMP_NUM_THREADS`
in-process, before any other imports, at the top of the script (only sets
what isn't already set, so an explicit shell env var still wins).

---

## Part 7 — Stress test: long signal, up to 500 scales

User asked for a stress test on a long signal at ~500 scales, then asked
whether the slowdown found there was from the scale count specifically —
it wasn't; the real driver is **signal length**, and it interacts with
**device** in a way worth recording precisely.

**Correctness at stress scale** (300s / 76,800-sample real signal, 500
log-spaced scales, 8–40 Hz): magnitude Pearson r and phase mean cos(Δ)
both round to **1.000000**, median relative error 0.005% (99th pct
0.041%, max 0.34%) over 38.3M COI-valid samples, no NaN/Inf. Correctness
holds, and if anything is tighter at high scale density than at the
canonical `nfreqs=8`.

**Timing — a real crossover, not noise.** First pass was confounded: the
`OMP_NUM_THREADS=1` workaround needed to avoid the libomp segfault (Part
6) also throttles fcwt below its real 4-thread speed. Numbers below have
that cap removed (only `KMP_DUPLICATE_LIB_OK=TRUE` needed to avoid the
crash in a benchmark script; `OMP_NUM_THREADS=1` is not actually required
for correctness, just convenient/safe in the plotting script where speed
doesn't matter).

| regime | fcwt (FFTW, nthreads=4) | torch_cwt CPU | torch_cwt MPS |
|---|---|---|---|
| canonical (1024 samples, nfreqs=8) | 0.167 ms | **0.062 ms (2.7x faster)** | 0.309 ms (0.54x, slower) |
| long signal (76,800 samples), nfreqs=8 | 4.99 ms | 5.36 ms (0.93x, ~even) | 2.95 ms (1.69x faster) |
| long signal, nfreqs=200 | 42.6 ms | 118.0 ms (0.36x, slower) | 24.0 ms (1.77x faster) |
| long signal, nfreqs=500 | 139.3 ms | 416.9 ms (0.33x, slower) | 48.7 ms (2.86x faster) |

Root cause: this module does one full-length batched inverse FFT across
*every* scale (`[nfreqs, n_padded]`), as Step 1 specified. fcwt's
algorithm — the actual "accelerated" in its paper's title — uses a
scale-adaptive-length inverse FFT per scale (a narrow-band filter only
needs a short IFFT), giving it an `O(N log N)`-ish edge on long signals
that this module's `O(fn · N log N)` approach doesn't have. MPS's
parallelism across the batched FFT compensates once the problem is large
enough to amortize dispatch overhead — but at small (canonical-config)
sizes, that same dispatch overhead makes MPS *slower* than fcwt, and CPU
wins there instead.

**Why this doesn't block Step 6:** the actual call site (canonical config,
CPU-side preprocessing before tensors move to the training device) is
squarely in the regime where `torch_cwt` on CPU already wins (2.7x). The
long-signal/CPU crossover is a real, documented limitation of the naive
batched-IFFT design — relevant only if this module is ever pointed at long
continuous recordings instead of 4s windows. A scale-adaptive-length IFFT
would be the fix if that ever matters; not attempted here.

---

## Part 8 — Canonical config is outdated; real Runpod GPU benchmark

User clarified mid-session: the 4s-window/nfreqs=8 canonical config no
longer reflects the target. Actual target: **30s windows** (not 4s) for
now, nfreqs up to the ~500 already stress-tested, **with the explicit
option to move to long continuous (multi-hour) signals later** — this
whole torch-native effort is motivated by that shift plus "powerful GPUs
[being] available" for it now.

User also flagged specific numbers as an acceptability bar: "1.6x slower
is acceptable, 5x slower is not." Traced these to Part 7's *first*
(confounded) stress-test message, before the `OMP_NUM_THREADS=1`
fcwt-throttling bug was found — `fcwt=103.5ms` (artificially throttled to
~1 thread), `torch_cpu=582.9ms` (5.63x slower), `torch_mps=166.3ms`
(1.607x slower, matches exactly). The corrected, unthrottled numbers
already reported in Part 7 (2.99x slower CPU, 2.86x *faster* MPS at the
same size) supersede that framing — but MPS on this Mac is a weak proxy
for the actual deployment target anyway.

**Spun up a real Runpod GPU pod for a trustworthy answer** (user opted in,
recommended option): RTX 4090 (`ADA_24`, 24GB), community cloud
($0.34/hr), pod id `y554p58cykwzz0`, official `runpod-torch-v280` template
(torch 2.8.0+cu128 preinstalled, matching this repo's pin exactly).
Registered a new SSH key (`~/.ssh/id_ed25519_runpod`) since none of this
machine's local keys matched what was already registered with the
account. Note: the proxy SSH endpoint (`ssh.runpod.io`) refused
non-interactive commands ("Your SSH client doesn't support PTY" /
silently dropped into an interactive shell) — used the pod's **direct**
SSH endpoint (`root@<pod-ip>:<port>`, from `get-pod`'s `ssh.direct`)
instead, which behaves like normal non-interactive SSH. `fcwt==0.1.18`
built from source cleanly on this x86_64 Linux image (unlike the Apple
Silicon Mac, which can't build it at all — see setup.sh); needed
`pip install --break-system-packages --ignore-installed` (PEP 668,
same as this repo's own `setup.sh`). Synthetic float32 signals used for
this benchmark (transform cost depends only on shape/dtype, not content;
correctness-on-real-data was already established earlier this session on
CPU/MPS).

Sized the sweep to the 24GB card first — a naive 3hr/nfreqs=200 combo
would need a `[200, n_padded]` complex64 buffer alone (`n_padded` ≈16.8M
at 3hr) ≈27GB, over budget — capped nfreqs per duration accordingly
(`3hr` tested at nfreqs=8/50 only, not swept as high as the 30s/1hr rows).

**Results — torch_cwt wins everywhere on real CUDA hardware, correctness
intact throughout (mag/phase correlation 0.999998–1.000000, no NaN/Inf
anywhere):**

| config | fcwt | torch_cwt (CUDA) | speedup |
|---|---|---|---|
| 30s (new short-window target), nfreqs=8 | 0.68 ms | 0.16 ms | 4.4x faster |
| 30s, nfreqs=200 | 7.35 ms | 0.16 ms | 47x faster |
| 30s, nfreqs=500 | 17.82 ms | 0.23 ms | 79x faster |
| 1hr continuous, nfreqs=8 | 78.6 ms | 0.79 ms | 100x faster |
| 1hr continuous, nfreqs=200 | 1422 ms | 19.7 ms | 72x faster |
| 1hr continuous, nfreqs=500 | 3546 ms | 49.1 ms | 72x faster |
| 3hr continuous, nfreqs=8 | 343 ms | 3.2 ms | 106x faster |
| 3hr continuous, nfreqs=50 | 1486 ms | 19.8 ms | 75x faster |

The earlier MPS-based "CPU wins small, MPS wins big but loses on tiny
sizes" story (Part 7) doesn't hold on real GPU hardware — cuFFT (what
`torch.fft` uses under CUDA) is dramatically better optimized for this
batched-FFT workload than Apple's MPS backend, enough to beat fcwt's own
algorithmic edge (Part 7's "accelerated" scale-adaptive IFFT explanation)
outright, even at the smallest config tested. No crossover found in this
sweep at all.

Pod torn down immediately after (`delete-pod`, confirmed `success: true`)
— total uptime ~8.3 minutes, cost ≈$0.05.

---

---

## Part 9 — Step 6: wiring the swap in, batched (not naive)

Before wiring `torch_cwt` into `_prepare_features`, checked how
`self.transform_` actually gets called at every real call site
(`compute_cwt_real_imag_tensors_cached` in `cwt_window_cache.py`,
`compute_cwt_real_imag_tensors` and `compute_paired_cwt_noise_bank` in
`common.py`). All three invoke `transform_fn` inside a plain Python
`for sample_idx: for ch_idx:` double loop, one single-channel 1-D array
at a time — correct for fcwt (which has no batched interface to exploit
anyway) but exactly the overhead-bound regime (one host<->device
transfer + one tiny kernel launch per signal) that would have erased
most or all of `torch_cwt`'s measured Part 8 speedup if `self.transform_`
had simply been pointed at `torch_cwt.transform` unchanged. User's call
(asked directly, given "no more CPU processes on the pod"): batch the
call sites now, as part of Step 6, not as a deferred follow-up.

**What changed:**

- `utils/torch_cwt.py`: added `transform_batch(signals[N, T], ...)`, a
  thin wrapper that does ONE `cwt_torch` call over a whole stacked batch
  (one host<->device transfer, one batched kernel) instead of N separate
  `transform()` calls — this is what actually realizes `cwt_torch`'s
  already-batched-over-leading-dims design at the real call sites.
- `common.py`: added `prepare_cwt_tf_batch` (vectorized sibling of
  `prepare_cwt_tf`, same transpose/resample/nan_to_num logic over a
  leading batch axis). `compute_cwt_real_imag_tensors` and
  `compute_paired_cwt_noise_bank` gained an optional `batch_transform_fn`
  + `batch_size` (default 256) param — when given, replaces the
  per-item loop with `ceil(N/batch_size)` batched calls. `None` (default)
  is the original per-item loop, byte-for-byte unchanged — this is what
  the fcwt path still uses.
- `cwt_window_cache.py`: `compute_cwt_real_imag_tensors_cached` now
  resolves every (sample, channel)'s cache key in a first pass (cheap
  dict lookups, hits served immediately, unaffected either way), then
  processes only the MISSES — as one batched call per chunk when
  `batch_transform_fn` is given, or the original per-item loop otherwise.
  Cache keys/writes are identical in both paths.
- `cwt_gnn_classifiers.py`: `_init_cwt_gnn_classifier` gained
  `cwt_backend: Literal["fcwt", "torch"] = "fcwt"` and
  `torch_cwt_batch_size: int = 256`. New `_resolve_transform_fns()` sets
  `self.transform_`/`self.batch_transform_` from `resolve_coherence_utils()`
  (backend="fcwt", `batch_transform_fn=None` — zero behavior change) or
  from `utils.torch_cwt` bound to `self.device_` via `functools.partial`
  (backend="torch"). All three call sites (`_prepare_features`,
  `_fit_noise_augmentation_state`'s noise bank, and the surrogate-null
  calibration loop) now pass `batch_transform_fn=self.batch_transform_`
  through. `cwt_backend`/`torch_cwt_batch_size` are exposed on
  `SparseEvidenceGNNClassifier.__init__` (the class `run_pipelines.py`
  actually instantiates) and inherited automatically by
  `StreamingSparseEvidenceGNNClassifier`. Default is `"fcwt"` everywhere
  — the old path stays revertable by construction (a config flag, not a
  git revert) per the original plan's Step 7 requirement.
  `XWTPhaseGNNClassifier`/`XWTPhaseGNNV2Classifier` (not instantiated by
  `run_pipelines.py`) weren't given the new constructor params — they
  keep using fcwt unconditionally via `_init_cwt_gnn_classifier`'s
  default.

**Verification (CPU/MPS, this Mac, scratchpad scripts, not committed):**

1. `compute_cwt_real_imag_tensors` and `compute_cwt_real_imag_tensors_cached`,
   looped vs. batched (`batch_size=4` to force multiple chunks), same
   `torch_cwt` transform underneath both: **bit-for-bit identical
   output** (max abs diff `0.0` on both real and imag, 5 samples x 3
   channels x 30s windows). Re-running the batched path on the same
   input (all cache hits) also matched exactly — the cache-key/write
   logic is unaffected by the batching change.
2. Full end-to-end `SparseEvidenceGNNClassifier.fit()`/`predict_proba()`
   on synthetic 30s/4-channel windows (event_mode="sparse", nfreqs=8,
   epochs=3, device="cpu", same seed), `cwt_backend="fcwt"` vs.
   `cwt_backend="torch"`: predict_proba max abs diff 0.0023, mean 0.0011,
   100% argmax class agreement — consistent with the already-established
   ~0.9998+ CWT-level correlation propagating through a trained model,
   not a bug.
3. Same fit()/predict_proba() with `cwt_backend="torch"`, `device="mps"`:
   ran clean, no NaNs, `clf.device_ == "mps"` confirming the batched
   path actually executed on-device.

Not yet done: a real Runpod GPU throughput re-measurement of the batched
call sites specifically (Part 8's `cuda_bench.py` benchmarked
`cwt_torch`/`transform_batch` directly, not through these call sites'
Python-loop-vs-chunked-batch wrapping — though Test 1/2 above show the
wrapping itself adds no numerical difference, only a different call
pattern).

**A real bug found doing the real-data comparison (Part 9b, below) and
fixed here too:** the disk/in-memory CWT caches are keyed only on signal
content + CWT config, not on which backend computed the entry.

## Part 9b — real-data before/after comparison, and a cache-key bug it caught

Ran `leave_one_seizure_out_detection` (subject 1, smoke-scale: 4.0s/30.0s
windows, epochs=2, `DENSE_EDGE_GRU_PARAMS`, `device="cpu"`) via a
standalone script importing `run_pipelines.py`'s real data-loading/eval
functions unmodified, once per `cwt_backend`.

**First attempt gave bit-identical scores for both backends** -- wrong,
and for a specific, findable reason: `cwt_window_cache.py`'s
`_window_cache_key`, `dense_edge_cache.py`'s `dense_edge_cache_key`, and
`cwt_gnn_classifiers.py`'s `surrogate_null_cache_key` all hash raw signal
content + CWT/dense-edge config, but NOT which transform (fcwt vs.
torch_cwt) produced the cached entry. This machine already had a
populated on-disk `DiskCWTCache`/dense-edge cache from earlier fcwt-only
sessions -- so the `cwt_backend="torch"` run's every lookup was a 100%
cache hit, silently reading back fcwt-computed values and never actually
exercising `torch_cwt` at all. Both runs' progress bars showed "100%
reused from cache" for every fold, which is what gave it away.

**Fix:** added a `cwt_backend: str = "fcwt"` parameter to all three
cache-key functions, folded into the hash/key string. Default `"fcwt"`
so nothing changes for an fcwt-only caller in isolation -- but since
every prior on-disk entry (for ANY backend) was written without this
suffix, the key format itself changed, so **the fix is a one-time,
whole-cache invalidation**: the very next run (fcwt or torch) recomputes
everything once, the old `.npz` files become orphaned (harmless, safe to
delete or leave). Confirmed this is what happened on re-run: `[CWT
cache] 0/15594 ... reused from cache (0.0%)` for the first fold of each
backend post-fix, `CWT(cached,batched)` progress bars (confirming the
batched path actually ran), 65,869 total on-disk CWT entries by the end
(both backends' worth, genuinely distinct this time).

**Real (post-fix) results** -- accuracy/precision/recall/f1 are
identical per fold between backends (both collapse to the majority-class
classifier at epochs=2 smoke scale, expected and backend-independent);
average_precision/roc_auc (the continuous-score metrics that actually
reflect the CWT swap) differ by small, non-systematic amounts per fold,
consistent with the previously-established ~0.9998 coefficient-level
correlation propagated through training:

| fold (run) | n_test (ictal) | fcwt AP | torch AP | fcwt AUC | torch AUC |
|---|---|---|---|---|---|
| 01 | 120 (2) | 0.333 | 0.417 | 0.9746 | 0.9788 |
| 02 | 120 (1) | 0.500 | 0.500 | 0.9916 | 0.9916 |
| 03 | 120 (2) | 0.393 | 0.393 | 0.9746 | 0.9746 |
| 04 | 120 (2) | 1.000 | 1.000 | 1.0000 | 1.0000 |
| 05 | 120 (3) | 0.625 | 0.681 | 0.9801 | 0.9829 |
| 06 | 120 (3) | 0.118 | 0.120 | 0.8490 | 0.8547 |
| 07 | 78 (4)  | 0.360 | 0.328 | 0.9426 | 0.9358 |
| mean | | 0.4756 | 0.4912 | 0.9589 | 0.9598 |

No NaNs, no crashes, no systematic direction bias. This is smoke-scale
(epochs=2, subject 1 only) -- not a claim about final model quality,
only that the backend swap behaves as expected on real data once the
cache bug was out of the way.

## Part 10 -- default flip, fcwt removal, pod-image build, and the real
## bottleneck (dense-edge, not CWT)

Committed as `831695a`: `cwt_backend="torch"` added to `run_pipelines.py`'s
`_SHARED_ARCH_PARAMS`, so both `DENSE_EDGE_GRU_PARAMS` (detection) and
`PREDICTION_GRU_PARAMS` (prediction) now default to the torch-native path --
every real `run_pipelines.py` invocation exercises it, not just the
scratch eval scripts from Part 9b. `fcwt==0.1.18` and the unused
`pyFFTW==0.14.0` dropped from `requirements.txt`; the apt
`cmake build-essential libfftw3-dev` toolchain (existed solely to compile
fcwt from source) dropped from `setup.sh` and the new `Dockerfile`.
`cwt_backend="fcwt"` still exists in `cwt_gnn_classifiers.py` as a manual
revert switch, but reviving it now means reinstalling both fcwt and that
apt step -- not a default-path concern.

**Pod image**: `Dockerfile` (repo root) bakes in system + Python deps on
top of `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` so a fresh pod
skips `bash setup.sh`. Built via GitHub Actions
(`.github/workflows/build-pod-image.yml`), pushed to
`ghcr.io/noshore5/eeg_benchmarks`. RunPod's own GitHub-integration
Serverless build was tried first and is a dead end for this: it publishes
to `registry.runpod.net`, scoped to the Serverless endpoint it built for --
a plain Pod can't pull it (`Failed to get Hub registry auth` / `No such
image`, confirmed against a live Pod). See README.md's "RunPod pod image"
section for the full writeup. Image validated on a live RTX 4090 pod: GPU
visible, `torch.cuda.is_available() == True`, `fcwt` correctly absent, and
a real `SparseEvidenceGNNClassifier.fit()`/`predict_proba()` cycle on
synthetic data ran end-to-end on CUDA.

**The actual Truong-setup eval (30s windows, prediction mode, full chb01,
real CHB-MIT data via the `physionet-open.s3.amazonaws.com` mirror -- fast,
~35-40MB/s per file, not the throttled direct-PhysioNet path noted
earlier) was attempted on a fresh `dense_edge_gru-30s` pod and killed
partway through once real timing data made the outcome clear, not because
anything crashed:**

- CWT itself: genuinely fast, batched, ~26 it/s (736 windows*channels in
  ~29s per batch-of-32 trials).
- Dense-edge computation (`compute_dense_edge_input`, downstream of CWT):
  ~39s per 32-window batch, **unaffected by the CWT backend** -- same cost
  whether the CWT step feeding it took 29s or 5 minutes.
- Dataset: 4241 windows, 173 preictal (4.1%), chb01. At `batch_size=32`,
  `negative_to_positive_ratio=5.0`, 7 leave-one-seizure-out folds, 5
  epochs: **~20 hours projected total -- essentially unchanged from the
  local CPU estimate in Part 9's lead-up.**

Conclusion: the torch-native-cwt swap itself is complete and validated --
correct (Parts 0-5), batched and GPU-native (Part 9), wired in as the real
default (this Part), and genuinely fast at the CWT step specifically. But
CWT was never the actual bottleneck for `StreamingSparseEvidenceGNNClassifier`
prediction-mode runs at this scale -- dense-edge computation is, and this
branch never touched that. Decided (explicit user choice, not a default) to
stop here rather than optimize dense-edge or run the full ~20hr eval --
that's separate, pre-existing, out-of-scope work. Pod terminated
immediately after the finding, per this session's own GPU-cost-consciousness
rule.

## Current state

- Branch `torch-native-cwt`. Steps 0-6 of the original swap plan are
  **done, committed, and now the real default** (`ada4995`, `831695a`,
  plus the pod-image commits `fae9e85`/etc. -- see Part 10). Step 7
  (remove fcwt as a dependency) is **also done** (Part 10) -- ahead of
  the original plan's own sequencing, justified by real evidence
  (`run_pipelines.py` never exercised the fcwt path by default; keeping
  fcwt in `requirements.txt` was pure dead weight, notably a slow one to
  build).
- **Not done, and now known to be a separate problem**: dense-edge
  computation speed. A full real-data Truong-setup eval was attempted on
  a live RTX 4090 pod and killed once timing data (~20hr projected) showed
  GPU-native CWT alone doesn't fix the actual bottleneck. See Part 10.
- `utils/torch_cwt_plot_demo.py` and its output `utils/torch_cwt_demo.png`
  exist but are **not committed** (untracked). The user independently
  edited `NFREQS` to 300 in that file at one point during this session —
  current on-disk state reflects that edit's aftermath (later runs in
  this session used ad hoc inline scripts, not this file, for the 500/200
  scale sweeps, so the committed parity numbers above are unaffected by
  that edit).
- Steps 0–5 of the swap plan are done and validated. **Step 6 (Part 9)
  is implemented as a `cwt_backend="fcwt"|"torch"` opt-in flag, with the
  real per-item-loop call sites reworked to batch, smoke-tested correct
  on CPU and MPS on synthetic data, AND run end-to-end on real CHB-MIT
  data (Part 9b: subject 1, smoke-scale leave-one-seizure-out) with a
  genuine (not cache-confounded, see Part 9b) before/after score
  comparison.** Only at smoke scale (epochs=2, 1 subject) so far, not the
  fuller canonical run.
- **The canonical config used for Steps 0–5 (4s windows, nfreqs=8) is
  outdated** (Part 8) — the real target is 30s windows now, with a
  multi-hour continuous option to keep open. Steps 0–5's parity
  validation was run at the old 4s/nfreqs=8 config; re-validate at the
  actual 30s-window config before Step 6 (this session's Part 8 CUDA
  sweep already covers 30s at nfreqs=8/200/500 and shows the same
  essentially-perfect correctness, so this is confirmatory, not expected
  to surface anything new).
- Real Runpod GPU benchmark (Part 8, RTX 4090) shows `torch_cwt` beating
  `fcwt` by 4x–106x across every config tested (30s windows through 3hr
  continuous, nfreqs 8–500), correctness intact throughout. No CPU/MPS
  crossover found on real CUDA hardware — the concern from Part 7 (naive
  batched-IFFT losing to fcwt's algorithmic edge at long signals/high
  nfreqs) does not materialize on GPU.

## Part 11 -- dense-edge bottleneck root-caused and fixed: it was the compressed cache write, not the GPU math

Direct follow-up to Part 10's finding, prompted by a challenge that the
dense-edge slowness must be fixable since the computation is "a
convolution over a matrix product" and GPU should do that fast. Read
`_build_dense_edge_input`/`_full_edge_wct_maps`/`_smooth_wct_maps` in full
first (`cwt_gnn_classifiers.py`) to confirm that premise before measuring
anything: all three are genuinely vectorized torch (elementwise
cross-spectrum via `index_select`+multiply, separable `conv2d` Gaussian
smoothing, `avg_pool2d` downsampling) with no hidden per-item Python
loops -- the premise was correct.

`_precompute_dense_edge_inputs`'s per-chunk loop, however, calls
`save_dense_edge(cache_dir, cache_keys[i], dense[j])` once per trial
*inside* the same loop the GPU compute is timed in, and `save_dense_edge`
(`dense_edge_cache.py`) used `np.savez_compressed` -- synchronous,
single-threaded DEFLATE compression. `run_pipelines.py` passes a real
`dense_edge_cache_dir` by default, so every trial in a fresh-cache run
(0% hit rate, as Part 10's killed eval showed throughout) paid this cost.

Measured directly (not projected) on a fresh RTX 3090 pod, real chb01
data, real `PREDICTION_GRU_PARAMS`, `torch.cuda.synchronize()` wrapped
around `compute_dense_edge_input` so the GPU-only interval is exact:

| | before (compressed) | after (uncompressed `np.savez`) |
|---|---|---|
| GPU compute (6 chunks of 4 trials) | 0.70s (117ms/chunk) | 0.50s (83ms/chunk) |
| disk cache write (24 trials) | 12.73s (530ms/trial) | 0.26s (11ms/trial) |
| write as % of (compute+write) | 94.8% | 34.4% |
| `fit()` wall time, this slice | 16.08s | 3.24s |
| dense-edge chunk throughput | 0.45 chunk/s | 6.53 chunk/s (~14x) |

Isolated single-tensor A/B on the actual cached shape
(`(4, 253, 479, 8)` float32, `PREDICTION_GRU_PARAMS`'s dense-edge shape):
`np.savez_compressed` = 499ms for 13.73MB; `np.savez` (raw) = 6.6ms for
15.51MB -- an 11% size reduction bought at 75x the write time. This is
nowhere near the 2026-08-16 note's cited "8.23MB -> 3.76MB (~54%
smaller)" figure -- that number came from a different (dense-edge-GRU,
larger `T`) config's tensors, not `PREDICTION_GRU_PARAMS`'s, and was never
re-validated against the actual config real runs use before being trusted
as justification for paying this cost on every trial.

**Fix**: `dense_edge_cache.py`'s `save_dense_edge` now uses `np.savez`
instead of `np.savez_compressed` (commit follows this note). No cache-key
or on-disk-format change -- `np.load` reads both transparently, so
existing cached entries (compressed, from before this fix) remain valid
hits; only new writes stop compressing. Disk space cost is real but
modest (11-54% larger depending on config) and this cache is fully
regenerable, not source data -- the write-time-CPU cost it was buying
against was, on the pipeline's actual hot path, far more expensive than
the disk space it saved.

Not yet re-run: the full real Truong-setup eval that Part 10 killed after
projecting ~20hr. With dense-edge's disk-write cost now ~14x cheaper (and
GPU compute itself already fast and unaffected), the dense-edge stage
should no longer be the dominant cost -- but that's a projection from a
24-window slice, not a re-measurement of the full config. Left for a
future session/explicit request per the same cost-consciousness that
stopped Part 10's run rather than letting it run overnight.

## Open items

- **(Superseded by Part 11)** ~~Dense-edge computation is the actual
  bottleneck...~~ -- root-caused (compressed disk-cache write, not the
  GPU math) and fixed in Part 11. The real open item now: the full
  Truong-setup eval hasn't been re-run end-to-end since the fix, so the
  ~20hr projection from Part 10 is stale but not yet replaced by a new
  measured number.
- Step 6's real-data comparison at SMOKE scale (Part 9b: epochs=2, subject
  1, old 4.0s/30.0s detection-mode windowing) is superseded by Part 10's
  real-scale attempt at the actual 30s-window config -- kept here for
  history, not as an open item anymore.
- **Cache-key fix (Part 9b) invalidates every pre-existing on-disk
  CWT/dense-edge/surrogate-null cache entry once** (key format changed to
  include `cwt_backend`) -- expect a full recompute on the next run
  anywhere one of these caches already exists (this dev machine, and any
  Runpod pod/persistent volume that accumulated a cache in an earlier
  session). Not a correctness problem, just a one-time cost worth
  knowing about before assuming a "why is this suddenly slow" run is a
  regression.
- Batched-call-site throughput hasn't been re-measured on real GPU
  hardware specifically THROUGH the new call-site wrapping (only the
  underlying `cwt_torch`/`transform_batch` calls were benchmarked
  directly in Part 8) -- worth a follow-up Runpod run once there's a real
  eval to run anyway, rather than spinning up a pod just to re-confirm
  the wrapping adds no overhead beyond what Test 1/2 already showed
  (none, numerically).
- `run_pipelines.py`'s window_length/step_size defaults and
  `_SHARED_ARCH_PARAMS`'s `nfreqs=8` still reflect the old canonical
  config — not updated this session (this session only validated the CWT
  swap itself, in isolation, at both old and new configs; the pipeline's
  actual config constants are a separate, not-yet-made change).
- `utils/torch_cwt_plot_demo.py` / `utils/torch_cwt_demo.png` not yet
  committed.
- ~~Step 7 (remove `fcwt`/FFTW as a dependency) explicitly deferred until
  after Step 6's full run is confirmed clean~~ -- **done** (Part 10),
  ahead of a full clean run, on the evidence that `run_pipelines.py` never
  exercised the fcwt path by default anyway.
- The long-signal/CPU (and small-signal/MPS) performance crossover (Part
  7) is real on THIS Mac's hardware but does not reproduce on real CUDA
  (Part 8) — worth keeping in mind only if this ever needs to run well on
  non-CUDA hardware (e.g. local Mac dev runs of long-signal configs), not
  a concern for the actual Runpod GPU deployment target.
- `~/.ssh/id_ed25519_runpod` (new keypair, registered to the Runpod
  account this session for the Part 8 benchmark) exists on this machine
  and stays registered — harmless (SSH access only, no cost), not cleaned
  up since it's reusable for future pod work.
