# Session notes — disk-backed caching, epoch/window tuning, first real 7-fold result (2026-08-16)

Branch: `main`. Repo: `/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`.

Follow-on to [2026-08-15's session](../2026_08_15/chb_mit_dataset_and_dense_edge_gru_pipeline.md),
which left two open items: the CWT cache measured net-negative at smoke
scale, and no real (non-`--smoke`) leave-one-seizure-out run had been
scored. Both resolved today — plus the first real result, which landed a
lot stronger than the smoke numbers suggested.

---

## Part 1 — Disk-backed caching for both CWT and dense-edge stages

Picked back up from yesterday's open question ("cache dense-edge inputs
too, or leave them uncached?"). Went through several iterations before
landing on the final shape:

1. **Investigated whether dense-edge caching is exact**, same style as
   yesterday's CWT proof: with `coherence_threshold_mode="fixed"`,
   `_precompute_dense_edge_inputs`
   ([sparse_evidence_gnn_classifier.py:4091](../../pipelines/sparse_evidence_gnn_classifier.py#L4091))
   runs on nothing but `w_real`/`w_imag`/`freqs` — no surrogate/cluster
   branch, no `raw_x_native` dependency. Coherence and phase are both
   exactly invariant to the CWT cache's `1/std` rescale, so the dense-edge
   output for a given physical window is identical regardless of which
   fold's normalization produced the CWT tensors feeding it. This only
   holds for `"fixed"` mode — `"surrogate"`/`"surrogate_cluster"` calibrate
   against `raw_x_native` directly and are excluded from the cache.
2. **First attempt (in-memory dict, like the CWT cache) was rejected** —
   asked before building it, given the memory math: `[4, 253, T, F]`
   float32 per trial (253 edges = `C(23,2)`, all-pairs for CHB-MIT's
   23-channel montage) is much bigger than a per-channel CWT entry, and
   caching every unique window in RAM alongside the CWT cache risked ~6GB+
   on the 16GB machine. User course-corrected mid-build ("can it just cache
   the CWTs, and build the WCT on demand... I thought that was the original
   plan?") — reverted to the original plan (CWT cached, dense-edge
   recomputed fresh every call), no new cache built at that point.
3. **Visibility fix instead**: `_precompute_dense_edge_inputs`'s progress
   bar was gated to `surrogate`/`surrogate_cluster` modes only
   (`show_progress = (surrogate_mode or cluster_mode) and self.verbose >= 1`),
   silent under `"fixed"` — which is exactly why this stage's cost went
   unnoticed during yesterday's CWT-caching investigation. One-line fix:
   `show_progress = self.verbose >= 1`. This is what surfaced, live, that
   dense-edge precompute really is the dominant per-fold cost (a
   `dense-edges[fixed]` bar chugging at ~1s/chunk while CWT sat at 100%
   cache hits).
4. **Re-requested after seeing it live**: watching a real 100%-CWT-cache-hit
   run still "take forever" on the now-visible dense-edge bar prompted
   "let's build both caches and keep them on disk" — this time disk-backed,
   sidestepping the RAM objection from step 2 entirely (disk, not memory,
   holds the working set).

**Built:**
- [cwt_window_cache.py](../../pipelines/cwt_window_cache.py)'s `DiskCWTCache` —
  drop-in for the plain `dict` `compute_cwt_real_imag_tensors_cached`
  already expected (`.get`/`__setitem__` only), backed by one `.npz` file
  per key under `<mne_data>/cwt_window_cache/`, atomic write (temp +
  `os.replace`), in-memory dict in front so repeat lookups within one
  process skip the filesystem.
- [dense_edge_cache.py](../../pipelines/dense_edge_cache.py) (new) —
  `dense_edge_cache_key()` (SHA256 of the *whole* raw trial, all channels —
  dense-edge output mixes channel pairs, no per-channel decomposition to
  key on separately, unlike CWT) + `load_dense_edge`/`save_dense_edge`,
  same atomic-write convention, under `<mne_data>/dense_edge_cache/`.
- Wired into `SparseEvidenceGNNClassifier` via a new `dense_edge_cache_dir`
  param (`None` default = old behavior unchanged) and a rewritten
  `_precompute_dense_edge_inputs` that splits trials into cache
  hits/misses up front, only chunks the misses through
  `compute_dense_edge_input`, and writes each new result back — restricted
  to `mode_label == "fixed"`.
- [run_pipelines.py](../../run_pipelines.py)'s `leave_one_seizure_out` now
  builds a `DiskCWTCache` + resolves `default_dense_edge_cache_root()` once
  and passes both to every fold's classifier instance.

**Verified exact, not just fast**: a fresh classifier instance, different
seed, different fold's normalization stats, reading a dense-edge cache
entry another instance wrote — **bit-identical** result (0.0 max abs/rel
diff), not just float32-noise-level agreement.

**Verified live**: smoke rerun (798 windows), all 7 folds, fold 1 cold
(0% hits, writes everything), **folds 2-7: 100% hit rate on both caches**,
`dense-edges[fixed]` showing `0chunk [00:00]` — zero work once warm.

---

## Part 2 — Disk budget: it didn't fit, so compression + resolution cuts

Scaling the smoke run's cache size up to the real config
(`step_size=4.0` → 5,981 windows vs. smoke's 798, confirmed by actually
running the paradigm, not estimated) projected **~62GB** (46GB dense-edge +
16GB CWT) against **~55GB free disk** at the time. Root-caused the exact
numbers by inspecting real cached files rather than guessing:

- dense-edge tensor: `[4, 253, 127, 16]` float32 = 8.23MB/window, **40.5%
  exactly zero** (COI-masked).
- CWT entry: `(1024, 16)` complex = 0.131MB/(window, channel).

Also checked whether the smoke cache was any real head start for a real
run: only **399/5,981 (6.7%)** of real-run windows are byte-identical to
something smoke cached (`step=30s` and `step=4s` grids only coincide every
`lcm(30,4)=60s` of file-relative time) — confirmed by hashing actual window
content, not estimated. Not a meaningful warm start.

**Fix — three levers, chosen together, each measured on real files:**

| lever | change | effect (dense-edge) | effect (CWT) |
|---|---|---|---|
| compression | `.npy`/plain `.npz` → `np.savez_compressed` | 54% smaller (COI zeros compress well) | 6% smaller (barely sparse) |
| `nfreqs` | 16 → 8 | 2x smaller | 2x smaller |
| `dense_edge_time_downsample` | 8 → 16 | 2x smaller (dense-edge only) | unaffected |

Combined: dense-edge **8.23MB → 0.91MB/window (9x)**, CWT **0.131MB →
0.062MB/entry (2.1x)**. Real-run projection dropped from ~62GB to **~14GB**
— comfortably inside the ~55-69GB free (freed up further by deleting the
now-orphaned old-config smoke cache, unreachable anyway since cache keys
include `nfreqs`/`dense_edge_time_downsample`).

Re-verified with another smoke rerun: new shapes `[4, 253, 63, 8]` /
`(1024, 8)`, all 7 folds clean, 100% hits folds 2-7, **1.7GB total** cache
for 798 windows (matches the ~14GB/5,981-window projection), epoch time
dropped ~11-18s → ~6-7s (smaller tensors help training too, not just
caching).

---

## Part 3 — First real (non-`--smoke`) result

With caching/disk sorted, ran for real: `step_size=4.0s` (original
default), `epochs=30` (original default), all 7 folds, no `--smoke`.
`X: (5981, 23, 1024)`, 115/5981 (1.9%) ictal windows.

**Fold 1 (`run=00`) revealed a real overfitting signature**: training loss
hit *exactly* `0.000000` by epoch 24 and stayed there through epoch 30 —
not a plateau, a literal flatline (`acc`/`roc_auc` pinned at 1.0/1.0 too).
Epochs 24-30 (~6 minutes) did nothing. This is training-set signal only
(`validation_split=0.0` — deliberate, each fold's training set is already
small) so it doesn't directly say where held-out performance peaks, but
the held-out result for that fold (`precision=1.0 recall=0.9 f1=0.947
auc_pr=1.0`, `n_test_ictal=10`) suggests whatever mattered was captured
well before the flatline (`roc_auc` hit ~1.0 by epoch ~19-20).

**Investigated whether device is a lever** (a separate question from
epoch count — could training itself go faster per-epoch, not just for
fewer epochs): this machine is Apple Silicon (arm64) with MPS available
and built (confirmed via `torch.backends.mps.is_available()`), but the
training loop's `device="auto"` (`resolve_torch_device`, `common.py:141`)
only ever checks CUDA, silently falling back to CPU. Not a bug —
`resolve_best_available_device` (used only for the surrogate-calibration
path, not the trainable model) has an existing docstring explicitly
documenting *why* the training loop's default stays CPU: small model
(`hidden_dim=8`), GRU's per-timestep ops could be dominated by MPS
kernel-launch overhead — but flags this was never actually measured for
this exact dense+GRU config. Also noted `torch.get_num_threads()=4` vs. 10
CPU cores available. **Neither lever was tested** — the live run was
already at ~98% CPU and testing alongside it would have both slowed it
down and produced contaminated timing numbers. Left as an open item.

**Config changed instead** (the tested, lower-risk lever) — both
`run_pipelines.py` argparse defaults, based on fold 1's evidence:
- `--epochs`: 30 → **20** (leaves margin before the observed flatline,
  not validated against every fold).
- `--step-size`: 4.0 → **8.0** (halves window count → roughly halves both
  cache size and training-set size, hence per-epoch cost too — a real
  data-density tradeoff, not free: fewer windows means fewer of the
  already-scarce ictal-labeled windows too).

**Raised but deliberately left unresolved**: `window_length` stayed at
`4.0` while `step_size` moved to `8.0`, which means non-overlapping windows
*with gaps* — roughly half the raw signal (the `[4,8)`-style stretches
between windows) is never in any window at all. Checked whether this risks
missing a seizure entirely — it doesn't, chb01's seizures run 27-101s, far
longer than a 4s gap — but it's still discarded signal and coarser
onset/offset localization. Setting `window_length=8.0` to match would
close the gap but roughly double the per-window CWT/dense-edge cost again
(raw samples per window doubles), largely offsetting the `step_size` win.
Flagged as a real, unresolved tradeoff — not changed.

**The user restarted the run under the new config** (`step_size=8.0`,
`epochs=20`) themselves; while this note was being written, it finished:

| run | n_train | n_test | n_test_ictal | accuracy | precision | recall | f1 | avg_precision | roc_auc |
|---|---|---|---|---|---|---|---|---|---|
| 01 | 2541 | 450 | 5 | 0.998 | 1.000 | 0.800 | 0.889 | 0.820 | 0.980 |
| 02 | 2541 | 450 | 4 | 0.998 | 1.000 | 0.750 | 0.857 | 0.800 | 0.991 |
| 03 | 2541 | 450 | 5 | 0.996 | 0.714 | 1.000 | 0.833 | 0.753 | 0.998 |
| 04 | 2541 | 450 | 7 | 0.996 | 0.857 | 0.857 | 0.857 | 0.843 | 0.999 |
| 05 | 2541 | 450 | 12 | 0.998 | 1.000 | 0.917 | 0.957 | 1.000 | 1.000 |
| 06 | 2541 | 450 | 12 | 0.996 | 0.857 | 1.000 | 0.923 | 0.988 | 1.000 |
| 07 | 2700 | 291 | 13 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **mean** | | | | **0.997** | **0.918** | **0.903** | **0.902** | **0.886** | **0.995** |

Total wall-clock: process start 11:26am → results written 1:07pm, **~1h42m**
for all 7 folds (down from the >3h the old `epochs=30`/`step_size=4.0`
config's fold-1-alone timing implied).

Caveat worth keeping attached to these numbers, same shape as the memorization
observation above: every fold hits exact/near-exact training-set memorization
by this point in training (same flatline pattern seen in fold 1's log), and
per-fold `n_test_ictal` is small (4-13 windows) — strong numbers, but on thin
denominators, and this is one subject's leave-one-seizure-out, not a
validated generalization claim yet.

Results: [leave_one_seizure_out_20260816-130740.csv](../../results/leave_one_seizure_out_20260816-130740.csv)

---

## Current state

- Both caches disk-backed, compressed, verified exact and fast in practice
  ([cwt_window_cache.py](../../pipelines/cwt_window_cache.py),
  [dense_edge_cache.py](../../pipelines/dense_edge_cache.py)).
- [run_pipelines.py](../../run_pipelines.py) defaults: `step_size=8.0`,
  `epochs=20`, `window_length=4.0` (unchanged, now creating signal gaps —
  see Part 3).
- First real 7-fold leave-one-seizure-out result exists and is strong
  (mean F1 0.902, ROC-AUC 0.995) — but on one subject, with the
  memorization/thin-denominator caveats above.

## Open items

- `window_length`/`step_size` gap (`4.0`/`8.0`) — real signal being
  discarded, not closed.
- Device: MPS available on this machine but untested for the training
  loop; `torch.set_num_threads` also untested (4 of 10 cores). Neither
  applied — no measurement yet, just identified.
- Train-loss-based early stopping proposed (in place of the guessed
  `epochs=20`) but not built.
- Frequency band (1-40Hz) still explicitly untuned, carried over from
  2026-08-15.
- Only subject 1, only chb01 — same deliberate scope limit as yesterday.
