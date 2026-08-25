# Repo context (read this first)

Living status doc, not a log. Says what's true *right now* -- for the
history/reasoning behind any of it, follow the pointers into
`Epilepsy/Session_notes/<date>/`. Update this file (not just the session
note) before ending a working session that changed the state below --
that's the whole point of it: this repo gets worked on from different
Claude/Grok shells that don't share context with each other, and this is
the one file meant to catch a new shell up without it re-reading
everything.

**Last updated:** 2026-08-25, by Claude (continuous-cwt-mamba work below).

---

## Right now

GRU vs Mamba (encoder-free, full 23-channel mesh, val split 0.2, early
stop, matched protocol) across all 6 chb01 leave-one-seizure-out folds —
**both runs finished, comparison written up.** Mamba wins on AP (+0.076,
0.499 vs 0.423) and both FAR/h numbers, loses on recall/f1 (driven mostly
by a recall collapse on one fold, `1_18_0`), same 5/6 k-of-n hit rate as
GRU but a different seizure missed (`1_18_0` for Mamba vs `1_26_0` for
GRU — see gotcha below, the "persistent miss" turned out to be
GRU-specific, not universal). Mamba costs ~4.6x the wall-clock/epoch on
this hardware (`mambapy` pure-PyTorch scan, no CUDA kernel). Full tables
in `Epilepsy/Session_notes/2026_08_25/
full_6fold_23ch_encoderfree_val_gru.md`. Also resolved there: the earlier
`smoke_test.py`-based Mamba run's exact-0.0-val_loss pattern did not
reproduce in this matched run (0 occurrences across all 6 folds) — looks
like an artifact of `smoke_test.py`'s small capped train/val split, not a
real Mamba training-path issue.

**`continuous-cwt-mamba` branch, separate thread from the GRU/Mamba
comparison above:** a new paradigm, not a tuning change to
`_DenseEdgeMambaTemporal`. Goal: CWT the whole recording once (not
arbitrary 30s windows), let Mamba's SSM state run continuously across the
ENTIRE timeline with no per-window reset, and only window at the
classification readout, at the end. Motivation: the existing
`_DenseEdgeMambaTemporal` resets state every window (each call to
`Mamba.forward()`'s parallel scan always starts h=0) -- that throws away
exactly the cross-window context Mamba is supposed to be good at.

Built and verified so far, all in `Epilepsy/pipelines/cwt_gnn_classifiers.py`:
- `_DenseEdgeMambaContinuous` -- drives `mambapy`'s OTHER scan entry point,
  `Mamba.step(x, cache)` (built upstream for autoregressive generation: one
  timestep at a time, explicit `(h, inputs)` cache), instead of
  `_DenseEdgeMambaTemporal`'s `Mamba.forward()` (parallel scan, always
  h=0, no way to feed in a starting state). Loop `step()` across a chunk,
  carry the returned (`.detach()`'d) cache into the next chunk -- classic
  truncated BPTT: forward state chain is unbroken across an entire
  recording, only the backward graph is cut at chunk boundaries.
- `pool_continuous_edge_stream_to_windows` -- turns a concatenated
  continuous `[B,C,E,T_total]` output stream into per-window
  `[B,C,E,1]` snapshots (pool="last", the continuous analogue of
  `_DenseEdgeMambaTemporal`'s own "pool to h_T" convention) by pure
  indexing, no model weights.
- `_dense_edge_features` split into `_dense_edge_conv_out` +
  `_dense_edge_features_from_conv_out` (behavior-preserving refactor) so
  the pooled continuous-stream windows feed into the EXISTING
  `sparse_message_mlp`/hop-propagation/`sparse_classifier` readout
  unmodified -- confirmed working end-to-end (unbound-method test, see
  `scripts/dense_edge_mamba_continuous_parity.py`'s sibling checks in
  session history; not yet re-saved as a standalone script).
- **Proven, not just plumbed:** `scripts/continuous_mamba_state_carryover_check.py`
  -- synthetic task where a signal is injected only into a recording's
  first chunk and every later window's label depends on it, so a
  state-reset-every-chunk model has literally zero information for those
  windows (fresh noise every step, never repeated -- memorization is
  impossible) while a carried-state model can solve it. Result: carried
  state hits 0.922 (5-epoch-rolling-avg accuracy on the signal-free
  windows), reset stays at 0.573 (chance = 0.5). This is the actual
  evidence the mechanism buys something real, not just "it runs."
- `scripts/dense_edge_mamba_continuous_parity.py` -- chunking is
  bit-exact vs. one big call; `step()`-driven output matches `mambapy`'s
  own `forward()` scan to float32 noise (~6e-8); a chunk's `backward()`
  genuinely doesn't reach into the previous chunk's freed graph.
- `scripts/continuous_mamba_gpu_scale_probe.py` -- one-shot real-scale
  (23ch full mesh, E=253, C_in=192, real defaults d_model=16/d_state=16/
  expand=2) fwd+bwd probe on this session's RTX 3070 Ti. **Memory turns
  out NOT to be the binding constraint**: linear at ~2.4 MiB/timestep,
  T_chunk=1024 -> 2.48GiB, comfortably fits. **Wall-clock time is the
  real open problem**: ~2.68ms/timestep (B=1) -- a genuine per-timestep
  Python-loop/kernel-launch cost with no batching-across-time the way a
  parallel scan gets for free. Scaled to a real ~1hr recording (tens of
  thousands of timesteps even downsampled) that's roughly a couple
  minutes of Mamba compute ALONE per recording per epoch -- likely
  slower than even the already-slow (~65s/epoch, 4.6x GRU) windowed
  `mambapy` scan this file's "Known gotchas" section documents. Not
  measured yet: whether batching multiple recordings' rows together
  (same idea as the existing `mamba_chunk_size` row-chunking) amortizes
  enough of the per-step launch overhead to matter.

**Not started:** the actual CHB-MIT data pipeline / LOSO loop rewrite --
everything above was deliberately built and verified in isolation
(synthetic tensors) before touching `_build_windowed_dataset`/
`leave_one_seizure_out_*` in `run_pipelines.py`, since that's the highest-
risk, hardest-to-verify part (recording-level continuous sequences instead
of independent per-window rows, TBPTT chunk boundaries, windowed labels
read off a continuous timeline). See "Open threads" below.

## Branch map

- `main` -- baseline, no dense-edge-mamba, no cwt node encoder.
- `mamba-temporal-edge-model` (remote, `origin/`) -- where
  `dense_edge_temporal_mode="mamba"` (`_DenseEdgeMambaTemporal`) was
  developed. Merged into `main` at `6d38573`.
- `continuous-cwt-mamba` -- **current branch**. Has `main` +
  `mamba-temporal-edge-model` both merged in (so `dense_edge_mamba` IS
  available here, HEAD has 11 references to `_DenseEdgeMambaTemporal`),
  plus a `--skip-folds` CLI flag for resuming partial multi-fold runs, plus
  the uncommitted continuous/streaming Mamba work described above.
- `tf-node-encoding`, `dynmaic_subset` -- exist locally, not investigated
  this session; don't assume they have dense-edge-mamba unless you check.
- If you're starting a session and need `dense_edge_mamba`: check you're
  on (or have merged) `continuous-cwt-mamba` or `mamba-temporal-edge-model`
  first. `git log --oneline -- Epilepsy/pipelines/cwt_gnn_classifiers.py`
  or `grep _DenseEdgeMambaTemporal` is a fast way to confirm rather than
  assuming from the branch name alone.

## Known gotchas (keep rediscovering these -- stop rediscovering them)

- **No `mamba-ssm` (CUDA kernel) on Windows.** PyPI ships sdist-only, needs
  nvcc + Linux toolchain to build. This repo uses `mambapy` (pure PyTorch,
  `requirements.txt`) instead, which runs everywhere but is much slower
  per epoch. On a Linux/CUDA box (e.g. RunPod), `mambapy`'s own
  `MambaConfig.use_cuda=True` will delegate to real `mamba-ssm` if it's
  importable -- currently forced `False` in this repo's code
  (`_DenseEdgeMambaTemporal.__init__`). See
  `Epilepsy/runpod_mamba_fast_image_brief.md` for the build+code-change
  plan to actually flip this on.
- **That `use_cuda=True` path is NOT compatible with (b)float16** per
  `mambapy`'s own docs -- this repo's matched comparison runs default to
  `--dense-edge-amp-bf16 --train-amp-bf16`. Don't mix them without
  checking; see the brief above.
- **`smoke_test.py`'s epoch_time is not a full-run predictor**, especially
  for the mamba backend. It defaults to capped data
  (`max_interictal_recordings=5`) and `mamba_chunk_size` overhead scales
  nonlinearly with batch composition -- measured smoke-scale Mamba/GRU
  slowdown ~14x vs. real full-mesh/uncapped ~4.6x, same code. To estimate
  a real run's epoch_time, run `run_pipelines.py` itself with `--max-folds
  1 --epochs 1` against real (uncapped) params, not `smoke_test.py`.
  (Caveat now also printed in `smoke_test.py`'s own output.)
- **Disk cache is off by default on CUDA** (`resolve_disable_disk_cache`
  in `run_pipelines.py`) -- every batch recomputes CWT/dense-edge features
  from scratch. Expected, not a bug; explains the repeated
  `CWT(batched,gpu-resident)` / `dense-edges[fixed]` progress-bar blocks in
  every log, one set per batch.
- **`--max-folds` only truncates from the front** of the leave-one-seizure-
  out loop; there's no flag to skip an already-done fold 1 and resume at
  fold 2 -- use the new `--skip-folds` flag (added on this branch,
  `continuous-cwt-mamba`) for that instead, or just rerun fold 1 (it's
  deterministic, seed 42, so a rerun is a reproducibility check more than
  wasted work).
- **`run_pipelines.py` defaults to `--device mps`** (author's Mac). Always
  pass `--device cuda` explicitly on a CUDA box.
- **chb01 is the only subject exercised by default anywhere in this repo**
  (`DEFAULT_SUBJECTS=[1]`). It's baked into the RunPod `Dockerfile` and
  redistributed via a GitHub Release (see `README.md`) specifically
  because PhysioNet's own server throttles hard. Other subjects still pull
  from PhysioNet's S3 mirror on demand, not baked into any image.
- **`1_26_0`** (a chb01 seizure) is a persistent k-of-n miss for every
  *GRU*/encoder-in-graph variant tried so far (encoder-in-graph,
  encoder-free, k=4 through full mesh) -- but the matched Mamba run
  (2026-08-25) hit it; Mamba's own miss was `1_18_0` instead. So "misses
  1_26_0" is GRU-specific, not universal -- don't assume it generalizes to
  a new backbone without checking.
- **chb01 has exactly 7 real seizures**: `1_03_0, 1_04_0, 1_15_0, 1_16_0,
  1_18_0, 1_21_0, 1_26_0` (confirmed from `chb01-summary.txt`, not from
  any run's output). `chb01_02.edf` has 0 seizures -- a seizure ID
  `1_02_0` appearing anywhere (e.g. an old `smoke_test.py`-driven run
  note) is a labeling bug, not a real held-out seizure. Today's 6-fold
  runs (GRU and Mamba both) cover 6/7, missing `1_21_0` specifically --
  it produces zero surviving preictal windows at SPH=300/SOP=900 (onset
  too close to its own recording's start), so it never enters
  `leave_one_seizure_out_prediction`'s fold list at all.

## Open threads

- ~~Mamba 6-fold run~~ -- done, see "Right now" and the session note.
- Mamba's `1_18_0` recall collapse (0.100 vs GRU's 0.767, same seizure) --
  not investigated, biggest single per-seizure divergence in the
  comparison.
- `Epilepsy/runpod_mamba_fast_image_brief.md` -- unexecuted brief for a
  fast RunPod image (real `mamba-ssm` kernel + baked dataset). Written,
  not built.
- Channel-subset-k sweep (`Epilepsy/Session_notes/2026_08_24/
  k_sweep_channel_subset_cuda.md`) still has the `ChannelSignalEncoder` in
  the graph (24-in MLP) -- not a clean same-model ablation against the
  encoder-free full-mesh runs. Needs an encoder-free rerun to close that
  comparison out.
- `continuous-cwt-mamba` paradigm (see "Right now" above) -- component
  pieces built and verified in isolation, real data pipeline not started.
  Next concrete steps, in the order they were being approached:
  1. Investigate the throughput problem before investing in the data
     pipeline around it: ~2.68ms/timestep (B=1) is slow enough at
     real-recording scale that it may need addressing first (row-batching
     multiple recordings together, or something else) rather than after.
  2. Design + build the continuous CHB-MIT loading path: whole-recording
     CWT (not `_build_windowed_dataset`'s fixed windows), TBPTT chunk
     boundaries, and windowed labels (SPH/SOP-derived) read off the
     continuous timeline via `pool_continuous_edge_stream_to_windows`.
  3. A parallel `leave_one_seizure_out_*`-equivalent loop in
     `run_pipelines.py` for recording-level continuous sequences instead
     of independent per-window rows -- the existing LOSO functions assume
     per-window rows in `metadata`/`X` throughout, not a bolt-on.
  4. Only then: a real GPU LOSO run to compare against the
     `_DenseEdgeMambaTemporal` (windowed) baseline above.
