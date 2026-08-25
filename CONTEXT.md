# Repo context (read this first)

Living status doc, not a log. Says what's true *right now* -- for the
history/reasoning behind any of it, follow the pointers into
`Epilepsy/Session_notes/<date>/`. Update this file (not just the session
note) before ending a working session that changed the state below --
that's the whole point of it: this repo gets worked on from different
Claude/Grok shells that don't share context with each other, and this is
the one file meant to catch a new shell up without it re-reading
everything.

**Last updated:** 2026-08-25, by Claude.

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

**Heads up, found mid-session:** there is uncommitted work in the working
tree on `continuous-cwt-mamba` (current branch) that this session did not
write -- 292 lines in `Epilepsy/pipelines/cwt_gnn_classifiers.py` adding a
new `_DenseEdgeMambaContinuous` class (streaming/TBPTT variant using
`.step()` + carried cache, distinct from the batch `_DenseEdgeMambaTemporal`
used in the comparison above), plus an untracked
`scripts/dense_edge_mamba_continuous_parity.py`. Looks like another
shell's in-flight work. Not committed, not evaluated by this session --
if you're picking this branch up, check `git status`/`git diff` before
assuming this file matches what's on disk.

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
- The uncommitted `_DenseEdgeMambaContinuous` work described under "Right
  now" -- status/intent unknown to this session, check with whoever
  (agent or you) was last working on `continuous-cwt-mamba` before
  touching it.
