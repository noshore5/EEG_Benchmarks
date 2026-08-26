# `mamba-temporal-edge-model` branch: what it was, and why it's archived

Housekeeping note: the `EEG_Benchmarks_mamba` worktree and the
`mamba-temporal-edge-model` branch behind it were removed 2026-08-26. This
records what was in there before cleanup, since the branch itself is gone
from `origin`.

## Why the worktree existed

Created 2026-08-24 to develop and full-6-fold-test the Mamba temporal
backend for the dense-edge model in isolation, without disturbing whatever
was checked out in the main working tree (`tf-node-encoding` at the time).
A second `git worktree` pointed at its own branch, same repo, own working
directory (`EEG_Benchmarks_mamba`) -- not a clone, not a fork.

## What happened on the branch, commit by commit

**`8e0a6bd` "Add Mamba temporal backend for dense-edge model" (2026-08-24)**
-- the actual feature work. Added `_DenseEdgeMambaTemporal` to
`cwt_gnn_classifiers.py`: a third, interchangeable `dense_edge_conv`
backend (`dense_edge_temporal_mode="mamba"`) alongside the existing `conv`
(Conv2d+pool) and `rnn` (GRU) backends, same
`[B, C_in, E, T] -> [B, out_channels, E, 1]` contract. Used **mambapy**
(pure PyTorch Mamba) rather than the official `mamba-ssm` package, because
`mamba-ssm` ships no Windows wheel and needs an nvcc/Linux toolchain to
build, and wouldn't support the MPS path either -- this repo runs across a
Windows box, WSL, and an M-series Mac. Added `mamba_chunk_size`
(gradient-checkpointed chunking over the `B*E` leading dim) purely to avoid
a CUDA OOM found during smoke testing (mambapy's scan materializes
`O(B*E*T*d_state)` tensors) -- verified numerically equivalent to an
unchunked call. Wired up `--pipeline dense_edge_mamba` in
`run_pipelines.py`/`smoke_test.py`, plus `tests/test_dense_edge_mamba.py`.
Smoke-tested on CUDA (RTX 3070 Ti): loss decreases monotonically, no
NaN/Inf, ~14x slower than GRU **at smoke scale** (chunking overhead, not
the architecture -- see below for why that ratio doesn't hold at full
scale).

**`5301b8c` "mamba" (2026-08-24)** -- minor: a `.gitignore` tweak plus the
session note that became `Epilepsy/Session_notes/2026_08_24/
dense_edge_mamba_k23_full_run.md` (a single-fold `smoke_test.py` timing
run, capped dataset -- 50.74s/epoch. Not the full-6-fold number; see next
commit for that).

**`2985233` "Full 6-fold dense_edge_mamba results (23ch, encoder-free, val
0.2)" (2026-08-25)** -- the actual matched-protocol full 6-fold run, same
protocol as the main repo's GRU 6-fold run from the same day (same 6 chb01
leave-one-seizure-out seizures, val split 0.2, both bf16 flags, full
23-channel mesh, encoder-free model). Results:

| metric | GRU | Mamba (this branch) |
|---|---|---|
| mean AP | 0.423 | 0.499 |
| FAR/h (raw -> smoothed) | 14.26 -> 9.21 | 11.73 -> 6.39 |
| k-of-n hit rate | 5/6 | 5/6 (different seizure missed: `1_26_0` for GRU, `1_18_0` here) |
| mean epoch_time | ~14.3s | ~66s (~4.6x) |

Full per-fold tables live in the main repo's own
`Epilepsy/Session_notes/2026_08_25/full_6fold_23ch_encoderfree_val_gru.md`
(that file predates the branch deletion and is unaffected by it). This
`~66s/epoch, mambapy pscan, no CUDA kernel` figure is the correct
apples-to-apples historical baseline for this workload on the RTX 3070 Ti
-- not the 50.74s single-fold smoke number, and not any number from a run
on different hardware (e.g. an RTX 3060 lands slower, ~74s/epoch, for
hardware reasons alone).

## Why it's safe to have deleted

`main` had already independently gained its own `_DenseEdgeMambaTemporal`
/ `dense_edge_mamba` pipeline, `main`'s version is a **superset** of what
this branch had -- it additionally supports the fused `mamba-ssm` CUDA
kernel (`--mamba-use-cuda-kernel`, auto-detected on CUDA when `mamba-ssm`
is importable), which this branch's code never had (it only ever used the
mambapy pscan path). So nothing on this branch was unmerged, load-bearing
work -- `main`'s Mamba support is strictly ahead of it.

## Recovering it if ever needed

The branch tip is preserved at the tag `archive/mamba-temporal-edge-model`
(pushed to `origin`, never deleted) -- `git checkout
archive/mamba-temporal-edge-model` restores the exact state described
above, including the `full6fold_mamba_23ch_20260825-113022.log` raw run
log and the per-fold/per-seizure result CSVs that `2985233` recorded.
