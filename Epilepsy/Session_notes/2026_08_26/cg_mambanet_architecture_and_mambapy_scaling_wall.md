# CG-MambaNet: architecture built + verified, real chb01 run BLOCKED by mambapy's scan performance

## Motivation

Following up on the SOTA-landscape research (`Session_notes/2026_08_25/` and
the `seizure-prediction-sota-landscape` memory): CG-MambaNet (arXiv
2606.08226, Chen et al., cross-patient LOPO AUC-ROC 0.8152 on CHB-MIT) is a
CNN-GCN-Mamba-BiLSTM architecture with no public code. Two-phase plan:

1. **This session**: build the architecture, test it under THIS REPO'S OWN
   chb01-only leave-one-seizure-out protocol (same as GRU/Mamba/DBConformer/
   SlimSeiz) for a fast, apples-to-apples read against `dense_edge_mamba` --
   explicitly NOT a reproduction of the paper's own 0.8152 number (that needs
   multi-patient leave-one-patient-out, deferred, see below).
2. **Deferred**: the full multi-patient LOPO reproduction, scoped in detail
   by three parallel exploration agents earlier this session (montage
   unification across CHB-MIT patients is the real blocker there -- no
   channel picking/reordering exists anywhere in `chb_mit.py`).

Phase 1 is DONE at the architecture/wiring level but BLOCKED at the "actually
run it" level -- see "The blocker" below. Everything in this note is on the
`cg-mambanet` branch (off `main`, not merged).

## What got built

`Epilepsy/pipelines/cg_mambanet_classifier.py` (new) -- built from the
paper's own Methods section (unusually detailed for a preprint -- full
formulas, hyperparameter table), NOT vendored code (no public implementation
exists). Every interpretive choice/deviation from the paper's literal spec is
flagged in that file's module docstring; the big ones:

- CNN front-end (two depthwise convs k=5/k=15 + pointwise fusion + residual,
  shared weights per (channel,patch) segment) -> `_PatchEmbedding` (NEW,
  added mid-session, see "The blocker") -> 2-layer learnable-adjacency GCN ->
  bidirectional 12-layer Mamba (built on this repo's own pinned `mambapy`,
  NOT SlimSeiz's from-scratch block) -> 2-layer BiLSTM -> MLP head.
- Runs under this repo's own `--sph`/`--sop` labeling (not the paper's
  unbuffered 30min preictal window), native 256Hz (no 200Hz resample), no
  bandpass/notch/artifact-rejection preprocessing, global-scalar
  normalization (not per-channel) -- all deliberate, matching DBConformer/
  SlimSeiz's own precedent for "run under this repo's protocol."
- Always applies the paper's fixed 16-channel bipolar montage
  (`CG_MAMBANET_CHANNEL_INDICES` in `run_pipelines.py`, resolved against the
  existing `CHB01_CHANNEL_NAMES` table) -- chb01-only, guarded the same way
  `--slimseiz-fixed-channels` is.
- Wired into `run_pipelines.py` as `--pipeline cg_mambanet`, following the
  DBConformer/SlimSeiz `_raw_classifier_family_*` pattern exactly (own
  PARAMS dicts, own `results/cg_mambanet/` dir, shares
  `leave_one_seizure_out_raw_classifier[_prediction]`).
- `tests/test_cg_mambanet_classifier.py` (new) -- synthetic smoke tests
  (wiring, fixed-channel slicing, patch_size divisibility check). Also fixed
  two PRE-EXISTING stale `--pipeline` choices assertions
  (`tests/test_dense_edge_mamba.py`, `tests/test_tf_node_encoder.py`) that
  predated the 2026-08-25 dbconformer/slimseiz addition and were never
  updated -- not something this session broke, just discovered while
  touching the same area. All 73 tests pass.
- Also had to `pip install mambapy==1.2.0` into `.venv` -- pinned in
  `requirements.txt` (already a `dense_edge_mamba` dependency) but wasn't
  actually installed in this checkout's venv.

**Real chb01 smoke test succeeded end-to-end** (`--smoke --max-folds 1
--device cpu`): dataset built (775 windows, 22.6% preictal), model trained
2 epochs, per-seizure CSV written, plausible-looking output (AP 0.311,
hit rate 1/1). Confirms the architecture is wired correctly. Results:
`Epilepsy/results/cg_mambanet/prediction/prediction_leave_one_seizure_out_20260826-083517.csv`
(+ per-seizure CSV, same timestamp).

## The blocker: mambapy's Mamba scan does not scale to this architecture's size, on CPU OR MPS

**First problem, found and partially fixed:** the paper's CNN-front-end ->
GCN handoff doesn't describe a projection, and reading `d=200` as "the
embedding width IS the per-patch sample count" (`d_model = patch_size`)
gave `d_model=256` at this repo's 256Hz/1-patch-per-second convention -- 16x
the `d_model=16` `dense_edge_mamba` uses elsewhere in this repo. Measured
(smoke-scale, patch_size=256, seq_len=64): **92.6s per forward+backward
pass**, 13.35M params. Fixed by adding an optional `_PatchEmbedding`
projection (`nn.Linear(patch_size, d_embed)`) decoupling embedding width
from patch_size -- `d_embed=64` default (in line with `dense_edge_mamba`),
`d_embed=None` preserves the paper-literal reading for later CUDA use. This
dropped smoke-scale batch time to **4.42s** (21x faster) and params to
1.87M -- looked like a full fix.

**Second, deeper problem, NOT fixable by cg_mambanet's own hyperparameters:**
at real-scale (30s prediction window -> 30 patches x 16 channels = seq_len
480, still `d_embed=64`), a single forward+backward pass of just the
`_BiMambaEncoder` (12 layers x 2 directions = 24 total Mamba module
instances) shows severely SUPER-LINEAR scaling with batch size on CPU:

| batch | forward | backward | total |
|---|---|---|---|
| 4  | 1.11s  | 1.97s   | 3.08s   |
| 8  | 4.01s  | 8.45s   | 12.47s (4.0x for 2x batch) |
| 16 | 12.08s | 28.19s  | 40.27s (3.2x for 2x batch) |
| 32 | -- | -- | did not finish in remaining ~4min budget |

Ruled out as the cause: NOT `d_model` width (already fixed above), NOT
leaf-vs-non-leaf input tensors (tested directly -- only a 2.6x difference at
batch=4, nowhere near enough to explain the batch scaling). Per-stage timing
of the full model (`no_grad`/`eval`, real scale) showed the forward pass
ALONE only takes ~11s total (front_end 1.2s + mamba 9.5s + everything else
negligible) -- so the pathology is specifically an autograd/backward-path
issue in `mambapy`'s scan, not raw forward compute.

**MPS is not a fix -- it's worse.** Same test on MPS: batch=4 took 4.06s,
batch=8 took **72.26s** (17.8x for 2x batch), and batch=16 **OOM'd outright**
(`RuntimeError: MPS backend out of memory (MPS allocated: 18.04 GiB, ... max
allowed: 18.13 GiB)`) inside `mambapy/pscan.py`'s `pad_npo2` -- the parallel
Blelloch scan pads the sequence to the next power of two and keeps
intermediate tensors at every scan level for backward, across all 24
directional Mamba instances at once. That's the actual root cause: it's a
memory/compute-graph-size problem inherent to `mambapy`'s `pscan=True` path
at this encoder size, not a CPU-specific slowness.

**`pscan=False` (mambapy's documented sequential-scan fallback) is not a fix
either -- it's worse.** Added a `pscan` passthrough to `_BiMambaEncoder` and
tested it directly: forward stays cheap (14.16s at batch=16) but **backward
explodes** (157.97s at batch=4, **960.89s at batch=16**). The sequential
scan's Python `for` loop over 480 timesteps, chained across 24 directional
Mamba instances, builds a backward graph with roughly 24*480 ≈ 11,500
sequential autograd nodes -- catastrophic under autograd regardless of the
parallel scan's memory issue.

**Conclusion: neither of `mambapy`'s two scan implementations is viable,
on CPU or MPS, for a 12-layer x 2-direction encoder at seq_len=480.** This
is below `cg_mambanet`'s own hyperparameters -- it's a `mambapy`
(pure-PyTorch Mamba) limitation at this depth/sequence-length/direction-
count combination. The fused `mamba-ssm` CUDA kernel is a genuinely
different code path (a custom kernel, not built from chained PyTorch ops)
and almost certainly doesn't hit this wall -- the whole reason
`ghcr.io/noshore5/eeg_benchmarks-mamba` exists.

## Decision: stop here, resume on CUDA later

Given the choice between shrinking `mamba_n_layers` (further deviation from
the paper, degrades what's being tested) vs. moving to the RunPod CUDA image
(preserves paper-faithful depth, needs infra), the user chose to STOP here
rather than compromise the architecture further. `cg_mambanet` is left at
the paper's own `mamba_n_layers=12`, `mamba_d_state=64` in
`_CG_MAMBANET_SHARED_PARAMS` -- these are NOT locally runnable at real scale
today (see table above), only smoke-scale (`--smoke`, 4s windows,
`seq_len=64`) verified working.

## Next steps (not started)

- Get `cg_mambanet` running on `ghcr.io/noshore5/eeg_benchmarks-mamba`
  (fused `mamba-ssm` kernel) -- `use_cuda_kernel=None` (auto-detect) is
  already wired via `_resolve_mamba_use_cuda_kernel`, same pattern
  `dense_edge_mamba`/continuous-mamba use, so no code change should be
  needed, just running it there. Worth a `scripts/`-style kernel-vs-pscan
  parity/timing check first (mirroring
  `scripts/dense_edge_mamba_cuda_kernel_parity.py`) before trusting numbers.
  Once on CUDA, `d_embed=None` (paper-literal, no projection) is worth
  reconsidering too, since the whole reason `d_embed=64` exists is CPU/MPS
  practicality.
- Once a real 6-fold chb01 run completes, add the row to the existing 4-way
  comparison table (`Session_notes/2026_08_25/
  pipeline_comparison_gru_mamba_dbconformer_slimseiz.md`).
- The full multi-patient LOPO reproduction (comparable to the paper's 0.8152
  AUC) is still separately deferred -- see this session's earlier
  exploration-agent findings (not written to a file yet, only in-conversation
  as of this note).
