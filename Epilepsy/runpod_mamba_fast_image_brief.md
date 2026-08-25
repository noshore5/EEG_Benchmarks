# Task brief: fast RunPod image for `dense_edge_mamba` (for Grok)

Written 2026-08-25 by Claude, for Grok to execute. Everything below was
verified against the live repo/environment as of this date (paths, line
numbers, commit hashes, installed package internals) -- not guessed.

## Context

`dense_edge_mamba` (`Epilepsy/pipelines/cwt_gnn_classifiers.py`,
`_DenseEdgeMambaTemporal`) currently runs on `mambapy` (pure-PyTorch Mamba
SSM), not the official `mamba-ssm` CUDA kernels, because `mamba-ssm` has no
Windows wheel and this repo also has to run on the author's Mac (MPS). A
same-protocol Mamba-vs-GRU comparison run on Windows/CUDA today
(`--pipeline dense_edge_mamba --channel-subset-k 23`, uncapped, val 0.2,
both bf16) measured Mamba at **~65s/epoch vs GRU's ~14s/epoch (~4.6x
slower)**, using `mambapy`'s pure-PyTorch selective scan.

Goal: a RunPod pod image, built and ready to `docker run`, that (a) trains
`dense_edge_mamba` meaningfully faster than that by actually using
`mamba-ssm`'s fused CUDA kernel, and (b) has the CHB-MIT dataset already
baked in so a pod boot needs zero download time before training starts.

**This is a from-scratch build, not an edit of the existing root
`Dockerfile`.** That file is deliberately dependency-only (no repo code
baked in, see its own header comment) and is used elsewhere -- leave it
untouched. Produce a new file, `Dockerfile.mamba` at repo root, that reuses
patterns from it but is a complete, ready-to-train image (code baked in
too).

## Repo facts you'll need

- Remote: `https://github.com/noshore5/EEG_Benchmarks`.
- `dense_edge_mamba` only exists on branch `mamba-temporal-edge-model`
  (not on `main`, not on `tf-node-encoding`). Tip commit as of this
  writing: `5301b8c9b604f634040004ffeb9d8c1ab942f90b`. Pin the image to
  this branch's current tip when you build (fetch fresh, don't hardcode
  this exact hash unless you want strict reproducibility of *today's*
  numbers specifically -- your call, just record whichever you pick).
- Local env this repo already verified: Python 3.11.9, `torch==2.8.0+cu128`
  (CUDA 12.8). The existing root `Dockerfile` uses base image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, confirmed against a
  live RTX 4090 pod (driver 570.133.20, CUDA 12.8,
  `torch.cuda.is_available() == True`). Start `Dockerfile.mamba` from the
  same base -- it already has the right torch/CUDA pairing.
- Dataset baking mechanism already exists and is proven working -- copy it
  verbatim from the existing `Dockerfile` (see its "2026-08-21" comment
  block): CHB-MIT subject chb01 (~1.6GB) is redistributed as a GitHub
  Release tarball (`chbmit-chb01-1.0.0` tag, ODC-By 1.0 license, see
  `THIRD_PARTY_NOTICES.md`), sha256-pinned
  (`bf91e579c8b61a6813442d9351fa6e111dd6078d43ab2b04fd66d4660324b6f9`),
  extracted into `/root/mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/` with
  `ENV MNE_DATA=/root/mne_data` set. `Epilepsy/run_pipelines.py`'s default
  `subjects=[1]` is the only subject exercised anywhere in this repo's
  benchmarks, so chb01-only is correct -- don't bake in other subjects
  unless asked.

## Task A -- `Dockerfile.mamba`

Build on the existing `Dockerfile`'s structure (same base image, same
dataset-baking block, same `requirements.txt` install), plus:

1. **Bake the actual repo code in** (the existing `Dockerfile` deliberately
   doesn't -- this one should, since the point is a ready-to-run image, not
   a shared dependency layer). Either `git clone --branch
   mamba-temporal-edge-model` at build time, or `COPY` a checkout -- your
   choice, but record in a comment which commit ended up in the image.
2. **Swap `mambapy`'s pure-PyTorch fallback for the real CUDA kernel** by
   installing `mamba-ssm` (which pulls in `causal-conv1d` as its own
   dependency) instead of/alongside `mambapy` (`mambapy` is still needed --
   see Task B, it isn't replaced, just configured to delegate). Both are
   plain `pip install` from a Linux/CUDA base, unlike Windows:
   - `mamba-ssm`'s selective-scan kernels are a compiled CUDA extension
     (`nvcc` + a C++ toolchain), not a wheel-only install on most
     platforms. Confirm the base image already has `nvcc` (`nvcc
     --version`); if not, add an apt step for `build-essential
     ninja-build` before the pip install (RunPod's `pytorch` template
     images usually ship the full CUDA devel toolkit, not a stripped
     runtime-only one, but verify rather than assume).
   - Set `ENV MAX_JOBS=<nproc>` (or similar) before this install step if
     the build machine has multiple cores -- `mamba-ssm`'s own install
     docs recommend this to parallelize the nvcc compile, which is
     otherwise slow.
   - Pin `causal-conv1d`/`mamba-ssm` versions compatible with
     `torch==2.8.0+cu128` / Python 3.11 -- check current PyPI/GitHub
     release compatibility yourself before pinning (not verified in this
     brief; don't blindly copy a version number from an old blog post).
   - **Fail the build loudly if this doesn't work**, don't let it silently
     degrade: add a `RUN` step after install that does
     `python -c "from mamba_ssm.ops.selective_scan_interface import
     selective_scan_fn"` and errors the build if that import fails.
     (`mambapy` itself swallows this ImportError with just a printed
     warning and silently falls back -- fine at runtime as a safety net,
     not fine as a silent build-time failure mode you'd only discover
     later by noticing training is still slow.)
3. Keep the dataset-baking layer early (before the code/mamba-ssm layers)
   so it's never invalidated by code changes, same ordering rationale the
   existing `Dockerfile` already documents.

## Task B -- code change to actually use the kernel

`mambapy`'s own `MambaConfig` (already a dependency, `mambapy==1.2.0`,
installed source at `mambapy/mamba.py`) has a **built-in** `use_cuda: bool
= False` flag that, when `True` and `mamba_ssm` is importable, delegates
to `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` (the fused
kernel) instead of its own pure-PyTorch parallel scan -- confirmed by
reading the installed package source. So this is a **flag flip + plumbing
change, not a rewrite** against a different API.

Current code (`Epilepsy/pipelines/cwt_gnn_classifiers.py`,
`_DenseEdgeMambaTemporal.__init__`, ~line 1891) constructs:

```python
self.mamba = Mamba(
    MambaConfig(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        expand_factor=expand,
        d_conv=d_conv,
        # Force the pure-PyTorch parallel-scan path on every device --
        # see class docstring. Never reaches for mamba-ssm's CUDA
        # kernels even if that package happens to also be installed.
        use_cuda=False,
    )
)
```

Change needed:

1. Add a constructor parameter (e.g. `use_cuda_kernel: Optional[bool] =
   None`, `None` = auto-detect: `True` iff `torch.cuda.is_available()` AND
   `mamba_ssm` importable, else `False`) and pass it through as
   `MambaConfig(..., use_cuda=<resolved value>)`. Keep the default
   behavior unchanged when `mamba_ssm` isn't installed (Windows/Mac dev
   machines must keep working exactly as today -- `mambapy`'s own
   try/except already no-ops gracefully back to `use_cuda=False` if the
   import fails, so auto-detect is safe even without a hasattr check on
   your end).
2. Wire it through to `Epilepsy/run_pipelines.py`'s CLI the same way every
   other `dense_edge_mamba_*` constructor knob already is (grep
   `mamba_d_model`/`mamba_d_state` in that file for the exact pattern to
   mirror -- PARAMS key + optional CLI flag, same convention
   `smoke_test.py`'s docstring describes).
3. **bf16 incompatibility -- important, don't skip this.** `mambapy`'s
   `MambaConfig.use_cuda` field's own inline comment says: "use official
   CUDA implementation when training (not compatible with (b)float16)".
   This repo's matched comparison runs use `--dense-edge-amp-bf16
   --train-amp-bf16` by default. Using the CUDA kernel path under bf16
   autocast is either going to error or silently produce numerically
   wrong results -- don't assume it's fine. Either: exclude the Mamba
   block from the bf16 autocast region specifically (keep it fp32), or
   document clearly that a `use_cuda_kernel=True` run must drop the bf16
   flags entirely. Whichever you pick, make it explicit and loud (a
   warning printed at model construction time if both are set
   simultaneously would be reasonable), not a silent footgun.
4. **`mamba_chunk_size` may need retuning.** The existing chunking
   mechanism (default 128) exists purely because the pure-PyTorch scan
   materializes large dense intermediate tensors that OOM'd an 8GB card
   (see the class docstring's 2026-08-24 note). The fused CUDA kernel does
   not materialize those tensors the same way, so the OOM pressure this
   was defending against may no longer apply at the same severity --
   worth re-measuring whether chunk_size can be raised (or the whole
   chunking path skipped) once `use_cuda_kernel=True`, as a secondary
   speed win. Not blocking -- ship correct-and-still-chunked first, tune
   after.
5. **Validate numerically before trusting any comparison number.** This is
   a genuinely different code path (fused kernel vs. pure-PyTorch scan),
   not the "same math, different memory layout" guarantee the existing
   chunk_size mechanism has. Before reporting any speed/accuracy numbers
   from this image as comparable to today's `mambapy` results: run a
   small fixed-seed forward pass through `_DenseEdgeMambaTemporal` with
   `use_cuda_kernel=True` vs `False` and confirm outputs match within a
   reasonable tolerance (this is a pure-PyTorch-vs-fused-kernel numerical
   equivalence check, small/fast, no full training run needed). Then run
   `python Epilepsy/run_pipelines.py --pipeline dense_edge_mamba
   --channel-subset-k 23 --device cuda --max-folds 1 --epochs 1` inside
   the built image as a real end-to-end sanity check before trusting
   anything larger.

## Acceptance checklist

- [ ] `docker build -f Dockerfile.mamba .` succeeds, including the
      `mamba_ssm` import-check `RUN` step (build fails loudly, not
      silently, if the kernel didn't compile).
- [ ] Image contains chb01 dataset pre-extracted at
      `/root/mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/` with `MNE_DATA`
      set -- verify a fresh container needs zero network access to start
      training subject 1.
- [ ] `_DenseEdgeMambaTemporal` accepts `use_cuda_kernel`, threaded through
      `run_pipelines.py`'s CLI, auto-detecting when unset.
- [ ] Numerical parity check (kernel vs. no-kernel) passes at a fixed seed.
- [ ] `--max-folds 1 --epochs 1` sanity run completes inside the image
      with `use_cuda_kernel` engaged (confirm via a printed log line, not
      just "no crash") and bf16 is either explicitly excluded from the
      Mamba block or explicitly disabled for the run, not silently mixed.
- [ ] Record the measured epoch_time against this same config's ~65s/epoch
      `mambapy` baseline -- report whatever the real number is, don't
      assume it'll land near GRU's ~14s/epoch.

## Explicitly out of scope

- Don't touch the existing root `Dockerfile` -- it's shared, dependency-
  only by design, used elsewhere.
- Don't bake in subjects other than chb01.
- Don't touch Windows/`requirements.txt`/WSL -- unrelated to this task
  (WSL was considered and rejected for the *current* Windows dev machine
  earlier this session; this brief is the RunPod path instead).
- Don't change `mamba_chunk_size`'s default for the existing (non-kernel)
  path -- only tune it under `use_cuda_kernel=True`, separately.
