# Session notes — `Dockerfile.mamba` + `use_cuda_kernel` plumbing

Branch: `continuous-cwt-mamba`. Executes
`Epilepsy/runpod_mamba_fast_image_brief.md` (written for Grok, previously
unexecuted). This Mac has no Docker; RunPod pods have no Docker daemon;
the image is meant to be built by GitHub Actions the same way the
dep-only `Dockerfile` already is.

Stale bit in the brief: it said pin `mamba-temporal-edge-model` tip
`5301b8c`. This branch already has that merge plus continuous-Mamba
work; the image COPYs this checkout.

## Task B (code) -- done, uncommitted

- `_resolve_mamba_use_cuda_kernel` / `_mamba_ssm_importable`
- `_DenseEdgeMambaTemporal(..., use_cuda_kernel=None)` →
  `MambaConfig(use_cuda=resolved)`. Auto = CUDA + importable mamba-ssm.
  Explicit True errors if the kernel isn't there (no silent fallback).
- When the kernel is on, `_mamba_pooled` disables autocast and runs fp32
  so `--train-amp-bf16` does not feed bf16 into a kernel mambapy documents
  as incompatible.
- Wired through Core, Classifier, `DENSE_EDGE_MAMBA_PARAMS` /
  `PREDICTION_MAMBA_PARAMS` (`mamba_use_cuda_kernel=None`),
  `--mamba-use-cuda-kernel` / `--no-...` on `run_pipelines.py` and
  `smoke_test.py`.
- `_DenseEdgeMambaContinuous` stays on the local pscan. `selective_scan_fn`
  has `return_last_state` but no `initial_state`.
- `scripts/dense_edge_mamba_cuda_kernel_parity.py` -- CUDA + mamba-ssm only.
- Unit tests cover auto=False without the kernel, and True raising.

## Task A (image) -- files written, not yet built

- `Dockerfile.mamba` -- dataset layer copied from the root Dockerfile,
  then requirements, then `causal-conv1d`/`mamba-ssm` with
  `--no-build-isolation`, then a listed COPY of repo code (results/BCI
  excluded via `.dockerignore`). Build-time check is "`.so` file exists"
  not `import selective_scan_fn`, because GHA has no `libcuda`.
- `.github/workflows/build-mamba-pod-image.yml` →
  `ghcr.io/noshore5/eeg_benchmarks-mamba:<date>-<sha>`.

**Still to do once the image is on GHCR:** boot a CUDA pod with that
`imageName`, run the parity script, then
`--pipeline dense_edge_mamba --channel-subset-k 23 --device cuda
--max-folds 1 --epochs 1` and record epoch_time vs ~65s/epoch mambapy.
Not done this session.
