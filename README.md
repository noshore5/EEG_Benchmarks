# EEG_Benchmarks
Benchmarking on MOABB BCI benchmarks along with additional paradigms

## CHB-MIT subjects

Default epilepsy pipelines use CHB-MIT subject `chb01`. Recordings for any
subject aren't in git; a subject with a known GitHub Release mirror is
fetched from there instead of PhysioNet's (throttled) S3 mirror --
currently `chb01`-`chb04`, see `GITHUB_RELEASE_SHA256` in
`datasets/epilepsy/chb_mit.py` for the live list:

- [`chbmit-chb01-1.0.0`](https://github.com/noshore5/EEG_Benchmarks/releases/tag/chbmit-chb01-1.0.0) -- `chb01.tar.gz` (~991MB, 42 EDFs + summary)
- [`chbmit-chb02-1.0.0`](https://github.com/noshore5/EEG_Benchmarks/releases/tag/chbmit-chb02-1.0.0) -- `chb02.tar.gz` (~859MB, 36 EDFs + summary)
- [`chbmit-chb03-1.0.0`](https://github.com/noshore5/EEG_Benchmarks/releases/tag/chbmit-chb03-1.0.0) -- `chb03.tar.gz` (~863MB, 38 EDFs + summary)
- [`chbmit-chb04-1.0.0`](https://github.com/noshore5/EEG_Benchmarks/releases/tag/chbmit-chb04-1.0.0) -- `chb04.part1.tar.xz` + `chb04.part2.tar.xz` (~1.6GB combined, 42 EDFs + summary split across 2 assets -- chb04's raw size doesn't fit GitHub's 2GiB-per-asset cap as one archive)

All PhysioNet 1.0.0, ODC-By 1.0 -- see `THIRD_PARTY_NOTICES.md`.

`datasets/epilepsy/chb_mit.py` downloads a registered subject's archive(s)
on first use and extracts them into the MNE cache
(`~/mne_data/MNE-chbmit-data/chbmit/1.0.0/chbXX/`). A subject NOT in
`GITHUB_RELEASE_SHA256` downloads from PhysioNet's S3 mirror instead, same
as before this mechanism existed. Override any subject's archive with
`CHBMIT_CHB{NN}_ARCHIVE_URL` / `CHBMIT_CHB{NN}_SHA256` (e.g.
`CHBMIT_CHB01_ARCHIVE_URL` / `CHBMIT_CHB01_SHA256`) if needed -- the env
override always describes a single archive, even for a subject normally
split into parts.

## RunPod pod image

`Dockerfile` (repo root) bakes in this repo's Python dependencies (`pip
install -r requirements.txt`, no apt build toolchain needed anymore -- see
"fcwt" below) on top of `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(confirmed 2026-08-20 against a live RTX 4090 pod: driver 570.133.20, CUDA
12.8, `torch==2.8.0+cu128` with `torch.cuda.is_available() == True`). It
deliberately does **not** bake in the repo's pipeline code -- code is synced
to the pod at launch (rsync/git clone) since it changes on nearly every
commit. As of 2026-08-21 the image **does** bake in CHB-MIT subject `chb01`
(~1.6GB uncompressed, ~991MB as `chb01.tar.gz`), landing at
`/root/mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/` -- the only subject
`Epilepsy/run_pipelines.py`'s `DEFAULT_SUBJECTS` exercises by default, so a
fresh pod never needs a first-run download for the common case. The image
and `datasets/epilepsy/chb_mit.py` both pull that archive from the
[`chbmit-chb01-1.0.0`](https://github.com/noshore5/EEG_Benchmarks/releases/tag/chbmit-chb01-1.0.0)
GitHub Release (PhysioNet 1.0.0 files, ODC-By 1.0). Other subjects still
download on demand from PhysioNet's S3 mirror. See
`THIRD_PARTY_NOTICES.md`.

**Building it:** RunPod pods can't run a Docker daemon themselves (they're
unprivileged containers, no `cap_sys_admin`), so the image is built by GitHub
Actions (`.github/workflows/build-pod-image.yml`) on a real Docker daemon and
pushed to GHCR (`ghcr.io/noshore5/eeg_benchmarks`). Triggers automatically on
any push to `main` that touches `Dockerfile`, `requirements.txt`, or
`THIRD_PARTY_NOTICES.md` (repo code isn't baked into the image, so code-only
commits don't need a rebuild), or manually via `workflow_dispatch`. Tagged
with date + short commit hash (e.g. `20260820-831695a`) plus a floating
`latest`.

RunPod's own GitHub-integration build (Settings → Connections → GitHub, then
Serverless → New Endpoint → Import Git Repository) was tried first and is a
dead end for this use case: it publishes to RunPod's internal
`registry.runpod.net`, scoped to the Serverless endpoint it built for -- a
plain Pod can't pull it (`Failed to get Hub registry auth` / `No such image`,
confirmed against a live Pod, not a guess). It also requires a queue-endpoint
handler (`runpod.serverless.start()`) to build at all, which doesn't fit a
Pod-based repo like this one. Not used.

**Using it:** once built, pass `ghcr.io/noshore5/eeg_benchmarks:latest` (or a
specific date-commit tag) as `imageName` when creating a pod (or bake it into
a Runpod template) in place of the stock `runpod/pytorch:...` image, and skip
`bash setup.sh` entirely. Public GHCR package, no registry credential needed
to pull.

### `dense_edge_mamba` + fused `mamba-ssm` kernel (`Dockerfile.mamba`)

A second image, `Dockerfile.mamba`, bakes in repo code **and** compiles
`mamba-ssm`/`causal-conv1d` against the same `torch==2.8.0+cu128` base so
`_DenseEdgeMambaTemporal`'s `use_cuda_kernel` auto-detect (mambapy
`MambaConfig.use_cuda=True`) has a real fused scan to call. Windows/Mac
keep the portable `mambapy` pscan; this image is Linux/CUDA only.

- Workflow: `.github/workflows/build-mamba-pod-image.yml` (`workflow_dispatch`
  or a push that touches `Dockerfile.mamba`). Tags:
  `ghcr.io/noshore5/eeg_benchmarks-mamba:<date>-<sha>` and `:latest`.
- On the pod: `--device cuda --pipeline dense_edge_mamba`. Kernel
  auto-engages; `--no-mamba-use-cuda-kernel` forces pscan. The Mamba block
  is excluded from `--train-amp-bf16` (fp32) because the fused kernel is
  not (b)float16-safe.
- First thing on a new pod: `python scripts/dense_edge_mamba_cuda_kernel_parity.py`
  then `python Epilepsy/run_pipelines.py --pipeline dense_edge_mamba
  --channel-subset-k 23 --device cuda --max-folds 1 --epochs 1`.
- `_DenseEdgeMambaContinuous` does **not** use this kernel (no initial-state
  API on `selective_scan_fn`).

See `Epilepsy/runpod_mamba_fast_image_brief.md`.

**Fallback:** `setup.sh` (manual bootstrap on a stock pod) is still the
documented path until a few real runs have gone through the new image
cleanly -- don't remove it yet.

**fcwt:** dropped from `requirements.txt` (2026-08-20) along with the apt
`cmake build-essential libfftw3-dev` toolchain it needed to compile from
source -- that compile step was the slowest part of both `setup.sh` and this
image's build, for a dependency the default pipeline no longer uses now that
`cwt_backend="torch"` (`utils/torch_cwt.py`) is the default in
`run_pipelines.py`'s `_SHARED_ARCH_PARAMS`. `cwt_backend="fcwt"` still exists
in `cwt_gnn_classifiers.py` as a manual revert switch; using it again means
restoring both `fcwt==0.1.18` in `requirements.txt` and that apt step.
