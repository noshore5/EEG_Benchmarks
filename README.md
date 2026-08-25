# EEG_Benchmarks
Benchmarking on MOABB BCI benchmarks along with additional paradigms

## CHB-MIT subject 1

Default epilepsy pipelines use CHB-MIT subject `chb01`. The recordings are
not in git; they live on the
[`chbmit-chb01-1.0.0`](https://github.com/noshore5/EEG_Benchmarks/releases/tag/chbmit-chb01-1.0.0)
GitHub Release as `chb01.tar.gz` (~991MB, 42 EDFs + summary, PhysioNet
1.0.0, ODC-By 1.0 -- see `THIRD_PARTY_NOTICES.md`).

`datasets/epilepsy/chb_mit.py` downloads that archive on first use and
extracts it into the MNE cache (`~/mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/`).
Other subjects still come from PhysioNet's S3 mirror. Override the archive
with `CHBMIT_CHB01_ARCHIVE_URL` / `CHBMIT_CHB01_SHA256` if needed.

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
