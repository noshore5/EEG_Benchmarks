# EEG_Benchmarks
Benchmarking on MOABB BCI benchmarks along with additional paradigms

## RunPod pod image

`Dockerfile` (repo root) bakes in this repo's Python dependencies (`pip
install -r requirements.txt`, no apt build toolchain needed anymore -- see
"fcwt" below) on top of `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(confirmed 2026-08-20 against a live RTX 4090 pod: driver 570.133.20, CUDA
12.8, `torch==2.8.0+cu128` with `torch.cuda.is_available() == True`). It
deliberately does **not** bake in the repo's pipeline code or any dataset --
code is synced to the pod at launch (rsync/git clone), and CHB-MIT data lives
on a RunPod Network Volume / downloads on first pipeline run, never in the
image.

**Building it:** RunPod pods can't run a Docker daemon themselves (they're
unprivileged containers, no `cap_sys_admin`), and RunPod's GitHub-connected
build is a Serverless-endpoint feature, not a plain-Pod one -- there's no
"build from GitHub" option scoped to Pods directly. One-time manual step
(RunPod dashboard, not scriptable via the API/MCP tools): Settings →
Connections → connect the `noshore5/EEG_Benchmarks` GitHub repo, then
Serverless → New Endpoint → Import Git Repository → pick the branch → deploy
(min workers = 0 so the endpoint itself doesn't idle-bill). That's why
`handler.py` exists at repo root: a throwaway `runpod.serverless.start()`
stub satisfying the queue-endpoint handler requirement so the build runs at
all -- nothing in the repo calls it. RunPod builds and hosts the resulting
image in its own registry; no external registry (Docker Hub/GHCR) needed.
**Rebuild whenever `requirements.txt` or `Dockerfile` changes** by pushing to
the connected branch (auto-triggers a build), and tag/label it with the date
+ short commit hash, e.g. `2026-08-20-ada4995`, so old/new builds stay
distinguishable.

**Using it:** once built, pass the resulting image name as `imageName` when
creating a pod (or bake it into a Runpod template) in place of the stock
`runpod/pytorch:...` image, and skip `bash setup.sh` entirely.

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
