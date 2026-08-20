# EEG_Benchmarks
Benchmarking on MOABB BCI benchmarks along with additional paradigms

## RunPod pod image

`Dockerfile` (repo root) bakes in this repo's system + Python dependencies
(the same steps `setup.sh` runs by hand: apt `cmake build-essential
libfftw3-dev` for building `fcwt` from source, then `pip install -r
requirements.txt`) on top of `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(confirmed 2026-08-20 against a live RTX 4090 pod: driver 570.133.20, CUDA
12.8, `torch==2.8.0+cu128` with `torch.cuda.is_available() == True`). It
deliberately does **not** bake in the repo's code or any dataset — code is
synced to the pod at launch (rsync/git clone), and CHB-MIT data lives on a
RunPod Network Volume / downloads on first pipeline run, never in the image.

**Building it:** RunPod pods can't run a Docker daemon themselves (they're
unprivileged containers, no `cap_sys_admin`), so this image is built via
RunPod's GitHub-connected build, not `docker build` on a pod. One-time manual
step (RunPod dashboard, not scriptable via the API/MCP tools): Settings →
connect the `noshore5/EEG_Benchmarks` GitHub repo → create a build pointed at
the root `Dockerfile` on the branch you want. RunPod builds and hosts the
image in its own registry; no external registry (Docker Hub/GHCR) needed.
**Rebuild whenever `requirements.txt` or `Dockerfile` changes** — trigger a
new build from the dashboard (or push a commit, if the connection is set to
auto-build) and tag it with the date + short commit hash, e.g.
`2026-08-20-ada4995`, so old/new builds stay distinguishable.

**Using it:** once built, pass the resulting image name as `imageName` when
creating a pod (or bake it into a Runpod template) in place of the stock
`runpod/pytorch:...` image, and skip `bash setup.sh` entirely.

**Fallback:** `setup.sh` (manual bootstrap on a stock pod) is still the
documented path until a few real runs have gone through the new image
cleanly — don't remove it yet.
