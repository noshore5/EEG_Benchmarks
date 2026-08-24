# syntax=docker/dockerfile:1
#
# Dependency-only image for RunPod pods running the Epilepsy pipelines in
# this repo. Bakes in system + Python dependencies (setup.sh's bootstrap)
# so a fresh pod skips `bash setup.sh` (a few minutes of apt + fcwt
# source build) entirely.
#
# Deliberately does NOT COPY the repo's Python code in. Code on this
# branch changes on nearly every commit (see git log) while
# requirements.txt/setup.sh change rarely -- baking code into the image
# would mean a multi-minute RunPod rebuild on every code change, for no
# benefit. Code is synced to the pod at launch instead (rsync/git clone
# into /workspace), same as today's manual flow; this image only removes
# the "install dependencies" step from that flow.
#
# Base: official Runpod image already matching the GPU driver/CUDA this
# repo's torch==2.8.0 pin needs -- confirmed against a live RTX 4090 pod
# on 2026-08-20 (driver 570.133.20, CUDA 12.8; torch reports
# torch.cuda.is_available() == True). Starting from this instead of a
# bare nvidia/cuda image also means torch is already installed, so the
# pip install of requirements.txt below is a fast no-op verify for torch
# specifically, not a multi-hundred-MB download.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# RUN steps below use bash-only syntax (set -euo pipefail); Docker's
# default /bin/sh -c doesn't support pipefail.
SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# 2026-08-21: CHB-MIT subject chb01 (~1.6GB), baked into the image --
# matches DEFAULT_SUBJECTS = [1] in Epilepsy/run_pipelines.py, the only
# subject actually exercised by default. Placed as the very first layer
# (before requirements.txt, which changes far more often) so it's never
# invalidated by a dependency bump, and before any code sync so a
# code-only pod launch never needs to touch the network for data.
#
# Downloaded as one GitHub Release tarball (same archive
# datasets/epilepsy/chb_mit.py prefetches for subject 1). PhysioNet's
# HTTPS server throttles to ~180KB/s/connection and the official S3
# mirror is still slow for a full 42-file chb01 fetch; the release is
# the same bytes, ODC-By 1.0, extracted into the S3-shaped MNE cache
# layout so a later per-file data_dl lookup hits disk. Other subjects
# still download on demand from S3.
#
# Lands at /root/mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/, MNE's
# get_dataset_path("CHBMIT", ...) convention for this dataset (verified
# against the author's own local cache) -- MNE_DATA is set explicitly
# below so this doesn't depend on HOME resolving to /root.
#
# CHB-MIT is distributed under ODC-By 1.0 (Open Data Commons Attribution
# License), which requires attribution on redistribution -- see
# THIRD_PARTY_NOTICES.md, baked into the image alongside the data, and
# README.md.
ENV MNE_DATA=/root/mne_data
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY THIRD_PARTY_NOTICES.md /workspace/THIRD_PARTY_NOTICES.md
RUN set -euo pipefail; \
    ARCHIVE_URL="https://github.com/noshore5/EEG_Benchmarks/releases/download/chbmit-chb01-1.0.0/chb01.tar.gz"; \
    ARCHIVE_SHA="bf91e579c8b61a6813442d9351fa6e111dd6078d43ab2b04fd66d4660324b6f9"; \
    DEST_ROOT="/root/mne_data/MNE-chbmit-data/chbmit/1.0.0"; \
    mkdir -p "$DEST_ROOT"; \
    curl -fSL --retry 5 --retry-delay 2 -o /tmp/chb01.tar.gz "$ARCHIVE_URL"; \
    echo "$ARCHIVE_SHA  /tmp/chb01.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/chb01.tar.gz -C "$DEST_ROOT"; \
    rm -f /tmp/chb01.tar.gz; \
    n_edf=$(find "$DEST_ROOT/chb01" -name '*.edf' | wc -l); \
    echo "Extracted $n_edf .edf files for chb01"; \
    [ "$n_edf" -ge 40 ]

# 2026-08-20: no apt build-toolchain step here anymore. It used to install
# cmake/build-essential/libfftw3-dev so pip could compile fcwt from source
# (PyPI only ships a macOS wheel for it) -- fcwt has since been dropped
# from requirements.txt entirely now that cwt_backend="torch"
# (utils/torch_cwt.py) is the default CWT path (see run_pipelines.py's
# _SHARED_ARCH_PARAMS). That apt step was the single biggest chunk of
# build time for a dependency nothing exercises by default anymore.
# cwt_backend="fcwt" still exists in cwt_gnn_classifiers.py as a manual
# revert switch, but using it again means `pip install fcwt` (and this
# apt block) coming back too -- not a default-path concern.

# Only requirements.txt is copied in at this stage, not the rest of the
# repo -- keeps this layer cached across code-only commits.
COPY requirements.txt /tmp/requirements.txt

# Same flags setup.sh uses, for the same reasons (see that file's own
# comments): --break-system-packages because this image's system Python
# is PEP-668-protected (Ubuntu 24.04 base); --ignore-installed because
# some deps (e.g. pyparsing) are pre-installed via apt/dpkg with no pip
# RECORD file, so pip can't safely uninstall them in place to upgrade --
# ignore-installed shadow-installs the pinned version instead of failing.
RUN pip install --break-system-packages --ignore-installed -r /tmp/requirements.txt

# Reminder baked into the image itself, matching setup.sh's own trailing
# note -- every run_pipelines.py invocation still needs --device cuda
# passed explicitly; its defaults are hardcoded to "mps" for the
# author's Mac.
RUN echo "Reminder: pass --device cuda to every Epilepsy/run_pipelines.py invocation." > /etc/motd

# 2026-08-20: deliberately no CMD/ENTRYPOINT override here. An earlier
# version of this file added one (running handler.py, a throwaway
# runpod.serverless.start() stub) to satisfy RunPod's GitHub-integration
# Serverless build, which was being used as a build mechanism for this
# image. That mechanism turned out to be a dead end: images it produces
# live in RunPod's internal registry.runpod.net scoped to their
# originating Serverless endpoint, and a plain Pod can't pull them
# ("Failed to get Hub registry auth" / "No such image" -- confirmed
# against a live Pod, not a guess). Building/pushing to GHCR via GitHub
# Actions instead (see .github/workflows/) -- a real registry a Pod CAN
# pull from -- so this image keeps the base runpod/pytorch image's own
# ENTRYPOINT/CMD intact (sshd + PUBLIC_KEY setup), which a Pod actually
# needs. handler.py still exists at repo root in case the Serverless
# endpoint that build mechanism created is still around, but nothing in
# this Dockerfile references it anymore.
