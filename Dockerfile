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

WORKDIR /workspace

# Same apt packages setup.sh installs: build toolchain fcwt needs to
# compile from source (PyPI only ships a macOS wheel for fcwt==0.1.18,
# so `pip install` falls back to building the sdist).
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake build-essential libfftw3-dev && \
    rm -rf /var/lib/apt/lists/*

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
