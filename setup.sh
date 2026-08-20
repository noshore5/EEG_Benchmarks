#!/usr/bin/env bash
# One-time environment setup for running the Epilepsy pipelines on a Runpod
# GPU pod. Run this once per fresh pod, from the repo root:
#
#   bash setup.sh
#
# What it does:
#   Installs everything in requirements.txt.
#
# NOT handled here (deliberately deferred, same reasoning as the Runpod
# agent-setup doc's own "install on demand" philosophy):
#   - CHB-MIT data download: happens automatically on first pipeline run,
#     pulled from an S3 mirror (see datasets/epilepsy/chb_mit.py).
#   - --device: torch==2.8.0 installs its CUDA-enabled Linux build
#     automatically via plain pip, but every run_pipelines.py invocation
#     still needs `--device cuda` passed explicitly (its defaults are
#     hardcoded to "mps" for the author's Mac -- see run_pipelines.py).
#
# 2026-08-20: this used to also apt-get install cmake/build-essential/
# libfftw3-dev, the toolchain fcwt needed to compile from source (PyPI only
# ships a macOS wheel for it). fcwt has been dropped from requirements.txt
# entirely -- cwt_backend="torch" (utils/torch_cwt.py) is now the default
# CWT path (see run_pipelines.py's _SHARED_ARCH_PARAMS), and that apt step
# was the slowest part of this script for a dependency nothing exercises by
# default anymore. cwt_backend="fcwt" still exists in cwt_gnn_classifiers.py
# as a manual revert switch; using it again means restoring both that apt
# step and `fcwt==0.1.18` in requirements.txt, not a default-path concern.

set -euo pipefail

echo "==> Installing Python requirements..."
# --break-system-packages: this pod's image PEP-668-protects the system Python
# (Ubuntu 24.04 base). Safe here -- the container is ephemeral and dedicated
# to this run, there's no system package manager state to protect.
# --ignore-installed: some deps (e.g. pyparsing) are pre-installed via apt/dpkg
# with no pip RECORD file, so pip can't safely uninstall them in place to
# upgrade -- ignore-installed makes it shadow-install the pinned version
# instead of failing on the uninstall step.
pip install --break-system-packages --ignore-installed -r requirements.txt

echo "==> Done. Remember: pass --device cuda to every Epilepsy/run_pipelines.py invocation."
