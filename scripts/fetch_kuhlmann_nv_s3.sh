#!/usr/bin/env bash
# Fetches the Kuhlmann NeuroVista (NV) contest data from the private S3
# mirror instead of Dropbox -- for boxes launched via eeg-run-spot.sh /
# the GitHub Actions eeg-run.yml workflow (including from a cloud Claude
# Code shell with no access to the local Mac's Dropbox-downloaded copy).
#
# The data is DUA-restricted (see CONTEXT.md's 2026-09-22 entry /
# download_kuhlmann_nv.py's module docstring) -- s3://noshore-eeg-
# benchmarks-827938107865 is private (public access fully blocked, see
# AWS_INFRA.md) and every eeg-run worker already carries the eeg-gpu IAM
# role's s3-eeg-bucket RW grant, so this needs no extra credentials on a
# box and doesn't touch Dropbox at all. Uploaded once from the Mac as a
# single tar.gz (2.9GB, 826 files) rather than synced file-by-file --
# S3 handles one large object far better than thousands of small ones.
#
# Usage (from repo root, on a box or locally):
#   scripts/fetch_kuhlmann_nv_s3.sh [Pat1Train]
#
# Currently only Pat1Train is mirrored (see CONTEXT.md: Pat2/Pat3/Test
# are out of scope per explicit user request). Re-run
# `aws s3 cp <local tar.gz> s3://.../datasets/kuhlmann_nv/<Subfolder>.tar.gz`
# by hand first if a future session needs another subfolder mirrored.

set -euo pipefail

SUBFOLDER="${1:-Pat1Train}"
BUCKET="noshore-eeg-benchmarks-827938107865"
S3_KEY="datasets/kuhlmann_nv/${SUBFOLDER}.tar.gz"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="$REPO_ROOT/datasets/epilepsy/kuhlmann_nv"
DEST_DIR="$DEST_ROOT/$SUBFOLDER"

if [ -d "$DEST_DIR" ] && [ -n "$(ls -A "$DEST_DIR" 2>/dev/null)" ]; then
  echo "[fetch_kuhlmann_nv_s3] $DEST_DIR already populated, skipping fetch."
  exit 0
fi

mkdir -p "$DEST_ROOT"
echo "[fetch_kuhlmann_nv_s3] downloading s3://$BUCKET/$S3_KEY ..."
aws s3 cp "s3://$BUCKET/$S3_KEY" /tmp/kuhlmann_nv_fetch.tar.gz

echo "[fetch_kuhlmann_nv_s3] extracting into $DEST_ROOT ..."
tar -xzf /tmp/kuhlmann_nv_fetch.tar.gz -C "$DEST_ROOT"
rm -f /tmp/kuhlmann_nv_fetch.tar.gz

n=$(find "$DEST_DIR" -type f | wc -l | tr -d ' ')
echo "[fetch_kuhlmann_nv_s3] done -- $n files in $DEST_DIR"
