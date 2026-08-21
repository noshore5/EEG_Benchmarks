#!/usr/bin/env bash
#
# Bounded RunPod session for the Epilepsy pipelines: launch a pod from the
# baked-dataset image -> verify it (Step 4 of tonight's plan) -> run the
# existing --smoke fast-path (Step 6) -> report real output -> ALWAYS tear
# the pod down, no matter how this script ends.
#
# Deliberately does NOT run the full-scale eval -- that's a separate,
# explicitly-approved follow-up (see this file's own final echo). This
# script's whole point is to be the safety net Step 5 asked for: one bounded
# pod lifecycle, cleaned up automatically, so nothing is left running and
# billing idle even if something above errors out or the session drops.
#
# Usage: bash scripts/launch_pod_smoke_test.sh
# Override any of the config vars below via env, e.g.:
#   WALL_CLOCK_HOURS=1 bash scripts/launch_pod_smoke_test.sh
#
# Requires: runpodctl (authenticated -- `runpodctl user` should succeed),
# an SSH key already registered on the account (`runpodctl ssh list-keys`),
# python3 + jq-free JSON parsing (uses python3 -c, no jq dependency).

set -euo pipefail

# ---- config (override via env) --------------------------------------------
IMAGE="${IMAGE:-ghcr.io/noshore5/eeg_benchmarks:20260821-f1eef36}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
MIN_CUDA_VERSION="${MIN_CUDA_VERSION:-12.8}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-50}"
WALL_CLOCK_HOURS="${WALL_CLOCK_HOURS:-2}"
POD_NAME="${POD_NAME:-eeg-benchmarks-smoke-$(date -u +%Y%m%d-%H%M%S)}"
REPO_URL="${REPO_URL:-https://github.com/noshore5/EEG_Benchmarks.git}"
SSH_TIMEOUT_S="${SSH_TIMEOUT_S:-300}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

# ---- Step 5 safety net, layer 1: local trap, fires on ANY exit ------------
# (success, error, Ctrl-C/SIGINT, or SIGTERM). Best-effort: if the pod is
# already gone (e.g. its own --terminate-after already fired) the delete
# call just fails harmlessly.
POD_ID=""
cleanup() {
  local exit_code=$?
  if [ -n "$POD_ID" ]; then
    echo "[cleanup] tearing down pod $POD_ID (script exit code $exit_code)..." >&2
    if runpodctl pod delete "$POD_ID" >/dev/null 2>&1; then
      echo "[cleanup] pod $POD_ID deleted." >&2
    else
      echo "[cleanup] WARNING: delete call failed -- check the console for pod $POD_ID," \
           "or wait for its --terminate-after backstop." >&2
    fi
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

ssh_exec() {
  # Non-interactive SSH, per runpod-usage/getting-started.md's own pattern.
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=10 \
      -i "$SSH_KEY_PATH" -p "$SSH_PORT" "root@$SSH_HOST" "$1"
}

echo "== Step 3: launching pod (image=$IMAGE gpu=$GPU_ID min_cuda=$MIN_CUDA_VERSION) =="

# ---- Step 5 safety net, layer 2: pod-side wall-clock cap -------------------
# Enforced by Runpod itself (--terminate-after), independent of whether this
# script or the orchestrating Claude Code session is even still running.
if date -v+1H >/dev/null 2>&1; then
  TERMINATE_AFTER="$(date -u -v+"${WALL_CLOCK_HOURS}"H '+%Y-%m-%dT%H:%M:%SZ')"   # BSD date (macOS)
else
  TERMINATE_AFTER="$(date -u -d "+${WALL_CLOCK_HOURS} hours" '+%Y-%m-%dT%H:%M:%SZ')"  # GNU date
fi
echo "terminate_after=$TERMINATE_AFTER (hard cap, enforced by Runpod regardless of this script)"

CREATE_JSON="$(runpodctl pod create \
  --name "$POD_NAME" \
  --image "$IMAGE" \
  --gpu-id "$GPU_ID" \
  --min-cuda-version "$MIN_CUDA_VERSION" \
  --ports "22/tcp" \
  --container-disk-in-gb "$CONTAINER_DISK_GB" \
  --terminate-after "$TERMINATE_AFTER" \
  --wait --wait-timeout 10m)"

POD_ID="$(echo "$CREATE_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
echo "pod created: $POD_ID"

echo "== Connecting =="
# `--wait` only proves a TCP banner answered on :22, not that sshd/the real
# shell is up yet (confirmed gotcha: "Running"/"READY" can lag real
# readiness by ~30-90s) -- poll `ssh info` + a real ssh round-trip instead
# of trusting the first response.
DEADLINE=$((SECONDS + SSH_TIMEOUT_S))
SSH_HOST="" SSH_PORT="" SSH_KEY_PATH=""
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  INFO_JSON="$(runpodctl ssh info "$POD_ID" 2>/dev/null || true)"
  SSH_HOST="$(echo "$INFO_JSON" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("ip",""))
except Exception:
    print("")' 2>/dev/null || true)"
  if [ -n "$SSH_HOST" ]; then
    SSH_PORT="$(echo "$INFO_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["port"])')"
    SSH_KEY_PATH="$(echo "$INFO_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["ssh_key"]["path"])')"
    if ssh_exec "echo ok" >/dev/null 2>&1; then
      echo "ssh reachable: root@$SSH_HOST:$SSH_PORT"
      break
    fi
  fi
  sleep 5
done
if [ -z "$SSH_HOST" ] || ! ssh_exec "echo ok" >/dev/null 2>&1; then
  echo "ERROR: pod never became SSH-reachable within ${SSH_TIMEOUT_S}s -- likely a bad-machine draw." >&2
  echo "Not retrying automatically; the trap above will delete pod $POD_ID on exit." >&2
  exit 1
fi

echo
echo "== Step 4: pre-flight verification =="

echo "-- torch.cuda.is_available() --"
CUDA_OK="$(ssh_exec 'python3 -c "import torch; print(torch.cuda.is_available())"')"
echo "$CUDA_OK"
[ "$CUDA_OK" = "True" ] || { echo "ERROR: CUDA not available on pod." >&2; exit 1; }

echo "-- pip show fcwt (must fail -- proves it's genuinely absent) --"
if ssh_exec 'pip show fcwt' >/tmp/fcwt_check.out 2>&1; then
  echo "ERROR: fcwt is installed on the pod (expected absent):" >&2
  cat /tmp/fcwt_check.out >&2
  exit 1
fi
echo "confirmed absent."

echo "-- cloning repo at $EXPECTED_COMMIT and checking git log -1 --"
ssh_exec "rm -rf /workspace/EEG_Benchmarks && git clone --quiet '$REPO_URL' /workspace/EEG_Benchmarks && cd /workspace/EEG_Benchmarks && git checkout --quiet '$EXPECTED_COMMIT'"
POD_COMMIT="$(ssh_exec 'cd /workspace/EEG_Benchmarks && git log -1 --format=%H')"
echo "expected=$EXPECTED_COMMIT pod=$POD_COMMIT"
[ "$POD_COMMIT" = "$EXPECTED_COMMIT" ] || { echo "ERROR: commit mismatch." >&2; exit 1; }

echo "-- dataset reachable at expected path --"
N_EDF="$(ssh_exec "find /root/mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01 -name '*.edf' 2>/dev/null | wc -l" | tr -d '[:space:]')"
echo "chb01 .edf files found: $N_EDF"
[ "$N_EDF" -ge 40 ] || { echo "ERROR: dataset not fully present (expected >=40 .edf files)." >&2; exit 1; }

echo
echo "All Step 4 checks passed clean."
echo
echo "== Step 6: smoke test (Epilepsy/run_pipelines.py --smoke) =="
ssh_exec "cd /workspace/EEG_Benchmarks && python3 Epilepsy/run_pipelines.py --smoke --device cuda" \
  2>&1 | tee /tmp/smoke_test.out
SMOKE_EXIT="${PIPESTATUS[0]}"

echo
if [ "$SMOKE_EXIT" -eq 0 ]; then
  echo "== Smoke test PASSED (exit 0). Output above is real, from this run. =="
  echo "Full-scale eval is a separate, explicitly-approved step -- not run by this script."
else
  echo "== Smoke test FAILED (exit $SMOKE_EXIT). See output above. =="
fi

# trap fires here regardless of $SMOKE_EXIT -- pod is deleted either way.
exit "$SMOKE_EXIT"
