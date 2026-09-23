#!/bin/bash
# eeg-run-spot.sh -- fire-and-forget a pipeline run on an interruptible spot
# GPU box. Single-shot, NOT resumable: unlike eeg-spot-train.sh's
# SpotTrainer-contract keepalive loop (checkpoint/resume/DONE sentinel --
# only train_detector.py implements that), this is for any plain
# run_pipelines.py-style command that just runs to completion once. If the
# spot box is reclaimed mid-run, the run is lost and nothing relaunches it --
# rerun by hand. In exchange for that, it's a drop-in --cmd launcher like
# eeg-run.sh, not a per-target integration.
#
# Combines eeg-run.sh's outcome handling (promote_results.sh commit on
# success, SNS on failure, ship logs to S3, self-terminate) with
# eeg-spot-train.sh's WORKING GPU bits: capacity-fallback across (instance
# type, AZ) candidates, and the DLAMI-safe dependency install (install into
# the DLAMI's existing torch+CUDA venv rather than eeg-run.sh's plain
# `pip install --break-system-packages -r requirements.txt`, which is known
# broken on the Ubuntu-22.04 GPU DLAMI's pip 22.0 -- see CONTEXT.md
# 2026-09-07).
#
# Fold-level checkpointing: this script always syncs s3://$BUCKET/checkpoints/
# <name>/ down to /root/checkpoint before the run and back up continuously
# during + once more on exit. If your --cmd points at
# leave_one_seizure_out_prediction (run_pipelines.py's own --checkpoint-dir
# flag, 2026-09-15), just add `--checkpoint-dir /root/checkpoint` to it -- a
# spot reclaim then loses progress only back to the last completed fold, and
# rerunning this SAME script (same --name) auto-resumes from S3, not from
# fold 0. For a --cmd that doesn't support --checkpoint-dir, the sync is a
# harmless no-op (empty dir) and behavior is identical to a plain one-shot
# run -- checkpointing is opt-in via what --cmd does with the directory, not
# forced by this script.
#
# Usage:
#   scripts/eeg-run-spot.sh --name tgm-pred-6fold \
#     --cmd 'python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba --label-mode prediction --device cuda --checkpoint-dir /root/checkpoint' \
#     [--session-note '...'] [--disk 150] [--branch main] [--keep]
#
# Needs AWS creds for account 827938107865.

set -euo pipefail

REGION=us-east-1
BUCKET=noshore-eeg-benchmarks-827938107865
REPO_URL=https://github.com/noshore5/EEG_Benchmarks.git
KEY=eeg-box
PROFILE_NAME=eeg-gpu
SNS_TOPIC_ARN=${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:827938107865:eeg-runs}
DEPLOY_KEY_SSM=${DEPLOY_KEY_SSM:-/eeg/github-deploy-key}

## 2026-09-16: g5/g6 (Ampere/Ada, real datacenter GPUs) tried before g4dn
## (Tesla T4, weak/older, launch-overhead-heavy workloads like this one's
## per-batch dense-edge precompute lose to it) -- see CONTEXT.md.
DEFAULT_CANDIDATES=(
  "g5.xlarge:us-east-1a"    "g5.xlarge:us-east-1b"     "g5.xlarge:us-east-1c"
  "g5.xlarge:us-east-1d"    "g5.xlarge:us-east-1f"
  "g6.xlarge:us-east-1a"    "g6.xlarge:us-east-1b"     "g6.xlarge:us-east-1c"
  "g4dn.xlarge:us-east-1c"  "g4dn.xlarge:us-east-1d"  "g4dn.xlarge:us-east-1a"
  "g4dn.xlarge:us-east-1b"  "g4dn.xlarge:us-east-1f"
)
# 2026-09-21 (FAILURE_LOG.md #14): every candidate above is 4 vCPU / 16GB
# HOST RAM regardless of GPU -- fine for GPU-VRAM-bound pipelines
# (temporal_graph_mamba's dense-edge cache), but nonstgm-mamba-smoke's
# host RSS hit 14.24GB after just ONE fold, killed (SIGKILL/rc=137, the
# OS OOM killer, not a CUDA exception) entering fold 1. --instance-types
# lets a caller override the (type, AZ) candidate list for a run that
# needs a bigger-RAM box (e.g. g5.2xlarge/g6.2xlarge, 32GB) without
# touching this default list for every other pipeline that's fine on
# 16GB. Comma-separated "type:az" pairs, same format as the array above.
CANDIDATES=("${DEFAULT_CANDIDATES[@]}")
subnet_for() {
  case "$1" in
    us-east-1a) echo subnet-057fcd8e8ed1ec050;;
    us-east-1b) echo subnet-07cdbc1058cf4f752;;
    us-east-1c) echo subnet-021f5ceeb4af26220;;
    us-east-1d) echo subnet-00252702f59bac48f;;
    us-east-1e) echo subnet-0d55ae5c23cf61927;;
    us-east-1f) echo subnet-0090cef9098097cb0;;
    *) echo "unknown AZ $1" >&2; return 1;;
  esac
}
maxprice_for() {
  case "$1" in
    g4dn.xlarge)  echo 0.526;;
    g5.xlarge)    echo 1.006;;
    g6.xlarge)    echo 0.8048;;
    # 2026-09-21: 2xlarge sizes double vCPU+RAM (32GB) for the same GPU --
    # added for pipelines that are host-RAM-, not GPU-VRAM-, bound (see
    # --instance-types above). Prices are a generous margin over typical
    # spot rates for these sizes, not a tight optimization -- capacity,
    # not price, is usually the binding constraint on this launcher.
    g4dn.2xlarge) echo 0.752;;
    g5.2xlarge)   echo 1.212;;
    g6.2xlarge)   echo 0.978;;
    *)            echo 1.00;;
  esac
}

NAME=""; CMD=""; NOTE=""; DISK=150; BRANCH=main; KEEP=0; DOCKER_IMAGE=""; DOCKER_ENV=""; INSTANCE_TYPES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name)         NAME=$2; shift 2;;
    --cmd)          CMD=$2; shift 2;;
    --session-note) NOTE=$2; shift 2;;
    --disk)         DISK=$2; shift 2;;
    --branch)       BRANCH=$2; shift 2;;
    --docker-image) DOCKER_IMAGE=$2; shift 2;;
    # 2026-09-18: repeatable KEY=VAL passthrough into the docker-run path's
    # -e flags (the only env vars the container gets today are PYTHONPATH,
    # hardcoded below) -- added for EEG_BENCHMARKS_PROFILE_STEPS=1
    # diagnostic launches without hand-editing the script each time.
    --docker-env)   DOCKER_ENV="$DOCKER_ENV -e $2"; shift 2;;
    # 2026-09-21: override the default (type, AZ) candidate list, e.g.
    # --instance-types "g5.2xlarge:us-east-1a,g5.2xlarge:us-east-1b" for a
    # host-RAM-bound run (FAILURE_LOG.md #14). Comma-separated "type:az".
    --instance-types) INSTANCE_TYPES=$2; shift 2;;
    --keep)         KEEP=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$NAME" ] || { echo "--name required" >&2; exit 2; }
[ -n "$CMD" ]  || { echo "--cmd required"  >&2; exit 2; }
NAME=$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9._-' '-')
if [ -n "$INSTANCE_TYPES" ]; then
  IFS=',' read -r -a CANDIDATES <<< "$INSTANCE_TYPES"
fi

# When --docker-image is given, use the pre-baked custom AMI
# (ami-0c131b0c97ed93cda -- env/deps only, no code baked in; code is bind-
# mounted live from a fresh git clone in the docker-run path below). This
# skips the ~20min `docker pull` on every launch, PROVIDED the AMI's cached
# image digest still matches :latest -- any Dockerfile.mamba rebuild
# invalidates every layer from the changed point onward, so the AMI needs
# re-baking after each such rebuild or launches pay a partial re-pull (see
# 2026-09-17: ami-042ff1af14b5afec6 was stale from the PIP_SRC fix, re-baked
# to this id from a box that pulled digest sha256:974a906a...). Falls back
# to the dynamic DLAMI lookup otherwise. See AWS_INFRA.md.
if [ -n "$DOCKER_IMAGE" ]; then
  AMI=ami-0c131b0c97ed93cda
else
  AMI=$(aws ec2 describe-images --owners amazon --region $REGION \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
              "Name=state,Values=available" \
    --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
fi
SG=$(aws ec2 describe-security-groups --region $REGION \
  --filters Name=group-name,Values=eeg-ssh --query 'SecurityGroups[0].GroupId' --output text)

PFX="s3://$BUCKET/exports/runs/$NAME"
SHUTDOWN_BEHAVIOR=terminate; [ "$KEEP" = 1 ] && SHUTDOWN_BEHAVIOR=stop

CMD_B64=$(printf '%s' "$CMD"  | base64 | tr -d '\n')
NOTE_B64=$(printf '%s' "$NOTE" | base64 | tr -d '\n')

UD=$(cat <<EOF
#!/bin/bash
set -x
exec > /var/log/eeg-run-spot.log 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin:/snap/bin
# hard watchdog -- terminate no matter what after 8h even if the trap never fires
( sleep 28800; shutdown -h now ) &
BUCKET=$BUCKET
PFX="$PFX"
NAME="$NAME"
CKPT_S3="s3://$BUCKET/checkpoints/$NAME"
CMD=\$(echo $CMD_B64 | base64 -d)
NOTE=\$(echo $NOTE_B64 | base64 -d)
STARTED=\$(date -u +%Y-%m-%dT%H:%M:%SZ)

finish() {
  RC=\${RC:-1}
  aws s3 cp /var/log/eeg-run-spot.log "\$PFX/boot.log" || true
  [ -f /root/pip.log ] && aws s3 cp /root/pip.log "\$PFX/pip.log" || true
  [ -f /root/run.log ] && aws s3 cp /root/run.log "\$PFX/run.log" || true
  [ -d /root/checkpoint ] && aws s3 sync /root/checkpoint "\$CKPT_S3" || true
  [ -d /root/repo/Epilepsy/results ] && aws s3 cp --recursive /root/repo/Epilepsy/results "\$PFX/results" || true
  if [ "\$RC" != 0 ]; then
    aws sns publish --region $REGION --topic-arn "$SNS_TOPIC_ARN" \
      --subject "eeg-run-spot FAILED/interrupted: \$NAME (rc=\$RC)" \
      --message "\$(printf 'run %s rc=%s (rc=143/137 = likely a spot reclaim, not a bug)\ncmd: %s\nlogs: %s\n\n--- run.log tail ---\n%s' \
        "\$NAME" "\$RC" "\$CMD" "\$PFX/" "\$(tail -n 40 /root/run.log 2>/dev/null)")" || true
  fi
  [ "$KEEP" = 1 ] || shutdown -h now || systemctl poweroff || halt -p
}
trap finish EXIT

apt-get update -y && apt-get install -y git python3-venv >> /root/pip.log 2>&1

DOCKER_IMAGE="$DOCKER_IMAGE"
DOCKER_ENV="$DOCKER_ENV"
if [ -n "\$DOCKER_IMAGE" ]; then
  # Docker path (2026-09-16, revised same day): the image holds ONLY
  # environment -- requirements.txt, compiled mamba-ssm/causal-conv1d fused
  # kernel, chb01 baked in (see Dockerfile.mamba) -- never repo code. Code
  # comes from a live git clone bind-mounted over /workspace at docker-run
  # time, so ANY code change (new pipeline, new flag, whatever) is
  # usable the instant it's pushed -- no image rebuild, no AMI re-bake.
  # Only a real dependency/environment change needs those. The dataset
  # (/root/mne_data inside the image) and installed packages live outside
  # /workspace, so the bind mount doesn't touch them.
  echo "DOCKER_IMAGE=\$DOCKER_IMAGE" >> /root/run.log
  nvidia-smi >> /root/run.log 2>&1 || echo "NO GPU" >> /root/run.log
  docker pull "\$DOCKER_IMAGE" >> /root/pip.log 2>&1 \
    && echo "docker pull ok" >> /root/run.log \
    || echo "docker pull FAILED (see pip.log)" >> /root/run.log
  git clone -b "$BRANCH" --depth 1 "$REPO_URL" /root/repo >> /root/pip.log 2>&1
  echo "code checkout: \$(cd /root/repo && git rev-parse --short HEAD)" >> /root/run.log
  mkdir -p /root/checkpoint
  aws s3 sync "\$CKPT_S3" /root/checkpoint || true
  echo "checkpoint dir has \$(ls /root/checkpoint 2>/dev/null | wc -l) file(s) at launch" >> /root/run.log
  ( while true; do
      aws s3 cp /root/run.log "\$PFX/run.log" 2>/dev/null
      aws s3 sync /root/checkpoint "\$CKPT_S3" 2>/dev/null
      sleep 20
    done ) &
  TAILER=\$!
  ARGS=\$(echo "\$CMD" | sed -E 's/^ *python[0-9.]* +//')
  term() { docker kill -s TERM eeg-run 2>/dev/null || true; }
  trap term TERM
  set +e
  # /root/repo mounted over /workspace shadows the image's own (absent)
  # code with the live checkout; --rm only discards the container layer,
  # not this host-side mount, so /root/repo/Epilepsy/results already has
  # whatever the run wrote -- no separate results_out mount/copy needed.
  docker run --name eeg-run --gpus all --rm \
    -v /root/checkpoint:/root/checkpoint \
    -v /root/repo:/workspace \
    -w /workspace -e PYTHONPATH=/workspace:/workspace/Epilepsy \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \$DOCKER_ENV "\$DOCKER_IMAGE" \
    python3 -u \$ARGS >> /root/run.log 2>&1 &
  JOB_PID=\$!
  wait \$JOB_PID
  RC=\$?
  set -e
  kill \$TAILER 2>/dev/null || true
  cd /root/repo
  REPO_DIR=/root/repo RC=\$RC RUN_NAME="$NAME" RUN_CMD="\$CMD" \
    RUN_LOG=/root/run.log RUN_STARTED_UTC="\$STARTED" \
    SESSION_NOTE="\$NOTE" DEPLOY_KEY_SSM="$DEPLOY_KEY_SSM" \
    bash scripts/promote_results.sh
  exit \$RC
fi

# DLAMI ships a torch+CUDA venv (usually /opt/pytorch) -- install the repo's
# extra deps INTO it rather than reinstalling torch from scratch (that path
# is broken: pip 22.0 on this AMI doesn't support --break-system-packages).
BASEPY=""
for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python \
         /opt/conda/bin/python /usr/local/bin/python3 python3; do
  if "\$c" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    BASEPY="\$c"; break
  fi
done
echo "BASEPY=\$BASEPY" >> /root/run.log

git clone -b "$BRANCH" "$REPO_URL" /root/repo
cd /root/repo
git rev-parse --short HEAD >> /root/run.log
nvidia-smi >> /root/run.log 2>&1 || echo "NO GPU" >> /root/run.log

grep -vE '^(torch|--extra-index-url|#|\$)' requirements.txt > /tmp/reqs.txt || true
if [ -n "\$BASEPY" ]; then
  PY="\$BASEPY"
  "\$PY" -m pip install -r /tmp/reqs.txt >> /root/pip.log 2>&1 \
    && echo "pip ok" >> /root/run.log || echo "pip FAILED (see pip.log)" >> /root/run.log
else
  echo "no torch python found -- bare venv fallback" >> /root/run.log
  python3 -m venv /root/venv && PY=/root/venv/bin/python
  "\$PY" -m pip install -U pip wheel >> /root/pip.log 2>&1 || true
  "\$PY" -m pip install -r requirements.txt >> /root/pip.log 2>&1 \
    && echo "pip ok" >> /root/run.log || echo "pip FAILED (see pip.log)" >> /root/run.log
fi
"\$PY" -c "import torch,moabb,mne;print('cuda',torch.cuda.is_available())" >> /root/run.log 2>&1

# Fused mamba-ssm CUDA kernel (2026-09-16) -- same recipe Dockerfile.mamba
# already proved works (versions, --no-build-isolation --no-deps to dodge
# the torch-ABI-mismatch bug documented there), just compiled directly on
# THIS box instead of baked into an image. Building here (not on a GPU-less
# CI runner) means no TORCH_CUDA_ARCH_LIST guess needed -- pip's build
# auto-detects whatever GPU is actually present (T4/A10G/L4, whichever
# candidate this launch landed on), so it isn't tied to Dockerfile.mamba's
# baked 8.0/8.6/8.9/9.0 list (which doesn't even cover T4's sm_75).
apt-get install -y --no-install-recommends build-essential ninja-build >> /root/pip.log 2>&1
nvcc --version >> /root/run.log 2>&1 || echo "NO NVCC" >> /root/run.log
export MAX_JOBS=4
"\$PY" -m pip install --no-build-isolation --no-deps \
  "causal-conv1d>=1.4.0" "mamba-ssm>=2.2.2" >> /root/pip.log 2>&1 \
  && echo "mamba-ssm compile: ok" >> /root/run.log \
  || echo "mamba-ssm compile: FAILED (see pip.log)" >> /root/run.log
"\$PY" -m pip install huggingface_hub transformers >> /root/pip.log 2>&1
# Loud, explicit check -- grep run.log for this exact marker, never assume.
"\$PY" -c "
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('MAMBA_SSM_CUDA_KERNEL_OK', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
" >> /root/run.log 2>&1 || echo "MAMBA_SSM_CUDA_KERNEL_FAILED" >> /root/run.log

mkdir -p /root/mne_data
# --exclude kuhlmann_nv/*: that subtree is fetched separately, into a
# different destination (/root/repo/datasets/epilepsy/kuhlmann_nv, see
# scripts/fetch_kuhlmann_nv_s3.sh) -- pulling its 2.9GB tarball here too
# would be pure waste (never read from /root/mne_data). --no-progress:
# without a TTY, aws cli prints a full new line per progress update
# instead of overwriting one line -- on a multi-GB sync that's tens of
# thousands of duplicate lines flooding boot.log (confirmed 2026-09-23,
# nv-pat1-smoke: buried the run.log section badly enough that
# eeg-tail.yml/get_job_logs couldn't surface the actual failure).
aws s3 sync --no-progress --exclude "kuhlmann_nv/*" "s3://$BUCKET/datasets" /root/mne_data || true
export MNE_DATA=/root/mne_data PYTHONPATH=/root/repo

# resume: pull any checkpoint left by a previous (interrupted) attempt at
# this same --name. Empty/nonexistent on a first launch -- fine, "sync" of
# nothing is a no-op and run_pipelines.py's --checkpoint-dir starts fresh.
mkdir -p /root/checkpoint
aws s3 sync "\$CKPT_S3" /root/checkpoint || true
echo "checkpoint dir has \$(ls /root/checkpoint 2>/dev/null | wc -l) file(s) at launch" >> /root/run.log

( while true; do
    aws s3 cp /root/run.log "\$PFX/run.log" 2>/dev/null
    aws s3 sync /root/checkpoint "\$CKPT_S3" 2>/dev/null
    sleep 20
  done ) &
TAILER=\$!

# spot reclaim sends SIGTERM ~2min before termination -- forward it so the
# job dies promptly and the trap/logs still ship, rather than being killed
# mid-upload by the hard poweroff.
term() { kill -TERM \$JOB_PID 2>/dev/null || true; }
trap term TERM

set +e
# strip a leading "python"/"python3" from --cmd; we supply the interpreter
ARGS=\$(echo "\$CMD" | sed -E 's/^ *python[0-9.]* +//')
# 2026-09-16 fix: was a truncating '>' redirect to /root/run.log -- silently wiped every
# boot-time diagnostic above (BASEPY, nvidia-smi, mamba-ssm compile result,
# the MAMBA_SSM_CUDA_KERNEL_OK/FAILED marker) the moment training started,
# so the one thing worth grepping for was never actually in the shipped log.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "\$PY" -u \$ARGS >> /root/run.log 2>&1 &
JOB_PID=\$!
wait \$JOB_PID
RC=\$?
set -e
kill \$TAILER 2>/dev/null || true

REPO_DIR=/root/repo RC=\$RC RUN_NAME="$NAME" RUN_CMD="\$CMD" \
  RUN_LOG=/root/run.log RUN_STARTED_UTC="\$STARTED" \
  SESSION_NOTE="\$NOTE" DEPLOY_KEY_SSM="$DEPLOY_KEY_SSM" \
  bash scripts/promote_results.sh
# finish() runs via trap: ships to S3, notifies on non-zero rc, terminates
EOF
)

BDM="[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"

for pair in "${CANDIDATES[@]}"; do
  ITYPE=${pair%%:*}; AZ=${pair##*:}
  MAXP=$(maxprice_for "$ITYPE")
  SUBNET_ID=$(subnet_for "$AZ")
  echo ">> trying spot $ITYPE in $AZ (max \$$MAXP)"
  IID=$(aws ec2 run-instances --region $REGION \
    --image-id "$AMI" --instance-type "$ITYPE" --key-name "$KEY" \
    --subnet-id "$SUBNET_ID" --security-group-ids "$SG" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --instance-market-options "MarketType=spot,SpotOptions={MaxPrice=$MAXP,SpotInstanceType=one-time}" \
    --instance-initiated-shutdown-behavior "$SHUTDOWN_BEHAVIOR" \
    --block-device-mappings "$BDM" \
    --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=eeg},{Key=Name,Value=eeg-run-spot-$NAME},{Key=role,Value=eeg-gpu}]" \
    --user-data "$UD" \
    --query 'Instances[0].InstanceId' --output text 2>/tmp/spoterr) && {
      echo ">> launched $IID ($ITYPE/$AZ)"
      echo "   watch:  aws s3 cp $PFX/run.log -"
      echo "   result: commit to origin/main on success; SNS on failure/interrupt; box self-terminates"
      exit 0
    }
  if grep -q "InsufficientInstanceCapacity\|MaxSpotInstanceCountExceeded" /tmp/spoterr; then
    echo "   no capacity, next candidate"; continue
  fi
  echo ">> launch failed (non-capacity):"; cat /tmp/spoterr; exit 1
done
echo ">> no spot capacity on any (type, AZ) candidate -- try again later."
exit 3
