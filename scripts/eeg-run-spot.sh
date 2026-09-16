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

CANDIDATES=(
  "g4dn.xlarge:us-east-1c"  "g4dn.xlarge:us-east-1d"  "g4dn.xlarge:us-east-1a"
  "g4dn.xlarge:us-east-1b"  "g4dn.xlarge:us-east-1f"
  "g5.xlarge:us-east-1a"    "g5.xlarge:us-east-1b"     "g5.xlarge:us-east-1c"
  "g5.xlarge:us-east-1d"    "g5.xlarge:us-east-1f"
  "g6.xlarge:us-east-1a"    "g6.xlarge:us-east-1b"     "g6.xlarge:us-east-1c"
)
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
    g4dn.xlarge) echo 0.526;;
    g5.xlarge)   echo 1.006;;
    g6.xlarge)   echo 0.8048;;
    *)           echo 1.00;;
  esac
}

NAME=""; CMD=""; NOTE=""; DISK=150; BRANCH=main; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name)         NAME=$2; shift 2;;
    --cmd)          CMD=$2; shift 2;;
    --session-note) NOTE=$2; shift 2;;
    --disk)         DISK=$2; shift 2;;
    --branch)       BRANCH=$2; shift 2;;
    --keep)         KEEP=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$NAME" ] || { echo "--name required" >&2; exit 2; }
[ -n "$CMD" ]  || { echo "--cmd required"  >&2; exit 2; }
NAME=$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9._-' '-')

AMI=$(aws ec2 describe-images --owners amazon --region $REGION \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
            "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
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

mkdir -p /root/mne_data
aws s3 sync "s3://$BUCKET/datasets" /root/mne_data || true
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
"\$PY" -u \$ARGS > /root/run.log 2>&1 &
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
