#!/bin/bash
# eeg-run.sh -- fire-and-forget a pipeline run on an on-demand EC2 box.
#
# The box: clones the repo, installs deps, runs your command, and then
#   - on success: commits the new result CSV(s) (+ an optional templated
#     session note) straight to origin/main and pushes  (promote_results.sh)
#   - always: ships run.log + full results tree to S3
#   - on failure: sends one SNS notification
#   - then TERMINATES itself (no babysitting shell needed)
#
# Usage:
#   scripts/eeg-run.sh --name godoy-pred-6fold \
#     --cmd 'python Epilepsy/run_pipelines.py --pipeline godoy_tmc --label-mode prediction --device cuda' \
#     [--session-note 'Why this run: checking whether prediction labels ...'] \
#     [--gpu | --cpu] [--type g5.2xlarge] [--disk 150] [--branch main] [--keep]
#
# Needs AWS creds for account 827938107865 (aws sts get-caller-identity).

set -euo pipefail

REGION=us-east-1
BUCKET=noshore-eeg-benchmarks-827938107865
REPO_URL=https://github.com/noshore5/EEG_Benchmarks.git
KEY=eeg-box
PROFILE_NAME=eeg-gpu                       # instance profile: S3 RW + SSM param read
SNS_TOPIC_ARN=${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:827938107865:eeg-runs}
DEPLOY_KEY_SSM=${DEPLOY_KEY_SSM:-/eeg/github-deploy-key}

NAME=""; CMD=""; NOTE=""; KIND=gpu; ITYPE=""; DISK=""; BRANCH=main; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name)         NAME=$2; shift 2;;
    --cmd)          CMD=$2; shift 2;;
    --session-note) NOTE=$2; shift 2;;
    --gpu)          KIND=gpu; shift;;
    --cpu)          KIND=cpu; shift;;
    --type)         ITYPE=$2; shift 2;;
    --disk)         DISK=$2; shift 2;;
    --branch)       BRANCH=$2; shift 2;;
    --keep)         KEEP=1; shift;;              # debug: don't self-terminate
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$NAME" ] || { echo "--name required" >&2; exit 2; }
[ -n "$CMD" ]  || { echo "--cmd required"  >&2; exit 2; }
NAME=$(echo "$NAME" | tr -c 'A-Za-z0-9._-' '-')

if [ "$KIND" = gpu ]; then
  ITYPE=${ITYPE:-g5.2xlarge}; DISK=${DISK:-150}
  AMI=$(aws ec2 describe-images --owners amazon --region $REGION \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
              "Name=state,Values=available" \
    --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
else
  ITYPE=${ITYPE:-c7i.2xlarge}; DISK=${DISK:-60}
  AMI=$(aws ec2 describe-images --owners 099720109477 --region $REGION \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
fi

SG=$(aws ec2 describe-security-groups --region $REGION \
  --filters Name=group-name,Values=eeg-ssh \
  --query 'SecurityGroups[0].GroupId' --output text)

PFX="s3://$BUCKET/exports/runs/$NAME"
SHUTDOWN_BEHAVIOR=terminate; [ "$KEEP" = 1 ] && SHUTDOWN_BEHAVIOR=stop

# base64 the free-text args so quotes/newlines/$ in them can't break the script
CMD_B64=$(printf '%s' "$CMD"  | base64 | tr -d '\n')
NOTE_B64=$(printf '%s' "$NOTE" | base64 | tr -d '\n')

# ---- user-data ---------------------------------------------------------------
UD=$(cat <<EOF
#!/bin/bash
set -x
exec > /var/log/eeg-run.log 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
export PATH=/usr/local/bin:/usr/bin:/bin:/snap/bin
BUCKET=$BUCKET
PFX="$PFX"
NAME="$NAME"
CMD=\$(echo $CMD_B64 | base64 -d)
NOTE=\$(echo $NOTE_B64 | base64 -d)
STARTED=\$(date -u +%Y-%m-%dT%H:%M:%SZ)

which aws >/dev/null 2>&1 || {
  apt-get update -y && apt-get install -y unzip curl git
  curl -sS https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/a.zip
  ( cd /tmp && unzip -q a.zip && ./aws/install )
}
hash -r

finish() {
  RC=\${RC:-1}
  aws s3 cp /var/log/eeg-run.log "\$PFX/eeg-run.log" || true
  [ -f /root/run.log ] && aws s3 cp /root/run.log "\$PFX/run.log" || true
  [ -d /root/repo/Epilepsy/results ] && aws s3 cp --recursive /root/repo/Epilepsy/results "\$PFX/results" || true
  if [ "\$RC" != 0 ]; then
    aws sns publish --region $REGION --topic-arn "$SNS_TOPIC_ARN" \
      --subject "eeg-run FAILED: \$NAME (rc=\$RC)" \
      --message "\$(printf 'run %s failed rc=%s\ncmd: %s\nlogs: %s\n\n--- run.log tail ---\n%s' \
        "\$NAME" "\$RC" "\$CMD" "\$PFX/" "\$(tail -n 40 /root/run.log 2>/dev/null)")" || true
  fi
  [ "$KEEP" = 1 ] || shutdown -h now
}
trap finish EXIT

git clone -b "$BRANCH" "$REPO_URL" /root/repo
cd /root/repo
SHA=\$(git rev-parse --short HEAD)

# deps: DLAMI/CPU both -> system python, torch wheel bundles CUDA
python3 -m pip install --break-system-packages --ignore-installed -r requirements.txt

mkdir -p /root/mne_data
aws s3 sync "s3://$BUCKET/datasets" /root/mne_data || true
export MNE_DATA=/root/mne_data PYTHONPATH=/root/repo

# prefetch chb01 (S3 datasets/ is empty -> pull from PhysioNet once, outside the
# timed run) so run.log timing reflects compute, not download
python3 -c "import sys; sys.path.insert(0,'.'); from datasets.epilepsy import CHBMIT; CHBMIT().get_data(subjects=[1])" || true

( while true; do aws s3 cp /root/run.log "\$PFX/run.log" 2>/dev/null; sleep 20; done ) &
TAILER=\$!

set +e
bash -lc "\$CMD" > /root/run.log 2>&1
RC=\$?
set -e
kill \$TAILER 2>/dev/null || true

REPO_DIR=/root/repo RC=\$RC RUN_NAME="$NAME" RUN_CMD="\$CMD" \
  RUN_LOG=/root/run.log RUN_STARTED_UTC="\$STARTED" \
  SESSION_NOTE="\$NOTE" DEPLOY_KEY_SSM="$DEPLOY_KEY_SSM" \
  bash scripts/promote_results.sh
# finish() runs via trap: ships to S3, notifies on failure, terminates
EOF
)

BDM="[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"

echo ">> launching $KIND $ITYPE from $AMI  (name=$NAME, disk=${DISK}G, on-exit=$SHUTDOWN_BEHAVIOR)"
IID=$(aws ec2 run-instances --region $REGION \
  --image-id "$AMI" --instance-type "$ITYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --iam-instance-profile "Name=$PROFILE_NAME" \
  --instance-initiated-shutdown-behavior "$SHUTDOWN_BEHAVIOR" \
  --block-device-mappings "$BDM" \
  --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=eeg},{Key=Name,Value=eeg-run-$NAME},{Key=role,Value=eeg-gpu}]" \
  --user-data "$UD" \
  --query 'Instances[0].InstanceId' --output text)

echo ">> $IID launched"
echo "   watch:   aws s3 cp $PFX/run.log -  (updates every ~20s once deps are in)"
echo "   result:  commit to origin/main on success; SNS on failure; box self-terminates"
