#!/bin/bash
# eeg-spot-train.sh -- launch ONE interruptible spot GPU box that (re)starts a
# resumable SzCORE detector training run and self-terminates only on a CLEAN
# finish. Designed to be called repeatedly by eeg-spot-keepalive.yml until the
# run writes its DONE sentinel to S3. See SPOT_TRAINING.md.
#
#   scripts/eeg-spot-train.sh --job szcore-detector \
#     --cmd 'python Epilepsy/szcore/train_detector.py --subjects $(seq 1 24) --epochs 25 --device cuda'
#
# The --cmd is run with cwd=/root/repo; the script appends
#   --checkpoint-dir /root/ckpt --s3-prefix s3://$BUCKET/checkpoints/<job>
# so you do NOT put those in --cmd.
#
# Behaviour on the box:
#   * fresh venv (avoids the DLAMI's Ubuntu-22.04 system-pip / PEP668 mess
#     that broke eeg-run.sh's GPU path 2026-09-07)
#   * training auto-resumes from s3://.../checkpoints/<job>/latest.pt
#   * clean finish  -> trainer writes .../DONE, box terminates
#   * SIGTERM (spot) -> trainer checkpoints + exits 0, NO DONE, box is reclaimed
#     by AWS; the keepalive workflow launches the next one
#   * crash (rc!=0)  -> SNS notify, terminate, NO DONE (keepalive relaunches;
#     its crash-loop guard stops after 3 no-progress launches)
#
# Needs AWS creds for account 827938107865.

set -euo pipefail

REGION=us-east-1
BUCKET=noshore-eeg-benchmarks-827938107865
REPO_URL=https://github.com/noshore5/EEG_Benchmarks.git
KEY=eeg-box
PROFILE_NAME=eeg-gpu
SNS_TOPIC_ARN=${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:827938107865:eeg-runs}

# (instanceType, AZ) candidates, tried in order until one has spot capacity.
# g4dn first: on 2026-09-07 it was the only G family with us-east-1 spot stock.
CANDIDATES=(
  "g4dn.xlarge:us-east-1c"  "g4dn.xlarge:us-east-1d"  "g4dn.xlarge:us-east-1a"
  "g4dn.xlarge:us-east-1b"  "g4dn.xlarge:us-east-1f"
  "g5.xlarge:us-east-1a"    "g5.xlarge:us-east-1b"     "g5.xlarge:us-east-1c"
  "g5.xlarge:us-east-1d"    "g5.xlarge:us-east-1f"
  "g6.xlarge:us-east-1a"    "g6.xlarge:us-east-1b"     "g6.xlarge:us-east-1c"
)
# plain functions instead of `declare -A` -- macOS ships bash 3.2, no assoc arrays
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

JOB=""; CMD=""; DISK=150
while [ $# -gt 0 ]; do
  case "$1" in
    --job)  JOB=$2; shift 2;;
    --cmd)  CMD=$2; shift 2;;
    --disk) DISK=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$JOB" ] || { echo "--job required" >&2; exit 2; }
[ -n "$CMD" ] || { echo "--cmd required" >&2; exit 2; }
JOB=$(printf '%s' "$JOB" | tr -c 'A-Za-z0-9._-' '-')

PFX_S3="s3://$BUCKET/checkpoints/$JOB"
LOG_S3="s3://$BUCKET/exports/spot-train/$JOB"

# already finished?
if aws s3 ls "$PFX_S3/DONE" --region $REGION >/dev/null 2>&1; then
  echo ">> $PFX_S3/DONE exists -- run already complete, nothing to launch."
  exit 0
fi
# already running?
RUNNING=$(aws ec2 describe-instances --region $REGION \
  --filters "Name=tag:Project,Values=eeg" "Name=tag:spot-job,Values=$JOB" \
            "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$RUNNING" ]; then
  echo ">> a box for $JOB is already $RUNNING -- not launching another."
  exit 0
fi

AMI=$(aws ec2 describe-images --owners amazon --region $REGION \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
            "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
SG=$(aws ec2 describe-security-groups --region $REGION \
  --filters Name=group-name,Values=eeg-ssh --query 'SecurityGroups[0].GroupId' --output text)

CMD_B64=$(printf '%s' "$CMD" | base64 | tr -d '\n')

UD=$(cat <<EOF
#!/bin/bash
set -x
exec > /var/log/eeg-spot-train.log 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin:/snap/bin
# hard watchdog: terminate no matter what after 8h, even if the trap never fires
( sleep 28800; shutdown -h now ) &
BUCKET=$BUCKET
PFX_S3="$PFX_S3"
LOG_S3="$LOG_S3"
JOB="$JOB"
CMD=\$(echo $CMD_B64 | base64 -d)

finish() {
  RC=\${RC:-1}
  aws s3 cp /var/log/eeg-spot-train.log "\$LOG_S3/boot.log" || true
  [ -f /root/run.log ] && aws s3 cp /root/run.log "\$LOG_S3/run.log" || true
  [ -f /root/pip.log ] && aws s3 cp /root/pip.log "\$LOG_S3/pip.log" || true
  if [ "\$RC" != 0 ] && [ "\$RC" != 42 ]; then
    aws sns publish --region $REGION --topic-arn "$SNS_TOPIC_ARN" \
      --subject "eeg-spot-train \$JOB rc=\$RC" \
      --message "\$(printf '%s\n\n%s' "\$CMD" "\$(tail -n 40 /root/run.log 2>/dev/null)")" || true
  fi
  shutdown -h now || systemctl poweroff || halt -p
}
trap finish EXIT

# The DLAMI ships a torch+CUDA venv (usually /opt/pytorch). Install the
# repo's extra deps INTO it rather than rebuilding torch+moabb from scratch
# in a bare venv (that was 2026-09-08's pip rc=1). Discover the torch python
# defensively; fall back to a --system-site-packages venv off it.
apt-get update -y && apt-get install -y git python3-venv >> /root/pip.log 2>&1
BASEPY=""
for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python \
         /opt/conda/bin/python /usr/local/bin/python3 python3; do
  if "\$c" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    BASEPY="\$c"; break
  fi
done
echo "BASEPY=\$BASEPY" >> /root/run.log

git clone -b main "$REPO_URL" /root/repo
cd /root/repo
git rev-parse --short HEAD >> /root/run.log
nvidia-smi >> /root/run.log 2>&1 || echo "NO GPU" >> /root/run.log

# everything the SzCORE trainer needs that the DLAMI venv lacks (NOT torch)
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
"\$PY" -c "import torch,moabb,mne,epilepsy2bids;print('cuda',torch.cuda.is_available())" >> /root/run.log 2>&1

mkdir -p /root/ckpt
export PYTHONPATH=/root/repo MNE_DATA=/root/mne_data
mkdir -p /root/mne_data

( while true; do aws s3 cp /root/run.log "\$LOG_S3/run.log" 2>/dev/null; sleep 20; done ) &
TAILER=\$!

# spot interruption: forward SIGTERM to the trainer so it checkpoints
term() { kill -TERM \$TRAIN_PID 2>/dev/null || true; }
trap term TERM

set +e
# strip a leading "python"/"python3" from --cmd; we supply the interpreter
ARGS=\$(echo "\$CMD" | sed -E 's/^ *python[0-9.]* +//')
"\$PY" -u \$ARGS --checkpoint-dir /root/ckpt --s3-prefix "\$PFX_S3" >> /root/run.log 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID
RC=\$?
set -e
kill \$TAILER 2>/dev/null || true
echo "trainer rc=\$RC" >> /root/run.log
# RC 0 = clean finish (DONE written by trainer). RC 0 after SIGTERM also fine.
# finish() trap ships logs + terminates either way.
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
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "$BDM" \
    --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=eeg},{Key=Name,Value=eeg-spot-$JOB},{Key=spot-job,Value=$JOB}]" \
    --user-data "$UD" \
    --query 'Instances[0].InstanceId' --output text 2>/tmp/spoterr) && {
      echo ">> launched $IID ($ITYPE/$AZ)"
      echo "   log:   aws s3 cp $LOG_S3/run.log -"
      echo "   ckpt:  aws s3 ls $PFX_S3/"
      exit 0
    }
  if grep -q "InsufficientInstanceCapacity\|MaxSpotInstanceCountExceeded" /tmp/spoterr; then
    echo "   no capacity, next candidate"; continue
  fi
  echo ">> launch failed (non-capacity):"; cat /tmp/spoterr; exit 1
done
echo ">> no spot capacity on any candidate -- keepalive will retry next tick."
exit 3
