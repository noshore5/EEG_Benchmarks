#!/bin/bash
# setup_spot_keepalive.sh -- ONE-TIME: grant the existing `eeg-gh-launcher`
# IAM role what .github/workflows/eeg-spot-keepalive.yml needs beyond what
# setup_gh_launcher.sh / setup_gh_launcher_tail.sh already gave it:
#   - s3 read+write+delete on checkpoints/*  (latest.pt, progress.json,
#     keepalive-state.json, DONE)
#   - s3:ListBucket restricted to checkpoints/*
#   - sns:Publish on the eeg-runs topic (crash-loop / give-up notification)
#
# The role already has (from the earlier setup scripts): RunInstances,
# ec2:Describe*, CreateTags-on-launch, Terminate/Stop gated to Project=eeg,
# PassRole eeg-gpu, and s3 read on exports/*. eeg-spot-train.sh's launch
# call reuses those; this script only adds the checkpoint bookkeeping.
#
# Run once, from a shell with admin AWS creds for account 827938107865.
# Idempotent -- overwrites just this one inline policy.
set -euo pipefail
ACCT=827938107865
BUCKET=noshore-eeg-benchmarks-827938107865
ROLE=eeg-gh-launcher
TOPIC=arn:aws:sns:us-east-1:${ACCT}:eeg-runs
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  echo "role $ROLE does not exist -- run setup_gh_launcher.sh first" >&2
  exit 1
fi

echo "== inline policy eeg-gh-spot-keepalive on $ROLE =="
cat > "$TMP/pol.json" <<JSON
{ "Version": "2012-10-17", "Statement": [
  { "Sid": "ListCheckpointsPrefixOnly", "Effect": "Allow",
    "Action": "s3:ListBucket",
    "Resource": "arn:aws:s3:::${BUCKET}",
    "Condition": { "StringLike": { "s3:prefix": "checkpoints/*" } } },
  { "Sid": "RwCheckpoints", "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
    "Resource": "arn:aws:s3:::${BUCKET}/checkpoints/*" },
  { "Sid": "NotifyOnGiveUp", "Effect": "Allow",
    "Action": "sns:Publish",
    "Resource": "${TOPIC}" }
] }
JSON
aws iam put-role-policy --role-name "$ROLE" --policy-name eeg-gh-spot-keepalive \
  --policy-document "file://$TMP/pol.json"
echo "  attached"

cat <<DONE

== done ==
eeg-spot-keepalive.yml can now run. The scheduled cron (*/15) fires
automatically once this workflow is on the default branch. To start a run:
  gh workflow run eeg-spot-keepalive.yml -f job=szcore-detector \\
    -f cmd='python Epilepsy/szcore/train_detector.py --subjects 1 2 3 --epochs 10 --device cuda'
To restart a finished/failed run: add  -f reset=true
DONE
