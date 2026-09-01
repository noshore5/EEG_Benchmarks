#!/usr/bin/env bash
# Run this ONCE from a machine with the `claude` admin IAM user (e.g. the Mac).
# Grants the eeg-box instance role the EC2-launch + SSM perms it needs, and
# gives the eeg-gpu role SSM so we can shell into the GPU box.
set -euo pipefail

BUCKET=s3://noshore-eeg-benchmarks-827938107865

aws s3 cp "$BUCKET/exports/eeg-box-ec2-policy.json" /tmp/eeg-box-ec2-policy.json

aws iam put-role-policy \
  --role-name eeg-box \
  --policy-name ec2-launch-and-ssm \
  --policy-document file:///tmp/eeg-box-ec2-policy.json

aws iam attach-role-policy \
  --role-name eeg-gpu \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

echo "Done. eeg-box can now launch/terminate Project=eeg instances and SSM into them."
