#!/bin/bash
# setup_gh_launcher_tail.sh -- ONE-TIME: grant the existing `eeg-gh-launcher`
# IAM role (see setup_gh_launcher.sh) read-only access to this project's S3
# exports, so a Claude session with no AWS creds can tail a running box's
# log via GitHub Actions (.github/workflows/eeg-tail.yml) instead of being
# handed real credentials.
#
# Run once, from a shell with admin AWS creds for account 827938107865.
# Idempotent (safe to re-run; just overwrites this one inline policy --
# does not touch the eeg-gh-launch policy from setup_gh_launcher.sh).
#
# Adds inline policy `eeg-gh-tail-read` on role eeg-gh-launcher:
#   - s3:GetObject on exports/* only (no datasets/, no checkpoints/)
#   - s3:ListBucket on the bucket, prefix-restricted to exports/*
#   - ec2:DescribeInstances (already covered by eeg-gh-launch's Discover
#     statement, but harmless/idempotent to restate here for clarity)
# No write, no delete, no access outside exports/.
set -euo pipefail
ACCT=827938107865
BUCKET=noshore-eeg-benchmarks-827938107865
ROLE=eeg-gh-launcher
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  echo "role $ROLE does not exist -- run setup_gh_launcher.sh first" >&2
  exit 1
fi

echo "== inline policy eeg-gh-tail-read on $ROLE =="
cat > "$TMP/pol.json" <<JSON
{ "Version": "2012-10-17", "Statement": [
  { "Sid": "ListExportsPrefixOnly", "Effect": "Allow",
    "Action": "s3:ListBucket",
    "Resource": "arn:aws:s3:::${BUCKET}",
    "Condition": { "StringLike": { "s3:prefix": "exports/*" } } },
  { "Sid": "ReadExportsOnly", "Effect": "Allow",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::${BUCKET}/exports/*" }
] }
JSON
aws iam put-role-policy --role-name "$ROLE" --policy-name eeg-gh-tail-read \
  --policy-document "file://$TMP/pol.json"
echo "  attached"

cat <<DONE

== done ==
$ROLE can now read (list+get) s3://${BUCKET}/exports/* only -- no write,
no delete, nothing outside exports/. Test with:
  gh workflow run eeg-tail.yml -f run_name=smoke-launcher-verify
DONE
