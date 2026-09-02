#!/bin/bash
# setup_gh_launcher.sh -- ONE-TIME setup so the `eeg-run` GitHub Actions
# workflow (.github/workflows/eeg-run.yml) can launch EC2 boxes with NO
# stored credentials, via GitHub OIDC federation.
#
# Run once, from a shell with admin AWS creds for account 827938107865.
# Idempotent (safe to re-run; overwrites the role policy, re-creates the
# OIDC provider only if missing).
#
# Creates:
#   - IAM OIDC provider for token.actions.githubusercontent.com (if absent)
#   - IAM role `eeg-gh-launcher`, assumable ONLY by this repo's workflows
#   - inline policy `eeg-gh-launch` on it: just enough to run scripts/eeg-run.sh
#     (RunInstances + Describe* + CreateTags on launch + PassRole eeg-gpu +
#      TerminateInstances limited to Project=eeg-tagged boxes)
#
# After this runs, launching from anywhere is:
#   gh workflow run eeg-run.yml -f name=<slug> -f kind=gpu \
#     -f cmd='python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba --label-mode prediction --device cuda --seed 42 --validation-split 0 --epochs 20'
#   ...or the GitHub mobile app / any browser: Actions -> "eeg-run" -> Run workflow.
set -euo pipefail
REGION=us-east-1
ACCT=827938107865
REPO=noshore5/EEG_Benchmarks
ROLE=eeg-gh-launcher
OIDC_HOST=token.actions.githubusercontent.com
OIDC_ARN="arn:aws:iam::${ACCT}:oidc-provider/${OIDC_HOST}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "== GitHub OIDC provider =="
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  echo "  exists: $OIDC_ARN"
else
  # thumbprint is ignored for this provider by STS now, but the API still wants one
  aws iam create-open-id-connect-provider \
    --url "https://${OIDC_HOST}" \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
    --query OpenIDConnectProviderArn --output text
  echo "  created: $OIDC_ARN"
fi

echo "== IAM role $ROLE (trusts only repo:$REPO) =="
# NB: this account's OIDC subject claim is customised -- the sub GitHub
# actually presents is  repo:<owner>@<ownerid>/<repo>@<repoid>:ref:refs/...
# not the vanilla  repo:<owner>/<repo>:...  -- so match on :repository (exact)
# plus a wildcard :sub (AWS requires a sub or job_workflow_ref condition).
cat > "$TMP/trust.json" <<JSON
{ "Version": "2012-10-17", "Statement": [
  { "Effect": "Allow",
    "Principal": { "Federated": "$OIDC_ARN" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "${OIDC_HOST}:aud": "sts.amazonaws.com",
        "${OIDC_HOST}:repository": "${REPO}"
      },
      "StringLike": { "${OIDC_HOST}:sub": "repo:${REPO%%/*}@*/${REPO#*/}@*:*" }
    } }
] }
JSON
if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE" --policy-document "file://$TMP/trust.json"
  echo "  trust policy updated"
else
  aws iam create-role --role-name "$ROLE" \
    --description "GitHub Actions eeg-run.yml -> launch EC2 training boxes (OIDC)" \
    --assume-role-policy-document "file://$TMP/trust.json" \
    --query 'Role.Arn' --output text
fi

echo "== inline policy eeg-gh-launch =="
cat > "$TMP/pol.json" <<JSON
{ "Version": "2012-10-17", "Statement": [
  { "Sid": "Discover", "Effect": "Allow",
    "Action": ["ec2:DescribeImages","ec2:DescribeSecurityGroups",
               "ec2:DescribeInstances","ec2:DescribeInstanceStatus",
               "ec2:DescribeSubnets","ec2:DescribeVpcs"],
    "Resource": "*" },
  { "Sid": "Launch", "Effect": "Allow",
    "Action": "ec2:RunInstances",
    "Resource": "*" },
  { "Sid": "TagOnLaunch", "Effect": "Allow",
    "Action": "ec2:CreateTags",
    "Resource": "arn:aws:ec2:${REGION}:${ACCT}:*/*",
    "Condition": { "StringEquals": { "ec2:CreateAction": "RunInstances" } } },
  { "Sid": "KillOwnBoxes", "Effect": "Allow",
    "Action": ["ec2:TerminateInstances","ec2:StopInstances"],
    "Resource": "arn:aws:ec2:${REGION}:${ACCT}:instance/*",
    "Condition": { "StringEquals": { "ec2:ResourceTag/Project": "eeg" } } },
  { "Sid": "PassInstanceProfileRole", "Effect": "Allow",
    "Action": "iam:PassRole",
    "Resource": "arn:aws:iam::${ACCT}:role/eeg-gpu",
    "Condition": { "StringEquals": { "iam:PassedToService": "ec2.amazonaws.com" } } }
] }
JSON
aws iam put-role-policy --role-name "$ROLE" --policy-name eeg-gh-launch \
  --policy-document "file://$TMP/pol.json"
echo "  attached"

cat <<DONE

== done ==
Role ARN: arn:aws:iam::${ACCT}:role/${ROLE}
  (already hardcoded in .github/workflows/eeg-run.yml -- nothing to paste)

Test it:
  gh workflow run eeg-run.yml -f name=smoke -f kind=cpu \\
    -f cmd='python -c "import torch,pandas; print(torch.__version__)"'
  gh run watch \$(gh run list --workflow=eeg-run.yml -L1 --json databaseId --jq '.[0].databaseId')

The workflow job just launches the box (~30 s). Then watch the box itself:
  aws s3 cp s3://noshore-eeg-benchmarks-827938107865/exports/runs/smoke/run.log -

Once this works you can terminate eeg-box (i-083e3b55993a13c13) -- the
GH launcher replaces it. Check first whether it holds anything not in the
repo or S3 (cached data, local commits, keys).
DONE
