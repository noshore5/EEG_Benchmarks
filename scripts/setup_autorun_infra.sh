#!/bin/bash
# setup_autorun_infra.sh -- ONE-TIME setup for autonomous eeg-run result promotion.
# Run once, from a shell with admin AWS creds for account 827938107865 and a
# `gh` logged in as a noshore5/EEG_Benchmarks admin. Idempotent-ish (safe to
# re-run; it overwrites the SSM params and re-attaches the IAM policy).
#
# Creates:
#   - SNS topic `eeg-runs` + email subscription (confirm the email afterwards)
#   - a GitHub deploy key (write) on noshore5/EEG_Benchmarks
#   - SSM SecureString /eeg/github-deploy-key  (the deploy key's private half)
#   - SSM SecureString /eeg/gemini-api-key     (placeholder -- see last step)
#   - inline policy `eeg-ssm-and-sns` on roles eeg-gpu and eeg-box
#     (ssm:GetParameter on /eeg/*, sns:Publish on eeg-runs)
set -euo pipefail
REGION=us-east-1
ACCT=827938107865
EMAIL=${EMAIL:-noahtshore@gmail.com}
REPO=noshore5/EEG_Benchmarks
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== SNS topic + email sub =="
TOPIC=$(aws sns create-topic --name eeg-runs --region $REGION --query TopicArn --output text)
aws sns subscribe --region $REGION --topic-arn "$TOPIC" --protocol email \
  --notification-endpoint "$EMAIL" --query SubscriptionArn --output text
echo "  $TOPIC  (confirm the subscription email)"

echo "== GitHub deploy key (write) =="
ssh-keygen -t ed25519 -N '' -C 'eeg-autorun' -f "$TMP/k" >/dev/null
gh api -X POST "repos/$REPO/keys" \
  -f title='eeg-autorun (SSM /eeg/github-deploy-key)' \
  -f "key=$(cat "$TMP/k.pub")" -F read_only=false --jq '"  deploy key id \(.id)"'

echo "== SSM params =="
aws ssm put-parameter --region $REGION --name /eeg/github-deploy-key \
  --type SecureString --overwrite --value "$(cat "$TMP/k")" --query Version --output text
aws ssm put-parameter --region $REGION --name /eeg/gemini-api-key \
  --type SecureString --overwrite --value PLACEHOLDER --query Version --output text
echo "  /eeg/github-deploy-key set ; /eeg/gemini-api-key = PLACEHOLDER"

echo "== IAM: let eeg-gpu / eeg-box read /eeg/* and publish to eeg-runs =="
cat > "$TMP/pol.json" <<JSON
{ "Version": "2012-10-17", "Statement": [
  { "Effect": "Allow", "Action": ["ssm:GetParameter","ssm:GetParameters"],
    "Resource": "arn:aws:ssm:$REGION:$ACCT:parameter/eeg/*" },
  { "Effect": "Allow", "Action": "sns:Publish",
    "Resource": "arn:aws:sns:$REGION:$ACCT:eeg-runs" }
] }
JSON
for R in eeg-gpu eeg-box; do
  aws iam put-role-policy --role-name "$R" --policy-name eeg-ssm-and-sns \
    --policy-document "file://$TMP/pol.json"
  echo "  attached to $R"
done

cat <<'DONE'

== done ==
Next:
  1. Click the confirmation link in the "AWS Notification - Subscription
     Confirmation" email.
  2. Put your real Gemini API key in (templated notes work without this;
     this is only for LLM-written notes):
       aws ssm put-parameter --region us-east-1 --name /eeg/gemini-api-key \
         --type SecureString --overwrite --value 'YOUR_GEMINI_KEY'
  3. GPU on-demand quota is currently 0 -- `eeg-run --gpu` will fail at
     run-instances until the pending increase lands. `eeg-run --cpu` works now.
  4. Smoke test (no results file -> promote is a no-op; verifies launch,
     deps, S3 upload, self-terminate, and the failure-SNS path if you flip
     the command to `false`):
       scripts/eeg-run.sh --cpu --name smoke --cmd 'python -c "import torch,pandas; print(torch.__version__)"'
DONE
