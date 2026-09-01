#!/bin/bash
# promote_results.sh -- run ON a job box after a pipeline finishes. If the job
# succeeded, commit the new result CSV(s) (+ an optional templated session note)
# straight to origin/main and push. Purely additive: only ever `git add`s new
# files under Epilepsy/results/ and Epilepsy/Session_notes/, refuses if anything
# else in the tree is dirty, and only commits on rc==0.
#
# Caller sets these env vars (see scripts/eeg-run.sh and the *_userdata.sh):
#   REPO_DIR             path to the repo checkout            (default /root/repo)
#   RC                   exit code of the pipeline run        (required)
#   RUN_NAME             short slug, used in the commit msg + note filename
#   RUN_CMD              the exact command that was run       (for the note)
#   RUN_LOG              path to the captured run log         (default $REPO_DIR/../run.log)
#   RUN_STARTED_UTC      ISO8601, when the pipeline started   (optional)
#   SESSION_NOTE         if non-empty, render a note; the value is its "why" para
#   DEPLOY_KEY_SSM       SSM param name holding the write deploy key
#                        (default /eeg/github-deploy-key)
#   GEMINI_KEY_SSM       SSM param with a Google Gemini API key (default
#                        /eeg/gemini-api-key). If it resolves, the session note
#                        is LLM-written from the run log + context; otherwise it
#                        falls back to the deterministic template. Note failure
#                        never blocks the commit.
#   GEMINI_MODEL         (default gemini-2.0-flash)
#   GIT_REMOTE_SSH       (default git@github.com:noshore5/EEG_Benchmarks.git)
#
# Exit codes: 0 promoted (or nothing to promote / rc!=0 -> deliberate no-op),
#             3 refused (tree dirty outside whitelist), 4 push failed.

set -uo pipefail

REPO_DIR=${REPO_DIR:-/root/repo}
RC=${RC:?promote_results.sh: RC (job exit code) not set}
RUN_NAME=${RUN_NAME:-run}
RUN_CMD=${RUN_CMD:-unknown}
RUN_LOG=${RUN_LOG:-$REPO_DIR/../run.log}
RUN_STARTED_UTC=${RUN_STARTED_UTC:-}
SESSION_NOTE=${SESSION_NOTE:-}
DEPLOY_KEY_SSM=${DEPLOY_KEY_SSM:-/eeg/github-deploy-key}
GEMINI_KEY_SSM=${GEMINI_KEY_SSM:-/eeg/gemini-api-key}
GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.0-flash}
GIT_REMOTE_SSH=${GIT_REMOTE_SSH:-git@github.com:noshore5/EEG_Benchmarks.git}

WL_RE='^(Epilepsy/results/|Epilepsy/Session_notes/)'   # promote whitelist

say() { echo "$(date -u +%H:%M:%S) promote: $*"; }

if [ "$RC" != "0" ]; then
  say "job rc=$RC -- not committing anything (results still go to S3 by the caller)"
  exit 0
fi

cd "$REPO_DIR" || { say "no repo at $REPO_DIR"; exit 3; }

# --- what did the run produce under the whitelist? ---
mapfile -t NEW < <(git status --porcelain -- Epilepsy/results/ Epilepsy/Session_notes/ \
                     | sed 's/^...//' | sed 's/^"//; s/"$//')

# --- refuse if the run dirtied anything tracked OUTSIDE the whitelist ---
DIRTY_OUTSIDE=$(git status --porcelain | sed 's/^...//' | sed 's/^"//; s/"$//' \
                 | grep -vE "$WL_RE" || true)
if [ -n "$DIRTY_OUTSIDE" ]; then
  say "REFUSING to commit -- run modified tracked files outside results/ & Session_notes/:"
  echo "$DIRTY_OUTSIDE" | sed 's/^/    /'
  exit 3
fi

# --- optional session note (LLM-written if a Gemini key is in SSM, else templated) ---
if [ -n "$SESSION_NOTE" ]; then
  D=$(date -u +%Y_%m_%d)
  mkdir -p "Epilepsy/Session_notes/$D"
  NOTE="Epilepsy/Session_notes/$D/${RUN_NAME}.md"
  SHA=$(git rev-parse --short HEAD)
  ENDED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  WALL="n/a"
  if [ -n "$RUN_STARTED_UTC" ]; then
    S=$(date -u -d "$RUN_STARTED_UTC" +%s 2>/dev/null || echo "")
    [ -n "$S" ] && WALL="$(( ($(date -u +%s) - S) / 60 )) min"
  fi
  HOST="$(nproc) vCPU, $(free -g 2>/dev/null | awk '/Mem:/{print $2}') GB RAM"
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  [ -n "$GPU" ] && HOST="$HOST, GPU $GPU"

  # facts block -- appended to the note in both modes, and fed to the LLM as context
  FACTS=$(mktemp)
  {
    echo "## Run"
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| command | \`$RUN_CMD\` |"
    echo "| repo | \`$SHA\` |"
    echo "| started | ${RUN_STARTED_UTC:-n/a} |"
    echo "| ended | $ENDED |"
    echo "| wall | $WALL |"
    echo "| host | $HOST |"
    echo "| exit | rc=$RC |"
    echo
    echo "## Result files"
    echo
    for f in "${NEW[@]}"; do
      [ "$f" = "$NOTE" ] && continue
      echo "### \`$f\`"
      echo
      if [[ "$f" == *.csv ]] && command -v python3 >/dev/null; then
        echo '```'
        python3 - "$f" <<'PY' 2>/dev/null || echo "(could not render)"
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print(df.to_string(index=False, max_rows=60))
if len(df) > 1:
    num = df.select_dtypes("number")
    if not num.empty:
        print("\nmean:")
        print(num.mean().to_string())
PY
        echo '```'
      fi
      echo
    done
    if [ -f "$RUN_LOG" ]; then
      echo "## run.log (tail)"
      echo
      echo '```'
      tail -n 60 "$RUN_LOG"
      echo '```'
    fi
  } > "$FACTS"

  # try LLM prose; on any failure fall back to the deterministic template
  GEMINI_KEY=$(aws ssm get-parameter --name "$GEMINI_KEY_SSM" --with-decryption \
                 --query 'Parameter.Value' --output text 2>/dev/null || true)
  LLM_BODY=""
  if [ -n "$GEMINI_KEY" ] && [ "$GEMINI_KEY" != "None" ] && command -v python3 >/dev/null; then
    PROMPT=$(printf '%s\n\nYou are writing a lab session note for the EEG_Benchmarks repo. Match the terse, factual style of an experienced ML researcher: what was tried, why, what the numbers were, what it means, what to do next. Markdown, start with "# %s", ~150-350 words, no preamble. The run intent and raw facts follow.\n\nINTENT: %s\n\nFACTS:\n%s\n' \
              "" "$RUN_NAME" "$SESSION_NOTE" "$(cat "$FACTS")")
    LLM_BODY=$(GEMINI_KEY="$GEMINI_KEY" GEMINI_MODEL="$GEMINI_MODEL" PROMPT="$PROMPT" python3 <<'PY' 2>/dev/null || true
import json, os, urllib.request
key, model, prompt = os.environ["GEMINI_KEY"], os.environ["GEMINI_MODEL"], os.environ["PROMPT"]
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=60))
print(r["candidates"][0]["content"]["parts"][0]["text"].strip())
PY
)
  fi

  {
    if [ -n "$LLM_BODY" ]; then
      echo "$LLM_BODY"
      echo
      echo "---"
      echo "_Autonomous \`eeg-run\`; note drafted by $GEMINI_MODEL, committed by \`promote_results.sh\`. Facts below are machine-generated._"
    else
      echo "# ${RUN_NAME}"
      echo
      echo "_Autonomous \`eeg-run\`; templated by \`promote_results.sh\` (no LLM key / call failed)._"
      echo
      echo "$SESSION_NOTE"
    fi
    echo
    cat "$FACTS"
  } > "$NOTE"
  rm -f "$FACTS"
  NEW+=("$NOTE")
  say "wrote $NOTE ($([ -n "$LLM_BODY" ] && echo "$GEMINI_MODEL" || echo templated))"
fi

if [ ${#NEW[@]} -eq 0 ]; then
  say "run produced no new files under the whitelist -- nothing to promote"
  exit 0
fi

# --- auth: pull the write deploy key from SSM, use it just for this push ---
mkdir -p /root/.ssh
if ! aws ssm get-parameter --name "$DEPLOY_KEY_SSM" --with-decryption \
       --query 'Parameter.Value' --output text > /root/.ssh/eeg_deploy 2>/dev/null; then
  say "could not read deploy key from SSM ($DEPLOY_KEY_SSM) -- cannot push"
  exit 4
fi
chmod 600 /root/.ssh/eeg_deploy
export GIT_SSH_COMMAND="ssh -i /root/.ssh/eeg_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
git remote set-url origin "$GIT_REMOTE_SSH"
git config user.name  "eeg-autorun"
git config user.email "eeg-autorun@users.noreply.github.com"

git add -- "${NEW[@]}"
git commit -q -m "auto: ${RUN_NAME} results" \
  -m "$(printf 'files:\n%s\n\nAuto-Run: %s\nInstance: %s' \
        "$(printf '  %s\n' "${NEW[@]}")" "$RUN_NAME" \
        "$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)")"

# --- push to main with pull-rebase retries (other shells push here too) ---
for i in $(seq 1 6); do
  if git fetch -q origin main && git rebase -q origin/main && git push -q origin HEAD:main; then
    say "pushed ${RUN_NAME} -> origin/main ($(git rev-parse --short HEAD))"
    exit 0
  fi
  git rebase --abort 2>/dev/null
  say "push attempt $i failed, retrying"
  sleep $(( (RANDOM % 8) + 3 ))
done
say "push failed after 6 attempts -- results are in S3, commit is local only"
exit 4
