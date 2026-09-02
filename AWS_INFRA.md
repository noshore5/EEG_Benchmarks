# AWS infrastructure

**Who this is for:** any agent shell that touches the shared AWS account --
whether it runs *on* an ephemeral AWS box (an `eeg-run` worker, a GPU box)
or drives the cloud remotely from a local Mac / Grok / cloud-Claude shell.
It describes shared cloud state, not repo state, and nothing in the
pipelines reads from it yet -- so if you're only editing code or running
tests, you can still skip it.

**The standing `eeg-box` (t3.small, ~$15/mo) was terminated 2026-09-02.**
Its job as a launcher is now done by GitHub Actions (`eeg-run.yml`, OIDC,
no stored creds -- see "Launch from anywhere" below); there is no longer a
persistent shell box. Every AWS box is now ephemeral and self-terminating.

**Can this shell touch AWS?** You need credentials for account
`827938107865` -- check with `aws sts get-caller-identity`.

- **On an ephemeral box** (`eeg-run` worker / GPU box): the instance
  profile (`eeg-gpu`) supplies them automatically, and *scoped* -- no keys
  on disk. See "IAM" for what the role can and can't do.
- **Local / off-AWS shell:** needs the `aws` CLI + a profile in
  `~/.aws/credentials`. The Mac this repo is usually worked from has the
  `claude` IAM user configured -- **AdministratorAccess**.
- **Grok / cloud-Claude shell with no creds:** can't launch anything
  directly -- trigger the `eeg-run.yml` workflow (`gh workflow run`, or
  GitHub mobile/web) or drop a job for the S3 runner.

**Account:** `827938107865` &nbsp; **Region:** `us-east-1`
**Last verified:** 2026-09-02, by Claude -- `eeg-run.yml` launch-from-
anywhere path verified end-to-end (OIDC -> launch -> deps -> run -> S3 ->
self-terminate, no orphan) on a CPU smoke run; `eeg-box` terminated the
same day. Spot SLR created by admin 2026-09-01. GPU quota-increase
requests still `CASE_OPENED` (AWS support queue; effective quota 0).

## Where to run cloud ops from

Launching or starting an instance is a one-shot API call -- the instance
doesn't care which shell made it. What matters is what happens *after* the
shell goes away:

- **Preferred: the `eeg-run.yml` workflow.** `gh workflow run
  eeg-run.yml -f name=... -f cmd=...` (or GitHub mobile/web). Needs no AWS
  creds anywhere -- authenticates via OIDC. The launched box does
  everything itself and self-terminates; nothing to babysit.
- **One-shot calls** (`run-instances`, `terminate-instances`, `aws s3
  cp`, `start-instances` the CPU box): run them from wherever you have
  creds. Local Mac is fine.
- **Anything that needs babysitting** (tailing a run, a poll loop): don't
  anchor it to a laptop. Use the S3 job-runner (survives disconnection) or
  let the `eeg-run` box handle it and watch the S3 `run.log`.
- **Credential hygiene:** the Mac authenticates as the `claude` user
  (**AdministratorAccess**) -- broad privilege sitting in a laptop's
  `~/.aws/credentials`. The workflow path avoids this entirely; prefer it.
  If direct local launches become routine, have an admin cut a scoped
  `eeg-launcher` IAM user mirroring the `eeg-gh-launcher` role's policy.

## Every launched box must self-terminate and ship results to S3

Non-negotiable for any instance you start (raw `run-instances` or a
`gpu-run`-style helper): it cleans up after itself **without needing the
launching shell to stay alive**. A laptop that kicks off a 3-hour GPU job
and then sleeps must not leave a paid box running.

**Preferred -- self-terminating user-data.** The box terminates *itself*
at the end of its user-data; nothing external has to notice it finished.
This is what `_setup/gpu_userdata.sh` already does. Launch it with:

- `--instance-initiated-shutdown-behavior terminate` -- so the script's
  `shutdown -h now` terminates, not stops. (The persistent `eeg-cpu-box`
  is the deliberate exception: it uses `stop` + an idle timer.)
- `--block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":<GB>,"VolumeType":"gp3","DeleteOnTermination":true}}]'`
  -- no orphaned EBS.
- instance profile `eeg-gpu` (S3 RW, no keys) and tag `Project=eeg`.

user-data skeleton -- the `trap ... EXIT` is the whole point: results ship
and the box dies **even if the job crashes or an earlier line errors**:

```bash
#!/bin/bash
set -x; exec > /var/log/eeg-job.log 2>&1
export HOME=/root PATH=/usr/local/bin:/usr/bin:/bin
BUCKET=noshore-eeg-benchmarks-827938107865
PFX="s3://$BUCKET/exports/<run-name>"
trap 'aws s3 cp /var/log/eeg-job.log "$PFX/eeg-job.log";
      aws s3 cp /root/run.log        "$PFX/run.log" 2>/dev/null;
      aws s3 cp --recursive /root/repo/Epilepsy/results "$PFX/results" 2>/dev/null;
      shutdown -h now' EXIT
which aws >/dev/null 2>&1 || { apt-get update -y && apt-get install -y unzip curl &&
  curl -sS https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/a.zip &&
  ( cd /tmp && unzip -q a.zip && ./aws/install ); }
git clone https://github.com/noshore5/EEG_Benchmarks.git /root/repo
cd /root/repo && python3 -m pip install --ignore-installed -r requirements.txt
( while true; do aws s3 cp /root/run.log "$PFX/run.log" 2>/dev/null; sleep 20; done ) &
PYTHONPATH=/root/repo python3 <JOB ...> > /root/run.log 2>&1
# trap fires here -> upload + terminate
```

**Fallback -- launcher-side trap** (`trap cleanup EXIT` terminates the box
when the launching script exits). **Do not rely on it from a laptop** --
if the shell dies the box leaks. Use self-terminating user-data instead
(which is what `eeg-run.sh` does).

**Results go to S3, never box-to-box.** There's no SSH trust or stable
address between ephemeral boxes. Everything lands in the bucket:
`exports/runs/<name>/` for `eeg-run` logs + results, `exports/<run-name>/`
for ad-hoc jobs, `checkpoints/` for weights. Your Mac then `aws s3 sync`s
from there.

**Orphan backstop.** Nothing sweeps a box orphaned by a hard crash
(user-data never ran, kernel panic). Until a terminator Lambda exists,
after launching check back with:
`aws ec2 describe-instances --filters Name=tag:Project,Values=eeg Name=instance-state-name,Values=running --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]' --output text`

## Autonomous runs -- `eeg-run` (fire-and-forget, self-committing)

`scripts/eeg-run.sh` is the fire-and-forget path: launch it and the result
CSV(s) (and, optionally, a session note) land as a commit on `origin/main`
with **no human step**. Flow:

```
scripts/eeg-run.sh --cpu --name godoy-pred-6fold \
  --cmd 'python Epilepsy/run_pipelines.py --pipeline godoy_tmc --label-mode prediction --device cpu' \
  --session-note 'Why: real 6-fold prediction-label run to confirm the val_split=0 leak fix holds.'
```

On the box: clone repo -> `pip install` -> run `--cmd` -> then
`scripts/promote_results.sh`:

- **`rc != 0`** -> ship logs to S3, one **SNS email** (`eeg-runs` topic),
  terminate. **No commit.**
- **`rc == 0`** -> `git add` **only** new files under `Epilepsy/results/`
  and `Epilepsy/Session_notes/` (refuses if the run dirtied any tracked
  file outside those two dirs), commit as `eeg-autorun`, `git push` to
  `main` with up to 6 pull-rebase retries. Then ship full logs + results
  tree to S3 and terminate.

Purely additive by construction -- pipelines write timestamped CSVs
(`..._<run_id>.csv`), so promotion never edits an existing file and
effectively never conflicts. A bad run can at worst add a results folder
you delete; it can't corrupt anything you pull.

**Session note** (`--session-note "<why>"`): LLM-written if
`/eeg/gemini-api-key` holds a real key, else a deterministic template
(run config + metrics table + `run.log` tail). Note generation never
blocks the commit.

**Credentials** (one-time, `scripts/setup_autorun_infra.sh`):
- SSM SecureString **`/eeg/github-deploy-key`** -- a `contents:write`
  GitHub deploy key, repo-scoped, revocable from the repo's Deploy Keys
  page. The box pulls it only for the push.
- SSM SecureString **`/eeg/gemini-api-key`** -- optional, for LLM notes.
- Roles `eeg-gpu` / `eeg-box` inline policy **`eeg-ssm-and-sns`**:
  `ssm:GetParameter` on `/eeg/*` + `sns:Publish` on `eeg-runs`.
- SNS topic **`eeg-runs`** -> email; the only notification you get (on
  failure). Success = the commit.

**GPU note:** `--gpu` fails at `run-instances` until the pending G/VT
quota increase lands (see "If you're bringing up the GPU box"). `--cpu`
works now (Ubuntu 24.04, on-demand, 8-vCPU limit applies).

### Launch from anywhere -- GitHub Actions (`.github/workflows/eeg-run.yml`)

No AWS creds needed on the launching machine. The workflow authenticates
to AWS via **GitHub OIDC** (role `eeg-gh-launcher`, assumable only by this
repo's workflows -- nothing stored anywhere) and runs `scripts/eeg-run.sh`
for you. The job just launches the box (~30 s) then exits; the box does
the rest exactly as above.

```
gh workflow run eeg-run.yml -f name=<slug> -f kind=gpu \
  -f cmd='python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba --label-mode prediction --device cuda --seed 42 --validation-split 0 --epochs 20' \
  -f session_note='Why: ...'
```

...or GitHub mobile app / any browser: **Actions -> "eeg-run" -> Run
workflow**. This is the path that replaces `eeg-box` as a launcher.

One-time setup: `scripts/setup_gh_launcher.sh` (admin AWS creds; creates
the OIDC provider + `eeg-gh-launcher` role + a tight inline policy --
RunInstances, Describe\*, CreateTags-on-launch, PassRole `eeg-gpu`,
Terminate/Stop limited to `Project=eeg` boxes).

### Tailing / killing a box from a shell with NO AWS creds

Two more workflows on the same OIDC role, for a Claude session (or anyone)
that can't run `aws` directly:

- **`eeg-tail.yml`** -- read-only snapshot: dumps the current S3 `run.log`
  (+ `eeg-job.log` tail) for a given run name, plus instance state and an
  orphan check (any `Project=eeg` box still `running`). Re-trigger to poll;
  it doesn't hold the job open. `gh workflow run eeg-tail.yml -f
  run_name=<slug> [-f instance_id=<id>]`. One-time setup:
  `scripts/setup_gh_launcher_tail.sh` -- adds inline policy
  `eeg-gh-tail-read` to `eeg-gh-launcher` (`s3:GetObject`/`ListBucket`
  scoped to `exports/*` only, no write/delete, nothing outside that
  prefix).
- **`eeg-terminate.yml`** -- kill switch: `aws ec2 terminate-instances`
  on one instance ID, via the same role's existing `Terminate/Stop`
  (tag-gated to `Project=eeg`, so it can't touch anything else). `gh
  workflow run eeg-terminate.yml -f instance_id=<id>`. No extra setup
  needed -- reuses the permission `setup_gh_launcher.sh` already grants.

Both need to exist on the repo's default branch to be dispatchable
(`workflow_dispatch` on a feature branch alone returns 404 from the API).

---

## Boxes

| Box | Instance | Spec | State | Address | Notes |
|---|---|---|---|---|---|
| ~~**eeg-box**~~ | `i-083e3b55993a13c13` | t3.small | **terminated 2026-09-02** | -- | was the standing shell/launcher box, ~$15/mo. Replaced by the `eeg-run.yml` GitHub Actions launcher (OIDC, no stored creds). No persistent shell box exists now. |
| **eeg-cpu-box** | `i-0a6100d4c303f52a2` | c7i.2xlarge (8 vCPU / 16 GB), 50 GB gp3, us-east-1a | **stopped** (persistent stop/start) | -- | on-demand CPU worker for pipeline runs, driven via the S3 job runner. Bootstrapped 2026-09-01 (pip env + chb01 data on the volume). `--instance-initiated-shutdown-behavior stop`. Idle ~$4/mo (volume only); ~$0.36/hr when running. First job: `godoy_tmc` 6-fold, ~41.5 s/epoch, 58 min wall (see session note 2026_09_01). NB: only ~3 of 8 cores used, but peak RSS 14/16 GB (raw-classifier LOSO path leaks per-fold) -- don't downsize below 16 GB. See "eeg-cpu-box job runner" below. This is a *separate* path from `eeg-run` (which spins up fresh ephemeral boxes). |
| `eeg-run` workers | ephemeral | c7i.2xlarge (cpu) / g5.2xlarge (gpu) | launched per job, self-terminate | -- | created by `scripts/eeg-run.sh` (usually via `eeg-run.yml`). Fresh box each run, tag `Project=eeg`, profile `eeg-gpu`. GPU blocked on quota (below). |
| GPU box | -- | -- | **not launched** | -- | IAM/instance-profile staged (`eeg-gpu`), nothing running |

## Launching / driving an EC2 box -- read before `run-instances`

Applies to the `eeg-run.yml` workflow (via the `eeg-gh-launcher` role),
the S3 job runner, and any direct `run-instances` from a local admin
shell. Items 1, 3, 4 and 6-8 are account-wide facts. Items 2 and 5 were
`eeg-box`-*role* limits -- the box is gone, but `eeg-gh-launcher` is
similarly scoped (no `CreateImage`, no `ssm:SendCommand`, `RunInstances`
tag-gated); a local shell on the `claude` admin user has none of these
limits.

Every one of these was hit for real standing up `eeg-cpu-box` 2026-09-01.
Skipping this list cost ~40 min of terminate/relaunch cycles.

1. **AMI has no `aws` CLI.** `ami-0d7f022123f8ff19d`
   (`ubuntu-noble-24.04-amd64-server`) -- and every stock Ubuntu 24.04
   image -- ships **no `aws`**, no `session-manager-plugin`, and there is
   **no `awscli` apt package** (not in the enabled repos). Any user-data
   that calls `aws` must install it *first*:
   ```
   apt-get update -y && apt-get install -y unzip curl
   curl -sS https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/a.zip
   ( cd /tmp && unzip -q a.zip && ./aws/install )   # -> /usr/local/bin/aws
   ```
   Symptom if you forget: user-data half-runs, nothing ever appears in
   S3, `describe-instances` just says `running`. (`eeg-run.sh`'s user-data
   already does this install step -- keep it first if you edit it.)
2. **Spot SLR: FIXED 2026-09-01.** An admin (user `claude`) ran `aws iam
   create-service-linked-role --aws-service-name spot.amazonaws.com`, so
   `AWSServiceRoleForEC2Spot` now exists and spot `RunInstances` no longer
   fails `AuthFailure.ServiceLinkedRoleCreationNotPermitted`. **But** the
   spot *quota* for GPU (`L-3819A6DF`, "All G and VT Spot Instance
   Requests") is still ~0 -> spot GPU now fails `MaxSpotInstanceCount
   Exceeded` instead. Increase-request to 8 filed 2026-09-01 ~14:03 UTC,
   PENDING. (Non-GPU spot families have their own separate quotas.)
3. **On-demand vCPU limit is 16** (Standard family: C/D/H/I/M/R/T/Z).
   With `eeg-box` gone the full 16 is free (minus `eeg-cpu-box`'s 8 when
   it's running). `c7i.2xlarge` (8) fine; `c7i.4xlarge` (16) fine only if
   nothing else is up. When replacing an 8-vCPU box, wait for the old one
   to reach **`terminated`** (not just `shutting-down`, ~2-4 min) before
   the replacement launches, or you may transiently exceed 16.
4. **AZ capacity varies.** `c7i.4xlarge` in us-east-1c returned
   `InsufficientInstanceCapacity`; 1a was fine. VPC
   `vpc-0ab36a88b3617669a` public subnets (all auto-assign public IP):
   1a `subnet-057fcd8e8ed1ec050`, 1b `subnet-07cdbc1058cf4f752`,
   1c `subnet-021f5ceeb4af26220`, 1d `subnet-00252702f59bac48f`,
   1e `subnet-0d55ae5c23cf61927`, 1f `subnet-0090cef9098097cb0`.
5. **Scoped launcher roles CANNOT** do admin ops. `eeg-gh-launcher` (the
   workflow role) = `RunInstances` + `Describe*` + `CreateTags`-on-launch
   + `PassRole eeg-gpu` + `Terminate/Stop` gated to `Project=eeg`, nothing
   else. No `ssm:*`, `CreateImage`, `iam:*`, `servicequotas:*`. For quota
   checks / AMI baking / SLR creation use a local admin (`claude`) shell.
   No `ssm:GetParameter` either -> resolve AMIs with `describe-images
   --owners 099720109477` / `amazon` (which `eeg-run.sh` does).
6. **Tag `Project=eeg` at launch** (`--tag-specifications
   'ResourceType=instance,Tags=[{Key=Project,Value=eeg}]'`) or you can't
   stop/terminate it afterward -- those perms are tag-gated.
7. **Ephemeral boxes need no shell.** `eeg-run` boxes self-terminate; the
   CPU box is driven through S3. If you *do* need an interactive shell,
   the `eeg-gpu` profile has `AmazonSSMManagedInstanceCore` -> `aws ssm
   start-session --target <id>` (needs `session-manager-plugin` locally,
   not installed on the Mac -- `brew install --cask session-manager-plugin`).
8. A correct bootstrap of `eeg-cpu-box` takes **~4 min** (apt + awscli +
   `pip install -r requirements.txt` ~2.5 min + chb01 prefetch). If it's
   taking much longer with no S3 output, it's item 1, not slowness.

## eeg-cpu-box job runner

The CPU box is driven through S3, not a shell -- **S3 control needs
nothing babysitting it**: queue a job, close the laptop, the box runs it
and archives results to S3 on its own. Preferred remote-control path from
*any* shell (local Mac, cloud). This is a *separate* path from `eeg-run`:
the CPU box is persistent (pre-bootstrapped env + cached chb01 data), so
it's faster to start for a quick job, whereas `eeg-run` spins up a clean
box and commits results automatically.

- **Start it:** `aws ec2 start-instances --instance-ids i-0a6100d4c303f52a2`
  (needs `Project=eeg` tag-gated start/stop -- local admin, or a role with
  that perm). A systemd unit `eeg-runner.service` starts on boot and
  polls S3.
- **Queue a job:** upload a bash script to
  `s3://noshore-eeg-benchmarks-827938107865/exports/eeg_box/jobs/next.sh`.
  The runner claims it (`s3 mv`), runs it with `cwd=/root/repo`, streams
  `output.log` to `.../jobs/latest.log` every 30 s, and on exit copies
  `output.log` + `rc` + `Epilepsy/results/` to `.../jobs/runs/<ts>/`.
- **Idle self-stop:** 30 min with no job -> the runner runs `shutdown -h
  now`, which *stops* (not terminates) the box. Restart + re-queue to use
  it again. Repo is at `/root/repo` (git clone, `git pull` in a job to
  update); chb01 data cached on the volume.
- Bootstrap progress on first launch: `.../exports/eeg_box/bootstrap-status`.
- user-data / job-runner source: `s3://…/exports/eeg_box/_setup/`
  (`persist_userdata.sh`, `job_godoy.sh`). Regenerate the box from
  `persist_userdata.sh` if the volume is ever lost. Terminating the box
  loses the bootstrapped env (no AMI -- `CreateImage` denied).
- **Gotcha:** the base Ubuntu 24.04 AMI has **no `aws` CLI** and no
  `session-manager-plugin` / `awscli` apt package. `persist_userdata.sh`
  installs the CLI (official v2 zip) before its first `aws` call -- keep
  that step first.

## Shared storage

**`s3://noshore-eeg-benchmarks-827938107865/`** -- created 2026-09-01,
versioning **enabled**, no lifecycle policy. Prefixes:

- `checkpoints/` -- model checkpoints to share between boxes
- `datasets/` -- preprocessed / cached data
- `exports/` -- results, figures, anything to pull back to the Mac

Old bucket `s3://coheriq-eeg-dense-edge-cache/` (referenced in
`_to_delete/aws_sync.log`) is **deleted** -- do not sync to it.

## IAM

- **User `claude`** -- the local CLI creds (Mac). Admin via group `cli`
  (`AdministratorAccess` + others).
- **Role `eeg-gh-launcher`** (created 2026-09-02) -- assumed by the
  `eeg-run.yml` workflow via GitHub OIDC, **not** an instance profile,
  nothing stored. Trust: `token.actions.githubusercontent.com`, scoped to
  `:repository = noshore5/EEG_Benchmarks` exact + `:sub` wildcard (this
  account presents a customised `sub`, `repo:<owner>@<id>/<repo>@<id>:...`).
  Inline policy `eeg-gh-launch`: `RunInstances` + EC2 `Describe*` +
  `CreateTags`-on-launch + `Terminate/Stop` gated to `Project=eeg` +
  `PassRole` for `eeg-gpu` only. Setup: `scripts/setup_gh_launcher.sh`.
- **Role `eeg-box`** -- the terminated box's instance-profile role. Still
  exists (unused); has `s3-eeg-bucket` RW, `ec2-launch-and-ssm`
  (`scripts/eeg-box-ec2-policy.json`), `eeg-box-extra` (servicequotas
  read+request, spot-SLR create, `ec2:CreateImage`). Safe to delete, or
  keep as a template for a future scoped launcher user.
- **Instance profile `eeg-gpu`** -> role `eeg-gpu` (created 2026-09-01) --
  attached to every `eeg-run` worker. `s3-eeg-bucket` RW, `eeg-ssm-and-sns`
  (`ssm:GetParameter` on `/eeg/*` + `sns:Publish` on `eeg-runs`), plus
  managed `AmazonSSMManagedInstanceCore` for keyless SSM shell access.

## Network

- SG **`eeg-ssh`** (`sg-013ae3c4f7d2405bb`) -- inbound TCP 22 from
  `0.0.0.0/0` (open; tighten to a home IP when convenient).
- SG `default` (`sg-0a356e30030ed3133`).
- Key pair: **`eeg-box`**.
- `AWSDataLifecycleManagerDefaultRole` exists (EBS snapshot automation)
  but no DLM schedules are configured.

## If you're bringing up the GPU box

**First read "Launching / driving an EC2 box" above** -- the awscli,
spot-SLR, vCPU-limit and tag gotchas all apply here too.

**GPU quota status (2026-09-01, late):** effective quotas on-demand G/VT
(`L-DB2E81BA`) = **0**, spot G/VT (`L-3819A6DF`) = **0**. Both
increase-requests to 8 (filed ~14:03 UTC by user `claude`) moved
`PENDING` -> **`CASE_OPENED`** -- AWS routed them to a human support
queue rather than auto-approving (normal for GPU-from-zero; hours to
~2 business days). Until one clears, *no* GPU instance launches
(`VcpuLimitExceeded` on-demand / `MaxSpotInstanceCountExceeded` spot).
Also `g5`/`g6`/`g4dn` **spot capacity was out region-wide** in us-east-1
that day -- us-east-1 is the most GPU-contended AWS region; if capacity
stays dry after the quota clears, fall back to firing the same two
requests in `us-east-2` and running there (pulls chb01 from the us-east-1
bucket, ~$0.14 egress). Check status from a **local admin shell**
(scoped launcher roles have no `servicequotas:*`):
`aws service-quotas list-requested-service-quota-change-history
--service-code ec2 --region us-east-1` and `get-service-quota
--service-code ec2 --quota-code L-DB2E81BA --region us-east-1` for the
effective value.

**Once the quota clears, GPU runs need no extra work** -- `eeg-run.sh`
already handles the GPU path (`--gpu` / `-f kind=gpu`): latest Deep
Learning OSS Nvidia PyTorch AMI, `g5.2xlarge`, 150 GB, profile `eeg-gpu`,
SG `eeg-ssh`, tag `Project=eeg`, self-terminating. Just run:
```
gh workflow run eeg-run.yml -f name=<slug> -f kind=gpu \
  -f cmd='python Epilepsy/run_pipelines.py --pipeline <p> --device cuda ...'
```
The older standalone `gpu_userdata.sh` under `exports/eeg_box/_setup/` is
superseded by `eeg-run.sh` -- ignore it.

To shell into a running GPU box for debugging: `aws ssm start-session
--target <id>` (profile `eeg-gpu` has the SSM managed policy) or SSH with
the `eeg-box` key pair. Update the box table + `CONTEXT.md` if you leave
one running with `--keep`.
