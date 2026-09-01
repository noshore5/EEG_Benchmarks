# AWS infrastructure

**Who this is for:** any agent shell that touches the shared AWS account --
whether it runs *on* an AWS box (`eeg-box`, a GPU box) or drives the cloud
remotely from a local Mac / Grok / cloud-Claude shell. It describes shared
cloud state, not repo state, and nothing in the pipelines reads from it
yet -- so if you're only editing code or running tests, you can still skip
it.

**Can this shell touch AWS?** You need credentials for account
`827938107865` -- check with `aws sts get-caller-identity`.

- **On an AWS box** (`eeg-box` / `eeg-gpu`): the instance profile supplies
  them automatically, and *scoped* -- no keys on disk. See "IAM" for what
  the role can and can't do.
- **Local / off-AWS shell:** needs the `aws` CLI + a profile in
  `~/.aws/credentials`. The Mac this repo is usually worked from has the
  `claude` IAM user configured -- **AdministratorAccess**, so it can do
  everything in this file, *including* the actions the `eeg-box` role is
  denied (item 5 below). `session-manager-plugin` is **not** installed on
  the Mac, so `aws ssm start-session` won't work from there -- use
  `ssh eeg-box` or the S3 job-runner instead.
- **Grok / cloud-Claude shell with no creds:** can't launch anything
  directly -- hand the work to `eeg-box` (scoped launch role) or drop a
  job for the S3 runner.

**Account:** `827938107865` &nbsp; **Region:** `us-east-1`
**Last verified:** 2026-09-01, by Claude (eeg-box shell) -- `eeg-cpu-box`
stood up + bootstrapped + ran its first job (`godoy_tmc` 6-fold), then
stopped; gotchas below all hit and confirmed. Spot SLR created by admin.
GPU quota-increase requests filed -> now `CASE_OPENED` (AWS support queue,
not auto-approved; effective quota still 0). Admin added inline policy
`eeg-box-extra` to the `eeg-box` role (servicequotas read + request,
spot-SLR create, `ec2:CreateImage`/`ModifyInstanceAttribute`) -- so the
eeg-box shell can now check quota status and bake an AMI itself.

## Where to run cloud ops from

Launching or starting an instance is a one-shot API call -- the instance
doesn't care which shell made it, so doing it from a laptop is fine. What
matters is what happens *after* the shell goes away:

- **One-shot calls** (`start-instances` the CPU box, `run-instances`,
  `terminate-instances`, `aws s3 cp`): run them from wherever you are.
  Local Mac is fine and usually faster than SSHing into eeg-box first.
- **Anything that needs babysitting** (tailing a run, a poll loop, an
  interactive `ssh`): don't anchor it to a laptop. Use the S3 job-runner
  (survives disconnection) or eeg-box's persistent tmux (`cc`).
- **eeg-box still earns its place** as: a launcher for shells with no AWS
  creds (Grok, cloud-Claude), a persistent tmux home for long sessions,
  and an in-region git/edit box (fast S3). It is *not* required merely to
  launch instances.
- **Credential hygiene:** the Mac currently authenticates as the `claude`
  user (**AdministratorAccess**) -- broad privilege sitting in a laptop's
  `~/.aws/credentials`. If launching from local shells becomes routine,
  have an admin cut a scoped `eeg-launcher` IAM user mirroring the
  `eeg-box` role's `ec2-launch-and-ssm` policy
  (`scripts/eeg-box-ec2-policy.json`) and put *those* keys on the Mac.

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

**Fallback -- launcher-side trap** (what `~/bin/gpu-run` on eeg-box does:
`trap cleanup EXIT` terminates the box when the launching script exits).
Fine when launched from **eeg-box's persistent tmux**. **Do not rely on it
from a laptop** -- if the shell dies the box leaks. From local, use
self-terminating user-data instead.

**Results go to S3, never box-to-box.** There's no SSH trust or stable
address between boxes (eeg-box's IP is dynamic). Everything lands in the
bucket: `exports/<run-name>/` for logs + figures to pull back,
`checkpoints/` for weights. eeg-box or your Mac then `aws s3 sync`s from
there. To hand a result to eeg-box specifically, drop it under
`exports/eeg_box/jobs/runs/<ts>/` -- don't try to push to the box.

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

---

## Boxes

| Box | Instance | Spec | State | Address | Notes |
|---|---|---|---|---|---|
| **eeg-box** | `i-083e3b55993a13c13` | t3.small, 30 GB EBS (`vol-0812ddb8bac762447`) | running | public `54.236.223.15`, private `172.31.22.104` | the box this agent usually lives on. 2 vCPU / 2 GB RAM -- too small to run pipelines (no deps installed) |
| **eeg-cpu-box** | `i-0a6100d4c303f52a2` | c7i.2xlarge (8 vCPU / 16 GB), 50 GB gp3, us-east-1a | **stopped** (persistent stop/start) | -- | on-demand CPU worker for pipeline runs. Bootstrapped 2026-09-01 (pip env + chb01 data on the volume). `--instance-initiated-shutdown-behavior stop`. Idle ~$4/mo (volume only); ~$0.36/hr when running. First job: `godoy_tmc` 6-fold, ~41.5 s/epoch, 58 min wall (see session note 2026_09_01). NB: only ~3 of 8 cores used, but peak RSS 14/16 GB (raw-classifier LOSO path leaks per-fold) -- don't downsize below 16 GB. See "eeg-cpu-box job runner" below. |
| GPU box | -- | -- | **not launched** | -- | IAM/instance-profile staged (`eeg-gpu`), nothing running |

## Launching / driving an EC2 box -- read before `run-instances`

Works from `eeg-box` **or** from a local shell with creds (see scope note
above). Items 1, 3, 4 and 6-8 are account-wide facts and bite the same
either way. Items 2 and 5 are `eeg-box`-*role* limits: a local shell on
the `claude` admin user *can* create the spot SLR (item 2) and *can* do
the item-5 actions (`CreateImage`, `ModifyInstanceAttribute`, `iam:*`,
`ssm:SendCommand`).

Every one of these was hit for real standing up `eeg-cpu-box` 2026-09-01
from eeg-box. Skipping this list cost ~40 min of terminate/relaunch
cycles.

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
   S3, `describe-instances` just says `running`. (eeg-box's own `aws` was
   hand-installed to `/usr/local/bin/aws`, so it's misleading.)
2. **Spot SLR: FIXED 2026-09-01.** An admin (user `claude`) ran `aws iam
   create-service-linked-role --aws-service-name spot.amazonaws.com`, so
   `AWSServiceRoleForEC2Spot` now exists and spot `RunInstances` no longer
   fails `AuthFailure.ServiceLinkedRoleCreationNotPermitted`. **But** the
   spot *quota* for GPU (`L-3819A6DF`, "All G and VT Spot Instance
   Requests") is still ~0 -> spot GPU now fails `MaxSpotInstanceCount
   Exceeded` instead. Increase-request to 8 filed 2026-09-01 ~14:03 UTC,
   PENDING. (Non-GPU spot families have their own separate quotas.)
3. **On-demand vCPU limit is 16** (Standard family: C/D/H/I/M/R/T/Z).
   `eeg-box` (t3.small) already burns 2 -> **<=14 free**. So:
   `c7i.2xlarge` (8) fine; `c7i.4xlarge` (16) -> `VcpuLimitExceeded`.
   Replacing an 8-vCPU box: 8+8 = 16 but with eeg-box's 2 that's 18 ->
   you must wait for the old one to reach **`terminated`** (not just
   `shutting-down`, ~2-4 min) before the replacement will launch.
4. **AZ capacity varies.** `c7i.4xlarge` in us-east-1c returned
   `InsufficientInstanceCapacity`; 1a was fine. VPC
   `vpc-0ab36a88b3617669a` public subnets (all auto-assign public IP):
   1a `subnet-057fcd8e8ed1ec050`, 1b `subnet-07cdbc1058cf4f752`,
   1c `subnet-021f5ceeb4af26220`, 1d `subnet-00252702f59bac48f`,
   1e `subnet-0d55ae5c23cf61927`, 1f `subnet-0090cef9098097cb0`.
5. **`eeg-box` role CANNOT:** `ssm:SendCommand` / `ssm:GetCommandInvocation`
   (only `StartSession`), `ssm:GetParameter` (can't resolve the Canonical
   AMI SSM alias -- use `describe-images --owners 099720109477` or the
   pinned id above), most `iam:*`, `iam:ListInstanceProfiles`.
   `ec2:CreateImage` + `ec2:ModifyInstanceAttribute` were **granted
   2026-09-01** via the `eeg-box-extra` inline policy (so AMI baking and
   instance-type change on a stopped box now work); `iam:CreateService
   LinkedRole` for `spot.amazonaws.com` and `servicequotas:*` (read +
   request) are in that policy too.
6. **Tag `Project=eeg` at launch** (`--tag-specifications
   'ResourceType=instance,Tags=[{Key=Project,Value=eeg}]'`) or you can't
   stop/terminate it afterward -- those perms are tag-gated.
7. **No `session-manager-plugin` on eeg-box** -> `aws ssm start-session`
   errors out. To drive a box: install the plugin (deb from AWS), or use
   `--instance-initiated-shutdown-behavior stop` + the S3 job-runner
   pattern below (no shell needed).
8. A correct bootstrap of `eeg-cpu-box` takes **~4 min** (apt + awscli +
   `pip install -r requirements.txt` ~2.5 min + chb01 prefetch). If it's
   taking much longer with no S3 output, it's item 1, not slowness.

## eeg-cpu-box job runner

The CPU box is driven through S3, not a shell -- the `eeg-box` role has no
`ssm:SendCommand` and a local shell has no `session-manager-plugin`, but
the real reason is that **S3 control needs nothing babysitting it**: queue
a job, close the laptop, the box runs it and archives results to S3 on its
own. This is the preferred remote-control path from *any* shell (eeg-box,
local Mac, cloud):

- **Start it:** `aws ec2 start-instances --instance-ids i-0a6100d4c303f52a2`
  (eeg-box role can start/stop it -- tag `Project=eeg`). A systemd unit
  `eeg-runner.service` starts on boot and polls S3.
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

- **User `claude`** -- the CLI creds on the boxes. Admin via group `cli`
  (`AdministratorAccess` + others).
- **Instance profile `eeg-box`** -> role `eeg-box`:
  - inline policy `s3-eeg-bucket`: RW (`Get`/`Put`/`DeleteObject` +
    `ListBucket`) on the shared bucket. So from eeg-box, `aws s3 ...`
    against the bucket works with no keys.
  - inline policy `ec2-launch-and-ssm` (added 2026-09-01, verified from
    eeg-box same day): EC2 `Describe*` + `RunInstances`/`CreateTags`
    (any), `TerminateInstances`/`StopInstances`/`StartInstances` gated to
    `ec2:ResourceTag/Project=eeg`, `iam:PassRole` for `eeg-gpu` only, and
    `ssm:StartSession`/etc. So eeg-box can launch/terminate the GPU box
    and SSM into it. Source: `scripts/eeg-box-ec2-policy.json`.
    **Always tag launched instances `Project=eeg`** or you can't
    terminate them from here.
  - inline policy `eeg-box-extra` (added 2026-09-01 by admin `claude`):
    `servicequotas:GetServiceQuota` / `ListServiceQuotas` /
    `ListRequestedServiceQuotaChangeHistory*` / `RequestServiceQuota
    Increase` (all `*`); `iam:CreateServiceLinkedRole` gated to
    `iam:AWSServiceName=spot.amazonaws.com`; `ec2:CreateImage` /
    `RegisterImage` / `ModifyInstanceAttribute` / `DescribeImages` (all
    `*`). Rationale: stop bouncing quota checks / SLR creation / AMI
    baking through the Mac. Deliberately NOT included: unrestricted
    `RunInstances` (the `Project=eeg` tag gate stays), `iam:*` beyond the
    one SLR, billing.
- **Instance profile `eeg-gpu`** -> role `eeg-gpu` (created 2026-09-01),
  same `s3-eeg-bucket` policy, plus managed policy
  `AmazonSSMManagedInstanceCore` (added 2026-09-01) so the GPU box
  registers with SSM for keyless shell access. Attach this profile at
  launch.

## Network

- SG **`eeg-ssh`** (`sg-013ae3c4f7d2405bb`) -- inbound TCP 22 from
  `0.0.0.0/0` (open; tighten to a home IP when convenient).
- SG `default` (`sg-0a356e30030ed3133`).
- Key pair: **`eeg-box`**.
- `AWSDataLifecycleManagerDefaultRole` exists (EBS snapshot automation)
  but no DLM schedules are configured.

## If you're bringing up the GPU box

**First read "Launching an EC2 box from eeg-box" above** -- the awscli,
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
bucket, ~$0.14 egress). Check status **from eeg-box directly** now (the
`eeg-box-extra` policy grants it):
`aws service-quotas list-requested-service-quota-change-history
--service-code ec2 --region us-east-1` and `get-service-quota
--service-code ec2 --quota-code L-DB2E81BA --region us-east-1` for the
effective value.

**A one-shot GPU runner is staged:**
`s3://noshore-eeg-benchmarks-827938107865/exports/eeg_box/gpu_run/` is the
output prefix; `.../exports/eeg_box/_setup/gpu_userdata.sh` is the
user-data -- launches `g5.2xlarge` on the Deep Learning OSS Nvidia Driver
AMI (`ami-012ba162b9cd2729c`, PyTorch 2.7 / Ubuntu 22.04), installs
`requirements.txt` (re-pulls `torch==2.8.0+cu128` ~2.5 GB), prefetches
chb01, runs `godoy_tmc --device cuda`, uploads results, **self-terminates**.
Est. bootstrap ~6-8 min (heavier AMI + EBS lazy-load).

1. Launch from eeg-box into `us-east-1`, attach instance profile
   **`eeg-gpu`** and SG **`eeg-ssh`**, key pair **`eeg-box`**, and tag
   **`Project=eeg`** (required -- terminate perms are gated on that tag).
2. It gets bucket RW for free via the instance profile -- pull data from
   `s3://noshore-eeg-benchmarks-827938107865/datasets/`, push checkpoints
   to `checkpoints/`.
3. Shell in via `aws ssm start-session --target <id>` from eeg-box (no
   key needed once the SSM agent registers), or SSH with the `eeg-box`
   key.
4. Recommended type: **g5.2xlarge spot** (A10G 24 GB, 8 vCPU, 32 GB RAM;
   ~$0.55-0.90/hr spot in us-east-1a).
5. **Launch it self-terminating** -- see "Every launched box must
   self-terminate and ship results to S3" above. Use self-terminating
   user-data + `--instance-initiated-shutdown-behavior terminate`; don't
   depend on a babysitting shell.
6. Update the table above (instance id, state) and note it in
   `CONTEXT.md`'s pointer line.
