# AWS infrastructure (read only if you're running on an AWS instance)

**Scope gate:** this file is for agent shells running *on* an AWS box
(`eeg-box` or a future GPU box). If you're a Mac / Grok / local shell just
working the repo, you can ignore this entirely -- it describes shared
cloud state, not repo state, and nothing in the pipelines reads from it
yet.

**Account:** `827938107865` &nbsp; **Region:** `us-east-1`
**Last verified:** 2026-09-01, by Claude (eeg-box shell) -- `eeg-cpu-box`
stood up + bootstrapped end-to-end; gotchas below all hit and confirmed.

---

## Boxes

| Box | Instance | Spec | State | Address | Notes |
|---|---|---|---|---|---|
| **eeg-box** | `i-083e3b55993a13c13` | t3.small, 30 GB EBS (`vol-0812ddb8bac762447`) | running | public `54.236.223.15`, private `172.31.22.104` | the box this agent usually lives on. 2 vCPU / 2 GB RAM -- too small to run pipelines (no deps installed) |
| **eeg-cpu-box** | `i-0a6100d4c303f52a2` | c7i.2xlarge (8 vCPU / 16 GB), 50 GB gp3, us-east-1a | **stop/start** (persistent) | -- | on-demand CPU worker for pipeline runs. Bootstrapped 2026-09-01 (pip env + chb01 data on the volume). `--instance-initiated-shutdown-behavior stop`. Idle ~$4/mo (volume only); ~$0.36/hr when running. See "eeg-cpu-box job runner" below. |
| GPU box | -- | -- | **not launched** | -- | IAM/instance-profile staged (`eeg-gpu`), nothing running |

## Launching an EC2 box from eeg-box -- read before `run-instances`

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
   S3, `describe-instances` just says `running`. (eeg-box's own `aws` was
   hand-installed to `/usr/local/bin/aws`, so it's misleading.)
2. **Spot is not available to this role.** `RunInstances` with
   `--instance-market-options MarketType=spot` fails
   `AuthFailure.ServiceLinkedRoleCreationNotPermitted` -- the
   `AWSServiceRoleForEC2Spot` SLR doesn't exist and `eeg-box` has neither
   `iam:CreateServiceLinkedRole` nor `CreateSpot...`. **Use on-demand**
   until an admin creates that SLR once (`aws iam
   create-service-linked-role --aws-service-name spot.amazonaws.com`).
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
5. **`eeg-box` role CANNOT:** `ec2:CreateImage` (no AMI baking),
   `ec2:ModifyInstanceAttribute` (no termination protection, no
   instance-type change on a stopped box -- pick the type you want up
   front), `ssm:SendCommand` / `ssm:GetCommandInvocation` (only
   `StartSession`), `ssm:GetParameter` (can't resolve the Canonical AMI
   SSM alias -- use `describe-images --owners 099720109477` or the pinned
   id above), any `iam:*`, `iam:ListInstanceProfiles`.
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

No `ssm:SendCommand` / `ec2:ModifyInstanceAttribute` / `ec2:CreateImage`
for the `eeg-box` role, so the CPU box is driven through S3, not a shell:

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
spot-SLR, vCPU-limit and tag gotchas all apply here too (a GPU instance
family has its own separate on-demand vCPU limit, likely 0 until raised).

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
5. **Terminate as soon as the run finishes** (RunPod discipline applies
   here too).
6. Update the table above (instance id, state) and note it in
   `CONTEXT.md`'s pointer line.
