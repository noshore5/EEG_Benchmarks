# AWS infrastructure (read only if you're running on an AWS instance)

**Scope gate:** this file is for agent shells running *on* an AWS box
(`eeg-box` or a future GPU box). If you're a Mac / Grok / local shell just
working the repo, you can ignore this entirely -- it describes shared
cloud state, not repo state, and nothing in the pipelines reads from it
yet.

**Account:** `827938107865` &nbsp; **Region:** `us-east-1`
**Last verified:** 2026-09-01, by Claude (eeg-box shell) -- EC2/SSM grant
to role `eeg-box` applied from the Mac and confirmed working from the box
(`RunInstances` dry-run succeeds).

---

## Boxes

| Box | Instance | Spec | State | Address | Notes |
|---|---|---|---|---|---|
| **eeg-box** | `i-083e3b55993a13c13` | t3.small, 30 GB EBS (`vol-0812ddb8bac762447`) | running | public `54.236.223.15`, private `172.31.22.104` | the box this agent usually lives on |
| GPU box | -- | -- | **not launched** | -- | IAM/instance-profile staged (`eeg-gpu`), nothing running |

There is currently **one** EC2 instance. "The other infra" is the shared
S3 bucket + the IAM scaffolding below, not a second machine.

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
