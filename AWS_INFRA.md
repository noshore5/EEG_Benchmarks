# AWS infrastructure (read only if you're running on an AWS instance)

**Scope gate:** this file is for agent shells running *on* an AWS box
(`eeg-box` or a future GPU box). If you're a Mac / Grok / local shell just
working the repo, you can ignore this entirely -- it describes shared
cloud state, not repo state, and nothing in the pipelines reads from it
yet.

**Account:** `827938107865` &nbsp; **Region:** `us-east-1`
**Last verified:** 2026-09-01, by Claude (Mac shell), via the `claude` IAM user.

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
- **Instance profile `eeg-box`** -> role `eeg-box`, inline policy
  `s3-eeg-bucket`: RW (`Get`/`Put`/`DeleteObject` + `ListBucket`) on the
  shared bucket. So from eeg-box, `aws s3 ...` against the bucket works
  with no keys.
- **Instance profile `eeg-gpu`** -> role `eeg-gpu` (created 2026-09-01),
  same `s3-eeg-bucket` policy. Pre-made so a GPU instance launched later
  can hit the bucket immediately -- attach this profile at launch.

## Network

- SG **`eeg-ssh`** (`sg-013ae3c4f7d2405bb`) -- inbound TCP 22 from
  `0.0.0.0/0` (open; tighten to a home IP when convenient).
- SG `default` (`sg-0a356e30030ed3133`).
- Key pair: **`eeg-box`**.
- `AWSDataLifecycleManagerDefaultRole` exists (EBS snapshot automation)
  but no DLM schedules are configured.

## If you're bringing up the GPU box

1. Launch into `us-east-1`, attach instance profile **`eeg-gpu`** and SG
   **`eeg-ssh`**, key pair **`eeg-box`**.
2. It gets bucket RW for free via the instance profile -- pull data from
   `s3://noshore-eeg-benchmarks-827938107865/datasets/`, push checkpoints
   to `checkpoints/`.
3. Update the table above (instance id, state) and note it in
   `CONTEXT.md`'s pointer line.
