# First AWS box pipeline run -- `godoy_tmc` 6-fold on a CPU instance

2026-09-01, `eeg-box` shell. Task: stand up an AWS CPU worker (not the
tiny `eeg-box` itself), run `godoy_tmc` prediction 6-fold LOSO on CHB-MIT
chb01, report per-epoch wall-clock. Then leave the box as a persistent
stop/start worker.

## TL;DR

- **~41.5 s/epoch** on 8 vCPU (`c7i.2xlarge`), rock-steady (range 39.8-43.6 s
  across all 82 epochs of the 6-fold). ~3-4x slower than MPS (7 s/epoch on
  the user's M-series Mac), ~1x-ish vs older heavy anchoring.
- **Full 6-fold: 58 min 54 s wall** (dataset build + 82 trained epochs +
  per-fold best-model restore + eval). `git reset --hard origin/main` ->
  ran at `51cf7a3`.
- Only **~300% CPU** (3 of 8 cores busy) -- the model is tiny
  (1 Transformer layer, d_model=32, 22 batches/epoch), doesn't parallelize.
  A 4-vCPU box would run this at ~the same speed for half the price...
- ...**except peak RSS was 14.1 GB on the 16 GB box.** Memory climbs
  fold-over-fold (the per-fold `del clf; gc.collect()` fix in `51cf7a3`
  only touches `leave_one_seizure_out_prediction`, NOT the raw-classifier
  path `leave_one_seizure_out_raw_classifier_prediction` that godoy/
  dbconformer/slimseiz/cg_mambanet use). An 8 GB box would OOM on fold ~4.
  Keep 16 GB until that path gets the same treatment.

## Per-fold timing

| Fold | Seizure | Epochs run (early-stop) | best ep | ~s/epoch | batches/ep |
|---|---|---|---|---|---|
| 1 | 1_03_0 | 17 | 12 | 41.5 | 22 |
| 2 | 1_04_0 | 12 |  7 | 41.2 | 22 |
| 3 | 1_15_0 | 10 |  5 | 40.4 | 22 |
| 4 | 1_16_0 | 17 | 12 | 43.1 | 23 |
| 5 | 1_18_0 | 11 |  6 | 41.9 | 22 |
| 6 | 1_26_0 | 15 | 10 | 41.6 | 22 |

82 epochs total. `User 6870 s + Sys 3772 s` over 3534 s wall = 3.0x
core-seconds, i.e. ~3 busy cores.

## Results (mean across 6 folds, prediction mode, chb01)

Not the headline of this exercise (timing was), but for the record:

| metric | this CPU run | MPS single-run 2026-08-31 | MPS 5-seed mean |
|---|---|---|---|
| average_precision | **0.535** | 0.619 | 0.556 +/- 0.056 |
| roc_auc | 0.949 | 0.968 | -- |
| f1 | 0.521 | 0.532 | -- |
| recall | 0.678 | 0.811 | -- |
| FAR/h (raw / smoothed) | 7.69 / 5.64 | 8.66 / -- | -- |
| event hit rate raw / k-of-n | 6/6 / 5/6 | 5/6 / 5/6 | -- |

Per-fold AP: .112 / .313 / .346 / 1.000 / 1.000 / .442.
CPU vs MPS differ well within the seed band (CPU 0.535 is ~0.4 sigma under
the 0.556 +/- 0.056 sweep mean) -- CPU/MPS use different BLAS reductions so
this is not bit-reproducible against the MPS runs anyway. Notable: this run
**hit 1_15** (AP .346, the fold the 2026-08-31 MPS run totally missed) and
whiffed harder on 1_03 (AP .112 vs .549). The two easy folds (1_16, 1_18)
stay perfect (AP 1.000, 0 false alarms). Hard-fold variance dominates.

Did **not** add a row to the comparison doc -- this is a CPU timing run,
not a new seed; the board number for godoy stays the MPS 5-seed 0.556.

## The box: `eeg-cpu-box` `i-0a6100d4c303f52a2`

`c7i.2xlarge` (8 vCPU / 16 GB), 50 GB gp3, us-east-1a, on-demand,
`--instance-initiated-shutdown-behavior stop`. Tagged `Project=eeg`.
Persistent stop/start worker; **stopped** at end of this session
(idle self-stop also armed at 30 min). Idle cost ~$4/mo (volume only),
~$0.36/hr running.

Driven via S3, not SSH (`eeg-box` role has no `ssm:SendCommand`): a
systemd `eeg-runner.service` polls
`s3://noshore-eeg-benchmarks-827938107865/exports/eeg_box/jobs/next.sh`,
claims it with `s3 mv`, runs it in `/root/repo`, streams `output.log` to
`.../jobs/latest.log`, and on exit archives to `.../jobs/runs/<ts>/`.
Full run log + CSVs for this run:
`.../exports/eeg_box/jobs/runs/20260901-131847/`.
user-data + job scripts stashed at `.../exports/eeg_box/_setup/`
(`persist_userdata.sh`, `job_godoy.sh`). Regenerate the box from
`persist_userdata.sh` if the volume is lost (no AMI -- `CreateImage`
denied to the role).

## The scenic route (documented so it doesn't recur)

Standing up the box cost ~40 min of terminate/relaunch. All 8 gotchas are
now written into **`AWS_INFRA.md` -> "Launching an EC2 box from eeg-box"**.
Short version:

1. Base Ubuntu 24.04 AMI has **no `aws` CLI**, no `session-manager-plugin`,
   no `awscli` apt package. Install awscli v2 zip as the *first* user-data
   step or nothing reaches S3 and `describe-instances` just says `running`.
   (Cost us 2 dead boxes -- `i-0c3c8d159a1fa6c2d` + one interim.)
2. Spot -> `AuthFailure.ServiceLinkedRoleCreationNotPermitted`; role can't
   create the SLR. Use on-demand. **(Fixed later this session -- admin ran
   `aws iam create-service-linked-role --aws-service-name spot.amazonaws.com`;
   SLR now exists, CreateDate 2026-09-01T14:03 UTC.)**
3. On-demand Standard-family vCPU limit is 16; `eeg-box` burns 2.
   `c7i.2xlarge` (8) OK, `c7i.4xlarge` (16) -> `VcpuLimitExceeded`.
   Replacing an 8-vCPU box: wait for `terminated`, not `shutting-down`.
4. AZ capacity varies (`c7i.4xlarge` dead in 1c, fine in 1a).
5. Role can't `CreateImage` / `ModifyInstanceAttribute` / `ssm:SendCommand`
   / `ssm:GetParameter` / any `iam:*`.
6. Tag `Project=eeg` at launch or stop/terminate is denied (tag-gated).
7. No `session-manager-plugin` on eeg-box.
8. A correct bootstrap is ~4 min. Longer with no S3 output = it's #1.

## GPU sidebar -- economics + why we couldn't run it

User asked: how much faster would CUDA have to be to beat the CPU box on
cost. Break-even = price ratio:
- vs `g5.2xlarge` on-demand ($1.21/hr): **3.4x** faster (12 s/epoch).
- vs `g5.2xlarge` spot (~$0.75/hr): **2.1x** (19 s/epoch).
Given the user's MPS does 7 s/epoch, a real A10G should clear both easily
(40/7 = 5.7x), so on-demand GPU ends up ~1.7x cheaper per job AND finishes
in minutes. For this pipeline the cheapest option is still the Mac ($0).
GPU math flips decisively for the heavy pipelines regardless.

Tried to run it anyway on GPU spot -- **blocked**:
- GPU on-demand vCPU quota (`L-DB2E81BA`) = **0** -> `VcpuLimitExceeded`
  even for a 4-vCPU `g5.xlarge`, all 5 AZs.
- GPU spot: after the SLR fix, `g5.*` -> genuine `InsufficientInstanceCapacity`
  region-wide (us-east-1 is the most GPU-contended AWS region;
  `g5`/`g6`/`g4dn` all out across all 6 AZs). `g6.*` ->
  `MaxSpotInstanceCountExceeded` = spot quota `L-3819A6DF` (also ~0),
  i.e. g6 spot likely *has* capacity and just needs the quota.

**Quota increase requests filed** (admin, user `claude`, 2026-09-01 ~14:03 UTC,
both `PENDING`, us-east-1):
- `L-DB2E81BA` "Running On-Demand G and VT instances" -> 8
- `L-3819A6DF` "All G and VT Spot Instance Requests" -> 8

When either approves: `gpu_userdata.sh` (staged at
`.../exports/eeg_box/_setup/gpu_userdata.sh`) is a one-shot -- launches
`g5.2xlarge` on the Deep Learning OSS Nvidia Driver AMI
(`ami-012ba162b9cd2729c`, PyTorch 2.7 Ubuntu 22.04), `pip install -r
requirements.txt` (still re-pulls `torch==2.8.0+cu128`, ~2.5 GB), prefetches
chb01, runs `godoy_tmc --device cuda`, pushes results to
`.../exports/eeg_box/gpu_run/`, self-terminates. Est. bootstrap ~6-8 min
(heavier AMI + EBS lazy-load), not the 4 min the CPU box hit.

Quotas are **per-region** -- if us-east-1 GPU capacity stays dry, the fallback
is to fire the same two requests in `us-east-2` and run the one-shot there
(pulls chb01 from the existing us-east-1 bucket, ~$0.14 egress).

## Cloud-pricing aside (user asked)

us-east-1 is cheapest not because of Virginia electricity but because it's
AWS's oldest/largest region and the pricing anchor every other region is
marked up from. Power is ~10-25% of DC opex and near-noise for a GPU
instance (accelerator capex dominates). Cheap-power regions (eu-north-1
hydro, Gulf states) still price at/above us-east-1 on scale. Being in
Tbilisi doesn't hurt this workflow (all batch/S3; ~150 ms to Virginia
irrelevant) -- would only matter for interactive IDE-remote work.
