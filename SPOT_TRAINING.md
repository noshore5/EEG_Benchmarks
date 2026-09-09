# Passive spot-instance training

**Status (2026-09-09): infrastructure works and is DORMANT. No active
training target.** The SzCORE-detector / CHB-MIT run this was first hung on
was abandoned (2026-09-09, user: "the spot training pipelines work, which
is the important thing, but I don't care about training godoy for
detection of CHB-MIT"). The `eeg-spot-keepalive.yml` cron is
**disabled** (`gh workflow disable`) -- re-enable only when there's a real
run to drive, and never with the 24-subject default command (needs the
dataset cache, §"What's NOT built" #2). Read `AWS_INFRA.md` first.

What was actually proven:
- `SpotTrainer` checkpoint / resume / SIGTERM-flush / config-guard --
  tested locally (chb01, cpu+mps), works.
- `eeg-spot-train.sh` bootstrap on a real g4dn spot box got through:
  spot launch w/ capacity fallback, `/opt/pytorch` venv + full dep
  install, git clone, `epilepsy2bids` import, `torch.cuda == True`,
  PhysioNet download, `build_dataset` running. No full training run ever
  completed an epoch on a box (every failure was bootstrap/plumbing or the
  24-subject OOM, never the training loop).
- Cost of getting there: ~6 spot boxes, ~$0.5-2 of AWS credit (2 were
  autonomous cron boxes running the 24-subject default -- a mistake:
  the workflow was pushed to main with an active `*/15` schedule).

To reuse this for a different model: point `--cmd` at another training
script that takes `--checkpoint-dir` / `--s3-prefix` / `--resume-from` and
follows the same DONE-sentinel + SIGTERM contract (copy `SpotTrainer`).

Built so far:
- `Epilepsy/szcore/trainer.py` -- `SpotTrainer`: per-epoch checkpoint
  (local + S3), `--resume-from` (auto-resumes from `s3://.../latest.pt`),
  SIGTERM -> checkpoint + exit 0, config-hash guard, sidecar
  `progress.json`. Verified locally: resume continues at the right epoch;
  SIGTERM mid-run exits 0 with a good checkpoint; changed hyperparams are
  refused.
- `Epilepsy/szcore/train_detector.py` -- rewired onto `SpotTrainer`;
  `--checkpoint-dir` / `--s3-prefix` / `--resume-from`; writes `DONE` on
  clean finish.
- `scripts/eeg-spot-train.sh` -- launches ONE spot GPU box with
  (type, AZ) capacity fallback + explicit `MaxPrice`, venv bootstrap
  (sidesteps the DLAMI pip bug), forwards SIGTERM to the trainer,
  self-terminates only on clean finish. Guards: won't launch if `DONE`
  exists or a box for the job is already running.
- `.github/workflows/eeg-spot-keepalive.yml` -- `*/15` cron; relaunches
  until `DONE`; crash-loop guard (3 stalls / 48 h -> FAILED `DONE` + SNS).
- `scripts/setup_spot_keepalive.sh` -- one-time IAM (checkpoints/* RW +
  sns:Publish on `eeg-gh-launcher`). **NOT yet run.**

## The idea

Spot instances are the same hardware as on-demand at ~60-90% off
(g4dn.xlarge: $0.526/hr on-demand vs ~$0.16/hr spot), but AWS can reclaim
them at any time with a 2-minute warning. If a training job

1. checkpoints its full state every epoch to S3, and
2. resumes from the latest checkpoint on startup, and
3. is relaunched automatically whenever the box dies,

then an interruption costs only the partial epoch since the last
checkpoint. A run that needs ~10 GPU-hours finishes for ~$1.60 instead of
~$5.30, spread over however many interruptions, with no one babysitting it.

## Target

The **SzCORE seizure-detection checkpoint** (`Epilepsy/szcore/train_detector.py`):
one model, tens of epochs, no leave-one-seizure-out loop. That shape fits
per-epoch checkpoint/resume exactly.

It does **not** fit `Epilepsy/run_pipelines.py`'s LOSO pipelines well --
those train 24 fresh short models (one per held-out seizure), so the
natural resume granularity is per-fold, which `run_pipelines.py`'s
existing fold-skip `--resume` behaviour already half-covers. Per-epoch
checkpointing buys little there.

## Components

### 1. Checkpoint every epoch  (`train_detector.py` / `GodoyTMCClassifier.fit`)

After each epoch, write a single file to
`s3://noshore-eeg-benchmarks-827938107865/checkpoints/<job>/latest.pt`
containing enough to resume bit-for-bit:

```python
{
  "epoch": e,                       # last COMPLETED epoch
  "model_state":     model.state_dict(),
  "optimizer_state": opt.state_dict(),
  "scheduler_state": sched.state_dict() if sched else None,
  "torch_rng":  torch.get_rng_state(),
  "numpy_rng":  np.random.get_state(),
  "best_metric": best_val,           # for early-stopping continuity
  "epochs_no_improve": patience_ctr,
  "config_hash": hash of args,       # refuse to resume a mismatched config
}
```

Write to a temp key then `s3 mv` (atomic) so a kill mid-write can't leave a
truncated checkpoint. Keep `latest.pt` + `best.pt`; optionally keep every
Nth epoch for a training curve.

### 2. Catch the 2-minute interruption warning

Spot interruption shows up two ways; handle both:

- the instance gets a **`SIGTERM`** ~2 min before termination -> install a
  handler that sets a flag; the training loop checkpoints and exits 0 at
  the next safe point (end of the current batch/epoch).
- IMDS: `GET http://169.254.169.254/latest/meta-data/spot/instance-action`
  returns 200 with a `time` once marked for termination -> a background
  thread can poll every ~5 s as a backstop.

On the flag: flush a checkpoint, upload, exit. Do **not** self-terminate
-- let AWS reclaim the box; the relaunch loop brings up the next one.

### 3. Resume on startup  (user-data)

Before starting training, user-data does:

```bash
aws s3 cp "s3://.../checkpoints/$JOB/latest.pt" /root/latest.pt 2>/dev/null \
  && RESUME="--resume-from /root/latest.pt" || RESUME=""
python3 Epilepsy/szcore/train_detector.py --subjects $(seq 1 24) --epochs 25 \
  --device cuda $RESUME
```

`--resume-from` (new flag): load the checkpoint, restore model/opt/rng,
set `start_epoch = ckpt["epoch"] + 1`, continue. If `config_hash`
mismatches, abort loudly (don't silently train a different thing).

### 4. Relaunch loop

Something must notice "box gone, job not done" and launch another. Job
signals completion by writing `s3://.../checkpoints/<job>/DONE` as its last
act on clean finish.

**Chosen approach: a scheduled GitHub Actions workflow** (`eeg-spot-keepalive.yml`),
every ~15 min, on the existing `eeg-gh-launcher` OIDC role:

```
if   s3 object .../DONE exists            -> exit (job finished)
elif a Project=eeg,spot=1 box is running  -> exit (still training)
else                                      -> launch one spot GPU box
```

- Zero always-on infra; reuses the OIDC launcher.
- Worst case ~15 min idle between an interruption and the resume. Fine for
  passive training.
- Capacity fallback: the launch step tries a list of
  `(instanceType, AZ)` pairs -- e.g. `g4dn.xlarge`, `g5.xlarge`,
  `g6.xlarge` x all 6 us-east-1 AZs -- and takes the first that isn't
  `InsufficientInstanceCapacity`. (us-east-1 g5/g6 spot was fully dry on
  2026-09-07; g4dn was the only family with stock.)
- Set an explicit spot `MaxPrice` (e.g. on-demand rate) so a price spike
  can't run up the bill.

**Alternative (not chosen): EventBridge spot-interruption event -> Lambda ->
relaunch.** Near-instant resume, but real infra to stand up and maintain.
Only worth it if the 15-min gap ever actually matters.

### 5. Guardrails

- **Wall-clock cap:** job refuses to start if
  `now - first_checkpoint_time > 48h` -> writes `DONE` with a FAILED
  marker and stops the relaunch loop. Prevents an infinite relaunch storm
  from a job that crashes every epoch.
- **Crash-loop detector:** keepalive workflow tracks consecutive launches
  with no epoch progress (compare `ckpt["epoch"]` across ticks); after 3,
  stop and SNS-notify.
- **Cost tripwire:** a CloudWatch billing alarm at e.g. $20/mo (there's
  already a $100 budget).
- **Orphan sweep:** the keepalive tick also terminates any `Project=eeg`
  box running longer than N hours with a stale checkpoint.

## Cost sketch (full 24-subject SzCORE detector, ~10 GPU-h of real compute)

| | spot (this design) | on-demand |
|---|---|---|
| g4dn.xlarge compute | ~$1.60 | ~$5.30 |
| wasted partial epochs (say 5 interruptions x ~half an epoch) | ~$0.10 | -- |
| S3 checkpoints (~200 MB x versions) | <$0.05 | <$0.05 |
| EBS root, per-launch, ephemeral | ~$0.01 | ~$0.01 |
| **total** | **~$1.8** | **~$5.4** |

Both are inside the AWS credit balance; the point of spot here is less
about the dollars than about not needing the on-demand GPU quota (still
0) and not babysitting.

## What's NOT built / still to do

1. **Run `scripts/setup_spot_keepalive.sh` once** (admin AWS creds) and
   merge the workflow to `main` so the `*/15` cron is live.
2. **Streaming-shard dataset cache.** `build_dataset` still
   `np.concatenate`s every window -- ~130 GB for full CHB-MIT, won't fit
   in a g4dn/g5.xlarge's RAM. Needs: per-record window extraction ->
   `.npy` shards on the box's disk (or S3 `datasets/`), then a
   `Dataset` that reads shards. Until this lands, `--subjects` is capped
   at whatever fits (~4-6 subjects on a 16 GB box).
3. **End-to-end spot run never executed** -- us-east-1 g5/g6 spot was dry
   2026-09-07; g4dn had stock. First real run will shake out the venv
   bootstrap timing and the SIGTERM-forward path on an actual eviction.
4. **`predict_proba` on the full training set** after fit re-runs the
   whole set through the model on the box -- fine at a few subjects,
   revisit for 24.
5. Optional: EventBridge spot-interruption -> Lambda for instant relaunch
   instead of the 15-min cron gap.

## Why spot isn't the universal default

- **Capacity, not price, is the usual blocker.** Popular GPU types in
  hot regions (us-east-1) are frequently just unavailable on spot.
- **Interruptions have real cost** for jobs that can't checkpoint cheaply
  or resume exactly (see below), and for anything latency-sensitive.
- **Engineering overhead:** checkpoint/resume/relaunch/guardrails is real
  work; for a job that runs once for 20 minutes it's not worth it.
- **Stateful / long-lived services** (a database, an API, an interactive
  session) can't tolerate a 2-minute-notice eviction at all.

### Tasks that can't (easily) use this pattern

- **Anything with no natural checkpoint** or where a checkpoint is as
  expensive as the work: some HPC simulations, large graph computations
  with no clean intermediate state.
- **Distributed training** where losing one node kills the whole job
  unless the framework does elastic/fault-tolerant training (possible,
  but more setup).
- **Real-time / serving / anything with an SLA.**
- **Work that must not be re-executed** (non-idempotent side effects:
  sending emails, charging cards, irreversible writes).
- **Proof-of-work mining (e.g. Bitcoin).** Not a checkpointing problem --
  a Bitcoin miner *is* trivially interruptible (each hash attempt is
  independent; losing the box loses nothing but the in-flight nonce).
  The reason you can't profitably "passively mine on spot" is economic:
  Bitcoin is SHA-256 on ASICs that are ~5-6 orders of magnitude more
  efficient per hash than any GPU, so a rented GPU's electricity-and-rent
  cost vastly exceeds the block reward it can expect to earn. Even a free
  GPU would earn ~nothing. Spot pricing doesn't change that math. (This
  was true for GPU-minable coins like Ethereum pre-Merge too, once you
  priced in cloud rates -- cloud providers ban or throttle mining partly
  because the only people who try it at scale are running stolen
  credentials.) The general lesson: spot economics only help if the
  output has more value than the discounted compute cost, which for
  mining it doesn't.
