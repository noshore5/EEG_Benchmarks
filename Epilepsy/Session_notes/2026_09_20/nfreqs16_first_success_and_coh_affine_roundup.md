# nfreqs=16 first verified 6-fold run, coh_affine_rescale, and the SSM push fix (2026-09-20)

Consolidation note for a long session's worth of work spread across three
individual run writeups (`tgm-nfreqs16-chunk2-seed42{,-v2}.md`,
`tgm-nfreqs16-cohaffine-seed42.md`) and 13 numbered entries in
`FAILURE_LOG.md`. This is the pointer-heavy summary; those files have the
full evidence trail.

## The headline result

After ~10 failed/killed spot-GPU attempts across this session and the one
before it, `temporal_graph_mamba` nfreqs=16 prediction completed a real
full 6-fold LOSO run for the first time, and it reproduced on a second
independent run:

| metric | chunk2-seed42 | chunk2-seed42-v2 |
|---|---|---|
| accuracy | 0.885 | 0.891 |
| precision | 0.404 | 0.419 |
| recall | 0.783 | 0.772 |
| f1 | 0.429 | 0.437 |
| average_precision | **0.484** | **0.481** |
| roc_auc | 0.941 | 0.944 |
| event-level hit rate (raw) | 6/6 | 6/6 |
| event-level hit rate (smoothed) | 5/6 | 5/6 |
| epoch_time | 3.45s | 3.46s |

Launch command now pinned in `LAUNCH_CHECKLIST.md` (commit `d175a9c`+):
`--dense-edge-gpu-cache --precompute-chunk-size 2
--temporal-graph-edge-complex-native --checkpoint-dir /root/checkpoint`.

## Why it took this long: the OOM was never a cache-sizing problem

Every attempt before this session (and several this session) died with
the same signature: fold 1 trains cleanly at ~100% dense-edge cache hit
rate, then something OOMs later -- either at the eval boundary, early in
fold 2, or entering fold 4. The instinct each time was to tune
`--dense-edge-gpu-cache-gb` (tried 6, 8, 10, 12, 14 across this session
and the last). **None of those were the right lever**:

- 10 and 12 destabilized *training itself* (hit rate churned back toward
  0%, epoch_time blew up 10-16x) -- the 15GB default isn't a comfortable
  ceiling with slack to trim, it's close to nfreqs=16's real working-set
  floor.
- 14 died at the **exact same** crash number (20.75GB allocated) as the
  default 15 -- proof a 1GB budget change did nothing, so the cache
  budget wasn't what was overflowing at all.

The two real fixes, once actually found:

1. **Clear `DenseEdgeMemCache` both before AND after `predict_proba` each
   fold** (`a3a776c`, `bf4339f`). Eval computes dense-edge tensors for a
   disjoint window set (guaranteed cache miss) and, since read/write on
   that cache was never train/eval-aware, its own computation refilled
   the cache with entries just as dead to the NEXT fold as training's
   entries were to eval. A diagnostic print added mid-session
   (`[post-fold teardown] cuda allocated=`) proved this wasn't a
   cross-fold leak at all -- every teardown read a flat 0.02GB across 3
   completed folds, right up until the fold-4 crash.
2. **`--precompute-chunk-size 2`** (default `min(batch_size,4)=4`) --
   lowers how many trials get computed concurrently during dense-edge
   chunk-building. This is what actually eliminated the fold-4 OOM; cache
   budget tuning never touched it because the transient peak wasn't
   coming from the cache's steady-state size at all.

Full attempt-by-attempt writeup, in order, with what should have been
inferred sooner at each step: `FAILURE_LOG.md` #1-13.

## coh_affine_rescale: bringing back "significance" honestly, but it didn't help

`temporal_graph_edge_drop_significance=True` (default since 2026-09-15)
permanently dropped the old 4th dense-edge channel because it was named
"significance" but is really just `(coh - threshold)/threshold` -- an
affine RESCALE of coherence under one fixed global threshold, not a real
significance test. Correctness-driven removal, but the 2026-08-31 ablation
(NEGATIVES.md item 3) had found it cost ~0.10 mean AP, because a
deterministic-but-differently-scaled view of the same signal is still
useful to an undertrained linear layer (it's a free instance-norm-style
recentering, not new information -- see the discussion in this session's
transcript for the fuller reasoning).

Added `--temporal-graph-edge-coh-affine-rescale` (`130aef7`) to get that
feature back under `--temporal-graph-edge-complex-native` WITHOUT paying
the memory cost of a 3rd cached channel -- `coherence == |re + i*im|^2`
exactly, so it's recomputed from values already resident in VRAM at
forward() time, concatenated onto `temporal_edge_cproj`'s output before
`temporal_edge_cproj_out` (widens that Linear by `nfreqs`).

Caught and fixed a real bug before running it for real (`1d57652`): the
cached `[re,im]` pair is already zeroed outside the cone of influence, but
`(0 - threshold)/threshold = -1`, not 0 -- the recomputed feature was
leaking a nonzero constant into every COI-invalid cell, unlike the legacy
channel which explicitly re-zeroed via `coi_valid`. Fixed by re-inferring
COI-validity from `coh_mag != 0` and re-zeroing to match.

**Result, once actually run: mean AP 0.491** -- within the run-to-run noise
already seen between the two baseline runs (0.484 vs 0.481, a 0.003
spread), NOT the ~0.10 AP recovery the original ablation measured.
epoch_time also went up ~11% (3.83s vs 3.45s) from the wider layer. Open,
unresolved question: whether concatenating this feature *after* the
complex projection (this implementation) is a structurally weaker
injection point than feeding it *before* projection as one of 4 stacked
real-valued channels (the original, now-permanently-dropped
architecture) -- those are genuinely different places in the network for
the same underlying scalar to enter, and this session didn't test the
alternative.

## Infra: the SSM auto-push had been silently broken for a while

Every nfreqs=16 GPU run this session finished with `could not read deploy
key from SSM (/eeg/github-deploy-key) -- cannot push` -- hit 3 times
before actually being root-caused rather than just recovered from S3 each
time. The parameter is encrypted with the AWS-managed default key
(`alias/aws/ssm`), but the `eeg-gpu` IAM role's `eeg-ssm-and-sns` policy
only granted `kms:Decrypt` on a different, specific customer-managed key
-- so every box could read the ciphertext but never decrypt it. Fixed via
`aws iam put-role-policy` (added a `kms:Decrypt` statement scoped to
`alias/aws/ssm`). **Not yet verified end-to-end** -- no run has completed
since the policy change landed; the next launch should confirm `boot.log`
shows a real push succeeding.

## What's actually new on `main` after this session

- `Epilepsy/pipelines/dense_edge_cache.py`: `DenseEdgeMemCache.clear()`.
- `Epilepsy/run_pipelines.py`: pre+post-eval cache clears, CUDA
  `empty_cache()` in per-fold teardown (was MPS-only), `--precompute-chunk-size`
  now load-bearing for nfreqs=16 (was previously a Runpod/high-RAM-only
  throughput knob), `--temporal-graph-edge-coh-affine-rescale` CLI flag.
- `Epilepsy/pipelines/cwt_gnn_classifiers.py`:
  `temporal_graph_edge_coh_affine_rescale` threaded through
  `SparseEvidenceGNNCore`/`SparseEvidenceGNNClassifier`, COI-correct.
- `FAILURE_LOG.md` (new file, repo root): chronological failure/root-cause
  log for this config, meant to be read before every future launch and
  appended to on every future failure.
- `LAUNCH_CHECKLIST.md`: current-best command updated to the verified-good
  `chunk2`-era flags.
- IAM: `eeg-gpu` role's `eeg-ssm-and-sns` policy now grants `kms:Decrypt`
  on `alias/aws/ssm`.

## Open threads for next session

1. Confirm the SSM/IAM fix actually works end-to-end (auto-push succeeds
   on a run that hasn't needed manual S3 recovery).
2. `coh_affine_rescale` as implemented doesn't recover the AP the old
   4-channel "significance" did -- either try feeding it in pre-projection
   (closer to the original architecture) instead of post-projection
   concatenation, or conclude the feature's value was architecture-position-
   dependent and drop this flag.
3. `LAUNCH_CHECKLIST.md`'s command is now the trusted default for nfreqs=16
   -- no more re-deriving cache-gb/chunk-size from scratch, see
   `FAILURE_LOG.md` for why that's expensive.
