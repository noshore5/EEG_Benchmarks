## BEFORE EVERY LAUNCH: read FAILURE_LOG.md

FAILURE_LOG.md (repo root) is the full chronological list of every failed
nfreqs=16 spot run, its root cause, and what should have been done
differently. Consult it before every launch to avoid repeating a mistake
already made once (shrinking cache-gb as a blind fix, launching on the
wrong branch, assuming a fix is deployed without grepping for it, etc.).
**Every time a run fails, add a new entry to FAILURE_LOG.md** (symptom,
root cause, what should have been done differently, fix if any) before
relaunching — this list only stays useful if it's kept current in real
time, not reconstructed from memory later.

## CURRENT BEST LAUNCH COMMAND (overwrite this block, don't append below it — last updated 2026-09-19)

branch: main
commit: d175a9c (CUDA per-fold empty_cache, pre-eval cache-clear, post-eval
  cache-clear, post-teardown diagnostic print, AND --precompute-chunk-size 2
  as the actual fold-4 OOM fix — all load-bearing)
verified-good full-6-fold run: CONFIRMED 2026-09-20.
  tgm-nfreqs16-chunk2-seed42 (i-05b0363c9f2a0ab5e) completed all 6 folds
  cleanly -- see Epilepsy/Session_notes/2026_09_20/tgm-nfreqs16-chunk2-seed42.md
  for full results (mean AP=0.484, roc_auc=0.941, 6/6 raw hit rate).
  Results committed manually (box's own auto-push failed: SSM deploy-key
  unreadable -- see note in that session-notes file, needs fixing before
  relying on auto-push again).

  Full attempt history (10 failed runs before this one) is in
  FAILURE_LOG.md entries #7-12 -- the short version: neither the eval-time
  OOM nor the fold-4 OOM were cache-BUDGET problems (shrinking
  --dense-edge-gpu-cache-gb to 10/12/14 either destabilized training or did
  nothing). The real fixes were (a) clearing the dense-edge mem cache both
  before AND after eval each fold (a3a776c, bf4339f) and (b)
  --precompute-chunk-size 2 to lower the transient VRAM peak during
  dense-edge chunk-building, which is what actually got past fold 4.

scripts/eeg-run-spot.sh \
  --name <run-name> \
  --branch main \
  --docker-image ghcr.io/noshore5/eeg_benchmarks-mamba:latest \
  --cmd 'python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba \
    --label-mode prediction --device cuda --seed 42 --nfreqs 16 \
    --dense-edge-gpu-cache --precompute-chunk-size 2 \
    --temporal-graph-edge-complex-native \
    --checkpoint-dir /root/checkpoint' \
  --session-note '<what changed since last verified run>'

Known-required flags for nfreqs=16 (all confirmed present on main as of
this commit — recheck with --help before trusting this list):
  --temporal-graph-edge-complex-native   (2ch cache, ~half footprint vs the old 4ch stack)
  --checkpoint-dir /root/checkpoint      (fold-level resume on spot reclaim)
  --dense-edge-gpu-cache                 (DO NOT pass --dense-edge-gpu-cache-gb --
    an explicit override of 10, tried 2026-09-19 to fix the eval OOM below,
    was the WRONG lever: it dropped train-time cache hit rate from 100% to
    62.5% and epoch_time from 3.45s to 56.23s. The real fix is the cache
    CLEAR before eval below, which needs the full training budget to still
    work. 2026-09-20: leaving this flag unset now AUTO-SIZES the budget
    from the box's actual VRAM (total - 8GB headroom, see --help) instead
    of the old hardcoded 15.0 default -- still ~15GB on a g5.xlarge's 23GB
    A10G, but now correct if the launcher lands on a different card too.)

EVAL-TIME OOM finding + fix (2026-09-19, tgm-nfreqs16-native-leakfix-seed42,
instance i-0afdc4455758fbdcd, g5.xlarge): the run did NOT hit the
cross-fold leak below -- it never got past fold 1. Training completed all
20 epochs cleanly (3.45s/epoch, 100% cache hit rate throughout). At the
train->eval boundary:
  [eval boundary] cuda allocated=13.48GB reserved=23.18GB mem_cache_nbytes=13.46GB
i.e. the dense-edge cache was already sitting near its 15GB budget from
training. Eval uses a different (held-out) window set than training, so
it's a cache MISS -- `[dense-edge mem cache] 0/32 trials reused (0.0%)` --
and computing fresh dense-edge tensors for the test set on top of an
already-13.46GB-full cache pushed allocated memory to 21.88GB and OOM'd
(`Tried to allocate 476.00 MiB` on the 22.06GB card). NOT the cross-fold
leak below -- a distinct failure mode.
  FIX (commit a3a776c): DenseEdgeMemCache.clear() + call it right before
  predict_proba each fold, in leave_one_seizure_out_prediction. Training's
  cache entries are guaranteed dead weight by eval time (disjoint window
  sets), so clearing trades away nothing already earned and gives eval the
  full cache-gb budget as headroom. Next fold's training refill falls back
  to the on-disk dense-edge cache (not full recompute) for windows shared
  across folds, so this should NOT reproduce the shrunk-budget slowdown
  seen with --dense-edge-gpu-cache-gb 10 -- but this is UNVERIFIED, watch
  the next run's per-fold epoch_time to confirm training speed holds up
  after the first eval clears the cache.

Cross-fold memory leak fix (commit 7e010af, 2026-09-19): the per-fold
teardown in leave_one_seizure_out_prediction called torch.mps.empty_cache()
but NEVER torch.cuda.empty_cache() -- on CUDA this let PyTorch's caching
allocator accumulate fold-over-fold (seed17 OOM'd with 19.23GB allocated,
not just fragmented, on a 22GB card) until a late fold OOM'd. This is
almost certainly why seed12/15/17/20 all OOM'd partway through despite
fold 1 running cleanly at near-100% cache reuse. Confirm any future OOM
traceback isn't this regressing before assuming it's a cache-sizing issue.

Env (baked into eeg-run-spot.sh, not passed by hand):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   (fixes fragmentation OOM;
    added 2026-09-19, confirmed it gets a fold past the batch-1 OOM that
    hit every earlier nfreqs=16 attempt, T4 and A10G alike)

Instance: g5.xlarge only. Do NOT accept g4dn.xlarge (T4, 16GB) for this
config — kill and retry if the launcher lands on one.

Evidence trail / do NOT re-derive from scratch:
  - Every nfreqs=16 attempt before 2026-09-19's fixes OOM'd or crashed --
    on T4 (14.56GB) AND on real 22GB A10G boxes (seed12, seed15, seed17).
    This was never a "T4 is just slower" story -- see AWS_INFRA.md/CONTEXT.md
    if that framing resurfaces.
  - seed10/15/17/18/20 (pre-native-complex, cache-gb=15 default, no
    empty_cache fix): fold 1 clean, 100% cache reuse, ~3.45-3.48s/epoch
    steady state -- then OOM'd in a later fold from the leak above.
  - --dense-edge-gpu-cache-gb 6 and =8 (2026-09-19, with native-complex):
    both too small -- hit rate peaked ~34% then churned back to 0% within
    one epoch, epoch_time stayed ~42-55s, no real speedup.
