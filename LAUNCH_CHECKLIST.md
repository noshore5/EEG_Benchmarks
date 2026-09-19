## CURRENT BEST LAUNCH COMMAND (overwrite this block, don't append below it — last updated 2026-09-19)

branch: main
commit: 7e010af (has the CUDA per-fold empty_cache fix — real but NOT what's
  currently failing; see eval-time OOM finding below)
verified-good full-6-fold run: NOT YET CONFIRMED — tgm-nfreqs16-native-leakfix-seed42
  OOM'd inside fold 1's OWN eval step (never reached fold 2). Root cause
  found 2026-09-19 below. Next attempt must lower --dense-edge-gpu-cache-gb.

scripts/eeg-run-spot.sh \
  --name <run-name> \
  --branch main \
  --docker-image ghcr.io/noshore5/eeg_benchmarks-mamba:latest \
  --cmd 'python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba \
    --label-mode prediction --device cuda --seed 42 --nfreqs 16 \
    --dense-edge-gpu-cache --dense-edge-gpu-cache-gb 10 \
    --temporal-graph-edge-complex-native \
    --checkpoint-dir /root/checkpoint' \
  --session-note '<what changed since last verified run>'

Known-required flags for nfreqs=16 (all confirmed present on main as of
this commit — recheck with --help before trusting this list):
  --temporal-graph-edge-complex-native   (2ch cache, ~half footprint vs the old 4ch stack)
  --checkpoint-dir /root/checkpoint      (fold-level resume on spot reclaim)
  --dense-edge-gpu-cache --dense-edge-gpu-cache-gb 10   (2026-09-19 REVISION:
    the code default of 15.0 gets train-time cache reuse to 100% at
    ~3.45s/epoch, BUT leaves no headroom for eval-time dense-edge computation
    -- see the eval-boundary OOM finding directly below. 10GB is an
    unverified first guess at leaving ~5-6GB headroom for eval; watch the
    next run's `[eval boundary] cuda allocated=` line and adjust. Do NOT
    go back to explicit 6 or 8 either -- those were tried 2026-09-19 BEFORE
    native-complex-edges was added to the same launch and caused train-time
    cache churn back to 0% reuse; not directly comparable to this new
    eval-headroom problem, but still too small for good train-time hit rate.)

EVAL-TIME OOM finding (2026-09-19, tgm-nfreqs16-native-leakfix-seed42,
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
(`Tried to allocate 476.00 MiB` on the 22.06GB card). This is a genuinely
different failure mode from the cross-fold leak below: the 15GB default
cache budget doesn't leave enough VRAM headroom for eval-time dense-edge
compute on uncached windows. Fix direction: lower --dense-edge-gpu-cache-gb
to leave headroom (untested guess: 10GB), or evict/shrink the cache before
eval starts, or chunk eval's dense-edge computation smaller. NOT a
leak/accumulation issue -- don't confuse with the empty_cache fix below.

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
