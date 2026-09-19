## CURRENT BEST LAUNCH COMMAND (overwrite this block, don't append below it — last updated 2026-09-19)

branch: main
commit: 7e010af (has the CUDA per-fold empty_cache fix below — do not
  launch nfreqs=16 on an older commit, it will OOM in a later fold)
verified-good full-6-fold run: NOT YET CONFIRMED — tgm-nfreqs16-native-leakfix-seed42
  is the first attempt with all fixes below combined, currently in flight.
  Update this line (and the commit above, if it moves) once a run actually
  completes all 6 folds and lands a commit on origin/main.

scripts/eeg-run-spot.sh \
  --name <run-name> \
  --branch main \
  --docker-image ghcr.io/noshore5/eeg_benchmarks-mamba:latest \
  --cmd 'python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba \
    --label-mode prediction --device cuda --seed 42 --nfreqs 16 \
    --dense-edge-gpu-cache \
    --temporal-graph-edge-complex-native \
    --checkpoint-dir /root/checkpoint' \
  --session-note '<what changed since last verified run>'

Known-required flags for nfreqs=16 (all confirmed present on main as of
this commit — recheck with --help before trusting this list):
  --temporal-graph-edge-complex-native   (2ch cache, ~half footprint vs the old 4ch stack)
  --checkpoint-dir /root/checkpoint      (fold-level resume on spot reclaim)
  --dense-edge-gpu-cache                 (DO NOT pass --dense-edge-gpu-cache-gb --
    the code default of 15.0 is close to the real working-set size and is
    what let seed10/15/17/18/20 reach 100% cache reuse in fold 1 at
    ~3.45-3.48s/epoch. Explicit overrides of 6 or 8 tried 2026-09-19 both
    UNDER-sized it and caused cache churn back to 0% reuse -- don't repeat
    that mistake. If a future run OOMs from the cache itself specifically
    -- distinguishable from the fold-to-fold leak below via the
    [dense-edge mem cache] WARNING about oversized_rejections -- retune
    from the default, don't just guess smaller.)

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
