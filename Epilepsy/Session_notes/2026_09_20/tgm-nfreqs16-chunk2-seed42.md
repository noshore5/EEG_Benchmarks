# tgm-nfreqs16-chunk2-seed42 — first verified full 6-fold nfreqs=16 run

Commit: d175a9c (dense-edge cache: pre+post-eval clear, CUDA per-fold
empty_cache, `--precompute-chunk-size 2`)
Instance: g5.xlarge, us-east-1a, i-05b0363c9f2a0ab5e
Command:
```
python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba \
  --label-mode prediction --device cuda --seed 42 --nfreqs 16 \
  --dense-edge-gpu-cache --precompute-chunk-size 2 \
  --temporal-graph-edge-complex-native --checkpoint-dir /root/checkpoint
```

## Why this run succeeded where ~10 prior attempts failed (see FAILURE_LOG.md)

The fold-4 OOM that killed every attempt at cache-gb=15, =14, and =12 was
never a cache-budget problem -- proven by cachegb14 dying at the exact same
20.75GB/chunk-3-of-8 spot as cachegb15 (identical crash regardless of a 1GB
budget change). The actual fix was `--precompute-chunk-size 2` (down from
the default `min(batch_size,4)=4`), which lowers how many trials are
computed concurrently during dense-edge chunk-building -- the real source
of the transient VRAM peak, not the cache's steady-state size.

## Results (mean across 6 folds)

| metric | value |
|---|---|
| accuracy | 0.8854 |
| precision | 0.4044 |
| recall | 0.7833 |
| f1 | 0.4290 |
| average_precision (AP) | 0.4842 |
| roc_auc | 0.9409 |
| false_alarms_per_hour | 13.21 |
| false_alarms_per_hour_smoothed | 8.51 |
| event-level hit rate (raw) | 6/6 (100%) |
| event-level hit rate (k-of-n smoothed) | 5/6 (83.3%) |

Per-fold breakdown in `fold_rows.csv`/`per_seizure_rows.csv` (also in
`Epilepsy/results/temporal_graph_mamba/prediction/
prediction_leave_one_seizure_out_20260920-133507.csv` /
`prediction_per_seizure_20260920-133507.csv`).

## Note on how these results got committed

The box's own auto-push (`scripts/promote_results.sh`) FAILED: "could not
read deploy key from SSM (/eeg/github-deploy-key) -- cannot push". Results
were recovered from S3
(`s3://noshore-eeg-benchmarks-827938107865/checkpoints/tgm-nfreqs16-chunk2-seed42/`)
and committed manually instead. **The SSM deploy-key parameter needs
fixing/checking before relying on auto-push for future runs** -- add to
FAILURE_LOG.md / AWS_INFRA.md if this recurs.
