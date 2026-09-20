# tgm-nfreqs16-chunk2-seed42-v2 — reproduction of chunk2-seed42

Same command as tgm-nfreqs16-chunk2-seed42 (commit d175a9c), run again to
confirm reproducibility before trusting the fix. Instance
i-0714f94707dba082b, g5.xlarge, us-east-1a.

## Results (mean across 6 folds)

| metric | v1 (chunk2-seed42) | v2 (this run) |
|---|---|---|
| accuracy | 0.8854 | 0.8906 |
| precision | 0.4044 | 0.4193 |
| recall | 0.7833 | 0.7722 |
| f1 | 0.4290 | 0.4368 |
| average_precision | 0.4842 | 0.4813 |
| roc_auc | 0.9409 | 0.9438 |
| event-level hit rate (raw) | 6/6 | 6/6 |
| event-level hit rate (smoothed) | 5/6 | 5/6 |

Small run-to-run variance (same seed, same code -- likely nondeterminism
from CUDA kernels / cuDNN autotuning), but confirms the fix is genuinely
reproducible, not a one-off lucky run.

## Note on auto-push

Same SSM deploy-key failure as v1 (`could not read deploy key from SSM
(/eeg/github-deploy-key) -- cannot push`) -- second occurrence, see
FAILURE_LOG.md #13. This needs actually fixing, not just working around
with manual recovery a third time.
