# tgm-nfreqs16-ssmverify-seed42 — retest of the SSM deploy-key fix (still broken)

Reproduction of `chunk2-seed42`'s exact command (commit `4155336`, launched
via the new `eeg-run-spot.yml` GitHub Actions workflow instead of a manual
shell), specifically to confirm whether last session's KMS `kms:Decrypt`
fix for the SSM deploy-key auto-push gap (FAILURE_LOG.md #13) actually
works end-to-end.

Instance: g5.xlarge, `i-0c1fcbd75f946e3ac`.

## Results (mean across 6 folds)

| metric | value |
|---|---|
| accuracy | 0.8827 |
| precision | 0.3873 |
| recall | 0.7778 |
| f1 | 0.4088 |
| average_precision (AP) | 0.4836 |
| roc_auc | 0.9431 |
| event-level hit rate (raw) | 6/6 |
| event-level hit rate (smoothed) | 5/6 |

Matches `chunk2-seed42`/`chunk2-seed42-v2` (AP 0.4842 / 0.4813) within
normal run-to-run noise — the underlying pipeline is unchanged and stable.

## SSM fix: still broken, 4th occurrence

`boot.log` showed the identical `could not read deploy key from SSM
(/eeg/github-deploy-key) -- cannot push` as the prior 3 occurrences. The
KMS `kms:Decrypt` policy change applied last session did **not** fix it.

Root cause of *that*: `promote_results.sh`'s SSM read piped stderr to
`/dev/null`, so every prior occurrence — including the one that prompted
the KMS "fix" — was diagnosed purely from theory, never from the actual
AWS error text. Fixed in this session (commit `25f22d4`): stderr is now
captured and surfaced in the failure message, so the next occurrence will
actually be diagnosable. See `FAILURE_LOG.md` #13 for the full history.

## Results recovery

Since auto-push failed, results were recovered manually via a new
`eeg-tail.yml` section (commit `6b37562`/`25f22d4`) that dumps
`exports/runs/<name>/results/*.csv` — the tail role can't read
`checkpoints/*` but can read this. CSVs committed alongside this note:
`results/temporal_graph_mamba/prediction/{prediction_leave_one_seizure_out,prediction_per_seizure}_20260920-173847.csv`.
