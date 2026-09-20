# tgm-nfreqs16-cohaffine-seed42 — first real test of --temporal-graph-edge-coh-affine-rescale

Commit: 1d57652 (COI-corrected coh_affine_rescale feature)
Instance: g5.xlarge, us-east-1a, i-0ddf477bd1a1ac4f4
Command: same as chunk2-seed42 plus `--temporal-graph-edge-coh-affine-rescale`

## Results (mean across 6 folds)

| metric | chunk2-seed42 (v1) | chunk2-seed42-v2 | cohaffine-seed42 |
|---|---|---|---|
| average_precision | 0.4842 | 0.4813 | 0.4907 |
| roc_auc | 0.9409 | 0.9438 | 0.9409 |
| epoch_time | 3.45s | 3.46s | 3.83s (~11% slower) |

## Verdict

AP moved from ~0.48 to ~0.49 -- within the run-to-run noise already seen
between the two baseline reproductions (0.4842 vs 0.4813, a 0.003 spread).
This is NOT the recovery toward ~0.7 AP the user remembered from an
earlier session on this seed. The affine-rescale feature, recomputed
downstream of the [re,im] cache and COI-corrected, does not reproduce the
~0.10 AP gain the 2026-08-31 ablation (NEGATIVES.md item 3) measured for
the ORIGINAL 4-channel-cached "significance" feature. Possible reasons
(not yet investigated): the original ablation's ~0.10 AP gain may have
included other differences between configs (different seed set, pre-
native-complex/pre-chunk2 code state), or the deterministic-affine-feature
mechanism benefits less when concatenated post-complex-projection (this
implementation) versus when it was one of 4 channels fed pre-projection
(the original architecture) -- these are two structurally different
places in the network for the same underlying feature to enter.

## Infra fix landed during this run

The recurring `could not read deploy key from SSM (/eeg/github-deploy-key)
-- cannot push` (3rd occurrence, see FAILURE_LOG.md #13) was root-caused
during this run: the SSM parameter is encrypted with the AWS-managed
default key (`alias/aws/ssm`), but the eeg-gpu IAM role's policy only
granted kms:Decrypt on a different customer-managed key. Fixed by adding a
kms:Decrypt statement for `alias/aws/ssm` to the eeg-ssm-and-sns role
policy. Untested end-to-end yet (this run's results were still recovered
manually since the fix landed after it finished) -- next run should
confirm auto-push actually works now.
