# tgm-nfreqs8-cohaffine-seed42 — does affine-rescale stack with nfreqs=8?

Combines the two independent gains found this session: `--nfreqs 8`
(beats nfreqs=16 on its own) and `--temporal-graph-edge-coh-affine-rescale`
(barely moved AP on its own, at nfreqs=16). Same infra otherwise
(complex-native 2ch `[re,im]` cache, `--precompute-chunk-size 2`,
`--dense-edge-gpu-cache`), seed 42.

Instance: g5.xlarge, `i-03f0cfe8eded5c537`.

## Results (mean across 6 folds)

| metric | nfreqs=16 (`ssmverify`) | +affine (`cohaffine`, nfreqs=16) | nfreqs=8 alone | **nfreqs=8 + affine (this run)** |
|---|---|---|---|---|
| accuracy | 0.8827 | -- | 0.9096 | 0.9007 |
| precision | 0.3873 | -- | 0.4674 | 0.4885 |
| recall | 0.7778 | -- | 0.7800 | 0.8000 |
| f1 | 0.4088 | -- | 0.5028 | 0.5402 |
| average_precision (AP) | 0.4836 | 0.4907 | 0.5831 | **0.5827** |
| roc_auc | 0.9431 | 0.9409 | 0.9610 | 0.9601 |
| event-level hit rate (raw) | 6/6 | -- | 6/6 | 6/6 |
| event-level hit rate (smoothed) | 5/6 | -- | 6/6 | 6/6 |

## Verdict: affine's benefit does NOT stack with nfreqs=8 -- it's redundant

AP (0.5827) and roc_auc (0.9601) are statistically indistinguishable from
nfreqs=8 alone (0.5831 / 0.9610) -- well within the run-to-run noise
already seen between baseline reproductions (~0.003 AP spread). f1 and
recall ticked up slightly but that's within the same noise band. Adding
affine-rescale on top of nfreqs=8 bought essentially nothing.

Combined with the nfreqs=16 finding (affine alone: AP 0.484->0.491, also
within noise), this affine-rescale feature does not appear to carry
useful signal in this architecture at either frequency resolution --
consistent with the overfitting-vs-genuine-signal distinction from the
nfreqs ablation: nfreqs=8's gain over nfreqs=16 came from reducing
overfitting (a capacity/generalization effect), which affine-rescale
(a hand-engineered feature) doesn't touch one way or the other.

**Practical implication:** nfreqs=8 without affine-rescale remains the
best config found so far (AP=0.583, simpler, no extra ~11% epoch-time
cost measured for affine at nfreqs=16). No reason to carry the affine
flag forward unless a cheaper/differently-computed version of it is
tried.

## Infra: SSM auto-push -- 5th occurrence, but finally the REAL error

`could not read deploy key from SSM (/eeg/github-deploy-key) -- cannot
push. AWS error: You must specify a region.` Thanks to last commit's
stderr-capture fix, this is the first occurrence where the actual AWS
error was visible -- and it turned out to have nothing to do with the
KMS/IAM theory from occurrence #3. Root cause: `promote_results.sh` never
set `AWS_DEFAULT_REGION`, and the box has no AWS CLI config file. Fixed by
exporting `AWS_DEFAULT_REGION=us-east-1` in the script (commit
`036da74`). Results recovered manually via `eeg-tail.yml`'s results-dump
section as before. Next run should confirm auto-push actually works.
