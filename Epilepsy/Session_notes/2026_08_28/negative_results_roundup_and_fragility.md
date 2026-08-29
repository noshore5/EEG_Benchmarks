# Negative / null-result roundup + why experiments on this setup are hard to read (2026-08-28)

Consolidation note. Over the last few days: two attempts to make
`temporal_graph_mamba` (AP 0.674, the prediction leader) *better* came
back clearly negative, one came back a wash, and the attempt to beat it
with a richer architecture (`hermitian_ssm`) failed outright. Collecting
them because the read-out problem is the same each time and it should
change how the next experiment is run.

## The results

| # | change | expectation | result | note |
|---|---|---|---|---|
| 1 | `hermitian_ssm`: Hermitian channel graph + top-2 eigenpairs + complex spectral encoder | richer than per-node temporal Mamba -> should beat 0.674 | 6-fold AP **0.253** (float16 run, `*_20260828-091710.csv`) -- weakest pipeline in the repo. **NEGATIVE** | `hermitian_ssm_first_6fold_and_eigh_fix.md` |
| 2 | wide band 8-124 Hz (nfreqs 15 then 10) into `temporal_graph_mamba` | 8-40 was a disk-budget pick, never swept -- expect headroom | fold 1_03 AP 0.792 -> **0.31 / 0.24**; killed after 1-2 folds both attempts. **NEGATIVE** | `temporal_graph_mamba_wideband_negative_and_aggregate_knob.md` |
| 3 | reg tuning: `weight_decay` 1e-4->3e-4, dropout 0->0.15 | reduce FAR/h at the fixed 0.5 threshold | fold 1_03 AP 0.792 -> **0.23**; fold 1_04 **failed to train** (restored epoch 1). **NEGATIVE** | `2026_08_27/temporal_graph_mamba_full_6fold_and_tuning_attempt.md` |
| 4 | `temporal_graph_aggregate="post"` (per-edge Mamba, keep the 253-edge graph through the temporal model) | pre-aggregation discards which channel *pair* carried a transient -> keeping it should help | full 6-fold: mean AP **0.639** vs 0.674, ROC-AUC **0.970** vs 0.94, 4/6 folds >= pre, gap is all fold 1_03. **WASH** -- not adopted (11x compute for a tie) | `temporal_graph_aggregate_post_6fold.md` |

## The common thread

Attempts 2, 3, 4 are the **same experiment**: give the model more to work
with -- more frequency detail, more graph detail -- or pull harder on
regularization. 2 and 3 clearly regressed. 4, run to a full 6-fold,
turned out to be a **wash** (mean AP 0.639 vs 0.674, entirely one fold;
actually a slightly *better* global ranker at ROC-AUC 0.970 vs 0.94).
`hermitian_ssm` (1) is the extreme richness bet: a much larger model that
never got off the ground.

Important correction from #4: while it was running, fold 1 (AP 0.41 vs
0.79) looked like a clean regression and got written up that way. The
full 6-fold showed that was **fold-1 variance**, not a capacity failure.
Lesson: partial-run reads on this setup are actively misleading, not just
noisy (see the corollary below).

The bottleneck is **not representational capacity**. The
`temporal_graph_mamba` "pre" pipeline already funnels the full
`[4, E=253, T', F=8]` edge stack through an 8-wide neck and an 8-dim
Mamba and still hits 0.674. Widening or enriching that path has at best
been a wash (#4) and mostly added things to overfit (#2, #3, #1).

Why: **the data budget per fold is tiny.** chb01 LOSO prediction = 1
subject, 6 seizures, one held-out seizure per fold, ~30 preictal windows
in the positive class per fold. Any change that adds effective capacity
(wider band through a linear mixer, per-edge temporal modeling, a
bigger complex encoder) trades against that ~30-sample signal. The
model trains fine on its own validation split (val_roc_auc ~0.97 in
every case) and then generalizes worse to the held-out seizure.

Corollary: **partial-run AP on this setup is worse than worthless -- it
is misleading.** The untuned "pre" baseline's own per-fold AP is 0.29,
0.33, 0.79, 0.80, 0.996 -- a 3x spread with nothing changed. #4's fold 1
(0.41 vs 0.79) read as a clean regression; the full 6-fold was a wash.
#2 and #3 were killed after 1-2 folds (correctly, on cost -- a full
6-fold is ~10-13h) and *were* genuinely bad -- but we can't actually
distinguish "killed early and genuinely bad" from "killed early and just
unlucky on fold 1" without paying for the full run. #4 is the one case
we did pay, and it flipped the verdict.

## What this implies for the next experiment

1. **Stop trying to add capacity / richness.** Bands, neck width,
   per-edge modeling, bigger encoders -- that lever is exhausted on this
   data budget. If `hermitian_ssm` is revived, the eigen-feature
   *quality* fixes (rank-1 collapse, power stripped by coherence
   normalization -- see its note) matter, not more parameters.

2. **The real open lever is the fixed 0.5 decision threshold, not the
   model.** Every "pre" fold has the same tell: high AP, terrible FAR/h,
   precision ~0.2 / recall 1.0. The model *ranks* preictal windows well
   and is scored at a badly-chosen operating point. Calibrating the
   threshold against the validation split's own PR curve is still
   unbuilt and is the most promising untried thing. (Open thread in
   `CONTEXT.md`.)

3. **Seed-repeat the 0.674 baseline before building on it.** Given the
   0.29-1.00 per-fold spread and that a mild reg change collapsed fold
   1_03, we don't actually know 0.674 is stable. 2-3 seeds of the
   untuned "pre" config, full 6-fold, would tell us the error bar we're
   working against -- and would make every future A/B interpretable
   instead of guessed.

4. **If an A/B must run, run it to completion or don't start it.** A
   killed-after-fold-2 result on this setup is not evidence -- #4 proved
   this directly (fold-1 "regression" -> full-run wash). Budget the full
   ~10-13h or use `--max-folds`/`--skip-folds` deliberately, and don't
   write a verdict off a prefix.

5. **More subjects would fix most of this** -- chb01-only is why every
   fold is a coin flip. Not in scope now, but the fragility is a
   data-quantity problem first and a modeling problem a distant second.

## Status of the code

- `main` is clean. `temporal_graph_aggregate` knob committed (`2e48dd2`),
  default `"pre"` -- confirmed the right default by the full #4 run.
  Wide-band revert committed. Reg-tuning edit was reverted (not committed).
- #4 "post" 6-fold CSVs written under
  `Epilepsy/results/temporal_graph_mamba/prediction/*20260828-161354*`.
- `hermitian_ssm` pipeline + 3 result sets on `main`, parked.
- Only uncommitted work anywhere: the new `architecture-diagram-
  temporal-graph-mamba.html` in the separate `noshore5.github.io` repo
  (untracked, user's call whether to deploy).
