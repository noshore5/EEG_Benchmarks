# `temporal_graph_aggregate="post"` 6-fold A/B -- statistical wash, not worth the 11x compute (2026-08-28)

**TL;DR:** full 6-fold done. post mean AP **0.639** vs pre **0.674**
(within the 0.29-1.00 per-fold noise band), post ROC-AUC **0.970** vs pre
**0.94**, post raw+smoothed hit rate 6/6 vs pre 6/6 raw / 5/6 smoothed.
post beats or ties pre on 4 of 6 folds; the whole AP gap is one fold
(`1_03`, 0.41 vs 0.79). Net: a genuine tie, arguably a marginally better
*global* ranker, but ~11x slower per epoch and not a clear enough win to
flip the default. **Keeping `"pre"`. Knob stays in the code, parked.**

Follow-on to `temporal_graph_mamba_wideband_negative_and_aggregate_knob.md`
(same session). That note describes the knob's implementation and
rationale; this note records the A/B result.

## The question

`temporal_graph_mamba` "pre" (the AP 0.674 leader) scatter-means the
253-edge coherence graph down to 23 node sequences **before** the Mamba,
so the temporal model only ever sees a per-timestep 23-node average.
"post" runs `_DenseEdgeMambaTemporal` over each of the ~253 edge
sequences first, **then** scatter-means the per-edge summaries to nodes
-- the temporal model keeps the full edge graph. Cache is byte-identical
(`[B, 4, E, T', F]`); this is a pure forward-path reordering. Cost is
~E/C (~11x) more Mamba rows -> ~7x slower epochs (433s vs ~60s).

Hypothesis: retaining the edge graph through the temporal model should
help, since the pre-aggregation throws away which channel *pair* carried
a coherence transient.

## Setup

- Only `temporal_graph_aggregate` changed: `"pre"` -> `"post"`. Baseline
  band (8-40 Hz, nfreqs=8), baseline widths (`temporal_graph_edge_dim=8`,
  `hidden_dim=8`), baseline reg (`weight_decay=1e-4`, dropout 0). Same
  `dst_idx` / `temporal_node_in_degree` divisor.
- `--pipeline temporal_graph_mamba --label-mode prediction --device cpu`,
  full 20-epoch budget, patience 5, 6-fold LOSO on chb01. pid 93605,
  log `scratchpad/tgm_post.log`.
- Launched from a working tree with `temporal_graph_aggregate="post"`
  set; the committed default on `main` is `"pre"` (reverted after
  launch -- Python doesn't re-read source post-import).

## Result

### Fold 1 (`1_03_0`): AP 0.407 -- clear regression (baseline 0.792)

| metric | "pre" baseline | "post" | 
|---|---|---|
| AP (auc_pr) | **0.792** | **0.407** |
| precision / recall | 0.213 / 1.000 | 0.219 / 1.000 |
| FAR/h raw->sm | 18.5->17.2 | 17.8->17.3 |
| hit raw/sm | True/True | True/True |

The operating point at the fixed 0.5 threshold is essentially unchanged
(precision/recall/FAR nearly identical). What got **worse is the ranking
itself** -- AP nearly halved. And this is despite healthy-looking
validation: fold 1's training climbed cleanly (val_roc_auc 0.97,
val_loss 0.149 at the restored epoch 11, early-stopped epoch 16). So
"post" trains fine and ranks its own validation split fine; it just
generalizes worse to the held-out seizure. Retaining the full edge graph
through the temporal model gave the model *more* to overfit on ~30
preictal windows, not a better inductive bias.

Timing: epoch 1 ~1043s (partial cold cache), epochs 2-16 ~433s each,
fold ~2h.

### Fold 2 (`1_04_0`): AP 0.837 -- a wash (baseline 0.830)

| metric | "pre" baseline | "post" |
|---|---|---|
| AP (auc_pr) | 0.830 | 0.837 |
| precision / recall | 0.140 / 1.000 | 0.131 / 1.000 |
| FAR/h raw->sm | 35.6->26.3 | 38.5->29.4 |
| hit raw/sm | True/True | True/True |

Ranking essentially tied; FAR/h slightly worse. So "post" is not
*uniformly* worse than "pre" -- fold 1 regressed hard, fold 2 is a tie.
Running mean after 2 folds: post 0.622 vs pre 0.811 -- but that's fold 1
doing all the damage, and fold-1 AP is one seizure's luck (see the
0.29-1.00 spread in the roundup note). Verdict still pending folds 3-6.

### Fold 3 (`1_15_0`): AP 0.487 -- a win (baseline 0.331)

precision 0.580 / recall 0.967 / f1 0.725, FAR/h 3.7->1.2 (baseline
0.396 / 0.633 / 0.487, FAR 5.1->1.6). Better ranking *and* better
operating point on this fold.

Running mean after 3 folds: post 0.577 vs pre 0.651. Fold 1 still the
whole gap.

### Fold 4 (`1_16_0`): AP 0.994 -- wash (baseline 0.996)

Both near-perfect. precision 0.920 / recall 1.000 / f1 0.958, FAR/h
0.33->0.00. Restored epoch 18 (ran full 20).

Running mean after 4 folds: post 0.681 vs pre 0.737.

### Fold 5 (`1_18_0`): AP 0.854 -- a win (baseline 0.802)

precision 0.917 / recall 0.733 / f1 0.815 (baseline 0.909 / 0.333 /
0.488 -- pre badly under-recalled this fold), FAR/h 0.33->0.00 both.

Running mean after 5 folds: post 0.716 vs pre 0.750.

### Fold 6 (`1_26_0`): AP 0.257 -- slight loss (baseline 0.293)

precision 0.300 / recall 0.600 / f1 0.400, FAR/h 8.4->5.4 (baseline
0.339 / 0.700 / 0.457, FAR 8.2->5.6). Both weak; near-tie.

## Full 6-fold result

| Fold | pre AP | post AP | delta |
|---|---|---|---|
| `1_03_0` | 0.792 | 0.407 | **-0.385** |
| `1_04_0` | 0.830 | 0.837 | +0.007 |
| `1_15_0` | 0.331 | 0.487 | +0.156 |
| `1_16_0` | 0.996 | 0.994 | -0.002 |
| `1_18_0` | 0.802 | 0.854 | +0.052 |
| `1_26_0` | 0.293 | 0.257 | -0.036 |
| **mean AP** | **0.674** | **0.639** | -0.035 |
| **mean ROC-AUC** | 0.94 | **0.970** | +0.03 |
| hit rate raw / sm | 6/6 / 5/6 | 6/6 / 6/6 | -- |
| mean FAR/h sm | 8.44 | 8.90 | +0.46 |
| mean f1 | ~0.49 | 0.582 | -- |

CSVs: `Epilepsy/results/temporal_graph_mamba/prediction/
prediction_leave_one_seizure_out_20260828-161354.csv` (+ per-seizure).

## Verdict

**A wash.** On 5 of 6 folds "post" is within noise of "pre" or better
(3 wins, 2 ~ties). The -0.035 mean-AP gap is entirely `1_03`, and one
fold's AP on this 1-subject / 6-seizure setup is a coin flip (the pre
baseline's own per-fold AP spans 0.29-1.00). "post" actually looks like
a slightly *better global ranker* -- mean ROC-AUC 0.970 vs 0.94, and it
fixed the one smoothed-hit miss (`1_18`, pre recall 0.33 -> post 0.73).

But it costs ~11x the Mamba rows / ~7x the epoch time, and "better AUC,
same-ish AP, worse FAR" is not a clear enough win to justify that or to
flip a default. **`"pre"` stays the committed default. Knob stays in
the code (documented, `"post"` opt-in).**

This also revises the "post regressed" framing from earlier in the run
(and in the roundup note): fold 1 looked like a clean regression; the
full 6-fold shows it was fold-1 variance, not a real capacity failure.
"post" belongs in the "tried, roughly equivalent, not adopted" bucket,
not the "negative" bucket with wide-band / reg-tuning.

## If resumed

- The ROC-AUC edge (0.970 vs 0.94) is the only real signal. If a *better
  ranker* is what's wanted (e.g. feeding a downstream threshold
  calibrator), "post" is mildly preferable -- but calibration should be
  built and tested on "pre" first since it's 7x cheaper to iterate on.
- Seed-repeat both "pre" and "post" (2-3 seeds each) if this ever needs
  to be more than "a wash" -- the per-fold noise is too large to
  distinguish 0.639 from 0.674 on one run.
- Not worth a capacity-compensation search (smaller `hidden_dim` etc.)
  -- low priority given it's already a tie.
