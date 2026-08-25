# Session notes — DBConformer baseline runs, chb01 prediction LOSO (2026-08-25)

Branch: `continuous-cwt-mamba`. Subject chb01, all 6 leave-one-seizure-out
folds, `label_mode=prediction` (preictal/interictal, sph=300s, sop=900s).
`--pipeline dbconformer` (`Epilepsy/pipelines/dbconformer_classifier.py`,
vendored from the DBConformer paper's own `models/DBConformer.py`).
`device="mps"` throughout — this machine is a 16GB-unified-memory Mac.

This note covers only the DBConformer runs and what they say about its own
hyperparameters. A pipeline-vs-pipeline comparison against GRU/Mamba
(`full_6fold_23ch_encoderfree_val_gru.md`) is a separate note, not this one.

Command for every run below (identical except where noted):
```
python Epilepsy/run_pipelines.py --pipeline dbconformer --label-mode prediction
```
All use `_DBCONFORMER_SHARED_PARAMS`/`PREDICTION_DBCONFORMER_PARAMS`
(`run_pipelines.py`) as the base config: `emb_size=40`, `patch_size=128`,
`spa_dim=16`, `gate_flag=False`, `posemb_flag=True`, `branch="all"`,
`chn_atten_flag=True`, `batch_size=32`, `lr=1e-3`, `weight_decay=1e-4`,
`validation_split=0.2`, `early_stopping_patience=5`, `epochs=20`
(`DEFAULT_PREDICTION_EPOCHS`) — this repo's own standard prediction-mode
training protocol, the same one GRU/Mamba use, not the DBConformer paper's
(paper's `max_epoch=100` is denominated in cross-subject-transfer
source-loader iterations, not comparable). Varying knob noted per run.

---

## Is this "the paper's baseline"? No — and it can't be

Before any numbers: DBConformer's own paper never reports a number this
reproduces. Its seizure-detection results are on the authors' CHSZ and
NICU datasets, not CHB-MIT; its LOSO script (`DBConformer_LOSO.py`) is
leave-one-*subject*-out across many subjects on MI/ERP data, not
leave-one-*seizure*-out on a single subject. What's here is the
DBConformer *architecture* (confirmed unmodified except plumbing — see
`dbconformer_classifier.py`'s module docstring) run under this repo's own
CHB-MIT protocol, at hyperparameters adapted from the paper's MI defaults
where a clean match existed. It's an off-the-shelf-architecture baseline,
not a paper reproduction — see the "Right now" entry in `CONTEXT.md` this
session updated, and the extended reasoning in the chat transcript this
note summarizes.

---

## Run 1 — baseline: depth=5, weighted (`175207`)

The first non-smoke run at this scale. `tem_depth=chn_depth=5` (an
unlabeled starting-point value, not the paper's, not deliberately chosen),
`use_class_weights=True` (this repo's automatic full inverse-frequency
weighting per fold, `common.py`'s `TorchEEGClassifier._criterion`).

CSVs: `prediction_leave_one_seizure_out_20260825-175207.csv`,
`prediction_per_seizure_20260825-175207.csv`.

| seizure | n_test (pre) | hit raw→sm | FAR/h raw→sm | prec | rec | f1 | **AP** | AUC |
|---|---|---|---|---|---|---|---|---|
| `1_03_0` | 750 (30) | T→T | 15.33→10.17 | 0.227 | 0.900 | 0.362 | 0.279 | 0.922 |
| `1_04_0` | 650 (30) | T→T | 27.29→22.45 | 0.175 | 1.000 | 0.299 | 0.405 | 0.951 |
| `1_15_0` | 718 (30) | T→T | 8.20→2.09 | 0.390 | 1.000 | 0.561 | 0.518 | 0.980 |
| `1_16_0` | 743 (23) | T→T | 9.17→6.17 | 0.295 | 1.000 | 0.455 | 0.883 | 0.994 |
| `1_18_0` | 750 (30) | T→**F** | 1.00→0.00 | 0.250 | 0.067 | 0.105 | 0.362 | 0.962 |
| `1_26_0` | 630 (30) | T→T | 9.40→6.80 | 0.299 | 0.667 | 0.412 | 0.203 | 0.904 |

**Mean:** accuracy 0.897, precision 0.273, recall 0.772, f1 0.366,
**AP 0.442**, AUC 0.952, FAR/h raw 11.73, FAR/h sm 7.95, hit raw 6/6,
hit k-of-n 5/6 (`1_18_0` misses under smoothing, same fold GRU/Mamba both
also struggle with in the separate comparison note).

Pattern worth flagging on its own: recall pinned at 1.0 on three folds
(`1_04_0`, `1_15_0`, `1_16_0`) while precision stays 0.18–0.39 — the
model over-predicts positive. Two follow-up runs below probe why.

---

## Run 2 — depth=6, weighted (`200637`)

Same config, `tem_depth=chn_depth=6` — the DBConformer authors' own
`DBConformer_LOSO.py` default for every MI/ERP dataset except two BNCI
ones (which get 2). Tried on the theory that 5 was nobody's deliberate
choice and 6 is at least *someone's* deliberate choice.

CSVs: `prediction_leave_one_seizure_out_20260825-200637.csv`,
`prediction_per_seizure_20260825-200637.csv`.

**Mean:** accuracy 0.889, precision 0.159, recall 0.583, f1 0.244,
**AP 0.290**, AUC 0.927, FAR/h raw 11.76, FAR/h sm 9.12, hit raw 4/6,
hit k-of-n 4/6.

**Regressed on every metric vs. depth=5** — most strikingly, `1_15_0` and
`1_18_0` both collapse to precision=recall=0 (the model predicts nothing
positive on either fold's test set). Consistent with more transformer
capacity overfitting a ~3,500-window-per-fold single-subject dataset the
paper's depth=6 default was never calibrated for (their MI/ERP datasets
pool many subjects' trials — much more data per training run than a
single CHB-MIT subject's LOSO folds). The paper's own choice to use
depth=2 for their two smallest datasets is a hint in the same direction.

---

## Run 3 — depth=3, weighted (`203614`)

Follow-up to Run 2: if 6 (more capacity) is worse than 5, is 5 just one
end of a monotonic "less capacity is better" trend, or a local optimum?
`tem_depth=chn_depth=3`, otherwise identical to the baseline.

CSVs: `prediction_leave_one_seizure_out_20260825-203614.csv`,
`prediction_per_seizure_20260825-203614.csv`.

**Mean:** accuracy 0.899, precision 0.254, recall 0.633, f1 0.325,
**AP 0.355**, AUC 0.938, FAR/h raw 10.76, FAR/h sm 7.33, hit raw 6/6,
hit k-of-n 4/6.

**Also worse than depth=5** (AP 0.442→0.355, f1 0.366→0.325, hit k-of-n
5/6→4/6), though less severely than depth=6. **Depth=5 is a local
optimum among {3, 5, 6}, not one end of a trend** — capacity below or
above it both cost performance on this task. No further depth sweep run
(2 and 4 untested) — diminishing returns on a 3-point result that already
answers the actual question (is 5 defensible; yes).

---

## Run 4 — depth=5, unweighted (`202350`)

Diagnostic on the baseline's own recall/precision pattern (see Run 1's
closing note): does turning off `use_class_weights` fix it?
`tem_depth=chn_depth=5` (back to baseline), `use_class_weights=False`.

CSVs: `prediction_leave_one_seizure_out_20260825-202350.csv`,
`prediction_per_seizure_20260825-202350.csv`.

**Mean:** accuracy 0.914, precision 0.322, recall 0.500, f1 0.293,
**AP 0.379**, AUC 0.934, FAR/h raw 8.12, FAR/h sm 5.23, hit raw 6/6,
hit k-of-n **3/6**.

Precision did improve (0.273→0.322) as the recall-saturation pattern
predicted — but not for free: recall fell hard, f1 fell (0.366→0.293), AP
fell (0.442→0.379), and the k-of-n smoothed hit rate dropped from 5/6 to
3/6 (lost `1_15_0` and `1_26_0` in addition to `1_18_0`). **A genuine
precision/recall tradeoff, not a bug fix** — on the metrics that matter
most for this task (AP, event-level reliability), weighted is better.

---

## Conclusion — reported baseline stays Run 1

`tem_depth=chn_depth=5`, `use_class_weights=True` (`175207`) is the
best-performing config found across all four runs, and it's also the one
matching GRU/Mamba's training protocol (same `use_class_weights`
mechanism, same `epochs`/`early_stopping_patience`/`validation_split`).
Both follow-up axes (depth, class weighting) were explored and both
pointed back to the original config, not away from it — this isn't
"we didn't get around to tuning it further," it's "the two most obvious
levers were tried and neither helped."

`run_pipelines.py`'s `_DBCONFORMER_SHARED_PARAMS`/
`PREDICTION_DBCONFORMER_PARAMS` comment block records both diagnostic
results inline for anyone who considers re-tuning either knob.

**175207's numbers are what should be cited as "the DBConformer number"
until/unless it's deliberately re-tuned** (a disclosed search, applied
evenly to GRU/Mamba too if the comparison is meant to stay fair — see the
chat transcript's discussion of why an undisclosed per-model search would
make the comparison meaningless).

---

## Aside: the kernel panic — see CONTEXT.md's gotcha, not re-litigated here

A real macOS kernel panic (`SOCD report: iBoot async abort`, an
AMCC/memory-controller-level panic — not an ordinary OOM) at 18:28:04
coincided with this session's early DBConformer activity. A different
concurrent session did the actual investigation (real profiling under an
RSS-limit watchdog, multiple reproduction attempts) and pinned it on
`--pipeline slimseiz` stage 1 (`select_slimseiz_channels`, real-scale
PCA/SMOTE/DecisionTree over 23 channels x 30 iterations) specifically —
not on running two jobs at once, and not on DBConformer. See
`CONTEXT.md`'s "Known gotchas" section for the full writeup, don't
duplicate it here. Runs 2–4 in this note were subsequently run under a
memory-pressure watch (auto-kill below 12% free, armed for the first of
them) as a precaution; none came close to triggering it (30%+ free
throughout), and no further panics occurred, including while running
concurrently with the other session's own slimseiz job (which was using
`--slimseiz-fixed-channels`, bypassing stage 1 entirely). DBConformer
itself is not implicated by this investigation.

## Open

- Depths 2 and 4 untested (3, 5, 6 covered; 5 is the local optimum found).
- `1_18_0` is a hard fold across every DBConformer variant tried here
  (misses smoothed in 3 of 4 runs) — same fold Mamba struggles with in
  the separate GRU/Mamba comparison note, different from GRU's own
  persistent miss (`1_26_0`). Worth a feature-level look if DBConformer
  is pursued further, not investigated here.
- No disclosed/pre-committed hyperparameter search has been run (only
  two single-knob diagnostics, both negative). If DBConformer is meant to
  be tuned for real rather than left as an off-the-shelf baseline, that's
  still open, and should be matched by equivalent tuning effort on
  GRU/Mamba if the result is meant to inform a model choice.
- Pipeline-vs-pipeline (DBConformer vs. GRU vs. Mamba) writeup: separate
  note, not started here.
