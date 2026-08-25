# Pipeline comparison — GRU vs Mamba vs DBConformer vs SlimSeiz, chb01 prediction LOSO (2026-08-25)

Deferred from `dbconformer_baseline_runs.md` ("we'll make another later
about the pipeline comparisons"). All four numbers below are chb01,
`label_mode=prediction` (sph=300s, sop=900s), the same 6 leave-one-seizure-out
folds (`1_03_0, 1_04_0, 1_15_0, 1_16_0, 1_18_0, 1_26_0`), and the same
shared training protocol in `run_pipelines.py`: `validation_split=0.2`,
`early_stopping_patience=5`, `epochs=20` (`DEFAULT_PREDICTION_EPOCHS`),
`use_class_weights=True` (automatic full inverse-frequency weighting,
`common.py`'s `TorchEEGClassifier._criterion`) — confirmed identical
across all four pipelines' param blocks, not eyeballed.

**Not identical in every dimension:** GRU/Mamba ran on a Windows/CUDA
(RTX 3070 Ti) box in `dense_edge_mamba`'s own worktree; DBConformer/
SlimSeiz ran on this Mac (`device="mps"`). That affects wall-clock only,
not the accuracy metrics below. More substantively, GRU/Mamba consume
this repo's own CWT+dense-edge-graph features while DBConformer/SlimSeiz
classify raw `(23, n_timepoints)` windows directly — that's the actual
axis being compared (architecture + input representation as a package),
not a confound to control away.

Best-known number per pipeline, per your instruction (DBConformer =
depth=5, weighted; the config each pipeline's own tuning converged on
rather than a fresh undisclosed search — see each source note):

| pipeline | source run | source note |
|---|---|---|
| GRU (dense-edge, encoder-free, full 23ch mesh) | `20260825-105659` | `full_6fold_23ch_encoderfree_val_gru.md` |
| Mamba (dense-edge, matched protocol) | `20260825-113049` | same note, "Matched Mamba comparison" |
| DBConformer (depth=5, weighted) | `20260825-175207` | `dbconformer_baseline_runs.md` |
| SlimSeiz (adaptive per-fold channel select) | `20260825-171651` | `slimseiz_fixed_channel_montage.md` |

SlimSeiz note: its paper-montage fixed-channel variant (`--slimseiz-fixed-channels`)
tied this adaptive run on aggregate f1/precision/FAR-h and is operationally
safer (skips the crash-implicated stage-1 selection entirely) — see that
note. It's not used as the headline number here only because its CSV for
folds 0-3 was lost to an unrelated watchdog-timeout kill, so AP/AUC aren't
available for a full 6-fold mean; the adaptive run has complete data. Treat
SlimSeiz's row below as representative of either config, not a case where
adaptive selection is known to be better.

---

## Mean across 6 folds

| metric | GRU | Mamba | DBConformer | SlimSeiz |
|---|---|---|---|---|
| accuracy | 0.878 | 0.896 | 0.897 | 0.913 |
| precision | 0.343 | 0.348 | 0.273 | 0.286 |
| recall | 0.807 | 0.750 | 0.772 | 0.556 |
| f1 | 0.436 | 0.389 | 0.366 | 0.340 |
| **AP** | 0.423 | **0.499** | 0.442 | 0.431 |
| AUC | 0.944 | 0.953 | 0.952 | 0.951 |
| FAR/h raw | 14.26 | 11.73 | 11.73 | 8.56 |
| FAR/h smoothed | 9.21 | 6.39 | 7.95 | **6.31** |
| hit rate, raw | **6/6** | **6/6** | **6/6** | 5/6 |
| hit rate, k-of-n (smoothed) | 5/6 | 5/6 | 5/6 | **4/6** |

## Per-fold AP (the metric that's mattered most across every prior note here)

| seizure | GRU | Mamba | DBConformer | SlimSeiz |
|---|---|---|---|---|
| `1_03_0` | 0.141 | 0.156 | 0.279 | 0.261 |
| `1_04_0` | 0.339 | 0.480 | 0.405 | 0.355 |
| `1_15_0` | 0.472 | 0.629 | 0.518 | 0.330 |
| `1_16_0` | 0.619 | 0.883 | 0.883 | **1.000** |
| `1_18_0` | 0.776 | 0.618 | 0.362 | 0.425 |
| `1_26_0` | 0.193 | 0.229 | 0.203 | 0.212 |

## Per-fold precision / recall / f1

AP (above) is threshold-independent and hides what's happening at the
actual 0.5 decision threshold `predict()` uses. This is where `1_18_0`'s
story actually lives — it doesn't look like an outlier fold in the AP
table (Mamba scores 0.618 there, its 2nd-best fold), but it's a near-total
recall collapse for every raw-EEG model at the decision threshold:

| seizure | GRU rec | Mamba rec | DBConformer rec | SlimSeiz rec |
|---|---|---|---|---|
| `1_03_0` | 0.900 | 0.800 | 0.900 | 0.967 |
| `1_04_0` | 1.000 | 0.967 | 1.000 | 0.867 |
| `1_15_0` | 1.000 | 0.900 | 1.000 | 0.100 |
| `1_16_0` | 0.739 | 1.000 | 1.000 | 1.000 |
| `1_18_0` | 0.767 | **0.100** | **0.067** | **0.000** |
| `1_26_0` | 0.433 | 0.733 | 0.667 | 0.400 |

| seizure | GRU f1 | Mamba f1 | DBConformer f1 | SlimSeiz f1 |
|---|---|---|---|---|
| `1_03_0` | 0.320 | 0.310 | 0.362 | 0.324 |
| `1_04_0` | 0.231 | 0.283 | 0.299 | 0.331 |
| `1_15_0` | 0.411 | 0.551 | 0.561 | 0.109 |
| `1_16_0` | 0.607 | 0.657 | 0.455 | 0.979 |
| `1_18_0` | **0.754** | **0.171** | **0.105** | **0.000** |
| `1_26_0` | 0.292 | 0.364 | 0.412 | 0.296 |

**`1_18_0` is GRU's single *best* fold on both recall and f1, and it's
every other pipeline's worst-or-tied-worst fold.** That's not a small
effect on the aggregate — recompute mean f1 with `1_18_0` dropped
(5 folds instead of 6) and GRU's "best f1" result inverts:

| pipeline | f1 (6 folds, reported) | f1 (5 folds, excl. `1_18_0`) |
|---|---|---|
| GRU | 0.436 | 0.372 |
| Mamba | 0.389 | **0.433** |
| DBConformer | 0.366 | 0.418 |
| SlimSeiz | 0.340 | 0.408 |

Same flip on recall itself, GRU vs. Mamba specifically (Mamba's the only
one of the other three where the gap is worth isolating — its 6-fold
recall trails GRU's by the widest margin of the three):

| pipeline | recall (6 folds, reported) | recall (5 folds, excl. `1_18_0`) |
|---|---|---|
| GRU | **0.807** | 0.814 |
| Mamba | 0.750 | **0.880** |

GRU's reported recall edge (+0.057) is almost entirely `1_18_0`: exclude
it and GRU barely moves (0.807→0.814, it was already doing fine on the
other five) while Mamba jumps 0.750→0.880 and the lead **flips to
Mamba by 0.066**. So on both recall and f1 — the two metrics GRU's
6-fold numbers beat Mamba's on — the gap is one fold's near-total
recall collapse for Mamba (`1_18_0`: 0.100), not a five-of-six-fold
pattern. Drop that fold and Mamba leads on every metric in this
comparison except accuracy.

Drop the one fold where GRU aces recall and every raw-EEG model
collapses it, and GRU goes from best f1 to worst — Mamba (and
DBConformer, and SlimSeiz) all pull ahead of it. **The headline "GRU has
the best recall/f1" finding above is substantially an artifact of one
fold, not a consistent five-of-six-fold advantage.** This cuts the other
way from a "Mamba was brought down by one bad fold" story: Mamba's f1
*was* dragged down by `1_18_0` in absolute terms (0.389 vs. its
otherwise-typical ~0.43), but so was DBConformer's and SlimSeiz's, by
similar amounts — it's GRU that's the outlier here, propped up by a fold
none of the raw-EEG models can get any recall on at all. AP-wise (the
metric treated as primary in this repo) Mamba was barely affected by
this fold either way (0.618, not a low score) — the recall collapse is
specific to the fixed 0.5 threshold, not to Mamba's underlying ranking
ability on that fold's windows.

Worth investigating directly if any of these get pursued further: what's
different about `1_18_0`'s preictal windows, in whichever feature space
each model sees, that only GRU's dense-edge CWT+graph representation
handles.

---

## Reading this

**Mamba wins on AP** (0.499, next-best DBConformer at 0.442, +0.057) and
ties for best hit-rate reliability (5/6 smoothed, same as GRU and
DBConformer) at the best FAR/h-raw among the two dense-edge models. It's
the strongest single number here on the metric this repo has consistently
treated as primary. Costs ~4.6x GRU's wall-clock per epoch on this
hardware (`mambapy` pure-PyTorch scan, no fused kernel) — a real cost,
just not one that changes which model performs best, per your "forget
training time" framing.

**GRU has the best recall/f1** (0.807 / 0.436) but the worst AP and
highest FAR/h of the four — it over-predicts positive more than the
others (same pattern flagged for DBConformer in its own note, more
pronounced here). Best raw hit-rate parity with Mamba/DBConformer (6/6),
so it's not missing seizures outright, just noisier around them. **This
f1 lead is mostly one fold, though** — see "Per-fold precision / recall /
f1" below: drop `1_18_0` and GRU's f1 (0.436) drops to worst of the four
(0.372) while Mamba's (0.389) becomes best (0.433). GRU's overall f1 edge
isn't a consistent five-of-six-fold pattern.

**DBConformer sits in the middle on every metric** — never best, never
worst, consistent with it being an unmodified off-the-shelf architecture
adapted to this protocol rather than one purpose-built or tuned for it
(see its own note's two negative diagnostics).

**SlimSeiz is a different operating point, not uniformly worse.** Best
FAR/h-smoothed (6.31, i.e. fewest false alarms per hour) and best raw
accuracy (0.913), but the lowest recall (0.556) and the only pipeline to
miss 2/6 seizures under smoothing (`1_18_0` and `1_15_0`) instead of 1/6.
It's the most conservative of the four, trading event-level reliability
for a quieter alarm stream. Its AP (0.431) is inflated somewhat by a
single outlier fold: `1_16_0` scores a perfect AP=1.000 (f1=0.979, its
easiest fold across all four pipelines by a wide margin) while `1_15_0`
(AP 0.330, recall 0.100) and `1_18_0` (AP 0.425, recall 0.000) are among
its weakest — the mean is doing some work to paper over a wide per-fold
spread.

**`1_18_0` is the hardest fold overall for recall/f1/hit-rate** — see
"Per-fold precision / recall / f1" above for the numbers: every raw-EEG
pipeline (Mamba, DBConformer, SlimSeiz) collapses recall there
(0.100 / 0.067 / 0.000) while GRU aces it (0.767, its own best fold).
On AP specifically it's less clean — DBConformer's AP there (0.362) is
also its worst fold, but Mamba's (0.618) isn't bad at all, so this is a
decision-threshold effect more than a ranking-ability one. Worth a closer
look at what's different about that fold's features if any of these
models get pursued further — flagged in three separate notes now without
being investigated.

**`1_26_0` is the second-hardest fold across the board** — every
pipeline's second- or third-lowest AP, and GRU's persistent k-of-n miss
specifically (the only smoothed miss GRU has here). Not a shared miss
across pipelines the way `1_18_0` is (Mamba, DBConformer, and SlimSeiz
all still count it as a raw+smoothed hit) — just a hard fold for
everyone in absolute AP terms.

## Caveats

- Four different training runs, four different random seeds/environments
  (two machines) — not a controlled statistical comparison, single-run
  point estimates per pipeline. Nothing here has error bars.
- "Best number per pipeline" means each pipeline's own best-known,
  already-disclosed config (see table above) — not a symmetric
  hyperparameter search applied evenly across all four. GRU/Mamba weren't
  put through diagnostics analogous to DBConformer's depth/class-weight
  sweep; it's possible either has unexplored headroom the same way
  DBConformer's depth=5 vs 3/6 sweep found a local optimum. Don't read
  this table as "the ceiling for each architecture," only as "the best
  each has produced so far under one shared protocol."
- SlimSeiz's headline number here (adaptive selection) is the one with
  complete data, not necessarily the one that's actually best — see the
  fixed-montage tie noted above.
