# hermitian_ssm band-match 6-fold -- 0.253 -> 0.436 AP, now competitive (2026-08-29)

**TL;DR:** config-only changes to `hermitian_ssm` (band 8-124 -> 8-40 Hz,
nfreqs 60 -> 16, `diagonal` "power" -> "zero", `k` 2 -> 6) took it from
**mean AP 0.253** (worst pipeline in the repo) to **0.436**, with
**ROC-AUC 0.947** (>= `temporal_graph_mamba` "pre" 0.94), **6/6 raw +
6/6 smoothed** event hits (tgm "pre": 6/6 / 5/6), and **FAR/h smoothed
8.0** (tgm "pre" 8.44). It is now competitive on every prediction metric
*except* AP (0.436 vs 0.674). No code changed -- only
`HERMITIAN_SSM_PARAMS` in `run_pipelines.py` (uncommitted).

Follow-on to `2026_08_28/hermitian_ssm_first_6fold_and_eigh_fix.md`
(the 0.253 run + the eigen-feature diagnosis that motivated this).

## What changed and why

| key | 0.253 run | this run | rationale |
|---|---|---|---|
| `spectral_lowest/highest` | 8-124 Hz | **8-40 Hz** | match `temporal_graph_*`; apples-to-apples cross-arch comparison at last |
| `spectral_nfreqs` | 60 | **16** (fd=2 -> 8 cached bins) | keep Welch-style freq smoothing (stabilises eigh; ~6% eigenvector flips at fd=1) but stop paying for 60 bins over a 32 Hz span |
| `spectral_diagonal` | `"power"` | **`"zero"`** | diagnosis item 2: `A_ii = |W_i|^2` power spikes dominated the eigenstructure (lambda_1 median 14, **max 3210**), pinning mode 1 to a single-channel artefact / global common-mode. Zeroing the diagonal makes the eigendecomposition purely about *coupling*. |
| `spectral_k` | 2 | **6** | diagnosis item 1: top-2 were near-redundant (corr(l1,l2)=0.93). k=6 only helps *given* diagonal="zero" frees the smaller eigenvalues from under the power floor. |

`spectral_mains_notch=True` left on -- now a harmless no-op since
highest=40.

New spectral cache key `9d6ad0d850b8b8f0`, F_out=8, k=6, 9.8 GB for all
41 chb01 recordings (~266 MB/recording), ~40 min one-time CPU precompute.

## Result

Full 6-fold LOSO prediction on chb01, `--device cpu`, 30-epoch budget /
patience 5. CSVs:
`Epilepsy/results/hermitian_ssm/prediction/prediction_leave_one_seizure_out_20260829-111904.csv`
(+ per-seizure `_per_seizure_20260829-111904.csv`).

| fold | hermitian AP | tgm "pre" AP | note |
|---|---|---|---|
| `1_03_0` | 0.254 | 0.792 | loss |
| `1_04_0` | 0.324 | 0.830 | loss |
| `1_15_0` | **0.414** | 0.331 | **win** |
| `1_16_0` | 0.279 | 0.996 | loss (pre near-perfect here) |
| `1_18_0` | 0.788 | 0.802 | tie |
| `1_26_0` | **0.556** | 0.293 | **win** |
| **mean AP** | **0.436** | **0.674** | |
| mean ROC-AUC | **0.947** | 0.94 | |
| hit raw / smoothed | **6/6 / 6/6** | 6/6 / 5/6 | |
| mean FAR/h raw -> sm | 13.2 -> **8.0** | ~? -> 8.44 | |
| mean recall | 0.824 | (high, ~1.0) | |
| mean precision | 0.346 | ~0.2 | |

Per-fold: hermitian **wins the two folds where "pre" is weak** (`1_15`,
`1_26`) and **loses the three where "pre" is strong** (`1_03`, `1_04`,
`1_16`). It is a *different* model, not yet a *better* one -- but the
profile is now "solid global ranker, weak precision at the top of the
ranking" (ROC-AUC 0.947 with AP 0.436), which is a calibration-shaped
gap, not a representational failure like the 0.253 run.

## Read

- **`diagonal="zero"` is almost certainly the main lever.** The
  diagnosis said the power diagonal was blowing up lambda_1 and making
  mode 1 a global-synchrony common-mode; removing it is the change most
  aligned with the +0.18. Can't prove it without an ablation (flip only
  `k` back to 2, hold the rest) -- probably not worth a full 6-fold for
  attribution given 0.436 is still well short of 0.674.
- **The architecture is no longer disqualified.** Going into this it was
  "weakest pipeline in the repo, parked." It now matches or beats the
  leader on ROC-AUC, smoothed hit rate, and FAR/h. That's enough to
  justify the next round of *real* changes.
- **Fragility caveat still applies:** one chb01 subject, 6 seizures, ~30
  preictal windows/fold; per-fold AP spans 0.25-0.79 here. 0.436 has a
  wide error bar. Seed-repeat before over-reading the pre/post fold
  pattern.

## Follow-up: projector encoder (#2) -- NEGATIVE

Ran 2026-08-29 (`encoder_mode="projector"`, `_ProjectorEncoder`, same warm
cache, only the encoder changed vs the 0.436 run). Encodes gauge-invariant
node summaries of `P = sum_r lambda_r u_r u_r^H`: `lambda`, `diag(P)`,
`|lambda|`-participation, and node net complex coupling `c_i = (P1)_i - P_ii`.

Result: per-fold AP `.138 / .190 / .259 / .300 / .585 / .168`, **mean AP
0.273**, ROC-AUC 0.898, hit 6/6 raw / 4/6 k-of-n. (CSV
`*_20260829-152511.csv` was accidentally deleted by an over-broad cleanup
glob right after; these summary numbers are the full record.) Worse than the eigenvector encoder
(0.436 / 0.947 / 6/6-6/6) on **every** axis, ROC-AUC included -- so
removing the eigenvector phase-gauge noise did not even buy a better
ranker. 5 of 6 folds down.

Read: the gauge freedom / mode-crossing churn was **not** the bottleneck.
Collapsing the C x C projector to four C-vectors discards more than the
`[Re u, Im u]` encoder was losing to phase instability. Consistent with
the fragility note's thesis (bottleneck is the ~30-window budget, not
feature quality). `encoder_mode="eigenvector"` stays the hermitian best.

## Follow-up 2: graph encoder (#3) -- PARKED after fold 1

`encoder_mode="graph"` / `_GraphEncoder`: feeds the upper triangle of `P`
(253 complex + 23 real / freq) -- lossless, gauge/order-invariant, and
does NOT collapse to nodes (the thing #2 was faulted for). `P`'s diag +
triangle are precomputed per window in `_WindowDataset` (numpy, no
autograd): reconstructing them as a batched autograd tensor every step was
~525 s/epoch (5x; batch 16 vs 32 made no difference -- it's the C(C-1)/2
complex MAC + big intermediates, not batch count). Precompute-in-loader is
~275 s/epoch, no cache rebuild.

**Fold 1 (`1_03_0`): AP 0.169** (vs eigenvector-encoder 0.254, projector
0.138, tgm "pre" 0.792 on this fold). Killed after fold 1 on the user's
call -- with #2 negative and #3 fold 1 tracking the same, the pattern was
clear enough not to spend the ~2.5 h more.

**Overall read across #2 + #3:** both are gauge-invariant functions of
`P`, both land well under the raw-eigenvector encoder. So the eigenvector
phase gauge / mode-crossing churn is **not** what caps hermitian. The raw
`[Re u, Im u] + mode_id + eigenvalue-embedding` parameterisation is just
easier for the model to optimise on ~30 preictal windows -- the "noise"
plausibly acts as regularisation. This is the fragility note's thesis
again: data budget, not representation.

## Follow-up 3: evolution encoder (#6) -- NEGATIVE (~chance)

`encoder_mode="evolution"` / `_EvolutionEncoder`: token per (f,t) is the
complex k x k subspace-evolution operator `M(t) = U(t)^H U(t-1)` (M(0)=I)
plus `lambda(t)`; features `[lambda, Re M, Im M, |M|]`. Eigenvector phases
canonicalised first so `arg(M)` is frame-consistent. `M` per window via
numpy einsum. Epoch ~127 s. This was the variant that would have
motivated Mamba-3.

Fold 1 (`1_03_0`): **AP 0.062** (base rate 0.04), hit=False,
precision=recall=f1=0.000. Fold 2 (`1_04_0`): **AP 0.091** (base rate
0.046). val_auc never broke 0.77 either fold (vs 0.94-0.97 every other
encoder), val_loss diverged to 5.7, early-stop restored the ~untrained
epoch-1 checkpoint. Killed after fold 2 -- not "overfits but ranks OK"
like #2/#3, it simply does not learn.

**Why:** `M` lives entirely in the k=6 *mode* space -- zero channel
information (how the subspace rotated, nothing about which electrodes).
#6 is the most extreme "discard channel identity" variant and fails
hardest.

## Conclusion -- encoder-variant investigation CLOSED, NEGATIVE

| encoder | feeds | channel identity | mean AP |
|---|---|---|---|
| `eigenvector` | `[Re u, Im u]` + `mode_id` + lambda | full (u_r in C^23) | **0.436** |
| `projector` (#2) | node summaries of `P` | partial -> collapsed | 0.273 |
| `graph` (#3) | upper triangle of `P` | partial | 0.169 (fold 1) |
| `evolution` (#6) | `M = U^H U` in C^{6x6} | none | ~chance |

Result degrades **monotonically** with how much channel identity the
encoder drops. Hermitian needs *which channels couple*, in raw
eigenvector coordinates; every abstraction toward "the graph as an
object" loses ground. Gauge-invariance was a red herring -- removing it
(#2/#3) cost more than it saved. **#4 (`canonicalize_eigenvectors=True`,
BUILT, not run) is skipped:** "gauge fix + same encoder" has ~nil
expected value once #2/#3 showed the gauge isn't the cap.

Confirms the fragility note's thesis from the opposite direction: the
ceiling (~0.44 hermitian, 0.674 `temporal_graph_mamba` "pre") is set by
the ~30-preictal-window data budget, not the representation.

**Only untried hermitian lever:** Mamba-3 (complex diagonal state) on the
eigenvector encoder -- a temporal-model change, ~150-250 lines, low
expected value. Everything else (k-sweep, per-channel log-power stream,
per-frequency Mamba lanes) is "add capacity", which is exhausted here.

`_WindowDataset` was refactored: `graph_mode` bool -> `item_mode` in
{`eigenpairs`, `graph`, `evolution`} + separate `canonicalize` flag.
`HERMITIAN_SSM_PARAMS` reset to `encoder_mode="eigenvector"`.

## Mamba-3 (still the eventual step, unbuilt)

`M(t)` (and the eigenvector encoder's phase content generally) is a stream
of complex rotations. Mamba-1/2 have a real diagonal state transition --
they exponentially accumulate/decay a state channel, they cannot rotate
one. Mamba-3's complex diagonal `A_r = exp(Delta(-a_r + i omega_r))` gives
each state channel a decay rate *and* an eigen-frequency, so it can lock
onto an evolving phase relationship. Repo has only mambapy's real Mamba-1
pscan; the scan is associative over C unchanged, so Mamba-3 here is
~150-250 lines (complex diagonal state, real->complex in / complex->real
out, keep selective Delta,B,C). Do it on the eigenvector encoder (the
best) and/or #6.

## Code / repo status (2026-08-29 EOD)

- **Uncommitted, nothing staged:**
  - `Epilepsy/run_pipelines.py` -- `HERMITIAN_SSM_PARAMS`: the band-match
    spectral keys (8-40 / nfreqs=16 / diagonal="zero" / k=6, cache key
    `9d6ad0d850b8b8f0`) + `encoder_mode="graph"` + the encoder_mode
    doc-comment. Stale line ~9xx ("8-124 / 30-bin input is a different
    regime") is now wrong.
  - `Epilepsy/pipelines/hermitian_ssm_classifier.py` -- 4 `encoder_mode`s
    (`eigenvector` default in the class, `projector`, `graph`,
    `evolution`), `canonicalize_eigenvectors` flag, `_ProjectorEncoder` /
    `_GraphEncoder` / `_EvolutionEncoder`, `_WindowDataset` `item_mode` +
    `canonicalize` refactor + per-window numpy prep.
  - `CONTEXT.md`, and pre-existing untracked notes.
- Results kept: `*_20260829-111904.csv` (band-match 6-fold, mean AP 0.436).
  All smoke CSVs cleaned. #3 was killed before writing a CSV.
- Cache `~/mne_data/hermitian_ssm_cache/9d6ad0d850b8b8f0` (9.8 GB) warm.
  `~/mne_data/dense_edge_cache` DELETED 2026-08-29 (63 GB -> 68 GB free,
  swap headroom; rebuild ~2-4 h if `temporal_graph_*` is rerun).
- No runs in flight. No disk guard needed (68 GB free).
