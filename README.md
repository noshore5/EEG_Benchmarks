# EEG_Benchmarks

A research bench for **EEG classification under leakage-resistant
evaluation**. Two domains share one codebase, one shared training loop,
and one results layout:

1. **BCI / motor imagery** — pipelines evaluated on
   [MOABB](https://moabb.neurotechx.com) benchmarks (`BCI/`).
2. **Epilepsy** — CHB-MIT scalp EEG, seizure **prediction** (preictal vs
   interictal) and seizure **detection** (ictal vs non-ictal), under a
   strict leave-one-seizure-out protocol (`Epilepsy/`).

The point of the repo is the *evaluation*, not any one model. Published
seizure-prediction/detection numbers are hard to compare because almost
every paper uses its own split, horizon, and preprocessing — and random
splits over overlapping windows leak. Here, faithful reimplementations of
published architectures and several original ones are run head-to-head
under a single protocol with the leakage paths closed.

<p align="center">
  <img src="docs/assets/coherence-graph.png" width="430"
       alt="Wavelet-coherence graph over the scalp montage for a chb01 preictal window">
  <img src="docs/assets/coherence-edge.png" width="430"
       alt="Coherence and phase across time and frequency for one edge">
</p>

Several pipelines here consume a **wavelet-coherence graph**: nodes are the
bipolar EEG channels, edges carry the time–frequency coherence and phase
between each channel pair. Left: that graph for one real chb01 preictal
window (strongest 45 of 253 edges). Right: what a single edge carries —
coherence magnitude and phase over the 4 s window, 8–40 Hz. Both are
computed by the real pipeline; regenerate with the scripts in
`docs/assets/`, full export (all 253 edges, raw arrays) in
`exports/coherence-graph/`.

## What's in it

### Epilepsy pipelines (`Epilepsy/run_pipelines.py`)

Reimplementations, each built from the paper's spec (not vendored code):

| pipeline | source | input |
|---|---|---|
| `slimseiz` | SlimSeiz (adaptive channel selection) | raw EEG |
| `dbconformer` | DBConformer | raw EEG |
| `cg_mambanet` | CG-MambaNet | raw EEG |
| `godoy_tmc` | Godoy et al. TMC-T (conv-tokenizer + Transformer) | raw EEG |
| `truong_stft_cnn` | Truong et al. 2018 (STFT + CNN) | STFT |

Original architectures built here:

| pipeline | idea | input |
|---|---|---|
| `temporal_graph_mamba` | wavelet coherence graph → per-channel selective SSM (Mamba) → logits | CWT coherence graph |
| `dense_edge` / `dense_edge_gru` / `dense_edge_mamba` | dense edge-feature graph with a Conv2d / per-edge GRU / Mamba temporal step | CWT coherence graph |
| `hermitian_ssm` | eigendecomposition of the Hermitian coherence matrix → selective SSM | CWT coherence graph |
| `continuous_cwt_mamba` | streaming (carried-state) variant for continuous recordings | CWT coherence graph |
| `nonstgm_gru` / `nonstgm_mamba` | time-local regularized covariance → precision → conditional-dependence graph, GRU/Mamba temporal step | covariance/precision graph |

Every pipeline respects `--label-mode {prediction,detection}` and writes
to a per-pipeline, per-mode results directory. Detection and prediction
numbers are **never pooled** — different tasks, different ceilings (see
`Epilepsy/results/README.md`).

### Evaluation protocol

- **Leave-one-seizure-out** cross-validation: hold out one seizure's
  recording, train on the rest, repeat.
- **Prediction** uses a configurable seizure-prediction horizon (SPH) and
  occurrence period (SOP), a postictal buffer excluded from interictal,
  and negative-window subsampling with per-recording stratification so no
  fold is starved.
- **Shuffled-label controls** (`*_shuffled_control/`) for every prediction
  pipeline — a sanity floor.
- Prediction runs additionally report event-level metrics (per-seizure
  hit/miss, false alarms per hour) that window-level F1/AP can't stand in
  for.

Current head-to-head board (chb01, prediction LOSO, single untuned run
per pipeline — treat gaps under ~0.05 mean AP as noise):
`Epilepsy/Session_notes/2026_08_31/pipeline_comparison_all_models.md`.

### NonStGM: nonstationary covariance/precision graph (`nonstgm_gru` / `nonstgm_mamba`)

A **practical EEG adaptation, not a reproduction**, of:

> Basu, S. and Subba Rao, S. "Graphical Models for Nonstationary Time
> Series." *Annals of Statistics*, 2023, 51(4), 1453–1483.
> [arXiv:2109.08709](https://arxiv.org/abs/2109.08709).

**1. Why coherence and precision graphs answer different questions.**
The existing WCT/coherence pipelines (`dense_edge*`, `temporal_graph_*`,
`hermitian_ssm`) build a graph from *pairwise* frequency-specific
coherence between channels `i` and `j` — a marginal, bivariate measure
that says nothing about whether that coupling is explained away by a
third channel. NonStGM instead builds a graph from the *inverse*
covariance (precision) matrix, which is a genuinely multivariate object:
its off-diagonal structure answers "are channels `i` and `j` still
coupled once every other channel is accounted for?", not just "are they
correlated?". These are different, complementary questions about the same
recording, which is why this pipeline is designed to be run head-to-head
against the WCT baselines under identical folds/labels/metrics rather than
replacing them.

**2. What covariance represents.** For a window `X(t) ∈ R^C`, the sample
covariance `Σ` over some span of time is a second-moment summary of how
channels co-vary — but it mixes direct and indirect coupling (`i`
correlated with `j` only because both are correlated with `k` shows up in
`Σ_ij` too).

**3. Why inverse covariance represents conditional dependence.** For a
(locally) Gaussian process, a zero in `Σ⁻¹` (the precision matrix `Θ`) is
*exactly* the classical Gaussian-graphical-model statement that channels
`i` and `j` are conditionally independent given every other channel
(Whittaker 1990). Inverting `Σ` therefore strips out the indirect
couplings that a raw covariance/coherence graph cannot distinguish from
direct ones.

**4. Why nonstationarity matters.** A single global covariance/precision
matrix over an entire 30s window (or worse, an entire recording) assumes
the coupling structure is constant over that span — exactly the
assumption Basu & Subba Rao's paper argues is wrong for physiological
signals like EEG, and exactly the assumption that would throw away any
signal specific to the run-up to a seizure. This pipeline never computes
one global covariance over a whole window; see the ablation below for how
that specific claim is tested rather than just asserted.

**5. How the EEG adaptation works (and how it differs from the paper).**
Per window, per **local, overlapping time segment** (`--nonstgm-temporal-
segments`, `--nonstgm-overlap`): estimate the plain sample covariance
`Σ_t`, ridge-regularize it (`Σ_t + λI`, `--nonstgm-regularization`, always
applied — never an unregularized inverse), invert via a Cholesky-based
solve to get `Θ_t`, and turn `Θ_t` into edge features:

```
precision magnitude:   |Θ_t,ij|
partial correlation:   ρ_ij|rest = −Θ_t,ij / sqrt(Θ_t,ii · Θ_t,jj)
```

(sign convention: the leading minus is the standard Gaussian-graphical-
model partial-correlation identity, Whittaker 1990 Prop 5.4.5 — a
positive `Θ_t,ij` maps to a negative raw ratio, so the minus sign restores
"positive ρ = positively conditionally coupled"; ρ ∈ [-1, 1] for any valid
precision matrix). Only the strict upper triangle (`i<j`, `C(C-1)/2`
edges) is emitted — no duplicate `(i,j)`/`(j,i)` pair. **This is
deliberately a simple, computationally cheap sample-covariance estimator,
NOT the paper's own kernel-smoothed nonparametric covariance estimator or
its (possibly frequency-domain) DGLASSO-style sparse precision estimate,
and carries none of the paper's asymptotic consistency theory or formal
edge-significance tests.** It is a pragmatic stand-in chosen to keep this
pipeline's cost comparable to the repo's existing WCT/dense-edge
pipelines, not a claim of statistical equivalence to the paper (see
`Epilepsy/pipelines/nonstgm_graph.py`'s module docstring for the full
math and this same "what is/isn't faithful" framing). A frequency-local
precision estimate (reusing this repo's CWT infrastructure, closer in
spirit to the paper's frequency-domain framework) is a documented
extensibility stub (`frequency_band` param) — **not implemented in this
pass**, per the "time-local version first, keep the interface extensible"
scoping this feature was built under.

The resulting per-segment edge sequence (`[B, n_edge_feat, E, T_segments]`)
is fed to a temporal model over the *exact same* `[B, C_in, E, T]` contract
`dense_edge_gru`/`dense_edge_mamba` already established
(`_DenseEdgeGRUTemporal`/`_DenseEdgeMambaTemporal`, reused unchanged — see
`Epilepsy/pipelines/nonstgm_classifier.py`) — `nonstgm_gru` and
`nonstgm_mamba` differ *only* in which of those two backends reads the
graph, an isolated architecture ablation, not two separately-tuned models.

**6. What information is used at each temporal segment.** Only the raw
voltage samples inside that segment's own time span (`--nonstgm-overlap`
controls how much segments share) — no information from other segments,
other windows, or other recordings leaks into a segment's own `Σ_t`/`Θ_t`.

**7. How leakage is prevented.** The only data-fit step in this pipeline
is the global z-score normalization (`NonStGMClassifier._prepare_features`,
same convention as `dbconformer`/`godoy_tmc`), fit strictly on the current
LOSO fold's *training* windows (`TorchEEGClassifier`'s `train_idx`
threading, `Epilepsy/pipelines/common.py`) — never on the held-out
subject/seizure. The covariance/precision estimator itself has **no
data-driven fit step at all**: `λ` (regularization), segment count, and
overlap are fixed hyperparameters set before training, not tuned against
validation/test data, so there is no additional leakage surface beyond
the existing normalization. The LOSO fold construction itself (one held-
out seizure's `seizure_id` excluded from training, per
`leave_one_seizure_out_raw_classifier_prediction`) is unchanged from
every other raw-EEG pipeline in this repo.

**8. Numerical stability.** Every forward pass computes (not opt-in):
min/max eigenvalue of each segment's raw `Σ_t`, worst-case condition
number, Cholesky-failure count, NaN/Inf count, and (when
`--nonstgm-adaptive-regularization` is set) the effective `λ` used per
segment after any 10×/100× retries. Surfaced as
`NonStGMClassifier.diagnostics_` after a `predict_proba` call. A failed
regularized Cholesky never silently produces NaN/Inf — it falls back to a
diagonal-only precision (`I/λ`) and counts the fallback, or (with
adaptive regularization) retries at a larger `λ`; either way the failure
is counted, never hidden.

**9. How to reproduce local experiments.**

```bash
# quick synthetic-data unit tests (no dataset needed)
pip install -r requirements.txt
pytest tests/test_nonstgm.py -v

# a real chb01 LOSO run (downloads CHB-MIT on first use)
python Epilepsy/run_pipelines.py --pipeline nonstgm_gru --label-mode prediction --subjects 1 --device cpu
python Epilepsy/run_pipelines.py --pipeline nonstgm_mamba --label-mode prediction --subjects 1 --device cpu

# via config file (spec-requested interface; any --nonstgm-* flag can
# also just be passed directly on the command line)
python Epilepsy/run_pipelines.py --config configs/nonstgm.yaml
```

Ablation switches: `--nonstgm-representation {covariance,precision,
partial_corr,both}`, `--nonstgm-temporal-segments`, `--nonstgm-overlap`,
`--nonstgm-regularization`, `--nonstgm-static` (collapses to one
whole-window segment — the static-vs-time-varying ablation),
`--nonstgm-edge-threshold`, `--nonstgm-adaptive-regularization`. Each is
wired as an isolated switch (spec: "do not silently combine all of these
into one feature representation") so a sweep can attribute any benefit to
covariance information, conditional dependence, or explicitly modeling
nonstationarity separately.

**10. How to launch AWS experiments, and expected compute/memory
(design-only — not executed in this session; no AWS credentials or CHB-MIT
dataset were available in the sandbox this pipeline was built in).**
Recommended initial target: `g4dn.xlarge` (1× NVIDIA T4, 16GB GPU memory,
4 vCPU, 16GB RAM) — this repo's existing `scripts/eeg-run-spot.sh` /
`Dockerfile.mamba` GPU-pod path applies unchanged (`nonstgm_mamba` needs
the same `mambapy`/CUDA-kernel setup `dense_edge_mamba` already documents
in `AWS_INFRA.md`). Mixed precision: bf16 for the model forward/backward
(`--train-amp-bf16`, already a repo-wide flag), but the covariance
accumulation and the regularization/inversion step should stay fp32 (this
is what `nonstgm_graph.py`'s `_local_covariance`/`_regularized_precision`
already force internally, regardless of the surrounding autocast context)
— ill-conditioned EEG channel covariances are exactly the case reduced
precision breaks first. S3 layout: `preprocessed/`, `cache/`,
`checkpoints/`, `results/`, `logs/`, `configs/` under the existing bucket
(`AWS_INFRA.md`), with a cache key following `hermitian_ssm_cache.py`'s
own pattern (`HermitianSpectralConfig.cache_key()`: a sorted-JSON hash of
a frozen, versioned config dataclass) — for NonStGM that config would
include `dataset, subject, window_length, sampling_rate, representation,
temporal_segments, overlap, regularization, static, version`. **None of
this was run** in the session this pipeline was built in (no AWS access,
no cached CHB-MIT data) — treat it as a reproducible recipe to launch from
the AWS-capable side of this repo's split-shell workflow (see
`CONTEXT.md`), not as a reported result.

**Validation performed so far / explicit limitations:** `tests/
test_nonstgm.py` (synthetic data, no dataset needed) checks covariance/
precision symmetry, positive-definiteness after regularization, exact
`C(C-1)/2` edge count, partial-correlation range, no NaN/Inf on
well-conditioned input, loud (never-silent) diagnostics on a deliberately
singular input, train-fold-only normalization (no leakage), and both
temporal backends' shape contracts. **A real CHB-MIT LOSO run and the
AWS comparison have not been executed** — see `CONTEXT.md` for what the
next session needs to do before any benchmark number from this pipeline
should be trusted.

### SzCORE submission (`Epilepsy/szcore/`)

Packages `godoy_tmc` as a container for
[SzCORE / EpilepsyBench](https://epilepsybenchmarks.com), the community
seizure-detection benchmark: EDF in → seizure-annotation TSV out, scored
on a private held-out set. See `Epilepsy/szcore/README.md`.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Epilepsy: seizure prediction on chb01 (auto-downloads the data)
python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba --label-mode prediction --device mps

# wiring-only smoke (fast, not a real result)
python Epilepsy/run_pipelines.py --pipeline godoy_tmc --label-mode prediction --smoke

# BCI: motor imagery on BNCI2014-001
python BCI/run_pipelines.py --pipeline dense_edge --subjects 1
```

Use `--device cpu` off Apple Silicon / without CUDA. `--param-names` /
`--param-values` override any pipeline parameter for a one-off run without
editing source.

## Repo layout

| path | what |
|---|---|
| `Epilepsy/` | seizure prediction/detection — pipelines, run harness, results, math notes |
| `Epilepsy/szcore/` | SzCORE detection-benchmark container |
| `BCI/` | MOABB motor-imagery pipelines (evidence GNNs, EEGNet) |
| `datasets/epilepsy/` | CHB-MIT loader (PhysioNet + GitHub-Release mirror, seizure-annotation parsing) |
| `paradigms/` | windowing / continuous-labeling paradigms |
| `utils/` | CWT backends, shared helpers |
| `scripts/` | AWS launch + result-promotion tooling |
| `CONTEXT.md` | living current-state doc — read before working in the repo |
| `Epilepsy/Session_notes/` | dated detailed record of every experiment |

## Data & infrastructure

- **CHB-MIT** subjects `chb01`–`chb04` are mirrored on this repo's GitHub
  Releases (faster than PhysioNet's throttled S3); others download on
  demand. `datasets/epilepsy/chb_mit.py` handles fetch + cache into
  `~/mne_data/`. Redistribution notice + citations: `THIRD_PARTY_NOTICES.md`.
- **RunPod / AWS**: `Dockerfile` (+ `Dockerfile.mamba` for the fused
  `mamba-ssm` CUDA kernel) build GPU pod images via GitHub Actions;
  `scripts/` + `.github/workflows/eeg-run.yml` launch fire-and-forget
  training runs. See `docs/infrastructure.md` and `AWS_INFRA.md`.

## Status / caveats

- Most head-to-head numbers are **single untuned runs on one subject
  (chb01)** — directional, not definitive. No error bars unless a session
  note gives a seed sweep.
- Cross-patient generalization is not solved here (or in the literature);
  the honest LOSO numbers are far below the subject-specific figures
  papers report.
- No `LICENSE` file yet — add one before sharing the repo publicly.
  CHB-MIT data is ODC-By 1.0 (`THIRD_PARTY_NOTICES.md`).
