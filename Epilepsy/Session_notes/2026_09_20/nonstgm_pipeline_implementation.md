# NonStGM: nonstationary covariance/precision graph pipeline

Branch: `claude/nonstgm-eeg-pipeline-o5cdcl`. Sandboxed cloud session, no
CHB-MIT dataset (`mne`/`mne_data` both absent) and no AWS credentials
available -- scope agreed with the user up front: implement the real
pipeline + docs, validate with synthetic-data unit tests, document the AWS
run as a recipe rather than executing it.

## What this is

A practical EEG adaptation (explicitly **not** a reproduction) of:

> Basu, S. and Subba Rao, S. "Graphical Models for Nonstationary Time
> Series." Annals of Statistics, 2023, 51(4), 1453-1483. arXiv:2109.08709.

Instead of the existing WCT/coherence graph (pairwise frequency-specific
coherence), this builds a graph from time-local regularized
covariance -> precision (inverse covariance) -> conditional-dependence
edges, and feeds that edge sequence to a GRU or this repo's existing
Mamba temporal backend. Full math/leakage/ablation writeup: README's new
"NonStGM" section. Exact estimator/regularization: `Epilepsy/pipelines/
nonstgm_graph.py`'s module docstring.

## Files added

- `Epilepsy/pipelines/nonstgm_graph.py` -- `LocalCovariancePrecisionGraph`
  (nn.Module): raw window `(B,C,T)` -> local segments -> ridge-regularized
  covariance `Σ_t+λI` -> Cholesky-based precision `Θ_t` -> edge features
  (precision magnitude / partial correlation / raw covariance / both),
  strict upper-triangle only (`E=C(C-1)/2`). Numerical diagnostics
  (min/max eigenvalue, condition number, Cholesky-failure count, NaN/Inf
  count, fallback-regularization fraction, effective λ) always computed,
  never silently dropped. `frequency_band` param is a documented
  extensibility stub (raises `NotImplementedError` if set) -- not
  implemented this pass, per spec's "time-local version first" scoping.
- `Epilepsy/pipelines/nonstgm_classifier.py` -- `NonStGMCore` (graph
  module -> reused `_DenseEdgeGRUTemporal`/`_DenseEdgeMambaTemporal` from
  `cwt_gnn_classifiers.py`, UNCHANGED -> edge-mean-pool -> MLP head) +
  `NonStGMClassifier(TorchEEGClassifier)` sklearn wrapper, same
  train-idx-only normalization pattern as `dbconformer_classifier.py`/
  `godoy_tmc_classifier.py`.
- `tests/test_nonstgm.py` -- 19 synthetic-data pytest tests (no dataset
  needed): covariance/precision symmetry, PD after regularization, exact
  edge count, partial-correlation range, no-NaN on well-conditioned input,
  loud diagnostics on a deliberately singular input (duplicated channel),
  adaptive-regularization effective-λ recording, static vs. time-varying
  segment counts, `frequency_band` NotImplementedError, unregularized-λ
  rejection, train-fold-only normalization (leakage check), and both GRU/
  Mamba backends' fit/predict_proba shape contracts. All 19 pass (verified
  in a scratch venv with CPU torch 2.14.0+cu130, mambapy 1.2.0, and the
  rest of `requirements.txt` minus the pinned `+cu128` torch line/
  `--extra-index-url`, which this sandbox's proxy 403s on -- plain PyPI's
  default CUDA build was used instead, fine for CPU-only unit tests).
- `configs/nonstgm.yaml` -- example config demonstrating the new
  `--config` flag.

## Files modified

- `Epilepsy/run_pipelines.py`:
  - `"nonstgm_gru"`/`"nonstgm_mamba"` added to `--pipeline` choices + a
    help-text block matching the file's existing per-pipeline style.
  - `NONSTGM_GRU_PARAMS`/`PREDICTION_NONSTGM_GRU_PARAMS` +
    `NONSTGM_MAMBA_PARAMS`/`PREDICTION_NONSTGM_MAMBA_PARAMS` (Mamba dict =
    exact copy of GRU dict + `temporal_backend="mamba"`, same isolated-
    ablation reasoning `DENSE_EDGE_MAMBA_PARAMS` already documents).
  - Extended `_raw_classifier_family_params`, the `classifier_cls` map in
    `main()` (both pipeline names -> `NonStGMClassifier`, backend picked
    via `clf_params["temporal_backend"]`), and `_apply_raw_classifier_cli_
    overrides` (new `--nonstgm-*` flags). Routed through the existing
    `leave_one_seizure_out_raw_classifier_prediction` -- no new fold logic.
  - New CLI flags: `--nonstgm-representation`, `--nonstgm-temporal-
    segments`, `--nonstgm-overlap`, `--nonstgm-regularization`,
    `--nonstgm-static`, `--nonstgm-edge-threshold`,
    `--nonstgm-adaptive-regularization`.
  - New **generic, pipeline-agnostic** `--config PATH` flag
    (`apply_config_file`) -- no such mechanism existed anywhere in the
    repo before (confirmed by grep). YAML keys matching an argparse dest
    apply as defaults; an explicit CLI flag always wins. Not specific to
    nonstgm -- any future pipeline can reuse it.
  - `_write_nonstgm_run_metadata` -- writes `config_<run_id>.yaml`/
    `git_commit_<run_id>.txt`/`environment_<run_id>.txt` into the nonstgm
    result dir (spec section 16), scoped to just this pipeline rather than
    threaded through every pipeline's own results writing.
  - **Known gap vs. spec**: `leave_one_seizure_out_raw_classifier_
    prediction` (shared by dbconformer/slimseiz/cg_mambanet/godoy_tmc/
    nonstgm_*) does not expose per-fold classifier instances or per-window
    predictions to its caller, so `NonStGMClassifier.diagnostics_`
    (per-fold numerical-stability numbers) and a `--dump-window-scores`-
    style predictions CSV are NOT currently written into the nonstgm
    results directory -- only available if `NonStGMClassifier` is driven
    directly (as the unit tests do). Extending that shared function to
    surface these was judged out of scope for this pass (would touch every
    pipeline sharing it); flagged here and in the README as a follow-up.
- `README.md`: new "NonStGM" subsection (all 10 documentation points from
  the spec, paper citation, explicit "what is/isn't faithful" framing) +
  a table row under "Original architectures built here".
- `CONTEXT.md`: new pointer entry (see there for the exact next-session
  checklist -- real CHB-MIT LOSO run + AWS comparison are NOT done yet).

## What was and wasn't verified this session

Verified (scratch venv, `pip install -r requirements.txt` minus the
`+cu128`/`--extra-index-url` line which this sandbox's proxy blocks):
- `tests/test_nonstgm.py`: 19/19 pass.
- Full existing repo `pytest` suite: 90 passed, 3 slow-deselected, 2
  pre-existing failures (`test_dense_edge_mamba_is_a_pipeline_choice`,
  `test_tf_encoder_is_a_flag_not_a_pipeline` -- both hardcode a stale
  exact `--pipeline` choices set that was already missing `godoy_tmc`/
  `hermitian_ssm`/`continuous_cwt_mamba`/`temporal_graph_*` BEFORE this
  branch touched anything, confirmed by `git stash` + re-running them on
  the unmodified tree). Not a regression from this work.
- `_build_argument_parser()` builds cleanly with both new pipeline names;
  `--config configs/nonstgm.yaml` merges correctly and an explicit CLI
  flag correctly overrides a config-file value (`--device cpu` over the
  config's `device: mps`).

NOT verified (no dataset/AWS access in this sandbox -- next session's
job, see CONTEXT.md):
- A real CHB-MIT LOSO run (even 1-2 folds).
- Prediction timestamp/label alignment against the existing benchmark's
  protocol (spec validation-experiment items 9-10).
- Any AWS run / cost / GPU-memory / runtime-per-epoch numbers -- the
  README's AWS section is a documented recipe, not a completed run.
- The ablation sweep itself (representation x static/time-varying x
  edge-threshold) -- switches are wired and unit-tested individually, but
  no comparative benchmark numbers exist yet.
