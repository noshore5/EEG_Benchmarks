# hermitian_ssm pipeline built (2026-08-27)

Built `--pipeline hermitian_ssm` end to end from the "Graph Spectral
Mamba-3" design doc (`Epilepsy/hermitian_ssm.md`). This is the
whole-graph-as-one-token branch of that design (Design B + Design C of
`graph_state_space_mamba_design.md`, promoted to a standalone spec and
generalised), NOT the STG-Mamba topology-in-the-scan branch (Design A).

## Decisions locked with the user before building

- **Temporal model**: reuse the existing `_DenseEdgeMambaTemporal` (Mamba-2
  / mambapy) to get end to end first. Mamba-3 is the documented next step
  -- `hermitian_ssm_classifier.py` carries a big `MAMBA-3 INTEGRATION
  POINT` comment block naming the two seams to change (`_SpectralEncoder`
  stops flattening the complex eigenvector to reals; `_MambaTemporalHead`
  swaps in a rotation-factored state transition).
- **Band**: 8-124 Hz, `nfreqs=60` (log grid, just below the 125-128 Hz
  Nyquist edge -- CONTEXT.md's band open-thread + the IOPscience
  covariance-eigenvalue precedent).
- **Mains notch**: 57-63 / 117-123 Hz zero-phase IIR notch in the
  precompute (band now straddles 60 Hz US mains + 120 Hz harmonic; the
  CWT/coherence path has none today, only `truong_stft_cnn` does).
- **Frequency decimation** (user's idea): run CWT + coherence smoothing at
  the full 60 bins, then mean-pool the *smoothed* cross-/auto-spectra over
  adjacent log-freq bin pairs BEFORE forming coherence (Welch-style),
  `freq_downsample=2` -> 30 cached freq bins. Halves the cache and the
  eigh cost, discards nothing before it is used.
- **Cache**: per-recording, on the continuous downsampled-time grid, keyed
  by `HermitianSpectralConfig.cache_key()`. The eigendecomposition depends
  only on `(frequency, downsampled timestep)` -- both recording
  properties, not windowing -- so any `window_length` / `step_size`
  re-slices one cache. Stored as one `.npy` per array per recording
  (`<key>/<subj>_<run>/eigenvectors.npy` etc.) so training can `mmap` the
  eigenvectors (a full 6-fold train set is >> RAM; only small window
  slices are ever touched).
- **Precompute runs inside the classifier** (`_prepare`-time), not a
  separate pass in `main()`.
- **Dataset**: recording-preserving build (`_build_continuous_dataset`,
  same as `continuous_cwt_mamba`) so the classifier can slice the
  per-recording cache. NOT because state is carried across windows -- it
  is not; windows are independent here.
- **Mamba hyperparams**: `DENSE_EDGE_MAMBA_PARAMS`' values
  (d_state/d_conv/expand/n_layers), except `d_model=256` (the doc's
  whole-graph token width, vs. 16 for dense_edge_mamba's per-edge seq).

## Files

- `Epilepsy/pipelines/hermitian_ssm_cache.py` -- deterministic precompute
  (`compute_recording_spectral`) + `HermitianSpectralConfig` +
  `HermitianSpectralCache` (per-recording `.npy` dir, mmap-able) +
  `estimate_cache_bytes_per_recording`. Own partial-band Morlet CWT
  (`_MorletCWT`) that transforms `freq_chunk` frequencies at a time so
  peak memory is bounded regardless of recording length; reuses
  `utils.torch_cwt`'s Morlet constants / boundary-pad rule.
- `Epilepsy/pipelines/hermitian_ssm_classifier.py` -- `_SpectralEncoder`
  (per-mode `[Re u, Im u]` proj + additive `lambda` embedding -> mode
  fuse -> freq fuse -> `[B,T,256]`), `_MambaTemporalHead`
  (`_DenseEdgeMambaTemporal` with E=1), `HermitianSSMClassifier`
  (recording-list in, per-recording `[n_windows,2]` proba out; lazy
  mmap-backed `_WindowDataset`). Self-contained -- does NOT subclass
  `SparseEvidenceGNNClassifier`.
- `run_pipelines.py` -- `--pipeline hermitian_ssm` choice + help,
  `HERMITIAN_SSM_PARAMS`, `_hermitian_ssm_clf_params`,
  `leave_one_seizure_out_hermitian_ssm` (copy of the continuous-mamba
  recording-level LOSO loop, class swapped), `main()` branch
  (prediction-only, prints projected cache size, results under
  `results/hermitian_ssm/prediction/`).
- `scripts/hermitian_ssm_numerical_validation.py` -- design doc Section 20
  checks (synthetic Hermitian: Hermiticity / real eigenvalues /
  orthogonality / full + rank-k reconstruction / discarded-Frobenius-energy
  identity / phase-gauge invariance / known low-rank spectrum; plus
  `compute_recording_spectral` internals + freq_chunk determinism).
  **PASS** as of build.
- `scripts/hermitian_ssm_smoke.py` -- synthetic-recording end-to-end
  (encoder -> Mamba -> head fwd/bwd). **PASS**.

## Encoder: frequency + mode identity features (2026-08-27, after first build)

`vec_proj` / `val_proj` are weight-shared across both the frequency axis
and the `k` (mode) axis, so out of the box the model can only tell
frequencies / modes apart by their fixed slot in `mode_fuse` /
`freq_fuse`'s concatenation. Two cheap features fix that, both
concatenated onto the per-mode eigenvector input, neither touching the
cache:

- **`freq_feature`** (`freq_feature=True`): two normalised Hz scalars
  (linear + log, in [0,1], highest-first to match the cache's `freqs`).
  Lets the encoder express a smooth function of frequency instead of
  memorising 30 slots.
- **`mode_feature`** (`mode_feature=True`): a learned
  `nn.Parameter([k, d_mode_id=4])`, one small vector per mode SLOT.
  Because eigenpairs are `|lambda|`-sorted every timestep, slot r is
  "rank r" ("currently strongest", "second", ...), NOT a stable physical
  mode -- but rank is what the slot means, and the model can learn the
  dominant mode behaves differently from the secondary.

Both flags in `HERMITIAN_SSM_PARAMS` and
`HermitianSSMClassifier(...)`; set `False` for ablation.
`vec_proj` in-features: `2C` (46) -> `2C + 2 + 4` (52) with both on.

## Not yet done (encoder ideas discussed, not built)

- **Per-frequency Mamba lanes**: stop collapsing the frequency axis in the
  encoder; feed `_DenseEdgeMambaTemporal` `E = F` so Mamba runs one
  weight-shared scan per frequency-graph (the edge-axis trick, same as
  `dense_edge_mamba`/`temporal_graph_mamba`), fuse the F lane outputs
  after the scan. Matches doc Section 16. Would be a config switch
  `temporal_lanes: "fused" | "per_freq"`. Caveat: no cross-frequency
  mixing inside the recurrence (that is Design A / STG-Mamba). Lanes
  should be frequencies (stable identity), not modes (rank slots).

## Deliberate deviations from the doc

- Coherence smoothing operator `S` is **time-only** (1-D Gaussian over the
  downsampled time axis), not the windowed WCT pipeline's 2-D `(5,3)`
  time x frequency kernel -- the doc (Section 16) keeps frequencies
  independent and fuses them in the learned encoder, so cross-frequency
  smoothing would pre-mix what the encoder is meant to combine.
- Frequency grid is **log-spaced** (`utils.torch_cwt`'s only grid), not the
  doc's `linspace`.
- `freq_downsample=2` (doc has no frequency decimation) -- user's addition.

## Status / next

- Numerical validation + synthetic smoke pass.
- **Real CLI smoke PASS** (`--pipeline hermitian_ssm --smoke --max-folds 1
  --device cpu`): ran clean end to end against real chb01 data, exit 0,
  wrote both prediction CSVs. `--smoke` auto-shrinks the spectral config
  (nfreqs=12, freq_downsample=2, time_downsample=32, highest=60) in
  `_hermitian_ssm_clf_params` -- the full-config precompute is exercised by
  `scripts/hermitian_ssm_numerical_validation.py`, not the CLI smoke. The
  shrunk-config smoke cache was deleted afterward (different `cache_key()`
  from a real run).
- **Precompute cost (measured, corrected):** batched CPU `torch.linalg.eigh`
  ~29 us / 23x23 matrix; full config is 30 freq bins x ~57600 timesteps
  per 1h recording -> ~1 min/recording, ~40 min one-time for a full
  6-fold (~40 recordings), then cached. `torch.linalg.eigh` has NO MPS
  implementation in this torch build (confirmed -- NotImplementedError),
  so precompute is CPU-only for now; a CUDA box would run it on GPU.
- Cache size: ~0.65 GB / 1h recording (30 freq bins, k=2); ~24 GB
  projected for all of chb01. `main()` prints the projected total before
  the run.
- Real 6-fold LOSO run: **NOT started** -- user chose to hold off after the
  build. Command when ready: `.venv/bin/python Epilepsy/run_pipelines.py
  --pipeline hermitian_ssm --device cpu` (no `--smoke`).
- To make room: `~/mne_data/dense_edge_cache` (63 GB recompute cache) was
  deleted 2026-08-27 (only ~21 GB was free; full-res hermitian_ssm cache
  is ~26 GB). 84 GB free after. The next `dense_edge*`/`temporal_graph_*`
  run recomputes that cache.
- Also not done: any tuning; Mamba-3; GPU precompute path.
- Branch: `graph-state-mamba` (same branch the aggregate-then-Mamba work
  is on). Not committed yet.
