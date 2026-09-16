# "What the SSM sees" figures — CWT/WCT + hermitian_ssm windows

**Date:** 2026-09-14 · Mac shell · branch `main` (docs-only, no pipeline
code changed)

## What was done

The user asked for a plot of one 30-s CWT window and one WCT window, then
(seeing a similar figure another shell made for a crypto-market SSM) asked
for the EEG analogue — a 7-panel "feature pipeline" diagram: aligned
signal → normalized input → causal Morlet CWT → complex wavelet coherence
→ Hermitian channel graph → eigendecomposition → the spectral feature
matrix the Mamba actually consumes. Built two render scripts + PNGs, all
computed from one real chb01_03 preictal window (t ≈ 2570–2600 s; seizure
onset 2996 s; SPH 300 / SOP 900), not synthetic data.

### `docs/assets/render_cwt_wct_windows.py`

Two variants each for CWT and WCT:

- `*_legible.png` — `nfreqs=300`, human-readable scalogram / coherence +
  phase.
- `*_model.png` — **exactly** what `temporal_graph_mamba` "pre" (the AP
  ≈0.67 leader) ingests: `nfreqs=8`, 8–40 Hz log grid, native `T=7680`
  CWT; the WCT panel is the real `[4, E, T_ds≈479, F=8]` dense-edge stack
  (coh, sinφ, cosφ, significance) replicating
  `SparseEvidenceGNNCore._full_edge_wct_maps` / `_smooth_wct_maps` /
  `_coi_valid_mask` / `_downsample_dense_edge_time` (separable Gaussian
  smooth kernel (5,3) σ=(2,1), COI mask, ×16 time avg-pool).

Caught and fixed mid-session: the first `nfreqs=8` render looked smooth/
blurred — that was `imshow`'s default bilinear interpolation blending 8
pixel-rows into a gradient, not the data. Fixed with
`interpolation="nearest"`; the 8 discrete log-spaced frequency rows are
now visible as hard bands.

Confirmed for the user: `coherence_threshold` (0.90, fixed mode) survives
only as the affine rescale of the dense-edge stack's 4th channel,
`significance = (coh − 0.9)/0.9` — there's no hard gate/threshold in this
(dense / temporal_graph) code path, only in the separate sparse-events
path this pipeline doesn't use. Visible in `wct_window_model.png`: ch3 is
literally a recolored copy of ch0.

### `docs/assets/render_hermitian_ssm_window.py`

Panel-for-panel analogue of the crypto figure, replicating
`hermitian_ssm_cache.compute_recording_spectral` (256 Hz, 8–40 Hz,
`nfreqs=16`, mains notch 60/120 Hz, `time_downsample=16` → 480 steps,
5-step Gaussian time smoothing, `freq_downsample=2` → `F_out=8`,
`diagonal="zero"`, `k=6`) on a padded (±12 s context, cropped after) real
segment.

Went through two correction rounds after the user pushed back:

1. **First pass** collapsed frequency into one `H(t) = Σ_f w_f Γ(f,t)`
   Hermitian graph per timestep (matching the crypto figure's single
   per-asset-graph convention) and eigendecomposed *that*. User asked
   "doesn't the model also have a graph at each frequency?" — correct:
   the real cache keeps `Γ(f,t)` **un-reduced**, batching the
   eigendecomposition over `[T_ds, F_out]` jointly
   (`eigenvalues[T,F,k]`, `eigenvectors[T,F,k,C]`), i.e. 480×8 = 3840
   independent small Hermitian graphs per window, not 480. The
   frequency collapse was something *I* invented for display, not
   what the model stores.
2. **Rewrote panels 5–7** to match: panel 5 is now a 4-timestep × 8-
   frequency grid of `|Γ(f,t)|` snapshots (no reduction); panel 6 is 8
   small per-frequency subplots each with its own top-4 `|λ(f,t)|`
   tracks; panel 7's feature-matrix rows are pulled directly from the
   real `(eigvals[T,F,k], eigvecs[T,F,k,C])` arrays (λ₁ per freq bin,
   `|u₁|²` channel loadings averaged over freq, mean `|Γ|` per freq) —
   no invented single-graph reduction anywhere in the final figure.
   Also removed an earlier fabricated `d_spec` formula from the title
   (the real encoder, `_ComplexSpectralEncoder`, consumes the
   `(eigvals, eigvecs)` arrays structurally through complex-weight
   layers, not as one flattened `d_spec`-wide vector — there's no
   single number to quote).

**Caveat flagged to the user and worth repeating here:** `hermitian_ssm`
is a **shelved / negative-result direction** (6-fold mean AP ≈0.44 vs
"pre"'s ≈0.67 — `NEGATIVES.md`, `temporal_graph_mamba_math.md` §4). This
figure is "what that SSM would see," not the production model. It also
answered a related user question directly: panel 4's coherence *is* the
real `hermitian_ssm` coherence math, but it is **not** the leader's — the
leader ("pre") uses a different config (nfreqs=8, 2-D (5,3) smoothing,
COI mask, no matrix/eigendecomposition at all); that pipeline's own
faithful figure is `wct_window_model.png` above.

## Files (all untracked as of this note — not yet `git add`ed)

- `docs/assets/render_cwt_wct_windows.py`
- `docs/assets/{cwt,wct}_window_{legible,model}.png`
- `docs/assets/render_hermitian_ssm_window.py`
- `docs/assets/hermitian_ssm_window.png`

## Next / open

- Nothing queued from this session — it was a documentation/visualization
  task, no experiment launched. Normal run-discipline (always keep a job
  queued) applies to the *next* experimental session, not here.
- If these get committed, consider linking `hermitian_ssm_window.png`
  from `temporal_graph_mamba_math.md` §4 (the `hermitian_ssm` section)
  since it's a direct visual companion to that math.
