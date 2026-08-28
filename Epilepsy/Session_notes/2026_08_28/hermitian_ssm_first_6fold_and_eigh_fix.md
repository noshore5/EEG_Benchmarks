# hermitian_ssm: first real 6-fold LOSO, eigh convergence fix, d_model speed diagnosis (2026-08-28)

Follow-up to `2026_08_27/hermitian_ssm_pipeline_built.md`. The build was
committed (`26dd172`) the same day; this session ran it for real.

## eigh convergence bug (found + fixed mid-run)

First two run attempts crashed in `compute_recording_spectral`
(`hermitian_ssm_cache.py`) on recording `1_06`:

```
torch._C._LinAlgError: linalg.eigh: (Batch element 200340): The algorithm
failed to converge because the input matrix is ill-conditioned or has too
many repeated eigenvalues (error code: 22).
```

Real CHB-MIT recordings have flatlined / dropped-electrode segments; the
Hermitian channel-graph matrix for those `(freq, timestep)` slices is ~0
or has (near-)repeated eigenvalues, and LAPACK's Hermitian
divide-and-conquer solver (`syevd`, what `torch.linalg.eigh` uses on CPU)
fails to converge on them -- and one bad element kills the whole batched
call.

Fix (in `compute_recording_spectral`, the eigh call):

1. `a = torch.nan_to_num(a, nan=0, posinf=0, neginf=0)` -- a NaN/Inf
   anywhere also fails the solve.
2. `try: torch.linalg.eigh(a)` / `except torch._C._LinAlgError:` fall back
   to `torch.linalg.eig(a)` (the general **geev** path) for that whole
   freq chunk. Different LAPACK routine, no `syevd` convergence mode.
   Hermitian input still gives real eigenvalues (drop the ~1e-7 imaginary
   residue) and unit-norm, near-orthonormal eigenvectors -- fine, because
   the code immediately re-sorts by `|lambda|` and phase-gauges anyway.

A diagonal-regulariser approach (Tikhonov jitter scaled to the diagonal
magnitude, escalating) was tried first and did **not** work -- the
failing matrices had ~0 diagonal (`diagonal="power"` on a flat segment),
so the relative jitter was ~0. The geev fallback is the actual fix.

`scripts/hermitian_ssm_numerical_validation.py` still PASS after the
change.

## First real 6-fold result (d_model=256, untuned)

`.venv/bin/python Epilepsy/run_pipelines.py --pipeline hermitian_ssm
--device cpu`, ~6.5 h wall on CPU (reused the 34-recording spectral cache
from the crashed attempts; ~7 test recordings computed at predict time in
fold 1). Results:
`results/hermitian_ssm/prediction/*_20260827-231141.csv`.

| Fold | AP | FAR/h raw->sm | hit raw->sm |
|---|---|---|---|
| 1_03 | 0.107 | 15.2->10.8 | T / F |
| 1_04 | 0.130 | 38.7->31.7 | T / T |
| 1_15 | 0.498 | 23.4->8.0  | T / T |
| 1_16 | 0.330 | 9.5->0.0   | T / T |
| 1_18 | 0.251 | 19.3->1.2  | T / T |
| 1_26 | 0.104 | 8.2->2.2   | T / F |

**Mean: AP 0.237, ROC-AUC 0.871, precision 0.147, recall 0.643,
FAR/h 19.0->9.0, hits 6/6 raw / 4/6 k-of-n.**

Against the field (all full 6-fold chb01 prediction, mean AP):

| pipeline | AP |
|---|---|
| temporal_graph_mamba | 0.674 |
| dense_edge_gru k=20 | 0.567 |
| dense_edge_mamba (mamba-ssm fused) | 0.499 |
| dbconformer (best) | 0.442 |
| dense_edge_mamba (mambapy) / slimseiz | ~0.42 |
| **hermitian_ssm** | **0.237** |

Weakest full-6-fold pipeline so far. Catches every seizure raw but rings
constantly (precision 0.15); AUC 0.87 trails temporal_graph_mamba's 0.94,
so it is a worse *ranker*, not just badly thresholded. No tuning yet
(d_model, frequency band, regularization).

## Speed diagnosis: it's entirely d_model=256

~4 min/epoch here vs `temporal_graph_mamba`'s ~1 min. Benchmarked the two
halves at the real config (numbers inflated ~2x by contention with the
live run, ratios clean):

| component (fwd+bwd, B=64) | s/batch |
|---|---|
| `_SpectralEncoder` (all the embeddings/fusions) | ~0.6 |
| `_MambaTemporalHead`, **d_model=256** | ~4 |
| `_MambaTemporalHead`, d_model=16 | ~0.3 |

So the Mamba step is ~85% of the epoch, and `d_model` is the whole story:

- `d_model=256` -> `d_inner = expand*d_model = 512`. mambapy's pure-PyTorch
  pscan materializes `[B, T, d_inner, d_state]` = `[64, 480, 512, 16]` =
  ~1 GB tensors, several live (`deltaA`, `deltaB`, `BX`, `hs`).
- `E=1` (whole-graph token) -> `n_rows = B*E = 64 <= chunk_size=128`, so
  `_DenseEdgeMambaTemporal` takes its single-call branch and **skips
  gradient checkpointing** -> all ~3-4 GB retained for backward ->
  swaps on 16 GB RAM.
- `temporal_graph_mamba` avoids both: `d_model=16` (inherits
  `_SHARED_ARCH_PARAMS`) and `E=23` nodes -> `n_rows = B*23` clears
  chunk_size -> chunked + checkpointed, ~126 MB/tensor, never swaps.

`d_model=256` was a design-doc default ("whole-graph token width"), not a
measured choice. Raw per-`(timestep, freq_bin)` feature width is only
**94** (2 eigenvalues + 2 modes * 23 channels * 2 real/imag), fused across
30 freq bins -- 256 is oversized.

## This session's changes

- `hermitian_ssm_cache.py`: eigh fix (above).
- `hermitian_ssm_classifier.py`:
  - per-epoch wall time in the training log line (`(12.3s)`), `import time`.
  - `freq_feat` build: `np.asarray(freqs, ...)` -> `np.array(freqs, ...)`
    to silence the "non-writable NumPy array" UserWarning (the mmap'd
    `freqs` is read-only; `np.array` copies).
- `CONTEXT.md` + this note.
- **Next (Option B): `d_model` 256 -> 64** in `HERMITIAN_SSM_PARAMS`
  (shrinks `freq_fuse` output + token + Mamba together). Run kicked off
  after this commit. Cache key is unchanged by `d_model`, so the ~20 GB
  spectral cache is reused.

## d_model=64 run (Option B, `*_20260828-055322.csv`)

Same command, same spectral cache (key unchanged by d_model), `d_model`
256 -> 64 (committed `1ffec2e`). ~2.5 h wall.

| metric | d_model=256 | d_model=64 |
|---|---|---|
| mean AP | 0.237 | 0.241 |
| mean ROC-AUC | 0.871 | 0.880 |
| mean FAR/h raw->sm | 19.0->9.0 | 12.4->5.0 |
| k-of-n hits | 4/6 | 3/6 |
| epoch time | ~240 s | ~130 s |

Per-fold AP (d256 -> d64): 1_03 .107->.099, 1_04 .130->.300,
1_15 .498->.204, 1_16 .330->.433, 1_18 .251->.309, 1_26 .104->.105.
Noisy fold-to-fold (different init + early-stop points), but the means
say **narrowing the token cost nothing** -- AP flat, AUC slightly up, FAR
down, 2x faster. 256 was oversized; **d_model=64 adopted as the default.**

Epoch time only halved (not to ~1 min) because the bottleneck moved:
the Mamba head is no longer dominant, but the ~17 GB/fold eigenvector
mmap doesn't fit 16 GB RAM and `num_workers=0` serializes the reads
(swap sat at ~10 GB during the run). float16 cache + `num_workers` is the
next speed pass -- deferred, see below.

hermitian_ssm is still the weakest full-6-fold pipeline (AP ~0.24 vs
temporal_graph_mamba 0.67), and AUC 0.88 vs 0.94 means it is genuinely a
worse ranker, not just miscalibrated. Whether to keep iterating (float16
cache, frequency-band sweep, k sweep, lr/reg) or park it is the open
decision.

## Open

- Option A fallback if d_model=64 is capacity-limited: keep the encoder
  token at 256 but add `Linear(256 -> 64)` before the Mamba (the
  `_DenseEdgeMambaTemporal.in_proj` seam -- currently `nn.Identity()`
  because `in_channels == d_model`). Needs `_MambaTemporalHead` to take
  `token_dim` and `mamba_d_model` separately.
- Frequency band is unswept across the *entire* benchmark: everything
  else runs 8-40 Hz / nfreqs=8, chosen for disk budget (see
  `_SHARED_ARCH_PARAMS` comment), not accuracy. hermitian_ssm's 8-124 /
  30-bin input is a different regime -> the architecture comparison
  isn't clean. Cheap experiment now that `dense_edge_cache` is gone:
  run `temporal_graph_mamba` (current leader, fast) with a wider/finer
  band.
- Decision-threshold calibration (already a CONTEXT open thread) would
  help every pipeline's FAR/h, hermitian_ssm included.
- mmap dataset (~17 GB/fold) still > 16 GB RAM even at d_model=64 -- the
  cache isn't shrunk by d_model. If epochs are still IO-bound after the
  d_model cut, add DataLoader `num_workers` / `prefetch` (currently 0).
