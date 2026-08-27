# Graph Spectral Mamba-3 — Design Reference

## Purpose

Transform a multichannel time series into a compact spectral representation
of its evolving channel-coupling graph, then model that representation with
Mamba-3.

The pipeline is **task-agnostic**. The downstream task may be binary or
multiclass classification, regression, or any other sequence-level task, and
the representation assumes nothing about class definitions. The spectral
cache is built once and reused across tasks and across temporal models
(Mamba, Transformer, GRU, LSTM, TCN, ...).

The central idea:

> Instead of passing the full channel graph through the temporal model,
> compress each graph into its dominant spectral coupling modes first.

---

## 1. Pipeline

```text
Raw multichannel time series  X ∈ ℝ^(N_channels × N_samples)
        ▼  windowing
Windowed signal
        ▼  continuous wavelet transform (per channel, complex)
W(channel, freq, time)  ∈ ℂ
        ▼  temporal downsampling
W(channel, freq, timestep)
        ▼  wavelet cross-spectrum   X_ij = W_i · conj(W_j)
        ▼  wavelet coherence        magnitude + relative phase
Complex Hermitian channel graph  A(f,t) ∈ ℂ^(N×N)
        ▼  Hermitian eigendecomposition   A = U Λ Uᴴ
        ▼  keep top-k eigenpairs
{(λ_r, u_r)}_{r=1..k}   per frequency, per timestep
        ▼  complex-aware spectral encoder  (modes → frequency → token)
[B, T, d_model]
        ▼  Mamba-3
        ▼  task-specific head
task output
```

There is a **hard boundary** between the precomputed representation (raw →
CWT → coherence → Hermitian graph → eigh → cache) and the learned model
(cache → spectral encoder → Mamba-3 → head). Everything left of the cache is
deterministic and computed once; everything right of it needs
forward/backward passes.

---

## 2. Raw data

Input is an arbitrary multichannel time series `X ∈ ℝ^(N_channels ×
N_samples)`. Do not hard-code the channel count. For CHB-MIT: `N_channels =
23`, `sampling_rate = 256`.

Carry enough metadata (`record_id`, `channel_names`, `sampling_rate`,
`start_time`, `duration`) to map any window back to its source recording and
time interval. Labels/targets stay separate from the representation.

---

## 3. Windowing

Divide the recording into temporal windows `X_window ∈ ℝ^(N_channels ×
N_samples)`. Window duration and stride are parameters of the surrounding
task; the spectral pipeline does not depend on them.

---

## 4. Continuous wavelet transform

For each channel independently, compute a complex CWT `W_i(f, t)`, giving a
`[channel, frequency, time]` tensor of `complex64` values. Retain the
complex coefficients (amplitude + phase) rather than collapsing to
magnitude, because the cross-channel construction uses complex
multiplication.

Frequency grid is configurable:

```python
lowest  = 8.0
highest = 40.0
nfreqs  = 60
frequencies = torch.linspace(lowest, highest, nfreqs)
```

Logarithmic grids, band-specific grids, and different limits/densities are
possible alternatives.

---

## 5. Temporal downsampling

The CWT inherits the full sampling-rate time resolution, which is finer than
the graph representation needs. Downsample along time after the CWT, applied
identically to all channels:

```python
time_downsample = 16     # 16/256 Hz ≈ 62.5 ms per graph timestep
```

Result: `W ∈ [N_channels, N_freqs, N_timesteps]`, complex.

---

## 6. Wavelet cross-spectrum

For channels `i`, `j`:

$$X_{ij}(f,t) = W_i(f,t)\,\overline{W_j(f,t)} = |X_{ij}|\,e^{i\phi_{ij}}, \qquad \phi_{ij} = \arg(W_i) - \arg(W_j).$$

The cross-spectrum has conjugate symmetry `X_ji = conj(X_ij)`.

---

## 7. Wavelet coherence

The raw cross-spectrum scales with the individual signal powers. Normalize:

$$C_{ij}(f,t) \sim \frac{|S(W_i\overline{W_j})|}{\sqrt{S(|W_i|^2)\,S(|W_j|^2)}}$$

where `S` is the smoothing operator. This yields, per (channel pair,
frequency, timestep), the **coherence magnitude** and **relative phase**
needed to build the graph.

---

## 8. Complex Hermitian channel graph

Off-diagonal entries:

$$A_{ij}(f,t) = C_{ij}(f,t)\,e^{i\phi_{ij}(f,t)} = C_{ij}\bigl(\cos\phi_{ij} + i\sin\phi_{ij}\bigr).$$

This keeps coupling strength and relative phase without representing phase
as a discontinuous scalar. Since `A_ji = conj(A_ij)`, the matrix is
Hermitian: `A = Aᴴ`.

**Diagonal.** A Hermitian matrix must have a real diagonal, so the raw
complex CWT coefficient cannot go there. Use channel-local wavelet power:

$$A_{ii}(f,t) = |W_i(f,t)|^2.$$

Diagonal = local activity, off-diagonal = inter-channel coupling. A
`diagonal = zero` ablation (`A.fill_diagonal_(0)`) should stay available to
test whether local power adds information beyond the coupling graph.

**Construction.** For each `(f, t)` build `A ∈ ℂ^(N×N)`; for CHB-MIT the
full tensor is `[N_freqs, N_timesteps, 23, 23]`. The graph is a complete
mesh with `N(N-1)/2` unique off-diagonal pairs (253 for N=23). Compute the
upper triangle, then reconstruct the lower triangle by conjugation (`A[j,i]
= conj(A[i,j])`) — do **not** zero it. For storage the graph can be packed
as (upper-triangle complex + real diagonal) and re-densified before
`eigh`.

---

## 9. Hermitian eigendecomposition

For each frequency and timestep:

$$A = U\Lambda U^H, \qquad \texttt{torch.linalg.eigh(A)}.$$

Because `A` is Hermitian: eigenvalues are real, eigenvectors are orthonormal
and generally complex. Output: eigenvalues `[N]`, eigenvectors `[N, N]`.

**Ordering.** Do not rely on the solver's ordering; reorder explicitly.
Default: descending `|λ|`,

$$|\lambda_1| \ge |\lambda_2| \ge \cdots \ge |\lambda_N|.$$

The coherence-plus-phase matrix with a real power diagonal is not guaranteed
positive semidefinite, so negative eigenvalues are possible; sorting by
`|λ|` is the correct convention for "largest spectral components" in the
sense of the Frobenius-optimal rank-k approximation. (For a PSD matrix this
coincides with descending `λ`.) The sort key stays configurable.

---

## 10. Top-k spectral compression

Retain the first `k` eigenpairs `{(λ_1, u_1), ..., (λ_k, u_k)}`. Default `k
= 2`; supported range `1 ≤ k ≤ N_channels`.

The rank-k approximation is

$$A_k = \sum_{r=1}^{k} \lambda_r u_r u_r^H,$$

but the model forward pass does not reconstruct `A_k` — it operates on the
eigenpairs directly.

**What a mode means.** `λ_r` = how strongly that coupling mode contributes;
`u_r` = which channels participate and with what relative complex
structure. Keeping only eigenvalues would discard the localization
information; the representation keeps both. Each mode is a distributed
pattern over all channels — rank-2 does not mean "two edges", it means the
graph is approximated by its two dominant coupling patterns.

**Optimality.** For a Hermitian matrix the truncated spectral
representation is the optimal rank-k approximation in the appropriate norm.
The discarded Frobenius energy is exactly

$$\|A - A_k\|_F^2 = \sum_{r>k} |\lambda_r|^2$$

when eigenvalues are ordered by magnitude, so the compression/error
tradeoff is well defined.

**Interpretation of k.** `k=1` one network-wide mode; `k=2` two dominant
coupling patterns; `k=4–8` most major coupling structure; `k=12` a rich
representation; `k=N` the complete graph, reconstructable exactly up to
floating point. Even at `k=N` the eigendecomposition is useful: it
re-expresses the graph as (mode strength × channel participation) rather
than (channel pair × channel pair). Rank reduction is a separate operation
from that change of coordinates.

`k=2` is an aggressive compression and should be treated as a hypothesis.
Weak modes can still carry task-relevant information despite low energy, so
build the representation with `k` as a clean architectural knob and sweep
it (1, 2, 4, 8, 12, N).

---

## 11. Eigenvector phase gauge

Eigenvectors are unique only up to a phase: if `u` is an eigenvector so is
`e^{iθ}u`. A deterministic phase convention is required before caching.

**Canonicalization:**

1. Find the eigenvector element with largest magnitude, `u_m`.
2. Compute `θ = arg(u_m)`.
3. Rotate the whole vector: `u' = u · e^{-iθ}`, so `u'_m` is real and
   non-negative.

**Degenerate / near-degenerate eigenvalues.** When `λ_r ≈ λ_{r+1}` the
individual eigenvectors are not well defined (the solver may rotate the
basis within the degenerate subspace) even though the subspace is stable.
This is mathematical, not a numerical bug. The initial approach: sort
deterministically, canonicalize phases, accept the resulting basis, and do
**not** attempt temporal eigenvector tracking.

Record spectral gaps `|λ_k − λ_{k+1}|` as a diagnostic — a large gap after
`k` suggests a natural truncation point; a flat spectrum suggests aggressive
low-rank compression is less justified.

Gauge-invariant encodings (e.g. `u uᴴ`, subspace/projector representations)
are a possible fallback if canonicalization proves fragile, but they move
back toward an N×N representation and partly undo the compression, so they
are deferred.

---

## 12. Spectral encoder

Goal: turn the per-timestep collection of eigenpairs into **one**
`d_model` token, preserving the semantic grouping `frequency → mode →
(eigenvalue, eigenvector)` rather than flattening every scalar into an
unstructured vector.

**Per mode.** For each `(f, r)`, encode `(λ_{f,r}, u_{f,r})`:

- Eigenvector `u ∈ ℂ^N` → complex-aware learned projection → complex mode
  embedding. (A real fallback encodes `[Re(u), Im(u)]` as `2N` reals; this
  is a baseline, not the preferred form.)
- Eigenvalue kept as an explicit real mode-strength feature, combined with
  the mode embedding **additively** (separate embeddings summed) rather
  than by forming `λu` directly, so the model has explicit access to both
  mode shape and mode strength. `z = λ·E(u)` is a testable alternative.

**Mode fusion.** Concatenate the `k` mode representations → linear
projection → per-frequency representation. Alternatives: attention over
modes, weighted pooling, mode-axis SSM.

**Frequency fusion.** Concatenate the `nfreqs` frequency representations →
linear projection → `d_model`. Alternatives: frequency attention, pooling,
frequency-axis SSM. The frequency dimension becomes part of each timestep's
content, not an extra temporal axis.

Result: `[B, T, d_model]`, one token per graph timestep.

---

## 13. Mamba-3

Mamba-3 receives `[B, T, d_model]` — `B` batch, `T` graph timesteps,
`d_model` model dimension. Each token is the entire channel graph at one
moment, so the model sees `G(t_1), G(t_2), ..., G(t_T)` rather than
scanning channels or edges independently.

Mamba-3's complex state transition is internal to the temporal model and
retains an efficient real-valued implementation; the complex values in the
spectral representation do **not** require feeding native complex tensors
into the recurrence. The exact conversion from the encoder's complex output
to Mamba-3's real input interface is an encoder implementation detail.

Start with a modest `d_model = 256` and a conservative Mamba-3 config close
to existing project conventions. Do not tune Mamba hyperparameters and
architecture simultaneously in the first implementation.

---

## 14. Task head and pooling

The representation does not depend on the target type.

```text
binary       Mamba-3 → Linear(d_model → 1) → logit
multiclass   Mamba-3 → Linear(d_model → N_classes)
regression   Mamba-3 → Linear(d_model → N_outputs)
```

If the task produces one output per window, aggregate the sequence after
Mamba-3. Default: take the last timestep. Alternatives (configurable): mean,
attention, learned CLS token, max pooling.

---

## 15. Spectral cache

Precompute the eigendecomposition once; it is deterministic for a given
graph and does not belong in the training loop.

**Contents:** eigenvalues, eigenvectors, metadata, and targets where
appropriate.

**Logical layout** (preserve semantic dimensions; let the encoder do the
grouping, flatten only just before a linear layer if needed):

```text
eigenvalues   [window, timestep, frequency, mode]
eigenvectors  [window, timestep, frequency, mode, channel]
```

**Precision:** `eigenvalues float32`, `eigenvectors complex64`. Do not
introduce reduced precision (`float16`, `bfloat16`) until correctness is
established; any reduced-precision cache must be validated against
Hermiticity reconstruction, spectral reconstruction, and downstream
stability.

**Storage:** contiguous dataset- or partition-level storage, not millions
of tiny files. Follow the project's existing caching infrastructure.

```text
cache/dataset/
    spectral_eigenvalues
    spectral_eigenvectors
    metadata
    labels
```

**Size example** (`N=23`, `nfreqs=60`, `k=2`): per timestep,
`60·(2 + 2·2·23) = 5640` real scalars. This is the real-equivalent storage
count, not a claim that Mamba must see 5640 independent features — the
encoder collapses the hierarchy first.

---

## 16. Frequency dimension

Each frequency has its own graph `A_f(t)`, so the representation is a
collection of frequency-specific spectral graphs. The model can learn how
coupling structure changes across both time and frequency.

The 60 frequency-specific 23×23 graphs are processed **independently** — do
**not** materialize a `1380 × 1380` block-diagonal supra-adjacency matrix,
which would be almost entirely structural zeros (no cross-frequency
coupling is computed). Frequency is fused later by the encoder.

Cross-frequency interactions (phase-amplitude coupling, cross-frequency
coherence, bispectral features) could create genuine off-diagonal frequency
blocks in a future multiplex formulation. Outside the initial scope.

---

## 17. Not a message-passing GNN

There is no `node state → aggregate neighbors → update` operation. The
graph enters through its spectral decomposition `A → U, Λ`: eigenvectors
encode channel relationships, eigenvalues encode dominant coupling
strengths. A conventional GNN learns `H' = σ(AHW)` by propagating through
edges; this architecture characterizes the graph through its spectral modes
and then models the temporal dynamics of those modes. It is a
**graph-spectral temporal model**, not GNN + Mamba.

This is attractive when the graph itself is the quantity of interest and
repeated message passing is not needed.

---

## 18. Cache vs learned compression

Spectral compression (`A → eigh → top-k → cache`) is deterministic, done
once, and imposes a strong prior: *represent the graph in its dominant
coupling modes*. Learned compression (`A → learned encoder → Mamba`) runs
every forward pass and is optimized by backprop. These are different
architectural choices — a spectral representation is not just a cheaper
learned projection.

Every retained spectral component has a mathematical meaning, which makes
`k` far more interpretable than an opaque `Linear(4424 → 512)`.

---

## 19. Scaling with channel count

A dense Hermitian graph has `N²` entries but only `~N(N-1)/2` unique
off-diagonal relationships. A rank-k representation needs `k` eigenvalues +
`k` eigenvectors of length `N`, i.e. `~kN` rather than `N²`. The
compression ratio strengthens as `N` grows for fixed `k`: useful at
`N=23, k=2`, potentially dramatic at `N=64, k=2`. Spectral compression is
especially attractive for higher-density montages.

---

## 20. Numerical validation

Before any real training, expose checks:

| Check | Verify |
|---|---|
| Hermiticity | `‖A − Aᴴ‖` near machine precision |
| Real eigenvalues | `eigh` eigenvalues real to precision |
| Orthogonality | `Uᴴ U ≈ I` |
| Full reconstruction | `A ≈ U Λ Uᴴ` |
| Rank-k reconstruction | `A_k = U_k Λ_k U_kᴴ` |
| Reconstruction error | measured `‖A − A_k‖_F` vs theoretical `Σ_{r>k}|λ_r|²` |
| Phase canonicalization | multiply `u` by arbitrary `e^{iθ}`; canonicalization returns the same representation |

**Synthetic smoke test.** Build small random Hermitian matrices (`N=5`:
random complex upper triangle + real diagonal + conjugate lower triangle),
confirm `A == Aᴴ`, and run the full `eigh → sort → canonicalize → rank-k
reconstruction` path. Also test known low-rank matrices with an obvious
expected spectrum.

---

## 21. Diagnostics

The preprocessing pipeline should optionally save, per graph/window:

- frequencies, spectral rank `k`, eigenvalues
- **spectral energy ratio** `R_k = Σ_{r≤k}|λ_r|² / Σ_{r≤N}|λ_r|²` for
  `k = 1, 2, 4, 8, 12` — the fraction of Frobenius energy retained. High
  energy retention does **not** guarantee task-information retention; a
  low-energy mode may still be task-relevant. Spectral compression and task
  performance are separate questions.
- **reconstruction error** `E_k = ‖A − A_k‖_F / ‖A‖_F`
- **spectral gaps** `|λ_k − λ_{k+1}|` (see §11)

Eigenvalues may also cross order over time: after sorting, "mode 1" always
means the currently strongest mode, which may not be the same physical mode
across timesteps. This is acceptable; explicit mode tracking can be added
later if temporal mode identity matters.

---

## 22. Initial configuration

```python
sampling_rate    = 256

lowest           = 8.0
highest          = 40.0
nfreqs           = 60
cwt_backend      = "torch"

time_downsample  = 16          # ≈ 62.5 ms per graph timestep

diagonal         = "power"     # or "zero"
eigenvalue_sort  = "abs"       # or "value"
k                = 2

eigenvalue_dtype  = torch.float32
eigenvector_dtype = torch.complex64

d_model          = 256
```

---

## 23. Configurable components

CWT frequency range / count / backend · temporal downsampling · diagonal
representation · eigenvalue sort convention · spectral rank `k` ·
eigenvector canonicalization · encoder dimension · mode-fusion method ·
frequency-fusion method · `d_model` · Mamba-3 config · sequence pooling ·
task head.

---

## 24. Settled for the first implementation

1. Existing CWT implementation.
2. Wavelet cross-spectrum for pairwise relationships.
3. Wavelet coherence magnitude + relative phase for off-diagonal entries.
4. Hermitian channel graph.
5. Real wavelet power on the diagonal.
6. `torch.linalg.eigh`.
7. Retain eigenvalues **and** eigenvectors.
8. `k = 2`.
9. 60 frequencies, 8–40 Hz.
10. Precompute and cache the eigendecomposition.
11. Cache independent of the downstream task.
12. Encode each eigenmode as a structured object, not unrelated scalars.
13. Fuse modes and frequencies into one token per timestep.
14. Feed the temporal sequence into Mamba-3.

---

## 25. Explicitly deferred

Choose the simplest defensible option for each; do not try to solve them all
at once.

- Exact complex encoder architecture (native complex linear layers vs
  paired real ops)
- Exact Mamba-3 config; optimal `d_model`
- Optimal mode-fusion and frequency-fusion mechanisms
- Sequence pooling
- Cache file format and compression precision
- Eigenvalue sort by value vs `|λ|`
- Treatment of near-degenerate eigenspaces; temporal eigenmode tracking
- Phase-invariant vs canonicalized eigenvector encoding
- Whether the diagonal improves the representation
- Appropriate spectral rank `k`

---

## 26. Summary

$$\text{time series} \to \text{complex CWT} \to \text{wavelet coherence} \to \text{Hermitian graph} \to \text{spectral modes} \to \text{complex spectral encoder} \to \text{Mamba-3}$$

The key compression:

$$A_{N\times N} \to \{(\lambda_1, u_1), \ldots, (\lambda_k, u_k)\}$$

Each retained pair is one dominant coupling mode — how strong it is and
which channels participate — letting the temporal model track the strongest
evolving coupling structures per frequency and timestep without carrying the
full pairwise graph.
