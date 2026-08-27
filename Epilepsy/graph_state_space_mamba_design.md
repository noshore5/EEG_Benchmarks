# Graph-aware state-space Mamba: design notes (not yet built)

Status: **decided as the next architecture to build (2026-08-27)**, ahead
of decision-threshold calibration. Not started -- scoping only so far, all
below is design discussion, no code exists yet. Moved out of `CONTEXT.md`
into its own file (2026-08-27) since the discussion outgrew a pointer-doc
entry; `CONTEXT.md`'s open-threads section keeps a one-line pointer here.

## Motivation

Current `dense_edge_mamba`/`temporal_graph_mamba` run one shared Mamba per
edge (or per node, for `temporal_graph_mamba` -- see
`Session_notes/2026_08_27/` for how that one actually works: graph
aggregation happens fresh every timestep via mean-pooling incoming edge
messages, THEN each node's own resulting sequence runs through an
independent, weight-shared Mamba scan). Either way, **the graph's topology
never enters the Mamba recurrence itself** -- time (the scan) and space
(the GNN's message-passing/aggregation) are fully separate stages that
only talk to each other at a handoff point (see the diagram at
`noshore5.github.io/resources/architecture-diagram-mamba.html`). A
graph-aware/spatiotemporal SSM would instead bake the channel-graph
topology into the scan's own dynamics.

Known real cost of the current mean-aggregation step, worth keeping in
mind while scoping any alternative: it collapses up to 22 incoming edge
messages into one flat average per node per timestep, discarding which
neighbor contributed what. Prior experiments in this codebase
(`event_aggregation="gated_softmax"`/`"concat"`, subject 1, 3 seeds,
non-`temporal_graph` modes -- see `cwt_gnn_classifiers.py`'s
`event_aggregation` docstring, ~line 2630) found smarter aggregation
*didn't help* (gated softmax converged to near-uniform weights; concat,
which keeps every edge distinct, performed worse than every pooled
variant). That's real evidence, but it was never tested in the
per-timestep-repeated-aggregation setting `temporal_graph_mamba` uses --
open question, not resolved either way.

## Literature scan (2026-08-27)

**STG-Mamba** (arXiv 2403.12418, "Spatial-Temporal Graph Learning via
Selective State Space Model") is the closest existing formulation. Its
mechanism: multiply the adjacency matrix directly into the SSM's
discretization step -- `Δ' = Δ ⊙ adjacency` (implemented as `Δ' =
matmul(Δ, adj_padded)`, i.e. a literal matrix multiply across the node
axis), then `ΔA = exp(Δ' · A)`. A node's own state-transition speed at
timestep *t* becomes a function of every other node's delta, weighted by
the adjacency -- genuinely coupled spatial+temporal evolution in one
recurrence step, not two stages glued at a handoff. A `KFGN`
(Kalman-filtering GNN) module produces that adjacency dynamically per
timestep from three parallel temporal branches (recent/periodic/trend),
rather than using a fixed one.

STG-Mamba's own inputs/outputs (traced from the paper, `ar5iv.labs.arxiv.org/html/2403.12418`):
- Input: windowed per-node time series, `(batch, num_nodes, seq_len, features)`
  (12 historical steps in their benchmarks), MinMax-normalized to `[0,1]`.
- Adjacency: dynamic, input-dependent, one learned per temporal branch via
  `KFGN`'s `DynamicFilter-GNN`.
- Output: regression forecast, `(batch, num_nodes, 12)`, trained on
  RMSE/MAE/MAPE.
- Validated on traffic-speed (307 nodes), metro ridership (80 nodes),
  weather (184 nodes) -- all forecasting tasks on graphs with genuinely
  time-varying connectivity (rush hour, weekday/weekend cycles, seasons).

Two adjacent but less relevant papers also surfaced:
- **Graph Mamba Networks / GMN** (arXiv 2402.08678): tokenizes a node's
  neighborhood into a sequence and scans over THAT as a message-passing
  replacement -- no time axis involved, not a spatiotemporal model.
- **DG-Mamba** (arXiv 2412.08160): targets learning how a graph's own
  topology changes over time -- a different problem from ours, since
  chb01's channel-graph topology (complete mesh, 253 edges for 23
  channels) is fixed per window; only the coherence *values* on those
  fixed edges change.

**Mamba-3** (arXiv 2603.15569, Tri Dao et al., 2026 -- separately relevant
to point 4 below, not part of the original STG-Mamba scan): builds a
genuinely complex-valued state transition for Mamba, motivated by
state-tracking expressivity in language modeling, not graph structure.
Mechanism: an N-sized complex diagonal state is losslessly represented as
a 2N-sized real state; the complex transition matrix factors into a
commuting scaling part (magnitude decay) and an oscillatory part
(rotation), implemented per-dimension as 2x2 rotation blocks
(`[cosΔθ, -sinΔθ; sinΔθ, cosΔθ]`). Key practical detail: a **"RoPE
trick"** folds the rotations into the `B`/`C` projection terms *before*
the recurrence runs, rather than rotating the hidden state directly --
so this **runs on standard real-valued kernels/pscan, no custom CUDA
kernel required**. Historical context the blog post notes: SSMs prior to
Mamba (the S4/S4D lineage) were frequently complex-valued by default and
excelled in vision/audio domains with explicit frequency content, but
complex state was found unhelpful for language and phased out in Mamba-1
-- Mamba-3 is described as the first modern recurrent model to bring
complex-valued state transitions back, for a different reason
(state-tracking) than ours (representing Hermitian graph structure), but
the reparameterization technique is directly reusable regardless of why
you want complex state.

## Applying this to chb01's channel graph

chb01's 23-channel montage is a **complete mesh already** -- 253 edges,
fixed every window. This matters for scoping STG-Mamba's mechanism here:

- **KFGN is very likely unnecessary.** Its whole point is *dynamic*
  connectivity (edges appearing/disappearing, traffic/ridership patterns
  shifting over the day) -- chb01's topology never changes, only the
  coherence *values* on fixed edges do. Porting KFGN would be solving a
  problem this dataset doesn't have. The existing per-window coherence
  computation likely substitutes directly for whatever `Δ'` needs.

### Building a genuinely information-preserving adjacency: Hermitian per frequency

The cross-spectrum already computed in `_full_edge_wct_maps`
(`cwt_gnn_classifiers.py:3799-3803`, `xwt = W_src * conj(W_dst)`) has a
property worth exploiting directly: swapping src/dst conjugates the
result, so **coherence (magnitude) is symmetric but phase is
antisymmetric** -- `A_ij = coh_ij * exp(i*phase_ij)` satisfies `A_ji =
conj(A_ij)`, i.e. **Hermitian**, not merely real-symmetric. This falls
straight out of how cross-spectral coherence/phase are already defined --
free information, not an added assumption. One such matrix per frequency
bin (currently `nfreqs=8`), refreshed every timestep, built by scattering
the existing canonical-upper-triangle edge list into dense `23x23` form
(mirroring into the conjugate lower triangle) -- no new spectral math,
just a repackaging of `xwt_real`/`xwt_imag`/`coh`/`phase`, which are
already computed.

**Diagonal (2026-08-27 addition, not yet in any code): use each channel's
own CWT value, not a placeholder.** Self-coherence is trivially ~1 and
carries no information -- instead put each channel's own complex CWT
coefficient (`w_real + i*w_imag` for that channel) on the diagonal.
`auto1`/`auto2` (`|W_i|^2`, `|W_j|^2`) are already computed as a byproduct
of coherence normalization in the same function, and the raw
`w_real`/`w_imag` are available even earlier -- cheap to wire in. Keeps
the whole matrix internally consistent: complex diagonal (each node's own
state) and complex off-diagonal (pairwise coherence/phase) treated the
same way, rather than a mismatched "real diagonal, complex off-diagonal"
object.

**Representation choice: settled as Cartesian, for the plain (no-Mamba-3)
16-plane budget (2026-08-27).** With exactly 2 real numbers per (edge,
frequency) slot to stay at 16 planes total (8 freq x 2), the options are:
`(mag, raw phase)` -- wraparound discontinuity, the exact problem
`_build_dense_edge_input` already avoids elsewhere by storing `(sin phi,
cos phi)` instead of raw phase for its own (non-diagonal, sparse/dense-
mode) features; `(mag, sin)` or `(mag, cos)` alone -- ambiguous, loses the
sign needed to recover phase uniquely; `(sin, cos)` alone -- wraparound-
safe but drops magnitude (coherence strength) entirely, unacceptable
here; `(mag, sin, cos)` together -- the only fully wraparound-safe polar
encoding, but that's 3 numbers per slot, blowing the budget to 24 planes,
not 16. **Cartesian (Re, Im)** is the only representation that is both
wraparound-safe and fits in 2 real numbers: smooth everywhere on the
complex plane, no discontinuity, no dropped information, no extra plane.
Not a coin flip between polar and Cartesian -- Cartesian by construction,
for this specific budget.

This entire question dissolves if Mamba-3's complex-state
reparameterization (above) is adopted instead: once phase is handled as
an actual rotation operator (the 2x2 rotation block, or direct complex
multiplication) rather than a raw scalar angle fed into a generic dense
layer, there is no wraparound problem to route around in the first place
-- rotation composition on the circle has no discontinuity. The
wraparound issue is an artifact of representing phase as a bare number
for a dense layer to interpret, not a property of phase itself, so
adopting Mamba-3's treatment sidesteps this representation question
entirely rather than resolving it in Cartesian's favor.

### Real degrees of freedom, per timestep

Per frequency bin: upper-triangle edges (253) x 2 (real/imag) + diagonal
nodes (23) x 2 (real/imag) = `553` real values (dropping the redundant
conjugate lower triangle, since it carries no new information). Across 8
frequency bins: `553 x 8 = 4,424` real values per timestep -- this is the
actual information content of "16 independent Hermitian matrices per
timestep" (8 frequency bins x {real, imag} planes, per the (1a) split
below), not an inflated count.

**Important: dropped, not zeroed.** The lower triangle isn't empty/
discardable in the sense of carrying no information -- it's the
*conjugate* of the upper triangle (same physical coherence value,
derived rather than independently measured), not a separate zero-valued
quantity. Two genuinely different needs, handled differently:
- **Packed/flat representation** (feeding an encoder, building a token --
  what Design B needs, and what the 553-real-value count above already
  is): store only the 276 unique complex values (253 edges + 23
  diagonal), skip the lower triangle entirely. Standard practice, not a
  novel trick -- this is exactly LAPACK's "Hermitian packed storage"
  convention (`CHPEV` and friends): half the memory, zero information
  loss.
- **Dense `[N,N]` matrix** (what Design A's `Δ' = Δ @ A` matmul actually
  needs -- real linear algebra over every entry): reconstruct it from the
  553 packed values by mirroring the lower triangle as the conjugate of
  the upper (`A = A_upper + A_upper.conj().transpose() -
  diag(A_upper.diag())`), NOT by zeroing it. Every one of the 529 entries
  ends up populated with a genuine value; zeroing the lower triangle
  instead would silently discard real signal (a strictly-upper-triangular
  operator is a different, deliberately asymmetric linear map, not a
  Hermitian one) rather than losslessly compressing it.

### Alternative packaging: supra-adjacency (multiplex network) framing

Instead of reasoning about 8 separate `23x23` Hermitian matrices, the
multiplex/multilayer-network formalism (Kivelä et al. 2014) unifies them
into ONE `184x184` (8x23) Hermitian supra-adjacency matrix: 8 diagonal
`23x23` blocks (each frequency's own Hermitian coherence matrix, exactly
as designed above, no new computation) plus 56 off-diagonal blocks
(coupling between frequency `f` and `f'`) that are currently all-zero, no
cross-frequency coupling is computed anywhere in this pipeline today.
Those off-diagonal blocks are a natural, motivated slot for a genuinely
new kind of information later -- e.g. phase-amplitude coupling between a
low-frequency phase and a high-frequency amplitude, an established EEG
phenomenon this codebase has never computed -- without restructuring
anything if that gets added.

**Real cost, not just a reframing, if implemented naively:** `184^2 =
33,856` entries vs. `8 x 23^2 = 4,232` for the 8 separate matrices --
exactly 8x, not a rounding difference, because a dense `184x184` tensor
stores all 64 `23x23` blocks in an 8x8 grid, but only the 8 diagonal
blocks carry real information while the other 56 are structural zeros.
Materializing this densely spends 7/8 of memory and compute on known
zeros. **Do not implement this as a literal dense `184x184` tensor unless
cross-frequency coupling terms actually get computed and fill in those
off-diagonal blocks with real information.** Until then, "8 separate
`23x23` matrices" and "one block-diagonal `184x184` Hermitian matrix" are
the same mathematical object -- the supra-adjacency framing is a genuine
conceptual unification (one clean description instead of a stack to
reason about separately) worth keeping in mind, but the block-list
storage (what's already scoped above, `553 x 8 = 4,424` real values) is
the only computationally sane implementation of it today.

## Two candidate architectures, genuinely different scope

**(1a), settled 2026-08-27:** represent each frequency's complex Hermitian
matrix as two real matrices (real and imaginary parts / equivalently
magnitude and phase), sidestepping complex arithmetic in the scan itself
-- `8 frequencies x 2 (real/imag) = 16` real matrix "planes" per timestep,
each `23x23`. (This was chosen over feeding complex numbers directly into
the SSM state -- see Mamba-3 note above for why that's no longer
obviously the harder path, though: the "no complex arithmetic" motivation
for (1a) may be revisitable now that Mamba-3's real-2N reparameterization
exists as a template. Worth reconsidering once Design A below is scoped
further.)

### Design A: node-axis-preserved, adjacency modulates delta directly (the actual STG-Mamba mechanism)

Requires keeping the node axis explicit all the way through the scan --
**a real blocker**: `_mixer_ssm_chunk`/`_DenseEdgeMambaTemporal`
(`cwt_gnn_classifiers.py:2242+`) currently folds the node/edge axis into
the batch dimension (`reshape(batch_size * num_edges, ...)`) specifically
to get weight-shared, mutually-independent per-unit scans. Once that
reshape happens, no axis remains to matmul an adjacency against -- STG-
Mamba's `Δ' = Δ @ A` needs all N nodes' deltas visible simultaneously in
one computation. This means rewriting the scan's internals, not adding a
config flag; every existing Mamba-based pipeline shares this code, so
changes need to be additive/opt-in.

`delta` is currently computed purely from a unit's own input (line
2260-2267: `deltaBC = mixer.x_proj(x)`, ..., `delta =
F.softplus(mixer.dt_proj(delta))`) with zero external signal -- the hook
point for adjacency modulation is right before `deltaA =
torch.exp(delta.unsqueeze(-1) * A)` (line 2268).

With 16 real planes (not STG-Mamba's single real adjacency), one real
matmul isn't literally available -- two options, genuinely different
architectures:
- **Widen `d_inner` into 16 sub-channels**, one per (frequency,
  real/imag) plane, each sub-channel's delta modulated by matmul against
  its own plane's adjacency, then concatenated. Preserves all information
  (no arbitrary compression), at real compute/memory cost -- `hidden_dim`
  is currently 8; this would multiply the Mamba-internal `d_inner`
  sharply, and would need its own `d_state`/`d_conv`/`expand` retuning
  pass to stay numerically stable.
- **Learn a weighted combination of the 16 planes into one adjacency**
  before a single matmul. Cheaper, keeps STG-Mamba's single-matmul
  mechanism intact, but is still a compression step -- just a learned one.
  Given the `gated_softmax` near-uniform-convergence result noted above
  (training given freedom to weight things differently found nothing
  worth differentiating, in a different but related aggregation context),
  there's a real chance this collapses to something close to a flat
  average anyway -- a hypothesis to test, not an assumed win.

Whichever is chosen, this genuinely "maintains the STG-Mamba graph
topology adjacency logic": the graph structure does real computational
work (a matmul against a structured operator), rather than being handed
to a generic dense layer that could learn to ignore it entirely.

### Design B: whole-graph-as-one-token, standard off-the-shelf Mamba

Simpler, no custom scan needed at all. Flatten one timestep's ~4,424 real
values (Hermitian upper-triangle + diagonal, all frequencies, both
real/imag planes) into a single vector, project with `Linear(4424 ->
d_model)`, and run one standard, single, shared Mamba over the sequence of
these per-timestep tokens -- one state trajectory for the WHOLE graph,
all 23 channels and their pairwise relationships visible at every step,
nothing folded into a batch dimension, nothing computed independently per
node or per edge. This recovers, architecturally, the "one big Mamba over
the whole graph" framing from earlier in this design discussion.

Real tradeoff: this drops the thing STG-Mamba's mechanism actually buys --
graph topology no longer shapes the recurrence's own dynamics (no `Δ' = Δ
⊙ A`); Mamba's learned `x_proj`/`dt_proj` would have to discover on its
own which of the ~4,424 flattened numbers matter and how they relate. The
"projection or not" question raised in discussion isn't really "skip
Mamba's input projection" (every Mamba block structurally has one,
`in_proj: d_model -> 2*d_inner`, unavoidable) -- it's whether the graph
enters as a *structured* multiply (Design A) or gets flattened into
generic learned weights with no built-in graph prior (Design B).

Much smaller build than Design A: a tensor-reshaping stage (repacking the
already-computed `[B, 4, E, T, F]` dense-edge tensor into the Hermitian
upper-triangle-plus-diagonal layout, not new spectral math), one linear
encoder, and reuse of a single standard Mamba block (no node-axis-folding
trick needed at all, since there's only one "unit" -- the whole graph).

### Design C: spectral (eigenvalue) compression of the coupling structure, node power kept raw

Added 2026-08-27, spun out of a spectral-theorem question mid-design.
Different axis of compression from A/B: instead of choosing *how* the
graph enters the scan, this asks whether the ~4,424-real-value-per-timestep
representation itself can be shrunk before it ever reaches Mamba, by
exploiting the Hermitian structure directly (spectral theorem: any
Hermitian matrix `A = U * Λ * U^H`, `Λ` real diagonal, `U` unitary).

**The split, not a single operation.** Diagonalizing the *whole* matrix
(diagonal node-power values and off-diagonal coherence together) smears
both into one joint spectrum and destroys the diagonal's own per-node
identity along with the off-diagonal's -- not what's wanted. The coherent
version splits the two pieces first:
- **Diagonal (node power/CWT, 23 complex values per freq bin): kept raw,
  untouched, fully node-indexed.** No eigendecomposition involved on this
  part at all -- this is exactly the "diagonal-as-node-CWT" feature
  already scoped above, unchanged.
- **Off-diagonal (253-edge coherence structure): diagonalized on its own**
  (zero the diagonal first, `eigvalsh` on the resulting Hermitian
  coupling-only matrix), keeping only the real eigenvalue spectrum `Λ`
  (23 real numbers), discarding `U` (the eigenvectors) entirely.

**Compression achieved:** off-diagonal goes from 253 complex edges (506
reals) down to 23 real eigenvalues per frequency bin -- roughly **22x**
on the coupling half of the representation. Combined with the untouched
23-complex-value diagonal (46 reals), one frequency bin's full per-timestep
footprint drops from `553` reals to `23 (diagonal, real) + 23 (diagonal,
imag) + 23 (eigenvalues) = 69` reals -- an **~8x** reduction overall, not
just on the coupling piece. Across 8 frequency bins: `4,424 -> 552` reals
per timestep.

**Expressiveness tradeoff, precisely:**
- *Survives fully*: per-node power/CWT (exact, node-indexed, no loss).
- *Survives as a summary*: total coupling "energy" (trace of eigenvalues
  ~ sum of squared coherences) and how concentrated vs. distributed it is
  -- one dominant eigenvalue reads as one global synchronized mode (the
  classic ictal-hypersynchrony signature this literature scan's
  IOPscience citation is built on), several comparable eigenvalues reads
  as multiple semi-independent clusters, a flat spectrum reads as diffuse
  unstructured pairwise coupling.
- *Lost entirely*: spatial localization of the coupling -- eigenvalues
  alone can't say *which* channel pairs are synchronized, only *how much*
  synchronization exists in aggregate. No seizure-focus localization
  signal survives in this piece of the representation. Also lost: any
  asymmetric/directional phase structure, since eigenvalues of a Hermitian
  matrix are real magnitudes of modes, not the mode shapes themselves.

**Tunable middle ground: keep top-`k` eigenvector/eigenvalue pairs
instead of eigenvalues only.** By the Eckart-Young theorem, truncating to
the top-`k` components is the provably optimal rank-`k` Hermitian
approximation in Frobenius norm -- e.g. top-2 modes costs `2 x 23 = 46`
extra reals (plus 2 eigenvalues) but recovers the dominant spatial
coupling pattern, not just its magnitude. This is a genuine dial between
"23 reals, zero locality" (`k=0`, eigenvalues only) and "506 reals, full
locality" (`k=23`, i.e. the untouched original matrix) -- worth sweeping
rather than committing to an endpoint.

**Compute cost, not free but genuinely small at this graph size.**
Eigendecomposition of an `n x n` Hermitian matrix is `O(n^3)`; for `n=23`
that's ~12K-120K flops per matrix depending on constant factor
(eigenvalues-only via `eigvalsh`/LAPACK `heevr` is cheaper than computing
eigenvectors too). Across 8 freq bins x every timestep x every window
across chb01's folds this totals roughly 10^11 flops in aggregate --
seconds, not minutes, of CPU time. Three things make this practically
fine rather than just theoretically small:
1. It's a data-prep step, computed once and cached (same place as
   `_full_edge_wct_maps` already runs today), not repeated per epoch.
2. `torch.linalg.eigvalsh`/`eigh` batch over an arbitrary leading
   dimension -- `(n_freqs * n_timesteps, 23, 23)` in, all eigenvalues out
   in one vectorized call, no Python-level loop over millions of individual
   solves.
3. The `O(n^3)` scaling is specifically cheap *because* chb01 only has 23
   channels -- this would stop being "small" fast on a larger montage
   (e.g. 64 channels is `~22x` more expensive per matrix than 23).
Caveat if this were ever made end-to-end differentiable rather than a
fixed precomputed feature: `eigh`'s backward pass is unstable near
degenerate/close eigenvalues (`1/(λ_i - λ_j)` blowup) -- not a concern for
the precompute-and-cache use case, but relevant if Design C ever grows
into something trained jointly.

**What this does and doesn't fix about caching cost.** This is a
*downstream feature* compression, not a fix for the existing
`dense_edge_cache`'s size -- `dense_edge`/`temporal_graph_mamba` consume
the full per-edge structure directly and can't be retrofitted onto
eigenvalues without losing the very information they operate on. Design C
would need its own, separate, smaller cache; it doesn't shrink the cache
any of the other pipelines already depend on. If the dense-edge cache's
disk footprint is the actual pain point, the more relevant levers are
lower `nfreqs`, fp16 storage, or Hermitian packed storage (553 unique
reals instead of a redundant full layout, if not already done) -- not
spectral decomposition.

Relationship to Design B: this is compatible with, not competing against,
Design B's "flatten one timestep into one token" packaging -- swap the
`~4,424`-real flattened vector for the `~552`-real spectrally-compressed
one and everything else about Design B (linear encoder, single shared
Mamba, no custom scan) is unchanged. Not compatible with Design A as
described (Design A's `Δ' = Δ @ A` needs the literal dense node-indexed
adjacency; reconstructing it from `Λ` alone is impossible without `U`,
which is exactly what Design C discards).

## Cost/risk summary

- STG-Mamba's own benchmarks are traffic/sensor graphs of comparable node
  count to chb01's 23-channel graph, so raw scale looks tractable for
  either design.
- No existing EEG/seizure implementation of any of this to adapt --
  porting/inventing is entirely on us either way.
- Design A's per-plane matmul has no existing kernel support in
  `mamba-ssm`'s fused CUDA path -- would run on the pure-PyTorch `pscan`
  path only, same throughput ceiling already measured for the
  continuous-mamba carried-state work (`Session_notes/2026_08_25/
  continuous_mamba_chunk_scan_throughput.md`). Design B has no such
  constraint (standard Mamba block, fused kernel eligible as usual).
- Mamba-3's real-2N complex-state reparameterization (RoPE trick) may be
  directly reusable for Design A regardless of which 16-plane-combination
  option is chosen, and specifically undercuts the earlier assumption
  that complex arithmetic forces a slow custom scan -- worth reading the
  actual Mamba-3 paper (not just the blog summary) before finalizing
  Design A's approach.

## Recommendation / next steps

1. Prototype Design B first -- much smaller build, tests whether the
   richer (but topology-agnostic) input representation alone beats
   `temporal_graph_mamba`'s current per-node-independent-scan baseline
   (mean AP 0.674, first real 6-fold run, `Session_notes/2026_08_27/
   temporal_graph_mamba_full_6fold_and_tuning_attempt.md`) before
   committing to Design A's much larger scan rewrite. Cheapest variant of
   this prototype: run Design B with Design C's spectrally-compressed
   `~552`-real token instead of the full `~4,424`-real flattened one --
   same architecture, smaller/faster build, and a natural first data point
   on whether the full coupling matrix's spatial detail matters at all or
   whether the coherence *spectrum* alone already carries most of the
   useful signal.
2. If Design B (either token variant) shows the richer representation
   matters, THEN invest in Design A -- decide the 16-plane-combination
   question (widen `d_inner` vs. learned-weighted-combination) and read
   the full Mamba-3 paper for whether its rotation trick simplifies the
   per-plane matmul implementation. Design C's eigenvalue-only compression
   does not carry over to Design A (no adjacency reconstruction without
   `U`), so this step stays Design A vs. B on the original representation,
   not vs. Design C.
3. Small/smoke-scale synthetic-data prototype before any real chb01 run,
   same "smoke-test before trusting a new pipeline wire-up" convention
   `temporal_graph_mamba` itself followed
   (`scripts/temporal_graph_mamba_smoke.py`).
4. The untuned `temporal_graph_mamba` 6-fold retrain
   (`/tmp/tg_mamba_retrain_full.log`, `--dump-window-scores`) referenced
   above as a blocking condition has since finished cleanly (2026-08-27,
   see `Session_notes/2026_08_27/
   temporal_graph_mamba_full_6fold_and_tuning_attempt.md`) -- no run is
   currently active, so the "wait for it to finish" condition no longer
   applies, though the general "no competing processes while a monitored
   run is active" convention still holds for any future run.
