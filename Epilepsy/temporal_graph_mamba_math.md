# `temporal_graph_mamba` "pre" — a linear-algebra description

What the prediction-leader pipeline (`--pipeline temporal_graph_mamba
--label-mode prediction`, `temporal_graph_aggregate="pre"`, mean 6-fold AP
≈ 0.67 on chb01 LOSO) actually computes, written as matrix operations
rather than ML vocabulary. Source: `Epilepsy/pipelines/cwt_gnn_classifiers.py`
(`SparseEvidenceGNNCore`, `event_mode="temporal_graph"`,
`_temporal_graph_node_states`, `_build_dense_edge_input`,
`_DenseEdgeMambaTemporal`). Canonical config: `_SHARED_ARCH_PARAMS` +
`PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS` in `run_pipelines.py`.

Notation: `C = 23` channels, `F = 8` frequency bins (Morlet CWT,
log-spaced 8–40 Hz), window 30 s @ 256 Hz `= 7680` samples, temporally
down-sampled by 16 to `T = 480`, `E = C(C-1)/2 = 253` unordered channel
pairs ("edges" — the graph is complete).

---

## 1. The Hermitian object at the core

For one window, the CWT gives each channel `c` a complex matrix
`W_c ∈ ℂ^{F×T}`. At each frequency–time cell `(f,t)` collect the channels
into `w(f,t) ∈ ℂ^{C}`.

**Instantaneous cross-spectral matrix** (rank-1, Hermitian, PSD):

$$S(f,t) \;=\; w(f,t)\,w(f,t)^{H} \;\in\; \mathbb{C}^{C\times C}.$$

**Smoothed** with a fixed separable Gaussian kernel `G` over a small
`(f,t)` neighbourhood (`smooth_kernel_size`, `smooth_kernel_sigma`):

$$\langle S\rangle(f,t) \;=\; \big(G * S\big)(f,t).$$

A Gaussian-weighted sum of rank-1 Hermitian PSD matrices is Hermitian
PSD of higher rank. Normalise by the diagonal `D = \operatorname{diag}\langle S\rangle`:

$$\boxed{\;\Gamma(f,t) \;=\; D^{-1/2}\,\langle S\rangle(f,t)\,D^{-1/2}\;}
\qquad \Gamma = \Gamma^{H},\quad \Gamma_{ii} = 1,\quad |\Gamma_{ij}|\le 1.$$

`Γ(f,t)` is the **frequency–time-resolved complex coherence matrix** — a
complex analogue of a correlation matrix.

- `|Γ_{ij}(f,t)|` — wavelet coherence: coupling strength of channels `i,j`
  at `(f,t)`.
- `arg Γ_{ij}(f,t)` — phase lead/lag between the channels.

Cells outside the wavelet cone of influence are zeroed (COI mask), so
`Γ` is only trusted where it has support.

This is the entire physical content the model is given. Everything below
is a fixed or learned map applied to `Γ`.

---

## 2. `pre` as a composition of maps

| # | operation | in → out shape | code |
|---|---|---|---|
| 1 | `Γ(f,t) = D^{-1/2}(G * w w^H)D^{-1/2}`, COI-masked | `w: ℂ^{C×F×T}` → `Γ: ℂ^{C×C}` per `(f,t)` | `_build_dense_edge_input` / `_smooth` |
| 2 | strict upper triangle of `Γ`, each complex entry in real polar coords, ↓-sample `t` by 16 | → `X ∈ ℝ^{E×T×4F}` | `_build_dense_edge_input`, `_downsample_dense_edge_time` |
| 3 | one learned affine map per `(edge, t)`, then GELU | `ℝ^{4F=32} → ℝ^{8}` | `temporal_edge_proj = Linear(32,8)∘GELU` |
| 4 | small MLP per `(edge, t)` | `ℝ^{8} → ℝ^{8}` (`Linear-GELU-Linear`) | `sparse_message_mlp` |
| 5 | row-normalised incidence average, edges → nodes | `Z(t) ∈ ℝ^{E×8} → H(t) ∈ ℝ^{C×8}` | scatter-mean in `_temporal_graph_node_states` |
| 6 | per channel (shared params): selective linear state-space recursion over `t`, keep last step | `H ∈ ℝ^{C×8×T} → \mathcal{E} ∈ ℝ^{C×8}` | `temporal_node_mamba = _DenseEdgeMambaTemporal` |
| 7 | flatten, one affine map to 2 logits | `ℝ^{C·8 = 184} → ℝ^{2}` | `sparse_classifier = Linear(184,2)` |

### Step 2 — the edge feature

For edge `e = (i,j)`, `i<j`, and each `f`:

$$x_e(f,t) \;=\; \Big(\;|\Gamma_{ij}|,\;\; \sin\arg\Gamma_{ij},\;\; \cos\arg\Gamma_{ij},\;\; \sigma_{ij}\;\Big),
\qquad \sigma_{ij} = \frac{|\Gamma_{ij}| - \theta}{\theta}.$$

Under `coherence_threshold_mode="fixed"` (what `pre` uses) `θ` is a
constant, so the 4th coordinate `σ` is an **affine function of the 1st** —
redundant, a fixed reparametrisation, no new information. (It exists for
`coherence_threshold_mode="surrogate"`, where `θ` is a per-edge
calibrated null and `σ` = "how far past the null this cell is." Not used
here.) Stacking over `F` gives the length-`4F` vector fed to step 3.

The map in step 2 is: take `Γ`, discard the diagonal (all ones anyway),
keep the strict upper triangle (`Γ` Hermitian ⇒ lower triangle is the
conjugate, no loss), write `ℂ ∋ z = r e^{i\varphi}` as `(r, \sin\varphi,
\cos\varphi)`.

### Steps 3–4 — the "encoder", as matrices

Let `A ∈ ℝ^{8×4F}` (step 3) and the `sparse_message_mlp` be
`M(u) = W_2\,\phi(W_1 u)` with `W_1, W_2 ∈ ℝ^{8×8}` (step 4), `\phi` =
GELU. Applied independently to every `(edge, timestep)`:

$$Z_e(t) \;=\; M\big(\phi(A\,x_e(t))\big) \;\in\; \mathbb{R}^{8}.$$

Same weights for all 253 edges and all 480 timesteps — a shared
`ℝ^{32} → ℝ^{8}` channel-pair-spectrum embedding.

### Step 5 — edges → nodes, as an incidence product

The 253 edges are the **undirected** pairs `(i,j)`, `i<j`, but the
aggregation is **directed**: each edge sends its message only to its
higher-indexed endpoint `j` (`self.dst_idx`; `self.src_idx` is unused in
this scatter). Let `B ∈ \{0,1\}^{C×E}` be that directed incidence —
`B_{c,e}=1` iff `c = \max(\text{endpoints of }e)`. Each **column** of `B`
has exactly one 1; **row** `c` has `c` ones (channel `c` is `j` for the
`c` edges `(0,c),\dots,(c-1,c)`). With `D =
\operatorname{diag}(1,1,2,3,\dots,C-1)` (channel 0's in-degree of 0
clamped to 1):

$$H(t) \;=\; D^{-1}\,B\,Z(t) \;\in\; \mathbb{R}^{C×8}.$$

So `H(t)_c` is the mean of `Z_e(t)` over the edges `(0,c),\dots,(c-1,c)`,
and **`H(t)_0` is always the zero vector**. This asymmetry — the node
aggregation depends on the (arbitrary) channel index order, and channel 0
carries no state — is a genuine quirk of "pre" as implemented, not a
choice anyone would design; it just never hurt the 0.674 number enough to
notice. `channel_subset_k` (fixed 2026-08-31) makes `D` per-window: only
the `k`-clique's edges are live, so `D_c` = that node's live in-degree,
and non-selected nodes get the zero vector.

This is `temporal_graph_aggregate="pre"`: **collapse the 253-edge graph
to 23 node sequences per timestep, then** run the temporal model. (`"post"`
runs step 6 on all 253 edge sequences first and averages after — `~11×`
more work, statistical wash, mean AP 0.639.)

### Step 6 — the selective state-space recursion

For each channel `c` independently, with shared parameters, input the
length-`T` sequence `u_1,\dots,u_T` with `u_t = H(t)_{c,:} ∈ ℝ^{8}`.
A learned `\text{in\_proj}: ℝ^{8} → ℝ^{16}` lifts each `u_t`; the state
`h_t ∈ ℝ^{16}` follows

$$h_t \;=\; \bar A_t\,h_{t-1} \;+\; \bar B_t\,\tilde u_t,
\qquad y_t \;=\; C_t\,h_t,$$

where

- `\bar A_t = \exp(\Delta_t \odot \Lambda)` is **diagonal**, `\Lambda ∈
  ℝ^{16}` a fixed learned vector with negative entries (a per-coordinate
  decay rate),
- `\Delta_t ∈ ℝ^{16}_{>0}`, `\bar B_t`, `C_t` are **affine functions of
  the current input `\tilde u_t`** (this input-dependence is the
  "selective" / gating part; a plain linear time-invariant SSM would have
  them constant).

Unrolled, the last output is an input-controlled linear combination of
the whole sequence:

$$y_T \;=\; \sum_{t \le T} \underbrace{\Big(C_T \prod_{t < s \le T} \bar A_s\Big)\bar B_t}_{\text{data-dependent kernel } K_{T,t}} \; \tilde u_t.$$

`pre` keeps only `y_T` (last timestep), passes it through
`\text{out\_proj}: ℝ^{16} → ℝ^{8}`, giving one vector `\mathcal{E}_c ∈
ℝ^{8}` per channel — an "end-of-window" summary of that channel's
coherence-averaged trajectory. Implementation: `mambapy` pure-PyTorch
parallel scan (portable; a fused CUDA kernel on Linux/CUDA is a flag
flip, same math).

### Step 7 — readout

$$\text{logits} \;=\; W\,\operatorname{vec}(\mathcal{E}) \;+\; b,
\qquad \mathcal{E} \in \mathbb{R}^{C\times 8},\;\; W \in \mathbb{R}^{2 \times 184}.$$

Note: a **flatten**, not a mean over channels — channel `c` sits in a
fixed slot of the 184-vector, so channel identity is preserved into the
final linear map. (Consistent with the `hermitian_ssm` encoder sweep
finding that discarding channel identity always costs accuracy.)

At the canonical config there is **no graph message passing** after step
5: `_propagate_hops` runs only when `n_hops > 1`, and `n_hops = 1`.

---

## 3. One-line summary

$$
\text{logits}
=
W \,\operatorname{vec}\!\Big[\,
\text{SSM}_\theta\!\Big(
D^{-1} B \; M\!\big(\phi(A\, \operatorname{triu}_{\text{polar}}\Gamma(\cdot,t))\big)
\Big)_{t=1:T}\,\Big]_{\text{last }t}
+ b
$$

with `Γ(f,t) = D_\Gamma^{-1/2}(G * w w^H)D_\Gamma^{-1/2}` the Hermitian
coherence matrix, `B` the directed (`i→j`, `i<j`) incidence matrix,
`D = \operatorname{diag}(1,1,2,\dots,C-1)` its row-sums, `A, M` shared
edge-spectrum maps, `SSM_θ` a per-channel selective (diagonal, gated)
state-space filter.

---

## 4. The matrix-native variant: `hermitian_ssm`

`--pipeline hermitian_ssm` keeps `Γ` **as a matrix** instead of
vectorising its upper triangle. It time-averages to `Γ(f)` (one Hermitian
`C×C` per frequency) and takes the eigendecomposition

$$\Gamma(f) \;=\; \sum_{r=1}^{C} \lambda_r(f)\, u_r(f)\, u_r(f)^{H},
\qquad \lambda_1 \ge \dots \ge \lambda_C \ge 0,\;\; u_r \in \mathbb{C}^{C}.$$

The "encoder" is then literally **rank-`k` truncation**: feed the top
`k = 6` eigenpairs `(\lambda_r ∈ ℝ, u_r ∈ ℂ^{C})` — with `u_r` written as
`[\operatorname{Re} u_r, \operatorname{Im} u_r] ∈ ℝ^{2C}` plus a
mode-index and `\lambda_r` — into the same kind of temporal model.

Results (`Session_notes/2026_08_29/`, `NEGATIVES.md`): mean 6-fold AP
**≈ 0.44 vs `pre`'s ≈ 0.67**. The encoder sweep tried four ways of
consuming `Γ` as an object — raw eigenvectors (0.44), gauge-invariant
projector summaries of `P = \sum_r \lambda_r u_r u_r^H` (0.27), the whole
upper triangle of `P` (0.17), pure mode-space evolution `M(t) = U(t)^H
U(t-1)` (≈ chance) — and accuracy **degrades monotonically with how much
channel identity the representation drops**. `pre`'s "flatten the
off-diagonal and hit it with a shared linear map, keep every channel in
its own slot" turns out to beat every attempt to exploit the spectral
structure of `Γ`. Design intent for a fuller matrix-native model:
`Epilepsy/graph_state_space_mamba_design.md`.

---

## 5. What the `channel_cwt` ablations remove

`channel_cwt` (`Epilepsy/pipelines/channel_cwt_mamba.py`, the null
baseline):

- **drops steps 1–2 and 5 entirely.** No cross-spectrum, no `Γ`, no
  incidence matrix. The per-channel feature is just `|W_c(f,t)|^2`
  (power), z-scored per `(c,f)`.
- keeps step 3 (a real `Linear` on the per-channel power spectrum),
  step 6 (the *same* `_DenseEdgeMambaTemporal`, now one sequence per
  channel), and a readout.

So it is `pre` with the Hermitian coherence matrix deleted and replaced
by the per-channel magnitude spectra. Result: mean AP collapses to
**0.149 / 0.263** on the first two folds (killed 2/6), ~0.5 below a
same-machine `pre` reproduction (`Session_notes/2026_08_31/
pre_repro_6fold.md`).

**Ablation 1** (`mamba_n_hops=2`) adds one round of learned pairwise
interaction back — but on the *post-SSM node vectors* `\mathcal{E}_c`, not
on `Γ`:

$$m_{i\to j} = \text{MLP}\big([\mathcal{E}_j, \mathcal{E}_i]\big),\qquad
\mathcal{E}_j \leftarrow \text{GRUCell}\Big(\textstyle\sum_i m_{i\to j},\; \mathcal{E}_j\Big)$$

over all `C(C-1)` ordered pairs (complete graph). It never forms the
cross-spectrum.

**Result (2026-08-31, full 6-fold): mean AP 0.173** (per-fold
0.150 / 0.190 / 0.180 / 0.185 / 0.189 / 0.147), ROC-AUC 0.878 — dead flat
every fold regardless of how easy that seizure is for `pre` (e.g. `pre`
scores 0.985 on fold 1_16, the hops model 0.185). At or *below* the
no-hops `channel_cwt` on the two folds that overlap. vs a same-machine
`pre` reproduction at 0.644.

So a learned interaction of per-channel *summaries* — even a full
253-edge message-passing graph — cannot substitute for measuring
`\arg\Gamma_{ij}`. The cross-channel phase structure is not recoverable
from the per-channel magnitude spectra: **forming the off-diagonal of
`Γ` (the cross-spectrum `S_{ij}=w_i\bar w_j` and its phase) is the entire
signal.** On the ~30-preictal-window budget the extra hop parameters
marginally hurt.
