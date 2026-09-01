# Mamba-3 complex-diagonal SSM backend + per-freq temporal mode (2026-08-29)

**TL;DR:** built `Epilepsy/pipelines/mamba3.py` -- a complex-diagonal
selective SSM ("Mamba-3" style, state eigenvalue `lambda = -exp(a) + i*omega`)
as a drop-in temporal backend for `hermitian_ssm`, so the temporal model
can integrate the coherence *phase* over time instead of a real-diagonal
SSM that can only accumulate/decay. Wired as `mamba_backend="mamba3"`,
made it the `HERMITIAN_SSM_PARAMS` default. Also built + verified +
**parked** `temporal_mode="per_freq"` (one weight-shared Mamba lane per
frequency; not a bug, just ~8x slower with no epoch-1 advantage). All in
commit `7d672e1` on `main` (pushed).

**Result: mamba3 is NEGATIVE** (6-fold mean AP 0.408 vs real-Mamba 0.436).
**Follow-up k=6->k=12 also NEGATIVE** (0.390 vs 0.436). All three hermitian
levers (encoders, mamba3, k axis) now closed negative -- the pipeline's
~0.44 AP ceiling on chb01 is a data-budget wall. Deliverable is
`temporal_graph_aggregate="post"` (0.639). `mamba_backend` reverted to
`"mamba"` in the working tree; `spectral_k` needs to go 12 -> 6 to restore
the 0.436 baseline (cache `9d6ad0d850b8b8f0` still on disk).

Follow-on to `2026_08_29/hermitian_ssm_bandmatch_6fold.md`, whose
conclusion was: encoder-variant search CLOSED NEGATIVE, **only untried
hermitian lever is Mamba-3** (a temporal-model change), low expected value
because the ceiling looks data-budget-bound.

## Why a complex-diagonal SSM

The eigenvector encoder feeds `[Re u_r, Im u_r]` per mode per frequency
per timestep -- a stream of complex numbers whose *phase* carries how the
coherence structure rotates over the window. Mamba-1/2 have a **real**
diagonal state transition `A_r = exp(Delta * (-a_r))`: each state channel
can only decay or accumulate, never rotate. Mamba-3's complex diagonal
`A_r = exp(Delta * (-a_r + i*omega_r))` gives every state channel a decay
rate `exp(-Delta a_r)` **and** an eigen-frequency `omega_r`, so the
recurrence can phase-lock onto an evolving complex relationship. The
selective part (input-dependent `Delta`, `B`, `C`; `C` complex) is kept.

The scan `h_t = A_t h_{t-1} + x_t` is associative over C unchanged, so
mambapy's O(T log T) Blelloch parallel scan works elementwise on complex
tensors as-is. Only the autograd needed fixing.

## `_ComplexPScan` -- the scan and its conjugation-correct backward

mambapy's `PScan.pscan` / `pscan_rev` are pure elementwise mul/add, dtype-
generic -- they run on complex tensors directly. But mambapy's own
`PScan.backward` is **wrong for complex A**: it does not conjugate the A
path. `_ComplexPScan(torch.autograd.Function)` reuses the forward scan and
supplies the correct backward.

Derivation (torch convention: `.grad` = conjugate cotangent; for `y = a*x`
-> `grad_x = conj(a) grad_y`, `grad_a = conj(x) grad_y`):

    h_t = A_t h_{t-1} + x_t
    G_t = grad_H[t] + conj(A_{t+1}) G_{t+1}     (reverse scan; weights conj(A) shifted left by 1)
    grad_x_t = G_t
    grad_A_t = conj(h_{t-1}) G_t                (h_0 = 0  ->  grad_A_1 = 0)

Implementation: forward pads to power-of-2, `transpose(2,1).contiguous()`
to `[B,D,Lp,N]`, `_RealPScan.pscan(A, X)` in place, saves `(A_in, X=H)`.
Backward builds `A_shift = pad(A_t[:,:,1:]).conj().contiguous()`, runs
`_RealPScan.pscan_rev(A_shift, grad)` to get `G`, then
`gradA[:,:,1:] = H[:,:,:-1].conj() * grad[:,:,1:]`.

**Verified 2026-08-29:** `torch.autograd.gradcheck` in float64 passes for
`_ComplexPScan` alone AND the full `_Mamba3Block`, at both power-of-2 and
non-power-of-2 L. Forward matches a naive Python loop to fp32 precision.
Cost ~2.1x a real mambapy pscan fwd+bwd (2060 vs 974 ms/step at B=64,
T=480, d_state=16) -- usable, epoch ~2x "mamba".

## Three scan attempts that did NOT work (before landing on the above)

Per the user's "keep building it" -- the fast+correct scan took three
dead ends:

1. **cumprod form** `h = P * cumsum(u / P)` where `P = cumprod(A)`:
   NaN at chunk_len=240. `P` underflows (|A| <= 1, product over T decays
   past fp32 denormal), division blows up.
2. **plain sequential Python loop over T=480**: correct but >2 min/step
   -- 30x real Mamba, unusable for a 30-epoch 6-fold.
3. **chunked log-space O(L^2) einsum** `[R,L,L,di,ds]`: 39 s/step
   (~30x). Also failed gradcheck -- a hardcoded `.to(torch.complex64)`
   broke fp64 finite-differences, and a `clamp(min=0)` put a
   non-differentiable kink on the diagonal.

Final fix also corrected the block's dtype handling:
`cdt = torch.complex128 if x.dtype == torch.float64 else torch.complex64`
(so gradcheck's fp64 path stays fp64 end to end).

## `_Mamba3Block` / `_Mamba3Temporal`

`_Mamba3Block`: `in_proj (d_model -> 2*d_inner) -> chunk (x, z) -> causal
depthwise conv1d -> silu -> x_proj -> split(dt_rank, d_state, 2*d_state)
-> delta = softplus(dt_proj(dt_lr)); Cm = complex(Cri[:ds], Cri[ds:])
-> y = _scan(delta, Br, Cm, x) -> y = y * silu(z) -> out_proj(y ->
d_model)`. (`out_proj` was a bug fix -- the block first returned d_inner
and broke the residual add.)

`_scan`: `lam = (-exp(A_log) + i*omega)`, `deltaA = exp(delta * lam)`
(|.| <= 1), `u = delta * x * Bm`, `h = _ComplexPScan(deltaA, u)`,
`return (h * Cm).sum(-1).real + D * x`.

Init tuned so the block starts near Mamba-1: `omega ~ 0.01*randn`
(tiny rotation), `A_log = log(arange(1, d_state+1))`, `dt_proj.bias` so
`softplus(bias) in [1e-3, 1e-1]` (mambapy convention).

`_Mamba3Temporal(in_channels, out_channels, *, d_model=64, d_state=16,
d_conv=4, expand=2, n_layers=1, dropout=0.0)`: exact same contract as
`_DenseEdgeMambaTemporal` -- `[B, C_in, E, T] -> [B, out_channels, E, 1]`,
weight-shared across E (folded into the row dim), last-timestep pool.
Drop-in; E can be edges / nodes / frequencies / 1.

## Wiring

- `hermitian_ssm_classifier.py`: new
  `_build_temporal_backend(backend, in_ch, out_ch, *, d_model, d_state,
  d_conv, expand, n_layers, dropout, chunk_size)` -> `_Mamba3Temporal`
  if `backend=="mamba3"` else `_DenseEdgeMambaTemporal`. Both
  `_MambaTemporalHead` and `_PerFreqMambaHead` take `backend="mamba"`
  kwarg and build through it.
- `HermitianSSMClassifier.__init__`: `mamba_backend: str = "mamba"`,
  validated in `("mamba", "mamba3")`. `fit()` threads it to the head.
- `run_pipelines.py` `HERMITIAN_SSM_PARAMS`: `mamba_backend="mamba3"`
  (with rationale comment -- graph-native temporal model, complex
  Blelloch pscan, gradchecked, ~2x real pscan).

## per_freq temporal mode -- BUILT, VERIFIED NOT-A-BUG, PARKED

`temporal_mode="per_freq"`: encoder skips `freq_fuse`, returns
`[B,T,F,d_freq]`; `_PerFreqMambaHead` runs one **weight-shared** Mamba
lane per frequency over T (E=F in the backend), then fuses the F
band-summaries with a `Linear(F*head_hidden, head_hidden)` before the
head. Contrast `"fused"`: encoder fuses F frequencies into one token,
one Mamba over T.

The user's prior read was "the weak epoch-1 val_auc is certainly a bug."
Investigated with `scratchpad/check_perfreq.py`:

- **Not a bug.** epoch-1 val_auc ~0.77 is normal -- the fused eigenvector
  encoder does 0.670 / 0.811 / 0.738 at epoch 1 across folds and still
  produces mean AP 0.436. Grads flow to every parameter, shapes correct.
- Bands are **not** collapsed: `band0 vs band7` feature corr 0.44 with
  `freqs` passed (vs ~1.0 with `freqs=None`), i.e. the per-frequency
  lanes see genuinely different inputs.
- Chunked scan path is bit-identical to the reference.
- **One real defect found + fixed:** `encoder.freq_fuse`
  (`Linear(n_freqs*d_freq, d_model)`, ~33k params) was still being
  built in `fuse_freq=False` mode -- dead weight getting weight decay.
  Now `if fuse_freq: self.freq_fuse = ...` (conditional) in all four
  encoders.

**Why parked:** ~831 s/epoch (~8x "fused"). 8 weight-shared Mamba lanes
through the gradient-checkpointed path, and no epoch-1 metric advantage
to justify it. The band-summary fusion also throws away the per-frequency
structure right before the head anyway. Not deleted -- `temporal_mode`
validates `("fused", "per_freq")` and the head is built.

## 6-fold result (hermitian eigenvector encoder + Mamba-3)

Run: `HERMITIAN_SSM_PARAMS` as committed + `mamba_backend="mamba3"`,
`--device cpu`, 30-epoch budget / patience 5, warm cache
`9d6ad0d850b8b8f0`. Log `scratchpad/herm_m3.log`. ~200-240 s/epoch.

Baseline to beat = the band-match run (same everything, real Mamba):
mean AP **0.436**, ROC-AUC 0.947, per-fold AP
`.254 / .324 / .414 / .279 / .788 / .556`.

| fold | baseline AP (real Mamba) | **mamba3 AP** | note |
|---|---|---|---|
| `1_03_0` | 0.254 | **0.399** | win; early-stop ep 11, val_loss noisy (0.31->1.07->0.45), val_auc held 0.93-0.95 |
| `1_04_0` | 0.324 | **0.342** | marginal win; cleanest curve of the run (val_loss 0.69->0.148 monotone, val_auc 0.987), yet held-out AP barely moved |
| `1_15_0` | 0.414 | **0.401** | ~tie (loss by 0.013); 21 epochs, noisiest curve; smoothed hit=False here (baseline: True, recall 1.0) |
| `1_16_0` | 0.279 | **0.218** | loss by 0.061; healthy curve (val_loss 0.182, val_auc 0.971) again -> weak held-out AP |
| `1_18_0` | 0.788 | **0.719** | loss by 0.069; still the strong fold (precision 1.0, FAR/h 0.0), raw hit / smoothed miss |
| `1_26_0` | 0.556 | **0.371** | loss by 0.185 (worst fold); healthy curve again |
| **mean AP** | **0.436** | **0.408** | **mamba3 loses by 0.028** |
| ROC-AUC | 0.947 | 0.939 | loss |
| hit raw / k-of-n | 6/6 / 6/6 | 6/6 / **4/6** | loss |
| FAR/h sm | 8.0 | 7.15 | ~tie |

**Verdict: mamba3 NEGATIVE.** Mean AP 0.408 vs real-Mamba 0.436, worse
on ROC-AUC and k-of-n hit rate too. Fold 1 (`1_03_0`) was the only real
win (+0.145); folds 3-6 gave it all back, fold 6 worst (-0.185). Per-fold
the two backends trade blows around the same mean with no consistent
edge -- the complex-diagonal state does not help at this data budget.

Fold-by-fold the pattern held every time: mamba3 trained to a *healthy*
internal curve (val_auc 0.94-0.99, best val_loss 0.15-0.31) on all 6
folds -- often cleaner than real Mamba -- and it made no difference to
held-out AP. This is the fragility thesis from
`hermitian_ssm_bandmatch_6fold.md` confirmed once more from the temporal-
model side: the ~30-preictal-window budget caps this, not whether the
recurrence can represent an evolving phase. The "integrate the coherence
PHASE over time" hypothesis is not wrong in principle -- there is just
not enough data here for the extra expressiveness to pay for itself.

CSVs: `Epilepsy/results/hermitian_ssm/prediction/*_20260829-2*.csv`
(mamba3 6-fold). Real-Mamba baseline: `*_20260829-111904.csv`.

## k axis (k=6 -> k=12) -- NEGATIVE

User call 2026-08-30: with both temporal-model levers spent, test the one
untried knob on the axis that gave the only big jump (k=2->6 = +0.18 AP).
Rebuilt the spectral cache at **k=12** (new key `293ac41c6e53675f`,
21 GB), real-Mamba backend, everything else = the 0.436 config. Clean
A/B: k is the only variable.

| fold | k=6 (real Mamba) | **k=12 (real Mamba)** |
|---|---|---|
| `1_03_0` | 0.254 | 0.255 |
| `1_04_0` | 0.324 | 0.254 |
| `1_15_0` | 0.414 | 0.405 |
| `1_16_0` | 0.279 | 0.162 |
| `1_18_0` | 0.788 | 0.712 |
| `1_26_0` | 0.556 | 0.551 |
| **mean AP** | **0.436** | **0.390** |
| ROC-AUC | 0.947 | 0.940 |
| k-of-n hits | 6/6 | 5/6 |

**k=12 loses on 5 of 6 folds** (only `1_03` flat), mean AP -0.046. Two
folds materially worse (`1_16` -0.12, `1_04` -0.07), none better. The
k=12 training curves were also visibly noisier (val_loss bouncing
0.3<->1.9 epoch to epoch vs k=6's steadier descent).

Read: **k=6 is the right truncation.** k=2->6 helped because modes 3-6
carry real coupling structure that `diagonal="zero"` freed from under the
power floor; modes 7-12 are estimation noise on ~30 preictal windows and
the encoder cannot learn to ignore them (mode_id + lambda-embed give it
the machinery in principle, not the data). k=23 is off the table -- the
trend is monotone the wrong way. CSV `*_20260830-005015.csv`.

## Where this leaves the deliverable

All three hermitian levers tried, all NEGATIVE:
- **encoder variants** -- `hermitian_ssm_bandmatch_6fold.md` (eigenvector
  0.436 is the ceiling; projector/graph/evolution worse).
- **`mamba_backend="mamba3"`** -- 0.408 vs 0.436, this note.
- **k axis (k=12)** -- 0.390 vs 0.436, this note.
- (`temporal_mode="per_freq"` -- parked, not a bug, 8x slower, no gain.)

hermitian_ssm's ceiling on chb01 is **~0.44 mean AP** and that is a
data-budget wall, not an architecture one -- every lever that added
representational power (complex state, more modes, richer encoders) came
back flat-to-negative while the internal val metrics stayed healthy. Per
the user's standing call, the deliverable is now
**`temporal_graph_aggregate="post"`** (mean AP 0.639, graph flows through
the temporal model). Real ceiling lever is more CHB-MIT subjects.

Revert note: `HERMITIAN_SSM_PARAMS` in the working tree already has
`mamba_backend` back to `"mamba"` and `spectral_k=12` (the k-axis run
config). To restore the 0.436 baseline: `spectral_k` -> 6 (cache key
reverts to `9d6ad0d850b8b8f0`, still on disk), `mamba_backend` stays
`"mamba"`. These live in `run_pipelines.py` alongside another shell's
`_CG_MAMBANET_SHARED_PARAMS` edit -- commit with `git add -p`.

## Code / repo status

- **Committed + pushed:** `7d672e1` on `main` -- `mamba3.py`,
  `hermitian_ssm_classifier.py` (backend wiring + `fuse_freq` conditional
  + per_freq head), `run_pipelines.py` (`mamba_backend="mamba3"` default
  + 13 lines of rationale), `CONTEXT.md`.
- **Uncommitted (mine):** `CONTEXT.md` -- post-fallback-deliverable
  additions (see `hermitian_ssm_bandmatch_6fold.md` and the message from
  the user: `temporal_graph_aggregate="post"` is the fallback if hermitian
  + mamba3 fail). Committed with this note.
- **Uncommitted (NOT mine -- another shell):** `run_pipelines.py`
  `_CG_MAMBANET_SHARED_PARAMS` capacity cut (12->3 Mamba layers,
  d_state 64->16, grad clip, etc.). Left for the cg_mambanet shell --
  that's the open thread flagged in commit `647ffb8`.
- Scratchpad diagnostics: `check_perfreq.py`, `check_chunk.py`,
  `check_mamba3.py`, `gc2.py` (gradcheck + speed), `herm_m3.log`.

## If mamba3 also fails to beat 0.436

Per the user (2026-08-29): revert the deliverable to
`temporal_graph_aggregate="post"` -- flip it in
`PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS`. "post" runs Mamba over each of
the ~253 edge sequences and aggregates to nodes *after*, so the graph
flows through the temporal model (meets the graph-native bar; "pre" does
not). 6-fold was a near-wash (mean AP 0.639 vs "pre" 0.674, ROC-AUC 0.970
vs 0.94) at ~11x compute. Real ceiling lever is more CHB-MIT subjects,
not architecture.
