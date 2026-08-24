# Session notes — Mamba temporal backend for the dense-edge GNN (2026-08-24)

Branch: `mamba-temporal-edge-model` (off `main` @ `6da7d37`). Repo:
`C:\Users\User\Documents\noshore5\EEG_Benchmarks`. Machine: local Windows
11, NVIDIA GeForce RTX 3070 Ti (8GB, driver 591.86, CUDA 13.1), `torch
2.8.0+cu128`. Subject chb01.

Adds `dense_edge_temporal_mode="mamba"` as a **third**, interchangeable
`dense_edge_conv` backend alongside the existing `"conv"`
(`_build_dense_feature_conv`, Conv2d+pool) and `"rnn"`
(`_DenseEdgeGRUTemporal`, per-edge GRU). Purely additive — neither
existing backend was touched; both still construct and run exactly as
before (verified: `test_gru_and_conv_backends_unaffected_by_mamba_addition`,
and the full pre-existing `tests/test_tf_node_encoder.py` suite still
passes unchanged bar one hardcoded pipeline-set assertion that legitimately
needed a third entry).

This is a **temporal-backend ablation**, not a new architecture: same
dense-edge/GNN pipeline, same graph, same training protocol, same dataset —
only the module that pools each edge's `[C_in, T]` coherence/phase/
significance sequence down to one `[out_channels]` vector per edge changes.

---

## 1–9. Where the sequence enters the existing temporal block

Read before writing anything (`Epilepsy/pipelines/cwt_gnn_classifiers.py`):

- `SparseEvidenceGNNCore._dense_edge_features` (event_mode="dense") builds
  `conv_in = dense_edge_raw.permute(0, 1, 4, 2, 3).reshape(B, 4*nfreqs, E, T)`
  from the precomputed, non-trainable `[B, 4, E, T, F]` coherence/sinφ/
  cosφ/significance stack (`_build_dense_edge_input`), then calls
  `self.dense_edge_conv(conv_in)`.
- **This is exactly where the GRU currently receives its input** — not a
  sequence of pooled graph embeddings, not pooled node features. It is a
  **per-edge, per-timestep raw feature sequence**, frequency already
  folded into the channel axis, computed *before* any GNN message passing
  runs. `_DenseEdgeGRUTemporal.forward` reshapes `[B, C_in, E, T] ->
  [B*E, T, C_in]` (edges folded into the batch dim, one `nn.GRU` instance
  with SHARED weights across every edge), takes the GRU's final hidden
  state `h_T` as the "whole sequence, summarized" output, and reshapes
  back to `[B, out_channels, E, 1]`.
- Downstream of `dense_edge_conv`, the resulting per-edge vector becomes
  `events_padded` and flows into the **existing, unmodified** GNN message-
  passing / aggregation / classifier head (`sparse_message_mlp`,
  `_aggregate_events`, `sparse_classifier`) — identical for all three
  `dense_edge_temporal_mode` values.

This does **not** match the prompt's assumed "GNN → graph representation
per timestep → Mamba" architecture. The actual GNN message passing runs
**once**, on the temporal block's *already-pooled* per-edge output — time
is resolved entirely inside `dense_edge_conv`, upstream of the graph. Per
the task's own instruction ("preserve the existing representation, don't
assume the prompt's architecture"), Mamba was slotted into the exact same
per-edge position the GRU already occupies — the `_DenseEdgeGRUTemporal`
docstring calls this out explicitly, and `_DenseEdgeMambaTemporal`'s
docstring cross-references it.

Tensor shape at that point, at this file's shared defaults
(`nfreqs=8`, `dense_edge_time_downsample=16`, 30s/256Hz window,
`channel_subset_k=4` → still full `E=253` canonical edges,
`_dense_edge_features` scatters into the full mesh regardless of k):
`conv_in = [B, 32, 253, 480]` (`C_in=4*nfreqs=32`, `T'=7680/16=480`) →
reshaped to `[B*253, 480, 32]` before entering either backend. Confirmed
live at `batch_size=32`: `SparseEvidenceGNN dimensions B=32 T=7680` printed
by `configure_summary_context`, and `dense_edge_conv.in_proj.weight:
shape=(16, 32)` in the parameter dump (`in_proj: 32 -> mamba_d_model=16`).

---

## Mamba implementation

**`mamba-ssm`** (official/upstream, `state-spaces/mamba`) was evaluated
first, per the task, and **rejected for this environment**:

- PyPI (`pypi.org/pypi/mamba-ssm/json`, checked live) ships only a source
  distribution (`mamba_ssm-2.3.2.post1.tar.gz`) — **no Windows wheel at
  all**, any Python/torch/CUDA version.
- Its selective-scan/causal-conv1d kernels are custom CUDA extensions
  requiring `nvcc` + a matching Linux build toolchain to compile from
  source. This repo runs on Windows (see `setup.sh` / `requirements.txt`'s
  own `cu128` extra-index-url comment about Windows torch-wheel gotchas)
  and — per the task's explicit constraint — must not break the existing
  MPS path (the author's macOS dev machine, see every prior session note
  under this folder). `mamba-ssm`'s CUDA-only kernels cannot run on MPS or
  CPU at all, regardless of build success.
- Attempting an nvcc build on Windows was judged not worth pursuing:
  high risk of a fragile, hours-long toolchain fight for a first, modest
  ablation, and it would still leave MPS unsupported either way.

**Chosen alternative: [`mambapy`](https://github.com/alxndrTL/mamba.py)**
(PyPI `mambapy==1.2.0`, added to `requirements.txt` with a comment
explaining this decision) — a pure-PyTorch reimplementation of the same
selective-SSM recurrence (parallel-scan by default; this integration
forces `use_cuda=False` in every `MambaConfig` so it *never* reaches for
`mamba-ssm`'s CUDA kernels even if that package is separately installed).
Only dependency: `torch` (already pinned). `py3-none-any` wheel — no
platform-specific build step, confirmed installs and imports cleanly on
this Windows/CUDA machine (`from mambapy.mamba import Mamba, MambaConfig`
→ `mambapy OK`). This is the "least invasive alternative" the task asked
for when the standard package doesn't fit the environment, not a
from-scratch reimplementation of the selective-scan math.

---

## Architecture: `_DenseEdgeMambaTemporal`

`Epilepsy/pipelines/cwt_gnn_classifiers.py` — new `nn.Module`, same file
(alongside `_DenseEdgeGRUTemporal`/`_build_dense_feature_conv`, all three
gated by `SparseEvidenceGNNCore.__init__`'s `dense_edge_temporal_mode`
param, now `Literal["conv", "rnn", "mamba"]`).

```
conv_in [B, C_in, E, T]                      (C_in = 4*nfreqs, unchanged input)
  -> reshape [B*E, T, C_in]                  (edges folded into batch, GRU's own convention)
  -> nn.Linear(C_in, d_model)  [in_proj]     (Identity if C_in == d_model)
  -> Mamba(MambaConfig(...))   [mambapy]     (n_layers stacked selective-SSM blocks)
  -> take last timestep [B*E, d_model]       (causal -> already summarizes 0..T, same
                                               "final state" role as GRU's h_T)
  -> nn.Dropout (mamba_dropout, default 0.0 = no-op; MambaConfig has no
     native dropout field)
  -> nn.Linear(d_model, out_channels) [out_proj]
  -> reshape/permute -> [B, out_channels, E, 1]   (identical contract every backend shares)
```

Everything downstream (`_dense_edge_features`'s squeeze/permute,
`sparse_message_mlp`, aggregation, hops, `sparse_classifier`) is
**unmodified, shared code** across all three `dense_edge_temporal_mode`
values — confirmed by `dense_edge_conv`-agnostic tests passing for
`"conv"`/`"rnn"`/`"mamba"` with the same assertions
(`test_all_three_dense_edge_backends_share_the_same_contract`).

### Hyperparameters (new, explicit — not hardcoded)

Added to both `SparseEvidenceGNNCore.__init__` and
`SparseEvidenceGNNClassifier.__init__` (forwarded through `_build_model`,
same pattern every other dense-family param already uses):

| param | default | meaning |
|---|---|---|
| `mamba_d_model` | 16 | Mamba's own working width — independent of `C_in` (32) and `out_channels` (`dense_conv_out_channels`, 8); projected in/out at the boundaries the same way Conv2d's first/last layers change channel width. |
| `mamba_d_state` | 16 | SSM state dimension `N`. |
| `mamba_d_conv` | 4 | Mamba's local causal Conv1d kernel width (mambapy default). |
| `mamba_expand` | 2 | Inner-width expansion factor (`mambapy`'s `expand_factor`); inner dim = `d_model*expand` = 32. |
| `mamba_n_layers` | 1 | Number of stacked Mamba blocks — deliberately 1 for this first, modest experiment. |
| `mamba_dropout` | 0.0 | Applied externally (see architecture above); 0.0 = bit-identical to no dropout. |
| `mamba_chunk_size` | 128 | **Memory tactic only, not an architecture change** — see below. |

`learning_rate`/`batch_size`/`epochs`/`validation_split`/
`early_stopping_patience` etc. are **untouched**: `run_pipelines.py`'s new
`DENSE_EDGE_MAMBA_PARAMS`/`PREDICTION_MAMBA_PARAMS` dicts start as an
**exact copy** of `DENSE_EDGE_GRU_PARAMS`/`PREDICTION_GRU_PARAMS`'s numbers
(same `_SHARED_ARCH_PARAMS` base, same `batch_size`, same `learning_rate`),
only `dense_edge_temporal_mode="mamba"` (+ the `mamba_*` block above)
differs — so any accuracy difference against GRU is attributable to the
temporal backend alone, per the task's fairness requirement. Dataset,
SPH/SOP, postictal exclusion, sampling, subject splits, leakage
protections, preprocessing, CWT frequencies, coherence threshold, graph
construction, and the prediction target are all **completely untouched**
— no line in `datasets/`, `paradigms/`, or the dense-edge feature-building
code (`_build_dense_edge_input`, `dense_edge_cache.py`) was modified.

### `mamba_chunk_size`: the one implementation issue found during smoke testing

First smoke attempt (`mamba_chunk_size` didn't exist yet, i.e. running the
whole `B*E` batch through `mambapy` in one call) hit:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.40 GiB.
GPU 0 has a total capacity of 8.00 GiB ...
```

Root cause, traced into `mambapy/mamba.py`'s `ssm()`: **both** the
`pscan=True` (parallel-scan) and `pscan=False` (sequential-loop) code
paths materialize three dense `[rows, T, d_model*expand, d_state]` float
tensors (`deltaA`, `deltaB`, `BX`) **before** either scan strategy even
runs — so switching `pscan` off does not help, the allocation happens in
shared setup code. `rows` here is this class's flattened `B*E` — and `E`
is the **full canonical edge count** (253 for the 23-channel mesh, *not*
reduced by `channel_subset_k` — every edge always "fires" in
`event_mode="dense"`, see `_dense_edge_features`'s docstring), so at
`batch_size=32` that's `32*253=8096` rows. At this file's shared defaults
(`T'=480`, `d_model*expand=32`, `d_state=16`), one intermediate tensor
alone needs `8096*480*32*16*4 bytes ≈ 7.96GB` — confirmed by the OOM's
exact "Tried to allocate 7.40 GiB" (close, matches the actual `A_log`
initialization skewing the number slightly) on an 8GB card, before even
counting the other two same-sized tensors or `hs`.

This is a real, structural cost of a pure-PyTorch selective scan (memory
scales with `rows*T*d_state` directly) — unlike Conv2d (bounded local
receptive field) or `nn.GRU` (cuDNN's fused kernel never materializes a
dense `[rows, T, ...]` tensor). It is **not** fixable by picking different
Mamba hyperparameters without changing the architecture (`d_state`/
`expand` only scale the constant, not the `rows*T` factor that's the real
problem here), and per the task's constraint ("do not change the data
representation to make Mamba work"), `E`/`T'`/`batch_size` were left
untouched.

**Fix**: `mamba_chunk_size` splits `seq`'s `B*E` leading dim into groups
of at most `mamba_chunk_size` rows, runs `self.mamba` on each group
separately, concatenates — same "chunking is a memory tactic, not a
numerical change" contract `CWTTimeFrequencyNodeEncoder`'s own
`chunk_size` already establishes elsewhere in this file (mirrored
directly, including gradient-checkpointing each chunk during training via
the same `torch.utils.checkpoint.checkpoint` import). Verified bit-close
(not just "doesn't crash"): `test_mamba_temporal_chunking_matches_full_batch`
checks a `chunk_size=2` run against a `chunk_size=4096` (effectively
unchunked) run on identical weights, `atol=1e-5`. Default 128 keeps each
of the three intermediate tensors under ~250MB at the shared defaults —
confirmed: the 8GB-card OOM was fully resolved, smoke run below completed
end to end with no OOM.

---

## Sequence-handling assertions/logging

- `_DenseEdgeMambaTemporal.forward` asserts `conv_in.shape[1] ==
  self.in_channels` before doing anything else (catches a `C_in` mismatch
  as a clear `AssertionError` at the temporal block itself, not a cryptic
  shape error three layers downstream).
- The existing `SparseEvidenceGNN dimensions B=... T=...` log line
  (`configure_summary_context`) and `print_custom_summary`'s config dump
  (extended to print `mamba_d_model=... mamba_d_state=... mamba_d_conv=...
  mamba_expand=... mamba_n_layers=... mamba_dropout=... mamba_chunk_size=...`
  only when `dense_edge_temporal_mode="mamba"`) already surface batch
  size / sequence length / feature dimension once per run — reused rather
  than duplicated.
- The sequence is `[B, C_in, E, T]` on entry and reshaped to `[B*E, T,
  C_in]` (batch, sequence length, feature dim) explicitly in code with an
  inline comment, matching the GRU path's own convention — this repo's
  dense-edge pipeline only ever uses this one convention (never `[T, B,
  F]`), so there was no implicit-vs-explicit ambiguity to resolve, just to
  document (done, mirroring `_DenseEdgeGRUTemporal`'s own comment style).

---

## Tests

New: `tests/test_dense_edge_mamba.py` (23 tests, mirrors
`tests/test_tf_node_encoder.py`'s synthetic-data-only, no-CHB-MIT-download
convention — same file the task's "integrate into existing model smoke
tests" instruction pointed at). Covers every category the task listed:

1. **Construction**: `test_mamba_temporal_module_constructs`,
   `..._projects_when_dims_differ`, `..._identity_projection_when_dims_match`,
   `..._dropout_is_noop_by_default`.
2. **Forward pass**: `test_core_forward_with_mamba_backend_end_to_end`.
3. **Tensor shape**: `test_mamba_temporal_forward_shape_across_sequence_lengths`.
4. **Different sequence lengths**: same test, parametrized `n_time in
   [4, 8, 16]`.
5. **Batch size > 1**: `test_mamba_temporal_forward_batch_sizes`,
   parametrized `batch_size in [1, 2, 5]`.
6. **Gradient propagation**: `test_mamba_temporal_gradients_flow` (module
   level) and `test_core_forward_with_mamba_backend_end_to_end` (full
   `SparseEvidenceGNNCore.forward` -> loss -> backward, checks
   `dense_edge_conv`'s own parameters receive nonzero gradient).
7. **Existing GRU/Conv backends still work unchanged**:
   `test_gru_and_conv_backends_unaffected_by_mamba_addition` (construction
   type check) + `test_all_three_dense_edge_backends_share_the_same_contract`
   (parametrized `"conv"`/`"rnn"`/`"mamba"`, same forward-pass contract) +
   full pre-existing `tests/test_tf_node_encoder.py` suite re-run
   unchanged (17/18 passed as-is; the 18th hardcoded the exact 3-pipeline
   `--pipeline` choice set and legitimately needed `"dense_edge_mamba"`
   added — same treatment any new pipeline addition would need).

Plus: input-validation tests (`test_invalid_dense_edge_temporal_mode_rejected`,
`test_mamba_mode_requires_event_mode_dense`,
`test_shuffle_time_order_rejected_for_mamba`), hyperparameter-forwarding
(`test_classifier_forwards_mamba_hyperparameters`), the chunking-numerical-
equivalence test above, and CLI/config-convention wiring
(`test_dense_edge_mamba_is_a_pipeline_choice` — checks `--pipeline` choices,
`_dense_family_params`/`_dense_family_result_dir`, and that
`DENSE_EDGE_MAMBA_PARAMS` matches `DENSE_EDGE_GRU_PARAMS` on every shared
training-dynamics key).

**Result**: `42 passed` (`tests/test_tf_node_encoder.py` +
`tests/test_dense_edge_mamba.py`), and `42 passed, 3 deselected` for the
repo's full non-slow suite (`pytest tests/ -m "not slow"` — the 3
deselected are `test_chb_mit_continuous_labeling.py`'s real-download
tests, unrelated to this change and already marked `slow` before this
session).

---

## Config wiring (repository's existing conventions, not a parallel system)

Followed the exact `"dense_edge"` vs `"dense_edge_gru"` pattern
`run_pipelines.py` already uses — added `"dense_edge_mamba"` as a **third**
`--pipeline` choice (not a new flag/config system):

- `DENSE_EDGE_MAMBA_PARAMS` (detection) / `PREDICTION_MAMBA_PARAMS`
  (prediction) — exact copies of the GRU dicts + `dense_edge_temporal_mode
  ="mamba"` + the `mamba_*` block, same "four independent copies, pipeline
  × label-mode" precedent `_SHARED_ARCH_PARAMS`'s own comment describes.
- `_dense_family_params("dense_edge_mamba", label_mode)` and
  `_dense_family_result_dir(..., "dense_edge_mamba", ...)` — results land
  under `results/dense_edge_mamba/` (never pooled with `dense_edge`'s or
  `dense_edge_gru`'s CSVs, same reasoning as the existing two).
- `smoke_test.py`'s `--pipeline` choices extended the same way.
- `--cwt-encoder` (the CWT time-frequency node encoder) works on
  `dense_edge_mamba` too, for free — it only checks `event_mode`, not
  `dense_edge_temporal_mode`, so no extra wiring was needed there.

---

## Smoke-test results

Reused `smoke_test.py` unchanged (1 fold, subject chb01, `label_mode=
prediction`, `window_length=30s`, `channel_subset_k=4`,
`max_interictal_recordings=5`, `batch_size=32`, `validation_split=0.2`,
`early_stopping_patience=5`, `seed=42`, `device=cuda`, `dense_edge_amp_bf16
=True`, `train_amp_bf16=True`). `total_params=13571` (GRU equivalent:
`10539` — Mamba's extra params are the `in_proj`/`out_proj`/SSM
parameters vs. GRU's single gate matrix).

### Mamba (`--pipeline dense_edge_mamba`)

| epoch | loss | train roc_auc | val_loss | val roc_auc | epoch_s |
|---|---|---|---|---|---|
| 1 | 0.8644 | 0.4798 | 0.6838 | 0.7795 | 42.69 |
| 2 | 0.6565 | 0.6543 | 0.5579 | 0.8165 | 41.95 |
| 3 | 0.4635 | 0.8578 | 0.6060 | 0.8244 | 42.77 |
| 4 | 0.3877 | 0.8949 | 0.5387 | 0.8182 | 42.76 |
| 5 | 0.3705 | 0.9050 | 0.4664 | 0.8355 | 43.24 |
| 6 | 0.3475 | 0.9119 | 0.4420 | 0.8518 | 43.28 |

`mean epoch_time=42.78s`, restored best (epoch 6, `val_loss=0.442`). Held-
out fold (seizure `1_02_0`, seed 42): `F1=0.688 AP=0.618 AUC=0.927
precision=0.647 recall=0.733`, `FAR raw=12.0/h -> smoothed=0.0/h`, hit=True
(raw and smoothed). Total 1-fold wall time: **264.03s**. No NaNs/Infs at
any point; loss monotonically decreasing; `dense_edge_conv`'s own
parameters confirmed nonzero-gradient (both via the unit test and by the
loss actually moving); ran end to end on CUDA.

### GRU (`--pipeline dense_edge_gru`), identical params — direct comparison run

| epoch | loss | train roc_auc | val_loss | val roc_auc | epoch_s |
|---|---|---|---|---|---|
| 1 | 0.8694 | 0.4907 | 0.6893 | 0.8254 | 3.34 |
| 2 | 0.6715 | 0.6313 | 0.6152 | 0.8240 | 2.92 |
| 3 | 0.5534 | 0.8213 | 0.5264 | 0.8270 | 2.90 |
| 4 | 0.4115 | 0.8883 | 0.4923 | 0.8382 | 2.88 |
| 5 | 0.3722 | 0.9038 | 0.4610 | 0.8481 | 2.91 |
| 6 | 0.3662 | 0.9023 | 0.4523 | 0.8592 | 2.87 |

`mean epoch_time=2.97s`. Same fold: `F1=0.697 AP=0.663 AUC=0.932
precision=0.639 recall=0.767`, `FAR raw=13.0/h -> smoothed=0.0/h`. Total
1-fold wall time: **21.16s**.

**Mamba is ~14.4x slower per epoch (42.78s vs. 2.97s) and ~12.5x slower
wall-clock overall (264s vs. 21s) at this smoke scale**, on this 1-fold/
6-epoch/`batch_size=32` config — this is the direct cost of
`mamba_chunk_size=128`'s ~63 sequential Python-loop chunks per forward
(each running mambapy's pure-PyTorch selective scan, then recomputed
again for backward via gradient checkpointing), vs. GRU's single fused
`nn.GRU`/cuDNN call over the whole `8096`-row batch. Prediction quality is
essentially indistinguishable at this trivial 1-fold/6-epoch scale (both
`F1≈0.69`, `AUC≈0.93`) — **not a meaningful accuracy comparison**, just
confirmation both backends train and converge similarly on the same tiny
slice. A real accuracy ablation needs the full leave-one-seizure-out
6-fold run, deliberately not launched this session per the task's own
instruction not to auto-launch a large run.

### Exact command to reproduce

```
.venv\Scripts\python.exe Epilepsy\smoke_test.py --pipeline dense_edge_mamba --device cuda --epochs 6
.venv\Scripts\python.exe Epilepsy\smoke_test.py --pipeline dense_edge_gru --device cuda --epochs 6   # GRU comparison run
```

(`PARAMS` in `smoke_test.py` supplies everything else — `label_mode`,
`subjects`, `window_length`, `channel_subset_k`, etc. — unchanged from
the file's existing defaults; only `--pipeline`/`--device`/`--epochs`
were overridden via its existing CLI-override layer.)

---

## Compatibility issues encountered (summary)

1. **`mamba-ssm` incompatible with this environment** — no Windows wheel,
   requires CUDA-kernel compilation, would break the MPS path. Resolved
   by using `mambapy` instead (see "Mamba implementation" above). This is
   a documented decision, not a silent substitution.
2. **CUDA OOM at default (unchunked) batch** — `mambapy`'s selective scan
   materializes `O(B*E*T*d_state)` intermediate tensors regardless of
   `pscan`; `B*E=8096` at this pipeline's shared defaults made that ~24GB
   on an 8GB card. Resolved with `mamba_chunk_size` (memory tactic,
   verified numerically equivalent — see Tests above), **not** by
   shrinking `E`/`T'`/`batch_size` or otherwise touching the shared data
   representation.
3. **~14x slower per epoch than GRU** at this smoke scale — a direct,
   understood consequence of #2's chunking (Python-loop overhead +
   gradient-checkpoint recompute), not evidence about Mamba's
   architectural merit. Flagged here for human review: whether to (a)
   accept this cost for a real 6-fold run, (b) raise `mamba_chunk_size` on
   a bigger GPU if one becomes available, or (c) revisit `mamba-ssm`'s
   CUDA kernels if this ever moves to a Linux pod (the RunPod deployment
   path several other session notes already use) — none of these were
   pursued this session; only documented as options.

No evaluation-protocol or data-representation changes were made to
accommodate Mamba, per the task's explicit constraint — if a real 6-fold
run shows Mamba losing to GRU/conv, that is intended to stand as a valid
result, not be tuned away.

---

## What this does and doesn't show

Do:

- `dense_edge_temporal_mode="mamba"` constructs, runs forward/backward,
  and trains end to end on CUDA with no NaNs/Infs, converging similarly
  to GRU on one smoke fold.
- The three `dense_edge_conv` backends are cleanly interchangeable behind
  one `Literal["conv", "rnn", "mamba"]` switch, sharing every line of code
  downstream and matching training dynamics (`DENSE_EDGE_MAMBA_PARAMS` ==
  `DENSE_EDGE_GRU_PARAMS` on every non-temporal-mode key).
- `mamba-ssm` vs. `mambapy` is a documented, reasoned dependency decision,
  not an unexplained substitution.

Don't:

- **Not an accuracy ablation.** One fold, six epochs, `channel_subset_k=4`
  smoke scale — this session deliberately did not launch the real 6-fold
  comparison (matching the task's "do not launch a large run
  automatically" instruction). The GRU numbers above exist only to
  confirm both backends behave sanely on identical data, not to claim
  Mamba is competitive or not.
- **Not a full-hour/continuous-CWT experiment.** This still uses the
  existing fixed-window dense-edge pipeline exactly as `dense_edge_gru`/
  `dense_edge` do — no continuous full-record CWT/edge-feature
  infrastructure was added or assumed, per the task's explicit scope
  limit.
- The ~14x epoch-time gap is real and should inform whether/how a 6-fold
  Mamba run gets launched next, not be hidden or optimized away silently.
