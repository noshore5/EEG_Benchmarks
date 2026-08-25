# Session notes — continuous-CWT first step, and a design doc for Mamba's continuous-state paradigm shift (2026-08-25)

Branch: `continuous-cwt` (off `main` @ `d680e28`, which itself is
`6d38573` + the `start_fold` addition — see that commit). Repo:
`C:\Users\User\Documents\noshore5\EEG_Benchmarks`. Machine: local Windows
11, NVIDIA GeForce RTX 3060 (12GB), `torch 2.8.0+cu128`.

Two things live in this branch:

1. **Implemented, validated on real data**: a standalone continuous
   (whole-recording) CWT building block, NOT yet wired into the real
   training pipeline. This is the "first easy change" — additive only,
   nothing existing was touched.
2. **NOT implemented, written up for later**: what it would actually take
   to give `dense_edge_mamba`'s Mamba temporal backend a real continuous
   sequential state across a whole recording instead of resetting every
   30s window. This turned out to be a much bigger, separate paradigm
   shift from (1) — see Part 2 below for why, and what it would require.

Context: this followed directly from comparing `dense_edge_gru` and
`dense_edge_mamba` at `channel_subset_k=23` (see
`Session_notes/2026_08_24/dense_edge_mamba_k23_full_run.md` and this
session's own earlier GRU k=23 fold-1 comparison, logged at
`Epilepsy/gru_k23_fold1_run.log`). The motivating question was whether
Mamba specifically — a sequential state-space model — was being wasted by
the pipeline's per-window-independent training setup. Short answer:
mostly yes, and fixing that is Part 2's scope, not Part 1's.

---

## Part 1 — Continuous CWT: implemented and validated

### What was built

- `Epilepsy/pipelines/continuous_cwt.py` (new file):
  - `compute_continuous_cwt(raw_channels, ...)` — one
    `utils.torch_cwt.transform_batch` call over an ENTIRE recording's raw
    signal (`[n_channels, n_time_full]`), instead of the real pipeline's
    current per-window independent calls
    (`cwt_window_cache.compute_cwt_real_imag_tensors_cached`). Returns
    `(real_full, imag_full, freqs)` at full recording length, same
    `[n_channels, n_time, nfreqs]` axis convention the per-window path's
    `w_real`/`w_imag` already use.
  - `slice_continuous_cwt_window(real_full, imag_full, start_sample, n_samples)`
    — pure slicing, extracts one window's CWT out of the continuous
    tensors. `ContinuousLabelingParadigm`'s existing metadata
    (`subject`, `run`, `window_start`) already identifies exactly which
    recording and absolute sample offset a window needs — no paradigm
    change required to make this work.
  - `continuous_coi_valid_mask(...)` — the absolute-offset counterpart to
    `SparseEvidenceGNNCore._coi_valid_mask` (`cwt_gnn_classifiers.py`),
    which assumes `time_offset=0` and validates against the WINDOW's own
    length. This one takes the window's absolute position within the
    full recording and the recording's full sample count, so an interior
    window (clear of both true edges by more than the widest wavelet's
    support) comes back fully COI-valid. **Not wired into
    `SparseEvidenceGNNCore` yet** — that class's own `_coi_valid_mask`
    is untouched.
- `scripts/continuous_cwt_parity.py` (new file, mirrors the existing
  `scripts/torch_cwt_parity.py` convention): loads a real CHB-MIT
  recording, computes the continuous CWT once, computes several 30s
  windows independently the current real-pipeline way, and compares
  window-slice-of-continuous vs. independent-per-window — pooled, and
  split into each window's own edge region vs. interior.

Neither `cwt_window_cache.py` nor `cwt_gnn_classifiers.py` — the actual
call sites every real classifier uses — were touched. This is a pure
addition; nothing about the existing training path can be affected by
having these files present.

### Real-data validation (chb01_03.edf, full recording)

```
python scripts/continuous_cwt_parity.py --device cpu --n-windows 5
```

`n_channels=23 n_time_full=921600 duration=3600.0s` (a genuine ~1hr
recording). Canonical prediction-mode config: `sampling_rate=256,
lowest=8.0, highest=40.0, nfreqs=8, window_length=30.0`.

5 windows sampled across the recording (including one touching the true
start and one touching the true end):

| window | full max&#124;dre&#124; | edge-region max&#124;dre&#124; | interior max&#124;dre&#124; |
|---|---|---|---|
| 0.0–30.0s (touches recording start) | 17.14 | 17.14 | 0.0803 |
| 892.5–922.5s | 21.71 | 21.71 | 0.0866 |
| 1785.0–1815.0s | 7.64 | 7.64 | 0.0248 |
| 2677.5–2707.5s | 7.57 | 7.57 | 0.0391 |
| 3570.0–3600.0s (touches recording end) | 8.60 | 8.60 | 0.0187 |

Pooled: real/imag Pearson r = 0.9979/0.9979 (pooled over ALL samples,
edges included — lower than the ~0.9998+ seen in the fcwt-vs-torch_cwt
parity note specifically BECAUSE this comparison deliberately includes
the region expected to differ). Pooled edge-region max error = 21.7
(real) / 23.5 (imag); pooled interior max error = 0.087 (real) / 0.127
(imag) — three orders of magnitude smaller, consistent with float32
noise, not a bug.

**This is exactly the predicted result, now measured, not assumed**:
away from any true or arbitrary window edge, per-window-independent CWT
and slice-of-continuous CWT agree to float32 noise. At every window's
own edge, they genuinely diverge — that divergence IS the cone-of-influence
loss the continuous path is designed to eliminate for interior window
boundaries (only the two windows genuinely touching the recording's true
start/end have a REAL edge to lose signal to; the continuous path pays
that same cost there too, correctly — no free lunch at the actual
boundary).

### An honest timing caveat (CPU only, not yet measured on CUDA)

```
continuous CWT (whole recording, one call): 3809.91ms (first/cold run: 6223.40ms)
per-window CWT (5 independent calls, sum):   171.20ms
```

On THIS machine's CPU, computing the continuous CWT over a full 23-channel/
1hr recording is much SLOWER than 5 small independent per-window calls —
not faster. This is a real measurement, not swept under the rug: the
2026-08-20 note's "1hr continuous, nfreqs=8 → 0.79ms" figure
(`Session_notes/2026_08_20/torch_native_cwt_module_and_parity_validation.md`,
Part 8) was measured on a real RTX 4090 pod via CUDA's cuFFT, and — as
far as this note's author can tell from that note's own benchmark
script description — was NOT confirmed at a full 23-channel batch
specifically through `transform_batch` (only through the lower-level
`cwt_torch` call, unclear at what batch width). This CPU number was
measured deliberately on CPU here (not CUDA) to avoid contending with the
`dense_edge_mamba` k=23 5-fold background run using the GPU at the same
time this session — it is NOT a claim that continuous CWT is slow on the
real CUDA deployment target, just an honest flag that the "continuous is
faster" framing from earlier in this session was NEVER actually
re-confirmed at the real 23-channel batch width, on either device, and
should be measured on CUDA (with the GPU otherwise idle) before being
assumed. Per the earlier conversation in this session: speed was never
the primary argument for continuous CWT anyway (CWT was already
established as not the pipeline's bottleneck — dense-edge computation is,
see the 2026-08-20 note's Part 10/11) — the actual argument is the
COI-edge-quality fix validated above, which holds regardless of which
device computes it.

### What's NOT done (still ahead of wiring this in for real)

- `_coi_valid_mask` in `cwt_gnn_classifiers.py` still uses window-relative
  offsets — `continuous_coi_valid_mask` above is the fix but isn't wired
  in.
- No lazy per-recording cache exists yet (the design discussed earlier
  this session: CPU-resident, bounded-size, keyed on `(subject, run,
  cwt_config)`, computed on first access via `dataset.get_data()` scoped
  to one recording) — `compute_continuous_cwt` has to be called explicitly
  by a caller right now, nothing calls it automatically.
- `cwt_window_cache.py`'s real call path
  (`compute_cwt_real_imag_tensors_cached`) is completely unaware this
  module exists.
- A real CUDA timing/memory measurement (see caveat above).
- Gating this behind an opt-in flag (`cwt_continuous=True` or similar,
  matching `cwt_backend`/`dense_edge_temporal_mode`'s existing "revertable
  by construction" precedent) once it IS wired in.

---

## Part 2 — Mamba continuous state: design summary, NOT implemented

This is the write-up requested to leave in this branch for later — a
scoping document, not code. Motivating question from this session: would
continuous CWT give `dense_edge_mamba`'s Mamba backend a cleaner
sequential state over an entire recording, instead of resetting every
30s window arbitrarily?

**Answer: no.** Continuous CWT only changes what VALUES populate a
window's CWT tensor (better edges, per Part 1) — it does not change
where the model's own forward-pass boundary is. Traced directly in
`cwt_gnn_classifiers.py`:

- `SparseEvidenceGNNCore._dense_edge_features` builds
  `conv_in = [B, C_in, E, T]` where **T is one window's own time axis**
  (480 timesteps post-downsample at the canonical config — this window's
  30s, nothing more), from the precomputed dense-edge coherence/phase
  stack.
- `_DenseEdgeMambaTemporal.forward` reshapes that to `[B*E, T, C_in]` and
  runs the SSM scan over T. That is the ENTIRE sequence Mamba ever sees
  in one call. It pools to `[B*E, d_model]` (last timestep) and hands off
  to the — identical across all three `dense_edge_temporal_mode`
  backends, and non-temporal — GNN message-passing/classifier head.
- No state is threaded between calls. Every window is an independent
  `forward()` invocation. `mambapy`'s `Mamba` block starts from zero
  hidden state every single call; nothing in this integration passes a
  previous call's final state in as this call's initial state.
- Training batches are fully shuffled every epoch
  (`_BatchIndexSampler.__iter__`: `torch.randperm(self.n, ...)`) — windows
  from different recordings and different times of the same recording
  land in the same batch in random order.
  `_SequentialBatchSampler` (validation only) is NOT a chronological/
  continuity mechanism — it just means "don't reshuffle the validation
  set" (an implementation detail so lazy feature computation doesn't
  re-materialize the whole val set every epoch), confirmed directly by
  reading it: `for start in range(0, self.n, self.batch_size): yield
  list(range(start, ...))` — plain index order, nothing about recording
  identity or chronology.
- GRU (`_DenseEdgeGRUTemporal`) has the IDENTICAL structure and the
  IDENTICAL per-window state reset — this isn't a Mamba-specific gap in
  today's pipeline, it's shared by every current temporal backend. Mamba
  isn't uniquely disadvantaged today; it's just the backend whose whole
  architectural point (long-range sequential state) is currently unused
  by design.

### What a real fix would require (not started, not scoped in detail — this is a first-pass inventory of what would need to move together)

1. **Redefine what a training example is.** Today: one window → one
   independent label → one independent loss
   (`leave_one_seizure_out_prediction`'s per-window
   precision/recall/f1/roc_auc). A state-carrying design needs a
   RECORDING's windows to become a dependent SEQUENCE of examples, closer
   to truncated backprop-through-time (BPTT) in language/audio modeling
   than to the current i.i.d.-window classifier.
2. **Remove/restructure the training shuffle.** `_BatchIndexSampler`'s
   per-epoch `randperm` is precisely what breaks state continuity — a
   state-carrying model needs one recording's windows processed in
   chronological order, unshuffled, with each window's Mamba call
   receiving the PREVIOUS window's final SSM state as its initial state.
   Cross-RECORDING parallelism (each recording's chain is independent)
   could still exist; cross-window parallelism WITHIN one recording
   could not, at least not without a chunked/truncated-BPTT compromise.
3. **Decide what happens at label-structure gaps.** SPH/SOP excludes
   windows near seizure onset; `postictal_buffer` excludes windows after
   offset (`paradigms/continuous_labeling.py`'s `_label_windows_prediction`).
   These are real temporal discontinuities in the window sequence a
   state-carrying model would traverse — does state reset at an excluded
   gap? Carry through it? This needs an explicit answer, not an implicit
   default.
4. **Interface change to `_DenseEdgeMambaTemporal`.** `forward()` would
   need to accept an optional incoming state and return its final state,
   not just run start-to-finish internally every call. `mambapy`'s public
   `Mamba`/`MambaConfig` API would need to be checked for whether it
   exposes/accepts recurrent state across separate calls at all (not
   confirmed either way this session — not investigated, since this is a
   scoping note, not an implementation attempt).
5. **Leave-one-seizure-out fold interaction.** Does a held-out
   recording's evaluation get a "cold start" (fairest test of real
   deployment, where the model has no prior context on a genuinely new
   patient/session) or a "warm-up" pass over some leading portion before
   the region actually scored? This is a real modeling-validity choice,
   not a detail — get it wrong and event-level hit/miss numbers stop
   meaning what they currently mean.
6. **Throughput characteristics change.** Today, all of a fold's training
   windows can be shuffled into arbitrary batches and trained with full
   batch-parallelism. A state-carrying model forces at least partial
   sequential dependency within each recording — cross-recording
   parallelism remains available, but per-recording throughput would
   look more like RNN/SSM sequence training than the current
   embarrassingly-parallel-over-windows setup.

None of this is started. This section exists so the next session (or
this one, later) doesn't have to re-derive "why doesn't continuous CWT
just fix this" or re-trace the `forward()`/sampler code to find the
actual reset point — that's now done and recorded here.

---

## Reproduce Part 1

```powershell
git checkout continuous-cwt
.venv\Scripts\python.exe scripts\continuous_cwt_parity.py --device cpu --n-windows 5
```

`--device cuda` once the GPU is free of other work, to get a real (not
CPU-only) timing number for the honest caveat above.
