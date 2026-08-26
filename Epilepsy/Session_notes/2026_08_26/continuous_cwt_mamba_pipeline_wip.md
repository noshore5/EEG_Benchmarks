# continuous-cwt-mamba: wired into run_pipelines.py -- WIP, one known bug blocking a real run

Executes CONTEXT.md's "Open threads" items 2+3 for the continuous-cwt-mamba
paradigm (whole-recording CWT loading path + a LOSO loop). **Not fully
working yet** -- see "Known bug" below before trusting any run of
`--pipeline continuous_cwt_mamba`.

## What's done and verified

**Phase A -- whole-recording chunked CWT/dense-edge feature extractor**
(`Epilepsy/pipelines/continuous_dense_edge.py`, new). Chunks a whole
recording in TIME (not just rows), pad-and-trim design reusing the
existing per-window `compute_dense_edge_input` pipeline unmodified --
see that module's docstring for the full correctness argument. Verified:

- `scripts/continuous_cwt_chunk_parity.py`: chunked-with-padding vs.
  one-shot CWT, both downsample=1 and downsample=4. Found and fixed a
  REAL bug along the way (not just numerical noise): for
  `dense_edge_time_downsample > 1`, each chunk's average-pooling grid was
  phase-shifted from the reference's own grid unless `pad` is rounded up
  to a multiple of `downsample` -- fixed, both cases now pass (~1.5-1.8e-2
  max diff, explained/bounded, see that script's comments for why it's
  not float32-noise-tight: two genuine, small, explicable sources --
  buffer-boundary FFT truncation tail and per-chunk-vs-one-shot FFT-length
  discretization differences in the frequency-domain Gaussian filter).
- `scripts/continuous_cwt_scale_probe.py`: real scale (23ch, E=253,
  nfreqs=48), RTX 3070 Ti. T_chunk=128 measured ~514ms/chunk, ~4GB peak
  (safe). T_chunk=512 measured ~8.75GB peak (unsafe on this 8GB card).
  T_chunk=256 measured an anomalous ~20x slowdown (~10.9s/chunk) -- not
  investigated, just avoid that value. `CONTINUOUS_CWT_MAMBA_PARAMS`
  defaults to `t_chunk=128`.

**Phase B -- recording-preserving data loading**
(`paradigms/continuous_labeling.py`'s new `get_continuous_data()` method).
Reuses `_window_starts`/`_label_windows[_prediction]` unchanged; `get_data()`
itself is untouched. Verified: `scripts/continuous_labeling_get_continuous_
data_parity.py` -- exact match against `get_data()` (bounds, labels, raw
content) for both label modes, on a synthetic recording.

**Phase C -- classifier + pipeline wiring** (`Epilepsy/pipelines/
continuous_cwt_mamba_classifier.py`, new; `run_pipelines.py` wiring:
`--pipeline continuous_cwt_mamba`, `CONTINUOUS_CWT_MAMBA_PARAMS`,
`leave_one_seizure_out_continuous_mamba`, `--continuous-mamba-t-chunk`/
`--continuous-mamba-scan`). `ContinuousCWTMambaClassifier` cannot subclass
`SparseEvidenceGNNClassifier`'s own `fit()` (random-batch, incompatible
with state-carrying) -- reuses its constructor/`_build_model` scaffold and
the existing readout tail (`_dense_edge_features_from_conv_out`,
`sparse_message_mlp`, `_aggregate_events`/`_propagate_hops`,
`sparse_classifier`) unmodified via a new `_continuous_logits` method.

`scripts/continuous_cwt_mamba_classifier_smoke.py` (tiny synthetic
recordings, CPU, `t_chunk=16`) passes: construct, fit 2 epochs, predict,
softmax rows sum to 1.

## Known bug -- blocks a real run, NOT yet root-caused

**Found during this session**, in order:

1. First real smoke run (`--pipeline continuous_cwt_mamba --smoke
   --continuous-mamba-t-chunk 64 --max-folds 1 --device cuda`) hit a real
   CUDA OOM during TRAINING: "13.60 GiB is allocated by PyTorch" on an 8GB
   card. Root cause understood and fixed: `fit()`'s original training loop
   forwarded an ENTIRE recording's chunks (accumulating every chunk's live
   autograd graph) before a single `loss.backward()` at the end -- exactly
   what `_DenseEdgeMambaContinuous`'s own `cache.detach()` TBPTT
   truncation exists to prevent, just not applied at the training-loop
   level. Fixed by `_train_recording_incremental` (new method,
   `continuous_cwt_mamba_classifier.py`): processes windows as soon as
   their covering chunks have streamed, `backward()`+`optimizer.step()`
   immediately per ready group, then drops buffered chunk output no
   longer needed by any pending window -- peak memory now bounded by the
   longest window's span in chunks, not the whole recording. This part
   IS fixed -- training completes without OOM.

2. **Rerunning the same smoke command after that fix now gets past
   training, then crashes during VALIDATION** (`_recording_logits_and_
   labels` -> `_stream_recording` -> `torch.cat(pieces, dim=-1)`) with:
   `torch.AcceleratorError: CUDA error: an illegal memory access was
   encountered`. Reproduced twice, including with `CUDA_LAUNCH_BLOCKING=1`
   (which should force synchronous, precisely-attributed CUDA errors) --
   still points at that same `torch.cat` call, which is itself an
   innocuous op, strongly suggesting the actual illegal access happened
   EARLIER (most likely inside `_train_recording_incremental`'s chunk
   streaming/slicing, or possibly `_DenseEdgeMambaContinuous`'s own pscan
   under repeated small-chunk calls) and only got flagged asynchronously
   once validation's next CUDA call ran. **Not yet isolated.**

Next step (was in progress when this session paused for GPU contention
with the user's other work): rerun the identical smoke command with
`--device cpu` instead of `cuda`. CPU execution turns a silent illegal
memory access into a clear, catchable Python exception (out-of-bounds
index, wrong tensor shape, etc.) instead of CUDA's opaque, often
misattributed async error -- the standard next move for this exact
failure mode. If CPU reproduces a clean Python traceback, that pinpoints
the real bug directly (most likely candidate: an off-by-one in
`_train_recording_incremental`'s buffer-trimming/slicing index math, or
in `_sample_to_output_index`'s "past the last chunk" fallback branch --
neither has been stress-tested against a real, many-chunks-per-window
recording yet, only the tiny synthetic smoke test where every window fit
in one or two chunks). If CPU does NOT reproduce it, that points at
something CUDA/mambapy-pscan-specific instead (e.g. repeated small-batch
kernel launches with an oddly-shaped tensor at some chunk).

**Do not trust any `--pipeline continuous_cwt_mamba` run's results until
this is fixed and re-verified** -- Phases A and B are solid; Phase C's
training-loop OOM is fixed; this second bug is real and open.

## CPU repro attempt (2026-08-26, different shell) -- does NOT reproduce

Ran the exact documented next step: `--pipeline continuous_cwt_mamba
--smoke --continuous-mamba-t-chunk 64 --max-folds 1 --device cpu` on a Mac
(no CUDA available here to instead try `CUDA_LAUNCH_BLOCKING=1` on the
original box). **Completed cleanly, exit code 0** -- fit (2 epochs),
validated, predicted, wrote both result CSVs
(`prediction_leave_one_seizure_out_20260826-173317.csv` /
`prediction_per_seizure_20260826-173317.csv`). No crash, no exception, at
any point (including `_stream_recording`'s `torch.cat`, the call the CUDA
error pointed at).

This lands on the branch of the note's own "if CPU does NOT reproduce it"
reasoning above: **rules out a general Python-level off-by-one** in
`_train_recording_incremental`'s buffer-trimming/slicing or
`_sample_to_output_index`'s fallback branch (both ran correctly here,
across a real 12-recording/775-window smoke dataset, all 8 training + 2
validation recordings, without incident) -- if either had a real
index/shape bug it would very likely have thrown a plain Python exception
here too, not just silently produced wrong numbers, given how it manifests
on CUDA (immediate crash, not a quiet miscalculation).

Points the remaining investigation at something CUDA/mambapy-pscan-
specific instead, as the note anticipated -- most likely `mambapy`'s
`pscan` under repeated small-batch/oddly-shaped-tensor kernel launches
across many chunk calls (one call per `t_chunk=64`-sized chunk per
recording, streamed sequentially -- CPU has no equivalent async/kernel-
launch failure mode to hit here). **Not yet re-investigated on CUDA** --
this Mac has none; next actual step needs a CUDA box (RunPod pod, or
whatever machine produced the original crash) rerunning the same smoke
command with `CUDA_LAUNCH_BLOCKING=1 --device cuda` and, if that still
just points at `torch.cat`, adding a `torch.cuda.synchronize()` after
every per-chunk `_DenseEdgeMambaContinuous` call inside the streaming loop
to force each chunk's CUDA errors to surface synchronously and pin down
exactly which chunk (recording, chunk index, tensor shape) triggers it --
cheaper than bisecting blind.

Smoke output also reproduced the already-documented `1_02_0` fabricated-
seizure-ID gotcha (chb01_02.edf has 0 real seizures --
`chb01-summary.txt`) -- expected under `--smoke`'s capped interictal
recording selection, not a new bug, see CONTEXT.md's "Known gotchas".
