# channel_cwt null baseline: making it a clean "pre" ablation (2026-08-30, PM)

Mac shell. Follows `less_engineered_encoders_and_anomaly_reframe.md` (same
day). Subject: getting `ChannelCWTMambaClassifier` (`channel_cwt_mamba.py`)
to run as a *true* ablation of `temporal_graph_mamba` "pre" -- same
temporal block, same hyperparams, same ~60 s/epoch on MPS -- so the only
difference is "coherence graph removed."

## What channel_cwt is

The null baseline for every coherence-graph pipeline in the repo:
per-channel CWT power `|w|**2`, z-scored per (channel, freq) over the
training set (NO log -- "pre" doesn't log either; NO phase; plain real
Linear). Fed to a shared-weight sequence model. Answers: does the
coherence graph buy anything over raw per-channel spectra + a temporal
model? Swaps in for `HermitianSSMClassifier` so the hermitian_ssm LOSO
loop + SPH/SOP eval run unchanged (`_to_delete/run_channel_cwt.py`).

## The equivalence fixes (this was the whole session)

Initial runs were ~4x slower than "pre" (220 s/epoch) and OOM'd the
machine repeatedly. Root causes, all fixed:

1. **Wrong temporal block.** `_ChannelCWTNet` used `mambapy.mamba.Mamba`
   directly -- naive full pscan, no row-chunking. With
   `HERMITIAN_SSM_PARAMS["batch_size"]=64` and 23 channels folded into
   batch = 1472 sequences -> `[1472, 512, 128, 16]` f32 = 6 GB per SSM
   tensor -> ~25 GB swap avalanche (RSS stayed low because macOS
   compressed/swapped as fast as pscan allocated). **Fix:** rewrote
   `_ChannelCWTNet` to use `cwt_gnn_classifiers._DenseEdgeMambaTemporal`
   -- "pre"'s actual block (128-row chunking + gradient checkpointing).
   Component test `[32,23,480,8]` fwd+bwd: maxrss 2.2 GB (was ~25 GB).
   Config now matches "pre" exactly: d_model=16, d_state=16, d_conv=4,
   expand=2, n_layers=1, chunk_size=128, use_cuda_kernel=None.

2. **`--device cpu` in the wrapper.** mambapy pure-PyTorch pscan over
   T=480 is ~10x slower on CPU. Every historical "pre" timing was MPS.
   **Fix:** wrapper forces `--device mps`. Now a repo rule -- see
   CONTEXT.md and memory `mps-default-on-this-mac`.

3. **No negative subsampling.** `DEFAULT_NEGATIVE_TO_POSITIVE_RATIO = 5.0`
   -- the dense-family LOSO loops subsample training negatives 5:1 via
   `_subsample_negative_windows`, but `leave_one_seizure_out_hermitian_ssm`
   does NOT. channel_cwt was training ~2800 windows/fold vs "pre"'s ~830.
   **Fix:** added `negative_to_positive_ratio: float | None = 5.0` param +
   a subsampling block in `fit()` (interictal TRAIN windows only;
   `predict_proba` untouched).

Supporting memory fixes found along the way (all real, all fixed):
- `_WinDS.__getitem__` re-`np.load(mmap_mode="r")` every call -> faulted
  mmap pages accumulate ~7 GB/epoch. Fixed with a bounded in-RAM LRU
  (`OrderedDict`, `_ram_cap=48`, float16 store -> zero redundant disk
  reads across a shuffled epoch).
- `_build_continuous_dataset` loads all 41 chb01 raws as float64 = 7 GB,
  held the whole run. The real `HermitianSSMClassifier` ignores `raw_x`
  (reads the eigenpair cache); mine used it then stacked model on top.
  Fixed: `rec["raw_x"] = None` after each `cache.ensure` in `_build_index`.
- `fchunk=16` in `ensure` -> the ifft on a ~2^20-padded hour recording
  was ~7 GB. Fixed: `fchunk=1` (one freq bin per `cwt.transform` call).

## Machine swap exhaustion + reboot

Repeated jetsam kills left stale swap; machine hit 35/35.8 GB swap,
749 MB free. Quitting GUI apps (Chrome/Slack/Claude.app/Spotify) freed
~4 GB but wasn't enough -- the run avalanched again alone (that was root
cause #1, not yet fixed at the time). **User physically rebooted.** The
`/private/tmp` scratchpad was wiped -- run wrappers recreated in
`_to_delete/` (gitignored, survives reboots). New discipline: quit GUI
apps before heavy local runs; MPS default.

## Current run

`_to_delete/run_channel_cwt.py`, pid 3102, MPS, monitor `bo4gc07kl`.
All three equivalence fixes in. **Epoch 1-4: 49-54 s** -- matches "pre"'s
~60 s. 6-fold, 30 epochs/fold. val_auc climbing 0.78 -> 0.88 in fold 1.

## Result: NEGATIVE, killed at 2 folds (conclusive)

| fold | "pre" AP | channel_cwt auc_pr |
|---|---|---|
| 1_03_0 | 0.792 | **0.149** |
| 1_04_0 | 0.830 | **0.263** |

~0.6 AP below "pre" on both folds. Training curves were healthy every
fold (val_auc 0.91-0.93) and did not transfer -- the same
train/val-vs-holdout gap every hermitian lever has shown, but far wider.
Killed after fold 2 on the user's call.

**Read:** this is a clean one-variable ablation of "pre" (same
aggregate-to-23, same `_DenseEdgeMambaTemporal`, no message-passing in
either -- `_propagate_hops` only fires at `n_hops>1`, and "pre" is
`n_hops=1`). The *only* moving part is coherence edge features vs
per-channel CWT power. So the collapse is attributable directly to the
coherence representation: it carries essentially all the signal; raw
per-channel spectra + the same temporal model cannot substitute.

## Follow-ups queued

1. **`pre` reproduction -- DONE 2026-08-31, REPRODUCES.**
   (`_to_delete/run_pre_repro.py`, MPS, ~8.5 h IO-bound.) 6-fold mean AP
   **0.644** vs historical 0.674, ROC-AUC **0.973** vs ~0.94. Per-fold
   this/historical: 1_03 .560/.792, 1_04 .811/.830, 1_15 .416/.331,
   1_16 .985/.996, 1_18 .811/.802, 1_26 .283/~.29. The whole 0.03 gap is
   fold 1_03 variance; every other fold matches or beats, global ranker
   is better. -> 0.674 is real. channel_cwt (.149/.263) collapsed ~0.5
   below a clean same-machine baseline. Conclusion stands.
   CSV `results/temporal_graph_mamba/prediction/*_20260830-211240.csv`.
2. **per-channel features + graph message-passing** (ablation 1) --
   **DONE 2026-08-31, NEGATIVE.** 6-fold mean AP **0.173** (ROC-AUC 0.878),
   per-fold .150/.190/.180/.185/.189/.147 -- dead flat ~0.18 every fold,
   at or below the no-hops channel_cwt on the 2 folds that overlap, vs
   pre_repro's 0.644. Learned complete-graph message passing on the
   post-SSM per-channel vectors (`MLP([h_j,h_i])` + `GRUCell`, all 23*22
   ordered pairs) recovers nothing; the extra params marginally hurt on
   ~30 preictal windows/fold. **Conclusion: it is coherence specifically
   -- the cross-spectrum `S_ij = w_i conj(w_j)` and its phase `arg S_ij`
   -- NOT cross-channel information in general.** A learned interaction of
   per-channel magnitude-spectrum summaries cannot stand in for measuring
   the pairwise phase. CSV `results/hermitian_ssm/prediction/*_20260831-054118.csv`.
   (was BUILT + smoke-passed, `_to_delete/run_channel_cwt_hops.py`.)
   `_HopMessagePassing` in `channel_cwt_mamba.py` replicates
   `SparseEvidenceGNNCore._propagate_hops` exactly (MLP message on
   `[h_dst, h_src]` + scatter-add + `GRUCell` update) over the complete
   23-channel graph (506 directed edges -- the repo's canonical topology
   is fully connected, NOT a 10-20 montage). `mamba_n_hops=2` = 1
   propagation round after the (identical) Mamba temporal block. Still no
   coherence. Tests: cross-channel info in general vs coherence
   specifically. NB `_propagate_hops` is MLP+GRUCell, NOT Mamba -- the
   temporal Mamba block is unchanged from "pre"/channel_cwt.
3. (later) **`pre` with edge features = `|coherence|` magnitude only,
   phase zeroed** -- tests whether the phase relationships are the signal.
4. **drop edge component 3 (significance)** -- feed `temporal_edge_proj`
   only `[coh, sinphi, cosphi]` (3 not 4). In `coherence_threshold_mode=
   "fixed"` (what "pre" uses) significance is a deterministic threshold
   indicator on `|coh|`, not a surrogate test. Clean 3-vs-4 ablation,
   distinct from `temporal_graph_edge_complex` (drops it + `_ComplexLinear`).

## Perf: the dense_edge cache is bigger than RAM on this machine

The pre-repro run went IO-bound on folds 2+ (fold 1 44 s/epoch, folds 2+
~95 s, CPU 77 % idle) -- the `event_mode="temporal_graph"` cache is
~15 MB/trial (`[4, 253, 480, 8]`), ~70 GB for a full 6-fold (2026-08-27
note), exceeds 16 GB RAM -> page-cache thrash. The original 0.674 was
itself an overnight run stitched across 3 process restarts.

**Fix (same model, real TODO):** don't cache the `[4,E,T,F]` edge stack;
cache only the per-channel CWT (~1 MB/trial, fits RAM) and regenerate
edges on MPS each forward. In `"fixed"` mode all 4 components are
deterministic functions of `(raw window, config)` (`dense_edge_cache.py`
docstring); `_build_dense_edge_input` is already device-agnostic torch,
no numpy/RNG. Same result (bit-identical at matched dtype/device). The
2026-08-27 note found recompute *faster* than cache-load on CUDA.

## Parked, not launched

- `_to_delete/run_channel_cwt_broadband.py` -- linear 1 Hz freq grid,
  2-124 Hz, 60 bins, td=64, fp16 (~3 GB cache). Broadband variant. Run
  ONLY if the default channel_cwt result is good (user directive).
- `continuous_cwt_mamba` 6-fold -- NOT to be launched (user directive),
  kept in context only.
- post-based jobs from the earlier note (edge_complex ~11h, anomaly_v2
  ~3-4h, post_repro ~11h) -- one-at-a-time on MPS after channel_cwt.

## Uncommitted (commit once results land, `git add -p`)

`channel_cwt_mamba.py` (new), `hermitian_ssm_classifier.py`,
`cwt_gnn_classifiers.py`, `hermitian_ssm_anomaly.py` (new),
`run_pipelines.py`, CONTEXT.md, this note. `mamba_backend` committed
default is stale `"mamba3"` -> revert to `"mamba"` in the next hermitian
commit. Do NOT commit any `_CG_MAMBANET_SHARED_PARAMS` edit (other shell).
