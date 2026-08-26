# Repo context (read this first)

Living status doc, not a log. Says what's true *right now* -- for the
history/reasoning behind any of it, follow the pointers into
`Epilepsy/Session_notes/<date>/`. Update this file (not just the session
note) before ending a working session that changed the state below --
that's the whole point of it: this repo gets worked on from different
Claude/Grok shells that don't share context with each other, and this is
the one file meant to catch a new shell up without it re-reading
everything.

**Last updated:** 2026-08-26, by Claude (wired continuous-cwt-mamba into
`run_pipelines.py` as `--pipeline continuous_cwt_mamba` -- Open threads'
items 2+3 below. **WIP, not working yet:** new
`Epilepsy/pipelines/continuous_dense_edge.py` (whole-recording chunked
CWT, Phase A) and `ContinuousLabelingParadigm.get_continuous_data()`
(Phase B) are both built and verified (see `scripts/continuous_cwt_chunk_
parity.py`, `scripts/continuous_cwt_scale_probe.py`,
`scripts/continuous_labeling_get_continuous_data_parity.py`, all passing).
`ContinuousCWTMambaClassifier` (Phase C, `Epilepsy/pipelines/
continuous_cwt_mamba_classifier.py`) hit and fixed a real training-time
CUDA OOM (whole-recording-then-one-backward kept every chunk's graph
alive -- fixed via incremental per-window-group backward+step, see that
class's `_train_recording_incremental`), but a real `--smoke` run on CUDA
now hits a SECOND bug, an unresolved "illegal memory access" during
validation, not yet root-caused -- full writeup and the planned next
diagnostic step (rerun on `--device cpu` for a clean Python traceback) in
`Session_notes/2026_08_26/continuous_cwt_mamba_pipeline_wip.md`. Session
paused here (GPU needed for the user's other work) before that diagnostic
ran. **Do not trust a real run of this pipeline until that's fixed.**).

Prior entry, still current: Claude (merged branch `cg-mambanet` into
`main` -- built `Epilepsy/pipelines/cg_mambanet_classifier.py`, a
CNN-GCN-Mamba-BiLSTM architecture reconstructed from the CG-MambaNet paper
(arXiv:2606.08226, no public code), wired as `--pipeline cg_mambanet`
under this repo's own chb01 leave-one-seizure-out protocol (NOT the
paper's own multi-patient LOPO -- that's still deferred). Real chb01
smoke test succeeded (wiring verified), but a REAL (non-smoke) run is
currently BLOCKED: `mambapy`'s Mamba scan (both `pscan=True` and
`pscan=False`) does not scale to this encoder's size (12 layers x 2
directions, seq_len=480) on CPU OR MPS -- severely super-linear batch-size
scaling, MPS OOMs outright at batch=16. Full investigation, numbers, and
next steps:
`Session_notes/2026_08_26/cg_mambanet_architecture_and_mambapy_scaling_wall.md`.
Decision: stop here rather than shrink `mamba_n_layers` further, resume once
this runs on the RunPod CUDA image (fused `mamba-ssm` kernel sidesteps the
whole problem, and per the entry directly below may already have a
mambapy-vs-fused-kernel comparison worth reading first). Added a new
"Known gotchas" entry below for this -- don't re-discover it.). 
Prior
entry, still current: Claude (chb02-04 dense-edge-GRU baseline extension +
GitHub-release mirror generalized from chb01-only to any subject --
separate thread from cg-mambanet, on a fresh branch off the same
pre-cg-mambanet commit; see "Right now (chb02-04 baseline + GitHub-release
mirror)" below -- this may also resolve part of cg-mambanet's own deferred
"montage unification across CHB-MIT patients" blocker, worth checking
before re-deriving it). 
Prior entry, still current: Claude (committed all outstanding
work on `continuous-cwt-mamba` -- SlimSeiz channel-select stage, the
`--slimseiz-fixed-channels` flag, the DBConformer depth/weight sweep, the
4-way pipeline comparison note, and this file's own recent edits --
pushed, fast-forward-merged into `main`, then deleted `continuous-cwt-mamba`
both locally and on `origin`. **`continuous-cwt-mamba` no longer exists --
any shell still referencing it as "the current branch" (including
`CLAUDE.md`'s "Facts that flipped" section, not yet updated) should treat
`main` as current instead.** No code changes in this update, just
branch/repo state.). Prior entry, still current: Claude (wrote the
deferred 4-way pipeline comparison -- GRU vs Mamba vs DBConformer vs
SlimSeiz, chb01 prediction, all under the same shared protocol; Mamba
leads on AP (0.499), see
`Session_notes/2026_08_25/pipeline_comparison_gru_mamba_dbconformer_slimseiz.md`
for the full table and per-fold breakdown. Also corrected two notes'
"root cause may be a second simultaneous Claude-shell job" framing for
the slimseiz crash -- confirmed isolated to stage 1 channel selection,
not concurrency; see "Known gotchas" below). Prior entry, still current:
Claude (the queued `--slimseiz-fixed-channels` 6-fold run finished --
essentially ties the adaptive per-fold selection on chb01 aggregate,
missing the same seizure on hit rate too; see "Right now" below for the
full comparison table. One operational hiccup along the way, not a
crash: the watchdog script's hardcoded ~700s wall-clock kill was sized
for one fold, not six, and killed the first launch mid-fold-5 -- fixed
and the remaining 2 folds rerun via `--skip-folds`, see that section).
Before that: Claude (ran `dbconformer` at real 6-fold scale plus two negative
hyperparameter diagnostics -- depth 5 is the reported baseline, see
`Session_notes/2026_08_25/dbconformer_baseline_runs.md`). Before that:
Claude (added `--slimseiz-fixed-channels` -- bypasses stage 1 entirely and
feeds stage 2 a literal channel list). Before that: Claude (profiled the slimseiz
channel-select crash under a memory watchdog, added a `max_samples` cap to
`select_slimseiz_channels` as a mitigation -- root cause NOT fully
confirmed, see "Known gotchas" below). Before that: Claude (`--pipeline
dbconformer`/`slimseiz` added). Before that: Grok (`scan="chunk"`,
`use_cuda_kernel`, GHCR `eeg_benchmarks-mamba` image live).

---

## Right now (chb02-04 baseline + GitHub-release mirror, 2026-08-26)

Two things, done on a fresh branch off this commit of `main`
(`continuous-cwt-mamba` no longer exists, see the entry above -- this
work started on it before that was known, then got cherry-picked onto
`main` directly once the divergence was discovered). Full detail:
`Epilepsy/Session_notes/2026_08_26/
chb02_chb03_chb04_gru_baseline_and_github_release_generalization.md`.

1. **Matched dense-edge-GRU baseline extended to chb02 and chb03** (same
   protocol as chb01's `full_6fold_23ch_encoderfree_val_gru.md`). chb02:
   **total collapse** -- 0/2 hit rate, AP 0.016, AUC 0.498 (chance),
   model predicts negative on every single preictal window in both folds.
   chb03: **3/7 hit rate** with a striking split -- the 4 earliest folds
   (`chb03_01`-`04`) collapse the same way chb02 did, the 3 latest
   (`chb03_34`-`36`, after a long seizure-free gap) train and predict
   well (recall 0.67-0.83, AUC 0.88-0.91). Leading theory (not tested):
   each LOSO fold on these subjects trains on very few other seizures'
   preictal windows (chb02 has only 2 usable seizures total after
   SPH/SOP filtering), so it's data starvation, not GRU-specific --
   proposed but didn't run the same protocol with `dense_edge`/
   `dense_edge_mamba` to check. **Also found, not fixed:** chb02's
   reported seizure IDs (`2_17_0`, `2_20_0`) are off by +1 real file
   number vs. `chb02-summary.txt` (real seizures are in `chb02_16+.edf`
   and `chb02_19.edf`) -- same "+`.edf` file shifts a position-based
   index" family of bug as chb01's fabricated `1_02_0`, but this time
   both seizures are real, just mislabeled. Root cause not traced (likely
   `run_pipelines.py`'s `unique_seizures` construction).
2. **GitHub-release mirror generalized from chb01-only to any subject.**
   `datasets/epilepsy/chb_mit.py`'s `GITHUB_RELEASE_SHA256` registry (was
   `CHB01_GITHUB_*` single-subject constants) now covers chb01-04, with
   multi-part support added for chb04 (its ~6.4GB raw doesn't fit
   GitHub's 2GiB-per-asset cap as one archive -- split into two xz parts).
   Four live releases: `chbmit-chb01-1.0.0` through `chbmit-chb04-1.0.0`.
   Two unrelated bugs found+fixed along the way: chb02's `chb02_16+.edf`
   404'd on PhysioNet S3 (unescaped `+` in the URL -- `_record_url` now
   `quote()`s the filename) and a pre-existing (confirmed via `git
   stash`, not introduced this session) Windows-only bug in
   `download_url`'s `file://` handling. 13/13 tests passing
   (`tests/test_chb_mit_github_prefetch.py`, rewritten for the
   generalization), plus a live check that every published asset URL
   resolves with the right `Content-Length`.

Also mid-session: disk hit 92% (77G/894G free) from this session's own
downloads+caches -- purged pip cache (3.6G) and deleted
`~/mne_data/dense_edge_cache` (16G, pure recompute cache, safe, rebuilds
automatically) at user's request. If dense-edge training looks slower on
the very next run after this, that's the recompute, not a regression.

---

## Right now

**`--slimseiz-fixed-channels` added to `run_pipelines.py` (2026-08-25).**
The upstream SlimSeiz repo's paper numbers turn out to most likely be
reported for one fixed 8-channel montage shared across the whole CHB-MIT
cohort, not a genuinely per-patient adaptive selection -- found by pulling
`Common_channesl.ipynb` from `github.com/guoruilu/SlimSeiz` (not vendored
into this repo as a file; not part of the two notebooks this repo already
ported/cited). That notebook loads each patient's own top-8 channel-select
output (`chbNN_sel_ch_30iter_with_SMOTE.json`, not published in the repo,
only their printed cell outputs are) and tallies which channels land in
most patients' top-8: `P3-O1, P8-O2, C3-P3, C4-P4, FZ-CZ, P4-O2, CZ-PZ,
F3-C3` (counts 18/18/18/17/17/17/15/14 out of 24 patients) --
`SLIMSEIZ_PAPER_FIXED_CHANNELS` in `run_pipelines.py`. For chb01
specifically its own top-8 overlaps this fixed set 7/8 (missing F3-C3
only). `--slimseiz-fixed-channels` (no args = that default list, or pass
explicit names) sets `SlimSeizClassifier.channel_select_fixed_indices`,
which **skips stage 1 entirely** (no PCA/SMOTE/DecisionTree call at all --
see `slimseiz_classifier.py`'s "FIXED CHANNELS" docstring section) and
just slices `X` to the given channels before stage 2 -- this is actually a
*lower*-risk slimseiz configuration than the default adaptive-selection
path, since it removes the crash-implicated stage from the run entirely.
Name-to-index resolution is chb01-only right now (`CHB01_CHANNEL_NAMES`,
read directly off `chb01_01.edf` via `mne.io.read_raw_edf`, confirmed
`datasets/epilepsy/chb_mit.py` does no picking/reordering so this is the
real channel order every fold's `X` uses) -- errors if `--subjects` isn't
`[1]`. Smoke-tested clean (`--smoke --slimseiz-fixed-channels --max-folds
1`, peak RSS ~1GB, resolved indices `[7,15,6,10,16,11,17,5]` matched a
hand-check against the EDF header). **A real (non-smoke) 6-fold run with
this flag is queued** (not yet started as of this writing) -- a background
watcher (`/private/tmp/.../scratchpad/wait_then_run_slimseiz_fixedch.sh`,
session-local, won't survive a reboot) is polling for PID 13592 (a
concurrent `--pipeline dbconformer` job, already running when this was
queued) to exit before launching, RSS-capped at 12GB via the same
watchdog wrapper used in the crash investigation below. Log will land at
`Epilepsy/results/slimseiz/prediction/full6fold_slimseiz_fixedch_
20260825-204111.log`. If that log doesn't exist yet and the watcher/
dbconformer processes are gone (check `ps`), the queued run silently
never fired (e.g. this Mac rebooted) -- just rerun
`.venv/bin/python Epilepsy/run_pipelines.py --pipeline slimseiz
--slimseiz-fixed-channels` directly (ideally still watchdog-wrapped, see
gotcha below).

**Fixed-channel 6-fold run DONE (2026-08-25) -- essentially ties the
adaptive per-fold selection on chb01.** Not memory-related at all this
time: the first launch died at fold 5/6 because the watchdog script
(`probe_with_watchdog.sh`, still in scratch, session-local) had a
hardcoded ~700s wall-clock kill sized for testing one fold during the
crash investigation, not a full 6-fold pass -- an operational mistake, not
a crash (RSS was ~6.6GB the whole time, nowhere near the 12GB cap). Fixed
by making the timeout a `TIMEOUT_S` env var and rerunning just the 2
remaining folds via `--skip-folds 0 1 2 3` (fold order is
`(subject,run,seizure_onset)`-sorted: 1_03_0, 1_04_0, 1_15_0, 1_16_0,
1_18_0, 1_26_0 -- indices 0-5). Combined result, fixed-channels vs. the
earlier real adaptive-selection run
(`prediction_leave_one_seizure_out_20260825-171651.csv`):

| metric | adaptive (per-fold) | fixed (paper's 8) |
|---|---|---|
| precision (mean) | 0.286 | 0.297 |
| recall (mean) | 0.556 | 0.539 |
| f1 (mean) | 0.340 | 0.351 |
| FAR/h raw/smoothed (mean) | 8.56 / 6.31 | 8.22 / 6.08 |
| hit rate raw | 5/6 | 5/6 |
| hit rate smoothed | 4/6 | 4/6 |

Both configurations miss the exact same seizure on hit rate (`1_18_0` --
mean preictal score ~0.0004 under both, a genuinely hard fold, not a
channel artifact) and the same smoothed-miss (`1_15_0`). Per-fold, two
folds move in opposite directions (`1_03_0` f1 0.324->0.246 under fixed,
worse; `1_26_0` f1 0.296->0.436 under fixed, better) that roughly cancel
in the aggregate. **Conclusion: for chb01, the crash-implicated stage-1
selection isn't buying anything over the paper's own fixed 8-channel
montage** -- `--slimseiz-fixed-channels` gets the same result, safer (no
PCA/SMOTE/DecisionTree at all) and faster. Untested whether this
generalizes past chb01 (`--subjects` is chb01-only in this repo today).
Results: `prediction_leave_one_seizure_out_20260825-171651.csv` (adaptive,
6 folds) + fixed-channel folds 0-3 (log only, CSV lost to the timeout
kill -- see `full6fold_slimseiz_fixedch_20260825-204111.log`) + folds 4-5
(`prediction_leave_one_seizure_out_20260825-205851.csv`).

**`--pipeline dbconformer` / `--pipeline slimseiz` added to
`run_pipelines.py` (2026-08-25, this branch).** Two new raw-EEG classifiers
vendored from the same upstream repo as each other (`../DBConformer/models/`
in this checkout) into `Epilepsy/pipelines/dbconformer_classifier.py` /
`slimseiz_classifier.py` -- DBConformer (dual temporal/spatial-Transformer)
and SlimSeiz (1-D conv stem + a self-contained sequential-scan Mamba block,
independent of this repo's own `mambapy`-based dense-edge-mamba). Neither
does CWT/STFT preprocessing -- they classify raw `(n_channels,
n_timepoints)` windows directly, so no disk cache either. Both respect
`--label-mode` (detection AND prediction, unlike `truong_stft_cnn` which
forces prediction-only) via their own shared leave-one-seizure-out loops
(`leave_one_seizure_out_raw_classifier[_prediction]` in `run_pipelines.py`).
Own hyperparameter blocks, own `results/dbconformer/` / `results/slimseiz/`
output dirs. `einops==0.8.2` added to `requirements.txt` (both models'
attention/Mamba code needs it); `timm` was NOT added -- DBConformer's one
`trunc_normal_` use was swapped for `torch.nn.init.trunc_normal_` instead
(see that module's docstring for this and every other vendoring adaptation).
Verified with `--smoke --max-folds 1 --device cpu` for both pipelines x
both label modes (wiring only, not model quality -- untrained/untuned
hyperparameters, see each PARAMS dict's own "starting point" comments).

**`dbconformer` (chb01, prediction mode) now has a real 6-fold baseline,
plus two negative diagnostic results (2026-08-25).** Reported number:
`tem_depth=chn_depth=5`, `use_class_weights=True` (this repo's standard
protocol, matching GRU/Mamba) — AP 0.442, f1 0.366, precision 0.273,
hit rate 5/6 k-of-n (`prediction_leave_one_seizure_out_20260825-175207.csv`).
Two follow-up axes tried, both regressed and were reverted: `tem_depth=
chn_depth=6` (the paper's own MI default) and `=3` both scored worse
than 5 (5 is a local optimum, not one end of a trend); `use_class_
weights=False` improved precision (0.273->0.322) but cost AP/f1/hit-rate
(a real tradeoff, not a bug fix). `_DBCONFORMER_SHARED_PARAMS`/
`PREDICTION_DBCONFORMER_PARAMS` in `run_pipelines.py` are back at the
175207 config; both diagnostics are documented inline in that file's
comment block. **Not** a reproduction of any DBConformer paper-reported
number -- their seizure-detection results are on CHSZ/NICU, not CHB-MIT,
and their own LOSO script is leave-one-*subject*-out on MI/ERP data, not
this repo's leave-one-*seizure*-out protocol. Full per-fold tables and
reasoning: `Epilepsy/Session_notes/2026_08_25/dbconformer_baseline_runs.md`.
`slimseiz` still has no equivalent real-scale run as of this entry.

GRU vs Mamba (encoder-free, full 23-channel mesh, val split 0.2, early
stop, matched protocol) across all 6 chb01 leave-one-seizure-out folds —
**both runs finished, comparison written up.** Mamba wins on AP (+0.076,
0.499 vs 0.423) and both FAR/h numbers, loses on recall/f1 (driven mostly
by a recall collapse on one fold, `1_18_0`), same 5/6 k-of-n hit rate as
GRU but a different seizure missed (`1_18_0` for Mamba vs `1_26_0` for
GRU — see gotcha below, the "persistent miss" turned out to be
GRU-specific, not universal). Mamba costs ~4.6x the wall-clock/epoch on
this hardware (`mambapy` pure-PyTorch scan, no CUDA kernel). Full tables
in `Epilepsy/Session_notes/2026_08_25/
full_6fold_23ch_encoderfree_val_gru.md`. Also resolved there: the earlier
`smoke_test.py`-based Mamba run's exact-0.0-val_loss pattern did not
reproduce in this matched run (0 occurrences across all 6 folds) — looks
like an artifact of `smoke_test.py`'s small capped train/val split, not a
real Mamba training-path issue.

**`continuous-cwt-mamba` branch, separate thread from the GRU/Mamba
comparison above:** a new paradigm, not a tuning change to
`_DenseEdgeMambaTemporal`. Goal: CWT the whole recording once (not
arbitrary 30s windows), let Mamba's SSM state run continuously across the
ENTIRE timeline with no per-window reset, and only window at the
classification readout, at the end. Motivation: the existing
`_DenseEdgeMambaTemporal` resets state every window (each call to
`Mamba.forward()`'s parallel scan always starts h=0) -- that throws away
exactly the cross-window context Mamba is supposed to be good at.

Built and verified so far, all in `Epilepsy/pipelines/cwt_gnn_classifiers.py`:
- `_DenseEdgeMambaContinuous` -- default `scan="chunk"` is a local
  reimplementation of unreleased mambapy `Mamba.chunk_step`
  (alxndrTL/mamba.py@67000c9; **not** in the pinned `mambapy==1.2.0` PyPI
  wheel): the same Blelloch pscan `_DenseEdgeMambaTemporal`'s
  `Mamba.forward()` already uses, but with the previous chunk's `h`
  injected at timestep 0 and the conv1d's left context taken from the
  carried `inputs` cache. `scan="step"` keeps the original per-timestep
  `Mamba.step()` Python loop as a parity/ablation path. Truncated BPTT
  unchanged: cache returned `.detach()`'d, so the forward state chain is
  unbroken across a recording and gradients for chunk N do not reach
  chunk N-1. See
  `Epilepsy/Session_notes/2026_08_25/continuous_mamba_chunk_scan_throughput.md`.
- `pool_continuous_edge_stream_to_windows` -- turns a concatenated
  continuous `[B,C,E,T_total]` output stream into per-window
  `[B,C,E,1]` snapshots (pool="last", the continuous analogue of
  `_DenseEdgeMambaTemporal`'s own "pool to h_T" convention) by pure
  indexing, no model weights.
- `_dense_edge_features` split into `_dense_edge_conv_out` +
  `_dense_edge_features_from_conv_out` (behavior-preserving refactor) so
  the pooled continuous-stream windows feed into the EXISTING
  `sparse_message_mlp`/hop-propagation/`sparse_classifier` readout
  unmodified -- confirmed working end-to-end (unbound-method test, see
  `scripts/dense_edge_mamba_continuous_parity.py`'s sibling checks in
  session history; not yet re-saved as a standalone script).
- **Proven, not just plumbed:** `scripts/continuous_mamba_state_carryover_check.py`
  -- synthetic task where a signal is injected only into a recording's
  first chunk and every later window's label depends on it, so a
  state-reset-every-chunk model has literally zero information for those
  windows (fresh noise every step, never repeated -- memorization is
  impossible) while a carried-state model can solve it. Result: carried
  state hits 0.922 (5-epoch-rolling-avg accuracy on the signal-free
  windows), reset stays at 0.573 (chance = 0.5). This is the actual
  evidence the mechanism buys something real, not just "it runs."
- `scripts/dense_edge_mamba_continuous_parity.py` -- T-chunking is
  bit-exact vs. one big call; `step()`-driven output matches `mambapy`'s
  own `forward()` scan to float32 noise (~1e-7); `scan="chunk"` matches
  `scan="step"` to the same tolerance (and matches `Mamba.forward()`
  bit-exactly on a fresh cache). A chunk's `backward()` genuinely
  doesn't reach into the previous chunk's freed graph.
- `tests/test_dense_edge_mamba_continuous.py` -- the above, plus n_layers=2
  and `pool_continuous_edge_stream_to_windows`.
- `scripts/continuous_mamba_gpu_scale_probe.py` -- real-scale (23ch full
  mesh, E=253, C_in=192, d_model=16/d_state=16/expand=2) fwd+bwd probe,
  now comparing both scans and sweeping B. **The step()-loop throughput
  problem is closed as a blocker:** CPU median ~40x (`step` ~82ms/step vs
  `chunk` ~2ms/step at T=256); MPS ~1.55x at T=512 (chunk flat in T at
  ~0.94ms/step, step growing). CUDA 3070 Ti was the original 2.68ms/step
  `scan="step"` measurement -- not re-run this session (Mac shell); the
  updated probe is what to run on that box. Row-batching (B>1) is linear
  in rows once T is fused, so it is **not** the fix and is not a
  prerequisite for the data pipeline. `scan="chunk"` does allocate the
  pscan `[rows,T,d_inner,d_state]` intermediates (~0.5GiB/tensor at
  B=1,E=253,T=1024) -- still fine at B=1 on 8GB; stacking many
  recordings will reintroduce Temporal's rows*T OOM and would need the
  same `mamba_chunk_size`-style row split.

**Not started:** the actual CHB-MIT data pipeline / LOSO loop rewrite --
everything above was deliberately built and verified in isolation
(synthetic tensors) before touching `_build_windowed_dataset`/
`leave_one_seizure_out_*` in `run_pipelines.py`, since that's the highest-
risk, hardest-to-verify part (recording-level continuous sequences instead
of independent per-window rows, TBPTT chunk boundaries, windowed labels
read off a continuous timeline). See "Open threads" below.

## Branch map

- `main` -- **current branch, everything is merged here as of 2026-08-26.**
  Has `dense_edge_mamba` (from `mamba-temporal-edge-model`, merged at
  `6d38573`), the cwt node encoder, the continuous-cwt-mamba paradigm
  plumbing (`aa3c565`, `4760de0`), `use_cuda_kernel` + `Dockerfile.mamba` +
  the live GHCR `eeg_benchmarks-mamba` image, SlimSeiz, dbconformer, the
  chb01-04 GitHub-release mirror generalization, AND (this update)
  `cg_mambanet` -- see the 2026-08-26 "Last updated" entries above for both.
- `mamba-temporal-edge-model` (remote, `origin/`) -- where
  `dense_edge_temporal_mode="mamba"` (`_DenseEdgeMambaTemporal`) was
  originally developed, before merging into `main` at `6d38573`. Archived
  and removed by another shell mid-2026-08-26 (see
  `Session_notes/2026_08_26/mamba_temporal_edge_model_branch_archived.md`)
  -- don't assume it still exists.
- `continuous-cwt-mamba` -- **no longer exists.** Merged into `main` and
  deleted (both locally and on `origin`) 2026-08-25; don't assume it's
  still there, and don't recreate it without checking `main` first (one
  shell this session accidentally did exactly that with a stale local
  copy -- see the "chb02-04 baseline + GitHub-release mirror" entry above
  for how that got sorted out).
- `cg-mambanet` -- **merged into `main` 2026-08-26** (fast-forward, this
  update), still exists as a branch (locally and on `origin`) but has
  nothing `main` doesn't now. Left un-deleted since a real run is still
  blocked on `mambapy`'s scan performance (see "Known gotchas" below) --
  may be worth resuming work on this branch specifically once that's
  unblocked, rather than starting fresh on `main`.
- `tf-node-encoding`, `dynmaic_subset` -- exist locally, not investigated
  recently; don't assume they have anything `main` doesn't unless you
  check.
- `dense_edge_mamba` is on `main` now, so there's no need to hunt for a
  branch that has it -- just confirm with `grep _DenseEdgeMambaTemporal
  Epilepsy/pipelines/cwt_gnn_classifiers.py` if in doubt.

## Known gotchas (keep rediscovering these -- stop rediscovering them)

- **`mambapy`'s Mamba scan (BOTH `pscan=True` and `pscan=False`) does not
  scale to a "many stacked/parallel Mamba instances x long-ish sequence"
  encoder, on CPU OR MPS -- measured 2026-08-26 building
  `cg_mambanet_classifier.py`'s bidirectional 12-layer encoder (24 total
  directional Mamba instances at seq_len=480).** `pscan=True` (default):
  severely super-linear batch-size scaling (batch 4/8/16 -> 3.1s/12.5s/
  40.3s total on CPU) and an outright MPS OOM at batch=16 (`pad_npo2`
  padding the sequence to a power of two and keeping intermediate tensors
  at every scan level for backward, across all 24 instances at once).
  `pscan=False` (mambapy's own documented fallback): forward stays cheap
  but BACKWARD explodes worse (960.89s at batch=16 on CPU) -- the
  sequential Python loop's ~11,500 chained autograd nodes (24 instances x
  480 timesteps) is its own catastrophic-backward failure mode. Neither
  mode is a fix for the other's problem; this is a `mambapy` (pure-PyTorch
  Mamba) limitation at this depth/sequence-length/instance-count
  combination, not something a classifier's own hyperparameters (d_model,
  batch_size within reason) can route around -- `_DenseEdgeMambaTemporal`
  never hit this because it's a single unidirectional instance, not 24 at
  once. Full numbers:
  `Session_notes/2026_08_26/cg_mambanet_architecture_and_mambapy_scaling_wall.md`.
  The fused `mamba-ssm` CUDA kernel (RunPod image) is a different code path
  (a real kernel, not chained PyTorch ops) and is expected to sidestep this
  -- not yet confirmed, no kernel-vs-pscan parity check run for this
  specific encoder shape yet.
- **`--pipeline slimseiz` (non-smoke) once blew up memory/crashed this
  Mac; root cause NOT cleanly pinned down despite real profiling; a
  hardening fix is in place but is a mitigation, not a proven fix.**
  Original incident: a full 6-fold LOSO pass coincided with a hard crash
  on 2026-08-25 ~18:28 (SOCD hardware watchdog reset, not a clean Python
  OOM-kill -- see
  `/Library/Logs/DiagnosticReports/panic-base+socd-2026-08-25-182804.panic`
  / paired `ResetCounter-*.diag`, "Boot faults: wdog,reset_in_1"; its log
  file was left at 0 bytes).

  **Investigation (2026-08-25 ~19:20-19:55), all under a custom RSS-limit
  watchdog wrapper (kills the process before it can take the OS down) --
  see the session note for the wrapper script if it's not still in
  scratch:**
  - `_build_windowed_dataset` alone (real args: `window_length=30.0`,
    `max_interictal_recordings=None`): safe, ~5s, peak ~5.8GB.
  - `select_slimseiz_channels` alone, on a real fold's actual training
    array (868, 23, 7680): safe, peak <1GB -- but **slow, ~200s/fold**
    (30 iter x 23 channels x PCA(60)+SMOTE+DecisionTree on 7680-wide
    real-scale windows, vs. --smoke's 1024-wide).
  - Stage 2 (network training) alone, real scale, stage 1 stubbed out:
    safe, peak ~8GB, recovers after.
  - **The real combined 1-fold run (both real stages, unstubbed)
    reproduced the fast blowup ONCE** (~8.8GB RSS + ~4.8GB swap within
    20s, empty log, killed) but **did NOT reproduce it on two further
    identical attempts** (one ran 300s+ peaking ~8.2GB before hitting a
    test timeout, unrelated to memory; one ran to completion in 463s,
    peak ~10.0GB, wrote results normally). So the failure is NOT reliably
    reproducible from this code alone under matching conditions. **This is
    isolated to stage 1 (`select_slimseiz_channels`, PCA/SMOTE/DecisionTree
    over 23 channels x 30 iterations on real-scale 7680-wide windows) --
    it is not a general "running two things at once on this Mac" risk.**
    Running other pipelines (DBConformer, GRU/Mamba, or another slimseiz
    job that bypasses stage 1 via `--slimseiz-fixed-channels`) alongside
    each other is not implicated by this investigation. Treat "fixed"
    claims about this pipeline skeptically until it's been run clean
    multiple more times, including a full 6-fold pass (not yet attempted
    post-fix as of this writing).

  **Mitigation applied regardless** (bounds worst-case cost even if the
  exact trigger stays unknown): `select_slimseiz_channels` gained
  `max_samples` (default 1000, one stratified subsample drawn up front,
  reused across all 30 iterations/23 channels) -- see
  `slimseiz_channel_select.py`'s own docstring. Wired through
  `SlimSeizClassifier(channel_select_max_samples=...)` and
  `--slimseiz-channel-select-max-samples` in `run_pipelines.py`. Note this
  cap does NOT activate on typical folds today (they run ~868-900 samples
  under the default 5:1 negative:positive subsample, under the 1000
  cap) -- it protects a fold with more positives / a looser ratio, not
  the exact fold used during this investigation.

  **Before running `--pipeline slimseiz` without `--smoke` (and without
  `--slimseiz-fixed-channels`, i.e. stage 1 actually runs) on this Mac
  again:** consider wrapping it in an RSS-limit watchdog (kill ~11GB)
  rather than running bare, especially for a multi-fold run -- a single
  real fold measured ~10GB peak and ~8min; a 6-fold run is untested
  post-fix and could run ~45-75min at proportionally higher cumulative
  risk if a larger fold pushes past what's been measured here.
- **No `mamba-ssm` (CUDA kernel) on Windows/Mac.** PyPI ships sdist-only,
  needs nvcc + Linux. Portable default is still `mambapy` pscan
  (`requirements.txt`). `_DenseEdgeMambaTemporal(use_cuda_kernel=None)`
  now **auto-detects**: True iff CUDA + `mamba-ssm` importable, else
  False. Explicit `--mamba-use-cuda-kernel` errors if the kernel isn't
  there (no silent fallback). The RunPod image
  `ghcr.io/noshore5/eeg_benchmarks-mamba:20260825-4760de0` (also `:latest`)
  has the kernel compiled in -- that is the box to use it on. Do not
  re-force `use_cuda=False` in `__init__`; that was the old state.
- **Fused kernel is NOT compatible with (b)float16** (mambapy's own
  docs). When `use_cuda_kernel` is on, `_mamba_pooled` disables autocast
  and runs that block in fp32, so `--train-amp-bf16` on the rest of the
  model is OK. Don't feed bf16 into the kernel yourself. Continuous
  `_DenseEdgeMambaContinuous` does **not** use this kernel
  (`selective_scan_fn` has no initial-state argument).
- **`smoke_test.py`'s epoch_time is not a full-run predictor**, especially
  for the mamba backend. It defaults to capped data
  (`max_interictal_recordings=5`) and `mamba_chunk_size` overhead scales
  nonlinearly with batch composition -- measured smoke-scale Mamba/GRU
  slowdown ~14x vs. real full-mesh/uncapped ~4.6x, same code. To estimate
  a real run's epoch_time, run `run_pipelines.py` itself with `--max-folds
  1 --epochs 1` against real (uncapped) params, not `smoke_test.py`.
  (Caveat now also printed in `smoke_test.py`'s own output.)
- **Disk cache is off by default on CUDA** (`resolve_disable_disk_cache`
  in `run_pipelines.py`) -- every batch recomputes CWT/dense-edge features
  from scratch. Expected, not a bug; explains the repeated
  `CWT(batched,gpu-resident)` / `dense-edges[fixed]` progress-bar blocks in
  every log, one set per batch.
- **`--max-folds` only truncates from the front** of the leave-one-seizure-
  out loop; there's no flag to skip an already-done fold 1 and resume at
  fold 2 -- use the new `--skip-folds` flag (added on this branch,
  `continuous-cwt-mamba`) for that instead, or just rerun fold 1 (it's
  deterministic, seed 42, so a rerun is a reproducibility check more than
  wasted work).
- **`run_pipelines.py` defaults to `--device mps`** (author's Mac). Always
  pass `--device cuda` explicitly on a CUDA box.
- **chb01 is still the only subject `DEFAULT_SUBJECTS` exercises and the
  only one baked into the RunPod `Dockerfile`**, but as of 2026-08-26 it's
  no longer the only subject mirrored on GitHub Releases -- chb02/03/04
  are too now (`GITHUB_RELEASE_SHA256` in `datasets/epilepsy/chb_mit.py`,
  see this file's 2026-08-26 section above and `README.md`'s "CHB-MIT
  subjects" section). A subject NOT in that registry still pulls from
  PhysioNet's S3 mirror on demand, same as before this existed.
- **`1_26_0`** (a chb01 seizure) is a persistent k-of-n miss for every
  *GRU*/encoder-in-graph variant tried so far (encoder-in-graph,
  encoder-free, k=4 through full mesh) -- but the matched Mamba run
  (2026-08-25) hit it; Mamba's own miss was `1_18_0` instead. So "misses
  1_26_0" is GRU-specific, not universal -- don't assume it generalizes to
  a new backbone without checking.
- **chb01 has exactly 7 real seizures**: `1_03_0, 1_04_0, 1_15_0, 1_16_0,
  1_18_0, 1_21_0, 1_26_0` (confirmed from `chb01-summary.txt`, not from
  any run's output). `chb01_02.edf` has 0 seizures -- a seizure ID
  `1_02_0` appearing anywhere (e.g. an old `smoke_test.py`-driven run
  note) is a labeling bug, not a real held-out seizure. Today's 6-fold
  runs (GRU and Mamba both) cover 6/7, missing `1_21_0` specifically --
  it produces zero surviving preictal windows at SPH=300/SOP=900 (onset
  too close to its own recording's start), so it never enters
  `leave_one_seizure_out_prediction`'s fold list at all.
- **chb02's reported seizure IDs are off by +1 real file number**
  (2026-08-26, not fixed). `2_17_0`/`2_20_0` are what the LOSO loop
  prints, but `chb02-summary.txt`'s actual seizures are in
  `chb02_16+.edf` and `chb02_19.edf` (onset/offset timestamps match
  exactly) -- a third real seizure in `chb02_16.edf` never enters the
  fold list at all (same too-close-to-recording-start reason as chb01's
  `1_21_0`). Same family of bug as chb01's fabricated `1_02_0` ID, but
  this time the seizures ARE real, just mislabeled. Root cause not
  traced -- likely `run_pipelines.py`'s `unique_seizures` construction
  building `seizure_id` from a recording-list position rather than the
  filename's numeric suffix, thrown off by `chb02_16+.edf` sitting
  between `chb02_16.edf` and `chb02_17.edf` in file order.

## Open threads

- ~~Mamba 6-fold run~~ -- done, see "Right now" and the session note.
- **chb02/chb03 GRU baseline extension (2026-08-26)**: chb02 total
  collapse (0/2 hit rate, AUC at chance), chb03 3/7 with a stark
  early-recordings-fail / late-recordings-work split -- see this file's
  2026-08-26 section above and its session note. Leading theory is data
  starvation (few preictal windows per LOSO fold on these subjects), not
  tested against conv/Mamba backbones yet -- that comparison is the
  obvious next step if this thread continues.
- chb02's seizure-ID off-by-one (immediately above) -- not fixed.
- Mamba's `1_18_0` recall collapse (0.100 vs GRU's 0.767, same seizure) --
  not investigated, biggest single per-seizure divergence in the
  comparison. Tried on 2026-08-25 (Mac shell) to pull raw `predict_proba`
  scores for that fold's 30 true preictal windows to check
  ranking-vs-threshold (are scores clustered just under 0.5, i.e. a
  calibration/threshold problem, not a ranking failure?) without
  retraining -- couldn't: confirmed **no run in this codebase ever
  persists a trained model** (`torch.save`/`pickle.dump`/`joblib.dump`
  all absent from `run_pipelines.py` and `pipelines/*.py`, checked on
  both `continuous-cwt-mamba` and `origin/mamba-temporal-edge-model`
  `2985233`, the actual commit that produced this run). `common.py`'s
  early-stopping `best_state = deepcopy(model_.state_dict())`
  (~line 2090) is RAM-only, restored into `self.model_` inside `fit()`,
  gone once the process exits. The `EEG_Benchmarks_mamba` worktree that
  ran it also only ever existed on the Windows/CUDA box from that
  session (`C:\Users\User\Documents\noshore5\EEG_Benchmarks`) -- not
  reachable from a Mac shell. Two ways to actually get this number next
  time: (a) retrain just this one fold (`--pipeline dense_edge_mamba
  --skip-folds <all but 1_18_0's index>`, seed 42 for same-protocol
  reproducibility -- note MPS vs CUDA won't reproduce bit-identical
  weights even at the same seed, so treat it as representative, not the
  literal original run) and call `clf.predict_proba(X_test)` once fit
  completes; or (b) add real checkpoint persistence
  (`torch.save(model_.state_dict(), ...)` after `fit()`) to
  `SparseEvidenceGNNClassifier`/`common.py` so future runs like this one
  don't hit the same dead end.
- `Epilepsy/runpod_mamba_fast_image_brief.md` -- Task A/B in tree.
  Image **built and pushed** 2026-08-25 (GHA run 32843334915, ~31 min):
  `ghcr.io/noshore5/eeg_benchmarks-mamba:20260825-4760de0` and `:latest`.
  **Not yet run on a pod.** Next: kernel-vs-pscan parity
  (`scripts/dense_edge_mamba_cuda_kernel_parity.py`) then
  `--max-folds 1 --epochs 1` on a CUDA pod, record epoch_time vs the
  ~65s/epoch mambapy baseline. Fused kernel is windowed Temporal only;
  continuous `scan="chunk"` cannot use `selective_scan_fn` (no
  initial-state argument).
- Channel-subset-k sweep (`Epilepsy/Session_notes/2026_08_24/
  k_sweep_channel_subset_cuda.md`) still has the `ChannelSignalEncoder` in
  the graph (24-in MLP) -- not a clean same-model ablation against the
  encoder-free full-mesh runs. Needs an encoder-free rerun to close that
  comparison out.
- `continuous-cwt-mamba` paradigm (see "Right now" above) -- component
  pieces built and verified in isolation, `scan="chunk"` throughput path
  in place (item 1 below done). Real data pipeline not started.
  Next concrete steps, in the order they were being approached:
  1. ~~Investigate the throughput problem~~ -- done. Default is now
     `scan="chunk"` (carried-state pscan). See the 2026-08-25 session
     note. CUDA 3070 Ti re-measure of the updated probe is optional
     confirmation, not a blocker.
  2. Design + build the continuous CHB-MIT loading path: whole-recording
     CWT (not `_build_windowed_dataset`'s fixed windows), TBPTT chunk
     boundaries, and windowed labels (SPH/SOP-derived) read off the
     continuous timeline via `pool_continuous_edge_stream_to_windows`.
  3. A parallel `leave_one_seizure_out_*`-equivalent loop in
     `run_pipelines.py` for recording-level continuous sequences instead
     of independent per-window rows -- the existing LOSO functions assume
     per-window rows in `metadata`/`X` throughout, not a bolt-on.
  4. Only then: a real GPU LOSO run to compare against the
     `_DenseEdgeMambaTemporal` (windowed) baseline above.
