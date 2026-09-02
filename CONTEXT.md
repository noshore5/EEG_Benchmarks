# Repo context (read this first)

Living status doc, not a log. Says what's true *right now* -- for the
history/reasoning behind any of it, follow the pointers into
`Epilepsy/Session_notes/<date>/`. Update this file (not just the session
note) before ending a working session that changed the state below --
that's the whole point of it: this repo gets worked on from different
Claude/Grok shells that don't share context with each other, and this is
the one file meant to catch a new shell up without it re-reading
everything.

**`Epilepsy/NEGATIVES.md`** is the running ledger of every lever we've
*tried* and shelved (early-killed or run to negative/wash) -- check it
before re-running an experiment; add a row when you kill or conclude one.

**`AWS_INFRA.md`** -- shared AWS cloud state (S3 bucket, EC2 boxes, IAM).
Read it before touching the AWS account from **any** shell -- on-box or
driving remotely from a local Mac / Grok shell (the Mac has admin creds).
Skip it if you're only editing code / running tests; nothing in the
pipelines depends on it yet.
2026-09-01: **`scripts/eeg-run.sh`** -- fire-and-forget a run on an
on-demand box; on `rc==0` it commits the new result CSV(s) (+ optional
`--session-note`) straight to `origin/main` via `scripts/promote_results.sh`
and self-terminates; SNS email only on failure. Needs one-time
`scripts/setup_autorun_infra.sh` (deploy key in SSM, IAM, SNS topic).
`--cpu` works now; `--gpu` blocked on the pending quota bump. See
`AWS_INFRA.md` "Autonomous runs".
2026-09-02: **launch from anywhere w/o AWS creds** --
`.github/workflows/eeg-run.yml` (`gh workflow run eeg-run.yml -f name=... -f
cmd=...`, or GitHub mobile/browser -> Actions -> "eeg-run"). GH OIDC ->
role `eeg-gh-launcher`, nothing stored. One-time `scripts/setup_gh_launcher.sh`.
VERIFIED end-to-end 2026-09-02 (CPU smoke run: OIDC -> launch -> deps ->
run -> S3 -> promote no-op -> self-terminate, no orphan). `eeg-box`
(`i-083e3b55993a13c13`, t3.small, ~$15/mo) TERMINATED 2026-09-02 -- the
GH launcher replaces it. NB this account presents a customised OIDC `sub`
(`repo:noshore5@<id>/EEG_Benchmarks@<id>:...`); the trust policy matches
`:repository` exact + `:sub` wildcard.
Also fixed `eeg-run.sh` bootstrap bugs: missing `python3-pip`; `/usr/sbin`
not on PATH so `shutdown` failed -> orphaned box; `NAME` sanitize trailing
`-`; bare Ubuntu has no `python` (only `python3`) -> symlink shim.
2026-09-01: `eeg-box` role now has EC2 launch/terminate + SSM perms
(verified). GPU box still **not launched** -- setup only, no runs started.
2026-09-01: **`eeg-cpu-box`** (`i-0a6100d4c303f52a2`, c7i.2xlarge, 8 vCPU/
16 GB) stood up as a persistent stop/start CPU worker -- bootstrapped pip
env + chb01 data on its 50 GB volume, driven via an S3 job-runner (drop a
script at `s3://.../exports/eeg_box/jobs/next.sh`), self-stops after 30 min
idle. **Currently STOPPED** (~$4/mo volume only). First job done:
`godoy_tmc` prediction 6-fold CPU -- **~41.5 s/epoch on 8 vCPU, 58 min
wall**, mean AP 0.535 (within the MPS 5-seed band; timing run, not a new
board number). Peak RSS 14.1/16 GB -- the raw-classifier LOSO path leaks
per-fold, don't shrink the box below 16 GB. See `AWS_INFRA.md` "eeg-cpu-box
job runner" + `Session_notes/2026_09_01/aws_cpu_box_first_run_godoy_tmc.md`.
GPU run blocked: on-demand G quota `L-DB2E81BA` = 0 and spot G quota
`L-3819A6DF` = 0 (both increase-requests now `CASE_OPENED` -- AWS support
queue, not auto-approved; hours-to-days). EC2-Spot SLR exists; `g5` spot
capacity was out region-wide in us-east-1. One-shot `gpu_userdata.sh`
staged at `.../_setup/` for when a quota clears. eeg-box role got inline
policy `eeg-box-extra` 2026-09-01 (servicequotas read+request, spot-SLR
create, `ec2:CreateImage`/`ModifyInstanceAttribute`) -- eeg-box shell can
now check quota status itself.

**`Epilepsy/temporal_graph_mamba_math.md`** -- `pre` written as matrix
operations (Hermitian coherence matrix `Gamma(f,t)` -> off-diag ->
incidence-average -> per-channel selective SSM -> flatten -> logits),
plus how `hermitian_ssm` (eigendecomp of `Gamma`) and the `channel_cwt`
ablations relate to it. Written 2026-08-31 for the linear-algebra view.

**Last updated:** 2026-08-30, by Claude (Mac shell). On `main`. Uncommitted
working tree (all this session, all smoke-tested, commit once the running/
queued results land):
- `run_pipelines.py`: `HERMITIAN_SSM_PARAMS` only -- `encoder_mode` ->
  `"complex"`, `mamba_backend` -> `"mamba"` (mamba3 revert), `spectral_k`
  comment. (The old `_CG_MAMBANET_SHARED_PARAMS` other-shell edit is NO
  LONGER in the tree.)
- `hermitian_ssm_classifier.py`: +`_ComplexSpectralEncoder`
  (`encoder_mode="complex"`), +`_ComplexMatrixEncoder` (`encoder_mode="matrix"`).
- `cwt_gnn_classifiers.py`: +`_ComplexLinear`, +`temporal_graph_edge_complex`
  knob (built genuinely last in `SparseEvidenceGNNCore.__init__` -- flag-off
  RNG is bit-identical, verified).
- Queued runs use scratchpad monkeypatch wrappers (`run_post.py`,
  `run_matrix.py`, `run_edge_complex.py`) -- NO further source edits needed.

**RUN DISCIPLINE (2026-08-30, user directive):** running on this Mac is
free. There must ALWAYS be a job queued/running -- do NOT stop and idle
waiting for the user's approval on "what's next" (that wasted ~6h of
overnight compute once). Queue the next experiment and start it; the user
will stop it if they don't want it. Keep `scratchpad/queue_runner.sh`
topped up.

**Active state 2026-08-30:** the supervised hermitian architecture search
is effectively DONE (every lever negative-to-wash). Live direction: the
task **reframed as graph-state anomaly detection** -- train on interictal
only, flag preictal as departure from normal. Early smoke is the best
signal the direction has produced (see below).
- **encoder variants CLOSED NEGATIVE** (eigenvector 0.436 is the ceiling;
  projector 0.273, graph 0.169, evolution ~chance). `Session_notes/2026_08_29/`.
- **`mamba_backend="mamba3"`** (complex-diagonal SSM) -- 6-fold **NEGATIVE
  0.408 vs 0.436**. `Session_notes/2026_08_29/mamba3_and_perfreq.md`.
- **k axis CLOSED NEGATIVE:** k=6->12 6-fold **0.390 vs 0.436** (lost 5/6
  folds). k=6 cache key `9d6ad0d850b8b8f0`. k=12 cache was `293ac41c6e53675f`
  -- **deleted** (20 GB freed).
- **`encoder_mode="complex"` (`_ComplexSpectralEncoder`) -- KILLED at 3/6.**
  De-engineered eigenvector encoder (`_ComplexLinear`, lambda-scaled,
  mean-pool, flatten once at end; 3.35x fewer params). Trailing:
  0.291/0.318/0.312 vs 0.254/0.324/0.414. Same wash-to-worse as every
  lever. Methods story ("matches baseline, far less engineering") is the
  user's call; not a win on the number.
- **`encoder_mode="matrix"` -- KILLED at 3/6, NEGATIVE.** 0.213 / 0.290 /
  0.349 vs eigenvector 0.254 / 0.324 / 0.414. Consistently ~0.06 under
  eigenvector but well above `graph` (0.169): the complex channel-mix
  beats the flatten hack, but the **rank-6 truncation is the binding
  ceiling** -- no encoding trick on 6 modes reaches 0.436. High FAR.
- **`HermitianSSMAnomaly` v1 -- FAILED, fold 1 AP 0.029.** Next-token MSE
  collapsed to persistence (val MSE 0.0009, score had no spread). The
  smoke's ROC-AUC 0.80 was an artifact of stopping at 2 epochs pre-collapse.
  **v2 fix:** predict the *direction* of the h-step (h=8) token change,
  not the next token -- persistence now costs a real error of 1.0.
  Re-queued as `anomaly_v2`. `Epilepsy/pipelines/hermitian_ssm_anomaly.py`.
- **`ChannelCWTMambaClassifier`** (`Epilepsy/pipelines/channel_cwt_mamba.py`,
  NEW): the missing null -- per-channel CWT (no coherence at all) ->
  per-channel Linear -> ONE weight-shared Mamba over each channel's T
  sequence -> mean over channels -> 2 logits. **Per-channel feature is
  pre-matched (2026-08-30): raw power `|w|**2`, z-scored per freq, plain
  real Linear, NO phase, NO log.** Rationale: "pre" only ever feeds the
  model a normalized magnitude-derived quantity (coherence in [0,1]),
  never a per-channel phase or complex value; per-channel absolute phase
  is meaningless (only cross-pair relative phase is, and this model has
  no pair); "pre" doesn't log either. Band-matched to the k=6 hermitian
  cache (8-40 Hz, F_out=8, td=16). Own per-recording complex-CWT cache
  (reuses `_MorletCWT`; unchanged by the feature rewrite -- feature is
  derived downstream, no recompute). Drops "pre"'s n-hop message passing.
- **`channel_cwt` OOM'd 16 GB (rc=137) at training entry, 2026-08-30.**
  `batch_size=32` x 23 channels = 736 sequences all through the mambapy
  pure-PyTorch pscan at once. (A concurrent "Godoy" smoke in another shell
  likely added pressure.) **Fix in `channel_cwt_mamba.py`:** `batch_size`
  default 32->8; freq-axis chunking in `_ChannelCWTCache.ensure`
  (`_MorletCWT.transform` has no internal chunking -> `[C, nfreqs,
  n_padded]` blows up at nfreqs=60); new `store_dtype` kwarg (fp16 cache
  option). Numerically identical for the existing nfreqs=16 log cache.
- **`channel_cwt_broadband`** (`scratchpad/run_channel_cwt_broadband.py`,
  NEW; was briefly named "truong"): same classifier, broad evenly-spaced
  spectral grid -- **LINEAR** freq spacing (not log), 2-124 Hz, **60
  output bins**, 60/120 Hz mains notch kept. `freq_spacing="linear"` added
  to `_freq_grid` / `_ChannelCWTCache` (in the cache-key hash). Disk
  safety: linear-60 at td=16/fp32 projected ~79 GB with <60 GB free ->
  forced `td=64` (~4 Hz frames) + `store_dtype="float16"` -> ~3 GB.
- **Swap thrash 2026-08-30 -- CAUSE: too many GUI apps on 16 GB, not a
  leak.** After the day's jetsam kills, baseline commit (Chrome many
  tabs ~2.7 GB, Pylance 1.2 GB, Slack, Claude.app, several claude CLIs)
  sat near 16 GB, so every training job instantly avalanched into swap
  (peaked 35 GB). User is remote (phone, Claude Code only -- no terminal,
  cannot `sudo reboot`). **Fix without reboot:** quit Chrome/Slack/
  Claude.app/Spotify via `osascript -e 'quit app "X"'` -> swap 29 -> 6 GB,
  ~5.7 GB RAM free. If this recurs, quit GUI apps first.
- **Machine REBOOTED 2026-08-30 ~19:06** to clear the swap thrash (user
  did it from physical access). `/private/tmp` scratchpad was wiped by the
  reboot -- all wrapper scripts + queue runners + logs GONE. **Wrappers
  recreated in `_to_delete/` (gitignored, survives reboot):**
  `_to_delete/run_channel_cwt.py` so far; `run_channel_cwt_broadband.py`,
  `run_edge_complex.py`, `run_anomaly.py`, `run_post.py` still need
  recreating from CONTEXT descriptions / the transcript if used.
- **channel_cwt memory: ROOT CAUSE #2 FOUND (2026-08-30).** The pipeline
  itself is light; the waste was in data handling. `_build_continuous_
  dataset` loads all 41 chb01 raw recordings as float64 = **7 GB**, held
  by the LOSO loop for the whole run. The *real* `HermitianSSMClassifier`
  never touches `raw_x` (reads the prebuilt eigenpair cache off disk) so
  it's just dead weight it survives; `ChannelCWTMambaClassifier` *uses*
  `raw_x` then stacks its feat-stats + model on top of the resident 7 GB
  -> ~12 GB -> swap avalanche (killed 5x). Probe confirmed the CWT is
  cheap (+87 MB, 3 s/rec). **Fixes in `channel_cwt_mamba.py`:**
  (a) `_build_index` sets `rec["raw_x"] = None` right after each recording
  is cached -- frees the 7 GB progressively; nothing downstream of `fit`
  reads it (all 41 caches built during fold 1, folds 2-7 then run tiny).
  (b) CWT freq-chunk `fchunk = 1` (was 16 = whole grid = no-op; the ifft
  on a ~2^20-padded hour-long recording was ~7 GB by itself). Plus the
  earlier mmap-leak LRU (`_ram_cap = 8`) and `batch_size` 8.
- **channel_cwt memory: REAL ROOT CAUSE (2026-08-30) + config bug the
  user caught.** Instrumented trace showed it always died in the FIRST
  training batch with `xb=(64, 23, 480, 8)`. Two problems: (1) my
  `_ChannelCWTNet` used `mambapy.mamba.Mamba` DIRECTLY (naive full pscan,
  no row-chunking) -- NOT "pre"'s Mamba. `batch_size=64` (from
  HERMITIAN_SSM_PARAMS, passed into the ctor) x 23 channels folded into
  the batch axis = 1472 sequences -> `[1472, 512, 128, 16]` f32 = 6 GB
  per SSM tensor -> 25 GB swap avalanche. (2) even the hyperparams were
  wrong (d_model=64 vs "pre"'s 16). **Fix: `_ChannelCWTNet` now uses
  `cwt_gnn_classifiers._DenseEdgeMambaTemporal` -- "pre"'s EXACT temporal
  block** (d_model=16, d_state=16, d_conv=4, expand=2, n_layers=1,
  chunk_size=128), which row-chunks at 128 + gradient-checkpoints. Now a
  clean coherence-ablation of "pre": each channel = its own weight-shared
  sequence in the node-axis slot, CWT power replaces the scatter-meaned
  edge messages, graph message-passing dropped (mean over 23 node
  embeddings -> 2-class head). Component test: `[32,23,480,8]` fwd+bwd,
  maxrss 2.2 GB (was ~25 GB), 4010 params. Also fixed earlier this
  session: raw_x=None after caching (frees the 7 GB float64 dataset
  load), fchunk=1 in the CWT (ifft on ~2^20-padded recording was ~7 GB),
  mmap-leak LRU. Debug instrumentation stripped.
- **ALWAYS DEFAULT TO `--device mps` ON THIS MACHINE.** It's Apple
  Silicon; `_SHARED_ARCH_PARAMS` / all the pipeline param dicts already
  default `device="mps"`, and that's what every past "pre"/hermitian
  timing (e.g. "pre" ~60 s/epoch) was measured on. `--device cpu` is
  ~10x slower here (mambapy pure-PyTorch pscan over T=480). Only fall
  back to cpu for a specific MPS bug, and say so explicitly.
- **channel_cwt equivalence to "pre" -- 2 more gaps found & fixed
  (2026-08-30), both making epochs ~4x "pre"'s 60 s/epoch-on-MPS:**
  (a) wrapper forced `--device cpu` -> `--device mps` (see MPS rule
  above); (b) the `hermitian_ssm` LOSO loop does NOT subsample training
  negatives, but "pre"/dense-family do (5:1,
  `DEFAULT_NEGATIVE_TO_POSITIVE_RATIO`). Added
  `negative_to_positive_ratio=5.0` to `ChannelCWTMambaClassifier.fit`
  (interictal TRAIN windows only; predict_proba untouched) -> ~830
  train windows/fold instead of ~2800. Also bumped `_ram_cap` 16 -> 48
  (f16) so a shuffled epoch does zero redundant disk reads.
  First MPS run (pid 2734, cpu->mps only): epoch 1 219 s, epoch 2 241 s,
  val_auc ~0.86 -- killed to add the subsampling fix.
- **`_to_delete/run_channel_cwt.py` 6-fold RUNNING (pid ~3102) on MPS +
  5:1 subsample**, pre-matched temporal block (`_DenseEdgeMambaTemporal`,
  d_model=16, chunk=128), bs 64. Epochs 49-59 s (matched "pre").
  **RESULT: NEGATIVE, killed at 2 folds (2026-08-30).** fold 1_03_0
  auc_pr 0.149, fold 1_04_0 0.263 -- vs "pre" 0.792 / 0.830. ~0.6 AP
  below "pre" on both; healthy training curves that don't transfer.
  Clean one-var ablation (both use aggregate-to-23 + `_DenseEdgeMambaTemporal`,
  neither has message-passing at n_hops=1) -> the coherence edge
  representation carries essentially all the signal. Full writeup:
  `Session_notes/2026_08_30/channel_cwt_null_baseline_equivalence.md`.
- **`pre` reproduction RUNNING (pid 5562, MPS, monitor `bku1i61m0`)** --
  `_to_delete/run_pre_repro.py`, `temporal_graph_mamba --label-mode
  prediction`, untuned params. Clean same-machine baseline for the
  channel_cwt collapse + the next ablation. Judge on the 6-fold MEAN vs
  0.674, NOT per-fold (fold 1_03 AP swings 0.29-1.00 on unchanged config).
  - **run1 (pid 4938) killed at fold 1** by my monitor's swap watchdog
    (13GB line, too low -- raised to 26GB for run2). fold 1_03_0 came in
    **auc_pr 0.560** (MPS) vs historical 0.792 -- within fold-1 variance,
    not conclusive.
  - **fold 1 = 44s/epoch (matches "fast" memory). folds 2+ = ~95s/epoch,
    IO-bound** (CPU 77% idle, RAM full). NOT a misconfig -- config is
    identical (nfreqs=8, tds=16, 8-40Hz). The `event_mode="temporal_graph"`
    dense_edge cache is ~15 MB/trial (raw `[4,253,480,8]` edge stack, needed
    for the learned `temporal_edge_proj`); a full 6-fold = ~70 GB (2026-08-27
    note), exceeds 16 GB RAM -> page-cache thrash on folds 2-6. The
    original 0.674 was itself an overnight run stitched across 3 process
    restarts -- never a fast clean 6-fold. run2 grinding to completion
    (~4-5h) to match config exactly. fold 1_03_0 = **0.560** (both runs,
    reproducible on MPS) vs historical 0.792.
  - **PERF PATH (same model, worth doing): don't cache the [4,E,T,F] edge
    stack -- cache only the per-channel CWT (~1 MB/trial, fits RAM) and
    regenerate edges on MPS each forward.** In `coherence_threshold_mode=
    "fixed"` (what "pre" uses) all 4 edge components -- coherence, sinphi,
    cosphi, AND significance (a deterministic threshold indicator on
    |coh|, NOT a surrogate test; surrogates are a separate opt-in mode) --
    are deterministic functions of `(raw window, config)` per
    `dense_edge_cache.py`'s docstring. `compute_dense_edge_input` /
    `_build_dense_edge_input` is already device-agnostic torch (no numpy,
    no RNG). Recompute == same result (bit-identical at matched dtype/
    device; ~1e-6 drift if cache built on CPU). 2026-08-27 note found
    recompute FASTER than cache-load on CUDA (10s vs 26s epochs). This
    fixes the >RAM cache IO wall on the 16 GB machine. Not done tonight
    (run2 grinds as-is to match config); real TODO.
  - **QUEUED ablation: drop edge component 3 (significance).** Feed
    `temporal_edge_proj` only `[coh, sinphi, cosphi]` (3 not 4). Tests
    whether the fixed-threshold significance channel carries anything.
    Distinct from `temporal_graph_edge_complex` (which drops it AND swaps
    in `_ComplexLinear`) -- this is the clean 3-vs-4 ablation.
- **ablation 1 BUILT + smoke-passed, QUEUED behind pre_repro**
  (`_to_delete/run_channel_cwt_hops.py`, MPS). `channel_cwt_mamba.py`:
  `_HopMessagePassing` (replicates `SparseEvidenceGNNCore._propagate_hops`
  -- MLP msg on [h_dst,h_src] + scatter-add + GRUCell, complete graph
  23*22=506 directed edges), `_ChannelCWTNet` gains `n_hops` (1=off,
  bit-identical A/B for 1 vs 2), classifier gains `mamba_n_hops` (wrapper
  sets =2). Per-channel CWT power + 1 round cross-channel mixing, still NO
  coherence. Isolates "cross-channel info in general" from "coherence
  math specifically". n_hops=1/2 nets are RNG-identical to each other;
  both differ by a tiny RNG shift from the already-run channel_cwt
  (0.149/0.263) which had no `self.hops` -- immaterial vs the 0.6 gap +
  fold variance. Runs one-at-a-time on MPS after pre_repro.
  **Wrappers in `_to_delete/` (gitignored)** -- scratchpad wiped by the
  reboot; only `run_channel_cwt.py` + `probe_ensure.py` recreated.
  No queue runner.
  (channel_cwt OOM history -- all fixed, run completed NEGATIVE, see
  above + the session note. Superseded paragraphs removed 2026-08-30.)
- **QUEUE (2026-08-30, one MPS job at a time -- 16 GB RAM OOMs (rc=137)
  any two heavy jobs at once):**
  1. **pre_repro -- DONE 2026-08-31 05:40. REPRODUCES.** 6-fold mean AP
     **0.644** vs historical 0.674, ROC-AUC **0.973** vs ~0.94 (better
     global ranker). Per-fold (this / historical): 1_03 .560/.792,
     1_04 .811/.830, 1_15 .416/.331, 1_16 .985/.996, 1_18 .811/.802,
     1_26 .283/~.29. Entire 0.03 gap = fold 1_03 variance; every other
     fold matches or beats. -> 0.674 is real, "pre" sits ~0.64-0.67.
     Confirms channel_cwt (0.149/0.263) collapsed ~0.5 below a clean
     same-machine baseline: coherence carries the signal.
     CSV: `results/temporal_graph_mamba/prediction/*_20260830-211240.csv`.
     LOG (this shell's scratchpad): `.../pre_repro.log`. MPS. Took ~8.5 h
     (IO-bound folds 2-6, see below). Full writeup:
     `Session_notes/2026_08_31/pre_repro_6fold.md`.
  2. **ablation 1 hops** -- `_to_delete/run_channel_cwt_hops.py`, built +
     smoke-passed. per-channel CWT power + `mamba_n_hops=2` cross-channel
     mixing, no coherence. Tests cross-channel-info-in-general vs coherence.
     **RUNNING 2026-08-31 05:41, pid 29039, MPS**, monitor `bkg4aeuk6`.
     LOG `scratchpad/channel_cwt_hops_20260831-054113.log`. The armed
     `queue_runner.sh` DID fire at pre_repro exit but launched WITHOUT
     `PYTHONPATH` -> `ModuleNotFoundError: No module named 'Epilepsy'`,
     died instantly (log `channel_cwt_hops_20260831-045945.log`).
     Relaunched by hand with `PYTHONPATH=. ... --label-mode prediction`.
     **FIXED 2026-08-31 (shell "The Que"):** all `_to_delete/run_*.py`
     wrappers now `sys.path.insert(0, <repo root>)` at the top;
     `queue_runner.sh` rewritten (generic `<pid> <wrapper> [args]`, no
     PYTHONPATH needed). "The Que" briefly double-launched a 2nd hops
     (pid 29317) before seeing this shell's note -- killed it + its
     watcher; **only pid 29039 runs.** A fresh watcher (pid 29471) polls
     29039 and fires the item-2b SMOKE (`run_temporal_graph_recompute.py
     --smoke --max-folds 1`) on exit -- do not double-arm.
     **DONE 2026-08-31 08:06. NEGATIVE.** 6-fold mean AP **0.173**
     (ROC-AUC 0.878), per-fold .150/.190/.180/.185/.189/.147 -- DEAD FLAT
     ~0.18 every fold regardless of pre's fold difficulty. At/below the
     no-hops channel_cwt (.149/.263 on its 2 folds). vs pre_repro 0.644.
     -> learned complete-graph message passing on per-channel spectral
     summaries recovers NOTHING; extra params marginally hurt on the ~30-
     window budget. **CONCLUSION: it's coherence specifically -- the
     cross-spectrum `S_ij = w_i conj(w_j)` and its phase `arg S_ij` --
     not cross-channel info in general.** CSV
     `results/hermitian_ssm/prediction/*_20260831-054118.csv`. Added to
     NEGATIVES.md + `Session_notes/2026_08_30/channel_cwt_null_baseline_
     equivalence.md` + `temporal_graph_mamba_math.md` sec 5.
  2b. **CWT-recompute perf change -- BUILT 2026-08-31 (Mac shell "The Que"),
      smoke ARMED behind hops (watcher pid 29471). Slotted after item 2.**
      Wrapper: `_to_delete/run_temporal_graph_recompute.py` (monkeypatches
      `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS["dense_edge_source"]=
      "recompute"`). Code: `dense_edge_source="disk_cache"|"recompute"` on
      `SparseEvidenceGNNClassifier` (+ `Literal`, __init__ validation
      scoping it to fixed / tds>1 / no time_avg / no cwt_encoder / no
      channel_subset_k); `dense_edge_cwt_predownsample` on
      `SparseEvidenceGNNCore` (baked in by `_build_model` when recompute);
      `_coi_valid_mask` gains `time_stride`; `_build_dense_edge_input`
      honours predownsample (skip `_downsample_dense_edge_time`, native
      `n_time_in`); new `_prepare_features_recompute` +
      `_predownsample_cwt` / `_expand_pooled_cwt_to_edges` /
      `_recompute_dense_edge_helper` / `_pooled_cwt_cache_dir`
      (`<dense_edge_cache_dir>/recompute_pooled_cwt`);
      `dense_edge_cache_key` gains `dense_edge_source` -- appended to the
      config tuple ONLY when != "disk_cache" so existing on-disk entries
      (pre_repro's cache) hash identically. CPU unit checks pass
      (`scratchpad/test_recompute_paths.py`): key namespacing, __init__
      guards, pooling math, edge rebuild `[B,4,E,T',F]` finite + coh in
      [0,1], COI stride. NEXT: watcher 29471 auto-runs the smoke on hops
      exit -> check its log (`run_temporal_graph_recompute_*.log`), then if
      clean launch the full `python3 _to_delete/run_temporal_graph_recompute.py`
      6-fold; judge mean AP vs 0.674 AND log s/epoch vs pre_repro's
      IO-bound 85-370 s (folds 2-6) to confirm the perf win.
      **SMOKE FIRED 2026-08-31 08:06 (pid 32705) AND CRASHED** --
      `ValueError: Expected freqs shape (32, 8) or (8,), got (29, 8)` in
      `_batched_freqs` <- `compute_dense_edge_input` <-
      `_expand_pooled_cwt_to_edges` <- `_prepare_features_recompute`
      (cwt_gnn_classifiers.py:3826/4791/6622/6671).
      **ROOT CAUSE + FIX 2026-08-31 08:10 ("The Que"):** not batch size per
      se -- `_prepare_features_recompute` cached `self._recompute_freqs` as
      the *batched* `(B,F)` row returned by `super()._prepare_features`
      (B=29 on some fold's last batch); a later cache-hit batch (B=32) then
      fed that stale `(29,8)` to `_batched_freqs`, which only accepts
      `(B_now,F)` or `(F,)`. Fix: store the 1-D `(F,)` grid --
      `_f[0] if _f.ndim==2 else _f`. CPU unit checks still pass.
      RE-ARMED: watcher pid 32870 polls godoy_tmc (pid 32725, another
      shell's MPS job) and re-runs the smoke on its exit. Do not double-arm.
      **SMOKE PASSED 2026-08-31 08:22** (godoy done 08:22, smoke pid 33571):
      clean end-to-end, `[recompute-cwt cache] N/N reused` every batch,
      edges rebuilt in-forward, ~4 s/epoch, no freqs error. CSV
      `*_20260831-082209.csv`.
      **FULL 6-fold RUNNING 2026-08-31 09:11, pid 33936, MPS.** LOG
      `scratchpad/run_temporal_graph_recompute_FULL_20260831-091127.log`.
      Judge: (a) mean AP vs 0.674 / pre_repro 0.644; (b) s/epoch vs
      pre_repro's IO-bound 85-370 s folds 2-6 -- THIS is the point of 2b.
      **PARTIAL 2026-08-31 09:56 (fold 3/6 running):** PERF WIN CONFIRMED
      -- ~40-45 s/epoch EVERY fold (fold-1 e1 44.7s then ~42s; fold-2/3
      e1 ~88-92s cache-warm then ~42-45s) vs pre_repro's 85-370 s on folds
      2-6. recompute-cwt cache 100% reused every batch, fits RAM, no
      page-cache thrash. BUT **AP trending DOWN**: 1_03 auc_pr 0.396
      (pre_repro .560), 1_04 0.404 (pre_repro .811). Training curves
      healthy (val_roc_auc 0.98+) -- it's test-set AP. Non-bit-identical
      coarse-grid smooth/xspec looks like it materially degrades the edge
      features. Wait for all 6 before final call, but leaning: fast but
      worse -> NOT a drop-in replacement for the leader.
      **DONE 2026-08-31 10:57. NEGATIVE (accuracy).** 6-fold mean AP
      **0.469** (ROC-AUC 0.942, k-of-n 4/6) vs pre_repro 0.644 / hist
      0.674 -- down **0.175**. Per-fold (2b / pre_repro): 1_03 .396/.560,
      1_04 .404/.811, 1_15 .279/.416, 1_16 .682/.985, 1_18 .842/.811,
      1_26 .208/.283 -- worse on 5/6, only 1_18 ~matches. Training curves
      healthy every fold (val_roc_auc 0.98+) -> pure test-set
      generalisation loss from the non-bit-identical coarse-grid
      recompute (`smooth@480 != smooth@7680-then-pool`,
      `xspec(downsample(w)) != downsample(xspec(w))`).
      **PERF WIN real but moot:** ~42-57 s/epoch EVERY fold (warmup
      ~88-97s) vs pre_repro's IO-bound 85-370s folds 2-6; recompute-cwt
      cache 100% reused, ~0.7 MB/trial, fits RAM.
      **VERDICT: keep `dense_edge_source="disk_cache"` as production. The
      recompute path is a dead end -- it buys a 2-6x epoch speedup at the
      cost of 0.175 mean AP.** `dense_edge_source` param + code stays
      (guarded, default disk_cache, no effect unless opted in); wrapper
      `_to_delete/run_temporal_graph_recompute.py` kept for the record.
      CSV `*_20260831-091141.csv`. Added to NEGATIVES.md.
      --- ORIGINAL NOTES ---
      Facts pinned 2026-08-31 (Mac shell "The Que"), config
      `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS` (the AP 0.674 leader):
      window 30 s @ 256 Hz = **7680 samples native**; `nfreqs=8` (8-40 Hz);
      `cwt_resample_n_time=None` -> **CWT cache stored at native T=7680**
      (`[2,23,7680,8]` ~11 MB/trial); `dense_edge_time_downsample=16` ->
      **dense-edge cache at T=480** (`[4,253,480,8]` full mesh
      ~**15 MB/trial** -- E=C*(C-1)/2=253 undirected, matches
      save_dense_edge docstring; >RAM across the 6-fold set = the IO wall);
      `coherence_threshold_mode="fixed"` (determinism precondition holds);
      `time_averaged_graph=False`.
      Streaming classifier calls `_precompute_dense_edge_inputs` once per
      BATCH every epoch (comments ~7448) -> disk cache hit every batch =
      the wall.
      **DECISION (user: "not worried about bit identical"): build VERSION
      B, run as a fresh experiment vs 0.674 (NOT a drop-in perf swap).**
      `_build_dense_edge_input` order is cross-spectrum(253 edges) ->
      `_smooth` (sep. Gaussian, stride 1) over full native 7680 ->
      `_downsample_dense_edge_time` x16 -> 480. Version A (cache native
      CWT, recompute the real pipeline) is bit-identical but the
      recompute is the "94.8% of forward() time" smooth-over-7680 paid
      every epoch -- likely NOT faster on MPS. Version B: cache CWT
      **pre-downsampled x16** (`[2,23,480,8]` ~**0.7 MB/trial**, ~1 GB/fold,
      fits RAM ~44x smaller); in-forward do only cross-spectrum + smooth
      + COI + stack at T=480 (cheap: ~62M complex elems/batch, few ms MPS);
      SKIP `_downsample_dense_edge_time` (CWT already coarse). NOT
      bit-identical -- smooth@480 != smooth@7680-then-pool, and
      xspec(downsample(w)) != downsample(xspec(w)). COI: keep the TRUE
      `n_time_in=7680` (stash it) with `T_out=480` so `_coi_valid_mask`'s
      cone scaling stays physical.
      **Call sites (cwt_gnn_classifiers.py unless noted):**
      - `_precompute_dense_edge_inputs` (~7231) -- branch on new knob;
        when "recompute": mean-pool `w_real/w_imag` over T by
        `dense_edge_time_downsample` BEFORE `compute_dense_edge_input`,
        cache THAT (new small cache namespace, not the `[4,E,T,F]` disk
        cache), return the freshly built `[B,4,E,480,8]` (do NOT persist
        the big stack).
      - `_build_dense_edge_input` / `compute_dense_edge_input` (~4596/
        ~4741) -- add `cwt_predownsampled: bool`; when True skip the
        trailing `_downsample_dense_edge_time`, pass real `n_time_in` to
        `_coi_valid_mask` (~4667).
      - `_temporal_graph_node_states(dense_edge_raw)` (~5206) -- unchanged;
        still receives full `[B,4,E,480,8]` (temporal_edge_proj needs all
        E, all T, 4*F). Only the *persistence* changes, not the shape.
      - `run_pipelines.py`: add `dense_edge_source="disk_cache"|"recompute"`
        to `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS`; plumb like
        `resolve_disable_disk_cache` (~163) / `shared_dense_edge_cache_dir`
        (~1427/1610/1716).
      - CWT reuse: native `w_real/w_imag` already come from the CWT window
        cache (`cwt_window_cache.py`), invariant to per-fold
        `scale=1/(std+eps)`; the downsample + xspec are the only new work.
      - Wrapper: `_to_delete/run_temporal_graph_recompute.py` (monkeypatch
        `dense_edge_source="recompute"` into the params dict, like the
        other `_to_delete` wrappers) + A/B timing harness (1 fold each
        mode, compare s/epoch + final fold-1 AP).
  3. **significance 3-vs-4 ablation -- BUILT 2026-08-31 ("The Que"),
     smoke ARMED behind 2b (watcher pid 34646).** Wrapper
     `_to_delete/run_edge_drop_significance.py` (monkeypatches
     `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS["temporal_graph_edge_drop_
     significance"]=True`). Code: `temporal_graph_edge_drop_significance:
     bool` on Core + Classifier (+ `_build_model` plumb). When True:
     `temporal_edge_proj` = `Linear(3*nfreqs -> edge_dim)` and
     `_temporal_graph_node_states` slices `dense_edge_raw[:, :3]` in-forward
     (cproj path already ignored ch3). **Runs on disk_cache path** (NOT 2b's
     recompute) -- bit-identical edge features, only ch3 dropped; per-epoch
     ~ pre_repro's IO-bound wall-clock. CPU checks pass (in_features 24 vs
     32, node_states forward OK both ways). Judge 6-fold mean AP vs 0.644 /
     0.674.
     **DONE 2026-08-31 12:39. MILD NEGATIVE / noisy.** 6-fold mean AP
     **0.542** (ROC-AUC 0.957, hit 6/6 raw + 5/6 k-of-n) vs pre_repro
     0.644 -- **down 0.10**, but MIXED per-fold (item3 / pre_repro):
     1_03 .237/.560, 1_04 .593/.811, 1_15 **.568/.416**, 1_16 .732/.985,
     1_18 **.872/.811**, 1_26 .251/.283 -- worse on 3, BETTER on 2, ~tie 1.
     The deficit is ~2x the fold noise, concentrated in 1_03. Well above
     2b's 0.469, so the significance channel matters more than the
     coarse-grid recompute did -- but the muddiness is exactly why the
     COI-mask follow-up (item 3b) is worth running. CSV
     `*_20260831-110945.csv`. LOG `run_edge_drop_significance_FULL_
     20260831-110934.log`. Added to NEGATIVES.md.
  3b. **`dense_edge_ch3="coi_mask"` -- BUILT 2026-08-31 ("The Que"),
      NOT yet queued (user: godoy 5-seed sweep goes first).** Tests the
      hypothesis from item 3: significance's only non-redundant content in
      fixed mode is the COI boundary (`sig_masked = coh/thr - coi_valid`),
      so replace stack component 3 with the RAW `coi_valid` mask -- same
      4-vector width, ch3 now carries only the cone boundary, zero
      redundancy with ch0. Code: `dense_edge_ch3: Literal["significance",
      "coi_mask"]` on Core (ch3 swap in `_build_dense_edge_input` after
      the COI-mask block) + Classifier (validation: fixed mode +
      coi_enabled + not drop_significance) + `_build_model` plumb +
      `dense_edge_cache_key` (namespaced `ch3=coi_mask`, appended only
      when != "significance" so pre_repro's cache is untouched -- but note
      a coi_mask run recomputes its own [4,E,T,F] stack from scratch, ~8h
      like pre_repro fold 2-6). CPU checks pass (cache-key namespacing,
      3 validation raises, build_model keeps 4-vector). Wrapper
      `_to_delete/run_edge_coi_channel.py` still TODO.
  3c. **`channel_subset_k` now works with "pre" -- FIXED 2026-08-31 (this
      shell), QUEUED (run after item 3b / the parked post jobs, or
      sooner on a free MPS slot -- it's cheap).** `--channel-subset-k K` was plumbed to
      `event_mode="temporal_graph"` but `_temporal_graph_node_states`
      (cwt_gnn_classifiers.py ~5348) summed ALL 253 edges' messages -- dead
      (non-clique) edges arrive as scattered zeros, but `temporal_edge_proj`
      / `sparse_message_mlp` biases turn a zero input into a nonzero
      CONSTANT, polluting every node's mean, and it divided by the fixed
      full-mesh `temporal_node_in_degree`. **Fix:** derive the live-edge
      mask from `dense_edge_raw` (dead edge = exactly all-zero), zero the
      dead messages, divide each node by its per-sample live in-degree.
      Bit-identical at full mesh (verified via `torch.equal` on an all-live
      sample); both `pre` and `post` aggregate branches. 29/30 dense-edge
      tests pass (1 fail is pre-existing `--pipeline` choices staleness).
      **Smoke first** (fast, 1-2 folds, 5 interictal recs, capped epochs):
      `python Epilepsy/run_pipelines.py --pipeline temporal_graph_mamba
      --label-mode prediction --device mps --channel-subset-k 12 --smoke`
      -- confirms the live-clique cache builds (look for `[4,66,...]` not
      `[4,253,...]` edge tensors) and a fold completes without NaN.
      **Then the real run:** `python Epilepsy/run_pipelines.py --pipeline
      temporal_graph_mamba --label-mode prediction --device mps
      --channel-subset-k 12` (full 6-fold; ~1-2h, cache fits RAM so no
      IO wall like pre_repro). Compare mean AP to pre_repro 0.644 /
      hist 0.674; log a NEGATIVES.md row + session note. Try K=8 next if
      K=12 is within fold noise. (K=12 -> 66 edges cached vs 253, ~4x
      smaller; K=8 -> 28, ~9x.) NB drops whole channels not just edges -- non-
      selected nodes ~zero per window, live clique is per-window, so the
      `Linear(184,2)` readout sees a varying active set (modeling risk, not
      a bug). Selection = deterministic cosine top-k on the raw per-window
      signal. Uncommitted edit to `cwt_gnn_classifiers.py` -- coordinate
      with "The Que" (also editing that file, items 3/3b).
  4. parked post-based jobs (other thread): `run_edge_complex.py` (~11 h)
     -> `run_anomaly.py` v2 (~3-4 h) -> `run_post.py` (~11 h). Wrappers
     need recreating in `_to_delete/` (scratchpad wiped by reboot).
  - `continuous_cwt_mamba` 6-fold: **not launched** (user, see below).
  5b. **godoy_tmc 5-seed sweep -- DONE 2026-08-31.** Seeds 42-46 6-fold
      mean AP: 0.619 / 0.548 / 0.590 / 0.550 / 0.470 -> **mean 0.556,
      sample std 0.056** (range 0.47-0.62). The single-run 0.619 was the
      TOP of the range: seed 42 drew a lucky `1_03` (AP 0.549 vs ~0.15
      for the other four seeds). All seed variance lives in folds `1_03`
      and `1_26`; `1_16` perfect every seed, `1_18` perfect on 4/5.
      godoy_tmc on MPS is deterministic given seed (seed-42 sweep run was
      bit-identical to `-080715`). Honest headline **0.556 ± 0.056** --
      a clear 3rd, ~0.08-0.09 AP behind "pre" (0.644) / "post" (0.639),
      NOT a near-tie. Session note
      `Session_notes/2026_08_31/godoy_tmc_seed_sweep.md`; comparison-doc
      board row + caveat updated. CSVs seed 42 `-124010`, 43 `-125636`,
      44 `-131322`, 45 `-133242`, 46 `-135037`.
  5c. **godoy_tmc d_model capacity mini-sweep -- DONE 2026-08-31.**
      Helper `_to_delete/_godoy_run_one.py <d> <heads> <seed>`
      (monkeypatches `_ConvTokenizer` last conv block + `n_heads`).
      Cut to **seed 42 only** at d16 and d64 (d16 gated seeds 43/44 off).
      Per-fold AP seed 42: d16 **0.519** (`*-140959.csv`), d32 **0.619**
      (`*-080715`), d64 **0.498** (`*-142719.csv`). Inverted-U, peak at
      the paper's d_model=32; both directions lose ~0.10 AP; d64 overfit
      the val split (val_roc_auc ~0.998, lost held-out seizure). No d8
      run (gate not met). **Capacity axis closed for godoy_tmc -- it is
      already at its best config, width does not rescue it toward "pre".**
      Session note: `Session_notes/2026_08_31/godoy_tmc_seed_sweep.md`
      (d_model capacity sweep section appended).

  5d. **MISSION (2026-08-31): faithful-to-"pre" but RAM-resident.**
      Goal = a `temporal_graph_mamba` "pre" variant whose dense-edge
      cache fits in 16 GB RAM (no page-cache thrash like item 2b's
      recompute), with mean AP as close as possible to pre_repro 0.644 /
      hist 0.674. Lever in flight: `--channel-subset-k` (top-K channels
      by `abs_cosine` -> C(K,2)-edge clique; live-clique cache only
      builds surviving edges). K=12 -> 66/253 edges (~3.8x cache cut,
      most faithful subset that fits); K=4 -> 6/253 edges (~42x, floor).
      **K=12 KILLED after fold 3 (2026-08-31, run-discipline gate) --
      NEGATIVE.** folds 1_03/1_04/1_15 AP 0.296/0.528/0.460 vs pre_repro
      0.560/0.811/0.416; 3-fold mean 0.428 vs 0.596, down ~0.17. Dropping
      187/253 edges is amputation. K=4 cancelled with it. NEGATIVES.md
      row logged. Log `_to_delete/pre_csk12_real_20260831-145254.log`.
      **Recomputed the RAM math (earlier 61 GB figure was WRONG -- that's
      the accumulated on-disk cache across many runs, not one run's
      working set).** One full-mesh 6-fold run: ~14 GB disk cache; the
      DataLoader re-streams only ~700 subsampled train windows/epoch =
      **~11 GB/epoch at fp32** -> marginal on the 16 GB Mac (mild thrash).
      **fp16 halves it to ~5.5 GB/epoch -- fits, zero edges dropped.**
      COI masks only 2.1% of (t,f) cells at 8-40 Hz (measured), so
      cell-sparsity is not worth it here. Significance channel (ch3) in
      fixed mode = `coi_valid * (coh/0.9 - 1)` -- exactly recomputable
      from ch0 at load, 25% more if dropped from storage (v2).

      **DISK CLEANUP 2026-08-31 ~15:55 (was 97% full, fp16 build ENOSPC'd
      twice):** deleted (all regenerable) -- item-2b `recompute_pooled_cwt`
      (~2.9 GB), all `channel_subset_k` compact entries (~13.6 GB, whole
      subset-k line rejected), `~/mne_data/hermitian_ssm_cache` (24 GB,
      encoder investigation CLOSED), and **fp32 full-mesh dense_edge_cache
      entries >12 h old (~59 GB)**. That last one means the **parked "post"
      jobs will have to regenerate their dense-edge cache** (~hours of
      precompute) when/if they run. Disk now 55% used, 87 GB free.

      **RUNNING 2026-08-31 ~16:10, MPS:** `_to_delete/run_pre_fp16_cache.py`
      (seed 42) -- full-mesh "pre" 6-fold with the dense-edge cache stored
      float16 (monkeypatches the cache layer, `-fp16` key suffix so it
      never collides with the fp32 cache, upcasts on load so downstream is
      unchanged). Log `_to_delete/pre_fp16_seed42_*.log`. Verified: fp16
      npz = 7.8 MB (half of 15.5), dtype float16, all 253 edges. Compare
      mean AP to pre_repro 0.644 -- MUST reproduce it (fp16 is pure
      compression, byte-exact after upcast). If it holds: this is the
      RAM-resident faithful deliverable; then item 3b + drop-ch3 for more.

      **UPDATE 2026-08-31 ~23:10 -- see `_to_delete/OVERNIGHT_STATUS.md`
      for the full running handoff.**
      - ROOT-CAUSE of the earlier fp16 thrash: the repo NEVER calls
        `torch.mps.empty_cache()`. On Apple Silicon the MPS reserved pool
        IS system RAM; without it, `_precompute_dense_edge_inputs`' GPU
        transients become permanent resident RAM -> swap. FIX (works,
        verified clean 42-56s/epoch): `_to_delete/run_pre_fp16_*` wrappers
        now monkeypatch `empty_cache()` into `_save_fp16` + a `_sync_device`
        wrapper. Consider upstreaming (ask user first).
      - fp16 seed 42 6-fold: **mean AP 0.617 vs pre_repro 0.644** (-0.027),
        one-sided (every fold <= pre_repro). Per-fold:
        .459/.791/.375/.985/.807/.284 vs .560/.811/.416/.985/.811/.283.
        1_16_0 bit-identical, 1_18_0/1_26_0/1_04_0 within noise; gap is
        1_03_0 (-0.10, a fold that swings 0.29-1.00) + 1_15_0.
      - fp16 SEED SWEEP (2026-09-01, `_to_delete/fp16_queue.sh`, one MPS
        job at a time, "-fp16" cache reused, no rebuild). 6-fold mean AP:
        seed42 .617 / seed43 .490 / seed43@30ep .464 / seed44 .609 /
        seed45 .510. vs pre_repro .644 (historical .674). Spread ~0.06
        std swamps the seed42 -0.027; NO isolated fp16 penalty visible.
        seed43 is a genuinely bad val-split/init draw (30 epochs did not
        rescue it, .464). "Stable" fold 1_16_0 stays .93-1.00 every seed;
        all volatility is in the tiny noise folds 1_03_0 / 1_04_0.
        Per-seed logs `_to_delete/pre_fp16_seed*_*.log`, means in
        `_to_delete/fp16_queue.out`. Plan: seeds 46/47, then `rm
        *-fp16.npz` (~31 GB), then forever fp32 seed sweep (Phase B,
        `run_pre_fp32_emptycache.py`) for matched fp16-vs-fp32 rows.
        Full write-up: `Epilepsy/Session_notes/2026_09_01/fp16_dense_edge_cache_seed_sweep.md`.
        Phase B fp32 sweep: DONE at seeds 42/43/44/45 (fp16-vs-fp32 =
        NULL, split-noise not a real dtype effect). `-fp16` cache
        deleted 09-01 (~31 GB freed); fp32 no-suffix files kept.
        30-epoch hypothesis = NEGATIVE. NEGATIVES.md seed-exposure row
        still TODO.
      - **PER-FOLD MEMORY LEAK FOUND + FIXED 09-01.** `temporal_graph_mamba`
        prediction (`leave_one_seizure_out_prediction`, run_pipelines.py
        ~L1672 loop) never released a fold's `clf` (model/optim/feature
        tensors) before the next fold -- footprint climbed monotonically
        across the 6-fold LOSO loop (fold 1 ~50s/epoch -> fold 4
        ~4000s/epoch as the 16 GB Mac hit swap; `footprint <pid>` showed
        phys_footprint 25 GB / peak 28 GB with RSS only 289 MB). Fix:
        explicit `del clf,...; _gc.collect(); torch.mps.empty_cache()` at
        end of each fold (committed on `main`). This was the real cause of
        every "page-cache thrash on folds 2-6" note above -- NOT just
        cache size. fp32 seed-43 job (pid 34495) + old val0 driver
        (41150) KILLED mid-fold-5 09-01 16:34; seed-43 fp32 partial
        discarded (fp16-vs-fp32 already null).
      - **val=0 sweep** -- epochs {12,20} x seeds {42..46},
        `--validation-split 0` (train 100% each fold, no early stop, no
        ckpt restore) to kill the val-split variance driver. Leak fix
        confirmed holding (flat ~55s/epoch through all 6 folds).
        Job def: `temporal_graph_mamba` "pre" prediction, chb01 LOSO,
        `--epochs <N> --validation-split 0 --seed <S>` (+ `--device mps`
        on Mac / `--device cpu` on AWS). Mac wrapper `_to_delete/run_pre_val0.py
        <seed> <epochs>` (MPS-only monkeypatch -- do NOT use on AWS; call
        run_pipelines directly with --device cpu there).
        STATUS (no compute running as of 09-01 20:00, user paused it):
          - ep12 seed42: DONE, mean AP **0.627** (88aff75).
          - ep12 seed43: DONE, mean AP **0.569** (aa2a60d).
          - ep12 seed44: DONE, mean AP **0.540** (d7f7683).
          - ep12 seed{45,46}: NOT RUN (Mac, resume via
            `_to_delete/run_pre_val0.py <seed> 12`).
          - **ep20 seed{42..46}: UNCLAIMED -> AWS cpu box to run.**
        Finding so far: seed spread at val=0 is 0.627/0.569/0.540 --
        pure optim noise (no split-luck channel), SAME magnitude as the
        val-split=0.2 sweep. Dropping the split did NOT tighten the error
        bar; it just moved variance from split-luck to init/SGD-luck.
        Per job: 6-fold mean AP over the `average_precision` column of
        `prediction_leave_one_seizure_out_*.csv`; commit both prediction
        CSVs. Compare 6-fold-mean spread across seeds vs the val-split=0.2
        sweep's ~0.05-0.13; ep20 rows are the fair comparison to the
        fp32 seed-42 ~0.64 anchor.
      - fp16 `_load` upcasts to fp32 on load, so fp16 halves DISK + the
        DataLoader re-stream READ but NOT the resident fp32 edge tensor.
      - NEXT (user priority): AWS S3 upload of `~/mne_data/dense_edge_cache`
        (~36 GB, $100 credit, be frugal, egress ~$3.24 so no yo-yo) to
        free local disk WITHOUT deleting the fp32 partial cache, then an
        fp32 6-fold "tracking" run + the nfreqs=16 band probe
        (`run_pre_fp16_nfreqs16.py`, DEFERRED, ~63 GB cold rebuild).
  4. parked post-based jobs (other thread): `run_edge_complex.py` (~11 h)
     -> `run_anomaly.py` v2 (~3-4 h) -> `run_post.py` (~11 h). Wrappers
     need recreating in `_to_delete/` (scratchpad wiped by reboot).
  - `continuous_cwt_mamba` 6-fold: **not launched** (user, see below).
  5. **`godoy_tmc` real 6-fold** chb01 prediction LOSO -- separate thread,
     plain CLI, no scratchpad wrapper needed (no source patching). See
     `Epilepsy/pipelines/godoy_tmc_classifier.py` (TMC-T, built from
     arXiv:2209.11172, commit `8e3945d`) and its addition to
     `_build_argument_parser`'s `--pipeline` help text for the full
     reasoning. Small model (1 Transformer layer, d_model=32) --
     expected fast on CPU/MPS, comparable to `dbconformer`/`slimseiz`'s
     own per-fold wall-clock, nowhere near `continuous_cwt_mamba`'s or
     `hermitian_ssm`'s. Run command:
     ```
     python3 Epilepsy/run_pipelines.py --pipeline godoy_tmc --label-mode prediction --device mps
     ```
     (swap `--device cpu` if MPS is busy with another queue item at the
     time). No `--subjects` override needed -- unlike `cg_mambanet`,
     `godoy_tmc` has no fixed channel montage, defaults to subject 1
     already matching this repo's other single-subject LOSO rows anyway.
     Results land under `Epilepsy/results/godoy_tmc/prediction/`. Once
     it completes, add its row to `pipeline_comparison_gru_mamba_
     dbconformer_slimseiz.md` next to the CG-MambaNet addendum -- same
     table, same protocol, same "not the paper's own reported number"
     framing (Godoy et al.'s own eval was a random 80/20 split, not
     LOSO -- see the classifier's module docstring).
     **DONE 2026-08-31 08:07 (other shell, pid 32725). 6-fold mean AP
     0.619**, ROC-AUC 0.968, f1 0.532, recall 0.811, FAR/h 8.66, hit 5/6
     raw + 5/6 k-of-n. Per-fold AP .549/.291/.311/1.000/1.000/.564 --
     perfect on 1_16 AND 1_18 (0 false alarms; 1_18 is the fold every
     other raw-EEG model collapses recall on), total miss on 1_15
     (predicts 0 preictal, but AP .311/AUC .957 -> threshold/calibration
     failure not representation). Strongest in that comparison doc on
     every aggregate metric bar FAR/h-smoothed; sits just under "pre"
     (0.644-0.674), above the doc's old encoder-free Mamba (0.499).
     WRITEUP DONE 2026-08-31: **new `Session_notes/2026_08_31/pipeline_
     comparison_all_models.md`** -- the full board, all 9 candidate
     pipelines + the ablation/variant table, refreshed from CSVs. (The
     2026_08_25 4-way doc is left untouched as the detailed write-up for
     those four; an earlier attempt to bolt Godoy + a "pre" row onto it
     was reverted per user -- "make a new writeup with all the models".)
     Board top: "pre" 0.644 / "post" 0.639 / godoy **0.556 ± 0.056**
     (5-seed sweep, see 5b; single-run 0.619 was the lucky top) /
     enc-free Mamba 0.499 / DBConf 0.442 / herm-eig 0.436 / SlimSeiz
     0.430 / GRU 0.423 / CG-MN 0.127.
- **`continuous_cwt_mamba` 6-fold -- NOT to be launched (user, 2026-08-30);
  kept in context only.** CPU smoke (subject 1, 1 fold) ran clean into
  training -- epoch 1 `loss=0.708 val_loss=0.743` at ~29 min (incl. one-time
  dense-edge CWT cache build), ~15-20 min/epoch steady-state -- then killed
  by hand at epoch 2 (rc=143, log's "smoke FAILED ... needs the bug fixed"
  is the script's generic message, not a real failure). Takeaway: the
  2026-08-26 CUDA "illegal memory access" during validation does NOT
  reproduce on CPU; the pipeline trains. A full CPU 6-fold is just too slow
  to be worth the machine time right now.
- **Commit pending** (once queued results land): `run_pipelines.py`
  (`HERMITIAN_SSM_PARAMS` comments + `encoder_mode` + `mamba_backend`
  revert from stale `"mamba3"` `7d672e1`), `hermitian_ssm_classifier.py`,
  `cwt_gnn_classifiers.py`, `hermitian_ssm_anomaly.py`, `channel_cwt_mamba.py`.
User's position (2026-08-29): `temporal_graph_mamba` "pre" (0.674) is NOT
an acceptable deliverable -- it collapses 253 edges -> 23 nodes *before*
the temporal model, failing the graph-native requirement regardless of the
number. Acceptable deliverables, in preference order:
  1. hermitian_ssm (eigenvector encoder; 0.436 real Mamba. mamba3 NEGATIVE
     0.408. k-axis in progress).
  2. **`temporal_graph_aggregate="post"`** -- FALLBACK if the hermitian
     experiments all fail (mamba3 already did; k-axis is the last one).
     "post" runs Mamba over each of the ~253
     EDGE sequences first, aggregates to nodes *after* -> the graph flows
     through the temporal model, which meets the graph-native bar. Its
     6-fold was a near-wash (mean AP 0.639 vs "pre" 0.674, ROC-AUC 0.970
     vs 0.94; see the post A/B thread below) and the user prefers it to
     "pre" on architecture despite the ~11x compute. Revert to it (flip
     `temporal_graph_aggregate="post"` in `PREDICTION_TEMPORAL_GRAPH_MAMBA_
     PARAMS`) if hermitian doesn't pan out.
Real ceiling lever is more CHB-MIT subjects, not architecture.

Also 2026-08-29 (separate thread, `main`, not this branch): stood up
`cg_mambanet` on a RunPod GPU end-to-end -- fixed a GHCR pull-rate-limit
(mirrored image to Docker Hub), a torch-ABI mismatch in `Dockerfile.mamba`
(`--no-deps`), and two missing runtime deps found live
(`huggingface_hub`, `transformers`) -- then ran the real 6-fold chb01
LOSO prediction test. Results are weak (see Known gotchas below and the
CG-MambaNet addendum in `pipeline_comparison_gru_mamba_dbconformer_slimseiz.md`);
pod terminated after the run. Results CSVs are in
`Epilepsy/results/cg_mambanet/prediction/`.

2026-08-29 session work:
- **`hermitian_ssm` band-match 6-fold -- big jump, now competitive.**
  Config-only changes vs the 0.253 run: band 8-124->**8-40 Hz**, nfreqs
  60->**16** (fd=2 -> 8 cached bins), `diagonal` **"power"->"zero"**, `k`
  **2->6**. Result: mean AP **0.436** (was 0.253, +72%), ROC-AUC **0.947**
  (>= tgm "pre" 0.94), hit rate **6/6 raw + 6/6 smoothed** (tgm "pre" is
  6/6 / 5/6), FAR/h smoothed **8.0** (tgm "pre" 8.44). Per-fold AP vs "pre":
  1_03 .254/.792, 1_04 .324/.830, 1_15 **.414/.331**, 1_16 .279/.996,
  1_18 .788/.802, 1_26 **.556/.293** -- hermitian wins the 2 folds where
  "pre" is weak, loses the 3 where it's strong. Strong ROC-AUC + weak AP =
  precision-at-top / calibration gap, not a ranking failure. `diag="zero"`
  is the likely main lever (removed the per-channel power spikes that
  pinned lambda_1 to a global-synchrony common mode). CSVs
  `Epilepsy/results/hermitian_ssm/prediction/*_20260829-111904.csv`. See
  `Session_notes/2026_08_29/hermitian_ssm_bandmatch_6fold.md`.
  - **Uncommitted:** the `HERMITIAN_SSM_PARAMS` edit in `run_pipelines.py`
    (spectral cache key -> new key `9d6ad0d850b8b8f0`). Not committed
    pending the user's call. Stale line ~899 ("8-124 / 30-bin input is a
    different regime") is now wrong -- the band IS matched.
  - **Cache:** `~/mne_data/hermitian_ssm_cache/9d6ad0d850b8b8f0` = 9.8 GB,
    all 41 recordings, warm. Old-key caches were already deleted 2026-08-28.
  - **Projector encoder (#2) -- NEGATIVE (2026-08-29).**
    `encoder_mode="projector"` (`_ProjectorEncoder`): gauge-invariant node
    summaries of `P = Sum lambda_r u_r u_r^H`. Same warm cache, only the
    encoder changed. Per-fold AP `.138/.190/.259/.300/.585/.168`, **mean
    0.273**, ROC-AUC 0.898 (`*_20260829-152511.csv`) -- worse than the
    eigenvector encoder (0.436/0.947) on every axis. Gauge noise was not
    the bottleneck; the C x C -> C-vector collapse loses more than it saves.
  - **Encoder-variant investigation -- CLOSED 2026-08-29, NEGATIVE.**
    `HermitianSSMClassifier.encoder_mode` now has 4 options (all BUILT +
    smoke-passed, all reuse the warm k=6 cache, all in
    `hermitian_ssm_classifier.py`; `_WindowDataset.item_mode` +
    `canonicalize` do the per-window numpy prep):
      | mode | what it feeds | channel identity | result |
      |---|---|---|---|
      | `"eigenvector"` (default) | `[Re u, Im u]` + `mode_id` + lambda | **full** (u_r in C^23) | **mean AP 0.436** -- hermitian best |
      | `"projector"` (#2) | gauge-inv node summaries of `P` | partial->collapsed | 0.273, 5/6 folds down |
      | `"graph"` (#3) | upper triangle of `P` (lossless) | partial | fold 1 = 0.169, killed |
      | `"evolution"` (#6) | complex k x k `M(t)=U(t)^H U(t-1)` + lambda | **none** (pure mode space) | folds 1-2 ~chance (.062,.091), val_auc stuck <0.77, killed |
    `canonicalize_eigenvectors=True` (#4) also BUILT, NOT run -- skipped:
    #2/#3 already show gauge noise isn't the cap, #4 = "gauge fix + same
    encoder" has ~nil expected value.
    **Conclusion:** result degrades monotonically with how much channel
    identity the encoder drops. What hermitian needs is *which channels
    couple*, in raw eigenvector coordinates. Every abstraction toward the
    "graph as an object" loses ground. Confirms the fragility thesis (data
    budget, not representation) from the opposite direction.
    **Ceiling: mean AP ~0.44.** `temporal_graph_mamba` "pre" (0.674) stays
    the prediction leader.
    **Only lever left for hermitian:** Mamba-3 (complex state) on the
    eigenvector encoder -- but that's a temporal-model change not a
    representation one, and expected value is low. Unbuilt (~150-250 lines).
    Uncommitted: `hermitian_ssm_classifier.py` (4 encoder modes),
    `run_pipelines.py` (`HERMITIAN_SSM_PARAMS`, `encoder_mode="eigenvector"`).
    See `Session_notes/2026_08_29/hermitian_ssm_bandmatch_6fold.md`.
  - **k-sweep (do AFTER 3/4/6, on whichever encoder wins, and add
    DataLoader `num_workers` first).** Cache is linear in k (`eigenvectors.npy`
    is 96% of it; `eigh` always computes full rank so precompute time is
    ~flat). k=6 -> 9.8 GB / ~100-120 s per epoch / ~2 h 6-fold. **k=12 ->
    ~19-20 GB / ~200 s / ~5 h** (batch 16 ok). k=23 -> ~37 GB / ~380 s /
    ~9-10 h (graph enc needs batch 8). New cache key each, additive to the
    k=6 cache (68 GB free). NB k=23 => P == A exactly, so graph-enc at k=23
    is just "feed raw coherence upper triangle" -- not compression. k=12 is
    the meaningful middle. Epochs are IO-bound purely because
    `num_workers=0`; parallelising the loader ~halves epoch time and should
    land before any k>6 run.

Prior (2026-08-28) session work:
- **`hermitian_ssm` 6-fold (float16, d_model=64, 8-124 Hz):** mean AP
  **0.253**, AUC 0.888 (`*_20260828-091710.csv`). Superseded by the
  band-match run above.
- **Wide-band experiment on `temporal_graph_mamba` -- ABANDONED,
  negative.** Widening 8-40->8-124 Hz (nfreqs=15/tds=32, then
  nfreqs=10/tds=16) made fold 1_03 AP collapse 0.792->0.307 then
  ->0.236. The wide band dilutes the 8-40 signal through
  `temporal_edge_proj`'s `4*nfreqs->edge_dim` mix. Reverted; kept as a
  comment in `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS`.
- **`temporal_graph_aggregate="post"` A/B -- DONE, a wash, not adopted**
  (2026-08-28, 8-40/nfreqs=8 baseline, only that knob changed): Mamba
  over the ~253 edge sequences first, aggregate to nodes after -- keeps
  the full edge graph. Full 6-fold: mean AP **0.639 vs "pre" 0.674**
  (gap is entirely fold 1_03), ROC-AUC **0.970 vs 0.94** (slightly
  better global ranker), hit rate 6/6 raw+sm vs 6/6 raw / 5/6 sm.
  4/6 folds >= pre. ~11x compute for a tie -> **committed default stays
  "pre"**, knob parked in code. CSVs `*20260828-161354*`. See
  `Session_notes/2026_08_28/temporal_graph_aggregate_post_6fold.md`.
  **2026-08-29: "post" is now the FALLBACK deliverable** -- the graph
  flows through the temporal model (edge Mambas, aggregate after), which
  meets the user's graph-native bar; "pre" does not. If hermitian_ssm +
  mamba3 all fail, revert to "post" (see the User's-position note at top).
- **Negative/null-results pattern (2026-08-28):** attempts to improve
  `temporal_graph_mamba` by adding capacity/richness: wide-band (NEG,
  0.79->0.24 fold 1), reg-tuning 2026-08-27 (NEG), "post" (WASH on full
  6-fold, looked NEG on fold 1). Bottleneck is not capacity -- it's ~30
  preictal windows/fold on chb01-only, and partial-run reads are
  actively misleading (the "post" fold-1 "regression" was fold variance).
  Next levers: decision-threshold calibration (not the model),
  seed-repeating the 0.674 baseline for an error bar. Full writeup:
  `Session_notes/2026_08_28/negative_results_roundup_and_fragility.md`.
Session notes: `Session_notes/2026_08_28/`.

Prior entry, still current: 2026-08-26, by Claude (ran the planned next
diagnostic step on `continuous_cwt_mamba`'s open CUDA bug, from a fresh Mac shell:
`--pipeline continuous_cwt_mamba --smoke --continuous-mamba-t-chunk 64
--max-folds 1 --device cpu`. **Completed cleanly, exit 0** -- fit (2
epochs), validated, predicted, wrote both result CSVs, no exception
anywhere including the `torch.cat` call CUDA's error pointed at. This
rules out a general Python-level off-by-one in
`_train_recording_incremental`'s slicing or `_sample_to_output_index`'s
fallback branch (both exercised correctly here, real 12-recording/
775-window smoke dataset) -- **the bug is CUDA/mambapy-pscan-specific,
not a portable logic bug.** Next actual step needs a CUDA box: rerun with
`CUDA_LAUNCH_BLOCKING=1 --device cuda`, and if that still just points at
`torch.cat`, add `torch.cuda.synchronize()` after every per-chunk
`_DenseEdgeMambaContinuous` call in the streaming loop to force each
chunk's error synchronous and pin down which (recording, chunk index,
tensor shape) triggers it. Full writeup: `Session_notes/2026_08_26/
continuous_cwt_mamba_pipeline_wip.md`'s "CPU repro attempt" section (new,
appended below the original bug writeup -- read both). **Still don't
trust a real run of this pipeline until the CUDA bug is actually fixed;**
this update only narrows where to look, no code changed.
Separately, also this session: another Mac shell had independently started
re-deriving `ContinuousLabelingParadigm.get_continuous_data()` from
scratch (as `iter_labeled_runs`/`label_run`, a different-shaped refactor)
before pulling and discovering Phase B already existed on `main` -- caught
before it was pushed, branch deleted, no trace left. Flagging only so a
future shell doesn't wonder why a `label_run` method was almost added:
it wasn't, `get_continuous_data()` is the one and only loader.).

Prior entry, still current: Claude (wired continuous-cwt-mamba into
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

- `main` -- **current branch, everything is merged here as of 2026-08-28.**
  Has `dense_edge_mamba` (from `mamba-temporal-edge-model`, merged at
  `6d38573`), the cwt node encoder, the continuous-cwt-mamba paradigm
  plumbing (`aa3c565`, `4760de0`), `use_cuda_kernel` + `Dockerfile.mamba` +
  the live GHCR `eeg_benchmarks-mamba` image, SlimSeiz, dbconformer, the
  chb01-04 GitHub-release mirror generalization, `cg_mambanet`, AND (2026-08-28,
  from `graph-state-mamba`) the `temporal_graph_gru`/`temporal_graph_mamba`
  `--pipeline` wiring, the `hermitian_ssm` pipeline, and the
  `temporal_graph_aggregate` pre/post knob.
- `graph-state-mamba` -- **merged into `main` (fast-forward) and deleted**
  (local + `origin`) 2026-08-28. Don't recreate it; its work is all on
  `main`. Was where `temporal_graph_*` wiring + `hermitian_ssm` +
  the aggregate pre/post knob were developed.
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
- `cg-mambanet` -- **merged into `main` 2026-08-26** (fast-forward), then
  deleted (both locally and on `origin`) 2026-08-26 once confirmed
  `main..cg-mambanet` was empty (nothing on it main didn't already have).
  Don't assume it still exists. Its architecture (`cg_mambanet_classifier.py`,
  `--pipeline cg_mambanet`) lives on `main`; see "Open threads" below for
  the still-outstanding real-run attempt.
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
  (a real kernel, not chained PyTorch ops) and does sidestep this --
  confirmed 2026-08-29 on a RunPod RTX 4090:
  `scripts/dense_edge_mamba_cuda_kernel_parity.py` passed
  (max|kernel-pscan|=1.2e-7) and a full 6-fold chb01 LOSO `cg_mambanet`
  prediction run completed in minutes. Results are weak (mean AP 0.127,
  AUC 0.797, worst of 5 pipelines compared) -- see
  `pipeline_comparison_gru_mamba_dbconformer_slimseiz.md`'s CG-MambaNet
  addendum -- but that's the architecture underperforming, not a
  scaling-wall or kernel-correctness issue; the wall itself is resolved.
- **RunPod pods: ALWAYS terminate (`delete-pod`) as soon as a run
  finishes or is confirmed dead-ended -- do not wait for the user to say
  so.** They're not always around to give that go-ahead and a pod left up
  just burns their money on idle GPU time. This applies even to pods hit
  mid-diagnosis (rate-limit loops, CUDA-driver-too-old boot failures,
  etc.) -- delete and recreate rather than leaving a stuck one running
  while you investigate.
- **A `dataCenterIds` pin does NOT guarantee driver/CUDA version --
  that's per physical host, not per datacenter (2026-08-29).** A pod
  pinned to EU-RO-1 that had previously booted fine on `cuda>=12.8` later
  got a different EU-RO-1 host reporting driver `12.4` and failed to boot
  (`nvidia-container-cli: requirement error: unsatisfied condition:
  cuda>=12.8`) on two separate re-creates. There's no way surfaced by
  these tools to request a minimum driver version directly -- if a pod's
  container is stuck retry-looping on this error, delete and recreate
  (whichever datacenter) rather than waiting; a different host usually
  has a current-enough driver within one or two tries.
- **RunPod's fused-kernel image (`Dockerfile.mamba`) needed two more
  runtime deps beyond torch/mamba-ssm/causal-conv1d, found live on
  2026-08-29: `huggingface_hub` and `transformers`** (mamba_ssm's
  `mamba2.py` -> `generation.py` import chain pulls both in, but they're
  runtime, not build, deps -- the `--no-deps` pip flag added earlier to
  fix the torch-ABI-mismatch incident skips them too). Fixed in the
  Dockerfile (commit `3382187`) by installing them in their own layer,
  separately from the `--no-deps` mamba-ssm/causal-conv1d install, so the
  ABI-safety property that flag exists for isn't reopened.
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

- **`cg_mambanet` capacity cut (2026-08-29, `run_pipelines.py` commit
  `b4ee0db`) re-run 2026-08-30 (RunPod, `-135306`) -- DID NOT FIX IT,
  basically a wash.** The first real 6-fold run (`20260829-114933`) was
  well behind every other pipeline in the comparison table (AP 0.127 vs.
  0.42-0.50) with an overfitting signature (train roc_auc -> ~1.0 by
  epoch 6-9, val_loss spiking, best checkpoint epoch 3-8/20) --
  diagnosed as the paper's cross-patient-scale architecture (12 Mamba
  layers, d_state=64, BiLSTM 128x2) overparameterized for this repo's
  per-subject LOSO protocol (~900-3500 windows/fold vs. the paper's own
  pooled ~12+-patient set). Cut to mamba_n_layers=3, d_state=16,
  expand_factor=1, d_embed=32, lstm_hidden=64/1-layer, head_dropout=0.5,
  weight_decay=3e-3, grad_clip_norm=1.0 (was unset), early_stopping_
  patience=3 and re-ran the same protocol: AP ticked up marginally
  (0.127->0.139) but accuracy/precision/recall/f1/AUC/FAR-h all got
  *worse* (e.g. AUC 0.797->0.774, FAR/h-smoothed 1.50->2.62), k-of-n hit
  rate unchanged at 3/6. Per-epoch train/val gap WAS visibly healthier
  this run (confirms the overfitting diagnosis wasn't wrong), but that
  didn't translate into better held-out numbers -- so capacity wasn't the
  whole story, or the cut cost more useful capacity than overfitting
  margin. See `pipeline_comparison_gru_mamba_dbconformer_slimseiz.md`'s
  CG-MambaNet addendum ("Capacity-cut follow-up") for the full before/
  after table. **Not resolved -- next step (if pursued further) would be
  a capacity SWEEP (a few sizes between the two extremes already tried),
  not another single before/after point.** Two other latent issues still
  flagged, not yet fixed: unconstrained/unsquashed learnable adjacency in
  `_LearnableGCN` (no NaN observed yet but no softmax/ReLU guards it
  either); weight_decay applied uniformly including Mamba's own SSM
  params A_log/D/dt_bias, which most Mamba training recipes exclude.
- **CWT frequency band (`lowest=8.0, highest=40.0, nfreqs=8`) is an
  untuned motor-imagery-BCI leftover, never revisited for epilepsy
  (traced 2026-08-27).** `Session_notes/2026_08_15/chb_mit_dataset_and_
  dense_edge_gru_pipeline.md` documents the actual origin: `lowest=8.0,
  highest=35.0` came from BNCI2014_001's mu/beta motor-imagery band (a
  completely different task -- decoding imagined movement, not seizure
  activity), loosely widened to `highest=40.0` when the codebase moved to
  CHB-MIT and flagged AT THE TIME as "an explicitly not-tuned broad
  placeholder... picking a real epilepsy-appropriate band is unstarted
  work." Never picked back up since. `nfreqs` was separately halved
  16->8 purely for disk-cache size (2026-08-16), not for spectral-
  resolution reasons. **35-40Hz is very likely too low a ceiling for
  epilepsy-relevant fast activity** -- worth a real, literature-informed
  band selection instead of the inherited placeholder.

  Real considerations before just widening the range (raised 2026-08-27,
  not yet resolved): (1) mains noise (CHB-MIT is US data, 60Hz + 120Hz
  harmonic) sits inside any wider band and is currently notch-filtered
  ONLY in `truong_stft_cnn_classifier.py` (57-63Hz/117-123Hz) -- not
  applied anywhere in the CWT/coherence path `dense_edge`/
  `temporal_graph_mamba`/`dense_edge_mamba` actually use, so widening
  through 60Hz without adding equivalent filtering there risks a
  purely-artifactual, highly-coherent-by-construction mains spike
  dominating part of the signal; (2) clinically-relevant high-frequency
  epilepsy biomarkers (ripples 80-250Hz, fast ripples 250-500Hz) are
  primarily an intracranial-EEG finding -- scalp recordings like CHB-MIT
  have real, physiologically-driven SNR loss at high frequency from
  volume conduction/skull attenuation, so pushing toward Nyquist (128Hz
  at `sampling_rate=256`) isn't obviously "free" extra signal on this
  modality; (3) CWT's time-frequency tradeoff (higher-frequency wavelets
  need shorter temporal support for the same cycle count) interacts with
  the existing cone-of-influence masking (`_coi_valid_mask`) -- worth
  checking how much high-frequency content actually survives COI-
  validity at real window lengths before assuming it's usable; (4)
  widening the band while keeping `nfreqs=8` fixed coarsens resolution-
  per-bin further -- real resolution at higher frequencies likely wants
  more bins, reopening the disk-budget constraint that halved `nfreqs`
  in the first place.

  Recommended approach: an actual ablation (current 8-40Hz vs. a wider
  band with mains notching added to the CWT/coherence path vs. a band
  chosen from real epilepsy/seizure-coherence literature), not reasoning
  from first principles alone.

  **Literature scan done 2026-08-27** -- found real, directly relevant
  precedent for going much wider than 40Hz on this exact dataset/sampling
  rate:
  - A covariance-matrix-eigenvalue-based seizure prediction paper
    (`iopscience.iop.org/article/10.1088/1741-2552/ac6063` --
    methodologically close to this codebase's own coherence-graph
    approach) uses CHB-MIT-style scalp EEG at `sampling_rate=256`
    (same as here) and analyzes **0-124Hz in 61 bins of 2Hz resolution**
    ("we cannot investigate frequencies higher than 128 [Nyquist]... we
    only focus on (0,124)Hz, excluding marginal frequencies 125-128Hz")
    -- i.e. essentially the full Nyquist range minus the unusable edge,
    not a truncated low/mid-frequency band. No explicit 60Hz mains notch
    reported -- only a general wavelet-based denoising step. Reported
    false-positive rate as low as 0.09/h (vs. this codebase's current
    `temporal_graph_mamba` mean FAR/h of 8.44 smoothed -- not apples to
    apples, different classifier entirely, but a strong existence proof
    that a wide, near-Nyquist band is workable and can support very low
    FAR on this kind of data).
  - A separate frequency-band-analysis paper for seizure prediction
    (found via search snippet only, full text not fetched --
    `researchgate.net/publication/258238854`) reports gamma-band
    (~32-120Hz) features giving the highest sensitivity/precision and
    lowest false-positive rate among the bands it tested on CHB-MIT,
    ahead of the lower/classic clinical bands.
  Both sources point the same direction: **the current 40Hz ceiling is
  very likely leaving real, usable signal on the table** for this task,
  and near-Nyquist ranges (approaching but not reaching the unusable
  125-128Hz edge at `sampling_rate=256`) have real precedent on
  comparable scalp-EEG/CHB-MIT-style data. Neither source resolves the
  mains-noise question definitively (neither reports an explicit 60Hz/
  120Hz notch), so whether this codebase needs to add one before
  widening remains an open, testable question, not settled by this scan
  alone -- worth trying the wider band both with and without an added
  notch to see whether it matters in practice, rather than assuming
  either way.
- **Calibrate the decision threshold instead of using `predict()`'s fixed
  0.5 cutoff -- UNBLOCKED 2026-08-27, not yet done.** On the real 6-fold
  `temporal_graph_mamba` run's fold 1 (`1_03_0`), AP (auc_pr) was 0.792 --
  5-6x every other pipeline's fold-1 AP (GRU 0.141, dense_edge_mamba
  0.156, DBConformer 0.279, SlimSeiz 0.261) -- but FAR/h was the *worst*
  of the group at the actual 0.5 threshold (18.5 raw, 17.2 smoothed, vs.
  GRU 18.7/14.8, DBConformer 15.3/10.2). This pattern held across nearly
  every fold in the completed 6-fold baseline (mean AP 0.674, the best of
  the compared pipelines, but FAR/h mid-pack) -- see
  `Session_notes/2026_08_27/temporal_graph_mamba_full_6fold_and_tuning_
  attempt.md`. AP being threshold-independent while FAR/h is often worst
  in the group suggests the model's score *ranking* is genuinely good but
  0.5 is a bad operating point for it -- worth picking a threshold from
  the validation split's own PR curve (e.g. targeting a FAR/h budget)
  instead of hardcoding 0.5, the way clinical seizure-prediction
  literature usually does.

  What changed 2026-08-27: `leave_one_seizure_out_prediction` now has a
  `dump_window_scores` param / `--dump-window-scores` CLI flag
  (`run_pipelines.py`) that persists one row per TEST window (`fold_i,
  held_out_seizure_id, subject, run, window_seizure_id, window_start,
  y_test, y_score, y_pred, y_pred_smoothed`) to
  `prediction/prediction_window_scores_<run_id>.csv` -- no retraining
  needed to test threshold/k-of-n/calibration changes anymore, just
  reprocessing the saved scores. A real, full untuned 6-fold
  `temporal_graph_mamba` run WITH this flag already completed cleanly
  end-to-end 2026-08-27 (no external kill, no `--skip-folds` stitching):
  `/tmp/tg_mamba_retrain_full/temporal_graph_mamba/prediction/
  prediction_window_scores_20260827-070913.csv`, 4,241 real test windows
  across all 6 chb01 folds, real score separation (not clustered near
  0.5 the way a `--smoke` run's scores are). This file has NOT been
  copied anywhere durable yet -- it's in `/tmp`, not committed to the
  repo or `Epilepsy/results/`; move/commit it (or regenerate) before it
  can be lost to a reboot/tmp-cleanup.

  Next step, not yet done: for each fold, sweep a threshold against that
  fold's own validation-split scores (need to add validation-split score
  logging too -- today's CSV only has TEST windows, not the val split
  `_train_loop` already holds out during training) and re-score the test
  windows at the chosen threshold instead of the hardcoded 0.5, comparing
  FAR/h and precision/recall/f1 before vs. after per fold.
- **Need to actually try `cg_mambanet` for real (2026-08-26).** Built and
  smoke-tested (`--pipeline cg_mambanet`, `cg_mambanet_classifier.py` on
  `main`), but a real (non-smoke) run has never completed -- blocked on
  `mambapy`'s scan not scaling to this encoder's size on CPU/MPS (see
  "Known gotchas" above for the numbers). Branch `cg-mambanet` itself is
  deleted (fully merged, nothing left on it); this is just about running
  the pipeline that's already on `main`. Next step: run it on the RunPod
  CUDA image (`ghcr.io/noshore5/eeg_benchmarks-mamba:20260825-4760de0`,
  fused `mamba-ssm` kernel sidesteps `mambapy`'s scan entirely) and see if
  a real chb01 LOSO pass actually completes and produces numbers worth
  comparing against the other pipelines.
- **`channel_subset_k` marks dead edges by zeroing, not an explicit
  liveness bit -- ambiguous in principle (2026-08-26)**:
  `_compute_live_dense_edge_scattered`/`_scatter_live_dense_edge`
  (`Epilepsy/pipelines/cwt_gnn_classifiers.py:6934-6937`) compute the real
  4-channel `[coh, sinφ, cosφ, significance]` stack only for the selected
  top-k clique's edges, then scatter into a full-`E` zeros tensor -- a
  dropped edge is `[0,0,0,0]` for every timestep, not flagged by a
  dedicated 5th "is this edge live" channel. This is implicit, not
  architecturally guaranteed unique: a genuinely live edge sitting exactly
  at the fixed coherence threshold (`significance=0`) with `phase=0` would
  look identical to a masked-out one. Never actually confused in practice
  as far as anyone's checked, but worth either (a) adding an explicit
  live/dead channel so the model doesn't have to infer it from an
  all-zero pattern, or (b) skipping compute for dead edges outright
  instead of computing zeros for them, per the next bullet.
- **Given (a), Mamba doesn't need to run on dead edges at all**: if
  liveness were explicit (or even just passed down as the same boolean
  mask `_scatter_live_dense_edge` already has as `live`), the per-edge
  Mamba call in `_DenseEdgeMambaTemporal.forward` could skip the
  known-dead rows of the `[B*E, T, C_in]` batch entirely instead of
  scanning all-zero input through the SSM and getting a zero (or
  near-zero, non-trivial-bias-dependent) output back out. Combined with
  the fact that edges are already the undirected (i<j) 253-edge topology
  (not 506 -- see the 2026-08-09 note in `SparseEvidenceGNNCore.__init__`)
  and `channel_subset_k` typically selecting well under the full channel
  mesh, this could be a real compute saving proportional to how sparse
  the selected clique is relative to the full 253-edge mesh -- untested,
  and would need `mamba_chunk_size`'s row-grouping (see that class's
  docstring) reworked to skip rows rather than just batch them.
- **Aggregate-then-Mamba (`temporal_graph_mamba`) -- on `main`, real
  6-fold done, current prediction leader.** `event_mode="temporal_graph"`
  (2026-08-11): aggregate every edge's per-timestep message to its
  destination node first, then walk a persistent per-node state forward
  through the sequence (`_temporal_graph_node_states`).
  `temporal_graph_mode="mamba"` swaps that per-node `nn.GRU` for
  `_DenseEdgeMambaTemporal` reused UNCHANGED with the node axis in the
  edge-axis slot. Wired as `--pipeline temporal_graph_gru`/
  `temporal_graph_mamba` (reuses the dense-family dispatch). Merged to
  `main` from `graph-state-mamba` 2026-08-28.
  **Real chb01 6-fold prediction (untuned, 8-40 Hz/nfreqs=8):** mean
  **AP 0.674**, ROC-AUC 0.94 -- beats dense_edge_gru k=20 (0.567),
  dense_edge_mamba (0.42-0.50), dbconformer (0.44), hermitian_ssm (0.25).
  Per-fold AP 0.79/0.83/0.33/1.00/0.80/0.29 -- **high variance** (1 subject,
  6 seizures, ~30 preictal windows/fold) and **fragile**: a mild reg
  change (weight_decay 1e-4->3e-4, dropout 0->0.15) collapsed fold 1_03
  0.792->0.232 and broke fold 1_04's training -- REVERTED, see
  `Session_notes/2026_08_27/temporal_graph_mamba_full_6fold_and_tuning_attempt.md`.
  Not yet done: seed-repeat to confirm 0.674 is stable; `temporal_graph_gru`
  (per-node GRU) 6-fold for the gru-vs-mamba comparison.
  **`temporal_graph_aggregate` pre/post knob (2026-08-28, on `main`):**
  "pre" (default) = the 0.674 run. "post" = run the Mamba over each of
  the ~253 EDGE sequences first (`_DenseEdgeMambaTemporal`'s original
  per-edge role), then scatter-mean to nodes -- keeps the full edge graph
  instead of a per-timestep 23-node average. ~11x more Mamba rows
  (n_rows=B*E >> chunk_size so it gradient-checkpoints, memory bounded);
  cache `[B,4,E,T,F]` unchanged. Guard: requires
  `temporal_graph_mode="mamba"`. 6-fold "post" A/B (2026-08-28, only this
  knob changed): **DONE, a wash.** Full 6-fold mean AP 0.639 vs "pre"
  0.674 (gap = fold 1_03 alone: 0.407 vs 0.792), ROC-AUC 0.970 vs 0.94,
  4/6 folds >= pre. Per-fold post AP: 0.407 / 0.837 / 0.487 / 0.994 /
  0.854 / 0.257. ~11x compute for a tie -> PARKED, default stays "pre".
  To reproduce, flip `temporal_graph_aggregate="post"` in
  `PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS`. CSVs `*20260828-161354*`. See
  `Session_notes/2026_08_28/temporal_graph_aggregate_post_6fold.md`
  and the roundup note in that dir.
- **`temporal_graph_mamba` neck is width-8, never swept.**
  `temporal_edge_proj` (`Linear(4*nfreqs -> temporal_graph_edge_dim=8)`)
  then everything at `hidden_dim=8` including the Mamba's `d_model` --
  the precomputed `[4,E,T,F]` edge stack (coh/sinphi/cosphi/significance,
  full detail) is funnelled through an 8-neck before the temporal model.
  Both widths are 2026-08-16 disk-budget choices. Candidate lever: bump
  `hidden_dim`/`temporal_graph_edge_dim` to ~32 in the
  `TEMPORAL_GRAPH_*_PARAMS` dicts (NOT `_SHARED_ARCH_PARAMS` -- shared
  with all dense-family pipelines). Or make the F-reduction a fixed band
  pool (theta/alpha/beta/gamma) at precompute time instead of a learned
  squeeze. See `Session_notes/2026_08_28/`.
- **Graph-aware state-space Mamba -- whole-graph-as-one-token branch BUILT
  2026-08-27 as `--pipeline hermitian_ssm`** (branch `graph-state-mamba`,
  committed `26dd172`; eigh fix + first 6-fold result 2026-08-28). This is the "Graph Spectral Mamba-3" design
  (`Epilepsy/hermitian_ssm.md`), i.e. Design B + Design C of
  `graph_state_space_mamba_design.md` taken to their conclusion: complex
  Hermitian per-frequency channel graph -> `torch.linalg.eigh` -> top-k=2
  eigenpairs (deterministic, disk-cached per recording) -> complex spectral
  encoder -> `_DenseEdgeMambaTemporal` (Mamba-2, reused; **Mamba-3 is the
  documented next step**, comment block in `hermitian_ssm_classifier.py`)
  -> head. NOT Design A (STG-Mamba `Delta' = Delta @ A` topology-in-the-
  scan) -- that is still unbuilt.
  Files: `Epilepsy/pipelines/hermitian_ssm_cache.py` (precompute + cache),
  `Epilepsy/pipelines/hermitian_ssm_classifier.py` (self-contained, does
  NOT subclass SparseEvidenceGNNClassifier), `leave_one_seizure_out_
  hermitian_ssm` + `HERMITIAN_SSM_PARAMS` + `--pipeline hermitian_ssm` in
  `run_pipelines.py`, `scripts/hermitian_ssm_{numerical_validation,smoke}.py`.
  Full build writeup + locked decisions: `Session_notes/2026_08_27/
  hermitian_ssm_pipeline_built.md`.
  Config (2026-08-29 band-match): **8-40 Hz, nfreqs=16 -> 8 bins**
  post-smoothing, time_downsample=16, **k=6**, **diagonal="zero"**,
  mains notch now a no-op (highest=40), **d_model=64**
  (was 256; cut 2026-08-28), **eigenvector_storage="float16"**.
  (Old 8-124 Hz / nfreqs=60->30 / k=2 / diagonal="power" config: the
  three PARKED runs on `main`, mean AP <=0.253.)
  Cache: per-recording `.npy` dir keyed by `HermitianSpectralConfig.
  cache_key()`, windowing-independent (any window/step re-slices it),
  mmap'd during training. ~0.65 GB / 1h recording, ~24 GB projected for
  all chb01. Precompute is CPU-only (`torch.linalg.eigh` has no MPS impl
  in this torch build) -- batched CPU eigh ~29us/matrix, so ~1 min/
  recording, ~40 min one-time for a full 6-fold, then cached.
  Encoder gained **frequency + mode identity features** (2026-08-27, after
  first build): `vec_proj`/`val_proj` are weight-shared across both the F
  and k axes, so identity was otherwise only positional (slot order in
  `freq_fuse`/`mode_fuse`). `freq_feature=True` concatenates normalised Hz
  (linear+log) onto the per-mode input; `mode_feature=True` concatenates a
  learned `[k, 4]` per-slot embedding (slot = `|lambda|`-rank, not a
  stable physical mode). Both flags in `HERMITIAN_SSM_PARAMS` /
  `HermitianSSMClassifier`, `False` for ablation. Not built: per-frequency
  Mamba lanes (`E=F` instead of `E=1`) -- discussed, left as a future
  config switch, see the session note.
  **Status (2026-08-29): band-match 6-fold is now the current run --
  mean AP 0.436, ROC-AUC 0.947, hit 6/6 raw+sm, FAR/h sm 8.0**
  (`*_20260829-111904.csv`; config: 8-40 Hz, nfreqs=16->8 bins,
  `diagonal="zero"`, k=6, d_model=64, float16). Competitive with
  `temporal_graph_mamba` on every metric except AP (0.436 vs 0.674).
  `diag="zero"` is the likely main lever of the +0.18 vs the old config.
  Prior PARKED runs (8-124 Hz, on `main`): d_model=256
  (`*_20260827-231141.csv`) AP 0.237; d_model=64 complex64
  (`*_20260828-055322.csv`) AP 0.241; d_model=64 float16
  (`*_20260828-091710.csv`) mean AP 0.253, AUC 0.888.
  The 2026-08-28 diagnosis (off the old cache) that drove the band-match
  config: eigen-features collapsed to ~1 effective dim -- corr(l1,l2)=0.93,
  top eigenvector near-uniform (participation ~19/23 = global synchrony),
  ~12% near-degenerate slices; the power diagonal dominated the
  eigenstructure (l1 median 14, max 3210). `diagonal="zero"` + k=6
  addressed the last two. Still untried: projector encoder (gauge-invariant
  `P = Sum l_r u_r u_r^H` node summaries -- kills the sign-flip / mode-
  crossing churn), Mamba-3 (complex state, tracks cross-spectral phase),
  explicit per-channel log-power stream. See
  `Session_notes/2026_08_29/hermitian_ssm_bandmatch_6fold.md`.
  **DONE this session:** (1) eigh-convergence fix -- `torch.linalg.eigh`
  (LAPACK syevd) fails on flatlined CHB-MIT segments (LinAlgError code
  22); `nan_to_num` + fall back to `torch.linalg.eig` (geev) per freq
  chunk. (2) d_model 256->64 -- cost nothing (256 was oversized, raw
  per-(t,f) width is 94), 2x faster; 256's `d_inner=512` pscan tensors
  (~1 GB, E=1 so no gradient-checkpointing) were swapping 16 GB RAM.
  (3) float16 eigenvector storage (`eigenvector_storage="float16"`,
  real/imag split, 4 B/comp vs 8) -- cache 24->13 GB, ~1e-4 abs error,
  near-lossless (AP 0.241->0.253 within fold noise). Old complex64 caches
  all orphaned by the new cache key; deletable.
  Still IO-bound at ~120 s/epoch (`num_workers=0`) -- float16 didn't fix
  that (was never swap, just serialized reads); DataLoader `num_workers`
  is the remaining speed lever if this thread resumes.
  **Also not done:** GPU precompute path, Mamba-3, per-frequency Mamba
  lanes, projector encoder. (Band-match DONE 2026-08-29 -- hermitian_ssm
  now runs the same 8-40 Hz band as `temporal_graph_*`, so the
  cross-architecture comparison is finally apples-to-apples.)
  Disk (as of 2026-08-29): **`~/mne_data/dense_edge_cache` DELETED**
  (freed 63 GB -> 68 GB free, for swap headroom during the heavier encoder
  experiments; the 2026-08 Mac hard-crash was disk/compute exhaustion).
  Rebuild ~2-4 h cold if `temporal_graph_*` is run again; its 0.674 result
  CSVs are on disk. `~/mne_data/hermitian_ssm_cache/9d6ad0d850b8b8f0` =
  9.8 GB (band-match config, all 41 recordings, warm) -- the encoder
  experiments (#2/#3/#4/#6) all reuse it unchanged, no new cache writes.
- **HOUSEKEEPING: prune stale `hermitian_ssm_cache` keys (~14 GB), TODO.**
  As of 2026-08-31 `~/mne_data/hermitian_ssm_cache` = 24 GB across 5 keys.
  ACTIVE = `9d6ad0d850b8b8f0` (9.8 GB, per-seizure SUBDIR layout +
  `config.json`: k=6, 8-40 Hz, nfreqs=16, fd=2, td=16, diagonal=zero) --
  KEEP. STALE (flat `1_XX.npy`, NO config.json, all mtime 08-30, from the
  closed encoder/backend sweep): `e17cc670654f6b76` 9.9 G,
  `1623f9b20fc41f57` 3.1 G, `f86554f585cbcdb9` 1.1 G, `60300779bc34771f`
  0.35 G. Before `rm -rf`: confirm queued `anomaly_v2` / `edge_complex`
  (share the hermitian eigenpair cache) resolve to `9d6ad0d850b8b8f0` --
  they should (eigenpair key = band/k/downsample only; encoder_mode +
  mamba_backend are downstream, don't touch it). NOT urgent (18 GiB free
  disk) and does NOT help the dense_edge >RAM IO wall (that's item 2b).
  Do it during a queue lull.
- **STG-Mamba Design A (topology modulates the SSM delta) -- still
  unbuilt.** See `Epilepsy/graph_state_space_mamba_design.md`; the
  `hermitian_ssm` pipeline above deliberately took the simpler
  whole-graph-token route first (the doc's own recommended order).
- **Significance channel may be redundant in the canonical config, untested
  (2026-08-26)**: `_build_dense_edge_input`'s 4th dense-edge channel is
  `significance = (coh - threshold) / threshold`
  (`Epilepsy/pipelines/cwt_gnn_classifiers.py:4506`). Under
  `coherence_threshold_mode="fixed"` -- what `dense_edge_gru`/
  `dense_edge_mamba`'s canonical configs actually use
  (`run_pipelines.py:285`, `coherence_threshold=0.90`, a single scalar for
  every edge/frequency/trial) -- this is a pure affine transform of `coh`
  alone (same rank ordering, no new information); it would only carry
  real independent information under `coherence_threshold_mode="surrogate"`/
  `"surrogate_cluster"`, where the threshold varies per (edge, frequency,
  trial). Never actually ablated. Worth trying: rerun `dense_edge_gru` or
  `dense_edge_mamba` with the significance channel dropped (3-channel
  `[coh, sinφ, cosφ]` dense-edge input instead of 4), same protocol/seed,
  to see whether it changes accuracy/AP at all and how much it saves
  (smaller `dense_edge_conv` `in_channels`, narrower per-edge feature
  stack into the temporal model -- not the dominant cost, which is the
  coherence/phase construction itself, but a real secondary saving).
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
