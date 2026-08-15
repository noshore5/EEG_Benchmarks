# Full session summary — Sparse-Evidence-GNN debugging & tuning (2026-08-06)

Branch: `channel_subset`. Repo: `/Users/noahshore/Documents/CoherIQs/moabb`.
Session ID: `eb5a0829-d1cd-47a6-ac24-8855c60507f4`.

**Last updated: 2026-08-06** (end of day). Update this file in place for
further 2026-08-06 work — append new arcs and revise "Where things stand /
open threads" as items resolve. Later days get their own dated file; see
[2026-08-07_cross_subject_eval_and_phase_gate_direction_fix.md](2026-08-07_cross_subject_eval_and_phase_gate_direction_fix.md)
for the next day's work (cross-subject eval, surrogate significance
calibration, the phase-gate direction bug).

NOTE (2026-08-07): this file's header briefly referenced a third concurrent
session (`da07884f-c9c8-4f51-80d0-cf96277c3771`, "Arcs 9-15") with no
matching arc content anywhere in the file — evidence of another Claude Code
session editing this same notes file concurrently. That reference has been
removed here since it pointed at content that doesn't exist in this file;
if that session's own arcs turn up elsewhere, reconcile rather than
re-deleting.

This reconstructs the entire day's session (spanning multiple `/compact` cycles),
not just the tail captured by the most recent compaction.

---

## Arc 1 — Coherence/CWT visualization and bug hunting

Started with a question about where the wavelet coherence array gets built in
the pipeline, which led to building debug visualizations (saved to
`debug_plots/`): raw time-series + wavelet transforms for a single electrode
pair ("edge 0"), overlaid with the coherence array.

Iterative visual debugging surfaced real problems:
- Initial coherence plots were nearly all bright yellow (coherence ~0.8-1.0
  everywhere) — not the sparse, discriminative map expected.
- Traced to **downsampling** — CWT coefficients were being resampled from
  ~1000 points down to ~200 before coherence was computed, smearing out real
  structure and producing spuriously high coherence even in bands with zero
  power in the raw wavelet transforms (verified: 35Hz band showed all-yellow
  coherence despite 0 magnitude in both channels' transforms at that band —
  a clear artifact, not real coherence).
- Rebuilt the plot **without resampling** (native CWT resolution) — produced
  a visibly correct, sparse coherence map.
- Tuned the smoothing kernel by eye: tried larger kernels and a 21×9 kernel,
  settled on **(5,3)** as the best balance (larger kernels caused bad
  "vertical smearing" across scales/frequencies).
- Added **cone-of-influence (COI) masking** — coherence values outside each
  wavelet's valid time-frequency support are now excluded rather than
  contributing spurious high-coherence edges.
- Fixed a panel-orientation bug (panel 5 was plotted upside down).
- Investigated why every event passing the coherence/phase gate had phase
  angle ≈ 90° — resolved as part of the native-resolution fix (was a
  downstream symptom of the same resampling artifact).
- Once the event/edge extraction looked correct, plugged it into the main
  `sparse_evidence_gnn_classifier.py` pipeline for real training runs.

## Arc 2 — Speed optimization: precomputed FCWT cache

Running the full continuous wavelet transform (FCWT) on every channel-pair
connection on every forward pass was identified as wasteful. Implemented a
**cache of all possible channel-pair connections' FCWT results**, computed
once per trial rather than recomputed every forward pass — significant
speedup, confirmed to produce **identical scores** to the uncached version
(i.e., a correctness-preserving optimization, not a behavior change).

## Arc 3 — The 0.80 → 0.76 → ~0.71 regression investigation

After combining native-resolution coherence + COI masking + the FCWT cache,
subject-1 mean score dropped from a remembered ~0.80 to ~0.76, and further
canonical-setup runs fluctuated down to ~0.71-0.72, with the user unsure
which specific tweak was responsible.

Systematically eliminated candidates (coherence/phase thresholds at matched
event density, COI on/off, smoothing kernel size re-tested post-fix,
frequency range 30 vs 35Hz, raw event density, `common.py`) — **none of
these moved the needle**, all results landed in a tight [0.7606, 0.7623]
band. Root cause eventually isolated to `ChannelSignalEncoder`: two stacked
`Conv1d(kernel_size=9)` layers give a fixed 17-sample receptive field
regardless of sampling rate — ~68ms at native 250Hz, shorter than one
mu-band cycle (8-12Hz, ~83-125ms). The *old, buggy* pipeline's
`cwt_resample_n_time=200` had accidentally been giving that same 17-sample
window a ~5x larger real-time span by resampling `raw_x` too — so the old
"good" 0.80 score depended on the same bug that was breaking the coherence
array.

**Fix**: added a `channel_encoder_dilation` parameter (dilation=5 → receptive
field 81 samples ≈ 324ms, ~3.2 mu cycles), which grows the encoder's
real-time receptive field without resampling/discarding any signal and
without adding parameters. This recovered accuracy to parity (~0.80 on
subject 1) while keeping coherence, COI, and raw_x all at correct native
resolution.

**Important nuance surfaced during this investigation**: going native-resolution
did **not** improve accuracy beyond the old buggy pipeline's ceiling — it only
recovered parity. Re-tested a much larger smoothing kernel (`(25,3)`, restoring
~100ms effective smoothing) after the dilation fix: 0.8008 vs 0.7991 — still
noise-level. Native resolution's correctness fixes are a wash for this
architecture/task: real bug fixes with zero accuracy cost, but not a source of
extra accuracy on their own.

Along the way, discovered a **separate, unrelated bug**: `_write_group_artifacts`
in `run_wct_gnn.py` writes `scores_<run-id>.csv`/`summary_<run-id>.md` via a
plain overwrite with no file lock (unlike MOABB's own HDF5 store, which does
lock). Running two invocations against the same `--run-id` concurrently (e.g.
a background run plus an IDE-triggered run) can interleave writes into one
torn CSV — observed as a subject's rows appearing twice with different
wall-clock times, silently skewing the printed "Per pipeline mean" (a
row-weighted, not subject-balanced, average) toward the duplicated subject.
**Not yet fixed** — diagnostic is: if "Per subject/pipeline mean" and "Per
pipeline mean" don't reconcile by hand-averaging, that's this bug, not a real
result.

4-subject canonical validation (`run_canonical_setup.py`, `CANONICAL_VARIANT
= "sparse"`, subjects 1-4) at epochs=50/batch_size=16: subj1=0.801, subj2=0.557,
subj3=0.947, subj4=0.538, pipeline mean=0.711. subj2/subj4 sitting near chance
is a property of those specific subjects, not a pipeline defect — confirmed by
comparison against **EEGNet** on subject 2 (100 epochs): 0.603, i.e. also
barely above chance, so this isn't a Sparse-Evidence-GNN-specific weakness.

## Arc 4 — Logging infrastructure

Requested a proper logging system: every run now writes a full parameter dump
+ per-epoch timing to a log file automatically (`experiment_logging.py`,
producing `~/mne_data/<run-id>/experiment_<timestamp>.log` +
`experiment_latest.log` symlink), including the random seed(s) used. This is
the automatic system, distinct from the **manually curated**
`BCI/run_logs/` docs folder (populated only on explicit
request) — this distinction was clarified again later in the session when the
user worried not all runs were being saved.

## Arc 5 — Epoch count, batch size, and benchmark-structure tuning

- Explored raising epochs 50 → 100 → 150, and where the epochs setting lives
  in the config.
- Confirmed epochs 50→100 was a **real gain**: 0.799→0.828 on subject 1
  (single seed), and this held up in the 4-subject canonical run too
  (subj1=0.838, subj2=0.583, subj3=0.942, subj4=0.624, mean=0.747 at
  epochs=100/batch_size=8 — every subject improved except subject 3, already
  near ceiling). Full parameters/timing documented in
  `BCI/run_logs/2026-08-06_canonical-sparse_epochs100_bs8.md`.
- Explored batch_size: tried 32 (worse), then found smaller was better, tried
  4 explicitly (confirmed batch_size does **not** need to be a power of 2),
  compared epoch time between Sparse-Evidence-GNN and EEGNet directly.
- Raised `coherence_threshold` from 0.5 while lowering `phase_threshold_deg`
  to preserve event sparsity (0.5 judged too low for coherence on its own).
- Discussed whether the `0train`/`1test` score discrepancy (using
  `CrossSessionEvaluation`, MOABB's own native session labels — not a split
  the pipeline chooses) was meaningful or a benchmark-setup artifact.
  Confirmed as **real cross-session non-stationarity** for the tested subject,
  not noise (0train ~0.87, 1test ~0.79 consistently across 3 independent
  seeds, once the seed-43 outlier is excluded — see Arc 6).
- Tried shrinking the highest searched frequency 35→30Hz — did not move the
  needle; decoupled raw_x resolution from coherence resolution to test
  whether resampling only the raw signal (keeping coherence native) could
  recover ~0.80 — this was the same investigation thread that led to the
  `channel_encoder_dilation` fix in Arc 3.
- At end of a work day, explicitly asked to have all of that day's changes
  well-documented — resulted in the `run_logs/` entries and this memory
  system being populated.

## Arc 6 — Seed variance / robustness investigation (most recent arc)

Prompted by wanting to test batch_size and seed as variables (clarified:
epochs should generally **not** be dropped to 25/50 for that kind of
comparison — it changes what's being measured, so use the canonical epoch
count for fair comparisons).

1. **Single-seed bs=8 vs bs=10 comparison on subject 1**: looked like a wash
   (0.8383 vs 0.8304 mean).
2. **4-seed × 4-batch-size sweep** (seeds 42-45 × batch_size {4,8,16,32},
   subject 1, epochs=100): found (a) the `0train` vs `1test` gap is mostly
   real (~0.87 vs ~0.79 across 3 clean seeds), and (b) **seed 43 is
   catastrophically bad specifically on the `0train` direction** — collapsed
   to ~0.57 (a ~30-point drop) at both batch_size=16 and batch_size=4, while
   its own `1test` score stayed completely normal (0.798-0.802, actually the
   *best* of the four seeds) at both. batch_size=8 was the only setting that
   didn't reproduce the failure across those 4 seeds.
3. **8-seed × 2-batch-size follow-up sweep** (seeds 42-49 × batch_size {8,10},
   subject 1, epochs=100, run 3-way parallel via `xargs -P3`, ~14.5 min wall
   clock for all 16 runs): seed 43 collapsed on `0train` again at bs=10
   (0.548, `1test` stayed normal at 0.804) — now the **third** batch size
   (after 4 and 16) where seed 43 specifically fails. bs=8 had **zero
   failures** across all 8 seeds (0train std 0.015, 1test std 0.024).
   Excluding seed 43, bs=10 and bs=8 are statistically indistinguishable.
   Documented in full in
   `BCI/run_logs/2026-08-06_sparse-evidence-gnn_bs8-vs-bs10_8seed-sweep.md`.
4. **Mechanistic explanation for the bad-seed effect** (conceptual, not a new
   test): small model (hidden_dim=8, channel_embed_dim=8) + small data (144
   trials/session) is inherently seed-sensitive; `seed` controls weight init,
   batch shuffle order, AND the composition of the tiny (~29-trial)
   `validation_split=0.2` slice used for "restore best model by val_loss"
   checkpoint selection — noise in that small slice can occasionally pick a
   checkpoint that looks good on validation but generalizes badly to the true
   held-out session. The fact that seed 43 fails only on `0train` (not
   `1test`, same seed) points to an *interaction* between seed 43's
   validation-split draw and the `0train` session's specific data, not a
   globally bad seed. bs=8's apparent immunity is most likely "routes around
   the trap" for this specific seed rather than structural immunity, since
   seed 43 fails across most *other* batch sizes tested. A concrete follow-up
   test was offered but not yet performed: compare per-epoch val_loss vs.
   true `1test`-accuracy trajectories for seed 43 @ bs=10, to check whether
   an earlier, unselected checkpoint would have generalized better (a
   "smoking gun" test for the checkpoint-selection theory).
5. Clarified infrastructure questions along the way: background Bash-tool
   runs are ordinary detached child processes on the user's own Mac (visible
   via `ps aux`/Activity Monitor), not hidden remote infrastructure; `run_logs/`
   is manually curated (only populated on request) vs. the automatic
   per-run experiment logs/results artifacts under `~/mne_data/` which are
   always written regardless.
6. **Dataset size question**: "is there more data we can use? I thought
   subject 1 had more samples" — verified via direct MOABB API calls that
   subject 1 does **not** have more raw trials than any other subject: every
   subject has 288 annotated events/session × 2 sessions, uniformly (verified
   for subjects 1 and 2, structurally guaranteed for all 9 by the dataset's
   fixed 4-class/6-run/48-trial-per-run protocol). Two real "more data"
   levers do exist, though: (a) `LeftRightImagery` currently discards half
   the trials (144 of 288/session) by dropping the `feet`/`tongue` classes —
   switching to `MotorImagery(n_classes=4)` would double per-session data but
   requires an architectural change (binary → 4-way classification); (b) all
   training/eval is currently per-subject in isolation
   (`CrossSessionEvaluation`) — pooling subjects would add data but answers a
   different question (cross-subject vs. cross-session generalization).
   Neither is a drop-in change; both were surfaced as options, not yet
   requested for implementation.

## Where things stand / open threads

- **Not yet implemented**: fix for the `run_wct_gnn.py` concurrent-write race
  (atomic write + lock around `_write_group_artifacts`).
- **Not yet performed** (offered, no answer yet): the val_loss-vs-true-accuracy
  trajectory check for seed 43 @ bs=10, to directly test the
  checkpoint-selection theory.
- **Not yet requested**: implementing a 4-class paradigm switch
  (`MotorImagery(n_classes=4)`) or multi-subject pooled training — both
  surfaced as "more data" options but not asked for yet.
- **Canonical config as of end of session**: epochs=100, batch_size=8 (grid
  default in `run_wct_gnn.py` was later changed to 10 by the user directly),
  `channel_encoder_dilation=5`, `coi_enabled=True`, native-resolution CWT/
  coherence (no resampling), `smooth_kernel_size=(5,3)`,
  `coherence_threshold=0.5`, `phase_threshold_deg=30.0`, `highest=35.0`,
  `lowest=8.0`, subjects 1-4 canonical validation mean ≈ 0.747.
- **Continued 2026-08-07**: see
  [2026-08-07_cross_subject_eval_and_phase_gate_direction_fix.md](2026-08-07_cross_subject_eval_and_phase_gate_direction_fix.md)
  for cross-subject evaluation, surrogate significance calibration, and the
  phase-gate direction bug (reported, "fixed" with `.abs()`, then reverted
  after root-causing the regression).

## Related persistent memory files (`~/.claude/projects/.../memory/`)

- `sparse-evidence-gnn-native-resolution-fix.md` — Arc 3 in full detail.
- `run-wct-gnn-concurrent-write-race.md` — the concurrent-write bug from Arc 3.
- `sparse-evidence-gnn-seed-variance.md` — Arc 6 in full detail, both sweeps.

## Related run_logs entries (repo, manually curated)

- `BCI/run_logs/2026-08-06_canonical-sparse_epochs100_bs8.md`
- `BCI/run_logs/2026-08-06_sparse-evidence-gnn_bs8-vs-bs10_8seed-sweep.md`
