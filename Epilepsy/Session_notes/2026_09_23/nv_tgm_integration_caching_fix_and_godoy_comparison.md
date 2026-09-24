# temporal_graph_mamba on NV Pat1: new integration, a caching-regime bug
# found and fixed same-day, and why a godoy_tmc comparison is still
# pending (2026-09-23)

Consolidation note for a long session spread across `scripts/run_nv_pat1_tgm.py`
(new file), two numbered `FAILURE_LOG.md` entries, and ~10 individual
`eeg-run` results/spot-launch attempts. This is the pointer-heavy summary;
`FAILURE_LOG.md` #17 and the individual result CSVs under
`Epilepsy/results/{godoy_tmc,temporal_graph_mamba}/nv/` have the full
evidence trail.

## What this session built

`scripts/run_nv_pat1_tgm.py`: applies `temporal_graph_mamba`'s dense-edge
CWT->Mamba classifier (`StreamingSparseEvidenceGNNClassifier`,
`event_mode="temporal_graph"`, same class `run_pipelines.py --pipeline
temporal_graph_mamba` uses on CHB-MIT) to the Kuhlmann NeuroVista (NV)
Pat1 dataset, instead of the raw-EEG `GodoyTMCClassifier` `run_nv_pat1.py`
already used. Same StratifiedGroupKFold/segment-or-block-grouping/
segment-level-scoring machinery as `run_nv_pat1.py` (deliberately not
re-derived); only the classifier and its cache wiring are new. Key
correctness fix baked in from the start: `PREDICTION_TEMPORAL_GRAPH_MAMBA_
PARAMS`'s `sampling_rate=256` default is CHB-MIT's native rate -- overridden
to NV's actual decimated rate (`400Hz/--decimate`) or CWT/coherence would
be computed against the wrong sample rate entirely.

## The caching bug: wrong twice before it was right

**Attempt 1** (script's first version) passed no cache at all
(`cwt_cache`/`dense_edge_cache_dir`/`dense_edge_mem_cache` all `None`), on
the stated assumption that NV's scale ("hundreds of windows") made
per-fold recompute acceptable. Wrong -- full-scale NV Pat1 is ~16,180
sub-windows, and with zero caching every batch of every epoch recomputed
its own dense-edge/CWT tensors from scratch, with no reuse even *within*
one `fit()` call. `nv-pat1-tgm-block-6fold` ran 30+ minutes still stuck on
fold 0's first epoch before being killed.

**Attempt 2** (the actual fix, commit `ab8f3cc`) did NOT just copy
CHB-MIT's `leave_one_seizure_out_prediction` regime either -- that relies
mostly on a GPU-resident `DenseEdgeMemCache` (~15-23GB typical budget).
Full-mesh dense-edge entries run ~15MB/trial at CHB-MIT's 23-channel
montage (`dense_edge_cache.py`'s own docstring, "247/253 edges"); at
16,180 unique NV windows that's ~243GB of unique entries -- 10-15x any
single GPU's VRAM, so a GPU-only cache would constantly evict, same
failure mode as no cache, just with extra bookkeeping. (NV is actually
16 channels, not 23 -- E scales ~C^2, so real NV per-trial cost is closer
to half that, ~7.5MB/trial / ~115GB total working set. Still far bigger
than VRAM.)

**The real fix is two-tier**: disk cache as the floor (`DiskCWTCache` +
`dense_edge_cache_dir`, content-addressed, bounded by the 150GB gp3 EBS
volume `eeg-run-spot.sh` already attaches by default, not VRAM), with the
GPU-resident `DenseEdgeMemCache` as an opt-in accelerator on top
(`--dense-edge-gpu-cache`) for whichever fold's windows are currently hot.
Cleared at each fold's train/eval boundary (disjoint window sets, so
training's entries are dead weight for eval -- same reasoning as
`run_pipelines.py`'s 2026-09-19 eval-boundary fix) but kept warm *across*
folds, since NV's folds share most of their training windows just like
CHB-MIT's LOSO folds do.

**Validated same-day**: `nv-pat1-tgm-cache-smoke-v2` (60-segment smoke)
went from 0% -> 100% disk-cache hit rate after one precompute pass;
epochs dropped from "never finishes" to ~2.5-3s each on that tiny slice.
**Caveat surfaced when asked directly:** that "few seconds" number does
NOT hold at full scale -- it reflects ~24 batches/epoch at smoke scale vs
~420 batches/epoch full-scale (~17x more), so warm full-scale epochs
landed at ~76-77s each once training stabilized (epoch 10+), which is
close to the expected ~17x scaling from batch count, not a broken promise
but a data point I should have qualified better the first time I reported
smoke-test speed as a general result.

## Spot capacity was the dominant time sink, not code

Once the cache fix worked, **4 of 5 launch attempts for the full-scale run
that day failed on "no spot capacity on any (type,AZ) candidate"** across
every `g5.2xlarge`/`g6.2xlarge` AZ tried in `us-east-1` -- this hit BOTH
the godoy_tmc retries (`nv-pat1-block-6fold-v2`, `-v3`) and the tgm retries
(`nv-pat1-tgm-block-6fold-v2`) before `-v3` finally landed on
`g6.2xlarge`/`us-east-1b`. Pure AWS availability, not a bug -- see
`FAILURE_LOG.md` #17 for the full timeline. Plain retries (same
instance-type list) were the only lever that worked; no instance-type
widening was needed once capacity freed up.

## No valid godoy_tmc vs temporal_graph_mamba comparison exists yet -- caught and corrected mid-session

`nv-pat1-tgm-block-6fold-v3` (temporal_graph_mamba, today) completed
cleanly: 6/6 folds, `roc_auc` 0.768-0.828, no stalls, no OOM, correctly
segment-level-scored (`n_train_windows`/`n_test_windows`/`n_test_segments`
schema). **First draft of this note compared it against
`nv_pat1_stratified_kfold_blockgrouped_20260922-141154.csv` (godoy_tmc,
roc_auc 0.882) and called godoy the winner -- that was wrong and got
corrected before this note was finalized.** That CSV uses the OLD
`n_train`/`n_test` schema (sub-window counts, ~2600-2860/fold), meaning it
predates the segment-level-aggregation scoring fix `CONTEXT.md`'s
2026-09-22 entry documents -- that entry explicitly says these sub-window-
scored numbers "must never be pooled" with corrected ones, and that a
corrected full block-grouped godoy run was still the "top pending task"
for a future session. Today's 3 failed godoy relaunch attempts
(`nv-pat1-block-6fold-v2`/`v3`, both killed by no-spot-capacity before
ever starting -- see `FAILURE_LOG.md` #17) never produced that corrected
replacement.

**So: there is currently no existing godoy_tmc full-scale block-grouped
run with correct (segment-level) scoring to compare tgm's 0.801 roc_auc
against.** The only correctly-scored godoy_tmc NV results on `main` are
smoke-scale (`--limit 60`, `segmentgrouped_20260922-13{3702,4024}.csv`
and `..._20260923-093547.csv`), not full-scale, not block-grouped, and
too small (~60 segments) to be a meaningful comparison point. A real
comparison needs a fresh full-scale `--grouping block` run of
`run_nv_pat1.py` (godoy_tmc) under the current, fixed scoring -- not yet
done as of this session's end.

## What's actually new on `main` after this session

- `scripts/run_nv_pat1_tgm.py` (new file): temporal_graph_mamba on NV
  Pat1, with `--disable-disk-cache`/`--dense-edge-gpu-cache`/
  `--dense-edge-gpu-cache-gb`/`--dense-edge-gpu-cache-headroom-gb` CLI
  flags mirroring `run_pipelines.py`'s conventions.
- `scripts/fetch_kuhlmann_nv_s3.sh`: `--no-progress` on the `aws s3 cp`
  (fixed log-spam-hides-real-errors, `FAILURE_LOG.md` #15), extraction
  robustness fix for tarballs without a `Pat<N>Train/` top-level folder
  (`FAILURE_LOG.md` #16).
- `scripts/eeg-run-spot.sh`: `--no-progress --exclude "kuhlmann_nv/*"` on
  the generic dataset sync (same log-spam fix, avoids a redundant
  re-pull).
- `datasets/epilepsy/kuhlmann_nv.py`: `_try_fetch_from_s3` self-fetch
  wiring (pre-existing this session, referenced for context).
- `FAILURE_LOG.md` #15-17: log-spam/tarball-layout/host-RAM-OOM/
  no-spot-capacity entries for this dataset's spot-launch history.
- Results: `nv_pat1_tgm_stratified_kfold_segmentgrouped_20260923-102247.csv`
  (first-ever tgm-on-NV smoke), `..._111835.csv` (cache-fix validation
  smoke), `..._blockgrouped_20260923-140142.csv` (full-scale
  temporal_graph_mamba, roc_auc 0.801 across 6 folds -- currently the only
  correctly-scored full-scale block-grouped NV Pat1 result on `main`,
  pending a matching godoy_tmc run to compare against).

## Open threads for next session

1. **Top pending task, carried over from 2026-09-22's CONTEXT.md entry,
   still not done:** run `run_nv_pat1.py` (godoy_tmc) full-scale,
   `--grouping block`, 6-fold, under the current (segment-level-correct)
   scoring, so there's finally a real apples-to-apples number against
   today's tgm result (roc_auc 0.801). Every existing godoy_tmc NV
   block-grouped result predates the scoring fix and must not be cited
   as current.
2. Once that comparison exists: is temporal_graph_mamba actually weaker
   than godoy_tmc on NV Pat1, or was this session's stale-CSV comparison
   just wrong in both directions (unit AND possibly which one wins)? CHB-
   MIT's nfreqs=16 tgm run scores much higher there (roc_auc 0.941-0.944,
   see 2026-09-20's note) -- worth knowing whether that architecture
   advantage holds, reverses, or vanishes on NV once compared correctly.
3. Before relaunching ANY benchmark, check `Epilepsy/results/<pipeline>/
   <dataset>/`'s CSV *schema* (not just that a file with a plausible name
   exists) for an existing comparable run first -- this session's first
   draft of this very note cited a real, existing file that turned out to
   be scored in the wrong unit, which is a worse trap than a missing file
   since it looks like a valid comparison until the schema is checked.
4. `run_nv_pat1_tgm.py` has no fold-level checkpoint/resume -- a spot
   reclaim mid-run loses all progress. Not yet a real cost (today's full
   run finished in one shot), but worth adding if this becomes a
   routinely-rerun pipeline the way CHB-MIT's `--checkpoint-dir` already
   is.
