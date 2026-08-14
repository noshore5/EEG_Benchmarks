# Session notes — surrogate-cache speedup confirmation + progress indicator (2026-08-09)

Branch: `channel_subset`. Repo: `/Users/noahshore/Documents/CoherIQs/moabb`.
Same-day sibling of
[2026-08-09_channel_encoder_event_locality_fix.md](2026-08-09_channel_encoder_event_locality_fix.md)
(unrelated thread — that file covers a `ChannelSignalEncoder` architecture
experiment that was tried and reverted same day; this one covers the
72→36-edge topology change's surrogate-cache behavior, landed earlier
today, separately from both).

---

## Arc 1 — What `ChannelSignalEncoder` is (context question)

Walked through the class for the user: a small dilated 1D-CNN
(`kernel_size=9`, `dilation=5` in the canonical config → 81-sample / ~324ms
receptive field) that embeds each channel's whole raw trial into one
`embed_dim`-length vector via `AdaptiveAvgPool1d(1)`, concatenated onto
every event's message before `sparse_message_mlp`. Tied back to
[[sparse-evidence-gnn-channel-encoder-dominates]] — this embedding is what
the earlier `feature_ablation` test showed drives ~all of this pipeline's
accuracy, not the surrogate-gated event content.

## Arc 2 — Confirmed the 72→36-edge topology change actually sped up / re-keyed the surrogate cache

Checked `~/mne_data/surrogate_null_cache/` directly rather than trusting
assertion:

- Old (72 directed edges, dated Aug 7–8): 1,444 files, avg ~827KB, 1.19GB
  total.
- New (36 canonical edges, dated Aug 9, keyed by `edge_topology=
  "canonical_undirected"` added to `surrogate_null_cache_key`): 1,479
  files, avg ~452KB, 669MB total.
- Clean bimodal size split (nothing between ~450KB and ~800KB) confirms no
  old/new key collisions — `values_grid`'s `[E, F, 201]` shape genuinely
  halved with `E`.

Live-timed a genuine cold cache miss (subject 6, trial 47, never touched
before): 19.55s cold vs 9.62s on an immediate rerun (cache hit) — same
MOABB-loading overhead in both, so the delta (~9.9s) isolates the real
surrogate-compute cost: 100 surrogates × 36 edges × 16 freqs × native-res
CWT + COI. Corrected an earlier, unverified claim from before this
session's compaction ("~2s for 144 trials") — that likely described a
mostly-warm cache, not a real from-scratch benchmark. The trustworthy
number is **~10s/trial cold**, so a fully-cold subject (144 trials) is
more realistically ~20–25 min serial, not seconds.

Noted but not acted on: the old 72-edge cache entries (~1.19GB) are now
permanently orphaned — nothing will read them under the new key.

## Arc 3 — Added a progress indicator for the surrogate cache-miss path; found 2 real bugs doing it

User asked for a progress indicator on "the caching script" (the ~10s/trial
cold computation from Arc 2). `_precompute_sparse_events` already had a
`tqdm` bar, but it's scoped to looping over *trials* — invisible to a
single-trial caller like `debug_sparse_evidence_gnn.py`, which never
reaches that loop.

- Added an announce-`print` + `tqdm` bar directly inside
  `_surrogate_null_percentile_grid`'s 5-chunk surrogate loop (20
  surrogates/chunk), gated on the existing `self.verbose >= 1` convention.
- Set `verbose=1` in `debug_sparse_evidence_gnn.py`'s
  `CANONICAL_SPARSE_KWARGS` so the debug script actually shows it. This
  surfaced a real collision: the script's `SparseEvidenceGNNClassifier(...)`
  call hardcoded `verbose=0` alongside `**kwargs` — fixed by dropping the
  hardcoded value.
- Found a second, sneakier bug: right after construction, the script calls
  `clf._init_cwt_gnn_classifier(..., verbose=0, ...)` to set up
  training-loop scaffolding — that call's own hardcoded `verbose=0`
  flows into `_init_torch_classifier`, which does `self.verbose = verbose`
  unconditionally, silently reverting the constructor's `verbose=1` right
  back to 0. Fixed by passing `verbose=clf.verbose` there instead of a
  second hardcoded literal.
- Verified end-to-end on 5 different never-before-cached subject/trial
  combos — announce print + live `5/5 chunk` tqdm bar both fire correctly
  on genuine cache misses (~1.4–2.6 chunk/s observed).

## Arc 4 — Transient `AttributeError`, diagnosed as a concurrent-edit artifact

First fresh-subject test (subject 7, trial 30) hit
`AttributeError: 'SparseEvidenceGNNClassifier' object has no attribute
'mu_band_surrogate_percentile'` inside `_percentile_vector`. Did not
reproduce on an immediate retry (same subject/trial) or on 4 subsequent
fresh subjects; a direct construction test confirmed the attribute is
always set unconditionally in `__init__`. User's explanation: the source
file was mid-save in another editor/terminal at that exact moment, so this
process's import briefly picked up a half-written file. Consistent with
the one-off, non-reproducible pattern — treated as resolved, no code
change needed.

## Where things stand

- Surrogate-cache progress feedback now works both for full-dataset
  precompute (`_precompute_sparse_events`, pre-existing bar) and
  single-trial debug-script use (this session's addition).
- Real per-trial cold-cache cost is ~10s at `surrogate_count=100`,
  36 edges — worth remembering next time a "why is this slow" question
  comes up instead of re-deriving it.
- Old 72-edge cache entries (~1.19GB) are dead weight on disk; cleanup not
  requested, not done.
