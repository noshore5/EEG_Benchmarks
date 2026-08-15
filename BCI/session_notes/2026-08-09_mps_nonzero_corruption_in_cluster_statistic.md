# Session notes — MPS `.nonzero()` corruption in run-consolidation (2026-08-09)

Branch: `channel_subset`. Repo: `/Users/noahshore/Documents/CoherIQs/moabb`.
Continues from
[2026-08-09_channel_encoder_event_locality_fix.md](2026-08-09_channel_encoder_event_locality_fix.md)
-- that file's own fix was reverted by a concurrent session (see its header
note); this file is about an unrelated, independent bug found while trying
to validate it, and is unaffected by that revert.

---

## Arc 1 — A crash while validating a (since-reverted) architecture change

While re-running the canonical 4-subject Sparse-Evidence-GNN suite
(`coherence_threshold_mode="surrogate"`, `surrogate_percentile=90.0`,
matching a same-day `sweep-pct90-4subj` baseline: subj1=0.808, subj2=0.560,
subj3=0.963, subj4=0.567, mean=0.725), subject 1 completed cleanly but the
run crashed partway into subject 2 with:

```
RuntimeError: index 7534 is out of bounds for dimension 0 with size 720
```

inside `SparseEvidenceGNNCore._max_cluster_statistic`'s
`scatter_reduce_(0, be_of_run, cluster_mass_cpu, reduce="amax")` call. This
is entirely unrelated to the ChannelSignalEncoder work in progress
elsewhere in the file -- `_max_cluster_statistic` is part of the surrogate
null-distribution calibration path
(`coherence_threshold_mode in {"surrogate", "surrogate_cluster"}`), a
pre-existing function this session didn't otherwise touch.

## Arc 2 — Root cause: a real PyTorch MPS backend bug in `.nonzero()`

Added temporary diagnostics and reproduced directly against real subject-2
data (`surrogate_cache_enabled=False` to force fresh computation, bypassing
the disk cache). Confirmed the corruption is present **immediately after**
`gate_r.nonzero(as_tuple=False)` returns -- not introduced by any of this
function's own arithmetic:

```
[DEBUG3] CORRUPTED right after nonzero()! R=11520 T=997
valid_pos.shape=(1598799, 2) row_idx max=122222 time_idx max=241502
```

`gate_r` at that point is a correctly-shaped, **contiguous** `(11520, 997)`
boolean tensor on `mps:0` (`surrogate_device="auto"` resolves to MPS on
this machine, and the surrogate/cluster-mode precompute pathway moves the
whole `helper` model + its CWT/coherence tensors there -- see
`resolve_best_available_device`'s docstring for why: the conv2d-heavy
coherence smoothing is ~10x faster on MPS here). `row_idx`/`time_idx` of
122222/241502 are mathematically impossible for a tensor with only 11520
rows and 997 columns -- `.nonzero()` on this real MPS tensor returned
indices that don't correspond to any real element position.

Ruled out via isolated tests before concluding this (each compared CPU vs.
MPS on the identical op):
- Plain `.nonzero()` on a large random boolean tensor -- matched.
- `torch.unique(..., return_inverse=True)` on realistic-scale int64 run-ids
  -- matched.
- The exact `permute(0,1,3,2).reshape(B*E*F, T)` pattern this code uses,
  then `.nonzero()`, on random data at the real shape (20, 36, 997, 16) --
  matched, and `gate_r.is_contiguous()` was `True` both times.

None of these synthetic repros triggered it -- only the real, chained
`(coh > threshold) & (phase.abs() > rad) & coi_valid` boolean expression on
real subject-2 coherence data did, and only sometimes (a first attempt with
`surrogate_count=10` and a 24-trial slice didn't crash; a full 288-trial,
`surrogate_count=100` fresh run against subject 1 alone also didn't crash;
subject 2 did, both in the original multi-subject run and in an isolated
full-288-trial repro). **Data-dependent, not a blanket "nonzero is broken on
MPS" issue** -- meaning on other inputs it could plausibly return silently
*wrong* (not out-of-bounds, so non-crashing) results instead of throwing,
which is the more concerning failure mode.

## Arc 3 — Second exposure: `_build_sparse_events` shares the same pattern

`SparseEvidenceGNNCore._build_sparse_events` -- the function that builds
the **real** (non-null) events actually used for training, not just this
null-calibration statistic -- has the identical
`permute().reshape().nonzero()` + `torch.unique(return_inverse=True)`
run-consolidation block. In `coherence_threshold_mode in {"surrogate",
"surrogate_cluster"}`, `SparseEvidenceGNNClassifier._precompute_sparse_events`
moves `helper` (and therefore this function's execution) to
`surrogate_torch_device` too -- so real training events were exposed to the
exact same risk, just without a bounds-check to crash on. This raises the
possibility that some already-recorded pipeline weirdness (not confirmed,
not chased further this session) could trace back to this rather than a
statistical/architectural cause -- worth keeping in mind, not claiming.

## Fix

Both functions now force the `nonzero()`/`unique()` index math onto CPU:

- `_max_cluster_statistic`: the whole run-consolidation block (from the
  `permute().reshape()` calls through the final `scatter_reduce_`) now runs
  entirely on CPU copies of `gate`/`coh`/`phase`/`cluster_forming_threshold`;
  only the final `[B, E]` result is moved back via `.to(coh.device)`. This
  extends a pattern the code already had -- `scatter_reduce_` was *already*
  forced to CPU "for MPS portability," per its existing comment; this just
  moves that same defensive treatment earlier, to where the bug actually is.
- `_build_sparse_events`: narrower fix, since much more of the function
  (the real event tensors, `self.dst_idx`/`self.src_idx`/`freqs_batched`
  indexing, the final padded output tensors) needs to stay on the original
  device. Only the `nonzero()`/`unique()` step runs on CPU copies of
  `gate_r`/`phase_r`; the resulting `row_idx`/`time_idx`/`inverse` are
  converted back to `gate.device` immediately after, and everything else in
  the function is unchanged.

Verified:
- A synthetic `_max_cluster_statistic` call on MPS (random data, same
  shapes) no longer errors.
- The exact real-data repro that crashed before (subject 2, fresh
  `surrogate_cache_enabled=False`, full 288 trials, `surrogate_count=100`,
  `surrogate_percentile=90.0`) now completes clean.
- A full 4-subject canonical run (`surrogate_percentile=90.0`, same config
  as the `sweep-pct90-4subj` baseline) completed without any crash across
  all 4 subjects.

## Arc 4 — Accuracy after the fix (confounded, but informative)

The validation run's `ChannelSignalEncoder` had already been reverted back
to `AdaptiveAvgPool1d(1)` by the time it ran (see this file's header), so
these numbers reflect **the original architecture + this MPS fix**, not the
event-locality change:

| subject | baseline (`sweep-pct90-4subj`, pre-fix) | post-MPS-fix |
| --- | --- | --- |
| 1 | 0.808 | 0.847 |
| 2 | 0.560 | 0.553 |
| 3 | 0.963 | 0.969 |
| 4 | 0.567 | 0.666 |
| **mean** | **0.725** | **0.759** |

Subject 4 moved the most (+0.099), subjects 1 and 3 improved modestly,
subject 2 is flat within noise. Plausible given the bug lived in the
surrogate-threshold/cluster-null machinery -- corrupted thresholds would
selectively distort which (edge, freq, time) cells pass the significance
gate, plausibly worse for some subjects' data than others. **Single seed,
single run** -- see [[sparse-evidence-gnn-seed-variance]] before treating
this delta as settled; the crash-fix itself is solid (mechanistically
understood, reproduced, and directly verified), but whether it robustly
moves accuracy in this specific direction needs a repeat run/seed to
confirm.

## Where things stand / open threads (this file)

- Only `_max_cluster_statistic` and `_build_sparse_events` were fixed --
  these were the two call sites found via this crash and a targeted search
  for the same `.nonzero()`/`permute+reshape` pattern
  (`grep -n "nonzero(as_tuple=False)"`) in this file. Not audited: whether
  other MPS-resident code elsewhere in the broader `moabb_pipelines/`
  package (`wct_evidence_gnn_classifier.py`, `xwt_phase_gnn_classifier.py`,
  etc.) has the same pattern -- none of those currently run their heavy
  tensor ops on MPS by default (see `resolve_best_available_device`'s own
  docstring on why only the surrogate-calibration path opted in), so the
  exposure is believed limited to this file, but not independently verified
  elsewhere.
- Not root-caused at the PyTorch/MPS-internals level (e.g. which specific
  op/dtype/size threshold triggers it) -- fixed by avoiding the affected
  device entirely for this specific computation, not by understanding
  Apple's/PyTorch's MPS kernel bug itself. If this resurfaces on a newer
  torch/macOS version, worth re-checking whether the CPU-forcing here is
  still necessary.
- The accuracy deltas in Arc 4 are a single seed/run and should be
  re-confirmed, ideally alongside whatever the next real
  `ChannelSignalEncoder` experiment turns out to be (see the other
  concurrent session's notes for that thread).
