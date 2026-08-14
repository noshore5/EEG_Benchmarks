# Session notes — ChannelSignalEncoder event-locality fix (2026-08-09)

**Superseded/reverted the same day.** A concurrent session ran this change
and found it regressed accuracy to near-chance (severe train/held-out gap,
suspected gradient-sparsity cause from gathering one timestep per event
instead of pooling the whole trial) — reverted `ChannelSignalEncoder` back
to `AdaptiveAvgPool1d(1)`. Full writeup:
[2026-08-09_mu_band_relaxation_scale_adaptive_kernel_and_channel_encoder_regression.md](2026-08-09_mu_band_relaxation_scale_adaptive_kernel_and_channel_encoder_regression.md).
Memory: [[sparse-evidence-gnn-channel-encoder-event-locality-fix]]. This
file is kept as-is below for the original architectural reasoning (still
correct) and the reverted implementation (preserved in
`session_notes/snapshots/2026-08-09_channel_encoder_per_timestep.md` for
reinstatement) -- just not the currently-active code.

Branch: `channel_subset`. Repo: `/Users/noahshore/Documents/CoherIQs/moabb`.
Continues from
[2026-08-08_sparse_evidence_gnn_surrogate_debug_plots.md](2026-08-08_sparse_evidence_gnn_surrogate_debug_plots.md).
Note: this file's canonical pipeline config already reflects the
2026-08-09 edge-topology change (72 directed edges -> 36 canonical
undirected pairs) documented in
[sparse_evidence_gnn_classifier.py](../moabb_pipelines/sparse_evidence_gnn_classifier.py)'s
module docstring -- that change landed earlier today, separately from
this note, and is not re-described here.

---

## Arc 1 — Architecture read-through: channel encoder was trial-global, not event-local

Asked to review `SparseEvidenceGNNCore`'s architecture end to end for
shape/parameter fit against the incoming data. Shapes all checked out (9
channels post-`channel_subset` -> `C(9,2)=36` canonical edges via
`upper_pair_indices`, `message_in = 5 + 2*channel_embed_dim = 21` feeding
`Linear(21, hidden_dim=8)`, `ChannelSignalEncoder`'s dilated-conv padding
preserves sequence length exactly, `edge_pair_idx`/`src_idx`/`dst_idx`
buffer overrides in `SparseEvidenceGNNCore.__init__` correctly propagate
into the parent's `_full_edge_wct_maps` since that method reads
`self.edge_pair_idx` dynamically rather than a captured local -- no bug
there).

One real architectural mismatch found:
[`ChannelSignalEncoder`](../moabb_pipelines/sparse_evidence_gnn_classifier.py)
ended in `nn.AdaptiveAvgPool1d(1)`, collapsing each channel's ENTIRE trial
to one static embedding vector. `SparseEvidenceGNNCore.forward()` then
broadcast that same vector onto every event on that channel, regardless of
when in the trial the event actually fired -- so `src_emb`/`dst_emb`
carried channel identity but zero timing information, while the
surrounding sparse-event/graph machinery (this pipeline's whole reason to
exist, per its own module docstring) is specifically about routing
time-localized evidence.

This gives a mechanistic explanation for something already flagged as an
unexplained empirical finding in
[[sparse-evidence-gnn-channel-encoder-dominates]]: the
`feature_ablation="zero_event_features"` test costing accuracy only ~2-3
points made sense once you see that the channel-embed block was the only
one of `sparse_message_mlp`'s two input blocks carrying a per-trial signal
the classifier could actually exploit per event -- the event-content block
(`t, freq, mag, sin, cos`) was doing comparatively little of the work not
because event content is inherently uninformative, but because the block
sitting next to it in the concat was structurally incapable of varying
with the event's own timing.

## Arc 2 — Fix: gather the encoder's per-timestep output at each event's own time index

Changed `ChannelSignalEncoder.forward` to return
`[batch, n_channels, embed_dim, n_time]` (dropped the final
`AdaptiveAvgPool1d(1)`) instead of `[batch, n_channels, embed_dim]`.
`SparseEvidenceGNNCore.forward()` now gathers `src_emb`/`dst_emb` per
event at that event's own (approximate) raw-sample time index rather than
indexing by channel alone:

```python
timestamp_frac = events_padded[..., 0]  # mean_time / (T_out - 1)
time_idx = (timestamp_frac * (n_time - 1)).round().long().clamp(0, n_time - 1)
src_emb = channel_feat[batch_idx, src_padded, :, time_idx]
dst_emb = channel_feat[batch_idx, dst_padded, :, time_idx]
```

Two things worth recording about this specific implementation:

- **The time-index mapping is an approximation, deliberately.**
  `events_padded[..., 0]` is a fraction of `_build_sparse_events`'
  post-smoothing coherence timeline (`T_out`), not raw `n_time` directly.
  `T_out` is at most `smooth_kernel_size[0] - 1` samples shorter than raw
  `n_time` (the smoothing conv2d is called with `pad_h=0` in
  `compute_events` -- valid-padding in time, confirmed by reading
  `common.make_gaussian_weight2d` and `_smooth_wct_maps`), so mapping the
  fraction onto raw `n_time` directly is off by at most a handful of
  samples out of ~1000 at this pipeline's `smooth_kernel_size=(5, 3)`.
  Negligible next to the encoder's own ~81-sample receptive field
  (`channel_encoder_dilation=5`), and it avoids threading a 6th tensor
  through the whole `TensorDataset`/fit-loop plumbing
  (`_prepare_features` -> `_build_model_from_features` ->
  `to_float_tensors` -> `forward`) just to carry an exact index. If
  `smooth_kernel_size` is ever widened a lot (e.g. back to the previously-
  tested `(25, 3)`), this approximation's error grows proportionally and
  is worth re-checking.
- **The combined `[batch_idx, src_padded, :, time_idx]` indexing avoids a
  large intermediate.** A naive `channel_feat[batch_idx, src_padded]`
  (channel-only gather, THEN a separate time-index gather) would
  materialize a `[B, max_count, embed_dim, n_time]` tensor first --
  wasteful at this pipeline's event counts. Indexing all three of
  `(batch, channel, time)` in one `__getitem__` call with a plain slice
  left on the `embed_dim` axis goes straight to `[B, max_count,
  embed_dim]`; verified this against a manual per-element reference gather
  in an interactive check before relying on it (mixed
  advanced-index/slice indexing reorders output dims per numpy's rules,
  so this was worth confirming rather than assuming).

Verified with a standalone smoke test (`SparseEvidenceGNNCore` on random
data, `compute_events` -> `forward` -> `.backward()`): forward pass shapes
correct, gradients flow into both `raw_x` and `channel_encoder`
parameters, and the zero-events edge case (`max_count=0`) doesn't crash.
Not yet run through an actual `moabb` training pass or the 4-subject
canonical suite quoted in `sparse_evidence_gnn_classifier.py`'s module
docstring -- that accuracy number (`subj1=0.801 subj2=0.557 subj3=0.947
subj4=0.539, mean=0.711`) predates this change and should be treated as
stale until re-run.

## Where things stand / open threads (this file)

- **Not yet validated end to end.** Next step is a real canonical run
  (`run_wct_gnn.py`'s `_make_sparse_evidence_gnn`, unchanged --
  `channel_encoder_dilation=5`, `hidden_dim=8`, `channel_embed_dim=8` all
  still apply, no config values needed to change for this fix) to see
  whether event-local embeddings move accuracy at all, and specifically
  whether a fresh `feature_ablation="zero_event_features"` run now costs
  MORE than the ~2-3 points measured under the old trial-pooled encoder
  (the mechanistic story above predicts it should, if event timing is
  genuinely informative once the channel embedding stops washing it out).
- `debug_sparse_evidence_gnn.py`'s architecture diagram
  (`_draw_message_diagram`) still describes the message path in terms
  that remain accurate (`concat -> [message_in] -> sparse_message_mlp`)
  but doesn't call `ChannelSignalEncoder`/`forward` directly, so it needed
  no changes -- flagged here only so a future reader doesn't assume the
  diagram was updated in lockstep and go looking for a diff that isn't
  there.
- [[sparse-evidence-gnn-seed-variance]] still applies to whatever the next
  validation run produces -- treat a single-seed number here with the
  same caution as every other number in this pipeline's history.
