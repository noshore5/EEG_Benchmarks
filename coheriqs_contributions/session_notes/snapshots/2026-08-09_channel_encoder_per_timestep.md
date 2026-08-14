# Snapshot: ChannelSignalEncoder per-timestep embedding (reverted 2026-08-09)

Landed today (2026-08-09) from a different Claude Code session working
against this same file concurrently with the one that produced this note --
not yet re-validated against the 4-subject canonical suite when it landed
(its own docstring said so). Reverted back to the pooled
(`AdaptiveAvgPool1d(1)`) version in `sparse_evidence_gnn_classifier.py`
after results across several single-variable threshold/kernel experiments
all came back near-chance for subject 2, and the user reported "almost no
results" more broadly -- suspected cause: gradient signal from
`channel_encoder`'s conv layers only reaches the ~handful of raw-sample
positions actual events land on per step (out of ~1000 timesteps), instead
of every position via the old pooled-average, which is a much sparser/
noisier training signal that could plausibly tank accuracy on its own,
independent of any coherence-threshold tuning.

**Rationale for the change (from its own docstring, still worth
revisiting later):** pooling the whole trial into one static per-channel
vector meant every event on a given channel got the IDENTICAL src/dst
embedding regardless of when in the trial it fired -- i.e. zero timing
information in that feature block, which is a real conceptual gap given
the whole point of the sparse-event architecture is routing
time-localized evidence. Consistent with feature_ablation=
"zero_event_features" barely moving accuracy (see
[[sparse-evidence-gnn-channel-encoder-dominates]] in project memory) --
under the OLD pooling, the channel-embed block was the only one of the two
message-MLP input blocks that could vary meaningfully trial-to-trial.

**If picking this back up:** the gradient-sparsity concern above is the
first thing to test/address -- e.g. an intermediate design (pool over a
LOCAL window around each event's time index, not the whole trial and not
a single raw sample) might get event-relative timing back without cutting
gradient coverage down to single points. Worth deciding whether the
"zero_event_features ablation barely moves accuracy" finding it was meant
to address is actually still a problem worth solving, or was itself
measured under other confounds active at the time.

## `ChannelSignalEncoder` (was `sparse_evidence_gnn_classifier.py` lines ~139-228)

```python
class ChannelSignalEncoder(nn.Module):
    """Lightweight learned per-channel signature: gives each graph node an
    actual representation of its raw signal shape, not just the
    coherence/timing scalars that arrive on its edges. A much smaller
    version of WCTEvidenceGNNCore's feature_conv, not a copy of it.

    `dilation` controls the receptive field in real time, independent of
    input length. Two stacked kernel_size=9 convs give a fixed 17-SAMPLE
    receptive field (RF = 1 + (9-1) + (9-1)) regardless of sampling rate --
    at native ~250Hz that's only ~68ms, shorter than a single mu-band cycle
    (8-12Hz, ~83-125ms), so at native resolution this encoder was
    architecturally blind to oscillatory envelope shape and could only see
    sub-cycle sample texture (empirically confirmed: resampling ONLY raw_x
    to 200 samples -- i.e. giving this encoder the same 17-sample window a
    ~5x larger real-time span -- recovered accuracy from ~0.76 to ~0.80
    while leaving coherence/COI untouched at native resolution). Dilation
    grows the *time* the kernel spans without growing kernel size (avoiding
    both extra parameters and a large-kernel's own smoothing effect), and
    without touching/resampling the input signal itself (so no real
    high-frequency content is discarded, unlike naively downsampling raw_x).
    dilation=5 with kernel_size=9 gives RF = 1 + 8*5 + 8*5 = 81 samples =
    ~324ms at 250Hz -- ~3.2 cycles of an 8-12Hz mu rhythm.

    2026-08-09: no longer ends in AdaptiveAvgPool1d(1). Pooling over the
    WHOLE trial collapsed each channel to one static vector that got
    broadcast onto every one of that channel's events regardless of when in
    the trial the event fired -- i.e. src/dst embeddings carried zero timing
    information, only channel identity, while the graph/event machinery
    around them (SparseEvidenceGNNCore.forward's whole reason to exist) is
    specifically about routing TIME-LOCALIZED evidence. This is consistent
    with the feature_ablation="zero_event_features" finding that accuracy
    barely moves without the event-content block (see class docstring above
    and [[sparse-evidence-gnn-channel-encoder-dominates]] in project
    memory): with the old pooling, the channel-embed block was the only one
    of the two message-MLP input blocks that could vary meaningfully trial-
    to-trial in a way the classifier could exploit per event, since it alone
    carried a real, un-pooled-away signal shape. forward() now gathers this
    encoder's PER-TIMESTEP output at each event's own (approximate) raw
    sample index instead (see forward()'s comment there), so two events on
    the same channel at different points in the trial get different
    embeddings. Not yet re-validated against the 4-subject canonical suite
    quoted in the module docstring above -- that number predates this
    change."""

    def __init__(self, embed_dim: int, dilation: int = 1):
        super().__init__()
        kernel_size = 9
        pad = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=kernel_size, padding=pad, dilation=dilation), nn.GELU(),
            nn.Conv1d(8, embed_dim, kernel_size=kernel_size, padding=pad, dilation=dilation), nn.GELU(),
        )

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        """Returns [batch, n_channels, embed_dim, n_time] -- one embedding
        PER RAW SAMPLE, not one pooled over the whole trial (see class
        docstring's 2026-08-09 note). Callers that want a per-event
        embedding should index the last (time) axis at that event's own
        sample index rather than averaging it away."""
        batch_size, n_channels, n_time = raw_x.shape
        x = raw_x.reshape(batch_size * n_channels, 1, n_time)
        feat = self.net(x)  # [batch*n_channels, embed_dim, n_time]
        return feat.reshape(batch_size, n_channels, -1, n_time)
```

## Event-local gather in `SparseEvidenceGNNCore.forward()` (was lines ~904-947)

```python
    def forward(self, raw_x, events_padded, src_padded, dst_padded, valid_mask):
        """Trainable-only forward pass over PRECOMPUTED sparse events (see
        compute_events()). `to_float_tensors` upstream casts everything to
        float for DataLoader/TensorDataset batching, so src_padded/dst_padded/
        valid_mask arrive as float and need casting back here."""
        src_padded = src_padded.long()
        dst_padded = dst_padded.long()
        valid_mask = valid_mask.bool()
        batch_size, _, n_time = raw_x.shape

        channel_feat = self.channel_encoder(raw_x)  # [B, n_channels, embed_dim, n_time]
        max_count = events_padded.shape[1]
        batch_idx = torch.arange(batch_size, device=raw_x.device).unsqueeze(1).expand(-1, max_count)

        # Event-local channel embedding: gather the encoder's per-timestep
        # output at (an approximation of) each event's own raw-sample time
        # index, instead of one embedding pooled over the whole trial and
        # broadcast onto every event regardless of when it fired -- see
        # ChannelSignalEncoder's 2026-08-09 docstring note.
        #
        # events_padded[..., 0] is timestamp_vals = mean_time / (T_out - 1)
        # (see _build_sparse_events), a fraction of the POST-SMOOTHING
        # coherence timeline. T_out is at most (smooth_kernel_size[0] - 1)
        # samples shorter than raw n_time (valid-padding smoothing conv --
        # common.make_gaussian_weight2d is called with pad_h=0 in
        # compute_events -- see _coi_valid_mask's time_offset for the exact
        # center-alignment this induces), so mapping this fraction onto raw
        # n_time directly is off by at most a few samples out of ~1000 --
        # negligible next to the encoder's own ~81-sample receptive field
        # (channel_encoder_dilation=5 in run_wct_gnn.py), and it avoids
        # threading a 6th tensor through the whole TensorDataset/fit-loop
        # plumbing (_prepare_features/_build_model_from_features/
        # to_float_tensors) just to carry an exact raw-sample index.
        timestamp_frac = events_padded[..., 0]
        time_idx = (timestamp_frac * (n_time - 1)).round().long().clamp(0, n_time - 1)

        # Combined advanced indexing over (batch, channel, time) with a
        # plain slice on the embed_dim axis -- PyTorch/numpy move the
        # (non-adjacent) advanced-indexed dims to the front, giving
        # [B, max_count, embed_dim] directly without ever materializing a
        # [B, max_count, embed_dim, n_time] intermediate (which, at this
        # pipeline's event counts, would be needlessly large).
        src_emb = channel_feat[batch_idx, src_padded, :, time_idx]
        dst_emb = channel_feat[batch_idx, dst_padded, :, time_idx]

        # ... (feature_ablation branch + rest of forward() unchanged either way)
```

Note: this indexing itself was verified correct against a slow-loop
reference during the 2026-08-09 investigation (not the bug, if there is
one) -- the suspected issue is the gradient-sparsity mechanism described
above, not a shape/gather error.
