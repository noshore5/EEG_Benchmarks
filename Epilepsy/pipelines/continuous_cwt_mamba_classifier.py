"""ContinuousCWTMambaClassifier: the continuous-cwt-mamba paradigm's own
sklearn-style classifier -- keeps `_DenseEdgeMambaContinuous`'s SSM state
running across an entire recording (Mamba state reset only between
recordings, never between a recording's own classification windows),
reading `Epilepsy.pipelines.continuous_dense_edge`'s chunked whole-
recording CWT/dense-edge stream and `paradigms.continuous_labeling.
ContinuousLabelingParadigm.get_continuous_data`'s recording-preserving
windows.

Cannot subclass `SparseEvidenceGNNClassifier`/`StreamingSparseEvidenceGNN
Classifier`'s own `fit()`: both shuffle window indices at random per batch
(`_BatchIndexSampler`, `cwt_gnn_classifiers.py`), fundamentally
incompatible with state-carrying, which needs a recording's own windows
processed strictly in chronological order. This class DOES subclass
`SparseEvidenceGNNClassifier` for its constructor/`_build_model` scaffold
(every CWT/coherence/graph hyperparameter is identical -- only
`dense_edge_conv` and the training/inference loop differ), and reuses the
model's existing readout tail (`_dense_edge_features_from_conv_out`,
`sparse_message_mlp`, `_aggregate_events`/`_propagate_hops`,
`sparse_classifier`) completely unmodified via `_continuous_logits`.

v1 scope, matching the plan this was built from: `event_mode="dense"` only,
no `cwt_encoder`, no `time_frequency_node_ablation`, no `channel_subset_k`
(continuous_dense_edge's chunking doesn't support the live-edge-subset
scatter path), no `mamba_use_cuda_kernel` (the fused kernel has no
initial-state API, see `_DenseEdgeMambaContinuous`'s docstring) -- __init__
raises clearly on any of these rather than silently ignoring them.

Batching: one recording at a time (matches CONTEXT.md's own row-batching
finding -- stacking multiple recordings' rows is "not the fix" for
throughput once `scan="chunk"` is in use, and reintroduces the rows*T OOM
risk `_DenseEdgeMambaTemporal`'s `mamba_chunk_size` exists to bound).
`t_chunk` (target output steps per streamed chunk) is a memory/TBPTT knob,
not a training-dynamics one -- see `scripts/continuous_cwt_scale_probe.py`
for measured memory/throughput at real (23ch) scale on an RTX 3070 Ti;
T_chunk=128 measured ~514ms/chunk at ~4GB peak (safe headroom on an 8GB
card), T_chunk=512 measured ~8.75GB (unsafe), T_chunk=256 measured an
anomalous ~20x slowdown (not investigated further, just avoid it).

One optimizer step per recording (loss summed over that recording's own
windows) -- simplest correct choice given each recording needs its own
independent cache-reset boundary; gradient accumulation across several
recordings before stepping is a documented possible future knob, not
implemented here.

validation_split applies at the RECORDING level (a fraction of whole
recordings held out), not the window level `SparseEvidenceGNNClassifier`
uses -- windows within one recording can't be split without breaking the
state-carrying chain that IS the paradigm.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from Epilepsy.pipelines.common import apply_global_zscore, resolve_torch_device, set_seed
from Epilepsy.pipelines.continuous_dense_edge import iter_continuous_dense_edge_chunks
from Epilepsy.pipelines.cwt_gnn_classifiers import (
    SparseEvidenceGNNClassifier,
    _DenseEdgeMambaContinuous,
    pool_continuous_edge_stream_to_windows,
)


def _fit_global_zscore_stats_recordings(recordings: list[dict]) -> tuple[float, float]:
    """`fit_global_zscore_stats`-equivalent for a list of variable-length
    recordings instead of one fixed-shape array -- accumulates sum/sumsq/
    count across recordings (population mean/std over every raw sample of
    every recording combined, NOT an average of per-recording means, which
    would silently misweight shorter recordings)."""
    total = 0
    s = 0.0
    ss = 0.0
    for rec in recordings:
        x = np.asarray(rec["raw_x"], dtype=np.float64)
        total += x.size
        s += float(x.sum())
        ss += float((x * x).sum())
    if total == 0:
        return 0.0, 1.0
    mean = s / total
    var = max(ss / total - mean * mean, 0.0)
    std = float(np.sqrt(var))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return float(mean), std


def _sample_to_output_index(sample: int, chunk_meta: list[tuple[int, int, int, int]]) -> int:
    """Maps a raw sample position to its approximate index in the
    concatenated continuous_out stream a recording's chunks build --
    `chunk_meta` is `[(raw_start, raw_end, output_start_idx, downsample),
    ...]` in chronological order, one entry per streamed chunk (see
    `iter_continuous_dense_edge_chunks`'s `start`/`end` return values --
    the kept output for a chunk corresponds to raw `[start, end)` to
    within `_time_offset_samples()` samples, negligible next to a
    `downsample`-sized output bucket -- see continuous_dense_edge.py's
    module docstring)."""
    for raw_start, raw_end, output_start_idx, downsample in chunk_meta:
        if raw_start <= sample < raw_end:
            return output_start_idx + (sample - raw_start) // downsample
    # At or past the last chunk's own raw_end (e.g. a window's end_sample
    # lands exactly on the recording's final chunk boundary) -- clamp to
    # that chunk's own last output step rather than raising.
    raw_start, raw_end, output_start_idx, downsample = chunk_meta[-1]
    n_steps = max(1, (raw_end - raw_start) // downsample)
    return output_start_idx + n_steps - 1


class ContinuousCWTMambaClassifier(SparseEvidenceGNNClassifier):
    """See module docstring."""

    model_label = "Continuous-CWT-Mamba"
    aux_metric_name = "bursts_per_row"

    def __init__(
        self,
        *,
        mamba_scan: str = "chunk",
        t_chunk: int = 128,
        early_stopping_patience: int | None = 5,
        **kwargs,
    ) -> None:
        if kwargs.get("event_mode", "dense") != "dense":
            raise NotImplementedError(
                "ContinuousCWTMambaClassifier v1 only supports event_mode='dense'."
            )
        kwargs.setdefault("event_mode", "dense")
        kwargs.setdefault("dense_edge_temporal_mode", "mamba")  # discarded, see _build_model
        if kwargs.get("cwt_encoder", False):
            raise NotImplementedError(
                "ContinuousCWTMambaClassifier v1 does not support cwt_encoder=True -- "
                "the node-embedding path needs per-window w_real/w_imag, which the "
                "continuous streaming path doesn't produce in that shape."
            )
        if kwargs.get("time_frequency_node_ablation", "none") != "none":
            raise NotImplementedError(
                "ContinuousCWTMambaClassifier v1 does not support "
                "time_frequency_node_ablation (requires cwt_encoder machinery)."
            )
        if kwargs.get("channel_subset_k") is not None:
            raise NotImplementedError(
                "ContinuousCWTMambaClassifier v1 does not support channel_subset_k -- "
                "continuous_dense_edge's chunked CWT has no live-edge-subset scatter "
                "path (see that module's docstring). Use a static channel_subset "
                "(a fixed channel list) instead if you want fewer channels."
            )
        if kwargs.get("mamba_use_cuda_kernel") is not None:
            raise NotImplementedError(
                "ContinuousCWTMambaClassifier: the fused mamba-ssm CUDA kernel has no "
                "initial-state API and can't be used for a carried-state scan -- see "
                "_DenseEdgeMambaContinuous's docstring. Always runs the mambapy pscan."
            )
        super().__init__(early_stopping_patience=early_stopping_patience, **kwargs)
        if mamba_scan not in ("chunk", "step"):
            raise ValueError(f"mamba_scan must be 'chunk' or 'step', got {mamba_scan!r}.")
        self.mamba_scan = mamba_scan
        self.t_chunk = int(t_chunk)

    def _build_model(self, n_channels: int, n_classes: int, **kwargs):
        core = super()._build_model(n_channels, n_classes, **kwargs)
        # Replace the windowed _DenseEdgeMambaTemporal dense_edge_conv
        # (built above just to get every OTHER submodule constructed --
        # dense_edge_conv is built LAST in SparseEvidenceGNNCore.__init__,
        # see that constructor's own comment, so nothing else's init/RNG
        # order depends on which backend ends up here) with the carried-
        # state continuous backend. Same in_channels/out_channels contract
        # every dense_edge_conv backend shares.
        core.dense_edge_conv = _DenseEdgeMambaContinuous(
            in_channels=4 * self.nfreqs,
            out_channels=self.dense_conv_out_channels,
            d_model=self.mamba_d_model,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand=self.mamba_expand,
            n_layers=self.mamba_n_layers,
            dropout=self.mamba_dropout,
            scan=self.mamba_scan,
        )
        return core

    def _stream_recording(
        self, core, recording: dict, mean: float, std: float
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        """Streams one recording's whole-signal chunks through
        `core.dense_edge_conv`, threading `cache` across calls (reset only
        at the start of THIS recording -- never mid-recording). Returns
        `(continuous_out [1, C, E, T_total], chunk_meta)` -- see
        `_sample_to_output_index`'s docstring for `chunk_meta`."""
        raw_x = recording["raw_x"]
        downsample = max(1, int(core.dense_edge_time_downsample))
        cache = None
        pieces: list[torch.Tensor] = []
        chunk_meta: list[tuple[int, int, int, int]] = []
        cum = 0
        for start, end, conv_in in iter_continuous_dense_edge_chunks(
            self, core, raw_x, self.t_chunk, mean=mean, std=std
        ):
            out_seq, cache = core.dense_edge_conv(conv_in.to(self.device_), cache)
            pieces.append(out_seq)
            chunk_meta.append((start, end, cum, downsample))
            cum += int(out_seq.shape[-1])
        continuous_out = torch.cat(pieces, dim=-1)  # [1, C, E, T_total]
        return continuous_out, chunk_meta

    def _train_recording_incremental(
        self, core, optimizer, loss_fn, recording: dict, mean: float, std: float,
        pool: str = "last",
    ) -> tuple[float, int]:
        """TRAINING counterpart to `_stream_recording` -- NOT just a
        no_grad-vs-grad variant, a genuinely different streaming strategy,
        because it needs one. `_stream_recording` forwards every chunk of
        a recording before concatenating and pooling ONCE at the end; under
        `torch.no_grad()` (validation, `predict_proba`) that's fine (no
        autograd graph is ever built). Under training it measured a real
        CUDA OOM ("13.60 GiB is allocated by PyTorch" reported on an 8GB
        card during a --smoke run): forwarding a whole recording before
        any `backward()` keeps EVERY chunk's own live computation graph
        alive simultaneously -- exactly what `_DenseEdgeMambaContinuous`'s
        own `cache.detach()` TBPTT truncation exists to avoid, just not
        applied at this training-loop level.

        Fix: process windows as soon as the chunks covering them have
        streamed, `backward()` and `optimizer.step()` for that group
        immediately, then drop (let Python free) any buffered chunk
        output no longer needed by a still-pending window. Peak memory
        this way is bounded by the LONGEST window's span in chunks, not
        the whole recording -- a real recording may be hundreds of chunks
        long; a classification window is typically a handful.

        This means potentially SEVERAL optimizer.step() calls per
        recording (one per group of windows that becomes ready together),
        not the one-step-per-recording the module docstring's original
        design assumed before this OOM was found and fixed -- an
        `optimizer_step_batch_*`-style credit/flush scheme
        (`SparseEvidenceGNNClassifier` has one) is a possible future
        refinement, not implemented here; each ready group gets its own
        immediate step, whatever size iter_continuous_dense_edge_chunks'
        chunking happens to produce.

        Returns (summed loss over every window in this recording weighted
        by window count, n_windows) for the caller's epoch-loss logging.
        """
        raw_x = recording["raw_x"]
        windows = recording["windows"]
        downsample = max(1, int(core.dense_edge_time_downsample))
        cache = None
        buffer: list[tuple[int, int, torch.Tensor]] = []  # (start_idx, end_idx, out_seq) -- NOT detached
        chunk_meta: list[tuple[int, int, int, int]] = []
        cum = 0
        next_w = 0
        total_loss = 0.0
        total_n = 0

        for start, end, conv_in in iter_continuous_dense_edge_chunks(
            self, core, raw_x, self.t_chunk, mean=mean, std=std
        ):
            out_seq, cache = core.dense_edge_conv(conv_in.to(self.device_), cache)
            out_len = int(out_seq.shape[-1])
            buffer.append((cum, cum + out_len, out_seq))
            chunk_meta.append((start, end, cum, downsample))
            cum += out_len

            ready = []
            while next_w < len(windows) and windows[next_w]["end_sample"] <= end:
                ready.append(windows[next_w])
                next_w += 1

            if ready:
                bounds = []
                for w in ready:
                    s_idx = _sample_to_output_index(w["start_sample"], chunk_meta)
                    e_idx = _sample_to_output_index(
                        max(w["start_sample"], w["end_sample"] - 1), chunk_meta
                    ) + 1
                    e_idx = max(e_idx, s_idx + 1)
                    e_idx = min(e_idx, cum)
                    s_idx = min(s_idx, e_idx - 1)
                    bounds.append((s_idx, e_idx))
                buf_start = min(b[0] for b in bounds)
                local_pieces = []
                for s, e, t in buffer:
                    if e <= buf_start:
                        continue
                    if s < buf_start:
                        t = t[..., buf_start - s :]
                        s = buf_start
                    local_pieces.append(t)
                local_out = torch.cat(local_pieces, dim=-1)
                local_bounds = [(s - buf_start, e - buf_start) for s, e in bounds]
                pooled = pool_continuous_edge_stream_to_windows(local_out, local_bounds, pool=pool)
                conv_out = pooled.squeeze(1)  # [n, C, E, 1]

                logits = self._continuous_logits(core, conv_out)
                y = torch.tensor(
                    [w["label"] for w in ready], dtype=torch.long, device=logits.device
                )
                loss = loss_fn(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(core.parameters(), self.grad_clip_norm)
                optimizer.step()
                total_loss += float(loss.item()) * len(ready)
                total_n += len(ready)

            # Drop buffered chunk output no longer needed by any still-
            # pending window -- the actual memory fix. `cache` (already
            # detached by _DenseEdgeMambaContinuous.forward) is untouched
            # by this and keeps carrying forward regardless.
            keep_from = (
                _sample_to_output_index(windows[next_w]["start_sample"], chunk_meta)
                if next_w < len(windows) else cum
            )
            buffer = [(s, e, t) for s, e, t in buffer if e > keep_from]

        if next_w < len(windows):
            raise RuntimeError(
                f"{len(windows) - next_w} window(s) in this recording were never covered by "
                "any streamed chunk -- a bug in chunk/window bounds bookkeeping (every window's "
                "end_sample should be <= the recording's own raw sample count, which "
                "get_continuous_data already guarantees), not expected input."
            )
        return total_loss, total_n

    def _continuous_logits(self, core, conv_out: torch.Tensor) -> torch.Tensor:
        """conv_out ([n_windows, dense_conv_out_channels, E, 1], one
        recording's pooled windows stacked in the batch dim) -> class
        logits, via the SAME readout tail SparseEvidenceGNNCore.forward
        uses after its own _dense_edge_features call -- see
        _dense_edge_features_from_conv_out's docstring for why this split
        exists. Only the event_mode="dense", cwt_encoder=False,
        feature_ablation!="zero_event_features" path (this class's own
        __init__ already rejects the unsupported configs)."""
        events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask, batch_idx = (
            core._dense_edge_features_from_conv_out(conv_out)
        )
        if core.feature_ablation == "zero_event_features":
            events_padded = torch.zeros_like(events_padded)
        full_features = events_padded
        msg = core.sparse_message_mlp(full_features)
        evidence = core._aggregate_events(
            msg, full_features, dst_padded, valid_mask, batch_idx, conv_out.dtype
        )
        if core.n_hops > 1:
            if core.freq_aware_hops:
                evidence = core._propagate_hops_freq_aware(
                    msg, full_features, dst_padded, freq_idx_padded, valid_mask,
                    batch_idx, conv_out.dtype,
                )
            else:
                evidence = core._propagate_hops(evidence)
        batch_size_actual = conv_out.shape[0]
        if core.event_aggregation == "concat":
            readout = evidence.reshape(
                batch_size_actual, core.n_channels * core.concat_max_degree * core.hidden_dim
            )
        else:
            readout = evidence.reshape(batch_size_actual, core.n_channels * core.hidden_dim)
        return core.sparse_classifier(readout)

    def _recording_logits_and_labels(
        self, core, recording: dict, mean: float, std: float, pool: str = "last"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recording, streamed once, pooled at every one of its
        windows -> (logits [n_windows, n_classes], labels [n_windows])."""
        continuous_out, chunk_meta = self._stream_recording(core, recording, mean, std)
        windows = recording["windows"]
        bounds = []
        labels = []
        for w in windows:
            start_idx = _sample_to_output_index(w["start_sample"], chunk_meta)
            end_idx = _sample_to_output_index(max(w["start_sample"], w["end_sample"] - 1), chunk_meta) + 1
            end_idx = max(end_idx, start_idx + 1)
            end_idx = min(end_idx, continuous_out.shape[-1])
            start_idx = min(start_idx, end_idx - 1)
            bounds.append((start_idx, end_idx))
            labels.append(w["label"])
        pooled = pool_continuous_edge_stream_to_windows(continuous_out, bounds, pool=pool)
        # [n_windows, 1, C, E, 1] (B=1 always here) -> [n_windows, C, E, 1]
        conv_out = pooled.squeeze(1)
        logits = self._continuous_logits(core, conv_out)
        y = torch.tensor(labels, dtype=torch.long, device=logits.device)
        return logits, y

    def fit(self, recordings: list[dict], validation_split: float | None = None, seed: int | None = None):
        """recordings: `ContinuousLabelingParadigm.get_continuous_data()`'s
        return value (a whole subject/fold's worth) -- NOT `(X, y)`
        windowed arrays, unlike every other classifier in this file. Split
        happens at the RECORDING level (see module docstring)."""
        set_seed(self.seed if seed is None else seed)
        self.device_ = resolve_torch_device(self.device)
        self._resolve_transform_fns()

        if not recordings:
            raise ValueError("fit() received no recordings.")
        n_channels = recordings[0]["raw_x"].shape[0]
        self.classes_ = np.array([0, 1])
        self.class_to_idx_ = {0: 0, 1: 1}

        if self.normalize_input:
            self.X_mean_, self.X_std_ = _fit_global_zscore_stats_recordings(recordings)
        else:
            self.X_mean_, self.X_std_ = 0.0, 1.0

        vs = self.validation_split if validation_split is None else validation_split
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(recordings))
        if vs and 0.0 < vs < 1.0 and len(recordings) > 1:
            n_val = max(1, int(round(vs * len(recordings))))
            val_idx = set(order[:n_val].tolist())
        else:
            val_idx = set()
        train_recordings = [r for i, r in enumerate(recordings) if i not in val_idx]
        val_recordings = [r for i, r in enumerate(recordings) if i in val_idx]
        self._vprint(
            1,
            f"[Train] {len(train_recordings)} training recordings, "
            f"{len(val_recordings)} validation recordings.",
        )

        self.model_ = self._build_model(n_channels=n_channels, n_classes=2).to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_since_improve = 0
        for epoch in range(self.epochs):
            self.model_.train()
            train_order = rng.permutation(len(train_recordings))
            epoch_loss, epoch_n = 0.0, 0
            for i in train_order:
                rec = train_recordings[int(i)]
                # _train_recording_incremental, NOT _recording_logits_and_
                # labels + one backward() -- see that method's own
                # docstring for the real CUDA OOM this avoids (a whole-
                # recording-then-one-backward design keeps every chunk's
                # live graph alive at once).
                rec_loss, rec_n = self._train_recording_incremental(
                    self.model_, optimizer, loss_fn, rec, self.X_mean_, self.X_std_
                )
                epoch_loss += rec_loss
                epoch_n += rec_n
            train_loss = epoch_loss / max(1, epoch_n)

            val_loss = None
            if val_recordings:
                self.model_.eval()
                v_loss, v_n = 0.0, 0
                with torch.no_grad():
                    for rec in val_recordings:
                        logits, y = self._recording_logits_and_labels(
                            self.model_, rec, self.X_mean_, self.X_std_
                        )
                        v_loss += float(loss_fn(logits, y).item()) * y.numel()
                        v_n += y.numel()
                val_loss = v_loss / max(1, v_n)

            msg = f"[Train][Epoch {epoch + 1}/{self.epochs}] loss={train_loss:.6f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.6f}"
            self._vprint(1, msg)

            if val_loss is not None and self.early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.early_stopping_patience:
                        self._vprint(1, f"[Train] early stopping at epoch {epoch + 1}.")
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, recordings: list[dict]) -> list[np.ndarray]:
        """Returns one `[n_windows, 2]` softmax-probability array PER
        recording (a plain list, not a single stacked array -- recordings
        may have different window counts) -- caller (the LOSO loop) knows
        which recording is which via the SAME `recordings` list order."""
        self.model_.eval()
        out = []
        with torch.no_grad():
            for rec in recordings:
                logits, _y = self._recording_logits_and_labels(
                    self.model_, rec, self.X_mean_, self.X_std_
                )
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return out
