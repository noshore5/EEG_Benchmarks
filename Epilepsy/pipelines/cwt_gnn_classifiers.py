"""CWT-based GNN classifiers.

Two families, merged into this one file 2026-08-16 (previously split across
xwt_phase_gnn_classifier.py and sparse_evidence_gnn_classifier.py) at the
user's request:

  - XWTPhaseGNNCore / XWTPhaseGNNClassifier / XWTPhaseGNNV2Core /
    XWTPhaseGNNV2Classifier -- level-0 XWT phase-conditioned message passing.
    Epilepsy/run_pipelines.py does not instantiate any of these; kept here
    (unpruned) for parity with BCI/moabb_pipelines' equivalent file and in
    case a future Epilepsy pipeline wants them.
  - _BaseCWTGNNClassifier -- shared sklearn/MOABB fit/predict/noise-
    augmentation scaffolding both families build on.
  - SparseEvidenceGNNCore / SparseEvidenceGNNClassifier (plus the
    surrogate-null-cache helpers below) -- full-resolution coherence ->
    events/dense-edge-features -> GNN message passing -> classify, with
    event_mode in {"sparse", "dense", "temporal_graph"}.
    Epilepsy/run_pipelines.py's DENSE_EDGE_GRU_PARAMS drives this class in
    exactly ONE of those configs (event_mode="dense",
    dense_edge_temporal_mode="rnn" -- informally "dense edge GRU"), but the
    class itself is broader than that name, hence this file isn't named
    after that one config either.

Straight merge, not a prune: no event_mode/class was removed or renamed.
SparseEvidenceGNNClassifier's own (much more detailed) original module
docstring follows, unchanged, as a comment block just above the surrogate-
null-cache section it introduces, so nothing written there is lost.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

try:
    from Epilepsy.pipelines.common import (
        TorchEEGClassifier,
        apply_global_zscore,
        augment_paired_cwt_batch,
        compute_paired_cwt_noise_bank,
        compute_cwt_real_imag_tensors,
        emit_initial_detail,
        fit_global_zscore_stats,
        is_experiment_logging_configured,
        make_gaussian_weight2d,
        ordered_pair_indices as _ordered_pair_indices,
        phase_rule_deadzone_sign as _phase_rule_deadzone_sign,
        print_torch_custom_model_summary,
        print_torch_parameter_hashes,
        print_torch_parameter_summary,
        resolve_coherence_utils,
        resolve_phase_rule as _resolve_phase_rule,
        resolve_torch_device,
        resolve_train_val_indices,
        set_seed,
        upper_pair_indices,
        validate_eeg_X,
        validation_groups_from_metadata,
    )
    from Epilepsy.pipelines.common import _count_eligible_tensor_batches, _min_accepted_batch_size
    from Epilepsy.pipelines.cwt_window_cache import (
        DISABLE_CWT_CACHE,
        compute_cwt_real_imag_tensors_cached,
        precompute_window_cache_keys,
    )
    from Epilepsy.pipelines.dense_edge_cache import (
        dense_edge_cache_key,
        load_dense_edge,
        save_dense_edge,
    )
except ModuleNotFoundError:
    from pipelines.common import (
        TorchEEGClassifier,
        apply_global_zscore,
        augment_paired_cwt_batch,
        compute_paired_cwt_noise_bank,
        compute_cwt_real_imag_tensors,
        emit_initial_detail,
        fit_global_zscore_stats,
        is_experiment_logging_configured,
        make_gaussian_weight2d,
        ordered_pair_indices as _ordered_pair_indices,
        phase_rule_deadzone_sign as _phase_rule_deadzone_sign,
        print_torch_custom_model_summary,
        print_torch_parameter_hashes,
        print_torch_parameter_summary,
        resolve_coherence_utils,
        resolve_phase_rule as _resolve_phase_rule,
        resolve_torch_device,
        resolve_train_val_indices,
        set_seed,
        upper_pair_indices,
        validate_eeg_X,
        validation_groups_from_metadata,
    )
    from pipelines.common import _count_eligible_tensor_batches, _min_accepted_batch_size
    from pipelines.cwt_window_cache import (
        DISABLE_CWT_CACHE,
        compute_cwt_real_imag_tensors_cached,
        precompute_window_cache_keys,
    )
    from pipelines.dense_edge_cache import (
        dense_edge_cache_key,
        load_dense_edge,
        save_dense_edge,
    )


# ============================================================================
# Originally xwt_phase_gnn_classifier.py ("Level-0 XWT phase-conditioned GNN
# classifiers.") -- XWTPhaseGNNCore, _BaseCWTGNNClassifier,
# XWTPhaseGNNClassifier, XWTPhaseGNNV2Core, XWTPhaseGNNV2Classifier.
# ============================================================================


def _resolve_torch_cwt():
    """Imports utils/torch_cwt.py (torch.fft-native CWT, drop-in-signature
    replacement for coherence_utils.transform -- see that module's own
    docstring) the same way resolve_coherence_utils (common.py) resolves
    utils/coherence_utils.py: repo_root/utils sits alongside
    Coherent_Multiplex, so getting repo_root onto sys.path once is enough.
    No fallback-root search (unlike resolve_coherence_utils) -- torch_cwt.py
    is this repo's own module, not a vendored external one, so there's only
    ever one place it can live.
    """
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from utils import torch_cwt  # type: ignore

    return torch_cwt


class XWTPhaseGNNCore(nn.Module):
    """Torch core for level-0 phase-gated XWT message passing."""

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
        hidden_dim: int = 64,
        message_dim: int = 64,
        theta_dead_deg: float = 45.0,
        time_stride: int = 1,
        state_mode: str = "per_node",
        phase_rule: str | Callable[[torch.Tensor, float], torch.Tensor] = "deadzone_sign",
        use_mag: bool = True,
        use_ang: bool = True,
        use_raw: bool = True,
        use_state_src: bool = True,
        use_state_dst: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if time_stride <= 0:
            raise ValueError("time_stride must be >= 1")
        if state_mode not in {"per_node", "per_node_per_freq"}:
            raise ValueError("state_mode must be one of {'per_node', 'per_node_per_freq'}")

        self.n_channels = n_channels
        self.nfreqs = nfreqs
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.theta_dead_rad = math.radians(theta_dead_deg)
        self.time_stride = time_stride
        self.state_mode = state_mode
        self.phase_rule_fn = _resolve_phase_rule(phase_rule)
        self.use_mag = use_mag
        self.use_ang = use_ang
        self.use_raw = use_raw
        self.use_state_src = use_state_src
        self.use_state_dst = use_state_dst

        src_idx, dst_idx = _ordered_pair_indices(n_channels)
        self.register_buffer("src_idx", src_idx, persistent=False)
        self.register_buffer("dst_idx", dst_idx, persistent=False)

        payload_dim = 0
        if self.use_mag:
            payload_dim += 1
        if self.use_ang:
            payload_dim += 1
        if self.use_raw:
            payload_dim += 2
        if self.use_state_src:
            payload_dim += hidden_dim
        if self.use_state_dst:
            payload_dim += hidden_dim
        if payload_dim == 0:
            raise ValueError("At least one payload component must be enabled.")

        self.message_mlp = nn.Sequential(
            nn.Linear(payload_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, hidden_dim),
        )
        self.state_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def _aggregate_per_node(self, msg: torch.Tensor) -> torch.Tensor:
        """Aggregate [B, E, H] messages to [B, C, H] by destination."""
        batch_size, num_edges, hidden_dim = msg.shape
        device = msg.device
        agg = torch.zeros(
            batch_size * self.n_channels,
            hidden_dim,
            device=device,
            dtype=msg.dtype,
        )
        batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * self.n_channels
        dst = (self.dst_idx.unsqueeze(0) + batch_offsets).reshape(-1)
        agg.index_add_(0, dst, msg.reshape(batch_size * num_edges, hidden_dim))
        return agg.view(batch_size, self.n_channels, hidden_dim)

    def _aggregate_per_node_per_freq(self, msg: torch.Tensor) -> torch.Tensor:
        """Aggregate [B, E, F, H] messages to [B, C, F, H] by destination."""
        batch_size, num_edges, nfreqs, hidden_dim = msg.shape
        device = msg.device
        agg = torch.zeros(
            batch_size * self.n_channels * nfreqs,
            hidden_dim,
            device=device,
            dtype=msg.dtype,
        )
        base_b = (
            torch.arange(batch_size, device=device).view(batch_size, 1, 1)
            * self.n_channels
            * nfreqs
        )
        base_f = torch.arange(nfreqs, device=device).view(1, 1, nfreqs)
        dst = self.dst_idx.view(1, num_edges, 1) * nfreqs + base_f + base_b
        agg.index_add_(
            0,
            dst.reshape(-1),
            msg.reshape(batch_size * num_edges * nfreqs, hidden_dim),
        )
        return agg.view(batch_size, self.n_channels, nfreqs, hidden_dim)

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}.")

        device = raw_x.device
        num_edges = self.src_idx.numel()
        if self.state_mode == "per_node":
            state = torch.zeros(batch_size, self.n_channels, self.hidden_dim, device=device)
        else:
            state = torch.zeros(
                batch_size,
                self.n_channels,
                self.nfreqs,
                self.hidden_dim,
                device=device,
            )

        gate_sum = 0.0
        gate_count = 0.0
        for t in range(0, n_time, self.time_stride):
            src_r = w_real[:, self.src_idx, t, :]
            src_i = w_imag[:, self.src_idx, t, :]
            dst_r = w_real[:, self.dst_idx, t, :]
            dst_i = w_imag[:, self.dst_idx, t, :]

            xwt_real = src_r * dst_r + src_i * dst_i
            xwt_imag = src_i * dst_r - src_r * dst_i
            mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
            ang = torch.atan2(xwt_imag, xwt_real)
            delta = torch.atan2(torch.sin(ang), torch.cos(ang))
            gate = self.phase_rule_fn(delta, self.theta_dead_rad)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)

            gate_sum += float(gate.sum().item())
            gate_count += float(gate.numel())
            mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
            ang = torch.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)

            features = []
            if self.use_mag:
                features.append(mag.unsqueeze(-1))
            if self.use_ang:
                features.append(ang.unsqueeze(-1))
            if self.use_raw:
                raw_t = raw_x[:, :, t]
                src_raw = raw_t[:, self.src_idx].unsqueeze(-1).unsqueeze(-1)
                dst_raw = raw_t[:, self.dst_idx].unsqueeze(-1).unsqueeze(-1)
                features.append(src_raw.expand(batch_size, num_edges, self.nfreqs, 1))
                features.append(dst_raw.expand(batch_size, num_edges, self.nfreqs, 1))

            if self.state_mode == "per_node":
                if self.use_state_src:
                    src_state = state[:, self.src_idx, :].unsqueeze(2)
                    features.append(
                        src_state.expand(batch_size, num_edges, self.nfreqs, self.hidden_dim)
                    )
                if self.use_state_dst:
                    dst_state = state[:, self.dst_idx, :].unsqueeze(2)
                    features.append(
                        dst_state.expand(batch_size, num_edges, self.nfreqs, self.hidden_dim)
                    )
            else:
                if self.use_state_src:
                    features.append(state[:, self.src_idx, :, :])
                if self.use_state_dst:
                    features.append(state[:, self.dst_idx, :, :])

            msg = self.message_mlp(torch.cat(features, dim=-1))
            msg = msg * gate.unsqueeze(-1)

            if self.state_mode == "per_node":
                agg = self._aggregate_per_node(msg.sum(dim=2))
                state = self.state_cell(
                    agg.reshape(batch_size * self.n_channels, self.hidden_dim),
                    state.reshape(batch_size * self.n_channels, self.hidden_dim),
                ).view(batch_size, self.n_channels, self.hidden_dim)
            else:
                agg = self._aggregate_per_node_per_freq(msg)
                state = self.state_cell(
                    agg.reshape(batch_size * self.n_channels * self.nfreqs, self.hidden_dim),
                    state.reshape(batch_size * self.n_channels * self.nfreqs, self.hidden_dim),
                ).view(batch_size, self.n_channels, self.nfreqs, self.hidden_dim)

        pooled = state.mean(dim=1) if self.state_mode == "per_node" else state.mean(dim=(1, 2))
        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return self.classifier(pooled), edge_density


class _BaseCWTGNNClassifier(TorchEEGClassifier):
    """Shared sklearn wrapper logic for XWT/WCT CWT-tensor GNNs."""

    _estimator_type = "classifier"
    model_label = "CWT-GNN"

    # 2026-08-22: when True, _prepare_features keeps CWT (and, via
    # SparseEvidenceGNNClassifier._precompute_dense_edge_inputs, dense-edge)
    # output resident on self.device_ instead of the CPU numpy round trip
    # every other subclass uses -- see StreamingSparseEvidenceGNNClassifier's
    # override docstring for why only it sets this. False here so every
    # eager (whole-training-set-at-once) subclass is completely unaffected.
    _keep_features_on_device = False

    def _init_cwt_gnn_classifier(
        self,
        *,
        sampling_rate: int,
        lowest: float,
        highest: float,
        nfreqs: int,
        cwt_resample_n_time: int | None,
        normalize_input: bool,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        grad_clip_norm: float | None,
        noise_augmentation_enabled: bool = False,
        noise_apply_prob: float = 0.0,
        noise_strength: float = 0.0,
        noise_bank_size: int = 128,
        noise_bank_seed: int | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        last_batch_min_ratio: float = 0.0,
        selector_alpha_val_update_rate: float = 1.0,
        optimizer_step_batch_size: int | None = None,
        optimizer_step_batch_mode: str = "credit",
        optimizer_step_remainder_policy: str = "flush",
        # NEW: restrict which channels are used, by integer index (or by
        # name if self.channel_names_ has been populated before fit()).
        channel_subset: list[int] | list[str] | None = None,
        # Epilepsy fork only: upstream (BCI/moabb_pipelines) hardcodes this
        # to False when calling _init_torch_classifier below -- fine for
        # motor imagery's balanced left/right trials, wrong here. CHB-MIT
        # windows are ~2% ictal, so leaving this False trains a classifier
        # that can minimize loss by always predicting interictal.
        # TorchEEGClassifier._criterion (common.py) already implements
        # inverse-class-frequency weighting; this just stops discarding it.
        # Default True since that's this fork's whole reason to exist.
        use_class_weights: bool = True,
        # Epilepsy fork only -- see cwt_window_cache.py's docstring. A plain
        # dict the caller owns; pass the SAME dict into multiple classifier
        # instances (e.g. one per leave-one-seizure-out fold) to reuse CWT
        # work for windows they share instead of recomputing per fold.
        # None (default) gives this instance its own private cache, i.e. no
        # behavior change from before this existed. Pass the
        # DISABLE_CWT_CACHE sentinel (cwt_window_cache.py) to skip CWT
        # caching entirely.
        cwt_cache: dict | None = None,
        # Step 6 (torch-native-cwt branch): selects which CWT implementation
        # self.transform_/self.batch_transform_ resolve to in
        # _prepare_features. "fcwt" (default) is the original FFTW/fcwt.cwt()
        # path, unchanged -- exists so the old path stays trivially
        # revertable (per that branch's plan) without a git revert: just
        # don't pass cwt_backend="torch". "torch" resolves to
        # utils/torch_cwt.py (torch.fft, GPU-native, batched -- see
        # _resolve_torch_cwt / torch_cwt.py's own docstring for the
        # correctness validation this rests on).
        cwt_backend: Literal["fcwt", "torch"] = "fcwt",
        # torch backend only: caps how many (sample, channel) windows one
        # batched torch_cwt.transform_batch call covers at once (see
        # compute_cwt_real_imag_tensors_cached's batch_transform_fn param).
        # Bounds peak device memory (roughly linear in this times nfreqs
        # times padded window length -- see the 2026-08-20 session notes'
        # Part 8 for the sizing math) independent of how many windows/
        # channels a given fit() call sees. Unused when cwt_backend="fcwt".
        torch_cwt_batch_size: int = 256,
        verbose: int = 0,
    ) -> None:
        self.cwt_cache = cwt_cache if cwt_cache is not None else {}
        self.sampling_rate = sampling_rate
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
        self.cwt_resample_n_time = cwt_resample_n_time
        self.normalize_input = normalize_input
        self.noise_augmentation_enabled = noise_augmentation_enabled
        self.noise_apply_prob = noise_apply_prob
        self.noise_strength = noise_strength
        self.noise_bank_size = noise_bank_size
        self.noise_bank_seed = noise_bank_seed
        # NEW
        self.channel_subset = channel_subset
        self.channel_names_: list[str] | None = None
        if cwt_backend not in ("fcwt", "torch"):
            raise ValueError(f"cwt_backend must be 'fcwt' or 'torch', got {cwt_backend!r}.")
        self.cwt_backend = cwt_backend
        self.torch_cwt_batch_size = int(torch_cwt_batch_size)
        self.transform_ = None
        self.batch_transform_ = None
        self.X_mean_: float | None = None
        self.X_std_: float | None = None
        self.noise_bank_: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.noise_channel_std_: torch.Tensor | None = None
        self.noise_bank_device_: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.noise_channel_std_device_: torch.Tensor | None = None
        self._validate_noise_augmentation_params()
        self._init_torch_classifier(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            last_batch_min_ratio=last_batch_min_ratio,
            selector_alpha_val_update_rate=selector_alpha_val_update_rate,
            optimizer_step_batch_size=optimizer_step_batch_size,
            optimizer_step_batch_mode=optimizer_step_batch_mode,
            optimizer_step_remainder_policy=optimizer_step_remainder_policy,
            use_class_weights=use_class_weights,
            verbose=verbose,
        )

    # NEW: resolves self.channel_subset (indices or names) into an index
    # list and slices the channel axis of X. Returns X unchanged if no
    # subset was requested.
    def _apply_channel_subset(self, X: np.ndarray) -> np.ndarray:
        if self.channel_subset is None:
            return X
        if len(self.channel_subset) == 0:
            raise ValueError("channel_subset must not be empty.")

        if all(isinstance(c, str) for c in self.channel_subset):
            if self.channel_names_ is None:
                raise ValueError(
                    "channel_subset was given as channel names, but "
                    "self.channel_names_ has not been set. Set "
                    "channel_names_ (e.g. from your MNE info['ch_names']) "
                    "before calling fit(), or pass channel_subset as "
                    "integer indices instead."
                )
            missing = [c for c in self.channel_subset if c not in self.channel_names_]
            if missing:
                raise ValueError(f"channel_subset names not found: {missing}.")
            idx = [self.channel_names_.index(name) for name in self.channel_subset]
        elif all(isinstance(c, (int, np.integer)) for c in self.channel_subset):
            idx = [int(c) for c in self.channel_subset]
        else:
            raise ValueError(
                "channel_subset must be a list of all str (names) or all "
                "int (indices), not a mix."
            )

        n_channels = X.shape[1]
        out_of_range = [i for i in idx if i < 0 or i >= n_channels]
        if out_of_range:
            raise ValueError(
                f"channel_subset indices out of range for {n_channels} "
                f"channels: {out_of_range}."
            )
        return X[:, idx, :]

    def _resolve_transform_fns(self) -> None:
        """Sets self.transform_ (always, single-signal, numpy in/out --
        the interface every call site was originally written against) and
        self.batch_transform_ (only for cwt_backend="torch" -- see
        compute_cwt_real_imag_tensors's batch_transform_fn param for why a
        separate batched entry point exists at all instead of just calling
        transform_ in a loop with device="cuda").

        Runs the transform on self.device_ when it's been resolved already
        (mid/post-fit(); see TorchEEGClassifier.fit, common.py, which sets
        it before the first _prepare_features call) -- CPU otherwise (e.g.
        a bare _prepare_features() probe call before fit(), which fcwt's
        own path also implicitly runs on CPU).
        """
        if self.cwt_backend == "fcwt":
            self.transform_, _ = resolve_coherence_utils()
            self.batch_transform_ = None
            return
        torch_cwt = _resolve_torch_cwt()
        device = getattr(self, "device_", None)
        self.transform_ = functools.partial(torch_cwt.transform, device=device)
        self.batch_transform_ = functools.partial(torch_cwt.transform_batch, device=device)

    def _prepare_features(self, X: np.ndarray, *, fit: bool, train_idx=None, window_keys=None):
        # NEW: slice channels first, before normalization/CWT/anything else
        # touches X, so every downstream step (z-score stats, CWT tensors,
        # noise bank, n_channels inference) only ever sees the subset.
        X = self._apply_channel_subset(X)

        if fit:
            self._validate_noise_augmentation_params()
        if self.normalize_input:
            if fit:
                ref = X if train_idx is None else X[train_idx]
                self.X_mean_, self.X_std_ = fit_global_zscore_stats(ref)
            if self.X_mean_ is None or self.X_std_ is None:
                raise ValueError("Input normalization stats are not initialized.")
            mean, std = self.X_mean_, self.X_std_
        else:
            mean, std = 0.0, 1.0

        if self.transform_ is None:
            self._resolve_transform_fns()
        # Epilepsy fork only: cached in place of compute_cwt_real_imag_tensors
        # -- X here is still RAW (pre-normalization); the cache applies the
        # mean/std rescale on retrieval instead (see cwt_window_cache.py's
        # docstring for why that's exact, not approximate). This is what
        # lets separate CV folds -- each a new classifier instance, each
        # with its own fold-specific mean/std -- reuse CWT work for windows
        # they have in common instead of recomputing the wavelet transform
        # from scratch every fold. self.cwt_cache is a private per-instance
        # {} by default (no cross-instance reuse unless a caller shares one
        # dict across fold instances -- see _init_cwt_gnn_classifier's
        # cwt_cache param); DISABLE_CWT_CACHE -> real None below.
        #
        # keep_on_device (2026-08-22): only StreamingSparseEvidenceGNNClassifier
        # sets _keep_features_on_device, and even there only actually takes
        # effect when the torch backend is active (batch_transform_ is None
        # under cwt_backend="fcwt") and cwt_resample_n_time is unset (the
        # real pipeline's own default) -- both required by
        # _compute_cwt_real_imag_tensors_device_resident, see its docstring.
        # Falls back to the always-correct CPU-numpy path otherwise, same as
        # before this existed.
        keep_on_device = (
            bool(self._keep_features_on_device)
            and self.batch_transform_ is not None
            and self.cwt_resample_n_time is None
            and getattr(self, "device_", None) is not None
        )
        features = compute_cwt_real_imag_tensors_cached(
            X,
            mean=mean,
            std=std,
            sampling_rate=self.sampling_rate,
            highest=self.highest,
            lowest=self.lowest,
            nfreqs=self.nfreqs,
            cwt_resample_n_time=self.cwt_resample_n_time,
            transform_fn=self.transform_,
            verbose=self.verbose,
            # DISABLE_CWT_CACHE -> real None: the sentinel is an identity
            # marker at this class's level (distinguishing "disabled" from
            # the constructor's own cwt_cache=None default, which means
            # "private per-instance dict"); compute_cwt_real_imag_tensors_cached
            # (cwt_window_cache.py) is what actually treats None specially.
            # Not passed at all on the keep_on_device path -- that path
            # never touches the cache (see cwt_window_cache.py's docstring).
            cache=None if keep_on_device else (None if self.cwt_cache is DISABLE_CWT_CACHE else self.cwt_cache),
            window_keys=None if keep_on_device else window_keys,
            batch_transform_fn=self.batch_transform_,
            batch_size=self.torch_cwt_batch_size,
            cwt_backend=self.cwt_backend,
            keep_on_device=keep_on_device,
            device=self.device_ if keep_on_device else None,
        )
        if fit:
            X_normalized = apply_global_zscore(X, mean, std) if self.normalize_input else X
            self._fit_noise_augmentation_state(features, X_normalized, train_idx)
        return features

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> nn.Module:
        raw_x = features[0] if isinstance(features, tuple) else features
        return self._build_model(n_channels=int(raw_x.shape[1]), n_classes=n_classes, **kwargs)

    def _validate_noise_augmentation_params(self) -> None:
        if not 0.0 <= float(self.noise_apply_prob) <= 1.0:
            raise ValueError("noise_apply_prob must be in [0.0, 1.0].")
        if float(self.noise_strength) < 0.0:
            raise ValueError("noise_strength must be >= 0.0.")
        if int(self.noise_bank_size) <= 0:
            raise ValueError("noise_bank_size must be > 0.")

    def _uses_noise_augmentation(self) -> bool:
        return (
            bool(self.noise_augmentation_enabled)
            and float(self.noise_apply_prob) > 0.0
            and float(self.noise_strength) > 0.0
        )

    def _fit_noise_augmentation_state(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        X: np.ndarray,
        train_idx,
    ) -> None:
        self.noise_bank_ = None
        self.noise_channel_std_ = None
        self.noise_bank_device_ = None
        self.noise_channel_std_device_ = None
        if not self._uses_noise_augmentation():
            return

        raw_x = features[0]
        if train_idx is None:
            ref = raw_x
        else:
            ref = raw_x[torch.as_tensor(train_idx, dtype=torch.long)]
        channel_std = torch.std(ref, dim=(0, 2), unbiased=False)
        channel_std = torch.nan_to_num(channel_std, nan=0.0, posinf=0.0, neginf=0.0)
        self.noise_channel_std_ = channel_std.float().contiguous()
        bank_seed = (
            int(self.noise_bank_seed)
            if self.noise_bank_seed is not None
            else int(self.seed or 0) + 10_003
        )
        self.noise_bank_ = compute_paired_cwt_noise_bank(
            bank_size=int(self.noise_bank_size),
            segment_length=int(X.shape[2]),
            sampling_rate=self.sampling_rate,
            highest=self.highest,
            lowest=self.lowest,
            nfreqs=self.nfreqs,
            cwt_resample_n_time=self.cwt_resample_n_time,
            transform_fn=self.transform_,
            seed=bank_seed,
            verbose=self.verbose,
            batch_transform_fn=self.batch_transform_,
            batch_size=self.torch_cwt_batch_size,
        )

    def _prepare_training_state_on_device(self) -> None:
        self.noise_bank_device_ = None
        self.noise_channel_std_device_ = None
        if not self._uses_noise_augmentation():
            return
        if self.device_ is None:
            raise ValueError("Torch device is not initialized.")
        if self.noise_bank_ is None or self.noise_channel_std_ is None:
            raise ValueError("Noise augmentation state is not initialized.")
        self.noise_bank_device_ = tuple(
            tensor.to(device=self.device_, dtype=torch.float32)
            for tensor in self.noise_bank_
        )
        self.noise_channel_std_device_ = self.noise_channel_std_.to(
            device=self.device_,
            dtype=torch.float32,
        )

    def _augment_train_batch_inputs(
        self, batch_inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        if not self._uses_noise_augmentation():
            return batch_inputs
        if self.noise_bank_device_ is None or self.noise_channel_std_device_ is None:
            raise ValueError("Noise augmentation device state is not initialized.")
        return augment_paired_cwt_batch(
            batch_inputs,
            noise_bank=self.noise_bank_device_,
            channel_std=self.noise_channel_std_device_,
            apply_prob=float(self.noise_apply_prob),
            strength=float(self.noise_strength),
        )


class XWTPhaseGNNClassifier(_BaseCWTGNNClassifier):
    """sklearn/MOABB wrapper around the level-0 XWT phase GNN core."""

    model_label = "XWT-V1"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 48,
        cwt_resample_n_time: int | None = None,
        time_stride: int = 1,
        theta_dead_deg: float = 45.0,
        state_mode: str = "per_node",
        phase_rule: str | Callable[[torch.Tensor, float], torch.Tensor] = "deadzone_sign",
        use_mag: bool = True,
        use_ang: bool = True,
        use_raw: bool = True,
        use_state_src: bool = True,
        use_state_dst: bool = True,
        hidden_dim: int = 64,
        message_dim: int = 64,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 0.1,
        normalize_input: bool = True,
        noise_augmentation_enabled: bool = False,
        noise_apply_prob: float = 0.0,
        noise_strength: float = 0.0,
        noise_bank_size: int = 128,
        noise_bank_seed: int | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        # NEW
        channel_subset: list[int] | list[str] | None = None,
        verbose: int = 0,
    ) -> None:
        self.time_stride = time_stride
        self.theta_dead_deg = theta_dead_deg
        self.state_mode = state_mode
        self.phase_rule = phase_rule
        self.use_mag = use_mag
        self.use_ang = use_ang
        self.use_raw = use_raw
        self.use_state_src = use_state_src
        self.use_state_dst = use_state_dst
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self._init_cwt_gnn_classifier(
            sampling_rate=sampling_rate,
            lowest=lowest,
            highest=highest,
            nfreqs=nfreqs,
            cwt_resample_n_time=cwt_resample_n_time,
            normalize_input=normalize_input,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            noise_augmentation_enabled=noise_augmentation_enabled,
            noise_apply_prob=noise_apply_prob,
            noise_strength=noise_strength,
            noise_bank_size=noise_bank_size,
            noise_bank_seed=noise_bank_seed,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            # NEW
            channel_subset=channel_subset,
            verbose=verbose,
        )

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> XWTPhaseGNNCore:
        return XWTPhaseGNNCore(
            n_channels=n_channels,
            nfreqs=self.nfreqs,
            n_classes=n_classes,
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
            theta_dead_deg=self.theta_dead_deg,
            time_stride=self.time_stride,
            state_mode=self.state_mode,
            phase_rule=self.phase_rule,
            use_mag=self.use_mag,
            use_ang=self.use_ang,
            use_raw=self.use_raw,
            use_state_src=self.use_state_src,
            use_state_dst=self.use_state_dst,
            **kwargs,
        )


class XWTPhaseGNNV2Core(nn.Module):
    """V2 core with channel-local temporal encoder and frequency-indexed state."""

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
        message_dim: int = 3,
        hidden_state_dim: int = 32,
        encoder_dim: int = 16,
        use_encoder_batch_norm: bool = True,
        encoder_dropout: float | None = 0.5,
        use_local_residual: bool = True,
        use_prev_state_mean: bool = True,
        gru_input_dropout: float | None = 0.0,
        readout_dropout: float | None = 0.0,
        time_stride: int = 1,
        theta_dead_deg: float = 45.0,
        use_raw_in_message: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if time_stride <= 0:
            raise ValueError("time_stride must be >= 1")
        for name, value in {
            "encoder_dropout": encoder_dropout,
            "gru_input_dropout": gru_input_dropout,
            "readout_dropout": readout_dropout,
        }.items():
            if value is not None and (float(value) < 0.0 or float(value) >= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0), or None.")

        self.n_channels = n_channels
        self.nfreqs = nfreqs
        self.message_dim = message_dim
        self.hidden_state_dim = hidden_state_dim
        self.encoder_dim = encoder_dim
        self.use_encoder_batch_norm = use_encoder_batch_norm
        self.encoder_dropout = None if encoder_dropout is None else float(encoder_dropout)
        self.use_local_residual = use_local_residual
        self.use_prev_state_mean = use_prev_state_mean
        self.gru_input_dropout = None if gru_input_dropout is None else float(gru_input_dropout)
        self.readout_dropout = None if readout_dropout is None else float(readout_dropout)
        self.time_stride = time_stride
        self.theta_dead_rad = math.radians(theta_dead_deg)
        self.use_raw_in_message = use_raw_in_message
        self.phase_rule_fn = _phase_rule_deadzone_sign

        src_idx, dst_idx = _ordered_pair_indices(n_channels)
        self.register_buffer("src_idx", src_idx, persistent=False)
        self.register_buffer("dst_idx", dst_idx, persistent=False)

        encoder_layers: list[nn.Module] = [nn.Conv1d(1, encoder_dim, kernel_size=5, padding=2)]
        if self.use_encoder_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(encoder_dim))
        encoder_layers.append(nn.ReLU())
        if self.encoder_dropout is not None and self.encoder_dropout > 0.0:
            encoder_layers.append(nn.Dropout(p=self.encoder_dropout))
        encoder_layers.append(nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, padding=2))
        if self.use_encoder_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(encoder_dim))
        encoder_layers.append(nn.ReLU())
        if self.encoder_dropout is not None and self.encoder_dropout > 0.0:
            encoder_layers.append(nn.Dropout(p=self.encoder_dropout))
        self.channel_encoder = nn.Sequential(*encoder_layers)

        message_in_dim = 1 + 2 * message_dim + (2 if use_raw_in_message else 0)
        self.message_mlp = nn.Sequential(
            nn.Linear(message_in_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )
        self.local_enc_proj = (
            nn.Linear(encoder_dim, nfreqs * message_dim) if use_local_residual else None
        )
        self.state_to_freq_proj = nn.Linear(hidden_state_dim, nfreqs * message_dim)
        self.gru_input_proj = nn.Linear(nfreqs * message_dim, hidden_state_dim)
        self.gru_input_dropout_layer = (
            nn.Dropout(self.gru_input_dropout)
            if self.gru_input_dropout is not None and self.gru_input_dropout > 0.0
            else None
        )
        self.readout_dropout_layer = (
            nn.Dropout(self.readout_dropout)
            if self.readout_dropout is not None and self.readout_dropout > 0.0
            else None
        )
        self.state_cell = nn.GRUCell(hidden_state_dim, hidden_state_dim)
        readout_dim = 2 * hidden_state_dim if self.use_prev_state_mean else hidden_state_dim
        self.classifier = nn.Linear(readout_dim, n_classes)

    def _aggregate_per_node_per_freq(self, msg: torch.Tensor) -> torch.Tensor:
        """Aggregate [B, E, F, M] to [B, C, F, M] by destination node."""
        batch_size, num_edges, nfreqs, message_dim = msg.shape
        device = msg.device
        agg = torch.zeros(
            batch_size * self.n_channels * nfreqs,
            message_dim,
            device=device,
            dtype=msg.dtype,
        )
        base_b = (
            torch.arange(batch_size, device=device).view(batch_size, 1, 1)
            * self.n_channels
            * nfreqs
        )
        base_f = torch.arange(nfreqs, device=device).view(1, 1, nfreqs)
        dst = self.dst_idx.view(1, num_edges, 1) * nfreqs + base_f + base_b
        agg.index_add_(
            0,
            dst.reshape(-1),
            msg.reshape(batch_size * num_edges * nfreqs, message_dim),
        )
        return agg.view(batch_size, self.n_channels, nfreqs, message_dim)

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}.")

        enc = raw_x.reshape(batch_size * n_channels, 1, n_time)
        enc = self.channel_encoder(enc)
        enc = enc.reshape(batch_size, n_channels, self.encoder_dim, n_time).permute(0, 1, 3, 2)

        state = torch.zeros(
            batch_size,
            self.n_channels,
            self.hidden_state_dim,
            device=raw_x.device,
        )
        gate_sum = 0.0
        gate_count = 0.0
        prev_state_sum = torch.zeros(batch_size, self.hidden_state_dim, device=raw_x.device)
        last_state_pooled = torch.zeros(batch_size, self.hidden_state_dim, device=raw_x.device)
        step_count = 0

        for t in range(0, n_time, self.time_stride):
            src_r = w_real[:, self.src_idx, t, :]
            src_i = w_imag[:, self.src_idx, t, :]
            dst_r = w_real[:, self.dst_idx, t, :]
            dst_i = w_imag[:, self.dst_idx, t, :]

            xwt_real = src_r * dst_r + src_i * dst_i
            xwt_imag = src_i * dst_r - src_r * dst_i
            xwt_mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
            xwt_mag_log = torch.log1p(torch.nan_to_num(xwt_mag, nan=0.0, posinf=0.0, neginf=0.0))

            ang = torch.atan2(xwt_imag, xwt_real)
            ang = torch.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)
            delta = torch.atan2(torch.sin(ang), torch.cos(ang))
            gate = self.phase_rule_fn(delta, self.theta_dead_rad)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
            gate_sum += float(gate.sum().item())
            gate_count += float(gate.numel())

            freq_state = self.state_to_freq_proj(state).reshape(
                batch_size,
                self.n_channels,
                self.nfreqs,
                self.message_dim,
            )
            src_state = freq_state[:, self.src_idx, :, :]
            dst_state = freq_state[:, self.dst_idx, :, :]
            feats = [xwt_mag_log.unsqueeze(-1), src_state, dst_state]
            if self.use_raw_in_message:
                raw_t = raw_x[:, :, t]
                src_raw = raw_t[:, self.src_idx].unsqueeze(-1).unsqueeze(-1)
                dst_raw = raw_t[:, self.dst_idx].unsqueeze(-1).unsqueeze(-1)
                feats.extend(
                    [
                        src_raw.expand(batch_size, self.src_idx.numel(), self.nfreqs, 1),
                        dst_raw.expand(batch_size, self.dst_idx.numel(), self.nfreqs, 1),
                    ]
                )

            msg = self.message_mlp(torch.cat(feats, dim=-1))
            msg = msg * gate.unsqueeze(-1)
            agg_msg = self._aggregate_per_node_per_freq(msg)

            if self.use_local_residual:
                if self.local_enc_proj is None:
                    raise RuntimeError("local_enc_proj is not initialized.")
                local_enc_term = self.local_enc_proj(enc[:, :, t, :]).reshape(
                    batch_size,
                    self.n_channels,
                    self.nfreqs,
                    self.message_dim,
                )
                update_in_freq = agg_msg + local_enc_term
            else:
                update_in_freq = agg_msg

            update_in = self.gru_input_proj(
                update_in_freq.reshape(batch_size, self.n_channels, self.nfreqs * self.message_dim)
            )
            if self.gru_input_dropout_layer is not None:
                update_in = self.gru_input_dropout_layer(update_in)

            state = self.state_cell(
                update_in.reshape(batch_size * self.n_channels, self.hidden_state_dim),
                state.reshape(batch_size * self.n_channels, self.hidden_state_dim),
            ).view(batch_size, self.n_channels, self.hidden_state_dim)

            pooled_nodes = state.mean(dim=1)
            if self.use_prev_state_mean and step_count > 0:
                prev_state_sum += last_state_pooled
            last_state_pooled = pooled_nodes
            step_count += 1

        readout = last_state_pooled.reshape(batch_size, self.hidden_state_dim)
        if self.use_prev_state_mean:
            prev_state_mean = (
                prev_state_sum / float(step_count - 1)
                if step_count > 1
                else torch.zeros_like(prev_state_sum)
            )
            readout = torch.cat([readout, prev_state_mean], dim=1)
        if self.readout_dropout_layer is not None:
            readout = self.readout_dropout_layer(readout)

        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return self.classifier(readout), edge_density


class XWTPhaseGNNV2Classifier(_BaseCWTGNNClassifier):
    """V2 sklearn/MOABB wrapper with channel-local encoder and freq-indexed state."""

    model_label = "XWT-V2"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 32,
        cwt_resample_n_time: int | None = None,
        time_stride: int = 1,
        theta_dead_deg: float = 45.0,
        message_dim: int = 3,
        hidden_state_dim: int = 32,
        encoder_dim: int = 16,
        use_encoder_batch_norm: bool = True,
        encoder_dropout: float | None = 0.5,
        use_local_residual: bool = True,
        use_prev_state_mean: bool = True,
        gru_input_dropout: float | None = 0.0,
        readout_dropout: float | None = 0.0,
        use_raw_in_message: bool = True,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 0.1,
        normalize_input: bool = True,
        noise_augmentation_enabled: bool = False,
        noise_apply_prob: float = 0.0,
        noise_strength: float = 0.0,
        noise_bank_size: int = 128,
        noise_bank_seed: int | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        # NEW
        channel_subset: list[int] | list[str] | None = None,
        verbose: int = 0,
    ) -> None:
        self.time_stride = time_stride
        self.theta_dead_deg = theta_dead_deg
        self.message_dim = message_dim
        self.hidden_state_dim = hidden_state_dim
        self.encoder_dim = encoder_dim
        self.use_encoder_batch_norm = use_encoder_batch_norm
        self.encoder_dropout = encoder_dropout
        self.use_local_residual = use_local_residual
        self.use_prev_state_mean = use_prev_state_mean
        self.gru_input_dropout = gru_input_dropout
        self.readout_dropout = readout_dropout
        self.use_raw_in_message = use_raw_in_message
        self._init_cwt_gnn_classifier(
            sampling_rate=sampling_rate,
            lowest=lowest,
            highest=highest,
            nfreqs=nfreqs,
            cwt_resample_n_time=cwt_resample_n_time,
            normalize_input=normalize_input,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            noise_augmentation_enabled=noise_augmentation_enabled,
            noise_apply_prob=noise_apply_prob,
            noise_strength=noise_strength,
            noise_bank_size=noise_bank_size,
            noise_bank_seed=noise_bank_seed,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            # NEW
            channel_subset=channel_subset,
            verbose=verbose,
        )

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> XWTPhaseGNNV2Core:
        return XWTPhaseGNNV2Core(
            n_channels=n_channels,
            nfreqs=self.nfreqs,
            n_classes=n_classes,
            message_dim=self.message_dim,
            hidden_state_dim=self.hidden_state_dim,
            encoder_dim=self.encoder_dim,
            use_encoder_batch_norm=self.use_encoder_batch_norm,
            encoder_dropout=self.encoder_dropout,
            use_local_residual=self.use_local_residual,
            use_prev_state_mean=self.use_prev_state_mean,
            gru_input_dropout=self.gru_input_dropout,
            readout_dropout=self.readout_dropout,
            time_stride=self.time_stride,
            theta_dead_deg=self.theta_dead_deg,
            use_raw_in_message=self.use_raw_in_message,
            **kwargs,
        )


# ============================================================================
# Originally sparse_evidence_gnn_classifier.py. That file's own module
# docstring, preserved verbatim as a comment (see this file's own module
# docstring above for why):
#
# 2026-08-22 correction (Epilepsy fork): the verbatim BCI/moabb_pipelines text
# below repeatedly says "36 canonical edges" (and "9-channel subset") as if
# that were a fixed constant of this code. It never was -- edge count is
# always C(n_channels, 2) (upper_pair_indices/_ordered_pair_indices), and 36
# was only ever true for the MOTOR-IMAGERY paradigm's specific 9-channel
# montage this file was ported from. This Epilepsy fork's actual default
# (channel_subset=None over CHB-MIT's 23-channel montage) gives
# C(23, 2) = 253 edges -- confirmed directly against a real run's own printed
# config ("n_channels=23 edges=253 nfreqs=8 ..."), not 36. Several of the same
# comments also assume nfreqs=16, halved to nfreqs=8 for this fork on
# 2026-08-16 (see run_pipelines.py's _SHARED_ARCH_PARAMS). Take any concrete
# cell-count/memory-size arithmetic in the historical text below as
# illustrating the METHOD (how to reason about scaling), not this fork's
# actual current numbers -- it was wrong once already (an earlier session
# reading this file cited "36 edges" as fact instead of checking the live
# config, and had to be corrected).
#
# Sparse/event-based WCT evidence GNN classifier.
#
# Instead of pooling coherence into fixed time windows (as WCTEvidenceGNN
# does), this computes coherence + phase at full time resolution, thresholds
# them, and CONSOLIDATES temporally-adjacent surviving samples (per channel
# pair, per frequency bin) into region-level "events" -- one event per burst,
# not one per sample. Each event carries (timestamp, frequency, magnitude,
# sin/cos(angle)) plus a learned per-channel signal embedding for its source
# and destination channel, and is routed into its destination node's evidence
# via the same graph topology as WCTEvidenceGNN.
#
# Validated in exploratory testing (BNCI2014-001, cross-session, canonical
# 4-subject run via run_canonical_setup.py): subj1=0.801 subj2=0.557 subj3=0.947
# subj4=0.539, pipeline mean=0.711. subj2/subj4 sitting near chance (0.5) is a
# property of those subjects, not this pipeline specifically -- EEGNet (100
# epochs) gets 0.603 on subject 2 too. Earlier single-subject number (0.750 on
# subject 1) is superseded by the above; not yet validated on subjects 5-9.
# See ChannelSignalEncoder's docstring below for the receptive-field fix
# (channel_encoder_dilation) that this accuracy depends on.
#
# 2026-08-09: edge topology changed from 72 directed pairs (i->j and j->i as
# two separate edges, for this pipeline's 9-channel subset) to 36 canonical
# (i<j) undirected pairs (upper_pair_indices), with the coherence/gate math
# updated to match. Motivated by the 2026-08-07 phase-gate investigation (see
# below): the full-edge cross-spectrum computed the exact-conjugate j->i copy
# of every i->j cross-spectrum independently (xwt_(j->i) = conj(xwt_(i->j))
# always), so every coh/phase/threshold/gate tensor downstream carried 2x
# redundant edges -- directly the "coh_all is ~4.3GB per trial, too large to
# cache" cost noted in 2026-08-08 session notes, Arc 6. Collapsing to one
# canonical edge per pair and gating on phase.abs() > threshold (instead of
# picking a direction via which of two duplicate edges fired) is exactly
# mathematically equivalent -- sin(mean_angle)'s sign now carries the
# direction bit that used to live in "which edge fired" -- and does NOT
# reintroduce the 2026-08-07 `.abs()` regression: that regression's cause was
# two INDEPENDENT edges both applying the same symmetric |phase|>threshold
# test and therefore firing together on every qualifying cell (since
# |phase_ij|=|phase_ji| always); with only one edge per pair there is no
# second edge left to double-fire. See _build_sparse_events and
# _max_cluster_statistic for the gate itself, and surrogate_null_cache_key's
# `edge_topology` argument for why this needed a cache-key change (raw_trial's
# bytes alone don't distinguish the two topologies -- same channel count
# either way -- so without it a pre-2026-08-09 72-edge cache entry could be
# loaded and misinterpreted as if it were the new 36-edge shape).
#
# Two fixes validated via debug_plots/edge0_*.png before being wired in here:
#   - cwt_resample_n_time now defaults to None (native resolution). Resampling
#     the complex CWT coefficients via scipy.signal.resample (the old default,
#     200) was destroying real signal above ~n_time/(2*trial_secs) Hz -- a
#     clean 30Hz test tone measured 0.81 magnitude natively vs 0.006 after
#     resample to 200 samples on a ~4s trial.
#   - SparseEvidenceGNNCore._build_sparse_events now ANDs a cone-of-influence
#     mask into its gate (see _coi_valid_mask): fcwt.cwt() returns no COI, and
#     without it, events could be built from time/freq cells where the wavelet
#     ran off the edge of the trial.
#   - _build_sparse_events (coherence/gate/COI/run-consolidation) is entirely
#     non-trainable and deterministic given fixed CWT features, yet forward()
#     used to call it on every (batch, epoch) -- profiling showed it was 94.8%
#     of forward()'s time. SparseEvidenceGNNClassifier._prepare_features now
#     calls it once per trial (see _precompute_sparse_events) and forward()
#     only does the trainable part (channel_encoder + sparse_message_mlp +
#     sparse_classifier) every step. Measured ~9x faster end to end.
# The kernel_size=(5,3) smoothing is deliberately left at this value: at native
# resolution (n_time~1001, 4.0ms/sample) it spans only ~20ms of time smoothing,
# vs ~100ms if the kernel were widened to (25,3) to match the smoothing width
# the old cwt_resample_n_time=200 pipeline had by accident. Re-tested (25,3)
# against (5,3) after the channel_encoder_dilation fix above -- 0.8008 vs
# 0.7991 on subject 1, still noise-level -- so widening the kernel buys nothing
# and (5,3) is kept for the time/frequency resolution this sparse-event
# architecture is built to exploit.
#
# 2026-08-09 reorganization: this used to be spread across three files --
# this module (subclassing WCTEvidenceGNNCore purely to borrow two of its
# non-trainable coherence methods), wct_evidence_gnn_classifier.py (that base
# class, whose windowed-pipeline machinery -- feature_conv, message_mlp,
# window layout, memory-estimate summaries -- this pipeline never used), and
# common.py (which held the surrogate-null-cache disk format and
# phase-randomization helpers that only this pipeline calls). It's now one
# self-contained file:
#   - SparseEvidenceGNNCore no longer subclasses WCTEvidenceGNNCore. It
#     subclasses nn.Module directly and carries its own copies of the three
#     small (parameter-free) coherence methods it actually used
#     (_full_edge_wct_maps, _smooth_wct_maps, _batched_freqs) instead of
#     inheriting a windowed-pipeline base class wholesale -- the parent's own
#     feature_conv/message_mlp/classifier submodules were constructed but
#     never called by this class's forward() (dead weight, now simply not
#     built), and its print_custom_summary reported windowed-pipeline
#     quantities (window_compute_mode, feature_conv shapes) that don't apply
#     here, replaced below with a summary of what this pipeline actually
#     computes.
#   - The surrogate-null-cache helpers (SURROGATE_NULL_CACHE_PERCENTILES,
#     default_surrogate_cache_root, surrogate_null_cache_key,
#     load_surrogate_null_cache, save_surrogate_null_cache,
#     phase_randomize_surrogates, resolve_best_available_device) moved in from
#     common.py verbatim -- nothing else in the repo called them, so this is a
#     move, not a duplicate copy that could drift.
#   - compute_cwt_real_imag_tensors, make_gaussian_weight2d, and
#     upper_pair_indices are genuinely shared with other pipelines (WCT/XWT,
#     Coherence-CNN, CWT-CNN) and stay imported from common.py rather than
#     being copied in, to avoid a second copy that could drift from theirs.
#     _BaseCWTGNNClassifier (the generic sklearn fit/predict/noise-
#     augmentation scaffolding shared with the other CWT-GNN pipelines) stays
#     imported from xwt_phase_gnn_classifier.py for the same reason.
#   - SparseEvidenceGNNCore.__init__ still accepts (and ignores)
#     `model_init_seed` for call-site compatibility with
#     SparseEvidenceGNNClassifier._build_model, which still passes it. This
#     mirrors pre-reorg behavior exactly: under the old subclassing scheme,
#     model_init_seed only ever scoped the (unused, now-removed)
#     feature_conv/message_mlp/classifier's initialization, never
#     channel_encoder/sparse_message_mlp/sparse_classifier's -- so it was
#     already a no-op for the submodules this class actually trains. Not
#     changed here; flagged as a possible real fix for a future change, not
#     bundled into this reorganization since it would alter training-run
#     reproducibility/results.
# ============================================================================

# --- Surrogate null-distribution cache -------------------------------------
#
# Computing surrogate_count surrogate CWTs + coherence maps per trial is the
# expensive part of coherence_threshold_mode="surrogate" (see
# SparseEvidenceGNNClassifier._surrogate_coherence_threshold). The
# percentile grid below is cached to disk keyed by everything that affects
# the null distribution EXCEPT the percentile itself, so trying a different
# surrogate_percentile against the same trial/config is a free cache hit
# rather than a recompute.
#
# Moved in from common.py 2026-08-09 -- this pipeline was the only caller of
# any of these.

SURROGATE_NULL_CACHE_PERCENTILES = np.linspace(0.0, 100.0, 201)  # every 0.5pt


def default_surrogate_cache_root() -> Path:
    configured = (
        os.environ.get("MNE_DATASETS_BNCI_PATH")
        or os.environ.get("MNE_DATA")
        or str(Path.home() / "mne_data")
    )
    return Path(configured).expanduser() / "surrogate_null_cache"


def surrogate_null_cache_key(
    raw_trial: np.ndarray,
    *,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: int | None,
    smooth_kernel_size: tuple,
    smooth_kernel_sigma: tuple,
    coi_enabled: bool,
    surrogate_count: int,
    surrogate_seed: int,
    edge_topology: str = "directed_ij_ji",
    scale_adaptive_smoothing: bool = False,
    scale_adaptive_cycles: float = 1.5,
    scale_adaptive_max_kernel: int = 101,
    cwt_backend: str = "fcwt",
) -> str:
    """Deterministic cache key covering every input that affects the null
    coherence distribution -- deliberately NOT surrogate_percentile, since
    that's just a lookup into the same cached distribution. Any change to
    the trial's own signal or to these config values changes the hash, so
    there's no separate invalidation logic needed: a stale cache entry is
    simply unreachable under its old key.

    `cwt_backend` (added 2026-08-20, Step 6 of the torch-native-cwt swap):
    the null distribution is built from `compute_cwt_real_imag_tensors`
    output, so switching backends must not silently serve a stale entry
    computed under the other one. Defaults to "fcwt" so existing on-disk
    entries key identically to before. (The CWT-window and dense-edge
    caches this originally cross-referenced were removed 2026-08-21 --
    this surrogate-null cache is a separate, opt-in mechanism, not on the
    default `coherence_threshold_mode="fixed"` path, and was left in
    place.)

    `edge_topology` (2026-08-09): distinguishes the cached grid's edge axis
    layout -- e.g. "directed_ij_ji" (the original 2*C(n,2) directed-pair
    scheme) vs "canonical_undirected" (C(n,2) canonical i<j pairs,
    upper_pair_indices, adopted by SparseEvidenceGNNCore to stop redundantly
    computing the exact-conjugate j->i copy of every i->j cross-spectrum).
    raw_trial's bytes alone don't distinguish these -- the channel count
    (hence trial shape) is identical either way -- so without this the two
    topologies would collide on the same cache key and a values_grid loaded
    under one edge count would be silently reshaped as if it were the
    other. Defaults to the pre-existing scheme so old callers/cache entries
    keep resolving to the same key unless they opt in.

    `scale_adaptive_smoothing`/`scale_adaptive_cycles`/`scale_adaptive_max_kernel`
    (2026-08-09): same reasoning as `edge_topology` -- these change the
    actual coherence values computed (per-frequency time-smoothing kernel
    width instead of one flat width), not just how they're interpreted, so
    a values_grid computed under one setting must never be loaded and
    reused under another. Defaults preserve the pre-existing flat-kernel
    cache key for old callers.
    """
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(raw_trial, dtype=np.float32).tobytes())
    config_tuple = (
        int(sampling_rate), float(highest), float(lowest), int(nfreqs),
        None if cwt_resample_n_time is None else int(cwt_resample_n_time),
        tuple(smooth_kernel_size), tuple(smooth_kernel_sigma),
        bool(coi_enabled), int(surrogate_count), int(surrogate_seed),
        str(edge_topology),
        bool(scale_adaptive_smoothing), float(scale_adaptive_cycles),
        int(scale_adaptive_max_kernel), str(cwt_backend),
    )
    hasher.update(repr(config_tuple).encode("utf-8"))
    return hasher.hexdigest()


def load_surrogate_null_cache(
    cache_dir: Path,
    key: str,
    phase_threshold_deg: float | None = None,
    forming_percentile: float | None = None,
) -> dict | None:
    """Returns a dict with the cached 'values' ([E, F, len(percentiles)]
    per-cell percentile grid, phase-independent) and, if present and valid,
    'cluster_null' ([N_surrogates, E] per-EDGE max-cluster-statistic null
    distribution, for coherence_threshold_mode="surrogate_cluster" -- see
    SparseEvidenceGNNCore._max_cluster_statistic's docstring for why this is
    scoped per-edge rather than pooled across the whole edge/freq/time
    search space), or None on a cache miss (file absent, unreadable, or
    written under an older percentile grid -- all treated as a plain miss,
    not an error, so a stale/corrupt entry just triggers a recompute).

    Unlike `values` (which never depends on phase or the forming
    percentile), `cluster_null` DOES depend on both: its cluster-forming
    gate is coherence-above-the-forming-percentile-threshold AND phase
    jointly, to match what real candidate clusters must clear (see
    SparseEvidenceGNNCore._max_cluster_statistic). So a cached cluster_null
    is only returned if BOTH `phase_threshold_deg` and `forming_percentile`
    match the values it was computed under; a mismatch on either returns
    'cluster_null': None (a plain recompute-just-that-part, not a full
    cache miss -- 'values' is still served from cache since it's
    unaffected by either). A `cluster_null` written under the older
    whole-trial-scalar scheme (shape [N_surrogates], 1-D -- pre-per-edge
    scoping) is also treated as a miss: shape is validated against
    `values`'s own edge count so a stale-format entry self-heals via the
    normal recompute+backfill path instead of silently feeding
    wrong-shaped data downstream.
    """
    path = cache_dir / f"{key}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            if not np.array_equal(data["percentiles"], SURROGATE_NULL_CACHE_PERCENTILES):
                return None
            values = data["values"]
            cluster_null = None
            if "cluster_null" in data:
                cached_phase_deg = float(data["cluster_null_phase_threshold_deg"])
                cached_forming_pct = float(data["cluster_null_forming_percentile"])
                phase_matches = phase_threshold_deg is not None and np.isclose(
                    cached_phase_deg, float(phase_threshold_deg)
                )
                forming_matches = forming_percentile is not None and np.isclose(
                    cached_forming_pct, float(forming_percentile)
                )
                candidate = data["cluster_null"]
                shape_matches = candidate.ndim == 2 and candidate.shape[1] == values.shape[0]
                if phase_matches and forming_matches and shape_matches:
                    cluster_null = candidate
            return {"values": values, "cluster_null": cluster_null}
    except Exception:
        return None


def save_surrogate_null_cache(
    cache_dir: Path,
    key: str,
    values: np.ndarray,
    cluster_null: np.ndarray | None = None,
    cluster_null_phase_threshold_deg: float | None = None,
    cluster_null_forming_percentile: float | None = None,
) -> None:
    """Atomic write (temp file + rename) so a concurrent reader never sees a
    half-written cache file. cluster_null (shape [N_surrogates, E], one
    max-cluster-mass value per surrogate PER EDGE -- see
    SparseEvidenceGNNCore._max_cluster_statistic) is optional -- only
    populated by coherence_threshold_mode="surrogate_cluster" (see
    SparseEvidenceGNNClassifier._surrogate_null_percentile_grid) -- and, if
    given, both cluster_null_phase_threshold_deg and
    cluster_null_forming_percentile must be given too (see
    load_surrogate_null_cache's docstring on why cluster_null depends on
    both while values depends on neither)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / f"{key}.npz"
    tmp_path = cache_dir / f".{key}.{os.getpid()}.tmp.npz"
    save_kwargs = dict(percentiles=SURROGATE_NULL_CACHE_PERCENTILES, values=values)
    if cluster_null is not None:
        save_kwargs["cluster_null"] = cluster_null
        save_kwargs["cluster_null_phase_threshold_deg"] = float(cluster_null_phase_threshold_deg)
        save_kwargs["cluster_null_forming_percentile"] = float(cluster_null_forming_percentile)
    np.savez(tmp_path, **save_kwargs)
    os.replace(tmp_path, final_path)


def phase_randomize_surrogates(
    x: np.ndarray, n_surrogates: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate phase-randomized surrogates of a single multi-channel trial.

    For each channel independently, replaces the real FFT's phase spectrum
    with fresh uniform-random phases while keeping the magnitude spectrum
    exactly -- so every surrogate has the same power spectrum / autocorrelation
    as the real channel, but any genuine cross-channel phase coupling (the
    thing coherence actually measures) is destroyed. This is the standard
    null-hypothesis generator for coherence significance testing (Theiler
    et al. 1992 surrogate-data framework). Used by
    SparseEvidenceGNNClassifier's coherence_threshold_mode="surrogate" to
    calibrate a per-(edge, frequency) significance threshold in place of a
    fixed magnitude cutoff.

    Uses rfft/irfft (real-signal FFT), so conjugate symmetry is automatic --
    only the non-negative-frequency bins need randomizing, and DC (and
    Nyquist, if n_time is even) must stay real-valued (zero phase) for
    irfft to return a real-valued signal.

    Parameters
    ----------
    x : ndarray, shape (n_channels, n_time)
        A single real trial's raw signal, already at whatever
        preprocessing stage (channel-subset, normalization) it will be fed
        through the real CWT.
    n_surrogates : int
        Number of independent surrogate copies to generate.
    rng : np.random.Generator

    Returns
    -------
    ndarray, shape (n_surrogates, n_channels, n_time), float32
    """
    n_channels, n_time = x.shape
    spectrum = np.fft.rfft(x, axis=-1)  # (n_channels, n_freq_bins)
    magnitude = np.abs(spectrum)
    n_freq_bins = spectrum.shape[-1]

    random_phase = rng.uniform(
        0.0, 2.0 * np.pi, size=(n_surrogates, n_channels, n_freq_bins)
    )
    random_phase[:, :, 0] = 0.0  # DC must stay real
    if n_time % 2 == 0:
        random_phase[:, :, -1] = 0.0  # Nyquist must stay real

    surrogate_spectrum = magnitude[None, :, :] * np.exp(1j * random_phase)
    surrogates = np.fft.irfft(surrogate_spectrum, n=n_time, axis=-1)
    return surrogates.astype(np.float32)


def resolve_best_available_device(device: str) -> torch.device:
    """cuda -> mps -> cpu, unlike a plain "auto" that only ever checks cuda
    (and so silently falls back to cpu on Apple Silicon even when MPS is
    available). Used for this pipeline's surrogate coherence/smoothing
    calibration (a real batched conv2d over dozens of edges x ~1000 time x
    16 freq, per surrogate_device below) -- NOT for the trainable model/
    training loop (`device` on SparseEvidenceGNNClassifier), which is small
    (hidden_dim~8, tiny batches) and where MPS's kernel-launch overhead can
    dominate real compute and may regress; that would need its own
    measurement before switching its default."""
    if device != "auto":
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        if str(device) == "mps" and not (
            torch.backends.mps.is_available() and torch.backends.mps.is_built()
        ):
            return torch.device("cpu")
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _sync_device(device: torch.device) -> None:
    """Blocks until all queued work on `device` has actually finished.

    Needed anywhere a `time.perf_counter()` window is meant to measure real
    GPU work: CUDA/MPS kernel launches return to Python immediately (the
    work is only queued), so an un-synced timer mostly measures launch
    overhead and silently attributes the real compute time to whatever
    later call happens to block (e.g. a `.cpu()` or `.item()`) -- see
    _precompute_dense_edge_inputs's phase-timing block, added specifically
    to stop guessing at where that stage's CUDA time goes (2026-08-22,
    following up on the unresolved profiling question in the 2026-08-21
    session notes)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _interp_percentile_grid(
    values_grid: np.ndarray, percentile: float | np.ndarray
) -> np.ndarray:
    """Interpolates a specific percentile out of a cached [E, F,
    len(SURROGATE_NULL_CACHE_PERCENTILES)] null-coherence grid. The grid
    points are evenly spaced 0..100, so this is a plain fractional-index
    lookup -- no per-(edge, freq) Python loop needed. Returns [E, F].

    `percentile` may be a scalar (applied uniformly to every frequency, the
    original behavior) OR a 1-D array of length F giving a DIFFERENT
    percentile per frequency bin -- e.g. SparseEvidenceGNNClassifier's
    mu_band_surrogate_percentile, which relaxes the significance bar
    specifically in the mu band where the 2026-08-08 session notes (Arc 5)
    found genuinely phase-consistent activity was being uniformly rejected
    by the flat threshold (verified on both subject 1 and subject 2, worse
    for subject 2 -- it loses the band entirely where subject 1 recovers at
    an adjacent bin)."""
    n_grid_points = values_grid.shape[-1]
    step = SURROGATE_NULL_CACHE_PERCENTILES[1] - SURROGATE_NULL_CACHE_PERCENTILES[0]
    percentile_arr = np.asarray(percentile, dtype=np.float64)
    frac_idx = np.clip(percentile_arr / step, 0, n_grid_points - 1)
    if percentile_arr.ndim == 0:
        lo, hi = int(np.floor(frac_idx)), int(np.ceil(frac_idx))
        w = frac_idx - lo
        return (values_grid[:, :, lo] * (1.0 - w) + values_grid[:, :, hi] * w).astype(np.float32)
    # Per-frequency percentile: lo/hi/w are each [F] -- gather per-frequency
    # grid slices via take_along_axis instead of a single scalar index.
    lo = np.floor(frac_idx).astype(np.int64)
    hi = np.ceil(frac_idx).astype(np.int64)
    w = (frac_idx - lo).astype(np.float32)
    n_edges, n_freqs, _ = values_grid.shape
    lo_b = np.broadcast_to(lo.reshape(1, n_freqs, 1), (n_edges, n_freqs, 1))
    hi_b = np.broadcast_to(hi.reshape(1, n_freqs, 1), (n_edges, n_freqs, 1))
    lo_vals = np.take_along_axis(values_grid, lo_b, axis=2)[:, :, 0]
    hi_vals = np.take_along_axis(values_grid, hi_b, axis=2)[:, :, 0]
    return (lo_vals * (1.0 - w.reshape(1, n_freqs)) + hi_vals * w.reshape(1, n_freqs)).astype(
        np.float32
    )


def _build_dense_feature_conv(
    *,
    in_channels: int,
    kernel_size: int,
    intermediate_channels: int,
    out_channels: int,
    pool_size: int,
    intermediate_channels_reduced: int | None = None,
) -> nn.Module:
    """event_mode="dense"'s learned per-edge conv stack -- modeled on
    WCTEvidenceGNNCore's _build_feature_conv (wct_evidence_gnn_classifier.py:
    same two-block conv+pool pattern, same kernel_size/pool_size/
    intermediate_channels/out_channels/intermediate_channels_reduced knobs),
    but NOT a copy of it: that function convolves WCT's raw per-channel
    signal; this one convolves SparseEvidenceGNNCore's own already-fixed
    coherence/phase/significance arrays (native resolution, no
    cwt_resample_n_time, post-COI-mask -- see _build_dense_edge_input),
    which is a different input entirely.

    Every conv/pool here uses kernel_size=(1, k) / pool_size=(1, p) -- height
    (the edge axis, size E) is always kernel/stride 1, so edges are
    convolved independently with SHARED weights, exactly how WCTEvidenceGNNCore's
    own feature_conv treats its channel axis (see that function's docstring
    reference in ChannelSignalEncoder above). Width (the time axis) is where
    kernel_size/pool_size actually reduce -- frequency is folded into
    `in_channels` by the caller (SparseEvidenceGNNCore.forward) BEFORE this
    stack ever sees it, so the first conv layer is free to learn
    cross-frequency combinations, unlike the edge axis which stays untouched
    per-edge throughout.

    Unlike WCTEvidenceGNNCore's version, this omits BatchNorm/Dropout --
    matches this file's own existing learned-conv precedent
    (ChannelSignalEncoder, the only other trainable conv block in this
    pipeline, uses plain Conv1d+GELU+pool with no normalization/
    regularization layers either) rather than importing WCT's heavier
    nn_components-based regularization stack for a first, not-yet-validated
    pathway.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, intermediate_channels, kernel_size=(1, kernel_size)),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
    ]
    conv2_in_channels = intermediate_channels
    if intermediate_channels_reduced is not None:
        conv2_in_channels = intermediate_channels_reduced
        layers += [
            nn.Conv2d(intermediate_channels, intermediate_channels_reduced, kernel_size=(1, 1)),
            nn.GELU(),
        ]
    layers += [
        nn.Conv2d(conv2_in_channels, out_channels, kernel_size=(1, kernel_size)),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
        # Collapses whatever time steps survive the two pool stages down to
        # one value per (edge, out_channel) -- same "pool the whole trial
        # into one vector" choice ChannelSignalEncoder's own
        # AdaptiveAvgPool1d(1) makes (see that class's 2026-08-09 revert
        # note: per-timestep/event-relative pooling was tried and regressed
        # accuracy). `None` in the first output_size slot means "leave the
        # edge (height) axis untouched" -- only width (time) is pooled.
        nn.AdaptiveAvgPool2d((None, 1)),
    ]
    return nn.Sequential(*layers)


class _DenseEdgeGRUTemporal(nn.Module):
    """dense_edge_temporal_mode="rnn" counterpart to _build_dense_feature_conv
    -- 2026-08-11, event_mode="dense" only. Replaces Conv2d's small, fixed
    local receptive field along the time axis (dense_conv_kernel_size,
    default 5) with a GRU that can integrate the FULL T' sequence with
    memory, motivated by the time_averaged_graph ablation showing
    time-collapsing T' to 1 doesn't hurt accuracy -- before drawing any
    conclusion about whether time matters, the architecture needs a
    mechanism that can actually see the whole sequence, which Conv2d's
    local kernel never could (see run_pipelines.py's DENSE_EDGE_GRU_PARAMS and
    dense_edge_temporal_mode docstring for the full rationale and
    [[sparse-evidence-gnn-time-averaged-graph-feature]] in project memory).

    Same "edges convolved/processed independently with SHARED weights"
    property _build_dense_feature_conv's own kernel_size=(1, k) / height-1-
    stride-1 convention gives it (see that function's docstring): the GRU
    runs once per edge, batched over (batch, edge) together, with identical
    weights for every edge. Frequency is folded into the GRU's per-timestep
    input vector the SAME way _build_dense_feature_conv folds it into
    Conv2d's in_channels -- SparseEvidenceGNNCore._dense_edge_features does
    that fold once, upstream of either path, so this is a per-EDGE GRU, NOT
    a per-(edge, frequency) GRU: a single edge's GRU hidden state mixes all
    nfreqs frequency bins (and all 4 raw stack channels: coh/sinφ/cosφ/
    significance) together at every timestep, exactly like Conv2d's own
    first layer already does. A genuinely per-(edge, frequency) GRU (nfreqs
    independent GRUs per edge, never mixing frequency until some later
    step) would be a bigger, un-requested architecture change -- flagged
    back to the user rather than silently built.

    Consumes/produces the exact same [B, C_in, E, T] -> [B, out_channels, E,
    1] shape contract _build_dense_feature_conv's nn.Sequential does (C_in =
    4 * nfreqs), so SparseEvidenceGNNCore._dense_edge_features'
    `.squeeze(-1).permute(0, 2, 1)` call downstream is completely unchanged
    regardless of which path built self.dense_edge_conv -- dense_conv_
    out_channels plays the exact same "per-edge feature width" role either
    way. Uses the GRU's own final hidden state h_T as the "T pooled to one
    value" counterpart to Conv2d's `nn.AdaptiveAvgPool2d((None, 1))` -- an
    actual summary of the whole sequence carried through memory, not a
    local-window average.

    dense_conv_kernel_size/dense_conv_pool_size/dense_conv_intermediate_
    channels/dense_conv_intermediate_channels_reduced (Conv2d-path-only
    hyperparameters) have no meaning here and are simply not read by this
    class -- unlike this file's other no-op-combination params, __init__
    does NOT reject non-default values of these alongside
    dense_edge_temporal_mode="rnn", since (unlike e.g.
    dense_edge_time_downsample) there is no way here to tell whether a
    given value was explicitly passed or is just its ordinary default.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=out_channels, batch_first=True)

    def forward(self, conv_in: torch.Tensor) -> torch.Tensor:
        """`conv_in`: [B, C_in, E, T] -- identical input _build_dense_feature_
        conv's Sequential receives (C_in = 4 * nfreqs, frequency already
        folded in by the caller; see class docstring). Reshapes to
        [B*E, T, C_in] (batch_first GRU convention, edges folded into the
        batch dim so one GRU instance processes every edge with shared
        weights), runs the GRU, and reshapes its final hidden state back to
        the [B, out_channels, E, 1] shape _dense_edge_features' downstream
        squeeze/permute expects.
        """
        batch_size, c_in, num_edges, n_time = conv_in.shape
        seq = conv_in.permute(0, 2, 3, 1).reshape(batch_size * num_edges, n_time, c_in)
        _, h_n = self.gru(seq)  # h_n: [1, B*E, out_channels]
        out = h_n.squeeze(0).reshape(batch_size, num_edges, -1)  # [B, E, out_channels]
        return out.permute(0, 2, 1).unsqueeze(-1)  # [B, out_channels, E, 1]


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

    2026-08-09: briefly tried dropping the AdaptiveAvgPool1d(1) below in
    favor of a per-timestep embedding gathered at each event's own time
    index (motivated by feature_ablation="zero_event_features" barely
    moving accuracy under the pooled version -- see
    [[sparse-evidence-gnn-channel-encoder-dominates]] in project memory).
    Reverted: every single-variable threshold/kernel experiment run right
    after landed near-chance for subject 2, and training broke down more
    broadly -- plausibly because gathering one timestep per event means
    channel_encoder's conv layers only get gradient at the handful of
    positions actual events land on per step, instead of every position via
    the old pooled average, a much sparser/noisier training signal.
    Full snapshot of that version (both this class and the forward()-side
    gather logic it needed) is kept in
    session_notes/snapshots/2026-08-09_channel_encoder_per_timestep.md for
    easy reinstatement if revisited -- e.g. with a local window pool around
    each event's time index instead of either whole-trial pooling or a
    single raw sample, to get event-relative timing back without cutting
    gradient coverage down to single points."""

    def __init__(self, embed_dim: int, dilation: int = 1):
        super().__init__()
        kernel_size = 9
        pad = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=kernel_size, padding=pad, dilation=dilation), nn.GELU(),
            nn.Conv1d(8, embed_dim, kernel_size=kernel_size, padding=pad, dilation=dilation), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_time = raw_x.shape
        x = raw_x.reshape(batch_size * n_channels, 1, n_time)
        emb = self.net(x).squeeze(-1)
        return emb.reshape(batch_size, n_channels, -1)


class SparseEvidenceGNNCore(nn.Module):
    """Torch core: full-resolution coherence -> region-consolidated sparse
    events -> per-channel-conditioned message passing -> flatten -> classify.

    Self-contained nn.Module (see the module docstring's 2026-08-09 note --
    this no longer subclasses WCTEvidenceGNNCore). _full_edge_wct_maps and
    _smooth_wct_maps below are copies of that class's (parameter-free)
    coherence computation methods, kept identical so the underlying wavelet
    math is unchanged from the windowed pipeline; everything else here is
    specific to this class.
    """

    # Cone-of-influence wavelet parameter. Must match coherence_utils.transform's
    # hardcoded `fcwt.Morlet(2.0)` -- fcwt.cpp's own edge-of-support formula is
    # getSupport(scale) = int(fb*scale*3.0), scale == sampling_rate/freq for this
    # wavelet. fcwt.cwt() returns no COI itself; this reproduces it from source.
    _COI_WAVELET_FB = 2.0

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
        hidden_dim: int = 8,
        channel_embed_dim: int = 8,
        coherence_threshold: float = 0.5,
        phase_threshold_deg: float = 30.0,
        smooth_kernel_sigma: tuple[float | None, float | None] = (None, None),
        smooth_kernel_size: tuple[int | None, int] = (5, 3),
        # See SparseEvidenceGNNClassifier's matching params for the full
        # rationale -- forwarded through unchanged by _build_model.
        scale_adaptive_smoothing: bool = False,
        scale_adaptive_cycles: float = 1.5,
        scale_adaptive_max_kernel: int = 101,
        # Accepted for call-site compatibility with SparseEvidenceGNNClassifier
        # ._build_model, which still passes it -- NOT currently used to seed
        # channel_encoder/sparse_message_mlp/sparse_classifier's
        # initialization. See the module docstring's 2026-08-09 note: this
        # was already a no-op for those submodules before this reorg (it
        # only ever scoped the old, unused, now-removed
        # feature_conv/message_mlp/classifier's init), so leaving it unwired
        # here changes nothing about existing behavior.
        model_init_seed: int | None = None,
        sampling_rate: int = 250,
        coi_enabled: bool = True,
        channel_encoder_dilation: int = 1,
        feature_ablation: str = "zero_channel_embed",
        # 2026-08-10: "mean" (default) is the original behavior -- every
        # event landing on a destination channel contributes an EQUAL share
        # to that channel's evidence (scatter_add then divide by active
        # count), regardless of how strongly sparse_message_mlp's own output
        # for that event might encode "this one matters more." There is no
        # mechanism by which one event can outweigh another in the
        # aggregation itself; the MLP can only push a message's magnitude
        # around, which then gets diluted (not selected) by however many
        # OTHER events happen to land on the same channel in that trial.
        # "gated_softmax" adds exactly that mechanism: event_gate (below)
        # produces one learned logit per event from the same features
        # sparse_message_mlp sees, softmax-normalized per (trial, destination
        # channel) group via the scatter-softmax below, and evidence becomes
        # a WEIGHTED sum instead of a flat mean -- weights already sum to 1
        # per group, so this replaces (not adds to) the /active_count step.
        # Added specifically to test whether the event pathway's
        # near-zero-cost feature_ablation="zero_event_features" result (see
        # that param's docstring / [[sparse-evidence-gnn-channel-encoder-
        # dominates]] in project memory) reflects the events genuinely
        # carrying little signal, or just this pathway being structurally
        # unable to express "which events matter" under flat mean-pooling.
        #
        # RESULT (2026-08-10, 3-seed comparison, subject 1, epochs=75):
        # "gated_softmax" changed nothing -- mean=0.8400 vs "mean"'s 0.8405
        # (feature_ablation="none"), and the zero_event_features cost stayed
        # ~0.006-0.007 either way, well inside seed noise. A follow-up check
        # of the trained event_gate's actual softmax weights found why: they
        # converged to near-perfectly uniform (entropy/max-entropy ratio
        # 0.993 across ~1150 (trial, channel) groups, weight std ~0.005 on a
        # ~0.005 uniform baseline) despite the mechanism being verified
        # capable of concentrating weight on a subset of events (see the
        # unit tests this was landed with). Given the freedom to
        # differentiate events, training found nothing worth differentiating
        # -- this is evidence the event pathway's low contribution is
        # upstream of aggregation (see [[sparse-evidence-gnn-frequency-
        # fragmentation-bias]]), not a fixable weighting-mechanism gap.
        #
        # "concat" (2026-08-10, event_mode="dense" AND n_hops=1 only --
        # __init__ raises otherwise): a third option, addressing "mean" and
        # "gated_softmax"'s shared property of collapsing every incident
        # edge into ONE hidden_dim vector per channel before
        # sparse_classifier sees it. Motivated by the dense-mode flat-
        # control finding that never pooling at all (flatten dense_edge_
        # conv's raw per-edge output straight to a linear classifier) beat
        # every graph/mean-pool variant tested -- "concat" is the graph-
        # topology-respecting middle ground: keeps every incident edge's own
        # message distinct (like the flat control) but still routes it
        # through sparse_message_mlp/per-node grouping (unlike the flat
        # control, which has no notion of channels at all). See
        # _aggregate_events' "concat" branch docstring for the mechanism and
        # [[sparse-evidence-gnn-capacity-confound-refuted]] in project
        # memory for the validated numbers (screening: 0.9184 seed 42;
        # end-to-end: 0.8973 seed 42, subject 1 -- still below every
        # flat-family number, so this is a working, validated OPTION, not a
        # pipeline-wide recommendation).
        event_aggregation: str = "mean",
        # 2026-08-10: message passing through _aggregate_events is single-hop
        # -- an event becomes a message that scatter_adds onto its
        # DESTINATION channel's evidence once, and that's the last time any
        # information moves between channels before flatten+classify. A
        # channel with no events landing on it directly (or whose real
        # discriminative signal lives on a NEIGHBOR's evidence, e.g. "C3 and
        # C4 both coupled to Cz" implying something about C3-C4 that neither
        # edge's own events encode) has no way to receive that information.
        # n_hops=1 (default) is exactly the pre-existing behavior -- forward()
        # never reaches _propagate_hops, so this is purely additive. n_hops=K>1
        # runs (K-1) additional rounds of message passing over the SAME
        # canonical src_idx/dst_idx topology real events use, propagating
        # BOTH directions of each edge (an undirected topology, same as
        # _build_sparse_events' gate -- see that method's 2026-08-09 note),
        # each round combining a node's incoming neighbor evidence with its
        # own prior-hop state via a GRUCell update (Gilmer et al. 2017
        # MPNN-style gated update -- lets the network learn how much of a
        # hop's neighbor evidence to fold in rather than overwriting
        # outright). See _propagate_hops.
        n_hops: int = 1,
        # 2026-08-10: _propagate_hops (above) mixes each node's ALREADY-
        # frequency-blended evidence vector -- _aggregate_events collapses
        # every event landing on a channel, any frequency, into one
        # hidden_dim vector before any hop ever runs (softmax/mean pooled
        # over events) -- so it cannot represent "channel A received
        # evidence AT FREQUENCY X, and A's outgoing hop message is
        # conditioned specifically on that freq-X evidence" (a directed,
        # frequency-matched chain, e.g. an event routing evidence INTO A at
        # freq X, A then routing evidence at that SAME freq X back OUT
        # toward a neighbor next hop): by the time hops run, A's one hidden
        # vector has no per-frequency identity left for that mechanism to
        # condition on.
        #
        # freq_aware_hops=True (only takes effect when n_hops>1 -- n_hops=1
        # never reaches any hop code either way) swaps the hop stage to
        # _propagate_hops_freq_aware instead of _propagate_hops: node state
        # keeps a separate hidden_dim slot PER FREQUENCY
        # ([B, n_channels, nfreqs, hidden_dim], via
        # _aggregate_events_freq_indexed) all the way through the hop
        # rounds, weight-tied across frequency, then pools back down to
        # [B, n_channels, hidden_dim] (self.freq_pool) so sparse_classifier's
        # input width is unaffected. This makes same-frequency chaining
        # structurally representable -- NOT guaranteed, still a learned
        # weighting rather than hardcoded chain-detection logic; see
        # _propagate_hops_freq_aware's docstring.
        #
        # False (default) never touches this pathway -- _propagate_hops (the
        # pre-existing, freq-blind implementation) is completely unchanged
        # and bit-identical, including when freq_aware_hops=True but
        # n_hops=1 (both are then no-ops, matching n_hops's own "1 is the
        # original behavior" contract).
        freq_aware_hops: bool = False,
        # 2026-08-10: event_mode="sparse" (default) is the entire pre-
        # existing pipeline above, bit-identical -- this and every
        # dense_conv_* param below are inert no-ops at the default. "dense"
        # replaces _build_sparse_events' hard threshold-and-consolidate step
        # with a LEARNED conv stack (dense_edge_conv, see
        # _build_dense_feature_conv) run over the same post-COI-mask
        # coherence/phase arrays _coherence_only already produces, PLUS a
        # continuous significance channel ((coh - surrogate_threshold) /
        # surrogate_threshold) in place of the hard surrogate gate -- see
        # _build_dense_edge_input/compute_dense_edge_input. The conv's
        # output is one feature vector PER EDGE (always present, unlike a
        # variable-count sparse event list), which forward() then feeds into
        # the exact same downstream graph machinery real events use
        # (sparse_message_mlp, _aggregate_events, _propagate_hops/
        # _propagate_hops_freq_aware) by constructing an "every edge is
        # always valid" events_padded/src_padded/dst_padded/valid_mask in
        # dense's own shape -- so this only replaces event BUILDING, not
        # anything downstream of it. Unlike sparse events, dense_edge_conv is
        # trainable and therefore cannot be precomputed once per trial the
        # way _build_sparse_events is (see SparseEvidenceGNNClassifier.
        # _precompute_dense_edge_inputs's docstring) -- only the (still
        # non-trainable) cross-spectrum/smoothing/COI/significance-channel
        # math is precomputed once; dense_edge_conv itself runs every
        # forward() call, like channel_encoder already does for raw_x.
        #
        # 2026-08-11: "temporal_graph" is a third option, distinct from both
        # "dense" (pools the whole T' axis away via dense_edge_conv's own
        # AdaptiveAvgPool2d before the graph ever sees a single vector per
        # edge) and n_hops>1 (adds DEPTH -- extra rounds of message passing
        # ACROSS THE GRAPH within one already-pooled snapshot, never touching
        # time). Neither is a genuine test of "evolving graph" propagation --
        # a mechanism that actually walks forward through time, updating
        # each node's state as new evidence arrives, the way an RNN over a
        # graph sequence would. "temporal_graph" reuses the exact same
        # precomputed, non-trainable _build_dense_edge_input stack "dense"
        # does (so dense_edge_time_downsample, coi_enabled, smoothing, and
        # the surrogate/fixed threshold math are all shared, unchanged
        # infrastructure -- see that param's own docstring), but processes
        # it STEP BY STEP through time instead of pooling/convolving over
        # the whole T' axis at once:
        #   1. temporal_edge_proj (a single small Linear+GELU, NOT the full
        #      Conv2d stack dense_edge_conv uses) folds frequency into
        #      features and produces a per-edge embedding at EVERY
        #      timestep, vectorized over (edge, time) in one call -- cheap,
        #      since it runs once per timestep rather than through a deep
        #      conv stack (see temporal_graph_edge_dim's docstring).
        #   2. Those per-timestep per-edge embeddings feed sparse_message_mlp
        #      (the SAME weights every other event_mode uses) and are
        #      aggregated to per-node evidence via the existing "mean"
        #      aggregation -- deliberately reusing that mechanism rather than
        #      "concat" (see event_aggregation's own docstring: concat is
        #      already a harder, currently-worse optimization target on its
        #      own; stacking it here would confound the temporal question
        #      with the aggregation question). event_mode="temporal_graph"
        #      therefore REQUIRES event_aggregation="mean" -- rejected
        #      explicitly otherwise, same "explicit no-op rejection"
        #      precedent as every other incompatible combination in this
        #      constructor.
        #   3. The resulting per-node sequence (one aggregated hidden_dim
        #      vector per node per surviving timestep) is fed into a single
        #      nn.GRU, weight-shared across nodes (nodes folded into the
        #      batch dim, same "shared weights across the axis that isn't
        #      time" spirit dense_edge_conv's own edge-axis sharing and
        #      _DenseEdgeGRUTemporal's edge-axis sharing both already use --
        #      see that class's docstring) -- a PERSISTENT hidden state
        #      genuinely updated timestep by timestep, unlike n_hops>1's
        #      GRUCell rounds (which never see more than one already-pooled
        #      snapshot; there is no "next timestep" for them to walk
        #      toward).
        #   4. The GRU's final hidden state per node IS `evidence`, in
        #      EXACTLY the shape _aggregate_events' "mean" branch already
        #      returns ([B, n_channels, hidden_dim]) -- so n_hops>1
        #      propagation (if also requested, ACROSS the graph, orthogonal
        #      to this mode's ACROSS TIME propagation) and sparse_classifier
        #      are the exact same shared code every other event_mode uses,
        #      not a duplicated path. See _temporal_graph_node_states.
        #
        # freq_aware_hops=True is rejected together with event_mode=
        # "temporal_graph", same reasoning/precedent as event_mode="dense":
        # temporal_edge_proj folds the whole frequency axis into its input
        # exactly like dense_edge_conv's first layer does, so a "temporal
        # graph event" has no discrete per-event frequency bin left to index
        # freq_aware_hops's per-frequency node state by.
        event_mode: Literal["sparse", "dense", "temporal_graph"] = "sparse",
        dense_conv_kernel_size: int = 5,
        dense_conv_pool_size: int = 4,
        dense_conv_intermediate_channels: int = 32,
        dense_conv_intermediate_channels_reduced: int | None = None,
        # Per-edge feature width dense_edge_conv outputs -- plays the same
        # role sparse events' fixed 5-wide [timestamp, freq, mag, sinφ,
        # cosφ] feature does when building sparse_message_mlp's input width
        # (message_in), but is a free constructor knob here since there's no
        # fixed discrete-event schema to match.
        dense_conv_out_channels: int = 8,
        # 2026-08-10: event_mode="dense"-only. dense_edge_conv's own final
        # layer (nn.AdaptiveAvgPool2d((None, 1)) in _build_dense_feature_conv)
        # average-pools EVERY surviving timestep down to one value per
        # (edge, out_channel) regardless of how many timesteps it's given --
        # and its two internal MaxPool2d(pool_size) stages already collapse
        # T~1001 down to ~T/16 before that final pool ever runs. The conv is
        # already discarding almost all of the time axis's resolution every
        # forward call; it's just doing so the expensive way, inside two real
        # Conv2d layers that have to process the full native T first (the
        # dominant per-epoch cost in event_mode="dense" -- see
        # run_pipelines.py's DENSE_EDGE_PARAMS).
        #
        # dense_edge_time_downsample=1 (default) changes nothing -- bit-
        # identical to before this param existed. A value k>1 average-pools
        # _build_dense_edge_input's [B, 4, E, T, F] output along T by that
        # factor (kernel=stride=k, trailing remainder samples dropped -- same
        # convention _build_dense_feature_conv's own MaxPool2d already uses)
        # BEFORE dense_edge_conv ever sees it, once per trial at precompute
        # time (SparseEvidenceGNNClassifier._precompute_dense_edge_inputs),
        # not every epoch -- moving (most of) the conv's own unavoidable time-
        # axis compression earlier and non-trainable, cutting both its
        # per-epoch FLOPs and dense_edge_raw's cached memory footprint
        # roughly linearly in k.
        #
        # Applied AFTER smoothing (smooth_kernel_size[0], a real low-pass
        # filter in time) and AFTER the COI mask, at native resolution --
        # NOT before, unlike cwt_resample_n_time (which resamples the raw
        # complex CWT coefficients pre-coherence and explicitly breaks the
        # COI mask's own timing assumptions; see that param's docstring).
        # Decimating an already-smoothed, already-COI-masked signal is the
        # textbook-correct order (filter, then downsample) rather than the
        # reverse. This does let one pooling window blend a few native COI-
        # valid/invalid timesteps together near the mask's edge -- but
        # dense_edge_conv's own first-layer kernel (kernel_size=(1,
        # dense_conv_kernel_size), unmasked) already blends across that same
        # boundary at native resolution today, so this isn't a new category
        # of approximation, just a coarser instance of one the architecture
        # already tolerates. Keep k modest relative to smooth_kernel_size[0]
        # (the actual anti-aliasing filter width) -- a large k decimates
        # faster than that filter's own bandwidth justifies and starts
        # reintroducing the same aliasing risk cwt_resample_n_time has, just
        # to a lesser degree (smoothed coherence, not a raw oscillating
        # carrier).
        dense_edge_time_downsample: int = 1,
        # 2026-08-11: the most extreme point on dense_edge_time_downsample's
        # own spectrum -- instead of pooling T down by a fixed integer
        # factor, collapse it to exactly 1 by averaging over EVERY COI-valid
        # timestep. This is the literal "single time-averaged coherence
        # graph per trial" ablation: a static, per-(edge, frequency)
        # functional-connectivity-style summary (mean coherence, mean
        # sinφ/cosφ, mean significance) with no within-trial temporal
        # structure left at all, replacing event BUILDING the same way
        # event_mode="dense" itself replaces _build_sparse_events' hard
        # threshold-and-consolidate step -- everything downstream
        # (dense_edge_conv, sparse_message_mlp, _aggregate_events,
        # _propagate_hops, sparse_classifier) is unchanged, so a score
        # difference from event_mode="dense" alone is attributable to
        # discarding time resolution, not to a different readout.
        #
        # Uses _coi_valid_mask's own per-(time, frequency) validity counts as
        # averaging weights (sum of valid cells / count of valid cells),
        # rather than a plain torch.mean over the raw T axis -- a plain mean
        # would systematically bias every average toward zero by however
        # much of the trial the COI happens to exclude at that frequency
        # (much more at low frequencies, which have wide cones), which isn't
        # "time-averaged coherence," it's "time-averaged coherence diluted by
        # an unrelated per-frequency constant." See _build_dense_edge_input.
        #
        # event_mode="dense" only (same requirement as
        # dense_edge_time_downsample, for the same reason: this only affects
        # _build_dense_edge_input, which sparse mode never calls). Mutually
        # exclusive with dense_edge_time_downsample != 1 -- averaging the
        # whole axis makes a partial-factor pool upstream of it meaningless,
        # rejected explicitly rather than silently ignored. Also requires
        # dense_conv_kernel_size == dense_conv_pool_size == 1: once T is 1,
        # dense_edge_conv's own kernel_size=(1, k>1) convolutions have
        # nothing left to convolve over (a k>1 Conv2d on a width-1 input
        # simply errors) -- rejected at construction time with a clear
        # message rather than as an opaque shape-mismatch deep inside
        # forward().
        time_averaged_graph: bool = False,
        # 2026-08-11, event_mode="dense" only. "conv" (default) is
        # dense_edge_conv's original Conv2d temporal stack
        # (_build_dense_feature_conv), bit-identical to before this param
        # existed. "rnn" swaps it for a GRU (_DenseEdgeGRUTemporal) that
        # integrates the FULL T' sequence with memory instead of Conv2d's
        # small fixed local window (dense_conv_kernel_size) -- see that
        # class's docstring for the full rationale and exactly what shape
        # contract it preserves. Purely additive: dense_edge_conv is built
        # from whichever path this selects, and everything else
        # (_dense_edge_features' squeeze/permute, sparse_message_mlp,
        # aggregation, hops, sparse_classifier) is unchanged either way.
        dense_edge_temporal_mode: Literal["conv", "rnn"] = "conv",
        # 2026-08-11, dense_edge_temporal_mode="rnn" only -- the negative
        # control for that mode: same architecture, same parameter count,
        # scrambled time order. When True, _dense_edge_features
        # independently permutes each (batch, edge, frequency)'s T' index
        # order (same permutation shared across the 4 coh/sinφ/cosφ/
        # significance stack channels, so a shuffled timestep's own values
        # stay physically consistent with each other -- only their ORDER
        # relative to other timesteps is scrambled) before dense_edge_conv
        # ever sees it. If the "rnn" path's accuracy depends on seeing
        # timesteps in their real order, shuffling should hurt it relative
        # to ordered "rnn"; if shuffling doesn't hurt, that's evidence the
        # GRU isn't actually exploiting temporal structure either. False
        # (default) changes nothing. Rejected (not silently ignored) when
        # True together with dense_edge_temporal_mode="conv" -- same
        # "explicit no-op rejection" precedent dense_edge_time_downsample/
        # time_averaged_graph/freq_aware_hops already use in this
        # constructor, since Conv2d's own receptive field has no notion of
        # "the whole sequence's order" to be sensitive to in the first
        # place.
        shuffle_time_order: bool = False,
        # 2026-08-11, event_mode="temporal_graph" only -- see that mode's
        # own docstring above. Per-edge, per-timestep embedding width
        # temporal_edge_proj outputs, playing the same "event_feature_dim"
        # role dense_conv_out_channels plays for event_mode="dense" (see
        # message_in's computation below) -- but unlike dense_conv_out_
        # channels, this is NOT the output of a deep Conv2d stack, just one
        # small Linear+GELU applied at every timestep (see event_mode's
        # docstring for why this stays cheap: temporal_edge_proj runs once
        # per timestep, so a heavy per-timestep layer would multiply cost by
        # T' in a way a single whole-trial conv never has to pay). Default
        # (8) matches dense_conv_out_channels's own default purely so the
        # two modes' sparse_message_mlp/sparse_classifier stay comparably
        # sized, not because it's been tuned.
        temporal_graph_edge_dim: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        if smooth_kernel_size[0] is not None and smooth_kernel_size[0] <= 0:
            raise ValueError("smooth_kernel_size[0] must be > 0")
        if smooth_kernel_size[1] is None or smooth_kernel_size[1] <= 0:
            raise ValueError("smooth_kernel_size[1] must be > 0 and not None")
        if smooth_kernel_sigma[0] is not None and smooth_kernel_sigma[0] <= 0.0:
            raise ValueError("smooth_kernel_sigma[0] must be > 0.0 or None")
        if smooth_kernel_sigma[1] is not None and smooth_kernel_sigma[1] <= 0.0:
            raise ValueError("smooth_kernel_sigma[1] must be > 0.0 or None")
        if event_aggregation not in ("mean", "gated_softmax", "concat"):
            raise ValueError(
                "event_aggregation must be 'mean', 'gated_softmax', or 'concat', got "
                f"{event_aggregation!r}."
            )
        if int(n_hops) < 1:
            raise ValueError(f"n_hops must be >= 1, got {n_hops!r}.")
        if event_mode not in ("sparse", "dense", "temporal_graph"):
            raise ValueError(
                "event_mode must be 'sparse', 'dense', or 'temporal_graph', got "
                f"{event_mode!r}."
            )
        if event_mode == "temporal_graph" and event_aggregation != "mean":
            # See event_mode's own docstring above -- "temporal_graph"
            # deliberately reuses ONLY the existing "mean" aggregation so
            # this experiment stays isolated to the temporal question,
            # rather than also compounding it with the (separately still
            # unresolved) aggregation question "concat"/"gated_softmax"
            # raise. Rejected explicitly, same "explicit no-op rejection"
            # precedent as every other incompatible combination below.
            raise ValueError(
                "event_mode='temporal_graph' requires event_aggregation='mean' "
                f"-- got event_aggregation={event_aggregation!r}. See event_mode's "
                "docstring above for why this experiment is deliberately scoped "
                "to the existing mean aggregation."
            )
        if event_aggregation == "concat" and event_mode != "dense":
            # "concat"'s per-node readout (_aggregate_events' "concat" branch)
            # gathers each destination channel's incident messages by FIXED
            # POSITION in msg's own edge axis (concat_slot_idx, built below
            # from the canonical dst_idx order) -- valid only because
            # event_mode="dense" guarantees msg's edge axis IS the canonical
            # edge list, every trial, in that exact order (_dense_edge_features
            # sets dst_padded = self.dst_idx broadcast, unconditionally).
            # event_mode="sparse" has no such guarantee: dst_padded there is a
            # variable-length, per-trial list of actual event positions (a
            # channel can have 0, 1, or several events land on it, packed in
            # arbitrary order) -- concat_slot_idx's fixed mapping would
            # silently gather the WRONG edges' messages into the wrong slots
            # if applied there. Rejecting explicitly rather than risking that.
            raise ValueError(
                "event_aggregation='concat' requires event_mode='dense' -- see "
                "_aggregate_events' 'concat' branch docstring for why sparse "
                "mode's variable-length, per-trial event lists can't use a "
                "fixed per-node slot mapping."
            )
        if event_aggregation == "concat" and int(n_hops) != 1:
            # _propagate_hops (n_hops>1) operates on a SINGLE hidden_dim
            # vector per node ([B, n_channels, hidden_dim]); concat's own
            # per-node representation is [B, n_channels, max_degree,
            # hidden_dim] -- every incident edge's message kept distinct,
            # nothing collapsed to one vector for a hop update to consume.
            # Combining the two would require either flattening concat's
            # representation back down (defeating the point of not
            # collapsing it) or redesigning _propagate_hops for a 4D node
            # state, neither of which this option does. Untested combination,
            # rejected explicitly rather than silently reinterpreted.
            raise ValueError(
                "event_aggregation='concat' requires n_hops=1 -- multi-hop "
                "propagation (_propagate_hops) assumes one hidden_dim vector "
                "per node, which concat's own per-node representation "
                "(every incident edge kept distinct) does not produce."
            )
        if event_mode in ("dense", "temporal_graph") and freq_aware_hops:
            # freq_aware_hops's whole mechanism (_aggregate_events_freq_indexed
            # / _propagate_hops_freq_aware) keys evidence by the discrete CWT
            # frequency BIN each sparse event's run landed on (f_of_run --
            # see _build_sparse_events). Neither "dense" nor "temporal_graph"
            # has such per-event frequency identity: dense_edge_conv (and
            # temporal_graph's own temporal_edge_proj, the same fold applied
            # per timestep instead of once) folds the whole frequency axis
            # into its input channels and mixes across it, so neither mode's
            # per-edge feature has a single bin to index by. Rejecting this
            # combination explicitly rather than silently misrouting every
            # feature into (arbitrary) freq slot 0.
            raise ValueError(
                f"freq_aware_hops=True has no meaning when event_mode={event_mode!r} -- "
                "dense/temporal_graph features have no discrete per-event "
                "frequency bin to index by (see event_mode's docstring above)."
            )
        if int(dense_edge_time_downsample) < 1:
            raise ValueError(
                "dense_edge_time_downsample must be >= 1, got "
                f"{dense_edge_time_downsample!r}."
            )
        if dense_edge_time_downsample != 1 and event_mode not in ("dense", "temporal_graph"):
            # Same "explicit no-op rejection" precedent as freq_aware_hops
            # just above -- _build_dense_edge_input (the only consumer of
            # this param) is never called in event_mode="sparse", so a
            # non-default value here would silently do nothing rather than
            # silently do the wrong thing; still rejected explicitly so it
            # can't be mistaken for a value that's actually taking effect.
            # "temporal_graph" DOES consume it (see that mode's own
            # docstring -- it reuses _build_dense_edge_input's output
            # unchanged), unlike the other dense-only knobs below.
            raise ValueError(
                "dense_edge_time_downsample != 1 has no meaning when "
                "event_mode='sparse' -- it only affects "
                "_build_dense_edge_input, which event_mode='sparse' never "
                "calls (see dense_edge_time_downsample's docstring above)."
            )
        if time_averaged_graph and event_mode != "dense":
            raise ValueError(
                "time_averaged_graph=True has no meaning when "
                "event_mode='sparse' -- it only affects _build_dense_edge_input, "
                "which event_mode='sparse' never calls (see "
                "time_averaged_graph's docstring above)."
            )
        if time_averaged_graph and dense_edge_time_downsample != 1:
            raise ValueError(
                "time_averaged_graph=True and dense_edge_time_downsample != 1 "
                "are mutually exclusive -- time_averaged_graph already collapses "
                "the whole time axis to 1, making a partial-factor pool "
                "upstream of it meaningless. Leave dense_edge_time_downsample=1 "
                "(its default) when using time_averaged_graph."
            )
        if time_averaged_graph and (dense_conv_kernel_size != 1 or dense_conv_pool_size != 1):
            raise ValueError(
                "time_averaged_graph=True requires dense_conv_kernel_size=1 and "
                "dense_conv_pool_size=1 -- once the time axis is collapsed to 1, "
                "dense_edge_conv's kernel_size=(1, k>1) convolutions/pools have "
                "nothing left to convolve/pool over. Got dense_conv_kernel_size="
                f"{dense_conv_kernel_size!r}, dense_conv_pool_size="
                f"{dense_conv_pool_size!r}."
            )
        if dense_edge_temporal_mode not in ("conv", "rnn"):
            raise ValueError(
                "dense_edge_temporal_mode must be 'conv' or 'rnn', got "
                f"{dense_edge_temporal_mode!r}."
            )
        if dense_edge_temporal_mode == "rnn" and event_mode != "dense":
            # Same "explicit no-op rejection" precedent as
            # dense_edge_time_downsample/time_averaged_graph above --
            # dense_edge_temporal_mode only affects how dense_edge_conv
            # itself is built, which only exists when event_mode="dense".
            raise ValueError(
                "dense_edge_temporal_mode='rnn' has no meaning when "
                "event_mode='sparse' -- dense_edge_conv (the only consumer "
                "of this param) is never built in event_mode='sparse' (see "
                "dense_edge_temporal_mode's docstring above)."
            )
        if shuffle_time_order and dense_edge_temporal_mode != "rnn":
            raise ValueError(
                "shuffle_time_order=True has no meaning when "
                "dense_edge_temporal_mode='conv' -- it only affects "
                "_dense_edge_features' input to the 'rnn' path (see "
                "shuffle_time_order's docstring above)."
            )

        self.n_channels = n_channels
        self.nfreqs = nfreqs
        self.hidden_dim = hidden_dim
        self.coherence_threshold = float(coherence_threshold)
        self.phase_threshold_rad = math.radians(phase_threshold_deg)
        # Fixed, not exposed as a constructor param -- this pipeline never
        # tuned it (matches WCTEvidenceGNNCore's own default, "reflect",
        # which SparseEvidenceGNNCore never overrode before this reorg).
        self.padding_mode = "reflect"
        self.smooth_kernel_sigma = smooth_kernel_sigma
        # smooth_kernel_size[0]=None used to mean "default to window_size"
        # in the windowed pipeline this borrowed from; this pipeline has no
        # window_size, so None falls back to 25 purely for parity with that
        # prior default -- every caller in this codebase always passes an
        # explicit (5, 3) anyway.
        self.smooth_kernel_size = (
            25 if smooth_kernel_size[0] is None else smooth_kernel_size[0],
            smooth_kernel_size[1],
        )
        self._summary_context: dict[str, object] | None = None

        # Edges are the canonical (i<j) undirected pairing (2026-08-09) --
        # one edge per channel pair, not two directed copies. See the module
        # docstring's 2026-08-09 note: xwt_(j->i) = conj(xwt_(i->j)) exactly
        # (verified bit-exact on real data), so a directed i->j/j->i scheme
        # would compute and carry every cross-spectrum/coherence/phase/
        # threshold/gate value twice for no extra information; the
        # direction bit that scheme encoded via "which of the two edges
        # fired" now lives instead in the SIGN of the per-event phase angle
        # (see _build_sparse_events's two-sided gate).
        pairs = upper_pair_indices(n_channels)
        src_idx = torch.tensor([i for i, _ in pairs], dtype=torch.long)
        dst_idx = torch.tensor([j for _, j in pairs], dtype=torch.long)
        self.register_buffer("src_idx", src_idx, persistent=False)
        self.register_buffer("dst_idx", dst_idx, persistent=False)
        self.register_buffer(
            "edge_pair_idx", torch.cat([src_idx, dst_idx]), persistent=False
        )
        # Bidirectional edge index for _propagate_hops -- src_idx/dst_idx
        # above is the canonical (i<j) UNDIRECTED topology (one edge per
        # pair), but multi-hop propagation must let evidence flow both
        # i->j and j->i, so hop_src_idx/hop_dst_idx are the 2*E directed
        # copies of it (concatenation, not a new pairing) purely for the
        # scatter_add in _propagate_hops -- unrelated to (and does not
        # reintroduce) the 2026-08-09 duplicate-edge coherence-gate issue
        # described above, since no gate/threshold math touches these.
        self.register_buffer(
            "hop_src_idx", torch.cat([src_idx, dst_idx]), persistent=False
        )
        self.register_buffer(
            "hop_dst_idx", torch.cat([dst_idx, src_idx]), persistent=False
        )
        # event_aggregation="concat"-only topology: concat_slot_idx[j, k] is
        # the position (in msg's own edge axis, i.e. into dst_idx/src_idx
        # above) of destination channel j's k-th incident edge, or
        # len(dst_idx) (a dedicated pad row _aggregate_events' "concat"
        # branch appends to msg) for a slot beyond that channel's actual
        # in-degree. Every channel's in-degree under this canonical (i<j)
        # dst-only topology is exactly its own channel index (channel 0 is
        # never a destination -- see _aggregate_events' "concat" docstring),
        # so degrees range 0..n_channels-1 and max_degree=n_channels-1 is
        # always enough slots for every channel, with no possible overflow.
        # Built unconditionally (like hop_src_idx/hop_dst_idx above) --
        # cheap, integer-only, no RNG draw, so building it regardless of
        # event_aggregation cannot shift any OTHER submodule's random init
        # (see event_gate's own comment below for why that class of bug is
        # taken seriously here).
        self.concat_max_degree = n_channels - 1
        concat_slot_idx = torch.full(
            (n_channels, self.concat_max_degree), len(dst_idx), dtype=torch.long
        )
        _concat_slot_counts = [0] * n_channels
        for _edge_pos, _j in enumerate(dst_idx.tolist()):
            concat_slot_idx[_j, _concat_slot_counts[_j]] = _edge_pos
            _concat_slot_counts[_j] += 1
        self.register_buffer("concat_slot_idx", concat_slot_idx, persistent=False)
        # event_mode="temporal_graph"-only: per-channel in-degree under this
        # same canonical (i<j) dst-only topology, used as the "mean"
        # aggregation's divisor at EVERY timestep (see
        # _temporal_graph_node_states). Fixed and precomputable because --
        # unlike sparse events, whose per-trial active-event count varies --
        # every canonical edge is always "active" every timestep in this
        # mode (same "every edge always fires" property event_mode="dense"
        # already has, see _dense_edge_features' valid_mask). clamp_min(1)
        # only guards channel 0 (never a destination -- see concat_slot_idx's
        # own comment above), whose in-degree is 0; that channel's
        # aggregated embedding is simply always zero at every timestep
        # (matches _aggregate_events' "mean" branch's own convention for a
        # channel with zero incident events). Built unconditionally, cheap,
        # integer-only, no RNG draw -- same "cannot shift any OTHER
        # submodule's random init" precedent as concat_slot_idx/hop_src_idx.
        temporal_node_in_degree = torch.bincount(
            dst_idx, minlength=n_channels
        ).clamp_min(1).to(torch.float32)
        self.register_buffer(
            "temporal_node_in_degree", temporal_node_in_degree, persistent=False
        )
        self.channel_embed_dim = channel_embed_dim
        self.channel_encoder = ChannelSignalEncoder(
            channel_embed_dim, dilation=channel_encoder_dilation
        )
        self.event_mode = event_mode
        self.dense_conv_out_channels = dense_conv_out_channels
        self.dense_edge_time_downsample = int(dense_edge_time_downsample)
        self.time_averaged_graph = bool(time_averaged_graph)
        self.temporal_graph_edge_dim = temporal_graph_edge_dim
        # event_mode="sparse": message_in unchanged (5 fixed event scalars +
        # src/dst embeds). "dense": dense_edge_conv's own out_channels
        # replaces the "5". "temporal_graph": temporal_edge_proj's own
        # output width plays the same role, one level cheaper -- see
        # event_mode's docstring above.
        if event_mode == "sparse":
            event_feature_dim = 5
        elif event_mode == "dense":
            event_feature_dim = dense_conv_out_channels
        else:  # "temporal_graph"
            event_feature_dim = temporal_graph_edge_dim
        message_in = event_feature_dim + 2 * channel_embed_dim
        self.sparse_message_mlp = nn.Sequential(
            nn.Linear(message_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.event_aggregation = event_aggregation
        # Tiny separate head (message_in -> 1), not folded into
        # sparse_message_mlp's own output -- keeps the message content and
        # the "how much should this event count" decision independently
        # learnable rather than forcing one shared trunk to represent both.
        #
        # 2026-08-10: built ONLY when actually used (event_aggregation=
        # "gated_softmax"), not unconditionally -- an earlier version always
        # constructed this here regardless of event_aggregation, which is a
        # real bug, not a cosmetic one: nn.Linear draws its random initial
        # weights from the global RNG at construction time, so unconditionally
        # inserting this layer BEFORE sparse_classifier silently shifted
        # sparse_classifier's own initial weights for every "mean"-mode model
        # too, even though event_gate itself is never touched by "mean"
        # mode's forward pass (confirmed directly: replaying the same
        # construction order with vs. without this line, same seed, gives
        # sparse_classifier different initial weights). On a pipeline this
        # seed-sensitive (see [[sparse-evidence-gnn-seed-variance]]), a
        # silently-shifted initialization is a real way to move results, not
        # noise -- this conditional restores "mean" mode's initialization to
        # true pre-event_aggregation-change behavior.
        self.event_gate = (
            nn.Linear(message_in, 1) if event_aggregation == "gated_softmax" else None
        )
        # "concat" keeps every incident edge's own hidden_dim-wide message
        # distinct instead of collapsing a channel's incident edges into one
        # vector (see _aggregate_events' "concat" branch) -- its readout is
        # therefore concat_max_degree times wider than "mean"/"gated_softmax"'s
        # n_channels*hidden_dim. This is the only place event_aggregation
        # changes a submodule's SHAPE (not just its forward-pass behavior),
        # so "mean"/"gated_softmax" get the exact same sparse_classifier
        # width/init they always have -- only "concat" (a brand new mode with
        # no prior behavior to preserve) differs.
        classifier_in_dim = (
            n_channels * self.concat_max_degree * hidden_dim
            if event_aggregation == "concat"
            else n_channels * hidden_dim
        )
        self.sparse_classifier = nn.Linear(classifier_in_dim, n_classes)
        self.n_hops = int(n_hops)
        # Constructed AFTER every module above (channel_encoder,
        # sparse_message_mlp, event_gate, sparse_classifier) so their own
        # random initialization draws the exact same values as before this
        # feature existed, regardless of n_hops -- matches event_gate's own
        # "constructed but unused at the default setting" precedent (see its
        # comment above). Unused (but still built -- negligible parameter
        # cost) when n_hops=1; see _propagate_hops and forward().
        self.hop_message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.hop_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.freq_aware_hops = bool(freq_aware_hops)
        # freq_aware_hops's own submodules (see __init__ docstring above /
        # _propagate_hops_freq_aware) -- separate weights from
        # hop_message_mlp/hop_update, not shared: freq-indexed states carry
        # a channel's SINGLE-frequency evidence rather than its
        # all-frequencies blend, a different enough distribution that tying
        # them to the same weights would force one set of weights to serve
        # both.
        #
        # Constructed here, AFTER every pre-existing submodule (channel_
        # encoder, sparse_message_mlp, event_gate, sparse_classifier,
        # hop_message_mlp, hop_update), unconditionally and regardless of
        # freq_aware_hops's own value -- same reasoning as hop_message_mlp/
        # hop_update's own comment above, and the exact bug this mirrors:
        # see [[sparse-evidence-gnn-event-gate-init-shift-bug]] in project
        # memory, where a new submodule built BEFORE an existing one
        # silently shifted that existing one's random initial weights via
        # RNG-draw order, even in configurations that never used the new
        # submodule. Building these last, always, keeps every pre-existing
        # submodule's initialization bit-identical across every
        # freq_aware_hops/n_hops combination. Negligible parameter cost
        # when unused (freq_aware_hops=False or n_hops=1).
        self.hop_message_mlp_freq = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.hop_update_freq = nn.GRUCell(hidden_dim, hidden_dim)
        # Learned softmax-attention pool collapsing the frequency axis back
        # to [B, n_channels, hidden_dim] at the end of
        # _propagate_hops_freq_aware, so sparse_classifier's input width
        # (n_channels * hidden_dim) -- and therefore its own parameter count
        # and initialization -- is identical whether freq_aware_hops is on
        # or off.
        self.freq_pool = nn.Linear(hidden_dim, 1)
        self._freq_lo = None  # set on first forward() call from observed freqs
        self._freq_hi = None
        self.sampling_rate = sampling_rate  # needed for the COI mask below
        self.scale_adaptive_smoothing = scale_adaptive_smoothing
        self.scale_adaptive_cycles = scale_adaptive_cycles
        self.scale_adaptive_max_kernel = scale_adaptive_max_kernel
        # Diagnostic toggle: COI support scales as 1/freq, so at this
        # pipeline's trial length it disproportionately crops the low-freq
        # (mu-band, most discriminative for motor imagery) end of the
        # spectrum -- up to ~37% of the trial at 8Hz vs ~8% at 35Hz. Measured
        # to shift the surviving events' frequency mix away from mu-band
        # (mu share 20.6%->17.6% of events, COI off->on, at native res).
        # Kept as a real constructor arg (not a monkeypatch) so it's a
        # legitimate, re-runnable pipeline configuration, not just a
        # debugging hack.
        self.coi_enabled = coi_enabled
        # 2026-08-09 ablation: isolates which of the two feature sources
        # concatenated into sparse_message_mlp's input (event_features:
        # [t, freq, mag, sinphi, cosphi] per consolidated burst, vs.
        # src/dst channel_encoder embeddings, present on EVERY event
        # regardless of that event's own gate-worthiness) is actually
        # driving classification -- motivated by dropping both gate
        # thresholds to ~0 (near-dense events) costing only ~2-3 accuracy
        # points in an initial single-seed check, which would be
        # surprising if the surrogate-calibrated event content itself were
        # doing most of the discriminating. Does NOT change which events
        # get built (compute_events/_build_sparse_events/the gate are
        # untouched) -- only zeros one feature block immediately before the
        # message MLP, in forward() below, so it's a pure ablation of what
        # the CLASSIFIER sees, not of the event-detection pipeline itself.
        #   "zero_channel_embed" -- (default, and the ONLY accepted value)
        #       src/dst ChannelSignalEncoder embeddings zeroed; message MLP
        #       sees only each event's own (t, freq, mag, phase) -- tests
        #       whether accuracy survives on event content + graph topology
        #       alone, with no raw-signal information.
        #
        # 2026-08-17: "none" and "zero_event_features" (which fed raw-signal
        # channel embeddings to the classifier) are hard-disabled -- this
        # kept getting switched on unintentionally (default was "none"), so
        # rather than rely on every call site remembering to opt out, the
        # capability itself is removed: no value other than
        # "zero_channel_embed" is accepted, at construction time, regardless
        # of what any pipeline config passes.
        if feature_ablation != "zero_channel_embed":
            raise ValueError(
                "feature_ablation must be 'zero_channel_embed' -- channel "
                "embeddings are hard-disabled (see 2026-08-17 comment above), "
                f"got {feature_ablation!r}."
            )
        self.feature_ablation = feature_ablation

        # dense_edge_conv: built LAST, and only when event_mode="dense" --
        # same "constructed after every pre-existing submodule, only when
        # actually used" precedent as event_gate/hop_message_mlp_freq above
        # (see [[sparse-evidence-gnn-event-gate-init-shift-bug]]). At
        # event_mode="sparse" (default) this attribute is simply never
        # created, so every pre-existing submodule's random init is
        # bit-identical to before event_mode existed.
        self.dense_conv_kernel_size = dense_conv_kernel_size
        self.dense_conv_pool_size = dense_conv_pool_size
        self.dense_conv_intermediate_channels = dense_conv_intermediate_channels
        self.dense_conv_intermediate_channels_reduced = dense_conv_intermediate_channels_reduced
        self.dense_edge_temporal_mode = dense_edge_temporal_mode
        self.shuffle_time_order = bool(shuffle_time_order)
        if event_mode == "dense" and dense_edge_temporal_mode == "rnn":
            # See _DenseEdgeGRUTemporal's own docstring -- same in_channels
            # (frequency folded in) / out_channels contract as the Conv2d
            # path below, built LAST for the same init-order reason (see
            # this block's own docstring just above).
            self.dense_edge_conv = _DenseEdgeGRUTemporal(
                in_channels=4 * nfreqs,
                out_channels=dense_conv_out_channels,
            )
        elif event_mode == "dense":
            # Frequency is folded into in_channels here (see
            # _build_dense_feature_conv's docstring): 4 raw channels (coh,
            # sinφ, cosφ, significance -- see _build_dense_edge_input) times
            # nfreqs, so the conv's first layer can learn cross-frequency
            # combinations while the edge axis stays untouched/weight-shared.
            self.dense_edge_conv = _build_dense_feature_conv(
                in_channels=4 * nfreqs,
                kernel_size=dense_conv_kernel_size,
                intermediate_channels=dense_conv_intermediate_channels,
                out_channels=dense_conv_out_channels,
                pool_size=dense_conv_pool_size,
                intermediate_channels_reduced=dense_conv_intermediate_channels_reduced,
            )
        else:
            self.dense_edge_conv = None

        # temporal_edge_proj/temporal_node_gru: event_mode="temporal_graph"
        # only -- built LAST, after dense_edge_conv above, for the exact
        # same init-order reason (see this block's own docstring). At every
        # other event_mode these attributes are simply never created, so no
        # pre-existing submodule's random init shifts because this feature
        # exists.
        #
        # temporal_edge_proj: the "small per-timestep linear/shallow layer"
        # from event_mode's docstring -- a single Linear+GELU (NOT
        # dense_edge_conv's deep two-block Conv2d stack), folding frequency
        # into features exactly the way dense_edge_conv's own first layer
        # does (4 raw stack channels x nfreqs -> temporal_graph_edge_dim),
        # but applied identically at every timestep rather than convolving
        # across time.
        #
        # temporal_node_gru: nn.GRU (not GRUCell) so the whole T' sequence
        # runs in one call per node rather than a Python-level per-timestep
        # loop -- same efficiency precedent _DenseEdgeGRUTemporal already
        # set for dense_edge_temporal_mode="rnn" (see that class's
        # docstring: recurrent processing can't parallelize across time the
        # way Conv2d can, so avoiding an extra Python loop on top of that
        # matters for wall-clock). input_size=hidden_dim because it consumes
        # sparse_message_mlp's OWN output (msg, already hidden_dim-wide)
        # after per-timestep mean aggregation to nodes -- see
        # _temporal_graph_node_states -- not temporal_edge_proj's raw
        # per-edge output directly.
        if event_mode == "temporal_graph":
            self.temporal_edge_proj = nn.Sequential(
                nn.Linear(4 * nfreqs, temporal_graph_edge_dim), nn.GELU()
            )
            self.temporal_node_gru = nn.GRU(
                input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
            )
        else:
            self.temporal_edge_proj = None
            self.temporal_node_gru = None

    def configure_summary_context(
        self,
        *,
        batch_size: int,
        n_time: int,
        dtype: torch.dtype,
        n_samples: int | None = None,
    ) -> None:
        """Stashes shapes for print_custom_summary, called once from
        SparseEvidenceGNNClassifier._build_model_from_features. Same
        interface as the (removed) WCTEvidenceGNNCore parent's version, so
        that call site didn't need to change."""
        self._summary_context = {
            "batch_size": int(batch_size),
            "n_time": int(n_time),
            "dtype": dtype,
            "n_samples": None if n_samples is None else int(n_samples),
        }

    def print_custom_summary(self, header: str = "Model") -> None:
        """Sparse-pipeline-relevant replacement for the (removed)
        WCTEvidenceGNNCore parent's print_custom_summary, which reported
        windowed-pipeline quantities (window_compute_mode, feature_conv
        shapes) that don't apply to this class."""
        num_edges = int(self.src_idx.numel())
        emit_initial_detail(
            f"[{header}] SparseEvidenceGNN config "
            f"n_channels={self.n_channels} edges={num_edges} nfreqs={self.nfreqs} "
            f"hidden_dim={self.hidden_dim} channel_embed_dim={self.channel_embed_dim} "
            f"coherence_threshold={self.coherence_threshold} "
            f"phase_threshold_rad={self.phase_threshold_rad:.4f} "
            f"smooth_kernel_size={self.smooth_kernel_size} "
            f"scale_adaptive_smoothing={self.scale_adaptive_smoothing} "
            f"coi_enabled={self.coi_enabled} feature_ablation={self.feature_ablation} "
            f"event_aggregation={self.event_aggregation} n_hops={self.n_hops} "
            f"freq_aware_hops={self.freq_aware_hops} event_mode={self.event_mode} "
            f"dense_conv_out_channels={self.dense_conv_out_channels} "
            f"time_averaged_graph={self.time_averaged_graph} "
            f"dense_edge_temporal_mode={self.dense_edge_temporal_mode} "
            f"shuffle_time_order={self.shuffle_time_order} "
            f"temporal_graph_edge_dim={self.temporal_graph_edge_dim} "
            f"dense_edge_time_downsample={self.dense_edge_time_downsample}"
        )
        context = self._summary_context
        if context is None:
            emit_initial_detail(
                f"[{header}] SparseEvidenceGNN dimensions unavailable: "
                "summary context was not configured."
            )
            return
        emit_initial_detail(
            f"[{header}] SparseEvidenceGNN dimensions "
            f"B={context['batch_size']} T={context['n_time']} "
            f"dtype={context['dtype']} n_samples={context['n_samples']}"
        )

    def _batched_freqs(self, freqs: torch.Tensor, batch_size: int) -> torch.Tensor:
        if freqs.ndim == 1:
            freqs = freqs.view(1, -1).expand(batch_size, -1)
        if freqs.shape != (batch_size, self.nfreqs):
            raise ValueError(
                f"Expected freqs shape {(batch_size, self.nfreqs)} or "
                f"{(self.nfreqs,)}, got {tuple(freqs.shape)}."
            )
        return freqs

    def _full_edge_wct_maps(
        self,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
        *,
        compute_mag: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_edges = self.src_idx.numel()
        # 2026-08-23: this stage's elementwise products below (index_select
        # + multiply/subtract on the raw [B, C, T, F] CWT coefficients) are
        # NOT autocast-eligible ops -- confirmed directly (`a*b-a` keeps its
        # input dtype under torch.autocast(dtype=bfloat16); only conv2d/
        # matmul-type ops get auto-downcast there). That means
        # dense_edge_amp_bf16's outer torch.autocast(...) context (see
        # compute_dense_edge_input's caller, _precompute_dense_edge_inputs)
        # never actually shrank THESE tensors -- only _smooth_wct_maps's
        # conv2d further downstream did. This is exactly the stage that
        # OOM'd at 23 channels/T=7680 native resolution (2026-08-23 session:
        # `xwt_imag = src_i * dst_r - src_r * dst_i` tried to allocate 1.85
        # GiB with nothing free). Explicitly casting to the ambient autocast
        # dtype here -- rather than relying on autocast's own (here,
        # ineffective) op-based casting -- is what actually halves this
        # stage's VRAM footprint. torch.is_autocast_enabled() is only True
        # inside an active CUDA autocast context (i.e. dense_edge_amp_bf16=
        # True on CUDA, the only caller that opens one around this code
        # path); a no-op everywhere else (CPU/MPS, or bf16 off), same
        # gating the outer context already provides -- no new flag needed.
        # Also casts `freqs` (not just w_real/w_imag): the final
        # `xwt_real * inv_scale` below is a plain elementwise multiply too
        # (same non-autocast-eligible category as everything else in this
        # function), and freqs/inv_scale start out fp32 regardless -- left
        # uncast, that multiply's ordinary bf16-times-fp32 type promotion
        # would silently upcast the RETURNED xwt_real/xwt_imag/auto1/auto2
        # back to fp32 right at the end, undoing the cast above for exactly
        # the tensors _smooth_wct_maps's conv2d (and its own `maps` stack)
        # receives next.
        if torch.is_autocast_enabled():
            amp_dtype = torch.get_autocast_dtype("cuda")
            w_real = w_real.to(amp_dtype)
            w_imag = w_imag.to(amp_dtype)
            freqs = freqs.to(amp_dtype)
        real_edges = w_real.index_select(1, self.edge_pair_idx)
        imag_edges = w_imag.index_select(1, self.edge_pair_idx)
        src_r, dst_r = real_edges.split(num_edges, dim=1)
        src_i, dst_i = imag_edges.split(num_edges, dim=1)

        xwt_real = src_r * dst_r + src_i * dst_i
        xwt_imag = src_i * dst_r - src_r * dst_i
        mag = None
        if compute_mag:
            mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
        auto1 = src_r * src_r + src_i * src_i
        auto2 = dst_r * dst_r + dst_i * dst_i

        inv_scale = freqs.view(freqs.shape[0], 1, 1, self.nfreqs)
        return (
            mag,
            xwt_real * inv_scale,
            xwt_imag * inv_scale,
            auto1 * inv_scale,
            auto2 * inv_scale,
        )

    def _smooth_wct_maps(
        self,
        xwt_real: torch.Tensor,
        xwt_imag: torch.Tensor,
        auto1: torch.Tensor,
        auto2: torch.Tensor,
        smooth_kernel_and_pad: tuple[torch.Tensor, tuple[int, int, int, int]],
        *,
        stride: tuple[int, int] = (1, 1),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smooth_kernel, pad = smooth_kernel_and_pad
        batch_size, num_edges, n_time, nfreqs = xwt_real.shape

        maps = torch.stack([xwt_real, xwt_imag, auto1, auto2], dim=2)
        maps = maps.view(batch_size * num_edges * 4, 1, n_time, nfreqs)

        # smooth_kernel (from make_gaussian_weight2d) is separable by
        # construction -- exp(-0.5*((h/sh)^2+(w/sw)^2)) is an outer product
        # of two 1D gaussians before the joint normalization -- so a single
        # conv2d over the full (kh, kw) kernel is mathematically identical
        # to two sequential 1D convs (H then W). We use the latter because
        # PyTorch's CPU conv2d im2col/unfold buffer scales with kh*kw*spatial,
        # which blows up for elongated kernels (e.g. (25, 3) on this
        # pipeline's edge-batched tensors materialized a ~21.6GB unfold and
        # nearly OOM'd the machine); the separable form scales with
        # (kh+kw)*spatial instead. The two 1D factors are recovered exactly
        # from the 2D kernel's marginals (each already sums to 1 by
        # construction: kernel[p,q] = kh_1d[p]*kw_1d[q]), so no extra
        # arguments are needed and every existing caller/signature is
        # unchanged.
        pad_w_left, pad_w_right, pad_h_top, pad_h_bottom = pad
        stride_h, stride_w = stride
        kh_1d = smooth_kernel.sum(dim=-1, keepdim=True)  # [1, 1, kh, 1]
        kw_1d = smooth_kernel.sum(dim=-2, keepdim=True)  # [1, 1, 1, kw]

        if pad_h_top or pad_h_bottom:
            maps = torch.nn.functional.pad(
                maps, (0, 0, pad_h_top, pad_h_bottom), mode=self.padding_mode
            )
        maps = torch.nn.functional.conv2d(maps, kh_1d, stride=(stride_h, 1))

        if pad_w_left or pad_w_right:
            maps = torch.nn.functional.pad(
                maps, (pad_w_left, pad_w_right, 0, 0), mode=self.padding_mode
            )
        smoothed = torch.nn.functional.conv2d(maps, kw_1d, stride=(1, stride_w))

        out_time, out_freq = smoothed.shape[-2:]
        smoothed = smoothed.view(batch_size, num_edges, 4, out_time, out_freq)
        # 2026-08-23: was `smooth_cross = torch.complex(smoothed[:, :, 0],
        # smoothed[:, :, 1])` + `coh = smooth_cross.abs() ** 2 / ...`,
        # returning smooth_cross itself for callers to run torch.angle() on.
        # Two problems under dense_edge_amp_bf16's
        # torch.autocast(dtype=torch.bfloat16): (1) torch.complex() only
        # accepts Half/Float/Double, rejecting the bf16 `smoothed` this
        # produces (RuntimeError, confirmed directly) -- forcing the two
        # slices to .float() fixed the crash but (2) complex64 itself is
        # ALWAYS 8 bytes/element (two fp32 lanes) regardless of the real
        # dtype fed in, so that fix silently re-inflated exactly the
        # [B, E, T, F]-at-native-resolution tensor this whole change exists
        # to shrink -- confirmed directly: the 23-channel/T=7680 OOM this
        # session persisted afterward, same 1.85GiB allocation size as
        # before the fix, just moved here from _full_edge_wct_maps.
        # real_c/imag_c below are the same values smooth_cross's real/imag
        # parts would have held, but as plain tensors: no complex64 dtype
        # to force a minimum 8 bytes/element, so they (and everything built
        # from them -- coh, and callers' phase) stay in `smoothed`'s own
        # dtype -- bf16 under autocast, bit-identical to the previous
        # fp32-only behavior otherwise. torch.angle(complex(r, i)) ==
        # torch.atan2(i, r) and complex(r,i).abs()**2 == r*r + i*i
        # algebraically -- every caller that used to do
        # `smooth_cross, coh, _ = self._smooth(...); phase =
        # torch.angle(smooth_cross)` now gets `phase` back directly instead
        # of `smooth_cross` (see this method's callers) and drops that
        # torch.angle() call, since there is no complex tensor left to call
        # it on.
        real_c = smoothed[:, :, 0]
        imag_c = smoothed[:, :, 1]
        smooth_auto1 = smoothed[:, :, 2]
        smooth_auto2 = smoothed[:, :, 3]
        coh = (real_c * real_c + imag_c * imag_c) / (smooth_auto1 * smooth_auto2 + 1e-12)
        phase = torch.atan2(imag_c, real_c)
        return phase, coh.clamp(min=0.0, max=1.0), smooth_kernel

    def _resolved_scale_adaptive_max_kernel(self) -> int:
        """scale_adaptive_max_kernel rounded up to odd, as every kernel-
        building/offset site needs it (an even width has no single center
        sample). Computed once per call rather than stored so a mid-run
        attribute edit (e.g. from a debug script) can't desync it from
        self.scale_adaptive_max_kernel."""
        max_k = int(self.scale_adaptive_max_kernel)
        return max_k if max_k % 2 == 1 else max_k + 1

    def _time_offset_samples(self) -> int:
        """Half-width (in raw samples) of whatever time-domain smoothing
        kernel _smooth (below) actually applies -- both the flat
        smooth_kernel_size[0] path and the scale-adaptive path are VALID
        (unpadded) convolutions, so coh/phase's time axis is shorter than
        the input by this offset on the leading edge (and the same on the
        trailing edge). _coi_valid_mask needs this to map T_out-space
        indices back to original-sample-space ones. The scale-adaptive
        path's actual shrink is scale_adaptive_max_kernel-1 (see
        _scale_adaptive_time_kernel -- every frequency's kernel is
        zero-padded out to that common width before the valid conv, so the
        shrink is uniform across frequency bins even though each bin's real
        kernel is narrower), NOT smooth_kernel_size[0]-1, which no longer
        describes the time width actually used once scale-adaptive
        smoothing is on."""
        if self.scale_adaptive_smoothing:
            return (self._resolved_scale_adaptive_max_kernel() - 1) // 2
        return (self.smooth_kernel_size[0] - 1) // 2

    def _coi_valid_mask(self, freqs_batched: torch.Tensor, n_time_in: int, T_out: int) -> torch.Tensor:
        """Cone-of-influence validity mask, aligned to coh/phase's time axis.

        NOT computed anywhere upstream in this pipeline (fcwt.cwt returns no
        COI array). Assumes native-resolution CWT coefficients -- i.e. the
        classifier's cwt_resample_n_time=None, so `n_time_in` (w_real's own
        time axis) already equals the original CWT's sample count and no
        extra rescale factor is needed. If cwt_resample_n_time is set to a
        non-None value upstream, this mask will be wrong (see the warning
        raised in SparseEvidenceGNNClassifier.__init__).
        """
        device, dtype = freqs_batched.device, freqs_batched.dtype
        scale = self.sampling_rate / freqs_batched  # [B, F], samples
        support = torch.floor(self._COI_WAVELET_FB * scale * 3.0)  # [B, F]
        time_offset = self._time_offset_samples()
        t_idx = torch.arange(T_out, device=device, dtype=dtype).view(1, T_out, 1) + time_offset
        support_b = support.unsqueeze(1)  # [B, 1, F]
        valid = (t_idx >= support_b) & (t_idx < (n_time_in - support_b))  # [B, T_out, F]
        return valid.unsqueeze(1)  # [B, 1, T_out, F] -- broadcasts over the edge dim

    def _scale_adaptive_time_kernel(
        self, freqs_1d: torch.Tensor, device, dtype
    ) -> torch.Tensor:
        """Builds a per-frequency time-domain Gaussian smoothing kernel
        whose width is proportional to THAT frequency's own oscillation
        period (scale_adaptive_cycles cycles), instead of one flat width
        for every frequency -- see scale_adaptive_smoothing's docstring on
        SparseEvidenceGNNClassifier for the full rationale (standard
        "scale-adaptive" wavelet-coherence smoothing, Torrence & Webster
        1999): a fixed raw-sample kernel under-smooths low frequencies (a
        far smaller fraction of one oscillation cycle there) much more than
        it under-smooths high ones, which is the mechanism behind this
        pipeline's mu-band coherence saturating near the surrogate null
        with almost no real separation.

        Every frequency's kernel is centered and zero-padded out to a
        common width (scale_adaptive_max_kernel, rounded to odd) so a
        single grouped `F.conv1d(..., groups=nfreqs)` call can apply all of
        them at once -- this also means the VALID-convolution time shrink
        (scale_adaptive_max_kernel - 1) is uniform across frequency bins
        even though each bin's real (non-zero) kernel support is narrower;
        see _time_offset_samples.

        Returns weight shaped [nfreqs, 1, max_k], ready for
        F.conv1d(input, weight, groups=nfreqs).
        """
        n_freqs = int(freqs_1d.shape[0])
        max_k = self._resolved_scale_adaptive_max_kernel()
        # Two steps, not a fused .to("cpu", dtype=torch.float64): MPS
        # doesn't support float64 at all, so casting BEFORE the device move
        # (as a single fused .to(device, dtype) call can do) raises even
        # though the tensor is about to leave MPS anyway.
        freqs_cpu = freqs_1d.detach().cpu().to(dtype=torch.float64)
        weight = torch.zeros(n_freqs, 1, max_k, dtype=torch.float64)
        for f in range(n_freqs):
            period_samples = self.sampling_rate / max(float(freqs_cpu[f].item()), 1e-6)
            k = int(round(self.scale_adaptive_cycles * period_samples))
            k = max(3, min(k, max_k))
            if k % 2 == 0:
                k += 1
            sigma = (k - 1) / 2
            offsets = torch.arange(k, dtype=torch.float64) - (k - 1) / 2
            g = torch.exp(-0.5 * (offsets / sigma) ** 2)
            g = g / g.sum()
            start = (max_k - k) // 2
            weight[f, 0, start : start + k] = g
        return weight.to(device=device, dtype=dtype)

    def _smooth_wct_maps_scale_adaptive(
        self, xwt_real, xwt_imag, auto1, auto2, freqs_1d, smooth_kernel_and_pad,
        stride: tuple[int, int] = (1, 1),
    ):
        """Scale-adaptive counterpart to _smooth_wct_maps: time-smooths each
        frequency bin with ITS OWN kernel width (see
        _scale_adaptive_time_kernel) instead of one flat width shared by
        every frequency, then applies the SAME (unchanged, flat)
        frequency-axis smoothing `smooth_kernel_and_pad` already carries --
        only the time half of the separable kernel is replaced. Only
        reached when self.scale_adaptive_smoothing=True (see _smooth's
        dispatch).
        """
        stride_h, stride_w = stride
        batch_size, num_edges, n_time, nfreqs = xwt_real.shape
        device, dtype = xwt_real.device, xwt_real.dtype

        time_weight = self._scale_adaptive_time_kernel(freqs_1d, device, dtype)  # [F,1,max_k]

        maps = torch.stack([xwt_real, xwt_imag, auto1, auto2], dim=2)  # [B,E,4,T,F]
        # Frequency needs to be conv1d's channel/group dim for a per-frequency
        # kernel: [B,E,4,T,F] -> [B*E*4, F, T]. No time padding (mirrors the
        # flat path's pad_h=0 convention -- a VALID convolution, output T
        # shrinks by max_k-1 uniformly across every frequency bin regardless
        # of that bin's own (<=max_k) real kernel width, since each row is
        # zero-padded out to max_k -- see _time_offset_samples).
        maps = maps.permute(0, 1, 2, 4, 3).reshape(batch_size * num_edges * 4, nfreqs, n_time)
        time_smoothed = torch.nn.functional.conv1d(
            maps, time_weight, stride=stride_h, groups=nfreqs
        )  # [B*E*4, F, T_out]

        freq_kernel, freq_pad = smooth_kernel_and_pad
        kw_1d = freq_kernel.sum(dim=-2, keepdim=True)  # [1,1,1,kw] -- same freq marginal as the flat path
        pad_w_left, pad_w_right, _pad_h_top, _pad_h_bottom = freq_pad
        time_smoothed = time_smoothed.permute(0, 2, 1).unsqueeze(1)  # [B*E*4, 1, T_out, F]
        if pad_w_left or pad_w_right:
            time_smoothed = torch.nn.functional.pad(
                time_smoothed, (pad_w_left, pad_w_right, 0, 0), mode=self.padding_mode
            )
        smoothed = torch.nn.functional.conv2d(time_smoothed, kw_1d, stride=(1, stride_w))

        out_time, out_freq = smoothed.shape[-2:]
        smoothed = smoothed.view(batch_size, num_edges, 4, out_time, out_freq)
        # 2026-08-23: same fix as _smooth_wct_maps's identical line (see
        # that method's comment for the full explanation of why the
        # torch.complex()-based version both crashed AND, once patched to
        # avoid the crash, defeated dense_edge_amp_bf16's own purpose) --
        # this is the scale_adaptive_smoothing=True path, not this
        # pipeline's actual config (scale_adaptive_smoothing=False), but
        # gets the same real/imag-without-complex64 treatment rather than
        # being left with either problem live.
        real_c = smoothed[:, :, 0]
        imag_c = smoothed[:, :, 1]
        smooth_auto1 = smoothed[:, :, 2]
        smooth_auto2 = smoothed[:, :, 3]
        coh = (real_c * real_c + imag_c * imag_c) / (smooth_auto1 * smooth_auto2 + 1e-12)
        phase = torch.atan2(imag_c, real_c)
        return phase, coh.clamp(min=0.0, max=1.0), time_weight

    def _smooth(self, xwt_real, xwt_imag, auto1, auto2, freqs_batched, smooth_kernel_and_pad,
                stride: tuple[int, int] = (1, 1)):
        """Dispatches to the flat-kernel _smooth_wct_maps (default) or the
        per-frequency _smooth_wct_maps_scale_adaptive (when
        scale_adaptive_smoothing=True) -- the single choke point every
        caller in this class should go through instead of calling
        _smooth_wct_maps directly, so the two paths can never drift apart at
        one call site and not another."""
        if self.scale_adaptive_smoothing:
            freqs_1d = freqs_batched[0]
            return self._smooth_wct_maps_scale_adaptive(
                xwt_real, xwt_imag, auto1, auto2, freqs_1d, smooth_kernel_and_pad, stride=stride,
            )
        return self._smooth_wct_maps(
            xwt_real, xwt_imag, auto1, auto2, smooth_kernel_and_pad, stride=stride,
        )

    def _coherence_only(self, w_real, w_imag, freqs_batched, smooth_kernel_and_pad):
        """Cross-spectrum + smoothing only -- no gate/COI/consolidation.

        Used to build the null coherence distribution for surrogate
        significance calibration (see
        SparseEvidenceGNNClassifier._surrogate_coherence_threshold); shares
        the exact same (parameter-free) math _build_sparse_events uses for
        real trials, so the null is computed under identical conditions.

        Returns (coh, phase) -- phase is needed by _max_cluster_statistic's
        cluster-forming gate for coherence_threshold_mode="surrogate_cluster",
        so the null is formed under the exact same (coherence AND phase) gate
        real events use, not coherence alone (a coherence-only null would
        systematically overstate cluster sizes relative to what real
        candidate clusters -- which must also clear the phase gate -- can
        ever achieve).
        """
        with torch.no_grad():
            _, xwt_real, xwt_imag, auto1, auto2 = self._full_edge_wct_maps(
                w_real, w_imag, freqs_batched, compute_mag=False
            )
            # 2026-08-23: _smooth's first return value used to be a complex
            # `smooth_cross` tensor this call site ran torch.angle() on --
            # see _smooth_wct_maps's 2026-08-23 comment for why that's now
            # `phase` directly (torch.atan2(imag, real), computed without
            # ever forming a complex64 tensor).
            phase, coh, _ = self._smooth(
                xwt_real, xwt_imag, auto1, auto2, freqs_batched, smooth_kernel_and_pad,
                stride=(1, 1),
            )
        return coh, phase

    def _max_cluster_statistic(self, coh, cluster_forming_threshold, coi_valid, phase):
        """Maris & Oostenveld-style cluster statistic: for each item along
        coh's batch dim (one per surrogate, when called during null-
        distribution calibration), forms candidate clusters via
        gate = (coh > cluster_forming_threshold) & (phase.abs() > self.phase_threshold_rad)
        & coi_valid -- the SAME two-part gate _build_sparse_events uses to
        form real events (coherence alone would systematically overstate
        cluster sizes relative to what real candidates, which must also
        clear the phase gate, can ever achieve; matching the gate exactly
        is what makes this a fair null). Consolidation additionally breaks a
        run wherever phase's SIGN flips (mirroring _build_sparse_events's
        own same-sign-continuation logic) -- necessary now that edges are
        the canonical (i<j) undirected pairing (2026-08-09) and the gate is
        two-sided: a run spanning a sign flip would silently average a
        positive-lag sample with a negative-lag one into a near-zero
        mean-angle that represents neither direction, inflating this
        null's cluster-mass statistic relative to what real (equally
        sign-broken) candidate clusters can ever achieve. Scores each
        cluster by its "cluster mass" (sum of coh-minus-threshold over its
        member samples), and returns the MAXIMUM cluster mass found
        anywhere -- across every frequency and run WITHIN EACH EDGE -- per
        (batch, edge) pair.

        Scoped per-edge, not pooled across the whole (edge, freq, time)
        search space, deliberately: empirically (2026-08-08 session notes,
        Arc 7, measured under the pre-2026-08-09 72-edge directed topology --
        the qualitative conclusion carries over to today's 36 canonical
        edges, but the exact figures quoted there predate this change),
        pooling the max across all edges x 16 freqs makes the null's
        "biggest spurious cluster anywhere" statistic far larger than any
        real, single-edge effect can plausibly beat -- a real cluster on
        one edge was never even close to the pooled null's MINIMUM across a
        forming-threshold sweep from the 70th to 99.5th percentile, i.e. the
        whole-graph version would reject every trial regardless of forming
        threshold. Correcting for multiple comparisons across each edge's
        own 16 freqs x T time cells (its natural "family," since a real
        coupling event lives on one specific channel pair) instead of
        across all edges' cells jointly keeps the correction meaningful
        without eliminating the pipeline's sensitivity entirely.

        This is the null-distribution statistic for cluster-based
        multiple-comparisons correction: pooling the per-surrogate maxima
        (within an edge) across many surrogates gives a null distribution
        of "the biggest spurious cluster you'd see on this edge by chance,"
        which a real candidate cluster on that edge must beat to be called
        significant -- unlike a flat per-cell percentile, this controls the
        false-positive rate across an edge's (freq, time) cells jointly
        rather than giving each cell its own independent false-positive
        budget. See coherence_threshold_mode="surrogate_cluster".

        `cluster_forming_threshold` must be broadcastable to coh's
        [B, E, T, F] shape (e.g. [1, E, 1, F] from the percentile grid).
        Returns a tensor shaped [B, E].
        """
        gate = (
            (coh > cluster_forming_threshold)
            & (phase.abs() > self.phase_threshold_rad)
            & coi_valid
        )
        B, E, T, F = gate.shape
        max_per_batch_edge = torch.zeros(B, E, device=coh.device, dtype=coh.dtype)
        if not gate.any():
            return max_per_batch_edge

        # 2026-08-09: the whole run-consolidation block below (permute+
        # reshape -> nonzero() -> unique(return_inverse=True)) is forced
        # onto CPU, not just the final scatter_reduce_ (which was ALREADY
        # CPU-only "for MPS portability" -- see below). Root-caused a real
        # crash (RuntimeError: index out of bounds in scatter_reduce_) to a
        # genuine PyTorch MPS backend bug, not a logic error here: reproduced
        # directly against real subject-2 data (surrogate_device="auto" ->
        # mps on this machine) with diagnostics confirming `gate_r.nonzero()`
        # on a correctly-shaped, CONTIGUOUS `(11520, 997)` bool tensor
        # returned row/col indices (122222, 241502) that are mathematically
        # impossible for that shape -- confirmed corrupted immediately after
        # nonzero() returns, not introduced by any of this function's own
        # arithmetic. A matched synthetic repro (same permute+reshape+
        # nonzero pattern, random data, same shape) did NOT reproduce it on
        # this machine/torch version, so it's data-dependent, not a blanket
        # "nonzero is always broken on MPS" issue -- meaning it can silently
        # give WRONG (not just crashing) results on other inputs too. Moving
        # this block to CPU is the safe fix; it also matches
        # _build_sparse_events's identical run-consolidation pattern, which
        # carries the SAME risk for real (non-null) events whenever this
        # precompute runs on MPS (coherence_threshold_mode in {"surrogate",
        # "surrogate_cluster"} moves `helper` to surrogate_device) -- see the
        # matching fix there. Cost: one extra device transfer of a handful of
        # ~11.5M-element tensors per surrogate sub-chunk, small next to the
        # conv2d smoothing that already dominates this precompute's runtime.
        gate_r = gate.permute(0, 1, 3, 2).reshape(B * E * F, T).cpu()
        coh_r = coh.permute(0, 1, 3, 2).reshape(B * E * F, T).cpu()
        phase_r = phase.permute(0, 1, 3, 2).reshape(B * E * F, T).cpu()
        threshold_full = cluster_forming_threshold.expand_as(coh)
        thr_r = threshold_full.permute(0, 1, 3, 2).reshape(B * E * F, T).cpu()
        R = gate_r.shape[0]

        # Break a run wherever gate_r is true but phase's sign differs from
        # the immediately preceding (also-gated) sample -- see docstring.
        # sign_r is always +-1 (never 0) wherever gate_r is true, since
        # gate_r requires phase.abs() > threshold >= 0.
        sign_r = torch.sign(phase_r)
        same_sign_continuation = torch.zeros_like(gate_r)
        same_sign_continuation[:, 1:] = (
            gate_r[:, :-1] & (sign_r[:, 1:] == sign_r[:, :-1])
        )
        starts = torch.zeros_like(gate_r)
        starts[:, 0] = gate_r[:, 0]
        starts[:, 1:] = gate_r[:, 1:] & (~same_sign_continuation[:, 1:])
        run_id_local = starts.cumsum(dim=1)
        row_offset = torch.arange(R, device=gate_r.device).view(R, 1) * (T + 1)
        global_run_id = run_id_local + row_offset

        valid_pos = gate_r.nonzero(as_tuple=False)
        row_idx, time_idx = valid_pos.unbind(1)
        run_ids_at_valid = global_run_id[row_idx, time_idx]
        excess_at_valid = coh_r[row_idx, time_idx] - thr_r[row_idx, time_idx]

        unique_runs, inverse = torch.unique(run_ids_at_valid, return_inverse=True)
        n_runs = unique_runs.shape[0]
        cluster_mass_cpu = torch.zeros(n_runs, dtype=coh.dtype).index_add_(
            0, inverse, excess_at_valid
        )
        row_of_run = torch.zeros(n_runs, dtype=torch.long)
        row_of_run[inverse] = row_idx
        # row_of_run indexes the flattened B*E*F rows; decompose to (b, e)
        # -- e is the middle factor since rows were built as
        # (b, e, f) -> b*E*F + e*F + f via the earlier .reshape(B*E*F, T).
        b_of_run = row_of_run // (E * F)
        e_of_run = (row_of_run % (E * F)) // F

        # scatter_reduce_(reduce="amax") over the flattened (b, e) index --
        # b_of_run/e_of_run/cluster_mass_cpu are already CPU tensors (the
        # whole block above now runs on CPU -- see the 2026-08-09 note).
        be_of_run = b_of_run * E + e_of_run
        max_per_be_cpu = torch.zeros(B * E, dtype=coh.dtype)
        max_per_be_cpu.scatter_reduce_(0, be_of_run, cluster_mass_cpu, reduce="amax")
        return max_per_be_cpu.view(B, E).to(coh.device)

    def _build_sparse_events(
        self, w_real, w_imag, freqs_batched, smooth_kernel_and_pad,
        coherence_threshold_override=None, cluster_mass_null_threshold=None,
    ):
        with torch.no_grad():
            _, xwt_real, xwt_imag, auto1, auto2 = self._full_edge_wct_maps(
                w_real, w_imag, freqs_batched, compute_mag=False
            )
            # 2026-08-23: see _coherence_only's identical call, above --
            # `phase` comes back directly now, not a complex tensor to run
            # torch.angle() on.
            phase, coh, _ = self._smooth(
                xwt_real, xwt_imag, auto1, auto2, freqs_batched, smooth_kernel_and_pad,
                stride=(1, 1),
            )
            threshold = (
                self.coherence_threshold
                if coherence_threshold_override is None
                else coherence_threshold_override
            )
            # Two-sided by design (2026-08-09), not the 2026-08-07 `.abs()`
            # regression re-appearing: THAT bug came from src_idx/dst_idx
            # instantiating BOTH directed copies of every channel pair
            # (i->j and j->i) while gating each independently on the SAME
            # symmetric |phase|>threshold test -- since xwt_(j->i) =
            # conj(xwt_(i->j)) exactly, |phase_ij| == |phase_ji| always, so
            # both edges' gates fired together on every qualifying cell,
            # roughly doubling event volume with duplicate, non-directional
            # signal. Edges are now the canonical (i<j) UNDIRECTED pairing
            # (see __init__) -- one edge per pair, not two -- so there is no
            # second edge left to double-fire, and phase.abs() is now the
            # correct, non-duplicating gate: a cell's phase can only be
            # positive or negative, never both, so it contributes to at
            # most one run either way. The direction bit that used to live
            # in "which of the two edges fired" now lives in the SIGN of
            # this event's stored angle (sin(mean_angle) below) instead.
            gate = (coh > threshold) & (phase.abs() > self.phase_threshold_rad)
            if self.coi_enabled:
                coi_valid = self._coi_valid_mask(
                    freqs_batched, n_time_in=w_real.shape[2], T_out=coh.shape[2]
                )
                gate = gate & coi_valid

            B, E, T, F = gate.shape
            gate_r = gate.permute(0, 1, 3, 2).reshape(B * E * F, T)
            coh_r = coh.permute(0, 1, 3, 2).reshape(B * E * F, T)
            phase_r = phase.permute(0, 1, 3, 2).reshape(B * E * F, T)
            R = gate_r.shape[0]

            if gate_r.any():
                # A run must also break wherever the phase SIGN flips, even
                # if the gate stays on across the flip -- otherwise a
                # consolidated event can silently average a positive-lag
                # sample with a negative-lag sample into a near-zero
                # mean_angle that represents neither direction. Live/load-
                # bearing now that the gate is two-sided (phase.abs() >
                # threshold, see above): a single canonical edge can
                # legitimately carry both directions' activity at different
                # (t, freq) cells, so consecutive gated samples are no
                # longer guaranteed same-signed the way they were under the
                # old one-sided, duplicate-edge scheme. sign_r is always
                # +-1 (never 0) wherever gate_r is true, since gate_r
                # requires |phase| > threshold >= 0.
                #
                # 2026-08-09: the nonzero()/unique() index math (only) is
                # done on CPU copies of gate_r/phase_r, then row_idx/
                # time_idx/inverse are converted back to gate.device right
                # after -- everything else in this function is unchanged.
                # See _max_cluster_statistic's matching note for the
                # root-caused MPS `.nonzero()` bug this works around:
                # reproduced directly against real data (surrogate_device
                # resolves to "mps" by default) -- `.nonzero()` on a
                # correctly-shaped, contiguous MPS bool tensor returned
                # row/col indices that were mathematically impossible for
                # that shape, confirmed corrupted immediately after
                # nonzero() returns (not introduced by this function's own
                # arithmetic). This function's own precompute pathway runs
                # on `surrogate_torch_device` in "surrogate"/
                # "surrogate_cluster" mode (see
                # SparseEvidenceGNNClassifier._precompute_sparse_events,
                # which moves `helper` there) and therefore builds the REAL
                # (non-null) events under the same risk _max_cluster_
                # statistic was crashing on -- without this fix it could
                # silently corrupt real training events on MPS rather than
                # crashing, since a less-severe corruption need not trip any
                # bounds check.
                gate_r_cpu = gate_r.cpu()
                phase_r_cpu = phase_r.cpu()
                sign_r = torch.sign(phase_r_cpu)
                same_sign_continuation = torch.zeros_like(gate_r_cpu)
                same_sign_continuation[:, 1:] = (
                    gate_r_cpu[:, :-1] & (sign_r[:, 1:] == sign_r[:, :-1])
                )
                starts = torch.zeros_like(gate_r_cpu)
                starts[:, 0] = gate_r_cpu[:, 0]
                starts[:, 1:] = gate_r_cpu[:, 1:] & (~same_sign_continuation[:, 1:])
                run_id_local = starts.cumsum(dim=1)
                row_offset = torch.arange(R).view(R, 1) * (T + 1)
                global_run_id = run_id_local + row_offset

                valid_pos = gate_r_cpu.nonzero(as_tuple=False)
                row_idx_cpu, time_idx_cpu = valid_pos.unbind(1)
                run_ids_at_valid = global_run_id[row_idx_cpu, time_idx_cpu]
                unique_runs, inverse_cpu = torch.unique(run_ids_at_valid, return_inverse=True)
                n_runs = unique_runs.shape[0]

                row_idx = row_idx_cpu.to(gate.device)
                time_idx = time_idx_cpu.to(gate.device)
                inverse = inverse_cpu.to(gate.device)

                mag_at_valid = coh_r[row_idx, time_idx]
                angle_at_valid = phase_r[row_idx, time_idx]
                time_at_valid = time_idx.to(coh.dtype)

                def scatter_mean(values):
                    s = torch.zeros(n_runs, dtype=coh.dtype, device=gate.device).index_add_(
                        0, inverse, values
                    )
                    c = torch.zeros(n_runs, dtype=coh.dtype, device=gate.device).index_add_(
                        0, inverse, torch.ones_like(values)
                    )
                    return s / c

                mean_mag = scatter_mean(mag_at_valid)
                mean_angle = scatter_mean(angle_at_valid)
                mean_time = scatter_mean(time_at_valid)

                row_of_run = torch.zeros(n_runs, dtype=torch.long, device=gate.device)
                row_of_run[inverse] = row_idx

                b_of_run = row_of_run // (E * F)
                rem = row_of_run % (E * F)
                e_of_run = rem // F
                f_of_run = rem % F

                if cluster_mass_null_threshold is not None:
                    # coherence_threshold_mode="surrogate_cluster": a
                    # candidate run formed above (via the per-cell
                    # cluster-forming threshold in `threshold`) only
                    # survives as a real event if its cluster mass (sum of
                    # coh-minus-threshold over its member samples) beats
                    # this trial's own null distribution of maximum cluster
                    # mass -- see SparseEvidenceGNNCore._max_cluster_statistic
                    # and SparseEvidenceGNNClassifier._precompute_sparse_events.
                    threshold_full = (
                        threshold if torch.is_tensor(threshold)
                        else torch.full_like(coh, float(threshold))
                    ).expand_as(coh)
                    thr_r = threshold_full.permute(0, 1, 3, 2).reshape(B * E * F, T)
                    excess_at_valid = mag_at_valid - thr_r[row_idx, time_idx]
                    cluster_mass = torch.zeros(
                        n_runs, dtype=coh.dtype, device=gate.device
                    ).index_add_(0, inverse, excess_at_valid)
                    # cluster_mass_null_threshold is [B, E] -- per-edge, not
                    # a single trial-wide scalar (see _max_cluster_statistic
                    # and _surrogate_cluster_thresholds's docstrings for why).
                    null_per_run = cluster_mass_null_threshold[b_of_run, e_of_run]
                    keep_run = cluster_mass > null_per_run

                    mean_mag = mean_mag[keep_run]
                    mean_angle = mean_angle[keep_run]
                    mean_time = mean_time[keep_run]
                    b_of_run = b_of_run[keep_run]
                    e_of_run = e_of_run[keep_run]
                    f_of_run = f_of_run[keep_run]
                    n_runs = int(keep_run.sum().item())

                dst_node = self.dst_idx[e_of_run]
                src_node = self.src_idx[e_of_run]
                # freq_node is f_of_run itself (the discrete CWT frequency
                # BIN index, 0..F-1) -- kept as its own value (not derived
                # later from event_features' continuous, min-max-normalized
                # freq_vals below) specifically for freq_aware_hops (see
                # __init__'s docstring / _aggregate_events_freq_indexed):
                # reverse-mapping a normalized float back to a bin index
                # would silently assume freqs_batched is evenly spaced,
                # which this pipeline's transform_fn is never guaranteed to
                # produce (e.g. log-spaced wavelet scales). f_of_run is
                # already the exact bin index the gate/threshold/event math
                # above used, so reusing it directly has no such risk.
                freq_node = f_of_run
                freq_vals_raw = freqs_batched[b_of_run, f_of_run]
                freq_vals = (freq_vals_raw - self._freq_lo) / max(
                    self._freq_hi - self._freq_lo, 1e-6
                )
                timestamp_vals = mean_time / float(max(T - 1, 1))
                b_idx = b_of_run

                event_features = torch.stack(
                    [timestamp_vals, freq_vals, mean_mag,
                     torch.sin(mean_angle), torch.cos(mean_angle)], dim=-1
                )
            else:
                n_runs = 0
                event_features = torch.zeros(0, 5, dtype=coh.dtype, device=gate.device)
                dst_node = torch.zeros(0, dtype=torch.long, device=gate.device)
                src_node = torch.zeros(0, dtype=torch.long, device=gate.device)
                freq_node = torch.zeros(0, dtype=torch.long, device=gate.device)
                b_idx = torch.zeros(0, dtype=torch.long, device=gate.device)

            counts = torch.bincount(b_idx, minlength=B)
            max_count = max(int(counts.max().item()) if n_runs > 0 else 1, 1)
            offsets = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=gate.device), counts.cumsum(0)[:-1]]
            )
            global_idx = torch.arange(n_runs, device=gate.device)
            pos_within_trial = global_idx - offsets[b_idx]

            events_padded = torch.zeros(B, max_count, 5, dtype=coh.dtype, device=gate.device)
            dst_padded = torch.zeros(B, max_count, dtype=torch.long, device=gate.device)
            src_padded = torch.zeros(B, max_count, dtype=torch.long, device=gate.device)
            # freq_idx_padded's fill value (0) for padding slots is never
            # read: those positions are always excluded by valid_mask=False
            # everywhere freq_idx_padded is actually consumed
            # (_aggregate_events_freq_indexed), same convention dst_padded/
            # src_padded already rely on.
            freq_idx_padded = torch.zeros(B, max_count, dtype=torch.long, device=gate.device)
            valid_mask = torch.zeros(B, max_count, dtype=torch.bool, device=gate.device)
            events_padded[b_idx, pos_within_trial] = event_features
            dst_padded[b_idx, pos_within_trial] = dst_node
            src_padded[b_idx, pos_within_trial] = src_node
            freq_idx_padded[b_idx, pos_within_trial] = freq_node
            valid_mask[b_idx, pos_within_trial] = True

        event_density = n_runs / max(B * E * F, 1)
        return events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask, event_density

    def compute_events(
        self, w_real, w_imag, freqs, coherence_threshold_override=None,
        cluster_mass_null_threshold=None,
    ):
        """Runs the full non-trainable pipeline (cross-spectrum -> smoothing
        -> gate -> COI -> run-consolidation) and returns padded per-trial
        events. This is what forward() used to do on every call; profiling
        showed it was 94.8% of forward()'s time despite being a deterministic
        function of these (fixed, precomputed) CWT features -- it's now
        called once per trial by SparseEvidenceGNNClassifier._prepare_features
        instead of once per (batch, epoch). Kept as its own method so it's
        still directly callable for debugging (e.g. debug_sparse_evidence_gnn.py
        calls the pieces of this directly).

        `coherence_threshold_override`, if given, replaces the fixed
        `self.coherence_threshold` scalar with a tensor broadcastable to
        coh's [B, E, T, F] shape (e.g. [B, E, 1, F], one calibrated
        per-edge/frequency threshold per trial in the batch) -- see
        SparseEvidenceGNNClassifier's coherence_threshold_mode="surrogate".

        `cluster_mass_null_threshold`, if given (shape [B, E] -- per edge,
        not a single trial-wide scalar), additionally filters formed runs
        by cluster mass against each trial's own per-edge null distribution
        of maximum cluster mass -- see
        coherence_threshold_mode="surrogate_cluster".
        """
        batch_size = w_real.shape[0]
        freqs_batched = self._batched_freqs(freqs, batch_size)
        if self._freq_lo is None:
            self._freq_lo = float(freqs_batched.min().item())
            self._freq_hi = float(freqs_batched.max().item())
        smooth_kernel_and_pad = make_gaussian_weight2d(
            kernel_size=self.smooth_kernel_size, sigma=self.smooth_kernel_sigma,
            pad_h=0, device=w_real.device, dtype=w_real.dtype,
        )
        events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask, _ = self._build_sparse_events(
            w_real, w_imag, freqs_batched, smooth_kernel_and_pad,
            coherence_threshold_override=coherence_threshold_override,
            cluster_mass_null_threshold=cluster_mass_null_threshold,
        )
        return events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask

    def _build_dense_edge_input(
        self, w_real, w_imag, freqs_batched, smooth_kernel_and_pad,
        coherence_threshold_override=None,
    ):
        """event_mode="dense" counterpart to _build_sparse_events: the exact
        same non-trainable cross-spectrum + smoothing math _coherence_only
        uses (so the null distribution _surrogate_coherence_threshold
        calibrates against stays valid for dense mode too -- nothing about
        the coherence/phase computation itself changes), but instead of
        thresholding + consolidating into a discrete event list, returns the
        full-resolution [coherence, sin(phase), cos(phase), significance]
        stack, post-COI-mask, for dense_edge_conv to process at forward()
        time.

        `coherence_threshold_override`, if given, is the SAME per-(edge,
        frequency) surrogate-calibrated threshold tensor compute_events'
        hard gate uses (see that method's docstring) -- here it instead
        feeds a CONTINUOUS significance channel,
        (coh - threshold) / threshold, so the model has access to "how far
        past the null this cell is" without a hard cutoff discarding
        everything below it. `threshold.clamp_min` guards only against a
        literal zero threshold (e.g. coherence_threshold_mode="fixed" left
        at its unused default of 0.0 -- see run_pipelines.py's SPARSE_FAMILY_PARAMS
        canonical config comment); coherence_threshold_mode="surrogate"'s
        real calibrated thresholds are never exactly zero.

        Returns a tensor shaped [B, 4, E, T, F] (channel dim = coh/sinφ/
        cosφ/significance), post-COI-mask (zeroed outside the cone of
        influence, same convention _build_sparse_events' gate uses via
        `& coi_valid`, rather than a discrete exclusion -- dense mode has no
        gate to AND into, so masking is applied to the values themselves).
        T is native resolution unless self.dense_edge_time_downsample > 1,
        in which case it's average-pooled down by that factor (see the
        constructor param's own docstring for why this is done HERE --
        post-smoothing, post-COI-mask -- rather than upstream on the raw CWT
        coefficients the way cwt_resample_n_time does), or T == 1 if
        self.time_averaged_graph is True (the whole axis collapsed to a
        single COI-valid-weighted average -- see that param's docstring and
        _time_average_dense_edge_input). __init__ rejects the two options
        combined, so at most one of them ever changes T here.
        """
        with torch.no_grad():
            _, xwt_real, xwt_imag, auto1, auto2 = self._full_edge_wct_maps(
                w_real, w_imag, freqs_batched, compute_mag=False
            )
            # 2026-08-23: see _coherence_only's identical call for the full
            # explanation -- `phase` comes back directly now, not a complex
            # tensor to run torch.angle() on. This is the event_mode="dense"
            # path DENSE_EDGE_GRU_PARAMS/PREDICTION_GRU_PARAMS actually use,
            # so this is the call site dense_edge_amp_bf16's VRAM fix
            # (see _full_edge_wct_maps/_smooth_wct_maps) was actually for.
            phase, coh, _ = self._smooth(
                xwt_real, xwt_imag, auto1, auto2, freqs_batched, smooth_kernel_and_pad,
                stride=(1, 1),
            )

            threshold = (
                self.coherence_threshold
                if coherence_threshold_override is None
                else coherence_threshold_override
            )
            threshold_full = (
                threshold if torch.is_tensor(threshold)
                else torch.full_like(coh, float(threshold))
            )
            significance = (coh - threshold_full) / threshold_full.clamp_min(1e-6)

            coi_valid = None
            if self.coi_enabled:
                coi_valid = self._coi_valid_mask(
                    freqs_batched, n_time_in=w_real.shape[2], T_out=coh.shape[2]
                ).to(coh.dtype)
                coh = coh * coi_valid
                phase = phase * coi_valid
                significance = significance * coi_valid

            stacked = torch.stack(
                [coh, torch.sin(phase), torch.cos(phase), significance], dim=1
            )  # [B, 4, E, T, F]

            if self.time_averaged_graph:
                stacked = self._time_average_dense_edge_input(stacked, coi_valid)
            elif self.dense_edge_time_downsample > 1:
                stacked = self._downsample_dense_edge_time(stacked)
        return stacked

    def _time_average_dense_edge_input(
        self, stacked: torch.Tensor, coi_valid: torch.Tensor | None
    ) -> torch.Tensor:
        """time_averaged_graph=True's own collapse of `stacked`'s ([B, 4, E,
        T, F]) time axis down to exactly 1 -- see that constructor param's
        docstring for the full rationale (this is the "single time-averaged
        coherence graph per trial" ablation itself).

        `coi_valid` ([B, 1, T, F], the SAME mask already applied to coh/
        phase/significance above -- or None when self.coi_enabled=False) is
        used as an averaging weight, not a plain torch.mean over T: dividing
        by the count of COI-valid timesteps per frequency (rather than by T
        itself) keeps the average an actual "mean coherence over the time
        this frequency bin has real support," instead of one that's
        systematically pulled toward zero by whatever fraction of the trial
        that frequency's cone excludes (worst at low frequencies, which have
        the widest cones). `clamp_min(1.0)` only guards a pathological
        all-invalid (edge, frequency) cell -- not expected in practice, since
        every canonical frequency bin's cone is far narrower than a full
        trial at this pipeline's epoch lengths.
        """
        if coi_valid is None:
            return stacked.mean(dim=3, keepdim=True)
        valid_counts = coi_valid.sum(dim=2, keepdim=True).clamp_min(1.0)  # [B, 1, 1, F]
        return stacked.sum(dim=3, keepdim=True) / valid_counts.unsqueeze(2)  # [B, 4, E, 1, F]

    def _downsample_dense_edge_time(self, stacked: torch.Tensor) -> torch.Tensor:
        """Average-pools `stacked`'s ([B, 4, E, T, F]) time axis by
        self.dense_edge_time_downsample -- see that param's own docstring
        for the full rationale. Only called when the factor is > 1 (a no-op
        otherwise, never reached).

        Reuses the exact fold-frequency-into-channels reshape
        SparseEvidenceGNNCore.forward's dense branch already uses to feed
        dense_edge_conv (permute F next to the channel axis, reshape into
        [B, C, E, T], run a (1, k)-kernel/stride 2D op, unfold back) --
        `nn.functional.avg_pool2d` here instead of dense_edge_conv's own
        `nn.MaxPool2d`, same (1, k) shape so only the edge (E) axis's height
        is left untouched and only T (width) is pooled. Trailing remainder
        timesteps that don't fill a full window are dropped (avg_pool2d's
        default ceil_mode=False) -- same convention _build_dense_feature_conv's
        own MaxPool2d stages already use, so this isn't a new truncation
        behavior for this pipeline to reason about.
        """
        batch_size, n_stack, num_edges, n_time, nfreqs = stacked.shape
        factor = self.dense_edge_time_downsample
        pool_in = stacked.permute(0, 1, 4, 2, 3).reshape(
            batch_size, n_stack * nfreqs, num_edges, n_time
        )  # [B, 4*F, E, T]
        pooled = torch.nn.functional.avg_pool2d(
            pool_in, kernel_size=(1, factor), stride=(1, factor)
        )  # [B, 4*F, E, T//factor]
        t_ds = pooled.shape[-1]
        return pooled.reshape(batch_size, n_stack, nfreqs, num_edges, t_ds).permute(
            0, 1, 3, 4, 2
        )  # [B, 4, E, T_ds, F]

    def compute_dense_edge_input(self, w_real, w_imag, freqs, coherence_threshold_override=None):
        """event_mode="dense" counterpart to compute_events -- runs the
        identical non-trainable cross-spectrum/smoothing/COI pipeline (see
        _build_dense_edge_input) once per trial, mirroring compute_events'
        own "non-trainable work runs once, not every forward()"
        optimization (see that method's docstring). Called by
        SparseEvidenceGNNClassifier._precompute_dense_edge_inputs.
        """
        batch_size = w_real.shape[0]
        freqs_batched = self._batched_freqs(freqs, batch_size)
        if self._freq_lo is None:
            self._freq_lo = float(freqs_batched.min().item())
            self._freq_hi = float(freqs_batched.max().item())
        smooth_kernel_and_pad = make_gaussian_weight2d(
            kernel_size=self.smooth_kernel_size, sigma=self.smooth_kernel_sigma,
            pad_h=0, device=w_real.device, dtype=w_real.dtype,
        )
        return self._build_dense_edge_input(
            w_real, w_imag, freqs_batched, smooth_kernel_and_pad,
            coherence_threshold_override=coherence_threshold_override,
        )

    def _aggregate_events(self, msg, full_features, dst_padded, valid_mask, batch_idx, dtype):
        """Combines each trial's per-event messages into per-(trial,
        destination channel) evidence -- the one place event_aggregation's
        modes actually differ (see __init__'s docstring for the full
        rationale).

        "mean" (original behavior): scatter_add every valid event's message
        onto its destination channel, then divide by how many valid events
        landed there. Every event counts equally; nothing in the aggregation
        itself lets one event outweigh another.

        "gated_softmax": event_gate produces one logit per event from the
        same `full_features` sparse_message_mlp sees, softmax-normalized
        across the events sharing a (trial, destination channel) group (a
        manual scatter-softmax: per-group max for numerical stability,
        exp, per-group sum, divide -- there's no built-in scatter_softmax in
        plain torch). Those weights already sum to 1 per group, so the
        weighted scatter_add below replaces the /active_count step entirely
        rather than adding to it. A channel with zero valid events simply
        never gets scattered into (evidence stays at its zero
        initialization) under either mode.

        "concat" (2026-08-10, event_mode="dense" only -- see __init__'s
        cross-validation): rather than pooling a channel's incident edges
        into one hidden_dim vector (lossy -- both "mean" and "gated_softmax"
        collapse ~n_channels-1 edges/channel down to one vector before
        sparse_classifier ever sees it), gathers every incident edge's own
        message into a distinct, fixed slot -- no information is discarded
        at aggregation time. Well-defined only because event_mode="dense"
        guarantees `msg`'s own edge axis IS the canonical dst_idx/src_idx
        order every trial (_dense_edge_features broadcasts self.dst_idx
        unchanged, unlike sparse mode's variable-length per-trial event
        list) -- concat_slot_idx (built in __init__ from that same dst_idx)
        maps each destination channel's k-th incident edge to its position
        in that fixed order, so no per-event dst_padded/valid_mask lookup is
        needed here at all, unlike "mean"/"gated_softmax" above. Motivated
        by the dense-mode flat-control finding that a flat (never-pooled)
        readout of dense_edge_conv's own output beat every graph/mean-pool
        variant tested -- see [[sparse-evidence-gnn-dense-flat-control-beats-
        graph]] and [[sparse-evidence-gnn-capacity-confound-refuted]] in
        project memory, and session_notes/run_logs/2026-08-10_sparse-
        evidence-gnn_dense-frozen-feature-capacity-aggregation-sweep.md for
        the screening/end-to-end numbers this option is built from
        (concat_h4: 0.9184 frozen-feature screening, seed 42, but only 0.8973
        end-to-end once dense_edge_conv co-adapts to it -- still below every
        flat-family number on record, so treat this as a validated, working
        OPTION, not a recommended default).
        """
        batch_size, max_count = dst_padded.shape
        valid_f = valid_mask.to(dtype)

        if self.event_aggregation == "mean":
            msg = msg * valid_f.unsqueeze(-1)
            evidence = torch.zeros(
                batch_size, self.n_channels, self.hidden_dim, dtype=dtype, device=msg.device
            )
            evidence.scatter_add_(
                1, dst_padded.unsqueeze(-1).expand(-1, -1, self.hidden_dim), msg
            )
            active = torch.zeros(batch_size, self.n_channels, dtype=dtype, device=msg.device)
            active.scatter_add_(1, dst_padded, valid_f)
            return evidence / active.clamp_min(1.0).unsqueeze(-1)

        if self.event_aggregation == "concat":
            assert max_count == self.dst_idx.numel(), (
                "event_aggregation='concat' assumes msg's edge axis is the "
                f"full canonical edge list ({self.dst_idx.numel()} edges); "
                f"got max_count={max_count}. Should be unreachable given "
                "__init__'s event_mode='dense' requirement for concat."
            )
            pad_row = torch.zeros(
                batch_size, 1, self.hidden_dim, dtype=dtype, device=msg.device
            )
            msg_padded = torch.cat([msg, pad_row], dim=1)  # [B, max_count+1, H]
            flat_slots = self.concat_slot_idx.reshape(-1)  # [n_channels*max_degree]
            gathered = msg_padded.index_select(1, flat_slots)  # [B, n_channels*max_degree, H]
            return gathered.reshape(
                batch_size, self.n_channels, self.concat_max_degree, self.hidden_dim
            )

        # "gated_softmax"
        gate_logits = self.event_gate(full_features).squeeze(-1)  # [B, max_count]
        gate_logits = gate_logits.masked_fill(~valid_mask, float("-inf"))

        group_idx = (batch_idx * self.n_channels + dst_padded).reshape(-1)
        logits_flat = gate_logits.reshape(-1)
        valid_flat = valid_mask.reshape(-1)
        n_groups = batch_size * self.n_channels

        group_max = torch.full((n_groups,), float("-inf"), dtype=dtype, device=msg.device)
        group_max.scatter_reduce_(0, group_idx, logits_flat, reduce="amax", include_self=True)
        max_per_event = group_max[group_idx]
        # A group with zero valid events has max_per_event=-inf; every entry
        # in it is invalid anyway and gets zeroed by valid_flat below, but
        # -inf - (-inf) is NaN, not just another -inf, so substitute a finite
        # placeholder before subtracting to keep the whole tensor NaN-free.
        safe_max = torch.where(
            torch.isfinite(max_per_event), max_per_event, torch.zeros_like(max_per_event)
        )
        shifted = torch.where(
            valid_flat, logits_flat - safe_max, torch.zeros_like(logits_flat)
        )
        exp_vals = torch.exp(shifted) * valid_flat.to(dtype)

        group_sum = torch.zeros(n_groups, dtype=dtype, device=msg.device)
        group_sum.scatter_add_(0, group_idx, exp_vals)
        sum_per_event = group_sum[group_idx]
        weights = (exp_vals / sum_per_event.clamp_min(1e-12)).reshape(batch_size, max_count)

        msg_weighted = msg * weights.unsqueeze(-1) * valid_f.unsqueeze(-1)
        evidence = torch.zeros(
            batch_size, self.n_channels, self.hidden_dim, dtype=dtype, device=msg.device
        )
        evidence.scatter_add_(
            1, dst_padded.unsqueeze(-1).expand(-1, -1, self.hidden_dim), msg_weighted
        )
        return evidence

    def _propagate_hops(self, node_state: torch.Tensor) -> torch.Tensor:
        """Runs (self.n_hops - 1) additional rounds of message passing over
        the 36 canonical edges, each round combining a node's current
        evidence with its neighbors' via the SAME edge topology
        _build_sparse_events routes real events through (src_idx/dst_idx) --
        propagating multi-hop coupling structure (e.g. channel A's evidence
        reaching channel C via B, two hops away) that the base (n_hops=1)
        pipeline cannot represent, since every event's message only ever
        lands on its own single destination channel (see _aggregate_events).

        Only called when self.n_hops > 1 (see forward()) -- n_hops=1 (the
        default) never reaches this method, so the pre-existing single-hop
        pipeline's output is bit-for-bit unchanged when this feature is left
        off.

        Each hop:
          1. Gathers every DIRECTED copy of each edge's two endpoint states
             (hop_src_idx/hop_dst_idx -- both i->j and j->i, since the
             canonical (i<j) edge list is an undirected topology, not a
             directed graph; see __init__'s buffer comment) and forms one
             message per direction via hop_message_mlp on
             [h_dst, h_src] (dst first -- matches the "message INTO dst,
             conditioned on src" framing below).
          2. Sums incoming messages per node (scatter_add) -- a node with no
             neighbors carrying evidence this hop simply receives an
             all-zero incoming message, not a crash or NaN.
          3. Updates each node's state via a GRUCell(incoming, prior_state) --
             a standard gated update (Gilmer et al. 2017 MPNN-style) that
             lets the network learn how much of a hop's neighbor evidence to
             fold in vs. keep its own prior-hop state, rather than
             overwriting outright.

        `node_state` is [B, n_channels, hidden_dim] (the same shape
        _aggregate_events returns); returns the same shape after n_hops-1
        rounds.
        """
        batch_size, n_channels, hidden_dim = node_state.shape
        src = self.hop_src_idx  # [2E], directed both ways
        dst = self.hop_dst_idx
        scatter_idx = dst.view(1, -1, 1).expand(batch_size, -1, hidden_dim)
        for _ in range(self.n_hops - 1):
            h_src = node_state.index_select(1, src)  # [B, 2E, H]
            h_dst = node_state.index_select(1, dst)  # [B, 2E, H]
            edge_msg = self.hop_message_mlp(torch.cat([h_dst, h_src], dim=-1))

            incoming = torch.zeros_like(node_state)
            incoming.scatter_add_(1, scatter_idx, edge_msg)

            flat_incoming = incoming.reshape(batch_size * n_channels, hidden_dim)
            flat_state = node_state.reshape(batch_size * n_channels, hidden_dim)
            node_state = self.hop_update(flat_incoming, flat_state).reshape(
                batch_size, n_channels, hidden_dim
            )
        return node_state

    def _aggregate_events_freq_indexed(
        self, msg, full_features, dst_padded, freq_idx_padded, valid_mask, batch_idx, dtype,
    ):
        """freq_aware_hops-only counterpart to _aggregate_events: identical
        "mean"/"gated_softmax" aggregation math (see that method's
        docstring), but grouped by (trial, destination channel, FREQUENCY
        BIN) instead of just (trial, destination channel) -- so a channel's
        events at different frequencies land in separate hidden_dim slots
        instead of being blended into one vector. Returns
        [B, n_channels, nfreqs, hidden_dim] instead of
        [B, n_channels, hidden_dim]. A (channel, freq) slot with zero valid
        events landing on it stays at its zero initialization, same
        convention as _aggregate_events.

        Only called from _propagate_hops_freq_aware (freq_aware_hops=True
        and n_hops>1); see that method and forward()'s branch. Deliberately
        NOT folded into _aggregate_events itself as an optional extra
        grouping key -- keeping this as its own method means the existing,
        already-relied-upon _aggregate_events is untouched by
        freq_aware_hops's addition, at the cost of duplicating its two
        aggregation-mode branches.
        """
        batch_size, max_count = dst_padded.shape
        valid_f = valid_mask.to(dtype)
        n_channels, nfreqs, hidden_dim = self.n_channels, self.nfreqs, self.hidden_dim
        n_groups = batch_size * n_channels * nfreqs
        # One flat group id per event, folding batch/channel/freq into a
        # single index -- group g's (batch, channel, freq) is recoverable as
        # (g // (n_channels*nfreqs), (g // nfreqs) % n_channels, g % nfreqs),
        # though nothing here needs that inverse; only the final reshape
        # does, implicitly, by construction order.
        group_idx = (batch_idx * n_channels + dst_padded) * nfreqs + freq_idx_padded
        group_flat = group_idx.reshape(-1)

        if self.event_aggregation == "mean":
            msg_v = (msg * valid_f.unsqueeze(-1)).reshape(-1, hidden_dim)
            flat = torch.zeros(n_groups, hidden_dim, dtype=dtype, device=msg.device)
            flat.index_add_(0, group_flat, msg_v)
            active = torch.zeros(n_groups, dtype=dtype, device=msg.device)
            active.index_add_(0, group_flat, valid_f.reshape(-1))
            evidence = flat / active.clamp_min(1.0).unsqueeze(-1)
            return evidence.reshape(batch_size, n_channels, nfreqs, hidden_dim)

        # "gated_softmax" -- same manual scatter-softmax as _aggregate_events,
        # just keyed by the (channel, freq) group_flat computed above instead
        # of a channel-only group.
        gate_logits = self.event_gate(full_features).squeeze(-1)  # [B, max_count]
        gate_logits = gate_logits.masked_fill(~valid_mask, float("-inf"))
        logits_flat = gate_logits.reshape(-1)
        valid_flat = valid_mask.reshape(-1)

        group_max = torch.full((n_groups,), float("-inf"), dtype=dtype, device=msg.device)
        group_max.scatter_reduce_(0, group_flat, logits_flat, reduce="amax", include_self=True)
        max_per_event = group_max[group_flat]
        safe_max = torch.where(
            torch.isfinite(max_per_event), max_per_event, torch.zeros_like(max_per_event)
        )
        shifted = torch.where(
            valid_flat, logits_flat - safe_max, torch.zeros_like(logits_flat)
        )
        exp_vals = torch.exp(shifted) * valid_flat.to(dtype)

        group_sum = torch.zeros(n_groups, dtype=dtype, device=msg.device)
        group_sum.scatter_add_(0, group_flat, exp_vals)
        sum_per_event = group_sum[group_flat]
        weights = (exp_vals / sum_per_event.clamp_min(1e-12)).reshape(batch_size, max_count)

        msg_weighted = (msg * weights.unsqueeze(-1) * valid_f.unsqueeze(-1)).reshape(-1, hidden_dim)
        flat = torch.zeros(n_groups, hidden_dim, dtype=dtype, device=msg.device)
        flat.index_add_(0, group_flat, msg_weighted)
        return flat.reshape(batch_size, n_channels, nfreqs, hidden_dim)

    def _propagate_hops_freq_aware(
        self, msg, full_features, dst_padded, freq_idx_padded, valid_mask, batch_idx, dtype,
    ):
        """freq_aware_hops counterpart to _propagate_hops -- see __init__'s
        docstring for the full motivation. _propagate_hops mixes each
        node's ALREADY-frequency-blended evidence vector (built by
        _aggregate_events, which pools every event on a channel -- any
        frequency -- into one hidden_dim vector before any hop ever runs),
        so it cannot represent "channel A received evidence at FREQUENCY X,
        and A's next-hop outgoing message is conditioned specifically on
        that freq-X evidence": by the time hops run, A's one hidden vector
        has no per-frequency identity left.

        This method instead builds a PER-FREQUENCY node state
        ([B, n_channels, nfreqs, hidden_dim], via
        _aggregate_events_freq_indexed) and runs (self.n_hops - 1) rounds
        of the same GRU-gated message passing as _propagate_hops, but
        independently per frequency slice (weight-tied across frequency,
        via hop_message_mlp_freq/hop_update_freq -- separate weights from
        hop_message_mlp/hop_update, see __init__'s comment on why they
        aren't shared). A hop's outgoing message for channel A's
        frequency-X slot is therefore a function only of A's OWN
        frequency-X state and its freq-X neighbors' states, never blended
        with A's other frequencies. That makes same-frequency directed
        chains structurally representable -- e.g. an event routing
        evidence INTO node A at freq X (direction is the sign of that
        event's stored angle -- see _build_sparse_events' two-sided gate)
        shapes A's freq-X slot, and the very next hop can route evidence
        OUT of A toward a neighbor as a function of that same slot. This is
        NOT a guarantee the model does this, and it has no explicit notion
        of event order/timing beyond what "hop count" already encodes --
        it is a learned weighting (hop_message_mlp_freq/hop_update_freq)
        given the STRUCTURAL capacity for such chains, same as every other
        aggregation choice in this pipeline (e.g. event_aggregation=
        "gated_softmax"'s "given the freedom, does training use it"
        precedent -- see that param's docstring).

        Finishes by pooling the frequency axis back down to
        [B, n_channels, hidden_dim] via a learned softmax-attention pool
        (self.freq_pool), so the return shape matches _propagate_hops' and
        sparse_classifier's input width is unaffected by freq_aware_hops.

        Only called when self.freq_aware_hops and self.n_hops > 1 (see
        forward()).
        """
        node_state = self._aggregate_events_freq_indexed(
            msg, full_features, dst_padded, freq_idx_padded, valid_mask, batch_idx, dtype,
        )  # [B, n_channels, nfreqs, hidden_dim]
        batch_size, n_channels, nfreqs, hidden_dim = node_state.shape
        src = self.hop_src_idx  # [2E], directed both ways
        dst = self.hop_dst_idx
        scatter_idx = dst.view(1, -1, 1, 1).expand(batch_size, -1, nfreqs, hidden_dim)
        for _ in range(self.n_hops - 1):
            h_src = node_state.index_select(1, src)  # [B, 2E, F, H]
            h_dst = node_state.index_select(1, dst)  # [B, 2E, F, H]
            edge_msg = self.hop_message_mlp_freq(torch.cat([h_dst, h_src], dim=-1))

            incoming = torch.zeros_like(node_state)
            incoming.scatter_add_(1, scatter_idx, edge_msg)

            flat_incoming = incoming.reshape(batch_size * n_channels * nfreqs, hidden_dim)
            flat_state = node_state.reshape(batch_size * n_channels * nfreqs, hidden_dim)
            node_state = self.hop_update_freq(flat_incoming, flat_state).reshape(
                batch_size, n_channels, nfreqs, hidden_dim
            )

        pool_weights = torch.softmax(self.freq_pool(node_state), dim=2)  # [B, C, F, 1]
        return (node_state * pool_weights).sum(dim=2)  # [B, n_channels, hidden_dim]

    def _shuffle_dense_edge_time(self, dense_edge_raw: torch.Tensor) -> torch.Tensor:
        """shuffle_time_order=True's negative control (dense_edge_temporal_
        mode="rnn" only -- see that param's __init__ docstring): independently
        permutes each (batch, edge, frequency) cell's T index order in
        `dense_edge_raw` ([B, 4, E, T, F]) before it ever reaches
        dense_edge_conv, so the GRU sees the same set of timesteps in a
        scrambled order rather than their real temporal sequence.

        The SAME permutation is applied across the 4 stack channels (coh,
        sinφ, cosφ, significance) for a given (batch, edge, frequency, T
        position) -- so a shuffled position's own 4 values stay the
        physically-consistent tuple they originally were, only their
        ordering RELATIVE TO OTHER timesteps is destroyed. This is
        deliberate: shuffling the 4 channels independently of each other
        would additionally break coherence/phase/significance's own
        internal relationship at every single timestep, confounding "does
        temporal ORDER matter" with "does the coherence/phase/significance
        relationship matter," which is not what this control is for.

        Draws from torch's global RNG (no dedicated generator/seed) --
        same precedent common.py's augment_paired_cwt_batch (torch.rand/
        torch.randint, unseeded) already uses for other forward-time
        stochastic augmentation in this codebase. Independent across
        (batch, edge, frequency), and independent every forward() call
        (train and eval alike) -- see shuffle_time_order's docstring for
        why this is a fair comparison to dense_edge_temporal_mode="rnn"
        without this flag.
        """
        batch_size, n_stack, num_edges, n_time, nfreqs = dense_edge_raw.shape
        # One independent random permutation of [0, T) per (B, E, F),
        # shared across the 4 stack channels -- argsort of iid random keys
        # is the standard vectorized "random permutation along an axis"
        # trick (avoids a python-level loop over B*E*F).
        perm = torch.argsort(
            torch.rand(batch_size, num_edges, nfreqs, n_time, device=dense_edge_raw.device),
            dim=-1,
        )  # [B, E, F, T], values in [0, T)
        gather_index = (
            perm.permute(0, 1, 3, 2)  # [B, E, T, F]
            .unsqueeze(1)
            .expand(batch_size, n_stack, num_edges, n_time, nfreqs)
        )
        return torch.gather(dense_edge_raw, dim=3, index=gather_index)

    def _dense_edge_features(self, dense_edge_raw):
        """event_mode="dense" forward()-time step: runs the TRAINABLE
        dense_edge_conv over the precomputed, non-trainable [B, 4, E, T, F]
        coherence/phase/significance stack (see _build_dense_edge_input --
        the conv itself must run every forward() call, unlike sparse mode's
        event-building, since it has learnable weights), producing one
        feature vector per edge. Packages that into the exact same
        (events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask,
        batch_idx) shape sparse mode's _aggregate_events/_propagate_hops
        already consume, so everything downstream of event-building
        (feature_ablation, sparse_message_mlp, aggregation, hops,
        sparse_classifier) is shared code, not a duplicated dense-mode copy
        -- see event_mode's __init__ docstring.

        Every one of the E canonical edges is "valid" every trial -- dense
        mode has no notion of an edge not firing, unlike a discrete event
        list that may be empty on some edges -- so max_count == E and
        valid_mask is all-True; src_padded/dst_padded are just
        self.src_idx/self.dst_idx broadcast across the batch (fixed
        topology, not computed per-trial the way sparse events' src/dst are).
        freq_idx_padded is an unused zero-filled placeholder: __init__
        rejects freq_aware_hops=True together with event_mode="dense" (dense
        features have no discrete per-event frequency bin), so nothing ever
        reads this array in dense mode.
        """
        device = dense_edge_raw.device
        batch_size_actual, c_in, num_edges, n_time, nfreqs = dense_edge_raw.shape
        if self.dense_edge_temporal_mode == "rnn" and self.shuffle_time_order:
            dense_edge_raw = self._shuffle_dense_edge_time(dense_edge_raw)
        # [B, C_in, E, T, F] -> [B, C_in, F, E, T] -> [B, C_in*F, E, T] --
        # folds frequency into the conv's input channels (see
        # _build_dense_feature_conv's docstring) while leaving E as the
        # untouched, weight-shared spatial axis.
        conv_in = dense_edge_raw.permute(0, 1, 4, 2, 3).reshape(
            batch_size_actual, c_in * nfreqs, num_edges, n_time
        )
        conv_out = self.dense_edge_conv(conv_in)  # [B, dense_conv_out_channels, E, 1]
        events_padded = conv_out.squeeze(-1).permute(0, 2, 1)  # [B, E, dense_conv_out_channels]

        src_padded = self.src_idx.unsqueeze(0).expand(batch_size_actual, -1)
        dst_padded = self.dst_idx.unsqueeze(0).expand(batch_size_actual, -1)
        freq_idx_padded = torch.zeros_like(src_padded)
        valid_mask = torch.ones(
            batch_size_actual, num_edges, dtype=torch.bool, device=device
        )
        batch_idx = torch.arange(batch_size_actual, device=device).unsqueeze(1).expand(
            -1, num_edges
        )
        return events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask, batch_idx

    def _temporal_graph_node_states(
        self, dense_edge_raw: torch.Tensor, channel_emb: torch.Tensor
    ) -> torch.Tensor:
        """event_mode="temporal_graph" forward()-time step: the genuine
        "evolving graph" counterpart to _dense_edge_features -- see
        event_mode's __init__ docstring for the full mechanism/rationale.
        Runs on the exact same precomputed, non-trainable [B, 4, E, T, F]
        stack `dense_edge_raw` that "dense" mode consumes (same
        dense_edge_time_downsample, COI mask, smoothing -- see
        _build_dense_edge_input), but instead of pooling T away with a
        Conv2d + AdaptiveAvgPool2d stack, walks forward through it one
        timestep at a time with a persistent per-node GRU hidden state.

        Returns `evidence`, shape [B, n_channels, hidden_dim] -- EXACTLY
        the shape _aggregate_events' "mean" branch returns, so the caller
        (forward()) can feed it straight into the same n_hops>1 propagation
        and sparse_classifier every other event_mode already shares.
        """
        batch_size_actual, c_in, num_edges, n_time, nfreqs = dense_edge_raw.shape

        # Same "fold frequency into channels, leave the edge axis untouched"
        # reshape _dense_edge_features' conv_in uses -- [B, 4, E, T, F] ->
        # [B, 4, F, E, T] -> [B, 4F, E, T] -> [B, E, T, 4F], the last
        # transpose putting the feature axis last so temporal_edge_proj's
        # per-position nn.Linear can process every (edge, timestep) at once
        # (no explicit Python loop needed for this step -- only the GRU
        # itself is inherently sequential, and nn.GRU handles that
        # internally rather than via a Python-level loop here).
        folded = dense_edge_raw.permute(0, 1, 4, 2, 3).reshape(
            batch_size_actual, c_in * nfreqs, num_edges, n_time
        )  # [B, 4F, E, T]
        edge_seq_in = folded.permute(0, 2, 3, 1)  # [B, E, T, 4F]
        edge_embed = self.temporal_edge_proj(edge_seq_in)  # [B, E, T, temporal_graph_edge_dim]

        # channel_emb is [B, n_channels, channel_embed_dim] -- fixed
        # topology (self.src_idx/self.dst_idx), so plain fancy-indexing
        # gathers each edge's endpoint embeddings without the batch_idx
        # dance sparse mode's variable-length per-trial event lists need.
        # Broadcast (not copy, via expand) across T since channel_emb has no
        # time axis of its own.
        src_emb = channel_emb[:, self.src_idx, :].unsqueeze(2).expand(-1, -1, n_time, -1)
        dst_emb = channel_emb[:, self.dst_idx, :].unsqueeze(2).expand(-1, -1, n_time, -1)

        # Same feature_ablation semantics forward()'s sparse/dense branch
        # applies (see that method's docstring) -- duplicated here rather
        # than shared, since this method's tensors carry an extra T axis
        # the shared code doesn't expect. See _aggregate_events_freq_indexed
        # for the same "isolated duplication over shared-code contortion"
        # precedent elsewhere in this file.
        if self.feature_ablation == "zero_event_features":
            edge_embed = torch.zeros_like(edge_embed)
        elif self.feature_ablation == "zero_channel_embed":
            src_emb = torch.zeros_like(src_emb)
            dst_emb = torch.zeros_like(dst_emb)

        full_features = torch.cat([edge_embed, src_emb, dst_emb], dim=-1)  # [B, E, T, message_in]
        msg = self.sparse_message_mlp(full_features)  # [B, E, T, hidden_dim] -- SAME weights
        # every other event_mode's message step uses; nn.Linear applies to
        # the last dim regardless of the extra T axis here.

        # Per-timestep "mean" aggregation to nodes -- the existing
        # _aggregate_events' "mean" branch generalized with an extra T axis
        # (event_mode="temporal_graph" requires event_aggregation="mean",
        # enforced in __init__, so there is no other branch to support
        # here). Every canonical edge is always "active" (same property
        # event_mode="dense" already has -- see _dense_edge_features), so
        # the divisor is the FIXED temporal_node_in_degree buffer rather
        # than a per-trial active-count, unlike _aggregate_events' own
        # scatter_add-based count.
        node_seq = torch.zeros(
            batch_size_actual, self.n_channels, n_time, self.hidden_dim,
            dtype=msg.dtype, device=msg.device,
        )
        dst_idx_expand = self.dst_idx.view(1, -1, 1, 1).expand(
            batch_size_actual, -1, n_time, self.hidden_dim
        )
        node_seq.scatter_add_(1, dst_idx_expand, msg)
        node_seq = node_seq / self.temporal_node_in_degree.view(1, -1, 1, 1)

        # Weight-shared-across-nodes GRU: nodes folded into the batch dim
        # (same spirit as dense_edge_conv's/_DenseEdgeGRUTemporal's own
        # edge-axis weight sharing -- see event_mode's docstring), producing
        # ONE persistent hidden state per node that's genuinely updated
        # timestep by timestep across the real T' sequence. h_n (final
        # hidden state) is the "T walked through to the end" counterpart to
        # dense mode's AdaptiveAvgPool2d((None, 1)) -- an actual summary
        # carried through memory, not a pooled/convolved snapshot.
        gru_in = node_seq.reshape(batch_size_actual * self.n_channels, n_time, self.hidden_dim)
        _, h_n = self.temporal_node_gru(gru_in)  # h_n: [1, B*n_channels, hidden_dim]
        evidence = h_n.squeeze(0).reshape(batch_size_actual, self.n_channels, self.hidden_dim)
        return evidence

    def forward(self, raw_x, *event_inputs):
        """Trainable forward pass. `event_inputs` depends on self.event_mode:

        "sparse" (default): (events_padded, src_padded, dst_padded,
        freq_idx_padded, valid_mask) -- PRECOMPUTED sparse events (see
        compute_events()); this branch is bit-identical to the pre-
        event_mode forward() signature. `to_float_tensors` upstream casts
        everything to float for DataLoader/TensorDataset batching, so
        src_padded/dst_padded/freq_idx_padded/valid_mask arrive as float and
        need casting back here.

        "dense": (dense_edge_raw,) -- the precomputed, non-trainable
        [B, 4, E, T, F] coherence/phase/significance stack (see
        _build_dense_edge_input/compute_dense_edge_input); dense_edge_conv
        (trainable, unlike sparse event-building) runs on it right here via
        _dense_edge_features, which also builds the same
        events_padded/src_padded/dst_padded/freq_idx_padded/valid_mask/
        batch_idx shape the "sparse" branch produces, so the rest of this
        method is identical between the two modes.

        "temporal_graph": (dense_edge_raw,) -- the SAME precomputed stack
        "dense" consumes (reused unchanged, see event_mode's own docstring),
        but processed step-by-step through time by
        _temporal_graph_node_states instead of pooled by dense_edge_conv.
        That method already returns `evidence` directly (in the same
        [B, n_channels, hidden_dim] shape _aggregate_events' "mean" branch
        produces), so this branch skips straight to the n_hops>1/readout
        tail every mode shares -- there is no per-edge events_padded/msg/
        _aggregate_events call to make here (see that method's own
        docstring for why: aggregation already happened once per timestep,
        inside the sequence the GRU walked through).
        """
        batch_size = raw_x.shape[0]
        channel_emb = self.channel_encoder(raw_x)

        if self.event_mode == "sparse":
            events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask = event_inputs
            src_padded = src_padded.long()
            dst_padded = dst_padded.long()
            freq_idx_padded = freq_idx_padded.long()
            valid_mask = valid_mask.bool()
            max_count = events_padded.shape[1]
            batch_idx = torch.arange(batch_size, device=raw_x.device).unsqueeze(1).expand(
                -1, max_count
            )
        elif self.event_mode == "dense":
            (dense_edge_raw,) = event_inputs
            events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask, batch_idx = (
                self._dense_edge_features(dense_edge_raw.to(raw_x.dtype))
            )
        else:  # "temporal_graph"
            (dense_edge_raw,) = event_inputs

        if self.event_mode == "temporal_graph":
            evidence = self._temporal_graph_node_states(
                dense_edge_raw.to(raw_x.dtype), channel_emb
            )
            # Every canonical edge is always "active" every timestep in
            # this mode too (same property event_mode="dense" already has
            # -- see _dense_edge_features) -- matches the event_density
            # formula below's existing "constant 1/self.nfreqs" convention
            # for that property, rather than introducing a different one.
            valid_edge_count = float(batch_size * self.src_idx.numel())
        else:
            src_emb = channel_emb[batch_idx, src_padded]
            dst_emb = channel_emb[batch_idx, dst_padded]

            # See __init__'s feature_ablation docstring -- zeroing happens
            # here, AFTER channel_encoder/event-building have already run,
            # so this is a pure ablation of what sparse_message_mlp sees,
            # not of which events exist or of gradient flow into the
            # zeroed-out submodule (channel_encoder/event-building still
            # run and still get real gradients through whichever branch
            # isn't zeroed... except the zeroed branch itself gets none,
            # since torch.zeros_like detaches it from the graph -- e.g.
            # "zero_event_features" means channel_encoder is the only thing
            # "coh"/"phase" thresholds route signal to via topology (which
            # edges/how many events, still real), not via feature content).
            if self.feature_ablation == "zero_event_features":
                events_padded = torch.zeros_like(events_padded)
            elif self.feature_ablation == "zero_channel_embed":
                src_emb = torch.zeros_like(src_emb)
                dst_emb = torch.zeros_like(dst_emb)

            full_features = torch.cat([events_padded, src_emb, dst_emb], dim=-1)
            msg = self.sparse_message_mlp(full_features)

            evidence = self._aggregate_events(
                msg, full_features, dst_padded, valid_mask, batch_idx, raw_x.dtype
            )
            valid_edge_count = float(valid_mask.sum().item())

        if self.n_hops > 1:
            if self.freq_aware_hops:
                # Entirely replaces `evidence` (the collapsed aggregation
                # above is simply unused in this branch, not a base to add
                # to) -- see _propagate_hops_freq_aware's docstring for why
                # a per-frequency hop pipeline needs to start from its own
                # per-frequency aggregation (_aggregate_events_freq_indexed)
                # rather than the already-blended `evidence` computed above.
                # Unreachable when event_mode="temporal_graph" -- __init__
                # rejects freq_aware_hops=True together with that mode, so
                # msg/full_features/dst_padded/freq_idx_padded/valid_mask/
                # batch_idx (only defined in the "sparse"/"dense" branch
                # above) are always available here.
                evidence = self._propagate_hops_freq_aware(
                    msg, full_features, dst_padded, freq_idx_padded, valid_mask,
                    batch_idx, raw_x.dtype,
                )
            else:
                evidence = self._propagate_hops(evidence)

        # "concat" evidence is [B, n_channels, concat_max_degree, hidden_dim]
        # (never touched by the n_hops>1 branch above -- __init__ rejects
        # event_aggregation="concat" with n_hops!=1, so this shape is exactly
        # what _aggregate_events' "concat" branch just returned); every other
        # mode (including "temporal_graph", which __init__ restricts to
        # event_aggregation="mean") is [B, n_channels, hidden_dim], same as
        # always.
        if self.event_aggregation == "concat":
            readout = evidence.reshape(
                batch_size, self.n_channels * self.concat_max_degree * self.hidden_dim
            )
        else:
            readout = evidence.reshape(batch_size, self.n_channels * self.hidden_dim)
        logits = self.sparse_classifier(readout)
        # matches the old event_density = n_runs / max(B*E*F, 1) exactly:
        # valid_mask.sum() over a batch IS that batch's n_runs. In "dense"
        # and "temporal_graph" modes every edge always "fires" (see
        # _dense_edge_features / _temporal_graph_node_states), so this
        # collapses to the constant 1/self.nfreqs every call -- not a
        # meaningful density signal in either mode, just kept so the
        # aux-metric return contract (a float) stays identical across modes.
        event_density = valid_edge_count / max(
            batch_size * self.src_idx.numel() * self.nfreqs, 1
        )
        return logits, event_density


class SparseEvidenceGNNClassifier(_BaseCWTGNNClassifier):
    """sklearn/MOABB wrapper around SparseEvidenceGNNCore."""

    model_label = "Sparse-Evidence"
    # This model's forward() aux value is n_runs / (batch*edges*freqs) -- an
    # unbounded average burst-count-per-row, NOT a bounded [0,1] fraction
    # like WCTEvidenceGNN's edge_density. Override the shared log label so
    # it doesn't misleadingly read as a percentage (it can exceed 1.0).
    aux_metric_name = "bursts_per_row"

    def __init__(
        self,
        # Epilepsy fork: upstream's 250/8-35Hz defaults are BNCI2014_001
        # motor-imagery specifics (native sampling rate; mu/beta ERD/ERS
        # band) that don't apply here. sampling_rate=256 matches CHB-MIT's
        # native rate. 1-40Hz is a broad, NOT dataset-tuned placeholder --
        # covers delta through low-gamma instead of assuming ictal activity
        # lives in any one MI-specific sub-band -- pick a real value once
        # there's evidence for one.
        sampling_rate: int = 256,
        lowest: float = 1.0,
        highest: float = 40.0,
        nfreqs: int = 16,
        # None = native resolution (no post-CWT resample). A non-None value
        # here previously destroyed real signal above ~n_time/(2*trial_secs)
        # Hz via scipy.signal.resample on the complex coefficients themselves
        # (verified with a clean 30Hz test tone: magnitude 0.81 natively ->
        # 0.006 after resample to 200). It would also make the COI mask in
        # SparseEvidenceGNNCore wrong (see the warning below).
        cwt_resample_n_time: int | None = None,
        coherence_threshold: float = 0.5,
        phase_threshold_deg: float = 30.0,
        hidden_dim: int = 8,
        channel_embed_dim: int = 8,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 0.1,
        normalize_input: bool = True,
        noise_augmentation_enabled: bool = False,
        noise_apply_prob: float = 0.0,
        noise_strength: float = 0.0,
        noise_bank_size: int = 128,
        noise_bank_seed: int | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        last_batch_min_ratio: float = 0.0,
        selector_alpha_val_update_rate: float = 1.0,
        optimizer_step_batch_size: int | None = None,
        optimizer_step_batch_mode: str = "credit",
        optimizer_step_remainder_policy: str = "flush",
        smooth_kernel_sigma: tuple[float | None, float | None] = (None, None),
        smooth_kernel_size: tuple[int | None, int] = (5, 3),
        # 2026-08-09 mu-band investigation: smooth_kernel_size's time width
        # is one FLAT number of raw samples applied at every frequency.
        # A fixed 5-sample window is ~16-24% of one mu-band (8-13Hz) cycle
        # at native 250Hz, but ~60% of one 30Hz cycle -- so it provides far
        # less real temporal averaging (fewer independent looks at the
        # phase relationship) at low frequency than at high frequency,
        # which is the mechanism behind coherence saturating near the
        # surrogate null specifically in the mu band (see the 2026-08-08
        # session notes' Arc 5 and this feature's own follow-up
        # investigation). Standard "scale-adaptive" wavelet-coherence
        # smoothing (Torrence & Webster 1999): a time-kernel width
        # proportional to each frequency's own oscillation PERIOD, so every
        # frequency gets smoothed over roughly the same NUMBER of cycles
        # instead of the same number of raw samples. When True, replaces
        # smooth_kernel_size[0] (the flat time width) with a per-frequency
        # width of `round(scale_adaptive_cycles * sampling_rate / freq)`,
        # clipped to [3, scale_adaptive_max_kernel] and rounded to odd;
        # smooth_kernel_size[1] (frequency-axis smoothing) is untouched
        # either way. False (default) preserves the original flat-kernel
        # behavior exactly -- this is purely additive and does not change
        # the windowed WCT-Evidence-GNN pipeline, only
        # SparseEvidenceGNNCore's own _smooth_wct_maps_scale_adaptive path.
        # See _scale_adaptive_time_kernel.
        scale_adaptive_smoothing: bool = False,
        scale_adaptive_cycles: float = 1.5,
        scale_adaptive_max_kernel: int = 101,
        coi_enabled: bool = True,
        # Diagnostic: independently resample ONLY raw_x (the signal fed to
        # ChannelSignalEncoder), leaving w_real/w_imag/coherence/events at
        # whatever resolution cwt_resample_n_time implies. Unlike
        # cwt_resample_n_time, this is safe to set alongside native-res
        # coherence -- compute_events() never touches raw_x, so this cannot
        # corrupt the COI mask or coherence estimate. Added specifically to
        # isolate whether the old pipeline's ~0.80 (vs ~0.76 now) traces to
        # ChannelSignalEncoder seeing a resampled (T=200) vs native (T~1001)
        # raw signal, independent of everything already ruled out in the
        # coherence/event pathway (density, COI, kernel, thresholds all
        # tested with no effect -- see run_pipelines.py's SPARSE_FAMILY_PARAMS).
        raw_x_resample_n_time: int | None = None,
        # Real fix for the same finding raw_x_resample_n_time diagnosed:
        # ChannelSignalEncoder's two kernel_size=9 convs give a fixed
        # 17-sample receptive field, too short (~68ms at native 250Hz) to
        # span even one mu-band cycle (~83-125ms). Dilation grows that
        # window in real time without resampling/discarding any of the raw
        # signal (unlike raw_x_resample_n_time) and without growing the
        # kernel's parameter count. See ChannelSignalEncoder's docstring.
        channel_encoder_dilation: int = 1,
        # 2026-08-09 ablation, forwarded to SparseEvidenceGNNCore -- see that
        # class's __init__ docstring. 2026-08-17: hard-disabled to
        # "zero_channel_embed" only (the only value SparseEvidenceGNNCore's
        # __init__ now accepts) -- channel embeddings kept getting switched
        # on unintentionally via the old "none" default, so nothing above
        # this class can turn them back on.
        feature_ablation: str = "zero_channel_embed",
        # 2026-08-10, forwarded to SparseEvidenceGNNCore -- see that class's
        # __init__ docstring. "mean" (default) is the original behavior;
        # "gated_softmax" lets sparse_message_mlp's events compete for
        # weight within each destination channel instead of contributing
        # equally, via a learned per-event gate. "concat" (event_mode="dense"
        # AND n_hops=1 only) keeps every incident edge's own message
        # distinct instead of pooling them into one vector -- see Core's
        # docstring for the full rationale and validated numbers.
        event_aggregation: str = "mean",
        # 2026-08-10, forwarded to SparseEvidenceGNNCore -- see that class's
        # __init__ docstring / _propagate_hops. 1 (default) is the original
        # single-hop behavior; n_hops=K>1 runs K-1 additional rounds of
        # GRU-gated message passing over the canonical edge topology after
        # the base per-event evidence is built, letting evidence reach
        # channels more than one edge away from where an event actually
        # landed.
        n_hops: int = 1,
        # 2026-08-10, forwarded to SparseEvidenceGNNCore -- see that class's
        # __init__ docstring / _propagate_hops_freq_aware. False (default) is
        # the original freq-blind hop behavior (_propagate_hops); only takes
        # effect when n_hops>1. True keeps each channel's evidence separate
        # PER FREQUENCY through the hop rounds instead of blending every
        # frequency into one vector before hops run, making same-frequency
        # directed chains (e.g. an event routing evidence INTO a channel at
        # freq X, that channel routing evidence back OUT at the same freq X
        # the next hop) structurally representable.
        freq_aware_hops: bool = False,
        # 2026-08-10, forwarded to SparseEvidenceGNNCore -- see that class's
        # __init__ docstring for the full rationale. "sparse" (default)
        # preserves this pipeline's existing hard-threshold-and-consolidate
        # event pipeline exactly, bit-identical. "dense" replaces event
        # BUILDING with a learned conv stack (dense_edge_conv) over the same
        # already-fixed (native-resolution, no cwt_resample_n_time, COI-
        # masked) coherence/phase arrays, plus a continuous significance
        # channel derived from whatever coherence_threshold_mode/
        # surrogate_percentile below resolve to -- everything downstream of
        # event-building (channel_encoder, sparse_message_mlp,
        # _aggregate_events, n_hops propagation, sparse_classifier) is
        # unaffected. The dense_conv_* params below only take effect when
        # event_mode="dense".
        #
        # 2026-08-11: "temporal_graph" is a third option -- see
        # SparseEvidenceGNNCore's own event_mode docstring for the full
        # mechanism/rationale. It reuses _build_dense_edge_input's [B, 4, E,
        # T, F] stack exactly like "dense" (so dense_edge_time_downsample
        # below applies to it too), but processes it step-by-step through
        # time via a per-node nn.GRU instead of pooling with dense_edge_conv
        # -- a genuine test of temporal graph propagation, distinct from
        # both "dense" (pools time away in one shot) and n_hops>1 (adds
        # depth ACROSS the graph within one already-pooled snapshot, never
        # touching time). Requires event_aggregation="mean" (raises
        # otherwise) and rejects freq_aware_hops=True, same as "dense". The
        # dense_conv_* params below have no meaning for this mode (it builds
        # its own temporal_edge_proj/temporal_node_gru instead of
        # dense_edge_conv) -- only temporal_graph_edge_dim (below) applies.
        event_mode: Literal["sparse", "dense", "temporal_graph"] = "sparse",
        dense_conv_kernel_size: int = 5,
        dense_conv_pool_size: int = 4,
        dense_conv_intermediate_channels: int = 32,
        dense_conv_intermediate_channels_reduced: int | None = None,
        dense_conv_out_channels: int = 8,
        # 2026-08-10, forwarded to SparseEvidenceGNNCore -- see that class's
        # __init__ docstring. 1 (default) is native resolution, bit-identical
        # to before this param existed; k>1 average-pools the already-
        # computed, already-COI-masked coherence/phase/significance stack's
        # time axis by that factor before dense_edge_conv ever sees it,
        # cutting its per-epoch cost roughly linearly in k. event_mode="dense"
        # only -- raises otherwise (see SparseEvidenceGNNCore's docstring).
        dense_edge_time_downsample: int = 1,
        # 2026-08-11, forwarded to SparseEvidenceGNNCore -- see that class's
        # docstring for the full rationale ("single time-averaged coherence
        # graph per trial"). False (default) is bit-identical to before this
        # param existed. event_mode="dense" only, mutually exclusive with
        # dense_edge_time_downsample != 1, and requires
        # dense_conv_kernel_size == dense_conv_pool_size == 1 -- all three
        # enforced here too (duplicated validation, same precedent as
        # event_aggregation="concat"'s own checks above) so they surface at
        # classifier construction time, not only once fit() builds the Core.
        time_averaged_graph: bool = False,
        # 2026-08-11, forwarded to SparseEvidenceGNNCore -- see that class's
        # docstring / _DenseEdgeGRUTemporal for the full rationale. "conv"
        # (default) is dense_edge_conv's original Conv2d temporal stack,
        # bit-identical to before this param existed. "rnn" swaps it for a
        # GRU that integrates the full T' sequence with memory instead of
        # Conv2d's small fixed local window. event_mode="dense" only --
        # raises otherwise (duplicated validation, same precedent as
        # time_averaged_graph's own checks above).
        dense_edge_temporal_mode: Literal["conv", "rnn"] = "conv",
        # 2026-08-11, forwarded to SparseEvidenceGNNCore -- see that class's
        # docstring / shuffle_time_order. The negative control for
        # dense_edge_temporal_mode="rnn": independently scrambles each
        # (edge, frequency)'s T' order before the GRU sees it. False
        # (default) changes nothing. dense_edge_temporal_mode="rnn" only --
        # raises otherwise.
        shuffle_time_order: bool = False,
        # 2026-08-11, forwarded to SparseEvidenceGNNCore -- see that class's
        # docstring / event_mode's own docstring for the full rationale.
        # event_mode="temporal_graph" only; a per-edge, per-timestep
        # embedding width (one small Linear+GELU, not a deep conv stack).
        # Default (8) matches dense_conv_out_channels's own default purely
        # for comparable sizing, not because it's been tuned.
        temporal_graph_edge_dim: int = 8,
        channel_subset: list[int] | list[str] | None = None,
        # Epilepsy fork only -- see _BaseCWTGNNClassifier._init_cwt_gnn_classifier's
        # docstring (xwt_phase_gnn_classifier.py) for why this defaults to
        # True here instead of upstream's hardcoded False.
        use_class_weights: bool = True,
        # Epilepsy fork only -- see cwt_window_cache.py's docstring / this
        # param's matching one on _init_cwt_gnn_classifier. Restored
        # 2026-08-22 (removed 2026-08-21, see cwt_window_cache.py's module
        # docstring for that removal's rationale and this restoration's
        # own reasoning -- the earlier removal was measured on a fast
        # Linux/Runpod GPU pod where recompute won; this reopens the
        # question for a different (Windows/WDDM) machine where disk I/O
        # and recompute cost may trade off differently).
        cwt_cache: dict | None = None,
        # Epilepsy fork only, 2026-08-15 (restored 2026-08-22) -- see
        # dense_edge_cache.py's docstring. Disk directory for the
        # coherence_threshold_mode="fixed" dense-edge-input cache
        # ([4, E, T, F] per trial, keyed by raw window content + config).
        # None (default) disables it -- every trial recomputed fresh, same
        # as before this param existed. Only ever consulted when
        # coherence_threshold_mode == "fixed" (see
        # _precompute_dense_edge_inputs): the invariance-to-per-fold-
        # normalization proof this cache relies on doesn't cover
        # "surrogate"/"surrogate_cluster", which calibrate against
        # raw_x_native directly.
        dense_edge_cache_dir: str | None = None,
        # 2026-08-19: Runpod deployment finding -- _precompute_sparse_events/
        # _precompute_dense_edge_inputs both hardcode chunk=min(batch_size,4)
        # trials per torch call, chosen (see those methods' comments) to keep
        # peak RSS around ~2GB on a ~16-17GB-RAM reference machine regardless
        # of batch_size. That's a real, deliberate memory guard, NOT a
        # parallelism tuning knob -- but on a machine with much more RAM
        # (measured: a 32-core/192GB Runpod pod), the same fixed chunk=4 cap
        # also caps how much work compute_dense_edge_input's batched torch
        # ops get per call, which caps how well torch's intra-op threading
        # (16-32 threads available, confirmed via torch.get_num_threads())
        # can actually parallelize -- observed ~1.3/32 cores utilized at
        # chunk=4. None (default) preserves the original min(batch_size, 4)
        # behavior unchanged everywhere this isn't explicitly overridden. A
        # caller on a high-RAM machine can raise this (e.g. to batch_size
        # itself) to trade memory headroom for CPU utilization; scale peak
        # RSS roughly linearly from the ~2GB-at-4 reference point when
        # choosing a value.
        precompute_chunk_size: int | None = None,
        # 2026-08-22: opt-in torch.compile of the throwaway dense-edge
        # helper's compute_dense_edge_input, backend="cudagraphs" --
        # captures the fixed-shape per-chunk kernel sequence and replays it
        # without re-dispatching through the CPU/WDDM driver each call,
        # directly targeting the many-small-ops launch overhead profiling
        # measured on this pipeline (see _sync_device's docstring and the
        # 2026-08-22 session notes). NOT the default mode="reduce-overhead"
        # (Inductor) path -- that needs Triton for its kernel-fusion codegen
        # (torch._inductor.exc.TritonMissing, confirmed on this Windows
        # box, no working wheel here), so this gets the launch-overhead win
        # (CUDA graph replay) without the fusion win (no Triton available to
        # do it). CUDA only (self.device resolves to "cuda") -- the
        # cudagraphs backend isn't meaningfully supported on MPS/CPU; False
        # there regardless of this flag. Off by default: first-call
        # compilation/graph capture is a real, one-time cost. A captured
        # graph is also locked to the exact input shape it was captured
        # for -- NOT a graceful recompile on a shape change, confirmed
        # directly (RuntimeError inside cudagraph_trees._copy_inputs_and_
        # remove_from_src when a short final chunk hit the graph captured
        # for a full one) -- so _precompute_dense_edge_inputs only routes
        # full-size chunks through the compiled path and always uses the
        # plain method for a short remainder chunk (routine here: the last
        # batch of a fold is rarely an exact multiple of the chunk size).
        compile_dense_edge_helper: bool = False,
        # 2026-08-22: opt-in torch.autocast(dtype=torch.bfloat16) around the
        # non-trainable dense-edge helper's compute_dense_edge_input call
        # (already under torch.no_grad() -- this only affects numerical
        # precision of the coherence/cross-spectrum math, never gradients).
        # Targets the OTHER lever compile_dense_edge_helper doesn't: most of
        # this stage (_full_edge_wct_maps's elementwise products,
        # _smooth_wct_maps's conv2d, COI masking, the stack) is memory-
        # bandwidth-bound on tensors shaped [chunk, edges, T, F] -- halving
        # bytes/element halves that traffic, and halves the per-chunk VRAM
        # footprint that made precompute_chunk_size=8/16/32 OOM earlier
        # (2026-08-22 session). bf16 (not fp16): full fp32 exponent range,
        # so no overflow/underflow risk on `coh`'s auto1*auto2 denominator
        # the way fp16's narrow range would risk -- see this pipeline's own
        # cwt_resample_n_time history for why a performance change here
        # gets verified numerically before being trusted, not just timed.
        # CUDA only; no-op elsewhere.
        #
        # STATUS (2026-08-23): fixed and actually verified now, in two
        # steps -- neither alone was enough:
        #   1. _full_edge_wct_maps (called BEFORE _smooth_wct_maps's conv2d,
        #      i.e. upstream of the only autocast-eligible op in this whole
        #      stage) does plain index_select + elementwise multiply/
        #      subtract on w_real/w_imag -- confirmed directly that
        #      autocast does NOT downcast those (only conv2d/matmul-type
        #      ops), so this flag's outer torch.autocast(...) context used
        #      to leave that stage's tensors at full fp32 regardless. This
        #      is exactly what OOM'd at 23 channels/T=7680 (2026-08-23
        #      session: `xwt_imag = src_i * dst_r - src_r * dst_i` tried to
        #      allocate 1.85GiB with nothing free) -- fixed by an explicit
        #      .to(amp_dtype) cast on w_real/w_imag/freqs at the top of
        #      _full_edge_wct_maps, gated on torch.is_autocast_enabled()
        #      (see that method's own comment).
        #   2. That alone still OOM'd at the same size, just moved later:
        #      _smooth_wct_maps used to build a complex64 `smooth_cross`
        #      tensor (torch.complex() rejects bf16 outright -- the
        #      original blocker) so callers could run torch.angle() on it.
        #      complex64 is ALWAYS 8 bytes/element regardless of the real
        #      dtype fed in, so forcing float32 in there just to satisfy
        #      torch.complex() silently re-inflated the memory this whole
        #      change exists to shrink. Fixed by dropping torch.complex()
        #      entirely -- torch.angle(complex(r,i)) == torch.atan2(i, r)
        #      and complex(r,i).abs()**2 == r*r + i*i algebraically, so
        #      _smooth_wct_maps/_smooth_wct_maps_scale_adaptive now return
        #      `phase` (via atan2) instead of a complex tensor, and every
        #      caller (_coherence_only, _build_sparse_events,
        #      _build_dense_edge_input) takes it directly instead of
        #      calling torch.angle() itself. coh/phase now stay in
        #      `smoothed`'s own dtype (bf16 under autocast) the whole way
        #      through, matching fp32 behavior bit-for-bit when this flag
        #      is off.
        # Both steps verified numerically (fp32-vs-bf16 dense-edge output
        # on a synthetic test-tone-like signal, not just timed) before
        # being trusted -- max abs diff ~0.007-0.008 on coh/significance
        # (range [0,1]), noise-level by this pipeline's own existing bar
        # (see the (5,3)-vs-(25,3) smoothing-kernel comparison above).

        # Dynamic per-window channel subset for dense-edge computation.
        # None / 0 = full mesh (current behaviour). When set, only the top-k
        # channels by absolute cosine are used for the expensive WCT/dense-edge
        # stage; channel embeddings stay size C and are indexed.
        channel_subset_k: int | None = None,
        channel_subset_metric: str = "abs_cosine",
        dense_edge_amp_bf16: bool = False,




        # 2026-08-23: opt-in torch.autocast(dtype=torch.bfloat16) around the
        # TRAINABLE forward pass (channel_encoder/dense_edge_conv/GRU/
        # classifier -- whatever self.model_(*batch_inputs) actually is),
        # i.e. the common.py TorchEEGClassifier.train_amp_bf16 flag this
        # constructor param sets. Distinct from dense_edge_amp_bf16 above:
        # that one casts the precomputed, non-trainable dense-edge feature
        # tensors (no autograd graph, torch.no_grad() the whole time); this
        # one casts the actual forward that backward()/optimizer.step()
        # differentiate through -- the ~60% of a training epoch that
        # dense_edge_amp_bf16 measurably left untouched (2026-08-23 session:
        # dense-edge precompute dropped ~11x per-call after that fix, but
        # epoch_time only dropped ~3.5x, because precompute is only part of
        # each epoch's wall clock -- see that session's notes). bf16 (not
        # fp16) for the same reason as dense_edge_amp_bf16: full fp32
        # exponent range, so no GradScaler/loss-scaling needed -- optimizer
        # state and master weights stay fp32 regardless; _model_forward
        # explicitly casts its returned logits back to float32 before
        # criterion() sees them (see that method's comment) so
        # CrossEntropyLoss never runs its log_softmax/nll_loss in raw bf16.
        # CUDA only; no-op elsewhere. False (default) changes nothing.
        # Independent of dense_edge_amp_bf16 -- either, both, or neither can
        # be set; nothing about this flag requires the other.
        train_amp_bf16: bool = False,
        # Surrogate-data significance gating: replaces the fixed
        # coherence_threshold magnitude cutoff with a per-trial, per-(edge,
        # frequency) threshold calibrated from that trial's own null
        # distribution. For each trial, generates surrogate_count
        # phase-randomized surrogates of its raw signal (see
        # phase_randomize_surrogates above -- destroys real cross-channel
        # coupling while preserving each channel's own power spectrum),
        # runs them through the identical (non-trainable) CWT -> cross-
        # spectrum -> smoothing pipeline the real trial uses, and sets the
        # threshold to the surrogate_percentile-th percentile of that null
        # coherence distribution (pooled over surrogates and time, COI-
        # valid cells only) instead of a single fixed number everywhere.
        # phase_threshold_deg's gate is unaffected -- this only replaces the
        # coherence half of the (coherence, phase) AND-gate in
        # _build_sparse_events. "fixed" (default) is the original behavior
        # and changes nothing about any existing run.
        # "surrogate_cluster" is a third mode: instead of gating each
        # (edge, freq, time) cell independently against its own percentile
        # (which gives every one of the ~0.58M cells tested per trial (36
        # canonical edges x 16 freqs x ~1001 time, as of the 2026-08-09
        # edge-topology change -- was ~1.15M under the old 72-edge directed
        # scheme) its own independent false-positive budget -- at
        # surrogate_percentile=95, that's ~5% expected to pass by chance
        # alone, before consolidation), it corrects for multiple comparisons
        # Maris & Oostenveld-style: candidate clusters are still formed
        # using the surrogate_percentile-th per-cell threshold, but a
        # candidate is only kept if its cluster mass (sum of
        # coherence-minus-threshold over its member samples) exceeds the
        # cluster_significance_percentile-th percentile of the null
        # distribution of MAXIMUM cluster mass seen anywhere within that
        # SAME EDGE's (freq, time) cells (not pooled across all edges --
        # empirically, pooling across the whole graph made the null so
        # large that no real single-edge effect could ever beat it, at any
        # forming threshold from the 70th to 99.5th percentile; see
        # 2026-08-08 session notes, Arc 7, measured under the pre-2026-08-09
        # 72-edge topology -- conclusion carries over, exact figures don't)
        # in surrogate_count surrogates.
        # This controls the false-positive rate across one edge's cells
        # jointly, not per cell. See SparseEvidenceGNNCore._max_cluster_statistic.
        coherence_threshold_mode: str = "fixed",
        surrogate_count: int = 100,
        surrogate_percentile: float = 95.0,
        # Targeted relaxation for coherence_threshold_mode in {"surrogate",
        # "surrogate_cluster"}: 2026-08-08 session notes, Arc 5 found that
        # genuinely phase-consistent mu-band (8-13Hz) cells never clear a
        # flat high surrogate_percentile in EITHER subject 1 or subject 2 --
        # subject 1 recovers at an adjacent frequency bin, subject 2 loses
        # the whole band. A single flat percentile-everywhere threshold is
        # the wrong tool for a band where the true-signal/noise-floor gap is
        # narrower than elsewhere in the spectrum. When set (not None),
        # frequency bins inside mu_band_range_hz use THIS percentile instead
        # of surrogate_percentile; every other bin is unaffected. None
        # (default) preserves the original flat-percentile-everywhere
        # behavior exactly. See _percentile_vector /
        # _interp_percentile_grid.
        mu_band_surrogate_percentile: float | None = None,
        mu_band_range_hz: tuple[float, float] = (8.0, 13.0),
        cluster_significance_percentile: float = 95.0,
        surrogate_seed: int | None = None,
        # Device for the surrogate calibration's coherence/smoothing math
        # ONLY -- independent of `device` (which governs the trainable
        # model/training loop). Measured separately: the trainable model is
        # tiny (hidden_dim~8, batch_size~8-10) and MPS's kernel-launch
        # overhead can dominate real compute there, so it isn't switched by
        # default (see resolve_best_available_device's docstring). The
        # surrogate coherence/smoothing step is a much larger batched conv2d
        # (36 canonical edges x ~1000 time x 16 freq, batched over
        # surrogate_count -- was 72 directed edges pre-2026-08-09) --
        # measured ~10x faster on this machine's MPS once warmed up (22ms vs
        # 232ms/call at surrogate_count=10, measured under the old 72-edge
        # topology; still the dominant-cost step today, ratio not
        # independently re-measured at 36), so "auto" is a safe default
        # here specifically. "cpu" forces CPU (e.g. for exact reproducibility
        # or on machines without MPS/CUDA).
        surrogate_device: str = "auto",
        # Disk cache for the per-trial null coherence distribution (see the
        # surrogate_null_cache_* helpers above). Keyed by the trial's own
        # raw signal plus every config value that affects the null
        # distribution (sampling_rate, highest, lowest, nfreqs,
        # cwt_resample_n_time, smooth_kernel_size/sigma, coi_enabled,
        # surrogate_count, surrogate_seed) -- deliberately NOT keyed by
        # surrogate_percentile, so re-running with a different percentile
        # against the same trials is a cache hit, not a recompute.
        # None (default) resolves to <mne_data_root>/surrogate_null_cache.
        surrogate_cache_dir: str | None = None,
        surrogate_cache_enabled: bool = True,
        # Step 6 (torch-native-cwt branch) -- see
        # _BaseCWTGNNClassifier._init_cwt_gnn_classifier's matching params
        # for the full rationale. "fcwt" (default) changes nothing.
        cwt_backend: Literal["fcwt", "torch"] = "fcwt",
        torch_cwt_batch_size: int = 256,
        verbose: int = 0,
    ) -> None:
        self.coherence_threshold = coherence_threshold
        self.phase_threshold_deg = phase_threshold_deg
        self.hidden_dim = hidden_dim
        self.channel_embed_dim = channel_embed_dim
        self.smooth_kernel_sigma = smooth_kernel_sigma
        self.smooth_kernel_size = smooth_kernel_size
        if scale_adaptive_cycles <= 0:
            raise ValueError(
                f"scale_adaptive_cycles must be > 0, got {scale_adaptive_cycles!r}."
            )
        if scale_adaptive_max_kernel < 3:
            raise ValueError(
                f"scale_adaptive_max_kernel must be >= 3, got {scale_adaptive_max_kernel!r}."
            )
        self.scale_adaptive_smoothing = scale_adaptive_smoothing
        self.scale_adaptive_cycles = scale_adaptive_cycles
        self.scale_adaptive_max_kernel = scale_adaptive_max_kernel
        self.coi_enabled = coi_enabled
        self.raw_x_resample_n_time = raw_x_resample_n_time
        self.channel_encoder_dilation = channel_encoder_dilation
        self.feature_ablation = feature_ablation
        if event_aggregation not in ("mean", "gated_softmax", "concat"):
            raise ValueError(
                "event_aggregation must be 'mean', 'gated_softmax', or 'concat', got "
                f"{event_aggregation!r}."
            )
        self.event_aggregation = event_aggregation
        if int(n_hops) < 1:
            raise ValueError(f"n_hops must be >= 1, got {n_hops!r}.")
        self.n_hops = n_hops
        self.freq_aware_hops = bool(freq_aware_hops)
        if event_mode not in ("sparse", "dense", "temporal_graph"):
            raise ValueError(
                "event_mode must be 'sparse', 'dense', or 'temporal_graph', got "
                f"{event_mode!r}."
            )
        if event_mode == "temporal_graph" and self.event_aggregation != "mean":
            # Same incompatibility SparseEvidenceGNNCore.__init__ enforces --
            # duplicated here so it surfaces at classifier construction
            # time, matching every other pre-existing check's own precedent.
            raise ValueError(
                "event_mode='temporal_graph' requires event_aggregation='mean' -- "
                "see SparseEvidenceGNNCore's event_mode docstring."
            )
        if event_mode in ("dense", "temporal_graph") and self.freq_aware_hops:
            # Same incompatibility SparseEvidenceGNNCore.__init__ enforces --
            # duplicated here (like event_aggregation/n_hops's own checks
            # above) so it surfaces at classifier construction time, not
            # only once fit() gets around to building the Core.
            raise ValueError(
                f"freq_aware_hops=True has no meaning when event_mode={event_mode!r} -- "
                "see SparseEvidenceGNNCore's event_mode docstring."
            )
        if self.event_aggregation == "concat" and event_mode != "dense":
            # Same incompatibility SparseEvidenceGNNCore.__init__ enforces --
            # see that class's event_aggregation docstring / _aggregate_events'
            # "concat" branch for why. Duplicated here so it surfaces at
            # classifier construction time, matching freq_aware_hops's own
            # precedent just above.
            raise ValueError(
                "event_aggregation='concat' requires event_mode='dense' -- see "
                "SparseEvidenceGNNCore's event_aggregation docstring."
            )
        if self.event_aggregation == "concat" and int(self.n_hops) != 1:
            raise ValueError(
                "event_aggregation='concat' requires n_hops=1 -- see "
                "SparseEvidenceGNNCore's event_aggregation docstring."
            )
        if int(dense_edge_time_downsample) < 1:
            raise ValueError(
                "dense_edge_time_downsample must be >= 1, got "
                f"{dense_edge_time_downsample!r}."
            )
        if dense_edge_time_downsample != 1 and event_mode not in ("dense", "temporal_graph"):
            # Same incompatibility SparseEvidenceGNNCore.__init__ enforces --
            # duplicated here for the same reason as freq_aware_hops/concat's
            # own checks above. "temporal_graph" DOES consume this param
            # (see that mode's own docstring), unlike the other dense-only
            # knobs checked below.
            raise ValueError(
                "dense_edge_time_downsample != 1 has no meaning when "
                "event_mode='sparse' -- see SparseEvidenceGNNCore's "
                "dense_edge_time_downsample docstring."
            )
        if time_averaged_graph and event_mode != "dense":
            raise ValueError(
                "time_averaged_graph=True has no meaning when event_mode='sparse' "
                "-- see SparseEvidenceGNNCore's time_averaged_graph docstring."
            )
        if time_averaged_graph and dense_edge_time_downsample != 1:
            raise ValueError(
                "time_averaged_graph=True and dense_edge_time_downsample != 1 are "
                "mutually exclusive -- see SparseEvidenceGNNCore's "
                "time_averaged_graph docstring."
            )
        if time_averaged_graph and (dense_conv_kernel_size != 1 or dense_conv_pool_size != 1):
            raise ValueError(
                "time_averaged_graph=True requires dense_conv_kernel_size=1 and "
                "dense_conv_pool_size=1 -- see SparseEvidenceGNNCore's "
                "time_averaged_graph docstring. Got dense_conv_kernel_size="
                f"{dense_conv_kernel_size!r}, dense_conv_pool_size="
                f"{dense_conv_pool_size!r}."
            )
        if dense_edge_temporal_mode not in ("conv", "rnn"):
            raise ValueError(
                "dense_edge_temporal_mode must be 'conv' or 'rnn', got "
                f"{dense_edge_temporal_mode!r}."
            )
        if dense_edge_temporal_mode == "rnn" and event_mode != "dense":
            raise ValueError(
                "dense_edge_temporal_mode='rnn' has no meaning when "
                "event_mode='sparse' -- see SparseEvidenceGNNCore's "
                "dense_edge_temporal_mode docstring."
            )
        if shuffle_time_order and dense_edge_temporal_mode != "rnn":
            raise ValueError(
                "shuffle_time_order=True has no meaning when "
                "dense_edge_temporal_mode='conv' -- see SparseEvidenceGNNCore's "
                "shuffle_time_order docstring."
            )
        self.event_mode = event_mode
        # 2026-08-16: SparseEvidenceGNNCore.forward's aux return (bursts_per_
        # row) is n_runs/(batch*edges*freqs) -- meaningful in "sparse" mode,
        # but every edge always "fires" in "dense"/"temporal_graph" mode (see
        # that forward()'s own comment), so it collapses to the CONSTANT
        # 1/nfreqs every call there -- not a real training signal, just
        # clutter in the epoch log. Left the value itself computed/returned
        # unchanged (still feeds edge_density_history_, and the forward()
        # return-type contract other code may depend on stays the same
        # shape across modes) -- only suppresses printing it when it's
        # provably constant. See common.py's epoch_message construction.
        self._log_aux_metric = event_mode not in ("dense", "temporal_graph")
        self.dense_conv_kernel_size = dense_conv_kernel_size
        self.dense_conv_pool_size = dense_conv_pool_size
        self.dense_conv_intermediate_channels = dense_conv_intermediate_channels
        self.dense_conv_intermediate_channels_reduced = dense_conv_intermediate_channels_reduced
        self.dense_conv_out_channels = dense_conv_out_channels
        self.dense_edge_time_downsample = int(dense_edge_time_downsample)
        self.time_averaged_graph = bool(time_averaged_graph)
        self.dense_edge_temporal_mode = dense_edge_temporal_mode
        self.shuffle_time_order = bool(shuffle_time_order)
        self.temporal_graph_edge_dim = temporal_graph_edge_dim
        if coherence_threshold_mode not in ("fixed", "surrogate", "surrogate_cluster"):
            raise ValueError(
                "coherence_threshold_mode must be 'fixed', 'surrogate', or "
                f"'surrogate_cluster', got {coherence_threshold_mode!r}."
            )
        self.coherence_threshold_mode = coherence_threshold_mode
        self.surrogate_count = surrogate_count
        self.surrogate_percentile = surrogate_percentile
        if mu_band_surrogate_percentile is not None:
            lo, hi = mu_band_range_hz
            if not (lo < hi):
                raise ValueError(
                    "mu_band_range_hz must be (lo, hi) with lo < hi, got "
                    f"{mu_band_range_hz!r}."
                )
        self.mu_band_surrogate_percentile = mu_band_surrogate_percentile
        self.mu_band_range_hz = mu_band_range_hz
        self.cluster_significance_percentile = cluster_significance_percentile
        self.surrogate_seed = surrogate_seed
        self.surrogate_device = surrogate_device
        self.surrogate_cache_dir = surrogate_cache_dir
        self.surrogate_cache_enabled = surrogate_cache_enabled
        self.dense_edge_cache_dir = dense_edge_cache_dir
        self.precompute_chunk_size = precompute_chunk_size
        self.compile_dense_edge_helper = compile_dense_edge_helper
        self.channel_subset_k = channel_subset_k
        self.channel_subset_metric = channel_subset_metric
        self.dense_edge_amp_bf16 = dense_edge_amp_bf16
        self.train_amp_bf16 = train_amp_bf16
        if cwt_resample_n_time is not None:
            import warnings
            warnings.warn(
                "SparseEvidenceGNNClassifier(cwt_resample_n_time=...) is set to a "
                "non-None value. SparseEvidenceGNNCore's COI mask assumes native "
                "resolution (cwt_resample_n_time=None) and will be computed "
                "incorrectly otherwise. Resampling the CWT coefficients also "
                "destroys real high-frequency signal (see class docstring / "
                "SparseEvidenceGNNCore._coi_valid_mask).",
                stacklevel=2,
            )
        if noise_augmentation_enabled:
            raise ValueError(
                "SparseEvidenceGNNClassifier does not support "
                "noise_augmentation_enabled=True together with its event-caching "
                "optimization: _build_sparse_events now runs once per trial during "
                "_prepare_features rather than once per training batch (see "
                "_precompute_sparse_events), so live per-batch CWT-domain noise "
                "injection (_augment_train_batch_inputs / augment_paired_cwt_batch) "
                "would have no effect on the cached events. Disable noise "
                "augmentation to use this classifier."
            )
        self._init_cwt_gnn_classifier(
            sampling_rate=sampling_rate,
            lowest=lowest,
            highest=highest,
            nfreqs=nfreqs,
            cwt_resample_n_time=cwt_resample_n_time,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            normalize_input=normalize_input,
            noise_augmentation_enabled=noise_augmentation_enabled,
            noise_apply_prob=noise_apply_prob,
            noise_strength=noise_strength,
            noise_bank_size=noise_bank_size,
            noise_bank_seed=noise_bank_seed,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            last_batch_min_ratio=last_batch_min_ratio,
            selector_alpha_val_update_rate=selector_alpha_val_update_rate,
            optimizer_step_batch_size=optimizer_step_batch_size,
            optimizer_step_batch_mode=optimizer_step_batch_mode,
            optimizer_step_remainder_policy=optimizer_step_remainder_policy,
            channel_subset=channel_subset,
            use_class_weights=use_class_weights,
            cwt_cache=cwt_cache,
            cwt_backend=cwt_backend,
            torch_cwt_batch_size=torch_cwt_batch_size,
            verbose=verbose,
        )

    def _prepare_features(self, X, *, fit: bool, train_idx=None, window_keys=None):
        # Channel-subset-applied but NOT z-score-normalized -- passed to
        # _precompute_sparse_events as raw_x_native so the surrogate cache
        # key hashes something that depends only on the physical trial +
        # config, not on which CV fold's normalization stats happened to be
        # active (see _precompute_sparse_events's docstring). Computed from
        # the still-untouched `X` argument, since super()._prepare_features
        # below reassigns its own local copy without mutating this one.
        raw_x_native = self._apply_channel_subset(np.asarray(X, dtype=np.float32))
        raw_x, w_real, w_imag, freqs = super()._prepare_features(
            X, fit=fit, train_idx=train_idx, window_keys=window_keys
        )
        # Sparse events are computed from w_real/w_imag/freqs alone (see
        # compute_events -- raw_x never enters that path), so resampling
        # raw_x here has zero effect on coherence/COI/events. This lets us
        # test ChannelSignalEncoder's sensitivity to raw-signal resolution
        # in isolation from the (already-ruled-out) coherence pathway.
        if self.raw_x_resample_n_time is not None and int(
            self.raw_x_resample_n_time
        ) != int(raw_x.shape[2]):
            from scipy.signal import resample

            # .cpu() no-ops (no copy) when raw_x is already on CPU -- the
            # common case -- and only actually matters for the
            # keep_on_device=True path (StreamingSparseEvidenceGNNClassifier),
            # where raw_x can be GPU-resident and plain .numpy() would raise.
            raw_np = resample(raw_x.detach().cpu().numpy(), int(self.raw_x_resample_n_time), axis=2)
            raw_np = np.nan_to_num(raw_np, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )
            raw_x = torch.from_numpy(raw_np).float().to(raw_x.device)
        if self.event_mode in ("dense", "temporal_graph"):
            # Dynamic per-window channel subset (top-k by absolute cosine,
            # see channel_subset_dynamic.py) -- applied HERE, before both the
            # dense-edge WCT precompute below AND the `raw_x` this method
            # returns, so the two stay consistent: forward()'s channel_encoder
            # (fed this method's raw_x) and its dense-edge path (fed
            # dense_edge_raw) must agree on n_channels, since
            # _build_model_from_features sizes the whole trainable model
            # (channel embeddings, src_idx/dst_idx pair buffers, everything)
            # off raw_x.shape[1] -- see that method and
            # SparseEvidenceGNNCore.__init__. None/0 (default) leaves raw_x/
            # w_real/w_imag untouched -- current full-mesh behavior.
            if self.channel_subset_k is not None and self.channel_subset_k > 0:
                from Epilepsy.pipelines.channel_subset_dynamic import select_channel_subset

                k = min(int(self.channel_subset_k), raw_x.shape[1])
                idx = select_channel_subset(
                    raw_x, k=k, metric=self.channel_subset_metric
                )  # (B, k) in practice (raw_x is always 3D here); (k,)
                # handled too since select_channel_subset's docstring allows
                # a 2D (C, T) input in general.
                if idx.dim() == 1:
                    # Same k channels for every trial in this call.
                    raw_x = raw_x[:, idx, :]
                    w_real = w_real[:, idx, :, :]
                    w_imag = w_imag[:, idx, :, :]
                    if raw_x_native is not None:
                        raw_x_native = raw_x_native[:, idx.detach().cpu().numpy(), :]
                else:
                    # Per-trial subset (B, k): "k slots", not fixed electrode
                    # identity -- slot j of the k-channel axis below is
                    # whichever physical channel select_channel_subset picked
                    # for THAT trial, so it can differ trial-to-trial. Every
                    # downstream consumer (channel_encoder, dense-edge WCT,
                    # the trainable model's src_idx/dst_idx) only ever sees
                    # positional slots, never a channel identity, so this is
                    # safe -- see this class's channel_subset_k docstring /
                    # the module's v1 scope note.
                    idx_raw = idx.unsqueeze(-1).expand(-1, -1, raw_x.shape[-1])
                    raw_x = torch.gather(raw_x, 1, idx_raw)
                    idx_w = idx.view(idx.shape[0], idx.shape[1], 1, 1).expand(
                        -1, -1, w_real.shape[2], w_real.shape[3]
                    )
                    w_real = torch.gather(w_real, 1, idx_w)
                    w_imag = torch.gather(w_imag, 1, idx_w)
                    if raw_x_native is not None:
                        idx_np = idx.detach().cpu().numpy()
                        raw_x_native = np.take_along_axis(
                            raw_x_native, idx_np[:, :, None], axis=1
                        )
            # "temporal_graph" reuses the exact same precomputed [B, 4, E,
            # T, F] stack "dense" does (see SparseEvidenceGNNCore's
            # event_mode docstring) -- only forward()-time processing of it
            # differs (_temporal_graph_node_states vs. _dense_edge_features),
            # so this precompute call is unchanged/shared between the two.
            dense_edge_raw = self._precompute_dense_edge_inputs(
                raw_x, w_real, w_imag, freqs, raw_x_native=raw_x_native
            )
            return raw_x, dense_edge_raw
        events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask = self._precompute_sparse_events(
            raw_x, w_real, w_imag, freqs, raw_x_native=raw_x_native
        )
        return raw_x, events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask

    def _resolved_surrogate_cache_dir(self):
        return (
            Path(self.surrogate_cache_dir)
            if self.surrogate_cache_dir is not None
            else default_surrogate_cache_root()
        )

    def _surrogate_null_percentile_grid(
        self, helper, raw_trial_np, smooth_kernel_and_pad, rng, device, need_cluster_null=False
    ):
        """Full null-coherence percentile grid for one trial -- either a
        cache hit (see surrogate_null_cache_* above) or freshly computed via
        phase-randomization surrogates. Cache-agnostic callers
        (_surrogate_coherence_threshold below) interpolate whatever specific
        percentile they need out of this grid, so a cache entry serves every
        surrogate_percentile, not just the one active when it was written.

        `need_cluster_null` only controls whether a CACHE HIT missing
        cluster_null is accepted as-is (False, the "surrogate" mode path --
        no reason to force a recompute for a caller that doesn't need it)
        or rejected as a miss (True, coherence_threshold_mode=
        "surrogate_cluster" -- see SparseEvidenceGNNCore.
        _max_cluster_statistic). Any FRESH (cache-miss) computation always
        computes and caches cluster_null regardless of `need_cluster_null`,
        since reusing the already-computed coh_all to do so costs only
        ~5% on top of the surrogate CWT/cross-spectrum/smoothing that
        already dominates the cost -- so any trial touched from now on, by
        either mode, becomes cluster-aware in the cache the first time. A
        cache entry written before this was added (i.e. missing
        cluster_null) is still treated as a miss if a later call needs it,
        triggering a one-time recompute+backfill, not a silent gap.

        Returns (values_grid: np.ndarray [E, F, 201],
                 cluster_null: np.ndarray [surrogate_count] or None,
                 from_cache: bool).
        Grid points are SURROGATE_NULL_CACHE_PERCENTILES (0..100,
        every 0.5).
        """
        cache_dir = self._resolved_surrogate_cache_dir()
        cache_key = None
        if self.surrogate_cache_enabled:
            cache_key = surrogate_null_cache_key(
                raw_trial_np,
                sampling_rate=self.sampling_rate,
                highest=self.highest,
                lowest=self.lowest,
                nfreqs=self.nfreqs,
                cwt_resample_n_time=self.cwt_resample_n_time,
                smooth_kernel_size=self.smooth_kernel_size,
                smooth_kernel_sigma=self.smooth_kernel_sigma,
                coi_enabled=self.coi_enabled,
                surrogate_count=self.surrogate_count,
                surrogate_seed=int(
                    self.surrogate_seed if self.surrogate_seed is not None else self.seed
                ),
                # 2026-08-09: this pipeline's edges are now the canonical
                # (i<j) undirected pairing (see SparseEvidenceGNNCore.
                # __init__), not surrogate_null_cache_key's directed-pair
                # default -- must be set explicitly so a pre-2026-08-09
                # 72-edge cache entry (same raw_trial bytes, same channel
                # count either way) is correctly treated as a miss instead
                # of being loaded and reshaped as if it had 36 edges.
                edge_topology="canonical_undirected",
                scale_adaptive_smoothing=self.scale_adaptive_smoothing,
                scale_adaptive_cycles=self.scale_adaptive_cycles,
                scale_adaptive_max_kernel=self.scale_adaptive_max_kernel,
                cwt_backend=self.cwt_backend,
            )
            cached = load_surrogate_null_cache(
                cache_dir, cache_key, phase_threshold_deg=self.phase_threshold_deg,
                forming_percentile=self.surrogate_percentile,
            )
            if cached is not None and (not need_cluster_null or cached["cluster_null"] is not None):
                return cached["values"], cached["cluster_null"], True

        surrogates = phase_randomize_surrogates(raw_trial_np, self.surrogate_count, rng)

        # Sub-chunk the surrogate CWT the same way _precompute_sparse_events
        # sub-chunks real trials, to bound memory regardless of
        # surrogate_count.
        sub_chunk = max(1, min(int(self.surrogate_count), 20))
        if self.verbose >= 1:
            print(
                f"[SparseEvidenceGNN] Surrogate null cache miss -- computing "
                f"{self.surrogate_count} surrogates in {math.ceil(self.surrogate_count / sub_chunk)} "
                f"chunk(s) of up to {sub_chunk} (this trial only; ~10s/trial "
                f"typical at surrogate_count=100)...",
                flush=True,
            )
        coh_chunks = []
        # This loop (full CWT + cross-spectrum + smoothing + COI, once per
        # sub-chunk of surrogates) is THE expensive part of a cache miss --
        # measured ~10s/trial at surrogate_count=100 on this machine (see
        # 2026-08-09 session notes) -- and it's invisible to
        # _precompute_sparse_events's own tqdm bar, which only tracks
        # progress across trials, not within one trial's surrogate
        # calibration (a single-trial caller like debug_sparse_evidence_gnn.py
        # never even reaches that outer loop). Gated on the same
        # self.verbose >= 1 convention as the rest of this pipeline's
        # console output.
        show_progress = self.verbose >= 1
        chunk_starts = list(range(0, self.surrogate_count, sub_chunk))
        for start in tqdm(
            chunk_starts, desc="surrogate-null[cache-miss]", unit="chunk",
            disable=not show_progress, leave=False,
        ):
            end = min(start + sub_chunk, self.surrogate_count)
            _, w_real_s, w_imag_s, freqs_s = compute_cwt_real_imag_tensors(
                surrogates[start:end],
                sampling_rate=self.sampling_rate,
                highest=self.highest,
                lowest=self.lowest,
                nfreqs=self.nfreqs,
                cwt_resample_n_time=self.cwt_resample_n_time,
                transform_fn=self.transform_,
                verbose=0,
                batch_transform_fn=self.batch_transform_,
                batch_size=self.torch_cwt_batch_size,
            )
            w_real_s = w_real_s.to(device)
            w_imag_s = w_imag_s.to(device)
            freqs_batched_s = helper._batched_freqs(freqs_s.to(device), end - start)
            with torch.no_grad():
                coh_s, phase_s = helper._coherence_only(
                    w_real_s, w_imag_s, freqs_batched_s, smooth_kernel_and_pad
                )
                if self.coi_enabled:
                    coi_valid = helper._coi_valid_mask(
                        freqs_batched_s, n_time_in=w_real_s.shape[2], T_out=coh_s.shape[2]
                    )
                    coh_s = torch.where(
                        coi_valid.expand_as(coh_s), coh_s, torch.full_like(coh_s, float("nan"))
                    )
                else:
                    coi_valid = torch.ones_like(coh_s, dtype=torch.bool)
            coh_chunks.append((coh_s, coi_valid, phase_s))

        coh_all = torch.cat([c for c, _, _ in coh_chunks], dim=0)  # [n_surrogates, E, T, F]

        _, n_edges, _, n_freqs = coh_all.shape
        pooled = coh_all.permute(1, 3, 0, 2).reshape(n_edges, n_freqs, -1)  # [E, F, N*T]
        grid_q = torch.as_tensor(
            SURROGATE_NULL_CACHE_PERCENTILES / 100.0, dtype=pooled.dtype, device=pooled.device
        )
        # torch.nanquantile accepts a vector of q values and returns them
        # all in one pass -- computing the whole percentile grid costs
        # basically the same as computing a single percentile, since the
        # cost here is dominated by the sort, not the number of quantiles
        # read off it afterward.
        values_grid_t = torch.nanquantile(pooled, grid_q, dim=-1)  # [201, E, F]
        values_grid = values_grid_t.permute(1, 2, 0).cpu().numpy()  # [E, F, 201]
        # A cell that's never COI-valid for any surrogate/time gets an
        # all-NaN pooled slice -> nanquantile returns NaN there. coh is
        # clamped to [0, 1] elsewhere, so filling with 1.0 makes that cell's
        # gate unconditionally False at every percentile, matching the fact
        # that the real trial's own COI mask independently zeroes it out
        # there anyway.
        values_grid = np.nan_to_num(values_grid, nan=1.0)

        # Always computed on a fresh (cache-miss) pass now, not just when
        # need_cluster_null=True for THIS call -- reusing coh_all here costs
        # ~5% extra on top of the surrogate CWT/cross-spectrum/smoothing
        # that's already the dominant cost (measured: 5.22s plain vs 5.29s
        # with cluster_null, one BNCI2014-001 trial, surrogate_count=100;
        # measured under the pre-2026-08-09 72-edge topology, not
        # independently re-measured at 36, but this stage's cost scales
        # with edge count so it should now be roughly half). Since coh_all
        # itself is never persisted to disk (it was ~4.3GB for this config
        # under the old 72-edge topology -- 100 surrogates x 72 edges x
        # ~1000 time x 16 freq -- now ~2.15GB (projected, not
        # re-measured) at today's 36 canonical edges -- still far too
        # large to cache per trial), a cache entry written WITHOUT
        # cluster_null can never have it backfilled later except by fully
        # regenerating the surrogates from scratch (load_surrogate_null_cache
        # treats a missing cluster_null as a cache miss whenever a caller
        # asks for one). Computing it unconditionally here means any trial
        # touched from now on -- by plain "surrogate" mode or
        # "surrogate_cluster" mode alike -- writes a cluster-aware cache
        # entry the first time, so switching mode later never re-pays this
        # cost for that trial. Entries already on disk from before this
        # change still need their one-time recompute+backfill. See
        # 2026-08-08 session notes, Arc 6.
        forming_threshold_np = _interp_percentile_grid(values_grid, self.surrogate_percentile)
        forming_threshold = torch.from_numpy(forming_threshold_np).to(
            coh_all.device
        ).view(1, n_edges, 1, n_freqs)

        cluster_max_chunks = []
        for coh_s, coi_valid_s, phase_s in coh_chunks:
            cluster_max_chunks.append(
                helper._max_cluster_statistic(coh_s, forming_threshold, coi_valid_s, phase_s)
            )
        cluster_null = torch.cat(cluster_max_chunks, dim=0).cpu().numpy()  # [surrogate_count]

        if cache_key is not None:
            save_surrogate_null_cache(
                cache_dir, cache_key, values_grid, cluster_null,
                cluster_null_phase_threshold_deg=self.phase_threshold_deg,
                cluster_null_forming_percentile=self.surrogate_percentile,
            )
        return values_grid, cluster_null, False

    def _percentile_vector(self, freqs_1d: np.ndarray | None):
        """Returns the percentile to interpolate out of a null-coherence
        grid: the plain scalar `self.surrogate_percentile` (original
        behavior), unless `mu_band_surrogate_percentile` is set, in which
        case frequency bins inside `mu_band_range_hz` get that looser
        percentile instead -- a 1-D [F] array _interp_percentile_grid knows
        how to consume. See mu_band_surrogate_percentile's docstring in
        __init__ for why this exists (Arc 5, 2026-08-08 session notes:
        genuinely phase-consistent mu-band cells never clear a flat
        high percentile in either subject 1 or 2, worse for subject 2).
        `freqs_1d` is None only when mu-band relaxation isn't configured
        (the common case) -- never dereferenced in that branch."""
        if self.mu_band_surrogate_percentile is None:
            return self.surrogate_percentile
        lo, hi = self.mu_band_range_hz
        in_band = (freqs_1d >= lo) & (freqs_1d <= hi)
        return np.where(in_band, self.mu_band_surrogate_percentile, self.surrogate_percentile)

    def _surrogate_coherence_threshold(
        self, helper, raw_trial_np, smooth_kernel_and_pad, rng, device, freqs_1d=None
    ):
        """Per-trial coherence significance threshold via phase-randomization
        surrogates (see coherence_threshold_mode's docstring in __init__).

        Fetches (cache hit) or computes (cache miss) the full null-coherence
        percentile grid for this trial via _surrogate_null_percentile_grid,
        then interpolates out the self.surrogate_percentile-th value (or,
        with mu_band_surrogate_percentile set, a per-frequency percentile --
        see _percentile_vector) -- one threshold per (edge, frequency) cell,
        pooled over surrogates and time within the grid computation.

        `device` must match `helper`'s current device (see
        _precompute_sparse_events, which moves `helper` there once for the
        whole surrogate-mode precompute pass -- measured ~10x faster on MPS
        for this step's conv2d-based smoothing; see surrogate_device).

        Returns a tensor shaped [1, E, 1, F] on `device`, ready to broadcast
        against a [B, E, T, F] coherence map (the "1, E, 1, F" batch/time
        dims are filled in later when trials are stacked back into a chunk).
        """
        values_grid, _, _ = self._surrogate_null_percentile_grid(
            helper, raw_trial_np, smooth_kernel_and_pad, rng, device
        )
        n_edges, n_freqs, _ = values_grid.shape
        threshold_np = _interp_percentile_grid(values_grid, self._percentile_vector(freqs_1d))
        threshold = torch.from_numpy(threshold_np).to(device)
        return threshold.view(1, n_edges, 1, n_freqs)

    def _surrogate_cluster_thresholds(
        self, helper, raw_trial_np, smooth_kernel_and_pad, rng, device, freqs_1d=None
    ):
        """Per-trial (cluster-forming threshold, cluster-mass null cutoff)
        pair for coherence_threshold_mode="surrogate_cluster" (see that
        mode's docstring in __init__ and
        SparseEvidenceGNNCore._max_cluster_statistic).

        The cluster-forming threshold is the same per-(edge,freq) grid
        lookup _surrogate_coherence_threshold does (used only to decide
        which samples are candidates for a cluster, not the final
        significance decision) -- including mu-band relaxation via
        _percentile_vector, if configured. NOTE: this is a DIFFERENT
        forming-threshold computation than the one _surrogate_null_
        percentile_grid uses internally to build/cache cluster_null itself
        (that one is deliberately left at the plain scalar
        self.surrogate_percentile -- see that method's call site -- since
        cluster_null's cache entry is keyed on
        cluster_null_forming_percentile=self.surrogate_percentile; feeding
        it a per-frequency array here would silently invalidate that
        cache-key/content correspondence). The cluster-mass null cutoff is
        now one value PER EDGE, not a single trial-wide scalar (see
        _max_cluster_statistic's docstring for why pooling across all edges
        made every trial reject regardless of forming threshold) -- the
        cluster_significance_percentile-th percentile of that trial's own
        null distribution of each edge's maximum cluster mass.

        Returns (cluster_forming_threshold [1,E,1,F], cluster_mass_cutoff
        [E]), both on `device`.
        """
        values_grid, cluster_null, _ = self._surrogate_null_percentile_grid(
            helper, raw_trial_np, smooth_kernel_and_pad, rng, device, need_cluster_null=True
        )
        n_edges, n_freqs, _ = values_grid.shape
        forming_threshold_np = _interp_percentile_grid(
            values_grid, self._percentile_vector(freqs_1d)
        )
        forming_threshold = torch.from_numpy(forming_threshold_np).to(device)
        forming_threshold = forming_threshold.view(1, n_edges, 1, n_freqs)

        # cluster_null is [surrogate_count, E] -- per-edge percentile, axis=0.
        cluster_mass_cutoff_np = np.percentile(
            cluster_null, self.cluster_significance_percentile, axis=0
        )
        cluster_mass_cutoff = torch.from_numpy(cluster_mass_cutoff_np).to(
            device=device, dtype=forming_threshold.dtype
        )
        return forming_threshold, cluster_mass_cutoff

    def _precompute_sparse_events(self, raw_x, w_real, w_imag, freqs, raw_x_native=None):
        """Runs SparseEvidenceGNNCore.compute_events (non-trainable: cross-
        spectrum, smoothing, gate, COI, run-consolidation) ONCE per trial,
        chunked to bound memory, instead of once per (batch, epoch) inside
        forward(). Profiling: this stage is 94.8% of a forward() call's time
        despite depending only on fixed CWT features + fixed hyperparameters,
        never on trainable weights -- so its output is identical on every
        epoch for a given trial. Padding (event count) is computed once here
        across the whole dataset, not per training mini-batch as before.

        `raw_x_native`, if given, is the channel-subset-applied signal
        *before* z-score normalization -- used (instead of `raw_x`, which IS
        normalized) as the surrogate CWT's input and as
        surrogate_null_cache_key's hash input. `raw_x` is normalized using
        stats refit per CV fold from that fold's own training data, so the
        same physical trial gets normalized to different float32 values
        depending on whether it's currently playing train or test role --
        and therefore hashes to a different (still "correct" but
        unreachable-next-time) cache entry every time its role changes.
        Coherence is invariant to a uniform affine shift/scale like
        z-scoring, so substituting the pre-normalization signal here changes
        nothing about the computed null distribution, only what it hashes
        to: now purely a function of the physical trial + config, matching
        surrogate_null_cache_key's own docstring. Falls back to
        `raw_x.detach().cpu().numpy()` if not given (keeps this method
        usable standalone, e.g. from debug_wct_evidence_gnn.py).

        Uses a throwaway SparseEvidenceGNNCore purely for its non-trainable
        buffers/thresholds (src_idx/dst_idx, coherence_threshold, etc.) --
        its trainable submodules are constructed but never used here. Its
        random init is RNG-isolated (torch.random.fork_rng) so it has no
        effect on the real model built later in _build_model_from_features.

        If coherence_threshold_mode == "surrogate", each trial additionally
        gets its own per-(edge, frequency) significance threshold calibrated
        via _surrogate_coherence_threshold, replacing the fixed
        coherence_threshold scalar for that trial's gate only.

        If coherence_threshold_mode == "surrogate_cluster", each trial gets
        both a per-(edge, frequency) cluster-forming threshold AND a
        per-edge cluster-mass null cutoff via _surrogate_cluster_thresholds
        (not a single trial-wide scalar -- see
        SparseEvidenceGNNCore._max_cluster_statistic's docstring for why),
        and formed candidate events are additionally filtered by cluster
        mass (see _build_sparse_events's cluster_mass_null_threshold).
        """
        n_channels = int(raw_x.shape[1])
        n_samples = int(raw_x.shape[0])
        with torch.random.fork_rng(devices=[]):
            helper = self._build_model(n_channels=n_channels, n_classes=2)
        helper.eval()
        helper._freq_lo = float(freqs.min().item())
        helper._freq_hi = float(freqs.max().item())
        # Every row of `freqs` is identical (common.py's compute_cwt_real_imag_tensors
        # expands one 1-D freqs array to [n_samples, nfreqs]) -- one row is
        # the real per-bin Hz values _percentile_vector needs to build a
        # mu-band-relaxed threshold, if mu_band_surrogate_percentile is set.
        freqs_1d_np = freqs[0].detach().cpu().numpy()

        surrogate_mode = self.coherence_threshold_mode == "surrogate"
        cluster_mode = self.coherence_threshold_mode == "surrogate_cluster"
        if surrogate_mode or cluster_mode:
            # Independent of `self.device` (the trainable model's device) --
            # see surrogate_device's docstring in __init__. `helper` moves
            # there for this whole precompute pass; real per-chunk tensors
            # are shuttled over and results brought back to CPU below so
            # everything downstream of this method is unaffected.
            surrogate_torch_device = resolve_best_available_device(self.surrogate_device)
            helper = helper.to(surrogate_torch_device)
            rng = np.random.default_rng(
                int(self.surrogate_seed if self.surrogate_seed is not None else self.seed)
            )
            smooth_kernel_and_pad = make_gaussian_weight2d(
                kernel_size=self.smooth_kernel_size, sigma=self.smooth_kernel_sigma,
                pad_h=0, device=surrogate_torch_device, dtype=w_real.dtype,
            )
            raw_x_np = (
                raw_x_native
                if raw_x_native is not None
                else raw_x.detach().cpu().numpy()
            )

        # Chunk by trials independently of self.batch_size (which governs
        # training, not this one-time precompute). _smooth_wct_maps's im2col
        # buffers scale ~linearly with trials-per-chunk * edges * 4; even
        # with the separable-conv fix, a full batch_size=16 chunk at this
        # pipeline's shapes (9 channels -> 36 canonical edges as of
        # 2026-08-09, was 72 directed edges; T~1001, nfreqs=16) measured
        # ~8GB peak RSS for large kernels like (25,3) under the old 72-edge
        # topology -- too close to this machine's ~17GB RAM (now roughly
        # half that, projected, not re-measured, at 36 edges). Capping at 4
        # trials/chunk keeps peak RSS in the ~2GB range regardless of
        # self.batch_size -- unless precompute_chunk_size raises that cap
        # (see __init__'s docstring on it): same memory-vs-throughput
        # tradeoff, opt-in for machines with room to spare.
        chunk_cap = 4 if self.precompute_chunk_size is None else int(self.precompute_chunk_size)
        chunk = max(1, min(int(self.batch_size), chunk_cap))
        all_events, all_src, all_dst, all_freq, all_valid = [], [], [], [], []
        chunk_starts = list(range(0, n_samples, chunk))
        # Surrogate/cluster calibration is the expensive path here (full
        # CWT -> cross-spectrum -> smoothing per surrogate, per trial --
        # see _surrogate_null_percentile_grid) and can take a while at
        # surrogate_count~100 with no cache hit, so give it a visible
        # progress indicator rather than sitting silent. Gated on verbose
        # like the rest of this pipeline's console output (cwt_progress_context
        # in common.py uses the same verbose>=1 convention).
        mode_label = "surrogate_cluster" if cluster_mode else "surrogate" if surrogate_mode else "fixed"
        show_progress = (surrogate_mode or cluster_mode) and self.verbose >= 1
        if show_progress:
            print(
                f"[SparseEvidenceGNN] Precomputing sparse events "
                f"(coherence_threshold_mode={mode_label!r}): {n_samples} trials, "
                f"chunk_size={chunk}, surrogate_count={self.surrogate_count}...",
                flush=True,
            )
        for start in tqdm(
            chunk_starts, desc=f"sparse-events[{mode_label}]", unit="chunk",
            disable=not show_progress, leave=False,
        ):
            end = min(start + chunk, n_samples)
            override = None
            cluster_cutoffs = None
            if surrogate_mode:
                per_trial = [
                    self._surrogate_coherence_threshold(
                        helper, raw_x_np[trial_idx], smooth_kernel_and_pad, rng,
                        surrogate_torch_device, freqs_1d=freqs_1d_np,
                    )
                    for trial_idx in range(start, end)
                ]
                override = torch.cat(per_trial, dim=0)  # [chunk, E, 1, F] on surrogate_torch_device
            elif cluster_mode:
                thresholds_and_cutoffs = [
                    self._surrogate_cluster_thresholds(
                        helper, raw_x_np[trial_idx], smooth_kernel_and_pad, rng,
                        surrogate_torch_device, freqs_1d=freqs_1d_np,
                    )
                    for trial_idx in range(start, end)
                ]
                override = torch.cat([t for t, _ in thresholds_and_cutoffs], dim=0)
                # Each per-trial cutoff is now [E] (per-edge, not a scalar --
                # see _surrogate_cluster_thresholds), so stack rather than
                # torch.tensor() a list of floats: [chunk, E].
                cluster_cutoffs = torch.stack(
                    [c for _, c in thresholds_and_cutoffs], dim=0
                ).to(dtype=w_real.dtype, device=surrogate_torch_device)
            if surrogate_mode or cluster_mode:
                chunk_w_real = w_real[start:end].to(surrogate_torch_device)
                chunk_w_imag = w_imag[start:end].to(surrogate_torch_device)
                chunk_freqs = freqs[start:end].to(surrogate_torch_device)
            else:
                chunk_w_real = w_real[start:end]
                chunk_w_imag = w_imag[start:end]
                chunk_freqs = freqs[start:end]
            with torch.no_grad():
                ev, sp, dp, fp, vm = helper.compute_events(
                    chunk_w_real, chunk_w_imag, chunk_freqs,
                    coherence_threshold_override=override,
                    cluster_mass_null_threshold=cluster_cutoffs,
                )
            if surrogate_mode or cluster_mode:
                ev, sp, dp, fp, vm = ev.cpu(), sp.cpu(), dp.cpu(), fp.cpu(), vm.cpu()
            all_events.append(ev)
            all_src.append(sp)
            all_dst.append(dp)
            all_freq.append(fp)
            all_valid.append(vm)

        max_count = max(t.shape[1] for t in all_events)

        def pad(t, fill=0):
            if t.shape[1] == max_count:
                return t
            pad_shape = list(t.shape)
            pad_shape[1] = max_count - t.shape[1]
            filler = torch.full(pad_shape, fill, dtype=t.dtype, device=t.device)
            return torch.cat([t, filler], dim=1)

        events_padded = torch.cat([pad(t) for t in all_events], dim=0)
        src_padded = torch.cat([pad(t) for t in all_src], dim=0)
        dst_padded = torch.cat([pad(t) for t in all_dst], dim=0)
        freq_idx_padded = torch.cat([pad(t) for t in all_freq], dim=0)
        valid_mask = torch.cat([pad(t, fill=False) for t in all_valid], dim=0)
        return events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask



    def _precompute_dense_edge_inputs(self, raw_x, w_real, w_imag, freqs, raw_x_native=None):
        """event_mode="dense" counterpart to _precompute_sparse_events:
        SAME once-per-trial, chunked, non-trainable precompute discipline
        (see that method's docstring for the full rationale -- everything
        there about raw_x_native/surrogate calibration/the throwaway
        `helper` model applies unchanged here), calling
        SparseEvidenceGNNCore.compute_dense_edge_input instead of
        compute_events. The surrogate-threshold calibration itself
        (_surrogate_coherence_threshold/_surrogate_cluster_thresholds, and
        therefore the on-disk null-distribution cache) is IDENTICAL between
        event_mode="sparse" and "dense" -- event_mode never enters that
        cache key -- so a warm cache from a prior sparse-mode run against
        the same trials/config is reused here too, not rebuilt.

        Unlike sparse events (a small, variable-count list per trial), the
        dense output is one full [4, E, T, F] array PER TRIAL with NO
        padding needed (every trial has the same T for a given dataset, so
        there's no variable-count dimension to pad, unlike sparse events'
        per-trial event count) -- but it is correspondingly much larger in
        memory: at this pipeline's native resolution (T~1001, E=36, F=16)
        that's ~9.2MB/trial (float32), vs. sparse events' few-KB/trial. Still
        chunked the same way (memory-bounded, independent of self.batch_size)
        since _smooth_wct_maps's im2col buffers are the same size either way
        -- only what's kept afterward (padded events vs. the full dense
        array) differs in scale.

        coherence_threshold_mode="surrogate_cluster" is supported in a
        REDUCED form: only the per-(edge, frequency) cluster-forming
        threshold feeds dense mode's significance channel (the same
        continuous-input role coherence_threshold_mode="surrogate"'s
        threshold plays); the per-edge cluster-MASS null cutoff
        (_surrogate_cluster_thresholds' second return value) has no
        counterpart here -- it's a property of a CONSOLIDATED discrete run
        (sum of coh-minus-threshold over the run's member samples), which
        dense mode never forms. Not exercised by this pipeline's canonical
        config (coherence_threshold_mode="surrogate"), so this reduction is
        untested against real cluster-mode runs; flagged here rather than
        silently ignored.
        """

        # raw_x: (B, C, T); w_real/w_imag: (B, C, T, F) -- channel axis is
        # dim 1 on all three (see compute_cwt_real_imag_tensors_cached).
        # Dynamic per-window channel subsetting (channel_subset_k) already
        # happened in the caller (_prepare_features), BEFORE raw_x/w_real/
        # w_imag/raw_x_native were passed in here -- so C below is already k
        # when channel_subset_k is set, and everything downstream (the
        # helper built from n_channels, its src_idx/dst_idx pair-index
        # buffers, the cache key) naturally sees k channels / k*(k-1)/2 edges
        # with no special-casing needed in this method. Done there rather
        # than here so the SAME subset also feeds raw_x's OTHER consumer --
        # the trainable model's channel_encoder -- which must agree with
        # dense_edge_raw on n_channels (see _prepare_features's comment).
        n_channels = int(raw_x.shape[1])
        n_samples = int(raw_x.shape[0])

        surrogate_mode = self.coherence_threshold_mode == "surrogate"
        cluster_mode = self.coherence_threshold_mode == "surrogate_cluster"
        mode_label = (
            "surrogate_cluster"
            if cluster_mode
            else "surrogate"
            if surrogate_mode
            else "fixed"
        )

        # 2026-08-22: skip the unconditional final .cpu() below when the
        # caller already wants this stage's output resident on self.device_
        # (StreamingSparseEvidenceGNNClassifier's per-batch path -- see
        # _keep_features_on_device) AND w_real/w_imag/raw_x already ARE on
        # that device (_prepare_features only sets keep_on_device=True in
        # that same case, so this should always hold when the flag is set;
        # checked directly here rather than trusted, since a mismatch would
        # otherwise silently do the .to(compute_device) transfers below for
        # nothing while still paying a needless .cpu() at the end). Scoped
        # to "fixed" mode only: surrogate/cluster mode's compute_device is
        # surrogate_torch_device, which need not equal self.device_, and
        # isn't used by this pipeline's actual configs anyway (see this
        # method's docstring).
        # .type-only comparison, not full device equality: an allocated
        # CUDA tensor's .device always carries a concrete index
        # (device(type='cuda', index=0)), while self.device_ here is
        # whatever resolve_torch_device(self.device) returned -- typically
        # the index-less device(type='cuda') -- and torch.device.__eq__
        # treats those as UNEQUAL even though every actual .to()/allocation
        # call resolves them to the same real device. Confirmed empirically
        # (2026-08-22): comparing the full device objects here silently
        # kept this False in every real run, so the .cpu() below never
        # actually got skipped despite keep_on_device's other conditions
        # all holding. This pipeline never runs multi-GPU-indexed configs,
        # so .type equality is exactly what "same device" means here.
        keep_on_device = (
            bool(getattr(self, "_keep_features_on_device", False))
            and not (surrogate_mode or cluster_mode)
            and w_real.device.type == getattr(self, "device_", torch.device("cpu")).type
        )

        if surrogate_mode or cluster_mode:
            with torch.random.fork_rng(devices=[]):
                helper = self._build_model(n_channels=n_channels, n_classes=2)
            helper.eval()

            surrogate_torch_device = resolve_best_available_device(
                self.surrogate_device
            )
            helper = helper.to(surrogate_torch_device)

            rng = np.random.default_rng(
                int(
                    self.surrogate_seed
                    if self.surrogate_seed is not None
                    else self.seed
                )
            )

            smooth_kernel_and_pad = make_gaussian_weight2d(
                kernel_size=self.smooth_kernel_size,
                sigma=self.smooth_kernel_sigma,
                pad_h=0,
                device=surrogate_torch_device,
                dtype=w_real.dtype,
            )

            raw_x_np = (
                raw_x_native
                if raw_x_native is not None
                else raw_x.detach().cpu().numpy()
            )
            # compile_dense_edge_helper is scoped to "fixed" mode only (see
            # that flag's docstring) -- surrogate/cluster mode always uses
            # the plain, uncompiled method.
            compute_dense_edge_input_fn = helper.compute_dense_edge_input

        else:
            # 2026-08-19: "fixed" mode (what DENSE_EDGE_GRU_PARAMS/
            # PREDICTION_GRU_PARAMS actually use) previously never moved
            # `helper` off CPU here -- unlike the surrogate/cluster branch
            # above, which already proves compute_dense_edge_input works
            # fine on CUDA. compute_dense_edge_input/_build_dense_edge_input
            # and everything they call (_full_edge_wct_maps, _smooth,
            # _coi_valid_mask, _downsample_dense_edge_time) are pure PyTorch
            # tensor ops that infer their device from their input tensors
            # (make_gaussian_weight2d(device=w_real.device, ...) inside
            # compute_dense_edge_input, _coi_valid_mask's
            # device=freqs_batched.device, avg_pool2d). There is no
            # .numpy()/scipy/forced .cpu() operation anywhere in that path.
            #
            # 2026-08-19 (part 2): naively moving to CUDA made things
            # SLOWER (epoch_time went up, not down). Raising the inner chunk
            # size made it WORSE rather than better, ruling out per-chunk
            # transfer overhead as the primary cause. The actual cost was
            # `helper = self._build_model(...)` + `.to(device)` running fresh
            # on EVERY call to this method. StreamingSparseEvidenceGNNClassifier
            # calls this once per training BATCH (not per epoch/fold), so that
            # created 20+ rebuild-and-transfer cycles per epoch.
            #
            # Fixed by caching the GPU-resident helper on `self`, keyed by
            # (n_channels, device). It is explicitly a non-trainable
            # "throwaway" model used only for forward-computation methods and
            # is never optimized, so reusing the same instance across calls
            # within one fit() is safe.
            fixed_torch_device = resolve_torch_device(self.device)
            cache_key = (n_channels, str(fixed_torch_device))

            cached_helper = getattr(self, "_dense_edge_helper_cache", None)

            if cached_helper is not None and cached_helper[0] == cache_key:
                helper = cached_helper[1]
            else:
                with torch.random.fork_rng(devices=[]):
                    helper = self._build_model(
                        n_channels=n_channels,
                        n_classes=2,
                    )

                helper.eval()
                helper = helper.to(fixed_torch_device)

                self._dense_edge_helper_cache = (cache_key, helper)
                # New helper instance -> any previously compiled callable
                # was compiled against the OLD instance's bound method and
                # is stale (torch.compile'd callables aren't shared across
                # different Python objects even with identical weights).
                self._dense_edge_compiled_fn_cache = None

            compute_dense_edge_input_fn = helper.compute_dense_edge_input
            if bool(self.compile_dense_edge_helper) and fixed_torch_device.type == "cuda":
                cached_compiled = getattr(self, "_dense_edge_compiled_fn_cache", None)
                if cached_compiled is not None and cached_compiled[0] == cache_key:
                    compute_dense_edge_input_fn = cached_compiled[1]
                else:
                    # 2026-08-22: mode="reduce-overhead" (the default
                    # Inductor backend) needs Triton for its kernel-fusion
                    # codegen -- unavailable on this Windows box
                    # (torch._inductor.exc.TritonMissing, confirmed
                    # directly). backend="cudagraphs" instead: captures and
                    # replays the fixed-shape kernel sequence without any
                    # Triton-based fusion, which is the specific overhead
                    # (per-launch CPU/WDDM-driver dispatch, not raw kernel
                    # fusion) this flag targets in the first place -- see
                    # this class's compile_dense_edge_helper docstring.
                    compute_dense_edge_input_fn = torch.compile(
                        helper.compute_dense_edge_input, backend="cudagraphs"
                    )
                    self._dense_edge_compiled_fn_cache = (cache_key, compute_dense_edge_input_fn)

        helper._freq_lo = float(freqs.min().item())
        helper._freq_hi = float(freqs.max().item())
        freqs_1d_np = freqs[0].detach().cpu().numpy()

        # Disk cache -- see dense_edge_cache.py's docstring. Restricted to
        # coherence_threshold_mode="fixed": that's the only mode where this
        # method's output is provably a pure function of (raw window,
        # config), independent of which fold's normalization stats produced
        # w_real/w_imag. self.dense_edge_cache_dir is None by default (opt-
        # in, same convention as cwt_cache/surrogate_cache_dir), so this is
        # a no-op unless a caller (e.g. run_pipelines.py) supplies a
        # directory.
        #
        # Re-implemented for keep_on_device (2026-08-23, was unconditionally
        # gated off here before): StreamingSparseEvidenceGNNClassifier
        # (label_mode="prediction") calls this method once per training
        # BATCH, every epoch -- with caching gated off, a real run (~20
        # epochs) recomputed the WCT/coherence/smoothing stage for the exact
        # same physical windows ~20 times over, with epoch>1 buying nothing
        # a cache wouldn't have given for free. keep_on_device's own
        # rationale (2026-08-22 comment, still true) is about avoiding a
        # per-batch CPU<->GPU bounce for `dense` ITSELF -- that's preserved
        # unchanged below (`if not keep_on_device: dense = dense.cpu()`
        # still gates the RESULT tensor's device). What changes here is only
        # that a cache MISS also gets written to disk (save_dense_edge
        # already does its own tensor.detach().cpu().numpy() internally --
        # see that function -- so it CPU-bounces the one trial being saved
        # regardless of `dense`'s device, not the whole batch), and a cache
        # HIT is loaded from CPU/disk and .to()'d onto the compute device
        # (below) instead of recomputed -- a small, one-off transfer, not
        # the per-batch bounce keep_on_device exists to avoid. Net effect:
        # epoch 1 pays the same (now-optimized, see dense_edge_amp_bf16)
        # compute cost plus a cheap uncompressed disk write per trial
        # (~6.6ms/trial, see save_dense_edge's docstring); every later
        # epoch's repeat windows become disk reads instead of WCT recompute.
        cache_dir = None
        cache_keys: list[str] | None = None
        if mode_label == "fixed" and self.dense_edge_cache_dir is not None:
            cache_dir = Path(self.dense_edge_cache_dir)
            # raw_x_native/raw_x are already channel-subset (both the static
            # self.channel_subset AND, when set, the dynamic channel_subset_k
            # -- see _prepare_features) by the time they reach here, so this
            # hashes exactly the bytes compute_dense_edge_input actually
            # consumed -- consistent with this method's existing "post-
            # channel-subset" convention (dense_edge_cache_key's docstring).
            # channel_subset_k/metric are ALSO hashed below (config_tuple)
            # purely so a full-mesh entry and a channel_subset_k entry can
            # never collide even in the (byte-identical-by-coincidence) case
            # where a k-subset happens to reproduce another run's raw bytes.
            raw_for_keys = (
                raw_x_native
                if raw_x_native is not None
                else raw_x.detach().cpu().numpy()
            )
            cache_keys = [
                dense_edge_cache_key(
                    raw_for_keys[i],
                    sampling_rate=self.sampling_rate, highest=self.highest, lowest=self.lowest,
                    nfreqs=self.nfreqs, cwt_resample_n_time=self.cwt_resample_n_time,
                    coherence_threshold=self.coherence_threshold,
                    smooth_kernel_size=self.smooth_kernel_size,
                    smooth_kernel_sigma=self.smooth_kernel_sigma,
                    coi_enabled=self.coi_enabled,
                    dense_edge_time_downsample=self.dense_edge_time_downsample,
                    time_averaged_graph=self.time_averaged_graph,
                    scale_adaptive_smoothing=self.scale_adaptive_smoothing,
                    scale_adaptive_cycles=self.scale_adaptive_cycles,
                    scale_adaptive_max_kernel=self.scale_adaptive_max_kernel,
                    cwt_backend=self.cwt_backend,
                    channel_subset_k=self.channel_subset_k,
                    channel_subset_metric=self.channel_subset_metric,
                )
                for i in range(n_samples)
            ]

        # results[i] is filled in from disk (cache hit) or computed below
        # (cache miss, or caching disabled entirely -- every trial is then a
        # "miss" and this reduces to the original always-compute behavior).
        results: list[torch.Tensor | None] = [None] * n_samples
        miss_indices = list(range(n_samples))
        if cache_keys is not None:
            miss_indices = []
            n_hits = 0
            for i, key in enumerate(cache_keys):
                cached = load_dense_edge(cache_dir, key)
                if cached is not None:
                    # load_dense_edge always returns a CPU tensor (disk-
                    # backed). keep_on_device's `results`/final stack are
                    # GPU-resident (see below), so a cache hit needs this one
                    # small transfer to match -- cheap next to the WCT/
                    # coherence recompute it's replacing.
                    if keep_on_device:
                        cached = cached.to(fixed_torch_device)
                    results[i] = cached
                    n_hits += 1
                else:
                    miss_indices.append(i)
            if self.verbose >= 1 and n_samples > 0:
                print(
                    f"[dense-edge cache] {n_hits}/{n_samples} trials reused from disk "
                    f"({100 * n_hits / n_samples:.1f}%)"
                )

        # Same memory-vs-throughput cap as _precompute_sparse_events above
        # (see that method's comment, and precompute_chunk_size's docstring
        # in __init__) -- chunk=4 by default, raisable per-machine.
        chunk_cap = (
            4
            if self.precompute_chunk_size is None
            else int(self.precompute_chunk_size)
        )
        chunk = max(1, min(int(self.batch_size), chunk_cap))
        chunk_starts = list(range(0, len(miss_indices), chunk))

        # Previously gated to surrogate/surrogate_cluster only, which left
        # this stage silent under coherence_threshold_mode="fixed" -- the mode
        # DENSE_EDGE_GRU_PARAMS actually uses. That silence meant there was no
        # visible per-chunk timing to compare against the CWT cache logging.
        show_progress = self.verbose >= 1

        if show_progress and miss_indices:
            extra = (
                f", surrogate_count={self.surrogate_count}"
                if (surrogate_mode or cluster_mode)
                else ""
            )

            print(
                f"[SparseEvidenceGNN] Precomputing dense edge inputs "
                f"(coherence_threshold_mode={mode_label!r}): "
                f"{len(miss_indices)}/{n_samples} trial(s) need compute, "
                f"chunk_size={chunk}{extra}...",
                flush=True,
            )

        # Per-phase timing breakdown (2026-08-22): the 2026-08-21 session
        # notes flagged "get a real CWT-vs-dense-edge-vs-model-step
        # breakdown" as unfinished, unresolved work -- this is that
        # breakdown for the dense-edge stage specifically. Every phase
        # boundary is bracketed by _sync_device so these numbers reflect
        # actual device work, not async kernel-launch return time (without
        # that, CUDA/MPS would silently misattribute real compute time to
        # whichever call happens to block next, e.g. the final .cpu()).
        # Gated at verbose>=2 (one level above the existing chunk-count
        # progress line above) since this adds a _sync_device call per
        # chunk -- a real, if small, cost of its own -- so it's opt-in, not
        # part of every run's overhead.
        profile = self.verbose >= 2
        t_transfer = 0.0
        t_compute = 0.0
        t_copy_back = 0.0
        t_threshold = 0.0

        for start in tqdm(
            chunk_starts,
            desc=f"dense-edges[{mode_label}]",
            unit="chunk",
            disable=not show_progress,
            leave=False,
        ):
            idx_chunk = miss_indices[start:start + chunk]
            override = None

            if profile:
                t0 = time.perf_counter()

            if surrogate_mode:
                per_trial = [
                    self._surrogate_coherence_threshold(
                        helper,
                        raw_x_np[trial_idx],
                        smooth_kernel_and_pad,
                        rng,
                        surrogate_torch_device,
                        freqs_1d=freqs_1d_np,
                    )
                    for trial_idx in idx_chunk
                ]

                override = torch.cat(per_trial, dim=0)

            elif cluster_mode:
                # Only the forming threshold is usable here -- see this
                # method's docstring on why the cluster-mass cutoff has no
                # dense-mode counterpart.
                per_trial = [
                    self._surrogate_cluster_thresholds(
                        helper,
                        raw_x_np[trial_idx],
                        smooth_kernel_and_pad,
                        rng,
                        surrogate_torch_device,
                        freqs_1d=freqs_1d_np,
                    )[0]
                    for trial_idx in idx_chunk
                ]

                override = torch.cat(per_trial, dim=0)

            # Both surrogate/cluster and fixed modes now compute on their
            # selected device. The final dense result is moved back to CPU
            # because `results` and the final torch.stack are CPU tensors --
            # UNLESS keep_on_device (above) applies, in which case it stays
            # on `compute_device` and `results`/the final stack end up
            # GPU-resident instead (StreamingSparseEvidenceGNNClassifier
            # only; see that flag's docstring).
            compute_device = (
                surrogate_torch_device
                if (surrogate_mode or cluster_mode)
                else fixed_torch_device
            )

            if profile:
                _sync_device(compute_device)
                t1 = time.perf_counter()
                t_threshold += t1 - t0

            chunk_w_real = w_real[idx_chunk].to(compute_device)
            chunk_w_imag = w_imag[idx_chunk].to(compute_device)
            chunk_freqs = freqs[idx_chunk].to(compute_device)

            if profile:
                _sync_device(compute_device)
                t2 = time.perf_counter()
                t_transfer += t2 - t1

            # 2026-08-22: the compiled (backend="cudagraphs") path captures
            # a graph for one fixed input shape and crashes -- not
            # gracefully recompiles -- when handed a different one
            # (RuntimeError: tensor size mismatch inside
            # cudagraph_trees._copy_inputs_and_remove_from_src; confirmed
            # directly). A short final chunk (n_samples % chunk != 0 --
            # routine here, e.g. the last training batch of a fold is
            # rarely an exact multiple of chunk) is exactly that case, so
            # only full-size chunks go through the compiled path; a short
            # remainder chunk always uses the plain helper method instead.
            active_compute_fn = (
                compute_dense_edge_input_fn
                if len(idx_chunk) == chunk
                else helper.compute_dense_edge_input
            )
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (bool(self.dense_edge_amp_bf16) and compute_device.type == "cuda")
                else contextlib.nullcontext()
            )
            with amp_ctx, torch.no_grad():
                dense = active_compute_fn(
                    chunk_w_real,
                    chunk_w_imag,
                    chunk_freqs,
                    coherence_threshold_override=override,
                )

            if profile:
                _sync_device(compute_device)
                t3 = time.perf_counter()
                t_compute += t3 - t2

            if not keep_on_device:
                dense = dense.cpu()

            if profile:
                t_copy_back += time.perf_counter() - t3

            for j, i in enumerate(idx_chunk):
                results[i] = dense[j]
                if cache_dir is not None:
                    save_dense_edge(cache_dir, cache_keys[i], dense[j])

        if profile and chunk_starts:
            n_chunks = len(chunk_starts)
            total = t_threshold + t_transfer + t_compute + t_copy_back
            self._vprint(
                2,
                f"[SparseEvidenceGNN] dense-edges[{mode_label}] phase timing "
                f"over {n_chunks} chunk(s) of <= {chunk} trial(s) "
                f"(device={compute_device}): "
                f"threshold={t_threshold:.3f}s transfer={t_transfer:.3f}s "
                f"compute={t_compute:.3f}s copy_back={t_copy_back:.3f}s "
                f"total={total:.3f}s "
                f"({total / n_chunks * 1000:.2f}ms/chunk, "
                f"{total / max(1, len(miss_indices)) * 1000:.2f}ms/trial)",
            )

        return torch.stack(results, dim=0)


    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> SparseEvidenceGNNCore:
        raw_x = features[0] if isinstance(features, tuple) else features
        model = self._build_model(n_channels=int(raw_x.shape[1]), n_classes=n_classes, **kwargs)
        model.configure_summary_context(
            batch_size=int(self.batch_size),
            n_time=int(raw_x.shape[2]),
            dtype=raw_x.dtype,
            n_samples=int(raw_x.shape[0]),
        )
        return model

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> SparseEvidenceGNNCore:
        return SparseEvidenceGNNCore(
            n_channels=n_channels,
            nfreqs=self.nfreqs,
            n_classes=n_classes,
            hidden_dim=self.hidden_dim,
            channel_embed_dim=self.channel_embed_dim,
            coherence_threshold=self.coherence_threshold,
            phase_threshold_deg=self.phase_threshold_deg,
            smooth_kernel_sigma=self.smooth_kernel_sigma,
            smooth_kernel_size=self.smooth_kernel_size,
            scale_adaptive_smoothing=self.scale_adaptive_smoothing,
            scale_adaptive_cycles=self.scale_adaptive_cycles,
            scale_adaptive_max_kernel=self.scale_adaptive_max_kernel,
            model_init_seed=self.seed,
            sampling_rate=self.sampling_rate,
            coi_enabled=self.coi_enabled,
            channel_encoder_dilation=self.channel_encoder_dilation,
            feature_ablation=self.feature_ablation,
            event_aggregation=self.event_aggregation,
            n_hops=self.n_hops,
            freq_aware_hops=self.freq_aware_hops,
            event_mode=self.event_mode,
            dense_conv_kernel_size=self.dense_conv_kernel_size,
            dense_conv_pool_size=self.dense_conv_pool_size,
            dense_conv_intermediate_channels=self.dense_conv_intermediate_channels,
            dense_conv_intermediate_channels_reduced=self.dense_conv_intermediate_channels_reduced,
            dense_conv_out_channels=self.dense_conv_out_channels,
            dense_edge_time_downsample=self.dense_edge_time_downsample,
            time_averaged_graph=self.time_averaged_graph,
            dense_edge_temporal_mode=self.dense_edge_temporal_mode,
            shuffle_time_order=self.shuffle_time_order,
            temporal_graph_edge_dim=self.temporal_graph_edge_dim,
            **kwargs,
        )


# ============================================================================
# 2026-08-17: StreamingSparseEvidenceGNNClassifier -- alternate, memory-
# bounded version of SparseEvidenceGNNClassifier for label_mode="prediction"
# 's much larger (full-subject) training sets. Added here as a SEPARATE
# class (at the user's request, colocated in this file rather than a
# separate module) -- nothing above this point is modified.
#
# Why this exists: the original class's fit() computes CWT + dense-edge
# features for the WHOLE training set in one call, before the epoch loop
# even starts (common.py:1499, `_prepare_features(X, fit=True,
# train_idx=train_idx)`) -- materializing every training window's feature
# tensors simultaneously. Measured directly from this pipeline's own disk
# cache: ~2.04MB/window (dense-edge) + ~1.51MB/window (CWT, real+imag x 23
# channels) = ~3.55MB/window. Detection mode's training sets (~2.5-3.8k
# windows/fold, 7-file scope) fit in that pattern; label_mode="prediction"'s
# full-subject training sets (~14-15k windows/fold before subsampling) do
# not -- ~52GB needed against a 16GB-RAM/15GB-swap machine, confirmed by an
# actual OOM kill (exit 137) on a real run. See the 2026-08-17 session note.
#
# Rather than touch _prepare_features/fit() on SparseEvidenceGNNClassifier
# (detection mode's already-tuned, working path) or TorchEEGClassifier in
# common.py (shared with the BCI pipeline too), this subclass overrides
# ONLY fit(). It computes features one small batch at a time (calling the
# exact same cache-aware _prepare_features every other call site already
# uses, just invoked per-batch instead of once-for-the-whole-training-set),
# then hands a standard, completely UNCHANGED _train_loop (common.py)
# exactly the same batch shape it already expects (`*batch_inputs,
# batch_y`, see common.py:1802-1803 / 2051-2052 / 2097-2098).
#
# Tradeoffs, both acceptable for this use case:
# - Batch ORDER (and therefore the exact training trajectory) will NOT
#   bit-match a run of the original class given the "same" seed --
#   verified directly: DataLoader(shuffle=True, generator=g) draws an
#   internal `_base_seed` from that SAME generator object before the
#   RandomSampler it also owns first calls torch.randperm on it
#   (torch/utils/data/dataloader.py's _BaseDataLoaderIter.__init__),
#   consuming one extra draw ahead of the sampler in a way this class's
#   own (simpler, directly-called torch.randperm) shuffling doesn't
#   replicate. Chasing bit-exact matching of that undocumented, version-
#   dependent internal ordering isn't a real correctness requirement --
#   a different-but-still-uniformly-random shuffle is exactly as valid a
#   training run as any other seed's. Verified instead on what actually
#   matters: every training window is visited exactly once per epoch, and
#   a given window's computed features are numerically identical to what
#   the original whole-set computation produces for that same window.
# - Loses _precompute_dense_edge_inputs's internal cross-window batching
#   for the COLD-cache case (that method already chunks internally at
#   `min(batch_size, 4)`; this streams at self.batch_size instead, calling
#   _prepare_features fresh per training batch) -- slightly different
#   chunking granularity on a cold cache, but every physical window still
#   only pays real compute cost ONCE per run (same disk caches, same cache
#   keys), so warm-cache epochs (2nd epoch onward, and any fold that shares
#   windows with an earlier one) are unaffected either way.
# - Assumes noise_augmentation_enabled=False (this pipeline's actual
#   config, in both DENSE_EDGE_GRU_PARAMS and PREDICTION_GRU_PARAMS) --
#   _fit_noise_augmentation_state's whole-training-set fitting isn't
#   reimplemented for the lazy per-batch path; fit() raises
#   NotImplementedError if augmentation is ever turned on with this class,
#   rather than silently doing the wrong thing.
# - validation_split > 0 isn't supported (also raises) -- this pipeline
#   always runs with validation_split=0.0 (see run_pipelines.py's params),
#   so nothing currently needs it; a lazy val_loader would need the same
#   treatment as the train loader below.
# ============================================================================


class _BatchIndexSampler(Sampler):
    """Yields lists of window indices, one list per batch, freshly
    reshuffled every time iteration starts (matching
    DataLoader(shuffle=True)'s per-epoch reshuffle) -- the index-selection
    half of what TensorDataset + DataLoader(shuffle=True) used to do,
    kept separate here from the (now lazy) feature computation below. See
    this section's header comment on why this doesn't bit-match
    DataLoader's own internal shuffle order (and why that's fine)."""

    def __init__(self, n: int, batch_size: int, generator: torch.Generator):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.generator = generator

    def __iter__(self):
        order = torch.randperm(self.n, generator=self.generator).tolist()
        for start in range(0, self.n, self.batch_size):
            yield order[start : start + self.batch_size]

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size


class _LazyFeatureBatchDataset(Dataset):
    """Given a LIST of window indices (from _BatchIndexSampler, via
    DataLoader(batch_size=None, sampler=...)), computes just that batch's
    (raw_x, dense_edge_raw, y) via the classifier's own cache-aware
    _prepare_features -- called on `len(indices)` windows, not the whole
    training set, so at most one batch's tensors (~batch_size * 3.55MB,
    e.g. ~114MB at batch_size=32) exist at a time instead of the whole
    training set's (~52GB at this pipeline's real scale).

    `batch_size=None` + a sampler that yields index LISTS (rather than
    DataLoader's usual per-sample-then-collate) is how this Dataset's
    __getitem__ ends up receiving a whole list of indices at once instead
    of PyTorch's default one-int-at-a-time contract -- a documented
    map-style-dataset pattern, not a workaround.
    """

    def __init__(self, classifier: "StreamingSparseEvidenceGNNClassifier", X_raw: np.ndarray, y_idx: np.ndarray):
        self.classifier = classifier
        self.X_raw = X_raw
        self.y_idx = y_idx
        # Precompute this fit() call's CWT cache keys ONCE here, instead of
        # __getitem__ re-hashing every raw channel on every batch (~24x/
        # epoch) -- see precompute_window_cache_keys's docstring
        # (cwt_window_cache.py) for the measured cost this removes. Same
        # _apply_channel_subset applied here as _prepare_features applies
        # internally to X before hashing/CWT (cwt_gnn_classifiers.py's base
        # _prepare_features), so these keys line up 1:1 with what
        # compute_cwt_real_imag_tensors_cached sees per batch -- subsetting
        # channels (axis 1) and selecting rows via `idx` (axis 0) are
        # independent and commute, so
        # _apply_channel_subset(X_raw)[idx] == _apply_channel_subset(X_raw[idx]).
        #
        # Skipped (window_keys left None) in two cases: DISABLE_CWT_CACHE
        # (nothing would ever consult these keys -- see that sentinel's
        # docstring), and whenever this classifier's keep_on_device
        # condition already holds for every batch (StreamingSparseEvidence-
        # GNNClassifier's default real-pipeline config: torch backend,
        # cwt_resample_n_time=None, device_ resolved) -- _prepare_features's
        # keep_on_device path forces cache=None/window_keys=None regardless
        # (caching and GPU-residency were never asked to compose, see that
        # method's comment), so hashing here would be pure waste in that
        # case. Recomputed as a plain bool check, not cached, since it's
        # cheap and this only runs once per fit() call.
        keep_on_device_always = (
            bool(getattr(classifier, "_keep_features_on_device", False))
            and classifier.batch_transform_ is not None
            and classifier.cwt_resample_n_time is None
            and getattr(classifier, "device_", None) is not None
        )
        if classifier.cwt_cache is DISABLE_CWT_CACHE or keep_on_device_always:
            self._window_keys = None
        else:
            X_subset = self.classifier._apply_channel_subset(X_raw)
            self._window_keys = precompute_window_cache_keys(
                X_subset,
                sampling_rate=classifier.sampling_rate,
                highest=classifier.highest,
                lowest=classifier.lowest,
                nfreqs=classifier.nfreqs,
                cwt_resample_n_time=classifier.cwt_resample_n_time,
                cwt_backend=classifier.cwt_backend,
            )

    def __getitem__(self, indices: Sequence[int]):
        idx = np.asarray(indices, dtype=np.int64)
        X_batch = self.X_raw[idx]
        batch_keys = None if self._window_keys is None else self._window_keys[idx]
        # fit=False: reuses self.X_mean_/self.X_std_ already fit ONCE on
        # the whole training set in fit() below -- must NOT refit per
        # batch, which would normalize each batch by its own, different,
        # wrong mean/std instead of a single consistent training-set one.
        raw_x, dense_edge_raw = self.classifier._prepare_features(
            X_batch, fit=False, window_keys=batch_keys
        )
        y_batch = torch.from_numpy(self.y_idx[idx]).long()
        return raw_x, dense_edge_raw, y_batch

    def __len__(self) -> int:
        return int(self.X_raw.shape[0])


class StreamingSparseEvidenceGNNClassifier(SparseEvidenceGNNClassifier):
    """Drop-in alternative to SparseEvidenceGNNClassifier for training sets
    too large to fit in memory as precomputed feature tensors -- same
    constructor, same predict()/predict_proba() (test sets here are small
    enough -- a few hundred to ~1k windows, ~2-4GB -- that the original
    eager path is fine for them; only training needed changing), same
    underlying _train_loop. Only fit()'s feature-materialization strategy
    differs. See this section's header comment for the full rationale.

    2026-08-22: overrides `_keep_features_on_device` to True -- unlike
    SparseEvidenceGNNClassifier's whole-training-set-at-once precompute
    (where holding every trial's CWT/dense-edge tensors in VRAM
    simultaneously would blow it), this class's `_LazyFeatureBatchDataset`
    already computes just ONE training batch (tens of trials) at a time,
    so there's no memory reason for that batch's features to ever touch
    host memory between CWT, dense-edge, and the trainable forward pass.
    Measured without this (2026-08-22 session): GPU utilization bursting
    to 90-100% then idling near 0% between chunks, ~86% of a 32-trial
    batch's wall time in CWT+dense-edge recompute alone even though the
    device compute inside each stage is fast -- the CPU<->GPU bounce
    between stages (a hard requirement of the eager classifier's own
    memory budget, inherited here for free even though it buys nothing)
    was the actual cost. See _prepare_features's `keep_on_device` gating
    and _compute_cwt_real_imag_tensors_device_resident's docstring
    (cwt_window_cache.py) for what this does and doesn't cover (torch
    backend + cwt_resample_n_time=None only -- both this pipeline's actual
    config; falls back to the original CPU-bounce path otherwise, same
    result either way, just slower).
    """

    _keep_features_on_device = True

    def fit(self, X, y, validation_groups: np.ndarray | None = None, metadata=None):
        X = validate_eeg_X(X)
        self._validate_batch_control_params()
        set_seed(self.seed)
        self.device_ = resolve_torch_device(self.device)

        self.classes_ = np.unique(y)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[cls] for cls in y], dtype=np.int64)
        n_classes = len(self.classes_)

        groups = validation_groups_from_metadata(
            metadata, self.validation_group_column, validation_groups, X.shape[0],
        )
        train_idx, val_idx, chosen_groups = resolve_train_val_indices(
            X.shape[0], y_idx, int(self.seed or 0), self.validation_split,
            self.validation_group_column, groups,
        )
        if val_idx.size == 0:
            self._vprint(1, "[Train] validation disabled.")
        else:
            # See this section's header comment -- not needed by this
            # pipeline (validation_split=0.0 always), not built.
            raise NotImplementedError(
                "StreamingSparseEvidenceGNNClassifier doesn't support "
                "validation_split > 0 -- a lazy val_loader would need the "
                "same batching treatment as the train loader, not built "
                "since nothing here currently sets validation_split != 0."
            )

        # ONE cheap, whole-training-set pass to fit normalization stats --
        # NOT the full _prepare_features (which would also CWT/dense-edge
        # the whole set, the exact cost this class exists to avoid).
        # Mirrors _BaseCWTGNNClassifier._prepare_features's own fit=True
        # branch above, up to (not including) the CWT call --
        # fit_global_zscore_stats is a cheap elementwise reduction, not a
        # wavelet transform, so holding X[train_idx] (raw signal only,
        # ~0.09MB/window) for this one call is fine.
        if self.normalize_input:
            X_subset_train = self._apply_channel_subset(X[train_idx])
            self.X_mean_, self.X_std_ = fit_global_zscore_stats(X_subset_train)
        else:
            self.X_mean_, self.X_std_ = 0.0, 1.0
        if self._uses_noise_augmentation():
            raise NotImplementedError(
                "StreamingSparseEvidenceGNNClassifier assumes "
                "noise_augmentation_enabled=False (this pipeline's actual "
                "config) -- _fit_noise_augmentation_state's whole-training-"
                "set fitting isn't implemented for the lazy per-batch path."
            )

        # Model construction only needs shape metadata (n_channels,
        # n_time), never actual feature VALUES -- see
        # _build_model_from_features above -- so a tiny probe stands in
        # for the whole training set here.
        probe_n = min(2, len(train_idx))
        probe_features = self._prepare_features(X[train_idx[:probe_n]], fit=False)
        self.model_ = self._build_model_from_features(probe_features, n_classes, device=self.device_).to(
            self.device_
        )
        self._prepare_training_state_on_device()
        if self.verbose >= 2 or is_experiment_logging_configured():
            model_label = getattr(self, "model_label", self.__class__.__name__)
            print_torch_parameter_summary(self.model_, header=model_label)
            print_torch_parameter_hashes(self.model_, header=model_label)
            print_torch_custom_model_summary(self.model_, header=model_label)

        optimizer, alpha_optimizer, selector_specs = self._build_training_optimizers()
        min_batch_size = _min_accepted_batch_size(int(self.batch_size), float(self.last_batch_min_ratio))

        # Same seeding convention as the original fit() -- see
        # common.py:1503-1524's comment on why this must be independent of
        # torch's global RNG state (model construction above already
        # consumed some of it).
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(
            int(self.seed) if self.seed is not None else int(torch.initial_seed())
        )

        train_dataset = _LazyFeatureBatchDataset(self, X[train_idx], y_idx[train_idx])
        train_sampler = _BatchIndexSampler(len(train_idx), self.batch_size, shuffle_generator)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, num_workers=0)
        val_loader = None  # val_idx.size == 0 enforced above

        if _count_eligible_tensor_batches(train_loader, min_batch_size) == 0:
            raise ValueError(
                "last_batch_min_ratio leaves no eligible training batches. "
                "Reduce last_batch_min_ratio or batch_size."
            )

        criterion = self._criterion(y_idx[train_idx])
        self._reset_histories()
        self._train_loop(
            train_loader,
            val_loader,
            optimizer,
            criterion,
            n_classes,
            selector_specs=selector_specs,
            alpha_optimizer=alpha_optimizer,
        )
        return self

    # 2026-08-23: overrides TorchEEGClassifier._predict_logits (common.py) --
    # that base version calls self._prepare_features(X, fit=False) on the
    # WHOLE of X in one shot (materializing every trial's CWT+dense-edge
    # tensors before any DataLoader batching even starts), then only
    # batches the already-built tensors for the forward pass. Fine for the
    # "eager" SparseEvidenceGNNClassifier's use case (test sets assumed to
    # be a few hundred windows) but WRONG for this streaming subclass: a
    # real, uncapped leave-one-seizure-out prediction test set is
    # deliberately never subsampled (see leave_one_seizure_out_prediction's
    # negative_to_positive_ratio docstring in run_pipelines.py), so it can
    # be as large as a training fold (~750 windows measured, chb01 alone) --
    # confirmed this session: a real 6-fold subject-1 run OOM'd here, on
    # fold 1's very first predict_proba call, right after that same fold's
    # 20 training epochs had completed cleanly. Fixed the same way fit()
    # above fixes it for training: reuse _LazyFeatureBatchDataset so only
    # one inference batch's features are ever materialized at a time,
    # instead of the whole test set's. Sequential (no _BatchIndexSampler
    # shuffle -- prediction order must match the caller's X row order) and
    # no labels (dummy zeros; _LazyFeatureBatchDataset's third return value
    # is unused here, same as it's unused by TorchEEGClassifier.predict()/
    # predict_proba(), which only ever consume this method's logits).
    def _predict_logits(self, X) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise ValueError("Model has not been fitted yet.")
        X = validate_eeg_X(X)
        n = X.shape[0]
        dataset = _LazyFeatureBatchDataset(self, X, np.zeros(n, dtype=np.int64))
        batch_size = max(1, int(self.batch_size))
        logits_list = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                indices = list(range(start, min(start + batch_size, n)))
                raw_x, dense_edge_raw, _ = dataset[indices]
                batch_inputs = tuple(
                    t.to(self.device_) for t in (raw_x, dense_edge_raw)
                )
                logits, _ = self._model_forward(batch_inputs)
                logits_list.append(logits.cpu().numpy())
        return np.concatenate(logits_list, axis=0)
