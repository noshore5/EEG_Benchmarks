"""NonStGM: nonstationary covariance/precision graph classifier.

Practical EEG adaptation (NOT a reproduction) of Basu & Subba Rao (2023),
"Graphical Models for Nonstationary Time Series", Annals of Statistics
51(4), 1453-1483, arXiv:2109.08709 -- see `nonstgm_graph.py`'s module
docstring for exactly what is and is not faithful to the paper.

Architecture: raw EEG window (B, C, T) -> `LocalCovariancePrecisionGraph`
(nonstgm_graph.py) turns it into a time-local conditional-dependence edge
sequence (B, n_edge_feat, E, T_segments), E = C*(C-1)/2 -> a temporal model
over that edge sequence -> mean-pool over edges -> small MLP head -> 2-class
logits.

Two temporal-backend variants (spec: "at least two model variants", "reuse
the repository's current Mamba infrastructure"), both REUSED UNCHANGED from
`cwt_gnn_classifiers.py` rather than reimplemented -- that module already
built exactly the `[B, C_in, E, T] -> [B, out_channels, E, 1]` per-edge,
shared-weight temporal contract this pipeline's edge tensor also needs:

    temporal_backend="gru"   -> `_DenseEdgeGRUTemporal`   (nn.GRU)
    temporal_backend="mamba" -> `_DenseEdgeMambaTemporal`  (mambapy Mamba SSM)

--pipeline nonstgm_mamba's param dict (run_pipelines.py) starts as an exact
copy of nonstgm_gru's, differing only in temporal_backend -- same isolated-
ablation reasoning dense_edge_mamba's docstring gives relative to
dense_edge_gru: this compares WHICH sequence model reads the graph, not a
capacity change bundled with it.

Leakage: `_prepare_features` z-score-normalizes the raw window using
train-fold-only statistics (`fit_global_zscore_stats`/`apply_global_zscore`,
the same convention `godoy_tmc_classifier.py`/`dbconformer_classifier.py`
use) -- identical mechanism `TorchEEGClassifier.fit()` already threads
`train_idx` through for every raw-EEG classifier in this repo. The
covariance/precision estimator itself has no data-driven fit step (the
regularization lambda is a fixed hyperparameter, not tuned on held-out
data), so there is no additional leakage surface beyond that normalization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from Epilepsy.pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats
    from Epilepsy.pipelines.cwt_gnn_classifiers import _DenseEdgeGRUTemporal, _DenseEdgeMambaTemporal
    from Epilepsy.pipelines.nonstgm_graph import LocalCovariancePrecisionGraph, Representation
except ModuleNotFoundError:
    from pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats
    from pipelines.cwt_gnn_classifiers import _DenseEdgeGRUTemporal, _DenseEdgeMambaTemporal
    from pipelines.nonstgm_graph import LocalCovariancePrecisionGraph, Representation


class NonStGMCore(nn.Module):
    """graph module -> temporal backend -> edge-mean-pool -> MLP head."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        representation: Representation = "precision",
        temporal_segments: int = 6,
        overlap: float = 0.5,
        regularization: float = 1e-2,
        static: bool = False,
        edge_threshold: float | None = None,
        adaptive_regularization: bool = False,
        temporal_backend: str = "gru",
        temporal_hidden: int = 16,
        head_hidden: int = 32,
        head_dropout: float = 0.2,
        mamba_d_model: int = 16,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_n_layers: int = 1,
        mamba_dropout: float = 0.0,
        mamba_chunk_size: int = 128,
        mamba_use_cuda_kernel: bool | None = None,
        **_unused,  # tolerate the device= kwarg TorchEEGClassifier._build_model_from_features passes through
    ) -> None:
        super().__init__()
        self.graph = LocalCovariancePrecisionGraph(
            representation=representation,
            temporal_segments=temporal_segments,
            overlap=overlap,
            regularization=regularization,
            static=static,
            edge_threshold=edge_threshold,
            adaptive_regularization=adaptive_regularization,
        )
        n_edge_feat = self.graph.n_edge_features

        if temporal_backend == "gru":
            self.temporal = _DenseEdgeGRUTemporal(in_channels=n_edge_feat, out_channels=temporal_hidden)
        elif temporal_backend == "mamba":
            self.temporal = _DenseEdgeMambaTemporal(
                in_channels=n_edge_feat,
                out_channels=temporal_hidden,
                d_model=mamba_d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                n_layers=mamba_n_layers,
                dropout=mamba_dropout,
                chunk_size=mamba_chunk_size,
                use_cuda_kernel=mamba_use_cuda_kernel,
            )
        else:
            raise ValueError(f"temporal_backend must be 'gru' or 'mamba', got {temporal_backend!r}")
        self.temporal_backend = temporal_backend

        self.head = nn.Sequential(
            nn.Linear(temporal_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, n_classes),
        )
        self._last_diagnostics: dict = {}

    def forward(self, x: torch.Tensor):
        edge_tensor, diagnostics = self.graph(x)  # [B, n_edge_feat, E, T_segments]
        self._last_diagnostics = diagnostics
        pooled = self.temporal(edge_tensor)  # [B, temporal_hidden, E, 1]
        pooled = pooled.squeeze(-1).mean(dim=-1)  # mean over edges -> [B, temporal_hidden]
        logits = self.head(pooled)
        # Second return value follows TorchEEGClassifier._model_forward's
        # (logits, aux) convention (common.py) -- aux must be a plain float,
        # not the diagnostics dict itself (that is instead read off
        # self._last_diagnostics / NonStGMClassifier.diagnostics_ after
        # each fit/predict call, see below). Report the fraction of
        # segments that needed fallback regularization as the per-batch
        # scalar aux, the single number closest to "is this batch
        # numerically healthy".
        n_seg = diagnostics.get("n_segments_total") or 0
        n_fallback = diagnostics.get("n_fallback_regularization") or 0
        aux = float(n_fallback) / n_seg if n_seg else 0.0
        return logits, aux


class NonStGMClassifier(TorchEEGClassifier):
    """sklearn-style wrapper around NonStGMCore -- same global z-score
    normalization convention as DBConformerClassifier/GodoyTMCClassifier
    (raw EEG in, no CWT/STFT preprocessing, no disk cache: the
    covariance/precision graph is computed on the fly inside the model's
    forward pass, once per batch)."""

    _estimator_type = "classifier"
    model_label = "NonStGM"
    aux_metric_name = "fallback_regularization_fraction"

    def __init__(
        self,
        representation: Representation = "precision",
        temporal_segments: int = 6,
        overlap: float = 0.5,
        regularization: float = 1e-2,
        static: bool = False,
        edge_threshold: float | None = None,
        adaptive_regularization: bool = False,
        temporal_backend: str = "gru",
        temporal_hidden: int = 16,
        head_hidden: int = 32,
        head_dropout: float = 0.2,
        mamba_d_model: int = 16,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_n_layers: int = 1,
        mamba_dropout: float = 0.0,
        mamba_chunk_size: int = 128,
        mamba_use_cuda_kernel: bool | None = None,
        normalize_input: bool = True,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        use_class_weights: bool = True,
        verbose: int = 0,
    ) -> None:
        self.representation = representation
        self.temporal_segments = temporal_segments
        self.overlap = overlap
        self.regularization = regularization
        self.static = static
        self.edge_threshold = edge_threshold
        self.adaptive_regularization = adaptive_regularization
        self.temporal_backend = temporal_backend
        self.temporal_hidden = temporal_hidden
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.mamba_d_model = mamba_d_model
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_n_layers = mamba_n_layers
        self.mamba_dropout = mamba_dropout
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_use_cuda_kernel = mamba_use_cuda_kernel
        self.normalize_input = normalize_input

        self.X_mean_: float | None = None
        self.X_std_: float | None = None
        self.diagnostics_: dict = {}

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
            use_class_weights=use_class_weights,
            verbose=verbose,
        )

    def _prepare_features(self, X, *, fit: bool, train_idx=None):
        if not self.normalize_input:
            return X
        if fit:
            ref = X if train_idx is None else X[train_idx]
            self.X_mean_, self.X_std_ = fit_global_zscore_stats(ref)
        if self.X_mean_ is None or self.X_std_ is None:
            raise ValueError("Normalization stats are not initialized -- call fit() first.")
        return apply_global_zscore(X, self.X_mean_, self.X_std_)

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> NonStGMCore:
        x = features[0] if isinstance(features, tuple) else features
        _, n_channels, _n_time = x.shape
        return NonStGMCore(
            n_channels=int(n_channels),
            n_classes=n_classes,
            representation=self.representation,
            temporal_segments=self.temporal_segments,
            overlap=self.overlap,
            regularization=self.regularization,
            static=self.static,
            edge_threshold=self.edge_threshold,
            adaptive_regularization=self.adaptive_regularization,
            temporal_backend=self.temporal_backend,
            temporal_hidden=self.temporal_hidden,
            head_hidden=self.head_hidden,
            head_dropout=self.head_dropout,
            mamba_d_model=self.mamba_d_model,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            mamba_n_layers=self.mamba_n_layers,
            mamba_dropout=self.mamba_dropout,
            mamba_chunk_size=self.mamba_chunk_size,
            mamba_use_cuda_kernel=self.mamba_use_cuda_kernel,
            **kwargs,
        )

    def predict_proba(self, X) -> np.ndarray:
        # NonStGMCore.forward stashes the covariance/precision numerical-
        # stability diagnostics for its most recent call on
        # self.model_._last_diagnostics (see NonStGMCore.forward) --
        # surfaced here after every predict_proba call (this reflects the
        # data just scored, e.g. a fold's held-out test set when called
        # from the LOSO loop, NOT necessarily the training set).
        proba = super().predict_proba(X)
        if self.model_ is not None:
            self.diagnostics_ = dict(getattr(self.model_, "_last_diagnostics", {}))
        return proba
