"""CG-MambaNet: a CNN-GCN-Mamba-BiLSTM architecture for cross-patient
epileptic seizure prediction, built from the paper's own description --

    Chen, M. et al. "CG-MambaNet: A spatiotemporal framework for
    cross-patient epileptic seizure prediction using CNN-GCN-Mamba-BiLSTM
    with event-level clinical evaluation." arXiv:2606.08226 (2026).

NOT vendored code -- CG-MambaNet has no public implementation as of this
writing (no code/checkpoint link anywhere on the arXiv listing, checked
2026-08-26). Everything below is reconstructed from the paper's own Methods
section (unusually detailed for a preprint -- full formulas and a
hyperparameter table), so treat this as "built to the paper's stated spec",
not "reproduces the paper's exact reported numbers".

**This file is run under THIS REPO'S OWN chb01-only leave-one-seizure-out
protocol and standard training recipe** (`--label-mode prediction`,
`validation_split=0.2`, `early_stopping_patience=5`, `epochs=20`,
`use_class_weights=True` -- the same recipe GRU/Mamba/DBConformer/SlimSeiz
already use), **NOT** the paper's own multi-patient leave-one-patient-out
evaluation or its own training regime (AdamW / cosine-with-warmup / 50
epochs / checkpoint-by-val-AUC / weighted BCE / per-channel normalization).
That fuller reproduction -- the only version whose number is actually
comparable to the paper's reported AUC-ROC 0.8152 -- is a separate, larger,
DEFERRED effort (this repo has no cross-patient channel-montage
unification yet; CHB-MIT patients differ in channel names/counts). This
pass exists to get a fast, apples-to-apples read on the ARCHITECTURE
against this repo's other pipelines (especially `dense_edge_mamba`) before
investing in that larger effort.

DEVIATIONS / INTERPRETIVE CHOICES from the paper's literal spec (flagged
here, not silently absorbed):

  - Labeling: this repo's own `--sph`/`--sop` defaults (5min/15min preictal
    window, buffered away from onset) are used, NOT the paper's own
    unbuffered "ends within 30 minutes of onset" definition -- keeps
    CG-MambaNet's labels identical to every other row in the comparison
    table, which is the whole point of this pass.
  - Sampling rate: this repo's native 256Hz (no resample to the paper's
    200Hz) and no bandpass/notch/artifact-rejection preprocessing --
    DBConformer/SlimSeiz don't do this preprocessing either; raw z-scored
    windows only, same as the rest of the raw-EEG classifier family.
  - Normalization: this repo's existing GLOBAL-scalar `fit_global_zscore_
    stats`/`apply_global_zscore` (train-fold-only stats), not the paper's
    PER-CHANNEL stats -- same convention DBConformer/SlimSeiz already use.
  - CNN front-end -> GCN handoff: the paper gives the CNN front-end's
    output as (B,C,S,P) (P = samples/patch) and the GCN's input as (B,C,S,d)
    with d=200, with no projection between them described. An earlier
    version of this file read that as "the embedding dimension IS the
    per-patch sample count, no extra linear projection" (d = patch_size
    directly) -- but that makes d_model scale with sample-rate/patch
    duration, not an independent architectural choice, and at this repo's
    patch_size=256 it produced a 13M-parameter, 12-layer x 2-direction
    Mamba stack that took ~93s per forward+backward pass at SMOKE scale on
    CPU (measured 2026-08-26) -- 16x the d_model of every other Mamba usage
    in this repo (`dense_edge_mamba` uses d_model=16) and completely
    impractical to train locally. A learned `_PatchEmbedding` projection
    (`nn.Linear(patch_size, d_embed)`) now sits between the CNN front-end
    and the GCN, decoupling the embedding width (`d_embed`) from
    patch_size/sequence length the way DBConformer's own
    `PatchEmbeddingTemporal` already does -- but it's OPTIONAL, not a
    replacement for the paper-literal reading: `d_embed=None` (either
    constructor) keeps the original "no projection, d=patch_size" behavior;
    a set `d_embed` (default 64 in `_CG_MAMBANET_SHARED_PARAMS`, in line
    with this repo's other Mamba usage, `dense_edge_mamba`'s d_model=16)
    projects down instead. Default ON here because this pass runs on
    CPU/MPS with `mambapy`'s portable pure-PyTorch pscan (no fused CUDA
    kernel) -- once this runs on the RunPod CUDA image
    (`ghcr.io/noshore5/eeg_benchmarks-mamba`, fused `mamba-ssm` kernel),
    `d_embed=None` becomes worth reconsidering as the more paper-faithful
    width, since that cost profile is completely different from what's
    practical here.
  - CNN front-end channel width: the paper reports "928 total params" for
    the front-end but doesn't state the two branches' intermediate channel
    width, so an exact param-count match isn't recoverable from the
    description alone -- `mid_channels` below is a reasonable small width,
    not a derived one.
  - Mamba bidirectionality: the paper says "forward and backward paths...
    gated and concatenated" with no formula. Implemented here as a learned
    sigmoid gate over the concatenated fwd/bwd features, blending them
    elementwise back down to `d_model` width (not a dimension-doubling
    concat), so the encoder's output width matches what feeds into it.
  - Mamba-sequence flattening order: the paper doesn't specify how
    (B,C,S,d) becomes the Mamba's (B, C*S, d) input sequence. This uses
    patch-major order -- (B,S,C,d) -> (B, S*C, d) -- so consecutive
    sequence steps are the fixed-montage channels at one time-patch,
    advancing across patches; the sequence's causal axis is real time.
  - Classification head: paper ends 64->1 + sigmoid, trained with weighted
    BCE. This repo's shared training loop (`TorchEEGClassifier`/
    `_train_loop` in common.py) trains via `CrossEntropyLoss` on
    `n_classes` logits -- head ends 64->2 instead, the same convention
    every other classifier in this repo already uses. Functionally
    equivalent for binary classification.
  - Mamba encoder built on this repo's own pinned `mambapy` package
    (already a hard dependency via `dense_edge_mamba`), NOT SlimSeiz's
    from-scratch single-block Mamba (sequential-scan only, `d_state` fixed
    at 16, no multi-layer stacking, `d_conv=3` not 4) -- that block was
    evaluated and rejected as the building block for this encoder.

Architecture pieces below (CNN front-end, GCN, bidirectional Mamba stack,
BiLSTM, MLP head) otherwise follow the paper's own stated shapes and
hyperparameters (12 Mamba layers, d_state=64, d_conv=4, 2 GCN layers,
2-layer BiLSTM hidden=128/direction dropout=0.3, MLP 256->64->n_classes)
as closely as the deviations above allow.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from Epilepsy.pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats
except ModuleNotFoundError:
    from pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats

try:
    from Epilepsy.pipelines.cwt_gnn_classifiers import _resolve_mamba_use_cuda_kernel
except ModuleNotFoundError:
    from pipelines.cwt_gnn_classifiers import _resolve_mamba_use_cuda_kernel


class _CGMambaCNNFrontEnd(nn.Module):
    """Two parallel depthwise temporal convs (k=5 ~ beta/gamma band, k=15 ~
    delta/theta/alpha band) + a pointwise fusion conv + residual, applied
    with SHARED weights to every (channel, patch) 1-D segment of length
    `patch_size` -- see this module's docstring for the (B,C,S,P) shape
    convention and the "CNN front-end -> GCN handoff" interpretive
    choice (a separate `_PatchEmbedding` module, not this one, handles the
    optional width projection down to `d_embed`)."""

    def __init__(self, patch_size: int, mid_channels: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(1, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
        )
        self.branch_k15 = nn.Sequential(
            nn.Conv1d(1, mid_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(mid_channels * 2, 1, kernel_size=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S, P) -- flatten (channel, patch) into the conv batch axis
        b, c, s, p = x.shape
        flat = x.reshape(b * c * s, 1, p)
        f1 = self.branch_k5(flat)
        f2 = self.branch_k15(flat)
        fused = self.fuse(torch.cat([f1, f2], dim=1))
        out = fused + flat  # residual (paper: "residual connection after fusion")
        return out.reshape(b, c, s, p)


class _PatchEmbedding(nn.Module):
    """Optional projection from the CNN front-end's per-patch sample width
    (`patch_size`) down to an independent embedding width (`d_embed`) --
    see module docstring's "CNN front-end -> GCN handoff" entry.
    `d_embed=None` is a no-op (`nn.Identity`), preserving the paper-literal
    "d = patch_size" reading for later use on a CUDA box with the fused
    Mamba kernel."""

    def __init__(self, patch_size: int, d_embed: int | None) -> None:
        super().__init__()
        self.out_dim = patch_size if d_embed is None else d_embed
        self.proj = nn.Identity() if d_embed is None else nn.Linear(patch_size, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _LearnableGCN(nn.Module):
    """2-layer GCN, fully learnable adjacency (uniform init, updated
    end-to-end), symmetric Kipf-Welling normalization recomputed each
    forward, residual + LayerNorm after the last layer. Operates over the
    channel axis (n_nodes) independently for every (batch, patch) pair."""

    def __init__(self, n_nodes: int, d: int, n_layers: int = 2) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.adj = nn.Parameter(torch.full((n_nodes, n_nodes), 1.0 / n_nodes))
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def _normalized_adjacency(self) -> torch.Tensor:
        a_tilde = self.adj + torch.eye(self.n_nodes, device=self.adj.device, dtype=self.adj.dtype)
        deg = a_tilde.sum(dim=-1).clamp_min(1e-8)
        d_inv_sqrt = torch.diag(deg.pow(-0.5))
        return d_inv_sqrt @ a_tilde @ d_inv_sqrt

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N, n_nodes, d), N = B*S
        a_hat = self._normalized_adjacency()
        residual = h
        for layer in self.layers:
            h = torch.relu(torch.einsum("ij,njd->nid", a_hat, layer(h)))
        return self.norm(h + residual)


class _BiMambaEncoder(nn.Module):
    """Bidirectional stack built on `mambapy.mamba.Mamba`/`MambaConfig`
    (already pinned, see requirements.txt) -- forward pass on `x`, a second
    pass on `x.flip(dims=[1])` with its output flipped back, blended via a
    learned sigmoid gate (see module docstring's "gated and concatenated"
    interpretive choice)."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 12,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        use_cuda_kernel: bool | None = None,
        pscan: bool = True,
    ) -> None:
        super().__init__()
        try:
            from mambapy.mamba import Mamba, MambaConfig
        except ImportError as exc:  # pragma: no cover -- environment-dependent
            raise ImportError(
                "cg_mambanet's bidirectional Mamba encoder requires the "
                "'mambapy' package (the same dependency dense_edge_mamba "
                "already needs -- see requirements.txt). Install with "
                "`pip install mambapy`."
            ) from exc
        resolved = _resolve_mamba_use_cuda_kernel(use_cuda_kernel)
        self.use_cuda_kernel = resolved
        cfg = dict(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            # None/auto: True only on Linux/CUDA with mamba-ssm importable.
            # Never silently mix with (b)float16; _run disables autocast
            # when this is True, same convention as _DenseEdgeMambaTemporal.
            use_cuda=resolved,
            # pscan=True (mambapy default) pads the sequence to a power of
            # two and keeps intermediate tensors at every scan level for
            # backward -- measured (2026-08-26) severely super-linear batch-
            # size scaling on CPU (batch 4->8->16: 3.1s/12.5s/40.3s) and an
            # outright MPS OOM at batch=16 (18GB against an 18.13GB ceiling)
            # with 24 total directional Mamba instances (12 layers x 2
            # directions) at this encoder's real-scale seq_len=480. pscan=
            # False falls back to mambapy's plain sequential selective_scan_
            # seq -- no padding/level materialization, portable, the
            # documented fallback for exactly this failure mode.
            pscan=pscan,
        )
        self.mamba_fwd = Mamba(MambaConfig(**cfg))
        self.mamba_bwd = Mamba(MambaConfig(**cfg))
        self.gate = nn.Linear(2 * d_model, d_model)

    def _run(self, mamba: nn.Module, seq: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_kernel:
            device_type = seq.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                return mamba(seq.float()).to(dtype=seq.dtype)
        return mamba(seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        fwd = self._run(self.mamba_fwd, x)
        bwd = self._run(self.mamba_bwd, x.flip(dims=[1])).flip(dims=[1])
        gate = torch.sigmoid(self.gate(torch.cat([fwd, bwd], dim=-1)))
        return gate * fwd + (1 - gate) * bwd


class CGMambaNet(nn.Module):
    """CNN front-end -> GCN -> bidirectional Mamba -> BiLSTM -> MLP head.
    Input `(B, C, T)` raw EEG windows (C = fixed montage size, T =
    window_length * sampling_rate); see module docstring for every
    deviation from the paper's literal spec."""

    def __init__(
        self,
        n_channels: int,
        n_time: int,
        n_classes: int = 2,
        patch_size: int = 256,
        d_embed: int | None = 64,
        gcn_layers: int = 2,
        mamba_n_layers: int = 12,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand_factor: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        head_dropout: float = 0.3,
        use_cuda_kernel: bool | None = None,
        **_unused,  # tolerate the device= kwarg TorchEEGClassifier._build_model_from_features passes through
    ) -> None:
        super().__init__()
        if n_time % patch_size != 0:
            raise ValueError(
                f"patch_size={patch_size} does not evenly divide the window length "
                f"{n_time} samples -- pick a patch_size that divides it, or change "
                "--window-length so window_length*sampling_rate does."
            )
        self.n_channels = n_channels
        self.n_patches = n_time // patch_size
        self.patch_size = patch_size

        self.front_end = _CGMambaCNNFrontEnd(patch_size)
        self.embed = _PatchEmbedding(patch_size, d_embed)  # see module docstring's "CNN front-end -> GCN handoff" note
        self.d = self.embed.out_dim
        self.gcn = _LearnableGCN(n_nodes=n_channels, d=self.d, n_layers=gcn_layers)
        self.mamba = _BiMambaEncoder(
            d_model=self.d,
            n_layers=mamba_n_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand_factor=mamba_expand_factor,
            use_cuda_kernel=use_cuda_kernel,
        )
        self.lstm = nn.LSTM(
            input_size=self.d,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        b, c, _t = x.shape
        x = x.view(b, c, self.n_patches, self.patch_size)  # (B,C,S,P)
        x = self.front_end(x)  # (B,C,S,P)
        x = self.embed(x)  # (B,C,S,d) -- d = P if d_embed is None, else d_embed

        # GCN over the channel axis, independently per (batch, patch).
        d = self.d
        x = x.permute(0, 2, 1, 3).reshape(b * self.n_patches, c, d)  # (B*S,C,d)
        x = self.gcn(x)  # (B*S,C,d)

        # Patch-major flatten into the Mamba sequence (see module docstring).
        x = x.reshape(b, self.n_patches, c, d)  # (B,S,C,d)
        x = x.reshape(b, self.n_patches * c, d)  # (B, S*C, d)

        x = self.mamba(x)  # (B, S*C, d)
        _, (h_n, _c_n) = self.lstm(x)  # h_n: (num_layers*2, B, hidden)
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        pooled = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*hidden)
        return self.head(pooled)


class CGMambaNetClassifier(TorchEEGClassifier):
    """sklearn-style wrapper around CGMambaNet -- global z-score
    normalization (same convention as DBConformerClassifier/
    SlimSeizClassifier) plus an optional fixed-channel slice.
    `channel_select_fixed_indices` is how run_pipelines.py applies
    CG-MambaNet's paper-specified 16-channel montage (the GCN's node count
    is read off whatever channel axis width `_prepare_features` produces,
    so leaving this unset falls back to whatever full montage `X` has --
    a deviation from the paper, not the intended default). No CWT/STFT
    preprocessing, no disk caching -- same reasoning as
    DBConformerClassifier/SlimSeizClassifier."""

    _estimator_type = "classifier"
    model_label = "CGMambaNet"

    def __init__(
        self,
        patch_size: int = 256,
        d_embed: int | None = 64,
        gcn_layers: int = 2,
        mamba_n_layers: int = 12,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand_factor: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        head_dropout: float = 0.3,
        use_cuda_kernel: bool | None = None,
        normalize_input: bool = True,
        channel_select_fixed_indices: list[int] | None = None,
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
        self.patch_size = patch_size
        self.d_embed = d_embed
        self.gcn_layers = gcn_layers
        self.mamba_n_layers = mamba_n_layers
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand_factor = mamba_expand_factor
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.head_dropout = head_dropout
        self.use_cuda_kernel = use_cuda_kernel
        self.normalize_input = normalize_input
        self.channel_select_fixed_indices = channel_select_fixed_indices

        self.X_mean_: float | None = None
        self.X_std_: float | None = None

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
        if self.channel_select_fixed_indices is not None:
            n_channels = X.shape[1]
            out_of_range = [i for i in self.channel_select_fixed_indices if i < 0 or i >= n_channels]
            if out_of_range:
                raise ValueError(
                    f"channel_select_fixed_indices out of range for {n_channels} channels: {out_of_range}."
                )
            X = X[:, self.channel_select_fixed_indices, :]
        if not self.normalize_input:
            return X
        if fit:
            ref = X if train_idx is None else X[train_idx]
            self.X_mean_, self.X_std_ = fit_global_zscore_stats(ref)
        if self.X_mean_ is None or self.X_std_ is None:
            raise ValueError("Normalization stats are not initialized -- call fit() first.")
        return apply_global_zscore(X, self.X_mean_, self.X_std_)

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> CGMambaNet:
        x = features[0] if isinstance(features, tuple) else features
        _, n_channels, n_time = x.shape
        return CGMambaNet(
            n_channels=int(n_channels),
            n_time=int(n_time),
            n_classes=n_classes,
            patch_size=self.patch_size,
            d_embed=self.d_embed,
            gcn_layers=self.gcn_layers,
            mamba_n_layers=self.mamba_n_layers,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand_factor=self.mamba_expand_factor,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            lstm_dropout=self.lstm_dropout,
            head_dropout=self.head_dropout,
            use_cuda_kernel=self.use_cuda_kernel,
            **kwargs,
        )
