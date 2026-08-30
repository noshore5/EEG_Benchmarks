"""TMC-T (Temporal Multi-Channel Transformer): a conv-tokenizer +
Transformer-encoder architecture for epileptic seizure prediction, built
from the paper's own description --

    Godoy, R. V. et al. "EEG-Based Epileptic Seizure Prediction Using
    Temporal Multi-Channel Transformers." arXiv:2209.11172 (2022).

NOT vendored code -- no public implementation is linked from the arXiv
listing (checked 2026-08-30). Reconstructed from the paper's Section III
(architecture figure + prose), not from a released repo, same status as
`cg_mambanet_classifier.py`'s CG-MambaNet build: "built to the paper's
stated spec", not "reproduces the paper's exact reported numbers".

WHY THIS PAPER: found while looking for a SOTA benchmark closer to this
repo's own per-subject leave-one-seizure-out (LOSO) protocol than
CG-MambaNet's cross-patient evaluation (see cg_mambanet_classifier.py's
module docstring and `pipeline_comparison_gru_mamba_dbconformer_
slimseiz.md`'s addenda). Godoy et al. IS patient-specific and reports a
chb01-specific number (accuracy 99.97%, sensitivity 100%, AUC 99.97% for
their best config) -- but their own evaluation is a random 80/20 train/
test split (+5-fold CV for hyperparameter tuning only), NOT
leave-one-seizure-out. Random splits on overlapping/adjacent seizure-
prediction windows are a well-documented leakage risk (adjacent windows
from the same seizure can land on both sides of the split) -- so their
chb01 number is not treated here as a target to hit, but as a number to
compare against once this architecture is run under this repo's own
honest LOSO protocol. That comparison (same architecture, weak split vs.
strict LOSO) is the actual point of this pipeline, not a reproduction
attempt.

ONLY TMC-T IS BUILT HERE, NOT TMC-ViT (the paper's other, slightly
better-performing variant): TMC-ViT reshapes the 23-channel EEG into a
21x21 2D "image" via 4 conv layers before patchifying it, and the paper
does not state how channels/time map onto that 21x21 grid (checked the
paper's full text, 2026-08-30) -- the same kind of unspecified-handoff gap
CG-MambaNet's CNN->GCN interface had. Building that faithfully would mean
inventing an axis mapping the paper never describes, which isn't
"reconstructing the paper's spec" so much as designing a new spec. TMC-T's
tokenizer, by contrast, is fully specified (exact filter counts, kernel
sizes, pooling) with no axis ambiguity.

DEVIATIONS / INTERPRETIVE CHOICES from the paper's literal spec:

  - Labeling: this repo's own `--sph`/`--sop` defaults (5min/15min) are
    used, NOT the paper's own 5min SPH / 30-or-60min SOP -- keeps this
    pipeline's labels identical to every other row in the comparison
    table, the same reasoning CG-MambaNet's docstring gives.
  - Sampling rate / preprocessing: this repo's native 256Hz, raw z-scored
    windows (this repo's existing GLOBAL-scalar `fit_global_zscore_stats`/
    `apply_global_zscore`), no bandpass/notch filtering -- the paper
    itself also uses raw signal ("We employ raw data relying on DL
    algorithms' ability to automatically learn discriminative features"),
    so this is a smaller deviation than most of this repo's other
    raw-EEG pipelines' normalization choices.
  - Conv tokenizer padding: the paper gives kernel widths (20, 20, 10) and
    pool widths (10, 6, 6) for the 3 conv blocks but not the padding used.
    `same`-style padding is used here (kernel_width // 2) so only the
    pooling stages reduce sequence length -- keeps the tokenizer's output
    length a deterministic function of window length, needed because this
    repo's window length (30s @ 256Hz = 7680 samples for
    `label_mode=prediction`) doesn't match either of the paper's own
    windows (5s/1280 samples or 20s/5120 samples).
  - Token construction: the paper's conv operates with kernel/pool HEIGHT
    always 1 (e.g. "1x20", "1x10"), meaning the 23-channel axis is never
    mixed or reduced by the tokenizer -- consistent with treating each
    channel independently until the Transformer's attention itself mixes
    them. This file follows that reading literally: conv2d over an
    input shaped (B, 1, C, T) with kernel (1, k) and pool (1, p), so the
    channel axis survives as a spatial dimension of the conv the whole
    way through, then (channel, reduced-time) cells are flattened into
    the Transformer's token sequence together -- i.e. a token is
    (one channel, one reduced time-step), sequence length = C * T'.
    Consistent with the paper's own architecture figure showing tokens
    fed to the Transformer as a per-channel-per-timestep grid, but the
    paper's prose doesn't spell out the flatten order (channel-major vs.
    time-major) -- this uses channel-major (all of one time-step's
    channels are adjacent in the sequence) since nothing in the paper
    suggests otherwise and it mirrors CG-MambaNet's own "consecutive
    sequence steps are the fixed-montage channels" reading for a
    similarly-shaped ambiguity.
  - Position embeddings: paper says "learnable embeddings" with no further
    detail -- an `nn.Parameter` of shape (1, seq_len, d_model), added to
    the token sequence before the encoder, added (not concatenated) same
    as ViT/BERT-style learnable position embeddings generally are.
  - Classification head: paper specifies TMC-T's head as "two dense layers
    with dropout 0.5" before a final 1-neuron sigmoid, but never states
    the two dense layers' widths (only TMC-ViT's head widths, 2048/1024,
    are given, and that's a materially larger model). This file uses a
    reasonable small width (128 -> 64) matching this repo's other raw-EEG
    classifiers' head sizes (DBConformer's is 64-wide, CG-MambaNet's is
    64-wide), not a value recoverable from the paper. Global average
    pooling over the token sequence feeds the head (the paper doesn't
    state a pooling method either -- no CLS token is mentioned in the
    architecture description, so mean-pooling over all tokens is used,
    the standard choice when no CLS token exists).
  - Loss / output: paper trains with binary cross-entropy on a single
    sigmoid output. This repo's shared training loop
    (`TorchEEGClassifier`/`_train_loop` in common.py) trains via
    `CrossEntropyLoss` on `n_classes` logits -- head ends 128->64->2
    instead of 128->64->1+sigmoid, same convention every other classifier
    in this repo already uses (CG-MambaNet, DBConformer, SlimSeiz).

Architecture otherwise follows the paper's own stated TMC-T shapes and
hyperparameters as closely as the deviations above allow: 3-block conv
tokenizer (16, 32, 32 filters; kernel widths 20, 20, 10; pool widths
10, 6, 6), embedding dim 32 (= final conv's filter count), 8 attention
heads, feedforward hidden 64, dropout 0.1 on the embedding+position sum
and on each sub-layer's residual, one Transformer encoder layer (the
paper's Figure 2 shows a single encoder block, not a stack -- unlike
TMC-ViT's stated "8 Transformer layers", TMC-T's layer count is never
given as > 1 anywhere in the text, so this uses 1 layer; see
`n_encoder_layers` if that reading turns out wrong and a stack is wanted).
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from Epilepsy.pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats
except ModuleNotFoundError:
    from pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats


class _ConvTokenizer(nn.Module):
    """3-block conv2d tokenizer, kernel/pool HEIGHT always 1 (channel axis
    never mixed/reduced) -- see module docstring's "Token construction"
    entry. Input (B, 1, C, T) -> output (B, C*T', d_model) token sequence,
    channel-major flatten order."""

    def __init__(
        self,
        mid_channels: tuple[int, int, int] = (16, 32, 32),
        kernel_widths: tuple[int, int, int] = (20, 20, 10),
        pool_widths: tuple[int, int, int] = (10, 6, 6),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, p in zip(mid_channels, kernel_widths, pool_widths):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, p)),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.d_model = mid_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.net(x)  # (B, F, C, T')
        b, f, c, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, c * t, f)  # (B, C*T', F) -- channel-major
        return x


class TMCTransformer(nn.Module):
    """Conv tokenizer -> learnable position embedding -> Transformer
    encoder -> mean-pool -> MLP head. Input `(B, C, T)` raw EEG windows;
    see module docstring for every deviation from the paper's literal
    spec."""

    def __init__(
        self,
        n_channels: int,
        n_time: int,
        n_classes: int = 2,
        d_model: int = 32,
        n_heads: int = 8,
        ffn_hidden: int = 64,
        n_encoder_layers: int = 1,
        dropout: float = 0.1,
        head_hidden: int = 128,
        head_dropout: float = 0.5,
        **_unused,  # tolerate the device= kwarg TorchEEGClassifier._build_model_from_features passes through
    ) -> None:
        super().__init__()
        self.tokenizer = _ConvTokenizer()
        assert self.tokenizer.d_model == d_model, (
            f"_ConvTokenizer's fixed output width ({self.tokenizer.d_model}) must match "
            f"d_model={d_model}."
        )

        with torch.no_grad():
            probe = torch.zeros(1, n_channels, n_time)
            seq_len = self.tokenizer(probe).shape[1]

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        tokens = self.tokenizer(x)  # (B, L, d_model)
        tokens = self.embed_dropout(tokens + self.pos_embed)
        encoded = self.encoder(tokens)  # (B, L, d_model)
        pooled = encoded.mean(dim=1)  # (B, d_model) -- no CLS token, see module docstring
        return self.head(pooled)


class GodoyTMCClassifier(TorchEEGClassifier):
    """sklearn-style wrapper around TMCTransformer -- global z-score
    normalization (same convention as DBConformerClassifier/
    SlimSeizClassifier/CGMambaNetClassifier). No CWT/STFT preprocessing,
    no disk caching -- same reasoning as those classifiers, and matches
    the paper's own choice to train on raw signal directly."""

    _estimator_type = "classifier"
    model_label = "GodoyTMC"

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 8,
        ffn_hidden: int = 64,
        n_encoder_layers: int = 1,
        dropout: float = 0.1,
        head_hidden: int = 128,
        head_dropout: float = 0.5,
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
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_hidden = ffn_hidden
        self.n_encoder_layers = n_encoder_layers
        self.dropout = dropout
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.normalize_input = normalize_input

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
        if not self.normalize_input:
            return X
        if fit:
            ref = X if train_idx is None else X[train_idx]
            self.X_mean_, self.X_std_ = fit_global_zscore_stats(ref)
        if self.X_mean_ is None or self.X_std_ is None:
            raise ValueError("Normalization stats are not initialized -- call fit() first.")
        return apply_global_zscore(X, self.X_mean_, self.X_std_)

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> TMCTransformer:
        x = features[0] if isinstance(features, tuple) else features
        _, n_channels, n_time = x.shape
        return TMCTransformer(
            n_channels=int(n_channels),
            n_time=int(n_time),
            n_classes=n_classes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            ffn_hidden=self.ffn_hidden,
            n_encoder_layers=self.n_encoder_layers,
            dropout=self.dropout,
            head_hidden=self.head_hidden,
            head_dropout=self.head_dropout,
            **kwargs,
        )
