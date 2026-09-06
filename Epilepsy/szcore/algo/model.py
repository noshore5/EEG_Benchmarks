"""Standalone copy of this repo's TMC-T architecture (godoy_tmc).

Vendored verbatim from ``Epilepsy/pipelines/godoy_tmc_classifier.py`` so the
SzCORE container can rebuild the network and load a checkpoint WITHOUT
importing the whole ``Epilepsy`` package (which pulls moabb, mne, the CWT
stack, ...). Keep the two classes below byte-identical to the source; if
that file's architecture changes, re-copy and retrain the checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn


class _ConvTokenizer(nn.Module):
    """3-block conv2d tokenizer, kernel/pool HEIGHT always 1 (channel axis
    never mixed/reduced). Input (B, 1, C, T) -> output (B, C*T', d_model)
    token sequence, channel-major flatten order."""

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
    encoder -> mean-pool -> MLP head. Input ``(B, C, T)`` raw EEG windows."""

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
        **_unused,
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
        pooled = encoded.mean(dim=1)  # (B, d_model)
        return self.head(pooled)
