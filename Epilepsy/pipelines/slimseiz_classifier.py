"""SlimSeiz: a lightweight 1-D conv + Mamba (selective state-space) EEG
classifier, vendored from:

    Wang, Z. et al., "SlimSeiz" (see ../../DBConformer/models/SlimSeiz.py in
    this repo for the author's original file, copied 2026-08-25 -- shipped
    alongside DBConformer in the same upstream repo).

The Mamba block here is a self-contained, sequential-scan implementation
(the same structure as github.com/johnma2006/mamba-minimal, credited in the
original file's own comments) -- it has no dependency on the `mamba-ssm`
CUDA package or on this repo's own `mambapy`-based
`_DenseEdgeMambaTemporal`/`_DenseEdgeMambaContinuous` (cwt_gnn_classifiers.py).
Kept as its own independent implementation rather than swapped for either of
those: it's architecturally different (a single MambaBlock mixing over the
model's own 32-channel conv feature map, not the per-edge/per-node sequence
those classes operate on) and swapping it would no longer be "the paper's
model", just something inspired by it.

ADAPTATIONS made when vendoring into this file:

  - Dropped `os`, `sys`, `random`, `mne`, `matplotlib.pyplot` imports --
    present in the original file (evidently left over from a larger
    training script it was extracted from) but unused by the model classes
    themselves.
  - `SlimSeiz.forward` no longer does `x = x.squeeze(1)` on entry -- same
    reasoning as DBConformerClassifier's identical adaptation (see that
    module's docstring): this repo's windows are already `(n_samples,
    n_channels, n_timepoints)`, no redundant leading singleton axis.
  - `SlimSeiz.__init__` gained an `n_classes` parameter (the original
    hardcoded `nn.Linear(32, 2)`, i.e. always binary) so
    `SlimSeizClassifier` can pass through however many classes
    `TorchEEGClassifier.fit()` actually sees, the same as every other
    classifier in this package -- CHB-MIT detection/prediction are binary
    today, but nothing here should silently assume that forever.
  - `forward` returns just the classification logits (`x_res`), not the
    original `(x_digits, x_res)` tuple -- `x_digits` (the pre-classifier
    pooled features) isn't used anywhere in this repo's training/eval loop,
    and returning it first would have been silently treated as the trained-
    against logits by `common.py`'s `_model_forward` (see
    DBConformerClassifier's docstring for the same finding on that model).

Everything else (the conv1/conv2_1+conv2_2/conv4_1+conv4_2 stem, the single
MambaBlock temporal mixer, RMSNorm, the adaptive-pool + linear head) is the
authors' original architecture, unmodified.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class MambaBlock(nn.Module):
    """A single Mamba block (selective SSM), as in Figure 3, Section 3.4 of
    the Mamba paper. Sequential (non-parallel-scan) reference implementation."""

    def __init__(self, input_channels):
        super().__init__()

        self.d_model = input_channels
        self.d_inner = self.d_model * 2
        self.dt_rank = math.ceil(self.d_model / 16)
        self.d_state = 16

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            groups=self.d_inner,
            padding=2,
        )

        # x_proj takes in `x` and outputs the input-specific Delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects Delta from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x):
        """x: (b, l, d) -> (b, l, d)"""
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # (b, l, 2*d_inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # (d_in, n) -- input-independent
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in) -- input-dependent (selective)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Discrete SSM recurrence: x(t+1) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t),
        with A discretized via zero-order hold and B via a simplified Euler
        step (see the Mamba paper, Section 2). Sequential over `l`, not the
        parallel/hardware-aware scan the official kernel uses."""
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SlimSeiz(nn.Module):
    def __init__(self, input_channels: int = 3, n_classes: int = 2):
        super().__init__()
        self.input_channels = input_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=21, stride=1, padding=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv4_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=2),
            nn.ReLU(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.mixer = MambaBlock(32)
        self.norm = RMSNorm(32)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.classifier = nn.Sequential(nn.Linear(32, n_classes))

    def forward(self, x):  # x: (B, C, T)
        x = self.conv1(x)
        x = self.pool3(self.conv2_1(x) + self.conv2_2(x))
        x = self.conv4_1(x) + self.conv4_2(x)
        # x: (batch, channels=32, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        x = self.mixer(self.norm(x)) + x
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.adaptive_avg_pool(x)
        x_digits = x.contiguous().view(x.size(0), -1)
        return self.classifier(x_digits)


try:
    from Epilepsy.pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats
except ModuleNotFoundError:
    from pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats


class SlimSeizClassifier(TorchEEGClassifier):
    """sklearn-style wrapper around SlimSeiz: optional global z-score
    normalization (same convention as DBConformerClassifier) feeding
    SlimSeiz directly on raw `(n_samples, n_channels, n_timepoints)`
    windows. No CWT/STFT preprocessing, no disk caching -- same reasoning
    as DBConformerClassifier (this model's per-window memory footprint is
    tiny compared to the CWT/dense-edge pipelines' tensors)."""

    _estimator_type = "classifier"
    model_label = "SlimSeiz"

    def __init__(
        self,
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

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> SlimSeiz:
        x = features[0] if isinstance(features, tuple) else features
        _, n_channels, _ = x.shape
        return SlimSeiz(input_channels=int(n_channels), n_classes=n_classes)
