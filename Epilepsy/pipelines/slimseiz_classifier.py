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

STAGE 1 -- CHANNEL SELECTION (added 2026-08-25): the paper's headline
numbers (94.8% accuracy / 95.5% sensitivity / 94.0% specificity on
CHB-MIT) are for this network fed an *adaptively selected* 8-of-22 channel
subset, not the full montage -- that selection is a separate stage in the
upstream repo (``Loop_select_ch_PCA_SMOTE_DT.ipynb``), not part of this
vendored model file. Every SlimSeiz run in this repo before this date fed
the network the full 23-channel chb01 montage, which is a different (and
weaker-performing, per the 2026-08-25 session comparison) condition than
what "SlimSeiz" refers to in the paper. `SlimSeizClassifier` now runs that
selection itself -- see `slimseiz_channel_select.py` for the ported
algorithm and its documented deviations from the notebook -- inside each
call to `fit()`, using only the windows passed to that call (so under this
repo's LOSO evaluation, only the fold's training seizures ever inform
which channels get selected for that fold; the held-out seizure never
does). Toggle with `select_channels=False` to fall back to the full
montage (the old behavior).

FIXED CHANNELS (added 2026-08-25): the upstream repo's own
``Common_channesl.ipynb`` (fetched from github.com/guoruilu/SlimSeiz, not
vendored as a file here) runs the per-patient stage-1 selection across all
24 CHB-MIT patients, then tallies which channels land in each patient's
own top-8 most often cohort-wide. The 8 that come out on top --
``P3-O1, P8-O2, C3-P3, C4-P4, FZ-CZ, P4-O2, CZ-PZ, F3-C3`` (counts 18, 18,
18, 17, 17, 17, 15, 14 out of 24 patients) -- read like a single fixed
montage used for the paper's headline numbers, not a genuinely per-patient
adaptive one; for chb01 specifically the notebook's own per-patient top-8
overlaps this set in 7 of 8 channels (missing only F3-C3). Pass
`channel_select_fixed_indices` to reproduce that fixed set exactly instead
of re-deriving channels from this repo's own PCA+SMOTE+DecisionTree port
each fold -- it bypasses stage 1 entirely (no PCA/SMOTE/DecisionTree call
at all, so none of that stage's memory profile applies), just slices `X`
to the given channel indices before stage 2. `select_channels` is ignored
when this is set. Resolving the 8 fixed *names* above to CHB-MIT channel
*indices* is the caller's job (channel order isn't visible from inside
this classifier) -- see run_pipelines.py's `--slimseiz-fixed-channels`.
"""

from __future__ import annotations

import math

import numpy as np
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
    from Epilepsy.pipelines.common import (
        TorchEEGClassifier,
        apply_global_zscore,
        fit_global_zscore_stats,
        validate_eeg_X,
    )
    from Epilepsy.pipelines.slimseiz_channel_select import select_slimseiz_channels
except ModuleNotFoundError:
    from pipelines.common import (
        TorchEEGClassifier,
        apply_global_zscore,
        fit_global_zscore_stats,
        validate_eeg_X,
    )
    from pipelines.slimseiz_channel_select import select_slimseiz_channels


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
        select_channels: bool = True,
        n_select_channels: int = 8,
        channel_select_iterations: int = 30,
        channel_select_pca_components: int = 60,
        channel_select_test_size: float = 0.3,
        channel_select_max_samples: int | None = 1000,
        channel_select_fixed_indices: list[int] | None = None,
    ) -> None:
        self.normalize_input = normalize_input

        self.X_mean_: float | None = None
        self.X_std_: float | None = None

        # Stage 1 (see this module's docstring and slimseiz_channel_select.py):
        # select_channels=True (default) reproduces the paper's adaptive
        # channel-selection condition its SOTA numbers are reported under,
        # re-run inside every fit() call on that call's own training
        # windows only. select_channels=False keeps the pre-2026-08-25
        # behavior (full montage, no selection) for anyone who wants that
        # comparison point.
        self.select_channels = select_channels
        self.n_select_channels = n_select_channels
        self.channel_select_iterations = channel_select_iterations
        self.channel_select_pca_components = channel_select_pca_components
        self.channel_select_test_size = channel_select_test_size
        self.channel_select_max_samples = channel_select_max_samples
        self.channel_select_fixed_indices = channel_select_fixed_indices
        self.selected_channel_idx_: np.ndarray | None = None

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

    def fit(self, X, y, validation_groups=None, metadata=None):
        """Stage 1 (channel selection) runs here, before TorchEEGClassifier's
        shared fit() logic -- see this module's docstring. `X`/`y` at this
        point are exactly what the caller passed in (this repo's LOSO loop
        calls fit() once per fold with that fold's training seizures only,
        the held-out seizure never included), so selection never sees the
        held-out seizure's windows."""
        X = validate_eeg_X(X)
        if self.channel_select_fixed_indices is not None:
            # Bypasses stage 1 (select_slimseiz_channels) entirely -- see
            # this module's "FIXED CHANNELS" docstring section. No PCA/
            # SMOTE/DecisionTree call happens on this path at all.
            n_channels = X.shape[1]
            out_of_range = [
                i for i in self.channel_select_fixed_indices if i < 0 or i >= n_channels
            ]
            if out_of_range:
                raise ValueError(
                    f"channel_select_fixed_indices out of range for "
                    f"{n_channels} channels: {out_of_range}."
                )
            self.selected_channel_idx_ = np.asarray(
                self.channel_select_fixed_indices, dtype=int
            )
        elif self.select_channels:
            self.selected_channel_idx_ = select_slimseiz_channels(
                X,
                y,
                n_select=self.n_select_channels,
                n_iterations=self.channel_select_iterations,
                pca_components=self.channel_select_pca_components,
                test_size=self.channel_select_test_size,
                seed=self.seed,
                verbose=self.verbose,
                max_samples=self.channel_select_max_samples,
            )
        else:
            self.selected_channel_idx_ = None
        return super().fit(X, y, validation_groups=validation_groups, metadata=metadata)

    def _prepare_features(self, X, *, fit: bool, train_idx=None):
        if self.selected_channel_idx_ is not None:
            X = X[:, self.selected_channel_idx_, :]
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
