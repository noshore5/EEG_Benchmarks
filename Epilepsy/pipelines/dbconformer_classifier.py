"""DBConformer: dual-branch (temporal + spatial) convolutional Transformer for
EEG decoding, vendored from:

    Wang, Z. et al. "DBConformer: Dual-Branch Convolutional Transformer for
    EEG Decoding." (see ../../DBConformer/models/DBConformer.py in this repo
    for the author's original file, copied 2026-08-25).

ADAPTATIONS made when vendoring the architecture into this file (all
behavior-preserving on the parts that are kept -- these are dependency/
plumbing cleanups, not an attempt to re-derive the model):

  - Dropped the `timm` dependency: the only thing `PatchEmbeddingTemporal.
    initParms` used from it was `trunc_normal_` weight init, which
    `torch.nn.init.trunc_normal_` already provides (torch is pinned in this
    repo's requirements.txt; timm is not, and pulling in a whole model-zoo
    package for one init function isn't worth the extra dependency).
  - Dropped the global `from torch.backends import cudnn; cudnn.benchmark =
    False; cudnn.deterministic = True` the original script set as an IMPORT
    SIDE EFFECT. That's reasonable in a standalone training script but not
    here -- this module is imported unconditionally by run_pipelines.py
    alongside every other pipeline, so a module-level cudnn.deterministic=True
    would silently slow down every OTHER pipeline's CUDA runs too. seed
    control for this classifier's own runs still goes through this repo's
    usual set_seed() (common.py), called once per fit().
  - Dropped `CrossAttention` and `SEBlock` (dead code in the original file --
    never instantiated by `DBConformer.forward`; `SEBlock`'s own docstring
    even says "bad performance") and the custom `GELU` class (unused --
    `FeedForwardBlock` already uses `nn.GELU()`).
  - Replaced the `args` namespace `DBConformer.__init__` originally expected
    (an argparse.Namespace from the authors' own CLI, carrying `data_name`,
    `chn`, `spa_dim`, `patch_size`, `time_sample_num`, `class_num`,
    `gate_flag`, `posemb_flag`, `branch`, `chn_atten_flag`) with plain
    keyword arguments -- this repo has no equivalent CLI namespace, and an
    opaque args object would just be a repackaging of the same values
    DBConformerClassifier already needs to hold as sklearn-style
    constructor params anyway.
  - `PatchEmbeddingTemporal`'s `time_points`/`num_classes` constructor
    params were accepted but never read anywhere in the original file --
    dropped rather than carried forward unused.
  - `Stem`'s `data_name != 'MI1-7'` branch (drop the last timepoint unless
    the dataset is literally the authors' own "MI1-7" one) is replaced with
    an explicit `trim_last_timepoint: bool` flag. The original check exists
    because some of the authors' datasets hand DBConformer one MORE
    timepoint than their own `time_sample_num` config value expects (their
    comment: "Example: 14001 has 1001 time points, we exclude the final
    point") -- a quirk of THEIR data loaders, not of the architecture. This
    repo's own windows (ContinuousLabelingParadigm) are exactly
    `window_length * sampling_rate` samples with no off-by-one, so
    DBConformerClassifier below always constructs the model with
    `trim_last_timepoint=False`; the flag is kept (not deleted) only so the
    original behavior is still reachable/documented if that ever changes,
    rather than silently lost.
  - `DBConformer.forward` no longer does `x = x.squeeze(1)` on entry --
    that assumed callers always hand it a redundant leading singleton
    "image channel" axis (i.e. `(B, 1, C, T)`), a convention from the
    authors' own moabb-style trial tensors. This repo's window arrays are
    already `(n_samples, n_channels, n_timepoints)` with no such axis, so
    the model here takes `(B, C, T)` directly.
  - `forward` returns just the classification logits (`out`), not the
    original `(x_fused, out)` tuple -- `x_fused` (the pooled fused
    embedding) isn't used anywhere in this repo's training/eval loop, and
    `common.py`'s `TorchEEGClassifier._model_forward` treats a tuple's
    FIRST element as the trained-against logits (that convention is for
    GNN classifiers whose second element is a scalar diagnostic like
    edge_density, not an embedding) -- returning the embedding first would
    have silently made IT the thing CrossEntropyLoss trains against instead
    of the real class logits. TruongSTFTCNNClassifier's TruongCNNCore
    (truong_stft_cnn_classifier.py) uses this same plain-tensor-return
    convention for exactly the same reason.

Everything else (Stem's multi-branch temporal conv stem, the dual temporal/
spatial Transformer encoders, the channel-attention pooling + fusion, the
classification head) is the authors' original architecture, unmodified.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

try:
    from Epilepsy.pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats
except ModuleNotFoundError:
    from pipelines.common import TorchEEGClassifier, apply_global_zscore, fit_global_zscore_stats


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Stem(nn.Module):
    """Multi-branch temporal conv stem -- `radix` parallel depthwise temporal
    convs at successively halved kernel sizes, summed (InterFre) then
    average-pooled into patches."""

    def __init__(
        self,
        in_planes,
        out_planes=64,
        kernel_size=63,
        patch_size=125,
        radix=2,
        trim_last_timepoint: bool = False,
    ):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.trim_last_timepoint = trim_last_timepoint

        self.sconv = Conv(
            nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups=radix),
            bn=nn.BatchNorm1d(self.mid_planes),
            activation=None,
        )

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(
                Conv(
                    nn.Conv1d(
                        self.out_planes,
                        self.out_planes,
                        kernel_size,
                        1,
                        groups=self.out_planes,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    bn=nn.BatchNorm1d(self.out_planes),
                    activation=None,
                )
            )
            kernel_size //= 2

        self.interFre = InterFre()

        self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]
        out = self.interFre(out)
        if self.trim_last_timepoint:
            out = out[:, :, :-1]
        out = self.downSampling(out)
        out = self.dp(out)
        return out


class PatchEmbeddingTemporal(nn.Module):
    """Outputs patch embeddings of shape (B, P, D) from (B, C, T)."""

    def __init__(self, in_planes, out_planes, kernel_size, radix, patch_size, trim_last_timepoint: bool = False):
        super().__init__()
        self.stem = Stem(
            in_planes=in_planes * radix,
            out_planes=out_planes,
            kernel_size=kernel_size,
            patch_size=patch_size,
            radix=radix,
            trim_last_timepoint=trim_last_timepoint,
        )
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, T)
        out = self.stem(x)  # (B, D, P)
        out = out.permute(0, 2, 1)  # -> (B, P, D)
        return out


class PatchEmbeddingSpatial(nn.Module):
    def __init__(self, spa_dim, emb_size=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spa_dim, emb_size),
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        x = x.unsqueeze(2)  # (B, C, 1, T)
        x = x.reshape(B * C, 1, T)  # -> (B*C, 1, T)
        x = self.encoder(x)  # -> (B*C, emb_size)
        x = x.view(B, C, -1)  # -> (B, C, emb_size)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)


class Gate_FC(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.fc = nn.Linear(emb_size * 2, emb_size)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)


class DBConformer(nn.Module):
    def __init__(
        self,
        chn: int,
        time_sample_num: int,
        n_classes: int = 2,
        emb_size: int = 40,
        tem_depth: int = 5,
        chn_depth: int = 5,
        patch_size: int = 125,
        spa_dim: int = 16,
        gate_flag: bool = False,
        posemb_flag: bool = True,
        branch: str = "all",
        chn_atten_flag: bool = True,
        trim_last_timepoint: bool = False,
        **_unused,  # tolerate the device= kwarg TorchEEGClassifier._build_model_from_features passes through
    ) -> None:
        super().__init__()

        self.embedding = PatchEmbeddingTemporal(
            in_planes=chn,
            out_planes=emb_size,
            kernel_size=63,
            radix=1,
            patch_size=patch_size,
            trim_last_timepoint=trim_last_timepoint,
        )
        self.channel_embedding = PatchEmbeddingSpatial(spa_dim=spa_dim, emb_size=emb_size)
        effective_time = time_sample_num - 1 if trim_last_timepoint else time_sample_num
        if effective_time % patch_size != 0:
            raise ValueError(
                f"patch_size={patch_size} does not evenly divide the (possibly trimmed) window "
                f"length {effective_time} samples (time_sample_num={time_sample_num}, "
                f"trim_last_timepoint={trim_last_timepoint}) -- pick a patch_size that divides it, "
                "or change --window-length so window_length*sampling_rate does."
            )
        self.P = effective_time // patch_size
        self.C = chn
        self.D = emb_size
        self.gate_flag = gate_flag
        self.posemb_flag = posemb_flag
        self.branch = branch
        self.chn_atten_flag = chn_atten_flag

        if posemb_flag:
            self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        self.temporal_transformer = TransformerEncoder(tem_depth, emb_size)
        self.spatial_transformer = TransformerEncoder(chn_depth, emb_size)
        if gate_flag or branch in ("temporal", "spatial"):
            self.gate_fc = Gate_FC(emb_size)
            self.classifier = ClassificationHead(emb_size, n_classes)
        else:
            self.classifier = ClassificationHead(emb_size * 2, n_classes)
            if chn_atten_flag:
                self.spatial_attn_pool = nn.Sequential(
                    nn.Linear(emb_size, emb_size),
                    nn.Tanh(),
                    nn.Linear(emb_size, 1),
                )

    def forward(self, x):  # x: (B, C, T)
        x_embed = self.embedding(x)  # -> (B, P, D)
        x_embed_spatial = self.channel_embedding(x)  # (B, C, D)
        if self.posemb_flag:
            x_embed = x_embed + self.pos_embedding_temporal
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial

        x_temporal = self.temporal_transformer(x_embed)  # (B, P, D)
        x_spatial = self.spatial_transformer(x_embed_spatial)  # (B, C, D)

        if self.branch == "temporal":
            x_fused = x_temporal.mean(dim=1)
            out = self.classifier(x_fused)
        elif self.branch == "spatial":
            x_fused = x_spatial.mean(dim=1)
            out = self.classifier(x_fused)
        elif self.branch == "all":
            if self.gate_flag:
                gate = torch.sigmoid(
                    self.gate_fc(torch.cat([x_temporal.mean(dim=1), x_spatial.mean(dim=1)], dim=-1))
                )
                x_fused = gate * x_spatial.mean(dim=1) + (1 - gate) * x_temporal.mean(dim=1)
            else:
                if self.chn_atten_flag:
                    x_t = x_temporal.mean(dim=1)
                    attn_scores = self.spatial_attn_pool(x_spatial)  # (B, C, 1)
                    attn_weights = torch.softmax(attn_scores, dim=1)
                    x_s = torch.sum(attn_weights * x_spatial, dim=1)  # (B, D)
                    x_fused = torch.cat([x_t, x_s], dim=-1)  # (B, 2*D)
                else:
                    x_fused = torch.cat([x_temporal.mean(dim=1), x_spatial.mean(dim=1)], dim=-1)
            out = self.classifier(x_fused)
        else:
            raise ValueError(f"Unsupported branch={self.branch!r} (expected 'all', 'temporal', or 'spatial').")
        return out


class DBConformerClassifier(TorchEEGClassifier):
    """sklearn-style wrapper around DBConformer: optional global z-score
    normalization (same `fit_global_zscore_stats`/`apply_global_zscore`
    convention `_BaseCWTGNNClassifier` uses) feeding DBConformer directly on
    raw `(n_samples, n_channels, n_timepoints)` windows -- no CWT/STFT
    preprocessing, no disk caching (this model's memory footprint per window
    is a few KB, nowhere near the CWT/dense-edge tensors' scale that made
    caching worthwhile for those pipelines).
    """

    _estimator_type = "classifier"
    model_label = "DBConformer"

    def __init__(
        self,
        emb_size: int = 40,
        tem_depth: int = 5,
        chn_depth: int = 5,
        # 2026-08-25: 128 (not the paper's own 125) -- divides both this
        # repo's detection-mode window (4s @ 256Hz = 1024 samples, 1024/128=8
        # patches) and its prediction-mode window (30s @ 256Hz = 7680
        # samples, 7680/128=60 patches) evenly, so the default works for
        # both --label-mode values without per-mode tuning. Not otherwise
        # validated -- see DBConformer's ValueError if a different
        # --window-length doesn't divide evenly.
        patch_size: int = 128,
        spa_dim: int = 16,
        gate_flag: bool = False,
        posemb_flag: bool = True,
        branch: str = "all",
        chn_atten_flag: bool = True,
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
        self.emb_size = emb_size
        self.tem_depth = tem_depth
        self.chn_depth = chn_depth
        self.patch_size = patch_size
        self.spa_dim = spa_dim
        self.gate_flag = gate_flag
        self.posemb_flag = posemb_flag
        self.branch = branch
        self.chn_atten_flag = chn_atten_flag
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

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> DBConformer:
        x = features[0] if isinstance(features, tuple) else features
        _, n_channels, n_time = x.shape
        return DBConformer(
            chn=int(n_channels),
            time_sample_num=int(n_time),
            n_classes=n_classes,
            emb_size=self.emb_size,
            tem_depth=self.tem_depth,
            chn_depth=self.chn_depth,
            patch_size=self.patch_size,
            spa_dim=self.spa_dim,
            gate_flag=self.gate_flag,
            posemb_flag=self.posemb_flag,
            branch=self.branch,
            chn_atten_flag=self.chn_atten_flag,
            trim_last_timepoint=False,  # see module docstring -- this repo's windows have no off-by-one artifact
            **kwargs,
        )
