"""Non-graph control for event_mode="dense" (2026-08-10).

Question: given dense_edge_conv's per-edge features (the thing that scored
0.8886 event-pathway-alone -- see session_notes/run_logs/2026-08-10_sparse-
evidence-gnn_dense-event-mode-sparse-vs-dense.md), is the graph/message-
passing layer around it (sparse_message_mlp -> _aggregate_events'
scatter-to-node -> _propagate_hops) adding anything at all, or is the win
entirely from dense feature extraction itself?

Reuses dense_edge_conv and the exact [B, 4, E, T, F] input
(_build_dense_edge_input/compute_dense_edge_input, via
SparseEvidenceGNNClassifier's own event_mode="dense" precompute) UNCHANGED
-- same architecture, same defaults (dense_conv_kernel_size=5,
dense_conv_pool_size=4, dense_conv_intermediate_channels=32,
dense_conv_out_channels=8). After dense_edge_conv produces
[B, E, dense_conv_out_channels], this SKIPS sparse_message_mlp,
_aggregate_events, and _propagate_hops entirely: flattens directly to
[B, E * dense_conv_out_channels] and feeds a single nn.Linear classifier
head (matching sparse_classifier's own "one Linear, no hidden layer"
simplicity -- so any difference in score isn't just "bigger classifier
head"). No scatter-to-node, no src/dst channel identity, no edge topology
used beyond "which flattened slot this edge's features land in" (a fixed,
arbitrary-but-consistent order, not a message-passing structure). Also,
deliberately, no ChannelSignalEncoder anywhere -- this isolates dense event
features from BOTH graph structure and raw-signal channel embeddings,
unlike feature_ablation="zero_channel_embed" (which keeps the graph, drops
channel embeddings).

Practical cost: expect similar wall-clock to event_mode="dense" itself
(~450-530s per seed, both CrossSessionEvaluation folds) -- dense_edge_conv
is the dominant cost in both cases (a real Conv2d forward+backward over the
full native-resolution [edge=36, time~1001, freq=16] array every training
step); removing sparse_message_mlp/_aggregate_events/_propagate_hops saves
comparatively little.

USAGE
-----
    python BCI/tests/run_dense_edge_flat_control.py smoke
    python BCI/tests/run_dense_edge_flat_control.py run
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline

from BCI.moabb_pipelines.common import set_seed
from BCI.moabb_pipelines.sparse_evidence_gnn_classifier import (
    SparseEvidenceGNNClassifier,
    _build_dense_feature_conv,
)


class DenseEdgeFlatCore(nn.Module):
    """dense_edge_conv (identical construction to event_mode="dense") ->
    flatten -> single Linear classifier head. See module docstring for the
    full rationale -- this deliberately omits every downstream piece
    SparseEvidenceGNNCore.forward() has after dense_edge_conv in "dense"
    mode (sparse_message_mlp, _aggregate_events, _propagate_hops) and every
    upstream piece it has alongside it (channel_encoder).
    """

    def __init__(
        self,
        n_edges: int,
        nfreqs: int,
        n_classes: int,
        dense_conv_kernel_size: int = 5,
        dense_conv_pool_size: int = 4,
        dense_conv_intermediate_channels: int = 32,
        dense_conv_intermediate_channels_reduced: int | None = None,
        dense_conv_out_channels: int = 8,
    ) -> None:
        super().__init__()
        self.n_edges = n_edges
        self.nfreqs = nfreqs
        self.dense_conv_out_channels = dense_conv_out_channels
        # Same 4x (coh, sinφ, cosφ, significance) x nfreqs input-channel
        # folding _dense_edge_features uses -- see that method's docstring
        # in sparse_evidence_gnn_classifier.py.
        self.dense_edge_conv = _build_dense_feature_conv(
            in_channels=4 * nfreqs,
            kernel_size=dense_conv_kernel_size,
            intermediate_channels=dense_conv_intermediate_channels,
            out_channels=dense_conv_out_channels,
            pool_size=dense_conv_pool_size,
            intermediate_channels_reduced=dense_conv_intermediate_channels_reduced,
        )
        # Single Linear, no hidden layer -- matches SparseEvidenceGNNCore's
        # own sparse_classifier (nn.Linear(n_channels * hidden_dim,
        # n_classes)) exactly in "shape" of readout, so a score difference
        # can't be attributed to this control simply having a bigger
        # classifier head than the graph pathway does.
        self.classifier = nn.Linear(n_edges * dense_conv_out_channels, n_classes)

    def forward(self, dense_edge_raw: torch.Tensor) -> torch.Tensor:
        batch_size, c_in, num_edges, n_time, nfreqs = dense_edge_raw.shape
        # Identical reshape to SparseEvidenceGNNCore._dense_edge_features:
        # [B, C_in, E, T, F] -> [B, C_in, F, E, T] -> [B, C_in*F, E, T].
        conv_in = dense_edge_raw.permute(0, 1, 4, 2, 3).reshape(
            batch_size, c_in * nfreqs, num_edges, n_time
        )
        conv_out = self.dense_edge_conv(conv_in)  # [B, dense_conv_out_channels, E, 1]
        feat = conv_out.squeeze(-1).permute(0, 2, 1)  # [B, E, dense_conv_out_channels]
        flat = feat.reshape(batch_size, -1)  # [B, E * dense_conv_out_channels]
        return self.classifier(flat)


# ---------------------------------------------------------------------------
# Canonical config for the throwaway SparseEvidenceGNNClassifier helper used
# purely for its event_mode="dense" precompute (_prepare_features ->
# _precompute_dense_edge_inputs -> compute_dense_edge_input) -- identical to
# the dense-mode write-up's config (session_notes/run_logs/2026-08-10_sparse-
# evidence-gnn_dense-event-mode-sparse-vs-dense.md), NOT read from
# run_sparse_evidence_gnn.py (known mid-edit by a concurrent session
# throughout today -- see that note's "Concurrent-edit check").
# ---------------------------------------------------------------------------
CANONICAL_HELPER_KWARGS: dict[str, object] = dict(
    sampling_rate=250,
    lowest=8.0,
    highest=30.0,
    nfreqs=16,
    cwt_resample_n_time=None,
    coherence_threshold=0.0,
    phase_threshold_deg=10.0,
    channel_encoder_dilation=5,
    hidden_dim=8,  # unused -- this control never builds SparseEvidenceGNNCore's own model
    channel_embed_dim=8,  # unused, same reason
    epochs=1,  # unused -- helper.fit() is never called, only _prepare_features
    batch_size=8,  # used for _precompute_dense_edge_inputs' chunk sizing
    learning_rate=1e-3,
    weight_decay=1e-4,
    grad_clip_norm=0.1,
    normalize_input=True,
    noise_augmentation_enabled=False,
    validation_split=0.0,
    validation_group_column=None,
    early_stopping_patience=None,
    device="auto",
    surrogate_seed=42,
    smooth_kernel_size=(5, 3),
    smooth_kernel_sigma=(None, None),
    coi_enabled=True,
    scale_adaptive_smoothing=False,
    scale_adaptive_cycles=1.5,
    scale_adaptive_max_kernel=101,
    raw_x_resample_n_time=None,
    feature_ablation="zero_channel_embed",  # irrelevant -- forward() on the helper is never called; "none" is no longer a valid value (2026-08-17)
    channel_subset=[1, 5, 7, 8, 9, 10, 11, 13, 17],
    coherence_threshold_mode="surrogate",
    surrogate_count=100,
    surrogate_percentile=99.0,
    mu_band_surrogate_percentile=None,
    mu_band_range_hz=(8.0, 13.0),
    cluster_significance_percentile=95.0,
    surrogate_device="auto",
    surrogate_cache_dir=None,
    surrogate_cache_enabled=True,
    verbose=0,
    n_hops=1,
    freq_aware_hops=False,
    event_mode="dense",
    dense_conv_kernel_size=5,
    dense_conv_pool_size=4,
    dense_conv_intermediate_channels=32,
    dense_conv_out_channels=8,
)


class DenseEdgeFlatClassifier(ClassifierMixin, BaseEstimator):
    # NOTE: ClassifierMixin MUST precede BaseEstimator in this MRO -- see
    # [[sklearn-classifiermixin-baseestimator-mro-order]] in project memory
    # (this exact bug, found and fixed while building the earlier non-graph
    # event-stats baseline today) -- the other order silently breaks
    # is_classifier() in this sklearn version and, in turn, moabb's ROC-AUC
    # scorer.
    """sklearn-compatible wrapper around DenseEdgeFlatCore: reuses
    SparseEvidenceGNNClassifier's event_mode="dense" precompute
    (_precompute_dense_edge_inputs) for the non-trainable
    [B, 4, E, T, F] input, then trains DenseEdgeFlatCore with a
    hand-rolled loop matching the canonical pipeline's own training
    hyperparameters (AdamW, CrossEntropyLoss, grad-norm clipping, a
    shuffle-order generator tied to `seed` independent of model-
    construction RNG draws -- see [[torch-eeg-classifier-dataloader-shuffle-
    seed-fix]]) as closely as a from-scratch loop reasonably can, so any
    score difference from event_mode="dense" is attributable to the
    architecture (graph vs. no graph), not to a different training regime.
    """

    def __init__(
        self,
        seed: int = 42,
        epochs: int = 75,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 0.1,
        dense_conv_kernel_size: int = 5,
        dense_conv_pool_size: int = 4,
        dense_conv_intermediate_channels: int = 32,
        dense_conv_out_channels: int = 8,
        # 2026-08-11: pass-through to the helper's own
        # time_averaged_graph (see SparseEvidenceGNNCore's docstring in
        # sparse_evidence_gnn_classifier.py) -- combines THIS control's own
        # ablation (no graph/message-passing) with that one's (no within-
        # trial time resolution, dense_edge_raw's T axis collapsed to 1 at
        # precompute time). False (default) is bit-identical to before this
        # param existed. True requires dense_conv_kernel_size=dense_conv_pool_size=1
        # -- NOT enforced here; the helper's own SparseEvidenceGNNClassifier
        # construction (_build_helper) already raises a clear error if that's
        # violated, so duplicating the check here would just be a second
        # place for the same message to go stale.
        time_averaged_graph: bool = False,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.dense_conv_kernel_size = dense_conv_kernel_size
        self.dense_conv_pool_size = dense_conv_pool_size
        self.dense_conv_intermediate_channels = dense_conv_intermediate_channels
        self.dense_conv_out_channels = dense_conv_out_channels
        self.time_averaged_graph = time_averaged_graph

    def _build_helper(self) -> SparseEvidenceGNNClassifier:
        kwargs = dict(CANONICAL_HELPER_KWARGS)
        kwargs["seed"] = self.seed
        kwargs["dense_conv_kernel_size"] = self.dense_conv_kernel_size
        kwargs["dense_conv_pool_size"] = self.dense_conv_pool_size
        kwargs["dense_conv_intermediate_channels"] = self.dense_conv_intermediate_channels
        kwargs["dense_conv_out_channels"] = self.dense_conv_out_channels
        kwargs["time_averaged_graph"] = self.time_averaged_graph
        return SparseEvidenceGNNClassifier(**kwargs)

    def fit(self, X, y):
        X = np.asarray(X)
        self.helper_ = self._build_helper()
        _raw_x, dense_edge_raw = self.helper_._prepare_features(
            X, fit=True, train_idx=np.arange(len(X))
        )
        n_edges = int(dense_edge_raw.shape[2])
        nfreqs = int(dense_edge_raw.shape[4])

        self.classes_ = np.unique(y)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[cls] for cls in y], dtype=np.int64)

        set_seed(self.seed)
        self.model_ = DenseEdgeFlatCore(
            n_edges=n_edges,
            nfreqs=nfreqs,
            n_classes=len(self.classes_),
            dense_conv_kernel_size=self.dense_conv_kernel_size,
            dense_conv_pool_size=self.dense_conv_pool_size,
            dense_conv_intermediate_channels=self.dense_conv_intermediate_channels,
            dense_conv_out_channels=self.dense_conv_out_channels,
        )

        dense_edge_raw = dense_edge_raw.float()
        y_tensor = torch.from_numpy(y_idx).long()

        # See [[torch-eeg-classifier-dataloader-shuffle-seed-fix]] -- own
        # generator tied to `seed`, independent of however many RNG draws
        # model construction above happened to consume.
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(int(self.seed))
        loader = DataLoader(
            TensorDataset(dense_edge_raw, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            generator=shuffle_gen,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for _epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                if self.grad_clip_norm is not None and float(self.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_.parameters(), max_norm=float(self.grad_clip_norm)
                    )
                optimizer.step()
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        _raw_x, dense_edge_raw = self.helper_._prepare_features(X, fit=False)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(dense_edge_raw.float())
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def smoke_test() -> None:
    torch.manual_seed(0)
    B, E, T, F, n_classes = 4, 36, 300, 16, 2
    print("=== DenseEdgeFlatCore smoke test ===")
    model = DenseEdgeFlatCore(
        n_edges=E, nfreqs=F, n_classes=n_classes,
        dense_conv_kernel_size=3, dense_conv_pool_size=2,
        dense_conv_intermediate_channels=16, dense_conv_out_channels=8,
    )
    dense_edge_raw = torch.randn(B, 4, E, T, F)
    y = torch.randint(0, n_classes, (B,))

    logits = model(dense_edge_raw)
    print("logits shape:", logits.shape)
    assert logits.shape == (B, n_classes)

    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    print("loss:", loss.item())

    def grad_norm(module, name):
        total = 0.0
        any_grad = False
        for p in module.parameters():
            if p.grad is not None:
                any_grad = True
                total += p.grad.norm().item()
        print(f"  {name}: any_grad={any_grad} total_grad_norm={total:.6f}")
        return any_grad, total

    ok1, n1 = grad_norm(model.dense_edge_conv, "dense_edge_conv")
    ok2, n2 = grad_norm(model.classifier, "classifier (Linear head)")
    assert ok1 and n1 > 0, "dense_edge_conv got no gradient!"
    assert ok2 and n2 > 0, "classifier head got no gradient!"
    print("PASS: forward/backward OK, gradients reach dense_edge_conv and classifier head.\n")
    print("ALL SMOKE TESTS PASSED")


def run(seeds: list[int]) -> None:
    from moabb.datasets import BNCI2014_001
    from moabb.evaluations import CrossSessionEvaluation
    from moabb.paradigms import LeftRightImagery

    warnings.filterwarnings("ignore")
    dataset = BNCI2014_001()
    dataset.subject_list = [1]
    paradigm = LeftRightImagery(fmin=8, fmax=35)
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        n_jobs=1,
        random_state=42,
        progress_bar=False,
        suffix="dense-edge-flat-control",
    )
    rows = []
    for seed in seeds:
        label = f"dense-flat-control-seed{seed}"
        pipelines = {label: make_pipeline(DenseEdgeFlatClassifier(seed=seed))}
        print(f"\n=== {label} ===", flush=True)
        results = evaluation.process(pipelines)
        by_session = results.set_index("session")["score"].to_dict()
        row = {"seed": seed, **by_session}
        row["mean"] = float(np.mean(list(by_session.values())))
        rows.append(row)
        print(row, flush=True)

    print("\n=== Summary: DenseEdgeFlatClassifier (no graph, no channel embed) ===")
    for r in rows:
        print(r)
    means = [r["mean"] for r in rows]
    print(
        f"mean={np.mean(means):.4f} "
        f"std={(np.std(means, ddof=1) if len(means) > 1 else 0.0):.4f}"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke_test()
    else:
        run([42])
