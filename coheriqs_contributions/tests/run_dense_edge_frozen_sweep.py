"""Freeze + cache dense_edge_conv's output, then sweep downstream heads fast
(2026-08-10). Three parts, selected via the CLI subcommand:

    python coheriqs_contributions/tests/run_dense_edge_frozen_sweep.py cache
    python coheriqs_contributions/tests/run_dense_edge_frozen_sweep.py sweep
    python coheriqs_contributions/tests/run_dense_edge_frozen_sweep.py validate --variant <name>

PART 1 ("cache"): trains DenseEdgeFlatClassifier (see
run_dense_edge_flat_control.py) exactly as the 2026-08-10 flat-control run
did -- seed 42, canonical config -- for BOTH CrossSessionEvaluation folds
(subject 1 has exactly two sessions, "0train"/"1test"; each fold trains on
one and tests on the other). That run's own driver never saved a checkpoint,
so this retrains rather than reloading -- see this module's own
"Reproduces, not reuses" note below for why that's expected to (and does)
land on the same numbers. For each fold's trained model, runs
dense_edge_conv once, torch.no_grad(), over ALL 288 subject-1 trials (not
just that fold's own train/test split) and caches the per-edge output via
moabb_pipelines.dense_conv_feature_cache. Then VERIFIES the cache: for that
fold's own held-out trials, reconstructs predict_proba by flattening the
CACHED features and running them through the model's OWN trained
`classifier` (nn.Linear) directly -- bypassing dense_edge_conv entirely --
and confirms this reproduces the fold's real score to numerical precision.
Both folds' scores must average to ~0.9291 (0train~0.8949, 1test~0.9633,
per the flat-control write-up) before Part 2 is trusted.

PART 2 ("sweep"): loads ONLY the cached features (no conv in the loop) and
fits small downstream heads directly on them, multi-seed (42-45), fast:
  - Flat heads (DenseEdgeFlatCore's own architecture family) at three
    capacities: no hidden layer (288->2 directly, the original flat
    control), hidden=72 (bottlenecked to match the graph pathway's own
    72-dim readout), hidden=288 (matched to the flat control's own full
    width, but now with an actual hidden layer + nonlinearity instead of a
    single linear projection).
  - Graph heads (FrozenGraphHead below -- ports SparseEvidenceGNNCore's
    exact sparse_message_mlp + "mean" _aggregate_events formula, with
    feature_ablation="zero_channel_embed" reproduced structurally by never
    building channel_encoder in the first place, since that ablation's own
    torch.zeros_like already detaches it from the graph -- see
    FrozenGraphHead's docstring) at hidden_dim=8 (current, 72-dim readout,
    reference), 16, and 32 (288-dim readout, matched to the flat control).

Compares every variant's multi-seed mean against the two reference numbers
this whole investigation has been chasing: 0.9291 (flat, current capacity)
and 0.8886/0.9048 (graph, current capacity, zero_channel_embed/full).

PART 3 ("validate"): takes the single best-performing variant from Part 2's
frozen-feature screening and trains it ONCE fully end-to-end (dense_edge_conv
UNFROZEN, jointly trained from scratch, exactly like the original flat
control) to confirm the frozen-feature screening number holds up when the
conv can co-adapt to that specific downstream head, rather than trusting the
frozen-feature number as final. As of 2026-08-10, this ALSO always caches
each fold's freshly-trained dense_edge_conv's frozen features (same
always-on hook Part 1 uses, cache_all_trial_features) and records every
(seed, head_kind, hidden_dim) run's fold/cache_key info in
PART3_MANIFEST_PATH -- Part 3 no longer requires a full retrain just to look
at its conv's features again.

PART 3b ("flat-on-frozen"): takes a Part 3 run's now-cached, jointly-trained
conv features and fits the ORIGINAL flat head on top of them instead of that
run's own graph/concat head -- isolates whether a Part 3 regression (e.g.
concat_h4's 0.8973) comes from the conv's own learned representations having
degraded under joint training with that head, or just from the head's
readout/aggregation step being the specific problem on top of otherwise-fine
features. Requires 'validate' to have been run first for the requested
seeds/head_kind/hidden_dim.

"Reproduces, not reuses" (Part 1 caveat): this retrains rather than loading
a saved checkpoint, since the original run never saved one. Same seed, same
hyperparameters, same code path (DenseEdgeFlatClassifier is imported
unchanged from run_dense_edge_flat_control.py, not reimplemented) -- PyTorch
is deterministic given a fixed seed and identical operation order on the
same device/build, so this is expected to land on bit-identical weights, not
just a similar score. Verified directly below (Part 1 prints both the
retrained score AND the recorded 2026-08-10 score side by side).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from coheriqs_contributions.moabb_pipelines.common import set_seed, upper_pair_indices
from coheriqs_contributions.moabb_pipelines.dense_conv_feature_cache import (
    compute_dense_conv_features,
    conv_weight_hash,
    default_dense_conv_feature_cache_root,
    dense_conv_feature_cache_key,
    load_dense_conv_feature_cache,
    save_dense_conv_feature_cache,
    trial_content_hash,
)
from coheriqs_contributions.tests.run_dense_edge_flat_control import (
    CANONICAL_HELPER_KWARGS,
    DenseEdgeFlatClassifier,
)

SUBJECT = 1
SEED_FOR_CACHE = 42  # the flat-control run's own seed -- see module docstring
SWEEP_SEEDS = [42, 43, 44, 45]
SESSIONS = ["0train", "1test"]
# Recorded 2026-08-10 flat-control numbers (seed 42), for the Part 1 sanity
# check -- see session_notes/run_logs/2026-08-10_sparse-evidence-gnn_dense-
# edge-flat-no-graph-control.md.
RECORDED_SCORES = {"0train": 0.8949, "1test": 0.9633}
RECORDED_MEAN = 0.9291

MANIFEST_PATH = default_dense_conv_feature_cache_root() / "manifest.json"
PART3_MANIFEST_PATH = default_dense_conv_feature_cache_root() / "part3_manifest.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_subject_data(subject: int = SUBJECT):
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    warnings.filterwarnings("ignore")
    dataset = BNCI2014_001()
    dataset.subject_list = [subject]
    paradigm = LeftRightImagery(fmin=8, fmax=35)
    X, y, metadata = paradigm.get_data(dataset, subjects=[subject])
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    return X, y_int, metadata


# ---------------------------------------------------------------------------
# ALWAYS-ON caching hook -- any code path in this script that trains a
# dense_edge_conv (Part 1's flat-control retrain, Part 3's end-to-end
# graph/concat training) runs its frozen features through here, the same way
# surrogate_null_cache gets written as an automatic side effect of the
# pipeline's own forward pass rather than as a separate opt-in step. Nothing
# downstream ever has to remember to cache -- if a conv got trained, its
# features are on disk.
# ---------------------------------------------------------------------------
def _dense_conv_kwargs_for_cache(
    kernel_size: int, pool_size: int, intermediate_channels: int, out_channels: int,
) -> dict:
    return dict(
        dense_conv_kernel_size=kernel_size,
        dense_conv_pool_size=pool_size,
        dense_conv_intermediate_channels=intermediate_channels,
        dense_conv_intermediate_channels_reduced=None,
        dense_conv_out_channels=out_channels,
    )


def _conv_feature_cache_key(dense_edge_conv: nn.Module, checkpoint_id: str, dense_conv_kwargs: dict) -> str:
    weight_hash = conv_weight_hash(dense_edge_conv)
    kw = CANONICAL_HELPER_KWARGS
    return dense_conv_feature_cache_key(
        sampling_rate=kw["sampling_rate"], highest=kw["highest"], lowest=kw["lowest"],
        nfreqs=kw["nfreqs"], cwt_resample_n_time=kw["cwt_resample_n_time"],
        smooth_kernel_size=kw["smooth_kernel_size"], smooth_kernel_sigma=kw["smooth_kernel_sigma"],
        coi_enabled=kw["coi_enabled"], coherence_threshold_mode=kw["coherence_threshold_mode"],
        surrogate_count=kw["surrogate_count"], surrogate_seed=kw["surrogate_seed"],
        surrogate_percentile=kw["surrogate_percentile"],
        scale_adaptive_smoothing=kw["scale_adaptive_smoothing"],
        scale_adaptive_cycles=kw["scale_adaptive_cycles"],
        scale_adaptive_max_kernel=kw["scale_adaptive_max_kernel"],
        channel_subset=tuple(kw["channel_subset"]),
        checkpoint_id=checkpoint_id,
        weight_hash=weight_hash,
        **dense_conv_kwargs,
    )


def cache_all_trial_features(
    helper, X_all: np.ndarray, dense_edge_conv: nn.Module,
    checkpoint_id: str, dense_conv_kwargs: dict,
) -> dict:
    """Runs `dense_edge_conv` once, torch.no_grad(), over every trial in
    `X_all` and writes the result to dense_conv_feature_cache, keyed on
    config + architecture + this specific conv's trained weight values (see
    dense_conv_feature_cache.py's module docstring for why the weight hash
    makes this safe against silent reuse across different training runs).
    Returns {'cache_key', 'n_edges', 'out_channels'}."""
    _raw_x_all, dense_edge_raw_all = helper._prepare_features(X_all, fit=False)
    features_all = compute_dense_conv_features(
        dense_edge_conv, dense_edge_raw_all.float()
    )  # [N, E, out_channels], cpu, float32

    raw_x_native_all = helper._apply_channel_subset(np.asarray(X_all, dtype=np.float32))
    trial_hashes = np.array(
        [trial_content_hash(raw_x_native_all[i]) for i in range(raw_x_native_all.shape[0])]
    )

    cache_dir = default_dense_conv_feature_cache_root()
    key = _conv_feature_cache_key(dense_edge_conv, checkpoint_id, dense_conv_kwargs)
    save_dense_conv_feature_cache(cache_dir, key, features_all.numpy(), trial_hashes)
    return {
        "cache_key": key,
        "n_edges": int(features_all.shape[1]),
        "out_channels": int(features_all.shape[2]),
    }


# ---------------------------------------------------------------------------
# PART 1: freeze + cache + verify
# ---------------------------------------------------------------------------
def build_and_cache_fold(X, y_int, metadata, test_session: str, seed: int = SEED_FOR_CACHE):
    """PART 1 for one fold: train, score, cache all-trial features, verify."""
    sessions = metadata["session"].values
    train_idx = np.where(sessions != test_session)[0]
    test_idx = np.where(sessions == test_session)[0]

    print(f"\n=== Fold: test_session={test_session} (train n={len(train_idx)}, test n={len(test_idx)}) ===")
    clf = DenseEdgeFlatClassifier(seed=seed)
    clf.fit(X[train_idx], y_int[train_idx])
    proba_test = clf.predict_proba(X[test_idx])
    score = roc_auc_score(y_int[test_idx], proba_test[:, 1])
    recorded = RECORDED_SCORES[test_session]
    print(f"  retrained score={score:.4f}  recorded(2026-08-10)={recorded:.4f}  "
          f"delta={score - recorded:+.4f}")

    # --- extract + cache dense_edge_conv's frozen output over ALL trials ---
    cache_info = cache_all_trial_features(
        clf.helper_, X, clf.model_.dense_edge_conv,
        checkpoint_id=f"flat_control_seed{clf.seed}_holdout={test_session}",
        dense_conv_kwargs=_dense_conv_kwargs_for_cache(
            clf.dense_conv_kernel_size, clf.dense_conv_pool_size,
            clf.dense_conv_intermediate_channels, clf.dense_conv_out_channels,
        ),
    )
    key = cache_info["cache_key"]
    cache_dir = default_dense_conv_feature_cache_root()
    print(f"  cached: {cache_dir / (key + '.npz')}")

    # --- verify: cached features -> trained classifier head reproduces score ---
    cached = load_dense_conv_feature_cache(cache_dir, key)
    features_all = torch.from_numpy(cached["features"]).float()
    n_edges = features_all.shape[1]
    flat_test = features_all[test_idx].reshape(len(test_idx), -1)
    clf.model_.eval()
    with torch.no_grad():
        logits_from_cache = clf.model_.classifier(flat_test)
        proba_from_cache = torch.softmax(logits_from_cache, dim=1).numpy()
    max_abs_diff = float(np.abs(proba_from_cache - proba_test).max())
    score_from_cache = roc_auc_score(y_int[test_idx], proba_from_cache[:, 1])
    print(f"  cache-reconstructed score={score_from_cache:.4f}  "
          f"max|proba_diff vs live forward|={max_abs_diff:.2e}")
    if max_abs_diff > 1e-4:
        raise RuntimeError(
            f"Cache verification FAILED for test_session={test_session}: "
            f"reconstructed proba diverges from live forward by {max_abs_diff:.2e}."
        )

    return {
        "test_session": test_session,
        "cache_key": key,
        "score": float(score),
        "score_from_cache": float(score_from_cache),
        "recorded_score": recorded,
        "n_edges": int(n_edges),
        "out_channels": int(clf.dense_conv_out_channels),
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }


def run_part1_cache():
    X, y_int, metadata = load_subject_data()
    results = []
    for test_session in SESSIONS:
        results.append(build_and_cache_fold(X, y_int, metadata, test_session))

    mean_score = float(np.mean([r["score"] for r in results]))
    mean_from_cache = float(np.mean([r["score_from_cache"] for r in results]))
    print(f"\n=== Part 1 summary ===")
    for r in results:
        print(f"  {r['test_session']}: live={r['score']:.4f} "
              f"cache={r['score_from_cache']:.4f} recorded={r['recorded_score']:.4f}")
    print(f"  mean(live)={mean_score:.4f}  mean(cache)={mean_from_cache:.4f}  "
          f"recorded_mean={RECORDED_MEAN:.4f}")
    if abs(mean_from_cache - RECORDED_MEAN) > 0.01:
        print(
            f"  WARNING: cache-reconstructed mean diverges from recorded "
            f"{RECORDED_MEAN:.4f} by more than 0.01 -- investigate before "
            f"trusting Part 2's sweep."
        )
    else:
        print(f"  OK: cache-reconstructed mean matches recorded 0.9291 within 0.01.")

    manifest = {
        "seed": SEED_FOR_CACHE,
        "subject": SUBJECT,
        "folds": results,
        "mean_score": mean_score,
        "mean_from_cache": mean_from_cache,
        "recorded_mean": RECORDED_MEAN,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest written: {MANIFEST_PATH}")
    return manifest


# ---------------------------------------------------------------------------
# PART 2: fast sweep on cached features
# ---------------------------------------------------------------------------
class FlatHead(nn.Module):
    """Same family as DenseEdgeFlatCore's own classifier -- flatten ->
    (optional hidden layer) -> n_classes. hidden_dim=None reproduces the
    original flat control exactly (single nn.Linear, no hidden layer)."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, n_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_classes)
            )

    def forward(self, flat_features: torch.Tensor) -> torch.Tensor:
        return self.net(flat_features)


class FrozenGraphHead(nn.Module):
    """SparseEvidenceGNNCore's sparse_message_mlp + "mean" _aggregate_events
    branch + sparse_classifier, ported to run directly on cached per-edge
    features -- reproduces feature_ablation="zero_channel_embed" exactly,
    but structurally (concatenating literal zero tensors of width
    channel_embed_dim) rather than by zeroing a computed channel_encoder
    output -- mathematically identical, since that ablation's own
    torch.zeros_like already detaches channel_encoder's output from the
    graph (see SparseEvidenceGNNCore.forward's feature_ablation docstring),
    so channel_encoder contributes nothing to either the forward value or
    the backward gradient under this ablation. Skipping it here isn't an
    approximation -- it's the same computation with the always-dead branch
    removed, and it's what keeps this sweep fast (no Conv1d over raw_x every
    step).

    Topology: same canonical (i<j) src_idx/dst_idx as SparseEvidenceGNNCore
    (upper_pair_indices, imported from the same common.py both use) -- NOT
    reimplemented independently, so this can't silently drift from
    production's actual edge ordering/direction.
    """

    def __init__(
        self, n_channels: int, edge_feature_dim: int, n_classes: int,
        hidden_dim: int = 8, channel_embed_dim: int = 8,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        pairs = upper_pair_indices(n_channels)
        self.register_buffer("dst_idx", torch.tensor([j for _, j in pairs], dtype=torch.long))
        message_in = edge_feature_dim + 2 * channel_embed_dim
        self.channel_embed_dim = channel_embed_dim
        self.sparse_message_mlp = nn.Sequential(
            nn.Linear(message_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.sparse_classifier = nn.Linear(n_channels * hidden_dim, n_classes)

    def forward(self, events_padded: torch.Tensor) -> torch.Tensor:
        batch_size, n_edges, _ = events_padded.shape
        zeros = torch.zeros(
            batch_size, n_edges, self.channel_embed_dim,
            dtype=events_padded.dtype, device=events_padded.device,
        )
        full_features = torch.cat([events_padded, zeros, zeros], dim=-1)
        msg = self.sparse_message_mlp(full_features)  # [B, E, H]

        dst = self.dst_idx.to(events_padded.device)
        evidence = torch.zeros(
            batch_size, self.n_channels, self.hidden_dim,
            dtype=events_padded.dtype, device=events_padded.device,
        )
        evidence.scatter_add_(
            1, dst.view(1, -1, 1).expand(batch_size, -1, self.hidden_dim), msg
        )
        active = torch.zeros(batch_size, self.n_channels, dtype=events_padded.dtype, device=events_padded.device)
        ones = torch.ones(batch_size, n_edges, dtype=events_padded.dtype, device=events_padded.device)
        active.scatter_add_(1, dst.view(1, -1).expand(batch_size, -1), ones)
        evidence = evidence / active.clamp_min(1.0).unsqueeze(-1)

        readout = evidence.reshape(batch_size, self.n_channels * self.hidden_dim)
        return self.sparse_classifier(readout)


class ConcatGraphHead(nn.Module):
    """Alternate aggregation, added after Part 2's capacity sweep found
    widening hidden_dim makes the graph pathway WORSE, not better (see
    module docstring) -- points at the "mean"-aggregation's forced
    averaging itself (not dimensionality) as the lossy step. This replaces
    mean-pooling with per-node CONCATENATION of every incident edge's own
    message, in a fixed canonical order, instead of averaging them into one
    vector -- no information is discarded at the aggregation step itself
    (every edge's message reaches the classifier distinctly), closer in
    spirit to the flat control's own "keep every edge's features distinct"
    design, but still routed through per-node structure (message MLP +
    topology-aware grouping) rather than a flat, node-blind concatenation.

    dense mode's topology is a FIXED complete graph over n_channels (every
    edge always "fires" -- see SparseEvidenceGNNCore._dense_edge_features),
    so each node's incident-edge SET is static across trials -- concatenation
    needs no padding logic beyond zero-filling nodes with fewer than
    max_degree incoming edges (node j's in-degree is exactly j under the
    canonical i<j dst-only scatter -- see _aggregate_events' docstring in
    sparse_evidence_gnn_classifier.py -- so degrees range 0 (node 0, never a
    dst) to n_channels-1 (the last node)). A dedicated all-zero "pad" row is
    appended to the per-batch message tensor and every missing slot indexes
    into it, rather than being left uninitialized.
    """

    def __init__(
        self, n_channels: int, edge_feature_dim: int, n_classes: int,
        hidden_dim: int = 8, channel_embed_dim: int = 8,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        pairs = upper_pair_indices(n_channels)
        dst_idx = [j for _, j in pairs]
        max_degree = n_channels - 1
        self.max_degree = max_degree
        # slot_idx[j, k] = index into the (E)-length edge/message axis of
        # node j's k-th incident edge, or E (the pad row) if node j has
        # fewer than max_degree incoming edges.
        n_edges = len(dst_idx)
        slot_idx = torch.full((n_channels, max_degree), n_edges, dtype=torch.long)
        counts = [0] * n_channels
        for edge_pos, j in enumerate(dst_idx):
            slot_idx[j, counts[j]] = edge_pos
            counts[j] += 1
        self.register_buffer("slot_idx", slot_idx)  # [n_channels, max_degree]
        message_in = edge_feature_dim + 2 * channel_embed_dim
        self.channel_embed_dim = channel_embed_dim
        self.sparse_message_mlp = nn.Sequential(
            nn.Linear(message_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.sparse_classifier = nn.Linear(
            n_channels * max_degree * hidden_dim, n_classes
        )

    def forward(self, events_padded: torch.Tensor) -> torch.Tensor:
        batch_size, n_edges, _ = events_padded.shape
        zeros = torch.zeros(
            batch_size, n_edges, self.channel_embed_dim,
            dtype=events_padded.dtype, device=events_padded.device,
        )
        full_features = torch.cat([events_padded, zeros, zeros], dim=-1)
        msg = self.sparse_message_mlp(full_features)  # [B, E, H]
        pad_row = torch.zeros(
            batch_size, 1, self.hidden_dim, dtype=msg.dtype, device=msg.device
        )
        msg_padded = torch.cat([msg, pad_row], dim=1)  # [B, E+1, H]

        slot_idx = self.slot_idx.to(events_padded.device)  # [n_channels, max_degree]
        flat_slots = slot_idx.reshape(-1)  # [n_channels * max_degree]
        gathered = msg_padded.index_select(1, flat_slots)  # [B, n_channels*max_degree, H]
        readout = gathered.reshape(batch_size, self.n_channels * self.max_degree * self.hidden_dim)
        return self.sparse_classifier(readout)


def train_head(
    model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
    seed: int, epochs: int = 75, batch_size: int = 8,
    learning_rate: float = 1e-3, weight_decay: float = 1e-4, grad_clip_norm: float = 0.1,
) -> nn.Module:
    """Same training regime as DenseEdgeFlatClassifier.fit() -- AdamW,
    CrossEntropyLoss, grad-norm clipping, a shuffle-order generator tied to
    `seed` independent of model-construction RNG draws (see
    [[torch-eeg-classifier-dataloader-shuffle-seed-fix]]) -- so any score
    difference between variants is attributable to architecture, not
    training-regime drift."""
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
        generator=shuffle_gen, num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
    return model


def _load_fold_features(manifest_fold: dict):
    cache_dir = default_dense_conv_feature_cache_root()
    cached = load_dense_conv_feature_cache(cache_dir, manifest_fold["cache_key"])
    if cached is None:
        raise RuntimeError(
            f"Cache miss for {manifest_fold['cache_key']} -- run Part 1 ('cache') first."
        )
    return torch.from_numpy(cached["features"]).float()  # [288, E, C]


def run_variant_across_folds(
    variant_name: str, build_model_fn, manifest: dict, y_int: np.ndarray, seeds: list[int],
    use_flat: bool,
):
    """Runs one architecture variant across both CrossSessionEvaluation
    folds and `seeds`, mirroring the same 2-fold cross-session structure
    the flat-control run itself used -- each fold's head trains on that
    FOLD'S OWN frozen features (extracted by that fold's own trained conv,
    never the other fold's) and is scored on that fold's held-out session,
    exactly like the original evaluation.process() call would score it."""
    per_seed_means = []
    per_fold_scores = {s: {} for s in seeds}
    for fold in manifest["folds"]:
        features_all = _load_fold_features(fold)  # [288, E, C]
        n_edges, out_channels = fold["n_edges"], fold["out_channels"]
        train_idx = np.array(fold["train_idx"])
        test_idx = np.array(fold["test_idx"])
        if use_flat:
            X_train = features_all[train_idx].reshape(len(train_idx), -1)
            X_test = features_all[test_idx].reshape(len(test_idx), -1)
        else:
            X_train = features_all[train_idx]
            X_test = features_all[test_idx]
        y_train = torch.from_numpy(y_int[train_idx]).long()
        y_test_np = y_int[test_idx]

        for seed in seeds:
            set_seed(seed)
            model = build_model_fn(n_edges=n_edges, out_channels=out_channels)
            train_head(model, X_train, y_train, seed=seed)
            model.eval()
            with torch.no_grad():
                proba = torch.softmax(model(X_test), dim=1).numpy()
            score = roc_auc_score(y_test_np, proba[:, 1])
            per_fold_scores[seed][fold["test_session"]] = score

    rows = []
    for seed in seeds:
        by_session = per_fold_scores[seed]
        mean = float(np.mean(list(by_session.values())))
        per_seed_means.append(mean)
        rows.append({"seed": seed, **by_session, "mean": mean})

    overall_mean = float(np.mean(per_seed_means))
    overall_std = float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0
    return {"variant": variant_name, "rows": rows, "mean": overall_mean, "std": overall_std}


def run_part2_sweep(seeds: list[int] = SWEEP_SEEDS):
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"No manifest at {MANIFEST_PATH} -- run Part 1 ('cache') first.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    _X, y_int, _metadata = load_subject_data()  # need y_int only; cheap-ish re-download-free (cached)

    n_channels = 9  # len(channel_subset) -- see CANONICAL_HELPER_KWARGS
    channel_embed_dim = 8

    variants = []

    # --- flat capacity sweep ---
    def make_flat(hidden_dim):
        def _build(n_edges, out_channels):
            return FlatHead(n_edges * out_channels, n_classes=2, hidden_dim=hidden_dim)
        return _build

    variants.append(("flat_direct_288", make_flat(None), True))       # original flat control
    variants.append(("flat_hidden72", make_flat(72), True))           # bottleneck matched to graph readout
    variants.append(("flat_hidden288", make_flat(288), True))         # matched to flat's own full width

    # --- graph capacity sweep (feature_ablation=zero_channel_embed equivalent) ---
    def make_graph(hidden_dim):
        def _build(n_edges, out_channels):
            return FrozenGraphHead(
                n_channels=n_channels, edge_feature_dim=out_channels, n_classes=2,
                hidden_dim=hidden_dim, channel_embed_dim=channel_embed_dim,
            )
        return _build

    variants.append(("graph_h8", make_graph(8), False))    # current default, 72-dim readout, reference
    variants.append(("graph_h16", make_graph(16), False))  # intermediate point
    variants.append(("graph_h32", make_graph(32), False))  # 288-dim readout, matched to flat

    # --- alternate aggregation: concat instead of mean-pool per node ---
    # Added because the capacity sweep above found widening hidden_dim makes
    # mean-pool aggregation WORSE, not better -- testing whether the lossy
    # forced-averaging step itself (not width) is the bottleneck. Readout
    # width = n_channels * max_degree * hidden_dim = 9*8*hidden_dim.
    def make_concat(hidden_dim):
        def _build(n_edges, out_channels):
            return ConcatGraphHead(
                n_channels=n_channels, edge_feature_dim=out_channels, n_classes=2,
                hidden_dim=hidden_dim, channel_embed_dim=channel_embed_dim,
            )
        return _build

    variants.append(("concat_h1", make_concat(1), False))  # 72-dim readout, matched to graph_h8
    variants.append(("concat_h4", make_concat(4), False))  # 288-dim readout, matched to flat

    results = []
    for name, build_fn, use_flat in variants:
        print(f"\n=== sweeping variant: {name} ===", flush=True)
        res = run_variant_across_folds(name, build_fn, manifest, y_int, seeds, use_flat)
        results.append(res)
        print(f"  mean={res['mean']:.4f} std={res['std']:.4f}  rows={res['rows']}")

    print("\n=== Part 2 summary (frozen-feature screening, seeds "
          f"{seeds}) ===")
    print(f"{'variant':<18} {'mean':>8} {'std':>8}")
    for r in results:
        print(f"{r['variant']:<18} {r['mean']:>8.4f} {r['std']:>8.4f}")
    print(f"\nReference: flat (current, single seed 42) = {RECORDED_MEAN:.4f}")
    print("Reference: graph zero_channel_embed (hidden_dim=8, seeds 42-45) = 0.8886 +/- 0.0067")
    print("Reference: graph full pipeline (hidden_dim=8, seeds 42-45)     = 0.9048 +/- 0.0031")

    sweep_out = MANIFEST_PATH.parent / "sweep_results.json"
    sweep_out.write_text(json.dumps(results, indent=2))
    print(f"\nsweep results written: {sweep_out}")
    return results


# ---------------------------------------------------------------------------
# PART 3: end-to-end validation of the winning variant (conv unfrozen)
# ---------------------------------------------------------------------------
HEAD_CLASSES = {"mean": FrozenGraphHead, "concat": ConcatGraphHead}


class DenseEdgeGraphCore(nn.Module):
    """End-to-end version of a graph-family downstream variant:
    dense_edge_conv (trainable, IDENTICAL construction to
    DenseEdgeFlatCore/event_mode="dense") -> either FrozenGraphHead
    (mean-pool aggregation) or ConcatGraphHead (concat aggregation, see that
    class's docstring for why it was added), all trained jointly. Only used
    by Part 3, so the conv CAN co-adapt to this specific downstream head,
    unlike Part 2's frozen-feature screening."""

    def __init__(
        self, n_edges: int, nfreqs: int, n_classes: int, hidden_dim: int,
        head_kind: str = "mean",
        channel_embed_dim: int = 8,
        dense_conv_kernel_size: int = 5, dense_conv_pool_size: int = 4,
        dense_conv_intermediate_channels: int = 32, dense_conv_out_channels: int = 8,
    ) -> None:
        super().__init__()
        from coheriqs_contributions.moabb_pipelines.sparse_evidence_gnn_classifier import (
            _build_dense_feature_conv,
        )

        if head_kind not in HEAD_CLASSES:
            raise ValueError(f"head_kind must be one of {list(HEAD_CLASSES)}, got {head_kind!r}.")
        self.n_edges = n_edges
        self.dense_conv_out_channels = dense_conv_out_channels
        self.dense_edge_conv = _build_dense_feature_conv(
            in_channels=4 * nfreqs, kernel_size=dense_conv_kernel_size,
            intermediate_channels=dense_conv_intermediate_channels,
            out_channels=dense_conv_out_channels, pool_size=dense_conv_pool_size,
            intermediate_channels_reduced=None,
        )
        n_channels = 9
        self.head = HEAD_CLASSES[head_kind](
            n_channels=n_channels, edge_feature_dim=dense_conv_out_channels,
            n_classes=n_classes, hidden_dim=hidden_dim, channel_embed_dim=channel_embed_dim,
        )

    def forward(self, dense_edge_raw: torch.Tensor) -> torch.Tensor:
        batch_size, c_in, num_edges, n_time, nfreqs = dense_edge_raw.shape
        conv_in = dense_edge_raw.permute(0, 1, 4, 2, 3).reshape(
            batch_size, c_in * nfreqs, num_edges, n_time
        )
        conv_out = self.dense_edge_conv(conv_in)
        feat = conv_out.squeeze(-1).permute(0, 2, 1)  # [B, E, out_channels]
        return self.head(feat)


class DenseEdgeGraphClassifier:
    """sklearn-compatible wrapper, same shape as DenseEdgeFlatClassifier but
    routing dense_edge_conv's output through a graph-family head
    (head_kind="mean" or "concat") instead of a flat classifier. Only
    instantiated by Part 3."""

    def __init__(
        self, seed: int = 42, hidden_dim: int = 32, head_kind: str = "mean",
        epochs: int = 75, batch_size: int = 8,
        learning_rate: float = 1e-3, weight_decay: float = 1e-4, grad_clip_norm: float = 0.1,
        dense_conv_kernel_size: int = 5, dense_conv_pool_size: int = 4,
        dense_conv_intermediate_channels: int = 32, dense_conv_out_channels: int = 8,
    ):
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.head_kind = head_kind
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        # DenseEdgeGraphCore's own defaults, exposed here (rather than left
        # implicit) so the always-on caching hook can build a correct cache
        # key without guessing the conv's architecture -- see
        # run_part3_validate.
        self.dense_conv_kernel_size = dense_conv_kernel_size
        self.dense_conv_pool_size = dense_conv_pool_size
        self.dense_conv_intermediate_channels = dense_conv_intermediate_channels
        self.dense_conv_out_channels = dense_conv_out_channels

    def _build_helper(self):
        from coheriqs_contributions.moabb_pipelines.sparse_evidence_gnn_classifier import (
            SparseEvidenceGNNClassifier,
        )
        kwargs = dict(CANONICAL_HELPER_KWARGS)
        kwargs["seed"] = self.seed
        return SparseEvidenceGNNClassifier(**kwargs)

    def fit(self, X, y):
        X = np.asarray(X)
        self.helper_ = self._build_helper()
        _raw_x, dense_edge_raw = self.helper_._prepare_features(X, fit=True, train_idx=np.arange(len(X)))
        n_edges = int(dense_edge_raw.shape[2])
        nfreqs = int(dense_edge_raw.shape[4])

        self.classes_ = np.unique(y)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[cls] for cls in y], dtype=np.int64)

        set_seed(self.seed)
        self.model_ = DenseEdgeGraphCore(
            n_edges=n_edges, nfreqs=nfreqs, n_classes=len(self.classes_),
            hidden_dim=self.hidden_dim, head_kind=self.head_kind,
            dense_conv_kernel_size=self.dense_conv_kernel_size,
            dense_conv_pool_size=self.dense_conv_pool_size,
            dense_conv_intermediate_channels=self.dense_conv_intermediate_channels,
            dense_conv_out_channels=self.dense_conv_out_channels,
        )
        train_head(
            self.model_, dense_edge_raw.float(), torch.from_numpy(y_idx).long(),
            seed=self.seed, epochs=self.epochs, batch_size=self.batch_size,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        _raw_x, dense_edge_raw = self.helper_._prepare_features(X, fit=False)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(dense_edge_raw.float())
        return torch.softmax(logits, dim=1).numpy()


PART3_MANIFEST_LOCK_PATH = PART3_MANIFEST_PATH.with_suffix(".lock")


def _load_part3_manifest() -> dict:
    if PART3_MANIFEST_PATH.is_file():
        return json.loads(PART3_MANIFEST_PATH.read_text())
    return {"runs": []}


def _save_part3_manifest(manifest: dict) -> None:
    """Atomic write (temp file + rename), same convention as
    save_dense_conv_feature_cache -- see
    [[run-wct-gnn-concurrent-write-race]] in project memory."""
    PART3_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PART3_MANIFEST_PATH.with_suffix(f".{os.getpid()}.tmp.json")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp_path, PART3_MANIFEST_PATH)


def _upsert_part3_manifest_run(run_record: dict) -> None:
    """Read-modify-write PART3_MANIFEST_PATH under an flock so parallel
    `validate` processes (e.g. different subjects run concurrently) never
    race on the shared manifest file and silently drop each other's updates
    -- this codebase has hit exactly that class of bug before on a
    different shared results file (see
    [[run-wct-gnn-concurrent-write-race]]), so it's treated as a real risk
    here, not a hypothetical one, even though each process's OWN cache
    (.npz) files never collide (checkpoint_id includes subject)."""
    PART3_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PART3_MANIFEST_LOCK_PATH.touch(exist_ok=True)
    with open(PART3_MANIFEST_LOCK_PATH, "r+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            manifest = _load_part3_manifest()
            manifest["runs"] = [
                r for r in manifest["runs"]
                if not (
                    r.get("subject", SUBJECT) == run_record["subject"]
                    and r["seed"] == run_record["seed"]
                    and r["head_kind"] == run_record["head_kind"]
                    and r["hidden_dim"] == run_record["hidden_dim"]
                )
            ]
            manifest["runs"].append(run_record)
            _save_part3_manifest(manifest)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def run_part3_validate(
    hidden_dim: int, head_kind: str = "mean", seeds: list[int] = SWEEP_SEEDS,
    subject: int = SUBJECT,
):
    """PART 3, now with the always-on caching hook: every fold's trained
    dense_edge_conv has its frozen features cached (via
    cache_all_trial_features) immediately after fit(), exactly like Part 1
    does for the flat control -- Part 3 is no longer a dead end that has to
    be fully retrained to look at its conv's features again. Each run
    (subject, seed, head_kind, hidden_dim) is recorded in
    PART3_MANIFEST_PATH so a later step (e.g. run_flat_on_frozen, "Part 3b")
    can reload the cached features by cache_key without retraining anything.
    `subject` defaults to SUBJECT (1, the flat-control's own subject) but is
    a real parameter -- see the 2026-08-10 cross-subject follow-up (subjects
    2-4) that first needed this."""
    X, y_int, metadata = load_subject_data(subject=subject)
    sessions = metadata["session"].values
    print(f"\n=== Part 3: end-to-end validation, subject={subject} head_kind={head_kind} "
          f"hidden_dim={hidden_dim}, seeds={seeds} ===")
    dense_conv_kwargs = _dense_conv_kwargs_for_cache(5, 4, 32, 8)  # DenseEdgeGraphCore's own defaults
    rows = []
    for seed in seeds:
        by_session = {}
        fold_records = []
        for test_session in SESSIONS:
            train_idx = np.where(sessions != test_session)[0]
            test_idx = np.where(sessions == test_session)[0]
            clf = DenseEdgeGraphClassifier(seed=seed, hidden_dim=hidden_dim, head_kind=head_kind)
            clf.fit(X[train_idx], y_int[train_idx])
            proba = clf.predict_proba(X[test_idx])
            score = roc_auc_score(y_int[test_idx], proba[:, 1])
            by_session[test_session] = float(score)
            print(f"  subject={subject} seed={seed} test_session={test_session} score={score:.4f}",
                  flush=True)

            cache_info = cache_all_trial_features(
                clf.helper_, X, clf.model_.dense_edge_conv,
                checkpoint_id=(
                    f"graph_{head_kind}_h{hidden_dim}_seed{seed}_subj{subject}_holdout={test_session}"
                ),
                dense_conv_kwargs=dense_conv_kwargs,
            )
            print(f"    cached: {cache_info['cache_key']}", flush=True)
            fold_records.append({
                "test_session": test_session,
                "cache_key": cache_info["cache_key"],
                "n_edges": cache_info["n_edges"],
                "out_channels": cache_info["out_channels"],
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "score": float(score),
            })
        mean = float(np.mean(list(by_session.values())))
        rows.append({"seed": seed, **by_session, "mean": mean})

        # lock-protected upsert (see _upsert_part3_manifest_run's docstring)
        # -- safe to call from multiple concurrently-running subjects/seeds.
        _upsert_part3_manifest_run({
            "subject": subject, "seed": seed, "head_kind": head_kind, "hidden_dim": hidden_dim,
            "folds": fold_records, "mean": mean,
        })

    means = [r["mean"] for r in rows]
    overall_mean = float(np.mean(means))
    overall_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    print(f"\n=== Part 3 summary: subject={subject} end-to-end {head_kind}_h{hidden_dim} ===")
    for r in rows:
        print(r)
    print(f"mean={overall_mean:.4f} std={overall_std:.4f}")
    print(f"part3 manifest written: {PART3_MANIFEST_PATH}")
    return {
        "subject": subject, "head_kind": head_kind, "hidden_dim": hidden_dim, "rows": rows,
        "mean": overall_mean, "std": overall_std,
    }


# ---------------------------------------------------------------------------
# PART 3b: flat head on top of a graph/concat-trained conv's frozen features
# ---------------------------------------------------------------------------
def run_flat_on_frozen(head_kind: str, hidden_dim: int, seeds: list[int], subject: int = SUBJECT):
    """Takes the (now-cached, via Part 3's always-on caching hook) frozen
    features from an end-to-end Part 3 run's dense_edge_conv -- e.g. the
    concat-trained conv from concat_h4 -- and fits the ORIGINAL flat head
    (flat_direct_288, no hidden layer, same architecture/training regime as
    the flat control) on top of them, INSTEAD of that run's own graph/concat
    head. Distinguishes two different explanations for a Part 3 regression
    like concat_h4's 0.8973 (below every flat-family number):

      - if flat-on-{head_kind}-features recovers close to flat's own
        ~0.92+, the {head_kind}-trained conv's representations were fine --
        the aggregation/readout step was the specific problem, not what the
        conv learned;
      - if flat-on-{head_kind}-features ALSO drops toward Part 3's own
        ~0.89-0.90, training jointly with {head_kind} degraded the conv's
        own representations, not just how they get pooled afterward -- a
        more fundamental problem, consistent with this investigation's own
        speculation that concat's gradient (routed through per-node
        grouping and a wide, redundant classifier input) may simply be a
        harder optimization target than flat's, not just a worse pooling
        choice on top of equally-good features.

    Requires run_part3_validate(hidden_dim, head_kind, seeds) to have been
    run first for every seed requested here (that's what populates
    PART3_MANIFEST_PATH and the underlying feature cache) -- raises if any
    requested (seed, head_kind, hidden_dim) combination is missing rather
    than silently skipping it.
    """
    if not PART3_MANIFEST_PATH.is_file():
        raise RuntimeError(f"No Part 3 manifest at {PART3_MANIFEST_PATH} -- run 'validate' first.")
    part3_manifest = _load_part3_manifest()
    cache_dir = default_dense_conv_feature_cache_root()

    _X, y_int, _metadata = load_subject_data(subject=subject)
    print(f"\n=== Part 3b: flat head on subject={subject} {head_kind}_h{hidden_dim}-trained "
          f"conv features, seeds={seeds} ===")

    rows = []
    graph_means = []
    for seed in seeds:
        run_entry = next(
            (r for r in part3_manifest["runs"]
             if r.get("subject", SUBJECT) == subject and r["seed"] == seed
             and r["head_kind"] == head_kind and r["hidden_dim"] == hidden_dim),
            None,
        )
        if run_entry is None:
            raise RuntimeError(
                f"No cached Part 3 run for subject={subject} head_kind={head_kind} "
                f"hidden_dim={hidden_dim} seed={seed} -- run `validate --subject {subject} "
                f"--head-kind {head_kind} --hidden-dim {hidden_dim} --seeds {seed}` first."
            )
        graph_means.append(run_entry["mean"])

        by_session = {}
        for fold in run_entry["folds"]:
            cached = load_dense_conv_feature_cache(cache_dir, fold["cache_key"])
            if cached is None:
                raise RuntimeError(
                    f"Cache miss for {fold['cache_key']} -- the manifest references a cache "
                    f"entry that no longer exists on disk; re-run 'validate' for this "
                    f"seed/head_kind/hidden_dim."
                )
            features_all = torch.from_numpy(cached["features"]).float()
            train_idx = np.array(fold["train_idx"])
            test_idx = np.array(fold["test_idx"])
            X_train = features_all[train_idx].reshape(len(train_idx), -1)
            X_test = features_all[test_idx].reshape(len(test_idx), -1)
            y_train = torch.from_numpy(y_int[train_idx]).long()
            y_test_np = y_int[test_idx]

            set_seed(seed)
            model = FlatHead(fold["n_edges"] * fold["out_channels"], n_classes=2, hidden_dim=None)
            train_head(model, X_train, y_train, seed=seed)
            model.eval()
            with torch.no_grad():
                proba = torch.softmax(model(X_test), dim=1).numpy()
            score = roc_auc_score(y_test_np, proba[:, 1])
            by_session[fold["test_session"]] = float(score)
            print(f"  seed={seed} test_session={fold['test_session']} "
                  f"flat-on-{head_kind}_h{hidden_dim}-features={score:.4f}  "
                  f"(that fold's own {head_kind} score was {fold['score']:.4f})", flush=True)
        mean = float(np.mean(list(by_session.values())))
        rows.append({"seed": seed, **by_session, "mean": mean})

    means = [r["mean"] for r in rows]
    overall_mean = float(np.mean(means))
    overall_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    graph_overall_mean = float(np.mean(graph_means))
    print(f"\n=== Part 3b summary: flat head on {head_kind}_h{hidden_dim}-trained conv features ===")
    for r in rows:
        print(r)
    print(f"mean={overall_mean:.4f} std={overall_std:.4f}")
    print(f"\nReference: flat-on-flat-features (original flat control, seed 42)   = {RECORDED_MEAN:.4f}")
    print(f"Reference: {head_kind}_h{hidden_dim} end-to-end own score (same seeds) = {graph_overall_mean:.4f}")
    print(f"flat-on-{head_kind}_h{hidden_dim}-features                             = {overall_mean:.4f}")

    out_path = cache_dir / f"flat_on_{head_kind}_h{hidden_dim}_results.json"
    out_path.write_text(json.dumps({
        "head_kind": head_kind, "hidden_dim": hidden_dim, "rows": rows,
        "mean": overall_mean, "std": overall_std,
        "reference_flat_on_flat": RECORDED_MEAN,
        "reference_graph_own_score": graph_overall_mean,
    }, indent=2))
    print(f"\nresults written: {out_path}")
    return {
        "head_kind": head_kind, "hidden_dim": hidden_dim, "rows": rows,
        "mean": overall_mean, "std": overall_std,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("cache")
    sweep_p = sub.add_parser("sweep")
    sweep_p.add_argument("--seeds", type=int, nargs="+", default=SWEEP_SEEDS)
    val_p = sub.add_parser("validate")
    val_p.add_argument("--hidden-dim", type=int, default=4)
    val_p.add_argument("--head-kind", choices=list(HEAD_CLASSES), default="concat")
    val_p.add_argument("--seeds", type=int, nargs="+", default=SWEEP_SEEDS)
    val_p.add_argument("--subject", type=int, default=SUBJECT)
    flat_p = sub.add_parser("flat-on-frozen")
    flat_p.add_argument("--hidden-dim", type=int, default=4)
    flat_p.add_argument("--head-kind", choices=list(HEAD_CLASSES), default="concat")
    flat_p.add_argument("--seeds", type=int, nargs="+", default=SWEEP_SEEDS)
    flat_p.add_argument("--subject", type=int, default=SUBJECT)
    args = parser.parse_args()

    if args.mode == "cache":
        run_part1_cache()
    elif args.mode == "sweep":
        run_part2_sweep(seeds=args.seeds)
    elif args.mode == "validate":
        run_part3_validate(
            hidden_dim=args.hidden_dim, head_kind=args.head_kind, seeds=args.seeds,
            subject=args.subject,
        )
    elif args.mode == "flat-on-frozen":
        run_flat_on_frozen(
            head_kind=args.head_kind, hidden_dim=args.hidden_dim, seeds=args.seeds,
            subject=args.subject,
        )
