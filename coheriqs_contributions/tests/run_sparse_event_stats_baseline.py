"""Non-GNN sanity-check baseline for SparseEvidenceGNNClassifier's event
pathway (2026-08-10).

Question: does the graph/message-passing machinery (channel embeddings,
sparse_message_mlp, scatter-add aggregation) add value beyond simple
per-edge event AGGREGATE STATISTICS? Reuses compute_events() -- the same
already-fixed, surrogate-calibrated, native-resolution, COI-masked event
pipeline every other Sparse-Evidence-GNN run uses -- via a throwaway
SparseEvidenceGNNClassifier instance (same technique
debug_sparse_evidence_gnn.py already uses to call this pipeline's real
methods directly), but replaces everything downstream of event-building
with fixed per-edge summary stats (count, time mean/std, freq mean/std,
mean coherence magnitude, circular mean phase) fed into a plain sklearn
classifier: no scatter-add, no learned channel embeddings, no message
passing, no graph structure beyond "which of the 36 canonical edges this
stat belongs to".

Directly comparable to the feature_ablation="zero_channel_embed" GNN result
(mean=0.628, seeds 42-45, same subject/config -- see session_notes/run_logs/
2026-08-10_sparse-evidence-gnn_zero-channel-embed-4seed-threshold-recheck.md):
that ablation already isolates "event content + graph topology alone, no
raw-signal channel embeddings" inside the GNN. This script asks the
complementary question -- how much of THAT number comes from the graph/
message-passing machinery itself, vs. from the raw event statistics a much
simpler model could read directly off compute_events()' output.

USAGE
-----
    python coheriqs_contributions/tests/run_sparse_event_stats_baseline.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery

from coheriqs_contributions.moabb_pipelines.common import upper_pair_indices
from coheriqs_contributions.moabb_pipelines.sparse_evidence_gnn_classifier import (
    SparseEvidenceGNNClassifier,
)

# ---------------------------------------------------------------------------
# Canonical event-extraction config -- copied VERBATIM from the "Full
# effective parameters" block in session_notes/run_logs/2026-08-10_sparse-
# evidence-gnn_zero-channel-embed-4seed-threshold-recheck.md (phase_
# threshold_deg=10, surrogate_percentile=99, subject 1, 9-channel motor
# subset), NOT from run_sparse_evidence_gnn.py's current on-disk state --
# that file was found mid-edit for an unrelated freq_aware_hops/subject-2
# experiment when this script was written (phase_threshold_deg=0.0,
# surrogate_percentile=99.999, epochs=100, subjects=[2] on disk at the
# time -- see this run's session note for the full concurrent-edit check).
# Only the keys that affect event BUILDING matter here (this script never
# touches sparse_message_mlp/hidden_dim/channel_embed_dim/n_hops/etc.,
# since it discards the model entirely after _prepare_features), but the
# full canonical dict is passed anyway so the throwaway helper's
# surrogate-null-cache key matches exactly (same cache the GNN runs already
# warmed) -- surrogate_seed is pinned at 42, independent of this script's
# own `seed` sweep (see CANONICAL_EVENT_KWARGS's own comment below).
# ---------------------------------------------------------------------------
CANONICAL_EVENT_KWARGS: dict[str, object] = dict(
    sampling_rate=250,
    lowest=8.0,
    highest=30.0,
    nfreqs=16,
    cwt_resample_n_time=None,
    coherence_threshold=0.0,
    phase_threshold_deg=10.0,
    channel_encoder_dilation=5,
    hidden_dim=8,
    channel_embed_dim=8,
    epochs=1,  # irrelevant -- this script never calls .fit()'s training loop
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-4,
    grad_clip_norm=0.1,
    normalize_input=True,
    noise_augmentation_enabled=False,
    validation_split=0.0,
    validation_group_column=None,
    early_stopping_patience=None,
    device="auto",
    # seed=42 here is NOT swept by this script and has no effect on events
    # (see surrogate_seed below, pinned independently) -- kept fixed purely
    # so the throwaway helper's own construction is deterministic.
    seed=42,
    surrogate_seed=42,
    smooth_kernel_size=(5, 3),
    smooth_kernel_sigma=(None, None),
    coi_enabled=True,
    scale_adaptive_smoothing=False,
    scale_adaptive_cycles=1.5,
    scale_adaptive_max_kernel=101,
    raw_x_resample_n_time=None,
    feature_ablation="none",  # irrelevant -- forward() is never called
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
    verbose=1,
    n_hops=1,
    freq_aware_hops=False,
)

SUBJECTS = [1]
SEEDS = [42, 43, 44, 45]
STAT_NAMES = ["count", "t_mean", "t_std", "f_mean", "f_std", "mag_mean", "phase_mean"]

# Process-local cache: event-stat feature extraction is DETERMINISTIC given
# fixed X (surrogate_seed is pinned above, independent of this script's
# `seed` sweep; z-score normalization -- the only other thing `seed` could
# touch via _prepare_features -- doesn't affect coherence either, see
# sparse_evidence_gnn_classifier.py's module docstring: "Coherence is
# invariant to a uniform affine shift/scale like z-scoring"). Keyed on a
# content hash of X so the same (train-session, test-session) fold's
# expensive compute_events() call happens ONCE across all 4 seeds x
# {logreg, mlp} model types in this process, not 8+ times -- only the cheap
# sklearn refit varies per seed.
_FEATURE_CACHE: dict[str, np.ndarray] = {}


def _edge_lookup(n_channels: int) -> tuple[np.ndarray, int]:
    """[n_channels, n_channels] LUT mapping (src, dst) -> canonical edge
    index (upper_pair_indices' own ordering -- the SAME ordering
    SparseEvidenceGNNCore.src_idx/dst_idx use), -1 for non-edges."""
    pairs = upper_pair_indices(n_channels)
    lut = -np.ones((n_channels, n_channels), dtype=np.int64)
    for idx, (i, j) in enumerate(pairs):
        lut[i, j] = idx
    return lut, len(pairs)


def _event_stats_features(
    events_padded, src_padded, dst_padded, valid_mask, n_channels: int
) -> np.ndarray:
    """Per trial, per canonical edge (36 for this pipeline's 9-channel
    subset): [count, t_mean, t_std, f_mean, f_std, mag_mean, phase_mean]
    from that edge's consolidated events (event_features columns are
    [timestamp, freq, mag, sin(angle), cos(angle)] -- see
    SparseEvidenceGNNCore._build_sparse_events). phase_mean is the CIRCULAR
    mean (atan2 of summed sin/cos), not a naive mean of an angle-like
    quantity, to avoid the wraparound artifact a plain mean would introduce.
    An edge with zero events in a trial gets an all-zero stat row (count=0
    included) rather than being omitted -- every trial's feature vector has
    the same fixed 36*7 length regardless of how many edges fired.

    Returns [n_trials, n_edges * 7], edges in upper_pair_indices' order,
    stats in STAT_NAMES' order within each edge's block.
    """
    events_np = events_padded.detach().cpu().numpy().astype(np.float64)
    src_np = src_padded.detach().cpu().numpy()
    dst_np = dst_padded.detach().cpu().numpy()
    valid_np = valid_mask.detach().cpu().numpy()
    n_trials, _max_count, _ = events_np.shape

    lut, n_edges = _edge_lookup(n_channels)
    edge_of_slot = lut[src_np, dst_np]  # [n_trials, max_count]; -1 at padding slots
    t, f, mag, sin_, cos_ = (events_np[..., k] for k in range(5))

    feats = np.zeros((n_trials, n_edges, len(STAT_NAMES)))
    for trial in range(n_trials):
        valid_slots = np.where(valid_np[trial])[0]
        if valid_slots.size == 0:
            continue
        edges_here = edge_of_slot[trial, valid_slots]
        for edge in np.unique(edges_here):
            sel = valid_slots[edges_here == edge]
            feats[trial, edge] = [
                sel.size,
                t[trial, sel].mean(),
                t[trial, sel].std(),
                f[trial, sel].mean(),
                f[trial, sel].std(),
                mag[trial, sel].mean(),
                np.arctan2(sin_[trial, sel].sum(), cos_[trial, sel].sum()),
            ]
    return feats.reshape(n_trials, n_edges * len(STAT_NAMES))


class SparseEventStatsBaseline(ClassifierMixin, BaseEstimator):
    # NOTE: ClassifierMixin MUST precede BaseEstimator in this MRO (not the
    # other, more commonly-typed-first order) -- this sklearn version's
    # is_classifier()/get_tags() resolve `estimator_type` via cooperative
    # __sklearn_tags__() calls up the MRO, and BaseEstimator-first silently
    # yields estimator_type=None (is_classifier() -> False) rather than an
    # error. That broke moabb's roc_auc scorer: with is_classifier() False,
    # sklearn.utils._response._get_response_values skips its binary
    # positive-class column selection entirely and hands roc_auc_score the
    # raw (n_samples, 2) predict_proba matrix, which fails deep inside
    # roc_curve with "y should be a 1d array, got an array of shape (N, 2)
    # instead." Confirmed by direct repro against this env's sklearn 1.9.0.
    """sklearn-compatible non-graph baseline: extracts per-edge event
    summary stats via the real, unmodified SparseEvidenceGNNClassifier
    event pipeline (compute_events(), through a throwaway helper instance's
    _prepare_features -- exactly what debug_sparse_evidence_gnn.py already
    does to call this pipeline's real methods directly), then fits a plain
    sklearn classifier on those fixed-length vectors. No SparseEvidenceGNNCore
    model is ever constructed or trained -- channel_encoder, sparse_message_mlp,
    _aggregate_events, and every hop-propagation mechanism are entirely
    absent from this pathway.
    """

    def __init__(self, model_type: str = "logreg", seed: int = 42):
        self.model_type = model_type
        self.seed = seed

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        key = hashlib.sha256(np.ascontiguousarray(X, dtype=np.float32).tobytes()).hexdigest()
        cached = _FEATURE_CACHE.get(key)
        if cached is not None:
            return cached
        helper = SparseEvidenceGNNClassifier(**CANONICAL_EVENT_KWARGS)
        raw_x, events_padded, src_padded, dst_padded, freq_idx_padded, valid_mask = (
            helper._prepare_features(X, fit=True, train_idx=np.arange(len(X)))
        )
        feats = _event_stats_features(
            events_padded, src_padded, dst_padded, valid_mask, n_channels=int(raw_x.shape[1])
        )
        _FEATURE_CACHE[key] = feats
        return feats

    def fit(self, X, y):
        X = np.asarray(X)
        feats = self._extract_features(X)
        if self.model_type == "logreg":
            self.pipeline_ = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, random_state=self.seed),
            )
        elif self.model_type == "mlp":
            self.pipeline_ = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    max_iter=2000,
                    random_state=self.seed,
                    alpha=1e-3,
                ),
            )
        else:
            raise ValueError(f"model_type must be 'logreg' or 'mlp', got {self.model_type!r}.")
        self.pipeline_.fit(feats, y)
        self.classes_ = self.pipeline_.classes_
        return self

    def predict(self, X):
        return self.pipeline_.predict(self._extract_features(np.asarray(X)))

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(self._extract_features(np.asarray(X)))


def run(model_type: str, seeds: list[int]) -> None:
    dataset = BNCI2014_001()
    dataset.subject_list = SUBJECTS
    paradigm = LeftRightImagery(fmin=8, fmax=35)
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        n_jobs=1,
        random_state=42,
        progress_bar=False,
        suffix=f"sparse-event-stats-baseline-{model_type}",
    )

    per_seed_rows = []
    for seed in seeds:
        label = f"SparseEventStats-{model_type}-seed{seed}"
        pipelines = {label: make_pipeline(SparseEventStatsBaseline(model_type=model_type, seed=seed))}
        print(f"\n=== {label} ===", flush=True)
        results = evaluation.process(pipelines)
        by_session = results.set_index("session")["score"].to_dict()
        row = {"seed": seed, **by_session}
        row["mean"] = float(np.mean(list(by_session.values())))
        per_seed_rows.append(row)
        print(row, flush=True)

    print(f"\n=== Summary: model_type={model_type} ===")
    means = [r["mean"] for r in per_seed_rows]
    for r in per_seed_rows:
        print(r)
    print(
        f"mean={np.mean(means):.4f} std={np.std(means, ddof=1):.4f} "
        f"min={np.min(means):.4f} max={np.max(means):.4f}"
    )


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logreg"
    run(model_type, SEEDS)
