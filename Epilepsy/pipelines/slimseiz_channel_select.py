"""SlimSeiz stage 1: per-subject channel selection, ported from the
upstream repo's ``Loop_select_ch_PCA_SMOTE_DT.ipynb``
(https://github.com/guoruilu/SlimSeiz) -- the half of SlimSeiz that
`slimseiz_classifier.py` doesn't vendor. That file only ships the stage-2
conv+Mamba network; the paper's headline numbers (94.8% accuracy / 95.5%
sensitivity / 94.0% specificity on CHB-MIT) are reported for that network
fed an *adaptively selected* 8-of-22 channel subset, not the full montage.
Without this stage, every SlimSeiz run in this repo was feeding the network
all 23 chb01 channels -- a different, and on this repo's own 2026-08-25
6-fold comparison, weaker-performing condition than what "SlimSeiz" refers
to in the paper (2026-08-25 session finding).

THE UPSTREAM ALGORITHM (as read from the notebook, cell-by-cell): for each
of 30 iterations, for each candidate channel independently: 5s-segment that
one channel, PCA to 60 components, SMOTE-balance the training split, fit a
DecisionTreeClassifier, score held-out accuracy. Tally which channels land
in each iteration's top-N most-accurate, and keep the N channels tallied
most often across all 30 iterations (``top_n_elements`` in the notebook --
a plain ``Counter`` over each iteration's ranked list).

DELIBERATE DEVIATIONS from the notebook, made when porting into this
repo's own window/label pipeline rather than re-deriving the paper's own
5s/PIL segmentation:

  - Operates directly on whatever windows/labels this repo's own LOSO fold
    training split already produced (this repo's SPH/SOP prediction
    windows, not the paper's 5s segments + 15-min PIL) -- the point of
    stage 1 here is "which channels are most individually informative
    under the label scheme this repo is actually being benchmarked on",
    which is the correct question for a fair comparison against this
    repo's GRU/Mamba dense-edge results, not a literal paper reproduction.
  - One stratified train/test split per iteration, reused across every
    channel within that iteration -- the notebook instead redraws an
    independent `train_test_split` (no `stratify=`) inside the per-channel
    loop, which is both slower (channel score reuses nothing) and noisier
    (each channel is scored against a different random split, so
    differences between channels partly reflect which split they happened
    to get). A stratified split also avoids the notebook's version
    occasionally producing a test fold with zero positives.
  - Ranks each iteration's channels by ROC-AUC, not raw accuracy. This
    repo's prediction task is far more imbalanced (~1:20-1:25
    preictal:interictal per LOSO fold, see the 2026-08-25 session notes)
    than accuracy can meaningfully rank under -- a channel that is
    genuinely uninformative can still post >90% test accuracy by
    predicting the majority class, which would make the notebook's own
    metric useless for ranking on this repo's folds specifically. AUC is
    threshold-independent and degrades gracefully under this repo's class
    balance. `top_n_elements`'s tally-across-iterations step is otherwise
    unchanged.
  - `SMOTE`'s `k_neighbors` is clamped to the minority class's actual
    training-split size (and skipped entirely below 2 samples) instead of
    the notebook's unguarded default (`k_neighbors=5`, which raises if the
    minority class has <6 samples) -- a real possibility on this repo's
    small per-fold training splits that the notebook's own script never
    had to handle (it ran channel selection on one patient's *entire*
    recording set, not a single LOSO fold's training seizures).

Selection is run separately inside every LOSO fold's `fit()`, using only
that fold's training windows (the held-out seizure's windows never
participate) -- see `slimseiz_classifier.py`'s `SlimSeizClassifier.fit()`
override. This is a deliberate divergence from the notebook's own
protocol (which selects once per patient using that patient's full
dataset, train and test pooled together) -- see the 2026-08-25 session
finding/decision: per-fold selection avoids leaking the held-out seizure's
own channel importance into that seizure's evaluation, which is the
scientifically correct choice for a LOSO benchmark even though it costs
extra compute and isn't a literal reproduction of the paper's script.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guidance
    raise ModuleNotFoundError(
        "slimseiz_channel_select requires `imbalanced-learn` (SMOTE), the "
        "same package the upstream SlimSeiz channel-selection notebook "
        "uses. Install it (`pip install imbalanced-learn`) or pass "
        "select_channels=False to SlimSeizClassifier to skip stage 1 and "
        "fall back to the full channel montage."
    ) from exc


def _score_channel(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    pca_components: int,
    seed: int | None,
) -> float:
    """PCA -> SMOTE -> DecisionTreeClassifier -> held-out ROC-AUC for one
    channel's (n_samples, n_timepoints) data, mirroring one cell of the
    upstream notebook's per-channel scoring loop (see module docstring for
    the ROC-AUC-vs-accuracy and SMOTE-k_neighbors deviations)."""
    n_comp = max(1, min(pca_components, x_train.shape[0] - 1, x_train.shape[1]))
    pca = PCA(n_components=n_comp, random_state=seed)
    # np.errstate: plain float32 PCA.fit_transform/transform on this
    # machine's Accelerate-BLAS matmul spuriously raises divide/overflow/
    # invalid RuntimeWarnings with no non-finite values actually produced
    # (verified directly: np.isfinite(...) all-True on the exact same call
    # that warns) -- a known Apple Accelerate false positive, not a signal
    # of a real problem here. The isfinite check right after is the actual
    # safety net; this just silences the known-spurious warning noise.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x_train_p = pca.fit_transform(x_train)
        x_test_p = pca.transform(x_test)
    if not (np.all(np.isfinite(x_train_p)) and np.all(np.isfinite(x_test_p))):
        # Genuinely degenerate channel (e.g. constant/zero-variance data) --
        # rank it last rather than let NaN/Inf propagate into SMOTE/the tree.
        return -1.0

    class_counts = np.bincount(y_train, minlength=2)
    minority_count = int(class_counts.min())
    if minority_count >= 2:
        k_neighbors = max(1, min(5, minority_count - 1))
        x_train_p, y_train = SMOTE(k_neighbors=k_neighbors, random_state=seed).fit_resample(
            x_train_p, y_train
        )
    # else: too few minority-class training samples to SMOTE meaningfully
    # (a real possibility on a single LOSO fold's training split, unlike
    # the notebook's whole-patient-dataset selection) -- fit on the
    # imbalanced split as-is rather than raising.

    clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(x_train_p, y_train)

    if len(np.unique(y_test)) < 2:
        # Degenerate stratified split (shouldn't happen in practice given
        # StratifiedShuffleSplit, but a channel-selection ranking pass
        # should never hard-fail a LOSO fold over it).
        return float(accuracy_score(y_test, clf.predict(x_test_p)))

    proba = clf.predict_proba(x_test_p)
    pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
    try:
        return float(roc_auc_score(y_test, proba[:, pos_idx]))
    except ValueError:
        return float(accuracy_score(y_test, clf.predict(x_test_p)))


def select_slimseiz_channels(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_select: int = 8,
    n_iterations: int = 30,
    pca_components: int = 60,
    test_size: float = 0.3,
    seed: int | None = 42,
    verbose: int = 0,
    max_samples: int | None = 1000,
) -> np.ndarray:
    """Rank channels the way the upstream SlimSeiz notebook does (repeated
    single-channel PCA+SMOTE+DecisionTree holdout scoring, tallied across
    iterations) and return the indices of the `n_select` most consistently
    top-ranked channels, sorted ascending.

    X: (n_samples, n_channels, n_timepoints) raw windows -- pass only the
    LOSO fold's training windows (the held-out seizure must not be in X).
    y: (n_samples,) binary labels, any two distinct values.

    max_samples : int | None (default 1000) -- caps how many of X's rows
    this function ever looks at, via one stratified subsample drawn before
    the iteration loop (not per-iteration -- every iteration/channel then
    reuses the same capped pool, same reasoning as the module docstring's
    "one split reused across channels" deviation: cheaper AND less noisy
    than each iteration seeing a different sample count).

    Added 2026-08-25 after this stage, run on a full (uncapped) LOSO
    fold's training windows (thousands of rows at this repo's real/non-
    smoke prediction scale, vs. --smoke's few hundred), was implicated in
    a real crash of the machine running it -- see CONTEXT.md's "Known
    gotchas". Channel ranking is a per-channel holdout-AUC comparison, not
    a fit that needs every available window to be meaningful, so capping
    the pool this function sees is a reasonable, correctness-preserving
    way to bound this stage's cost independent of fold size. `None`
    disables the cap (pre-2026-08-25 behavior, the crash-implicated path
    -- avoid on a memory-constrained machine at real/non-smoke scale).
    """
    n_samples, n_channels, _ = X.shape
    if max_samples is not None and n_samples > max_samples:
        y_enc_for_cap = LabelEncoder().fit_transform(np.asarray(y))
        cap_splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=max_samples, random_state=seed
        )
        cap_idx, _ = next(cap_splitter.split(np.zeros(n_samples), y_enc_for_cap))
        cap_idx = np.sort(cap_idx)
        if verbose >= 1:
            print(
                f"[SlimSeiz stage 1] capping input from {n_samples} to "
                f"{len(cap_idx)} rows (max_samples={max_samples}) before "
                "channel-selection iterations -- see this function's "
                "max_samples docstring."
            )
        X, y = X[cap_idx], y[cap_idx]
        n_samples = len(cap_idx)
    if n_channels <= n_select:
        if verbose >= 1:
            print(
                f"[SlimSeiz stage 1] n_channels={n_channels} <= "
                f"n_select={n_select}; skipping selection, using all "
                "channels."
            )
        return np.arange(n_channels)

    y_enc = LabelEncoder().fit_transform(np.asarray(y))
    if len(np.unique(y_enc)) != 2:
        raise ValueError(
            "select_slimseiz_channels requires binary labels (this repo's "
            "CHB-MIT detection/prediction tasks are binary today); got "
            f"{len(np.unique(y_enc))} distinct classes."
        )

    top_counts = np.zeros(n_channels, dtype=np.int64)
    mean_scores = np.zeros(n_channels, dtype=np.float64)

    for it in range(n_iterations):
        split_seed = None if seed is None else int(seed) * 10_000 + it
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=split_seed
        )
        train_idx, test_idx = next(splitter.split(np.zeros(n_samples), y_enc))
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        scores = np.empty(n_channels, dtype=np.float64)
        for c in range(n_channels):
            scores[c] = _score_channel(
                X[train_idx, c, :],
                y_train,
                X[test_idx, c, :],
                y_test,
                pca_components=pca_components,
                seed=seed,
            )

        ranked = np.argsort(-scores)
        top_counts[ranked[:n_select]] += 1
        mean_scores += scores / n_iterations

    # Primary key: how often a channel landed in the per-iteration top-N.
    # Tie-break: its mean score across all iterations (np.lexsort applies
    # the LAST key as primary).
    order = np.lexsort((-mean_scores, -top_counts))
    selected = np.sort(order[:n_select])

    if verbose >= 1:
        print(
            f"[SlimSeiz stage 1] selected {len(selected)}/{n_channels} "
            f"channels after {n_iterations} iterations: "
            f"{selected.tolist()} "
            f"(top-N tallies: {top_counts[selected].tolist()}, "
            f"mean AUC: {[round(s, 3) for s in mean_scores[selected]]})"
        )

    return selected
