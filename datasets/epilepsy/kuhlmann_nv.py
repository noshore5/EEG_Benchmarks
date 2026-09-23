"""Kuhlmann NeuroVista (NV) seizure-prediction contest data -- gated, DUA-
restricted (see scripts/download_kuhlmann_nv.py's module docstring and
CONTEXT.md before redistributing anything loaded here).

Format: AES/Kaggle "Melbourne University AES-MathWorks-NIH Seizure
Prediction" contest -- 3 patients, one 10-minute intracranial-EEG segment
per .mat file, 16 channels @ 400Hz (data shape (240000, 16), time x
channel -- transposed here to this repo's (n_channels, n_timepoints)
convention). Filenames are ``Pat<N><Train|Test>_<idx>_<label>.mat``; label
is real (0=interictal, 1=preictal) for Train, always a placeholder ``_0``
for Test (confirmed empirically 2026-09-22 -- Kaggle never shipped the
private answer key with the contest bundle, so Test is not usable for any
local evaluation).

NO seizure/event-grouping metadata survives in the released files -- each
.mat has only a ``data`` key (checked all fields present, 2026-09-22).
The six preictal segments that the contest's own data-generation pipeline
cut from one hour before a seizure are NOT identifiable as a group from
what's released: there's no sequence/hour/event id, and filenames are
shuffled (index order carries no timing information). This means a
genuine leave-one-SEIZURE-out split -- holding out all segments from one
seizure's lead-up together, as run_pipelines.py's
leave_one_seizure_out_raw_classifier[_prediction] do for CHB-MIT via its
seizure_id metadata column -- cannot be reconstructed for this dataset.
See scripts/run_nv_pat1.py for the fold regime actually used instead
(stratified k-fold over individual segments) and why.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal as sps

_FNAME_RE = re.compile(r"^Pat(\d+)(Train|Test)_(\d+)_(\d+)\.mat$")

DEFAULT_ROOT = Path(__file__).resolve().parent / "kuhlmann_nv"


def _try_fetch_from_s3(subfolder: str) -> None:
    """Best-effort pull of a missing Pat<N><Train|Test> subfolder from the
    private S3 mirror (scripts/fetch_kuhlmann_nv_s3.sh) -- lets a spot box
    run --cmd 'python scripts/run_nv_pat1.py ...' with no separate fetch
    step, same as chb_mit.py's own auto-download for CHB-MIT. No-op (never
    raises) if the script or S3 object isn't there; the caller's own
    FileNotFoundError below is the real error message either way."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "fetch_kuhlmann_nv_s3.sh"
    if not script.is_file():
        return
    subprocess.run(["bash", str(script), subfolder], check=False)


def list_segments(patient: int, split: str, root: Path = DEFAULT_ROOT) -> list[dict]:
    """Return [{"path", "segment_id", "label"}, ...] for one patient/split
    subfolder, sorted by segment_id. label is int(0/1) parsed from the
    filename -- real for split="Train", a meaningless placeholder for
    split="Test" (see module docstring)."""
    subdir = root / f"Pat{patient}{split}"
    if not subdir.is_dir():
        _try_fetch_from_s3(f"Pat{patient}{split}")
    if not subdir.is_dir():
        raise FileNotFoundError(
            f"{subdir} not found -- run scripts/download_kuhlmann_nv.py Pat{patient}{split} first."
        )
    out = []
    for p in subdir.glob("*.mat"):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        pat, split_tag, idx, label = m.groups()
        if int(pat) != patient or split_tag != split:
            continue
        out.append({"path": p, "segment_id": int(idx), "label": int(label)})
    out.sort(key=lambda e: e["segment_id"])
    return out


def load_segment(path: Path, decimate: int = 1) -> np.ndarray:
    """One segment -> (n_channels, n_timepoints) float32, this repo's raw-
    classifier convention (transposed from the .mat's native (time,
    channel) layout). decimate > 1 anti-alias-filters and downsamples by
    that integer factor (scipy.signal.decimate) before returning -- see
    load_patient_train's docstring for why this exists by default."""
    mat = sio.loadmat(path)
    data = mat["data"]  # (n_timepoints, n_channels)
    arr = np.ascontiguousarray(data.T.astype(np.float32))
    if decimate > 1:
        arr = sps.decimate(arr, decimate, axis=1, zero_phase=True).astype(np.float32)
    return arr


def load_patient_train(
    patient: int,
    root: Path = DEFAULT_ROOT,
    limit: int | None = None,
    decimate: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every Train segment for one patient.

    Returns (X, y, segment_ids):
      X: (n_segments, n_channels, n_timepoints) float32
      y: (n_segments,) int, 0=interictal / 1=preictal (real labels)
      segment_ids: (n_segments,) int, the contest's own file index --
        carries no recoverable timing/grouping information (see module
        docstring), kept only for traceability back to the source file.

    A small fraction of segments (17/826 = 2.1% for Pat1Train, confirmed
    2026-09-22) are shorter than the canonical 240,000 samples -- 236000/
    232000/228000, i.e. missing the tail 1-3 minutes. This is a known
    intracranial-recording dropout in the source contest data, not a
    download/parsing bug (verified: the short files load cleanly, they're
    just genuinely truncated). Rather than zero-pad (which would hand the
    model an artificial silent tail it could learn to key off of) or
    truncate every segment down to the shortest one (which would throw
    away 1-3 minutes from 97.9% of the data to accommodate 2.1%), these
    off-length segments are dropped here (shape-matching is done on the
    POST-decimation shape, so this still works with decimate > 1). Logged
    whenever it happens.

    decimate=4 (400Hz -> 100Hz) by default: full-resolution Pat1Train is
    ~12.7GB of float32 (825 segments x 16ch x 240,000 samples) -- loading
    that into one array plus this function's earlier list-then-np.stack
    pattern transiently DOUBLED that to ~25GB and drove this laptop's
    system memory into its compressor / near-zero-free territory
    (confirmed 2026-09-22, process hit 17GB RSS still mid-load, system
    down to 163MB unused -- killed before it could crash the machine like
    the earlier slimseiz-channel-select run did). Fixed two ways: (1) this
    function now fills one preallocated array in place instead of
    building a Python list and then np.stack-ing a second copy of it, and
    (2) decimate=4 cuts the resident footprint to ~3.2GB, comfortably
    inside this machine's headroom, while still leaving a 100Hz Nyquist
    (50Hz) well above where seizure-prediction spectral features live in
    every raw-classifier baseline this repo runs. Pass decimate=1 to load
    at native 400Hz if a run genuinely needs it and memory allows.
    """
    entries = list_segments(patient, "Train", root)
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise ValueError(f"No Train segments found for patient {patient} under {root}")

    n_entries = len(entries)
    canonical_shape: tuple[int, int] | None = None
    X: np.ndarray | None = None
    kept_mask = np.zeros(n_entries, dtype=bool)
    n_dropped = 0

    for i, e in enumerate(entries):
        arr = load_segment(e["path"], decimate=decimate)
        if canonical_shape is None:
            # First file sets the canonical shape (post-decimation) --
            # segments are already sorted by segment_id at this point, not
            # some adversarial order, so "first" is a fine guess; any
            # off-length file discovered later just gets dropped below
            # rather than silently invalidating everything loaded so far.
            canonical_shape = arr.shape
            X = np.empty((n_entries, *canonical_shape), dtype=np.float32)
        if arr.shape != canonical_shape:
            n_dropped += 1
            continue
        X[i] = arr
        kept_mask[i] = True

    if n_dropped:
        print(
            f"[kuhlmann_nv] dropped {n_dropped}/{n_entries} Pat{patient}Train segments "
            f"with off-canonical length (recording dropout, not a bug -- see load_patient_train's docstring)"
        )

    X = X[kept_mask]
    y = np.array([e["label"] for e, k in zip(entries, kept_mask) if k], dtype=np.int64)
    segment_ids = np.array([e["segment_id"] for e, k in zip(entries, kept_mask) if k], dtype=np.int64)
    return X, y, segment_ids


def reconstruct_preictal_blocks(
    X: np.ndarray,
    y: np.ndarray,
    segment_ids: np.ndarray,
    boundary_samples: int = 20,
    gap_threshold: float = 0.6,
) -> np.ndarray:
    """Reconstruct the contest's original 1-hour preictal blocks (~6
    consecutive 10-minute segments cut from one continuous pre-seizure
    recording) from boundary continuity, since no grouping metadata
    survives in the released files (see module docstring). Only touches
    Train data we already have -- no withheld label is read or inferred,
    this recovers which ALREADY-LABELED segments share a source
    recording, nothing else.

    v2 (2026-09-22), REPLACING a first attempt that failed its own
    validation check: raw-amplitude L2 distance between boundary windows
    showed no separation between genuine continuations and coincidence
    (see git history / NEGATIVES.md for that attempt). The fix was
    per-channel z-scoring each segment (using that segment's own
    mean/std) before comparing boundaries -- undoing whatever per-segment
    amplitude normalization the contest's own data prep applied, which
    was masking real continuity under attempt v1 -- plus cosine
    similarity instead of raw distance (scale/offset-invariant on top of
    the z-scoring). This DOES show real separation on Pat1: the
    best-match-cosine-similarity minus each segment's median similarity
    ("gap") has a clear cluster near 0.95-0.97 distinct from a bulk
    clustered near 0.4, and thresholding at gap>0.6 recovers ~24 edges
    forming chains of plausible size (2-5, up to a few short of the
    paper's stated 6) rather than either no matches or one giant blob --
    validated 2026-09-22, see scripts/ dev notes / this session's
    transcript for the diagnostic that found this.

    Chains are necessarily an UNDER-reconstruction, not a complete one:
    only ~24/244 valid segments produce an edge at this threshold, so
    most segments end up as their own singleton "chain" (no evidence of a
    partner, not evidence there ISN'T one -- a real adjacent segment
    might simply score below threshold due to signal noise at the cut).
    That's fine for fold-grouping purposes -- a singleton just behaves
    like plain segment-level grouping for that segment, so this can only
    ever do LESS damage than segment-level grouping, never more: it group
    some genuinely-linked segments together (which segment-level grouping
    would incorrectly split across folds) and changes nothing else.

    Only applied to preictal (y==1) segments -- interictal segments are
    independently sampled by the contest's own generator, not part of a
    block, so each keeps its own segment_id as its group. Segments with
    a flat/clipped channel anywhere in their boundary window (a real
    recording-dropout artifact -- confirmed during v1's diagnostic) are
    excluded from matching entirely (kept as singletons) since a
    degenerate (zero-variance) channel can't be z-scored and produces
    spurious "perfect" matches against other degenerate segments.

    Returns group_ids, same length/order as segment_ids: reconstructed
    chain id for preictal segments with an accepted match, offset well
    above any real segment_id to avoid collision; raw segment_id
    unchanged for interictal segments and for unmatched/degenerate
    preictal ones.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    group_ids = segment_ids.copy()
    preictal_idx = np.where(y == 1)[0]
    n = len(preictal_idx)
    if n < 2:
        return group_ids

    seg_x = X[preictal_idx]
    mean = seg_x.mean(axis=2, keepdims=True)
    std = seg_x.std(axis=2, keepdims=True)
    degenerate = (std < 1e-6).any(axis=(1, 2))
    valid_local = np.where(~degenerate)[0]
    n_valid = len(valid_local)
    if n_valid < 2:
        print(f"[kuhlmann_nv] {degenerate.sum()}/{n} preictal segments degenerate -- nothing left to match")
        return group_ids

    normed = (seg_x[valid_local] - mean[valid_local]) / std[valid_local]
    tails = normed[:, :, -boundary_samples:].reshape(n_valid, -1)
    heads = normed[:, :, :boundary_samples].reshape(n_valid, -1)
    tails = tails - tails.mean(axis=1, keepdims=True)
    heads = heads - heads.mean(axis=1, keepdims=True)
    tails /= np.linalg.norm(tails, axis=1, keepdims=True)
    heads /= np.linalg.norm(heads, axis=1, keepdims=True)
    sim = tails @ heads.T
    np.fill_diagonal(sim, np.nan)

    best_j = np.nanargmax(sim, axis=1)
    best_sim = sim[np.arange(n_valid), best_j]
    median_sim = np.nanmedian(sim, axis=1)
    gap = best_sim - median_sim
    is_match = gap > gap_threshold

    rows = np.where(is_match)[0]
    cols = best_j[is_match]
    edge_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_valid, n_valid))
    n_components, labels = connected_components(edge_matrix, directed=True, connection="weak")

    chain_sizes = np.bincount(labels)
    nontrivial = chain_sizes[chain_sizes > 1]
    print(
        f"[kuhlmann_nv] reconstructed {len(nontrivial)} non-trivial preictal chains "
        f"({int(nontrivial.sum())} segments) from {n_valid} valid segments "
        f"({int(degenerate.sum())} excluded for a degenerate/flat boundary) -- "
        f"chain sizes: {sorted(nontrivial, reverse=True)[:10]}"
    )

    chain_offset = int(segment_ids.max()) + 1
    global_idx = preictal_idx[valid_local]
    group_ids[global_idx] = chain_offset + labels
    return group_ids


def window_segments(
    X: np.ndarray, y: np.ndarray, segment_ids: np.ndarray, window_length: float, fs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice each (n_channels, n_timepoints) 10-minute segment into
    non-overlapping window_length-second sub-windows, inheriting the
    parent segment's label. Needed because raw-classifier architectures
    in this repo (GodoyTMCClassifier, DBConformerClassifier, SlimSeiz...)
    are tuned around this repo's CHB-MIT window sizes (30s @ 256Hz =
    7,680 samples) -- feeding a whole 240,000-sample NV segment through
    GodoyTMC's channel-major conv tokenizer directly blows up its
    Transformer's token count to n_channels * (T / 360) (~10,656 tokens
    for a full 400Hz segment vs. ~483 for a 30s CHB-MIT window),
    OOM-ing attention (confirmed empirically 2026-09-22: a 13.5GB single
    attention-buffer allocation on a 4-sample batch). Windowing to the
    same ballpark length CHB-MIT already uses sidesteps that without
    touching the architecture.

    Returns (X_win, y_win, group_ids) where group_ids equals the
    ORIGINATING segment's id, repeated per sub-window -- callers MUST use
    a grouped split (e.g. StratifiedGroupKFold(groups=group_ids)), never
    a plain per-window split: sub-windows from the same segment are highly
    correlated (same 10-minute recording), so letting them land on both
    sides of a train/test split is the same adjacent-window leakage risk
    godoy_tmc_classifier.py's module docstring flags for the paper's own
    random-split evaluation.
    """
    win_samples = int(round(window_length * fs))
    n_segments, n_channels, n_time = X.shape
    n_windows_per_segment = n_time // win_samples
    if n_windows_per_segment < 1:
        raise ValueError(f"window_length={window_length}s ({win_samples} samples) exceeds segment length {n_time}")

    # Preallocate and fill in place -- same "don't build a Python list and
    # then np.concatenate a second copy of it" reasoning as
    # load_patient_train, and this array is the same order of magnitude as
    # X (windowing reshapes, it doesn't shrink the byte count).
    n_windows_total = n_segments * n_windows_per_segment
    X_win = np.empty((n_windows_total, n_channels, win_samples), dtype=np.float32)
    y_win = np.empty(n_windows_total, dtype=np.int64)
    group_ids = np.empty(n_windows_total, dtype=np.int64)

    for i, (seg_x, seg_y, seg_id) in enumerate(zip(X, y, segment_ids)):
        trimmed = seg_x[:, : n_windows_per_segment * win_samples]
        windows = trimmed.reshape(n_channels, n_windows_per_segment, win_samples).transpose(1, 0, 2)
        lo, hi = i * n_windows_per_segment, (i + 1) * n_windows_per_segment
        X_win[lo:hi] = windows
        y_win[lo:hi] = seg_y
        group_ids[lo:hi] = seg_id

    return X_win, y_win, group_ids
