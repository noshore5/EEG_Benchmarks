"""Train the SzCORE seizure-DETECTION checkpoint (TMC-T / godoy_tmc arch).

Pools CHB-MIT recordings into 4 s bipolar-montage windows labelled
ictal/non-ictal, fits this repo's ``GodoyTMCClassifier``, and writes
``algo/model_weights.pt`` for the container.

    .venv/bin/python Epilepsy/szcore/train_detector.py --subjects 1 --epochs 20 --device mps
    .venv/bin/python Epilepsy/szcore/train_detector.py --subjects 1 2 3 5 --epochs 25

Note this is seizure DETECTION (ictal windows are positive), a different
task from the repo's headline preictal-PREDICTION rows -- do not compare
the numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `algo` package

from algo.common import FS, N_CHANNELS, WINDOW_S, TRAIN_STEP_S, load_bipolar_eeg, make_windows  # noqa: E402

from datasets.epilepsy.chb_mit import CHBMIT  # noqa: E402
from Epilepsy.pipelines.godoy_tmc_classifier import GodoyTMCClassifier  # noqa: E402

WEIGHTS_OUT = Path(__file__).resolve().parent / "algo" / "model_weights.pt"

MODEL_KWARGS = dict(
    d_model=32, n_heads=8, ffn_hidden=64, n_encoder_layers=1,
    dropout=0.1, head_hidden=128, head_dropout=0.5,
)


def _seizure_mask(n_samples: int, fs: float, seizures) -> np.ndarray:
    m = np.zeros(n_samples, dtype=bool)
    for onset, offset in seizures:
        a, b = int(round(onset * fs)), int(round(offset * fs))
        m[max(0, a) : min(n_samples, b)] = True
    return m


def build_dataset(subjects, window, step, pos_overlap=0.5, max_records=None):
    ds = CHBMIT()
    Xs, ys, groups = [], [], []
    for subj in subjects:
        records = ds.list_records(subj)
        # Keep every seizure record; cap the seizure-free ones if asked.
        if max_records is not None:
            sz = [r for r in records if r["seizures"]]
            free = [r for r in records if not r["seizures"]][: max(0, max_records - len(sz))]
            records = sz + free
        for rec in records:
            fname = rec["filename"]
            try:
                path = ds._local_file(subj, fname)
                data, fs = load_bipolar_eeg(str(path))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {fname}: {exc}")
                continue
            windows, starts = make_windows(data, window, step)
            if windows.shape[0] == 0:
                continue
            sm = _seizure_mask(data.shape[1], fs, rec["seizures"])
            frac = np.array([sm[s : s + window].mean() for s in starts])
            y = (frac >= pos_overlap).astype(np.int64)
            Xs.append(windows)
            ys.append(y)
            groups.append(np.full(y.shape[0], f"{subj}:{fname}"))
            print(f"  {fname}: {windows.shape[0]} windows, {int(y.sum())} ictal")
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    g = np.concatenate(groups)
    return X, y, g


def subsample_negatives(X, y, g, ratio, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    target = int(ratio * pos.size)
    if target < neg.size:
        neg = rng.choice(neg, size=target, replace=False)
    keep = np.sort(np.concatenate([pos, neg]))
    return X[keep], y[keep], g[keep]


def pick_threshold(probs, y):
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.30, 0.96, 0.02), 2):
        pred = probs >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        f1 = tp / (tp + 0.5 * (fp + fn)) if tp else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--neg-ratio", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-records-per-subject", type=int, default=None,
                    help="cap seizure-free recordings per subject (smoke/memory)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the auto-picked operating point")
    ap.add_argument("--out", default=str(WEIGHTS_OUT))
    args = ap.parse_args()

    window = int(round(WINDOW_S * FS))
    step = int(round(TRAIN_STEP_S * FS))

    print(f"[data] subjects={args.subjects} window={window} step={step}")
    X, y, g = build_dataset(
        args.subjects, window, step, max_records=args.max_records_per_subject
    )
    print(f"[data] total {X.shape[0]} windows, {int(y.sum())} ictal ({y.mean():.4%})")
    X, y, g = subsample_negatives(X, y, g, args.neg_ratio, args.seed)
    print(f"[data] after negative subsample: {X.shape[0]} windows, {int(y.sum())} ictal")
    assert X.shape[1] == N_CHANNELS

    clf = GodoyTMCClassifier(
        **MODEL_KWARGS,
        normalize_input=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        validation_split=0.2,
        early_stopping_patience=5,
        device=args.device,
        seed=args.seed,
        use_class_weights=True,
        verbose=1,
    )
    clf.fit(X, y)

    probs = clf.predict_proba(X)[:, 1]
    auto_t, f1 = pick_threshold(probs, y)
    threshold = args.threshold if args.threshold is not None else auto_t
    print(f"[thr] auto operating point {auto_t} (train sample-F1 {f1:.3f}); using {threshold}")

    ckpt = dict(
        arch="godoy_tmc_tmct",
        task="detection",
        state_dict={k: v.cpu() for k, v in clf.model_.state_dict().items()},
        model_kwargs=MODEL_KWARGS,
        n_channels=N_CHANNELS,
        n_time=window,
        fs=FS,
        window_s=WINDOW_S,
        x_mean=float(clf.X_mean_),
        x_std=float(clf.X_std_),
        threshold=float(threshold),
        train_subjects=list(args.subjects),
        train_epochs=args.epochs,
    )
    import torch

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
