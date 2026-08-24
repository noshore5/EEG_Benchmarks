"""
Editable-params smoke test for the dense_edge_gru / dense_edge pipelines.

PARAMS below is the primary way to configure a run -- edit it directly and:

    python Epilepsy/smoke_test.py

The point is a single block of constants to tweak (window_length,
channel_subset_k, cache on/off, ...) instead of re-typing/remembering CLI
flags for every ad hoc timing comparison (see the 2026-08-23 session's
cache-vs-no-cache and channel_subset_k=4-vs-8 checks -- this is that
workflow, scripted).

Every PARAMS key also has an optional CLI flag (2026-08-23, added after a
user reached for one instinctively) that overrides just that key for one
run, e.g.:

    python Epilepsy/smoke_test.py --channel-subset-k 12 --epochs 4

Flags left unpassed keep PARAMS's own value -- `python smoke_test.py` with
no flags at all runs PARAMS completely unchanged, same as before this
existed. The flags are a convenience layer on top of PARAMS, not a
replacement for it -- there's no flag for anything not already a PARAMS
key.

Reuses run_pipelines.py's own dataset-building
(_build_windowed_dataset) and leave-one-seizure-out loop
(leave_one_seizure_out_prediction/leave_one_seizure_out_detection, both via
their max_folds param) rather than duplicating that logic -- so the numbers
here match what a real run_pipelines.py invocation would produce for the
same params, not an approximation of it. Each fold builds its own fresh
classifier internally and neither loop function returns it, so per-epoch
timing isn't reachable as a return value -- common.py's TorchEEGClassifier
DOES now record it on self.epoch_time_history_ (added alongside this
script, useful to any caller that DOES hold a classifier reference, e.g.
pipeline_debug.py), but this script instead tees stdout and greps out each
printed "[Train][Epoch N/M] ... epoch_time=X.XXs" line (see
_EpochTimeCapture below) so it works across every fold without touching
either loop function's signature.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the actual pipeline module rather than duplicating its
# configuration (dataset building, param dicts, fold loop) -- same
# convention as pipeline_debug.py.
import Epilepsy.run_pipelines as rp


# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
PARAMS = dict(
    pipeline="dense_edge_gru",        # "dense_edge_gru" | "dense_edge"
    label_mode="prediction",          # "prediction" or "detection"
    cwt_encoder=False,  # True: learned CWT node embeddings
                                       # alongside WCT edges. Works for
                                       # either dense_edge pipeline.
    cwt_encoder_ablation="none",     # "none" | "node_only" | "zero_node_embed"
                                       # node_only = CWT encoder, no WCT.
    subjects=[1],
    window_length=30.0,               # seconds/window. Real default: 30.0
                                       # (prediction) / 4.0 (detection).
    step_size=30.0,                   # seconds between window starts.
    epochs=2,
    max_folds=1,                      # None = every leave-one-seizure-out
                                       # fold (slow -- see run_pipelines.py's
                                       # --max-folds docstring).
    max_interictal_recordings=5,      # label_mode="prediction" only. None =
                                       # every seizure-free recording for the
                                       # subject (~33 files/~1.7GB for chb01).
    channel_subset_k=4,               # None = full mesh. k channels ->
                                       # clique of k*(k-1)/2 live edges
                                       # scattered into full-E zeros.
    channel_subset_metric="abs_cosine",
    dense_edge_amp_bf16=True,
    train_amp_bf16=True,
    disable_disk_cache=None,          # None = True on cuda, False on mps/cpu
                                       # (see run_pipelines.resolve_disable_disk_cache).
                                       # True = force-recompute every trial.
    device="mps",                     # "cuda" / "mps" / "cpu" / "auto".
    verbose=2,                        # 2 = per-chunk dense-edge phase timing
                                       # too, not just per-epoch.
    seed=42,
    validation_split=0.2,             # 0.0 = no val; 0.2 tracks val_loss.
    early_stopping_patience=5,        # consecutive epochs without val_loss
                                       # improvement before stop. None = run
                                       # all epochs (then restore best if
                                       # validation_split > 0). Does nothing
                                       # unless validation_split > 0.
)
# ---------------------------------------------------------------------------


_EPOCH_LINE_RE = re.compile(
    r"\[Train\]\[Epoch (\d+)/(\d+)\] loss=([\d.]+).*?epoch_time=([\d.]+)s(?: val_loss=([\d.]+))?"
)


class _EpochTimeCapture:
    """Tees stdout to the real terminal while grepping each printed
    "[Train][Epoch N/M] ... epoch_time=X.XXs" line (common.py's own
    training-loop print) into a flat, fold-boundary-order-preserved list --
    see this module's docstring for why stdout-scraping instead of a
    classifier attribute."""

    def __init__(self, real_stdout):
        self.real_stdout = real_stdout
        # (epoch, of, train_loss, val_loss|None, seconds)
        self.epoch_times: list[tuple[int, int, float, float | None, float]] = []
        self._buf = ""

    def write(self, text: str) -> None:
        self.real_stdout.write(text)
        # Progress bars (tqdm) rewrite the current line with '\r', so a
        # single printed line can arrive across several write() calls with
        # no trailing '\n' until the line is actually done -- buffer until
        # a newline (or '\r', tqdm's own line-end) shows up before matching.
        self._buf += text
        while True:
            for sep in ("\n", "\r"):
                if sep in self._buf:
                    line, self._buf = self._buf.split(sep, 1)
                    match = _EPOCH_LINE_RE.search(line)
                    if match:
                        val_loss = (
                            float(match.group(5)) if match.group(5) is not None else None
                        )
                        self.epoch_times.append(
                            (
                                int(match.group(1)),
                                int(match.group(2)),
                                float(match.group(3)),
                                val_loss,
                                float(match.group(4)),
                            )
                        )
                    break
            else:
                break

    def flush(self) -> None:
        self.real_stdout.flush()


def _int_or_none(value: str) -> int | None:
    """argparse type= for the PARAMS keys that accept None (max_folds,
    max_interictal_recordings, channel_subset_k, early_stopping_patience)
    -- lets a CLI override explicitly null one out (`--channel-subset-k none`),
    not just set an int."""
    return None if value.lower() == "none" else int(value)


# Sentinel default for every CLI flag below, distinct from `None` -- three
# PARAMS keys (max_folds, max_interictal_recordings, channel_subset_k) have
# None as a REAL, meaningful value ("every fold" / "full mesh"), so using
# bare `None` as the "flag not passed" default (as a first cut of this did)
# made `--max-folds none` indistinguishable from not passing --max-folds at
# all -- confirmed directly: it silently left PARAMS's max_folds=1 in place
# instead of nulling it out. This sentinel is applied uniformly to every
# flag below (not just the three nullable ones) so main()'s override loop
# has one consistent "was this passed?" check for all of them.
_UNSET = object()


def _build_arg_parser() -> argparse.ArgumentParser:
    """One optional flag per PARAMS key, default _UNSET everywhere -- _UNSET
    means "not passed, keep PARAMS's own value" (see main()'s override
    loop), so this can never silently change a value the user didn't ask to
    change. Dest names (argparse auto-converts '-' to '_') match PARAMS
    keys exactly on purpose, so the override loop needs no name mapping."""
    parser = argparse.ArgumentParser(
        description=(
            "Optional per-run overrides on top of the PARAMS dict at the top "
            "of this file -- PARAMS is still the default for anything not "
            "passed here. `python smoke_test.py` with no flags runs PARAMS "
            "completely unchanged."
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=["dense_edge_gru", "dense_edge"],
        default=_UNSET,
    )
    parser.add_argument("--label-mode", choices=["prediction", "detection"], default=_UNSET)
    parser.add_argument(
        "--cwt-encoder",
        action=argparse.BooleanOptionalAction,
        default=_UNSET,
    )
    parser.add_argument(
        "--cwt-encoder-ablation",
        choices=["none", "zero_node_embed", "node_only"],
        default=_UNSET,
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=_UNSET)
    parser.add_argument("--window-length", type=float, default=_UNSET)
    parser.add_argument("--step-size", type=float, default=_UNSET)
    parser.add_argument("--epochs", type=int, default=_UNSET)
    parser.add_argument("--max-folds", type=_int_or_none, default=_UNSET)
    parser.add_argument("--max-interictal-recordings", type=_int_or_none, default=_UNSET)
    parser.add_argument("--channel-subset-k", type=_int_or_none, default=_UNSET)
    parser.add_argument("--channel-subset-metric", default=_UNSET)
    parser.add_argument("--dense-edge-amp-bf16", action=argparse.BooleanOptionalAction, default=_UNSET)
    parser.add_argument("--train-amp-bf16", action=argparse.BooleanOptionalAction, default=_UNSET)
    parser.add_argument("--disable-disk-cache", action=argparse.BooleanOptionalAction, default=_UNSET)
    parser.add_argument("--device", default=_UNSET)
    parser.add_argument("--verbose", type=int, default=_UNSET)
    parser.add_argument("--seed", type=int, default=_UNSET)
    parser.add_argument("--validation-split", type=float, default=_UNSET)
    parser.add_argument("--early-stopping-patience", type=_int_or_none, default=_UNSET)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    p = dict(PARAMS)
    for key, value in vars(args).items():
        if value is not _UNSET:
            p[key] = value
    p["disable_disk_cache"] = rp.resolve_disable_disk_cache(
        p["device"], p["disable_disk_cache"],
    )

    print("=== smoke_test.py params ===")
    for key, value in p.items():
        print(f"  {key}={value!r}")
    print()

    t_build0 = time.perf_counter()
    X, y, metadata = rp._build_windowed_dataset(
        p["subjects"],
        p["window_length"],
        p["step_size"],
        label_mode=p["label_mode"],
        sph=rp.DEFAULT_SPH,
        sop=rp.DEFAULT_SOP,
        postictal_buffer=rp.DEFAULT_POSTICTAL_BUFFER,
        max_interictal_recordings=(
            p["max_interictal_recordings"] if p["label_mode"] == "prediction" else None
        ),
    )
    t_build = time.perf_counter() - t_build0
    print(
        f"X: {X.shape}  preictal/ictal windows: {int(y.sum())}/{len(y)} "
        f"({100 * y.mean():.1f}%)"
    )
    print(f"[smoke_test] dataset build: {t_build:.2f}s\n")

    base_params = rp._dense_family_params(p["pipeline"], p["label_mode"])
    clf_params = dict(base_params)
    clf_params["seed"] = p["seed"]
    clf_params["device"] = p["device"]
    clf_params["dense_edge_amp_bf16"] = p["dense_edge_amp_bf16"]
    clf_params["train_amp_bf16"] = p["train_amp_bf16"]
    clf_params["channel_subset_k"] = p["channel_subset_k"]
    clf_params["channel_subset_metric"] = p["channel_subset_metric"]
    clf_params["verbose"] = p["verbose"]
    clf_params["validation_split"] = p["validation_split"]
    clf_params["early_stopping_patience"] = p["early_stopping_patience"]
    clf_params["cwt_encoder"] = p["cwt_encoder"]
    clf_params["time_frequency_node_ablation"] = p["cwt_encoder_ablation"]
    if p["cwt_encoder_ablation"] != "none":
        clf_params["cwt_encoder"] = True
    print(
        f"[smoke_test] dense_edge_temporal_mode="
        f"{clf_params['dense_edge_temporal_mode']!r} "
        f"cwt_encoder="
        f"{clf_params.get('cwt_encoder', False)!r} "
        f"time_frequency_node_ablation="
        f"{clf_params.get('time_frequency_node_ablation', 'none')!r}\n"
    )

    real_stdout = sys.stdout
    capture = _EpochTimeCapture(real_stdout)
    sys.stdout = capture
    try:
        t0 = time.perf_counter()
        if p["label_mode"] == "prediction":
            results, _per_seizure = rp.leave_one_seizure_out_prediction(
                X, y, metadata, clf_params, p["epochs"], p["window_length"],
                disable_disk_cache=p["disable_disk_cache"],
                max_folds=p["max_folds"],
            )
        else:
            results = rp.leave_one_seizure_out_detection(
                X, y, metadata, clf_params, p["epochs"],
                disable_disk_cache=p["disable_disk_cache"],
                max_folds=p["max_folds"],
            )
        total = time.perf_counter() - t0
    finally:
        sys.stdout = real_stdout

    print(f"\n[smoke_test] total leave-one-seizure-out wall time: {total:.2f}s")
    print(results.to_string())

    print("\n=== epoch loss ===")
    if not capture.epoch_times:
        print("  (no epoch lines captured -- was clf_params['verbose'] >= 1?)")
    else:
        fold_i = 0
        prev_epoch = 0
        print(f"  {'fold':>4} {'epoch':>7} {'loss':>10} {'val_loss':>10} {'epoch_s':>8}")
        for epoch, of, loss, val_loss, seconds in capture.epoch_times:
            if epoch <= prev_epoch:
                fold_i += 1
            prev_epoch = epoch
            val_s = f"{val_loss:.6f}" if val_loss is not None else "     --"
            print(f"  {fold_i:4d} {epoch:3d}/{of:<3d} {loss:10.6f} {val_s:>10} {seconds:8.2f}")
        all_times = [t for _, _, _, _, t in capture.epoch_times]
        print(
            f"  mean epoch_time={sum(all_times) / len(all_times):.2f}s  "
            f"min={min(all_times):.2f}s  max={max(all_times):.2f}s  "
            f"n={len(all_times)}"
        )


if __name__ == "__main__":
    main()
