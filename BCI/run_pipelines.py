"""Single entry point for every active pipeline in this repo.

    python BCI/run_pipelines.py --pipeline sparse_evidence --subjects 1 2 3 4
    python BCI/run_pipelines.py --pipeline eegnet wct_evidence_gnn --subjects 1

Pipelines: eegnet, wct_evidence_gnn, sparse_evidence, dense_edge, dense_edge_gru,
evolving_graph. dense_edge/dense_edge_gru/evolving_graph are all
SparseEvidenceGNNClassifier under different event_mode/dense_edge_temporal_mode
combinations (see PIPELINE_SPECS below) -- dense_edge_gru swaps dense_edge's
Conv2d temporal step for a per-edge GRU (still no cross-node propagation over
time); evolving_graph is a genuinely different mechanism, a per-node GRU that
walks the whole graph timestep-by-timestep. dense_edge also has an opt-in
DENSE_EDGE_TIME_AVERAGED module toggle (see its own comment above
DENSE_EDGE_PARAMS) that discards within-trial time resolution entirely
instead of dense_edge_time_downsample's partial pool.

Use --param-names/--param-values to override any pipeline's params for a
one-off run without editing this file, e.g.:
    --pipeline sparse_evidence --param-names epochs seed --param-values 50 43
"""

from __future__ import annotations

import argparse
import ast
import logging
import numbers
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from mne import get_config
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation, CrossSubjectEvaluation

try:
    from moabb.evaluations import GlobalFutureSessionEvaluation
except ImportError:
    GlobalFutureSessionEvaluation = None
from moabb.paradigms import LeftRightImagery

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BCI.moabb_pipelines.EEGNet import EEGNetClassifier
from BCI.moabb_pipelines.wct_evidence_gnn_classifier import (
    WCTEvidenceGNNClassifier,
)
from BCI.moabb_pipelines.sparse_evidence_gnn_classifier import (
    SparseEvidenceGNNClassifier,
)
from BCI.moabb_pipelines.common import (
    append_run_ledger_rows,
    run_ledger_rows_from_group,
)
from BCI.experiment_logging import (
    EventCategory,
    add_console_arguments,
    configure_experiment_logging,
    console_policy_from_args,
    log_event,
    resolve_experiment_log_path,
)

log = logging.getLogger(__name__)

# Default subject list, used as the --subjects default and anywhere else in
# this file that needs "the subjects for this run" without going through args.
DEFAULT_SUBJECTS = [1]


class RunConfigurationError(ValueError):
    """Configuration error the CLI should report as an argument error."""


# ---------------------------------------------------------------------------
# Pipeline parameters: GLOBAL_PARAMS applies to every pipeline; SPARSE_FAMILY_
# PARAMS is shared by the four SparseEvidenceGNNClassifier variants. Each
# pipeline's own dict only needs to state what's different from those.
# ---------------------------------------------------------------------------

GLOBAL_PARAMS = dict(seed=42)

SPARSE_FAMILY_PARAMS = dict(
    GLOBAL_PARAMS,
    sampling_rate=250,
    lowest=8.0,
    highest=30.0,
    nfreqs=16,
    cwt_resample_n_time=None,  # native resolution -- resampling destroys real signal and breaks the COI mask
    coherence_threshold=0.99,
    phase_threshold_deg=10.0,
    channel_encoder_dilation=5,  # widens ChannelSignalEncoder's receptive field to ~1 mu-band cycle
    hidden_dim=8,
    channel_embed_dim=8,
    epochs=60,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-4,
    grad_clip_norm=0.1,
    normalize_input=True,
    validation_split=0.0,  # internal val split measured worse than 0.0 for cross-session scoring here
    device="auto",
    surrogate_seed=42,
    smooth_kernel_size=(5, 3),
    coi_enabled=True,
    feature_ablation="none",
    channel_subset=[1, 5, 7, 8, 9, 10, 11, 13, 17],
    coherence_threshold_mode="fixed",
    surrogate_count=100,
    surrogate_percentile=99.0,
    verbose=2,
)

SPARSE_EVIDENCE_PARAMS = dict(
    SPARSE_FAMILY_PARAMS,
    event_mode="sparse",
)

DENSE_EDGE_PARAMS = dict(
    SPARSE_FAMILY_PARAMS,
    event_mode="dense",
    event_aggregation="concat",
    dense_edge_time_downsample=8,
    dense_edge_temporal_mode="conv",
    # dense_conv_kernel_size/dense_conv_pool_size are listed explicitly here
    # (at their ordinary classifier defaults, 5/4 -- no behavior change) so
    # --param-names can actually reach them: _build_estimator only allows
    # overriding keys already present in a pipeline's own params dict, and
    # time_averaged_graph=True below requires both set to 1.
    dense_conv_kernel_size=5,
    dense_conv_pool_size=4,
    # Opt-in toggle (ZERO_CHANNEL_EMBED/USE_CONCAT_DENSE-style, from before the
    # run_pipelines.py consolidation): discards within-trial time resolution
    # entirely instead of dense_edge_time_downsample's fixed-factor partial
    # pool -- collapses the [B, 4, E, T, F] stack to a single COI-weighted
    # average per (edge, frequency) before dense_edge_conv ever runs (see
    # time_averaged_graph's own docstring on SparseEvidenceGNNCore). Requires
    # dense_edge_time_downsample=1 (mutually exclusive -- averaging the whole
    # axis makes a partial pool upstream of it meaningless) and
    # dense_conv_kernel_size=dense_conv_pool_size=1 (once T=1, a k>1 Conv2d has
    # nothing left to convolve over) -- both enforced by the classifier's own
    # __init__, so leaving this False (default) is the only way to keep
    # dense_edge_time_downsample=8/dense_conv_kernel_size=5/
    # dense_conv_pool_size=4 in effect above. 2026-08-11, 4 subjects/1 seed:
    # ties/slightly beats the time-resolved baseline (0.7999 vs 0.7897 cohort
    # mean) -- see [[sparse-evidence-gnn-time-averaged-graph-feature]] in
    # project memory.
    time_averaged_graph=False,
)

DENSE_EDGE_GRU_PARAMS = dict(
    DENSE_EDGE_PARAMS,
    dense_edge_temporal_mode="rnn",
)

EVOLVING_GRAPH_PARAMS = dict(
    SPARSE_FAMILY_PARAMS,
    event_mode="temporal_graph",
    event_aggregation="mean",  # required by event_mode="temporal_graph"
    dense_edge_time_downsample=8,
    coherence_threshold_mode="surrogate",
    coherence_threshold=0.0,
    surrogate_percentile=99.0,
)

EEGNET_PARAMS = dict(
    GLOBAL_PARAMS,
    validation_split=0.0,
    verbose=1,
)

WCT_EVIDENCE_GNN_PARAMS = dict(
    GLOBAL_PARAMS,
    cwt_resample_n_time=200,
    coherence_threshold=0.5,
    use_mag=False,
    use_raw=False,
    readout_mode="flatten",
    evidence_norm="active_slots",
    epochs=50,
    batch_size=16,
    weight_decay=1e-2,
    noise_augmentation_enabled=True,
    noise_apply_prob=1.0,
    noise_strength=0.15,
    noise_bank_size=20000,
    noise_bank_seed=33,
    last_batch_min_ratio=0.5,
    selector_alpha_val_update_rate=0.5,
    optimizer_step_remainder_policy="carry",
    select_message_mlp_gate={"mode": "gumbel_hard"},
    message_mlp_selector_mode="separate_val",
    feature_conv_intermediate_channels_reduced=4,
    feature_conv_feature_dim=2,
    smooth_kernel_size=(5, 3),
    window_compute_mode="chunked",
    max_windows_per_chunk=1,
    channel_subset=[1, 5, 7, 8, 9, 10, 11, 13, 17],
    verbose=2,
)

PIPELINE_SPECS: dict[str, dict] = {
    "eegnet": dict(display="EEGNet", cls=EEGNetClassifier, params=EEGNET_PARAMS),
    "wct_evidence_gnn": dict(
        display="WCT-Evidence-GNN", cls=WCTEvidenceGNNClassifier, params=WCT_EVIDENCE_GNN_PARAMS
    ),
    "sparse_evidence": dict(
        display="Sparse-Evidence-GNN", cls=SparseEvidenceGNNClassifier, params=SPARSE_EVIDENCE_PARAMS
    ),
    "dense_edge": dict(display="Dense-Edge", cls=SparseEvidenceGNNClassifier, params=DENSE_EDGE_PARAMS),
    "dense_edge_gru": dict(
        display="Dense-Edge-GRU", cls=SparseEvidenceGNNClassifier, params=DENSE_EDGE_GRU_PARAMS
    ),
    "evolving_graph": dict(
        display="Evolving-Graph", cls=SparseEvidenceGNNClassifier, params=EVOLVING_GRAPH_PARAMS
    ),
}


# ---------------------------------------------------------------------------
# --param-names/--param-values: safe literal overrides for a one-off run.
# ---------------------------------------------------------------------------


def _parse_param_value(raw_value: str):
    normalized = raw_value.strip()
    named_literals = {"true": True, "false": False, "none": None, "null": None}
    if normalized.lower() in named_literals:
        return named_literals[normalized.lower()]
    try:
        return ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        return normalized  # bare estimator strings, e.g. "auto"


def _normalize_param_names(param_names):
    if param_names is None:
        return None
    normalized = [name.strip() for token in param_names for name in token.split(",")]
    if any(not name for name in normalized):
        raise ValueError("--param-names contains an empty parameter name.")
    return normalized


def _normalize_param_values(raw_param_values):
    if raw_param_values is None:
        return None
    if len(raw_param_values) == 1:
        parsed = _parse_param_value(raw_param_values[0])
        return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    return [_parse_param_value(raw_value) for raw_value in raw_param_values]


def _values_are_type_compatible(value, reference) -> bool:
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and not math.isfinite(value):
        return False
    if isinstance(reference, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(reference, bool)
    if isinstance(reference, numbers.Integral):
        return isinstance(value, numbers.Integral) and not isinstance(value, bool)
    if isinstance(reference, numbers.Real):
        return isinstance(value, numbers.Real) and not isinstance(value, bool)
    return isinstance(value, type(reference))


def _parse_overrides(param_names, raw_param_values) -> dict:
    if param_names is None and raw_param_values is None:
        return {}
    names = _normalize_param_names(param_names)
    values = _normalize_param_values(raw_param_values)
    if not names or not values:
        raise ValueError("--param-names and --param-values must both be given at least one value.")
    if len(names) != len(values):
        raise ValueError("--param-names and --param-values must contain the same number of entries.")
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError("--param-names must not contain duplicates: " + ", ".join(duplicates))
    return dict(zip(names, values))


def _build_estimator(pipeline_key: str, fixed_overrides: dict):
    spec = PIPELINE_SPECS[pipeline_key]
    params = dict(spec["params"])
    unsupported = sorted(set(fixed_overrides) - set(params))
    if unsupported:
        raise RunConfigurationError(
            f"Pipeline '{pipeline_key}' has no parameter(s): {', '.join(unsupported)}"
        )
    for name, value in fixed_overrides.items():
        if value is not None and not _values_are_type_compatible(value, params[name]):
            raise RunConfigurationError(
                f"Value {value!r} for parameter '{name}' is not type-compatible with "
                f"pipeline '{pipeline_key}' (expected {type(params[name]).__name__})."
            )
    params.update(fixed_overrides)
    return spec["cls"](**params)


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------


def _resolve_eval_modes(global_hyperparam_fit_mode: str, cross_subject: bool) -> list[str]:
    mode = str(global_hyperparam_fit_mode).lower()
    if mode == "false":
        modes = ["cross"]
    elif mode == "true":
        modes = ["global"]
    elif mode == "both":
        modes = ["cross", "global"]
    else:
        raise ValueError(f"Unsupported global_hyperparam_fit='{global_hyperparam_fit_mode}'.")
    if cross_subject:
        modes.append("subject")
    return modes


def _make_evaluation(eval_mode: str, eval_kwargs: dict):
    if eval_mode == "global":
        if GlobalFutureSessionEvaluation is None:
            raise ValueError(
                "GlobalFutureSessionEvaluation not available in this moabb version; "
                "use --global-hyperparam-fit false instead."
            )
        return GlobalFutureSessionEvaluation(**eval_kwargs)
    if eval_mode == "cross":
        return CrossSessionEvaluation(**eval_kwargs)
    if eval_mode == "subject":
        # Leave-one-subject-out: trains on all subjects but one, tests on the
        # held-out subject; repeats per subject.
        return CrossSubjectEvaluation(**eval_kwargs)
    raise ValueError(f"Unsupported eval_mode='{eval_mode}'.")


def _safe_run_id(value) -> str:
    run_id = value or os.environ.get("LOCAL_ORCHESTRATION_EXECUTION_ID")
    if run_id is None:
        run_id = datetime.now().strftime("manual-%Y%m%d-%H%M%S-%f")
    run_id = str(run_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "Run ID must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen."
        )
    return run_id


def _configured_data_root() -> Path:
    configured = (
        os.environ.get("MNE_DATASETS_BNCI_PATH")
        or get_config("MNE_DATASETS_BNCI_PATH")
        or os.environ.get("MNE_DATA")
        or get_config("MNE_DATA")
    )
    if configured is None:
        configured = Path.home() / "mne_data"
    data_root = Path(configured).expanduser().resolve()
    log_event(log, EventCategory.CONFIG, f"[Data] configured MOABB/MNE root: {data_root}")
    return data_root


def _configured_moabb_result_root() -> Path:
    configured = (
        os.environ.get("MOABB_RESULTS")
        or get_config("MOABB_RESULTS")
        or os.environ.get("MNE_DATA")
        or get_config("MNE_DATA")
        or str(Path.home() / "mne_data")
    )
    return Path(configured).expanduser().resolve()


def _markdown_table(frame, columns) -> list[str]:
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return lines


def _write_group_artifacts(
    *,
    evaluation,
    group_results,
    group_id: str,
    run_id: str,
    subjects: list[int],
    eval_mode: str,
    effective_params: dict,
    data_root: Path,
    fixed_overrides: dict | None,
    experiment_log_path,
    console_policy,
    overwrite: bool,
) -> None:
    """Write compact human-readable companions beside MOABB's HDF5 store.

    NOTE: `group_results.to_csv` below overwrites, unguarded by a lock (unlike
    MOABB's own HDF5 store). Don't run this script twice against the same
    --run-id concurrently -- see [[run-wct-gnn-concurrent-write-race]].
    """
    hdf5_path = Path(evaluation.results.filepath).resolve()
    artifact_dir = hdf5_path.parent
    scores_path = artifact_dir / f"scores_{group_id}.csv"
    summary_path = artifact_dir / f"summary_{group_id}.md"
    if not overwrite:
        existing = [path for path in (scores_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Run artifact already exists; refusing to overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    group_results.to_csv(scores_path, index=False)

    outer_columns = ["subject", "session", "pipeline", "score"]
    means = (
        group_results.groupby(["subject", "pipeline"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_score"})
    )
    lines = [
        "# Run summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Group: `{group_id}`",
        f"- Evaluation mode: `{eval_mode}`",
        f"- Subjects: `{', '.join(str(subject) for subject in subjects)}`",
        f"- Configured data root: `{data_root}`",
        f"- HDF5 store: `{hdf5_path}`",
        f"- Scores CSV: `{scores_path}`",
    ]
    if experiment_log_path is not None:
        lines.append(f"- Experiment log: `{experiment_log_path}`")
    if console_policy is not None:
        lines.append(f"- Console output policy: `{console_policy!r}`")
    lines.append(f"- Overwrite existing outputs: `{overwrite}`")
    lines.append(f"- Fixed parameter overrides: `{fixed_overrides or {}}`")
    lines.extend(["", "## Outer-CV rows", ""])
    lines.extend(_markdown_table(group_results, outer_columns))
    lines.extend(["", "## Subject/pipeline means", ""])
    lines.extend(_markdown_table(means, ["subject", "pipeline", "mean_score"]))
    lines.extend(["", "## Effective estimator parameters", ""])
    for label in sorted(effective_params):
        lines.append(f"### {label}")
        lines.append("")
        for name in sorted(effective_params[label]):
            lines.append(f"- `{name}`: `{effective_params[label][name]!r}`")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log_event(log, EventCategory.ARTIFACT, f"[Artifact] HDF5: {hdf5_path}")
    log_event(log, EventCategory.ARTIFACT, f"[Artifact] scores CSV: {scores_path}")
    log_event(log, EventCategory.ARTIFACT, f"[Artifact] summary: {summary_path}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pipeline",
        action="append",
        default=["dense_edge"],
        choices=sorted(PIPELINE_SPECS),
        metavar="PIPELINE",
        help="Repeatable. One of: " + ", ".join(sorted(PIPELINE_SPECS)),
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--global-hyperparam-fit",
        default="false",
        choices=["false", "true", "both"],
        help="'false' => CrossSessionEvaluation, 'true' => GlobalFutureSessionEvaluation, 'both' => both.",
    )
    parser.add_argument(
        "--cross-subject",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run a leave-one-subject-out CrossSubjectEvaluation. Requires 2+ --subjects.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace outputs for an existing run ID (default). --no-overwrite rejects existing ones.",
    )
    parser.add_argument("--experiment-log", default=None)
    add_console_arguments(parser)
    parser.add_argument(
        "--param-names",
        "--param_names",
        nargs="+",
        default=None,
        metavar="PARAM",
        help="Estimator parameters to override. One token per name, or a quoted comma-separated list.",
    )
    parser.add_argument(
        "--param-values",
        "--param_values",
        nargs="+",
        default=None,
        metavar="VALUE",
        help="Safe literal values paired with --param-names.",
    )
    return parser


def parse_parameters(arguments: list[str] | None = None) -> argparse.Namespace:
    return _build_argument_parser().parse_args(arguments)


def main(args: argparse.Namespace) -> None:
    if not args.pipeline:
        raise RunConfigurationError(
            "--pipeline is required. Choose from: " + ", ".join(sorted(PIPELINE_SPECS))
        )
    selected_pipelines = list(dict.fromkeys(args.pipeline))
    run_id = _safe_run_id(args.run_id)
    try:
        fixed_overrides = _parse_overrides(args.param_names, args.param_values)
    except ValueError as exc:
        raise RunConfigurationError(str(exc)) from exc
    if args.cross_subject and len(args.subjects) < 2:
        raise RunConfigurationError(
            f"--cross-subject requires 2+ --subjects; got {args.subjects}."
        )
    eval_modes = _resolve_eval_modes(args.global_hyperparam_fit, args.cross_subject)

    console_policy = console_policy_from_args(args)
    moabb_results_root = _configured_moabb_result_root()
    try:
        experiment_log_path = resolve_experiment_log_path(
            args.experiment_log, run_id=run_id, moabb_results_root=moabb_results_root, overwrite=args.overwrite
        )
    except FileExistsError as exc:
        raise RunConfigurationError(str(exc)) from exc
    configure_experiment_logging(experiment_log_path, console_policy=console_policy, overwrite=args.overwrite)
    log_event(log, EventCategory.ARTIFACT, f"[Artifact] experiment log: {experiment_log_path}")
    log_event(log, EventCategory.STATUS, f"Run ID: {run_id}")
    log_event(log, EventCategory.CONFIG, f"Pipelines: {selected_pipelines}")
    log_event(log, EventCategory.CONFIG, f"Subjects: {args.subjects}")
    if fixed_overrides:
        log_event(log, EventCategory.CONFIG, f"Fixed parameter overrides: {fixed_overrides}")

    dataset = BNCI2014_001()
    dataset.subject_list = args.subjects
    data_root = _configured_data_root()
    log_event(log, EventCategory.CONFIG, f"[Results] MOABB root: {moabb_results_root}")
    paradigm = LeftRightImagery(fmin=8, fmax=35)

    base_eval_kwargs = dict(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=args.overwrite,
        n_jobs=1,
        random_state=42,
        progress_bar=console_policy.moabb_progress,
    )

    results_chunks = []
    for eval_mode in eval_modes:
        log_event(log, EventCategory.STATUS, f"=== Starting eval={eval_mode} ===")
        group_id = f"{run_id}__{eval_mode}"
        evaluation = _make_evaluation(eval_mode, dict(base_eval_kwargs, suffix=group_id))

        pipelines = {
            PIPELINE_SPECS[key]["display"]: _build_estimator(key, fixed_overrides)
            for key in selected_pipelines
        }
        for label, estimator in pipelines.items():
            log_event(
                log,
                EventCategory.INITIAL_DETAILS,
                f"Effective estimator parameters for '{label}': {estimator.get_params(deep=True)}",
            )

        group_results = evaluation.process(pipelines)
        log_event(log, EventCategory.STATUS, f"Completed group rows: {len(group_results)}")

        effective_params = {label: est.get_params(deep=True) for label, est in pipelines.items()}
        _write_group_artifacts(
            evaluation=evaluation,
            group_results=group_results,
            group_id=group_id,
            run_id=run_id,
            subjects=args.subjects,
            eval_mode=eval_mode,
            effective_params=effective_params,
            data_root=data_root,
            fixed_overrides=fixed_overrides,
            experiment_log_path=experiment_log_path,
            console_policy=console_policy,
            overwrite=args.overwrite,
        )
        ledger_rows = run_ledger_rows_from_group(
            run_id=run_id,
            group_id=group_id,
            eval_mode=eval_mode,
            inner_group=None,
            group_results=group_results,
            effective_params=effective_params,
            experiment_log_path=experiment_log_path,
            fixed_overrides=fixed_overrides,
        )
        append_run_ledger_rows(ledger_rows)
        results_chunks.append(group_results)

    results = pd.concat(results_chunks, ignore_index=True)

    log_event(log, EventCategory.FINAL_RESULTS, "=== Outer CV results ===")
    log_event(
        log,
        EventCategory.FINAL_RESULTS,
        results[["subject", "session", "pipeline", "score"]].to_string(index=False),
    )

    log_event(log, EventCategory.FINAL_RESULTS, "=== Per subject/pipeline mean scores ===")
    per_subject_pipeline = (
        results.groupby(["subject", "pipeline"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_score"})
        .sort_values(["subject", "pipeline"])
    )
    log_event(log, EventCategory.FINAL_RESULTS, per_subject_pipeline.to_string(index=False))

    log_event(log, EventCategory.FINAL_RESULTS, "=== Per pipeline mean scores ===")
    per_pipeline = (
        results.groupby(["pipeline"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_score"})
        .sort_values(["pipeline"])
    )
    log_event(log, EventCategory.FINAL_RESULTS, per_pipeline.to_string(index=False))


if __name__ == "__main__":
    argument_parser = _build_argument_parser()
    try:
        main(argument_parser.parse_args())
    except RunConfigurationError as exc:
        argument_parser.error(str(exc))
