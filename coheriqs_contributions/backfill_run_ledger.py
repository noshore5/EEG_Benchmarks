"""One-time backfill of the cross-run results ledger from historical
`experiment_*.log` files under ~/mne_data (or $MNE_DATA).

Built 2026-08-08 alongside `common.py`'s run-ledger helpers -- see that
file's "Cross-run results ledger" section for why this exists: per-run-id
`summary_<group_id>.md` / `scores_<group_id>.csv` files overwrite on every
reuse of a run-id, so historical runs under a reused run-id (e.g.
"canonical-sparse", reused ~47 times for both the actual canonical baseline
and unrelated one-off scratch tests) are only recoverable from the raw
timestamped experiment logs -- confirmed the hard way, when "did you already
run subject 2 at phase_threshold_deg=10" required grepping ~120 run
directories and hand-parsing log text instead of one query.

This script parses each log's "Effective estimator parameters" and
"=== Outer CV results ===" blocks (the same text `run_pipelines.py's main()`
prints at the end of every run) and appends one ledger row per historical
outer-CV row, via the exact same `append_run_ledger_rows` helper new runs
write to going forward, so old and new rows share one schema.

Best-effort: logs with no completed "=== Outer CV results ===" block
(crashed/interrupted runs -- e.g. the 2026-08-07 3-way-parallel
canonical-sparse-seed43/44/45 sweep, whose logs end mid-epoch) are skipped
and counted, not guessed at.

Usage:
    python -m coheriqs_contributions.backfill_run_ledger [--dry-run]
    python -m coheriqs_contributions.backfill_run_ledger --data-root ~/mne_data
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

import pandas as pd

from coheriqs_contributions.moabb_pipelines.common import (
    RUN_LEDGER_CONFIG_KEYS,
    append_run_ledger_rows,
    default_run_ledger_path,
)

EFFECTIVE_PARAMS_RE = re.compile(
    r"Effective estimator parameters for '([^']+)':\s*(\{.*\})\s*$"
)
FIXED_OVERRIDES_RE = re.compile(r"Fixed parameter overrides:\s*(\{.*\})\s*$")
STARTING_GROUP_RE = re.compile(
    r"=== Starting group: eval=(?P<eval_mode>\S+), inner_group=(?P<inner_group>\S+) ==="
)
OUTER_ROW_RE = re.compile(
    r"^\s*(?P<subject>\S+)\s+(?P<session>\S+)\s+(?P<pipeline>.+?)\s+"
    r"(?P<score>-?\d+\.\d+)\s+(?P<best_params>\S+)\s*$"
)
LOG_LINE_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _to_iso(ts: str) -> str:
    """'2026-08-07 17:20:07,298' -> '2026-08-07T17:20:07.298' (sortable)."""
    return ts.replace(",", ".").replace(" ", "T")


def _strip_log_prefix(line: str) -> str:
    """Drop '<timestamp> <LEVEL> <logger> [<category>] ' leaving the
    message. Only the *first* physical line of a multi-line log_event call
    (e.g. the "=== Outer CV results ===" header, or the to_string() header
    row) carries this prefix -- every subsequent line of that same
    log_event's multi-line message is written raw, with no prefix at all.
    So: only strip if the line actually starts with a timestamp; otherwise
    it's already a bare continuation line and splitting on '] ' would wrongly
    truncate content like a pipeline label ('Foo [eval=cross]')."""
    if not LOG_LINE_TS_RE.match(line):
        return line
    return line.split("] ", 1)[-1] if "] " in line else line


def parse_log(path: Path) -> tuple[list[dict], int, bool]:
    """Returns (ledger_rows, n_unmatched_candidate_lines, block_found)."""
    run_id = path.parent.name
    lines = path.read_text(errors="replace").splitlines()

    fallback_ts = None
    for line in lines:
        m = LOG_LINE_TS_RE.match(line)
        if m:
            fallback_ts = _to_iso(m.group(1))
            break

    effective_params: dict[str, dict] = {}
    fixed_overrides: dict = {}
    group_meta = {"eval_mode": None, "inner_group": None}
    for idx, line in enumerate(lines):
        m = EFFECTIVE_PARAMS_RE.search(line)
        if m:
            try:
                effective_params[m.group(1)] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                pass
            continue
        m = FIXED_OVERRIDES_RE.search(line)
        if m:
            try:
                fixed_overrides = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                pass
            continue
        m = STARTING_GROUP_RE.search(line)
        if m:
            group_meta = m.groupdict()

    try:
        start = next(
            i for i, line in enumerate(lines) if "=== Outer CV results ===" in line
        )
    except StopIteration:
        return [], 0, False  # never reached final_results -- crashed or interrupted

    row_ts_match = LOG_LINE_TS_RE.match(lines[start])
    logged_at = _to_iso(row_ts_match.group(1)) if row_ts_match else fallback_ts

    try:
        end = next(
            i
            for i, line in enumerate(lines[start + 1 :], start + 1)
            if "=== Per subject" in line
        )
    except StopIteration:
        end = len(lines)

    inner_group = group_meta["inner_group"]
    inner_group = None if inner_group in (None, "None") else inner_group
    eval_mode = group_meta["eval_mode"]
    group_id = (
        f"{run_id}__{eval_mode}__inner-{(inner_group or 'none').replace('_', '-')}"
        if eval_mode
        else None
    )

    rows = []
    unmatched = 0
    for line in lines[start + 2 : end]:  # skip the "===" marker + header row
        content = _strip_log_prefix(line)
        if not content.strip():
            continue
        m = OUTER_ROW_RE.match(content)
        if not m:
            unmatched += 1
            continue
        pipeline = m.group("pipeline")
        params = effective_params.get(pipeline, {})
        row = {
            "logged_at": logged_at,
            "run_id": run_id,
            "group_id": group_id,
            "eval_mode": eval_mode,
            "inner_group": inner_group,
            "pipeline": pipeline,
            "subject": m.group("subject"),
            "session": m.group("session"),
            "score": float(m.group("score")),
            "best_params": m.group("best_params"),
            "experiment_log_path": str(path),
            "effective_params_json": json.dumps(params, default=str),
            "fixed_overrides_json": json.dumps(fixed_overrides, default=str),
        }
        for key in RUN_LEDGER_CONFIG_KEYS:
            row[key] = params.get(key)
        rows.append(row)
    return rows, unmatched, True


def find_logs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.glob("*/experiment_*.log") if p.name != "experiment_latest.log"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root containing per-run-id directories (default: $MNE_DATA or ~/mne_data).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse and report but don't write."
    )
    parser.add_argument(
        "--ledger-path",
        default=None,
        help="Ledger CSV to append to / dedupe against (default: default_run_ledger_path()).",
    )
    args = parser.parse_args()

    data_root = Path(
        args.data_root or os.environ.get("MNE_DATA") or (Path.home() / "mne_data")
    ).expanduser()
    ledger_path = Path(args.ledger_path) if args.ledger_path else default_run_ledger_path()

    logs = find_logs(data_root)
    print(f"Found {len(logs)} experiment logs under {data_root}")

    # Idempotency: skip logs whose path is already recorded in the ledger,
    # so re-running this script (e.g. after new logs accumulate) only
    # ingests what's new instead of duplicating all 298+ historical rows.
    already_ingested = set()
    if ledger_path.exists():
        try:
            already_ingested = set(
                pd.read_csv(ledger_path, usecols=["experiment_log_path"])[
                    "experiment_log_path"
                ].dropna()
            )
        except (ValueError, pd.errors.EmptyDataError):
            pass
    if already_ingested:
        before = len(logs)
        logs = [p for p in logs if str(p) not in already_ingested]
        print(f"Skipping {before - len(logs)} logs already present in {ledger_path}")

    all_rows: list[dict] = []
    completed = 0
    crashed_or_interrupted = 0
    empty_block = []
    total_unmatched = 0
    for log_path in logs:
        rows, unmatched, block_found = parse_log(log_path)
        total_unmatched += unmatched
        if rows:
            completed += 1
            all_rows.extend(rows)
        elif block_found:
            empty_block.append(log_path)  # block found but 0 rows parsed -- suspicious
        else:
            crashed_or_interrupted += 1

    print(f"Completed runs (had a final Outer CV block): {completed}")
    print(f"Skipped (crashed/interrupted, no final Outer CV block): {crashed_or_interrupted}")
    print(f"Parsed {len(all_rows)} outer-CV rows total")
    if empty_block:
        print(
            f"WARNING: {len(empty_block)} logs had an Outer CV block but 0 rows "
            f"parsed from it -- likely a real parser bug, not a crashed run: "
            + ", ".join(str(p) for p in empty_block[:5])
        )
    if total_unmatched:
        print(
            f"WARNING: {total_unmatched} candidate result lines didn't match the "
            "expected format and were skipped (e.g. non-'None' best_params from "
            "an actual grid search) -- spot-check if this number looks large."
        )

    if args.dry_run:
        print("\n--dry-run: not writing. Sample rows:")
        with pd.option_context("display.max_columns", 8, "display.width", 160):
            print(pd.DataFrame(all_rows).head(10))
        return

    if not all_rows:
        print("\nNothing new to append.")
        return

    written_path = append_run_ledger_rows(all_rows, ledger_path=ledger_path)
    print(f"\nAppended {len(all_rows)} rows to {written_path}")


if __name__ == "__main__":
    main()
