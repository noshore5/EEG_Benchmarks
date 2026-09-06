"""Run the container's inference path on a local CHB-MIT EDF, without Docker.

    .venv/bin/python Epilepsy/szcore/test_local.py <chbXX_YY.edf> [out.tsv]

Also scores the result against the recording's summary-file seizures using
`timescoring` (event-based) when available.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algo.infer import run  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    edf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(edf).with_suffix(".hyp.tsv"))
    run(edf, out)
    print(f"wrote {out}\n")
    print(Path(out).read_text())


if __name__ == "__main__":
    main()
