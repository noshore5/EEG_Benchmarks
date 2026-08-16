"""CHB-MIT Scalp EEG Database.

https://physionet.org/content/chbmit/1.0.0/

Scalp EEG recordings from pediatric subjects with intractable seizures,
collected at Boston Children's Hospital. Recordings are continuous (mostly
1-hour .edf files per subject) rather than trial-locked, and each subject
has a companion ``chbXX-summary.txt`` file documenting, per recording, how
many seizures it contains and their onset/offset (in seconds from the start
of that recording).

Files are plain .edf, read directly with :func:`mne.io.read_raw_edf` -- no
custom binary parser needed. The dataset-specific work is: (1) discovering
which files exist for a subject (the summary file is the only listing --
filenames aren't contiguously numbered, e.g. chb01 skips 28, 35, 44, 45),
and (2) parsing the summary into :class:`mne.Annotations` attached to each
Raw.

See :mod:`paradigms.continuous_labeling` for the paradigm that turns this
into windowed ``(X, y)`` arrays -- MOABB's own paradigms all assume
fixed-length, event-locked trials, which doesn't fit seizure data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import mne
from moabb.datasets.base import BaseDataset
from moabb.datasets.download import data_dl, get_dataset_path
from moabb.datasets.preprocessing import FixedPipeline


log = logging.getLogger(__name__)

# 2026-08-16: PhysioNet's own HTTPS server throttles to ~180KB/s per
# connection regardless of local link speed (confirmed: the same
# connection hit ~3.4MB/s against a CDN and ~3.9MB/s against this mirror) --
# not fixable from the client side. PhysioNet also publishes this exact
# dataset on a public, no-auth-required S3 bucket (see the "AWS S3" bulk-
# download option on https://physionet.org/content/chbmit/1.0.0/); verified
# byte-identical (sha256) against a file already downloaded from the
# physionet.org URL before switching. ~20x faster in practice.
BASE_URL = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/"
SIGN = "CHBMIT"

_FILE_RE = re.compile(r"File Name:\s*(\S+)")
# Single-seizure files write "Seizure Start Time: N seconds"; files with
# more than one seizure number them "Seizure 1 Start Time: ...", "Seizure 2
# Start Time: ...". This matches both.
_ONSET_RE = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds")
_OFFSET_RE = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds")


def parse_summary(text: str) -> List[dict]:
    """Parse a ``chbXX-summary.txt`` file into a list of per-recording records.

    Parameters
    ----------
    text : str
        Contents of the summary file.

    Returns
    -------
    records : list of dict
        One dict per recording, in file order, with keys:

        - ``filename`` : str, e.g. ``"chb01_03.edf"``
        - ``seizures`` : list of (onset_sec, offset_sec) float tuples,
          relative to the start of that recording. Empty if the recording
          has no documented seizures.
    """
    file_matches = list(_FILE_RE.finditer(text))
    records = []
    for i, match in enumerate(file_matches):
        block_start = match.end()
        block_end = (
            file_matches[i + 1].start() if i + 1 < len(file_matches) else len(text)
        )
        block = text[block_start:block_end]
        onsets = [float(x) for x in _ONSET_RE.findall(block)]
        offsets = [float(x) for x in _OFFSET_RE.findall(block)]
        if len(onsets) != len(offsets):
            log.warning(
                "Mismatched seizure onset/offset count for %s: %d onsets, %d offsets",
                match.group(1),
                len(onsets),
                len(offsets),
            )
        records.append(
            {"filename": match.group(1), "seizures": list(zip(onsets, offsets))}
        )
    return records


class CHBMIT(BaseDataset):
    """CHB-MIT Scalp EEG Database.

    Parameters
    ----------
    records : dict of int -> list of str | None
        Optional filter restricting which recordings are used per subject,
        e.g. ``{1: ["chb01_03.edf", "chb01_04.edf"]}``. If a subject has no
        entry (or ``records`` is None), all recordings listed in that
        subject's summary file are used. Mainly useful to avoid downloading
        an entire subject (~40 one-hour files) when only a few recordings
        are needed, e.g. for testing against :meth:`list_seizure_records`.
    """

    def __init__(self, records: Optional[Dict[int, List[str]]] = None):
        super().__init__(
            subjects=list(range(1, 25)),
            sessions_per_subject=1,
            events={"seizure": 1},
            code="CHBMIT",
            # Not a fixed per-trial interval -- CHB-MIT recordings run
            # continuously for ~1 hour each. Unused: _create_process_pipeline
            # is overridden below so BaseDataset never builds an
            # interval-based epoching step out of it.
            interval=[0, 1],
            paradigm="epilepsy",
            doi="10.13026/C2K01R",
        )
        self._records_filter = records or {}

    def _create_process_pipeline(self):
        # The inherited default (SetRawAnnotations) rewrites every
        # annotation's duration to a single fixed `interval[1] - interval[0]`
        # span -- correct for MOABB's fixed-length trials, but it would
        # silently overwrite each seizure's real (variable) duration with a
        # constant. _get_single_subject_data already attaches correct
        # mne.Annotations, so skip further processing entirely.
        return FixedPipeline([])

    @staticmethod
    def _subject_dir(subject: int) -> str:
        return f"chb{subject:02d}"

    def _summary_url(self, subject: int) -> str:
        subject_dir = self._subject_dir(subject)
        return f"{BASE_URL}{subject_dir}/{subject_dir}-summary.txt"

    def _record_url(self, subject: int, filename: str) -> str:
        return f"{BASE_URL}{self._subject_dir(subject)}/{filename}"

    def _summary_path(
        self,
        subject: int,
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
    ) -> Path:
        return Path(data_dl(self._summary_url(subject), SIGN, path, force_update))

    def _list_records(
        self,
        subject: int,
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
    ) -> List[dict]:
        """Download (if needed) and parse the subject's summary file.

        Applies the ``records`` filter passed to ``__init__``, if any.
        """
        summary_path = self._summary_path(subject, path, force_update)
        records = parse_summary(summary_path.read_text())
        wanted = self._records_filter.get(subject)
        if wanted is not None:
            records = [r for r in records if r["filename"] in wanted]
        return records

    def list_records(
        self,
        subject: int,
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
    ) -> List[dict]:
        """Return every recording documented for `subject` (seizure-containing
        and seizure-free alike). Cheap to call on its own (only downloads
        the summary file) -- public counterpart to list_seizure_records for
        callers that need the seizure-free entries too (e.g. label_mode=
        "prediction"'s need for genuine interictal recordings; see
        Epilepsy/run_pipelines.py's _build_windowed_dataset)."""
        return self._list_records(subject, path, force_update)

    def list_seizure_records(
        self,
        subject: int,
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
    ) -> List[dict]:
        """Return only the recordings that contain at least one documented seizure.

        Cheap to call on its own (only downloads the summary file), useful
        for building a ``records`` filter that skips the seizure-free
        recordings for a subject.
        """
        return [
            r
            for r in self._list_records(subject, path, force_update)
            if r["seizures"]
        ]

    def data_path(
        self,
        subject: int,
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        verbose=None,
    ) -> List[Path]:
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        get_dataset_path(SIGN, path)
        records = self._list_records(subject, path, force_update)
        paths = [self._summary_path(subject, path, force_update)]
        for record in records:
            paths.append(
                Path(
                    data_dl(
                        self._record_url(subject, record["filename"]),
                        SIGN,
                        path,
                        force_update,
                    )
                )
            )
        return paths

    def _get_single_subject_data(
        self, subject: int
    ) -> Dict[str, Dict[str, "mne.io.BaseRaw"]]:
        records = self._list_records(subject)

        runs = {}
        for idx, record in enumerate(records, start=1):
            edf_path = data_dl(self._record_url(subject, record["filename"]), SIGN)
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

            onsets = [onset for onset, _ in record["seizures"]]
            durations = [offset - onset for onset, offset in record["seizures"]]
            descriptions = ["seizure"] * len(onsets)
            seizure_annotations = mne.Annotations(
                onset=onsets, duration=durations, description=descriptions
            )
            raw.set_annotations(raw.annotations + seizure_annotations)

            runs[f"{idx:02d}"] = raw

        # CHB-MIT is one continuous multi-day monitoring session per subject,
        # split across many recordings (runs); there's no natural multi-day
        # "session" boundary in the summary file, so we use a single session.
        return {"0": runs}
