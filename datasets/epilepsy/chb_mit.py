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

import hashlib
import logging
import os
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import quote, urlparse

import mne
from mne.utils import _url_to_local_path
from moabb.datasets.base import BaseDataset
from moabb.datasets.download import data_dl, get_dataset_path
try:
    from moabb.datasets.preprocessing import FixedPipeline
except ImportError:  # moabb < 1.3 (no FixedPipeline); identity is enough here
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    class FixedPipeline(Pipeline):
        def __init__(self, steps=None):
            super().__init__(list(steps) if steps else [("identity", FunctionTransformer())])


log = logging.getLogger(__name__)

# 2026-08-16: PhysioNet's own HTTPS server throttles to ~180KB/s per
# connection regardless of local link speed (confirmed: the same
# connection hit ~3.4MB/s against a CDN and ~3.9MB/s against this mirror) --
# not fixable from the client side. PhysioNet also publishes this exact
# dataset on a public, no-auth-required S3 bucket (see the "AWS S3" bulk-
# download option on https://physionet.org/content/chbmit/1.0.0/); verified
# byte-identical (sha256) against a file already downloaded from the
# physionet.org URL before switching. ~20x faster in practice.
#
# 2026-08-24: that S3 mirror is still the per-file fallback (and the only
# source for subjects other than chb01), but a full chb01 fetch (~42 EDFs,
# ~1.6GB) is the default pipeline's first-run bottleneck. Subject 1 is
# redistributed as a single GitHub Release archive (ODC-By 1.0; see
# THIRD_PARTY_NOTICES.md) and extracted into the same MNE cache layout
# data_dl would have used for the S3 URLs, so an existing S3 cache is
# reused and GitHub-populated files still satisfy later data_dl lookups.
BASE_URL = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/"
SIGN = "CHBMIT"

# Subjects with a known GitHub Release mirror, keyed by subject number ->
# sha256 of that subject's chbXX.tar.gz (tag "chbmit-chbXX-1.0.0", same
# ODC-By 1.0 provenance as chb01 -- see THIRD_PARTY_NOTICES.md). A subject
# NOT in this dict falls through to PhysioNet S3 exactly as if this
# registry didn't exist -- adding a subject here is the only thing that
# turns GitHub-first fetching on for it (originally chb01-only, 2026-08-24;
# generalized to any subject 2026-08-26).
#
# Value is either a single sha256 string (one asset, "chbXX.tar.gz") or a
# list of sha256 strings for a subject split across multiple assets
# ("chbXX.partN.tar.xz", 1-indexed) -- GitHub Releases caps a single asset
# at 2 GiB, which chb04's ~6.4GB raw recordings don't fit in one gzipped
# archive. All parts share one release/tag and extract into the same
# subject directory (they're disjoint file sets, so extraction order
# doesn't matter); the subject's cache only counts as complete once every
# part has landed.
GITHUB_RELEASE_SHA256 = {
    1: "bf91e579c8b61a6813442d9351fa6e111dd6078d43ab2b04fd66d4660324b6f9",
    2: "d0985a0213815c235e320222552a1dba35455225fcc2a2d1cf03130025cd12d6",
    3: "40ebd8e66a8d4ed24cb70b569dd33e1f64dd6aaacd8fb502c8b93ab29ea6219b",
    # chb04's ~6.4GB raw recordings don't fit GitHub's 2GiB-per-asset cap
    # as one gzipped archive (confirmed: gzip got it to 3.7GB) -- split
    # into two xz-compressed, size-balanced parts instead (~1.6-1.7GB
    # each). See GITHUB_RELEASE_SHA256's own docstring above.
    4: [
        "1d7d02d78fea2708af36d8ceab66a8649773e6d934bf53adc4b7b90914dd3d6c",
        "4be4083432f4a4dfefbf24696e9ec5f565fd6a7afdc406c27b811b1c51ac74df",
    ],
}

# One failed GitHub prefetch per subject per process: don't retry on every
# missing EDF within the same run.
_GITHUB_PREFETCH_FAILED: set = set()

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


def _github_release_tag(subject: int) -> str:
    return f"chbmit-chb{subject:02d}-1.0.0"


def _github_release_parts(subject: int) -> List[str]:
    """sha256 list for this subject's archive part(s), in order.

    Empty list means "no registered release" (caller falls back to
    PhysioNet S3). The CHBMIT_CHB{:02d}_SHA256 env override always
    describes a single archive -- it's meant for tests / a one-off rebuilt
    archive, not for reproducing a multi-part split.
    """
    env = os.environ.get(f"CHBMIT_CHB{subject:02d}_SHA256")
    if env:
        return [env]
    value = GITHUB_RELEASE_SHA256.get(subject)
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _github_archive_part_url(subject: int, part_index: int, n_parts: int) -> str:
    """URL for the ``part_index``-th (0-based) of ``n_parts`` archive(s).

    n_parts == 1 -> ``chbXX.tar.gz`` (the common case, matches chb01).
    n_parts > 1 -> ``chbXX.partN.tar.xz`` (1-indexed N), the same asset
    names uploaded to that subject's release when it needed a split.
    """
    env = os.environ.get(f"CHBMIT_CHB{subject:02d}_ARCHIVE_URL")
    if env:
        return env
    tag = _github_release_tag(subject)
    fname = (
        f"chb{subject:02d}.tar.gz"
        if n_parts == 1
        else f"chb{subject:02d}.part{part_index + 1}.tar.xz"
    )
    return f"https://github.com/noshore5/EEG_Benchmarks/releases/download/{tag}/{fname}"


def cache_destination(url: str, path: Optional[Union[str, Path]] = None) -> Path:
    """Local path ``data_dl`` would use for ``url`` without downloading."""
    root = Path(get_dataset_path(SIGN, path)) / f"MNE-{SIGN.lower()}-data"
    return Path(_url_to_local_path(url, str(root)))


def subject_cache_dir(subject: int, path: Optional[Union[str, Path]] = None) -> Path:
    subject_dir = f"chb{subject:02d}"
    return cache_destination(f"{BASE_URL}{subject_dir}/{subject_dir}-summary.txt", path).parent


def subject_cache_complete(dest_dir: Path, subject: int) -> bool:
    """True if dest_dir has the subject's summary and every EDF it lists."""
    summary = dest_dir / f"chb{subject:02d}-summary.txt"
    if not summary.is_file():
        return False
    try:
        records = parse_summary(summary.read_text())
    except OSError:
        return False
    if not records:
        return False
    return all((dest_dir / rec["filename"]).is_file() for rec in records)


# Back-compat aliases (subject-1-only names predate the 2026-08-26
# generalization to any subject; kept since they're still the clearest
# spelling when subject is always 1, e.g. in the RunPod Dockerfile's docs).
def chb01_cache_dir(path: Optional[Union[str, Path]] = None) -> Path:
    return subject_cache_dir(1, path)


def chb01_cache_complete(dest_dir: Path) -> bool:
    return subject_cache_complete(dest_dir, 1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.lower()
    if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return value
    return None


def download_url(url: str, dest: Path, sha256: Optional[str] = None) -> Path:
    """Fetch ``url`` to ``dest``. Optional sha256 is hex, checked after write."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha256 = _valid_sha256(sha256)
    if dest.is_file() and (sha256 is None or _sha256_file(dest) == sha256):
        return dest

    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()

    log.info("Downloading %s -> %s", url, dest)
    parsed = urlparse(url)
    if parsed.scheme == "file":
        # unquote(parsed.path) alone is wrong on Windows: urlparse() leaves
        # a leading "/" before the drive letter ("/C:/Users/...") that
        # shutil.copy2 can't open. url2pathname strips that correctly on
        # every platform (same helper MNE's _url_to_local_path uses).
        shutil.copy2(urllib.request.url2pathname(parsed.path), tmp)
    else:
        try:
            from pooch import retrieve

            known_hash = f"sha256:{sha256}" if sha256 else None
            got = retrieve(
                url,
                known_hash=known_hash,
                fname=tmp.name,
                path=str(tmp.parent),
                progressbar=True,
            )
            if Path(got) != tmp:
                Path(got).replace(tmp)
        except ImportError:
            request = urllib.request.Request(
                url, headers={"User-Agent": "EEG_Benchmarks/chbmit"}
            )
            with urllib.request.urlopen(request, timeout=3600) as resp, tmp.open(
                "wb"
            ) as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
    if sha256 is not None and _sha256_file(tmp) != sha256:
        tmp.unlink()
        raise ValueError(f"sha256 mismatch for {url}")
    tmp.replace(dest)
    return dest


def _safe_tar_members(tar: tarfile.TarFile):
    for member in tar.getmembers():
        parts = Path(member.name).parts
        if member.name.startswith("/") or ".." in parts:
            raise ValueError(f"Refusing tar member with unsafe path: {member.name}")
        yield member


def extract_subject_archive(archive: Path, extract_root: Path, subject: int) -> Path:
    """Extract ``chbXX/...`` members of ``archive`` into ``extract_root``.

    ``extract_root`` is the PhysioNet version directory (``.../1.0.0``), so
    members named ``chbXX/chbXX_01.edf`` land at the S3 cache path.

    Does NOT check ``subject_cache_complete`` itself -- a multi-part
    subject (see GITHUB_RELEASE_SHA256) is legitimately incomplete after
    any one part. ``prefetch_subject_from_github`` checks completeness
    once, after every part has extracted.
    """
    subject_dir = f"chb{subject:02d}"
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = list(_safe_tar_members(tar))
        try:
            tar.extractall(extract_root, members=members, filter="data")
        except TypeError:
            tar.extractall(extract_root, members=members)
    return extract_root / subject_dir


def extract_chb01_archive(archive: Path, extract_root: Path) -> Path:
    return extract_subject_archive(archive, extract_root, 1)


def prefetch_subject_from_github(
    subject: int,
    path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> bool:
    """Populate one subject's MNE cache from its GitHub Release archive(s).

    Returns True if the cache is complete afterwards. Returns False if the
    subject has no known release (not in GITHUB_RELEASE_SHA256 / no
    CHBMIT_CHB{:02d}_SHA256 override) or an archive could not be fetched
    (caller should fall back to PhysioNet S3 either way). Summary-only
    lookups should not call this -- it pulls up to a few GB. A subject with
    multiple parts (see GITHUB_RELEASE_SHA256) fetches+extracts each in
    turn before the completeness check runs.
    """
    parts = _github_release_parts(subject)
    if not parts:
        return False
    dest_dir = subject_cache_dir(subject, path)
    if not force and subject_cache_complete(dest_dir, subject):
        return True
    if subject in _GITHUB_PREFETCH_FAILED and not force:
        return False
    if force:
        _GITHUB_PREFETCH_FAILED.discard(subject)

    extract_root = dest_dir.parent
    n_parts = len(parts)
    for part_index, sha256 in enumerate(parts):
        url = _github_archive_part_url(subject, part_index, n_parts)
        archive_name = Path(urlparse(url).path).name or f"chb{subject:02d}.tar.gz"
        archive = (
            Path(get_dataset_path(SIGN, path))
            / f"MNE-{SIGN.lower()}-data"
            / "github-releases"
            / archive_name
        )
        try:
            download_url(url, archive, sha256=sha256)
            extract_subject_archive(archive, extract_root, subject)
        except Exception as exc:
            _GITHUB_PREFETCH_FAILED.add(subject)
            log.warning(
                "GitHub chb%02d archive (part %d/%d) failed (%s); falling back to PhysioNet S3",
                subject,
                part_index + 1,
                n_parts,
                exc,
            )
            return False
        finally:
            if archive.is_file():
                try:
                    archive.unlink()
                except OSError:
                    pass
    if not subject_cache_complete(dest_dir, subject):
        _GITHUB_PREFETCH_FAILED.add(subject)
        log.warning(
            "GitHub chb%02d: all %d part(s) extracted but cache is still incomplete; "
            "falling back to PhysioNet S3",
            subject,
            n_parts,
        )
        return False
    _GITHUB_PREFETCH_FAILED.discard(subject)
    return True


def prefetch_chb01_from_github(
    path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> bool:
    return prefetch_subject_from_github(1, path=path, force=force)


class CHBMIT(BaseDataset):
    """CHB-MIT Scalp EEG Database.

    Parameters
    ----------
    records : dict of int -> list of str | None
        Optional filter restricting which recordings are used per subject,
        e.g. ``{1: ["chb01_03.edf", "chb01_04.edf"]}``. If a subject has no
        entry (or ``records`` is None), all recordings listed in that
        subject's summary file are used. For a subject with no registered
        GitHub release (see ``GITHUB_RELEASE_SHA256``) this still skips
        unused S3 downloads. A subject that IS registered is filled from
        one GitHub archive, so a filter no longer saves the first-run fetch
        for that subject specifically.
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
        # quote() is required for filenames the S3 mirror won't serve
        # unescaped -- e.g. chb02's "chb02_16+.edf" 404s as a literal "+"
        # but 200s as "%2B". _url_to_local_path (via request.url2pathname)
        # unquotes the path again when deriving the local cache filename,
        # so the file still lands on disk as "chb02_16+.edf", unchanged.
        return f"{BASE_URL}{self._subject_dir(subject)}/{quote(filename)}"

    def _local_file(
        self,
        subject: int,
        filename: str,
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
    ) -> Path:
        """Return the cached path for one CHB-MIT file, downloading if needed.

        EDFs (not the tiny summary) for a subject registered in
        GITHUB_RELEASE_SHA256 are filled from that subject's GitHub Release
        archive when the S3-shaped cache is incomplete. Unregistered
        subjects, and any GitHub failure, use PhysioNet S3 via ``data_dl``.
        """
        url = self._record_url(subject, filename)
        dest = cache_destination(url, path)
        if dest.is_file() and not force_update:
            return dest
        is_edf = filename.lower().endswith(".edf")
        if is_edf and _github_release_parts(subject):
            prefetch_subject_from_github(subject, path=path, force=force_update)
            if dest.is_file():
                return dest
        return Path(data_dl(url, SIGN, path, force_update))

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
                self._local_file(
                    subject, record["filename"], path=path, force_update=force_update
                )
            )
        return paths

    def _get_single_subject_data(
        self, subject: int
    ) -> Dict[str, Dict[str, "mne.io.BaseRaw"]]:
        records = self._list_records(subject)

        runs = {}
        for idx, record in enumerate(records, start=1):
            edf_path = self._local_file(subject, record["filename"])
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
