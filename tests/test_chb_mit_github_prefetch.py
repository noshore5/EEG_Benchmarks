"""GitHub Release prefetch for CHB-MIT subjects.

Synthetic: no PhysioNet / GitHub network. Builds a tiny tar.gz, points the
downloader at it, and checks the S3-shaped MNE cache is filled (and that
summary-only calls do not pull the archive).

Generalized 2026-08-26 from a chb01-only mechanism to any subject in
GITHUB_RELEASE_SHA256 (or with a CHBMIT_CHB{:02d}_SHA256 override) -- see
chb_mit.py's module-level docstring on GITHUB_RELEASE_SHA256.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.epilepsy import chb_mit  # noqa: E402
from datasets.epilepsy.chb_mit import (  # noqa: E402
    BASE_URL,
    CHBMIT,
    parse_summary,
    subject_cache_complete,
    subject_cache_dir,
)


def _summary_for(subject: int) -> str:
    subject_dir = f"chb{subject:02d}"
    return (
        f"File Name: {subject_dir}_03.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure Start Time: 10 seconds\n"
        "Seizure End Time: 20 seconds\n"
        f"File Name: {subject_dir}_04.edf\n"
        "Number of Seizures in File: 0\n"
    )


SUMMARY = _summary_for(1)


def _tiny_archive(path: Path, subject: int = 1) -> Path:
    subject_dir = f"chb{subject:02d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in (
            (f"{subject_dir}/{subject_dir}-summary.txt", _summary_for(subject).encode()),
            (f"{subject_dir}/{subject_dir}_03.edf", b"EDF-FAKE-03"),
            (f"{subject_dir}/{subject_dir}_04.edf", b"EDF-FAKE-04"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


@pytest.fixture(autouse=True)
def _reset_prefetch_state():
    chb_mit._GITHUB_PREFETCH_FAILED.clear()
    yield
    chb_mit._GITHUB_PREFETCH_FAILED.clear()


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../evil.edf")
        payload = b"nope"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe path"):
        chb_mit.extract_subject_archive(archive, tmp_path / "out", 1)


def test_prefetch_fills_s3_cache_layout(tmp_path, monkeypatch):
    archive = _tiny_archive(tmp_path / "upstream" / "chb01.tar.gz")
    sha = chb_mit._sha256_file(archive)
    monkeypatch.setenv("CHBMIT_CHB01_ARCHIVE_URL", archive.as_uri())
    monkeypatch.setenv("CHBMIT_CHB01_SHA256", sha)

    dest_dir = subject_cache_dir(1, tmp_path)
    assert not dest_dir.exists()
    assert chb_mit.prefetch_subject_from_github(1, path=tmp_path) is True
    assert subject_cache_complete(dest_dir, 1)
    assert (dest_dir / "chb01_03.edf").read_bytes() == b"EDF-FAKE-03"
    # Archive is deleted after extract so a pod does not keep a spare 1GB.
    assert not (dest_dir.parents[2] / "github-releases" / "chb01.tar.gz").exists()


def test_prefetch_skipped_when_cache_complete(tmp_path, monkeypatch):
    dest_dir = subject_cache_dir(1, tmp_path)
    dest_dir.mkdir(parents=True)
    (dest_dir / "chb01-summary.txt").write_text(SUMMARY)
    (dest_dir / "chb01_03.edf").write_bytes(b"EDF-FAKE-03")
    (dest_dir / "chb01_04.edf").write_bytes(b"EDF-FAKE-04")

    def boom(*_args, **_kwargs):
        raise AssertionError("download_url should not run when cache is complete")

    monkeypatch.setattr(chb_mit, "download_url", boom)
    assert chb_mit.prefetch_subject_from_github(1, path=tmp_path) is True


def test_github_failure_falls_back_to_s3(tmp_path, monkeypatch):
    monkeypatch.setenv("CHBMIT_CHB01_ARCHIVE_URL", "http://127.0.0.1:1/missing.tar.gz")
    monkeypatch.setenv(
        "CHBMIT_CHB01_SHA256",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )

    fetched = []

    def fake_data_dl(url, sign, path=None, force_update=False, verbose=None):
        dest = chb_mit.cache_destination(url, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"from-s3:" + urlparse(url).path.encode())
        fetched.append(url)
        return str(dest)

    monkeypatch.setattr(chb_mit, "data_dl", fake_data_dl)
    out = CHBMIT()._local_file(1, "chb01_03.edf", path=tmp_path)
    assert out.is_file()
    assert out.read_bytes().startswith(b"from-s3:")
    assert fetched == [f"{BASE_URL}chb01/chb01_03.edf"]
    assert 1 in chb_mit._GITHUB_PREFETCH_FAILED


def test_local_file_uses_github_not_s3_on_miss(tmp_path, monkeypatch):
    archive = _tiny_archive(tmp_path / "upstream" / "chb01.tar.gz")
    monkeypatch.setenv("CHBMIT_CHB01_ARCHIVE_URL", archive.as_uri())
    monkeypatch.setenv("CHBMIT_CHB01_SHA256", chb_mit._sha256_file(archive))

    def boom(*_args, **_kwargs):
        raise AssertionError("S3 data_dl should not run after a GitHub hit")

    monkeypatch.setattr(chb_mit, "data_dl", boom)
    out = CHBMIT()._local_file(1, "chb01_03.edf", path=tmp_path)
    assert out.read_bytes() == b"EDF-FAKE-03"


def test_summary_only_does_not_prefetch_archive(tmp_path, monkeypatch):
    summary_url = f"{BASE_URL}chb01/chb01-summary.txt"

    def fake_data_dl(url, sign, path=None, force_update=False, verbose=None):
        dest = chb_mit.cache_destination(url, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(SUMMARY)
        return str(dest)

    monkeypatch.setattr(chb_mit, "data_dl", fake_data_dl)

    def boom(*_args, **_kwargs):
        raise AssertionError("summary-only lookup must not pull the archive")

    monkeypatch.setattr(chb_mit, "prefetch_subject_from_github", boom)
    records = CHBMIT().list_seizure_records(1, path=tmp_path)
    assert [r["filename"] for r in records] == ["chb01_03.edf"]
    assert parse_summary((chb_mit.cache_destination(summary_url, tmp_path)).read_text())


def test_unregistered_subject_does_not_use_github(tmp_path, monkeypatch):
    """Subject 5 has no entry in GITHUB_RELEASE_SHA256 (as of this writing;
    subjects 1-4 are registered, see the dict's own comment) and no env
    override -- must behave exactly as before the 2026-08-26
    generalization: straight to PhysioNet S3, no GitHub attempt at all."""
    assert chb_mit._github_release_parts(5) == []

    fetched = []

    def fake_data_dl(url, sign, path=None, force_update=False, verbose=None):
        dest = chb_mit.cache_destination(url, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"s5")
        fetched.append(url)
        return str(dest)

    monkeypatch.setattr(chb_mit, "data_dl", fake_data_dl)

    def boom(*_args, **_kwargs):
        raise AssertionError("subject 5 has no registered release -- must not attempt one")

    monkeypatch.setattr(chb_mit, "prefetch_subject_from_github", boom)
    out = CHBMIT()._local_file(5, "chb05_01.edf", path=tmp_path)
    assert out.read_bytes() == b"s5"
    assert fetched == [f"{BASE_URL}chb05/chb05_01.edf"]


def test_registered_subject_other_than_one_uses_github(tmp_path, monkeypatch):
    """A subject with a CHBMIT_CHB{:02d}_SHA256 override (simulating a
    registered release without editing GITHUB_RELEASE_SHA256 itself) is
    fetched from GitHub exactly like subject 1 is."""
    archive = _tiny_archive(tmp_path / "upstream" / "chb02.tar.gz", subject=2)
    monkeypatch.setenv("CHBMIT_CHB02_ARCHIVE_URL", archive.as_uri())
    monkeypatch.setenv("CHBMIT_CHB02_SHA256", chb_mit._sha256_file(archive))

    def boom(*_args, **_kwargs):
        raise AssertionError("S3 data_dl should not run after a GitHub hit")

    monkeypatch.setattr(chb_mit, "data_dl", boom)
    out = CHBMIT()._local_file(2, "chb02_03.edf", path=tmp_path)
    assert out.read_bytes() == b"EDF-FAKE-03"


def test_multi_part_subject_fetches_all_parts(tmp_path, monkeypatch):
    """A subject registered with a list of sha256s (chb04's real shape)
    fetches and extracts every part before the cache counts as complete --
    incomplete after only one part must not short-circuit as success."""
    subject_dir = "chb04"
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def _archive(name: str, files: dict) -> Path:
        path = upstream / name
        with tarfile.open(path, "w:gz") as tar:
            for fname, payload in files.items():
                info = tarfile.TarInfo(f"{subject_dir}/{fname}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        return path

    part1 = _archive(
        "chb04.part1.tar.gz",
        {"chb04-summary.txt": _summary_for(4).encode(), "chb04_03.edf": b"EDF-FAKE-03"},
    )
    part2 = _archive("chb04.part2.tar.gz", {"chb04_04.edf": b"EDF-FAKE-04"})
    sha1, sha2 = chb_mit._sha256_file(part1), chb_mit._sha256_file(part2)

    monkeypatch.setattr(chb_mit, "GITHUB_RELEASE_SHA256", {**chb_mit.GITHUB_RELEASE_SHA256, 4: [sha1, sha2]})
    urls = {1: part1.as_uri(), 2: part2.as_uri()}
    monkeypatch.setattr(
        chb_mit,
        "_github_archive_part_url",
        lambda subject, part_index, n_parts: urls[part_index + 1],
    )

    dest_dir = subject_cache_dir(4, tmp_path)
    assert chb_mit.prefetch_subject_from_github(4, path=tmp_path) is True
    assert subject_cache_complete(dest_dir, 4)
    assert (dest_dir / "chb04_03.edf").read_bytes() == b"EDF-FAKE-03"
    assert (dest_dir / "chb04_04.edf").read_bytes() == b"EDF-FAKE-04"


def test_multi_part_subject_incomplete_after_one_part_falls_back(tmp_path, monkeypatch):
    """If only the first of N parts is reachable, the subject must NOT be
    reported complete/successful -- prefetch should fail closed so the
    caller falls back to PhysioNet S3, not silently serve a partial cache."""
    subject_dir = "chb04"
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    part1 = upstream / "chb04.part1.tar.gz"
    with tarfile.open(part1, "w:gz") as tar:
        for fname, payload in (
            ("chb04-summary.txt", _summary_for(4).encode()),
            ("chb04_03.edf", b"EDF-FAKE-03"),
        ):
            info = tarfile.TarInfo(f"{subject_dir}/{fname}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    sha1 = chb_mit._sha256_file(part1)

    monkeypatch.setattr(chb_mit, "GITHUB_RELEASE_SHA256", {**chb_mit.GITHUB_RELEASE_SHA256, 4: [sha1, "0" * 64]})
    monkeypatch.setattr(
        chb_mit,
        "_github_archive_part_url",
        lambda subject, part_index, n_parts: part1.as_uri() if part_index == 0 else "http://127.0.0.1:1/missing.tar.gz",
    )

    assert chb_mit.prefetch_subject_from_github(4, path=tmp_path) is False
    assert 4 in chb_mit._GITHUB_PREFETCH_FAILED
