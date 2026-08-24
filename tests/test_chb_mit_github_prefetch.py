"""GitHub Release prefetch for CHB-MIT subject 1.

Synthetic: no PhysioNet / GitHub network. Builds a tiny tar.gz, points the
downloader at it, and checks the S3-shaped MNE cache is filled (and that
summary-only calls do not pull the archive).
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
    chb01_cache_complete,
    chb01_cache_dir,
    extract_chb01_archive,
    parse_summary,
)


SUMMARY = (
    "File Name: chb01_03.edf\n"
    "Number of Seizures in File: 1\n"
    "Seizure Start Time: 10 seconds\n"
    "Seizure End Time: 20 seconds\n"
    "File Name: chb01_04.edf\n"
    "Number of Seizures in File: 0\n"
)


def _tiny_archive(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in (
            ("chb01/chb01-summary.txt", SUMMARY.encode()),
            ("chb01/chb01_03.edf", b"EDF-FAKE-03"),
            ("chb01/chb01_04.edf", b"EDF-FAKE-04"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


@pytest.fixture(autouse=True)
def _reset_prefetch_flag():
    chb_mit._CHB01_PREFETCH_FAILED = False
    yield
    chb_mit._CHB01_PREFETCH_FAILED = False


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../evil.edf")
        payload = b"nope"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe path"):
        extract_chb01_archive(archive, tmp_path / "out")


def test_prefetch_fills_s3_cache_layout(tmp_path, monkeypatch):
    archive = _tiny_archive(tmp_path / "upstream" / "chb01.tar.gz")
    sha = chb_mit._sha256_file(archive)
    monkeypatch.setenv("CHBMIT_CHB01_ARCHIVE_URL", archive.as_uri())
    monkeypatch.setenv("CHBMIT_CHB01_SHA256", sha)

    dest_dir = chb01_cache_dir(tmp_path)
    assert not dest_dir.exists()
    assert chb_mit.prefetch_chb01_from_github(path=tmp_path) is True
    assert chb01_cache_complete(dest_dir)
    assert (dest_dir / "chb01_03.edf").read_bytes() == b"EDF-FAKE-03"
    # Archive is deleted after extract so a pod does not keep a spare 1GB.
    assert not (dest_dir.parents[2] / "github-releases" / "chb01.tar.gz").exists()


def test_prefetch_skipped_when_cache_complete(tmp_path, monkeypatch):
    dest_dir = chb01_cache_dir(tmp_path)
    dest_dir.mkdir(parents=True)
    (dest_dir / "chb01-summary.txt").write_text(SUMMARY)
    (dest_dir / "chb01_03.edf").write_bytes(b"EDF-FAKE-03")
    (dest_dir / "chb01_04.edf").write_bytes(b"EDF-FAKE-04")

    def boom(*_args, **_kwargs):
        raise AssertionError("download_url should not run when cache is complete")

    monkeypatch.setattr(chb_mit, "download_url", boom)
    assert chb_mit.prefetch_chb01_from_github(path=tmp_path) is True


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
    assert chb_mit._CHB01_PREFETCH_FAILED is True


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
        raise AssertionError("summary-only lookup must not pull the 1GB archive")

    monkeypatch.setattr(chb_mit, "prefetch_chb01_from_github", boom)
    records = CHBMIT().list_seizure_records(1, path=tmp_path)
    assert [r["filename"] for r in records] == ["chb01_03.edf"]
    assert parse_summary((chb_mit.cache_destination(summary_url, tmp_path)).read_text())


def test_other_subjects_do_not_use_github(tmp_path, monkeypatch):
    fetched = []

    def fake_data_dl(url, sign, path=None, force_update=False, verbose=None):
        dest = chb_mit.cache_destination(url, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"s2")
        fetched.append(url)
        return str(dest)

    monkeypatch.setattr(chb_mit, "data_dl", fake_data_dl)

    def boom(*_args, **_kwargs):
        raise AssertionError("subject 2 must not fetch the chb01 archive")

    monkeypatch.setattr(chb_mit, "prefetch_chb01_from_github", boom)
    out = CHBMIT()._local_file(2, "chb02_01.edf", path=tmp_path)
    assert out.read_bytes() == b"s2"
    assert fetched == [f"{BASE_URL}chb02/chb02_01.edf"]
