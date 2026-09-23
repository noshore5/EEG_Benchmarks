"""One-off downloader for the Kuhlmann NeuroVista (NV) seizure-prediction
contest data, shared via a Dropbox folder link (gated by a DUA the user
agreed to -- see CONTEXT.md / session notes before redistributing anything
pulled by this script).

Uses the Dropbox API directly against the shared-link folder (not the
user's own Dropbox root -- this folder was shared as a link, not mounted),
since `dbx-downloader`/browser download of a 27GB zip was unreliable.
Requires DROPBOX_APP_KEY/DROPBOX_APP_SECRET/DROPBOX_REFRESH_TOKEN in .env
(2026-09-22: switched from a hand-pasted DROPBOX_ACCESS_TOKEN, which is
short-lived (~4h) and was expiring mid-session -- see chat history same
date for the one-time oauth2/authorize flow that produced the refresh
token). get_access_token() below exchanges the refresh token for a fresh
~4h access token at process startup; for a run longer than that, rerun
get_access_token() rather than caching -- not currently needed since even
the full 3-patient download finishes well inside one token's lifetime.

Usage:
    python scripts/download_kuhlmann_nv.py Pat1Train
    python scripts/download_kuhlmann_nv.py Pat1Train --limit 20   # smoke test
    python scripts/download_kuhlmann_nv.py Pat1Test Pat2Train ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

SHARED_LINK_URL = (
    "https://www.dropbox.com/scl/fo/jy001qlrne8gqkalbp5e8/"
    "AAs05zU9ji1aWqJBPSmoAjs?rlkey=uboofbzlvwq5eyukdf9gpeq62&dl=0"
)
DEST_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "epilepsy" / "kuhlmann_nv"


def get_access_token() -> str:
    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not (app_key and app_secret and refresh_token):
        # Fall back to a hand-pasted short-lived token if the refresh flow
        # hasn't been set up yet in this checkout.
        token = os.environ.get("DROPBOX_ACCESS_TOKEN")
        if token:
            return token
        sys.exit(
            "Need either DROPBOX_REFRESH_TOKEN (+ APP_KEY/APP_SECRET) or "
            "DROPBOX_ACCESS_TOKEN in .env."
        )
    resp = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": app_key,
            "client_secret": app_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def list_folder(token: str, subfolder: str) -> list[dict]:
    entries: list[dict] = []
    resp = requests.post(
        "https://api.dropboxapi.com/2/files/list_folder",
        headers={"Authorization": f"Bearer {token}"},
        json={"path": f"/{subfolder}", "shared_link": {"url": SHARED_LINK_URL}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    entries.extend(data["entries"])
    while data.get("has_more"):
        resp = requests.post(
            "https://api.dropboxapi.com/2/files/list_folder/continue",
            headers={"Authorization": f"Bearer {token}"},
            json={"cursor": data["cursor"]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        entries.extend(data["entries"])
    return [e for e in entries if e[".tag"] == "file"]


def download_one(token: str, subfolder: str, name: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return  # resume-friendly: skip already-downloaded files
    # files/download rejects shared_link+path combos here with 409 path/
    # not_found (confirmed empirically 2026-09-22) -- sharing/get_shared_
    # link_file is the endpoint that actually works for a file living
    # inside a shared *folder* link (as opposed to a link pointing straight
    # at one file), addressed by its path relative to the shared folder.
    api_arg = {
        "url": SHARED_LINK_URL,
        "path": f"/{subfolder}/{name}",
    }
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://content.dropboxapi.com/2/sharing/get_shared_link_file",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Dropbox-API-Arg": __import__("json").dumps(api_arg),
                },
                stream=True,
                timeout=(30, 300),  # (connect, read) -- files run several MB, a
                # stalled read can legitimately need more than 120s total.
            )
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subfolders", nargs="+", help="e.g. Pat1Train Pat1Test")
    parser.add_argument("--limit", type=int, default=None, help="cap files per subfolder (smoke test)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token = get_access_token()

    for subfolder in args.subfolders:
        dest_dir = DEST_ROOT / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{subfolder}] listing...", flush=True)
        entries = list_folder(token, subfolder)
        if args.limit:
            entries = entries[: args.limit]
        total = len(entries)
        print(f"[{subfolder}] {total} files to fetch", flush=True)
        t0 = time.time()
        for i, e in enumerate(entries, 1):
            dest = dest_dir / e["name"]
            try:
                download_one(token, subfolder, e["name"], dest)
            except requests.HTTPError as exc:
                print(f"[{subfolder}] FAILED {e['name']}: {exc}", flush=True)
                continue
            if i % 10 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"[{subfolder}] {i}/{total} ({elapsed:.0f}s elapsed)", flush=True)
        print(f"[{subfolder}] done.", flush=True)


if __name__ == "__main__":
    main()
