# chb02/chb03 GRU baseline extension + GitHub-release mirror generalized to any subject (2026-08-26)

Branch: `continuous-cwt-mamba`, working tree **uncommitted** as of this
writing (`git status` shows `datasets/epilepsy/chb_mit.py`,
`tests/test_chb_mit_github_prefetch.py`, `README.md`,
`THIRD_PARTY_NOTICES.md` modified, plus result CSVs/logs and two scratch
fetch scripts untracked) -- next session should `git status`/`git diff`
rather than assume any of this landed. Separate thread from the
continuous-cwt-mamba paradigm work described elsewhere in `CONTEXT.md`;
nothing in that thread was touched this session.

Machine: same local Windows box, RTX 3070 Ti, as every other session note.

---

## Part 1 -- matched dense-edge-GRU protocol extended to chb02 and chb03

Same command/protocol as the chb01 baseline
(`Epilepsy/Session_notes/2026_08_25/full_6fold_23ch_encoderfree_val_gru.md`),
just `--subjects 2` / `--subjects 3`:

```
python Epilepsy/run_pipelines.py --subjects N --device cuda --validation-split 0.2 \
    --dense-edge-amp-bf16 --train-amp-bf16
```

Neither subject was cached locally before this session -- both downloaded
from PhysioNet's S3 mirror on first use (chb01 is the only subject baked
into any image/release before this session; see Part 2).

### chb02: total collapse, not a partial miss

| seizure | n_test (pre) | precision | recall | f1 | AP | AUC | hit (raw/sm) |
|---|---|---|---|---|---|---|---|
| `2_17_0`* | 2070 (30) | 0.0 | 0.0 | 0.0 | 0.015 | 0.472 | F / F |
| `2_20_0`* | 1962 (30) | 0.0 | 0.0 | 0.0 | 0.017 | 0.524 | F / F |

Mean: accuracy 0.985, precision/recall/f1 all **0.0**, AP **0.016**, AUC
**0.498** (chance), hit rate **0/2**. `n_preictal_windows_predicted_positive`
is 0 for both folds -- not a near-miss, the model predicted negative for
every single preictal window in both folds. AUC ~0.50 means it wasn't even
internally ranking preictal above interictal. Likely cause: each LOSO fold
trains on exactly ONE other seizure's 30 preictal windows (chb02 only has
2 seizures that survive SPH=300/SOP=900 filtering, see below) -- that's
one event's temporal window sequence, not a diverse positive class, so
there's essentially nothing to generalize a "preictal signature" from.
`best epoch=1` in the early-stopping log line for at least one fold is
consistent with this -- the val split (35/175 samples, ~6 positives) may
itself be too small/noisy to pick a meaningful checkpoint.

**\*Seizure IDs are wrong by +1 file number -- found, not yet fixed.**
`chb02-summary.txt` documents 3 real seizures: `chb02_16.edf` (130-212s,
excluded from the fold list -- same reason as chb01's `1_21_0`, onset too
close to recording start for the full SPH/SOP lead time),
`chb02_16+.edf` (2972-3053s), `chb02_19.edf` (3369-3378s). The run's own
`2_17_0`/`2_20_0` onset/offset timestamps match `chb02_16+`/`chb02_19`
exactly -- so the *data* in each fold is presumably still the real
seizure's windows, but the *label* attached to it is off by one file
number, almost certainly the same "+`.edf` file shifts a position-based
index" artifact `CONTEXT.md` already documents for chb01's fabricated
`1_02_0` ID. Not investigated further this session (mid-trace when the
session moved on to something else) -- the exact code path is
`leave_one_seizure_out_prediction`'s `unique_seizures` construction in
`run_pipelines.py` (~line 815-820), building `seizure_id` from
`metadata["seizure_id"]`, which presumably derives from a recording-list
position rather than parsing the numeric suffix out of the filename
itself. Worth fixing before trusting any per-seizure ID in future
multi-subject reporting, chb02 or otherwise.

### chb03: 3/7 hit rate, striking early/late split

| seizure | n_test_preictal | recall | AP | AUC | hit |
|---|---|---|---|---|---|
| `3_01_0` | 2 | 0.0 | 0.004 | 0.359 | F |
| `3_02_0` | 14 | 0.0 | 0.020 | 0.434 | F |
| `3_03_0` | 4 | 0.0 | 0.014 | 0.672 | F |
| `3_04_0` | 30 | 0.0 | 0.037 | 0.237 | F |
| `3_34_0` | 30 | 0.800 | 0.496 | 0.878 | **T** |
| `3_35_0` | 30 | 0.833 | 0.467 | 0.901 | **T** |
| `3_36_0` | 30 | 0.667 | 0.538 | 0.907 | **T** |

Mean: AP 0.225, AUC 0.627, hit rate 3/7 (42.9%). The first 4 folds
(earliest recordings, `chb03_01`-`04`) collapse the same way chb02 did
(0 recall, near-chance AUC); the last 3 (`chb03_34`-`36`, much later in
the recording sequence, after a long seizure-free gap `chb03_05`-`33`)
train and predict well -- recall 0.67-0.83, AUC 0.88-0.91, comparable to
chb01's good folds. All 7 folds have a similar-sized training pool
(~3250-3350 windows) so this isn't simply a data-volume ceiling the way
chb02's total collapse might be -- something specific to the early
recordings/seizures (or the gap) looks like the better explanation, but
**not investigated further this session**, just flagged. Open thread.

Neither the seizure-ID bug above nor this early/late split were checked
against other backbones (conv `dense_edge`, `dense_edge_mamba`) --
whether GRU specifically struggles here vs. it being a genuine
data-difficulty signal is unknown. Discussed with user; agreed the most
likely explanation for chb02/chb03's failing folds is data starvation
(each LOSO fold trains on only one-or-few other seizures' preictal
windows for these subjects) rather than a GRU-specific weakness, but this
is reasoning from the numbers, not a tested claim -- the actual test
(same protocol run with `dense_edge`/`dense_edge_mamba` on chb02/chb03)
was proposed but not run this session.

---

## Part 2 -- GitHub-release mirror generalized from chb01-only to any subject

User asked to publish each newly-downloaded subject (chb02, chb03, chb04)
to a GitHub Release, the same mechanism chb01 already used (see
`README.md`'s "CHB-MIT subjects" section and `Epilepsy/runpod_mamba_fast_image_brief.md`'s antecedent for why: PhysioNet's own S3 mirror throttles
hard). User explicitly chose "generalize the loader" over "archive only"
when asked -- i.e. `chb_mit.py` should actually prefer GitHub over
PhysioNet for any subject with a registered release, not just chb01.

### Code changes (`datasets/epilepsy/chb_mit.py`)

- `CHB01_GITHUB_TAG`/`_ARCHIVE_URL`/`_SHA256`/`_SUBJECT` (singular,
  chb01-only) replaced by `GITHUB_RELEASE_SHA256: Dict[int, str | List[str]]`
  -- a subject NOT in this dict behaves exactly as if the registry never
  existed (straight to PhysioNet S3). A subject's value is either one
  sha256 (single `chbXX.tar.gz` asset) or a list of sha256s (multi-part,
  see chb04 below).
- `_chb01_archive_url`/`_chb01_archive_sha256` -> `_github_archive_part_url(subject, part_index, n_parts)` / `_github_release_parts(subject) -> List[str]`.
  Env override naming (`CHBMIT_CHB01_ARCHIVE_URL`/`_SHA256`) turned out to
  already BE the generalized pattern (`CHBMIT_CHB{subject:02d}_...`) by
  coincidence of chb01's own `{:02d}` formatting -- no back-compat shim
  needed there.
- `chb01_cache_dir`/`chb01_cache_complete`/`extract_chb01_archive`/
  `prefetch_chb01_from_github` -> `subject_cache_dir`/`subject_cache_complete`/
  `extract_subject_archive`/`prefetch_subject_from_github`, all now take an
  explicit `subject: int`. The four chb01-named functions still exist as
  one-line wrappers calling the generic ones with `subject=1` (kept since
  nothing outside `chb_mit.py`/its own tests referenced them by name, but
  they're a reasonable spelling for chb01-specific callers).
- `_local_file`'s `if subject == CHB01_GITHUB_SUBJECT and is_edf:` ->
  `if is_edf and _github_release_parts(subject):`.
- **Multi-part support** (needed for chb04, see below):
  `extract_subject_archive` no longer asserts completeness itself (a
  multi-part subject is legitimately incomplete after any single part);
  `prefetch_subject_from_github` now loops over every part, downloading +
  extracting each into the same subject directory (disjoint file sets, so
  extraction order doesn't matter), and only checks
  `subject_cache_complete` once, after the last part -- fails closed
  (returns `False`, falls back to PhysioNet S3) if still incomplete after
  every part extracted, rather than silently reporting a partial cache as
  success.
- `_GITHUB_PREFETCH_FAILED` (bool, chb01-only) -> a `set[int]` of subjects
  that failed this process -- same "don't retry every missing EDF" intent,
  now per-subject.

### Unrelated bug found + fixed while verifying this: Windows `file://` URIs

`download_url`'s `file://` scheme branch did
`shutil.copy2(unquote(parsed.path), tmp)` -- on Windows, `urlparse()`
leaves a leading `/` before the drive letter (`file:///C:/...` ->
`/C:/Users/...`), which `shutil.copy2` can't open
(`[Errno 22] Invalid argument`). Confirmed via `git stash` that 2 of the
existing GitHub-prefetch tests were already failing on this Windows
machine before this session touched anything -- not something introduced
here, just found while re-verifying the suite. Fixed with
`urllib.request.url2pathname(parsed.path)` (same helper MNE's own
`_url_to_local_path` already uses, handles the drive-letter case
correctly on every platform). `unquote` import dropped (no longer used).

### Unrelated bug found + fixed earlier in the session: chb02's `+` filename 404s

`chb02_16+.edf` is a real, documented file in `chb02-summary.txt` (not a
`smoke_test.py`-style indexing artifact like `1_02_0`) but 404'd when
downloaded -- `_record_url` built the URL by raw string concatenation, and
S3 rejects a literal unescaped `+` in the path (confirmed via `curl`:
literal `+` -> 404, `%2B` -> 200). Fixed with `urllib.parse.quote(filename)`
in `_record_url`. Verified the local cache path is unaffected --
`_url_to_local_path` (via `request.url2pathname`) unquotes the path again
when deriving the local filename, so the file still lands on disk as
`chb02_16+.edf`.

### chb04 needed a 2-part split -- GitHub's 2GiB-per-asset cap

chb04's raw recordings are ~6.4GB (larger files than chb01-03, ~177MB
each vs ~170MB). A single gzipped archive came to 3.7GB -- `gh release
create` rejected it: `HTTP 422 ... size must be less than 2147483648`
(2 GiB exactly). Tested `xz -T0` compression ratio on one representative
177MB file first rather than guessing: gzip got 0.59 (compressed/raw),
xz -6 got 0.47, xz -9 only marginally better (0.4673) for ~9x the time --
not worth it. Split the 42 EDFs (+ summary) into two size-balanced groups
(~3.43GB raw each, chosen via a simple greedy largest-file-first
balance) and xz -6 compressed each separately -> ~1.5GB per part, safely
under the cap. Two build attempts failed before this worked:

1. Plain `tar -czf` with a `C:/...` path errored `Cannot connect to C:
   resolve failed` -- GNU/MSYS tar misparses a Windows drive-letter path
   as a remote `host:path` spec unless given `--force-local`.
2. The file-list `.txt` files (fed to `tar -T`) were written by Python's
   default text-mode `open(..., 'w')`, which translates `\n` -> `\r\n` on
   Windows -- `tar -T` then looked for filenames with a literal trailing
   `\r`, failing on every single line. Fixed by rewriting those files with
   `open(..., 'w', newline='\n')`.

Release `chbmit-chb04-1.0.0` ships `chb04.part1.tar.xz` (1.47GiB) +
`chb04.part2.tar.xz` (1.42GiB), each with its own `.sha256` sidecar, same
`THIRD_PARTY_NOTICES.md` as the other three. `GITHUB_RELEASE_SHA256[4]` is
a 2-element list; `_github_archive_part_url` names parts
`chbXX.partN.tar.xz` (1-indexed) whenever `n_parts > 1`.

### Verification before calling this done

- Full test suite (`tests/test_chb_mit_github_prefetch.py` --
  rewritten/generalized, `tests/test_chb_mit_continuous_labeling.py`
  unaffected): 13/13 passing, including two new tests for the multi-part
  path (`test_multi_part_subject_fetches_all_parts`,
  `test_multi_part_subject_incomplete_after_one_part_falls_back` -- the
  second specifically checks a subject is NOT reported successful if only
  some parts are reachable, i.e. the fail-closed behavior actually works,
  not just the happy path).
- Live check against the real published URLs (not just local files):
  `curl -sIL` on all 5 assets (chb01/02/03's single tarball + chb04's two
  parts) -- all 200, and `Content-Length` matches the local file size
  exactly for every asset.
- `gh auth status` confirmed logged in as `noshore5` before any upload
  (blocked earlier in the session; user ran `gh auth login` themselves).

### What's live now

| subject | tag | asset(s) | registry entry |
|---|---|---|---|
| chb01 | `chbmit-chb01-1.0.0` | `chb01.tar.gz` | existing, unchanged |
| chb02 | `chbmit-chb02-1.0.0` | `chb02.tar.gz` | `d0985a02...cd12d6` |
| chb03 | `chbmit-chb03-1.0.0` | `chb03.tar.gz` | `40ebd8e6...ea6219b` |
| chb04 | `chbmit-chb04-1.0.0` | `chb04.part1.tar.xz` + `chb04.part2.tar.xz` | `["1d7d02d7...4dd3d6c", "4be40834...c74df"]` |

`README.md` and `THIRD_PARTY_NOTICES.md` updated to describe all four (the
latter re-uploaded as an asset on chb01/02/03's releases too, since it was
edited after those three were created -- all four releases now carry the
same up-to-date notice).

### Not done / open

- **chb02 seizure-ID off-by-one is unfixed** (see Part 1) -- the
  underlying cause is almost certainly in `run_pipelines.py`'s
  `unique_seizures` construction, not in this session's `chb_mit.py`
  changes, but wasn't traced to a root cause or patched.
- **chb03's early/late fold split is unexplained** -- flagged, not
  investigated (montage/recording-quality difference? something about the
  seizure-free gap? not checked).
- **Nothing this session is committed.** Working tree has real, tested,
  verified changes sitting uncommitted -- `git status`/`git diff` before
  assuming any of it landed, and before running `git stash` for any other
  reason (a bare stash was used mid-session to check pre-existing test
  failures; it was popped back immediately, but worth being deliberate
  about repeating that near this much uncommitted work).
- Disk cleanup happened mid-session at user's request: purged pip cache
  (3.6G -> ~0) and deleted `~/mne_data/dense_edge_cache` (16G, pure
  recompute cache, rebuilds automatically) when the machine hit 92% disk
  usage (94G total, 77G free at the low point). Also cleaned up the
  now-redundant local release-staging tarballs (~8.3G) after confirming
  every asset uploaded and verified correctly. If dense-edge training
  feels slower on the next run, that's the recompute happening, not a
  regression.
- Never got to re-running the GRU vs conv vs Mamba backbone comparison
  proposed for chb02/chb03's failing folds (see Part 1) -- next concrete
  step if this thread continues.
