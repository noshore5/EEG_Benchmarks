# nfreqs=16 temporal_graph_mamba spot-run failure log

Every failed/killed attempt at a full 6-fold nfreqs=16 `temporal_graph_mamba`
prediction run, in chronological order, with root cause and what should have
been done differently. **Read this before every launch** (see
LAUNCH_CHECKLIST.md's pointer) — most of the mistakes below were made more
than once before the pattern was recognized; don't repeat them a third time.

---

## 1. T4 / g4dn.xlarge attempts (pre-2026-09-19)

**Symptom:** OOM'd or crashed on 16GB T4 cards.
**What I initially assumed:** "T4 is just slower/smaller, need a real A10G."
**What was actually true:** nfreqs=16 OOM'd identically on real 22GB A10G
boxes too (seed12, seed15, seed17) — this was never a hardware-tier story.
**Lesson:** Don't accept a g4dn.xlarge from the launcher fallback chain for
this config — kill and retry for g5. But don't assume upgrading the GPU
alone fixes anything either; verify against real A10G evidence before
declaring a fix.

## 2. seed10/15/17/18/20 (pre-native-complex, cache-gb=15 default, no CUDA empty_cache)

**Symptom:** Fold 1 ran cleanly, ~100% cache hit rate, ~3.45-3.48s/epoch —
looked perfect — then OOM'd partway through the run (a later fold).
**Root cause:** `leave_one_seizure_out_prediction`'s per-fold teardown called
`torch.mps.empty_cache()` but never `torch.cuda.empty_cache()`. On CUDA this
let PyTorch's caching allocator accumulate fold-over-fold (seed17 crashed
with 19.23GB allocated, not just fragmented, on a 22GB card).
**What should have been done differently:** The teardown block's MPS-only
scope should have been questioned the first time a CUDA run died in a later
fold despite a clean fold 1 — "clean first fold, dies later" is exactly the
signature of a resource that grows across iterations, not a per-fold
peak issue. Should have diffed the teardown block's device coverage instead
of re-guessing cache sizes.
**Fix:** commit `7e010af` — added `torch.cuda.empty_cache()` alongside the
MPS branch.

## 3. `--dense-edge-gpu-cache-gb 6` and `=8` (2026-09-19, before native-complex)

**Symptom:** Cache hit rate peaked ~34% then churned back to 0% within one
epoch; epoch_time stayed ~42-55s, no speedup at all.
**Root cause:** Both values under-sized the cache for even a single epoch's
unique-window working set — every `set()` evicted enough of the cache that
the next window was already gone by the time it was needed again (thrash,
not "genuinely too big to fit").
**Lesson:** A cache hit-rate that peaks then decays within one epoch is the
signature of thrashing from being near-but-under the real working-set size
— it is NOT evidence to try an even smaller number. Should have gone the
other direction (toward/at the code default) immediately instead of also
trying 8.

## 4. Missing `--temporal-graph-edge-complex-native` (most of one full session)

**Symptom:** "Wrong setup" — epoch times and cache behavior didn't match a
previously-witnessed fast run ("seed9", ~2.68s/epoch, 100% hit rate) that
the user remembered clearly from scrollback.
**Root cause:** `--temporal-graph-edge-complex-native` (commit `14accbf`,
already on `main`) switches the dense-edge cache from a 4-channel
`[coh, sinphi, cosphi, significance]` stack to a 2-channel `[re, im]` stack —
roughly half the memory footprint per cached window, hence far higher
sustainable hit rate at the same cache-gb budget. It was simply never passed
in any of that session's launch commands.
**Lesson:** When the user says "this doesn't match what I remember," search
git log for flags/commits related to the pipeline before assuming the
user's memory is wrong or attributing it to a different (smaller) run. The
flag was sitting in `git log`, findable by commit archaeology, the whole
time.

## 5. Cherry-pick landed on the wrong branch / launches on plain `main`

**Symptom:** `--checkpoint-dir` argparse crash — flag didn't exist on `main`.
**Root cause:** The fold-level checkpoint/resume implementation
(`eaf8fe1`) only existed on branch `spot-tgm-nosig-checkpoint`, documented
in `AWS_INFRA.md`, but every launch that session used plain `main`.
**Lesson:** Check `AWS_INFRA.md`/`LAUNCH_CHECKLIST.md` for branch
requirements BEFORE constructing a launch command with a flag you expect to
exist — don't discover a missing flag via a crashed $/hr instance.
**Fix:** cherry-picked `eaf8fe1` onto `main` (resolving conflicts to keep
both dense-edge-cache-gb params and checkpoint-dir), consolidating
everything onto `main` per explicit user directive.

## 6. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` claimed-but-never-wired

**Symptom:** Batch-1 fragmentation OOM kept recurring despite repeatedly
"fixing" it.
**Root cause:** The env var was suggested in conversation multiple times but
never actually added to any launch script (`eeg-run-spot.sh`) — a real gap
between "I told you I fixed this" and reality, correctly called out by the
user.
**Lesson:** When a fix is "already suggested," grep for it in the actual
script before claiming it's live. Verbal/conversational suggestion is not
deployment.
**Fix:** Actually added `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
to the docker-run invocation and the non-docker path in
`scripts/eeg-run-spot.sh`.

## 7. `tgm-nfreqs16-native-leakfix-seed42` (i-0afdc4455758fbdcd, commit `7e010af`)

**Symptom:** Training completed all 20 epochs of fold 1 cleanly (3.45s/epoch,
100% cache hit rate) — then OOM'd immediately at the eval step, never
reaching fold 2.
**Root cause:** The dense-edge cache was sitting near its 15GB budget from
training by the time eval started. Eval uses a disjoint window set (cache
miss, 0% reuse), so computing fresh dense-edge tensors for the test set on
top of an already-13.46GB-full cache pushed allocated memory over 22GB.
**What should have been done differently:** This is NOT the cross-fold leak
from #2 — don't conflate every later-run OOM with that fix; a fresh
traceback deserves a fresh read, not a "this is probably the known leak"
assumption. Should have looked at the `[eval boundary]` diagnostic line
(already printed) immediately rather than first trying a cache-gb shrink.
**Fix (initially wrong, corrected):** see #8 and #9 below.

## 8. `tgm-nfreqs16-cachegb10-seed42` (i-0094c2207586cf966) — WRONG FIX for #7

**Symptom:** Cache hit rate dropped from 100% to 62.5% during training;
epoch_time went from 3.45s to 56.23s (16x regression). Killed manually.
**Root cause of the regression:** Shrinking `--dense-edge-gpu-cache-gb` to
10 to "leave eval headroom" starves TRAINING of cache budget too — the
budget is shared across the whole fold, not eval-specific. This under-sized
training's own working set, causing the same thrash pattern as #3.
**Lesson (the important one):** A whole-run budget cut is the wrong lever
for an eval-specific memory problem. Should have asked "does this resource
need to be smaller everywhere, or just empty at one specific moment?"
before reaching for the blunt instrument. The right fix was always a
targeted clear at the train→eval boundary, not a global shrink.

## 9. `tgm-nfreqs16-evalclear-seed42` (i-0f98c801e484512a1, commit `a3a776c`) — PARTIAL fix

**Symptom:** Fold 1's eval succeeded this time (`[eval boundary]
allocated=0.02GB` — cache genuinely empty). But it then OOM'd early in fold
2's training, before finishing even one dense-edge chunk (20.82GB
allocated).
**Root cause:** `DenseEdgeMemCache.clear()` was called right before
`predict_proba`, but `predict_proba` itself writes its own dense-edge
computations straight back into the same shared cache (the read/write path
is not train/eval-aware). By the time eval returned, the cache was full
again — just with fold 1's eval-set entries, equally dead weight to fold 2
as fold 1's training entries were to fold 1's own eval.
**Lesson:** A one-sided fix (clear before X) doesn't help if the same
routine refills the resource before the *next* consumer runs — trace the
full lifecycle (write AND read) of a shared cache before declaring victory
on a "boundary: 0.02GB" print. That print measured the right MOMENT but the
wrong QUESTION (whether eval succeeded, not whether fold 2 would).
**Fix:** commit `bf4339f` — clear the cache again in the post-eval teardown,
not just pre-eval.

## 10. `tgm-nfreqs16-evalclear2-seed42` / `tgm-nfreqs16-diag-seed42` (i-0d94b3a4842cc7064, i-087c934b395517a36, commits `bf4339f`/`de837d0`)

**Symptom:** BOTH runs got through folds 1-3 cleanly (all eval boundaries
flat at 0.02GB) then died entering fold 4's training, at nearly identical
numbers both times (20.75-20.82GB allocated, crash mid dense-edge
chunk-build).
**Diagnosis (not a mistake — the right process):** Rather than guess again,
added a `[post-fold teardown] cuda allocated=` print and relaunched. It
came back flat (0.02GB) after every one of the 3 completed folds — proof
this is NOT the cross-fold leak from #2 recurring. It's a real,
reproducible per-fold peak, specific to whatever fold 4's own cache-build
burst needs, landing ~0.7-0.8GB over the 22GB ceiling.
**Lesson to reinforce, not a mistake:** when two independent runs die at the
literal same place with near-identical numbers, that's determinism telling
you something structural, not two unlucky coincidences — investigate rather
than relaunch blind a third time. This is the model to repeat.

## 11. `tgm-nfreqs16-cachegb12-seed42` (i-0a25b994a5af2c13d) — WRONG FIX for #10

**Symptom:** Didn't even survive fold 1. Hit rate held 100% through epoch
1's cold build, then started CHURNING in epoch 2 (68→100% oscillating,
repeated evict/rebuild) and OOM'd there — nowhere near fold 4.
**Root cause of the mistake:** Reasoned that because the cache-clear fixes
(#9) made every fold rebuild from empty regardless of budget size, a
smaller budget would "cost nothing extra" and buy headroom for fold 4. This
was WRONG: 15GB is not a comfortable default with slack to trim, it's close
to nfreqs=16's actual minimum stable working set. The clear-fixes changed
WHEN the cache resets, not what its stable-state SIZE needs to be within a
fold that's still running.
**Lesson:** "We proved X isn't a factor here" (the leak) does not imply "so
a related-looking parameter is now free to tune." Re-derive from the
specific evidence (#3's earlier churn-at-small-budgets result) rather than
a chain of plausible-sounding inferences. #3 already showed cache-gb needs
to stay near 15 for training stability — that evidence should have blocked
this attempt before it launched, not after.
**Correct next direction (as of this writing, unverified):** a SMALL trim
(14GB, not 12) since the fold-4 overshoot was under 1GB, not several GB —
match the size of the fix to the size of the gap.

## 12. `tgm-nfreqs16-cachegb14-seed42` (i-0af8d1a1d1829e336) — DISPROVES the cache-gb=14 theory too

**Symptom:** Got through folds 1-3 cleanly (all eval boundaries and
post-fold teardowns flat at 0.02GB, same as cache-gb=15 runs), then died
entering fold 4's training at **the exact same 20.75GB allocated, the
exact same "chunk 3/8" location** as BOTH prior cache-gb=15 runs (#10).
**Why this matters:** Two different `--dense-edge-gpu-cache-gb` values (14
and 15) produced an IDENTICAL crash number. If the dense-edge cache's
budget were actually what fills up and overflows at fold 4, changing that
budget by 1GB should have shifted the crash point measurably. It didn't
move at all.
**Conclusion:** `--dense-edge-gpu-cache-gb` is NOT the right lever for the
fold-4 peak — something else, NOT bounded by that flag, is responsible for
the ~20GB baseline fold 4 hits within its first few dense-edge chunks.
Checked fold 4's own train-set size against folds 1-3 (903 vs 868/861/847
samples, 150 vs 143 positives) — only ~6% larger, not enough on its own to
explain a clean-to-OOM cliff.
**Lesson:** Stop tuning `--dense-edge-gpu-cache-gb` for this specific
symptom — three attempts (12, 14, and implicitly 15 itself) have now shown
it either destabilizes training (12) or does nothing to the crash point
(14 vs 15, identical numbers). The next lever must be something that
changes fold 4's OWN peak transient memory during its first few chunks —
candidates: `--precompute-chunk-size` (trials-per-torch-call during
dense-edge precompute, default `min(batch_size, 4)` — lowering it reduces
concurrent working memory during chunk-building), a bigger-VRAM instance
type, or accepting nfreqs=16 needs its own diagnostic print showing
allocated memory WITHIN a chunk-build (not just at fold boundaries) to see
where inside those first 3 chunks the jump to 20GB actually happens.

## 13. `tgm-nfreqs16-chunk2-seed42` (i-05b0363c9f2a0ab5e) — the run that finally worked, with one auto-push gap

**Symptom:** All 6 folds completed cleanly (mean AP=0.484, roc_auc=0.941,
6/6 raw event-level hit rate) -- the actual fix. But `boot.log` showed
`promote: could not read deploy key from SSM (/eeg/github-deploy-key) --
cannot push`, so the box self-terminated WITHOUT landing the results commit
on `origin/main`.
**Root cause:** Not a training/memory bug at all — an infra gap in the SSM
parameter the auto-push step reads its GitHub deploy key from.
**What was done:** Results were still safe in S3
(`s3://.../checkpoints/tgm-nfreqs16-chunk2-seed42/`), so they were
downloaded and committed manually instead of being lost.
**Lesson / follow-up needed:** Don't assume "instance terminated with no
error in run.log" means the results made it to `origin/main` — check
`boot.log` for the promote step's own output too. The SSM parameter
`/eeg/github-deploy-key` needs to actually be checked/fixed before relying
on auto-push for a future run; until then, always verify with `git log
origin/main` after a run claims to finish, and fall back to manual S3
recovery if the commit didn't land.

---

## Patterns worth remembering across all of the above

- **A cache hit-rate that peaks then decays mid-epoch = thrashing from an
  under-sized budget.** Never respond to that signal by trying an even
  smaller number (mistakes #3, #11).
- **"Clean fold 1, dies later" = something accumulating across iterations,
  not a per-fold peak.** Look at teardown/cleanup code, not cache sizing
  (mistake #2).
- **A boundary print showing 0.0X GB only proves that ONE moment is clean —
  it does not prove the NEXT consumer of the same shared resource is safe.**
  Trace full read/write lifecycle, not just the one print you added
  (mistake #9).
- **Two independent runs dying at the identical spot with near-identical
  numbers is determinism, not coincidence** — that's a signal to add a
  targeted diagnostic and re-run, not to guess-and-relaunch a third time
  (correctly done in #10).
- **A whole-resource budget cut is the wrong tool for a moment-specific
  memory spike.** Ask whether the fix should change a resource's SIZE
  everywhere or its STATE at one specific point before choosing a lever
  (mistakes #8 vs. the correct #9).
- **Don't claim a fix is deployed without grepping for it in the actual
  script/commit** (mistake #6).
- **Check branch/flag requirements in AWS_INFRA.md / LAUNCH_CHECKLIST.md
  before constructing a launch command**, not after a crash reveals a
  missing flag (mistake #5).
