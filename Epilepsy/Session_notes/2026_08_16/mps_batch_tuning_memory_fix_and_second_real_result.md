# Session notes — MPS/batch-size tuning, dead-metric cleanup, memory fix, second real result (2026-08-16)

Branch: `main`. Repo: `/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`.

Follow-on to [today's earlier session](disk_caching_epoch_tuning_and_first_real_result.md),
which left device (CPU vs MPS) and batch size flagged as untested,
unexploited levers. Tested both, found and fixed a real device-selection
bug and a real memory leak along the way, and got a second real 7-fold
result to compare against the first.

---

## Part 1 — MPS + batch size: measured, not guessed

Ran an isolated timing comparison (`device` × `batch_size`, 2×2, real
cached data — chb01_03.edf, 450 windows, 4 epochs each) once the earlier
real run had finished and nothing else was competing for CPU:

| config | steady-state epoch time (epochs 2-4 avg) | vs. `cpu`/`batch=8` |
|---|---|---|
| cpu, batch=8 (old default) | 4.01s | baseline |
| cpu, batch=32 | 3.82s | ~5% faster |
| mps, batch=8 | 2.53s | ~1.6x faster |
| **mps, batch=32** | **0.96s** | **~4.2x faster** |

MPS and batch size compound: MPS alone helps (~1.6x, overturning
`common.py`'s existing-but-unmeasured "might regress" comment on why the
training loop defaulted to CPU), and a bigger batch amortizes MPS's
per-kernel-launch overhead much more than it does on CPU (barely moves
CPU's number, but **2.6x**s MPS's). Correctness checked too, not just
speed — CPU and MPS produced numerically matching loss curves (e.g.
`0.822201` vs `0.822204`), so this is a real device change, not
approximate.

**Applied**: `DENSE_EDGE_GRU_PARAMS`: `device="mps"`, `batch_size=32`
(was `"auto"`→cpu, `8`).

**Real bug found applying it**: the change had no effect on an actual run
at first. `main()` unconditionally overwrites `clf_params["device"] =
args.device`
([run_pipelines.py:290](../../run_pipelines.py#L290)), and `--device`'s
argparse default was still `"auto"` — which only checks CUDA
(`resolve_torch_device`, `common.py:141`), silently falling back to CPU on
this Apple Silicon machine regardless of what `DENSE_EDGE_GRU_PARAMS` said.
Same class of bug as `epochs`/`step_size` being out of sync earlier today
-- fixed by changing `--device`'s default to `"mps"` too, so a plain run
(no explicit flag) actually gets what the params dict says.

**Learning rate**: batch size went 8→32 (4x). This pipeline uses `AdamW`
(`common.py:1632`), not plain SGD — the classic linear-scaling rule
(`lr × batch_ratio`, from SGD-with-momentum literature) is less directly
justified for Adam-family optimizers, which already adapt per-parameter
step size. Used the more commonly-cited Adam heuristic instead: **√ratio
scaling** (`lr × √4 = lr × 2`) → `learning_rate` `1e-3` → `2e-3`. Explicitly
a heuristic starting point, not validated for this model.

---

## Part 2 — `bursts_per_row`: a dead metric, cleaned up

Investigated what `bursts_per_row=0.125000` (constant across every epoch,
every fold, every run so far) actually means
([sparse_evidence_gnn_classifier.py:3049-3063](../../pipelines/sparse_evidence_gnn_classifier.py#L3049)):
`event_density = valid_edge_count / (batch_size * n_edges * nfreqs)`. In
`event_mode="sparse"` (not what this pipeline runs) it's a real, unbounded
signal -- average detected coherence "burst" events per row. In
`event_mode="dense"` (what `DENSE_EDGE_GRU_PARAMS` uses), **every edge
always fires**, so it collapses to the constant `1/nfreqs` every call --
`1/8 = 0.125`, exactly what's been printed. The code's own comment already
flagged this ("not a meaningful density signal... kept so the aux-metric
return contract stays identical across modes").

**Fix, scoped to logging only** -- didn't touch `forward()`'s return
shape (avoiding exactly the kind of cross-mode contract break that
comment was already guarding against): added
`self._log_aux_metric = event_mode not in ("dense", "temporal_graph")` to
`SparseEvidenceGNNClassifier.__init__`, and gated `common.py`'s epoch-log
`aux_suffix` construction on `getattr(self, "_log_aux_metric", True)` --
defaults to `True` (old behavior, unchanged) for every model that doesn't
set the flag; only this pipeline's dense/temporal_graph modes suppress it.
Verified: `_log_aux_metric` is `False` for the real config, and
`bursts_per_row` no longer appears in a fresh run's log. The underlying
value is still computed and still feeds `edge_density_history_` --
nothing downstream that might consume it was touched.

---

## Part 3 — Real memory leak found and fixed mid-run

While a real run (`--device=mps --epochs=10`) was going, epoch time crept
from ~6s up to ~20s+ within the *same* run/fold -- not explained by epoch
count, device, or anything from Parts 1-2. Checked `ps` first: only one
`run_pipelines.py` process running, so not multi-run contention. Checked
the machine directly instead of guessing further:

```
vm.swapusage: total = 20480.00M  used = 20086.00M  free = 394.00M
Pages free: 4130  (≈66MB of actual free RAM)
Swapins: 113,998,962   Swapouts: 158,876,582
```

Nearly all 20GB of swap in use, ~66MB of real RAM free, huge swap
in/out counts -- the machine was thrashing, and the process was observed
in "U" (uninterruptible sleep) state at 3.3% CPU, consistent with I/O-wait
from paging rather than compute.

**Root cause**: `DiskCWTCache`'s in-memory front-cache (`self._mem`,
added [earlier today](disk_caching_epoch_tuning_and_first_real_result.md)
to avoid re-reading disk on repeat lookups within one process) never
evicted, and is shared across all 7 folds of one `leave_one_seizure_out()`
call. Folds mostly overlap, so by fold 2-3 the union of windows touched is
nearly the whole dataset -- **2,991 windows × 23 channels ≈ 68,800
entries**, each a decompressed `(1024, 8)` float32 real+imag pair
(~64KB), all held simultaneously: **≈4.4GB**, growing monotonically,
never shrinking, on a 16GB machine already running other things.

**Fix**: dropped the in-memory layer entirely
([cwt_window_cache.py](../../pipelines/cwt_window_cache.py)) -- `.get()`
is now always a real disk read, `__setitem__` always writes straight
through, `__len__` counts `.npz` files on disk instead of dict entries
(changes its meaning slightly: now reflects everything ever cached under
`cache_dir`, including from prior runs, not just this process's touched
set -- updated the one place that prints it,
[run_pipelines.py:226](../../run_pipelines.py#L226), to say so). Verified
with a round-trip write/read test and confirmed `hasattr(cache, "_mem")`
is now `False`. Trades a cheap disk read per repeat hit for a hard bound
on memory growth -- the dense-edge cache never had this problem (always
disk-only, no front-cache was ever added to it), so this was the one place
that needed it.

---

## Part 4 — Second real result

The `--device=mps --epochs=10 --step-size=8.0` run that surfaced the
memory issue actually finished all 7 folds despite the slowdown (~22
minutes total, `1:52pm` → results at `14:13:45`) --
[leave_one_seizure_out_20260816-141345.csv](../../results/leave_one_seizure_out_20260816-141345.csv):

| run | f1 | average_precision | roc_auc |
|---|---|---|---|
| 01 | 0.833 | 0.943 | 0.999 |
| 02 | 0.667 | 0.875 | 0.998 |
| 03 | 0.833 | 1.000 | 1.000 |
| 04 | 0.833 | 0.948 | 0.999 |
| 05 | 0.957 | 1.000 | 1.000 |
| 06 | 0.923 | 1.000 | 1.000 |
| 07 | 1.000 | 1.000 | 1.000 |
| **mean** | **0.864** | **0.967** | **0.999** |

Compared to [the earlier `epochs=20` run](disk_caching_epoch_tuning_and_first_real_result.md)
(mean f1 0.902, average_precision 0.886, roc_auc 0.995): **not simply
better or worse**. F1 (a fixed-0.5-threshold metric) dropped in several
folds (notably run 02: 0.857→0.667), but average_precision and roc_auc
(threshold-independent, measuring the whole ranking) both went *up* across
nearly every fold. Likely explanation: `epochs` dropped 20→10 while
`batch_size` stayed at 32, so total optimizer steps roughly halved
(`10×80=800` vs `20×80=1600`) -- plausibly enough to shift where the
default 0.5 cutoff lands relative to an underlying ranking that's just as
good or better, without being enough to hurt the ranking itself. Not
disentangled from the swap-induced slowdown either (a fold trained while
the machine was thrashing isn't a clean comparison point) -- this result
should be treated as suggestive, not conclusive, until rerun on the
now-fixed memory behavior.

Runtime: **~22 minutes** for all 7 folds (vs. ~1h42m for the earlier
`epochs=20` run) -- despite the swap slowdown partway through, still a
large net win from MPS + batch=32 + fewer epochs combined.

---

## Current state

- [run_pipelines.py](../../run_pipelines.py) defaults: `device="mps"`,
  `batch_size=32`, `learning_rate=2e-3`, `epochs=20`, `step_size=8.0`,
  `window_length=4.0` (gap issue from earlier today still unresolved).
- [cwt_window_cache.py](../../pipelines/cwt_window_cache.py)'s
  `DiskCWTCache` is now memory-bounded (disk-only, no front-cache).
- `bursts_per_row` no longer printed for this pipeline's dense-mode runs.
- Two real 7-fold results exist now (`epochs=20`/cpu-ish timing vs.
  `epochs=10`/mps/batch=32), not directly comparable due to the
  epochs/batch confound and the memory issue affecting the second run's
  timing -- see Part 4.

## Open items

- Re-run at `epochs=20` (or whatever's decided) now that the memory leak
  is fixed, to get a clean comparison point not confounded by swapping.
  mid-run.
- `learning_rate=2e-3` is a heuristic (√-scaling for Adam), not validated
  against this model/data -- was going to test candidates empirically,
  not done yet.
- `window_length`/`step_size` gap (`4.0`/`8.0`) from earlier today, still
  open.
- Frequency band (1-40Hz) still untuned, carried over from 2026-08-15.
- Only subject 1, only chb01 -- same deliberate scope limit as before.
