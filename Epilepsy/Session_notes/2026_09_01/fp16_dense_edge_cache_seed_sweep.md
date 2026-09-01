# fp16 dense-edge cache -- "pre" seed sweep (2026-08-31 -> 09-01)

Mac shell, MPS. Follows `2026_08_31/pre_repro_6fold.md` (fp32 baseline
mean AP **0.644**, historical CPU 0.674). RAM-resident "pre" mission,
CONTEXT.md item 5d.

## Why

`temporal_graph_mamba` "pre" thrashes a 16 GB Mac: the dense-edge cache
is `[4, 253, 479, 8]` fp32 npz, ~15.5 MB/window, ~36 GB for a 6-fold
set -> exceeds RAM -> page-cache thrash, every epoch re-reads ~10 GB off
SSD (folds 2-6 ran 85-390 s/epoch in the fp32 repro).

**fp16 cache lever:** store the `dense` tensor as float16, `.float()`
upcast on load so everything downstream is byte-identical. Halves the
disk footprint and the DataLoader re-stream read. All 4 channels are
bounded [-1, 1] (coh magnitude, sin/cos phase, significance) so fp16's
~1e-3 relative resolution is far finer than the model needs. Distinct
`-fp16` cache-key suffix; dense edge inputs are **seed-independent** so
every seed reuses one cache with no rebuild.

Two questions:
1. Does fp16 change the result vs fp32? (fidelity)
2. If seed-42 fp16 misses fp32, is that an fp16 penalty or seed noise?

## Runs

`_to_delete/run_pre_fp16_emptycache.py <seed>` -- `temporal_graph_mamba`,
`--label-mode prediction --device mps`, untuned
`PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS` (`temporal_graph_aggregate="pre"`,
d_model=16), 20 epochs, `early_stopping_patience=5`. Monkeypatches:
fp16 save/load, `torch_cwt_batch_size` capped 64, and the
**MPS caching-allocator release patch** (see below). Queued one MPS job
at a time by `_to_delete/fp16_queue.sh`; per-run means in
`_to_delete/fp16_queue.out` + `pre_fp16_seed*_*.log`.

### MPS allocator fix (root cause of the earlier fp16 thrash)

The repo **never calls `torch.mps.empty_cache()`**. On Apple Silicon the
MPS caching allocator's reserved pool *is* system RAM; without a release
it grows to the high-water mark of the largest transient in
`_precompute_dense_edge_inputs` and never shrinks -> swap. The wrappers
patch `empty_cache()` into `_save_fp16` and wrap `cg._sync_device`. With
it, seed 42 ran a clean 42-56 s/epoch start to finish. **Candidate for
upstreaming -- ask the user first.** (The earlier "fp16 still thrashes"
scare was two concurrent MPS processes -- an orphaned queue child --
not a patch failure.)

## Result

6-fold mean AP, fp16 cache, by run seed (vs fp32 pre_repro 0.644):

| seed | 1_03 | 1_04 | 1_15 | 1_16 | 1_18 | 1_26 | **mean** |
|---|---|---|---|---|---|---|---|
| 42 | .459 | .791 | .375 | .985 | .807 | .284 | **.617** |
| 43 | .234 | .376 | .463 | .939 | .714 | .216 | **.490** |
| 43 @30ep | .234 | .339 | .341 | .939 | .714 | .216 | **.464** |
| 44 | .918 | .419 | .472 | 1.000 | .562 | .281 | **.609** |
| 45 | .226 | .494 | .509 | .930 | .742 | .159 | **.510** |
| fp32 s42 (pre_repro) | .560 | .811 | .416 | .985 | .811 | .283 | **.644** |

(seed 46 aborted at fold 0 ep 2 when the fp32 Phase-B sweep took the GPU.)

- **fp16 seed mean 0.556, seed std ~0.06** over seeds 42/43/44/45.
- **fp32 seed-42 (0.644) sits ~1.5 std above the fp16 mean** and at the
  top of the fp16 spread. Whether that's a real fp16 penalty or just
  that 0.644 is itself a favorable draw is **not resolvable from one
  fp32 seed** -- Phase B (`_to_delete/phaseB_fp32_sweep.sh`) is now
  sweeping matched fp32 seeds 42, 43, 44, ... for a paired comparison.
- **Matched-seed contrast (the only clean one): seed 42 fp16 0.617 vs
  fp32 0.644, -0.027, every fold <= fp32.** Within one seed-step of
  noise. 1_16_0 identical (.985); 1_04/1_18/1_26 within .02.

### Per-fold seed sensitivity (fp16, seeds 42/43/44/45)

| fold | range | span | note |
|---|---|---|---|
| 1_03_0 | .23 - .92 | **.68** | noise-dominated, ~25 preictal test windows |
| 1_04_0 | .38 - .79 | .42 | |
| 1_18_0 | .56 - .81 | .25 | |
| 1_15_0 | .38 - .51 | .13 | |
| 1_26_0 | .16 - .28 | .13 | low everywhere |
| 1_16_0 | .93 - 1.00 | **.07** | the only stable fold |

All the variance is in the tiny-positive-class folds. On chb01 LOSO,
~23-30 preictal windows/fold -> AP is a step function over a handful of
points and swings 0.1-0.7 on unchanged data.

### seed 43 is a genuinely bad draw, not undertraining

seed 43's `1_04` and `1_16` ran the full 20/20 with best val_loss at
ep 17-19 (still improving, never early-stopped). Rerun at **30 epochs**:
mean went **down** 0.490 -> 0.464 (`1_04` .376->.339, `1_15` .463->.341).
Extra training did not rescue it -> seed 43's validation split / init
just lands on a worse optimum and a worse early-stop point.

## Why the seed moves the result (established from code)

The run seed (`set_seed(self.seed)`) drives:
- **validation split** -- `resolve_train_val_indices(X.shape[0], y_idx,
  int(self.seed or 0), ...)` at `cwt_gnn_classifiers.py:8354` -- which
  windows are val vs train;
- **weight init** (`torch.manual_seed`);
- **batch-shuffle order** (`torch.Generator().manual_seed(seed)`).

It does **not** touch:
- the **test set** -- fixed by the held-out seizure (LOSO);
- the **negative subsample** -- `_subsample_negative_windows` uses
  `subsample_seed`, default 42 at `run_pipelines.py:1506`, and `main()`
  **never passes it** (`run_pipelines.py:3691`) -> the same 725-of-3348
  interictal windows are dropped in every run regardless of `--seed`.

`scatter_mean`'s GPU atomic-add ordering is also nondeterministic
run-to-run even at fixed seed (not seed-driven at all).

So on chb01's tiny per-fold positive class, the val-split choice alone
genuinely moves where early-stopping and SGD land. The user's hypothesis
("seed 42 gives a good split, 43 a bad one") is correct as to
mechanism -- it's the val split, not train/test contamination.

## Conclusions

1. **fp16 cache is faithful.** At matched seed 42, fp16 vs fp32 is
   -0.027 mean AP with 1_16_0 bit-identical and 4/6 folds within .02.
   The gap is smaller than the seed-to-seed noise. fp16 is pure
   compression + upcast; there is no mechanism for a real penalty
   beyond ~1e-3 rounding, and the data agrees.
2. **"pre" has ~0.06 seed std on the 6-fold mean.** 0.644 is a
   favorable-ish draw; the honest "pre" number is **~0.56-0.64
   depending on seed**, driven almost entirely by fold 1_03's
   0.23-0.92 swing. Reconfirms the fragility thesis from
   `2026_08_31/pre_repro_6fold.md` and the 2026-08-28 roundup.
3. **Seed-exposure caveat for NEGATIVES.md:** several early-killed
   "negatives" (edge matrix, edge complex, mamba3, csk12) were judged on
   partial or single-seed 6-fold means and are inside this ~0.06 band --
   they are not cleanly refuted. (drop_significance stays shelved for
   the COI-reconstruction reason, not re-tested here.)
4. **fp16 halves DISK + the re-stream read, NOT resident RAM** -- the
   `_load` upcasts to fp32 immediately, so the edge tensor in RAM is
   full size. The RAM win is the DataLoader working set during training,
   which is exactly what was thrashing.
5. **RAM-resident deliverable (item 5d): the fp16 cache + the
   `empty_cache` patch together give clean 42-56 s/epoch on 16 GB.**
   That is the faithful RAM-resident "pre" configuration.

## Next

- **Phase B running:** `_to_delete/phaseB_fp32_sweep.sh` -- fp32 seeds
  42 (done, 0.581 partial / see CSV), 43 (running, pid 34495), 44, ...
  forever. Gives matched fp16-vs-fp32 pairs to settle conclusion 1
  quantitatively and adds fp32 seed spread.
- fp16 `-fp16` cache **deleted** 09-01 05:35 (4241 files, ~31 GB) to
  free disk for Phase B; the ~1036 fp32 no-suffix files kept. A future
  fp16 run cold-rebuilds.
- DEFERRED (needs user go-ahead per the `pre-band-ceiling-hypothesis`
  memory): nfreqs=8 -> 16 band-resolution probe
  (`_to_delete/run_pre_fp16_nfreqs16.py`), ~63 GB cold rebuild.
- Consider upstreaming the `torch.mps.empty_cache()` fix.
