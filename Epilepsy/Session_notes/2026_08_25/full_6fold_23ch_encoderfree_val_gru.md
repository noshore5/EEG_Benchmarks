# Session notes — 23ch (full mesh) encoder-free GRU, val 0.2, all 6 folds (2026-08-25)

Branch: `tf-node-encoding` (`d76b7c9` plus uncommitted `run_pipelines.py` /
`smoke_test.py` changes, +50/-1 lines, not otherwise touched this session).
Repo: `C:\Users\User\Documents\noshore5\EEG_Benchmarks`. Machine: local
Windows box, RTX 3070 Ti (8GB VRAM). Subject chb01.

Follow-on to
[2026-08-24's encoder-free 1-fold run](../2026_08_24/encoder_ablation_amp_and_23ch_onefold.md#part-5--23-channel-1-fold-encoder-free-model)
(Part 5), which only covered fold 1 (`1_03_0`). That note's Open section
asked for the same model on all 6 folds so k=20 (from
[the k-sweep](../2026_08_24/k_sweep_channel_subset_cuda.md)) vs full mesh
would be an apples-to-apples ablation. This run is that — same command,
`--max-folds` dropped.

```
python Epilepsy/run_pipelines.py --device cuda --validation-split 0.2 \
    --dense-edge-amp-bf16 --train-amp-bf16
```

No `--channel-subset-k` → full 253-edge WCT, all 23 channels live.
Encoder-free (`sparse_message_mlp` 8-in, `total_params=10539`, same model
as Part 5). `--max-folds` was not passed, so **all 6 leave-one-seizure-out
folds ran**, not just fold 1 — there is no CLI flag to start at fold 2 and
skip fold 1 (`--max-folds` only truncates from the front), so fold 1 was
deliberately rerun rather than patched around; being deterministic
(seed 42), it's a reproducibility check as much as new data.

Disk cache off (CUDA default). Exit 0. Log:
`Epilepsy/results/prediction/full6fold_23ch_encoderfree_20260825-105629.log`.
CSVs: `prediction_leave_one_seizure_out_20260825-105659.csv`,
`prediction_per_seizure_20260825-105659.csv`.

---

## Per-fold

| seizure | n_test (pre) | stop / best epoch | best val_loss | hit raw→sm | FAR/h raw→sm | prec | rec | f1 | **AP** |
|---|---|---|---|---|---|---|---|---|---|
| `1_03_0` | 750 (30) | 17 / 12 | 0.095022 | T→T | 18.667→14.833 | 0.194 | 0.900 | 0.320 | 0.141 |
| `1_04_0` | 650 (30) | 13 / 8 | 0.133858 | T→T | 38.710→28.258 | 0.130 | 1.000 | 0.231 | 0.339 |
| `1_15_0` | 718 (30) | 10 / 5 | 0.189388 | T→T | 15.000→9.593 | 0.259 | 1.000 | 0.411 | 0.472 |
| `1_16_0` | 743 (23) | 20 / 15 (ran full) | 0.138695 | T→T | 2.667→0.000 | 0.515 | 0.739 | 0.607 | 0.619 |
| `1_18_0` | 750 (30) | 13 / 8 | 0.177033 | T→T | 1.333→0.000 | 0.742 | 0.767 | 0.754 | 0.776 |
| `1_26_0` | 630 (30) | 13 / 8 | 0.123495 | T→**F** | 9.200→2.600 | 0.220 | 0.433 | 0.292 | 0.193 |

`1_03_0` matches Part 5 exactly (same seed, same fold) — confirms
determinism, not new information. `1_26_0` is the k-sweep's known
persistent miss, still a miss here.

## Mean across 6 folds

| accuracy | precision | recall | f1 | **AP** | AUC | FAR/h raw | FAR/h sm | hit raw | hit k-of-n |
|---|---|---|---|---|---|---|---|---|---|
| 0.878 | 0.343 | 0.807 | 0.436 | **0.423** | 0.944 | 14.26 | 9.21 | 6/6 | 5/6 |

---

## Vs k=20 sweep and yesterday's other 23ch runs

**Not a clean ablation either, same caveat the k-sweep note already
flagged.** The k-sweep (`k=20` row below) still had `ChannelSignalEncoder`
constructed and zeroed in the graph (24-in `sparse_message_mlp`) — this
run is the encoder-free 8-in model. Same six seizures, same val split,
same bf16 flags in both.

| setup | model | AP | AUC | prec | rec | f1 | FAR/h raw→sm | hit k-of-n | protocol |
|---|---|---|---|---|---|---|---|---|---|
| **full mesh, encoder-free (this run)** | 8-in MLP | **0.423** | 0.944 | 0.343 | 0.807 | 0.436 | 14.26→9.21 | 5/6 | val 0.2, early stop, both bf16 |
| k=20 GRU (2026-08-24 sweep) | encoder in graph, 24-in MLP | 0.567 | 0.953 | 0.420 | 0.845 | 0.513 | 12.02→5.64 | 5/6 | same as above, different model |
| k=16 GRU (sweep) | encoder in graph | 0.534 | 0.950 | — | — | — | — | 5/6 | same |
| k=12 GRU (sweep) | encoder in graph | 0.457 | 0.932 | — | — | — | — | **6/6** | same |
| 23ch GRU, no val, 20 ep (`20260823-132639`) | encoder in graph | 0.413 | 0.916 | — | — | — | — | 3/6 | no val split, all 20 epochs |

Full mesh (encoder-free, this run) sits essentially on top of the
old no-val 23ch run (0.423 vs 0.413) despite being a different model
(encoder-free) and a different training protocol (val split + early
stopping vs 100%-of-train/20-epochs) — those two factors partly cancel.
It is well below every k-subset point in the sweep, including k=12's
weaker AP (0.457). k=20 vs full-mesh is **still not a same-model
ablation** — the sweep needs a rerun without the encoder to finish that
comparison cleanly (repeating the open item from the 2026-08-24 note).

---

## Mamba comparison — requested, not fulfilled cleanly

Asked to compare against "the dense edge mamba run, fold 1." That run
exists but on `origin/mamba-temporal-edge-model` (merged to
`origin/main` at `6d38573`, **not** in this branch's history) — session
note
[`dense_edge_mamba_k23_full_run.md`](https://github.com/noshore5/EEG_Benchmarks)
(2026-08-24, commit `8e0a6bd`), read via `git show` rather than checked
out.

**Not comparable to this run's fold 1 (`1_03_0`):**

- Different held-out seizure. It used `smoke_test.py`'s seizure
  enumeration and held out `1_02_0` — not in this run's 6-seizure set
  (`03/04/15/16/18/26`) at all.
- Different test-set size. `smoke_test.py` caps interictal at 5
  recordings → `n_test=150`. This run is uncapped → `n_test=650–750`.

Its result (k=23 full mesh, `smoke_test.py --epochs 20`, best epoch 10,
stopped epoch 15): `F1=0.690 AP=0.682 AUC=0.9417 precision=0.526
recall=1.000 FAR raw=27.0/h→smoothed=0.0/h`.

**Concern, not celebration:** that run's `loss`/`val_loss` hit exactly
`0.000000` and `val_acc`/`val_roc_auc` hit exactly `1.000`/`1.0000` by
epoch 8 and held through the early-stop window (epoch 10–15), on 623
train / 125 val windows. User separately reports a fold-3 Mamba run on a
different machine that "scored 100% on everything" (raw FAR and stop
epoch not recorded). Same signature twice. Given 30s windows drawn from
continuous EEG can be near-duplicates across adjacent time, this reads
like a candidate train/val split leak or degenerate-split artifact in
the Mamba path specifically — not evidence Mamba beats GRU. Worth
checking whether consecutive windows from one recording land on both
sides of the 623/125 split before trusting any Mamba AP number.

User decision this session: write up GRU only; do not launch a matched
Mamba 6-fold run yet. If/when that changes: `dense_edge_mamba` at k=23
full mesh measured ~50.74s/epoch vs this run's ~14.3s/epoch (~3.5x, not
smoke-scale's ~14x — the smoke number was inflated by `mamba_chunk_size`
overhead at a batch composition that no longer applies at full mesh), so
a real uncapped 6-fold run is a rough 60–90 min job on this GPU, and
would need `origin/mamba-temporal-edge-model` pulled/merged into this
branch first.

**Also worth correcting, found later the same day:** the `1_02_0` seizure
ID that run's writeup cites doesn't exist. chb01's `chb01-summary.txt`
lists exactly 7 real seizures (`03, 04, 15, 16, 18, 21, 26`);
`chb01_02.edf` has 0 documented seizures. So `smoke_test.py`'s Mamba run
wasn't testing "a different but valid seizure" — its seizure-ID
enumeration produced an ID that isn't real, most likely an indexing bug
tied to `max_interictal_recordings` capping which recordings survive into
`positive_meta` before `unique_seizures` is built (same capped-dataset
code path flagged elsewhere as unreliable, see `smoke_test.py`'s own
docstring caveat). The 7th real seizure that's absent from *this* run's
6-fold set is `1_21_0`, not `1_02_0` — it produces zero surviving
preictal windows under SPH=300/SOP=900 (its onset is too close to its own
recording's start for the full lead time to fit), so it never enters
`leave_one_seizure_out_prediction`'s `unique_seizures` list at all. Not a
bug in the real run's fold selection — just a naming/indexing artifact in
`smoke_test.py`'s capped path specifically. Treat that old note's numbers
as unverified beyond "Mamba ran and produced output."

---

## Matched Mamba comparison (same day, run completed later)

The matched run did get launched after all — same command, `--pipeline
dense_edge_mamba`, run in the `EEG_Benchmarks_mamba` worktree (branch
`mamba-temporal-edge-model` tip `5301b8c9b`) to avoid disturbing this
branch's working tree:

```
python Epilepsy/run_pipelines.py --device cuda --validation-split 0.2 \
    --dense-edge-amp-bf16 --train-amp-bf16 --pipeline dense_edge_mamba \
    --channel-subset-k 23
```

Same 6 seizures, same val split, same bf16 flags, same encoder-free
model — genuinely matched protocol this time. Exit 0. Log:
`Epilepsy/results/dense_edge_mamba/prediction/
full6fold_mamba_23ch_20260825-113022.log` (worktree). CSVs:
`prediction_leave_one_seizure_out_20260825-113049.csv`,
`prediction_per_seizure_20260825-113049.csv`.

**The exact-0.0 val_loss / 100%-everything pattern did not reproduce.**
Checked every epoch line in the full 6-fold log: zero occurrences of
`val_loss=0.000000` or `val_acc`/`val_roc_auc`=`1.0000`. val_loss curves
look ordinary (bottoms out 0.07–0.19 depending on fold, non-monotonic,
early-stops normally). This supports the theory above — that pattern was
specific to `smoke_test.py`'s small capped train/val split, not a Mamba
training-path issue. Consider that open item closed.

### Per-fold (Mamba)

| seizure | n_test (pre) | stop / best epoch | best val_loss | hit raw→sm | FAR/h raw→sm | prec | rec | f1 | **AP** |
|---|---|---|---|---|---|---|---|---|---|
| `1_03_0` | 750 (30) | 12 / 7 | 0.071131 | T→T | 16.833→13.167 | 0.192 | 0.800 | 0.310 | 0.156 |
| `1_04_0` | 650 (30) | 12 / 7 | 0.114545 | T→T | 28.258→17.419 | 0.166 | 0.967 | 0.283 | 0.480 |
| `1_15_0` | 718 (30) | 11 / 6 | 0.188221 | T→T | 7.151→1.570 | 0.397 | 0.900 | 0.551 | 0.629 |
| `1_16_0` | 743 (23) | 14 / 9 | 0.077140 | T→T | 4.000→0.000 | 0.489 | 1.000 | 0.657 | 0.883 |
| `1_18_0` | 750 (30) | 14 / 9 | 0.101044 | T→**F** | 0.333→0.000 | 0.600 | 0.100 | 0.171 | 0.618 |
| `1_26_0` | 630 (30) | 10 / 5 | 0.139702 | T→T | 13.800→6.200 | 0.242 | 0.733 | 0.364 | 0.229 |

Mean epoch_time this run: 65.97s (n=73 epochs across 6 folds) vs GRU's
~14.3s → **~4.6x slower**, matching the earlier single-fold estimate.

Notably, Mamba's smoothed miss is `1_18_0`, not `1_26_0` — GRU's
persistent miss (`1_26_0`) is a **hit** here (raw and smoothed both
True). First model variant tried that hits `1_26_0`. Doesn't mean the
"persistent miss" pattern is broken — it means it's specific to GRU (or
to whatever GRU and the earlier encoder-in-graph runs share), not
universal across temporal backbones. Worth flagging, not over-reading
from a single run.

### Mean across 6 folds — GRU vs Mamba

| metric | GRU | Mamba | Δ (Mamba − GRU) |
|---|---|---|---|
| accuracy | 0.878 | 0.896 | +0.018 |
| precision | 0.343 | 0.348 | +0.004 |
| recall | 0.807 | 0.750 | −0.057 |
| f1 | 0.436 | 0.389 | −0.046 |
| **AP** | **0.423** | **0.499** | **+0.076** |
| AUC | 0.944 | 0.953 | +0.008 |
| FAR/h raw | 14.26 | 11.73 | −2.53 |
| FAR/h smoothed | 9.21 | 6.39 | −2.82 |
| hit rate (raw) | 6/6 | 6/6 | — |
| hit rate (k-of-n) | 5/6 | 5/6 | — |
| mean epoch_time | ~14.3s | ~65.97s | ~4.6x slower |

Mamba comes out ahead on the metric that matters most for this task (AP,
+0.076) and on both false-alarm-rate numbers, at essentially the same
hit-rate reliability (5/6 both, different seizure missed). Costs ~4.6x
the wall-clock per epoch on this hardware (`mambapy` pure-PyTorch scan,
no CUDA kernel — see `Epilepsy/runpod_mamba_fast_image_brief.md` for the
path to close that gap). Recall/f1 favor GRU, driven almost entirely by
Mamba's near-total recall collapse on `1_18_0` (rec 0.100 vs GRU's
0.767 on the same seizure) — that one fold's contrast is worth a closer
look before concluding Mamba is a strict upgrade.

---

## Open

- k-sweep needs an encoder-free rerun (`k=4..20`) before k-vs-full-mesh
  is a real same-model ablation — this run only fixed the full-mesh side.
- ~~Mamba: no matched run exists yet~~ — done, see above.
- ~~Mamba exact-0.0 val_loss pattern unexplained~~ — did not reproduce in
  the matched run; looks like a `smoke_test.py`-capped-split artifact,
  not a real Mamba training-path issue. Closed.
- `1_26_0` is the persistent k-of-n miss for GRU/encoder-in-graph variants
  specifically, but was a **hit** in this Mamba run — so "persistent
  across every model variant" (as stated in earlier notes) is now known
  to be false; it's GRU-specific, at least as of this result. Mamba's own
  miss was `1_18_0` instead. Worth watching whether that holds up across
  more Mamba runs (e.g. once the `use_cuda_kernel` RunPod path exists) or
  was a one-off.
- Mamba's `1_18_0` recall collapse (0.100 vs GRU's 0.767 on the same
  seizure, same fold) is the single biggest per-seizure divergence in
  this comparison — not investigated, worth a look if Mamba is pursued
  further (e.g. does that seizure's preictal window look unusually
  different in the CWT/dense-edge features Mamba is fed, vs whatever GRU
  is more robust to).
- Real Mamba speed comparison against `mamba-ssm`'s fused CUDA kernel
  still blocked on the RunPod image (`runpod_mamba_fast_image_brief.md`,
  unexecuted as of this writing).
