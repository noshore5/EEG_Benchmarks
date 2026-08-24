# Session notes — dead channel encoder removed, bf16 vs fp32, 23ch 1-fold (2026-08-24)

Branch: `main` (`c1c7278` plus uncommitted cache/CLI/encoder wiring). Repo:
`C:\Users\User\Documents\noshore5\EEG_Benchmarks`. Machine: local Windows
box, RTX 3070 Ti (8GB VRAM). Subject chb01.

Follow-on to
[today's k-sweep](k_sweep_channel_subset_cuda.md) and
[compact-cache note](windows_cuda_compact_dense_edge_cache_and_smoke_times.md).
The k-sweep and all epoch-time numbers in those notes still had
`ChannelSignalEncoder` in the graph (constructed, run, then zeroed). This
note is everything after that.

---

## Part 1 — Channel encoder was on, and it did nothing

`feature_ablation="zero_channel_embed"` has been locked since 2026-08-17
(any other value raises). `ChannelSignalEncoder` still ran every train and
predict batch: two dilated `Conv1d`s (kernel 9, dilation 5) over raw
`[B, 23, 7680]`, then `AdaptiveAvgPool1d(1)`. The embeddings were replaced
with `zeros_like` before `sparse_message_mlp`. `zeros_like` detaches, so
there was no gradient either.

The classifier actually uses CWT → WCT/coherence → `dense_edge_conv` (GRU)
→ graph aggregate. Channel identity was not in that path.

Two-step removal in `Epilepsy/pipelines/cwt_gnn_classifiers.py`:

1. Skip the forward (keep constructing the module so later-submodule RNG
   stayed bit-identical).
2. User: we do not care about matching old seeds. Stopped constructing
   it (`self.channel_encoder = None`) and shrunk `sparse_message_mlp` from
   `event_feature_dim + 2*channel_embed_dim` (24-in at dense/GRU) to
   **event features only (8-in)**. No zero-concat.

`channel_embed_dim` / `channel_encoder_dilation` remain accepted constructor
args so call sites do not change; they do nothing. `ChannelSignalEncoder`
class is still in the file.

**This morning's k-sweep weights will not load** into `sparse_message_mlp`
(shape change) and later layers get different RNG draws. New runs are a
new model. Dummy forward: encoder never called, perturbing its old weights
was n/a after (1); after (2) `mlp_in=8`, `total_params=10539`.

---

## Part 2 — Epoch time: smoke lied, uncapped k=4 / k=20 did not move

Smoke (`--device cuda --channel-subset-k 4`, both bf16, cache off) after
the removal: **3.44 / 4.27 s**, mean **3.85 s**, train wall 10.87 s. That
looked like a win vs ~10 s/epoch smoke from earlier in the session.

It was not. The ~10 s figure was smoke vs a **cache-bound** smoke, not vs
the k-sweep. User: "I was getting like 3.x seconds/epoch with k=4."

Same uncapped protocol as the sweep (`--validation-split 0.2`, both bf16,
`--max-folds 1`):

| | this morning (encoder in graph) | after removal |
|---|---|---|
| k=4, 22 steps/epoch | ~3.8–4.0 s | 4.80, **4.02, 3.95, 3.97 s** |
| k=20, 22 steps/epoch | ~12.1 s (12.81 / 12.03 / 12.16 / 12.14 / 12.38) | 14.07, **11.88, 12.18, 12.57 s** |

Steady state is the same. At k=4 with cache off the floor is **CWT on 23
channels every batch**. Live WCT is ~2.3 ms/trial. The encoder was not
that slice. At k=20 the extra ~8 s vs k=4 is live WCT (190 edges) plus the
253-edge GRU.

---

## Part 3 — bf16 vs fp32 (killed after 2/6 folds)

Matched k=20 6-fold, val 0.2, encoder still in the graph (process started
before the removal). fp32 = both `--dense-edge-amp-bf16` and
`--train-amp-bf16` omitted.

| | fp32 | bf16 k=20 sweep |
|---|---|---|
| s/epoch | ~33 | ~12 |
| `1_03_0` | T→T, FAR 20.8→15.3, prec 0.194 rec 1.000 f1 0.324 **AP 0.368** | same hits/FAR/prec/rec/f1, **AP 0.361** |
| `1_04_0` | T→T, FAR 31.4→16.3, prec 0.152 rec 0.967 f1 0.262 **AP 0.447** | same, **AP 0.437** |
| early stop | 10 best 5; 12 best 7 | same |

Thresholded metrics matched exactly. AP differed by ~0.01. User stopped
the run: smoke was ~10 s/epoch **either way**, and two folds were enough
to see scores do not move in a way that matters. Autocast only shows up
when WCT+GRU fill the GPU (uncapped k=20), not at smoke scale.

---

## Part 4 — VRAM: encoder-off does not buy a feature cache

Encoder weights were ~664 params. What went away was peak **activations**
(two dilated convs on `[32×23, 8, 7680]`, a few hundred MB), not a
persistent pool.

Keeping the train-fold dense-edge tensors on the 8GB card (so CWT/WCT do
not rerun every epoch) still does not fit at the k we care about. ~704
windows after 5:1 subsample:

| | per window | ×704 |
|---|---|---|
| full E=253 fp32 | 15.5 MB | ~11 GB |
| compact k=4 | 0.37 MB | ~260 MB |
| compact k=12 | ~4 MB | ~2.9 GB |
| compact k=20 fp32 | ~11.7 MB | ~8.2 GB |
| compact k=20 bf16 | ~5.8 MB | ~4.1 GB |

k=4 compact fits in VRAM or RAM easily. k=20 fp32 does not. Disk cache
already lost on this box because 15 MB `np.load` was slower than GPU
recompute; an **in-RAM compact live-edge cache** is the next real speed
lever, not a bigger batch and not encoder VRAM. Raising `batch_size` 32→64
without that cache just makes a bigger CWT batch.

---

## Part 5 — 23-channel 1-fold (encoder-free model)

Last ~10 min on the GPU. The hole from the k-sweep note: no full-mesh run
on val 0.2 + early stopping.

```
python Epilepsy/run_pipelines.py --device cuda --validation-split 0.2 \
    --dense-edge-amp-bf16 --train-amp-bf16 --max-folds 1
```

No `--channel-subset-k` → full 253-edge WCT. Encoder already gone.
Wall **307 s** (~5.1 min). Fold `1_03_0` only.

Early stop epoch **17**, best epoch **12** (`val_loss=0.095022`).
Epoch time **14.2–16.6 s** (first 14.85; later ~14.3).

| | 23ch this fold | k=20 sweep `1_03_0` | k=20 4-epoch after removal | yesterday 23ch `132639` `1_03_0` |
|---|---|---|---|---|
| model | encoder-free, 8-in MLP | encoder in graph, 24-in MLP | encoder-free, 4 epochs only | encoder in graph, no val, 20 ep |
| s/epoch | ~14.3 | ~12.1 | ~12.2 | ~13.8–15.4 |
| AP | 0.141 | 0.361 | 0.209 (4 ep) | 0.108 |
| prec / rec / f1 | 0.194 / 0.900 / 0.320 | 0.194 / 1.000 / 0.324 | 0.207 / 0.933 / 0.339 | 0.038 / 0.067 / 0.049 |
| FAR/h | 18.7 → 14.8 | 20.8 → 15.3 | 17.8 → 12.2 | 8.3 → 2.2 |
| hit | T → T | T → T | T → T | T → F |

**Timing** is the fair comparison: full mesh is ~2–4 s/epoch slower than
k=20 on this protocol (253 vs 190 live WCT edges; CWT is 23ch either way).
That sits next to yesterday's uncapped 23ch ~13.8–15.4 s (no val, all 20
epochs).

**AP is not a k-vs-23 ablation.** This fold is a different model than the
k-sweep. One fold is not a mean. Treat 0.141 as "1_03_0 on the encoder-free
full mesh," not as "23ch is worse than k=20."

CSV: `prediction_leave_one_seizure_out_20260824-104710.csv`.

---

## Open

- In-RAM compact live-edge cache for the fold (disk lost; VRAM only at
  small k).
- 23ch **6-fold** with val 0.2 + encoder-free MLP, so k=20 vs full mesh is
  actually the same model.
- k-sweep AP table was the encoder-in-graph model; needs a rerun if that
  table is going to be the encoder-free one.
