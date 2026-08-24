# Session notes — CWT node encoder, `node_only` ablation, fold 1_03_0 (2026-08-24)

Branch: `tf-node-encoding` (`2a5fbe9`). Repo:
`/Users/noahshore/Documents/CoherIQs/EEG_Benchmarks`. Machine: local
macOS, Apple Silicon, MPS. Subject chb01.

This is a **different encoder** from
[today's ChannelSignalEncoder removal](encoder_ablation_amp_and_23ch_onefold.md).
That note killed the unused raw-time `Conv1d` that was constructed, run,
and zeroed under `feature_ablation="zero_channel_embed"`. This note is
the learned **CWT time-frequency node encoder** added on `tf-node-encoding`
(`CWTTimeFrequencyNodeEncoder`), the `node_only` ablation, and one
uncapped prediction fold.

Follow-on to that encoder-removal note and to
[yesterday's 23ch 6-fold](../2026_08_23/full_6fold_subject1_run_results.md)
(`20260823-132639`).

---

## What was built

Learned per-channel CWT embeddings on top of the existing WCT dense-edge
GNN, not a replacement for WCT. Shared `Conv2d` (2→16→32, k=5, GELU)
over each channel's CWT real/imag, then `AdaptiveAvgPool2d(1)` +
`Linear(32, 8)`. Messages become `m_ij = f(h_i, h_j, e_ij)`.

It is a **flag**, not a pipeline. `--pipeline dense_edge_gru_tf_node` was
removed. `--cwt-encoder` (default off) works on `dense_edge` and
`dense_edge_gru`. Encoder-on CSVs nest under `cwt_encoder/` so they
cannot be globbed with WCT-only numbers.

Native 30s CWT is `[T=7680, F=8]`. Conv2d on that T OOMed / crawled, so
the encoder average-pools **time only** by 16 (same grain as
`dense_edge_time_downsample`) before the CNN. Frequency is not pooled
before the convs; the CNN then globally pools to one 8-d vector per
channel.

Default-off. Encoder-on still needs GPU-resident CWT every batch: dense-
edge cache hits cannot skip CWT, because the trainable encoder consumes
`w_real` / `w_imag`. That is the main epoch-time cost, not building the
fully-connected graph.

`channel_subset_k` still does **not** shrink `C`. CWT + the node encoder
always run on all 23 channels. `k` only limits how many WCT pairs are
computed. `k=0` / `k<=0` / `k>=C` / unset are all full mesh — not "no
edges."

---

## `node_only` ablation

User wanted CWT with no help from WCT. That is
`time_frequency_node_ablation="node_only"`, not `--channel-subset-k 0`.

```
python Epilepsy/run_pipelines.py --pipeline dense_edge_gru \
    --cwt-encoder-ablation node_only
```

Turns `--cwt-encoder` on. Messages are `f(h_i, h_j, 0)`. WCT is not
computed; `dense_edge_conv` / the edge GRU does not run. The complete
undirected mesh is still there (`E=253`): every pair still gets a
message, concat aggregation still fills a 23 × 22 × 8 readout. Topology
without edge features, not "no graph."

Results nest under `cwt_encoder/node_only/` so they are not pooled with
joint CWT+WCT runs.

---

## Fold `1_03_0` — CWT-only vs WCT vs Truong

Uncapped prediction, chb01, MPS, val 0.2, patience 5, 20-epoch cap,
`negative_to_positive_ratio=5.0`, no `--channel-subset-k` (full mesh
topology; WCT skipped because `node_only`).

```
python Epilepsy/run_pipelines.py --pipeline dense_edge_gru \
    --cwt-encoder-ablation node_only
```

`X=(4241, 23, 7680)`, 173 preictal. Fold 1 train: 3348 negatives → 725
(143 positives kept). 22 steps/epoch, batch 32.
`trainable_params=24579` (node encoder 13912). First-forward:
`cwt_node_encoding=841ms`, `graph_message_passing=37ms`.

Early stop epoch 8, restored epoch 3 (`val_loss=0.656752`,
`val_acc=0.8333`). Train ROC stayed ~0.50–0.53 all eight epochs.
`val_acc` bounced `0.17 ↔ 0.83` (always-positive vs always-negative on a
~16.7% positive val split). Epoch time ~9–10 s after the first
(~13.6 s).

Held-out `1_03_0`: `n_test=750` (30 preictal / 720 interictal),
precision=recall=f1=0, `auc_pr=0.032`, hit=False, FAR/h=0. Chance AP is
`30/750=0.040`. The restored checkpoint is a majority-class classifier.

Same fold, same task, WCT-only and Truong STFT+CNN (full interictal, not
smoke):

| | AP | ROC | prec / rec / f1 | hit raw → k-of-n | FAR/h raw → k-of-n |
|---|---|---|---|---|---|
| **CWT `node_only` (this run, MPS)** | **0.032** | ~0.5 train | 0 / 0 / 0 | F → F | 0 → 0 |
| WCT-only 23ch, val 0.2 (`20260824-104710`, CUDA) | **0.141** | 0.883 | 0.194 / 0.900 / 0.320 | T → T | 18.7 → 14.8 |
| WCT-only 23ch, no val, 20 ep (`20260823-132639`) | **0.108** | 0.831 | 0.038 / 0.067 / 0.049 | T → F | 8.3 → 2.2 |
| Truong STFT+CNN (`20260819-103235`) | **0.147** | 0.887 | 0.206 / 0.867 / 0.333 | T → T | 16.7 → 13.0 |

WCT and Truong both rank this fold (ROC ~0.83–0.89, AP well above 0.04).
CWT-encoder-only does not. One fold, MPS vs CUDA, not a bit-identical
twin — the gap is still chance vs a real ranking.

Unit tests still pass: encoder grads flow, `node_only` zeros the edge
slots, `_prepare_features` skips WCT. This is not an obviously
disconnected backward pass. The encoder *runs* (log: `node_embed=(32,
23, 8)`, 841 ms of CWT conv). Train ROC stuck at 0.5 means those 8-d
codes never produced a ranking, even on the 868-window train fold,
before early stopping killed the run.

---

## Not "Truong's view + a fancier classifier"

That was the hoped-for reading of `node_only`. The tensors are not the
same.

Truong: `[23, 59, 114]` STFT magnitude. Conv1 is **3-D** with kernel
`(n_channels, 5, 5)` — mixes **all electrodes on local time–frequency
patches** — then two 2-D convs, flatten, FC. The classifier still sees a
TF map. 114 frequency bins (~full spectrum minus line noise).

CWT `node_only`: 8 CWT bins, 8–40 Hz. Each channel encoded **alone**,
time-pooled by 16, two 5×5 convs, then **`AdaptiveAvgPool2d(1)` → 8
numbers**. After that there is no spectrogram. Those scalars go through
the WCT GNN's concat-of-253-pairs head with zeros in the edge slots.
That head was built to flatten WCT edge vectors (see the 2026-08-10
dense-mode concat work), not to classify spectrograms.

So: still a GNN, still 253 pairings, no WCT features. Not Truong. A
fair CWT-vs-Truong ablation would classify from the CWT map the way
Truong classifies from STFT (shared conv over time–frequency, channels
mixed in conv, no GNN). Not implemented; user said skip that.

---

## Can `--cwt-encoder` help the joint (CWT+WCT) model?

Unlikely, given this.

Default joint messages are `concat(h_i, h_j, e_ij)`. `e_ij` is the
WCT/GRU vector the WCT-only model already uses successfully. `h_i` is
the same 8-d global pool that could not rank train windows in
`node_only`. Concatenating a feature with no ranking signal onto a
feature that already works does not inject Truong-style spectrogram
information. The MLP can ignore the extra 16 input slots.

Two further reasons it probably cannot catch up:

1. **WCT is already a fast, good feature.** On this fold WCT-only gets
   AP ~0.14 / ROC ~0.88 and val loss drops quickly. Early stopping
   restores that checkpoint. The encoder is a randomly initialized CNN
   that would need many more steps to learn filters. By the time WCT has
   a good checkpoint, the encoder is still near init.
2. **The head is an edge readout.** Concat-of-253-pairs is the wrong
   place to use per-channel TF maps even if `h_i` had some information.

Joint vs WCT-only can still *move* numbers for non-CWT reasons: first
`Linear` of `sparse_message_mlp` goes `8 → 24` in, ~14k extra params,
different init. Any such delta is a capacity/init confound until proven
otherwise.

The expensive part of `--cwt-encoder` is recomputing CWT every batch so
those 8-d codes exist. WCT-only remains the working model.

---

## Smoke vs real (why encoder epochs looked cheap)

`smoke_test.py` defaults are not the real pipeline: `k=4`, 5 interictal
files (~800 windows), 1 fold, MPS cache on, both bf16 flags on.
`run_pipelines.py` defaults: full ~33 interictal files (`X=4241`),
`k=None` (253 live WCT edges unless `node_only`), all 6 folds, bf16 off
unless passed, disk cache **off on CUDA**. Encoder always pays 23-channel
CWT. `k` does not change that.

---

## Open

- Joint `--cwt-encoder` (CWT+WCT) 6-fold never run at real scale. Low
  priority given the above: expected to look like WCT-only plus noise /
  extra params.
- A Truong-like CWT spectrogram CNN (no GNN, no global-pool-to-8) was
  discussed as the fair "does CWT add anything" test and explicitly
  deferred.
- `node_only` is still a GNN on a featureless complete graph. A true
  "node embeddings only" readout (concat/mean of `h_i`, no pair MLP)
  was not built.
