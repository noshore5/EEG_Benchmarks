# Less-engineered hermitian encoders + the anomaly-detection reframe (2026-08-30)

Mac shell. Follows `2026_08_29/mamba3_and_perfreq.md`. Started with the
temporal-model levers (mamba3, per_freq) and the k-axis already closed
negative; this session is about *representation* -- whether a
less-hand-engineered encoding holds the 0.436 line -- and a bigger
reframe of the task itself as graph-state anomaly detection.

## TL;DR

| experiment | status | result |
|---|---|---|
| k=6 -> k=12 (real Mamba) | DONE | **NEGATIVE** mean AP 0.390 vs 0.436, lost 5/6 folds |
| `encoder_mode="complex"` (`_ComplexSpectralEncoder`) | 3/6 folds, killed | **trailing** 0.291/0.318/0.312 vs 0.254/0.324/0.414 |
| `encoder_mode="matrix"` (`_ComplexMatrixEncoder`) | BUILT, queued | -- |
| `temporal_graph_edge_complex` (post + `_ComplexLinear`) | BUILT, queued | -- |
| `continuous_cwt_mamba` 6-fold (CPU) | queued | the abandoned run, finally |
| **`HermitianSSMAnomaly`** (interictal-only, next-token surprise) | BUILT, smoke PASSED, queued | **smoke: mean ROC-AUC 0.80, AP 0.50** (2 epochs, degraded cfg) |

## k=6 -> k=12

Rebuilt the spectral cache at k=12 (key `293ac41c6e53675f`, 20 GB; since
deleted). Clean A/B, real-Mamba backend, everything else the 0.436 config.
Result: mean AP **0.390 vs 0.436**, lost 5 of 6 folds, noticeably noisier
training. Modes 7-12 are estimation noise on ~30 preictal windows/fold.
k=6 is the right truncation; k=23 is off the table. k axis CLOSED.

## `encoder_mode="complex"` -- `_ComplexSpectralEncoder`

The user's actual objection to the `"eigenvector"` encoder was never
"phase tracking" -- it was `torch.cat([u.real, u.imag])` at the front:
"a more natural representation than flattening all the reals and imag to
one vector." A real `Linear(2C -> d)` on that concat has four free Re/Im
blocks; nothing ties them to complex multiplication.

`_ComplexSpectralEncoder` (hermitian_ssm_classifier.py): strips every
hand-built part. `_ComplexLinear` (two real `Linear`s, `W = W_re + i W_im`,
the four blocks *tied* to `Re(Wx)=W_re x_re - W_im x_im` etc. -- half the
params, "this is a complex number" as inductive bias) maps `u_r` C -> d;
scaled by `lambda_r`; split-GELU; second `_ComplexLinear`; mean-pool over
modes then frequencies; `view_as_real` flatten ONCE at the very end. No
`mode_id`, no normalised-Hz feature, no learned concat-fuse. 14,016 params
vs `_SpectralEncoder`'s 46,968 (3.35x fewer).

6-fold (real k=6 cache, real Mamba). Killed after 3 folds on the user's
call once it was clearly trailing:

| fold | complex | eigenvector baseline |
|---|---|---|
| 1_03 | 0.291 | 0.254 |
| 1_04 | 0.318 | 0.324 |
| 1_15 | 0.312 | 0.414 |
| mean(3) | 0.307 | 0.331 |

Every fold had a healthy internal curve (val_loss 0.08-0.25, val_auc
0.96-0.99) that again did not transfer. Fold 3 also missed the smoothed
detector (hit_smoothed=False). ~280-330 s/epoch (the `torch.complex` ops
cost more than the engineered path despite 3x fewer params).

Verdict: the de-engineered encoder is a *wash-to-slightly-worse*, same as
every other lever. It does NOT rescue the ceiling. Whether it's an
acceptable *methods* story (matches ~baseline with far less engineering)
is the user's call; on the number alone it's not a win.

## Still queued (2026-08-30, `scratchpad/queue_runner.sh` + `queue_runner2.sh`)

1. **post 0.639 reproduction** (`run_post.py`) -- clean same-machine
   baseline for judging (3).
2. **`encoder_mode="matrix"`** (`_ComplexMatrixEncoder`,
   hermitian_ssm_classifier.py) -- the rank-k Hermitian graph
   `A = sum_r lambda_r u_r u_r^H` that `encoder_mode="graph"` (0.169)
   feeds, but instead of flattening all C^2 entries into one real
   `Linear`, a complex channel map `W` contracts `M = W A W^H` (still
   Hermitian, `d_ch x d_ch`), computed as
   `sum_r lambda_r (W u_r)(W u_r)^H` -- no C x C matrix materialised.
   Tests: was `graph`'s 0.169 the flatten hack or the rank-k bottleneck?
3. **`temporal_graph_edge_complex`** (`run_edge_complex.py`) -- the "post"
   architecture (the graph-native 0.639 leader) with its
   `Linear(4*nfreqs -> edge_dim)` edge projection replaced by a
   `_ComplexLinear(nfreqs -> edge_dim)` over the per-frequency complex
   wavelet coherence `coh_f = (Re X + i Im X)/sqrt(|W_i|^2|W_j|^2)`. The
   de-engineered representation applied to the architecture that actually
   scores. Built genuinely last in `SparseEvidenceGNNCore.__init__` --
   flag-off RNG is bit-identical (verified).
4. **`continuous_cwt_mamba` 6-fold on CPU** -- the LOSO that was abandoned
   2026-08-26 on a CUDA "illegal memory access" during validation. This
   IS the planned "rerun on CPU for a clean traceback" step. Smoke first.
5. **`HermitianSSMAnomaly` 6-fold** -- see below.

## The reframe: seizure prediction as graph-state anomaly detection

User: "I'm thinking about reframing this model for detecting graph state
normality ... identify when it is in an abnormal state" / "what if we
only trained it on interictal windows, then see if it can find the
preictal?"

Rationale: every supervised lever has been capped by the ~30-preictal-
window data budget. Anomaly detection never learns preictal features --
it learns the abundant interictal graph-state manifold and flags
departures. No class balancing, no 5:1 negative subsampling, no preictal
labels in the loss. Most de-engineered framing available.

`Epilepsy/pipelines/hermitian_ssm_anomaly.py` -- `HermitianSSMAnomaly`:
- Shares the whole deterministic half with `hermitian_ssm_classifier`
  (cache key, `_WindowDataset`, every spectral encoder imported).
- `fit`: filter the window index to `label == 0` ONLY. Encoder -> token
  stream `[B,T,d]` -> `F.layer_norm` (no affine; kills the
  "collapse to a constant stream" shortcut once the target is detached)
  -> `_SeqMamba` (mambapy Mamba returning the FULL causal sequence, not
  last-pooled) -> `Linear` next-token predictor. Loss =
  `MSE(pred[:, :-1], tokens[:, 1:].detach())`. Early-stop on val MSE.
- After fit: score every training interictal window by mean next-token
  MSE, calibrate a sigmoid centred at the 90th percentile (so the
  argmax>0.5 operating point the LOSO hit/FAR uses = "more surprising
  than 90% of normal windows").
- `predict_proba`: per-window surprise -> z vs the interictal calib ->
  sigmoid -> `[1-p, p]`. Plugs into `leave_one_seizure_out_hermitian_ssm`
  unchanged (a wrapper swaps `rp.HermitianSSMClassifier`).

**Smoke (`run_anomaly.py --smoke --max-folds 2`, 2 epochs, degraded
4s-window / F_out=6 / 60 Hz config):**
- interictal filter works (dropped 145 preictal, trained on 384/96).
- next-token MSE trains: val 0.51 -> 0.24 in 2 epochs.
- **diagnostic: fold AP 0.572 / 0.437, mean ROC-AUC 0.80, mean AP 0.50**
  for separating preictal from interictal on the held-out seizures.
- hit/FAR 0/0 -- q90 threshold too conservative for the degraded smoke;
  AP/ROC are the real "is there signal" readout and they're positive.

This is the most encouraging early signal the hermitian direction has
produced. Real 6-fold (30 epochs, real config, `encoder_mode="complex"`)
is queued as job 5.

If the real run holds up: next steps are (a) carry SSM state across
windows (continuous, not per-window), (b) the raw 253-edge stream instead
of eigenpairs, (c) proper threshold calibration for the operational
hit/FAR numbers.

## Run discipline (user directive, now in CONTEXT.md + memory)

"running on my mac is free, so always run. if i dont want it i will stop
it." One night a session idled ~6h waiting for approval on the next run.
Rule: there must ALWAYS be a job queued/running; queue the next
experiment and start it, tell the user as a notification not a request.
`scratchpad/queue_runner.sh` is the mechanism; keep it topped up.

## Housekeeping

- Deleted the k=12 cache (`293ac41c6e53675f`, 20 GB, negative result) and
  the anomaly smoke cache (`8ca067d0cea13fd0`, 1.1 GB). 83 GB free.
- Uncommitted (commit once the queued results land): `run_pipelines.py`
  (`HERMITIAN_SSM_PARAMS` encoder_mode/mamba_backend/k comments),
  `hermitian_ssm_classifier.py` (+`_ComplexSpectralEncoder`,
  +`_ComplexMatrixEncoder`), `cwt_gnn_classifiers.py` (+`_ComplexLinear`,
  +`temporal_graph_edge_complex`), `hermitian_ssm_anomaly.py` (new).
  The old `_CG_MAMBANET_SHARED_PARAMS` other-shell edit is no longer in
  the tree.
