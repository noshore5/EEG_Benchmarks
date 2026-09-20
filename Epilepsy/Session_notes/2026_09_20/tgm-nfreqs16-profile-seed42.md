# tgm-nfreqs16-profile-seed42 — per-step timing breakdown (EEG_BENCHMARKS_PROFILE_STEPS=1)

1 fold, 5 epochs, nfreqs=16, complex-native, same command as
`ssmverify`/`chunk2-seed42` otherwise. Purpose: answer directly "why
isn't nfreqs=8 twice as fast as nfreqs=16" instead of guessing at
cache-size knobs (both nfreqs=8 and nfreqs=16 already hit 100% dense-edge
cache reuse, so cache residency was never the bottleneck).

Instance: g5.xlarge, `i-0dc74a97e83355ec3`.

## Steady-state per-epoch timing (epochs 2-5, cache fully warm)

```
[Train][Epoch 2/5] step timing: cwt=0.838s dense_edge=0.000s forward_total=0.968s
  (of which edge_to_node=1.014s mamba=0.177s) backward=1.052s optimizer=0.028s
  accounted=2.886s epoch_time=3.706s
```

(Epoch 1 is an outlier: dense_edge=49.9s, epoch_time=55.6s — that's the
one-time cache-build cost, not representative of steady state.)

## Answer: why nfreqs halving doesn't halve epoch time

| stage | time | share | scales with nfreqs? |
|---|---|---|---|
| cwt front-end | 0.838s | 23% | no -- recomputed per step regardless |
| edge_to_node projection | 1.014s | 27% | only weakly |
| **mamba forward** | **0.177s** | **5%** | **yes -- the actual F-dependent piece** |
| backward pass | 1.052s | 28% | no -- backprop through the whole net |
| optimizer step | 0.028s | 1% | no -- fixed |
| dense_edge cache lookup | 0.000s | 0% | no -- 100% cache hit, pure memory read |

`nfreqs` only widens the dense-edge tensor's last dim, which feeds the
`mamba` SSM forward pass and (weakly) `edge_to_node`. That's ~5-30% of
the per-step budget, not the whole thing. cwt, the backward pass, and the
optimizer step are fixed overhead independent of frequency resolution.
Halving nfreqs 16->8 can only ever shrink that 5-30% slice, which lines
up with the observed real-world numbers: nfreqs=8 measured 2.81s/epoch
mean vs nfreqs=16's 3.46-3.7s -- about a 24% speedup, not 2x.

## Auto-push: same SSM failure, but undiagnosed (stale commit)

`could not read deploy key from SSM` again -- but this box launched at
commit `4155336`, *before* the stderr-capture fix (`25f22d4`), so this
occurrence is still diagnostically blind. The next spot run launched
after `25f22d4` will be the first to show the real AWS error.
