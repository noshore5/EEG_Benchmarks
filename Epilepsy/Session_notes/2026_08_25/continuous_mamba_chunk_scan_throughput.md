# Session notes — continuous Mamba: carried-state pscan (scan="chunk")

Branch: `continuous-cwt-mamba`. Machine: Mac (MPS + CPU) in this session;
the earlier ~2.68ms/timestep number this closes out was from the Windows
RTX 3070 Ti probe (`scripts/continuous_mamba_gpu_scale_probe.py` as of
the previous session).

Follow-on to the isolated `_DenseEdgeMambaContinuous` work already in
`CONTEXT.md` (state-carryover proof, TBPTT detach, step()/forward()
parity). That session's open item 1 was: **the Python `Mamba.step()` loop
is too slow at real-recording scale; measure whether row-batching (or
something else) fixes it before investing in the CHB-MIT pipeline.**

## What the bottleneck actually was

Pinned `mambapy==1.2.0` has two scan entry points:

- `Mamba.forward()` — Blelloch pscan, always starts `h=0`. This is what
  `_DenseEdgeMambaTemporal` uses. No way to feed in a carried state.
- `Mamba.step(x, caches)` — one timestep, explicit `(h, inputs)` cache.
  Built for autoregressive generation. This is what
  `_DenseEdgeMambaContinuous` used.

`mambapy` **main** grew a third, `Mamba.chunk_step` (commit `67000c9`,
2025-12-15): pscan over a whole chunk **with** carried-in `h0` injected
at timestep 0 (`BX[:,0] += deltaA[:,0] * h0`) and the conv1d's left
context taken from the `inputs` cache. It is **not in the 1.2.0 PyPI
wheel** this repo pins (1.2.0 is still the latest on PyPI).

So the previous session's "must loop `step()`" was true of 1.2.0's public
API, and that loop is what measured ~2.68ms/timestep on the 3070 Ti.

## What changed

`_DenseEdgeMambaContinuous` now defaults to `scan="chunk"`: a local
reimplementation of that unreleased `chunk_step` against 1.2.0 internals
(same ResidualBlock + mixer weights, same cache contract). `scan="step"`
keeps the original Python loop as a parity/ablation path.

Same truncated-BPTT detach as before. `scan="chunk"` does materialize the
pscan `[rows, T, d_inner, d_state]` intermediates Temporal's
`mamba_chunk_size` exists to bound — at B=1, E=253, T_chunk=1024 that is
~0.5GiB/tensor, still fine on 8GB. Batching many recordings together
reintroduces the rows×T scaling that OOM'd Temporal.

## Correctness

`tests/test_dense_edge_mamba_continuous.py` + the existing parity script
(claim 3 added):

- `scan="chunk"` vs `scan="step"`: max |diff| ~1e-7 (float32 noise),
  including carried cache across uneven chunks, n_layers=1 and 2.
- Fresh-cache `scan="chunk"` vs `Mamba.forward()`: **bit-exact** (same
  pscan, no step() loop involved).
- TBPTT: backward on chunk N does not populate chunk N-1's `input.grad`
  (`cache` is detached).
- Existing Temporal tests still pass (36 tests, 2.8s).

`python scripts/dense_edge_mamba_continuous_parity.py` OK.

## Throughput

Median of 3–4 repeats after warmup, real scale (E=253, C_in=192,
d_model=16, d_state=16, expand=2), fwd+bwd, B=1:

| device | scan | T | ms/timestep |
|---|---|---|---|
| CPU | step | 64 / 128 / 256 | 79.2 / 81.2 / 81.9 |
| CPU | chunk | 64 / 128 / 256 | 2.90 / 2.25 / 1.99 |
| MPS | step | 64 / 128 / 256 / 512 | 0.94 / 1.00 / 1.21 / 1.46 |
| MPS | chunk | 64 / 128 / 256 / 512 | 0.95 / 0.94 / 0.95 / 0.94 |

CPU: **~28–41×**. MPS: a wash at small T, **1.55× at T=512**, with
chunk's per-step cost flat in T (as a scan should be) and step's growing.
fwd-only on MPS is roughly equal (~0.31 vs ~0.34 ms/step) — MPS is cheap
per small kernel, so the original CUDA diagnosis (launch-bound `step()`
loop at 2.68ms/step on the 3070 Ti) is the one that matters, and it is
the same pscan Temporal already trains with on that GPU.

**CUDA 3070 Ti was not re-measured this session** (this shell is a Mac).
That's the one number still worth grabbing with
`python scripts/continuous_mamba_gpu_scale_probe.py` on the Windows box;
the probe now compares both scans and sweeps B.

~1hr CHB recording (256Hz / `dense_edge_time_downsample=16` → T=57600) at
B=1, extrapolated from MPS T=512 chunk: **~54s of Mamba fwd+bwd per
recording**. Not free, but no longer "minutes of Mamba alone, slower than
the windowed baseline."

## Row-batching: not the fix

The previous session asked whether stacking several recordings' rows
(B>1) would amortize `step()`'s per-timestep launch cost. On `scan="chunk"`
the T-loop is gone, so there is nothing left to amortize that way: a B
sweep on MPS at T=256 was linear in rows (~4–5 µs/row-step from B=1 to
B=8). B>1 is still available later for occupancy, and will reintroduce
the rows×T memory scaling — not a prerequisite for the data pipeline.

## Not done

The CHB-MIT / LOSO rewrite is still not started. Next is item 2 of the
ordered list in `CONTEXT.md`: whole-recording CWT, TBPTT chunk
boundaries, SPH/SOP window labels read off the continuous timeline via
`pool_continuous_edge_stream_to_windows`. Existing
`_build_windowed_dataset` / `leave_one_seizure_out_*` assume independent
per-window rows in `X`/`metadata` throughout — not a bolt-on.
