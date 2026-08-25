"""Memory/timing probe for _DenseEdgeMambaContinuous at real dense-edge
scale (23-channel full mesh, E=253; real nfreqs=48 -> C_in=4*48=192;
real defaults: d_model=16, d_state=16, d_conv=4, expand=2).

Compares the two scan implementations:
  scan="step"  -- original per-timestep Python loop (mambapy 1.2.0's only
                  carried-state API). The 2026-08-25 RTX 3070 Ti probe
                  measured ~2.68ms/timestep at B=1; that number is the
                  reason scan="chunk" exists.
  scan="chunk" -- default: Blelloch pscan with carried-in h (local
                  reimplementation of unreleased mambapy Mamba.chunk_step).

Also sweeps B (recordings stacked in the batch dim, i.e. the row-batching
CONTEXT.md asked about) on scan="chunk", to see whether occupancy still
moves the needle once the Python timestep loop is gone.

One forward+backward per (scan, B, T_chunk) cell, not a training loop.
CUDA if visible, else MPS, else CPU. Stops a T sweep at the first OOM.

Run: python scripts/continuous_mamba_gpu_scale_probe.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Epilepsy.pipelines.cwt_gnn_classifiers import _DenseEdgeMambaContinuous  # noqa: E402

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "mps"
else:
    device = torch.device("cpu")
    device_name = "cpu"

E, C_IN, C_OUT = 253, 192, 8
D_MODEL, D_STATE, D_CONV, EXPAND = 16, 16, 4, 2


def _sync():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_mem():
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _peak_mib():
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")


def _free_mib():
    if device.type == "cuda":
        return torch.cuda.mem_get_info()[0] / (1024 ** 2)
    return float("nan")


def _make(scan: str) -> _DenseEdgeMambaContinuous:
    torch.manual_seed(0)
    model = _DenseEdgeMambaContinuous(
        in_channels=C_IN, out_channels=C_OUT, d_model=D_MODEL, d_state=D_STATE,
        d_conv=D_CONV, expand=EXPAND, n_layers=1, scan=scan,
    ).to(device)
    model.train()
    return model


def _run_one(model, batch: int, t_chunk: int):
    conv_in = torch.randn(batch, C_IN, E, t_chunk, device=device)
    _sync()
    t0 = time.perf_counter()
    out_seq, cache = model(conv_in, cache=None)
    loss = out_seq.sum()
    loss.backward()
    _sync()
    dt = time.perf_counter() - t0
    peak = _peak_mib()
    free = _free_mib()
    model.zero_grad(set_to_none=True)
    del conv_in, out_seq, cache, loss
    return dt, peak, free


print(f"device: {device_name}")
print(
    f"E={E} C_in={C_IN} d_model={D_MODEL} d_state={D_STATE} expand={EXPAND}\n"
)

# --- 1. scan="step" vs scan="chunk" at B=1 (the original bottleneck) -----
print("=== B=1  scan=step vs scan=chunk (fwd+bwd) ===")
print(
    f"{'scan':>6} {'T_chunk':>8} {'fwd+bwd_s':>10} {'ms/step':>10} "
    f"{'peak_mem_MiB':>13} {'free_MiB':>10}"
)
step_ms = {}
chunk_ms = {}
# step() is the known-slow path -- keep T modest so a CPU/MPS box finishes.
# chunk is cheap; run it out to the original probe's T=1024/2048.
t_grid_step = (32, 64, 128)
t_grid_chunk = (32, 64, 128, 256, 512, 1024, 2048)
if device.type == "cuda":
    t_grid_step = (32, 64, 128, 256, 512)  # still slow; skip 1024+

for scan, t_grid in (("step", t_grid_step), ("chunk", t_grid_chunk)):
    model = _make(scan)
    # One tiny fwd+bwd so MPS/CUDA kernel compile isn't billed to T=32.
    try:
        _run_one(model, batch=1, t_chunk=min(t_grid))
        model.zero_grad(set_to_none=True)
    except Exception:
        pass
    for t_chunk in t_grid:
        _reset_mem()
        try:
            dt, peak, free = _run_one(model, batch=1, t_chunk=t_chunk)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # MPS/CPU raise RuntimeError on OOM, CUDA has its own class.
            msg = str(exc).splitlines()[0]
            if "out of memory" in msg.lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
                print(f"{scan:>6} {t_chunk:8d}  OOM: {msg[:80]}")
                _reset_mem()
                break
            raise
        ms_per = 1000.0 * dt / t_chunk
        print(
            f"{scan:>6} {t_chunk:8d} {dt:10.3f} {ms_per:10.3f} "
            f"{peak:13.1f} {free:10.1f}"
        )
        (step_ms if scan == "step" else chunk_ms)[t_chunk] = ms_per
    del model
    _reset_mem()

overlap = sorted(set(step_ms) & set(chunk_ms))
if overlap:
    print("\nchunk / step speedup at overlapping T (B=1):")
    for t in overlap:
        print(f"  T={t:4d}  {step_ms[t] / chunk_ms[t]:6.1f}x  "
              f"(step {step_ms[t]:.3f} ms/step, chunk {chunk_ms[t]:.3f} ms/step)")

# --- 2. row-batching (B>1) on the default scan, now that T is fused ------
print("\n=== scan=chunk  B-sweep (row-batching recordings) ===")
print(
    f"{'B':>4} {'T_chunk':>8} {'rows':>6} {'fwd+bwd_s':>10} {'ms/step':>10} "
    f"{'us/row-step':>12} {'peak_mem_MiB':>13}"
)
model = _make("chunk")
t_for_b = 256 if device.type != "cpu" else 64
for batch in (1, 2, 4, 8):
    _reset_mem()
    try:
        dt, peak, _ = _run_one(model, batch=batch, t_chunk=t_for_b)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        msg = str(exc).splitlines()[0]
        if "out of memory" in msg.lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
            print(f"{batch:4d} {t_for_b:8d}  OOM: {msg[:80]}")
            _reset_mem()
            break
        raise
    rows = batch * E
    ms_per = 1000.0 * dt / t_for_b
    us_row = 1e6 * dt / (t_for_b * rows)
    print(
        f"{batch:4d} {t_for_b:8d} {rows:6d} {dt:10.3f} {ms_per:10.3f} "
        f"{us_row:12.2f} {peak:13.1f}"
    )
del model
_reset_mem()

# Back-of-envelope: 1hr CHB recording at 256Hz / dense_edge_time_downsample=16
# -> T=3600*16=57600. At T_chunk=1024 that's 57 chunks.
if 1024 in chunk_ms:
    t_rec = 57600 * (chunk_ms[1024] / 1000.0)
    print(
        f"\n~1hr recording (T=57600) at B=1, T_chunk=1024, scan=chunk: "
        f"~{t_rec:.1f}s of Mamba fwd+bwd (this device; linear extrapolation)."
    )
elif overlap:
    t = overlap[-1]
    t_rec = 57600 * (chunk_ms[t] / 1000.0)
    print(
        f"\n~1hr recording (T=57600) at B=1, extrapolated from T={t} "
        f"scan=chunk: ~{t_rec:.1f}s of Mamba fwd+bwd (this device)."
    )

print("\ndone.")
