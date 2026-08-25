"""GPU-only, one-shot memory/timing probe for _DenseEdgeMambaContinuous at
REAL dense-edge scale (23-channel full mesh, E=253; real nfreqs=48 ->
C_in=4*48=192; real defaults: d_model=16, d_state=16, d_conv=4, expand=2,
dense_conv_out_channels=8 -- see cwt_gnn_classifiers.py's constructor
defaults). One forward+backward per T_chunk value, not a training loop --
just enough to get real peak-memory/wall-time numbers to replace the
back-of-envelope estimates from earlier in this session. Stops at the first
OOM. B=1 (one recording at a time, per this branch's current plan).

Run: python scripts/continuous_mamba_gpu_scale_probe.py
"""
import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Epilepsy.pipelines.cwt_gnn_classifiers import _DenseEdgeMambaContinuous  # noqa: E402

assert torch.cuda.is_available(), "no CUDA device visible"
device = torch.device("cuda")

E, C_IN, C_OUT = 253, 192, 8
D_MODEL, D_STATE, D_CONV, EXPAND = 16, 16, 4, 2
B = 1

torch.manual_seed(0)
model = _DenseEdgeMambaContinuous(
    in_channels=C_IN, out_channels=C_OUT, d_model=D_MODEL, d_state=D_STATE,
    d_conv=D_CONV, expand=EXPAND, n_layers=1,
).to(device)
model.train()

print(f"device: {torch.cuda.get_device_name(0)}")
print(f"E={E} C_in={C_IN} d_model={D_MODEL} d_state={D_STATE} expand={EXPAND} B={B}\n")
print(f"{'T_chunk':>8} {'fwd+bwd_s':>10} {'peak_mem_MiB':>13} {'free_MiB':>10}")

for t_chunk in (32, 64, 128, 256, 512, 1024, 2048):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    conv_in = torch.randn(B, C_IN, E, t_chunk, device=device)
    try:
        torch.cuda.synchronize()
        t0 = time.time()
        out_seq, cache = model(conv_in, cache=None)
        loss = out_seq.sum()
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        free_mib = torch.cuda.mem_get_info()[0] / (1024 ** 2)
        print(f"{t_chunk:8d} {dt:10.3f} {peak_mib:13.1f} {free_mib:10.1f}")
        model.zero_grad(set_to_none=True)
        del conv_in, out_seq, cache, loss
    except torch.cuda.OutOfMemoryError as e:
        print(f"{t_chunk:8d}  OOM: {str(e).splitlines()[0]}")
        torch.cuda.empty_cache()
        break

print("\ndone.")
