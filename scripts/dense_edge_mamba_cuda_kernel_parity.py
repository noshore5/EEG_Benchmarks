"""CUDA-only numerical check: _DenseEdgeMambaTemporal with use_cuda_kernel=True
(mamba-ssm fused scan) vs False (mambapy pscan) on the same weights.

This is a genuinely different code path, not the "same math, different
memory layout" guarantee mamba_chunk_size has. Run inside the
Dockerfile.mamba image (or any Linux/CUDA box with mamba-ssm installed)
before trusting any kernel-vs-pscan speed/accuracy comparison.

Run: python scripts/dense_edge_mamba_cuda_kernel_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Epilepsy.pipelines.cwt_gnn_classifiers import (  # noqa: E402
    _DenseEdgeMambaTemporal,
    _mamba_ssm_importable,
)

assert torch.cuda.is_available(), "no CUDA device visible"
assert _mamba_ssm_importable(), (
    "mamba-ssm is not importable -- this script is for the Dockerfile.mamba "
    "image (or a Linux/CUDA box with `pip install --no-build-isolation mamba-ssm`)."
)

torch.manual_seed(0)
device = torch.device("cuda")
B, C_IN, E, T, C_OUT = 2, 32, 6, 48, 8

kernel = _DenseEdgeMambaTemporal(
    in_channels=C_IN, out_channels=C_OUT, d_model=16, d_state=16,
    d_conv=4, expand=2, n_layers=1, chunk_size=128, use_cuda_kernel=True,
).to(device)
pscan = _DenseEdgeMambaTemporal(
    in_channels=C_IN, out_channels=C_OUT, d_model=16, d_state=16,
    d_conv=4, expand=2, n_layers=1, chunk_size=128, use_cuda_kernel=False,
).to(device)
pscan.load_state_dict(kernel.state_dict())
kernel.eval()
pscan.eval()

conv_in = torch.randn(B, C_IN, E, T, device=device)
with torch.no_grad():
    out_k = kernel(conv_in)
    out_p = pscan(conv_in)
max_diff = (out_k - out_p).abs().max().item()
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"max|kernel - pscan| = {max_diff:.3e}  shape={tuple(out_k.shape)}")
# Fused kernel vs PyTorch pscan is not bit-exact; 1e-3 is the "same
# recurrence" bar, not the 1e-5 mamba_chunk_size bar.
assert torch.allclose(out_k, out_p, atol=1e-3, rtol=1e-3), (
    f"kernel and pscan disagree by {max_diff:.3e} -- too far to treat as "
    "the same model."
)
print("OK -- fused kernel matches mambapy pscan within 1e-3.")
