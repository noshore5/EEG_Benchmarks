"""CPU-only correctness check for _DenseEdgeMambaContinuous (continuous-cwt-mamba
branch). Two claims to verify, both purely about mambapy's step() recurrence,
no GPU involved:

  1. Chunking is a pure memory/TBPTT tactic, not a numerical change: running
     one long sequence through forward() in one call must produce the exact
     same output as running it through several smaller chunked calls with
     the cache carried across them.
  2. mambapy's two scan entry points -- Mamba.forward() (parallel scan,
     always h=0) and Mamba.step() looped per-timestep (this class's
     mechanism) -- are two implementations of the SAME recurrence, so on a
     cache=None (state-reset) call they should agree with each other too.

Run: python scripts/dense_edge_mamba_continuous_parity.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Epilepsy.pipelines.cwt_gnn_classifiers import _DenseEdgeMambaContinuous  # noqa: E402

torch.manual_seed(0)
device = torch.device("cpu")

B, C_IN, E, T = 2, 8, 3, 20
D_MODEL, D_STATE, D_CONV, EXPAND = 6, 4, 3, 2

module = _DenseEdgeMambaContinuous(
    in_channels=C_IN, out_channels=4, d_model=D_MODEL, d_state=D_STATE,
    d_conv=D_CONV, expand=EXPAND, n_layers=1,
).to(device)
module.eval()  # dropout=0.0 anyway (nn.Identity), but be explicit

conv_in = torch.randn(B, C_IN, E, T, device=device)

with torch.no_grad():
    # Claim 1: one full-length call vs. chunked calls with carried cache.
    out_full, _ = module(conv_in)

    chunk_bounds = [0, 7, 14, 20]  # uneven chunk sizes on purpose: 7, 7, 6
    pieces = []
    cache = None
    for start, end in zip(chunk_bounds[:-1], chunk_bounds[1:]):
        piece, cache = module(conv_in[:, :, :, start:end], cache=cache)
        pieces.append(piece)
    out_chunked = torch.cat(pieces, dim=-1)

    max_diff = (out_full - out_chunked).abs().max().item()
    print(f"[chunking parity] max|full - chunked| = {max_diff:.3e}  shape={tuple(out_full.shape)}")
    assert torch.allclose(out_full, out_chunked, atol=1e-5), "chunking changed the forward value!"
    assert out_full.shape == (B, 4, E, T)

    # Claim 2: step()-driven recurrence (this class, cache=None) agrees with
    # mambapy's own parallel-scan forward() on the same weights/input.
    rows = B * E
    seq = conv_in.permute(0, 2, 3, 1).reshape(rows, T, C_IN)
    seq = module.in_proj(seq)
    pscan_out = module.mamba(seq)  # [rows, T, d_model] -- Mamba.forward(), h always starts at 0

    step_out = []
    cache2 = module.init_cache(rows, seq.device)  # Mamba.step() (unlike this class's own
    # forward()) has no cache=None convenience -- it does caches[i] directly, so it needs
    # an already-initialized list, not None.
    for t in range(T):
        y, cache2 = module.mamba.step(seq[:, t, :], cache2)
        step_out.append(y)
    step_out = torch.stack(step_out, dim=1)  # [rows, T, d_model]

    max_diff2 = (pscan_out - step_out).abs().max().item()
    print(f"[scan-vs-step parity] max|pscan - step| = {max_diff2:.3e}  shape={tuple(pscan_out.shape)}")
    assert torch.allclose(pscan_out, step_out, atol=1e-4), "step() recurrence disagrees with forward()'s scan!"

print("OK -- chunking is exact, and the step()-based continuous mechanism matches mambapy's own scan.")
