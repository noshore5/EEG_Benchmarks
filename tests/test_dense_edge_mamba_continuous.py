"""Unit tests for _DenseEdgeMambaContinuous (continuous-cwt-mamba branch).

Synthetic, no CHB-MIT -- same convention as test_dense_edge_mamba.py.
Covers the scan="chunk" default (carried-state pscan) against the original
scan="step" Python loop, TBPTT detach across chunks, and
pool_continuous_edge_stream_to_windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Epilepsy.pipelines.cwt_gnn_classifiers import (  # noqa: E402
    _DenseEdgeMambaContinuous,
    pool_continuous_edge_stream_to_windows,
)

pytest.importorskip("mambapy")

IN_CHANNELS = 8
OUT_CHANNELS = 4
N_EDGES = 3
BATCH = 2
KWARGS = dict(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    d_model=6,
    d_state=4,
    d_conv=3,
    expand=2,
    n_layers=1,
)


def _paired_modules(n_layers: int = 1, **extra):
    kwargs = dict(KWARGS, n_layers=n_layers, **extra)
    chunk = _DenseEdgeMambaContinuous(**kwargs, scan="chunk")
    step = _DenseEdgeMambaContinuous(**kwargs, scan="step")
    step.load_state_dict(chunk.state_dict())
    chunk.eval()
    step.eval()
    return chunk, step


def test_default_scan_is_chunk():
    mod = _DenseEdgeMambaContinuous(**KWARGS)
    assert mod.scan == "chunk"


def test_invalid_scan_rejected():
    with pytest.raises(ValueError, match="scan"):
        _DenseEdgeMambaContinuous(**KWARGS, scan="parallel")


def test_forward_shape():
    mod = _DenseEdgeMambaContinuous(**KWARGS)
    conv_in = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 11)
    out, cache = mod(conv_in)
    assert out.shape == (BATCH, OUT_CHANNELS, N_EDGES, 11)
    assert torch.isfinite(out).all()
    assert len(cache) == 1
    h, inputs = cache[0]
    rows = BATCH * N_EDGES
    assert h.shape == (rows, mod.config.d_inner, mod.config.d_state)
    assert inputs.shape == (rows, mod.config.d_inner, mod.config.d_conv - 1)
    assert not h.requires_grad
    assert not inputs.requires_grad


def test_chunk_scan_matches_step_scan():
    """The throughput path (pscan with h0 injected) is the same recurrence
    as looping mambapy's step() -- float32 noise only."""
    torch.manual_seed(0)
    chunk, step = _paired_modules()
    conv_in = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 20)
    with torch.no_grad():
        out_c, cache_c = chunk(conv_in)
        out_s, cache_s = step(conv_in)
    assert torch.allclose(out_c, out_s, atol=1e-5, rtol=1e-5)
    assert torch.allclose(cache_c[0][0], cache_s[0][0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(cache_c[0][1], cache_s[0][1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n_layers", [1, 2])
def test_chunk_scan_matches_step_scan_carried_across_uneven_chunks(n_layers):
    torch.manual_seed(1)
    chunk, step = _paired_modules(n_layers=n_layers)
    conv_in = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 20)
    bounds = [0, 7, 14, 20]
    with torch.no_grad():
        out_full, _ = chunk(conv_in)
        cache_c = cache_s = None
        pieces_c, pieces_s = [], []
        for start, end in zip(bounds[:-1], bounds[1:]):
            oc, cache_c = chunk(conv_in[:, :, :, start:end], cache=cache_c)
            os, cache_s = step(conv_in[:, :, :, start:end], cache=cache_s)
            pieces_c.append(oc)
            pieces_s.append(os)
        cat_c = torch.cat(pieces_c, dim=-1)
        cat_s = torch.cat(pieces_s, dim=-1)
    assert torch.allclose(cat_c, cat_s, atol=1e-5, rtol=1e-5)
    assert torch.allclose(cat_c, out_full, atol=1e-5, rtol=1e-5)


def test_fresh_chunk_scan_matches_mamba_forward_exactly():
    """cache=None + scan=chunk is the same pscan Mamba.forward() runs, so
    the pre-out_proj activations must agree bit-exactly (no step() loop
    involved)."""
    torch.manual_seed(2)
    mod = _DenseEdgeMambaContinuous(**KWARGS, scan="chunk")
    mod.eval()
    conv_in = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 16)
    rows = BATCH * N_EDGES
    seq = conv_in.permute(0, 2, 3, 1).reshape(rows, 16, IN_CHANNELS)
    seq = mod.in_proj(seq)
    with torch.no_grad():
        gold = mod.mamba(seq)
        got, _ = mod._chunk_step(seq, mod.init_cache(rows, seq.device))
    assert torch.equal(got, gold)


def test_tbptt_backward_does_not_cross_chunk_boundary():
    """Returned cache is detached: backward on chunk N must not populate
    chunk N-1's input.grad. If the detach were dropped, the carried h
    would chain the graphs and first.grad would be non-None here."""
    torch.manual_seed(3)
    mod = _DenseEdgeMambaContinuous(**KWARGS, scan="chunk")
    mod.train()
    first = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 6, requires_grad=True)
    _out1, cache = mod(first)
    assert cache[0][0].requires_grad is False
    second = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 6, requires_grad=True)
    out2, _ = mod(second, cache=cache)
    out2.sum().backward()
    assert second.grad is not None and second.grad.abs().sum() > 0
    assert first.grad is None


def test_gradients_flow_through_chunk_scan():
    torch.manual_seed(4)
    mod = _DenseEdgeMambaContinuous(**KWARGS, scan="chunk")
    conv_in = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 8, requires_grad=True)
    out, _ = mod(conv_in)
    out.sum().backward()
    assert conv_in.grad is not None and conv_in.grad.abs().sum() > 0
    param_grads = [p.grad for p in mod.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in param_grads)


def test_pool_last_is_final_timestep():
    stream = torch.randn(BATCH, OUT_CHANNELS, N_EDGES, 10)
    pooled = pool_continuous_edge_stream_to_windows(
        stream, [(0, 4), (4, 10)], pool="last"
    )
    assert pooled.shape == (2, BATCH, OUT_CHANNELS, N_EDGES, 1)
    assert torch.equal(pooled[0], stream[:, :, :, 3:4])
    assert torch.equal(pooled[1], stream[:, :, :, 9:10])


def test_pool_mean_averages_window():
    stream = torch.randn(BATCH, OUT_CHANNELS, N_EDGES, 6)
    pooled = pool_continuous_edge_stream_to_windows(stream, [(1, 4)], pool="mean")
    assert torch.allclose(pooled[0], stream[:, :, :, 1:4].mean(dim=-1, keepdim=True))


def test_pool_rejects_bad_bounds():
    stream = torch.randn(1, 2, 3, 5)
    with pytest.raises(ValueError, match="out of range"):
        pool_continuous_edge_stream_to_windows(stream, [(0, 6)])
    with pytest.raises(ValueError, match="pool"):
        pool_continuous_edge_stream_to_windows(stream, [(0, 2)], pool="max")
