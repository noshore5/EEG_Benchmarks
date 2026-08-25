"""Unit tests for the Mamba temporal backend (dense_edge_temporal_mode="mamba").

Synthetic, no CHB-MIT download -- same convention as test_tf_node_encoder.py.
Checks the architectural contract of _DenseEdgeMambaTemporal directly (the
exact [B, C_in, E, T] -> [B, out_channels, E, 1] shape contract
_DenseEdgeGRUTemporal / _build_dense_feature_conv already establish -- see
that class's docstring in cwt_gnn_classifiers.py), that
SparseEvidenceGNNCore/SparseEvidenceGNNClassifier wire
dense_edge_temporal_mode="mamba" through correctly end to end, that the
existing "conv"/"rnn" backends are unaffected by its addition, and that
--pipeline dense_edge_mamba is wired the same way "dense_edge"/
"dense_edge_gru" already are in run_pipelines.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Epilepsy.pipelines.cwt_gnn_classifiers import (  # noqa: E402
    SparseEvidenceGNNClassifier,
    SparseEvidenceGNNCore,
    _DenseEdgeMambaTemporal,
)

# mambapy is a pinned requirements.txt dependency (see that file's 2026-08-24
# comment / this session's note for why it -- not mamba-ssm -- was chosen),
# but importorskip keeps this file from hard-failing collection on an
# environment that hasn't run `pip install -r requirements.txt` yet.
pytest.importorskip("mambapy")

N_CHANNELS = 4
NFREQS = 4
HIDDEN = 8  # dense_conv_out_channels
N_CLASSES = 2
BATCH = 2
N_EDGES = N_CHANNELS * (N_CHANNELS - 1) // 2
IN_CHANNELS = 4 * NFREQS  # frequency folded into C_in, same convention as GRU/conv


def _tiny_core(**overrides) -> SparseEvidenceGNNCore:
    kwargs = dict(
        n_channels=N_CHANNELS,
        nfreqs=NFREQS,
        n_classes=N_CLASSES,
        hidden_dim=HIDDEN,
        event_mode="dense",
        event_aggregation="concat",
        dense_edge_temporal_mode="mamba",
        dense_conv_out_channels=HIDDEN,
        mamba_d_model=8,
        mamba_d_state=8,
        mamba_d_conv=2,
        mamba_expand=2,
        mamba_n_layers=1,
        mamba_chunk_size=4,  # exercise the chunking path at this toy scale
        cwt_encoder=False,
        sampling_rate=64,
        n_hops=1,
    )
    kwargs.update(overrides)
    return SparseEvidenceGNNCore(**kwargs)


def _synthetic_dense_edge(n_time: int, n_channels: int = N_CHANNELS) -> torch.Tensor:
    n_edges = n_channels * (n_channels - 1) // 2
    return torch.randn(BATCH, 4, n_edges, n_time, NFREQS)


# ---------------------------------------------------------------------------
# 1. Module construction
# ---------------------------------------------------------------------------

def test_mamba_temporal_module_constructs():
    mod = _DenseEdgeMambaTemporal(
        in_channels=IN_CHANNELS, out_channels=HIDDEN,
        d_model=8, d_state=8, d_conv=2, expand=2, n_layers=1,
    )
    assert isinstance(mod, nn.Module)
    assert mod.in_channels == IN_CHANNELS
    assert mod.out_channels == HIDDEN
    assert mod.d_model == 8


def test_mamba_temporal_module_projects_when_dims_differ():
    mod = _DenseEdgeMambaTemporal(in_channels=IN_CHANNELS, out_channels=HIDDEN, d_model=8)
    assert isinstance(mod.in_proj, nn.Linear)
    assert mod.in_proj.in_features == IN_CHANNELS
    assert mod.in_proj.out_features == 8


def test_mamba_temporal_module_identity_projection_when_dims_match():
    mod = _DenseEdgeMambaTemporal(in_channels=8, out_channels=HIDDEN, d_model=8)
    assert isinstance(mod.in_proj, nn.Identity)


def test_mamba_temporal_chunking_matches_full_batch():
    """2026-08-24 CUDA-OOM finding -- mamba_chunk_size is a memory tactic
    only (splits the B*E leading dim into groups run through self.mamba
    separately, see that param's docstring). Chunked and full-batch calls
    on the SAME weights must agree, same contract
    test_encoder_chunking_matches_full_batch already establishes for
    CWTTimeFrequencyNodeEncoder's own chunk_size."""
    torch.manual_seed(0)
    chunked = _DenseEdgeMambaTemporal(
        in_channels=IN_CHANNELS, out_channels=HIDDEN, d_model=8, d_conv=2,
        n_layers=1, chunk_size=2,
    )
    full = _DenseEdgeMambaTemporal(
        in_channels=IN_CHANNELS, out_channels=HIDDEN, d_model=8, d_conv=2,
        n_layers=1, chunk_size=4096,
    )
    full.load_state_dict(chunked.state_dict())
    chunked.eval()
    full.eval()
    conv_in = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 8)  # B*E=8 rows > chunk_size=2
    with torch.no_grad():
        out_chunked = chunked(conv_in)
        out_full = full(conv_in)
    assert torch.allclose(out_chunked, out_full, atol=1e-5, rtol=1e-5)


def test_mamba_temporal_module_dropout_is_noop_by_default():
    mod = _DenseEdgeMambaTemporal(in_channels=IN_CHANNELS, out_channels=HIDDEN, d_model=8)
    assert isinstance(mod.dropout, nn.Identity)
    mod_dropout = _DenseEdgeMambaTemporal(
        in_channels=IN_CHANNELS, out_channels=HIDDEN, d_model=8, dropout=0.3
    )
    assert isinstance(mod_dropout.dropout, nn.Dropout)


# ---------------------------------------------------------------------------
# 2/3/4. Forward pass, tensor shape, different sequence lengths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_time", [4, 8, 16])
def test_mamba_temporal_forward_shape_across_sequence_lengths(n_time):
    mod = _DenseEdgeMambaTemporal(
        in_channels=IN_CHANNELS, out_channels=HIDDEN,
        d_model=8, d_state=8, d_conv=2, n_layers=1,
    )
    conv_in = torch.randn(BATCH, IN_CHANNELS, N_EDGES, n_time)
    out = mod(conv_in)
    assert out.shape == (BATCH, HIDDEN, N_EDGES, 1)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 5. Batch size > 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_mamba_temporal_forward_batch_sizes(batch_size):
    mod = _DenseEdgeMambaTemporal(
        in_channels=IN_CHANNELS, out_channels=HIDDEN, d_model=8, d_conv=2, n_layers=1,
    )
    conv_in = torch.randn(batch_size, IN_CHANNELS, N_EDGES, 6)
    out = mod(conv_in)
    assert out.shape == (batch_size, HIDDEN, N_EDGES, 1)


def test_mamba_temporal_rejects_wrong_in_channels():
    mod = _DenseEdgeMambaTemporal(in_channels=IN_CHANNELS, out_channels=HIDDEN, d_model=8)
    bad = torch.randn(BATCH, IN_CHANNELS + 1, N_EDGES, 8)
    with pytest.raises(AssertionError):
        mod(bad)


# ---------------------------------------------------------------------------
# 6. Gradient propagation
# ---------------------------------------------------------------------------

def test_mamba_temporal_gradients_flow():
    torch.manual_seed(0)
    mod = _DenseEdgeMambaTemporal(
        in_channels=IN_CHANNELS, out_channels=HIDDEN, d_model=8, d_conv=2, n_layers=1,
    )
    conv_in = torch.randn(BATCH, IN_CHANNELS, N_EDGES, 8, requires_grad=True)
    out = mod(conv_in)
    out.sum().backward()
    assert conv_in.grad is not None and conv_in.grad.abs().sum() > 0
    param_grads = [p.grad for p in mod.parameters() if p.requires_grad]
    assert param_grads, "mamba temporal block has no trainable parameters"
    assert any(g is not None and g.abs().sum() > 0 for g in param_grads), (
        "no gradient reached the mamba temporal block's own parameters"
    )


# ---------------------------------------------------------------------------
# SparseEvidenceGNNCore / SparseEvidenceGNNClassifier wiring
# ---------------------------------------------------------------------------

def test_core_accepts_mamba_temporal_mode():
    core = _tiny_core()
    assert core.dense_edge_temporal_mode == "mamba"
    assert isinstance(core.dense_edge_conv, _DenseEdgeMambaTemporal)


def test_core_forward_with_mamba_backend_end_to_end():
    torch.manual_seed(1)
    core = _tiny_core()
    core.train()
    raw_x = torch.randn(BATCH, N_CHANNELS, 32)
    dense_edge = _synthetic_dense_edge(n_time=8)
    logits, _ = core(raw_x, dense_edge)
    assert logits.shape == (BATCH, N_CLASSES)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    grads = [p.grad for p in core.dense_edge_conv.parameters() if p.requires_grad]
    assert grads
    assert any(g is not None and g.abs().sum() > 0 for g in grads), (
        "mamba dense_edge_conv received no gradient from the full forward pass"
    )


def test_invalid_dense_edge_temporal_mode_rejected():
    with pytest.raises(ValueError, match="dense_edge_temporal_mode"):
        _tiny_core(dense_edge_temporal_mode="not-a-real-mode")


def test_mamba_mode_requires_event_mode_dense():
    with pytest.raises(ValueError, match="dense_edge_temporal_mode"):
        SparseEvidenceGNNCore(
            n_channels=N_CHANNELS, nfreqs=NFREQS, n_classes=N_CLASSES,
            hidden_dim=HIDDEN, event_mode="sparse",
            dense_edge_temporal_mode="mamba",
        )


def test_shuffle_time_order_rejected_for_mamba():
    """shuffle_time_order is 'rnn's own negative control -- not yet
    implemented for 'mamba' (see _DenseEdgeMambaTemporal's docstring)."""
    with pytest.raises(ValueError, match="shuffle_time_order"):
        _tiny_core(shuffle_time_order=True)


def test_classifier_forwards_mamba_hyperparameters():
    clf = SparseEvidenceGNNClassifier(
        sampling_rate=64, lowest=8.0, highest=24.0, nfreqs=NFREQS,
        hidden_dim=HIDDEN, event_mode="dense", event_aggregation="concat",
        dense_edge_temporal_mode="mamba",
        mamba_d_model=8, mamba_d_state=8, mamba_d_conv=2, mamba_expand=2,
        mamba_n_layers=1, mamba_dropout=0.1, mamba_chunk_size=3,
        cwt_backend="torch", device="cpu", verbose=0,
        dense_edge_cache_dir=None, cwt_cache=None, normalize_input=False,
        epochs=1, batch_size=2,
    )
    model = clf._build_model(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    assert isinstance(model.dense_edge_conv, _DenseEdgeMambaTemporal)
    assert model.dense_edge_conv.d_model == 8
    assert model.dense_edge_conv.chunk_size == 3
    assert isinstance(model.dense_edge_conv.dropout, nn.Dropout)


# ---------------------------------------------------------------------------
# 7. Existing GRU/Conv backends still work unchanged (regression)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["conv", "rnn", "mamba"])
def test_all_three_dense_edge_backends_share_the_same_contract(mode):
    core = _tiny_core(dense_edge_temporal_mode=mode, mamba_d_conv=2)
    core.eval()
    raw_x = torch.randn(BATCH, N_CHANNELS, 32)
    # n_time=64: large enough for "conv"'s default two-stage Conv2d(kernel=5)
    # + MaxPool2d(pool=4) stack to have something left to pool at the end
    # (this is a pre-existing constraint of that backend, unrelated to
    # "rnn"/"mamba" -- both of those accept any n_time >= 1).
    dense_edge = _synthetic_dense_edge(n_time=64)
    with torch.no_grad():
        logits, _ = core(raw_x, dense_edge)
    assert logits.shape == (BATCH, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_gru_and_conv_backends_unaffected_by_mamba_addition():
    """dense_edge_temporal_mode='conv'/'rnn' construction is still exactly
    _build_dense_feature_conv / _DenseEdgeGRUTemporal -- adding 'mamba' as a
    third branch must not have touched either existing path."""
    from Epilepsy.pipelines.cwt_gnn_classifiers import _DenseEdgeGRUTemporal

    conv_core = _tiny_core(dense_edge_temporal_mode="conv")
    assert isinstance(conv_core.dense_edge_conv, nn.Sequential)

    rnn_core = _tiny_core(dense_edge_temporal_mode="rnn")
    assert isinstance(rnn_core.dense_edge_conv, _DenseEdgeGRUTemporal)


# ---------------------------------------------------------------------------
# CLI / config-convention wiring (run_pipelines.py)
# ---------------------------------------------------------------------------

def test_dense_edge_mamba_is_a_pipeline_choice():
    from Epilepsy.run_pipelines import (
        _build_argument_parser,
        _dense_family_params,
        _dense_family_result_dir,
    )

    parser = _build_argument_parser()
    pipeline_action = next(
        action for action in parser._actions if "--pipeline" in action.option_strings
    )
    assert set(pipeline_action.choices) == {
        "dense_edge_gru", "dense_edge", "dense_edge_mamba", "truong_stft_cnn",
    }

    for label_mode in ("prediction", "detection"):
        params = _dense_family_params("dense_edge_mamba", label_mode)
        assert params["dense_edge_temporal_mode"] == "mamba"
        # Same training dynamics as the GRU dict for that label_mode --
        # this must stay an isolated temporal-backend ablation (see
        # DENSE_EDGE_MAMBA_PARAMS's own module-level comment).
        gru_params = _dense_family_params("dense_edge_gru", label_mode)
        for key in ("batch_size", "learning_rate", "validation_split", "hidden_dim", "nfreqs"):
            assert params[key] == gru_params[key], key

    root = Path("/tmp/results")
    mamba_pred = _dense_family_result_dir(root, "dense_edge_mamba", "prediction", False)
    mamba_det = _dense_family_result_dir(root, "dense_edge_mamba", "detection", False)
    assert mamba_pred == root / "dense_edge_mamba" / "prediction"
    assert mamba_det == root / "dense_edge_mamba"
    gru_pred = _dense_family_result_dir(root, "dense_edge_gru", "prediction", False)
    assert mamba_pred != gru_pred  # never pooled with the GRU CSVs
