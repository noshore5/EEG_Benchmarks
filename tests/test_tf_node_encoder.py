"""Unit tests for the learned CWT time-frequency node encoder.

These are synthetic: they do not download CHB-MIT. They check the
architectural contract of the `tf-node-encoding` branch:

  * CWT real/imag reach a shared per-channel Conv2d encoder
  * node embeddings are [B, C, D] and channel-isolated before message passing
  * encoder weights are shared across electrodes
  * messages see (h_src, h_dst, e_ij)
  * gradients flow through both the new node encoder and the existing edge GRU
  * the WCT-only baseline (cwt_encoder=False) is unchanged
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
    CWTTimeFrequencyNodeEncoder,
    SparseEvidenceGNNClassifier,
    SparseEvidenceGNNCore,
)


N_CHANNELS = 4
NFREQS = 4
HIDDEN = 8
N_CLASSES = 2
CWT_T = 32
EDGE_T = 8
BATCH = 2


def _tiny_core(**overrides) -> SparseEvidenceGNNCore:
    kwargs = dict(
        n_channels=N_CHANNELS,
        nfreqs=NFREQS,
        n_classes=N_CLASSES,
        hidden_dim=HIDDEN,
        event_mode="dense",
        event_aggregation="concat",
        dense_edge_temporal_mode="rnn",
        dense_conv_out_channels=HIDDEN,
        cwt_encoder=False,
        sampling_rate=64,
        n_hops=1,
    )
    kwargs.update(overrides)
    return SparseEvidenceGNNCore(**kwargs)


def _synthetic_cwt(n_channels: int = N_CHANNELS) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel CWT with known different time-frequency content.

    Channel i has a tone at frequency-bin (i % nfreqs) and a unique
    amplitude, so embeddings must differ across channels if the encoder
    is actually looking at the CWT (and not mixing channels).
    """
    t = torch.linspace(0.0, 1.0, CWT_T)
    freqs = torch.arange(1, NFREQS + 1, dtype=torch.float32)
    w_real = torch.zeros(BATCH, n_channels, CWT_T, NFREQS)
    w_imag = torch.zeros(BATCH, n_channels, CWT_T, NFREQS)
    for ch in range(n_channels):
        f_idx = ch % NFREQS
        amp = float(ch + 1)
        phase = 0.4 * ch
        w_real[:, ch, :, f_idx] = amp * torch.cos(2 * torch.pi * freqs[f_idx] * t + phase)
        w_imag[:, ch, :, f_idx] = amp * torch.sin(2 * torch.pi * freqs[f_idx] * t + phase)
    return w_real, w_imag


def _synthetic_dense_edge() -> torch.Tensor:
    n_edges = N_CHANNELS * (N_CHANNELS - 1) // 2
    return torch.randn(BATCH, 4, n_edges, EDGE_T, NFREQS)


def _n_params(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def test_encoder_output_shape_and_weight_sharing() -> None:
    enc = CWTTimeFrequencyNodeEncoder(embed_dim=HIDDEN)
    w_real, w_imag = _synthetic_cwt()
    out = enc(w_real, w_imag)
    assert out.shape == (BATCH, N_CHANNELS, HIDDEN)

    # Same weights applied independently: encoding channel 0 alone matches
    # channel 0 of the batched-all-channels call.
    out_ch0 = enc(w_real[:, 0:1], w_imag[:, 0:1])
    assert torch.allclose(out[:, 0], out_ch0[:, 0], atol=1e-6)

    # A single module -- not one network per electrode.
    convs = [m for m in enc.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 2
    assert convs[0].in_channels == 2
    assert convs[0].out_channels == 16
    assert convs[1].out_channels == 32


def test_encoder_channel_isolation() -> None:
    enc = CWTTimeFrequencyNodeEncoder(embed_dim=HIDDEN)
    enc.eval()
    w_real, w_imag = _synthetic_cwt()
    with torch.no_grad():
        base = enc(w_real, w_imag)
        w_real_a = w_real.clone()
        w_real_a[:, 0] = w_real_a[:, 0] + 1.5
        perturbed = enc(w_real_a, w_imag)

    assert not torch.allclose(base[:, 0], perturbed[:, 0], atol=1e-6), (
        "channel 0 embedding must change when only channel 0's CWT changes"
    )
    assert torch.allclose(base[:, 1], perturbed[:, 1], atol=1e-6), (
        "channel 1 embedding must be unchanged when only channel 0's CWT changes "
        "(encoder must not mix channels)"
    )
    assert torch.allclose(base[:, 2], perturbed[:, 2], atol=1e-6)
    assert torch.allclose(base[:, 3], perturbed[:, 3], atol=1e-6)


def test_encoder_chunking_matches_full_batch() -> None:
    """Microbatching must be a memory tactic, not a numerical change."""
    torch.manual_seed(0)
    chunked = CWTTimeFrequencyNodeEncoder(
        embed_dim=HIDDEN, chunk_size=2, time_downsample=1
    )
    full = CWTTimeFrequencyNodeEncoder(
        embed_dim=HIDDEN, chunk_size=1024, time_downsample=1
    )
    full.load_state_dict(chunked.state_dict())
    chunked.eval()
    full.eval()
    w_real, w_imag = _synthetic_cwt()
    with torch.no_grad():
        assert torch.allclose(
            chunked(w_real, w_imag), full(w_real, w_imag), atol=1e-5, rtol=1e-5
        )


def test_encoder_time_downsample_does_not_mix_channels() -> None:
    enc = CWTTimeFrequencyNodeEncoder(embed_dim=HIDDEN, time_downsample=4)
    enc.eval()
    w_real, w_imag = _synthetic_cwt()
    with torch.no_grad():
        base = enc(w_real, w_imag)
        w_real_a = w_real.clone()
        w_real_a[:, 0] = w_real_a[:, 0] + 1.5
        perturbed = enc(w_real_a, w_imag)
    assert not torch.allclose(base[:, 0], perturbed[:, 0], atol=1e-6)
    assert torch.allclose(base[:, 1], perturbed[:, 1], atol=1e-6)


def test_encoder_uses_real_and_imag_as_ordinary_channels() -> None:
    enc = CWTTimeFrequencyNodeEncoder(embed_dim=HIDDEN)
    w_real, w_imag = _synthetic_cwt()
    with torch.no_grad():
        base = enc(w_real, w_imag)
        imag_flipped = enc(w_real, -w_imag)
    assert not torch.allclose(base, imag_flipped, atol=1e-5), (
        "encoder must actually consume the imaginary CWT channel"
    )


def test_baseline_core_has_no_node_encoder_and_edge_only_message() -> None:
    core = _tiny_core(cwt_encoder=False)
    assert core.cwt_node_encoder is None
    assert core.channel_encoder is None
    first = core.sparse_message_mlp[0]
    assert isinstance(first, nn.Linear)
    assert first.in_features == HIDDEN  # dense_conv_out_channels only


def test_cwt_encoder_core_widens_message_mlp() -> None:
    core = _tiny_core(cwt_encoder=True)
    assert core.cwt_node_encoder is not None
    first = core.sparse_message_mlp[0]
    assert isinstance(first, nn.Linear)
    assert first.in_features == HIDDEN + 2 * HIDDEN  # e_ij + h_src + h_dst


def test_cwt_encoder_parameter_count_stays_small() -> None:
    baseline = _tiny_core(cwt_encoder=False)
    new = _tiny_core(cwt_encoder=True)
    n_base = _n_params(baseline)
    n_new = _n_params(new)
    n_enc = _n_params(new.cwt_node_encoder)
    # Encoder is Conv2d(2,16,5)+Conv2d(16,32,5)+Linear(32,8) ≈ 13.9k.
    # It is shared across channels, so this cost does not grow with C --
    # relative to a 4-node toy graph it dominates; relative to the 23-node
    # smoke model (~11k) it is about +14k. Cap the whole toy model well
    # below anything transformer/ResNet-sized.
    assert 13_000 <= n_enc <= 15_000, n_enc
    mlp_delta = n_new - n_base - n_enc
    assert 0 < mlp_delta < 500, mlp_delta  # first Linear widens by 2*D
    assert n_new > n_base
    assert n_new < 25_000, (n_base, n_new, n_enc)


def test_node_and_edge_gradients_flow() -> None:
    torch.manual_seed(0)
    core = _tiny_core(cwt_encoder=True)
    core.train()
    raw_x = torch.randn(BATCH, N_CHANNELS, CWT_T)
    dense_edge = _synthetic_dense_edge()
    w_real, w_imag = _synthetic_cwt()
    logits, _ = core(raw_x, dense_edge, w_real, w_imag)
    loss = logits.sum()
    loss.backward()

    enc_grads = [
        p.grad for p in core.cwt_node_encoder.parameters() if p.requires_grad
    ]
    assert enc_grads, "node encoder has no trainable parameters"
    assert any(g is not None and g.abs().sum() > 0 for g in enc_grads), (
        "node encoder parameters received no gradient -- CWT is not in the "
        "trainable graph"
    )

    edge_grads = [
        p.grad for p in core.dense_edge_conv.parameters() if p.requires_grad
    ]
    assert edge_grads, "edge GRU has no trainable parameters"
    assert any(g is not None and g.abs().sum() > 0 for g in edge_grads), (
        "edge-network parameters received no gradient -- WCT path is missing"
    )


def test_message_sees_src_dst_and_edge() -> None:
    torch.manual_seed(1)
    core = _tiny_core(cwt_encoder=True)
    core.eval()
    dense_edge = _synthetic_dense_edge()
    w_real, w_imag = _synthetic_cwt()
    events, src, dst, *_ = core._dense_edge_features(dense_edge)
    node_embed = core._cwt_node_embeddings(w_real, w_imag)
    base_msg = core._compose_message_features(events, src, dst, node_embed)

    assert base_msg.shape[-1] == 2 * HIDDEN + HIDDEN
    d = HIDDEN
    h_src, h_dst, e_ij = base_msg[..., :d], base_msg[..., d : 2 * d], base_msg[..., 2 * d :]
    assert torch.allclose(e_ij, events)
    # src/dst slots are the gathered node embeddings, not a copy of the edge.
    assert not torch.allclose(h_src, e_ij)
    assert not torch.allclose(h_dst, e_ij)

    # Perturb only node 0 -- incident-edge messages change, others do not.
    perturbed = node_embed.clone()
    perturbed[:, 0] = perturbed[:, 0] + 3.0
    new_msg = core._compose_message_features(events, src, dst, perturbed)
    incident = (src == 0) | (dst == 0)
    unrelated = ~incident
    assert not torch.allclose(base_msg[:, incident[0]], new_msg[:, incident[0]]), (
        "messages on edges incident to the perturbed node must change"
    )
    if unrelated.any():
        assert torch.allclose(
            base_msg[:, unrelated[0]], new_msg[:, unrelated[0]], atol=1e-6
        ), "messages on edges not incident to the perturbed node must not change"


def test_zero_node_embed_ablation_matches_zero_slots() -> None:
    core = _tiny_core(
        cwt_encoder=True,
        time_frequency_node_ablation="zero_node_embed",
    )
    w_real, w_imag = _synthetic_cwt()
    embed = core._cwt_node_embeddings(w_real, w_imag)
    assert torch.equal(embed, torch.zeros_like(embed))


def test_node_only_ablation_zeros_edge_features_in_forward() -> None:
    torch.manual_seed(2)
    core = _tiny_core(
        cwt_encoder=True,
        time_frequency_node_ablation="node_only",
    )
    core.eval()
    raw_x = torch.randn(BATCH, N_CHANNELS, CWT_T)
    dense_edge = _synthetic_dense_edge()
    w_real, w_imag = _synthetic_cwt()
    # Capture composed features via a hook on the message MLP.
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod, inputs, _out):
        captured["full_features"] = inputs[0].detach()

    handle = core.sparse_message_mlp.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            core(raw_x, dense_edge, w_real, w_imag)
    finally:
        handle.remove()
    feats = captured["full_features"]
    edge_part = feats[..., 2 * HIDDEN :]
    assert torch.equal(edge_part, torch.zeros_like(edge_part)), (
        "node_only ablation must zero the WCT edge slots of the message"
    )
    node_part = feats[..., : 2 * HIDDEN]
    assert node_part.abs().sum() > 0, "node slots must still carry CWT embeddings"
    # WCT GRU/conv must not run: no edge-network gradient.
    core.zero_grad(set_to_none=True)
    core.train()
    logits, _ = core(raw_x, dense_edge, w_real, w_imag)
    logits.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.cwt_node_encoder.parameters()
    )
    edge_grads = [
        p.grad for p in core.dense_edge_conv.parameters() if p.requires_grad
    ]
    assert edge_grads, "edge network has no trainable parameters"
    assert all(g is None or g.abs().sum() == 0 for g in edge_grads), (
        "node_only must not train the WCT edge network"
    )


def test_baseline_forward_rejects_cwt_tensors() -> None:
    core = _tiny_core(cwt_encoder=False)
    raw_x = torch.randn(BATCH, N_CHANNELS, CWT_T)
    dense_edge = _synthetic_dense_edge()
    w_real, w_imag = _synthetic_cwt()
    with pytest.raises(ValueError):
        core(raw_x, dense_edge, w_real, w_imag)


def test_tf_forward_uses_cwt_and_runs_under_optional_cuda_bf16() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = _tiny_core(cwt_encoder=True).to(device)
    core.train()
    raw_x = torch.randn(BATCH, N_CHANNELS, CWT_T, device=device)
    dense_edge = _synthetic_dense_edge().to(device)
    w_real, w_imag = _synthetic_cwt()
    w_real = w_real.to(device)
    w_imag = w_imag.to(device)

    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with amp:
        logits, _ = core(raw_x, dense_edge, w_real, w_imag)
    assert logits.device.type == device.type
    loss = logits.float().sum()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.cwt_node_encoder.parameters()
    )


def test_prepare_features_keeps_cwt_for_node_encoder() -> None:
    """Classifier _prepare_features must return CWT tensors when the encoder
    is on, even though the dense-edge cache only stores WCT features."""
    torch.manual_seed(3)
    n_time = 64
    X = torch.randn(2, 3, n_time).numpy().astype("float32")
    clf = SparseEvidenceGNNClassifier(
        sampling_rate=64,
        lowest=8.0,
        highest=24.0,
        nfreqs=4,
        hidden_dim=8,
        event_mode="dense",
        event_aggregation="concat",
        dense_edge_temporal_mode="rnn",
        cwt_encoder=True,
        cwt_backend="torch",
        device="cpu",
        verbose=0,
        dense_edge_cache_dir=None,
        cwt_cache=None,
        normalize_input=False,
        epochs=1,
        batch_size=2,
    )
    clf.device_ = torch.device("cpu")
    clf.X_mean_, clf.X_std_ = 0.0, 1.0
    features = clf._prepare_features(X, fit=False)
    assert len(features) == 4, (
        "TF-node _prepare_features must return (raw_x, dense_edge_raw, "
        f"w_real, w_imag); got {len(features)} tensors"
    )
    raw_x, dense_edge_raw, w_real, w_imag = features
    assert raw_x.shape[:2] == (2, 3)
    assert w_real.shape[:2] == (2, 3) and w_imag.shape[:2] == (2, 3)
    assert w_real.ndim == 4 and w_imag.ndim == 4
    assert dense_edge_raw.ndim == 5 and dense_edge_raw.shape[1] == 4

    model = clf._build_model(n_channels=3, n_classes=2)
    model.train()
    logits, _ = model(raw_x, dense_edge_raw, w_real, w_imag)
    logits.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.cwt_node_encoder.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.dense_edge_conv.parameters()
    )


def test_node_only_prepare_features_skips_wct() -> None:
    """node_only must not compute WCT; encoder still trains; edge net does not."""
    torch.manual_seed(4)
    n_time = 64
    X = torch.randn(2, 3, n_time).numpy().astype("float32")
    clf = SparseEvidenceGNNClassifier(
        sampling_rate=64,
        lowest=8.0,
        highest=24.0,
        nfreqs=4,
        hidden_dim=8,
        event_mode="dense",
        event_aggregation="concat",
        dense_edge_temporal_mode="rnn",
        cwt_encoder=True,
        time_frequency_node_ablation="node_only",
        cwt_backend="torch",
        device="cpu",
        verbose=0,
        dense_edge_cache_dir=None,
        cwt_cache=None,
        normalize_input=False,
        epochs=1,
        batch_size=2,
    )
    clf.device_ = torch.device("cpu")
    clf.X_mean_, clf.X_std_ = 0.0, 1.0

    def _wct_should_not_run(*_args, **_kwargs):
        raise AssertionError("node_only must not call _precompute_dense_edge_inputs")

    clf._precompute_dense_edge_inputs = _wct_should_not_run
    features = clf._prepare_features(X, fit=False)
    raw_x, dense_edge_raw, w_real, w_imag = features
    assert torch.equal(dense_edge_raw, torch.zeros_like(dense_edge_raw))
    assert w_real.ndim == 4 and w_imag.ndim == 4

    model = clf._build_model(n_channels=3, n_classes=2)
    model.train()
    logits, _ = model(raw_x, dense_edge_raw, w_real, w_imag)
    logits.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.cwt_node_encoder.parameters()
    )
    assert all(
        p.grad is None or p.grad.abs().sum() == 0
        for p in model.dense_edge_conv.parameters()
        if p.requires_grad
    )


def test_tf_encoder_works_with_conv_temporal_mode() -> None:
    """The encoder is a flag on either dense-family temporal mode, not a GRU-only pipeline."""
    core = _tiny_core(
        cwt_encoder=True,
        dense_edge_temporal_mode="conv",
    )
    assert core.cwt_node_encoder is not None
    first = core.sparse_message_mlp[0]
    assert isinstance(first, nn.Linear)
    assert first.in_features == HIDDEN + 2 * HIDDEN
    # Construction is the contract here. The toy EDGE_T is too short for
    # the default Conv2d+pool stack; GRU tests already cover encoder
    # forward + grads, which are temporal-mode independent.


def test_tf_encoder_is_a_flag_not_a_pipeline() -> None:
    from Epilepsy.run_pipelines import (
        _build_argument_parser,
        _dense_family_params,
        _dense_family_result_dir,
    )

    parser = _build_argument_parser()
    pipeline_action = next(
        action for action in parser._actions if "--pipeline" in action.option_strings
    )
    assert "dense_edge_gru_tf_node" not in pipeline_action.choices
    assert set(pipeline_action.choices) == {
        "dense_edge_gru",
        "dense_edge",
        "dense_edge_mamba",
        "truong_stft_cnn",
        "dbconformer",
        "slimseiz",
        "cg_mambanet",
    }
    encoder_action = next(
        action
        for action in parser._actions
        if "--cwt-encoder" in action.option_strings
        and "--cwt-encoder-ablation" not in action.option_strings
    )
    assert "--no-cwt-encoder" in encoder_action.option_strings
    ablation_action = next(
        action
        for action in parser._actions
        if "--cwt-encoder-ablation" in action.option_strings
    )
    assert set(ablation_action.choices) == {"none", "zero_node_embed", "node_only"}

    for pipeline in ("dense_edge_gru", "dense_edge", "dense_edge_mamba"):
        for label_mode in ("prediction", "detection"):
            params = _dense_family_params(pipeline, label_mode)
            assert params["cwt_encoder"] is False

    with pytest.raises(ValueError, match="not a dense-family pipeline"):
        _dense_family_params("dense_edge_gru_tf_node", "prediction")

    root = Path("/tmp/results")
    gru_off = _dense_family_result_dir(root, "dense_edge_gru", "prediction", False)
    gru_on = _dense_family_result_dir(
        root, "dense_edge_gru", "prediction", False, cwt_encoder=True
    )
    conv_off = _dense_family_result_dir(root, "dense_edge", "prediction", False)
    conv_on = _dense_family_result_dir(
        root, "dense_edge", "prediction", False, cwt_encoder=True
    )
    gru_node_only = _dense_family_result_dir(
        root,
        "dense_edge_gru",
        "prediction",
        False,
        cwt_encoder=True,
        time_frequency_node_ablation="node_only",
    )
    assert gru_off == root / "prediction"
    assert gru_on == root / "cwt_encoder" / "prediction"
    assert conv_off == root / "dense_edge" / "prediction"
    assert conv_on == root / "dense_edge" / "cwt_encoder" / "prediction"
    assert gru_node_only == root / "cwt_encoder" / "node_only" / "prediction"
    assert "cwt_encoder" not in gru_off.parts
    assert "cwt_encoder" not in conv_off.parts
