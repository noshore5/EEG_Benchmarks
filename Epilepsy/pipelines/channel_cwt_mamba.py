"""ChannelCWTMambaClassifier -- the missing baseline (2026-08-30).

Every graph pipeline in this repo (dense_edge, temporal_graph "pre"/"post",
hermitian_ssm) computes a cross-channel coherence / cross-spectrum before
the model sees anything. None of them has ever been compared against the
obvious null: **per-channel CWT fed straight to a shared-weight sequence
model, no coherence at all.**

This is that baseline. Per 30 s window:
  raw EEG [C, T]
   -> Morlet CWT per channel  [C, F, T]   (reusing hermitian_ssm_cache's
      _MorletCWT + the same 8-40 Hz / nfreqs=16 / fd=2 -> F_out=8 /
      time_downsample=16 grid, so this is band-matched to the k=6
      hermitian cache and to temporal_graph_mamba)
   -> per (channel, timestep) feature = raw power |w|**2, z-scored per
      freq over the training set (pre-matched: "pre" only ever feeds the
      model a normalized magnitude-derived quantity -- coherence in
      [0,1] -- never a phase or a complex value per channel; per-channel
      absolute phase is meaningless, only cross-pair relative phase is,
      and this model has no pair. No log: "pre" doesn't log either.)
   -> per-channel Linear -> [C, T', H] node sequences
   -> ONE weight-shared Mamba over each channel's own T' sequence
      (channels folded into the batch dim -- the same "shared temporal
      model over a structured axis" trick "pre"/"post" use, which is
      most of why they beat hermitian; see Session_notes/2026_08_30/)
   -> mean over channels -> Linear -> 2 logits

Deliberately NOT wired into SparseEvidenceGNNCore: it needs raw per-channel
CWT, which that core has no cache/serve path for under event_mode=
"temporal_graph". Standalone, its own small per-recording CWT cache. Drops
"pre"'s n-hop graph message passing -- a secondary component; if this
baseline is competitive, the full "exact same downstream" version is
worth building.

Interface: fit(recordings) / predict_proba(recordings) / classes_ -- so a
wrapper can swap it in for HermitianSSMClassifier and reuse
leave_one_seizure_out_hermitian_ssm unchanged.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from Epilepsy.pipelines.common import resolve_torch_device, set_seed
from Epilepsy.pipelines.hermitian_ssm_cache import (
    HermitianSpectralConfig,
    _apply_mains_notch,
    _gaussian_kernel1d,
    _mean_pool_time,
    _MorletCWT,
    _smooth_time,
    default_hermitian_ssm_cache_root,
)


def _freq_grid(
    cfg: HermitianSpectralConfig, spacing: str = "log"
) -> tuple[torch.Tensor, torch.Tensor]:
    """(freqs_full [nfreqs], freqs_out [nfreqs//fd]), both highest-first.
    spacing="log" is the hermitian-cache default; spacing="linear" gives a
    Truong-style evenly-spaced grid (STFT-like) over [lowest, highest]."""
    fd = int(cfg.freq_downsample)
    f_out = cfg.nfreqs // fd
    if spacing == "linear":
        freqs_full = torch.linspace(
            cfg.highest, cfg.lowest, cfg.nfreqs, dtype=torch.float64
        ).to(torch.float32)
        freqs_out = freqs_full.double().reshape(f_out, fd).mean(dim=1).to(torch.float32)
        return freqs_full, freqs_out
    ratio = cfg.lowest / cfg.highest
    exps = torch.linspace(0.0, 1.0, cfg.nfreqs, dtype=torch.float64)
    freqs_full = (cfg.highest * ratio ** exps).to(torch.float32)
    freqs_out = torch.exp(
        torch.log(freqs_full.double()).reshape(f_out, fd).mean(dim=1)
    ).to(torch.float32)
    return freqs_full, freqs_out


class _ChannelCWTCache:
    """Per-recording disk cache of the smoothed, time-downsampled,
    freq-pooled complex CWT ``w_ds`` [C, F_out, T_ds] -- exactly the
    tensor hermitian_ssm_cache computes right before it forms coherence,
    just kept per-channel instead."""

    def __init__(self, cfg: HermitianSpectralConfig, root: str | None = None,
                 device: str = "cpu", verbose: bool = True,
                 freq_spacing: str = "log", store_dtype: str = "float32") -> None:
        self.cfg = cfg
        self.device = device
        self.verbose = verbose
        self.freq_spacing = freq_spacing
        self.store_dtype = np.dtype(store_dtype)
        self._ram: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        base = Path(root) if root else default_hermitian_ssm_cache_root().parent / "channel_cwt_cache"
        key_fields = [cfg.sampling_rate, cfg.lowest, cfg.highest, cfg.nfreqs,
                      cfg.freq_downsample, cfg.time_downsample, cfg.smooth_time_steps,
                      cfg.mains_notch]
        if freq_spacing != "log":
            key_fields.append(("freq_spacing", freq_spacing))
        if self.store_dtype != np.float32:
            key_fields.append(("store_dtype", str(self.store_dtype)))
        key = hashlib.sha1(repr(tuple(key_fields)).encode()).hexdigest()[:16]
        self.root = base / key
        self.root.mkdir(parents=True, exist_ok=True)
        self._freqs_full, self.freqs_out = _freq_grid(cfg, freq_spacing)

    def _path(self, subject, run) -> Path:
        return self.root / f"{subject}_{run}.npy"

    def ensure(self, subject, run, raw_x: np.ndarray) -> None:
        p = self._path(subject, run)
        if p.exists():
            return
        cfg = self.cfg
        dev = torch.device(self.device)
        x = _apply_mains_notch(np.asarray(raw_x), cfg)
        signal = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
        cwt = _MorletCWT(signal, cfg)
        fd = int(cfg.freq_downsample)
        smooth_kernel = _gaussian_kernel1d(cfg.smooth_time_steps, dev)
        t0 = time.perf_counter()
        # Chunk the freq axis ONE bin at a time: _MorletCWT.transform
        # materialises [C, fc, n_padded] complex with no internal chunking
        # and then an equally large ifft + work buffers. On an hour-long
        # recording n_padded ~ 2^20, so even fc=16 is a ~7-8 GB transient
        # per recording -> instant swap avalanche on a 16 GB Mac (killed
        # 4x, 2026-08-30). fc=1 keeps the peak ~0.5 GB; ~8 s/recording.
        freqs_full = self._freqs_full.to(dev)
        fchunk = 1
        parts = []
        for s in range(0, len(freqs_full), fchunk):
            c = cwt.transform(freqs_full[s:s + fchunk])             # [C, fc, N]
            parts.append(_mean_pool_time(c, cfg.time_downsample))   # [C, fc, T_ds]
            del c
        w_ds = torch.cat(parts, dim=1)                              # [C, nfreqs, T_ds]
        del parts
        # freq-pool (geometric-mean grid already; pool the complex coeffs)
        C, nf, td = w_ds.shape
        w_ds = w_ds.reshape(C, nf // fd, fd, td).mean(dim=2)        # [C, F_out, T_ds]
        # light time-smoothing of the complex coeffs (same kernel the
        # hermitian cache applies to the auto-spectra)
        w_ds = torch.complex(
            _smooth_time(w_ds.real.contiguous(), smooth_kernel),
            _smooth_time(w_ds.imag.contiguous(), smooth_kernel),
        )
        arr = torch.view_as_real(w_ds).cpu().numpy().astype(self.store_dtype)  # [C, F_out, T_ds, 2]
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.rename(p)
        if self.verbose:
            print(f"  [channel_cwt cache] {subject}_{run}: {arr.shape} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    # In-RAM LRU of recording arrays, held as float16 (~85 MB each; the
    # feature path in _WinDS re-widens to float32 before squaring). The
    # old np.load(mmap_mode="r") per __getitem__ leaked faulted pages into
    # RSS -> jetsam OOM. Cap large enough to hold a whole subject's ~45
    # recordings (~3.8 GB f16) so a shuffled epoch does ZERO redundant
    # disk reads -- a small cap + shuffle=True was re-reading 100s of GB
    # per epoch and dominated the MPS epoch time.
    _ram_cap = 48

    def get(self, subject, run) -> np.ndarray:
        k = (subject, run)
        a = self._ram.get(k)
        if a is not None:
            self._ram.move_to_end(k)
            return a
        a = np.load(self._path(subject, run))
        if a.dtype != np.float16:
            a = a.astype(np.float16)
        self._ram[k] = a
        if len(self._ram) > self._ram_cap:
            self._ram.popitem(last=False)
        return a


class _WinDS(torch.utils.data.Dataset):
    def __init__(self, cache: _ChannelCWTCache, index: list[dict], y: np.ndarray,
                 feat_mu: np.ndarray, feat_sd: np.ndarray) -> None:
        self.cache = cache
        self.index = index
        self.y = y
        self.mu = feat_mu   # [F_out]  (power stats)
        self.sd = feat_sd

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        e = self.index[i]
        arr = self.cache.get(e["subject"], e["run"])          # [C, F_out, T_ds, 2]
        t0, tw = e["t0"], e["tw"]
        C, F, T_tot, _ = arr.shape
        t1 = min(t0 + tw, T_tot)
        feat = np.zeros((C, tw, F), dtype=np.float32)
        n = max(0, t1 - t0)
        if n > 0:
            s = np.asarray(arr[:, :, t0:t1, :], dtype=np.float32).transpose(0, 2, 1, 3)  # [C, n, F, 2]
            power = s[..., 0] ** 2 + s[..., 1] ** 2                     # [C, n, F]
            feat[:, :n] = (power - self.mu) / self.sd
        return torch.from_numpy(feat), torch.tensor(int(self.y[i]), dtype=torch.long)


class _HopMessagePassing(nn.Module):
    """Spatial message passing across the C channels, replicating
    ``SparseEvidenceGNNCore._propagate_hops`` exactly: per directed edge a
    message MLP on ``[h_dst, h_src]``, scatter-add incoming per node, a
    GRUCell-gated state update. Complete graph -- all ``C*(C-1)`` directed
    pairs -- which *is* the repo's canonical edge topology for 23 CHB
    channels (fully connected, 253 undirected edges; there is no montage
    adjacency). ``n_rounds`` propagation rounds; 0 is a no-op (bit-identical
    to the coherence-free baseline).

    ablation 1 (2026-08-30): per-channel CWT power features + this, to
    separate "cross-channel mixing helps" from "coherence specifically
    helps". Everything else matches "pre" / the channel_cwt baseline."""

    def __init__(self, n_channels: int, hidden_dim: int, n_rounds: int) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_rounds = int(n_rounds)
        src, dst = [], []
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self.register_buffer("hop_src_idx", torch.tensor(src, dtype=torch.long))
        self.register_buffer("hop_dst_idx", torch.tensor(dst, dtype=torch.long))
        # SAME shapes/activation as SparseEvidenceGNNCore's hop_message_mlp/
        # hop_update -- see cwt_gnn_classifiers.py.
        self.hop_message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hop_update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_state: torch.Tensor) -> torch.Tensor:
        # node_state: [B, C, H]
        if self.n_rounds <= 0:
            return node_state
        b, c, h = node_state.shape
        src, dst = self.hop_src_idx, self.hop_dst_idx
        scatter_idx = dst.view(1, -1, 1).expand(b, -1, h)
        for _ in range(self.n_rounds):
            h_src = node_state.index_select(1, src)                  # [B, 2E, H]
            h_dst = node_state.index_select(1, dst)
            edge_msg = self.hop_message_mlp(torch.cat([h_dst, h_src], dim=-1))
            incoming = torch.zeros_like(node_state)
            incoming.scatter_add_(1, scatter_idx, edge_msg)
            node_state = self.hop_update(
                incoming.reshape(b * c, h), node_state.reshape(b * c, h)
            ).reshape(b, c, h)
        return node_state


class _ChannelCWTNet(nn.Module):
    """Temporal block is *exactly* "pre"'s: `cwt_gnn_classifiers.
    _DenseEdgeMambaTemporal` with the `temporal_graph_mamba` params
    (d_model=16, d_state=16, d_conv=4, expand=2, n_layers=1,
    chunk_size=128). This makes the baseline a clean ablation -- only the
    coherence graph is removed: each of the 23 channels is its own
    weight-shared sequence (the node axis sits in the slot "pre" puts its
    node axis), the CWT power replaces the scatter-meaned edge messages as
    the per-(node,timestep) feature, and the graph message-passing "pre"
    runs *after* the temporal block is dropped -- a plain mean over the 23
    node embeddings feeds the 2-class head. `_DenseEdgeMambaTemporal`'s
    128-row chunking + gradient checkpointing is what keeps this within
    memory at batch_size=64 (23 channels folded into the batch axis)."""

    def __init__(self, n_freq_out: int, *, d_model: int = 16, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, n_layers: int = 1,
                 out_channels: int = 8, head_hidden: int = 32,
                 n_channels: int = 23, n_hops: int = 1) -> None:
        super().__init__()
        from Epilepsy.pipelines.cwt_gnn_classifiers import _DenseEdgeMambaTemporal

        self.temporal = _DenseEdgeMambaTemporal(
            in_channels=n_freq_out, out_channels=out_channels,
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            n_layers=n_layers, dropout=0.0, chunk_size=128, use_cuda_kernel=None,
        )
        # n_hops matches "pre" semantics: 1 = single-hop (message passing
        # OFF, bit-identical to the coherence-free baseline); n_hops=K runs
        # K-1 rounds of spatial mixing across the 23 channels AFTER the
        # temporal block. Built unconditionally so n_hops=1's RNG stream is
        # identical to n_hops>1's -- same precedent as SparseEvidenceGNNCore.
        self.hops = _HopMessagePassing(n_channels, out_channels, max(0, n_hops - 1))
        self.head = nn.Sequential(nn.GELU(), nn.Linear(out_channels, head_hidden),
                                  nn.GELU(), nn.Linear(head_hidden, 2))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C=23, T, F] -> _DenseEdgeMambaTemporal wants [B, C_in=F, E=C, T]
        x = feat.permute(0, 3, 1, 2)                    # [B, F, C, T]
        node = self.temporal(x)                         # [B, out_channels, C, 1]
        node = node.squeeze(-1).permute(0, 2, 1)        # [B, C, out_channels]
        node = self.hops(node)                          # spatial mixing (no-op if n_hops<=1)
        node = node.mean(dim=1)                         # mean over the 23 channels
        return self.head(node)                          # [B, 2]


class ChannelCWTMambaClassifier:
    model_label = "Channel-CWT-Mamba"

    def __init__(
        self,
        *,
        epochs: int = 30,
        # b*c sequences (batch x 23 channels) all go through the mambapy
        # pure-PyTorch pscan at once; at bs=32 that OOM'd 16 GB on the real
        # 6-fold right at training entry (2026-08-30, rc=137). Keep small.
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 1.0,
        validation_split: float = 0.2,
        early_stopping_patience: int | None = 5,
        use_class_weights: bool = True,
        negative_to_positive_ratio: float | None = 5.0,  # matches DEFAULT_NEGATIVE_TO_POSITIVE_RATIO
        seed: int = 42,
        device: str = "cpu",
        verbose: int = 1,
        spectral_config: HermitianSpectralConfig | None = None,
        freq_spacing: str = "log",
        store_dtype: str = "float32",
        cache_root: str | None = None,
        precompute_device: str = "cpu",
        head_hidden: int = 32,
        mamba_d_model: int = 16,   # "pre" (_DenseEdgeMambaTemporal) defaults --
        mamba_d_state: int = 16,   # temporal_graph_mamba params, matched so this
        mamba_d_conv: int = 4,     # baseline is a clean coherence ablation of "pre"
        mamba_expand: int = 2,
        mamba_n_layers: int = 1,
        mamba_out_channels: int = 8,   # == dense_conv_out_channels / temporal_graph_edge_dim
        mamba_n_hops: int = 1,   # 1 = no spatial message passing (pure coherence-free
                                 # baseline). n_hops=2 = ablation 1: per-channel features
                                 # + 1 round of complete-graph mixing across the 23 channels.
        n_channels: int = 23,
        **_ignored: object,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.use_class_weights = use_class_weights
        self.negative_to_positive_ratio = negative_to_positive_ratio
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.cfg = spectral_config or HermitianSpectralConfig()
        self.freq_spacing = freq_spacing
        self.store_dtype = store_dtype
        self.cache_root = cache_root
        self.precompute_device = precompute_device
        self.head_hidden = head_hidden
        self.mamba_d_model = mamba_d_model
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_n_layers = mamba_n_layers
        self.mamba_out_channels = mamba_out_channels
        self.mamba_n_hops = mamba_n_hops
        self.n_channels = n_channels

        self.model_ = None
        self.classes_ = np.array([0, 1])
        self.device_ = None
        self._cache: _ChannelCWTCache | None = None
        self._feat_mu = None
        self._feat_sd = None

    def _get_cache(self) -> _ChannelCWTCache:
        if self._cache is None:
            self._cache = _ChannelCWTCache(
                self.cfg, root=self.cache_root, device=self.precompute_device,
                verbose=bool(self.verbose), freq_spacing=self.freq_spacing,
                store_dtype=self.store_dtype,
            )
        return self._cache

    def _build_index(self, recordings: list[dict]):
        cache = self._get_cache()
        td = int(self.cfg.time_downsample)
        index, y = [], []
        for rec in recordings:
            cache.ensure(rec["subject"], rec["run"], rec["raw_x"])
            # Free the ~170 MB raw signal now it's cached to disk. Subject
            # 1's 41 recordings are ~7 GB of raw held resident by the
            # LOSO loop for the whole run -- dead weight once the per-rec
            # CWT cache exists (fit/predict only touch the .npy after
            # this). Nothing downstream of fit reads raw_x.
            rec["raw_x"] = None
            wins = rec["windows"]
            tw = max(1, int(round((wins[0]["end_sample"] - wins[0]["start_sample"]) / td)))
            for w in wins:
                index.append({"subject": rec["subject"], "run": rec["run"],
                              "t0": w["start_sample"] // td, "tw": tw})
                y.append(int(w["label"]))
        return index, np.asarray(y, dtype=np.int64)

    def _feat_stats(self, index: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        cache = self._get_cache()
        seen = set()
        s = ss = 0.0
        n = 0
        for e in index:
            key = (e["subject"], e["run"])
            if key in seen:
                continue
            seen.add(key)
            a = np.asarray(cache.get(*key), dtype=np.float64)     # [C, F, T, 2]
            power = a[..., 0] ** 2 + a[..., 1] ** 2               # [C, F, T]
            s += power.sum(axis=(0, 2))
            ss += (power * power).sum(axis=(0, 2))
            n += power.shape[0] * power.shape[2]
        mu = (s / max(1, n)).astype(np.float32)                   # [F]
        var = np.maximum(ss / max(1, n) - mu.astype(np.float64) ** 2, 1e-12)
        sd = np.sqrt(var).astype(np.float32)
        return mu, np.where(sd > 1e-6, sd, 1.0)

    def fit(self, recordings: list[dict]) -> "ChannelCWTMambaClassifier":
        set_seed(self.seed)
        self.device_ = resolve_torch_device(self.device)
        index, y = self._build_index(recordings)
        self._feat_mu, self._feat_sd = self._feat_stats(index)
        n_freq_out = self._feat_mu.shape[0]

        rng = np.random.default_rng(self.seed)
        # Match "pre"/dense-family training: subsample interictal TRAIN
        # windows to negative_to_positive_ratio:1 (DEFAULT_NEGATIVE_TO_
        # POSITIVE_RATIO = 5.0). The hermitian_ssm LOSO loop this rides
        # does NOT subsample, which made epochs ~3-4x "pre"'s. Test
        # windows (predict_proba) are never touched.
        r = self.negative_to_positive_ratio
        if r is not None:
            pos = np.flatnonzero(y == 1)
            neg = np.flatnonzero(y == 0)
            cap = int(round(r * len(pos)))
            if 0 < cap < len(neg):
                keep = np.sort(np.concatenate([pos, rng.choice(neg, cap, replace=False)]))
                index = [index[i] for i in keep]
                y = y[keep]

        dataset = _WinDS(self._get_cache(), index, y, self._feat_mu, self._feat_sd)
        perm = rng.permutation(len(index))
        n_val = int(round(self.validation_split * len(index))) if self.validation_split else 0
        val_idx, tr_idx = perm[:n_val].tolist(), perm[n_val:].tolist()

        self.model_ = _ChannelCWTNet(
            n_freq_out, d_model=self.mamba_d_model, d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv, expand=self.mamba_expand,
            n_layers=self.mamba_n_layers, out_channels=self.mamba_out_channels,
            head_hidden=self.head_hidden,
            n_channels=self.n_channels, n_hops=self.mamba_n_hops,
        ).to(self.device_)

        y_tr = y[tr_idx]
        w = None
        if self.use_class_weights and (y_tr == 1).any() and (y_tr == 0).any():
            n_pos, n_neg, n_all = float((y_tr == 1).sum()), float((y_tr == 0).sum()), float(len(y_tr))
            w = torch.tensor([n_all / (2 * n_neg), n_all / (2 * n_pos)],
                             dtype=torch.float32, device=self.device_)
        criterion = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)

        def _loader(idx, shuffle):
            return torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, idx),
                batch_size=self.batch_size, shuffle=shuffle, num_workers=0,
            )

        train_loader = _loader(tr_idx, True)
        val_loader = _loader(val_idx, False) if n_val > 0 else None

        best_val, best_state, bad = float("inf"), None, 0
        for epoch in range(self.epochs):
            t0 = time.perf_counter()
            self.model_.train()
            tot, seen = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                opt.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip_norm)
                opt.step()
                tot += float(loss.item()) * len(yb); seen += len(yb)
            tr_loss = tot / max(1, seen)

            if val_loader is not None:
                vl, vauc = self._eval(val_loader, criterion)
                improved = vl < best_val - 1e-5
                if improved:
                    best_val = vl
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if self.verbose:
                    print(f"  [channel_cwt] epoch {epoch + 1}/{self.epochs} "
                          f"train_loss={tr_loss:.4f} val_loss={vl:.4f} val_auc={vauc:.3f} "
                          f"({time.perf_counter() - t0:.1f}s)" + (" *" if improved else ""), flush=True)
                if self.early_stopping_patience and bad >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"  [channel_cwt] early stop at epoch {epoch + 1}", flush=True)
                    break
            elif self.verbose:
                print(f"  [channel_cwt] epoch {epoch + 1}/{self.epochs} train_loss={tr_loss:.4f} "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _eval(self, loader, criterion) -> tuple[float, float]:
        from sklearn.metrics import roc_auc_score
        self.model_.eval()
        ls, seen, ys, ps = 0.0, 0, [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                logits = self.model_(xb)
                ls += float(criterion(logits, yb).item()) * len(yb); seen += len(yb)
                ys.append(yb.cpu().numpy())
                ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        y_all, p_all = np.concatenate(ys), np.concatenate(ps)
        try:
            auc = roc_auc_score(y_all, p_all) if len(np.unique(y_all)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        return ls / max(1, seen), auc

    def predict_proba(self, recordings: list[dict]) -> list[np.ndarray]:
        if self.model_ is None:
            raise ValueError("not fitted")
        self.model_.eval()
        out = []
        for rec in recordings:
            index, y = self._build_index([rec])
            ds = _WinDS(self._get_cache(), index, y, self._feat_mu, self._feat_sd)
            loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=0)
            probs = np.zeros((len(index), 2), dtype=np.float32)
            b0 = 0
            with torch.no_grad():
                for xb, _yb in loader:
                    p = torch.softmax(self.model_(xb.to(self.device_)), 1).cpu().numpy()
                    probs[b0:b0 + len(p)] = p
                    b0 += len(p)
            out.append(probs)
        return out
