"""HermitianSSMClassifier -- the learned half of the ``--pipeline hermitian_ssm``
paradigm ("Graph Spectral Mamba-3", ``Epilepsy/hermitian_ssm.md``).

Deterministic half (CWT -> coherence -> Hermitian graph -> eigh -> top-k)
lives in ``hermitian_ssm_cache.py`` and is disk-cached per recording. This
module consumes that cache:

    cached eigenpairs  [Tw, F, k] + [Tw, F, k, C]
      -> complex-aware spectral encoder   (per mode -> per frequency -> token)
      -> [B, Tw, d_model]
      -> Mamba temporal block             (reused _DenseEdgeMambaTemporal, E=1)
      -> last timestep
      -> linear head -> 2-class logits

Self-contained on purpose (own training loop, does NOT subclass
``SparseEvidenceGNNClassifier``) -- only shares small stateless helpers from
``common.py``.

Recording-level interface (like ``ContinuousCWTMambaClassifier``): ``fit``
takes a list of ``get_continuous_data()`` recording dicts, ``predict_proba``
returns one ``[n_windows, 2]`` array per recording in chronological window
order. Windows are independent here (no state carried across them -- that is
the ``continuous_cwt_mamba`` paradigm, not this one); the recording-level
interface exists only so the classifier can slice the per-recording
spectral cache, which is what makes that cache windowing-independent.

=========================================================================
MAMBA-3 INTEGRATION POINT (next step -- not done here, see design doc):
The encoder below collapses each complex eigenvector to ``[Re u, Im u]``
reals and feeds a plain real ``d_model`` token into a standard (Mamba-2
style) recurrence. Mamba-3 (arXiv 2603.15569) adds a genuinely
complex-valued diagonal state, losslessly carried as a 2N real state, with
per-dimension 2x2 rotation blocks and a "RoPE trick" that folds the
rotation into the B/C projections *before* the scan (so it still runs on
real kernels/pscan -- no custom CUDA kernel).

To adopt it here:
  * The encoder should stop flattening phase into a bare scalar. Emit the
    per-frequency token as a complex vector (or interleaved 2N real) so the
    eigenvector's relative phase structure is preserved into the recurrence.
  * Replace ``_DenseEdgeMambaTemporal`` with a Mamba-3 block whose state
    transition factors into (magnitude decay) x (rotation); the rotation is
    where the Hermitian graph's phase belongs, handled as an actual
    rotation operator rather than a number a dense layer must interpret.
  * ``_encode`` and ``_MambaTemporalHead`` are the two seams to change;
    the cache format and everything left of it stays fixed.
=========================================================================
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from Epilepsy.pipelines.common import resolve_torch_device, set_seed
from Epilepsy.pipelines.cwt_gnn_classifiers import _DenseEdgeMambaTemporal
from Epilepsy.pipelines.hermitian_ssm_cache import (
    HermitianSpectralCache,
    HermitianSpectralConfig,
)


# --------------------------------------------------------------------------
# Spectral encoder:  eigenpairs at one timestep  ->  one d_model token
# --------------------------------------------------------------------------

class _SpectralEncoder(nn.Module):
    """Per graph timestep: a collection of ``(lambda_r, u_r)`` over ``F``
    frequencies and ``k`` modes -> one ``d_model`` vector, preserving the
    semantic grouping ``frequency -> mode -> (eigenvalue, eigenvector)``
    (design doc Section 12) rather than flattening every scalar at once.
    """

    def __init__(
        self,
        n_channels: int,
        n_freqs: int,
        k: int,
        d_model: int,
        d_mode: int,
        d_freq: int,
        *,
        freqs: "np.ndarray | None" = None,
        mode_feature: bool = True,
        d_mode_id: int = 4,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.k = k

        # Mode-as-feature (2026-08-27). vec_proj / val_proj are weight-shared
        # across the k axis, so without this the model can only tell mode 0
        # from mode 1 by their fixed slot in mode_fuse's concatenation. A
        # small learned embedding per mode SLOT (concatenated onto the
        # per-mode input) makes vec_proj mode-aware. NB: eigenpairs are
        # |lambda|-sorted every timestep, so slot r tags "rank r"
        # ("currently strongest", "second", ...), not a stable physical
        # mode -- but rank is what the slot actually means, and the model
        # can learn that the dominant coupling mode behaves differently from
        # the secondary one. Ablate with mode_feature=False.
        self.use_mode_feature = bool(mode_feature)
        n_mf = 0
        if self.use_mode_feature:
            self.mode_id = nn.Parameter(torch.randn(k, d_mode_id) * 0.02)  # [k, d_mode_id]
            n_mf = d_mode_id

        # Frequency-as-feature (2026-08-27). Without this the per-mode
        # projections are weight-shared across all F frequencies, so the
        # model can only tell frequencies apart by their fixed slot in
        # freq_fuse's concatenation. Feeding the (normalised) Hz value into
        # the per-mode eigenvector input makes every layer from vec_proj
        # onward frequency-aware, and lets it express smooth functions of
        # frequency instead of memorising 30 independent slots. Two
        # features: linear-normalised and log-normalised frequency, both in
        # [0, 1]. Cheap, does not touch the cache; ablate by passing
        # freqs=None (freq_feature=False at the classifier level).
        self.use_freq_feature = freqs is not None
        n_ff = 0
        if self.use_freq_feature:
            f = torch.from_numpy(np.array(freqs, dtype=np.float32))        # [F], highest-first (np.array copies -> writable)
            f_lin = (f - f.min()) / (f.max() - f.min() + 1e-8)
            lg = torch.log(f.clamp_min(1e-6))
            f_log = (lg - lg.min()) / (lg.max() - lg.min() + 1e-8)
            self.register_buffer("freq_feat", torch.stack([f_lin, f_log], dim=-1))  # [F, 2]
            n_ff = 2

        # Per mode: eigenvector [Re u, Im u] (2C reals, + frequency feature)
        # and eigenvalue (1 real) get SEPARATE embeddings, summed -- the
        # model keeps explicit access to both "mode shape" and "mode
        # strength" (doc Section 12).
        self.vec_proj = nn.Linear(2 * n_channels + n_ff + n_mf, d_mode)
        self.val_proj = nn.Linear(1, d_mode)
        self.mode_fuse = nn.Linear(k * d_mode, d_freq)
        self.freq_fuse = nn.Linear(n_freqs * d_freq, d_model)
        self.act = nn.GELU()

    def forward(self, eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        """eigvals: [B, T, F, k] real. eigvecs: [B, T, F, k, C] complex.
        Returns tokens [B, T, d_model]."""
        b, t, f, k = eigvals.shape
        # ---- MAMBA-3 seam: this is where the complex eigenvector is
        # flattened to reals. A Mamba-3 encoder would keep it complex. ----
        vec_real = torch.cat([eigvecs.real, eigvecs.imag], dim=-1)      # [B,T,F,k,2C]
        if self.use_freq_feature:
            ff = self.freq_feat.view(1, 1, f, 1, -1).expand(b, t, f, k, -1)
            vec_real = torch.cat([vec_real, ff], dim=-1)               # [B,T,F,k,2C+2]
        if self.use_mode_feature:
            mf = self.mode_id.view(1, 1, 1, k, -1).expand(b, t, f, k, -1)
            vec_real = torch.cat([vec_real, mf], dim=-1)               # [B,T,F,k,...+d_mode_id]
        vec_emb = self.vec_proj(vec_real)                              # [B,T,F,k,d_mode]
        val_emb = self.val_proj(eigvals.unsqueeze(-1))                 # [B,T,F,k,d_mode]
        mode_emb = self.act(vec_emb + val_emb)                        # [B,T,F,k,d_mode]
        freq_emb = self.act(self.mode_fuse(mode_emb.reshape(b, t, f, -1)))  # [B,T,F,d_freq]
        token = self.freq_fuse(freq_emb.reshape(b, t, -1))            # [B,T,d_model]
        return token


class _MambaTemporalHead(nn.Module):
    """Spectral tokens [B, T, d_model] -> 2-class logits. Reuses
    ``_DenseEdgeMambaTemporal`` with the edge axis set to 1 (its
    ``[B, C_in, E, T] -> [B, out, E, 1]`` contract does not care what E
    indexes -- same trick ``temporal_graph_mamba`` uses for the node axis).
    Last-timestep pooling is that block's built-in behaviour and matches the
    design doc's default (Section 14).
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        dropout: float,
        chunk_size: int,
        head_hidden: int,
    ) -> None:
        super().__init__()
        self.mamba = _DenseEdgeMambaTemporal(
            in_channels=d_model,
            out_channels=head_hidden,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers,
            dropout=dropout,
            chunk_size=chunk_size,
            use_cuda_kernel=None,  # auto: fused only on Linux/CUDA + mamba-ssm
        )
        self.head = nn.Sequential(nn.GELU(), nn.Linear(head_hidden, 2))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t, d = tokens.shape
        conv_in = tokens.permute(0, 2, 1).unsqueeze(2)   # [B, d_model, 1, T]
        pooled = self.mamba(conv_in).reshape(b, -1)       # [B, head_hidden]
        return self.head(pooled)                          # [B, 2]


class _HermitianSSMNet(nn.Module):
    def __init__(self, encoder: _SpectralEncoder, temporal: _MambaTemporalHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal

    def forward(self, eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.encoder(eigvals, eigvecs))


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------

class HermitianSSMClassifier:
    """See module docstring. sklearn-ish (``fit`` / ``predict_proba`` /
    ``classes_``) but recording-list in, not ``(X, y)``."""

    model_label = "Hermitian-SSM"

    def __init__(
        self,
        *,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 1.0,
        validation_split: float = 0.2,
        early_stopping_patience: int | None = 5,
        use_class_weights: bool = True,
        seed: int = 42,
        device: str = "cpu",
        verbose: int = 1,
        # spectral config (the cache key)
        spectral_config: HermitianSpectralConfig | None = None,
        cache_root: str | None = None,
        precompute_device: str = "cpu",
        # encoder
        d_model: int = 64,   # 2026-08-28: was 256 (doc default); see run_pipelines HERMITIAN_SSM_PARAMS
        d_mode: int = 32,
        d_freq: int = 64,
        freq_feature: bool = True,   # feed normalised Hz into the per-mode encoder
        mode_feature: bool = True,   # learned per-mode-slot (rank) embedding into the per-mode encoder
        # temporal (mamba) -- dense_edge_mamba's config, d_model from the doc
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_n_layers: int = 1,
        mamba_dropout: float = 0.0,
        mamba_chunk_size: int = 128,
        head_hidden: int = 64,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.use_class_weights = use_class_weights
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.cfg = spectral_config or HermitianSpectralConfig()
        self.cache_root = cache_root
        self.precompute_device = precompute_device
        self.d_model = d_model
        self.d_mode = d_mode
        self.d_freq = d_freq
        self.freq_feature = freq_feature
        self.mode_feature = mode_feature
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_n_layers = mamba_n_layers
        self.mamba_dropout = mamba_dropout
        self.mamba_chunk_size = mamba_chunk_size
        self.head_hidden = head_hidden

        self.model_ = None
        self.classes_ = None
        self.device_ = None
        self._cache: HermitianSpectralCache | None = None
        self._val_norm = (0.0, 1.0)  # eigenvalue standardisation (fit on train)

    # -- feature building (lazy, mmap-backed) -----------------------------

    def _get_cache(self) -> HermitianSpectralCache:
        if self._cache is None:
            self._cache = HermitianSpectralCache(
                self.cfg,
                root=self.cache_root,
                device=self.precompute_device,
                verbose=bool(self.verbose),
            )
        return self._cache

    def _build_index(self, recordings: list[dict]):
        """Ensure every recording's spectral cache exists, then return a
        flat window index (no feature arrays materialised) + labels + the
        recording dicts keyed by (subject, run)."""
        cache = self._get_cache()
        recs_by_key: dict[tuple, dict] = {}
        index: list[dict] = []
        y: list[int] = []
        for rec in recordings:
            key = (rec["subject"], rec["run"])
            recs_by_key[key] = rec
            cache.ensure(rec["subject"], rec["run"], rec["raw_x"])
            meta = cache.get(rec["subject"], rec["run"], rec["raw_x"], mmap=True)
            ds = int(meta["time_downsample"])
            wins = rec["windows"]
            tw = max(1, int(round((wins[0]["end_sample"] - wins[0]["start_sample"]) / ds)))
            for w in wins:
                index.append({"subject": rec["subject"], "run": rec["run"],
                              "t0": w["start_sample"] // ds, "tw": tw})
                y.append(int(w["label"]))
        return recs_by_key, index, np.asarray(y, dtype=np.int64)

    def _eigenvalue_norm(self, recs_by_key: dict) -> tuple[float, float]:
        """population mean/std of the cached eigenvalues over these
        recordings (small arrays; eigenvectors are unit-norm, left alone)."""
        cache = self._get_cache()
        s = ss = 0.0
        n = 0
        for (subj, run), rec in recs_by_key.items():
            ev = np.asarray(cache.get(subj, run, rec["raw_x"], mmap=True)["eigenvalues"], dtype=np.float64)
            s += float(ev.sum())
            ss += float((ev * ev).sum())
            n += ev.size
        if n == 0:
            return 0.0, 1.0
        m = s / n
        std = float(np.sqrt(max(ss / n - m * m, 0.0)))
        return m, (std if std > 1e-8 else 1.0)

    def _loader(self, dataset, indices, *, shuffle: bool):
        sub = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(
            sub, batch_size=self.batch_size, shuffle=shuffle, num_workers=0
        )

    # -- fit -------------------------------------------------------------

    def fit(self, recordings: list[dict]) -> "HermitianSSMClassifier":
        set_seed(self.seed)
        self.device_ = resolve_torch_device(self.device)
        recs_by_key, index, y = self._build_index(recordings)
        uniq = np.unique(y)
        self.classes_ = uniq if len(uniq) > 1 else np.array([0, 1])
        self._val_norm = self._eigenvalue_norm(recs_by_key)

        cache = self._get_cache()
        first_key = next(iter(recs_by_key))
        probe = cache.get(first_key[0], first_key[1], recs_by_key[first_key]["raw_x"], mmap=True)
        n_freqs, k = probe["eigenvalues"].shape[1], probe["eigenvalues"].shape[2]
        n_channels = probe["eigenvectors"].shape[3]
        freqs = np.asarray(probe["freqs"]) if self.freq_feature else None

        dataset = _WindowDataset(cache, recs_by_key, index, y, self._val_norm)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(index))
        n_val = int(round(self.validation_split * len(index))) if self.validation_split else 0
        val_idx, tr_idx = perm[:n_val].tolist(), perm[n_val:].tolist()

        encoder = _SpectralEncoder(
            n_channels, n_freqs, k, self.d_model, self.d_mode, self.d_freq,
            freqs=freqs, mode_feature=self.mode_feature,
        )
        temporal = _MambaTemporalHead(
            self.d_model, d_state=self.mamba_d_state, d_conv=self.mamba_d_conv,
            expand=self.mamba_expand, n_layers=self.mamba_n_layers, dropout=self.mamba_dropout,
            chunk_size=self.mamba_chunk_size, head_hidden=self.head_hidden,
        )
        self.model_ = _HermitianSSMNet(encoder, temporal).to(self.device_)

        y_tr = y[tr_idx]
        if self.use_class_weights and (y_tr == 1).any() and (y_tr == 0).any():
            n_pos, n_neg, n_all = float((y_tr == 1).sum()), float((y_tr == 0).sum()), float(len(y_tr))
            w = torch.tensor([n_all / (2 * n_neg), n_all / (2 * n_pos)], dtype=torch.float32, device=self.device_)
        else:
            w = None
        criterion = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        train_loader = self._loader(dataset, tr_idx, shuffle=True)
        val_loader = self._loader(dataset, val_idx, shuffle=False) if n_val > 0 else None

        best_val, best_state, bad = float("inf"), None, 0
        for epoch in range(self.epochs):
            epoch_t0 = time.perf_counter()
            self.model_.train()
            tot, seen = 0.0, 0
            for ev_b, ur_b, ui_b, y_b in train_loader:
                evb = ev_b.to(self.device_)
                ub = torch.complex(ur_b, ui_b).to(self.device_)
                yb = y_b.to(self.device_)
                opt.zero_grad()
                loss = criterion(self.model_(evb, ub), yb)
                loss.backward()
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip_norm)
                opt.step()
                tot += float(loss.item()) * len(yb)
                seen += len(yb)
            tr_loss = tot / max(1, seen)

            if val_loader is not None:
                val_loss, vauc = self._eval_loader(val_loader, criterion)
                improved = val_loss < best_val - 1e-5
                if improved:
                    best_val = val_loss
                    best_state = {kk: vv.detach().cpu().clone() for kk, vv in self.model_.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if self.verbose:
                    print(f"  [hermitian_ssm] epoch {epoch + 1}/{self.epochs} "
                          f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_auc={vauc:.3f} "
                          f"({time.perf_counter() - epoch_t0:.1f}s)"
                          + (" *" if improved else ""))
                if self.early_stopping_patience and bad >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"  [hermitian_ssm] early stop at epoch {epoch + 1}")
                    break
            elif self.verbose:
                print(f"  [hermitian_ssm] epoch {epoch + 1}/{self.epochs} train_loss={tr_loss:.4f} "
                      f"({time.perf_counter() - epoch_t0:.1f}s)")

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _eval_loader(self, loader, criterion) -> tuple[float, float]:
        self.model_.eval()
        loss_sum, seen, ys, ps = 0.0, 0, [], []
        with torch.no_grad():
            for ev_b, ur_b, ui_b, y_b in loader:
                evb = ev_b.to(self.device_)
                ub = torch.complex(ur_b, ui_b).to(self.device_)
                yb = y_b.to(self.device_)
                logits = self.model_(evb, ub)
                loss_sum += float(criterion(logits, yb).item()) * len(yb)
                seen += len(yb)
                ys.append(yb.cpu().numpy())
                ps.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        y_all, p_all = np.concatenate(ys), np.concatenate(ps)
        try:
            auc = roc_auc_score(y_all, p_all) if len(np.unique(y_all)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        return loss_sum / max(1, seen), auc

    # -- predict --------------------------------------------------------

    def predict_proba(self, recordings: list[dict]) -> list[np.ndarray]:
        if self.model_ is None:
            raise ValueError("not fitted")
        self.model_.eval()
        cache = self._get_cache()
        out: list[np.ndarray] = []
        for rec in recordings:
            recs_by_key, index, y = self._build_index([rec])
            dataset = _WindowDataset(cache, recs_by_key, index, y, self._val_norm)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
            probs = np.zeros((len(index), 2), dtype=np.float32)
            b0 = 0
            with torch.no_grad():
                for ev_b, ur_b, ui_b, _y in loader:
                    evb = ev_b.to(self.device_)
                    ub = torch.complex(ur_b, ui_b).to(self.device_)
                    logits = self.model_(evb, ub)
                    p = torch.softmax(logits, dim=1).cpu().numpy()
                    probs[b0 : b0 + len(p)] = p
                    b0 += len(p)
            out.append(probs)
        return out


class _WindowDataset(torch.utils.data.Dataset):
    """Lazily slices one window's spectral features out of the per-recording
    mmap-backed cache. COI-invalid ``(timestep, frequency)`` entries and any
    right-padding (edge windows) contribute a zero token."""

    def __init__(self, cache: HermitianSpectralCache, recs_by_key: dict, index: list[dict],
                 y: np.ndarray, val_norm: tuple[float, float]) -> None:
        self.cache = cache
        self.recs = recs_by_key
        self.index = index
        self.y = y
        self.m, self.s = val_norm
        self._open: dict[tuple, dict] = {}

    def _arrays(self, key: tuple) -> dict:
        a = self._open.get(key)
        if a is None:
            rec = self.recs[key]
            a = self.cache.get(rec["subject"], rec["run"], rec["raw_x"], mmap=True)
            self._open[key] = a
        return a

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        e = self.index[i]
        a = self._arrays((e["subject"], e["run"]))
        ev_all, u_all, coi = a["eigenvalues"], a["eigenvectors"], a["coi_valid"]
        t0, tw = e["t0"], e["tw"]
        t_total, f, k = ev_all.shape[0], ev_all.shape[1], ev_all.shape[2]
        c = u_all.shape[3]
        t1 = min(t0 + tw, t_total)
        n = max(0, t1 - t0)

        ev = np.zeros((tw, f, k), dtype=np.float32)
        ur = np.zeros((tw, f, k, c), dtype=np.float32)
        ui = np.zeros((tw, f, k, c), dtype=np.float32)
        if n > 0:
            ev_s = np.array(ev_all[t0:t1], dtype=np.float32)
            packed = u_all.ndim == 5                         # float16 [n,F,k,C,2] vs complex64 [n,F,k,C]
            u_s = np.asarray(u_all[t0:t1]).astype(np.float32 if packed else np.complex64)
            invalid = ~np.array(coi[t0:t1])                  # [n, F]
            ev_s = (ev_s - self.m) / self.s
            ev_s[invalid] = 0.0
            u_s[invalid] = 0.0
            ev[:n] = ev_s
            if packed:
                ur[:n] = u_s[..., 0]
                ui[:n] = u_s[..., 1]
            else:
                ur[:n] = u_s.real
                ui[:n] = u_s.imag
        return (
            torch.from_numpy(ev),
            torch.from_numpy(ur),
            torch.from_numpy(ui),
            torch.tensor(int(self.y[i]), dtype=torch.long),
        )
