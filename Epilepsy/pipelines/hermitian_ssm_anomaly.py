"""HermitianSSMAnomaly: seizure *prediction as graph-state anomaly
detection* (2026-08-30).

Reframing of ``hermitian_ssm``. Instead of a supervised 2-class
preictal/interictal head trained on a handful (~30/fold) of preictal
windows, this trains a self-supervised next-token predictor on the
*interictal* coherence-graph stream ONLY -- what "normal" graph dynamics
look like -- and scores every test window by how badly the model predicts
its own next graph-state token. Preictal windows should be harder to
predict (higher surprise) if seizure onset is preceded by a graph-state
regime the model has never seen.

Why: every supervised lever on this pipeline has been capped by the
~30-preictal-window data budget (see CONTEXT.md fragility thesis). Anomaly
detection never tries to learn preictal features -- it learns the
abundant interictal manifold and flags departures. No class balancing, no
negative subsampling, no preictal labels in the loss.

Shares the whole deterministic half with ``hermitian_ssm_classifier``:
the disk cache (``HermitianSpectralConfig`` key), ``_WindowDataset``, and
every spectral encoder. Only the head, the loss, and train/score differ.

Interface: ``fit(recordings)`` / ``predict_proba(recordings)`` /
``classes_`` -- so ``leave_one_seizure_out_hermitian_ssm`` in
run_pipelines.py drives it unchanged (a wrapper swaps the class).
``predict_proba`` returns one ``[n_windows, 2]`` array per recording;
column 1 is the normalised anomaly score in (0, 1).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Epilepsy.pipelines.common import resolve_torch_device, set_seed
from Epilepsy.pipelines.hermitian_ssm_cache import (
    HermitianSpectralCache,
    HermitianSpectralConfig,
)
from Epilepsy.pipelines.hermitian_ssm_classifier import (
    _ComplexMatrixEncoder,
    _ComplexSpectralEncoder,
    _GraphEncoder,
    _ProjectorEncoder,
    _SpectralEncoder,
    _WindowDataset,
)


class _SeqMamba(nn.Module):
    """mambapy Mamba over ``[B, T, d]`` returning the FULL causal output
    sequence ``[B, T, d]`` (not last-timestep-pooled like
    ``_MambaTemporalHead``). Plus a linear predictor head."""

    def __init__(self, d_model: int, *, d_state: int, d_conv: int, expand: int,
                 n_layers: int) -> None:
        super().__init__()
        from mambapy.mamba import Mamba, MambaConfig

        self.mamba = Mamba(MambaConfig(
            d_model=d_model, n_layers=n_layers, d_state=d_state,
            expand_factor=expand, d_conv=d_conv, use_cuda=False,
        ))
        self.predict = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.predict(self.mamba(tokens))         # [B, T, d]


class _AnomalyNet(nn.Module):
    """Predicts the ``horizon``-step *change* in the (scale-fixed) token
    stream from the SSM state. Predicting the h-step delta rather than the
    next token defeats the "just copy the current token" collapse: with a
    slow, time-smoothed coherence stream, next-token MSE -> ~0 by
    persistence and carries no anomaly signal (confirmed 2026-08-30, fold
    1 AP 0.029). The h-step delta has real structure and nonzero
    magnitude, so persistence (predict 0) is a genuine error, and a window
    whose dynamics differ from interictal norms scores higher surprise."""

    def __init__(self, encoder: nn.Module, seq_model: _SeqMamba, *, horizon: int = 8) -> None:
        super().__init__()
        self.encoder = encoder
        self.seq = seq_model
        self.horizon = int(horizon)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        tokens = self.encoder(feat_a, feat_b)          # [B, T, d_model]
        tokens = F.layer_norm(tokens, (tokens.shape[-1],))   # scale-fix
        pred = self.seq(tokens)                         # [B, T, d_model]
        h = self.horizon
        delta = tokens[:, h:] - tokens[:, :-h]          # h-step change
        # Per-step unit-normalise the target so the task is predicting the
        # *direction* of change, not its (encoder-dependent, often tiny)
        # magnitude -- persistence (predict 0) then costs mean(target^2)=1,
        # a real error, and surprise measures shape mismatch.
        target = F.layer_norm(delta, (delta.shape[-1],)).detach()
        return pred[:, :-h], target


def _per_window_surprise(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared h-step-delta prediction error over (T-h, d) -> [B]."""
    return ((pred - target) ** 2).mean(dim=(1, 2))


class HermitianSSMAnomaly:
    model_label = "Hermitian-SSM-Anomaly"

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
        seed: int = 42,
        device: str = "cpu",
        verbose: int = 1,
        spectral_config: HermitianSpectralConfig | None = None,
        cache_root: str | None = None,
        precompute_device: str = "cpu",
        encoder_mode: str = "eigenvector",
        canonicalize_eigenvectors: bool = False,
        d_model: int = 64,
        d_mode: int = 32,
        d_freq: int = 64,
        freq_feature: bool = True,
        mode_feature: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_n_layers: int = 1,
        anomaly_horizon: int = 8,   # predict the h-step token delta, not next-token
        **_ignored: object,   # swallow classify-only knobs (mamba_backend, head_hidden, ...)
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.cfg = spectral_config or HermitianSpectralConfig()
        self.cache_root = cache_root
        self.precompute_device = precompute_device
        if encoder_mode not in ("eigenvector", "complex", "matrix", "projector", "graph"):
            raise ValueError(
                "anomaly encoder_mode must be 'eigenvector', 'complex', 'matrix', "
                f"'projector' or 'graph' (not 'evolution'), got {encoder_mode!r}"
            )
        self.encoder_mode = encoder_mode
        self.canonicalize_eigenvectors = bool(canonicalize_eigenvectors)
        self.d_model = d_model
        self.d_mode = d_mode
        self.d_freq = d_freq
        self.freq_feature = freq_feature
        self.mode_feature = mode_feature
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_n_layers = mamba_n_layers
        self.anomaly_horizon = int(anomaly_horizon)

        self.model_ = None
        self.classes_ = np.array([0, 1])
        self.device_ = None
        self._cache: HermitianSpectralCache | None = None
        self._val_norm = (0.0, 1.0)
        self._score_mu = 0.0
        self._score_sd = 1.0

    # -- shared cache / index plumbing (mirrors HermitianSSMClassifier) --

    def _get_cache(self) -> HermitianSpectralCache:
        if self._cache is None:
            self._cache = HermitianSpectralCache(
                self.cfg, root=self.cache_root, device=self.precompute_device,
                verbose=bool(self.verbose),
            )
        return self._cache

    def _build_index(self, recordings: list[dict]):
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
        cache = self._get_cache()
        s = ss = 0.0
        n = 0
        for (subj, run), rec in recs_by_key.items():
            ev = np.asarray(cache.get(subj, run, rec["raw_x"], mmap=True)["eigenvalues"], dtype=np.float64)
            s += float(ev.sum()); ss += float((ev * ev).sum()); n += ev.size
        if n == 0:
            return 0.0, 1.0
        m = s / n
        std = float(np.sqrt(max(ss / n - m * m, 0.0)))
        return m, (std if std > 1e-8 else 1.0)

    def _item_mode(self) -> str:
        return {"graph": "graph"}.get(self.encoder_mode, "eigenpairs")

    def _build_encoder(self, n_channels: int, n_freqs: int, k: int, freqs):
        m = self.encoder_mode
        if m == "projector":
            return _ProjectorEncoder(n_channels, n_freqs, k, self.d_model, self.d_freq,
                                     freqs=freqs, fuse_freq=True)
        if m == "graph":
            return _GraphEncoder(n_channels, n_freqs, k, self.d_model, self.d_freq,
                                 freqs=freqs, fuse_freq=True)
        if m == "complex":
            return _ComplexSpectralEncoder(n_channels, n_freqs, k, self.d_model,
                                           self.d_mode, self.d_freq, fuse_freq=True)
        if m == "matrix":
            return _ComplexMatrixEncoder(n_channels, n_freqs, k, self.d_model, self.d_freq,
                                         fuse_freq=True)
        return _SpectralEncoder(n_channels, n_freqs, k, self.d_model, self.d_mode, self.d_freq,
                                freqs=freqs, mode_feature=self.mode_feature, fuse_freq=True)

    # -- fit: interictal windows only ----------------------------------

    def fit(self, recordings: list[dict]) -> "HermitianSSMAnomaly":
        set_seed(self.seed)
        self.device_ = resolve_torch_device(self.device)
        recs_by_key, index, y = self._build_index(recordings)
        self._val_norm = self._eigenvalue_norm(recs_by_key)
        if self.encoder_mode in ("projector", "graph", "matrix"):
            self._val_norm = (0.0, self._val_norm[1])

        cache = self._get_cache()
        fk = next(iter(recs_by_key))
        probe = cache.get(fk[0], fk[1], recs_by_key[fk]["raw_x"], mmap=True)
        n_freqs, k = probe["eigenvalues"].shape[1], probe["eigenvalues"].shape[2]
        n_channels = probe["eigenvectors"].shape[3]
        freqs = np.asarray(probe["freqs"]) if self.freq_feature else None

        dataset = _WindowDataset(
            cache, recs_by_key, index, y, self._val_norm,
            item_mode=self._item_mode(), canonicalize=self.canonicalize_eigenvectors,
        )

        # INTERICTAL ONLY -- the whole point. No preictal in train or val.
        interictal = np.where(y == 0)[0]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(interictal)
        n_val = int(round(self.validation_split * len(interictal))) if self.validation_split else 0
        val_idx = interictal[:n_val].tolist()
        tr_idx = interictal[n_val:].tolist()
        if self.verbose:
            print(f"  [anomaly] train interictal windows={len(tr_idx)} val={len(val_idx)} "
                  f"(dropped {int((y == 1).sum())} preictal from training)")

        encoder = self._build_encoder(n_channels, n_freqs, k, freqs)
        seq = _SeqMamba(self.d_model, d_state=self.mamba_d_state, d_conv=self.mamba_d_conv,
                        expand=self.mamba_expand, n_layers=self.mamba_n_layers)
        self.model_ = _AnomalyNet(encoder, seq, horizon=self.anomaly_horizon).to(self.device_)
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
            for ev_b, ur_b, ui_b, _y in train_loader:
                evb = ev_b.to(self.device_)
                ub = torch.complex(ur_b, ui_b).to(self.device_)
                opt.zero_grad()
                pred, target = self.model_(evb, ub)
                loss = ((pred - target) ** 2).mean()
                loss.backward()
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip_norm)
                opt.step()
                tot += float(loss.item()) * len(evb); seen += len(evb)
            tr_loss = tot / max(1, seen)

            if val_loader is not None:
                vl = self._val_loss(val_loader)
                improved = vl < best_val - 1e-6
                if improved:
                    best_val = vl
                    best_state = {kk: vv.detach().cpu().clone()
                                  for kk, vv in self.model_.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if self.verbose:
                    print(f"  [anomaly] epoch {epoch + 1}/{self.epochs} "
                          f"train_mse={tr_loss:.5f} val_mse={vl:.5f} "
                          f"({time.perf_counter() - t0:.1f}s)" + (" *" if improved else ""))
                if self.early_stopping_patience and bad >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"  [anomaly] early stop at epoch {epoch + 1}")
                    break
            elif self.verbose:
                print(f"  [anomaly] epoch {epoch + 1}/{self.epochs} train_mse={tr_loss:.5f} "
                      f"({time.perf_counter() - t0:.1f}s)")

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Calibrate the sigmoid against the training interictal score
        # distribution. Centre at the 90th percentile so the p>0.5
        # operating point (argmax -> "predicted preictal", what the LOSO
        # driver's hit/FAR uses) means "more surprising than 90% of normal
        # windows" -- a sane ~10% interictal-FAR baseline, not the median
        # split. AP / ROC-AUC use the continuous score and are unaffected.
        scores = self._raw_scores(_loader(tr_idx, False))
        q50, q90 = np.percentile(scores, [50, 90])
        self._score_mu = float(q90)
        self._score_sd = float(max(q90 - q50, 1e-6))
        if self.verbose:
            print(f"  [anomaly] interictal score calib: q50={q50:.5f} q90={q90:.5f} "
                  f"-> centre={self._score_mu:.5f} scale={self._score_sd:.5f}")
        return self

    def _val_loss(self, loader) -> float:
        self.model_.eval()
        tot, seen = 0.0, 0
        with torch.no_grad():
            for ev_b, ur_b, ui_b, _y in loader:
                evb = ev_b.to(self.device_)
                ub = torch.complex(ur_b, ui_b).to(self.device_)
                pred, target = self.model_(evb, ub)
                loss = ((pred - target) ** 2).mean()
                tot += float(loss.item()) * len(evb); seen += len(evb)
        return tot / max(1, seen)

    def _raw_scores(self, loader) -> np.ndarray:
        self.model_.eval()
        out = []
        with torch.no_grad():
            for ev_b, ur_b, ui_b, _y in loader:
                evb = ev_b.to(self.device_)
                ub = torch.complex(ur_b, ui_b).to(self.device_)
                pred, target = self.model_(evb, ub)
                out.append(_per_window_surprise(pred, target).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    # -- predict: per-window anomaly score in (0, 1) ------------------

    def predict_proba(self, recordings: list[dict]) -> list[np.ndarray]:
        if self.model_ is None:
            raise ValueError("not fitted")
        cache = self._get_cache()
        out: list[np.ndarray] = []
        for rec in recordings:
            recs_by_key, index, y = self._build_index([rec])
            dataset = _WindowDataset(
                cache, recs_by_key, index, y, self._val_norm,
                item_mode=self._item_mode(), canonicalize=self.canonicalize_eigenvectors,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
            raw = self._raw_scores(loader)
            z = (raw - self._score_mu) / self._score_sd
            p = 1.0 / (1.0 + np.exp(-z))
            probs = np.stack([1.0 - p, p], axis=1).astype(np.float32)
            out.append(probs)
        return out
