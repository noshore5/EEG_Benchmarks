"""Checkpoint-aware training loop for the SzCORE detection model.

Purpose-built (not `GodoyTMCClassifier`) so a run can survive a spot-
instance eviction: every epoch it writes a resume-complete checkpoint, and
`--resume-from` picks the run back up exactly where it stopped. On SIGTERM
(the ~2-minute spot interruption warning) it flushes one last checkpoint
and exits 0 without writing the DONE sentinel, so the relaunch loop knows
to bring up another box. See ``SPOT_TRAINING.md``.

Kept dependency-free beyond torch/numpy: S3 sync shells out to the `aws`
CLI (present on every EEG box) and is a no-op if it's missing, so local
runs work unchanged.

The training recipe mirrors what `train_detector.py` used to get from
``GodoyTMCClassifier``: global scalar z-score (stats fit on train only),
class-weighted cross-entropy, Adam(lr, weight_decay), grad-norm clip,
a random validation split, and "restore best val-loss epoch" early
stopping.
"""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from algo.model import TMCTransformer

CKPT_FORMAT = "szcore-trainer-v1"


# --------------------------------------------------------------------------- #
# S3 helpers -- shell out to the aws CLI; degrade to no-op when unavailable.
# --------------------------------------------------------------------------- #
def _have_aws() -> bool:
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _s3_cp(src: str, dst: str) -> bool:
    try:
        subprocess.run(["aws", "s3", "cp", src, dst], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:  # noqa: BLE001
        print(f"[s3] cp {src} -> {dst} failed: {exc}", flush=True)
        return False


def _s3_exists(uri: str) -> bool:
    try:
        r = subprocess.run(["aws", "s3", "ls", uri], capture_output=True, text=True)
        return r.returncode == 0 and bool(r.stdout.strip())
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Config / checkpoint
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    model_kwargs: dict
    n_channels: int
    n_time: int
    fs: int
    window_s: float
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    seed: int = 42
    use_class_weights: bool = True
    train_subjects: list = field(default_factory=list)

    def hash(self) -> str:
        """Stable digest of everything that must match to resume a run.
        Deliberately excludes `epochs` (extending a run is allowed) and
        `train_subjects` (checked separately with a clearer message)."""
        keys = (
            "model_kwargs n_channels n_time fs window_s batch_size lr "
            "weight_decay grad_clip_norm validation_split seed use_class_weights"
        ).split()
        payload = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _device(spec: str) -> torch.device:
    if spec not in ("auto", None):
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class SpotTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        *,
        device: str = "auto",
        checkpoint_dir: str | Path = "checkpoints",
        s3_prefix: str | None = None,
        verbose: int = 1,
    ) -> None:
        self.cfg = cfg
        self.device = _device(device)
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.s3_prefix = s3_prefix.rstrip("/") if s3_prefix else None
        self._s3_ok = bool(self.s3_prefix) and _have_aws()
        self.verbose = verbose

        self._interrupted = False
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._on_sigterm)

        self.model: nn.Module | None = None
        self.opt: torch.optim.Optimizer | None = None
        self.x_mean_ = 0.0
        self.x_std_ = 1.0
        self.start_epoch = 0
        self.best_val = float("inf")
        self.best_epoch = -1
        self.no_improve = 0
        self.best_model_state: dict | None = None
        self.history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_ap": []}

    # -- signal ------------------------------------------------------------- #
    def _on_sigterm(self, signum, frame):  # noqa: ANN001
        self._log(0, "[spot] SIGTERM received -- will checkpoint and exit after this epoch")
        self._interrupted = True

    def _log(self, level: int, msg: str) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    # -- checkpoint io ---------------------------------------------------- #
    @property
    def _latest_local(self) -> Path:
        return self.ckpt_dir / "latest.pt"

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "format": CKPT_FORMAT,
            "epoch": epoch,  # last COMPLETED epoch (0-indexed)
            "config": self.cfg.__dict__,
            "config_hash": self.cfg.hash(),
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optim_state": self.opt.state_dict(),
            "best_model_state": self.best_model_state,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": list(np.random.get_state()),
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
            "no_improve": self.no_improve,
            "x_mean": self.x_mean_,
            "x_std": self.x_std_,
            "history": self.history,
        }
        tmp = self.ckpt_dir / "latest.pt.tmp"
        torch.save(ckpt, tmp)
        tmp.replace(self._latest_local)  # atomic on POSIX
        # cheap sidecar the keepalive workflow can read without torch
        prog = self.ckpt_dir / "progress.json"
        prog.write_text(json.dumps({
            "epoch": epoch, "best_epoch": self.best_epoch,
            "best_val": self.best_val, "config_hash": self.cfg.hash(),
        }))
        if self._s3_ok:
            # upload straight to latest.pt -- a killed upload just means we
            # resume from the previous epoch, which is fine.
            _s3_cp(str(self._latest_local), f"{self.s3_prefix}/latest.pt")
            _s3_cp(str(prog), f"{self.s3_prefix}/progress.json")
        self._log(1, f"[ckpt] epoch {epoch} saved{' + S3' if self._s3_ok else ''}")

    def _try_load_resume(self, resume_from: str | None) -> dict | None:
        if not resume_from:
            # auto-resume: if an S3 latest.pt exists, use it
            if self._s3_ok and _s3_exists(f"{self.s3_prefix}/latest.pt"):
                resume_from = f"{self.s3_prefix}/latest.pt"
            elif self._latest_local.exists():
                resume_from = str(self._latest_local)
            else:
                return None
        local = self._latest_local
        if resume_from.startswith("s3://"):
            if not _s3_cp(resume_from, str(local)):
                self._log(0, f"[resume] could not fetch {resume_from} -- starting fresh")
                return None
        else:
            local = Path(resume_from)
        if not local.exists():
            return None
        ckpt = torch.load(local, map_location="cpu", weights_only=False)
        if ckpt.get("format") != CKPT_FORMAT:
            raise ValueError(f"checkpoint format {ckpt.get('format')!r} != {CKPT_FORMAT!r}")
        if ckpt.get("config_hash") != self.cfg.hash():
            raise ValueError(
                "checkpoint config_hash mismatch -- refusing to resume a run with "
                "different hyperparameters. Delete the checkpoint or fix the config.\n"
                f"  checkpoint: {ckpt.get('config_hash')}\n  current:    {self.cfg.hash()}"
            )
        return ckpt

    # -- data ------------------------------------------------------------- #
    def _split_and_normalize(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.cfg.seed)
        n = X.shape[0]
        perm = rng.permutation(n)
        n_val = int(round(self.cfg.validation_split * n)) if self.cfg.validation_split else 0
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        self.x_mean_ = float(X[train_idx].mean())
        self.x_std_ = float(X[train_idx].std()) or 1.0

        def _ds(idx):
            xb = (X[idx] - self.x_mean_) / (self.x_std_ + 1e-8)
            return TensorDataset(
                torch.from_numpy(xb.astype(np.float32)),
                torch.from_numpy(y[idx].astype(np.int64)),
            )

        g = torch.Generator().manual_seed(self.cfg.seed)
        train_loader = DataLoader(
            _ds(train_idx), batch_size=self.cfg.batch_size, shuffle=True,
            generator=g, drop_last=False,
        )
        val_loader = (
            DataLoader(_ds(val_idx), batch_size=self.cfg.batch_size, shuffle=False)
            if n_val > 0 else None
        )
        return train_loader, val_loader, y[train_idx]

    # -- fit ------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray, *, resume_from: str | None = None) -> None:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        train_loader, val_loader, y_train = self._split_and_normalize(X, y)

        self.model = TMCTransformer(
            n_channels=self.cfg.n_channels, n_time=self.cfg.n_time, n_classes=2,
            **self.cfg.model_kwargs,
        ).to(self.device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        if self.cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=2).astype(np.float64)
            w = counts.sum() / (2.0 * np.maximum(counts, 1.0))
            class_w = torch.tensor(w, dtype=torch.float32, device=self.device)
        else:
            class_w = None
        criterion = nn.CrossEntropyLoss(weight=class_w)

        ckpt = self._try_load_resume(resume_from)
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state"])
            self.opt.load_state_dict(ckpt["optim_state"])
            torch.set_rng_state(ckpt["torch_rng"])
            st = ckpt["numpy_rng"]
            np.random.set_state((st[0], np.array(st[1], dtype=np.uint32), st[2], st[3], st[4]))
            self.start_epoch = ckpt["epoch"] + 1
            self.best_val = ckpt["best_val"]
            self.best_epoch = ckpt["best_epoch"]
            self.no_improve = ckpt["no_improve"]
            self.best_model_state = ckpt["best_model_state"]
            self.x_mean_ = ckpt["x_mean"]
            self.x_std_ = ckpt["x_std"]
            self.history = ckpt.get("history", self.history)
            self._log(0, f"[resume] continuing from epoch {self.start_epoch} "
                         f"(best_val={self.best_val:.5f} @ epoch {self.best_epoch})")
        else:
            self._log(0, f"[train] fresh start, {self.cfg.epochs} epochs, device={self.device}")

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.perf_counter()
            tr_loss = self._run_epoch(train_loader, criterion, train=True)
            self.history["train_loss"].append(tr_loss)

            if val_loader is not None:
                val_loss, val_ap = self._evaluate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_ap"].append(val_ap)
                improved = val_loss < self.best_val - 1e-12
                if improved:
                    self.best_val = val_loss
                    self.best_epoch = epoch
                    self.no_improve = 0
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    self.no_improve += 1
                self._log(1, f"[epoch {epoch}] train_loss={tr_loss:.5f} "
                             f"val_loss={val_loss:.5f} val_ap={val_ap:.4f} "
                             f"({time.perf_counter()-t0:.1f}s)"
                             + ("  *best*" if improved else ""))
            else:
                self._log(1, f"[epoch {epoch}] train_loss={tr_loss:.5f} "
                             f"({time.perf_counter()-t0:.1f}s)")

            self._save_checkpoint(epoch)

            if self._interrupted:
                self._log(0, "[spot] exiting 0 after checkpoint -- relaunch loop will resume")
                sys.exit(0)

            if (
                val_loader is not None
                and self.cfg.early_stopping_patience is not None
                and self.no_improve >= self.cfg.early_stopping_patience
            ):
                self._log(0, f"[early-stop] no val improvement in "
                             f"{self.no_improve} epochs; best @ {self.best_epoch}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self._log(0, f"[train] restored best epoch {self.best_epoch} "
                         f"(val_loss={self.best_val:.5f})")
        signal.signal(signal.SIGTERM, self._prev_sigterm)

    def _run_epoch(self, loader, criterion, *, train: bool) -> float:
        self.model.train(train)
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            if train:
                self.opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = self.model(xb)
                loss = criterion(logits, yb)
            if train:
                loss.backward()
                if self.cfg.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def _evaluate(self, loader, criterion) -> tuple[float, float]:
        self.model.eval()
        total, n = 0.0, 0
        probs, ys = [], []
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            total += criterion(logits, yb).item() * xb.size(0)
            n += xb.size(0)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            ys.append(yb.cpu().numpy())
        p = np.concatenate(probs)
        yv = np.concatenate(ys)
        return total / max(n, 1), _average_precision(yv, p)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        xb = (X - self.x_mean_) / (self.x_std_ + 1e-8)
        out = []
        for i in range(0, xb.shape[0], 512):
            b = torch.from_numpy(xb[i : i + 512].astype(np.float32)).to(self.device)
            out.append(torch.softmax(self.model(b), dim=1)[:, 1].cpu().numpy())
        p1 = np.concatenate(out) if out else np.empty(0, dtype=np.float32)
        return np.stack([1.0 - p1, p1], axis=1)


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """AP without sklearn: area under the step-wise precision-recall curve."""
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y_true.sum()
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))
