"""Numerical validation for the hermitian_ssm spectral precompute
(``Epilepsy/hermitian_ssm.md`` Section 20).

Two parts:
  1. Synthetic Hermitian smoke -- small random Hermitian matrices, plus a
     known low-rank matrix, run through the eigh -> |lambda|-sort ->
     phase-gauge -> rank-k reconstruction path.
  2. Real-pipeline checks -- run ``compute_recording_spectral`` on a short
     synthetic multichannel signal and verify Hermiticity /
     reconstruction / gauge invariance on the graphs it builds internally.

Run:  python scripts/hermitian_ssm_numerical_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Epilepsy.pipelines.hermitian_ssm_cache import (  # noqa: E402
    HermitianSpectralConfig,
    _MorletCWT,
    _gaussian_kernel1d,
    _mean_pool_time,
    _smooth_time,
    compute_recording_spectral,
)

TOL = 1e-4


def _canon_phase(vecs: torch.Tensor) -> torch.Tensor:
    """Same gauge as compute_recording_spectral: rotate so the largest-|.|
    component is real, non-negative. vecs: [..., k, C]."""
    amax = torch.argmax(vecs.abs(), dim=-1, keepdim=True)
    anchor = torch.gather(vecs, -1, amax)
    return vecs * torch.exp(-1j * torch.angle(anchor))


def part1_synthetic() -> bool:
    print("=== Part 1: synthetic Hermitian matrices ===")
    torch.manual_seed(0)
    ok = True
    N, B = 5, 64
    tri = torch.randn(B, N, N, dtype=torch.complex64)
    A = torch.triu(tri, diagonal=1)
    A = A + A.conj().transpose(-1, -2)
    A = A + torch.diag_embed(torch.randn(B, N))  # real diagonal

    herm_err = (A - A.conj().transpose(-1, -2)).abs().max().item()
    print(f"  Hermiticity  ||A - A^H||_max = {herm_err:.2e}")
    ok &= herm_err < TOL

    evals, evecs = torch.linalg.eigh(A)
    imag_ev = evals.imag.abs().max().item() if evals.is_complex() else 0.0
    print(f"  eigenvalues real to {imag_ev:.2e}")

    ortho = (evecs.conj().transpose(-1, -2) @ evecs - torch.eye(N)).abs().max().item()
    print(f"  orthogonality  ||U^H U - I||_max = {ortho:.2e}")
    ok &= ortho < TOL

    recon = (evecs @ torch.diag_embed(evals.to(torch.complex64)) @ evecs.conj().transpose(-1, -2) - A).abs().max().item()
    print(f"  full reconstruction  ||U L U^H - A||_max = {recon:.2e}")
    ok &= recon < TOL

    # |lambda|-desc sort + rank-k, check discarded Frobenius energy identity.
    order = torch.argsort(evals.abs(), dim=-1, descending=True)
    evals_s = torch.gather(evals, -1, order)
    evecs_s = torch.gather(evecs, -1, order.unsqueeze(-2).expand(-1, N, -1))
    for k in (1, 2, 3):
        Uk = evecs_s[..., :k]
        Lk = torch.diag_embed(evals_s[..., :k].to(torch.complex64))
        Ak = Uk @ Lk @ Uk.conj().transpose(-1, -2)
        measured = ((A - Ak).norm(dim=(-1, -2)) ** 2)
        theoretical = (evals_s[..., k:] ** 2).sum(-1)
        rel = (measured - theoretical).abs().max().item()
        print(f"  rank-{k}: max|measured - sum_r>k lambda_r^2| = {rel:.2e}")
        ok &= rel < 1e-3

    # Phase-gauge idempotence: multiply by arbitrary e^{i theta}, re-canon.
    v = evecs_s[..., :2].transpose(-1, -2)  # [B, k, C]
    c1 = _canon_phase(v)
    theta = torch.rand(v.shape[0], 2, 1) * 7.0
    c2 = _canon_phase(v * torch.exp(1j * theta))
    gauge = (c1 - c2).abs().max().item()
    print(f"  phase-gauge invariance  max|canon(u) - canon(e^{{i0}} u)| = {gauge:.2e}")
    ok &= gauge < TOL

    # Known low-rank matrix: A = 3 u u^H + 1 v v^H, u,v orthonormal.
    q, _ = torch.linalg.qr(torch.randn(N, 2, dtype=torch.complex64))
    A_lr = 3.0 * (q[:, [0]] @ q[:, [0]].conj().T) + 1.0 * (q[:, [1]] @ q[:, [1]].conj().T)
    lr_vals = torch.linalg.eigvalsh(A_lr)
    top2 = torch.sort(lr_vals.abs(), descending=True).values[:2]
    print(f"  known low-rank spectrum top-2 |lambda| = {top2.tolist()} (expect ~[3, 1])")
    ok &= abs(top2[0].item() - 3.0) < 1e-3 and abs(top2[1].item() - 1.0) < 1e-3

    print(f"  Part 1: {'PASS' if ok else 'FAIL'}")
    return ok


def part2_pipeline() -> bool:
    print("\n=== Part 2: compute_recording_spectral internals ===")
    rng = np.random.default_rng(1)
    cfg = HermitianSpectralConfig(nfreqs=12, highest=60.0, mains_notch=False)
    C, N = 8, int(256 * 20)
    # Two coupled channels + noise, so some real coherence structure exists.
    t = np.arange(N) / cfg.sampling_rate
    base = np.sin(2 * np.pi * 20 * t)
    raw = rng.standard_normal((C, N)) * 5.0
    raw[0] += 10 * base
    raw[1] += 10 * np.roll(base, 7)
    raw = raw.astype(np.float32)

    out = compute_recording_spectral(raw, cfg, device="cpu", freq_chunk=4)
    ev, U = out["eigenvalues"], out["eigenvectors"]
    ok = True

    finite = np.isfinite(ev).all() and np.isfinite(U).all()
    print(f"  outputs finite: {finite}")
    ok &= bool(finite)

    # Rebuild rank-k A from cache, check it is Hermitian and that its
    # top-k spectrum matches the stored eigenvalues.
    Ut = torch.from_numpy(U[:50])                       # [T,F,k,C]
    Lt = torch.from_numpy(ev[:50]).to(torch.complex64)  # [T,F,k]
    Ak = torch.einsum("tfkc,tfk,tfkd->tfcd", Ut, Lt, Ut.conj())
    herm = (Ak - Ak.conj().transpose(-1, -2)).abs().max().item()
    print(f"  rank-k A Hermiticity  ||Ak - Ak^H||_max = {herm:.2e}")
    ok &= herm < 1e-3

    re_vals = torch.linalg.eigvalsh(Ak)                 # ascending
    re_top = torch.sort(re_vals.abs(), dim=-1, descending=True).values[..., : cfg.k]
    stored_top = torch.sort(torch.from_numpy(ev[:50]).abs(), dim=-1, descending=True).values
    spec_err = (re_top - stored_top).abs().max().item()
    print(f"  rank-k spectrum vs stored eigenvalues  max err = {spec_err:.2e}")
    ok &= spec_err < 1e-2

    # Gauge determinism: recompute, expect bit-identical eigenvectors.
    out2 = compute_recording_spectral(raw, cfg, device="cpu", freq_chunk=6)
    dv = np.abs(out["eigenvectors"] - out2["eigenvectors"]).max()
    dl = np.abs(out["eigenvalues"] - out2["eigenvalues"]).max()
    print(f"  determinism across freq_chunk (4 vs 6): d|U|={dv:.2e}  d|lambda|={dl:.2e}")
    ok &= dv < 1e-4 and dl < 1e-4

    print(f"  Part 2: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    a = part1_synthetic()
    b = part2_pipeline()
    print(f"\nOVERALL: {'PASS' if (a and b) else 'FAIL'}")
    sys.exit(0 if (a and b) else 1)
