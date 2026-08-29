"""Complex-diagonal selective SSM ("Mamba-3" style) -- an alternative
temporal backend for ``hermitian_ssm``.

Difference from mambapy's ``Mamba`` (Mamba-1/2, *real* diagonal state):
the per-channel state eigenvalue is complex,

    lambda = -exp(a) + i * omega          a, omega learned per (d_inner, d_state)

so every state channel has an exponential decay rate ``exp(-a)`` **and** a
rotation frequency ``omega``. A real-diagonal SSM can only accumulate or
decay a state channel; a complex one can track an *evolving phase* -- which
is exactly what a sequence of cross-spectral coherence matrices (a Hermitian
graph over time), or subspace-evolution operators, carries. See the
"MAMBA-3 INTEGRATION POINT" note in ``hermitian_ssm_classifier.py``.

Selective (Mamba) part kept: ``Delta``, ``B``, ``C`` are input-dependent
(``C`` complex so the readout can resolve phase).

Contract mirrors ``_DenseEdgeMambaTemporal`` exactly:
``[B, C_in, E, T] -> [B, out_channels, E, 1]`` (last-timestep pool), so it
is a drop-in temporal backend -- ``E`` may index edges, nodes, frequencies
or be 1, the block does not care.

Scan = ``_ComplexPScan``: mambapy's O(T log T) Blelloch parallel scan
(``PScan.pscan`` / ``pscan_rev`` are dtype-generic -- pure elementwise
mul/add) wrapped in a complex-correct ``autograd.Function``. mambapy's own
``PScan.backward`` is wrong for complex ``A`` -- it doesn't conjugate the
``A`` path. Derivation (torch's convention: ``.grad`` = conjugate cotangent;
for ``y = a*x`` -> ``grad_x = conj(a) grad_y``, ``grad_a = conj(x) grad_y``):

    h_t = A_t h_{t-1} + x_t
    G_t = grad_H[t] + conj(A_{t+1}) G_{t+1}          (reverse scan, weights conj(A) shifted)
    grad_x_t = G_t
    grad_A_t = conj(h_{t-1}) G_t                     (h_0 = 0 -> grad_A_1 = 0)

Verified 2026-08-29: ``_ComplexPScan`` and full ``_Mamba3Block`` pass
``torch.autograd.gradcheck`` in float64 (po2 and non-po2 L); forward
matches a naive loop to fp32 precision; ~2.1x a real mambapy pscan
fwd+bwd at B=64, T=480, d_state=16 (usable -- epoch ~2x "mamba").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.pscan import PScan as _RealPScan, npo2, pad_npo2


class _ComplexPScan(torch.autograd.Function):
    """H_t = A_t * H_{t-1} + X_t (elementwise, complex). A, X: [B, L, D, N]."""

    @staticmethod
    def forward(ctx, A_in, X_in):
        L = X_in.size(1)
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)
        A = A.transpose(2, 1).contiguous()          # [B, D, Lp, N]
        X = X.transpose(2, 1).contiguous()
        _RealPScan.pscan(A, X)                      # in-place; H now in X
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_in):
        A_in, H = ctx.saved_tensors                 # H: [B, D, Lp, N]
        L = grad_in.size(1)
        if L == npo2(L):
            grad = grad_in.clone()
        else:
            grad = pad_npo2(grad_in)
            A_in = pad_npo2(A_in)
        grad = grad.transpose(2, 1).contiguous()    # [B, D, Lp, N]
        A_t = A_in.transpose(2, 1)
        # weight for G_t is conj(A_{t+1}): shift left by 1, conjugate, materialise
        A_shift = F.pad(A_t[:, :, 1:], (0, 0, 0, 1)).conj().contiguous()
        _RealPScan.pscan_rev(A_shift, grad)         # grad now = G
        gradA = torch.zeros_like(H)
        gradA[:, :, 1:] = H[:, :, :-1].conj() * grad[:, :, 1:]
        return gradA.transpose(2, 1)[:, :L], grad.transpose(2, 1)[:, :L]


class _Mamba3Block(nn.Module):
    """One complex-diagonal selective-SSM layer. Operates on ``[rows, T, d_model]``."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True,
        )
        # x -> (Delta_lowrank, B [real, d_state], C [complex, 2*d_state])
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # lambda = -exp(A_log) + i * omega
        a_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(a_init.clone())                         # [d_inner, d_state]
        self.omega = nn.Parameter(0.01 * torch.randn(self.d_inner, d_state))  # small: ~Mamba-1 at init
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt_proj bias init so softplus(bias) ~ dt in [1e-3, 1e-1] (mambapy convention)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3)
        ).clamp_min(1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _scan(self, delta: torch.Tensor, Bm: torch.Tensor, Cm: torch.Tensor,
              x: torch.Tensor) -> torch.Tensor:
        """delta [R,T,di], Bm [R,T,ds] real, Cm [R,T,ds] complex, x [R,T,di].
        Returns y [R,T,di] real."""
        cdt = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        lam = (-torch.exp(self.A_log) + 1j * self.omega).to(cdt)                  # [di,ds]
        deltaA = torch.exp(delta.unsqueeze(-1).to(cdt) * lam)                     # [R,T,di,ds], |.|<=1
        u = (delta.unsqueeze(-1) * x.unsqueeze(-1) * Bm.unsqueeze(2)).to(cdt)     # [R,T,di,ds]
        h = _ComplexPScan.apply(deltaA, u)                                       # [R,T,di,ds]
        return (h * Cm.unsqueeze(2)).sum(-1).real + self.D * x                   # [R,T,di]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [rows, T, d_model] -> [rows, T, d_model]."""
        r, t, _ = seq.shape
        xz = self.in_proj(seq)                                      # [R,T,2*di]
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :t].transpose(1, 2)  # causal
        x = F.silu(x)

        proj = self.x_proj(x)                                       # [R,T,dt_rank+3*ds]
        dt_lr, Br, Cri = torch.split(proj, [self.dt_rank, self.d_state, 2 * self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_lr))                     # [R,T,di]
        Cm = torch.complex(Cri[..., :self.d_state], Cri[..., self.d_state:])  # [R,T,ds]

        y = self._scan(delta, Br, Cm, x)                            # [R,T,di]
        y = y * F.silu(z)
        return self.out_proj(y)                                     # [R,T,d_model]


class _Mamba3Temporal(nn.Module):
    """``[B, C_in, E, T] -> [B, out_channels, E, 1]`` -- same contract as
    ``_DenseEdgeMambaTemporal``, complex-diagonal SSM inside. Weight-shared
    across ``E`` (folded into the row dim), last-timestep pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_proj = nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()
        self.layers = nn.ModuleList([
            _Mamba3Block(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, out_channels)

    def forward(self, conv_in: torch.Tensor) -> torch.Tensor:
        b, c_in, e, t = conv_in.shape
        assert c_in == self.in_channels, (
            f"_Mamba3Temporal built with in_channels={self.in_channels}, got {c_in}"
        )
        seq = conv_in.permute(0, 2, 3, 1).reshape(b * e, t, c_in)   # [B*E, T, C_in]
        seq = self.in_proj(seq)
        for layer, norm in zip(self.layers, self.norms):
            seq = seq + layer(norm(seq))
        pooled = self.dropout(seq[:, -1, :])                        # [B*E, d_model]
        out = self.out_proj(pooled).reshape(b, e, -1)               # [B, E, out]
        return out.permute(0, 2, 1).unsqueeze(-1)                   # [B, out, E, 1]
