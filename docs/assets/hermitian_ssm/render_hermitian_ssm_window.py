"""'What the SSM sees' — the hermitian_ssm feature pipeline for ONE real
chb01 preictal 30-s window, panel-for-panel analogous to the crypto-repo
figure (aligned series -> returns -> causal Morlet CWT -> complex wavelet
coherence -> Hermitian channel graph -> eigendecomposition -> spectral
feature matrix that the Mamba slides over).

Config = Epilepsy/run_pipelines.py HERMITIAN_SSM_PARAMS spectral_* keys
(HermitianSpectralConfig): 256 Hz, 8-40 Hz, nfreqs=16, mains notch 60/120,
time_downsample=16, smooth_time_steps=5, freq_downsample=2 -> F_out=8,
diagonal="zero", k=6. The deterministic math here mirrors
hermitian_ssm_cache.compute_recording_spectral EXACTLY, including that the
Hermitian graph and its eigendecomposition are NOT frequency-reduced: there
is one C×C Γ(f,t) per (timestep, frequency-bin) pair, eigendecomposed
independently (`eigenvalues [T_ds, F_out, k]`, `eigenvectors [T_ds, F_out,
k, C]` in the real cache) -- panels 5-7 show that per-frequency structure
directly instead of collapsing it to a single graph per timestep.

    .venv/bin/python docs/assets/hermitian_ssm/render_hermitian_ssm_window.py

Output: docs/assets/hermitian_ssm/hermitian_ssm_window.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils.torch_cwt import cwt_torch

FS = 256
WIN_S = 30.0
PAD_S = 12.0                       # context for notch + CWT edges, cropped after
LOW, HIGH, NFREQS = 8.0, 40.0, 16
TDS = 16                           # time_downsample -> 62.5 ms/step
FD = 2                             # freq_downsample -> F_out = 8
SMOOTH_STEPS = 5
K = 6                             # spectral_k
NOTCH = (60.0, 120.0)
EDF = Path.home() / "mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/chb01_03.edf"
END_S = 2600.0                     # chb01_03 onset 2996 s; SPH 300 / SOP 900 -> preictal
BOLD_CH = "FP1-F7"
OUT = Path(__file__).resolve().parent / "hermitian_ssm_window.png"

# ------------------------------------------------------------------ load
import mne
from scipy.signal import iirnotch, tf2sos, sosfiltfilt

raw = mne.io.read_raw_edf(str(EDF), preload=True, verbose="ERROR")
assert abs(raw.info["sfreq"] - FS) < 1e-6
names = raw.ch_names
data = raw.get_data() * 1e6                                   # [C, N] uV
C = len(names)
e0 = int((END_S - WIN_S - PAD_S) * FS)
e1 = int((END_S + PAD_S) * FS)
seg_pad = data[:, e0:e1]                                      # padded

# mains notch (zero-phase IIR), same as _apply_mains_notch
x = seg_pad.astype(np.float64)
for f0 in NOTCH:
    if f0 < FS / 2:
        q = f0 / (2.0 * 3.0)
        b, a = iirnotch(w0=f0, Q=q, fs=FS)
        x = sosfiltfilt(tf2sos(b, a), x, axis=-1)
sig = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)   # [C, Npad]

# ------------------------------------------------------------------ CWT
coeffs, freqs = cwt_torch(sig, FS, LOW, HIGH, NFREQS)         # [C, F, Npad], freqs high->low
# crop to the 30-s window
c0 = int(PAD_S * FS)
c1 = c0 + int(WIN_S * FS)
coeffs = coeffs[:, :, c0:c1].contiguous()                     # [C, 16, 7680]
raw_win = sig[:, c0:c1].numpy()
T_raw = coeffs.shape[-1]
freqs = freqs                                                 # [16]

# ------------------------------------------------------------------ Γ(f,t)  (cache math)
def _gauss1d(w):
    w = w + 1 - w % 2
    s = (w - 1) / 2
    xs = torch.arange(w, dtype=torch.float32) - s
    k = torch.exp(-0.5 * (xs / s) ** 2)
    return k / k.sum()

def _smooth_t(z, k):                                          # z [..., T]
    lead, t = z.shape[:-1], z.shape[-1]
    zf = F.pad(z.reshape(-1, 1, t), (k.numel() // 2,) * 2, mode="reflect")
    return F.conv1d(zf, k.view(1, 1, -1)).reshape(*lead, t)

def _pool_t(z, f):                                            # mean pool last axis
    t = (z.shape[-1] // f) * f
    return z[..., :t].reshape(*z.shape[:-1], t // f, f).mean(-1)

def _pool_f(z, f):                                            # mean pool axis -2 (freq)
    lead, nf, t = z.shape[0], z.shape[1], z.shape[2]
    return z.reshape(lead, nf // f, f, t).mean(2)

w_ds = _pool_t(coeffs, TDS)                                   # [C, 16, T_ds] complex
T_ds = w_ds.shape[-1]
kern = _gauss1d(SMOOTH_STEPS)
auto = _smooth_t(w_ds.real ** 2 + w_ds.imag ** 2, kern)      # [C, 16, T_ds]
iu, ju = torch.triu_indices(C, C, offset=1)
xwt = w_ds[iu] * torch.conj(w_ds[ju])                        # [P, 16, T_ds]
xr = _smooth_t(xwt.real, kern)
xi = _smooth_t(xwt.imag, kern)
xr, xi, auto = _pool_f(xr, FD), _pool_f(xi, FD), _pool_f(auto, FD)   # -> F_out=8
F_out = auto.shape[1]
freqs_out = torch.exp(torch.log(freqs.double()).reshape(F_out, FD).mean(1)).float()
denom = torch.sqrt(auto[iu] * auto[ju] + 1e-12)
coh = (torch.sqrt(xr ** 2 + xi ** 2 + 1e-20) / denom).clamp(0, 1)   # [P, 8, T_ds]
phase = torch.atan2(xi, xr)
gamma_off = coh * torch.exp(1j * phase)                      # [P, 8, T_ds] complex

# assemble Γ [T_ds, F_out, C, C], diagonal = 0
G = torch.zeros(T_ds, F_out, C, C, dtype=torch.complex64)
off = gamma_off.permute(2, 1, 0)                             # [T_ds, F_out, P]
G[:, :, iu, ju] = off
G[:, :, ju, iu] = torch.conj(off)

# per-freq eigendecomposition (k modes) — the actual cache output
evals_f, evecs_f = torch.linalg.eigh(G)                      # [T,F,C], [T,F,C,C]
order = torch.argsort(evals_f.abs(), -1, descending=True)[..., :K]
lam_k = torch.gather(evals_f, -1, order)                     # [T,F,k]
vec_k = torch.gather(evecs_f, -1, order.unsqueeze(-2).expand(-1, -1, C, -1)).permute(0, 1, 3, 2)  # [T,F,k,C]

# ------------------------------------------------------------------ feature matrix [d_spec, T]
# All from lam_k [T,F,k] / vec_k [T,F,k,C] -- the REAL cached per-(t,f) top-k
# eigenpairs, no frequency reduction. d_spec = F_out*k (λ_1..k per freq) +
# F_out (|u_1|^2 summed over channels -- always 1, dropped) ... kept simple:
# block A = λ_1(f,t) for each of the 8 freq bins; block B = |u_1(f,t)|^2
# averaged over the 8 freq bins, per channel (23 rows); block C = mean |Γ|
# per freq bin (8 rows, same panel-4 quantity, for scale reference).
def _bn(a):                                                 # per-ROW min-max to [0,1]
    return (a - a.min(1, keepdims=True)) / (np.ptp(a, axis=1, keepdims=True) + 1e-9)

feat_lam1_f = _bn(lam_k[:, :, 0].abs().T.numpy())            # [F_out, T]  top eigenvalue, per freq
feat_u1 = _bn((vec_k[:, :, 0, :].abs() ** 2).mean(dim=1).T.numpy())  # [C, T] |u_1|^2, freq-averaged
feat_coh = _bn(coh.mean(0).numpy())                          # [F_out, T]  mean pair coherence per freq
feat = np.vstack([feat_lam1_f, feat_u1, feat_coh])          # [8+23+8, T]
NFEAT_LAM = F_out
row_bounds = [0, NFEAT_LAM, NFEAT_LAM + C, NFEAT_LAM + C + F_out]

# ================================================================ plot
t_raw = np.arange(T_raw) / FS
t_ds = (np.arange(T_ds) + 0.5) * TDS / FS
per_hi, per_lo = FS / freqs[0].item(), FS / freqs[-1].item()
fig = plt.figure(figsize=(12, 24))
gs = GridSpec(7, 1, height_ratios=[1.1, 1.0, 1.3, 1.1, 2.6, 2.0, 1.6], hspace=0.7,
              top=0.965, bottom=0.03, left=0.1, right=0.92)

# 1 · aligned EEG
ax = fig.add_subplot(gs[0])
zr = (raw_win - raw_win.mean(1, keepdims=True)) / (raw_win.std(1, keepdims=True) + 1e-9)
for c in range(C):
    ax.plot(t_raw, zr[c] * 0.9 + c * 2.2, lw=0.25, color="0.55", alpha=0.7)
bi = names.index(BOLD_CH)
ax.plot(t_raw, zr[bi] * 0.9 + bi * 2.2, lw=0.6, color="C3", label=BOLD_CH)
ax.set_xlim(0, WIN_S); ax.set_ylim(-3, C * 2.2 + 2); ax.set_yticks([])
ax.legend(loc="upper left", fontsize=7)
ax.set_title("1 · aligned bipolar EEG  [23 ch]  — 256 Hz, 30-s window, per-channel z-scored (offset)",
             fontsize=9, loc="left")

# 2 · returns / normalized input
ax = fig.add_subplot(gs[1])
im = ax.imshow(zr, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
               extent=[0, WIN_S, C - 0.5, -0.5], interpolation="nearest")
ax.set_ylabel("channel"); ax.set_title(
    "2 · normalized signal  r  [C × T]  (per-channel unit variance; CWT input)", fontsize=9, loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="σ")

# 3 · CWT scalogram (one channel)
ax = fig.add_subplot(gs[2])
power = np.log10(np.abs(coeffs[bi].numpy()) ** 2 + 1e-12)   # [16, T_raw]  high->low freq
im = ax.imshow(power, aspect="auto", cmap="magma", origin="upper",
               extent=[0, WIN_S, NFREQS - 0.5, -0.5], interpolation="nearest")
ax.set_yticks(range(NFREQS))
ax.set_yticklabels([f"{f:.0f} Hz" if i % 2 == 0 else "" for i, f in enumerate(freqs.numpy())], fontsize=6)
ax.set_ylabel("frequency"); ax.set_title(
    f"3 · causal Morlet CWT — {BOLD_CH} scalogram  [F=16 × T]  (log₁₀|W|²; one per channel)",
    fontsize=9, loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="log₁₀|W|²")

# 4 · complex wavelet coherence, mean magnitude over pairs
ax = fig.add_subplot(gs[3])
im = ax.imshow(coh.mean(0).numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1,
               origin="upper", extent=[0, WIN_S, F_out - 0.5, -0.5], interpolation="nearest")
ax.set_yticks(range(F_out))
ax.set_yticklabels([f"{v:.0f} Hz" for v in freqs_out.numpy()], fontsize=6)
ax.set_ylabel("freq bin"); ax.set_title(
    "4 · complex wavelet coherence  Γ_ij(f,t) = |Γ|·e^{iφ}  [P=253 pairs × F=8 × T]  — mean |Γ| shown; phase kept",
    fontsize=9, loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="mean |Γ| over pairs")

# 5 · Hermitian graph snapshots — one per (timestep, frequency): the cache
# never reduces frequency, so this is a genuine grid, not 4 single graphs.
snap_t = np.linspace(0, T_ds - 1, 4).round().astype(int)
gs5 = gs[4].subgridspec(4, F_out, wspace=0.06, hspace=0.25)
Gmag = G.abs().numpy()
vmax5 = np.percentile(Gmag, 99)
for r, ti in enumerate(snap_t):
    for f in range(F_out):
        a = fig.add_subplot(gs5[r, f])
        a.imshow(Gmag[ti, f], cmap="cividis", vmin=0, vmax=vmax5, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title(f"{freqs_out[f]:.0f}Hz", fontsize=6)
        if f == 0:
            a.set_ylabel(f"t={t_ds[ti]:.1f}s", fontsize=6)
fig.text(0.5, gs5.get_grid_positions(fig)[1][0] + 0.012,
         "5 · Hermitian channel graph  |Γ(f,t)|  [C×C]  — one PER (timestep, frequency bin), no reduction "
         "(4 of 480 timesteps × all 8 freq bins shown)",
         ha="center", fontsize=9)

# 6 · eigenvalue tracks — small multiples, one panel per frequency bin
gs6 = gs[5].subgridspec(2, 4, wspace=0.25, hspace=0.55)
for f in range(F_out):
    a = fig.add_subplot(gs6[f // 4, f % 4])
    for m in range(4):
        a.plot(t_ds, lam_k[:, f, m].abs().numpy(), lw=0.9 if m == 0 else 0.5,
               label=f"λ{m+1}" if f == 0 else None)
    a.set_title(f"{freqs_out[f]:.0f} Hz", fontsize=7)
    a.tick_params(labelsize=6)
    if f == 0:
        a.legend(loc="upper right", fontsize=5, ncol=2)
fig.text(0.5, gs6.get_grid_positions(fig)[1][0] + 0.012,
         "6 · eigendecomposition  Γ(f,t) = Σ_m λ_m(f,t) u_m(f,t) u_m(f,t)ᴴ  — top-4 |λ| tracks, PER frequency bin "
         "(k=6 cached; λ₁≫rest ⇒ channels coupled as one at that band)",
         ha="center", fontsize=9)

# 7 · spectral feature matrix
ax = fig.add_subplot(gs[6])
im = ax.imshow(feat, aspect="auto", cmap="magma", vmin=0,
               extent=[0, WIN_S, feat.shape[0] - 0.5, -0.5], interpolation="nearest")
for b in row_bounds[1:-1]:
    ax.axhline(b - 0.5, color="w", lw=0.8)
ax.set_yticks([NFEAT_LAM / 2, NFEAT_LAM + C / 2, NFEAT_LAM + C + F_out / 2])
ax.set_yticklabels(["λ₁(f,t)  (8 freq bins)", "|u₁(f,t)|²  freq-avg  (23 ch)", "mean|Γ(f,t)| / freq"], fontsize=7)
ax.set_xlabel("time (s)")
ax.set_title(f"7 · deterministic spectral features  [{feat.shape[0]} rows × T={T_ds}]  (per-row scaled) — a "
             f"legible slice of the raw (eigvals [T,F,k], eigvecs [T,F,k,C]) the encoder actually consumes whole",
             fontsize=9, loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

fig.suptitle(
    "hermitian_ssm feature pipeline — one 30-s (7680-sample) window, all 23 channels, "
    f"chb01_03 preictal ending {END_S:.0f} s  (seizure onset 2996 s)\n"
    f"nfreqs=16  freq_downsample=2 → F_out=8   time_downsample=16 → {T_ds} steps   k={K}   mains-notch 60/120 Hz   diagonal=zero",
    fontsize=11, y=0.995)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
