"""One 30-s CWT window (one channel) and one WCT window (one channel pair)
from a real chb01 preictal segment, rendered two ways:

  *_legible.png  -- nfreqs=300, raw |coh|/phase: shows the time-frequency
                    structure a human wants to see.
  *_model.png    -- EXACTLY what the "pre" temporal_graph_mamba prediction
                    pipeline ingests: nfreqs=8, 8-40 Hz log grid, native
                    T=7680 CWT; the WCT is the [4, E, T_ds, F] dense-edge
                    stack (coh, sinφ, cosφ, significance), separable
                    Gaussian smooth kernel (5,3) sigma (2,1), COI-masked,
                    time-average-pooled x16 -> T_ds=479, F=8.

Config source: Epilepsy/run_pipelines.py _SHARED_ARCH_PARAMS +
PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS. The model-faithful math replicates
SparseEvidenceGNNCore._full_edge_wct_maps / _smooth_wct_maps /
_coi_valid_mask / _downsample_dense_edge_time.

    .venv/bin/python docs/assets/temporal_graph_mamba/render_cwt_wct_windows.py

Outputs: docs/assets/temporal_graph_mamba/{cwt,wct}_window_{legible,model}.png
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils import torch_cwt
from utils.coherence_utils import coherence

FS = 256
WIN_S = 30.0
LOW, HIGH = 8.0, 40.0
NFREQS_MODEL = 8            # _SHARED_ARCH_PARAMS.nfreqs
TDS = 16                    # dense_edge_time_downsample
SMOOTH_KH, SMOOTH_KW = 5, 3           # smooth_kernel_size (time, freq)
SMOOTH_SH, SMOOTH_SW = 2.0, 1.0       # (kh-1)/2, (kw-1)/2  (sigma=(None,None))
COH_THRESH = 0.90          # coherence_threshold -> significance = (coh-thr)/thr
COI_FB = 2.0               # _COI_WAVELET_FB

EDF = Path.home() / "mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/chb01_03.edf"
START_S = 2900.0           # chb01_03 seizure onset 2996 s -> preictal
CH_CWT = "F7-T7"
CH_PAIR = ("F7-T7", "T7-P7")
OUT = Path(__file__).resolve().parent

import mne
raw = mne.io.read_raw_edf(str(EDF), preload=True, verbose="ERROR")
assert abs(raw.info["sfreq"] - FS) < 1e-6, raw.info["sfreq"]
names = raw.ch_names
data = raw.get_data() * 1e6  # uV
s0, n = int(START_S * FS), int(WIN_S * FS)
seg = data[:, s0:s0 + n]
t = np.arange(n) / FS

xa = seg[names.index(CH_PAIR[0])]
xb = seg[names.index(CH_PAIR[1])]


# ------------------------------------------------------------------ legible
def _scalo(ax, mag, freqs, title, cmap="magma", vmin=0, vmax=None):
    f = freqs[::-1]
    im = ax.imshow(mag[::-1], aspect="auto", origin="lower",
                   extent=[0, WIN_S, f[0], f[-1]], cmap=cmap,
                   vmin=vmin, vmax=vmax if vmax is not None else mag.max())
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(title, fontsize=9, loc="left")
    return im


cx300, f300 = torch_cwt.transform(xa, FS, HIGH, LOW, nfreqs=300)
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                       gridspec_kw={"height_ratios": [1, 2]})
ax[0].plot(t, xa, lw=0.5, color="black"); ax[0].set_xlim(0, WIN_S)
ax[0].set_ylabel("µV"); ax[0].set_title(f"raw signal · {CH_CWT}", fontsize=9, loc="left")
im = _scalo(ax[1], np.abs(cx300), f300, "CWT scalogram |W(f,t)|  (Morlet fb=2, nfreqs=300)")
ax[1].set_xlabel("time (s)")
fig.colorbar(im, ax=ax[1], fraction=0.04, pad=0.02, label="|W|")
fig.suptitle(f"30-s CWT window · chb01_03 preictal (t={START_S:.0f}–{START_S+WIN_S:.0f} s; "
             "seizure 2996 s)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "cwt_window_legible.png", dpi=150, bbox_inches="tight")
print("wrote cwt_window_legible.png")

ca300, _ = torch_cwt.transform(xa, FS, HIGH, LOW, nfreqs=300)
cb300, _ = torch_cwt.transform(xb, FS, HIGH, LOW, nfreqs=300)
coh_l, _, S12_l = coherence(ca300.astype(np.complex128), cb300.astype(np.complex128), f300)
fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
im0 = _scalo(ax[0], coh_l, f300, "wavelet coherence |WCT(f,t)|", vmin=0, vmax=1)
fig.colorbar(im0, ax=ax[0], fraction=0.04, pad=0.02, label="0–1")
im1 = _scalo(ax[1], np.angle(S12_l), f300, "relative phase arg(W₁·conj W₂) (rad)",
             cmap="twilight", vmin=-np.pi, vmax=np.pi)
ax[1].set_xlabel("time (s)")
fig.colorbar(im1, ax=ax[1], fraction=0.04, pad=0.02, label="−π–+π")
fig.suptitle(f"30-s WCT window · {CH_PAIR[0]} ↔ {CH_PAIR[1]} · chb01_03 preictal", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "wct_window_legible.png", dpi=150, bbox_inches="tight")
print("wrote wct_window_legible.png")


# ------------------------------------------------------------------ model
# CWT: nfreqs=8, native T, complex coeffs. freqs come back high->low.
ca8, f8 = torch_cwt.transform(xa, FS, HIGH, LOW, nfreqs=NFREQS_MODEL)
cb8, _ = torch_cwt.transform(xb, FS, HIGH, LOW, nfreqs=NFREQS_MODEL)
freqs = torch.tensor(f8, dtype=torch.float32)                      # [F]
ar = torch.tensor(ca8.real, dtype=torch.float32).T                 # [T, F]
ai = torch.tensor(ca8.imag, dtype=torch.float32).T
br = torch.tensor(cb8.real, dtype=torch.float32).T
bi = torch.tensor(cb8.imag, dtype=torch.float32).T
T0 = ar.shape[0]

# --- CWT window the model sees (|W|, nfreqs=8) ---
fig, axm = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True,
                        gridspec_kw={"height_ratios": [1, 2]})
axm[0].plot(t, xa, lw=0.5, color="black"); axm[0].set_xlim(0, WIN_S)
axm[0].set_ylabel("µV"); axm[0].set_title(f"raw signal · {CH_CWT}", fontsize=9, loc="left")
mag8 = np.abs(ca8)                                                 # [F, T], high->low
im = axm[1].imshow(mag8[::-1], aspect="auto", origin="lower", interpolation="nearest",
                   extent=[0, WIN_S, -0.5, NFREQS_MODEL - 0.5], cmap="magma", vmin=0)
axm[1].set_yticks(range(NFREQS_MODEL)); axm[1].set_yticklabels([f"{v:.1f}" for v in f8[::-1]])
axm[1].set_ylabel("frequency (Hz)"); axm[1].set_xlabel("time (s)")
axm[1].set_title("CWT scalogram |W(f,t)|  —  nfreqs=8, native T=7680 (what the model ingests)",
                 fontsize=9, loc="left")
fig.colorbar(im, ax=axm[1], fraction=0.04, pad=0.02, label="|W|")
fig.suptitle(f"30-s CWT window (model resolution) · {CH_CWT} · chb01_03 preictal", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "cwt_window_model.png", dpi=150, bbox_inches="tight")
print("wrote cwt_window_model.png")

# --- dense-edge WCT stack (replicates SparseEvidenceGNNCore) ---
# _full_edge_wct_maps: cross-spectrum, then * inv_scale (= freqs) per bin.
xwt_real = (ar * br + ai * bi) * freqs
xwt_imag = (ai * br - ar * bi) * freqs
auto1 = (ar * ar + ai * ai) * freqs
auto2 = (br * br + bi * bi) * freqs

# _smooth_wct_maps: separable Gaussian. time = VALID (pad_h=0 -> shrink kh-1),
# freq = "same" (reflect pad kw-1, split). sigma=((kh-1)/2, (kw-1)/2).
def _g1d(k, s):
    x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-0.5 * (x / s) ** 2)
    return g / g.sum()

kt = _g1d(SMOOTH_KH, SMOOTH_SH).view(1, 1, SMOOTH_KH, 1)
kf = _g1d(SMOOTH_KW, SMOOTH_SW).view(1, 1, 1, SMOOTH_KW)
pf = SMOOTH_KW - 1
pf_l, pf_r = pf // 2, pf - pf // 2

def _smooth(m):                       # m: [T, F] -> [T-(kh-1), F]
    m = m[None, None]
    m = F.conv2d(m, kt, stride=1)                       # VALID in time
    m = F.pad(m, (pf_l, pf_r, 0, 0), mode="reflect")    # SAME in freq
    m = F.conv2d(m, kf, stride=1)
    return m[0, 0]

sr, si = _smooth(xwt_real), _smooth(xwt_imag)
sa1, sa2 = _smooth(auto1), _smooth(auto2)
coh = ((sr * sr + si * si) / (sa1 * sa2 + 1e-12)).clamp(0, 1)      # [T1, F]
phase = torch.atan2(si, sr)
T1 = coh.shape[0]
time_offset = (SMOOTH_KH - 1) // 2

# _coi_valid_mask
scale = FS / freqs                                                 # [F]
support = torch.floor(COI_FB * scale * 3.0)                        # [F]
t_idx = torch.arange(T1).view(T1, 1) + time_offset
coi_valid = ((t_idx >= support) & (t_idx < (T0 - support))).float()
coh = coh * coi_valid
phase = phase * coi_valid
sig = ((coh - COH_THRESH) / COH_THRESH) * coi_valid

stack = torch.stack([coh, torch.sin(phase), torch.cos(phase), sig], 0)  # [4, T1, F]

# _downsample_dense_edge_time: avg pool T by TDS (floor)
stack = F.avg_pool2d(stack.permute(0, 2, 1)[None], kernel_size=(1, TDS),
                     stride=(1, TDS))[0].permute(0, 2, 1)          # [4, T_ds, F]
Tds = stack.shape[1]
S = stack.numpy()

titles = ["ch0  coherence  |coh|", "ch1  sin φ", "ch2  cos φ",
          f"ch3  significance  (coh−{COH_THRESH})/{COH_THRESH}"]
cmaps = ["magma", "twilight", "twilight", "coolwarm"]
lims = [(0, 1), (-1, 1), (-1, 1), (-1.1, 0.15)]
fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for k, (a, ttl, cm, (vmn, vmx)) in enumerate(zip(axs, titles, cmaps, lims)):
    im = a.imshow(S[k].T[::-1], aspect="auto", origin="lower", interpolation="nearest",
                  extent=[0, WIN_S, -0.5, NFREQS_MODEL - 0.5], cmap=cm, vmin=vmn, vmax=vmx)
    a.set_yticks(range(NFREQS_MODEL)); a.set_yticklabels([f"{v:.0f}" for v in f8[::-1]])
    a.set_ylabel("Hz"); a.set_title(ttl, fontsize=9, loc="left")
    fig.colorbar(im, ax=a, fraction=0.03, pad=0.02)
axs[-1].set_xlabel("time (s)")
fig.suptitle(f"30-s WCT window (model input) · {CH_PAIR[0]} ↔ {CH_PAIR[1]} · chb01_03 preictal\n"
             f"dense-edge stack [4, T={Tds}, F={NFREQS_MODEL}]  —  one edge of temporal_edge_proj's input",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "wct_window_model.png", dpi=150, bbox_inches="tight")
print("wrote wct_window_model.png")
