"""Full preprocessing sequence for temporal_graph_mamba "pre" (the AP≈0.67
prediction leader), same 7-panel structure as
docs/assets/hermitian_ssm/render_hermitian_ssm_window.py, for one real
chb01 preictal 30-s window.

Config = Epilepsy/run_pipelines.py _SHARED_ARCH_PARAMS +
PREDICTION_TEMPORAL_GRAPH_MAMBA_PARAMS: 256 Hz, 8-40 Hz, nfreqs=8, native
T=7680, smooth_kernel_size=(5,3), COI-masked, dense_edge_time_downsample=16
-> T_ds~479. Math reference: Epilepsy/temporal_graph_mamba_math.md.

Panels 1-5 replicate SparseEvidenceGNNCore's deterministic, non-trainable
preprocessing exactly (_full_edge_wct_maps / _smooth_wct_maps /
_coi_valid_mask / _downsample_dense_edge_time) -- real numbers, no trained
weights needed. Panel 6 draws the FIXED (data-independent) directed
incidence structure step 5 of the math doc uses to collapse 253 edges to
23 nodes. Panel 7 is deliberately NOT a data panel: steps 3/4/6/7 (edge
encoder, message MLP, per-channel Mamba SSM, readout) are LEARNED, and this
pipeline never persists a trained checkpoint to disk (best-val state lives
in-memory only during a training run) -- there is nothing on disk to load
real weights from, and rendering untrained/random weights would misrepresent
"what the model sees" rather than show it, so panel 7 states that boundary
instead of faking numbers.

    .venv/bin/python docs/assets/temporal_graph_mamba/render_temporal_graph_mamba_window.py

Output: docs/assets/temporal_graph_mamba/temporal_graph_mamba_window.png
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
LOW, HIGH, NFREQS = 8.0, 40.0, 8      # _SHARED_ARCH_PARAMS
SMOOTH_KH, SMOOTH_KW = 5, 3           # smooth_kernel_size (time, freq)
SMOOTH_SH, SMOOTH_SW = 2.0, 1.0       # sigma = ((kh-1)/2, (kw-1)/2)
COH_THRESH = 0.90
COI_FB = 2.0
TDS = 16                              # dense_edge_time_downsample
EDF = Path.home() / "mne_data/MNE-chbmit-data/chbmit/1.0.0/chb01/chb01_03.edf"
END_S = 2600.0                        # same window as the hermitian_ssm figure (onset 2996s; SPH300/SOP900)
BOLD_CH = "FP1-F7"
CH_PAIR = ("F7-T7", "T7-P7")
OUT = Path(__file__).resolve().parent / "temporal_graph_mamba_window.png"

# ------------------------------------------------------------------ load
import mne

raw = mne.io.read_raw_edf(str(EDF), preload=True, verbose="ERROR")
assert abs(raw.info["sfreq"] - FS) < 1e-6
names = raw.ch_names
C = len(names)
data = raw.get_data() * 1e6
s0 = int((END_S - WIN_S) * FS)
n = int(WIN_S * FS)
seg = data[:, s0:s0 + n]                                    # [C, 7680] uV
t = np.arange(n) / FS

zr = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-9)   # normalize_input=True
sig = torch.tensor(zr, dtype=torch.float32)

# ------------------------------------------------------------------ CWT, nfreqs=8, native T
coeffs, freqs = cwt_torch(sig, FS, LOW, HIGH, NFREQS)        # [C, 8, 7680] complex, freqs high->low
T0 = coeffs.shape[-1]
ar, ai = coeffs.real, coeffs.imag                            # [C, F, T]

# ------------------------------------------------------------------ Γ + edge stack (_build_dense_edge_input math)
iu, ju = torch.triu_indices(C, C, offset=1)                  # 253 edges
src_r, dst_r = ar[iu], ar[ju]
src_i, dst_i = ai[iu], ai[ju]
xwt_real = (src_r * dst_r + src_i * dst_i)
xwt_imag = (src_i * dst_r - src_r * dst_i)
auto1 = (src_r ** 2 + src_i ** 2)
auto2 = (dst_r ** 2 + dst_i ** 2)
inv_scale = freqs.view(1, NFREQS, 1)
xwt_real, xwt_imag, auto1, auto2 = (z * inv_scale for z in (xwt_real, xwt_imag, auto1, auto2))
# -> each [E=253, F=8, T=7680], matches _full_edge_wct_maps

def _g1d(k, s):
    x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-0.5 * (x / s) ** 2)
    return g / g.sum()

kt = _g1d(SMOOTH_KH, SMOOTH_SH).view(1, 1, SMOOTH_KH, 1)
kf = _g1d(SMOOTH_KW, SMOOTH_SW).view(1, 1, 1, SMOOTH_KW)
pf_l, pf_r = (SMOOTH_KW - 1) // 2, (SMOOTH_KW - 1) - (SMOOTH_KW - 1) // 2

def _smooth(m):                       # m: [E, F, T] -> permute to [E,1,T,F] conv, VALID time / reflect freq
    m = m.permute(0, 2, 1).unsqueeze(1)               # [E,1,T,F]
    m = F.conv2d(m, kt, stride=1)                     # VALID time: T -> T-4
    m = F.pad(m, (pf_l, pf_r, 0, 0), mode="reflect")
    m = F.conv2d(m, kf, stride=1)                     # SAME freq
    return m[:, 0].permute(0, 2, 1)                   # [E, F, T1]

sr, si = _smooth(xwt_real), _smooth(xwt_imag)
sa1, sa2 = _smooth(auto1), _smooth(auto2)
coh = ((sr ** 2 + si ** 2) / (sa1 * sa2 + 1e-12)).clamp(0, 1)   # [E, F, T1]
phase = torch.atan2(si, sr)
T1 = coh.shape[-1]
time_offset = (SMOOTH_KH - 1) // 2

scale = FS / freqs
support = torch.floor(COI_FB * scale * 3.0)           # [F]
t_idx = torch.arange(T1).view(T1, 1) + time_offset    # [T1, 1]
coi_valid = ((t_idx >= support.view(1, NFREQS)) & (t_idx < (T0 - support.view(1, NFREQS)))).float()  # [T1, F]
coi_valid_ft = coi_valid.T.unsqueeze(0)               # [1, F, T1] -> broadcasts over edges
coh = coh * coi_valid_ft
phase = phase * coi_valid_ft
sig_ch3 = ((coh - COH_THRESH) / COH_THRESH) * coi_valid_ft

stack = torch.stack([coh, torch.sin(phase), torch.cos(phase), sig_ch3], 1)  # [E, 4, F, T1]
pooled = F.avg_pool2d(stack.reshape(-1, NFREQS, T1).unsqueeze(0), kernel_size=(1, TDS), stride=(1, TDS))
pooled = pooled.squeeze(0).reshape(stack.shape[0], 4, NFREQS, -1)   # [E, 4, F, T_ds]
T_ds = pooled.shape[-1]

edge_pair_idx = {(int(iu[e]), int(ju[e])): e for e in range(iu.numel())}
i0, j0 = sorted((names.index(CH_PAIR[0]), names.index(CH_PAIR[1])))
e_show = edge_pair_idx[(i0, j0)]

# ------------------------------------------------------------------ incidence structure (step 5, math doc)
# B[c,e] = 1 iff c == max(endpoints of e); D = diag(in-degree), D[0] clamped to 1.
B = torch.zeros(C, iu.numel())
B[ju, torch.arange(iu.numel())] = 1.0                  # dst = higher-indexed endpoint
indeg = B.sum(1).clamp(min=1.0)

# ================================================================ plot
t_raw = np.arange(T0) / FS
t_ds = (np.arange(T_ds) + 0.5) * TDS / FS
fig = plt.figure(figsize=(12, 23))
gs = GridSpec(7, 1, height_ratios=[1.1, 1.0, 1.3, 1.1, 1.3, 1.1, 1.6], hspace=0.65,
              top=0.96, bottom=0.03, left=0.09, right=0.93)

# 1 · aligned EEG
ax = fig.add_subplot(gs[0])
for c in range(C):
    ax.plot(t_raw, zr[c] * 0.9 + c * 2.2, lw=0.25, color="0.55", alpha=0.7)
bi = names.index(BOLD_CH)
ax.plot(t_raw, zr[bi] * 0.9 + bi * 2.2, lw=0.6, color="C3", label=BOLD_CH)
ax.set_xlim(0, WIN_S); ax.set_ylim(-3, C * 2.2 + 2); ax.set_yticks([])
ax.legend(loc="upper left", fontsize=7)
ax.set_title("1 · aligned bipolar EEG  [23 ch]  — 256 Hz, 30-s window", fontsize=9, loc="left")

# 2 · normalized input
ax = fig.add_subplot(gs[1])
im = ax.imshow(zr, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
               extent=[0, WIN_S, C - 0.5, -0.5], interpolation="nearest")
ax.set_ylabel("channel")
ax.set_title("2 · normalized signal  r  [C × T]  (normalize_input=True; CWT input)", fontsize=9, loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="σ")

# 3 · CWT scalogram, nfreqs=8, native T
ax = fig.add_subplot(gs[2])
mag = np.abs(coeffs[bi].numpy())
im = ax.imshow(mag, aspect="auto", origin="upper", extent=[0, WIN_S, NFREQS - 0.5, -0.5],
               cmap="magma", vmin=0, interpolation="nearest")
ax.set_yticks(range(NFREQS)); ax.set_yticklabels([f"{v:.1f}" for v in freqs.numpy()], fontsize=7)
ax.set_ylabel("frequency (Hz)")
ax.set_title(f"3 · causal Morlet CWT — {BOLD_CH} scalogram  |W(f,t)|  [F=8 × T=7680]  (native resolution the model ingests)",
             fontsize=9, loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="|W|")

# 4 · coherence, mean over all 253 pairs, post-smooth/COI, pre-downsample
ax = fig.add_subplot(gs[3])
mean_coh = coh.mean(0).numpy()             # [F, T1]
im = ax.imshow(mean_coh, aspect="auto", origin="upper", extent=[0, WIN_S, NFREQS - 0.5, -0.5],
               cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
ax.set_yticks(range(NFREQS)); ax.set_yticklabels([f"{v:.1f}" for v in freqs.numpy()], fontsize=7)
ax.set_ylabel("frequency (Hz)")
ax.set_title("4 · complex wavelet coherence  Γ_ij(f,t)  [P=253 pairs × F=8 × T]  — mean |Γ| shown, "
             "smoothed (5,3)/σ(2,1) + COI-masked (hatch-free zeros at edges)", fontsize=9, loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="mean |Γ| over pairs")

# 5 · one edge's dense-edge feature stack (post ×16 downsample) — the literal model input
titles5 = ["ch0  |coh|", "ch1  sinφ", "ch2  cosφ", f"ch3  significance"]
cmaps5 = ["magma", "twilight", "twilight", "coolwarm"]
lims5 = [(0, 1), (-1, 1), (-1, 1), (-1.1, 0.15)]
gs5 = gs[4].subgridspec(1, 4, wspace=0.3)
edge_stack = pooled[e_show].numpy()        # [4, F, T_ds]
for k in range(4):
    a = fig.add_subplot(gs5[k])
    im = a.imshow(edge_stack[k], aspect="auto", origin="upper", extent=[0, WIN_S, NFREQS - 0.5, -0.5],
                  cmap=cmaps5[k], vmin=lims5[k][0], vmax=lims5[k][1], interpolation="nearest")
    a.set_yticks(range(NFREQS)); a.set_yticklabels([f"{v:.0f}" for v in freqs.numpy()] if k == 0 else [], fontsize=6)
    a.set_title(titles5[k], fontsize=7)
    a.set_xlabel("s", fontsize=6)
fig.text(0.5, gs5.get_grid_positions(fig)[1][0] + 0.012,
         f"5 · dense-edge feature  x_e(f,t)  for edge {CH_PAIR[0]}↔{CH_PAIR[1]}  [4 × F=8 × T_ds={T_ds}]  "
         "— literal per-edge Linear(32,8) input, ×16 time-pooled", ha="center", fontsize=9)

# 6 · incidence structure (fixed, data-independent)
gs6 = gs[5].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
a1 = fig.add_subplot(gs6[0])
a1.imshow(B.numpy(), aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
a1.set_xlabel("edge e = (i,j), i<j  [253]"); a1.set_ylabel("channel c  [23]")
a1.set_title("directed incidence  B[c,e]=1 iff c=max(i,j)", fontsize=8, loc="left")
a2 = fig.add_subplot(gs6[1])
a2.barh(range(C), indeg.numpy(), color="0.4")
a2.set_ylim(C - 0.5, -0.5); a2.set_yticks([]); a2.set_xlabel("in-degree")
a2.set_title("D = diag(1,1,2,…,22)", fontsize=8, loc="left")
fig.text(0.5, gs6.get_grid_positions(fig)[1][0] + 0.012,
         "6 · edges → nodes  H(t) = D⁻¹ B Z(t)  — FIXED directed incidence (channel 0 has in-degree 0 ⇒ always the zero vector); "
         "structural, independent of this window's data", ha="center", fontsize=9)

# 7 · explicit boundary: everything past here is learned, no checkpoint on disk
ax = fig.add_subplot(gs[6]); ax.axis("off")
ax.text(0.5, 0.95,
        "7 · past this point: LEARNED, not shown with real numbers",
        ha="center", va="top", fontsize=10, fontweight="bold", transform=ax.transAxes)
ax.text(0.5, 0.78,
        r"$Z_e(t) = M(\phi(A\,x_e(t)))$   —   edge-spectrum embedding,  $A{:}\ \mathbb{R}^{32}{\to}\mathbb{R}^{8}$,  "
        r"$M$ = Linear-GELU-Linear  (steps 3-4)"
        "\n"
        r"$H(t) = D^{-1}B\,Z(t)$   —   incidence average, $\mathbb{R}^{8}$ per node  (step 5, structure shown in panel 6)"
        "\n"
        r"$\mathcal{E}_c = \mathrm{SSM}_\theta(H_c(1{:}T))$   —   per-channel selective Mamba, shared weights  (step 6)"
        "\n"
        r"$\mathrm{logits} = W\,\mathrm{vec}(\mathcal{E}) + b$   —   flatten, one Linear(184,2)  (step 7)",
        ha="center", va="top", fontsize=9.5, transform=ax.transAxes, linespacing=2.2)
ax.text(0.5, 0.28,
        "These 4 steps are trained parameters (temporal_edge_proj, sparse_message_mlp,\n"
        "temporal_node_mamba, sparse_classifier). This pipeline restores its best-val weights\n"
        "in-memory during training and never persists a checkpoint to disk (no torch.save() call\n"
        "in cwt_gnn_classifiers.py) — there is no trained state file in this repo to load and run a\n"
        "real forward pass from. Rendering these steps with untrained/random weights would show\n"
        "numbers that don't correspond to any trained model, not \"what the model sees\" — so this\n"
        "panel states the math instead of fabricating a plot.",
        ha="center", va="top", fontsize=8.5, transform=ax.transAxes, color="0.25", linespacing=1.6)

fig.suptitle(
    "temporal_graph_mamba \"pre\" feature pipeline — one 30-s (7680-sample) window, all 23 channels, "
    f"chb01_03 preictal ending {END_S:.0f} s  (seizure onset 2996 s)\n"
    f"nfreqs=8   smooth_kernel=(5,3) σ=(2,1)   COI-masked   dense_edge_time_downsample=16 → T_ds={T_ds}   "
    "6-fold mean AP ≈ 0.67 (leader)",
    fontsize=11, y=0.995)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
