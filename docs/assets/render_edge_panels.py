"""Render the README's per-edge coherence/phase figure from the real
export (exports/coherence-graph/raw/coherence.npz, produced by
exports/coherence-graph/build_coherence_export.py on a real chb01 preictal
window). Output: docs/assets/coherence-edge.png.

One edge of the wavelet-coherence graph: |coherence| and phase across the
4 s window, 8-40 Hz. No thresholding -- both panels are the raw computed
arrays.

    .venv/bin/python docs/assets/render_edge_panels.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "exports" / "coherence-graph" / "raw" / "coherence.npz"
OUT = ROOT / "docs" / "assets" / "coherence-edge.png"

EDGE = ("F8-T8", "FT10-T8")  # a neighbouring-channel pair -- strong, structured

d = np.load(NPZ, allow_pickle=True)
src, dst = d["src_channel"], d["dst_channel"]
idx = next(i for i in range(len(src))
           if {src[i], dst[i]} == set(EDGE))

# freqs_hz is stored high -> low; flip everything to ascending so the
# y-axis reads low frequency at the bottom.
flip = slice(None, None, -1)
coh = d["coh"][idx].T[flip]      # (F, T)
phase = d["phase"][idx].T[flip]  # (F, T)
coi = d["coi_valid"].T[flip]     # (F, T)  True = valid
freqs = d["freqs_hz"][flip]
fs = int(d["sampling_rate"])
t_max = coh.shape[1] / fs
extent = [0, t_max, float(freqs[0]), float(freqs[-1])]

fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)

panels = [
    ("coherence  |coh|", coh, "magma", dict(vmin=0, vmax=1), "0 – 1"),
    ("phase  (rad)", phase, "twilight", dict(vmin=-np.pi, vmax=np.pi), "-π – +π"),
]
for ax, (label, arr, cmap, kw, cblabel) in zip(axes, panels):
    im = ax.imshow(arr, aspect="auto", origin="lower", extent=extent, cmap=cmap, **kw)
    # Hatch the cone-of-influence (edge-effect) region.
    ax.contourf(np.linspace(0, t_max, coi.shape[1]), freqs, (~coi).astype(float),
                levels=[0.5, 1.5], colors="none", hatches=["xxxx"])
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(label, fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cblabel, fontsize=7)
    cb.ax.tick_params(labelsize=6)

axes[-1].set_xlabel("time (s)")
fig.suptitle(
    f"Wavelet coherence, one edge  ·  {EDGE[0]} ↔ {EDGE[1]}\n"
    "chb01 preictal window (seizure 1_16_0); hatched = cone of influence",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
