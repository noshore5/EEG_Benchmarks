"""Render the top-level README's coherence-graph figure from the real
export in exports/coherence-graph/ (positions.json + graph.json, produced
by exports/coherence-graph/build_coherence_export.py on a real chb01
preictal window). Output: docs/assets/coherence-graph.png.

    .venv/bin/python docs/assets/render_coherence_graph.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "exports" / "coherence-graph"
OUT = ROOT / "docs" / "assets" / "coherence-graph.png"

TOP_N_EDGES = 45  # strongest coherence pairs to draw (252/253 are "significant")

records = json.loads((EXPORT / "positions.json").read_text())
pos = {d["channel"]: np.array([d["x"], d["y"]]) for d in records}
edges = json.loads((EXPORT / "graph.json").read_text())

bipolar = [d["channel"] for d in records if "-" in d["channel"]]

# Spread the 23 bipolar-channel nodes onto a clean circle, ordered by the
# angle of their real electrode-pair midpoint -- keeps the anatomical
# left/right/front/back layout without the label pile-ups you get when
# adjacent pairs' midpoints nearly coincide.
ctr = np.mean([pos[c] for c in bipolar], axis=0)
ang = {c: np.arctan2(*(pos[c] - ctr)[::-1]) for c in bipolar}
order = sorted(bipolar, key=lambda c: -ang[c])
R = 1.0
node = {c: np.array([R * np.sin(2 * np.pi * i / len(order)),
                     R * np.cos(2 * np.pi * i / len(order))])
        for i, c in enumerate(order)}

sig = sorted((e for e in edges if e["significant"]),
             key=lambda e: e["mean_coherence"], reverse=True)[:TOP_N_EDGES]
cmin = min(e["mean_coherence"] for e in sig)
cmax = max(e["mean_coherence"] for e in sig)

fig, ax = plt.subplots(figsize=(6.6, 6.6))
ax.set_aspect("equal")
ax.axis("off")

cmap = plt.get_cmap("plasma")
for e in sig:
    p1, p2 = node[e["src"]], node[e["dst"]]
    t = (e["mean_coherence"] - cmin) / (cmax - cmin + 1e-9)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=cmap(0.25 + 0.7 * t),
            lw=0.7 + 3.0 * t, alpha=0.4 + 0.5 * t, zorder=2,
            solid_capstyle="round")

for c in order:
    p = node[c]
    ax.scatter(*p, s=120, color="white", edgecolor="0.2", lw=1.3, zorder=4)
    lp = p * 1.16
    ax.annotate(c, lp, fontsize=6.5, ha="center", va="center", zorder=5,
                color="0.15")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin, cmax))
cb = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.04, shrink=0.62)
cb.set_label("mean wavelet coherence, 8–40 Hz", fontsize=8)
cb.ax.tick_params(labelsize=7)

ax.set_title(
    f"Wavelet-coherence graph — chb01 preictal window\n"
    f"top {TOP_N_EDGES} of 253 channel-pair edges",
    fontsize=10,
)
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
