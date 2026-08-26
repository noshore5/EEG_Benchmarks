"""Memory/timing probe for `Epilepsy.pipelines.continuous_dense_edge` at
real dense-edge scale (23-channel full mesh, E=253, nfreqs=48 -- this
pipeline's real defaults), sibling to `continuous_mamba_gpu_scale_probe.py`
(which measured the SSM stage; this measures the CWT/dense-edge stage the
same paradigm's own roadmap flagged as unmeasured -- CONTEXT.md: "the one
real unknown... CWT-over-a-continuous-hour-long-recording's own cost has
never been measured").

Sweeps T_chunk (output steps) at real scale, one recording's worth of
synthetic raw EEG (~1hr at 256Hz), reporting wall time and peak CUDA memory
per chunk. CUDA if visible, else CPU (MPS untested here -- fine, this is a
probe, not a correctness check).

Run: python scripts/continuous_cwt_scale_probe.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Epilepsy.pipelines.continuous_dense_edge import (  # noqa: E402
    continuous_chunk_pad_samples,
    iter_continuous_dense_edge_chunks,
)
from Epilepsy.pipelines.cwt_gnn_classifiers import SparseEvidenceGNNClassifier  # noqa: E402

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")
    device_name = "cpu"

N_CHANNELS = 23  # real full mesh
SAMPLING_RATE = 256
N_TOTAL = SAMPLING_RATE * 3600  # ~1hr recording, real scale
NFREQS = 48
LOWEST, HIGHEST = 1.0, 40.0
DOWNSAMPLE = 16  # this pipeline's real dense_edge_time_downsample default


def _reset_mem():
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else float("nan")


def _make_classifier() -> SparseEvidenceGNNClassifier:
    clf = SparseEvidenceGNNClassifier(
        sampling_rate=SAMPLING_RATE, lowest=LOWEST, highest=HIGHEST, nfreqs=NFREQS,
        event_mode="dense", dense_edge_temporal_mode="conv",
        coherence_threshold_mode="fixed", coherence_threshold=0.5,
        dense_edge_time_downsample=DOWNSAMPLE, time_averaged_graph=False,
        cwt_backend="torch", device=str(device),
    )
    clf.device_ = device
    clf._resolve_transform_fns()
    return clf


print(f"device: {device_name}")
print(f"N_CHANNELS={N_CHANNELS} SAMPLING_RATE={SAMPLING_RATE} N_TOTAL={N_TOTAL} "
      f"({N_TOTAL / SAMPLING_RATE / 60:.0f} min) nfreqs={NFREQS} downsample={DOWNSAMPLE}\n")

clf = _make_classifier()
core = clf._build_model(n_channels=N_CHANNELS, n_classes=2)
core.eval().to(device)
pad = continuous_chunk_pad_samples(clf, core)
print(f"pad (raw samples, each side) = {pad}\n")

raw_x = (np.random.randn(N_CHANNELS, N_TOTAL).astype(np.float32) * 20.0)

print(f"{'T_chunk':>8} {'n_chunks':>9} {'total_s':>9} {'ms/chunk':>10} {'peak_MiB':>10}")
for t_chunk in (64, 128, 256, 512, 1024):
    _reset_mem()
    n_chunks = 0
    t0 = time.perf_counter()
    oom = False
    try:
        with torch.no_grad():
            for _start, _end, conv_in in iter_continuous_dense_edge_chunks(clf, core, raw_x, chunk_size=t_chunk):
                n_chunks += 1
                if n_chunks >= 20:  # enough chunks for a stable per-chunk estimate, not the whole hour
                    break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # pragma: no cover -- hardware-dependent
        msg = str(exc).splitlines()[0]
        if "out of memory" in msg.lower():
            oom = True
        else:
            raise
    dt = time.perf_counter() - t0
    if oom:
        print(f"{t_chunk:8d}  OOM")
        _reset_mem()
        continue
    ms_per_chunk = 1000.0 * dt / max(1, n_chunks)
    print(f"{t_chunk:8d} {n_chunks:9d} {dt:9.3f} {ms_per_chunk:10.3f} {_peak_mib():10.1f}")

print("\nExtrapolation: a real ~1hr recording at dense_edge_time_downsample="
      f"{DOWNSAMPLE} has ~{N_TOTAL // DOWNSAMPLE} output steps total; multiply "
      "ms/chunk above by (that / T_chunk) for a rough whole-recording CWT-stage estimate "
      "(add to _DenseEdgeMambaContinuous's own separately-measured ~54s/recording "
      "Mamba-stage cost, continuous_mamba_chunk_scan_throughput.md, for the total).")
