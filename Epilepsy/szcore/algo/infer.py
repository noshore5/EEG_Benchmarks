"""SzCORE inference entrypoint: EDF in -> seizure-annotation TSV out.

Contract (see https://epilepsybenchmarks.com/challenge-description/):
  * input EDF: 256 Hz, 19-channel unipolar common-average montage
  * output TSV columns: onset duration eventType confidence channels
    dateTime recordingDuration  (written by epilepsy2bids.Annotations)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from algo.common import (
    FS,
    INFER_STEP_S,
    WINDOW_S,
    load_bipolar_eeg,
    make_windows,
    probs_to_mask,
)
from algo.model import TMCTransformer

_WEIGHTS = Path(__file__).with_name("model_weights.pt")
_BATCH = 256


def _load_model(device: torch.device):
    ckpt = torch.load(_WEIGHTS, map_location="cpu", weights_only=False)
    model = TMCTransformer(
        n_channels=ckpt["n_channels"],
        n_time=ckpt["n_time"],
        n_classes=2,
        **ckpt["model_kwargs"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt


@torch.no_grad()
def _predict_window_probs(model, windows: np.ndarray, mean: float, std: float, device) -> np.ndarray:
    if windows.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    x = (windows - mean) / (std + 1e-8)
    out = []
    for i in range(0, x.shape[0], _BATCH):
        batch = torch.from_numpy(x[i : i + _BATCH]).float().to(device)
        logits = model(batch)
        out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def run(edf_path: str, out_path: str) -> None:
    from epilepsy2bids.annotations import Annotations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(device)

    data, fs = load_bipolar_eeg(edf_path)
    window = int(round(WINDOW_S * fs))
    step = int(round(INFER_STEP_S * fs))
    assert window == ckpt["n_time"], (window, ckpt["n_time"])

    windows, starts = make_windows(data, window, step)
    probs = _predict_window_probs(model, windows, ckpt["x_mean"], ckpt["x_std"], device)
    mask = probs_to_mask(
        probs, starts, data.shape[1], fs, float(ckpt["threshold"]), step
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Annotations.loadMask(mask, fs).saveTsv(out_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=os.environ.get("INPUT"))
    parser.add_argument("output", nargs="?", default=os.environ.get("OUTPUT"))
    args = parser.parse_args()
    if not args.input or not args.output:
        parser.error("input and output paths are required (positional or $INPUT/$OUTPUT)")
    run(args.input, args.output)


if __name__ == "__main__":
    main()
