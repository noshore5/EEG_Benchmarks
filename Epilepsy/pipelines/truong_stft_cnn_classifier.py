"""STFT + CNN preictal/interictal classifier, replicating the architecture in:

    Truong, N. D., Nguyen, A. D., Kuhlmann, L., Bonyadi, M. R., Yang, J.,
    Ippolito, S., & Kavehei, O. (2018). "Convolutional neural networks for
    seizure prediction using intracranial and scalp electroencephalogram."
    Neural Networks, 105, 104-111. (arXiv preprint: 1707.01976, "A
    Generalised Seizure Prediction with Convolutional Neural Networks for
    Intracranial and Scalp Electroencephalogram Data Analysis")

Read directly from the arXiv preprint (Section II.B-D, Fig. 3) on
2026-08-18. What's replicated exactly, and what's a deliberate, documented
deviation:

REPLICATED (architecture + preprocessing the paper actually specifies):
  - STFT preprocessing: 1s sub-window, 0.5s hop (0.5s overlap) over each
    (paper: 30s) EEG window, per channel. Power-line noise removed by
    excluding the STFT frequency bins within +/-3Hz of the power-line
    frequency and its 2nd harmonic (57-63Hz/117-123Hz for 60Hz mains, as
    used for CHB-MIT); the DC (0Hz) bin is also dropped. At CHB-MIT's
    256Hz sampling rate this yields exactly the paper's reported 59
    time-bins x 114 freq-bins per channel (see compute_stft_features's
    docstring for the arithmetic).
  - Per-frequency standardization ("a standardization step ... applied on
    STFT components across the whole frequency range to prevent high
    frequencies features being influenced by those at lower frequencies"),
    fit on the training split only.
  - CNN: three conv blocks, each (batch norm on the block's input) -> (conv
    + ReLU) -> (max pool), exactly as Fig. 3 / Section II.C describes:
    C1 is a 3-D conv with 16 kernels of size n_channels x 5 x 5 (n_channels
    = ALL EEG channels folded into one conv step -- this is what turns a
    per-channel spectrogram stack into a single-channel 2-D feature map),
    stride 1x2x2, pool 1x2x2. C2/C3: 32 and 64 2-D kernels, size 3x3,
    stride 1x1, pool 2x2. Flatten -> dropout(0.5) -> FC(256, sigmoid) ->
    dropout(0.5) -> FC(2, softmax).
  - k-of-n alarm smoothing post-processing (Section II.D): k=8 of the last
    n=10 window predictions, applied to one recording's predictions in
    chronological order (k_of_n_alarm below).
  - SPH/SOP prediction-horizon semantics: this repo's own
    ContinuousLabelingParadigm (label_mode="prediction") already implements
    the identical SPH/SOP window-labeling rule the paper defines in its
    Fig. 5 -- see that module's docstring. The paper's own defaults
    (SPH=5min, SOP=30min) are exposed as this repo's DEFAULT_SPH/DEFAULT_SOP
    in run_pipelines.py (DEFAULT_SOP has since been narrowed to 15min for
    CHB-MIT recording-length reasons documented there, not because the
    30min paper value is architecturally wrong).

DELIBERATE DEVIATIONS (things the paper does that this file does NOT do,
because they're training-procedure/dataset-scale choices, not "the
architecture", and reproducing them exactly would either duplicate work
this repo already does differently on purpose or isn't verifiable without
the paper's own code):
  - Class imbalance: the paper generates extra overlapping preictal windows
    per-subject (Fig. 2) to roughly balance classes before training. This
    repo instead uses TorchEEGClassifier's built-in inverse-class-frequency
    loss weighting (use_class_weights=True, the same mechanism
    SparseEvidenceGNNClassifier already uses for the same problem here --
    see cwt_gnn_classifiers.py's comment on why) plus the existing
    negative_to_positive_ratio subsampling in run_pipelines.py. Simpler,
    and consistent with how the rest of this codebase already handles this
    exact issue; not a claim that it matches the paper's numbers.
  - Validation split: the paper holds out the chronologically LAST 25% of
    each class's training-fold samples (Fig. 4), specifically to avoid
    leaking future-in-time information into early-stopping decisions. This
    file uses TorchEEGClassifier's standard (random) validation_split
    instead -- the temporal-holdout variant isn't implemented here.
  - Optimizer/learning-rate/batch-size/epoch-count: the paper doesn't state
    these (just "Keras 2.0 / TensorFlow 1.4, 4x K80 GPUs"). This file uses
    TorchEEGClassifier's AdamW + whatever epochs/batch_size/learning_rate
    the caller passes, same as every other classifier in this package.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view

try:
    from Epilepsy.pipelines.common import TorchEEGClassifier
except ModuleNotFoundError:
    from pipelines.common import TorchEEGClassifier


def compute_stft_features(
    X: np.ndarray,
    sampling_rate: int,
    stft_window_sec: float = 1.0,
    stft_hop_sec: float = 0.5,
    powerline_freq: float = 60.0,
    notch_halfwidth_hz: float = 3.0,
    remove_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel STFT magnitude spectrogram, Truong et al. 2018's
    preprocessing (Section II.B).

    X : (n_samples, n_channels, n_timepoints) raw EEG windows.

    Returns (stft, freqs):
      stft  : (n_samples, n_channels, n_time_bins, n_freq_bins) float32
              magnitude spectrogram, with the DC bin and the power-line
              notch bands already excluded from the frequency axis.
      freqs : (n_freq_bins,) the Hz value of each retained frequency bin
              (for introspection/plotting -- not needed downstream).

    At CHB-MIT's defaults (sampling_rate=256, 30s windows, stft_window_sec=
    1.0, stft_hop_sec=0.5, powerline_freq=60.0): a 1s sub-window at 256Hz
    gives 1Hz frequency resolution and rfft bins 0..128Hz (129 bins).
    Removing the DC bin (1) and the 57-63Hz/117-123Hz notch bands (7 bins
    each) leaves 129 - 1 - 7 - 7 = 114 frequency bins -- exactly the paper's
    reported input depth. Sliding a 1s window with a 0.5s hop across a 30s
    window gives (30*256 - 256) // 128 + 1 = 59 time bins -- exactly the
    paper's other reported dimension (Fig. 3: "n x 59 x 114").
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_channels, n_timepoints)")
    n_samples, n_channels, n_timepoints = X.shape

    win_len = int(round(stft_window_sec * sampling_rate))
    hop_len = int(round(stft_hop_sec * sampling_rate))
    if win_len <= 1 or hop_len <= 0:
        raise ValueError(f"stft_window_sec/stft_hop_sec too small for sampling_rate={sampling_rate}.")
    if n_timepoints < win_len:
        raise ValueError(
            f"windows have only {n_timepoints} samples, shorter than the "
            f"{win_len}-sample ({stft_window_sec}s @ {sampling_rate}Hz) STFT sub-window."
        )

    freqs = np.fft.rfftfreq(win_len, d=1.0 / sampling_rate)
    keep_mask = np.ones(len(freqs), dtype=bool)
    if remove_dc:
        keep_mask &= freqs > 0.0
    for harmonic in (1, 2):  # 57-63Hz and 117-123Hz at 60Hz mains
        center = powerline_freq * harmonic
        keep_mask &= ~((freqs >= center - notch_halfwidth_hz) & (freqs <= center + notch_halfwidth_hz))
    freq_idx = np.flatnonzero(keep_mask)
    if freq_idx.size == 0:
        raise ValueError("Notch filtering removed every frequency bin -- check powerline_freq/sampling_rate.")

    # Sliding sub-windows along the time axis without materializing an
    # explicit python loop over time bins: shape becomes
    # (n_samples, n_channels, n_windows_full, win_len), then keep every
    # hop_len-th one.
    windows = sliding_window_view(X, win_len, axis=-1)[:, :, ::hop_len, :]
    hann = np.hanning(win_len).astype(np.float32)
    spectrum = np.fft.rfft(windows * hann, axis=-1)
    magnitude = np.abs(spectrum)[..., freq_idx].astype(np.float32)
    return magnitude, freqs[freq_idx]


def fit_frequency_zscore_stats(stft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frequency-bin mean/std, pooled over (samples, channels, time) --
    Truong et al. 2018's "standardization ... across the whole frequency
    range" (Section II.B), meant to stop the naturally much larger
    low-frequency EEG power from swamping high-frequency bins. Fit on the
    training split only; apply_frequency_zscore then reuses these same
    stats for validation/test data.
    """
    axes = (0, 1, 2)
    mean = stft.mean(axis=axes)
    std = stft.std(axis=axes)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_frequency_zscore(stft: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (stft - mean) / std  # broadcasts over the trailing frequency axis


def k_of_n_alarm(predictions: np.ndarray, k: int = 8, n: int = 10) -> np.ndarray:
    """Rolling k-of-n alarm smoothing (Truong et al. 2018, Section II.D):
    for a chronologically-ordered sequence of per-window binary
    predictions from ONE continuous recording, the smoothed alarm at
    window i is 1 iff at least `k` of the last `n` raw predictions
    (windows i-n+1 .. i) are positive. Reduces isolated false positives
    during interictal periods without a Kalman filter. Paper defaults
    k=8, n=10 for 30s windows (a ~5-minute rolling context).

    predictions : 1-D array of 0/1, ALREADY restricted to one recording
        and in time order -- concatenating windows from different
        recordings/folds before calling this lets one recording's tail
        smear into the next recording's head.
    """
    predictions = np.asarray(predictions).astype(np.int64)
    if predictions.ndim != 1:
        raise ValueError("predictions must be 1-D (one recording's windows, in chronological order).")
    cumsum = np.concatenate([[0], np.cumsum(predictions)])
    idx = np.arange(len(predictions))
    lo = np.maximum(0, idx - n + 1)
    counts = cumsum[idx + 1] - cumsum[lo]
    return (counts >= k).astype(np.int64)


class TruongCNNCore(nn.Module):
    """The CNN in Fig. 3 / Section II.C: three (batch norm -> conv+ReLU ->
    max pool) blocks over an STFT spectrogram volume, then two
    fully-connected layers.

    Input is (batch, n_channels, n_time, n_freq) -- one STFT magnitude
    spectrogram per EEG channel. C1 treats n_channels as a genuine 3-D conv
    axis (kernel spans it completely, stride 1, so it always collapses to
    depth=1 in one step -- this is what fuses all channels into a single
    stack of 2-D feature maps); C2/C3 are then ordinary 2-D convs over
    those feature maps.
    """

    def __init__(
        self,
        n_channels: int,
        n_time: int,
        n_freq: int,
        n_classes: int = 2,
        c1_filters: int = 16,
        c2_filters: int = 32,
        c3_filters: int = 64,
        fc1_dim: int = 256,
        dropout: float = 0.5,
        **_unused,  # tolerate the device= kwarg TorchEEGClassifier._build_model_from_features passes through
    ) -> None:
        super().__init__()
        if n_time < 5 or n_freq < 5:
            raise ValueError(
                f"n_time={n_time}, n_freq={n_freq} too small for C1's 5x5 kernel -- "
                "use a longer STFT window or a shorter STFT sub-window/hop."
            )

        self.bn0 = nn.BatchNorm3d(1)
        self.conv1 = nn.Conv3d(1, c1_filters, kernel_size=(n_channels, 5, 5), stride=(1, 2, 2))
        # MaxPool2d, not MaxPool3d(kernel=(1,2,2)) -- mathematically identical
        # here (conv1's kernel spans n_channels exactly with stride 1, so that
        # axis is always length 1 right after conv1; pooling kernel=1 along a
        # length-1 axis is a no-op regardless of squeeze order), but
        # aten::max_pool3d_with_indices isn't implemented for the MPS backend
        # as of this writing (confirmed: conv1/bn0 -- Conv3d/BatchNorm3d --
        # run fine on MPS, only max_pool3d doesn't). Squeezing before pooling
        # (see _conv_features) instead of after avoids the gap entirely.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(c1_filters)
        self.conv2 = nn.Conv2d(c1_filters, c2_filters, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn2 = nn.BatchNorm2d(c2_filters)
        self.conv3 = nn.Conv2d(c2_filters, c3_filters, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # eval() for this probe pass -- in train() mode the BatchNorm layers
        # would normalize using (and update their running stats from) this
        # single all-zero dummy batch, silently polluting them before real
        # training ever starts. eval() uses (and leaves untouched) their
        # freshly-initialized running stats instead, so this is a pure
        # shape probe with no side effect on the model's learnable state.
        with torch.no_grad():
            self.eval()
            dummy = torch.zeros(1, n_channels, n_time, n_freq)
            flat_dim = self._conv_features(dummy).shape[1]
            self.train()
        if flat_dim == 0:
            raise ValueError(
                f"n_time={n_time}, n_freq={n_freq} collapse to an empty feature map after "
                "the three conv/pool blocks -- use a longer STFT window."
            )

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(flat_dim, fc1_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_dim, n_classes)

    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, n_channels, T, F) -- Conv3d wants an explicit feature-channel axis
        x = self.bn0(x)
        x = F.relu(self.conv1(x))
        x = x.squeeze(2)  # conv1's kernel spans n_channels exactly -> that axis is always length 1 here
        x = self.pool1(x)  # now a plain 2-D max pool over (T, F)

        x = self.bn1(x)
        x = self.pool2(F.relu(self.conv2(x)))

        x = self.bn2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_features(x)
        x = self.dropout1(x)
        x = torch.sigmoid(self.fc1(x))  # paper: sigmoid activation on the 256-unit FC layer
        x = self.dropout2(x)
        return self.fc2(x)  # raw logits; softmax is applied by CrossEntropyLoss/predict_proba, not here


class TruongSTFTCNNClassifier(TorchEEGClassifier):
    """sklearn-style wrapper: STFT preprocessing (compute_stft_features +
    per-frequency standardization) feeding TruongCNNCore, fit/predict via
    TorchEEGClassifier's shared training loop (see common.py) -- the same
    pattern _BaseCWTGNNClassifier uses for the CWT-GNN pipelines, with STFT
    in place of CWT and TruongCNNCore in place of a GNN.
    """

    _estimator_type = "classifier"
    model_label = "Truong-STFT-CNN"

    def __init__(
        self,
        sampling_rate: int = 256,
        stft_window_sec: float = 1.0,
        stft_hop_sec: float = 0.5,
        powerline_freq: float = 60.0,
        notch_halfwidth_hz: float = 3.0,
        c1_filters: int = 16,
        c2_filters: int = 32,
        c3_filters: int = 64,
        fc1_dim: int = 256,
        dropout: float = 0.5,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        use_class_weights: bool = True,
        verbose: int = 0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.stft_window_sec = stft_window_sec
        self.stft_hop_sec = stft_hop_sec
        self.powerline_freq = powerline_freq
        self.notch_halfwidth_hz = notch_halfwidth_hz
        self.c1_filters = c1_filters
        self.c2_filters = c2_filters
        self.c3_filters = c3_filters
        self.fc1_dim = fc1_dim
        self.dropout = dropout

        self.freq_mean_: np.ndarray | None = None
        self.freq_std_: np.ndarray | None = None
        self.stft_freqs_: np.ndarray | None = None

        self._init_torch_classifier(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            use_class_weights=use_class_weights,
            verbose=verbose,
        )

    def _prepare_features(self, X: np.ndarray, *, fit: bool, train_idx=None):
        # X arrives here already validated -- fit()/_predict_logits() in
        # common.py's TorchEEGClassifier both call validate_eeg_X() before
        # invoking this method.
        stft, freqs = compute_stft_features(
            X,
            sampling_rate=self.sampling_rate,
            stft_window_sec=self.stft_window_sec,
            stft_hop_sec=self.stft_hop_sec,
            powerline_freq=self.powerline_freq,
            notch_halfwidth_hz=self.notch_halfwidth_hz,
        )
        if fit:
            self.stft_freqs_ = freqs
            ref = stft if train_idx is None else stft[train_idx]
            self.freq_mean_, self.freq_std_ = fit_frequency_zscore_stats(ref)
        if self.freq_mean_ is None or self.freq_std_ is None:
            raise ValueError("Frequency standardization stats are not initialized -- call fit() first.")
        return apply_frequency_zscore(stft, self.freq_mean_, self.freq_std_)

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> TruongCNNCore:
        stft = features[0] if isinstance(features, tuple) else features
        _, n_channels, n_time, n_freq = stft.shape
        return TruongCNNCore(
            n_channels=int(n_channels),
            n_time=int(n_time),
            n_freq=int(n_freq),
            n_classes=n_classes,
            c1_filters=self.c1_filters,
            c2_filters=self.c2_filters,
            c3_filters=self.c3_filters,
            fc1_dim=self.fc1_dim,
            dropout=self.dropout,
            **kwargs,
        )
