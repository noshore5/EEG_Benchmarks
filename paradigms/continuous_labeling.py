"""Continuous labeling paradigm.

Adapted from the MOABB paradigm pattern (a class that turns a dataset's raw
recordings into (X, y) ready for a model), but built for continuous
recordings rather than event-locked trials. motor_imagery/p300/ssvep all
assume a stim channel firing discrete, fixed-length trials; seizure data
(and other continuous-monitoring data) has none of that -- a recording is
hours long with sparse, variable-length ictal spans documented as
:class:`mne.Annotations`.

This paradigm instead slides a fixed-length window across each Raw and
labels every window by how much it overlaps an annotation of interest
(e.g. "seizure"). It returns plain ``(X, y)`` arrays, not ``mne.Epochs`` --
downstream models here (wavelet-coherence graphs etc.) consume arrays, and
there's no meaningful "trial" to epoch around.

This module intentionally does *not* subclass ``moabb.paradigms.base.
BaseProcessing`` -- that pipeline (``FixedPipeline``/``StepType``/
``RawToEvents``/``SetRawAnnotations``) is built entirely around a stim
channel firing discrete, fixed-length trials, which continuous seizure
recordings don't have. Datasets it consumes (e.g.
:class:`datasets.epilepsy.CHBMIT`) do subclass the real
``moabb.datasets.base.BaseDataset``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from moabb.datasets.base import BaseDataset


log = logging.getLogger(__name__)


class ContinuousLabelingParadigm:
    """Slide a fixed-length window across continuous recordings and label each
    window ictal/interictal by its overlap with the dataset's annotations.

    Parameters
    ----------
    window_length : float (default 4.0)
        Length of each window, in seconds.
    step_size : float (default 1.0)
        Step between consecutive window starts, in seconds. In practice,
        pick this to match the step of whatever transform runs on the
        windows downstream (e.g. a wavelet-coherence hop size), so windows
        line up with the features computed on them.
    label_event : str (default "seizure")
        The annotation description treated as the positive ("ictal") class.
        Everything else is negative ("interictal"). Datasets that also want
        a "preictal" class (SOP/SPH-style) can attach a differently-named
        annotation and run this paradigm twice with different
        ``label_event`` values, or subclass to add a 3-way label rule.
    min_overlap : float (default 0.0)
        Minimum overlap (in seconds) between a window and an annotation span
        required to label the window positive. The default, 0.0, means any
        overlap at all is enough.
    fmin, fmax : float | None
        Optional bandpass filter applied to each Raw before windowing.
    resample : float | None
        Optional resample rate (Hz) applied to each Raw before windowing.
    """

    def __init__(
        self,
        window_length: float = 4.0,
        step_size: float = 1.0,
        label_event: str = "seizure",
        min_overlap: float = 0.0,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        resample: Optional[float] = None,
    ):
        if window_length <= 0:
            raise ValueError("window_length must be > 0")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        if min_overlap < 0:
            raise ValueError("min_overlap must be >= 0")
        self.window_length = window_length
        self.step_size = step_size
        self.label_event = label_event
        self.min_overlap = min_overlap
        self.fmin = fmin
        self.fmax = fmax
        self.resample = resample

    def _window_starts(self, duration: float) -> np.ndarray:
        """Start times (seconds) of every full window that fits in ``duration``."""
        n_windows = int(np.floor((duration - self.window_length) / self.step_size)) + 1
        if n_windows <= 0:
            return np.array([], dtype=float)
        return np.arange(n_windows) * self.step_size

    def _label_windows(self, starts: np.ndarray, annotations) -> np.ndarray:
        spans = [
            (onset, onset + duration)
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description
            )
            if description == self.label_event
        ]
        labels = np.zeros(len(starts), dtype=int)
        if not spans:
            return labels
        for i, start in enumerate(starts):
            end = start + self.window_length
            for onset, offset in spans:
                overlap = min(end, offset) - max(start, onset)
                if overlap > self.min_overlap:
                    labels[i] = 1
                    break
        return labels

    def get_data(
        self, dataset: BaseDataset, subjects: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Window every raw recording in ``dataset`` and label ictal/interictal.

        Parameters
        ----------
        dataset : BaseDataset
            A dataset whose ``get_data()`` returns
            ``{subject: {session: {run: raw}}}`` with annotations already
            attached to each raw (see e.g. ``datasets.epilepsy.CHBMIT``).
        subjects : list of int | None
            Subjects to include. If None, all of ``dataset.subject_list``.

        Returns
        -------
        X : np.ndarray, shape (n_windows, n_channels, n_samples)
        y : np.ndarray, shape (n_windows,)
            1 for windows overlapping a ``label_event`` annotation, 0
            otherwise.
        metadata : list of dict
            One dict per window (same order as X/y) with ``subject``,
            ``session``, ``run``, ``window_start``, ``window_end`` (the
            latter two in seconds, relative to the start of that run's
            recording).
        """
        data = dataset.get_data(subjects=subjects)

        X_parts: List[np.ndarray] = []
        y_parts: List[int] = []
        metadata: List[dict] = []

        for subject, sessions in data.items():
            for session, runs in sessions.items():
                for run, raw in runs.items():
                    raw = raw.copy().load_data(verbose="ERROR")
                    if self.fmin is not None or self.fmax is not None:
                        raw.filter(self.fmin, self.fmax, verbose="ERROR")
                    if self.resample is not None:
                        raw.resample(self.resample, verbose="ERROR")

                    sfreq = raw.info["sfreq"]
                    duration = raw.n_times / sfreq
                    starts = self._window_starts(duration)
                    if len(starts) == 0:
                        continue
                    labels = self._label_windows(starts, raw.annotations)

                    n_samples = int(round(self.window_length * sfreq))
                    # MOABB paradigms scale by dataset.unit_factor (default
                    # 1e6, volts -> microvolts); match that convention so
                    # output here is on the same scale as other paradigms.
                    raw_data = raw.get_data() * dataset.unit_factor
                    for start, label in zip(starts, labels):
                        start_sample = int(round(start * sfreq))
                        window = raw_data[:, start_sample : start_sample + n_samples]
                        if window.shape[1] != n_samples:
                            continue  # last partial window, skip
                        X_parts.append(window)
                        y_parts.append(int(label))
                        metadata.append(
                            {
                                "subject": subject,
                                "session": session,
                                "run": run,
                                "window_start": float(start),
                                "window_end": float(start + self.window_length),
                            }
                        )

        if X_parts:
            X = np.stack(X_parts, axis=0)
        else:
            X = np.empty((0, 0, 0))
        y = np.array(y_parts, dtype=int)
        return X, y, metadata
