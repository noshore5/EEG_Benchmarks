"""Continuous labeling paradigm.

Adapted from the MOABB paradigm pattern (a class that turns a dataset's raw
recordings into (X, y) ready for a model), but built for continuous
recordings rather than event-locked trials. motor_imagery/p300/ssvep all
assume a stim channel firing discrete, fixed-length trials; seizure data
(and other continuous-monitoring data) has none of that -- a recording is
hours long with sparse, variable-length ictal spans documented as
:class:`mne.Annotations`.

This paradigm instead slides a fixed-length window across each Raw and
labels every window according to ``label_mode``:

  - ``"detection"`` (default): binary ictal/interictal, by overlap with a
    ``label_event`` annotation. This is the original behavior of this class.
  - ``"prediction"``: SPH/SOP-style seizure *prediction* labeling (see
    ``__init__``'s docstring for the exact interval rules). This is a
    genuinely different task from detection -- predicting a seizure before
    it starts from a preictal window, not recognizing one that's already
    happening -- with a different, 3-way per-window label rule (positive/
    negative/excluded) rather than a relabeling of the same windows.

It returns plain ``(X, y)`` arrays, not ``mne.Epochs`` -- downstream models
here (wavelet-coherence graphs etc.) consume arrays, and there's no
meaningful "trial" to epoch around.

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

LABEL_MODES = ("detection", "prediction")


class ContinuousLabelingParadigm:
    """Slide a fixed-length window across continuous recordings and label
    each window according to ``label_mode`` (detection or prediction).

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
        The annotation description identifying seizure spans. In
        ``label_mode="detection"`` this is the positive class directly; in
        ``label_mode="prediction"`` these spans are the reference points
        (onset/offset) the SPH/SOP windows below are computed from.
    min_overlap : float (default 0.0)
        ``label_mode="detection"`` only. Minimum overlap (in seconds)
        between a window and an annotation span required to label the
        window positive. The default, 0.0, means any overlap at all is
        enough.
    label_mode : str (default "detection")
        ``"detection"``: binary ictal/interictal by overlap with
        ``label_event`` (original behavior, unchanged).

        ``"prediction"``: SPH/SOP-style labeling. For every seizure
        ``(onset, offset)`` documented in a recording:

        - **positive** (preictal): window overlaps
          ``[onset - sph - sop, onset - sph)`` -- the seizure occurrence
          period (SOP), the horizon during which a real prediction system
          would need to have already raised an alarm.
        - **excluded** (dropped from the dataset entirely -- not counted as
          either class): window overlaps ``[onset - sph, onset)`` (the SPH
          itself: too close to onset for a warning fired here to be
          useful), the ictal span ``[onset, offset)``, or the postictal
          buffer ``(offset, offset + postictal_buffer]``.
        - **negative** (interictal): everything else, provided the window
          also falls entirely outside ``interictal_buffer`` seconds of
          padding around the above (preictal-start, postictal-end) span for
          every seizure in the recording -- windows just outside the
          excluded zone but still close to a seizure are ambiguous enough
          to drop rather than trust as clean baseline.

        A window that is excluded by any seizure is excluded overall, even
        if it would otherwise be positive for a different (e.g. clustered)
        seizure -- see ``_label_windows_prediction``'s body for the exact
        combination rule.

        Positive windows get a ``seizure_id`` in the returned metadata
        (``f"{subject}_{run}_{k}"``, ``k`` = that seizure's index within its
        recording) plus ``seizure_onset``/``seizure_offset`` (seconds,
        relative to that recording's start) -- used downstream for
        leave-one-seizure-out fold grouping and event-level (per-seizure
        hit/miss) evaluation, not just window-level metrics. Negative
        windows get ``seizure_id=None``.
    sph : float (default 300.0)
        ``label_mode="prediction"`` only. Seizure Prediction Horizon,
        seconds -- the minimum lead time a real alarm needs; the SPH-long
        span immediately before onset is excluded (a warning fired there is
        too late to act on), not counted negative either. Starting point,
        not tuned.
    sop : float (default 1800.0)
        ``label_mode="prediction"`` only. Seizure Occurrence Period,
        seconds -- the span before the SPH during which a fired alarm
        counts as a (positive/preictal) prediction. Starting point, not
        tuned.
    postictal_buffer : float (default 1800.0)
        ``label_mode="prediction"`` only. Seconds after a seizure's offset
        excluded from the dataset (brain state right after a seizure isn't
        representative interictal baseline).
    interictal_buffer : float | None (default None)
        ``label_mode="prediction"`` only. Extra padding (seconds) around
        each seizure's full preictal-through-postictal span, beyond which a
        window is required to fall to be labeled negative. ``None`` (the
        default) reuses ``postictal_buffer``'s value.
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
        label_mode: str = "detection",
        sph: float = 300.0,
        sop: float = 1800.0,
        postictal_buffer: float = 1800.0,
        interictal_buffer: Optional[float] = None,
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
        if label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}")
        if sph < 0:
            raise ValueError("sph must be >= 0")
        if sop <= 0:
            raise ValueError("sop must be > 0")
        if postictal_buffer < 0:
            raise ValueError("postictal_buffer must be >= 0")
        if interictal_buffer is not None and interictal_buffer < 0:
            raise ValueError("interictal_buffer must be >= 0")
        self.window_length = window_length
        self.step_size = step_size
        self.label_event = label_event
        self.min_overlap = min_overlap
        self.label_mode = label_mode
        self.sph = sph
        self.sop = sop
        self.postictal_buffer = postictal_buffer
        self.interictal_buffer = interictal_buffer
        self.fmin = fmin
        self.fmax = fmax
        self.resample = resample

    def _window_starts(self, duration: float) -> np.ndarray:
        """Start times (seconds) of every full window that fits in ``duration``."""
        n_windows = int(np.floor((duration - self.window_length) / self.step_size)) + 1
        if n_windows <= 0:
            return np.array([], dtype=float)
        return np.arange(n_windows) * self.step_size

    def _seizure_spans(self, annotations) -> List[Tuple[float, float]]:
        """Chronologically-sorted (onset, offset) seconds for `label_event`
        annotations, shared by both label_mode's window-labeling methods."""
        return sorted(
            (onset, onset + duration)
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description
            )
            if description == self.label_event
        )

    def _label_windows(self, starts: np.ndarray, annotations) -> np.ndarray:
        """label_mode="detection": binary ictal/interictal by overlap."""
        spans = self._seizure_spans(annotations)
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

    def _label_windows_prediction(
        self, starts: np.ndarray, annotations
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
        """label_mode="prediction": 3-way SPH/SOP labeling -- see this
        class's own docstring for the exact interval rules.

        Returns
        -------
        labels : np.ndarray[int], shape (len(starts),)
            -1 (excluded), 0 (negative/interictal), or 1 (positive/preictal).
        seizure_idx : np.ndarray[int], shape (len(starts),)
            Index into ``spans`` of the seizure each positive window's
            preictal period belongs to; -1 for every non-positive window.
        spans : list of (onset, offset)
            The seizure spans this run's labels were computed from, in the
            same order ``seizure_idx`` indexes into.
        """
        spans = self._seizure_spans(annotations)
        n = len(starts)
        labels = np.zeros(n, dtype=int)
        seizure_idx = np.full(n, -1, dtype=int)
        if not spans:
            # No documented seizure in this recording under label_mode=
            # "prediction": nothing to predict, nothing to exclude --
            # everything here is valid interictal baseline.
            return labels, seizure_idx, spans

        interictal_buffer = (
            self.postictal_buffer if self.interictal_buffer is None else self.interictal_buffer
        )
        ends = starts + self.window_length

        excluded = np.zeros(n, dtype=bool)
        positive = np.zeros(n, dtype=bool)
        positive_seizure = np.full(n, -1, dtype=int)
        negative_ok = np.ones(n, dtype=bool)

        for k, (onset, offset) in enumerate(spans):
            preictal_start = onset - self.sph - self.sop
            preictal_end = onset - self.sph  # == warn zone start
            postictal_end = offset + self.postictal_buffer
            excl_start = preictal_start - interictal_buffer
            excl_end = postictal_end + interictal_buffer

            def overlaps(lo: float, hi: float) -> np.ndarray:
                # Same "any positive-length overlap" convention as
                # _label_windows's `overlap > self.min_overlap` default.
                return (starts < hi) & (ends > lo)

            overlap_preictal = overlaps(preictal_start, preictal_end)
            overlap_warn = overlaps(preictal_end, onset)
            overlap_ictal = overlaps(onset, offset)
            overlap_postictal = overlaps(offset, postictal_end)

            excluded |= overlap_warn | overlap_ictal | overlap_postictal

            newly_positive = overlap_preictal & ~positive
            positive |= overlap_preictal
            positive_seizure[newly_positive] = k

            negative_ok &= ~overlaps(excl_start, excl_end)

        # Excluded wins over positive (e.g. clustered seizures where a
        # window is simultaneously preictal for one and postictal for
        # another) -- an ambiguous window is dropped, not guessed at.
        labels = np.where(excluded, -1, np.where(positive, 1, 0))
        # A window that's neither positive nor excluded is only trustworthy
        # as negative/interictal if it also clears interictal_buffer around
        # every seizure; otherwise drop it too rather than mislabel it.
        labels = np.where((labels == 0) & (~negative_ok), -1, labels)
        seizure_idx = np.where(labels == 1, positive_seizure, -1)
        return labels, seizure_idx, spans

    def get_data(
        self, dataset: BaseDataset, subjects: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Window every raw recording in ``dataset`` and label it per ``label_mode``.

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
            ``label_mode="detection"``: 1 for windows overlapping a
            ``label_event`` annotation, 0 otherwise.
            ``label_mode="prediction"``: 1 for preictal windows, 0 for
            interictal windows. Excluded windows are dropped -- they never
            appear in X/y/metadata at all.
        metadata : list of dict
            One dict per window (same order as X/y) with ``subject``,
            ``session``, ``run``, ``window_start``, ``window_end`` (the
            latter two in seconds, relative to the start of that run's
            recording), and ``run_start_time`` (that recording's
            ``raw.info["meas_date"]`` as an ISO string, or ``None`` if the
            file doesn't carry one -- CHB-MIT/PhysioNet files sometimes
            anonymize this). ``label_mode="prediction"`` additionally adds
            ``seizure_id`` (``None`` for negative windows), ``seizure_onset``,
            and ``seizure_offset`` (seconds, relative to that recording's
            start; ``None`` for negative windows).
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

                    meas_date = raw.info.get("meas_date")
                    run_start_time = meas_date.isoformat() if meas_date is not None else None

                    sfreq = raw.info["sfreq"]
                    duration = raw.n_times / sfreq
                    starts = self._window_starts(duration)
                    if len(starts) == 0:
                        continue

                    if self.label_mode == "prediction":
                        labels, seizure_idx, spans = self._label_windows_prediction(
                            starts, raw.annotations
                        )
                    else:
                        labels = self._label_windows(starts, raw.annotations)
                        seizure_idx = None
                        spans = None

                    n_samples = int(round(self.window_length * sfreq))
                    # MOABB paradigms scale by dataset.unit_factor (default
                    # 1e6, volts -> microvolts); match that convention so
                    # output here is on the same scale as other paradigms.
                    raw_data = raw.get_data() * dataset.unit_factor
                    for i, (start, label) in enumerate(zip(starts, labels)):
                        if label < 0:
                            continue  # excluded (prediction mode only): drop, don't count either class
                        start_sample = int(round(start * sfreq))
                        window = raw_data[:, start_sample : start_sample + n_samples]
                        if window.shape[1] != n_samples:
                            continue  # last partial window, skip
                        X_parts.append(window)
                        y_parts.append(int(label))
                        window_meta = {
                            "subject": subject,
                            "session": session,
                            "run": run,
                            "window_start": float(start),
                            "window_end": float(start + self.window_length),
                            "run_start_time": run_start_time,
                        }
                        if self.label_mode == "prediction":
                            k = int(seizure_idx[i])
                            if k >= 0:
                                onset, offset = spans[k]
                                window_meta["seizure_id"] = f"{subject}_{run}_{k}"
                                window_meta["seizure_onset"] = float(onset)
                                window_meta["seizure_offset"] = float(offset)
                            else:
                                window_meta["seizure_id"] = None
                                window_meta["seizure_onset"] = None
                                window_meta["seizure_offset"] = None
                        metadata.append(window_meta)

        if X_parts:
            X = np.stack(X_parts, axis=0)
        else:
            X = np.empty((0, 0, 0))
        y = np.array(y_parts, dtype=int)
        return X, y, metadata
