"""Datasets, structured like MOABB's ``moabb.datasets`` package.

Each dataset subclasses :class:`moabb.datasets.base.BaseDataset` and knows
how to download/cache its own files and return :class:`mne.io.Raw` objects
keyed by subject/session/run.
"""
