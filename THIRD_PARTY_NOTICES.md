# Third-party data notices

## CHB-MIT Scalp EEG Database

This repository redistributes subjects `chb01`, `chb02`, `chb03`, and
`chb04` from the CHB-MIT Scalp EEG Database (version 1.0.0), collected at
Boston Children's Hospital and distributed via PhysioNet. `chb01`'s files
also ship in the RunPod image; all four ship as `chbXX.tar.gz` assets on
their own `chbmit-chbXX-1.0.0` GitHub Releases (see
`datasets/epilepsy/chb_mit.py`'s `GITHUB_RELEASE_SHA256` for the current
list of subjects mirrored this way).

- Source: https://physionet.org/content/chbmit/1.0.0/
- License: Open Data Commons Attribution License v1.0 (ODC-By 1.0) --
  https://physionet.org/content/chbmit/view-license/1.0.0/ -- permissive,
  requires attribution on redistribution. This notice is that attribution.

Please cite:

- Ali Shoeb. *Application of Machine Learning to Epileptic Seizure Onset
  Detection and Treatment*. PhD Thesis, Massachusetts Institute of
  Technology, September 2009.
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
  Mietus JE, Moody GB, Peng CK, Stanley HE. PhysioBank, PhysioToolkit, and
  PhysioNet: Components of a new research resource for complex physiologic
  signals. *Circulation* 101(23):e215-e220, 2000.

`chb01` is the only subject `Epilepsy/run_pipelines.py`'s `DEFAULT_SUBJECTS`
exercises by default, and the only one baked into the RunPod image; `chb02`,
`chb03`, and `chb04` are mirrored on GitHub Releases only, as a faster
alternative to PhysioNet's (throttled) S3 mirror for anyone who runs those
subjects. Subjects beyond these four still download on demand via
`datasets/epilepsy/chb_mit.py` from PhysioNet's S3 mirror; this notice
covers the data actually shipped in the image and on GitHub Releases.
