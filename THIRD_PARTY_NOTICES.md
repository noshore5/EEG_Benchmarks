# Third-party data notices

## CHB-MIT Scalp EEG Database

This repository redistributes subject `chb01` from the CHB-MIT Scalp EEG
Database (version 1.0.0), collected at Boston Children's Hospital and
distributed via PhysioNet. The same files ship in the RunPod image and as
the `chb01.tar.gz` asset on the `chbmit-chb01-1.0.0` GitHub Release.

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

Only `chb01` is redistributed here -- it's the only subject
`Epilepsy/run_pipelines.py`'s `DEFAULT_SUBJECTS` exercises by default. Other
subjects still download on demand via `datasets/epilepsy/chb_mit.py` from
PhysioNet's S3 mirror; this notice covers the data actually shipped in the
image and the GitHub Release.
