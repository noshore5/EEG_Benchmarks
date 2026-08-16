# Results: detection vs. prediction are NOT comparable

This directory holds output from two deliberately separate testing paradigms (see Epilepsy/run_pipelines.py's module docstring):

- `leave_one_seizure_out_*.csv` (this directory): `--label-mode detection` -- binary ictal/interictal, recognizing a seizure that's already happening.
- `prediction/prediction_leave_one_seizure_out_*.csv` + `prediction/prediction_per_seizure_*.csv`: `--label-mode prediction` -- SPH/SOP-style preictal/interictal, predicting a seizure before it starts.

**A lower prediction score is not necessarily a worse model.** These are different tasks with different achievable ceilings -- detection's positive windows sit inside the seizure's own abnormal signal, while prediction's positive windows are, by definition, signal recorded before anything overtly abnormal has happened yet. Do not average, rank, or otherwise pool rows from `detection` and `prediction` files together in any downstream script or summary table.

Prediction results also carry event-level metrics (`hit`, `false_alarms_per_hour` per held-out seizure) that detection's output doesn't have and that window-level precision/recall/F1/average_precision/roc_auc can't substitute for -- see `leave_one_seizure_out_prediction`'s docstring in run_pipelines.py.
