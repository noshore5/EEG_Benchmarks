# Results: detection vs. prediction are NOT comparable

This directory holds output from two deliberately separate testing paradigms (see Epilepsy/run_pipelines.py's module docstring):

- `leave_one_seizure_out_*.csv` (this directory): `--label-mode detection` -- binary ictal/interictal, recognizing a seizure that's already happening.
- `prediction/prediction_leave_one_seizure_out_*.csv` + `prediction/prediction_per_seizure_*.csv`: `--label-mode prediction` -- SPH/SOP-style preictal/interictal, predicting a seizure before it starts.

**A lower prediction score is not necessarily a worse model.** These are different tasks with different achievable ceilings -- detection's positive windows sit inside the seizure's own abnormal signal, while prediction's positive windows are, by definition, signal recorded before anything overtly abnormal has happened yet. Do not average, rank, or otherwise pool rows from `detection` and `prediction` files together in any downstream script or summary table.

Prediction results also carry event-level metrics (`hit`, `false_alarms_per_hour` per held-out seizure) that detection's output doesn't have and that window-level precision/recall/F1/average_precision/roc_auc can't substitute for -- see `leave_one_seizure_out_prediction`'s docstring in run_pipelines.py.

- `truong_stft_cnn/truong_stft_cnn_leave_one_seizure_out_*.csv` + `truong_stft_cnn/truong_stft_cnn_per_seizure_*.csv`: `--pipeline truong_stft_cnn` -- a DIFFERENT ARCHITECTURE (STFT+CNN, replicating Truong et al. 2018) on the same prediction task, not just a different label rule. Also not comparable to `detection/`'s numbers, and not directly comparable to `prediction/`'s either (different model, different window length by default) even though both solve the same task -- see Epilepsy/pipelines/truong_stft_cnn_classifier.py's module docstring.

- `dense_edge/leave_one_seizure_out_*.csv` and `dense_edge/prediction/`: `--pipeline dense_edge` -- same SparseEvidenceGNNClassifier and leave-one-seizure-out loops as `dense_edge_gru`, but `dense_edge_temporal_mode="conv"` (Conv2d over time) instead of `"rnn"` (per-edge GRU). Do not pool these with `leave_one_seizure_out_*.csv` / `prediction/` -- those are the GRU pipeline's historical layout.

- `cwt_encoder/` (GRU) and `dense_edge/cwt_encoder/`: `--cwt-encoder` on the corresponding dense-family pipeline -- learned CWT node embeddings on top of WCT edges. Do not pool these with the WCT-only CSVs in `prediction/` / `dense_edge/prediction/`.
- `cwt_encoder/node_only/`: `--cwt-encoder-ablation node_only` -- CWT encoder with WCT edge features (and WCT compute) off. Not comparable to joint CWT+WCT runs under `cwt_encoder/`.
