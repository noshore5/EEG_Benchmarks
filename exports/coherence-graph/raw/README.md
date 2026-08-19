# Raw coherence arrays

`coherence.npz` -- the full, real, unthresholded pipeline output for the one chosen preictal window (chb01, seizure 1_16_0, t=704s), computed by SparseEvidenceGNNCore._coherence_only + compute_events (see build_coherence_export.py alongside this file for the exact call sequence). Arrays:

- `coh` [E, T, F]: coherence magnitude, 0-1, per (edge, time, freq) cell.
- `phase` [E, T, F]: phase angle, radians.
- `significance` [E, T, F]: (coh - coherence_threshold) / coherence_threshold, COI-masked (same convention as `coh`/`phase`) -- the continuous significance channel event_mode="dense" (dense_edge_gru's real mode) actually consumes at forward() time; see SparseEvidenceGNNCore._build_dense_edge_input. NOT derived from compute_events' discrete event list (that's event_mode="sparse"'s mechanism, used here only for the PNGs' arrow placement).
- `coi_valid` [T, F]: True outside the cone of influence (shared across all edges).
- `freqs_hz` [F]: the F frequency bins' real Hz values.
- `src_channel`/`dst_channel` [E]: edge e connects src_channel[e] <-> dst_channel[e] (same order/names as graph.json).
- `coherence_threshold`, `phase_threshold_rad`, `lowest_hz`, `highest_hz`, `sampling_rate`: the real config this was computed under (coherence_threshold/lowest_hz were overridden from DENSE_EDGE_GRU_PARAMS's trained values of 0.99/1.0 at the user's request -- see build_coherence_export.py's own comments).
- `window_start_s`, `seizure_id`, `subject`, `run`: which real window this is.

`edges/{src}__{dst}.npz` -- per significant edge, the same edge's `coh`/`phase`/`significance`/`coi_valid`/`freqs_hz` slice plus its consolidated real events (`event_t`, `event_freq_idx`, `event_sin`, `event_cos`, `event_mean_mag` -- what the PNG's arrows are drawn from) and `mean_coherence` (same number as graph.json's row for this edge).
