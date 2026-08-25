# For Claude

Read `CONTEXT.md` first, before anything else in this repo -- it's the
living current-state doc (branch map, active work, known gotchas). This
repo is worked on from multiple Claude/Grok shells that don't share
context with each other, so `CONTEXT.md` is what keeps a new session from
re-deriving things a previous one already figured out.

Before ending a session that changed anything `CONTEXT.md` describes
(branch state, an in-flight run's status, a newly discovered gotcha, an
open thread getting resolved or a new one starting), update `CONTEXT.md`
to match -- not just a session note. Session notes
(`Epilepsy/Session_notes/<date>/`) are the detailed historical record and
should still be written; `CONTEXT.md` is the short pointer-heavy summary
of what's still true *right now*, kept current on top of them.

Facts that flipped in 2026-08-25 and are easy to reverse by reading old
comments -- confirm in `CONTEXT.md` rather than the file-level docstring
that introduced them:

- Current branch is `continuous-cwt-mamba`.
- `_DenseEdgeMambaTemporal.use_cuda_kernel` is auto-detect, not forced
  `False`. Fused-kernel image:
  `ghcr.io/noshore5/eeg_benchmarks-mamba:20260825-4760de0`.
- Continuous Mamba default scan is `"chunk"` (carried-state pscan), not
  the Python `step()` loop. Real CHB-MIT / LOSO pipeline for that
  paradigm is not started.
