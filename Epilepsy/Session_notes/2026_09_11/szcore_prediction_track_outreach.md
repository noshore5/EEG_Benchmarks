# SzCORE prediction-track outreach -- recon + a second contact thread

**Date:** 2026-09-11 · Mac shell · branch `main` (no code changes)

## What was done (all advisory/recon, no repo changes)

- Confirmed `esl-epfl/szcore` has **Discussions disabled**
  (`gh api repos/esl-epfl/szcore --jq '.has_discussions'` -> `false`), so
  Step 2 of `PREDICTION_TRACK_PROPOSAL.md` is settled: post a regular
  **Issue**, not a Discussion.
- Pulled the repo's label list (`gh label list`): `enhancement` exists,
  no `proposal` label. No `evaluation`/`datasets`/`competition` labels fit
  as well. Since we're not a collaborator, decided to open the issue
  **without** self-applying labels -- let a maintainer label it.
- Talked through toning the RFC body down given the (correctly) expected
  low odds of a reply -- leaning toward a shorter ask (context + one
  question) rather than the full 4-question template, to keep the cost of
  being ignored near zero. Not yet edited into the proposal doc.
- Corrected a mistake I (Claude) made: conflated `train_detector.py`'s
  detection-repurposing of the `godoy_tmc`/TMCTransformer architecture
  with Godoy's actual role in this repo, which is a **prediction**
  architecture (`Epilepsy/pipelines/godoy_tmc_classifier.py`) that beat
  every other benchmarked model except the user's own, under this repo's
  strict LOSO protocol. Worth remembering: SzCORE's own detector reuses
  the *architecture* only, not the *task* -- don't compare the two.
- Researched (moderate confidence, not independently verified against
  primary sources) My Seizure Gauge (MSG) vs. NeuroVista: MSG is the
  ongoing multi-institutional forecasting program (Kuhlmann, Freestone,
  Cook et al.) that grew out of the NeuroVista implant trial/company
  (defunct ~2013); MSG data is hosted via epilepsyecosystem.org and spans
  wearables (non-invasive) + sub-scalp/intracranial implants. NeuroVista
  data specifically = 16-ch canine iEEG, variable-channel-count human
  iEEG (the data behind Kaggle 2014/2016). Flagged to the user that this
  should be confirmed against primary sources before it goes in anything
  public.
- Drafted a short, low-commitment outreach email to **Levin Kuhlmann**
  (Monash, MSG) -- separate from the existing Jonathan Dan
  (ESL/EPFL/SzCORE) thread. Ask type: "just a conversation/advice," not a
  role or collaboration pitch. Final short version:

  > Subject: Question about getting involved in seizure-forecasting research
  >
  > Hi Professor Kuhlmann,
  >
  > I've been independently benchmarking seizure-prediction architectures
  > (including a reconstruction of Godoy et al.'s TMC-Transformer) under
  > strict leave-one-seizure-out evaluation on CHB-MIT. I'm interested in
  > My Seizure Gauge and wondering if there's room for outside
  > involvement, even informally -- would you have 15-20 minutes to talk
  > about where the gaps are?
  >
  > Happy to share results/code if useful.
  >
  > Thanks,
  > Noah Shore

  Not yet sent.

## State / next

- Two parallel, independent outreach threads now exist: Jonathan Dan
  (SzCORE/ESL, re: a prediction track -- no reply yet) and Levin Kuhlmann
  (MSG, re: general involvement -- drafted, not sent). Track them
  separately; a reply on one doesn't resolve the other.
- Neither the shortened RFC issue body nor the Kuhlmann email has been
  sent/posted as of this note. `PREDICTION_TRACK_PROPOSAL.md`'s decision
  gates (3-4 week no-reply window) still apply to the Dan thread; no
  equivalent gate yet defined for Kuhlmann since it's a lower-stakes ask.
