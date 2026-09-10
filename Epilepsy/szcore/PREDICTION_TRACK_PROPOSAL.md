# Plan: proposing a seizure-*prediction* track for SzCORE

**Status (2026-09-10): not started. This is the playbook, not a record of
work done.** SzCORE / EpilepsyBench (`esl-epfl/szcore`) is today a
*detection* benchmark only (event F1, FP/24h; EDF in -> seizure-annotation
TSV out; container evaluated against held-out private data). This doc is
how to pitch and build a *forecasting* track without it dying as an
unsolicited drive-by PR.

The people: David Atienza's Embedded Systems Laboratory (ESL) at EPFL owns
the repo; Jonathan Dan (postdoc, ESL) is the EEG/wearable-seizure lead and
the person already contacted (no reply yet). Clinical side is CHUV
(Lausanne university hospital).

---

## The core problem, stated plainly

The benchmark's value is the **curated held-out evaluation set**, which the
maintainers control. A PR full of scaffolding with `TODO: collect data`
where the eval set goes gives them nothing they can run, and commits them
to a direction they haven't agreed to. Someone with long continuous
recordings (CHUV, or a data consortium) has to agree to hold them out.
That is institutional buy-in, not a code contribution -- and it is the
real blocker, not the plumbing.

So: **lead with a proposal and a working prototype on public data, not
with a PR.**

---

## Sequencing (each step gated on the previous)

### Step 0 -- recon (before touching their repo)

- Read `CONTRIBUTING.md` / the repo's issue + PR history. Gauge: do they
  take external contributions? How big? Is there an open discussion about
  a prediction/forecasting track already? (If there is, join it instead of
  opening a new one.)
- Skim the SzCORE paper (Dan et al.) and `timescoring` docs -- know
  exactly what the detection contract and scoring are, so the proposal
  speaks their vocabulary (SPH/SOP, event vs sample scoring, the
  BIDS-like format).
- Check the `epilepsyecosystem.org` portal (Levin Kuhlmann) and the
  NeuroVista / Kaggle-2014 / Melbourne-2016 data situation -- know what
  public forecasting data actually exists and its licence.

### Step 1 -- bump Jonathan Dan (low effort, do first)

Reply in the existing thread. Short. Don't re-explain the whole idea --
just resurface it and add the one new thing (you've started building a
prototype). Draft below.

### Step 2 -- open a GitHub issue / Discussion (the RFC)

**Where:** GitHub *Discussions* on `esl-epfl/szcore` if enabled
(category: Ideas), else a regular *Issue* labelled `enhancement` /
`proposal`. Discussions is better -- it signals "let's talk about
direction" not "here's a bug / here's a demand."

**Title:** `Proposal: a seizure-forecasting (prediction) evaluation track`

**Body:** use the template in the next section.

**Tone:** you are asking whether they *want* this and offering to do the
work, not announcing it. End with explicit questions so there's something
to reply to.

### Step 3 -- build the prototype in *your* fork (parallel with Step 2)

A running demo with real numbers on public data. Not merged, not proposed
for merge yet -- linked from the issue as "here's what I mean, concretely."
Scope:

- Forecasting output contract (a modified `infer.py`: continuous
  per-window seizure probability, or discrete alarms, with SPH/SOP).
- A scoring module: sensitivity, time-in-warning (or FPR/h), **and** the
  statistical test against a chance predictor (analytic Schelter/Snyder,
  or seizure-time surrogates). This is the rigor bar in the subfield --
  without it the track isn't credible.
- One baseline model end-to-end (a circadian-prior forecaster is a good
  cheap one; or a small CNN).
- A data adapter for at least one public dataset (CHB-MIT is weak but
  present; Epilepsyecosystem / Melbourne-2016 if the licence allows).
- A short doc: "how a challenge organiser plugs in held-out data."

This repo already has the reusable pieces to move fast: the
`Epilepsy/szcore/` detection scaffold (container, `infer.py`, Dockerfile),
and `trainer.py` for resumable training. The forecasting `infer.py` is a
variant of the detection one.

### Step 4 -- the PR (only after a "yes, and structure it like X")

Scope it tight:

- The scoring module (ideally as an extension to `timescoring` or a
  sibling package `timescoring`-style), with tests.
- The forecasting container contract + a reference container.
- Docs.
- **Not** the eval data -- that stays a maintainer/consortium
  responsibility; the PR just defines the slot it drops into.

Keep it reviewable: one coherent addition, tests, no repo-wide
refactor.

---

## The RFC issue -- body template

> ### Proposal: a seizure-forecasting (prediction) evaluation track
>
> **Context.** SzCORE today evaluates seizure *detection* (event F1,
> FP/24h). Seizure *forecasting* -- predicting that a seizure will occur
> within a future window -- is a distinct task with its own large
> literature (Kaggle 2014, Melbourne/NeuroVista 2016, My Seizure Gauge,
> Epilepsyecosystem) but no standardised, containerised, reproducible
> benchmark. Those past efforts were one-off competitions whose data is
> now hard or impossible to obtain. A SzCORE-style track would fix that.
>
> **What would need to change (sketch).**
> - *Task contract:* model outputs, at time t, a probability (or alarm)
>   that a seizure falls in `[t + SPH, t + SPH + SOP]`. New `infer.py`
>   output shape: a forecast time series, not events.
> - *Scoring:* new module -- sensitivity, time-in-warning / FPR-per-hour,
>   and a statistical test that the forecaster beats a chance predictor
>   (analytic, or seizure-time surrogates). Not covered by `timescoring`.
> - *Data:* long continuous recordings with extensive interictal baseline
>   and ideally multi-day context (circadian/multidien seizure cycles
>   matter for forecasting). This is the hard part -- see below.
>
> **The data question.** The benchmark's value is the held-out eval set.
> I can build the task contract, scoring, a reference container, and a
> baseline, and validate the whole pipeline on public data (CHB-MIT,
> Epilepsyecosystem). I can't provide a credible held-out forecasting
> set -- that needs institutional data (CHUV, or a consortium). Is that
> something ESL/CHUV would be positioned to contribute or curate?
>
> **What I'm offering.** I've built a SzCORE detection submission scaffold
> already and would prototype the forecasting pipeline in a fork, linked
> here for discussion before any PR.
>
> **Questions.**
> 1. Is a forecasting track something you'd want in SzCORE at all, or is
>    it deliberately out of scope?
> 2. If yes -- would you want it as part of this repo, a sibling repo, or
>    an extension to `timescoring`?
> 3. Is there existing/planned work on this I should join instead?
> 4. Who would own the held-out-data side?

---

## The follow-up to Jonathan Dan -- draft

> Hi Jonathan,
>
> Following up on my note from [when] about a seizure-forecasting track
> for SzCORE -- no worries if it got buried.
>
> Since then I've built a SzCORE detection submission scaffold, and I'd
> like to prototype a forecasting evaluation pipeline (task contract +
> scoring with a chance-predictor test + a baseline) in a fork, then
> open a Discussion on the repo to see whether it's a direction the ESL
> group would want.
>
> Before I do -- is a prediction track something SzCORE is open to, or is
> it intentionally detection-only? And is there work on this already
> underway that I should join instead?
>
> Thanks,
> Noah

(Adjust the "when" and trim to fit the original thread. Keep it this
short.)

---

## Prior art to cite (so the pitch looks informed)

| Effort | Year | Data | Held-out? | Task framing |
|---|---|---|---|---|
| AES Seizure Prediction Challenge (Kaggle) | 2014 | iEEG, 5 dogs + 2 humans (NeuroVista + EMU) | Yes -- Kaggle public/private split, AUC | 10-min clips, preictal vs interictal classification |
| Melbourne/NeuroVista (Kaggle) | 2016 | long-term human iEEG, 3 patients | Yes, closer to prospective | contiguous held-out, alarm-style |
| My Seizure Gauge / Epilepsyecosystem | ongoing | multimodal, wearable + iEEG | portal-gated | prospective forecasting |

Key caveat to raise: the 2014 contest's segment-based design **leaked**
(clip adjacency + time-of-day were predictive on their own). A modern
track must evaluate on long continuous held-out recordings with a
prospective alarm/forecast contract, and always report
improvement-over-chance -- not curated preictal-vs-interictal clips scored
on AUC. Brinkmann et al. (Brain, 2016) is the reference for doing the
honest prospective re-analysis.

---

## Decision gates (don't skip ahead)

- No reply to Step 1 + Step 2 within ~3-4 weeks -> one more contact
  (a different ESL person active on the repo, referencing the open
  Discussion), then treat it as "not wanted / not now" and either build
  it as a standalone benchmark under our own name or shelve it.
- "Interested but no held-out data" -> the track is a spec + scoring
  library other people can use with their own data; still worth shipping,
  lower ambition.
- "Yes, and we have data / a consortium" -> then the full PR is worth the
  effort.
