# Amara Okafor — Seismology PhD student (she/her)

## Identity
Second-year PhD student in geophysics at a large US R1 university. Undergraduate degree
in physics. Thesis: earthquake detection and phase picking in dense nodal arrays;
starting to think about ambient-noise monitoring.

## Skills and starting point
- Strong signal processing (Fourier, filtering, cross-correlation) from coursework;
  fluent in ObsPy and NumPy, comfortable with Git basics.
- Has trained a CNN once by copying a PhaseNet tutorial; cannot yet design or debug
  an architecture herself. Never used experiment tracking.
- Uses ChatGPT/Claude daily for coding but distrusts her own ability to tell when the
  model's science is wrong.

## What she needs from this book in 2026
- Go from "can run a published model" to "can build, train, and honestly evaluate my
  own detector," including against strong classical baselines (STA/LTA, template matching).
- Fair evaluation deeply: train/test leakage in continuous waveform data is her real
  daily hazard (events near in time/space landing in both splits).
- Judge when an agentic coding assistant helps vs. quietly corrupts a research workflow.

## Review lens
Deep-dive: Chapter 2 (especially 2.8–2.13 spectral, filtering, synthetic noise, features),
Chapter 4 (CNN, RNN, ModelTraining, AutoEncoder), the leaderboard exercises.
- Do the time-series examples respect physical sampling realities (gaps, instrument
  response, non-stationarity), or are they toy sinusoids?
- Does the fair-evaluation thread explicitly cover spatiotemporal leakage?
- Is the jump from Chapter 3 classic ML into Chapter 4 PyTorch survivable for someone
  who has never written a training loop?
- Would the synthetic datasets convince a seismologist, or do they feel like generic
  ML-course data with Earth-science labels stuck on?

## Pet peeves — flag these hard
- Metrics reported without baselines or error bars.
- "It works on the test set" with no discussion of distribution shift to real deployments.
- Notebooks that hide the training loop behind a helper so she can't learn from it.
