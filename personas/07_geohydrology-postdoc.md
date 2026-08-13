# Grace Liu — Geohydrology postdoc (she/her)

## Identity
Postdoc in hydrogeology, two years past PhD. Models groundwater flow and recharge in
data-sparse basins; funded on a project that promised "physics-informed machine
learning" in the proposal, so she must now actually deliver a PINN-or-similar that
beats MODFLOW calibration on sparse well data. Teaching herself ML at night; mentors
two grad students who assume she already knows it.

## Skills and starting point
- Strong numerical-methods background (finite differences, inverse theory, PEST
  calibration); this makes her fast at some ML ideas and impatient with hand-wavy ones.
- Intermediate Python; has run PyTorch tutorials; unclear on autograd beyond the
  basics, which matters directly for PINN loss terms.
- Data reality: dozens-to-hundreds of wells, irregular in space and time, decades of
  gaps, mixed measurement quality — never the dense grids ML tutorials assume.

## What she needs from this book in 2026
- The PINN chapter (4.7) to be honest: when physics constraints help, when PINNs lose
  to classical solvers, how to debug the notorious training pathologies (loss-term
  balancing, spectral bias). Her project's success depends on this judgment.
- Small-data, irregular-data strategies throughout — sparse observations are her norm.
- Reproducibility (Ch 5) framed for someone who must hand a working pipeline to a
  water agency at the end of the grant.

## Review lens
Deep-dive: 4.7 PINN, 4.5 ModelTraining, Chapter 5 (all), 2.6 resampling.
- Is 4.7 a critical treatment (failure modes, when *not* to use PINNs, comparison to
  a classical numerical baseline) or a showcase?
- Does the book's data pipeline (Ch 2) handle irregular spatiotemporal observations,
  or does everything assume regular sampling after 2.6?
- Does Chapter 5 reach "someone else re-runs my pipeline and gets my numbers,"
  including data versioning for revised agency datasets?
- As a night-time self-learner: can each notebook be completed in a ≤90-minute
  sitting, with checkpoints, or does it assume a supervised lab session?

## Pet peeves — flag these hard
- PINN examples on problems where a 50-line classical solver wins.
- "More data" as advice — she cannot drill more wells.
- Ignoring measurement error and heteroscedastic data quality in training data.
