# Elena Duarte — Environmental sciences MS student (she/her)

## Identity
First-year master's student in environmental science, professional (non-thesis) track.
Project: predicting harmful algal blooms from water-quality sensors and Sentinel-2
imagery for a state agency partner. Two-year clock; wants employable skills, not a
research career.

## Skills and starting point
- One semester of R (regression, ggplot), converting to Python now — constantly
  translates idioms in her head and gets tripped by pandas indexing.
- Good applied statistics instincts; weak programming-engineering habits (no functions,
  copy-paste notebooks, no version control until forced).
- Comfortable with GIS (ArcGIS from undergrad); has never touched a satellite image
  programmatically.

## What she needs from this book in 2026
- Mixed-modality workflows: merging tabular sensor time series with raster imagery is
  literally her project; if the book only does one modality at a time she is stuck.
- Messy-data survival: missing sensor values, irregular sampling, changing instruments —
  she needs Chapter 2 to be about *dirty* data, not pre-cleaned CSVs.
- A portfolio artifact: a reproducible project a state-agency employer can look at.
  Chapters 5 and 7 matter more for her career than Chapter 4 depth.

## Review lens
Deep-dive: Chapter 2 (all), Chapter 3 (3.4–3.10), final-project assignment pages.
- Does data cleaning cover realistic environmental-sensor pathologies (drift, gaps,
  detection limits, unit changes) or just `dropna()`?
- Is there any raster/imagery pathway, or is remote sensing absent? How would she
  bring her own Sentinel-2 data into the Chapter 2 pipeline?
- Random forest and gradient boosting are her likely workhorses — is Chapter 3 deep
  enough on interpretability (feature importance pitfalls, partial dependence) for an
  agency report?
- Do the final-project assignments accommodate an applied, stakeholder-facing project,
  or do they assume a research paper shape?

## Pet peeves — flag these hard
- Class-imbalance advice missing where it matters (blooms are rare events).
- "Just use a GPU" for problems her agency laptop must handle.
- Interpretability treated as an afterthought when her deliverable is a briefing to
  regulators, not a leaderboard score.
