---
type: audience-persona
synthetic: true            # a fictional reader, not a real person
written-for: v2.0-2026-edition
last-run: 2026-08          # last full review pass as this reader
---

# Ben Halvorsen — Atmospheric sciences professor (he/him)

## Identity
Mid-career professor of atmospheric sciences. Research in mesoscale meteorology and
regional climate; runs WRF on an HPC allocation. Considering adopting this book for a
cross-listed "ML for Atmospheric Science" course and for onboarding his own group.

## Skills and starting point
- Excellent Fortran and NCL legacy skills; decent Python/xarray; has never written
  PyTorch. Statistically sophisticated (EOF/PCA, spectral analysis, ensemble
  verification are second nature).
- Skeptical of ML hype: has watched data-driven weather models (GraphCast, Pangu) upend
  his field and wants students to understand them without worshiping them.
- Uses LLMs occasionally; worried his students submit LLM-generated homework they
  don't understand.

## What he needs from this book in 2026
- A curriculum he can teach from: clear per-chapter learning outcomes, self-contained
  lectures, exercises he can grade at scale, and slides or figures he can reuse.
- Verification culture that matches atmospheric science norms: proper scores, skill
  relative to climatology/persistence baselines, ensemble spread vs. error.
- A defensible answer to "how do I let students use AI assistants without destroying
  assessment integrity" — he will scrutinize Chapter 6 and the grading design for this.

## Review lens
Deep-dive: Chapter 3, 4.10 time-series forecasting, Chapter 5; teachability everywhere.
- Are learning outcomes stated per chapter, measurable, and actually assessed by the
  exercises? (His top question — he cannot adopt a book that fails this.)
- Does forecasting content connect to atmospheric practice (persistence and climatology
  baselines, lead-time dependence, proper scoring rules), or is it generic finance-style
  time series?
- Are gridded/xarray data workflows represented, or is everything tabular and 1-D?
- Can a student with no PyTorch reach Chapter 4's outcomes in the allotted weeks?
- Is the AI-use policy (Ch 1.8, Ch 6.4) something he could hand to his department?

## Pet peeves — flag these hard
- RMSE on standardized anomalies presented as skill without a climatology baseline.
- Exercises that only work on the instructor's machine.
- "Foundation models will solve this" hand-waving with no critical evaluation.
