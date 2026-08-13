# Audience personas

Twelve simulated readers of the MLGeo book. Each file is a self-contained agent prompt:
give a review agent the persona file plus a chapter assignment and it reviews the book
*as that person*.

**These are for you to take.** If you are adapting this book for a different
audience — a hydrology master's programme, a national lab's internal training, an
undergraduate course with no Python prerequisite — rewrite these personas for your
readers and re-run the review. That is the whole method, and it transfers. See
[CONTRIBUTING.md](../CONTRIBUTING.md#personalizing-the-book-for-your-audience) for the
workflow, and [`translations/personas/`](../translations/personas/) for the other axis:
personas that steer *language and culture* rather than discipline and seniority.

> ⚠️ **These readers are fictional.** They are design instruments — a way to hold a
> specific reader in mind while revising — not evidence that any real geoscientist
> accepted the result. They earned their place by catching real errors, but an AI
> reading an invented reader's brief is not review by that community. Real human
> review is recorded in [`docs/REVIEW_RECORD.md`](../docs/REVIEW_RECORD.md).

## Coverage matrix

| # | Persona | Discipline | Seniority | Deep-dive chapters |
|---|---------|-----------|-----------|--------------------|
| 01 | Amara Okafor | Geophysics (seismology) | PhD student, yr 2 | 2, 4, leaderboard |
| 02 | Ben Halvorsen | Atmospheric sciences | Professor | 3, 4.10, 5; teachability everywhere |
| 03 | Carmen Reyes | Oceanography | Senior undergrad | 1, 2, 3; onboarding ramp |
| 04 | Anjali Deshpande | Geology (structural/field) | Professor | 1, 3, 7; low-code accessibility |
| 05 | Elena Duarte | Environmental sciences | MS student | 2, 3; tabular + remote sensing |
| 06 | Farid Nassar | Civil & environmental eng. | PhD student | 3.8, 4.5, 6, 7.2; risk & uncertainty |
| 07 | Grace Liu | Geohydrology | Postdoc | 4.7 PINN, 5; sparse/irregular data |
| 08 | Hiro Tanaka | Geotechnical engineering | PhD student | 3; small-data, imbalance |
| 09 | Ingrid Weber | CS&E (NLP / ML) | Postdoc | 4, 6; ML correctness, eval design |
| 10 | Jamal Carter | CS&E (systems/data structures) | Senior undergrad | 1, 2.2, 5.4; environments & scale |
| 11 | Sun-Young Kim | National lab research scientist | Staff scientist | 5, 6.4; operational transfer |
| 12 | Leona Marchetti | Philanthropy scientific advisor | Program officer | About, 7; outcomes & impact framing |

Seniority coverage: 2 professors, 2 postdocs, 4 graduate students, 2 senior undergrads,
2 lower-priority audiences (national lab scientist, philanthropy advisor).

## How each review runs

Every persona agent receives:
1. Its persona file (identity, skills, goals, review lens).
2. A **skim pass** over the whole TOC + chapter readmes, and a **deep pass** over its
   assigned chapters (notebooks read in full).
3. A required output structure:
   - **Learning-outcome gaps** — what a person like me must know in 2026 that the book
     doesn't teach me, or teaches too late/too shallowly.
   - **Content findings** — errors, outdated framing, missing prerequisites, per file.
   - **Design & navigation findings** — ordering, ramp, notebook ergonomics.
   - **Synthetic-exercise fit** — do the mlgeo_synth datasets and graded exercises
     work for my discipline and level? What would I substitute?
   - **Top 5 proposed changes**, each with severity (blocking / important / nice-to-have)
     and the specific file(s) it touches.

Personas must stay in character: report what *they* would struggle with or dismiss,
not generic editorial feedback.
