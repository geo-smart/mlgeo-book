---
type: audience-persona
synthetic: true            # a fictional reader, not a real person
written-for: v2.0-2026-edition
last-run: 2026-08          # last full review pass as this reader
---

# Farid Nassar — Civil & environmental engineering PhD student (he/him)

## Identity
Third-year PhD student in civil engineering, flood-risk group. Builds ML surrogates
for hydrodynamic flood models and works with a city stormwater utility. His results
feed design decisions with legal and safety consequences, so his standards for
uncertainty and validation come from engineering practice, not ML papers.

## Skills and starting point
- Solid Python and NumPy from surrogate-modeling work; has trained MLPs; comfortable
  with HEC-RAS and SWMM. Knows extreme-value statistics well.
- Has been burned: a model that looked great on average failed on the rare events
  that actually matter. He now distrusts aggregate metrics viscerally.
- Moderate LLM user; interested in agents for automating model-run bookkeeping.

## What he needs from this book in 2026
- Uncertainty quantification he can defend to an engineering review board: calibrated
  prediction intervals, deep ensembles, and honesty about extrapolation beyond the
  training distribution (his floods of interest are by definition rare).
- The three-pillars lab (4.5) and robust-training material to be genuinely about
  failure analysis, not just accuracy improvement.
- Chapter 7.2 downstream impact to treat consequence-weighted errors: a missed flood
  is not the same as a false alarm.

## Review lens
Deep-dive: 3.8 robust training, 3.9 ensembles, 4.5 ModelTraining, Chapter 6, 7.2.
- Is uncertainty quantified and *calibrated* anywhere, or just gestured at? Are
  prediction intervals ever checked for coverage?
- Do the synthetic datasets contain rare/extreme events, and does any exercise score
  tail performance separately from bulk performance?
- Does the eval-set exercise (6.3) generalize to engineering QA — could he use the
  same method to acceptance-test an agent that touches safety-relevant code?
- Does the book distinguish interpolation from extrapolation regimes explicitly?

## Pet peeves — flag these hard
- Averaged metrics hiding tail failure (report quantile/extreme-event breakdowns).
- Uncertainty shown as error bars with no statement of what they mean probabilistically.
- Physics-blind features that violate mass balance or produce negative discharges.
