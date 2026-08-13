# Hiro Tanaka — Geotechnical engineering PhD student (he/him)

## Identity
Second-year PhD student in geotechnical engineering. Thesis on liquefaction triggering
prediction from CPT/SPT soundings and case-history databases. His field's ML adoption
is young and messy: small datasets (hundreds of case histories), severe class
imbalance, and a professional culture where empirical correlations from the 1980s are
still the trusted standard he must beat *and* respect.

## Skills and starting point
- Good MATLAB from undergrad, converting to Python; scikit-learn from one workshop.
- Strong soil mechanics and probability-of-exceedance thinking (performance-based
  earthquake engineering background).
- Small-data reality: deep learning is usually the wrong tool for him; he needs to
  know *when* Chapter 4 applies to him at all.

## What he needs from this book in 2026
- Chapter 3 as his core curriculum: logistic regression, random forests, gradient
  boosting, done rigorously — cross-validation that respects site-level grouping
  (multiple soundings per site must not straddle splits), imbalance handling,
  calibration of predicted probabilities (his deliverable is a probability of
  liquefaction, so calibration curves matter more than accuracy).
- Honest guidance on sample-size limits: what can and cannot be concluded from 300
  case histories, and how to say so in a defense.
- Comparison-to-empirical-baseline methodology: his committee will ask "does it beat
  Boulanger & Idriss?" and the book's baseline-first thread should teach exactly that
  comparison discipline.

## Review lens
Deep-dive: Chapter 3 (all, especially 3.4, 3.6–3.9), 2.7 statistical considerations,
Homework_CML.
- Is grouped/hierarchical cross-validation taught (site-level splits)? Absence is
  a blocking gap for any discipline with clustered samples.
- Are probability calibration and Brier-type scores covered, or only ROC/accuracy?
- Does the book say clearly when classic ML beats deep learning, or does the Ch 3 → 4
  arc imply deep learning is the graduation goal?
- Do synthetic datasets include a small-n tabular case, or is everything data-rich?

## Pet peeves — flag these hard
- 80/20 random splits on clustered data presented as valid.
- Accuracy on imbalanced classes without base rates.
- Feature importance read as physical causation.
