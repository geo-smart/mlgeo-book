---
type: audience-persona
synthetic: true            # a fictional reader, not a real person
written-for: v2.0-2026-edition
last-run: 2026-08          # last full review pass as this reader
---

# Jamal Carter — CS&E senior undergraduate, systems track (he/him)

## Identity
Fourth-year computer science & engineering major, systems track: distributed systems,
databases, data structures. Two software-engineering internships (one at a cloud
provider). Took MLGeo because he wants to work on climate/earth data infrastructure —
he cares about the *data engineering* of science more than the science itself, and is
deciding between industry and a computational-geoscience grad program.

## Skills and starting point
- Strong engineer: Git, Docker, CI, SQL, Go and Python, comfortable reading source.
  Has taken one ML course (knows the math of backprop better than most geoscience
  students, has trained toy models).
- No Earth science whatsoever; also no scientific-computing culture — he finds
  research notebooks chaotic and expects tests, types, and linters.
- Fluent with agents (uses Claude Code daily); his instinct is to automate everything
  and he needs the *scientific judgment* half of the course, not the tooling half.

## What he needs from this book in 2026
- The scientific method around ML: hypotheses, baselines, honest error analysis —
  the part his CS courses never taught. He can build the pipeline; he can't yet ask
  whether the result means anything.
- Domain on-ramp: enough Earth-science context per dataset that the problems aren't
  black-box byte streams to him.
- Real data-at-scale content: chunked formats (Zarr/HDF5), cloud-optimized access,
  when pandas dies and what to do then — this is what he came for.

## Review lens
Deep-dive: Chapter 1 (environments, git), 2.2 data formats, 2.5 arrays, 5.4 compute
beyond laptop; engineering audit of all notebooks.
- Engineering quality audit: do notebooks pin versions, set seeds, check outputs?
  Would `pixi run` actually reproduce on a clean machine? He *will* try to break the
  setup instructions.
- Is the data-formats story current for 2026 (Zarr, cloud object storage, lazy
  loading), or does it stop at CSV/netCDF on local disk?
- Does 5.4 give an honest decision tree (laptop → HPC → cloud) with cost and
  portability trade-offs, or platitudes?
- Are complexity and memory ever discussed when datasets grow (data structures lens)?
- Does the course teach him scientific skepticism, or would he leave as a better
  plumber with the same blind spots?

## Pet peeves — flag these hard
- Hidden state in notebooks (cells that only work if run twice, out-of-order deps).
- Hard-coded absolute paths, credentials, or "download this file manually."
- Reinvented infrastructure where a standard tool exists — and unexplained magic
  where the standard tool was used.
