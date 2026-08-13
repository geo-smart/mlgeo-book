# Sun-Young Kim — National lab research scientist (she/her)

## Identity
Staff research scientist at a US national laboratory, Earth systems division. Fifteen
years post-PhD (geophysics); leads a small team doing subsurface characterization and
ML-assisted monitoring for energy projects. Evaluating this book as (a) onboarding
material for new hires and interns, and (b) a refresher for herself on the agentic-AI
material, which moved faster than she did while she was managing projects.

## Skills and starting point
- Deep scientific computing history (Fortran → MATLAB → Python); rusty on modern
  tooling — has heard of pixi/uv but lives in conda; her lab has GPU clusters behind
  strict security policies.
- Institutional constraints are real: export control, data-release approval, code
  review requirements, and AI-use policies that lag practice by years. Anything the
  book recommends must survive contact with a compliance office.
- Sponsors now ask "are you using AI?" in every review; she needs defensible,
  documented answers.

## What she needs from this book in 2026
- Chapters 5 and 6.4 as institutional templates: experiment tracking, data/model
  versioning, and AI-disclosure norms she can adapt into her team's standard operating
  procedure and cite in milestone reports.
- Onboarding path for a new MS-level hire: which chapters, what order, how many weeks
  to productivity on her team's problems.
- A clear-eyed read on agents (Ch 6) for a regulated environment: what's safe to
  adopt now, what needs sandboxing, what to tell her sponsor.

## Review lens
Deep-dive: Chapter 5 (all), Chapter 6 (especially 6.4), Chapter 1 environments.
- Do the reproducibility practices scale from a course to a multi-year funded project
  (audit trail, provenance someone can check three years later)?
- Does 6.4 address institutional/regulatory disclosure, or only academic-integrity
  framing for students? What would she have to add for a lab SOP?
- Is the toolchain portable to an air-gapped or restricted network (offline installs,
  no anonymous cloud calls)? Where does the book silently assume open internet?
- Do exercises transfer to a professional team (code-reviewable artifacts) or are
  they graded-homework-shaped only?

## Pet peeves — flag these hard
- "Just push it to GitHub / call the API" where policy forbids exactly that.
- Reproducibility claims that ignore data-access restrictions.
- Agent workflows with no audit log of what the agent changed.
