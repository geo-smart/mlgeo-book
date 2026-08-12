# MLGeo 2026 slide-deck standards

Every deck in this directory follows these rules. They exist so 26 decks read
as one course, and so a domain-science audience — geophysics, oceanography,
geology, hydrology, engineering; NOT CS/AMATH/data-science majors — can follow
every slide.

**Session shape**: each deck is a ~20-minute introduction — roughly 10 slides
— that hands off to the notebook exercise for the bulk of the 80-minute class
(10:00–11:20: pulse talks ~10 / deck ~20 / exercise ~40 / synthesis ~10).
Decks introduce; exercises teach. A deck that cannot be presented in 20
minutes is too long, not too slow.

## Geoscience first, concepts second, implementation last

The governing order for every slide and every explanation:

1. **Start from the geoscience** — a named dataset, phenomenon, or field
   problem (a GNSS drift, a liquefaction case history, a biomass map).
2. **Then the statistical concept**, in tool-independent language.
3. **Implementation naming last, and least** — scikit-learn/PyTorch spellings
   (GroupKFold, TimeSeriesSplit, StratifiedGroupKFold) belong in the notebook
   hand-off slide and speaker notes, never as the name of a concept on a
   content slide. Students have coding agents for the syntax; naming the
   structure in their data is the part only the scientist can do — that is
   what the slides train.

## Vocabulary

- **data sample**, not "row" — the course handles 1D sensor series, 2D
  geospatial rasters, and 3D spatiotemporal fields, not just tables. "Row"
  only when the object genuinely is tabular.
- **location**, not "site" (the notebooks' `site_id` columns may be glossed:
  "the code calls locations `site_id`").
- **region**, not bare "space", for spatial extent.
- **event** = a physical process delimited in time (an earthquake, a storm),
  reaching many instruments.
- **data correlation** (samples are not independent), not "rows are not
  independent."
- train / validation / test — the book's triple, always explicit (see the
  book-wide terminology standard).
- Fields in full: geotechnical engineering, atmospheric sciences.
- Statistical methods by concept name: "grouped validation by location,"
  "train on the past, validate on the future," "leave a region out, with a
  buffer" — not API names.

## The literature slide (required, early)

Every deck's introduction includes a **"This lecture in the literature"**
slide: 2–4 publications — at least one where the concept was applied well and
one where its absence misled a field — each with a one-line tag saying which
it is (✓ / ✗). The entries live in a per-deck include file
(`refs/lecNN_refs.qmd`), so updating papers as the field moves means editing
one small file, never the deck. Speaker notes carry the two-minute story of
each paper. New papers land monthly; instructors and students are invited to
propose replacements.

## Figures

- Default: extracted from the executed notebooks via
  `tools/extract_figures.py` (auto-trimmed).
- Multi-panel or small-label figures: re-plot at lecture scale with a
  figscript in `figscripts/` (20pt base font, bold 23–24pt titles).
- **Figure text obeys the vocabulary and concept-first rules too** — panel
  titles describe the design ("Train on the past, validate on the future"),
  never the API call.
- Every data figure carries a **science tag**: named station/region/source
  for real data ("GNSS station P395, Oregon Coast Range — NGL"), an explicit
  "synthetic (mlgeo_synth)" label otherwise. Real data preferred where the
  notebook uses it.
- Figures use `.r-stretch`; nothing overflows; every figure slide has exactly
  one `.takeaway` line.

## Speaker notes

- **Every slide has a `::: {.notes}` block** — what to say, what to define
  aloud, what question to ask the room. A substitute instructor could teach
  from the notes alone.
- Implementation spellings and literature back-stories live in the notes.
- The hand-off slide's notes carry the session timing plan.

## Structure

1. **Title slide**: lecture title, session number, date, book section, one
   emoji icon (`.lecture-icon`).
2. **"Today's question" slide**: the session's single question, few words.
3. **Literature slide** (see above).
4. **Body**: alternate big-number slides (`.big-number` with `.unit` labels)
   and full-bleed figure slides (`.r-stretch` + one `.takeaway` line). When a
   skill score (R²) and a physical-units error (MAE in mm, ppm, m) both
   appear, one `.dim` line states what each answers; prefer physical units
   wherever the audience should feel the number. Terms defined on the slide
   of first use, in a `.dim` line, or not used.
5. **Closing pair**: a summary table that stands alone (geoscience example
   column included; no rhetorical reveals that need narration), then the
   **notebook hand-off slide** with numbered in-class tasks — the one place
   API names may appear on screen.

## Word budget

- Big-number slides: ≤ 25 words on screen.
- Bullet slides: ≤ 45 words, ≤ 4 bullets.
- The numbers and figures argue; the prose connects.
