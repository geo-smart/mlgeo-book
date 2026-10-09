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

## The literature slides (required, early)

Every deck's introduction carries two literature slides, both from the
geosciences.

1. **One paper told in full.** Pick the paper where the lecture's concept
   changed a geoscience result. Stage it with: a citation card stating the
   finding in one sentence; a map or picture of the geoscience domain; a few
   real records from the data table (data samples as rows, the key columns,
   including the metadata columns that matter); and the figure where the
   problem shows. Rebuild every visual from the open data behind the paper with
   a figscript; never paste screenshots or figures from the publisher (the repo
   is CC BY and publisher figures are not). If today's data no longer
   reproduce the published number, say so on the slide and in the notes.
   Example: lecture 5, sea surface temperatures in the 1940s
   (figscripts/sst_1945_discontinuity.py).
2. **Breadth: the same problem elsewhere.** Three or four cards, each a
   number or a short term first, one sentence, then the citation, drawn from
   different Earth systems where possible. The cards live in the per-deck
   include file (`refs/lecNN_refs.qmd`), so updating papers means editing one
   small file, never the deck.

Check every citation (DOI resolves, title and authors match) before it goes on
a slide, and check every number against the paper or a rerun. Speaker notes
carry the full story of each paper.

## Titles, takeaways, and code on slides

- **No slogans.** A title names what the slide shows ("Zeros in three
  columns"), not a punchline ("A zero is not one thing"). A takeaway states
  one specific finding, with its number where there is one. No "not X, it's
  Y" constructions, no one-word fragments, no em-dashes.
- **Short code is welcome** where it shows the operation on the data: two to
  four lines, copied from the session's notebook so it runs, on a content
  slide after the geoscience and the concept. Longer code stays in the
  notebook.

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
