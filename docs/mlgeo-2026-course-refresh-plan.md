# MLGeo 2026 course refresh: visual system, figure provenance, and Canvas delivery

**Status:** selective Canvas migration completed; content revision pending

**Course:** ESS 469/569, Machine Learning in the Geosciences

**Audience:** senior undergraduates, graduate students, academic researchers, and applied researchers

**Canonical public book:** <https://geo-smart.github.io/mlgeo-book/>

**New Canvas course:** <https://canvas.uw.edu/courses/1918250>

**Archived reference course:** <https://canvas.uw.edu/courses/1747993>

This document records what was found, what has already been implemented in the
book, and the safest sequence for completing the 2026 Canvas course. It is also
an issue-ready implementation brief for collaborators.

## 1. August 2026 poster and reusable visual layers

### Located source

The poster created in August 2026 is:

```text
/Users/marinedenolle/Downloads/mlgeo-fall-2026-uw-flyer.png
```

The filesystem timestamp is **August 12, 2026 at 13:02:09**. It is a
2550 × 3300 RGB PNG. It is flattened: there are no recoverable text, vector,
image, or adjustment layers, and the file has no embedded Canva, Figma, PSD, or
SVG source reference. The original elements therefore cannot be extracted as
true editable layers from this PNG.

### Reconstructed reusable assets

Two clean, text-free backgrounds were reconstructed from the poster's visual
direction and added to the book:

| Asset | Intended use |
|---|---|
| `book/img/brand/mlgeo-scientific-landscape-hero.png` | Book homepage, Canvas course card/banner, slides, wide web panels |
| `book/img/brand/mlgeo-scientific-landscape-portrait.png` | Posters, mobile crops, vertical announcements |

The assets preserve the dark navy scientific landscape, mountains, ocean/ice,
geologic strata, and cyan/gold neural-network motif, but contain no baked-in
course title, QR code, logos, or UI panels. Those elements should remain live
HTML or editable vector content.

The homepage implementation is already present in:

- `book/about_this_book/about_this_book.md` — semantic hero content and calls to action;
- `book/_static/styles.css` — responsive visual treatment and design tokens;
- `book/myst.yml` — site-wide custom stylesheet registration;
- `book/img/brand/README.md` — asset provenance and reuse guidance.

The book builds successfully with this implementation. The live text remains
selectable, searchable, translatable, and usable by assistive technology.

### Layer model for future materials

Because the source flyer is flat, recreate future course materials with this
explicit layer stack:

1. **Background artwork:** one of the text-free PNGs above.
2. **Atmospheric overlay:** a navy-to-transparent gradient to ensure contrast.
3. **Live title and tagline:** HTML, SVG text, or editable presentation text.
4. **Content cards:** translucent navy panels with visible borders.
5. **Technical diagram:** original SVG using the course's diagram conventions.
6. **Institutional marks:** separate official UW/GeoSMART assets with their own
   usage and license records.
7. **Action layer:** live hyperlinks and a freshly generated QR code.

Do not flatten the editable master. Export PNG/PDF derivatives only after saving
the source in Figma, SVG, PPTX, or another layer-preserving format.

## 2. Visual system for textbook schematics and cartoons

### Design principles

The book serves higher education and working researchers, so the diagrams
should feel scientifically serious without becoming sterile. Use the poster's
visual language selectively:

- deep navy (`#070b36`) for structure and high-contrast backgrounds;
- indigo (`#2b1763`) for learned or latent state;
- cyan (`#59d7ff`) for data and information flow;
- gold (`#f7c843`) for objectives, decisions, thresholds, or selected results;
- warm orange for Earth processes, forcing, or energy;
- off-white for labels and neutral geometry.

Use these colors as semantics, not decoration. A model's input should not change
color between figures simply to make a page more varied.

For technical content:

- redraw diagrams as original **SVG**, with live text and view boxes;
- keep code and API identifiers exactly as written (`fit`, `predict`,
  `train_test_split`, `PyTorch`, `scikit-learn`);
- write descriptive alt text and a concise caption that explains the teaching
  purpose, not merely the appearance;
- preserve mathematical notation and show directionality explicitly;
- avoid decorative characters or cartoons when they imply a human role,
  identity, or cultural stereotype;
- test at mobile width, projector size, grayscale, and common forms of color
  vision deficiency;
- place source and license metadata in a machine-readable manifest.

AI-generated raster images are appropriate for atmospheric covers, chapter
openers, and non-technical illustration. They are not the preferred source for
network architectures, signal-processing operations, validation schemes, or
plots that students may interpret quantitatively.

### Current figure inventory and proposed replacements

This is a first-pass inventory of the highest-value schematic/cartoon work. It
does not include ordinary plots generated by notebook code.

| Priority | Chapter/source | Current asset or source | Concern | Proposed original replacement |
|---|---|---|---|---|
| P0 | `Chapter4-DeepLearning/mlgeo_4.1_NN.ipynb` | `img/TLU.png`, adapted from Géron | Third-party textbook composition; central teaching figure | SVG threshold logic unit: inputs, weights, weighted sum, threshold/activation, output; cyan inputs, violet learned weights, gold decision |
| P0 | `Chapter3-MachineLearning/3.9_ensemble_learning.ipynb` | `votingclassifier.png`, attributed to Géron | Third-party instructional diagram | SVG ensemble showing diverse learners, predictions, voting/averaging, final prediction; include hard vs soft voting inset |
| P0 | `Chapter3-MachineLearning/3.8_robust_training.ipynb` | `ValsetApproach.png`, `grid_search_cross_validation.png`, `Kfold.png`, `LOOCV.png` | Mixed sources/styles; some attributed to scikit-learn | A coordinated four-panel validation family showing holdout, K-fold, grouped/spatial split, and nested CV; add leakage warnings relevant to geoscience |
| P0 | `Chapter4-DeepLearning/mlgeo_4.4_RNN.ipynb` | `rnn.svg`, `lstm-2.svg`, D2L-derived | Externally authored architecture diagrams | Original unrolled sequence diagram plus LSTM gate detail, using consistent state/input/output conventions |
| P0 | `Chapter4-DeepLearning/mlgeo_4.3_CNN.ipynb` | external D2L `lenet.svg`; external convolution GIFs | Remote dependency, mixed licensing/provenance, inconsistent style | Local SVG LeNet/CNN architecture and original frame sequence or CSS/SVG animation for convolution; include boundary/padding semantics |
| P0 | `Chapter2-DataManipulation/2.9_filtering_data.ipynb` | `filters.png`, NI tutorial attribution | Third-party tutorial graphic | Generate filter-response plots from code and pair with an original signal-flow schematic; quantitative plots should be reproducible |
| P1 | `Chapter4-DeepLearning/mlgeo_4.2_MLP.ipynb` | `mlps.svg`, `MLPReg.png`, `MLPClass.png` | Mixed visual language | One reusable SVG component system for regression and classification heads |
| P1 | `Chapter4-DeepLearning/mlgeo_4.5_ModelTraining.ipynb` | `GD_cartoon.jpeg`, `GD_non_global.png`, `GD_AlphaTooSmall.png`, `GD_AlphaTooLarge.png` | Cartoon/bitmap style and uncertain reuse history | Reproducible loss-surface figures plus an original SVG optimizer path; distinguish learning-rate behavior without anthropomorphic cartoons |
| P1 | `Chapter4-DeepLearning/mlgeo_4.6_AutoEncoder.ipynb` | `autoen_architecture.png`, `Unet.png` | Bitmap architecture graphics | Coordinated SVG encoder/latent/decoder and U-Net with skip connections; add tensor-shape labels |
| P1 | `Chapter2-DataManipulation/2.2_data_formats_rendered.ipynb` | `hdf5_structure4.jpeg` | Raster schematic; provenance review needed | Original SVG tree with groups, datasets, attributes, and chunking callout |
| P1 | `Chapter2-DataManipulation/2.7_statistical_considerations.ipynb` | `mean.png`, `variance.png`, `skewness.png`, `kurtosis.png` | Separate bitmap illustrations | Generate all four distributions from one documented script; use identical axes and sample size |
| P1 | `Chapter2-DataManipulation/2.8_data_spectral_transforms.ipynb` | `Wavelet-Out1.jpeg`, `wavelet_families.png` | Mixed raster sources | Reproducible time-frequency example and selected wavelet families generated from code |
| P1 | `Chapter3-MachineLearning/3.4_binary_classification.ipynb` | `roc-curve-v2-glassbox.png`, Wikimedia/xkcd-style | Open license but stylistically distinct and potentially distracting | Original ROC/PR conceptual figure generated from a small synthetic geoscience example |
| P2 | Chapter 1 and readmes | externally hosted comics/Turing Way illustrations | Remote availability and mixed visual identity | Retain only when the external work itself is pedagogically important; otherwise summarize in original diagrams and preserve attribution links |

### IP and provenance workflow

Create `book/img/FIGURE_SOURCES.yml` with one record for every non-generated
figure:

```yaml
- id: tlu-threshold-unit
  file: book/img/tlu-threshold-unit.svg
  used_in: book/Chapter4-DeepLearning/mlgeo_4.1_NN.ipynb
  creator: MLGeo contributors
  source_url: null
  license: CC-BY-4.0
  derived_from: null
  status: original
  alt_text_reviewed: true
```

For an adapted figure, record the upstream work, author, URL, exact license,
date accessed, and what was changed. “Found online,” “from tutorial,” or a
caption-only attribution is not enough. An open license reduces permission risk
but does not remove the obligations of attribution, license compatibility, and
indicating modifications.

Add a build check that fails when:

- an image referenced by Markdown or a notebook is missing from the manifest;
- a remote image is embedded without a deliberate exception;
- an instructional image lacks alt text;
- a derivative lacks source and license fields.

### Replacement sequence

1. Establish an SVG template, symbol library, and figure manifest.
2. Replace the six P0 families and obtain instructor scientific review.
3. Replace P1 images chapter by chapter, beginning with figures visible in the
   2026 teaching schedule.
4. Audit P2 external illustrations and retain only those whose specific
   cultural or pedagogical value exceeds the maintenance cost.
5. Add visual-regression thumbnails to pull requests and verify the full MyST
   build before merging.

## 3. Canvas 2026 course architecture

### Findings

Course 1918250 is an unpublished, fresh Autumn 2026 shell. At inspection it had
no modules, announcements, assignments, quizzes, or pages, and an empty syllabus
body. The archived 2024 course contains useful assignments, quizzes/question
banks, rubrics, files, and syllabus language, but also stale dates, personnel,
Slack references, Zoom/calendar events, and obsolete course-navigation choices.

The 2026 book schedule is the canonical calendar:

- Monday/Wednesday in **ECE 003** and Friday in **JHN 175**, **10:00–11:20**;
- instruction September 30–December 11, 2026;
- no class November 11 or November 27;
- final report and repository due December 16.

### Migration strategy

Use Canvas **Copy a Canvas Course → Select specific content**, copying from
1747993 into 1918250. Import only:

- syllabus body as an editable starting point;
- active assignments worth revising;
- quizzes and question banks;
- rubrics;
- selected reusable files.

Do not import:

- calendar events or Zoom meetings;
- announcements;
- obsolete modules or navigation configuration;
- stale pages and discussions;
- unpublished “OLD” assignment variants unless needed as instructor reference.

After import, keep all copied items unpublished until their dates, links,
instructions, accessibility, and 469/569 differentiation have been reviewed.

### Student navigation model

Use **Modules** as the primary student path, with a small persistent set of
navigation destinations:

1. Home
2. Modules
3. Syllabus
4. Announcements
5. Assignments
6. Grades
7. People

Hide unused or duplicative tools from student navigation. Keep Files and Pages
available to instructors but reach student-facing material through Modules.

The Home page should answer four questions immediately:

- What should I do before the next class?
- Where is the textbook/notebook for today?
- Where do I submit work?
- Where do I ask a question?

### Module structure

Create one orientation module and one module per teaching week:

| Module | Dates | Core navigation |
|---|---|---|
| Start Here | Before Sep 30 | Welcome, syllabus, schedule, Gaia, book, computing setup, AI-use policy, accessibility/support, introductions |
| Week 1 — Open, reproducible science | Sep 30–Oct 2 | Ch 1 readings, workbench lab, HW1 |
| Week 2 — Agents, then data | Oct 5–9 | AI policy/mechanism, data formats, pandas, Ch 1 quiz, reading arc launch |
| Week 3 — Signals | Oct 12–16 | arrays, resampling, statistics, spectra, filtering; HW1 due |
| Week 4 — AI-ready data | Oct 19–23 | synthetic data, feature engineering, dimensionality reduction, critical-evaluation lab |
| Week 5 — Classic ML begins | Oct 26–30 | supervision, clustering, binary classification, Ch 2 quiz, project proposal |
| Week 6 — Classification, honestly | Nov 2–6 | multiclass, probability calibration, trees/ensembles, leaderboard, HW-CML |
| Week 7 — Fair evaluation | Nov 9–13 | robust training, agent eval sets, Ch 3 quiz, reading-arc stage 3 |
| Week 8 — Deep learning begins | Nov 16–20 | project check-in, MLPs, CNNs, Ch 6 quiz, HW-CML due |
| Week 9 — Sequence models | Nov 23–25 | RNN/LSTM/attention, model-training lab I, HW-DL |
| Week 10 — Uncertainty and forecasting | Nov 30–Dec 4 | model-training lab II, forecasting, workflow discussion, Ch 4/5 quizzes, HW-DL due |
| Week 11 — Synthesis | Dec 7–11 | dry-run clinic, review agent, buffer, presentations |
| Finals | Dec 12–18 | presentation slot, final report and repository due Dec 16 |

Within each module use the same order: **Overview → Prepare → Attend/participate
→ Practice → Submit → Check understanding**. Label book links by section and
topic rather than “click here.”

### Communication model with Gaia

Replace all 2024 Slack language with:

> **Gaia is our primary course community workspace.** Use it for conceptual and
> technical questions, peer help, shared resources, and course discussion. Use
> Canvas Announcements for official or time-sensitive course notices. Use
> private Canvas message or UW email for grades, accommodations, or personal
> matters. Do not post private student information, credentials, restricted
> data, or unpublished research data in Gaia.

The exact Gaia workspace and course-channel URL must be supplied before this
text can become a live link. Also decide whether students may use direct
messages for course support or should keep all non-private questions in public
course channels.

### Imported assignment mapping

The archived course includes useful starting points: AI-ready dataset, HW1,
Homework 2, CML work, final project components, reading assignments, an open and
reproducible science quiz, and an AI research/writing activity. Map and rename
them to the 2026 canonical schedule rather than retaining 2024 labels.

Recommended 2026 assignment groups:

- **Workbench and homework** — HW1, HW-CML, HW-DL;
- **Reading arc and participation** — four stages, paper-pulse presentation,
  peer feedback completion;
- **Quizzes** — Chapters 1, 2, 3, 4, 5, and 6;
- **Leaderboards/labs** — classification and forecasting;
- **Final project** — proposal, check-ins, presentation, report, repository.

The archived course's overall weights can be used as evidence, but should not
be silently carried forward. Confirm the 2026 weights and whether they differ
between 469 and 569 before configuring Canvas assignment groups. Likewise,
confirm staff names, office hours, grading turnaround, late-work policy, and
the registrar-assigned finals slot.

### Canvas quality-control checklist

Before publishing the course:

- [ ] every module has dates and an overview;
- [ ] all graded items appear once in Modules and once in the Canvas calendar;
- [ ] due dates match `schedule_fall2026.md`;
- [ ] requirements for 469 and 569 are visible at the point of work;
- [ ] assignment groups and weights total 100% for each enrollment path;
- [ ] copied rubrics match the revised instructions;
- [ ] quizzes are unpublished until reviewed and previewed;
- [ ] external links open and are labeled meaningfully;
- [ ] images have alt text and PDFs are selectable/tagged;
- [ ] book, GitHub, Gaia, and submission links are distinct;
- [ ] no 2024 dates, Slack references, old TA names, Zoom events, or room
  numbers remain;
- [ ] Student View can complete the Start Here module end to end;
- [ ] the course remains unpublished until the instructor signs off.

## 4. Decisions required before live Canvas editing

1. Exact Gaia workspace/course-channel URL and preferred channel names.
2. 2026 instructional team, contact methods, and office hours.
3. 2026 assignment-group weights for ESS 469 and ESS 569.
4. Late-work and extension policy.
5. Registrar-assigned final exam/presentation slot.
6. Whether the classification and forecasting leaderboards submit through
   Canvas, GitHub, Gaia, or a separate service.

## 5. Recommended execution order

1. Selectively import reusable 2024 Canvas material, excluding calendar/Zoom
   and announcements.
2. Review imported items in the unpublished shell and archive/delete obsolete
   copies.
3. Create the Start Here and weekly module skeletons.
4. Paste and customize the 2026 syllabus draft.
5. Enter the canonical due dates from the book schedule.
6. Add the Gaia link and communication norms.
7. Configure assignment groups after weights are confirmed.
8. Run link, accessibility, Student View, and date audits.
9. Publish only after instructor review.

The selective import was executed after explicit confirmation from the course
owner. Subsequent syllabus, module, date, grading, and navigation changes remain
pending until their content is reviewed and the open decisions above are
resolved.

## 6. Migration log — September 30, 2026

With the course owner's confirmation, a selective course copy from Canvas course
1747993 into course 1918250 completed successfully. The destination course
remained unpublished.

Imported:

- syllabus body;
- `AI-ready data set`;
- `Homework #1`;
- `Homework 2`;
- `Final Project CML Part`;
- `Final Project Finale`;
- `How to Research and Write Using Generative AI Tools`;
- `Open and Reproducible Science`;
- `Assignment #6: Project Proposal`;
- both archived quizzes (`Open and Reproducible Science` and
  `Feedback on the computing level`);
- the `Unfiled Questions` question bank;
- `Paper_Review_Template.docx` and `template_presentation.pptx`.

Excluded:

- course settings and old modules;
- all announcements and calendar events;
- discussion topics and pages;
- the dated October 11, 2024 activity;
- the old reading series;
- `Homework 2 OLD`, `Homework 3`, `Homework 7`, and the superseded final-project
  assignment;
- 2023 lecture slides, old presentation guidelines, the syllabus screenshot,
  and template-resource folders.

Canvas did not expose a Rubrics category in the source-course selection dialog,
and the destination Course Rubrics page remained empty after import. All eight
imported assignment/quiz items that retained a published state from 2024 were
immediately unpublished for review. Their copied 2024 due dates are still
present and must be replaced with the canonical 2026 dates before publication.

### Date and publication update

The copied dates were subsequently replaced with the 2026 curriculum dates and
the imported graded items were published inside the still-unpublished course:

| Item | Available | Due | Until |
|---|---|---|---|
| Computing-level survey | Sep 30, 8:00 a.m. | Oct 2, 11:59 p.m. | Oct 5, 11:59 p.m. |
| Generative-AI research activity | Sep 30, 8:00 a.m. | Oct 4, 11:59 p.m. | Oct 5, 11:59 p.m. |
| Open and Reproducible Science quiz | Oct 6 | Oct 8, 11:59 p.m. | Oct 8, 11:59 p.m. |
| Homework #1 | Sep 30, 8:00 a.m. | Oct 12, 11:59 p.m. | Oct 14, 11:59 p.m. |
| AI-ready data set | Oct 19 | Oct 25, 11:59 p.m. | Oct 28, 11:59 p.m. |
| Project proposal | Oct 12 | Oct 30, 11:59 p.m. | Nov 2, 11:59 p.m. |
| Final Project CML Part | Nov 2 | Nov 16, 10:00 a.m. | Nov 20, 11:59 p.m. |
| Homework 2 / HW-CML | Nov 4 | Nov 20, 11:59 p.m. | Nov 23, 11:59 p.m. |
| Final Project Finale | Dec 7 | Dec 16, 11:59 p.m. | Dec 18, 11:59 p.m. |

The course itself remains unpublished. Course-card image upload and the branded
homepage are pending browser file-upload permission.
