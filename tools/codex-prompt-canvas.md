You are preparing Canvas quizzes for ESS 469/569 "Machine Learning in the
Geosciences", Autumn 2026, University of Washington.

INPUT
The directory tools/quizzes/ contains six question banks I will use
(ch1.md, ch2.md, ch3.md, ch4.md, ch5.md, ch6.md) plus ch7.md (NOT used as
a quiz — skip it) and a README.md describing the format. Each bank is
plain markdown: numbered questions, four options a–d with the correct
option marked by an asterisk inline, and an answer key with one-sentence
rationales at the bottom of the file. Do not change any question content,
options, keys, or rationales — your job is packaging, not editing.

TASK
1. For each of the six banks, produce a text2qti-format source file
   (https://github.com/gpoore/text2qti): strip the answer-key/rationale
   section at the bottom, keep the asterisk-marked correct options
   (that is text2qti's convention), and add a quiz title header. Attach
   each question's one-sentence rationale from the answer key as
   text2qti general feedback for that question, so students see the
   rationale after the quiz closes.
2. Run text2qti (pip install text2qti) on each source file to produce six
   QTI .zip packages. Fix any conversion errors by adjusting formatting
   only, never content. Validate that each package contains the same
   number of questions as its source bank.
3. Produce a file CANVAS_SETUP.md with per-quiz import instructions and
   the exact settings below, so I can import each zip
   (Canvas > Settings > Import Course Content > QTI .zip) and configure
   it in a few clicks.

QUIZ TITLES, DATES, AND SETTINGS
All quizzes: timed 25 minutes, one attempt, shuffle answer options,
question order shuffled, results (score + feedback) released after the
window closes. "Available from" opens 00:00 PT on the open date and
"Until"/due closes 23:59 PT on the close date.

| Canvas quiz title                                | Bank   | Available from | Until (due) |
|--------------------------------------------------|--------|----------------|-------------|
| Quiz 1 — Ch 1: Open Source Ecosystem             | ch1.md | Tue Oct 6      | Thu Oct 8   |
| Quiz 2 — Ch 2: AI-Ready Geoscience Data          | ch2.md | Mon Oct 26     | Wed Oct 28  |
| Quiz 3 — Ch 3: Classic ML (incl. 3.10 reading)   | ch3.md | Tue Nov 10     | Thu Nov 12  |
| Quiz 4 — Ch 6: Agentic AI for Science            | ch6.md | Mon Nov 16     | Wed Nov 18  |
| Quiz 5 — Ch 5: Workflows & Reproducibility       | ch5.md | Mon Nov 30     | Thu Dec 3   |
| Quiz 6 — Ch 4: Deep Learning (incl. forecasting) | ch4.md | Thu Dec 3      | Mon Dec 7   |

All dates are 2026. Note the deliberate ordering: the Chapter 6 quiz runs
in November (the chapter is taught early) and the Chapter 5 quiz closes
BEFORE its in-class discussion on Dec 4 — it enforces flipped reading.
Chapter 7 has no quiz by design.

ALSO CREATE
A file CANVAS_ASSIGNMENTS.md listing the non-quiz graded items to create
as Canvas assignments (submission type noted), so the Canvas calendar is
complete:
- HW1 Workbench Setup — due Mon Oct 12 (website URL submission: repo link)
- Reading Arc Stage 1: AI-Assisted Lit Review + citation-verification log — due Wed Oct 21 (file upload)
- Final Project Proposal — due Fri Oct 30 (file upload)
- Reading Arc Stage 2: Anatomy of Good Papers — due Wed Nov 4 (file upload)
- Reading Arc Stage 3: Your Quality Rubric — due Fri Nov 13 (file upload)
- Project Check-in 1: Data-Audit Studio — in class Mon Nov 16 (no submission; on-paper grade)
- HW-CML — due Fri Nov 20 (file upload, .ipynb)
- Classification Leaderboard — closes Tue Nov 24 (no submission; scored via GitHub PR, grade entered manually)
- HW-DL — due Fri Dec 4 (file upload, .ipynb)
- Project Check-in 2: Dry-Runs + Agent Clinic — in class Mon Dec 7 (no submission)
- Forecasting Leaderboard — closes Wed Dec 9 (no submission; scored via GitHub PR)
- Reading Arc Stage 4: Your Pre-Submission Review Agent — due Thu Dec 10 (file upload: eval report + repo link)
- Final Presentations — Fri Dec 11 in class + finals slot (no submission)
- Final Report + Repository — due Wed Dec 16 (file upload + URL)

ALSO CREATE: PAPER-PULSE PEER-FEEDBACK SURVEYS
From week 3 (Mon Oct 12) through week 10 (Fri Dec 4), most class sessions
open with two 4-minute student "paper-pulse" talks, and every student in
the room completes a short feedback survey per session. Build the survey
materials:

1. One master survey template (as a text2qti-compatible file or plain
   markdown, whichever imports cleanest as a Canvas Classic "Graded
   Survey") with two identical blocks, one per presenter. Each block:
   - Presenter name (short-answer field; I will fill real names per date
     after sign-ups)
   - Five rating items, scale 1–5: (a) the central claim was stated
     clearly for a non-specialist; (b) the supporting evidence (figure or
     number) was identified; (c) the strength and flaw were specific
     (cited a line/figure/number, not a vibe); (d) pacing and delivery;
     (e) the AI-workflow disclosure was present and concrete.
   - Two required essay boxes: "One specific strength" and "One specific
     improvement."
2. Nineteen dated copies, one per pulse session. The pulse dates are
   every class meeting from Oct 12 through Dec 4 EXCEPT Nov 16
   (check-in), Nov 25 and Nov 30 (lab days): that is Oct 12, 14, 16, 19,
   21, 23, 26, 28, 30; Nov 2, 4, 6, 9, 13, 18, 20, 23; Dec 2, 4. Name
   each "Pulse Feedback — <Day> <Date>". If a CANVAS_API_TOKEN is
   available in the environment, create them directly via the Canvas
   API as graded surveys (complete/incomplete, 0 points toward a
   "Participation" assignment group, available only on their date
   10:00–23:59 PT); otherwise produce the 19 import files plus a
   DUPLICATION.md with the per-date settings so I can import them in
   bulk.
3. Settings for every survey: anonymous responses ON where Canvas
   supports it for graded surveys (if the instance forces named
   responses, note in DUPLICATION.md that anonymization happens at
   export — I strip names before forwarding feedback to presenters);
   one submission per student; completion counts toward participation.
4. A sign-up roster CSV (pulse_signup.csv): columns date, slot (1 or 2),
   student_name (blank) — 38 rows from the 19 dates. Plus a short Canvas
   announcement text (announce_pulse.md) explaining the talk format
   (claim / evidence / strength+flaw vs the rubric / AI-disclosure
   sentence, 4 minutes + 1 question), that sign-ups open Mon Oct 5, and
   that survey completion counts as participation.
5. The same master template, retitled "Final Presentation Feedback", as
   two additional dated surveys for Fri Dec 11 and the finals-week
   presentation slot (date TBD — leave a placeholder).

CONSTRAINTS
- Never publish, upload, or share the bank contents anywhere; these files
  contain answer keys and are deliberately kept out of the public course
  repository.
- Work only inside tools/quizzes/ and a new tools/quizzes/canvas_export/
  output directory; do not commit anything to git.
- End by printing a checklist of what you produced and a per-quiz
  question count so I can verify against the sources (expected: ch1=10,
  ch2=12, ch3=12, ch4=13, ch5=12, ch6=14), plus the survey count
  (expected: 19 pulse surveys + 2 final-presentation surveys + roster +
  announcement).
