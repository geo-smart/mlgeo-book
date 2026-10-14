# Reproducibility rules for <your track or project data>

<!--
Session 7, submission item (b). Copy this file to s07_my_files/ before you edit it
(that folder is ignored by git, so `git pull` never collides with your work).

What this file is: the rules an AI tool must follow when it works on your data. They
are your own reproducibility rubric, written as instructions. Start from the six
headings of the AGENTS.md you wrote for HW1 (book 1.9, Step 7) and from the class
rubric (0 missing, 1 present but incomplete, 2 another team can verify it).

Two tests for every rule, the same as in 1.9:
  1. Can someone settle it yes or no by reading the record of the conversation or
     the files the tool produced?
  2. Does it protect a rerun from the raw file?
"Be careful with outliers" fails test 1. "Never delete a value flagged as a real
extreme; list each one with its date" passes both.

Keep or rewrite the nine given rules and add at least five of your own: one under
each rule heading, three of them specific to your track's data (see the hints at the
end). Delete these comments and the hints before you submit.
-->

## Project

<One short paragraph: the system, the question, the raw data and their provider, and
the derived quantity this file protects. Example from a system outside the four tracks
(book 1.7): "Daily positions of the GNSS station P395 from the Nevada Geodetic
Laboratory (NGL). The quantity is the east velocity in mm/yr from a straight-line fit.
Reproducible here means that the same raw file and these rules give the same velocity
to 0.1 mm/yr.">

## Environment and compute

- Run every step as code that you show in full; do not report a number you did not compute with code shown in the conversation.
- <your rule>

## Data and provenance

- Use the raw file from the provider, never a copy edited by hand; state its URL or query, the retrieval date and time (UTC), its size in bytes and its SHA-256 fingerprint.
- Take values that the provider publishes (coordinates, units, missing-value codes, flag meanings) from the provider's documentation and quote the sentence; do not infer them from the data.
- <your rule>

## Code and workflow

- Never overwrite the raw file; write every cleaned or selected table to a new file with a different name.
- For every value changed, removed or flagged, report the column, the first and last date (or row), the number of values, and the physical reason.
- <your rule>

## Reporting and credit

- State the unit, the time window and the number of values behind every reported number.
- Report the derived quantity twice: on the file as given and after cleaning.
- <your rule>

## Verification

- Before reporting, recompute the row count (or sample count) of the raw file and compare it with the expected count (for example the number of calendar days) or with the course copy; say whether they agree.
- If a step cannot be done (no network, a format that cannot be read, a missing column), say so and stop that step; do not substitute an estimate.
- <your rule>

---

## Track hints (delete before you submit)

Pick the hints for your track and turn them into rules in your own words. Each hint names something you met in the notebook.

**Volcano (Mount St Helens, ComCat catalog and HSR seismogram)**
- The seismogram is in raw counts. A rule on when, and with which instrument response, counts become ground velocity: the response must be the one valid on 29 November 2004 (the channel metadata show a gain change on 5 October 2004).
- HSR records clip at +3582 and −3584 counts. A rule that marks those samples as clipped and keeps their amplitudes out of any magnitude or amplitude statistic.
- Large earthquakes are real extremes, not errors (the largest in the catalog is Md 3.9 on 2004-10-01). A rule that forbids removing events by magnitude alone.
- Every magnitude carries its type (`magType`); a rule that never mixes types without saying so.
- The catalog is revised by the provider (`updated` column). A rule that reports the latest `updated` date in the file you used.

**Soil moisture (USCRN, Turnbull)**
- Missing values are coded −99.000 in the soil moisture columns and −9999.0 in every other column (temperature, precipitation, solar radiation, humidity) of the daily file. A rule that converts them to missing before any statistic and reports how many there were in each column.
- The probes cannot measure water in frozen soil, so frozen days are blank. A rule that forbids filling them.
- Precipitation includes snow, which does not reach the soil the same day. A rule about which days enter the rain response (for example, soil temperature at 5 cm above 0 °C that day and the day before).
- A rule on one-day shifts when joining precipitation and soil tables: dates are Local Standard Time; state the time zone of every column you join.

**Land-surface model (CPC soil water and PREC/L precipitation, PSL)**
- Longitudes are 0 to 360 east and latitudes are stored north to south. A rule that states the convention of every grid it writes.
- Soil water is in mm in a one-layer column with a holding capacity of 760 mm. A rule that flags any value outside 0 to 760 mm and any block of months that looks like metres instead of mm.
- Coastal cells with little land behave like ocean (PSL fills ocean with zeros). A rule that uses the land-percent file to say which cells are kept and why.
- The drought threshold (20th percentile of each cell and calendar month) is computed on the baseline years only. A rule that names the baseline years and forbids using the evaluation years in it.
- The model has no snowpack and is driven by gauges whose number changes through time. A rule that the report says so when it interprets winter values.

**Planetary (TESS light curve of HD 209458 and the NASA Exoplanet Archive tables)**
- Name the flux column used (`PDCSAP_FLUX` or `SAP_FLUX`) and why; a rule that reports which `QUALITY` bits were removed and how many cadences each removed.
- Transits and stellar flares are real. A rule that forbids symmetric sigma-clipping through the transits, and that keeps the transit cadences in the depth.
- Units differ between tables (transit depth in % in one, ppm in the other; radii in Earth or Jupiter radii). A rule that states the unit of every column used.
- Masses in the planet table can come from a mass-radius relation rather than a measurement (`pl_bmassprov`). A rule that keeps only measured masses for any mass-radius analysis.
- A planet can appear more than once with different stellar radii. A rule on which row is kept, with the reference it comes from.
