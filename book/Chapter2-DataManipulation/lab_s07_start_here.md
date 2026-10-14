---
title: "Session 7: explore your system's data"
---

**Wednesday 14 October 2026.** Today is self-guided: the instructor is away and there is no TA. You work alone or with your project team through one notebook, in the track that matches your project's domain. Each question has a self-check that answers at once, and a worked solution, collapsed, to open after you have tried.

## Learning outcomes

After this session you will be able to:

- tell whether a data file is the one the provider served, unmodified and complete, from its size, its number of records, its first and last records and its fingerprint;
- say what an instrument or a model actually samples, and which processing stands between the sensor and the number in the file;
- find problems in a record, decide for each whether to flag, remove or keep the values, and measure how each problem biases a quantity your field cares about, including the damage that cleaning cannot undo;
- write rules and instructions that let another person, or an AI tool, rerun your data chain from the raw file, and check whether it did.

## Objectives for today

In this session you will:

- pull your track's raw data from the provider and compare it with the course copy;
- read one table, one time series and one 2D array of your system: units, missing values, sampling, gaps, one real extreme against one error;
- hunt the problems planted in a synthetic twin of your data, clean them, and compute your track's derived quantity three ways (truth, as given, after cleaning);
- test one correlation against the physics, and write the chain of steps from raw file to a learning problem;
- start the two agent skill files you submit on Friday.

Each notebook states the same outcomes and objectives for its own system.

## Choose your track

Take the track of your project's domain. If your domain is not one of these four, take the track whose data look most like yours: a seismometer or an event catalog, volcano; a station time series, soil moisture; a model or reanalysis grid, land-surface model; a telescope or images, planetary. A team may split across two tracks and compare on Friday. The soil moisture and land-surface tracks look at the same water from two sides: one station that measures it, and a model that computes it on a grid.

| Track and notebook | Guiding question | Data (provider) | Labels ([1.7](../Chapter1-GettingStarted/1.7_get_geodetic_gnss.ipynb)) |
|---|---|---|---|
| **Volcano**: Mount St Helens, 2004 to 2008 ([lab_s07_volcano](lab_s07_volcano.ipynb)) | How regular was the drumbeat clock under the growing lava dome, and how much of that clock do a single seismometer near the vent and the regional catalog each record? | Earthquake catalog around the volcano, 2004 to 2008, 11,305 events (U.S. Geological Survey (USGS) Comprehensive Catalog, ComCat). Six hours of the seismometer HSR, 2.5 km from the vent, on 29 November 2004, 100 samples per second in raw counts, with its instrument response (EarthScope). | 📍 In-situ sensor · 🪨 Solid Earth · 🌋 Volcanology; the catalog is 🗂️ Catalog & compiled product |
| **Soil moisture**: water in the top metre of soil at Turnbull ([lab_s07_soil_moisture](lab_s07_soil_moisture.ipynb)) | When it rains at Turnbull, how much of the water does the top of the soil take up, how does that water move down through the profile, and how does this change through the year? | Daily soil moisture at 5, 10, 20, 50 and 100 cm, precipitation and temperature at the U.S. Climate Reference Network (USCRN) station WA Spokane 17 SSW, Turnbull National Wildlife Refuge, 2015 to 2024, 3,653 days (NOAA National Centers for Environmental Information, NCEI). | 📍 In-situ sensor · 💧 Land water · 💧 Hydrology & land-surface modeling |
| **Land-surface model**: soil water in a model of the western United States ([lab_s07_land_surface_model](lab_s07_land_surface_model.ipynb)) | When and where does soil water in the western United States fall into drought, and how much of that does precipitation alone explain? | Monthly soil water from the one-layer model of the Climate Prediction Center (CPC) and the gauge precipitation that drives it, on a 0.5° grid from 31 to 49 N and 125 to 109 W, 1991 to 2020 (NOAA Physical Sciences Laboratory, PSL). | 🖥️ Model & simulation · 💧 Land water · 💧 Hydrology & land-surface modeling |
| **Planetary science**: planets around other stars ([lab_s07_planetary](lab_s07_planetary.ipynb)) | How large are the planets around other stars, and which patterns in planet tables come from the planets themselves and which from what telescopes can detect? | The table of 6,445 confirmed planets and the table of 8,148 TESS Objects of Interest (NASA Exoplanet Archive). The 2-minute light curve of the star HD 209458 from the Transiting Exoplanet Survey Satellite (TESS), sector 56, September 2022, and the pixels around one transit (Mikulski Archive for Space Telescopes, MAST). | 🔭 Telescope · 🪐 Planets & exoplanets · 🪐 Planetary science & astrobiology; the planet tables are 🗂️ Catalog & compiled product |

All four tracks have the same parts in the same order, so teammates on different tracks can compare part by part.

## Set up before you start (about 5 minutes)

Nothing today touches your HW1 repository `MLGEO2026_UWNETID` (due tonight, 11:59 pm) or changes the pixi environment.

1. Update your clone of the course book:

        cd mlgeo-book
        git pull

2. Check that the course data on your disk are the files the notebooks expect:

        pixi run python book/Chapter2-DataManipulation/snapshots/check_snapshots.py

   The script recomputes the size and the SHA-256 fingerprint of each course data file and compares them with the values recorded at download. The last line should read `All 26 data files match their recorded fingerprints: none was changed or damaged.`

3. Copy your track's notebook to a personal copy with your NetID in its name, in the same folder. Replace `TRACK` with `volcano`, `soil_moisture`, `land_surface_model` or `planetary`, and `UWNETID` with your NetID:

        cp book/Chapter2-DataManipulation/lab_s07_TRACK.ipynb book/Chapter2-DataManipulation/lab_s07_TRACK_UWNETID.ipynb

   Work only in your copy. Git ignores it, and the folder `s07_my_files/` where your downloads and files go, so a later `git pull` never collides with your work.

4. Start Jupyter from `mlgeo-book` and open your copy from `book/Chapter2-DataManipulation`:

        pixi run jupyter lab

   Set `NETID = "..."` (`MY_NETID` on the planetary track) in the first code cell to your NetID and run the cell. Your NetID chooses the problems planted in your synthetic twin, so your set differs from your teammates'.

Do not commit, push or stash anything in the book clone.

## The parts

| Part | What you do | Rough time (min) |
|---|---|---|
| 0 | The system and the question: the physical process, a map, the guiding question | 3 |
| 1 | Pull the raw data; check your file against the course copy; native resolution and processing | 10 |
| 2 | A table: what a row records, units, missing values and flags, limits, one real extreme and one error | 8 |
| 3 | A 1D array: a time series, its sampling, gaps and one real sensor problem | 8 |
| 4 | A 2D array: axes, coordinates, resolution, the structure the physics predicts (*if you have time*) | 8 |
| 5 | A synthetic twin with planted problems: list, hunt, clean, measure the damage, reflect | 15 |
| 6 | Correlations on the real data, with the trap that matters in your system | 8 |
| 7 | From data to a learning problem: the chain of steps, the task and target, balancing | 10 |
| 8 | A short comparison with an AI tool on your twin, and a start on the Friday files (*if you have time*) | 7 |
| Wrap-up | Three written answers, then the summary cell | 3 |

The times are rough estimates of how long each part takes, not a schedule. Parts 4 and 8 are optional; skip them if you want more time for Part 5.

**How a question works.** Replace the empty answer (`None`, `...` or `""`) in an answer cell and run it. The self-check prints `correct`, a hint (never the answer), or `not answered yet`. The collapsed cell below it, "Worked solution, open after you tried", shows one way to the answer.

**Downloads.** If you have no network or the provider does not answer, the cell says so and you continue with the course copy, which holds the same data. PSL's subset service refused every request while the lab was built (HTTP error 429, too many requests), so the land-surface notebook downloads the whole files in the background (about 545 MB, about 18 minutes) while you work on the course copy.

**Synthetic data.** The twin in Part 5 is built to look like your real data, with a known truth. Keep it apart from the real record, and say so whenever you quote a number from it.

**AI assistance.** Two labels from [1.8](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md) mark the steps where the mode matters: 🔒 *By hand* (no AI tool for the judgment in that step) and 🔍 *Agent, then verify* (Part 8). A step without a label allows an AI tool.

## Submit by Friday 16 October, 9:00

Upload to the Canvas assignment **"Session 7: explore your system's data"**. Each person submits, also when the team worked together. Completion only: the work does not count toward your grade, but a missing submission costs 10 points.

| | What | How |
|---|---|---|
| (a) | Your notebook copy, `lab_s07_TRACK_UWNETID.ipynb` | Kernel, Restart Kernel and Run All Cells, then save. The last cell prints a block headed `SESSION 7 SUMMARY`; check that it shows your NetID. An HTML export is optional. |
| (b) | `reproducibility_rules.md` | Your own reproducibility rubric, written as rules an AI tool must follow when it works on your track's data. Build on the six AGENTS.md headings from HW1 ([1.9](../Chapter1-GettingStarted/1.9_workbench_setup_hw1.md), Step 7) and the class rubric. Each rule must be checkable yes or no. |
| (c) | `pipeline_instructions.md` | Instructions for an AI tool to run your track's chain from the raw file to the derived quantity: get the raw file from the provider, confirm it was not modified or corrupted, clean and select with a reason for each step (flag, remove, or keep a real extreme), compute the derived quantity, and report what it could not do. |
| (d) | `ai_conversation.md`; the transcript may be a PDF beside it | The record of giving (b), (c) and the real data to UW Purple, or to Claude or GitHub Copilot in VS Code, and what you checked and corrected. Purple cannot run code or read binary files: upload the two files and the data as text (the template names the file for your track). This is a separate conversation from the twin test of Part 8. |
| (e) | A Canvas comment | Two or three sentences on that interaction: what the tool got right, what it got wrong, and what you changed because of it. |

Templates for (b), (c) and (d), with hints for each track and the code that writes text copies of the data for Purple, are in your clone, in `book/Chapter2-DataManipulation/lab_s07_templates/`. Copy them into `book/Chapter2-DataManipulation/s07_my_files/` and fill them in there:

    mkdir -p book/Chapter2-DataManipulation/s07_my_files
    cp book/Chapter2-DataManipulation/lab_s07_templates/*.md book/Chapter2-DataManipulation/s07_my_files/

Write them like a methods section: short and specific. Cut any rule that cannot be checked and any step without a reason.

## When you are stuck

1. Read the "If you are stuck" box at the end of the part: it lists the two most likely problems and their fixes.
2. Ask a teammate.
3. Post on the course discussion board on Canvas: your track, the part and question, and the last lines of the error message.

```{admonition} If you are stuck during setup
:class: tip
- *`git pull` refuses because a book file has local changes:* you edited a book file instead of your copy. Copy it to a new name (for example add `_mine` before `.ipynb`), restore the original with `git checkout -- <that file>`, then run `git pull` again. Do not use `git stash`.
- *`ModuleNotFoundError` or `FileNotFoundError` in the first cell:* your copy is not in `book/Chapter2-DataManipulation`, or Jupyter was not started with `pixi run jupyter lab` from `mlgeo-book`. Move the copy to that folder, restart the kernel and run the first cell again.
- *You have no clone of the course book, or it is broken:* the notebooks need the files next to them (the course data in `snapshots/` and the `lab_s07_*.py` helpers), so a notebook downloaded alone from this website will not run. Make a fresh clone in a new folder and set it up as in HW1 step 4: `git clone --depth 1 https://github.com/geo-smart/mlgeo-book.git mlgeo-book-s07`, then `cd mlgeo-book-s07` and `pixi install`. The course data are also browsable on GitHub: <https://github.com/geo-smart/mlgeo-book/tree/main/book/Chapter2-DataManipulation/snapshots>.
- *The data check prints a line with `FAIL`:* a course data file on your disk differs from the recorded one. `git status` shows which; `git checkout -- book/Chapter2-DataManipulation/snapshots/` restores the course copies.
```
