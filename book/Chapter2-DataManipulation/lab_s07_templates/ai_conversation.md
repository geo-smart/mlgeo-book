# AI conversation: <date>, <tool> (model: <name>), Session 7, <track>

<!--
Session 7, submission item (d). Copy this file to s07_my_files/ before you edit it.
A PDF of the conversation is also accepted (ai_conversation.pdf); then submit this
file too, with the two tables and "What I checked and corrected" filled in.

What you do: give an AI tool your reproducibility_rules.md, your
pipeline_instructions.md and your track's real data, ask it to follow your instructions
from the raw file to the derived quantity, then check its answer against your rules
and against the real-data values that Part 8 of your notebook prints. This is a
separate conversation from the twin test in class, which stays in your notebook. Use UW Purple, or
Claude or GitHub Copilot in VS Code. The instructions for each are below; delete
them before you submit.
-->

Files given to the tool: <names and sizes>
Request (paste it exactly):

> Attached are my reproducibility rules (reproducibility_rules.md), my instructions for the data chain (pipeline_instructions.md) and the data. Follow the instructions step by step under every rule. Show all code. At the end, list each rule and say whether you followed it, and list what you could not do.

## Transcript

<the full conversation, exported or pasted>

## Did the tool follow my rules?

| Rule (heading and first words) | Followed? (yes / no / could not) | Evidence: a line of the transcript or of its output |
|---|---|---|
| | | |

## Did it get my numbers?

| Quantity (real data) | My notebook (Part 8) | The tool | Same? If not, why |
|---|---|---|---|
| Rows or samples in the raw file | | | |
| Derived quantity, file as read | | | |
| Derived quantity, after cleaning | | | |

## What I checked and corrected

- Accepted: <what you kept as is>
- Changed: <what you edited, and why>
- Rejected: <what you threw out, and why; for example a real extreme it removed, a problem it invented, a number it gave without code>
- Verified: <how you checked what you kept: reran its code in your notebook copy, compared with a value printed in the notebook, read the provider's documentation>
- Changed in my rules or instructions because of this: <one rule or step you rewrote, or "none" with the reason>

---

## How to run it (delete before you submit)

### With UW Purple

Sign in at [purple.uw.edu](https://purple.uw.edu/) with your NetID, as in HW1 Step 8c. Purple reads text files (`.md`, `.csv`, `.txt`). It cannot run code, cannot download files, and cannot read binary formats (MiniSEED, NetCDF, FITS). Three consequences:

1. Upload `reproducibility_rules.md`, `pipeline_instructions.md` and the text files listed below for your track. Add one sentence to the request: "You cannot run code or download: say so at each step that needs it, write the code, and compute only what you can read from the attached files."
2. Expect it to report steps 2 and 3 of `pipeline_instructions.md` as not done. That is a correct answer, not a failure of your instructions.
3. To check its code, run it in a new cell of your notebook copy and paste the output back into the conversation. Record that you did.

Text files per track (run the code in a new cell of your notebook copy; it writes to `s07_my_files/`):

- **Volcano.** `snapshots/comcat_msh_2004-2008.csv` is already text. For the seismogram, upload a one-second summary of the HSR record (minimum, maximum and mean count, number of samples; about 0.9 MB):

  ```python
  from pathlib import Path
  import pandas as pd
  from obspy import read
  Path("s07_my_files").mkdir(exist_ok=True)
  parts = []
  for tr in read("snapshots/msh_UW.HSR.EHZ_2004-11-29T15-21.mseed"):
      t = pd.Timestamp(tr.stats.starttime.datetime) + pd.to_timedelta(tr.times(), unit="s")
      parts.append(pd.Series(tr.data, index=t))
  g = pd.concat(parts).sort_index().resample("1s")
  pd.DataFrame({"min": g.min(), "max": g.max(), "mean": g.mean().round(1),
                "n_samples": g.count()}).to_csv("s07_my_files/HSR_EHZ_2004-11-29_1s.csv")
  ```

- **Soil moisture.** The course copy is already text: upload `snapshots/uscrn_WA_Spokane_17_SSW_2015-2024.txt` with `snapshots/uscrn_daily01_HEADERS.txt` and `snapshots/uscrn_daily01_readme.txt`. If Purple refuses `.txt`, upload the CSV that Part 8 of your notebook writes to `s07_my_files/`.
- **Land-surface model.** Upload `snapshots/cpc_soilw_precl_spokane_cell_1991-2020.csv` (one cell, 19 KB). It gives the drought frequency of that cell only; say so in the request. For the whole box (992 land cells, about 13 MB, which Purple may refuse):

  ```python
  from pathlib import Path
  import xarray as xr
  Path("s07_my_files").mkdir(exist_ok=True)
  ds = xr.open_dataset("snapshots/cpc_soilw_precl_westUS_1991-2020.nc")
  df = ds[["soilw", "precip"]].to_dataframe().dropna(how="all").reset_index()
  df.to_csv("s07_my_files/cpc_westUS_1991-2020.csv", index=False, float_format="%.2f")
  ```

- **Planetary.** `snapshots/nea_pscomppars_2026-10-14.csv` is already text. For the light curve (about 1.2 MB):

  ```python
  from pathlib import Path
  import lab_s07_planetary_checks as c
  Path("s07_my_files").mkdir(exist_ok=True)
  lc, _ = c.read_lightcurve()
  cols = ["TIME", "SAP_FLUX", "SAP_BKG", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "QUALITY"]
  lc[cols].to_csv("s07_my_files/HD209458_tess_s56_lightcurve.csv", index=False)
  ```

Saving the conversation: use Purple's export or share option if your version shows one. Otherwise select the whole conversation, copy it and paste it under "Transcript", or print the page to PDF from the browser (File, Print, Save as PDF) and submit `ai_conversation.pdf`.

### With Claude or GitHub Copilot in VS Code

These tools can run code and download files, so every step can run. Open your `mlgeo-book` folder in VS Code and start the tool there. Add three lines to the request:

> Run all code with `pixi run python`. Write files only in `book/Chapter2-DataManipulation/s07_my_files/`. Do not edit, commit, push or stash anything in this repository.

After it finishes, `git status` should list no changed book file. If it does, restore the file with `git checkout -- <file>` and record that in "Rejected".

Check the commands it ran, not only its summary: does each command it says it ran appear in the record, and do its fingerprints match yours?

Saving the conversation:
- **Claude Code** (in the VS Code terminal or panel): type `/export` and save the file, as in HW1. If your version has no `/export`, copy the conversation from the panel.
- **GitHub Copilot Chat**: right-click in the chat panel and choose Copy All, then paste under "Transcript". The Command Palette command "Chat: Export Chat..." saves the same conversation as a JSON file, which you may submit as well.

Read the transcript before you submit it: remove access keys, passwords and personal details.
