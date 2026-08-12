# Instructor notes: leaderboard and hidden test sets

Not rendered in the book. This file documents how to run the fair-evaluation
infrastructure each year.

## Public leaderboard

Students submit via PR:
- `results/predictions_<uwnetid>.csv` (`row_id,prediction`) — Chapter 3.5 track,
  scored with macro-F1 against the canonical split (seed 2026, stratified 25%
  test) of the Zenodo seismic-source dataset.
- `results/forecast_<uwnetid>.csv` — Chapter 4.10 **diagnostic** track, scored
  with MASE against `leaderboard/co2_holdout.csv`, scaled by the naive one-step
  MAE of `leaderboard/co2_train_tail.csv` (textbook MASE: training-series
  denominator, not the holdout's own naive error).
- `results/forecast_hidden_<uwnetid>.csv` — Chapter 4.10 **graded** track,
  scored with MASE against `leaderboard/private/synth_forecast_truth.csv`,
  scaled by the naive one-step MAE of the committed
  `leaderboard/synth_forecast_history.csv`.

`.github/workflows/leaderboard.yaml` scores each PR (job summary) and updates
`book/leaderboard_standings.md` on merge to main. `results/` is gitignored for
local runs but tracked submissions arrive through PRs — keep the directory with
a `.gitkeep`.

## Hidden test sets

The public split protects against nothing once students have seen it all term.
Hold a hidden set and score it at the end:

1. **Classification track**: the Zenodo dataset is fixed, so the hidden set is a
   *different* stratified split: rebuild with `random_state=<private seed>` and
   score submissions' MODELS (ask students to submit a predict script or rerun
   their repo), or simply compare their claimed CV numbers to your rerun.
2. **Forecast diagnostic track (CO2)**: the holdout is public NOAA data cached
   on every student's disk — copying `leaderboard/co2_holdout.csv` into a
   submission scores MASE = 0.0. That exploit is deliberate pedagogy (4.10 says
   so in print): the public score is a diagnostic and carries **no grading
   weight**. Do not try to harden it; the hidden track below is the grade.
3. **Forecast graded track (hidden synthetic series)**: students are given
   `leaderboard/synth_forecast_history.csv` (committed) and forecast the next
   90 days. The truth lives in `leaderboard/private/` (gitignored) and exists
   only on your machine, so public CI renders the section as "pending" and
   your local `pixi run python leaderboard/score.py` produces the real table.
   Regenerate each year with a fresh private seed:

   ```bash
   SEED=<private>  # store outside the repo (e.g. leaderboard/private/SEED.txt)
   pixi run python - <<EOF
   import pandas as pd, mlgeo_synth
   df = mlgeo_synth.gnss_series(n_years=8.0, seed=$SEED)
   hist, hold = df.iloc[:-90], df.iloc[-90:]
   for part, path in [(hist, "leaderboard/synth_forecast_history.csv"),
                      (hold, "leaderboard/private/synth_forecast_truth.csv")]:
       pd.DataFrame({"date": part["date"].dt.strftime("%Y-%m-%d"),
                     "disp_mm": part["disp_mm"].round(3)}).to_csv(path, index=False)
   EOF
   ```

   Commit only the history file. Sanity checks before week 1: the truth file is
   NOT in `git status`, and a persistence submission (repeat the last history
   value 90 times) scores a plausible MASE (~1-3 depending on seed).
4. **Other synthetic tracks** (homework, Ch4 labs): regenerate with a private
   seed, e.g. `mlgeo_synth.geochem_table(n=10000, seed=<private>)`. Any
   generator in `mlgeo_synth` accepts a seed; the ground-truth columns make
   scoring free.

Report both public and hidden scores to the class: the gap is the lesson.

## Yearly rotation checklist

- [ ] Pick new private seeds; store them outside the repo.
- [ ] Refresh `leaderboard/co2_holdout.csv` horizon AND
      `leaderboard/co2_train_tail.csv` (last 120 pre-holdout months) to match
      the holdout defined in 4.10 Section 1.2.
- [ ] Regenerate the hidden forecast track (recipe above) with the new seed;
      commit the new history file, keep the truth in `leaderboard/private/`.
- [ ] Clear `results/` on a new-year branch.
- [ ] Verify `pixi run python leaderboard/score.py` runs clean before week 1.
