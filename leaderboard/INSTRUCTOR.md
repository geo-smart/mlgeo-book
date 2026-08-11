# Instructor notes: leaderboard and hidden test sets

Not rendered in the book. This file documents how to run the fair-evaluation
infrastructure each year.

## Public leaderboard

Students submit via PR:
- `results/predictions_<uwnetid>.csv` (`row_id,prediction`) — Chapter 3.5 track,
  scored with macro-F1 against the canonical split (seed 2026, stratified 25%
  test) of the Zenodo seismic-source dataset.
- `results/forecast_<uwnetid>.csv` — Chapter 4.10 track, scored with MASE
  against `leaderboard/co2_holdout.csv`.

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
2. **Forecast track**: the hidden set is time itself — score against months that
   had not been released when the assignment opened.
3. **Synthetic tracks** (homework, Ch4 labs): regenerate with a private seed,
   e.g. `mlgeo_synth.geochem_table(n=10000, seed=<private>)`. Any generator in
   `mlgeo_synth` accepts a seed; the ground-truth columns make scoring free.

Report both public and hidden scores to the class: the gap is the lesson.

## Yearly rotation checklist

- [ ] Pick new private seeds; store them outside the repo.
- [ ] Refresh `leaderboard/co2_holdout.csv` horizon.
- [ ] Clear `results/` on a new-year branch.
- [ ] Verify `pixi run python leaderboard/score.py` runs clean before week 1.
