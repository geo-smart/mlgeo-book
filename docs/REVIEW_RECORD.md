# Translation review record

Who actually read the translated editions, what they reviewed, and what they
decided. This file exists because "persona-reviewed" is not a quality claim.

The personas in [`translations/personas/`](../translations/personas/) are
**synthetic** — written to steer the translation, not to certify it. They did
useful work: they caught a wrong eligibility rule for Mexico's LANCAD compute
allocations, an overstatement about French mésocentres, an ice-cap claim about
a volcano whose glaciers are gone, and a superlative about the Brest tide-gauge
series. But an AI reading a fictional reader's brief is not a francophone or
hispanophone community accepting the terminology. Only the table below counts
as review.

## Reviews completed

*None yet.* The editions were published for community review on 2026-08-13.

| Reviewer | Language variety / region | Discipline & career stage | Chapters | Date | Source commit | Outcome |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

## How to record a review

Copy the block below into `## Reviews completed` as a new entry, or open a pull
request adding it. Reviewers may use a declared anonymous ID instead of a name.

```markdown
### <Name or anonymous ID>
- **Language variety / region:** e.g. French (France, Rhône-Alpes) / Spanish (México, centro)
- **Discipline and career stage:** e.g. seismology, maître de conférences
- **Chapters reviewed:** e.g. fr Chapters 1–3, glossary
- **Date / source commit:** 2026-09-01 / `abc1234`
- **Contested terms accepted:** …
- **Contested terms rejected, and why:** …
- **Conflicts of interest:** e.g. none / co-author of a cited dataset
```

## Panels we are seeking

The audit in [`ai-logs/2026-08-13-codex-language-audit.md`](ai-logs/2026-08-13-codex-language-audit.md)
recommends these, and we agree:

- **French** — one ML/computer-science lecturer, one geoscience lecturer, one
  research-software or HPC practitioner, one graduate student.
- **Spanish** — at least four regions (México/Central America, Andean region,
  Southern Cone, Spain), plus one research-software practitioner.
- **Cross-language** — one statistics/verification specialist to audit the
  accuracy / precision / recall / calibration / skill vocabulary, which is
  where translation alone cannot disambiguate meaning.

Contested rows are marked ⚑ in [`translations/GLOSSARY.md`](../translations/GLOSSARY.md);
those are the rows most in need of a real ruling. To volunteer, open an issue
titled `[translation] review: <language> <chapters>`.
