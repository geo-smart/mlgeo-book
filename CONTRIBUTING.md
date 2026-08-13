# Contributing

Contributions are welcome and credited. There are three quite different things you
might want to do, and they have different workflows:

1. **[Fix or improve the book](#fixing-and-improving-the-book)** — an error, a
   clearer explanation, a new lesson.
2. **[Personalize the book for your audience](#personalizing-the-book-for-your-audience)**
   — retarget it to a different discipline, level, or institution.
3. **[Work on a translation](#working-on-a-translation)** — improve the French or
   Spanish edition, or start another language.

The book is CC BY 4.0 and the code MIT, so you may fork and adapt freely with
attribution. We would rather you contribute back, but taking it and running is a
legitimate outcome — that is what open educational resources are for.

## Setting up

The environment is managed with [pixi](https://pixi.sh); there is no conda step.

```sh
git clone https://github.com/geo-smart/mlgeo-book.git
cd mlgeo-book
pixi install         # install the pinned environment

pixi run build-fast  # render from committed outputs — seconds
pixi run serve-fast  # live preview, no execution
pixi run build       # execute every notebook, then build — minutes
pixi run serve       # live preview with execution
```

**Use `build-fast` unless you changed code.** Notebooks ship with their outputs
committed, and that is what the published site renders, so the fast build shows you
exactly what readers will see — in about 17 seconds instead of several minutes. Run the
executing variants when you have edited a code cell, or before publishing.

Translated editions never execute (their notebooks carry the English edition's outputs
by design):

```sh
pixi run build-fr    # French edition
pixi run build-es    # Spanish edition
pixi run check       # the same gates CI runs
```

## Fixing and improving the book

1. Fork, then branch: `git checkout -b fix-short-description`
2. Edit under `book/`. Notebooks execute at build time — a failing cell fails the build.
3. Build locally and check the page you changed.
4. Push and open a pull request.

If your change touches a page that has translations, say so in the PR. CI records
which English commit each translated page was built from, so maintainers can see what
went stale; you are not expected to update the translations yourself.

## Personalizing the book for your audience

This book was written for ESS 469/569 at the University of Washington, and it shows:
the schedule, the quarter, the leaderboard, and most of the data are ours. Adapting it
is expected. Two independent axes exist, and each has its own set of personas:

| Axis | Directory | Steers |
|---|---|---|
| **Scientific audience** | [`personas/`](personas/) | Discipline, seniority, prior coding skill, what a reader needs to be able to do afterwards |
| **Language and culture** | [`translations/personas/`](translations/personas/) | Register, terminology, tolerance for English jargon, regional institutions and hazards |

They compose. A Chilean hydrology master's programme is the *hydrology master's student*
audience persona crossed with the *Southern Cone Spanish* language persona.

### The method

1. **Write your readers.** Copy two or three files from `personas/` and rewrite them for
   the people actually in your room: what they already know, what they will use this for,
   what would make them close the book. Keep them specific — "a second-year PhD student
   who can run a published model but cannot debug one" is useful; "a graduate student" is
   not. Ten personas that all sound alike are worth less than three that disagree.
2. **Review the book as them.** Give an AI agent one persona file plus a chapter and ask
   for the gaps: what must this reader know that the book does not teach, teaches too
   late, or teaches too shallowly. `personas/README.md` documents the output structure we
   use. Do this per persona — the disagreements between them are the signal.
3. **Decide, then edit.** Personas surface candidates; you decide. Expect roughly a third
   of any AI review to be wrong or not worth doing.
4. **Verify anything factual.** Persona reviews of this book produced confident, wrong
   claims — a broken image that was actually a code example, an institutional eligibility
   rule stated backwards. Check institutional and scientific claims against primary
   sources before you ship them.
5. **Swap the data.** Regional data is what makes a book feel like it is about the
   reader's world. `translations/README.md` describes the two-tier policy we use, and
   notebook 1.7 in the French and Spanish editions is a worked example: same loader, same
   physics, local stations, notebook re-executed.

### Versioning your personas

Do not give personas their own version number. A git tag pins the whole repository, so
the book's release is also the personas' release — cite the book version and you have
cited the personas that produced it. Each persona instead records **provenance** in its
frontmatter: `written-for` (which edition it was written against) and `last-run` (when
it last reviewed the book). Run `python tools/persona_status.py` to see which personas
have fallen behind the current tag; update `last-run` when you re-run one, and
`written-for` when you revise it for a new edition.

If you fork the book for your own audience, the same applies: tag your edition, and let
your personas ride that tag.

### A caution worth repeating

Personas are **fictional**. They are a way to hold a specific reader in mind while
revising, not evidence that a real community accepts the result. Do not describe a
persona-reviewed book as reviewed by the people the personas depict, and record real
human review separately — see [`docs/REVIEW_RECORD.md`](docs/REVIEW_RECORD.md) for the
template we use.

### Contributing your adaptation back

- **New or better personas** — open a PR adding to `personas/` or
  `translations/personas/`. Personas for audiences we have missed are especially useful.
- **Regional datasets** — the highest-value contribution. If you have a dataset that
  drops into an existing notebook and teaches the same lesson on local ground, open an
  issue titled `[data] <chapter>: <region>`. See issues #45 and #46 for the shape.
- **A whole adapted edition** — tell us in an issue; we will link it from the README so
  others find it. It does not need to live in this repository.

## Working on a translation

Read [`translations/README.md`](translations/README.md) first — it explains the
invariant that makes this work: **code cells, executed outputs, and code fences stay
byte-identical to the English edition; only prose is translated.** This is enforced in
CI, not just requested.

- **Never edit a translated notebook directly.** Use
  `python tools/nb_translate.py extract|inject`, which moves markdown cells only. Pass
  `--base-dst` for the data-localized notebook 1.7, or you will silently revert its
  localization.
- **Terminology** lives in [`translations/GLOSSARY.md`](translations/GLOSSARY.md)
  (trilingual) plus the per-language tables. It is a *usage guide, not an authority over
  the chapters* — where a chapter and a glossary row disagree, the chapter is usually
  reporting real usage. Fix the row and open an issue; do not quietly rewrite prose.
- **Before pushing:** `pixi run python tools/check_translations.py` and
  `pixi run python tools/lint_terminology.py` must both pass.
- **Contested terms** are marked ⚑ in the glossary. Rulings from native-speaker
  researchers are the single most useful thing you can contribute; open an issue titled
  `[translation] term: …`.
- **A new language** — open an issue titled `[translation] new language: …`. The tooling
  is language-agnostic; what a new edition needs is a glossary, a persona set, and
  someone who will own it.

## Reporting problems

Open a [GitHub issue](https://github.com/geo-smart/mlgeo-book/issues). Useful titles:
`[bug]`, `[content]`, `[translation]`, `[data]`, `[persona]`. If you found a factual or
scientific error, say what the correct claim is and where you checked it — that turns a
report into a fix.
