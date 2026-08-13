# MLGeo in Spanish and French — community translation effort

*Machine Learning in the Geosciences* (ESS 469/569, University of Washington) is
being translated into **Spanish** and **French**. This directory is the public
home of that effort: the reader personas that guide it, the terminology
glossaries, and the translated chapters as they land. **We are asking the
community — especially native-speaker geoscientists — to help fine-tune both.**

## How this works

- **English is authoritative.** Translations are views over the same
  computational truth: code cells, executed outputs, and figures stay identical
  to the English edition; prose and notebook markdown are translated. Each
  translated file records the English commit it was based on, and CI will flag
  translations that go stale as the English edition evolves.
- **Synthetic personas as a design aid — not as review.** Thirteen fictional
  readers (eight from France, five from Spanish-speaking Latin America) steer
  the translation: each file is written in its target language and states what
  that reader needs — terminology conventions, register, tolerance for
  anglicisms, and the localization pitfalls of their field and country. They
  earned their place by catching real errors, including a wrong eligibility
  rule for Mexico's LANCAD compute allocations and an ice-cap claim about a
  volcano whose glaciers are gone. But they are **invented**, and an AI reading
  an invented reader's brief is not a francophone or hispanophone community
  accepting the terminology. Every persona file says so at the top, and real
  human review is recorded separately in
  [`docs/REVIEW_RECORD.md`](../docs/REVIEW_RECORD.md) — currently empty, which
  is the honest status.
- **Pilot, then fleet.** Chapter 1 was the pilot in both languages; the rest of
  the book followed once its glossary and style decisions were settled, chapter
  by chapter, each with a persona pass afterwards.
- **Tooling keeps the invariant honest.** `tools/nb_translate.py` moves only
  notebook markdown cells, so a translated notebook cannot pick up a code or
  output change by construction; `tools/check_translations.py` verifies that
  and the toc↔manifest coverage; `tools/gen_manifest.py` pins each page to its
  English source commit; `tools/copy_translation_assets.py` mirrors figures
  into each language tree.

## How to help

1. **Critique the personas** ([`personas/fr/`](personas/fr/), [`personas/es/`](personas/es/)):
   are these readers real? What did we get wrong about your country's academic
   register, your field's vocabulary, your students' English? Open a GitHub
   issue titled `[translation] persona: …` or a PR.
2. **Fight about the glossary** ([`GLOSSARY_fr.md`](GLOSSARY_fr.md),
   [`GLOSSARY_es.md`](GLOSSARY_es.md)): terminology is where translations live
   or die. Every row is contestable, especially the *keep-in-English* column.
   Issues titled `[translation] term: …`.
3. **Review the pilot** (`fr/Chapter1-GettingStarted/`,
   `es/Chapter1-GettingStarted/` once merged): read one page as the reader you
   are, and tell us where the register breaks, where a term jars, where an
   example fails to travel.
4. **Spanish reviewers**: we would love to partner with communities such as
   [GeoLatinas](https://geolatinas.org). Reviewers are credited as
   contributors to the translated edition.

## Ground rules the translators follow

- Code, variable names, executed outputs, and figure text remain in English —
  reading scientific code in English is itself a skill the course serves.
- Spanish targets neutral pan-regional Spanish (no voseo, no
  country-specific idiom in instructional prose); the multi-country personas
  exist precisely to catch regionalisms.
- French follows current French academic usage in quantitative science:
  established French terms where they are genuinely standard
  (*apprentissage automatique*, *surapprentissage*), the English term where
  French usage keeps it (*machine learning* in running text is acceptable on
  first mention with the French gloss; *dropout*, *transformer*).
- The instructional register is the impersonal/2nd-person-formal standard of
  each language's textbooks (French *vous*; Spanish *usted*-neutral
  imperative), matching the English edition's direct-but-professional voice.
- **Localization is part of the translation.** Two tiers:
  - *Prose localization (everywhere)*: hazard framings, institutions, and
    analogies speak to the reader's region — the French edition reasons about
    Alpine seismicity, Rhône floods, and Météo-France; the Spanish edition
    about SSN seismicity, Popocatépetl, hurricanes, and CONAGUA — wherever the
    English edition reaches for a Pacific-Northwest example in prose.
  - *Data localization (where the pipeline permits)*: when a local dataset
    drops into the same code with only an identifier change and teaches the
    same lesson, the translated notebook uses it and is re-executed. The
    flagship: the GNSS lesson (1.7) pulls its station from the same global
    Nevada Geodetic Laboratory archive — a French station for the French
    edition, a Mexican (TLALOCNet) station for the Spanish edition, same
    loader, same physics, local ground. Localized notebooks carry their own
    executed outputs; all other notebooks keep outputs identical to English.

## Licensing across countries

The book's licenses — **MIT** for code and **CC BY 4.0** for content — are
international instruments and are valid and enforceable in France, Mexico,
and the rest of the francophone and hispanophone world; no country-specific
license is needed, and the translated editions carry the same licenses as the
English original (a translation is an adaptation, which CC BY 4.0 expressly
permits with attribution). Three nuances, for the record:

- **CC BY 4.0 is "unported" by design** (unlike the old 3.0 jurisdiction
  ports) and has official license translations: the translated editions link
  the [French deed](https://creativecommons.org/licenses/by/4.0/deed.fr) and
  the [Spanish deed](https://creativecommons.org/licenses/by/4.0/deed.es) so
  readers see the license in their own language.
- **France**: the CeCILL license family (CEA/CNRS/Inria) exists for authors
  who want a license drafted under French law, and some French institutions
  historically preferred it; MIT is nonetheless standard and fully valid in
  France (CeCILL-B is explicitly MIT/BSD-compatible), so we do not
  dual-license unless a French institutional partner ever requires it. Note
  also that French *droit moral* (attribution, integrity) applies regardless
  of license and is satisfied by CC BY's attribution requirement. French
  public-sector *data* often ships under the Licence Ouverte/Etalab — that
  governs some data sources we may cite, not our own licensing.
- **Mexico**: the Ley Federal del Derecho de Autor recognizes these licenses;
  MIT and CC BY 4.0 are the norm in Mexican open science. No local
  alternative is customary.

## Status

| Piece | State |
|---|---|
| Personas (8 FR, 5 ES) | published, labeled synthetic — critique welcome |
| Glossaries (starter, ~50 terms each) | published — contested rows marked; new terms from the full-book pass pending merge |
| French edition, full book | translated, AI-reviewed against synthetic personas — **no human community review yet** |
| Spanish edition, full book | translated, AI-reviewed against synthetic personas — **no human community review yet** |
| Human review record | [`docs/REVIEW_RECORD.md`](../docs/REVIEW_RECORD.md) — empty; volunteers wanted |
| Translated sites at `/fr/`, `/es/` | CI builds both editions into the main site; flag links (🇺🇸/🇫🇷/🇪🇸) in every edition's header switch languages |
| Data localization beyond notebook 1.7 | not started — candidates identified per chapter (a RENAG/Alpine station and a TLALOCNet station are the strongest); each needs notebook re-execution |

### What is deliberately not translated

"Full book" means the 73 pages in each translated table of contents. Three
things sit outside it, on purpose rather than by omission:

| Asset | Why | Revisit when |
|---|---|---|
| `book/reference/bibliography.md` | A `{bibliography}` directive over `references.bib`. Entries are titles of published works and must stay in their publication language; translating them would misquote the literature. | Never, unless MyST gains localized bibliography styling worth using. |
| `book/leaderboard_standings.md` | Generated by CI from submitted results. A translated copy would be stale the moment the workflow next runs, and there is no per-language generation step. | If the leaderboard workflow gains language-aware output. |
| `book/slides/` | Reveal.js decks for the UW course sessions. The badges in every edition link to the English decks; deck content changes each quarter and the translated editions are not tied to a UW teaching calendar. | If a francophone or hispanophone instructor adopts the course and wants decks in their language — see the [adoption guide](../book/about_this_book/adopting_this_book.md). |

Both editions cover the whole book: front matter, chapters 1–7, and the
glossary — 73 pages each. Every page records the English commit it was
translated from, and `tools/check_translations.py` enforces the core
invariant in CI: code cells, executed outputs, and code fences are
byte-identical to the English edition. Prose is localized (French: Alpine
and Cévennes examples, Météo-France, Epos-France/RENAG, BRGM, Data Terra,
GENCI/Jean Zay; Spanish: SSN, SASMEX, CONAGUA, Protección Civil, Grijalva
basins, LANCAD/SNCAD/NLHPC compute).
