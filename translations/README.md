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
- **Personas first.** Ten fictional but realistic readers — five from France,
  five from Spanish-speaking Latin America — steer the translation the same way
  twelve personas steered the book's 2026 revision. Each persona file is
  written in its target language and states what that reader needs from a
  translation: terminology conventions, register, tolerance for anglicisms,
  and the localization pitfalls of their field and country. AI agents
  translate against these personas and the glossaries; humans review.
- **Pilot, then fleet.** Chapter 1 is the pilot in both languages. After
  instructor and community review of the pilot, the remaining chapters are
  translated with the corrected glossary and style decisions baked in.

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

## Status

| Piece | State |
|---|---|
| Personas (5 FR, 5 ES) | published — critique welcome |
| Glossaries (starter, ~50 terms each) | published — contested rows marked |
| Chapter 1 pilot, French | in progress |
| Chapter 1 pilot, Spanish | in progress |
| Remaining chapters | after pilot review |
| Translated site at `/fr/`, `/es/` | after pilot review |
