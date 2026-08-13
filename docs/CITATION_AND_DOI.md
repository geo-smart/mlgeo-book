# Citing this book, and how to get a DOI

## Where things stand

The book has an author list and a `CITATION.cff`, but **no DOI yet**. The JOSE
manuscript (`JOSE_PAPER/paper.md`) was never peer reviewed, so it cannot serve
as the citation: citing an unreviewed manuscript as if it were a published
article misrepresents its status, and JOSE assigns no DOI until acceptance.
`CITATION.cff` therefore lists it as `status: in-preparation` and points
citers at the software release instead.

## The recommended alternative: Zenodo

Zenodo (operated by CERN) mints DOIs for free and integrates with GitHub
releases. It is the right choice here for three reasons: it is what this book
already teaches students to do in [1.1](../book/Chapter1-GettingStarted/1.1_open_reproducible_science.md);
it versions cleanly, giving both a per-release DOI and a **concept DOI** that
always resolves to the newest version; and it needs no editorial review, so it
does not block on the JOSE outcome. When JOSE is eventually published, the
article citation can be added alongside — the two are complementary, not
competing.

**Three steps, all requiring the repository owner's account:**

1. Sign in to [zenodo.org](https://zenodo.org) with GitHub, open
   **Settings → GitHub**, and switch `geo-smart/mlgeo-book` **on**. Zenodo only
   archives releases created *after* the switch is flipped, so do this first.
2. Publish a GitHub release from the tag (`v2.0-2026-edition`). Zenodo archives
   it and mints two DOIs: one for that version, one concept DOI for the record
   as a whole.
3. Put the **concept** DOI into `CITATION.cff` (uncomment the `doi:` line) and
   add the Zenodo badge to `README.md`. Commit; no re-release needed.

`.zenodo.json` in the repository root already carries the authors, ORCIDs,
license, and description, so the Zenodo record is populated correctly on the
first try rather than needing manual editing afterwards.

## If Zenodo is not wanted

- **OSF** (osf.io) issues DOIs for projects and components and handles mixed
  material (data, slides, manuscripts) well. Weaker GitHub integration:
  archiving is manual per version.
- **Figshare** issues DOIs and is institutionally supported at many
  universities. Also manual, and oriented toward single artifacts rather than
  an evolving repository.
- **A university library repository** — the UW ResearchWorks service issues
  DOIs and is the right home if the book should sit in the institutional
  record. Slower, involves a librarian, and versioning is coarser.
- **Software Heritage** archives the repository and gives a permanent SWHID,
  but a SWHID is not a DOI and is not accepted where a DOI is required.

Zenodo is the recommended default; the others are listed so the choice is
informed rather than assumed.

## How to cite, meanwhile

Until the DOI exists, cite the versioned release and its URL:

> Denolle, M., Cristea, N., Mehra, A., Ducellier, A., Sun, Z., Todoran, S.,
> Henderson, S., & Jensen, C. (2026). *MLGeo: Machine Learning in the
> Geosciences* (v2.0-2026-edition) [Curriculum book].
> https://github.com/geo-smart/mlgeo-book

## Authorship

`CITATION.cff` lists the eight authors of the JOSE manuscript, each with an
ORCID. The repository history contains additional contributors who are not on
that list; if any of them should be credited as authors of the book itself
rather than acknowledged in
[acknowledgements](../book/about_this_book/acknowledgements.md), add them to
`CITATION.cff` and `.zenodo.json` together, since Zenodo reads the latter.
