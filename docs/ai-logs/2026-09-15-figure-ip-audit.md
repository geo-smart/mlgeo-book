# Figure provenance audit

**Repository:** `geo-smart/mlgeo-book`, branch `audit-followups`
**Date:** 2026-09-15
**Method:** every image reference in the prose channel of the 79 English pages
(markdown pages and notebook markdown cells; `tools/extract_channels.py`),
matched to the attribution text within ±400 characters, then each unattributed
image inspected by eye and traced to its source. 47 references on 21 pages.
Two of the 47 are code-block examples (`glass.png`, `<yourname>.png`), not images.

The book is CC BY 4.0. A figure copied from a source that does not permit
redistribution under that licence is a defect regardless of how common the
figure is in teaching slides. The fix chosen here is to **redraw** rather than
seek permission, because a redrawn figure can be tied to the lesson's own
notation and regenerated from a script.

## Verdict per image

| Image | Page | Source found | Licence | Action |
|---|---|---|---|---|
| `TLU.png` | 4.1 | Géron, *Hands-On ML* fig. 10-4 | O'Reilly, all rights reserved | **redrawn** |
| `MLPReg.png`, `MLPClass.png` | 4.2 | Géron fig. 10-7 / 10-9 | same | **redrawn** |
| `GD_cartoon.jpeg`, `GD_non_global.png`, `GD_AlphaTooSmall.png`, `GD_AlphaTooLarge.png` | 4.5 | Géron fig. 4-3 to 4-6 | same | **redrawn** |
| `votingclassifier.png` | 3.9 | Géron fig. 7-2 (caption said so) | same | **redrawn** |
| `ValsetApproach.png`, `Kfold.png`, `LOOCV.png` | 3.8 | ISLR fig. 5.1 / 5.5 / 5.3 (the first was captioned "From scikit-learn", wrongly) | Springer | **redrawn** |
| `autoen_architecture.png` | 4.6 | Lilian Weng, "From Autoencoder to Beta-VAE" | personal blog, no licence | **redrawn** |
| `Unet.png` | 4.6 | Ronneberger et al. 2015, fig. 1 | Springer / authors | **redrawn** |
| `mean/variance/skewness/kurtosis.png` | 2.7 | Gregory Gundersen's blog (caption said so) | personal blog, no licence | **regenerated** |
| `Wavelet-Out1.jpeg`, `wavelet_families.png` | 2.8 | Ahmet Taspinar's blog (caption said so) | personal blog, no licence | **regenerated** |
| `filters.png` | 2.9 | untraceable | unknown | **regenerated** from `scipy.signal` |
| `hdf5_structure4.jpeg` | 2.2 | HDF Group tutorial (linked) | HDF Group | **redrawn** with the lesson's own groups |
| `overview-github-collaboration.png` | 1.5 | CU Boulder Earth Lab (caption said so) | CC BY-NC-SA 4.0: NC is incompatible with CC BY | **redrawn** |
| `phdcomics_finaldoc.gif` | 1.5 | PhD Comics (Jorge Cham) | classroom use permitted, redistribution is not | **removed**, replaced by our own `versioning-by-filename.png` |
| `teaching.png` | 2.13 | untraceable icon | unknown | **removed**, badge is now a text link |
| `mlps.svg`, `rnn.svg`, `lstm-2.svg` | 4.2, 4.4 | *Dive into Deep Learning* (d2l.ai) | CC BY-SA 4.0 | keep; **credit + licence added** (rnn had credit, no licence; the other two had neither) |
| `correlation.svg`, `pooling.svg`, `lenet.svg` (hotlinked) | 4.3 | d2l.ai | CC BY-SA 4.0 | keep; credit added for pooling and lenet |
| `Convolution_of_box_signal_with_itself2.gif`, `2D_Convolution_Animation.gif` (hotlinked) | 4.3 | Wikimedia Commons; Michael Plotke | CC BY-SA 3.0 | keep; **credit added** (required by the licence, was missing) |
| `roc-curve-v2-glassbox.png` | 3.4 | Wikimedia Commons, MartinThoma | CC0 | keep, already attributed |
| `grid_search_cross_validation.png` | 3.8 | scikit-learn docs | BSD-3 | keep, licence added to the credit |
| `TW_main-branch.png`, `TW_sub-branch.png` | 1.5 | The Turing Way | CC BY 4.0 | keep, attributed via footnote |
| `GitHubIssue.png` | 1.5 | screenshot of our own repo's UI | GitHub permits UI screenshots | keep |
| `Open-Science.svg` | 1.1 | Denolle & Gabriel 2023 slide title | ours | keep |
| `geocast-alldata.png`, `jensen.png`, `GeoSMART_logo.svg` | 1.6, 7.3, 1.2 | ours (jensen: co-author Claire Jensen) | ours | keep |
| `Dalle-geoscientific-data.png` | 2.1 | DALL·E output (caption says so) | OpenAI terms assign output to the user | keep |
| `Google_Slides_Logo.svg` | 2.1 | Google trademark, used as a link badge to a Google Slides deck | nominative use | keep |
| `colab-badge.svg` (hotlinked) | 1.4 | Google, "Open in Colab" badge | intended use | keep |
| `maxresdefault.jpg` (hotlinked YouTube thumbnail) | 1.1 | frame of a third-party video, as a link preview | standard embed practice | keep, noted |

Totals: 22 figures redrawn or regenerated, 2 removed, 6 credits added or completed, 14 kept as-is.

## What was not checked

Figures produced by the notebooks' own code (matplotlib outputs in executed
cells) are ours by construction and were not inventoried. Slide decks under
`book/slides/2026/` were not audited here; they carry their own `figs/` and
`figscripts/` and are a separate pass.

## Regeneration

Scripts in `tools/figures/`; see its README. After regenerating, run
`tools/copy_translation_assets.py` — it skips existing files, so delete the
stale copies under `translations/{fr,es}/img/` first.

## Unreferenced images removed from `book/img`

25 files that no page, slide or README referenced were deleted from the tree
(they remain in git history). Most were 2024-edition leftovers, and several are
recognisable published figures with no redistribution licence: `ConvNetQuake.jpg`
(Perol et al. 2018), `DeepDenoiser_*.png` (Zhu & Beroza 2019), `EqT.png`
(Mousavi et al. 2020), `cnn_rouet-leduc.png`, `CNN_StanfordCS230.gif`,
`denoising-autoencoder-architecture.png` (Weng), `capacity-vs-error.svg`,
`dropout2.svg`, `lenet-vert.svg`, `singleneuron.svg` (d2l.ai). Also removed:
`AC29_map.png`, `wavedecompnet.jpeg` (ours, unused), `GD_2D.png`, `SGD.png`,
`Convolution.png`, `max_pooling.png`, `cnn2.jpeg`, `fashion_mnist.png`,
`auto-sklearn.png`, `autosklearn.png`, `Google_Slides_logo.svg` (lower-case
duplicate), `YouTube-Logo.wine.svg`, `logo.png`, `student_version_badge.svg`.
Unreferenced files are not copied into the built site by MyST, so this changes
nothing a reader sees; it changes what the repository distributes.
