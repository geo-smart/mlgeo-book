# First-class active-learning sprint: paper → evidence → project

**Time:** 25 minutes in pairs

**Purpose:** introduce reproducibility through a paper students genuinely care
about, while generating structured candidate projects for the class data
gallery.

## Research landscape prompt

Use this matrix to choose a starting lane. It is a prompt, not a restriction:
methods and data modalities routinely cross domain boundaries.

| Research domain | Data modalities | Plausible first ML tasks |
|---|---|---|
| Earthquakes and volcanoes | waveforms, catalogs, sensor time series | detect events, classify signals, locate sources |
| Climate and weather | gridded fields, time series, ensembles | forecast, emulate simulations, quantify extremes |
| Cryosphere and remote sensing | imagery, rasters, point clouds | segment features, classify surfaces, detect change |
| Hydrology and hazards | station records, terrain, maps | forecast flow, map susceptibility, detect anomalies |
| Solid Earth and geodesy | GNSS/InSAR time series, spatial fields | estimate trends, detect transients, invert parameters |
| Oceans and ecosystems | profiles, trajectories, imagery, grids | discover regimes, forecast states, classify observations |

## Student prompt

Choose a geoscience question or domain that interests both partners. Use
ChatGPT Search to discover one recent research paper that applies machine
learning to that topic. Ask for:

> Find a recent peer-reviewed paper that applies machine learning to
> **[your geoscience topic]**. Prefer a paper with accessible data or code.
> Give the title, authors, year, journal, DOI, scientific question, data type,
> ML task, and links to the publisher page, data, and code. Clearly label
> anything you could not verify.

ChatGPT's response is a search lead, not evidence. Open the publisher or
archival landing page and independently confirm the title, authors, year, and
DOI before continuing.

## Reproducibility trace

Record direct links and one sentence of evidence for each item.

1. **Scientific claim:** Which exact figure, table, or numerical result would
   you attempt to reproduce?
2. **Data:** Where are the data? Are they public, licensed, documented, and
   small enough to use this quarter?
3. **Code:** Is there a repository or supplement? Does it identify a license
   and a version or release?
4. **Environment:** Are package versions, an environment file, container, or
   computing requirements provided?
5. **Evaluation:** What was held out? What baseline and metric were used? Is
   the split credible for the spatial, temporal, or grouped structure?
6. **Gap:** What single missing or ambiguous artifact most threatens an
   independent rerun?

Use **found**, **partial**, **missing**, or **not applicable** for each artifact.
A missing artifact is a legitimate finding, not a failed activity.

## Project-gallery entry

Submit one structured entry per pair:

- verified paper title and DOI;
- one-sentence scientific question;
- geoscience domain;
- data modality: time series, table/points, images/rasters, gridded fields, or
  waveforms;
- first ML task: forecast, classify, discover clusters, or detect events;
- data-access status: in hand, public source identified, restricted, or data
  still needed;
- exact result to reproduce;
- one extension or new question for a course project;
- largest reproducibility risk;
- partner names and preferred contribution: data, modeling, evaluation, or
  interpretation.

## Whole-room synthesis

Ask for two 60-second reports:

1. a paper with an unusually complete evidence trail; and
2. an exciting paper whose trail stopped at a missing dataset, codebase, or
   evaluation detail.

Project the class landscape and discuss where ideas cluster, which teams could
share a data source or method, and which ideas need a data-access rescue before
they can become viable projects.

## What this activity establishes

- AI-assisted search is allowed and useful, but independently verified sources
  carry the claim.
- Reproducibility is a chain of artifacts, not a statement in a methods
  section.
- A promising project needs a scientific question, tractable data, and an
  evaluation design before it needs a model.
- The gallery is a living class research map, not a list of polished examples.
