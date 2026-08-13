# [i18n/editorial] Comprehensive academic-language audit of `geo-smart/mlgeo-book`

**Repository:** `geo-smart/mlgeo-book`  
**Snapshot audited:** `main` at `4d085399b1ac8ca0bf8bffddc208ab42612784fb` (2026-08-13)  
**Audit date:** 2026-08-13  
**Requested output:** one issue-ready audit; no GitHub write actions performed

## Executive summary

The French and Spanish editions are technically much stronger than a typical first translation pass. Both cover the complete published teaching corpus: 73 pages each (36 Markdown files and 37 notebooks), spanning the front matter, Chapters 1–7, and the glossary. The repository's own `tools/check_translations.py` reports `fr: OK` and `es: OK`. An independent cell-level comparison also found no notebook code-identifier drift. No translated notebook has an extra or missing code cell, and English-looking prose detected in both editions was limited to a bibliographic title that should remain in its publication language.

The remaining work is therefore not “translate the missing book.” It is to align the editions with their actual audiences and with the professional cultures in which the terminology is used:

1. **Higher-education textbook materials**, where a term must teach both the concept and the vocabulary students will meet in code, papers, seminars, and internships.
2. **Academic and applied researchers**, who will reject technically correct but socially unnatural terminology.
3. **Culturally aware technical prose**, which must distinguish a useful bilingual bridge from over-translation.

The highest-priority finding is that the French glossary is more prescriptive than the French chapters and the intended readership warrant. It currently makes **apprentissage automatique** the default and treats *machine learning* as a tolerated first-mention form. Current French higher-education and research usage is genuinely mixed: Paris-Saclay programs use both *apprentissage automatique* and *machine learning* in the same program, CNRS training is titled “Machine Learning et Deep Learning,” and Inria research pages alternate between the two. For this book's research-facing audience, the most natural policy is:

> **First substantive occurrence per chapter:** *machine learning* (**ML**; « apprentissage automatique »).  
> **Thereafter:** **ML** or *machine learning* for the field and research practice; use the French **apprentissage…** family for named learning paradigms (*apprentissage supervisé, non supervisé, auto-supervisé, par renforcement*).

This is not a claim that *apprentissage automatique* is wrong. It is a register decision: the field name used by many francophone researchers is *machine learning* or **ML**, while the conceptual subtypes are very naturally expressed with *apprentissage*. The book should teach both without pretending that one has displaced the other.

The second priority is to replace one-word equivalence tables with **context rules**. *Workflow*, *pipeline*, *repository*, and *archive* do not have one translation each:

- a Git repository is a **dépôt Git**, not an archive;
- a preservation service such as HAL or Zenodo is an **archive** or preservation repository;
- a conceptual research workflow may be a **démarche** or **protocole**;
- a reproducible computational workflow may naturally remain a **workflow de calcul** for researchers;
- a signal/data pipeline is a **chaîne de traitement**;
- `sklearn.pipeline.Pipeline` is an API identifier and must remain `Pipeline`.

The English source also needs an editorial pass before future translations propagate its defects. Chapter 2 contains several clear errors (*equialent*, *Euclidian*, *metadat*, *arras*, *Xarrrays*, *refered*, *costum*) and awkward passages. The filename spelling `Assignement` is repeated across source and translations and should be corrected with redirect/alias handling.

## Scope, method, and limits

### Corpus coverage

| Section | English pages | French pages | Spanish pages | Status |
|---|---:|---:|---:|---|
| Chapter 1 — Getting Started | 10 | 10 | 10 | complete |
| Chapter 2 — Data Manipulation | 15 | 15 | 15 | complete |
| Chapter 3 — Machine Learning | 13 | 13 | 13 | complete |
| Chapter 4 — Deep Learning | 12 | 12 | 12 | complete |
| Chapter 5 — Model Workflows | 6 | 6 | 6 | complete |
| Chapter 6 — Agentic AI | 6 | 6 | 6 | complete |
| Chapter 7 — Use Cases | 4 | 4 | 4 | complete |
| Front matter / about | 6 | 6 | 6 | complete |
| Reference glossary | 1 translated page | 1 | 1 | complete |
| Published teaching corpus | **73** | **73** | **73** | complete |

`book/reference/bibliography.md`, `book/leaderboard_standings.md`, and the slide sources are not represented as translated pages. They are not listed in the translated MyST tables of contents and should be classified explicitly as either shared/non-translatable assets or future translation scope; they should not remain an undocumented exception to “full book.”

### Checks performed

- Inventoried every Markdown and notebook page in `book/`, `translations/fr/`, and `translations/es/`.
- Compared source and translated notebook cell structure.
- Compared Python identifier token sequences in every corresponding code cell.
- Scanned Markdown cells separately from code cells and outputs for untranslated English prose.
- Audited the current trilingual and language-specific glossaries against actual chapter usage.
- Searched the French and Spanish prose for high-risk ML/statistics and scientific-computing terms.
- Ran `pixi run python tools/check_translations.py`: **French OK; Spanish OK**.
- Reviewed representative passages from every chapter family, with detailed attention to Chapters 1–5 where terminology density is highest.
- Checked current usage against official or primary institutional examples from French and Spanish-speaking universities and research bodies.

This is an editorial and terminological audit, not a certification by a national standards body. The repository's fictional personas are useful design prompts, but they are not evidence of community acceptance. Final contested terminology should be reviewed by identifiable francophone and hispanophone subject-matter experts.

---

## 1. French academic terminology and house style

### 1.1 Recommended register

Target educated readers directly. Use natural contemporary French research prose, not administrative French and not literal localization. Preserve the book's concise, second-person plural instructional voice. Avoid both extremes:

- **over-francization**, which invents terms researchers do not use and weakens bridges to the literature;
- **uncontrolled franglais**, which leaves students without conceptual support.

The right unit is not “one English word → one French word.” It is “one concept in one discourse setting → the form a francophone researcher or instructor would use there.”

### 1.2 Revise the ML policy

Current glossary rule:

> `Machine learning | apprentissage automatique † | Bilingual at first mention; the native term dominates afterwards.`

Recommended rule:

> `Machine learning | machine learning (ML) ; apprentissage automatique as pedagogical/formal gloss | First occurrence: « machine learning (ML ; apprentissage automatique) ». Thereafter use ML/machine learning for the field and applied research practice. Retain apprentissage for named paradigms.`

Rationale and evidence:

- [Université Paris-Saclay, M1 Mathématiques et Apprentissage Statistique](https://www.universite-paris-saclay.fr/formation/master/mathematiques-et-applications/m1-mathematiques-et-apprentissage-statistique) freely alternates *machine learning*, *apprentissage automatique*, *apprentissage statistique*, and *data science* in a French graduate program.
- [CNRS Formation Entreprises](https://cnrsformations.cnrs.fr/catalogue/formation/36/machine-learning-et-deep-learning-pour-la-vision-par-ordinateur/) titles a course “Machine Learning et Deep Learning” while using *apprentissage automatique* inside the description.
- [Inria's TAU description](https://www.inria.fr/fr/tailor-reseau-europe-intelligence-artificielle) uses both *apprentissage automatique* and *machine learning* in research-facing prose.
- [Collège de France](https://www.college-de-france.fr/fr/chaire/yann-lecun-informatique-et-sciences-numeriques-chaire-annuelle/biography) uses *apprentissage automatique (machine learning)*, demonstrating that the French form is valid but commonly paired with the English field name.

Concrete repository changes:

- `translations/GLOSSARY.md`: rewrite the `Machine learning` row and the introductory claim that native French “dominates afterwards.”
- `translations/GLOSSARY_fr.md`: change the first row to the audience-sensitive rule above.
- `translations/README.md`: replace “established French terms where genuinely standard (*apprentissage automatique*)” with a mixed-register explanation.
- `translations/fr/reference/glossary.md`: define the concept under `Machine learning (ML) — apprentissage automatique`, rather than hiding the term students will hear most often.
- Chapter openings should introduce **ML** once; do not mechanically replace the well-formed subtype phrases *apprentissage supervisé*, *apprentissage profond*, etc.

### 1.3 Workflow / démarche / chaîne de traitement / pipeline

| Meaning | Recommended French | Avoid |
|---|---|---|
| Research process at the conceptual level | **démarche de recherche**, **protocole**, sometimes **processus** | automatic use of *flux de travail* |
| Reproducible computational process | **workflow de calcul** or **workflow reproductible**; gloss once if needed | pretending researchers never say *workflow* |
| Ordered signal/data transformations | **chaîne de traitement** | using *démarche* for executable transformations |
| ML preprocessing + estimator object | **pipeline**, and `Pipeline` for the scikit-learn class | translating the identifier |
| CI automation file | **workflow GitHub Actions**; preserve `.github/workflows/...` | *archive* or *protocole* |
| Narrow movement of work between people | **flux de travail** | using this as the only translation for every sense |

The current repository contradicts itself. `translations/GLOSSARY.md` says `Workflow → flux de travail`, while `translations/fr/MANIFEST.yml` explicitly says a senior pass selected community-standard *workflow*, and Chapters 5.3–5.4 use *workflow* repeatedly. The chapters are often more culturally accurate than the glossary.

Concrete decisions:

- [Chapter 5.3, line 15](https://github.com/geo-smart/mlgeo-book/blob/4d085399b1ac8ca0bf8bffddc208ab42612784fb/translations/fr/Chapter5-ModelWorkflows/5.3_data_model_versioning.md#L15): keep **workflow Git**, not *flux de travail Git*.
- [Chapter 5.3, lines 78–102](https://github.com/geo-smart/mlgeo-book/blob/4d085399b1ac8ca0bf8bffddc208ab42612784fb/translations/fr/Chapter5-ModelWorkflows/5.3_data_model_versioning.md#L78): keep **workflow GitHub Actions** and **workflow d'évaluation**; add the term to the glossary.
- `translations/fr/Chapter2-DataManipulation/2.13_MLready_data.ipynb`, markdown cell 2: the prose correctly uses **flux de travail condensé** for a broad method and **chaîne de traitement reproductible** for executable steps; preserve this as the model example.
- `translations/fr/Chapter3-MachineLearning/3.2_classification_regression.ipynb`, markdown cell 24: replace “un pipeline qui développe…” with “une **chaîne de modélisation** qui développe…”, unless the passage is specifically introducing the scikit-learn `Pipeline` object.
- `translations/fr/Chapter3-MachineLearning/3.9_ensemble_learning.ipynb`, markdown cells 9 and 16: keep *pipeline* because the object wraps `StandardScaler` and an estimator; write “un *pipeline* scikit-learn (`Pipeline`)” on first occurrence.
- `translations/fr/about_this_book/about_this_book.md`, line 51: “réexécute **la chaîne de calcul**” is more natural than “réexécute le pipeline” in a learning-outcome table.

### 1.4 Repository / dépôt / archive / entrepôt

| English concept | Recommended French | Explanation |
|---|---|---|
| Git repository | **dépôt Git** | Active version-controlled project with commits and branches. |
| GitHub repository | **dépôt GitHub** | Use GitHub's untranslated UI label only when telling users exactly what to click. |
| Repository, abstract software sense | **dépôt**; **référentiel** only when the institutional/product vocabulary requires it | GitHub's French docs alternate between *référentiel* and *dépôt*; developer prose strongly supports *dépôt*. |
| Institutional/open archive | **archive ouverte**, **archive institutionnelle** | HAL explicitly identifies itself as France's national open archive. |
| Long-term data archive | **archive de données** | Preservation and stable access are foregrounded. |
| Managed data repository/portal | **entrepôt de données**, **dépôt de données**, or named portal | Choose by function, not English spelling. |
| Object store | **stockage objet** / **magasin d'objets** | Not *archive* unless preservation is the function. |

The current distinction in [Chapter 1.1, line 81](https://github.com/geo-smart/mlgeo-book/blob/4d085399b1ac8ca0bf8bffddc208ab42612784fb/translations/fr/Chapter1-GettingStarted/1.1_open_reproducible_science.md#L81) is good: GitHub hosts the **dépôt**, while a GitHub release is **archivée** on Zenodo. Preserve it and encode the distinction in the glossary.

[HAL](https://about.hal.science/) calls itself an **archive ouverte multidisciplinaire** and the national open archive; this supports *archive* for preservation/dissemination, not for an active Git project. [GitHub's French documentation](https://docs.github.com/fr/get-started/using-git/about-git) states directly that a Git project is a **dépôt**, even though other interface pages also use *référentiel*. The book should use **dépôt** in prose and reproduce UI labels verbatim in bold when required.

### 1.5 French ML/statistics canonical choices

| English | Recommended research-textbook French | Notes |
|---|---|---|
| feature | **variable**, **variable explicative**, or **caractéristique** | Use *feature* parenthetically where it bridges to code; avoid forcing *caractéristique* where statisticians would say *variable*. |
| feature engineering | **feature engineering (construction de variables/caractéristiques)** on first use | “Ingénierie des caractéristiques” is comprehensible but can sound translated; discipline-dependent wording is better. |
| target | **variable cible**, **cible** | Preserve `target` as a column/key name. |
| label | **étiquette**, sometimes **classe** | Do not collapse a numeric target into an étiquette. |
| fitting | **ajustement** / **ajuster** | Preserve `.fit()` and `fit`. |
| training | **entraînement** | Preserve `train()`, `training_loss`, etc. |
| inference | **inférence** for deployment/prediction phase; **prédiction** when that is the concrete act | Do not use inference in the statistical-estimation sense without context. |
| clustering | **clustering** or **partitionnement**, defined together | For the target audience, *clustering* is normal. Do not mandate a single purist substitute. |
| cluster | **groupe**, **classe**, **cluster**, or **partition** depending on method | “Grappe” is not a safe universal default for modern ML prose. |
| classification | **classification** | Stable. |
| regression | **régression** | Stable. |
| loss | **perte** / **fonction de perte** | Distinguish from a broader objective and from decision-theoretic cost. |
| objective | **fonction objectif** | May contain multiple loss and regularization terms. |
| cost | **coût** only when a genuine cost is meant | Do not ban the word; ban semantic conflation. |
| validation | **validation** | Distinguish model validation, validation set, and software verification. |
| accuracy | **exactitude** or **taux de bonnes classifications** | Never *précision* when the metric is `accuracy`. |
| precision | **précision** | Reserve for `TP/(TP+FP)`. |
| recall | **rappel** or **sensibilité** depending field | State the formula; medical readers may expect *sensibilité*. |
| dataset | **jeu de données** | *Dataset* may be mentioned as ecosystem jargon, not forced away from code/product names. |
| baseline | **baseline (modèle de référence)** first, then either form consistently | Researchers commonly use *baseline*. |
| benchmark | **benchmark (banc d'essai / jeu de référence)** by context | A benchmark dataset is not always a banc d'essai. |
| notebook | **notebook Jupyter (carnet)** first, then **notebook** for research-facing prose | “Carnet Jupyter” is pedagogically clear but less natural as the sole working term for many researchers. |

### 1.6 Correct a real loss/cost inconsistency

The glossary says “never *fonction de coût*,” but three French passages use it. The answer is not a blind replacement; each passage needs a semantic edit:

- `translations/fr/Chapter3-MachineLearning/3.2_classification_regression.ipynb`, markdown cell 14: replace the conflation “fonction de perte, de coût ou objectif” with a short hierarchy: “Une **fonction de perte** mesure l'erreur sur un exemple; la **fonction objectif** agrège la perte et, le cas échéant, des termes de régularisation. Le terme **coût** est réservé ici aux conséquences ou pondérations auxquelles on attribue un coût.” Then introduce MSE as a loss/objective as appropriate.
- `translations/fr/Chapter3-MachineLearning/3.7_randomForest_regression.ipynb`, markdown cell 1: replace “fonction de coût” with **critère de partition**; Gini impurity, entropy, and squared error are split criteria in this passage.
- `translations/fr/Chapter4-DeepLearning/mlgeo_4.0_perceptrons.ipynb`, markdown cell 16: use **fonction de perte quadratique** or **fonction objectif**, depending on the subsequent code.

### 1.7 Scientific-computing French

| English | Recommended French | Register note |
|---|---|---|
| environment | **environnement (logiciel/Python/conda)** | Stable; preserve filenames such as `environment.yml`. |
| package | **paquet** or **bibliothèque** | A package and a library are not always identical; follow the ecosystem object. |
| dependency | **dépendance** | Stable. |
| lockfile | **fichier de verrouillage**; `pixi.lock` unchanged | Good in current text. |
| build | **build** in CI/research-engineering speech; **construction** when describing the published book | Gloss once; do not force one form. |
| job | **job** or **tâche** depending scheduler/CI context | HPC users commonly say *job*. |
| cloud | **cloud**, **informatique cloud** | For this audience, *informatique en nuage* is not the natural default. |
| HPC | **HPC (calcul haute performance)** first, then HPC | Keep acronym. |
| cluster | **cluster de calcul** or **grappe de calcul** | Both occur; *cluster* is natural among practitioners. Avoid confusing this with an ML cluster. |
| node | **nœud** | Stable. |
| scratch | **espace scratch (temporaire)** | Keep filesystem/scheduler vocabulary. |
| home | **répertoire personnel (`$HOME`)** | Preserve `$HOME`. |
| object storage | **stockage objet** | Prefer over literal *magasin* in general prose. |

---

## 2. French full-book language audit

### Overall assessment

The French edition is coherent, technically careful, and substantially localized. Its best passages already model the correct strategy: natural French explanation, English API names preserved, and a bilingual bridge where needed. The main defect is not pervasive mistranslation; it is that the glossary sometimes declares normal chapter usage “tolerated” or wrong.

### Chapter-level findings

| Section | Assessment | Concrete action |
|---|---|---|
| Front matter | Strong, localized, research-aware | Change the ML and notebook register in `about_this_book.md`; use *ML* and *notebook* naturally after a bilingual first mention. |
| Chapter 1 | Strong pedagogy and appropriate French institutions | Preserve **dépôt vs archive** distinction; revise `cloud` and `HPC` house rules to reflect actual practitioner language; preserve UI labels such as **New repository** verbatim. |
| Chapter 2 | Good translation, but inherits source defects and overglosses repeated terms | After fixing English source cells, retranslate affected cells. Use *feature engineering* as a bilingual bridge and permit *notebook*. |
| Chapter 3 | Scientifically strong; most terminology is consistent | Fix loss/cost/objective hierarchy; treat *clustering* and *pipeline* as legitimate research jargon rather than exceptions; preserve metric distinction exactitude/précision. |
| Chapter 4 | Strong and technically detailed | Add an explicit rule for *inférence* vs *prédiction*; preserve `input`, `target`, `DataLoader`, and loss class names; do not translate labels embedded in code outputs. |
| Chapter 5 | Best evidence that current glossary policy is wrong | Keep *workflow*, *job*, *build*, *cluster/HPC*, and *cloud* where culturally natural; update glossary to match. |
| Chapter 6 | Good contemporary agent/evaluation register | Keep *prompt*, LLM, agent, eval only where defined; prefer **jeu d'évaluation** in pedagogy but allow *eval* in engineering references. |
| Chapter 7 | Natural applied-research voice | Retain domain localization; verify institutional names annually because these can change. |
| Glossary | Valuable but over-prescriptive | Convert from one-to-one translation authority into contextual usage guide; resolve internal contradictions. |

### Concrete French replacement queue

| Priority | Location | Current | Proposed |
|---|---|---|---|
| P0 | `translations/GLOSSARY.md`, Machine learning row | native term dominates | ML/*machine learning* for field; *apprentissage…* for paradigms; bilingual first occurrence |
| P0 | `translations/GLOSSARY.md`, Workflow row | `flux de travail` only | contextual row covering workflow/démarche/chaîne/pipeline |
| P0 | `translations/GLOSSARY.md`, Jupyter notebook row | `carnet Jupyter` default | `notebook Jupyter (carnet)` for research-facing edition |
| P0 | `translations/GLOSSARY.md`, Cluster row | `grappe` | context: cluster/groupe/classe; reserve *grappe* for compute where desired |
| P0 | Chapter 3.2, markdown cell 14 | loss/cost/objective treated as synonyms | define loss, objective, and cost separately |
| P0 | Chapter 3.7, markdown cell 1 | `fonction de coût` for tree split | `critère de partition` |
| P1 | Chapter 1.5, lines 253/263 | first gloss then repeated *workflows* | keep *workflow* but normalize singular/plural and glossary status |
| P1 | Chapter 2.20, lines 22/31/45/89 | `notebook(s)` not aligned with glossary | keep as intended research register or normalize to new `notebook Jupyter (carnet)` rule |
| P1 | Chapter 3.2, cell 24 | generic `pipeline` | `chaîne de modélisation`, unless immediately tied to `Pipeline` |
| P1 | Chapter 5.3, lines 15/78/84/102 | *workflow* conflicts with glossary | keep chapter; change glossary |
| P2 | Chapter 1.4 | mixed *cloud*, grappe/cluster, HPC | preserve mixed practitioner register but define it once at section opening |

---

## 3. Spanish academic terminology and house style

### 3.1 Register and international scope

Use internationally intelligible scientific Spanish, with a Latin American center of gravity appropriate to the existing examples and personas, while avoiding caricatured “neutral Spanish.” A global edition can choose consistent forms without erasing legitimate variation.

Primary institutional usage is also mixed. [CONICET](https://icate.conicet.gov.ar/inteligencia-artificial/) uses **Aprendizaje Automático (Machine Learning en inglés)**; a [UNAM postgraduate biology course](https://pcbiol.posgrado.unam.mx/programas_cursos/2026-2/introduccion_a_la_inteligencia_artificial.pdf) uses **aprendizaje automático (machine learning, ML)**; other UNAM programs put *machine learning* in titles and use **ML** throughout. Therefore:

> First occurrence: **aprendizaje automático (*machine learning*, ML)**.  
> Thereafter: **aprendizaje automático** in explanatory prose and **ML** in compact technical prose, diagrams, comparisons, and references to the ecosystem.

Unlike the French edition, Spanish does not need to make the English field name the normal prose default. But it should teach **ML** explicitly and should not imply that professional Spanish usage excludes English terms.

### 3.2 Spanish canonical choices

| English | Recommended international Spanish | Notes |
|---|---|---|
| feature | **característica** or **variable explicativa** | Do not mandate one across statistics and deep learning. Preserve `feature` in identifiers. |
| target | **variable objetivo**, **objetivo** | Preserve `target` key/column. |
| label | **etiqueta** | Use **clase** only when it really means class. |
| fitting | **ajuste** / **ajustar** | Preserve `.fit()`. |
| training | **entrenamiento** | Stable. |
| inference | **inferencia** or **predicción** by phase/action | Same conceptual distinction as French. |
| clustering | **agrupamiento (*clustering*)** first | *Clustering* is common; do not repeatedly re-gloss. |
| cluster | **conglomerado**, **grupo**, or **clúster** by community | A single mandated word will not travel equally well across regions. |
| classification | **clasificación** | Stable. |
| loss | **pérdida** / **función de pérdida** | Keep separate from omission/miss. |
| objective | **función objetivo** | Stable. |
| validation | **validación** | Distinguish validation set from scientific validation. |
| accuracy | **exactitud** | Never *precisión* for this metric. |
| precision | **precisión** | Reserve for the metric. |
| recall | **exhaustividad**, **sensibilidad**, or **recall** depending domain | Formula and domain must decide; current universal mandate is too rigid. |
| dataset | **conjunto de datos** | *Dataset* may appear in names or a first bilingual bridge. |
| notebook | **notebook de Jupyter (cuaderno)** first; then choose one per chapter | UNAM materials themselves use *notebooks*; do not label that usage culturally wrong. |
| baseline | **modelo de referencia (*baseline*)** first | Either form thereafter, consistently. |
| benchmark | **benchmark**, **banco de pruebas**, **conjunto de referencia** by function | Avoid automatic *punto de referencia*. |
| random forest | **bosque aleatorio** | Stable. |
| deep ensemble | **ensamble profundo** | Pan-Americanly intelligible; avoid French *ensemble*. |

### 3.3 Scientific-computing Spanish

| English | Recommended Spanish | Notes |
|---|---|---|
| workflow | **flujo de trabajo**; **procedimiento/metodología** for conceptual research workflow | More natural than in French, but still context-sensitive. |
| pipeline | **cadena de procesamiento**, **flujo**, or *pipeline* | Preserve `Pipeline`; gloss technical *pipeline* once. |
| repository | **repositorio Git/GitHub** | Never *archivo* for active Git repository. |
| archive | **archivo** / **archivo de datos** | Preservation/collection. |
| environment | **entorno** | Stable. |
| package | **paquete** | Stable. |
| dependency | **dependencia** | Stable. |
| cloud | **nube**, **computación en la nube** | Product names such as Google Cloud unchanged. |
| HPC | **HPC (cómputo de alto rendimiento)** | *Computación* is acceptable too; choose one house form. |
| cluster | **clúster de cómputo** | More natural than forcing *conglomerado* in compute. |
| job | **trabajo**, **tarea**, or *job* according to scheduler culture | Explain scheduler terminology once. |

---

## 4. Spanish full-book language audit

### Overall assessment

The Spanish edition is consistent, clear, and more restrained than many technical translations. It successfully avoids systematic Spain-only forms such as *ordenador*. Its remaining risks are over-standardizing terms that are genuinely regional or domain-specific and retaining *pipeline* without an explicit glossary policy.

### Chapter-level findings

| Section | Assessment | Concrete action |
|---|---|---|
| Front matter | Strong and accessible | Introduce ML acronym explicitly; avoid implying English jargon is merely tolerated. |
| Chapter 1 | Good international register | Keep **repositorio** for Git and **archivo** for preservation; preserve UI labels and product names. |
| Chapter 2 | Strong; inherits English source defects | Re-translate only affected cells after English cleanup; add contextual `pipeline` rule. |
| Chapter 3 | Good metric discipline | Keep exactitud/precisión distinction; allow domain note for sensibilidad/exhaustividad/recall; normalize pipeline introduction. |
| Chapter 4 | Strong | Preserve `input`, `target`, and API terms; explain them in prose rather than translating code. |
| Chapter 5 | Good, culturally localized compute section | Keep flujo de trabajo in general prose; preserve path `/workflows/`; define HPC and regional allocation systems. |
| Chapter 6 | Contemporary and clear | Keep LLM; consider **modelo de lenguaje de gran tamaño** alongside current “gran escala,” because both circulate. |
| Chapter 7 | Strong applied register | Continue multi-country review of hazard and institutional examples. |
| Glossary | Useful but some universal claims are too strong | Make recall, cluster, notebook, score, and transformer explicitly audience/domain dependent. |

### Concrete Spanish replacement queue

| Priority | Location | Current issue | Proposed action |
|---|---|---|---|
| P0 | `translations/GLOSSARY.md`, notebook row | native form presented as sole written default | allow `notebook de Jupyter (cuaderno)` for research-facing prose |
| P0 | `translations/GLOSSARY.md`, recall row | `exhaustividad` treated as edition-wide answer | document domain split: exhaustividad/sensibilidad/recall |
| P0 | `translations/GLOSSARY.md`, cluster row | `conglomerado` universal | allow grupo/clúster/conglomerado by field; keep clúster for computing infrastructure |
| P1 | Chapter 3.10, cells 10/16/17 | repeated ungoverned *pipeline* | define *pipeline* once as cadena de modelado/procesamiento, then retain if desired |
| P1 | Chapter 3.9, cells 9/16 | *pipeline* wrapping sklearn objects | write “*pipeline* de scikit-learn (`Pipeline`)” first |
| P1 | Chapter 4.5, cell 22 | `pipelines (flujos de procesamiento)` | prefer **cadenas de procesamiento** in signal/data ingestion context |
| P1 | Chapter 4.6, cells 25/39 | operational `pipeline` | **cadena operativa de procesamiento/detección** where the object is not an API class |
| P2 | Chapter 4.6, cells 6/10 | `(input, target)` explained correctly | preserve identifiers; consider typographic code formatting consistently |

---

## 5. English source editorial audit

Translations should not become a patch layer over weak English. Fix the English source first, then re-run translation staleness checks and update only affected prose cells.

### High-confidence corrections

| Priority | Location | Current text/problem | Proposed replacement |
|---|---|---|---|
| P0 | `book/Chapter2-DataManipulation/2.5_Arrays.ipynb`, markdown cell 32 | “Similarity is **equialent**…”; “**Euclidian** distance” | “Similarity corresponds to proximity…”; **Euclidean distance** |
| P0 | same, markdown cell 34 | “Xarrays allows to attach metadat”; “data arras”; “Xarrrays” | “Xarray lets users attach metadata…”; “data arrays”; “Xarray datasets/objects” |
| P0 | same, markdown cell 87 | “refered” | **referred** |
| P0 | `book/Chapter2-DataManipulation/2.3_pandas_rendered.ipynb`, markdown cell 5 | “Pandas are composed of…” | “The core pandas objects are `Series` and `DataFrame`.” |
| P0 | same, markdown cell 68 | “costum functions” | **custom functions** |
| P0 | `book/Chapter3-MachineLearning/3.3_clustering.ipynb`, markdown cell containing raw line 918 | “It consists in performing…” | “It consists of running…” or “It runs the clustering algorithm…” |
| P1 | `book/Chapter1-GettingStarted/1.2_jupyter_environment.md`, line 9 | “merges … in a single document”; “composed of cells” | “combines … in a single document”; “consists of cells” |
| P1 | `book/Chapter2-DataManipulation/2.5_Arrays.ipynb`, cell 32 | “distance … between the two data”; “proximity of two data” | “distance between two observations/arrays”; “proximity of the observations” |
| P1 | same, cell 34 | long run-on paragraph; inconsistent Xarray singular/plural | split into short paragraphs; treat **Xarray** as package and `DataArray`/`Dataset` as objects |
| P1 | `book/Chapter2-DataManipulation/2.3_pandas_rendered.ipynb`, cell 5 | says `Series` are “columns with attributes or keys” | define `Series` accurately as one-dimensional labeled arrays |
| P1 | filenames `2.20_Final_Project_Assignement.md` and `mlgeo_4.20_final_project_assignement.md` | **Assignement** misspelling | rename to **Assignment** with MyST redirect/alias so external links do not break |

### Editorial policy for the source

- Use singular **data** consistently only if that matches the book's selected style; otherwise do not alternate randomly between singular and plural agreement.
- Prefer direct verbs: “lets users attach,” not “allows to attach.”
- Use package/product capitalization consistently: **pandas**, **NumPy**, **Xarray**, **NetCDF**, **Git**, **GitHub**.
- Define mathematical terms precisely before pedagogical shorthand.
- Remove executed-image/base64 output from spellchecking and language linting; scan Markdown cells, captions, and code comments separately.
- Never use blind spell correction on code identifiers (`Alow`, domain acronyms, station codes, etc.).

---

## 6. Cross-language ML/statistics terminology audit

This table should become the semantic core of the trilingual glossary. The “concept” column prevents different English words from collapsing into one translated word.

| Concept | English/code-facing term | French research-textbook form | International Spanish form | Invariant |
|---|---|---|---|---|
| Field | machine learning, ML | machine learning (ML; apprentissage automatique) | aprendizaje automático (machine learning, ML) | Teach ML acronym. |
| Input predictor | feature | variable/caractéristique; feature bridge | característica/variable explicativa | Preserve names such as `n_features`. |
| Response | target | variable cible | variable objetivo | Do not call every target a label. |
| Class annotation | label | étiquette/classe | etiqueta/clase | Preserve `y`, `labels`. |
| Parameter estimation | fitting | ajustement | ajuste | Preserve `.fit()`. |
| Learning phase | training | entraînement | entrenamiento | Preserve `train()`. |
| Deployment phase | inference | inférence/prédiction | inferencia/predicción | Choose by meaning, not word matching. |
| Unsupervised task | clustering | clustering/partitionnement | agrupamiento/clustering | Resulting groups need context-specific names. |
| Discrete prediction | classification | classification | clasificación | Stable. |
| Continuous prediction | regression | régression | regresión | Stable. |
| Per-example error | loss | perte | pérdida | Separate from omission/miss. |
| Optimized aggregate | objective | fonction objectif | función objetivo | May include regularization. |
| Real-world consequence | cost | coût | costo/coste | Cultural/domain stakes, not synonym for loss. |
| Model-development subset | validation set | ensemble de validation | conjunto de validación | Not final test. |
| Final held-out subset | test set | ensemble de test | conjunto de prueba | Never use repeatedly for tuning. |
| Correct fraction | accuracy | exactitude/taux de bonnes classifications | exactitud | Never precision/précision. |
| Positive predictive value | precision | précision | precisión | Formula controls meaning. |
| Sensitivity | recall | rappel/sensibilité | exhaustividad/sensibilidad/recall | Domain controls preferred label. |
| Information contamination | data leakage | fuite de données | fuga de datos | Keep separate from spectral leakage. |
| Spectral phenomenon | spectral leakage | fuite spectrale | fuga espectral | Always qualify. |

Recommended addition: every metric entry should include its formula or a link to the first formula-bearing lesson. Translation alone cannot disambiguate `precision`, `accuracy`, `recall`, `skill`, and `score` across disciplines.

---

## 7. Scientific-computing terminology audit

### Decision table

| Context | English | French | Spanish | Preserve literally? |
|---|---|---|---|---|
| Active version control | repository | dépôt Git/GitHub | repositorio Git/GitHub | Repository name/path yes. |
| Long-term preservation | archive | archive | archivo | Service/product name yes. |
| General research method | workflow | démarche/protocole | metodología/procedimiento | No automatic one-word mapping. |
| Automated computation | workflow | workflow de calcul | flujo de trabajo computacional | Paths/API unchanged. |
| Signal/data transforms | pipeline | chaîne de traitement | cadena de procesamiento | `Pipeline` unchanged. |
| Interactive document | notebook | notebook Jupyter (carnet) | notebook de Jupyter (cuaderno) | `.ipynb`, filenames unchanged. |
| Software isolation | environment | environnement | entorno | `environment.yml` unchanged. |
| Installable unit | package | paquet | paquete | Package name unchanged. |
| Required software relation | dependency | dépendance | dependencia | Config key unchanged. |
| Remote infrastructure | cloud | cloud/informatique cloud | nube/computación en la nube | AWS/GCP/Azure unchanged. |
| Parallel infrastructure | HPC | HPC (calcul haute performance) | HPC (cómputo de alto rendimiento) | HPC unchanged. |
| Scheduler execution | job | job/tâche | trabajo/tarea/job | Command and scheduler fields unchanged. |
| Reproducible resolution | lockfile | fichier de verrouillage | archivo de bloqueo | Filename unchanged. |

### Code and API preservation rules

Never translate:

- Python/R/Julia identifiers, function names, class names, arguments, exceptions, module paths, and shell commands;
- filenames and paths such as `.github/workflows/build.yaml`, `environment.yml`, `pixi.lock`, and `notebooks/*.ipynb`;
- JSON/YAML/TOML keys unless the format explicitly permits localized display text;
- dataframe columns used by code, including `target`, `label`, `uncertainty`, or `pick_weight`;
- printed model reports and executed outputs unless the underlying code is deliberately localized and re-executed;
- established product/project names: GitHub, Jupyter, NumPy, pandas, scikit-learn, PyTorch, Zarr, Dask, MyST, Zenodo, HAL.

The correct teaching pattern is: “la variable cible `target`,” “la variable objetivo `target`,” not a renamed column that breaks the notebook.

---

## 8. Translation consistency and maintenance workflow

### 8.1 Keep the existing technical invariants

The repository already has a strong foundation:

- source commit recorded per translated page;
- cell-preserving notebook translation;
- CI checks for table-of-contents and manifest coverage;
- code/output invariance;
- explicit exceptions for localized data notebooks.

Keep these controls. The current audit confirmed they pass.

### 8.2 Add linguistic QA that understands notebooks

Do not run ordinary spellcheck over raw `.ipynb` JSON. It produces false positives from base64 images, binary-like outputs, model reports, station codes, and identifiers. Add a tool that extracts and labels:

1. Markdown prose cells;
2. code comments;
3. user-facing string literals;
4. code identifiers;
5. executed outputs.

Apply different rules to each channel. Only prose should receive general spelling and style checks. Identifier drift should be an error. User-facing strings may be translated only if the code cell is intentionally localized and re-executed.

### 8.3 Replace rigid glossary enforcement with contextual linting

Useful automatic errors:

- `accuracy` translated as *précision/precisión* in a metric context;
- Git repository translated as *archive/archivo*;
- `.fit`, `Pipeline`, or `target` identifier changed inside code;
- bare *fuite/fuga* where data or spectral leakage is meant;
- English UI labels translated even though screenshots/interface remain English;
- mixed *ordenador/computadora* within one Spanish edition.

Do **not** automatically error on:

- *machine learning*, ML, workflow, pipeline, notebook, cloud, cluster, baseline, or benchmark in French;
- *machine learning*, ML, pipeline, notebook, recall, score, or dataset in Spanish;
- legitimate domain variants such as rappel/sensibilité or exhaustividad/sensibilidad.

Instead, require first-use glossing and within-chapter consistency.

### 8.4 Human review structure

Replace “persona-reviewed” as a quality claim with a traceable review record:

- reviewer name or declared anonymous reviewer ID;
- language variety and country/region;
- discipline and career context;
- chapters reviewed;
- date and source commit;
- contested terms accepted/rejected;
- conflicts disclosed.

Recommended panels:

- French: one ML/computer-science lecturer, one geoscience lecturer, one research software/HPC practitioner, one graduate student.
- Spanish: at least four regions (Mexico/Central America, Andean region, Southern Cone, Spain), plus one research-software practitioner.
- Cross-language: one statistics/verification specialist to audit accuracy/precision/recall/calibration/skill terminology.

Fictional personas can remain as design aids, but should be labeled explicitly as synthetic and never substituted for community review.

### 8.5 Proposed CI sequence

1. English source edit merged.
2. Manifest tool flags stale translations.
3. Translator edits prose/Markdown cells only.
4. Structural checker verifies code cells, outputs, links, and manifests.
5. Terminology linter checks high-risk semantic rules.
6. French and Spanish builds run.
7. Native subject reviewer signs off on changed chapters.
8. Changelog records terminology decisions and rejected alternatives.

### 8.6 Acceptance criteria

- [ ] Rewrite French ML policy around *machine learning/ML* as actual research usage, with *apprentissage automatique* retained as a pedagogical/formal equivalent.
- [ ] Add context-sensitive rows for workflow, démarche, chaîne de traitement, pipeline, repository, dépôt, archive, notebook, cloud, HPC, job, and build.
- [ ] Correct the French loss/cost/objective passages.
- [ ] Relax over-prescriptive cluster, recall, notebook, and pipeline rules in both languages.
- [ ] Fix the high-confidence English Chapter 2 errors before updating translations.
- [ ] Correct `Assignement` filenames with redirects or aliases.
- [ ] Document non-translated bibliography/leaderboard/slides scope.
- [ ] Keep `tools/check_translations.py` passing for both languages.
- [ ] Add notebook-aware prose linting and semantic terminology checks.
- [ ] Record real human subject-matter review; label fictional personas as synthetic.
- [ ] Preserve every code/API identifier unless an explicitly localized notebook is re-executed and documented.

---

## Suggested implementation order

### PR 1 — English source cleanup

Fix the high-confidence Chapter 2/3 editorial issues and assignment filenames. This prevents translations from deliberately preserving bad source prose.

### PR 2 — Audience-aware glossary rewrite

Update `translations/GLOSSARY.md`, `GLOSSARY_fr.md`, `GLOSSARY_es.md`, both language glossaries, and `translations/README.md`. Do not mass-replace chapter prose yet.

### PR 3 — Targeted French consistency pass

Apply the concrete French queue, especially ML register, workflow/pipeline distinctions, notebook register, and loss/cost/objective semantics.

### PR 4 — Targeted Spanish consistency pass

Apply the pipeline, notebook, recall, cluster, and international-register changes.

### PR 5 — Maintenance tooling

Add notebook-aware linguistic linting, contextual terminology rules, documented review provenance, and an explicit shared-assets policy.

## Final assessment

The editions are technically complete and structurally trustworthy. They should not be rewritten wholesale. The next revision should be a targeted, audience-aware editorial pass that treats scientific language as a community practice rather than a dictionary substitution exercise.

The central French change is decisive: for this higher-education and research audience, **ML / machine learning is not an error to tolerate**. It is a normal working term that should be taught alongside **apprentissage automatique**, while the excellent French *apprentissage supervisé/non supervisé/profond* family remains in use. The same cultural principle should govern *workflow*, *pipeline*, *notebook*, *cloud*, and *cluster*: preserve the jargon researchers actually need, explain it, and translate only where translation improves precision.
