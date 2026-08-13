# Trilingual glossary — English / Français / Español

One table, one order, three languages. Every term in the book's English
glossary (`book/reference/glossary.md`) appears here, plus the terms the
French and Spanish translation passes had to settle along the way.

**What this document is.** A usage guide for translators and readers, not an
authority that outranks the chapters. It reports how francophone and
hispanophone researchers write about these concepts, and it tells a translator
which form to pick *in which context*. Where a chapter and this table
disagree, the chapter is usually reporting real usage and the table is the
thing to fix — that is how most of the rows below got rewritten. Open an issue
`[translation] term: …` rather than silently rewriting chapter prose to match
a row you disagree with.

**English is authoritative for code.** Function names, arguments, library
documentation, error messages and the literature are all in English, and the
book never translates them. What the French and Spanish columns fix is the
*prose*: the word a reader meets in a sentence.

**One concept in one setting, not one word for one word.** Several English
words in this book cover two or three different concepts — *workflow*,
*pipeline*, *repository*, *cluster*, *notebook*, *build* — and collapsing them
into a single translated word destroys a distinction the reader needs. Those
terms get **one row per sense**, with the sense named in the first column.
Read the sense, then the cell.

**Reading the cells.**

| Form | Meaning |
|---|---|
| `terme` | The community writes this. It is the working term in that language. |
| `*term* (glose)` | The community keeps the **English** word in prose. The parenthesis is the gloss given once at the first occurrence of each chapter — the reader gets both the working term and its meaning. |
| `terme †` | **Register split, not a ban.** Both forms are correct and both circulate; the native term is the written default in explanatory prose, the English term is normal in speech, seminars, code comments, and compact technical prose. Gloss once per chapter, then stay consistent within the chapter. |
| **Error** in Notes | A hard invariant. Not a preference — getting it wrong changes what the sentence means. These are the rows a terminology linter should fail on. |
| ⚑ in Notes | Contested. The row states what is in dispute and who should rule on it. Open an issue `[translation] term: …`. |

The two languages do **not** keep the same English words, and that difference
is the most useful thing in this table. French francizes *autoencoder* into
« auto-encodeur » while Spanish prose runs on *autoencoder*; Spanish
translates *test set* into «conjunto de prueba» while French keeps the noun in
« ensemble de test ». Neither community is being sloppy — each is reporting
what its own ML and geoscience people actually say. Both keep *dropout*,
*gradient boosting*, *transformer* and *prompt*, which is a real consensus and
worth trusting.

**The two editions differ on the field name on purpose.** French prose in this
book says *machine learning* / **ML** for the field and keeps the
« apprentissage… » family for the named paradigms; Spanish prose says
«aprendizaje automático» and teaches **ML** as the acronym everyone will meet
in code, slides and papers. That is not an inconsistency to be normalized
away. It is what the two research communities do, and the *Machine learning*
row below gives the evidence for each.

**Metrics carry their formula.** Translation alone cannot disambiguate
*accuracy*, *precision*, *recall*, *skill* or *score* across statistics,
meteorological verification, seismology and medicine — the same French or
Spanish word serves different quantities in different fields. Every metric row
states the quantity, so a reader can check which one is meant before choosing
a word.

*FR — Ce tableau est un guide d'usage, pas une autorité qui prime sur les
chapitres : il donne, pour chaque terme anglais et pour chaque contexte
d'emploi, la forme qu'emploierait un chercheur francophone, et la forme
espagnole en regard. L'anglais fait foi dans le code. Les termes en italique
sont ceux que la communauté conserve en anglais dans la prose, glosés à la
première occurrence de chaque chapitre. Plusieurs lignes ont été scindées par
sens : un « workflow » de calcul, une démarche de recherche et un fichier
GitHub Actions ne se disent pas de la même manière.*

*ES — Esta tabla es una guía de uso, no una autoridad por encima de los
capítulos: da, para cada término en inglés y para cada contexto, la forma que
usaría una investigadora hispanohablante, y al lado la forma francesa. El
inglés manda en el código. Los términos en cursiva son los que la comunidad
conserva en inglés en la prosa, glosados en su primera aparición de cada
capítulo. Varias filas se dividieron por sentido: un flujo de trabajo
computacional, una metodología de investigación y un archivo de GitHub Actions
no se dicen igual.*

Detailed, translator-facing notes and the history of contested rows live in
[`GLOSSARY_fr.md`](GLOSSARY_fr.md) and [`GLOSSARY_es.md`](GLOSSARY_es.md).

---

| English term | Français | Español | Notes |
|---|---|---|---|
| Ablation | ablation | ablación | |
| Accuracy | exactitude — aussi « taux de bonnes classifications » | exactitud | `(TP+TN)/(TP+TN+FP+FN)` — the fraction of predictions that are correct. **Error:** never « précision » / «precisión», which are reserved for `TP/(TP+FP)`. A false friend that silently swaps one metric for another; flagged as disqualifying by both the Chilean and the UNAM reviewers. First formula-bearing lesson: 3.4. |
| Active learning | apprentissage actif | aprendizaje activo | |
| Agent | agent | agente | Chapter 6 vocabulary; the word survives unchanged in all three. |
| AI-ready data | données prêtes pour l'IA | datos listos para IA | FR uses the French acronym IA for AI throughout. |
| Aliasing | repliement de spectre † | solapamiento espectral † | Both write the native term; *aliasing* stays current in lab speech. |
| Archive (preservation) | archive ouverte, archive institutionnelle, archive de données | archivo, archivo de datos | The *deposit-and-preserve* sense: HAL, Zenodo, a long-term data archive. Distinct from the Git sense below — see *Repository — Git / GitHub*. HAL describes itself as an « archive ouverte multidisciplinaire », which is where the French word belongs. Service names (HAL, Zenodo) unchanged. |
| Attribute | attribut | atributo | ES: «atributo» is reserved for *attribute* and nothing else — it is what keeps the *feature* row below workable. |
| Autoencoder | auto-encodeur | *autoencoder* (autocodificador) | Clean divergence. FR francizes with a hyphen and never looks back; ES prose (chapter 4.6) runs on the English noun with «autocodificador» as the gloss. |
| Baseline | *baseline* (modèle de référence) | *baseline* (modelo de referencia) | Researchers say *baseline*. Gloss once per chapter, then use either form consistently within the chapter. Not a term to hunt down and replace. |
| Batch | lot † | lote † | *batch* current in code-speech in both. |
| Benchmark | *benchmark* — glose selon la fonction : « jeu de référence » (données) ou « banc d'essai » (protocole de comparaison) | *benchmark* — glosa según la función: «conjunto de referencia» (datos) o «banco de pruebas» (protocolo) | Two different objects wear the same English word. A benchmark **dataset** is not a *banc d'essai* / *banco de pruebas*; a benchmark **suite or protocol** is. Choose by what the sentence points at. ES: avoid the automatic «punto de referencia», which reads as a landmark. |
| Brier score | score de Brier | score de Brier | Mean squared error of a probabilistic forecast against the binary outcome: `mean((p − o)²)`, `o ∈ {0,1}`. Lower is better. First formula-bearing lesson: 3.6. ⚑ ES: «puntaje de Brier» circulates, but professional verification speech says «score». FR is settled Météo-France usage. The ES verification reviewer should rule once, together with the CRPS and Score rows. |
| Build (CI, packaging) | *build* † | *build*, «compilación» † | Practitioner form. FR: keep *build* in CI and research-engineering speech; use « construction (du livre) » when the sentence is about publishing the book itself, not about a CI step. ES: *build* in CI prose, «compilación» when describing the published output. Gloss once; do not force one form on both senses. |
| Calibration | calibration | calibración | FR: « étalonnage » belongs to instruments, never to probabilities. |
| Classification | classification | clasificación | |
| Climatology (reference) | climatologie de référence | valor climatológico de referencia | Operational verification vocabulary; native in both. |
| Cloud (infrastructure) | *cloud*, « informatique cloud » | «nube», «computación en la nube» | Practitioner form. FR: « informatique en nuage » is correct French but is not what this audience says; do not impose it. Product names (AWS, Google Cloud, Azure) unchanged in both. |
| Cluster — ML, the resulting group | « groupe », « classe », « partition » selon la méthode ; *cluster* en prose technique | «grupo», «conglomerado», «clúster» según la comunidad | ⚑ Changed: « grappe » and «conglomerado» were previously given as the universal defaults, and neither travels. « Grappe » is not current in modern francophone ML prose; «conglomerado» is statistics-flavoured and does not read the same way across regions. Pick by method and by what the surrounding sentence calls the objects, and stay consistent within a chapter. FR/ES teaching reviewers to rule. |
| Cluster — compute infrastructure | « cluster de calcul » (« grappe de calcul » possible) | «clúster de cómputo» | A different object from the ML row above, and the reader must not have to guess which one is meant. Name it « de calcul » / «de cómputo» on first use in a chapter that also does clustering. |
| Clustering (the task) | *clustering* ou « partitionnement », définis ensemble | «agrupamiento» / *clustering* | ⚑ Relaxed: *clustering* is normal in francophone and hispanophone ML speech and writing. Introduce both once — « le *clustering* (partitionnement) » / «el agrupamiento (*clustering*)» — then use one consistently within the chapter. Do not re-gloss at every occurrence, and do not mandate a purist single substitute. The Paris-Saclay ML reviewer should still rule on FR. |
| Coda | coda | coda | Seismological term, unchanged in all three. |
| Computer | ordinateur | computadora | ES fixes the American variant pan-regionally; never alternate with «ordenador» inside one edition. FR has no such split to manage. |
| Convolutional neural network (CNN) | réseau de neurones convolutif — sigle CNN | red neuronal convolucional — sigla CNN | The acronym stays English in both. |
| Corner frequency | fréquence coin | frecuencia de esquina | |
| Cost | coût | costo / coste | The real-world consequence attached to an error — what a missed alert or a false evacuation costs. **Error:** not a synonym for *loss* or *objective*; see those rows. The word is not banned, the conflation is. |
| Coverage (of intervals) | couverture empirique des intervalles | cobertura empírica de los intervalos | Météo-France and SMN both use the empirical-coverage phrasing. |
| Cross-validation | validation croisée | validación cruzada | |
| CRPS | score de probabilité classé continu — sigle CRPS | puntaje de probabilidad clasificada continua — sigla CRPS | `∫ (F(x) − 1{x ≥ y})² dx` — the integrated squared distance between the forecast CDF and the step function at the observation; it reduces to absolute error for a deterministic forecast, which is what makes it comparable across the two. First formula-bearing lesson: 4.10. ⚑ ES: the shipped text says «puntaje», but the «score de Brier» precedent in the same glossary argues for «score de probabilidad clasificada continua». One of the two has to give. ES verification reviewer (SMN/atmospheric-science persona) to rule. The acronym CRPS is kept in both. |
| Data card | fiche de données | ficha de datos | |
| Data leakage | fuite de données | fuga de datos | **Error** if left unqualified. Always « de données » / «de datos»; a bare « fuite » / «fuga» collides with spectral leakage, a completely unrelated phenomenon that this book also teaches. See *Spectral leakage*. |
| Dataset | « jeu de données » ; *dataset* admis en prose technique et dans les noms | «conjunto de datos»; *dataset* admitido en nombres y en un puente bilingüe inicial | The native phrase is the prose default in both. But *dataset* is ecosystem jargon researchers genuinely use (`Dataset`, HF datasets, Xarray `Dataset`), and product or class names are never translated. Do not force the English form out of a sentence that is naming an object. |
| Deep ensemble | ensemble profond | ensamble profundo | ⚑ FR: « ensemble de réseaux » still has defenders. ES fixes «ensamble», not the French-looking «ensemble». |
| Deep learning | apprentissage profond | aprendizaje profundo | A **named paradigm**, and the French and Spanish forms are the natural ones — this is exactly the family the *Machine learning* row keeps in the native language even though the field name itself is often English. *deep learning* is fine at a first mention or in a course title (CNRS titles a course « Machine Learning et Deep Learning pour la vision par ordinateur »), but the paradigm name in running prose is « apprentissage profond » / «aprendizaje profundo». |
| Dimensionality reduction | réduction de dimension | reducción de dimensionalidad | |
| Dropout | *dropout* (extinction aléatoire de neurones) | *dropout* (apagado aleatorio de neuronas) | Kept English in both. The glosses differ; the working word does not. |
| Embedding | plongement | *embedding* (incrustación) | ⚑ ES: the shipped chapters use «incrustación» and *embedding* in different places — not settled. FR « plongement » is stable, though persona 06 warns that unglossed « plongement lexical » reads as translator jargon. |
| Ensemble forecast | prévision d'ensemble | pronóstico por ensambles | Established operational vocabulary in both (Météo-France, SMN). |
| Epistemic vs. aleatoric uncertainty | incertitude épistémique / aléatoire | incertidumbre epistémica / aleatoria | |
| Epoch | époque | época | |
| Eval set | jeu d'évaluation | conjunto de evaluación | Chapter 6 vocabulary; both communities translate it rather than keeping *eval set*. |
| Expected calibration error (ECE) | erreur de calibration attendue — sigle ECE | error de calibración esperado — sigla ECE | Bin the predictions by confidence, then average `abs(accuracy(b) − confidence(b))` weighted by bin size. The number depends on the binning, so state it. First formula-bearing lesson: 4.5. |
| F1 score | score F1 | puntaje F1 / score F1 | `2·P·R/(P+R)` — the harmonic mean of precision and recall, so it moves only when both do. Keep the name `F1` unchanged; it is a label, not a word. ⚑ ES: inherits the «puntaje» vs «score» question from the *Score* row. |
| False alarm | fausse alarme | falsa alarma | Pairs with *miss*; native in both, no dispute. |
| Feature (model input) | « variable », « variable explicative », ou « caractéristique » selon qui parle | «característica» o «variable explicativa» | Choose by discipline, not by house rule. A statistician writing about a regression says « variable explicative » / «variable explicativa»; a deep-learning chapter talking about learned representations says « caractéristique » / «característica». Use *feature* parenthetically where it bridges to code. **Error:** ES must not use «atributo», which is reserved for *attribute*. Identifiers such as `n_features`, `feature_names` unchanged. |
| Feature engineering | *feature engineering* (construction de variables / de caractéristiques) | *feature engineering* (ingeniería de características) | Gloss once per chapter, then either form. « Ingénierie des caractéristiques » is comprehensible but reads as translated; the discipline-dependent wording of the *Feature* row applies to the gloss too. |
| Fine-tuning | affinage (*fine-tuning*) | ajuste fino (*fine-tuning*) | ⚑ FR: « affinage » versus « réglage fin » is unresolved; the book uses « affinage ». English tolerated and glossed in both. |
| Flicker noise | bruit de scintillement (bruit en 1/f) | ruido de parpadeo | |
| Fold | pli | pliegue | ⚑ Both are correct and both are rare in speech. FR alternative « bloc de validation », ES «partición de validación». Each edition's teaching reviewer should rule. |
| Git | Git | Git | Product name, unchanged. |
| GitHub | GitHub | GitHub | Product name, unchanged. |
| Gradient boosting | *gradient boosting* (renforcement par gradient) | *gradient boosting* (potenciación de gradiente) | Kept English in both — one of the clearest agreements in the table. |
| Gradient descent | descente de gradient | descenso de gradiente | |
| Ground truth | vérité terrain | verdad de referencia | FR uses the compact « vérité terrain »; ES needs the fuller phrase. |
| Grouped cross-validation | validation croisée par groupes | validación cruzada por grupos | |
| Heteroscedastic noise | bruit hétéroscédastique | ruido heterocedástico | |
| Hidden test set | ensemble de test caché | conjunto de prueba oculto | Course vocabulary (leaderboard, 4.10). |
| HPC | « HPC (calcul haute performance) » à la première occurrence, puis HPC | «HPC (cómputo de alto rendimiento)», luego HPC | Keep the acronym — it is what allocation calls, schedulers and centres use. Gloss once per chapter. |
| Hyperparameter | hyperparamètre | hiperparámetro | |
| Instrument response | réponse instrumentale | respuesta instrumental | FR keeps *counts* in English for the recorded units. |
| Interpolation vs extrapolation | interpolation / extrapolation | interpolación / extrapolación | |
| Job (scheduler, CI) | *job* † ou « tâche » | *job* / «trabajo» / «tarea» † | Practitioner form. HPC users say *job*, and the scheduler's own fields and commands (`sbatch`, `job id`) are never translated. Use « tâche » / «tarea» when the sentence is about work in general rather than a submitted scheduler job. Explain the scheduler vocabulary once. |
| Jupyter notebook | notebook Jupyter (carnet) | notebook de Jupyter (cuaderno) | Changed: « carnet » / «cuaderno» were the sole written defaults, which does not match research-facing prose in either language. Give the gloss once per chapter, then use *notebook*. `.ipynb`, filenames and paths unchanged; the book's own filenames are English. |
| Label | étiquette | etiqueta | |
| Large language model (LLM) | grand modèle de langage — sigle LLM | modelo de lenguaje de gran escala — sigla LLM | The acronym LLM is kept in both; only the expansion is translated. |
| Lead time | échéance | horizonte de pronóstico | ⚑ ES: «plazo de pronóstico» circulates. FR « échéance » is settled Météo-France usage — a single word where ES needs a phrase. |
| Leaderboard | classement † | tabla de clasificación † | *leaderboard* tolerated and glossed in both. |
| Leakage (spatial and temporal) | fuite spatiale / temporelle | fuga espacial / temporal | Always qualified. See *data leakage* and *spectral leakage*. |
| Learning rate | taux d'apprentissage | tasa de aprendizaje | |
| Linear probe | sonde linéaire | sonda lineal | FR: **not** « sondage linéaire » — that is hash-table linear probing, an unrelated computing term. « Sonde » is the instrument metaphor the book means. |
| Loss function | fonction de perte | función de pérdida | The error on **one example**. **Error:** do not write « fonction de coût » / «función de costo» for it, and do not treat *loss*, *objective* and *cost* as three words for one thing — the book distinguishes all three; see those rows. A tree's split criterion is a « critère de partition », not a loss. ES: this row is exactly why *miss* below cannot be «pérdida». |
| Machine learning | *machine learning* (**ML**) — « apprentissage automatique » comme glose ; la famille « apprentissage… » pour les paradigmes | **aprendizaje automático** (*machine learning*, **ML**) | **The two editions differ on purpose.** FR — first substantive occurrence per chapter: « le *machine learning* (**ML** ; apprentissage automatique) » ; thereafter **ML** or *machine learning* for the field and for research practice, and the French « apprentissage… » family for the named paradigms (*apprentissage supervisé, non supervisé, auto-supervisé, par renforcement, profond*). This is a **register decision, not a claim that « apprentissage automatique » is wrong**: it is correct, current, and kept as the gloss. Evidence for the mixed register: CNRS Formation Entreprises titles a course « Machine Learning et Deep Learning pour la vision par ordinateur » while its description body writes « apprentissage automatique » and « apprentissage profond »; Université Paris-Saclay's M1 *Mathématiques et Apprentissage Statistique* page mixes untranslated *machine learning* and *data science* with « apprentissage statistique »; Inria publishes in French both « Allier mathématiques et machine learning… » and « Apprentissage automatique et réseaux d'information »; the Collège de France chair biography writes « l'apprentissage automatique » (*machine learning*) — the French form paired with the English field name. ES — first occurrence: «aprendizaje automático (*machine learning*, ML)»; thereafter «aprendizaje automático» in explanatory prose and **ML** in compact technical prose, diagrams, tables and references to the ecosystem. Spanish does **not** need the English field name as its prose default, but it must teach the acronym, which is what students meet in code, slides and papers. CONICET's own Spanish page writes «Aprendizaje Automático (Machine Learning en inglés)». |
| MASE | MASE — erreur absolue moyenne mise à l'échelle | MASE — error absoluto medio escalado | Acronym kept, expansion translated once. |
| MC dropout (Monte Carlo dropout) | *MC dropout* | *MC dropout* | Kept English in both, following *dropout*. |
| Miss (error type) | manqué — « les événements manqués » | omisión | ES: **not** «pérdida», which would collide with «función de pérdida». FR nominalizes the participle (« les manqués »). Pairs with *false alarm*; both editions must name the two error directions separately when discussing who bears the cost. |
| Model card | fiche de modèle | ficha de modelo | Parallel to *data card* in both. |
| MyST | MyST | MyST | Product name; expansion *Markedly Structured Text* kept in English. |
| Neural network | réseau de neurones | red neuronal | |
| Object store | « stockage objet » | «almacenamiento de objetos» | S3, MinIO, cloud buckets. **Error:** not an *archive* / « archive » / «archivo» unless preservation is genuinely the function. FR: « magasin d'objets » exists but « stockage objet » is the general-prose form. |
| Objective function | fonction objectif | función objetivo | What the optimizer actually minimizes: the aggregated loss **plus** any regularization terms. **Error:** not interchangeable with *loss*. When a passage says « perte, coût ou objectif » as if the three were synonyms, that is the defect — write the hierarchy instead. |
| Overfitting | surapprentissage | sobreajuste | ⚑ FR: « surajustement » exists and has defenders; the book fixes « surapprentissage ». ES «sobreajuste» is uncontested. |
| Persistence (forecast) | persistance | persistencia | Météo-France and SMN usage; never renamed with a neologism. |
| Physics-informed neural network (PINN) | réseau de neurones informé par la physique — sigle PINN | red neuronal informada por la física — sigla PINN | |
| Pipeline — signal / data transforms | « chaîne de traitement » | «cadena de procesamiento» | An ordered sequence of transformations applied to data: detrend, taper, filter, resample. FR: **not** « démarche », which is conceptual, and not « flux de travail ». When the object is a modelling sequence rather than a signal chain, « chaîne de modélisation » / «cadena de modelado». |
| Pipeline — scikit-learn object | *pipeline* — `Pipeline` inchangé | *pipeline* — `Pipeline` sin traducir | The preprocessing-plus-estimator object. Write « un *pipeline* scikit-learn (`Pipeline`) » / «un *pipeline* de scikit-learn (`Pipeline`)» on first occurrence in a chapter, then *pipeline*. **Error:** never translate the class name. This is a genuine research term here, not an anglicism to be scrubbed. |
| pixi | pixi | pixi | Tool name, lowercase, unchanged. |
| pooch | pooch | pooch | Tool name, lowercase, unchanged. |
| Precision | précision | precisión | `TP/(TP+FP)` — of the cases predicted positive, the fraction that are. **Error:** reserved for this quantity only; never used for *accuracy*. First formula-bearing lesson: 3.4. |
| Pretraining | pré-entraînement | preentrenamiento | FR hyphenates, ES does not. |
| Prompt | *prompt* (consigne) | *prompt* (instrucción) | Kept English in both. |
| Random forest | forêt aléatoire | bosque aleatorio | Translated in both, unlike *gradient boosting* — the split inside the tree-model family is real, not an oversight. |
| Recall | « rappel » ou « sensibilité » selon le domaine | «exhaustividad», «sensibilidad» ou *recall* selon el dominio | `TP/(TP+FN)` — of the actual positives, the fraction found. **The formula decides which quantity is meant; the domain decides which word names it.** Information retrieval and ML say « rappel » / «exhaustividad»; medical, epidemiological and detection-oriented readers expect « sensibilité » / «sensibilidad»; hispanophone ML practitioners often just say *recall*. ⚑ Changed: the previous edition-wide mandate («exhaustividad» in ES, « rappel » in FR) was too rigid. State the formula on first use in a chapter, then keep one word within that chapter. First formula-bearing lesson: 3.4. |
| Recurrent neural network (RNN) | réseau récurrent — sigle RNN | red recurrente — sigla RNN | |
| Regression | régression | regresión | |
| Reinforcement learning | apprentissage par renforcement | aprendizaje por refuerzo | |
| Reliability diagram | diagramme de fiabilité | diagrama de confiabilidad | ⚑ ES: «fiabilidad» (Spain) versus «confiabilidad» (Americas). The edition fixes «confiabilidad» for pan-regional neutrality; a Spain-facing edition would flip it. |
| Repository — Git / GitHub | « dépôt Git », « dépôt GitHub » | «repositorio Git», «repositorio GitHub» | An active version-controlled project with commits and branches. **Error:** never « archive » / «archivo» for this — see *Archive*. FR: « référentiel » appears in some GitHub interface pages, but developer prose says « dépôt »; reproduce a UI label verbatim in bold when telling the reader exactly what to click (**New repository**), and use « dépôt » in prose. Repository names and paths unchanged. |
| Repository — managed data portal | « entrepôt de données » ou « dépôt de données » | «repositorio de datos» | A curated portal you deposit into and query (Data Terra, a national data centre). Choose by function, not by the English spelling: preservation → *Archive*; version control → *Repository (Git)*; buckets → *Object store*. |
| Reproducibility | reproductibilité | reproducibilidad | |
| Sampling rate | fréquence d'échantillonnage | tasa de muestreo | FR speaks of frequency, ES of rate — both are the field's own idiom, not a translation slip. |
| Score | score | puntaje † | *Score* names no single quantity — accuracy, F1, Brier, CRPS and skill are all "scores", and they point in opposite directions (some higher-is-better, some lower). Never write « le score » / «el puntaje» without naming which one. ⚑ ES: «puntaje» is the written default, yet professional verification speech says «score» — which is why «score de Brier» stands. The inconsistency is visible in the CRPS and F1 rows and should be ruled on once, for all of them. |
| Self-supervised learning | apprentissage auto-supervisé | aprendizaje autosupervisado | FR hyphenates, ES writes it solid. |
| Skill (forecast verification) | *skill* (score de compétence) | *skill* (habilidad de pronóstico) | `1 − score/score_ref` — a **number** computed against a reference forecast (persistence or climatology), so 0 means "no better than the reference" and 1 means perfect. Both communities keep the English word in operational prose. FR glosses it « score de compétence ». State the reference forecast whenever you state a skill number — the same word names a different quantity against a different reference. |
| Skill (ordinary model capability) | compétence | habilidad / desempeño | **FR has two registers and they must not be mixed.** When the sentence means how good a model is in general, write plain « compétence » — no italics, no English, no « score ». Reserve *skill* / « score de compétence » for the verification quantity in the row above. « Compétence de prévision » as a translation of *forecast skill* is the error the Météo-France reviewer closes the document over. ⚑ ES: the two-register split was never fixed explicitly; «habilidad» and «desempeño» both appear. ES verification reviewer to rule. |
| Spectral leakage | fuite spectrale | fuga espectral | **Error** if left unqualified. Always « spectrale » / «espectral». Energy spreading between frequency bins because a finite window truncates the signal — nothing whatever to do with *data leakage*, and the book says so at every occurrence in both languages. |
| STA/LTA | STA/LTA | STA/LTA | Acronym unchanged everywhere; expansion translated at first use. |
| Supervised learning | apprentissage supervisé | aprendizaje supervisado | |
| Tapering (windowing) | fenêtrage † | ventana de suavizado † | *tapering* kept in speech and glossed in both. |
| Test set | ensemble de test | conjunto de prueba | ⚑ ES: «conjunto de test» circulates; the edition fixes «prueba». Note the split — FR keeps the English noun inside the French phrase, ES translates it out. |
| Tolerance-based reproducibility | reproductibilité à tolérance | reproducibilidad basada en tolerancia | Chapter 5 vocabulary. |
| Training set | ensemble d'entraînement | conjunto de entrenamiento | Never « set d'entraînement » / «set de entrenamiento», however common in speech. |
| Transformer | *transformer* | *transformer* | ⚑ ES: «transformador» circulates; the edition keeps English. FR keeps English without dispute. |
| Underfitting | sous-apprentissage | subajuste | |
| Unsupervised learning | apprentissage non supervisé | aprendizaje no supervisado | |
| Validation set | ensemble de validation | conjunto de validación | |
| Workflow — conceptual research process | « démarche (de recherche) », « protocole », parfois « processus » | «metodología», «procedimiento» | The way a scientist proceeds, not a thing you execute. Do not reach for « flux de travail » by reflex. |
| Workflow — executable / reproducible computation | « workflow de calcul », « workflow reproductible » | «flujo de trabajo computacional» | Changed: the single row `workflow → flux de travail` contradicted the French chapters and `translations/fr/MANIFEST.yml`, where a senior pass chose community-standard « workflow ». FR researchers say *workflow*; keep it, gloss it once per chapter. ES: «flujo de trabajo» is genuinely the natural form and stays. |
| Workflow — CI automation file | « workflow GitHub Actions » | «workflow de GitHub Actions» / «flujo de trabajo de GitHub Actions» | The `.github/workflows/` path is **never** translated, in either language. **Error:** never « archive » or « protocole » for a CI file. |
| Workflow — passing work between people | « flux de travail » | «flujo de trabajo» | The narrow organizational sense, and the only French sense where « flux de travail » is the right first choice. |
| Zarr | Zarr | Zarr | Format name, unchanged. |

---

## The hard invariants

Everything else in this table is a preference a reviewer may overturn. These
are not. They are the rows where the wrong word changes the meaning of the
sentence, and a terminology linter should fail the build on them:

1. **`accuracy` is never « précision » / «precisión».** Two different formulas.
2. **A Git repository is never an « archive » / «archivo».** Version control and
   preservation are different services with different guarantees.
3. **`fuite` / `fuga` is always qualified** — « de données » or « spectrale »,
   «de datos» or «espectral». Never bare.
4. **Loss, objective and cost are three things**, not three words for one. Per
   example; aggregate plus regularization; real-world consequence.
5. **Identifiers are never translated** — API and class names (`Pipeline`,
   `.fit()`, `DataLoader`), filenames and paths (`environment.yml`,
   `pixi.lock`, `.github/workflows/`, `*.ipynb`), config keys, and dataframe
   column names (`target`, `label`, `uncertainty`, `pick_weight`). The teaching
   pattern is « la variable cible `target` » / «la variable objetivo `target`»,
   never a renamed column that breaks the notebook.

## Contested rows at a glance

⚑ rows, and who should settle them:

- **Verification scores** — Brier score, CRPS, F1 score, Score (ES «puntaje» vs
  «score»). One ruling from the Spanish-edition verification reviewer settles
  all of them; they cannot be decided separately without contradicting each
  other.
- **Spanish regional variants** — Reliability diagram («confiabilidad» vs
  «fiabilidad»), Test set. The pan-regional-neutrality reviewers (Argentina,
  Mexico) rule.
- **French ML register** — Deep ensemble, Fine-tuning, Overfitting. The
  Paris-Saclay ML reviewer rules; the standing instruction is to prefer what
  the French AI community says over what a purist would coin.
- **Both editions** — Fold, Embedding, Skill (ordinary capability), Transformer
  (ES side), Lead time (ES side).

New ⚑ rows opened by the 2026-08 usage-guide revision, all of them cases where
this table used to mandate a single word and now names a range:

- **Cluster — ML, the resulting group.** « Grappe » and «conglomerado» were the
  mandated defaults and neither is a safe universal. Now: groupe/classe/
  partition, grupo/conglomerado/clúster, by method. FR and ES teaching
  reviewers.
- **Clustering (the task).** *Clustering* is now explicitly allowed alongside
  « partitionnement » / «agrupamiento». Paris-Saclay ML reviewer for FR.
- **Recall.** The edition-wide mandate is replaced by a domain split
  (rappel/sensibilité; exhaustividad/sensibilidad/recall) governed by the
  formula. Needs the cross-language statistics/verification reviewer, together
  with the metric rows.
- **Machine learning.** The FR and ES editions now differ deliberately. Both
  language reviewers should confirm their own edition's register — this is the
  single change most likely to be argued about, and it should be.

Rows that stopped being contested because the dispute was an artifact of
over-prescription: Baseline, Benchmark, Dataset, Jupyter notebook, Feature,
Cloud, HPC, Job, Build. The answer in every case is "both forms, gloss once,
be consistent inside a chapter."
