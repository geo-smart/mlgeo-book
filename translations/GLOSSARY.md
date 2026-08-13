# Trilingual glossary — English / Français / Español

One table, one order, three languages. Every term in the book's English
glossary (`book/reference/glossary.md`) appears here, plus the terms the
French and Spanish translation passes had to settle along the way.

**English is authoritative for code.** Function names, arguments, library
documentation, error messages and the literature are all in English, and the
book never translates them. What the French and Spanish columns fix is the
*prose*: the word a reader meets in a sentence, and the word a translator must
use identically in every chapter.

**Reading the cells.**

| Form | Meaning |
|---|---|
| `terme` | The community writes this. It is the working term in that language. |
| `*term* (glose)` | The community keeps the **English** word in prose. The parenthesis is the gloss given once at the first occurrence of each chapter — the reader gets both the working term and its meaning. |
| `terme †` | The native term is written, but the English word is current in speech and in code comments. Tolerated, glossed, never the written default. |
| ⚑ in Notes | Contested. The row states what is in dispute and who should rule on it. Open an issue `[translation] term: …`. |

The two languages do **not** keep the same English words, and that difference
is the most useful thing in this table. French francizes *autoencoder* into
« auto-encodeur » while Spanish prose runs on *autoencoder*; Spanish
translates *test set* into «conjunto de prueba» while French keeps the noun in
« ensemble de test ». Neither community is being sloppy — each is reporting
what its own ML and geoscience people actually say. Both keep *dropout*,
*gradient boosting*, *transformer* and *prompt*, which is a real consensus and
worth trusting.

*FR — Ce tableau donne, pour chaque terme anglais, le terme français et le
terme espagnol retenus par le livre ; l'anglais fait foi dans le code, et les
termes en italique sont ceux que la communauté conserve en anglais dans la
prose, glosés à la première occurrence.*

*ES — Esta tabla da, para cada término en inglés, el término francés y el
término español fijados por el libro; el inglés manda en el código, y los
términos en cursiva son los que la comunidad conserva en inglés en la prosa,
glosados en su primera aparición.*

Detailed, translator-facing notes and the history of contested rows live in
[`GLOSSARY_fr.md`](GLOSSARY_fr.md) and [`GLOSSARY_es.md`](GLOSSARY_es.md).

---

| English term | Français | Español | Notes |
|---|---|---|---|
| Ablation | ablation | ablación | |
| Accuracy | exactitude | exactitud | **Never** « précision » / «precisión» — those are reserved for *precision*. Flagged as a disqualifying false friend by both the Chilean and the UNAM reviewers. |
| Active learning | apprentissage actif | aprendizaje activo | |
| Agent | agent | agente | Chapter 6 vocabulary; the word survives unchanged in all three. |
| AI-ready data | données prêtes pour l'IA | datos listos para IA | FR uses the French acronym IA for AI throughout. |
| Aliasing | repliement de spectre † | solapamiento espectral † | Both write the native term; *aliasing* stays current in lab speech. |
| Attribute | attribut | atributo | ES: «atributo» is reserved for *attribute* and nothing else — it is what keeps the *feature* row below workable. |
| Autoencoder | auto-encodeur | *autoencoder* (autocodificador) | Clean divergence. FR francizes with a hyphen and never looks back; ES prose (chapter 4.6) runs on the English noun with «autocodificador» as the gloss. |
| Baseline | modèle de référence † | modelo de referencia † | *baseline* tolerated and glossed in both. |
| Batch | lot † | lote † | *batch* current in code-speech in both. |
| Brier score | score de Brier | score de Brier | ⚑ ES: «puntaje de Brier» circulates, but professional verification speech says «score». FR is settled Météo-France usage. The ES verification reviewer should rule once, together with the CRPS row. |
| Calibration | calibration | calibración | FR: « étalonnage » belongs to instruments, never to probabilities. |
| Classification | classification | clasificación | |
| Climatology (reference) | climatologie de référence | valor climatológico de referencia | Operational verification vocabulary; native in both. |
| Cluster (the resulting group) | grappe | conglomerado | ES settled in the full-book pass: the *group* is «conglomerado», distinct from the *task* below. |
| Clustering (the task) | partitionnement † | agrupamiento † | ⚑ FR floats between « partitionnement » and « regroupement », and *clustering* is what most people say; the book fixes « partitionnement ». The Paris-Saclay ML reviewer should rule. |
| Coda | coda | coda | Seismological term, unchanged in all three. |
| Computer | ordinateur | computadora | ES fixes the American variant pan-regionally; never alternate with «ordenador» inside one edition. FR has no such split to manage. |
| Convolutional neural network (CNN) | réseau de neurones convolutif — sigle CNN | red neuronal convolucional — sigla CNN | The acronym stays English in both. |
| Corner frequency | fréquence coin | frecuencia de esquina | |
| Coverage (of intervals) | couverture empirique des intervalles | cobertura empírica de los intervalos | Météo-France and SMN both use the empirical-coverage phrasing. |
| Cross-validation | validation croisée | validación cruzada | |
| CRPS | score de probabilité classé continu — sigle CRPS | puntaje de probabilidad clasificada continua — sigla CRPS | ⚑ ES: the shipped text says «puntaje», but the «score de Brier» precedent in the same glossary argues for «score de probabilidad clasificada continua». One of the two has to give. ES verification reviewer (SMN/atmospheric-science persona) to rule. The acronym CRPS is kept in both. |
| Data card | fiche de données | ficha de datos | |
| Data leakage | fuite de données | fuga de datos | **Always** qualified « de données » / «de datos». Bare « fuite » / «fuga» would collide with spectral leakage; see both rows. |
| Deep ensemble | ensemble profond | ensamble profundo | ⚑ FR: « ensemble de réseaux » still has defenders. ES fixes «ensamble», not the French-looking «ensemble». |
| Deep learning | apprentissage profond † | aprendizaje profundo † | *deep learning* tolerated at first mention in both. |
| Dimensionality reduction | réduction de dimension | reducción de dimensionalidad | |
| Dropout | *dropout* (extinction aléatoire de neurones) | *dropout* (apagado aleatorio de neuronas) | Kept English in both. The glosses differ; the working word does not. |
| Embedding | plongement | *embedding* (incrustación) | ⚑ ES: the shipped chapters use «incrustación» and *embedding* in different places — not settled. FR « plongement » is stable, though persona 06 warns that unglossed « plongement lexical » reads as translator jargon. |
| Ensemble forecast | prévision d'ensemble | pronóstico por ensambles | Established operational vocabulary in both (Météo-France, SMN). |
| Epistemic vs. aleatoric uncertainty | incertitude épistémique / aléatoire | incertidumbre epistémica / aleatoria | |
| Epoch | époque | época | |
| Eval set | jeu d'évaluation | conjunto de evaluación | Chapter 6 vocabulary; both communities translate it rather than keeping *eval set*. |
| Expected calibration error (ECE) | erreur de calibration attendue — sigle ECE | error de calibración esperado — sigla ECE | |
| False alarm | fausse alarme | falsa alarma | Pairs with *miss*; native in both, no dispute. |
| Feature | caractéristique † | característica † | ES: **not** «atributo», which is reserved for *attribute*. *feature* is current in code-speech in both languages, but neither writes it. |
| Feature engineering | ingénierie des caractéristiques | ingeniería de características | |
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
| Hyperparameter | hyperparamètre | hiperparámetro | |
| Instrument response | réponse instrumentale | respuesta instrumental | FR keeps *counts* in English for the recorded units. |
| Interpolation vs extrapolation | interpolation / extrapolation | interpolación / extrapolación | |
| Jupyter notebook | carnet Jupyter † | cuaderno de Jupyter † | Both editions write the native word throughout; *notebook* is what everyone says aloud. |
| Label | étiquette | etiqueta | |
| Large language model (LLM) | grand modèle de langage — sigle LLM | modelo de lenguaje de gran escala — sigla LLM | The acronym LLM is kept in both; only the expansion is translated. |
| Lead time | échéance | horizonte de pronóstico | ⚑ ES: «plazo de pronóstico» circulates. FR « échéance » is settled Météo-France usage — a single word where ES needs a phrase. |
| Leaderboard | classement † | tabla de clasificación † | *leaderboard* tolerated and glossed in both. |
| Leakage (spatial and temporal) | fuite spatiale / temporelle | fuga espacial / temporal | Always qualified. See *data leakage* and *spectral leakage*. |
| Learning rate | taux d'apprentissage | tasa de aprendizaje | |
| Linear probe | sonde linéaire | sonda lineal | FR: **not** « sondage linéaire » — that is hash-table linear probing, an unrelated computing term. « Sonde » is the instrument metaphor the book means. |
| Loss function | fonction de perte | función de pérdida | FR: never « fonction de coût » (reserved for *cost*). ES: this row is exactly why *miss* below cannot be «pérdida». |
| Machine learning | apprentissage automatique † | aprendizaje automático † | Bilingual at first mention in both editions; the native term dominates afterwards. |
| MASE | MASE — erreur absolue moyenne mise à l'échelle | MASE — error absoluto medio escalado | Acronym kept, expansion translated once. |
| MC dropout (Monte Carlo dropout) | *MC dropout* | *MC dropout* | Kept English in both, following *dropout*. |
| Miss (error type) | manqué — « les événements manqués » | omisión | ES: **not** «pérdida», which would collide with «función de pérdida». FR nominalizes the participle (« les manqués »). Pairs with *false alarm*; both editions must name the two error directions separately when discussing who bears the cost. |
| Model card | fiche de modèle | ficha de modelo | Parallel to *data card* in both. |
| MyST | MyST | MyST | Product name; expansion *Markedly Structured Text* kept in English. |
| Neural network | réseau de neurones | red neuronal | |
| Overfitting | surapprentissage | sobreajuste | ⚑ FR: « surajustement » exists and has defenders; the book fixes « surapprentissage ». ES «sobreajuste» is uncontested. |
| Persistence (forecast) | persistance | persistencia | Météo-France and SMN usage; never renamed with a neologism. |
| Physics-informed neural network (PINN) | réseau de neurones informé par la physique — sigle PINN | red neuronal informada por la física — sigla PINN | |
| pixi | pixi | pixi | Tool name, lowercase, unchanged. |
| pooch | pooch | pooch | Tool name, lowercase, unchanged. |
| Precision | précision | precisión | Reserved for *precision* only. See *accuracy*. |
| Pretraining | pré-entraînement | preentrenamiento | FR hyphenates, ES does not. |
| Prompt | *prompt* (consigne) | *prompt* (instrucción) | Kept English in both. |
| Random forest | forêt aléatoire | bosque aleatorio | Translated in both, unlike *gradient boosting* — the split inside the tree-model family is real, not an oversight. |
| Recall | rappel | exhaustividad | ⚑ ES: «sensibilidad» circulates and *recall* is tolerated glossed. FR « rappel » is settled. |
| Recurrent neural network (RNN) | réseau récurrent — sigle RNN | red recurrente — sigla RNN | |
| Regression | régression | regresión | |
| Reinforcement learning | apprentissage par renforcement | aprendizaje por refuerzo | |
| Reliability diagram | diagramme de fiabilité | diagrama de confiabilidad | ⚑ ES: «fiabilidad» (Spain) versus «confiabilidad» (Americas). The edition fixes «confiabilidad» for pan-regional neutrality; a Spain-facing edition would flip it. |
| Reproducibility | reproductibilité | reproducibilidad | |
| Sampling rate | fréquence d'échantillonnage | tasa de muestreo | FR speaks of frequency, ES of rate — both are the field's own idiom, not a translation slip. |
| Score | score | puntaje † | ⚑ ES: «puntaje» is the written default, yet professional verification speech says «score» — which is why «score de Brier» stands. The inconsistency is visible in the CRPS row and should be ruled on once, for both. |
| Self-supervised learning | apprentissage auto-supervisé | aprendizaje autosupervisado | FR hyphenates, ES writes it solid. |
| Skill (forecast verification) | *skill* (score de compétence) | *skill* (habilidad de pronóstico) | The quantity you compute against a reference forecast. Both communities keep the English word in operational prose. FR glosses it « score de compétence » — a *number*. |
| Skill (ordinary model capability) | compétence | habilidad / desempeño | **FR has two registers and they must not be mixed.** When the sentence means how good a model is in general, write plain « compétence » — no italics, no English, no « score ». Reserve *skill* / « score de compétence » for the verification quantity in the row above. « Compétence de prévision » as a translation of *forecast skill* is the error the Météo-France reviewer closes the document over. ⚑ ES: the two-register split was never fixed explicitly; «habilidad» and «desempeño» both appear. ES verification reviewer to rule. |
| Spectral leakage | fuite spectrale | fuga espectral | **Always** qualified « spectrale » / «espectral». Unrelated to *data leakage*, and the book says so at every occurrence in both languages. |
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
| Workflow | flux de travail | flujo de trabajo | Native and stable in both. |
| Zarr | Zarr | Zarr | Format name, unchanged. |

---

## Contested rows at a glance

⚑ rows, and who should settle them:

- **Verification scores** — Brier score, CRPS, Score (ES «puntaje» vs «score»).
  One ruling from the Spanish-edition verification reviewer settles all three;
  they cannot be decided separately without contradicting each other.
- **Spanish regional variants** — Reliability diagram («confiabilidad» vs
  «fiabilidad»), Recall («exhaustividad» vs «sensibilidad»), Test set. The
  pan-regional-neutrality reviewers (Argentina, Mexico) rule.
- **French ML register** — Clustering, Deep ensemble, Fine-tuning,
  Overfitting. The Paris-Saclay ML reviewer rules; the standing instruction is
  to prefer what the French AI community says over what a purist would coin.
- **Both editions** — Fold, Embedding, Skill (ordinary capability), Transformer
  (ES side), Lead time (ES side).
