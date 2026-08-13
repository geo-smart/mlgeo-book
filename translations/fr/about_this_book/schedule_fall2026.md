# Calendrier du cours — automne 2026

ESS 469/569, University of Washington. Le cours a lieu **les lundi, mercredi et vendredi, de 10:00 à 11:20 en salle SIG 227** (Sieg Hall). Venez avec un portable chargé : la salle a des tablettes d'écriture et peu de prises, et les séances de TP durent les 80 minutes entières. L'enseignement court du 30 septembre au 11 décembre 2026 ; il n'y a pas cours le **mercredi 11 novembre** (Veterans Day) ni le **vendredi 27 novembre** (Native American Heritage Day), et Thanksgiving tombe le jeudi 26 novembre. La semaine d'examens finaux va du 12 au 18 décembre.

Cette page est la couche « mise en œuvre du cours » posée sur le livre : quelles séances couvrent quelles sections, et quand les travaux notés sont à rendre. Le livre porte toute la profondeur ; la séance en présentiel y sélectionne (voir les notes de différenciation 469/569 dans chaque devoir).

Deux choix de conception structurent le trimestre. D'abord, **le fil des agents traverse tout le cours** au lieu d'attendre la fin : ce qu'est un agent et comment nous l'utilisons (semaine 2), retourner vos compétences en données contre les affirmations d'une IA (semaine 4), et construire des jeux d'évaluation pour les agents (semaine 7) — ainsi le matériau distinctif du cours est enseigné tôt et appliqué souvent, et le projet de synthèse dispose d'un mois de marge. Ensuite, **décembre ne contient aucun contenu nouveau** : seulement de l'application, des cliniques, une séance tampon délibérée et les présentations, de sorte que rien d'important n'est perdu si une semaine glisse.

**Le rythme du « pouls d'articles ».** De la semaine 3 à la semaine 10, la plupart des séances s'ouvrent sur deux exposés étudiants de 4 minutes — un article tiré de la revue de littérature de l'[arc de lecture](../Chapter6-AgenticAI/6.5_reading_arc.md) de l'intervenant, disséqué à l'aune de la grille que la classe construit au fil du trimestre. Chaque étudiant présente une fois (les inscriptions ouvrent le lundi 5 octobre sur Canvas) et tout le monde dans la salle note chaque exposé via l'enquête permanente d'évaluation par les pairs : cinq notes de grille plus un point fort et un point à améliorer, anonymisés pour l'intervenant, la complétion comptant comme participation. Les exposés « pouls » sautent les deux journées de TP 4.5 et le point d'étape n° 1 ; le même instrument d'enquête revient pour les présentations finales.

## Calendrier des travaux notés, en un coup d'œil

| Item | Opens | Due / window |
|---|---|---|
| HW1 — workbench setup ([1.9](../Chapter1-GettingStarted/1.9_workbench_setup_hw1.md)) | Sep 30 | Mon Oct 12 |
| Ch 1 quiz (Canvas, timed) | Tue Oct 6 | Thu Oct 8 |
| Reading arc stage 1 — AI-assisted lit review ([6.5](../Chapter6-AgenticAI/6.5_reading_arc.md)) | Oct 5 | Wed Oct 21 |
| Ch 2 quiz | Mon Oct 26 | Wed Oct 28 |
| Final project proposal ([1.10](1.10_MLGEO_FinalProject.md)) | Oct 12 | Fri Oct 30 |
| Classification leaderboard ([3.5](../Chapter3-MachineLearning/3.5_multiclass_classification.ipynb)) | Mon Nov 2 | closes Tue Nov 24 |
| Reading arc stage 2 — anatomy of good papers | Oct 21 | Wed Nov 4 |
| Ch 3 quiz (includes the flipped 3.10 reading) | Tue Nov 10 | Thu Nov 12 |
| Reading arc stage 3 — your quality rubric | Nov 4 | Fri Nov 13 |
| Project check-in #1 — data-audit studio | — | Mon Nov 16, in class |
| Ch 6 quiz | Mon Nov 16 | Wed Nov 18 |
| HW-CML ([Homework_CML](../Chapter3-MachineLearning/Homework_CML.ipynb)) | Nov 4 | Fri Nov 20 |
| HW-DL ([Homework_DL](../Chapter4-DeepLearning/Homework_DL.ipynb)) | Nov 23 | Fri Dec 4 |
| Ch 5 quiz (checks the flipped Ch 5 reading) | Mon Nov 30 | Thu Dec 3 |
| Ch 4 quiz (after the forecasting session) | Thu Dec 3 | Mon Dec 7 |
| Forecasting leaderboard ([4.10](../Chapter4-DeepLearning/mlgeo_4.10_timeseriesforecast.ipynb)) | Wed Dec 2 | closes Wed Dec 9 |
| Project check-in #2 — dry-runs + agent clinic | — | Mon Dec 7, in class |
| Reading arc stage 4 — your pre-submission review agent | Nov 13 | Thu Dec 10 |
| Final presentations | — | Fri Dec 11 in class + assigned finals slot |
| Final report + repository | — | Wed Dec 16 |

Les colonnes et intitulés du tableau ci-dessus restent en anglais pour rester alignés avec le site anglais et avec les noms de rendus sur Canvas. En clair : *HW1* est le devoir 1 (poste de travail), les *quiz* sont les quiz de chapitre sur Canvas, *reading arc stage N* désigne l'étape N de l'arc de lecture, *leaderboard* le classement, *project check-in* le point d'étape de projet, et *Opens* / *Due* les dates d'ouverture et de remise.

Le chapitre 7 n'a pas de quiz : ses résultats d'apprentissage (traduction pour les publics, conséquences en aval) sont évalués directement par les livrables du projet final. Les quiz sont chronométrés, corrigés automatiquement sur Canvas, ouverts pendant la fenêtre indiquée, et tirés de banques de scénarios : ils testent le jugement face à une situation, non la mémorisation.

## Semaine par semaine

**Semaine 1 — Science ouverte et reproductible** (2 séances)
- Mer. 30 sept. — Introduction au cours ; pourquoi le ML en géosciences ; science ouverte et reproductible ([1.1](../Chapter1-GettingStarted/1.1_open_reproducible_science.md)). Devoir HW1 donné ; l'aide à l'installation continue à la clinique d'installation de la semaine 1 (heures de permanence), pas en cours. · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec01_why_ml_geosciences.html)
- Ven. 2 oct. — TP poste de travail : contrôle de version et environnements ([1.2](../Chapter1-GettingStarted/1.2_jupyter_environment.md)–[1.5](../Chapter1-GettingStarted/1.5_version_control_git.md), avec [1.9](../Chapter1-GettingStarted/1.9_workbench_setup_hw1.md) comme parcours en autonomie) ; répétition à blanc de la *pull request*. · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec02_your_workbench.html)

**Semaine 2 — Les agents, puis les données** (quiz du ch. 1, mar.–jeu.)
- Lun. 5 oct. — **Travailler avec des agents : politique et mécanisme** ([1.8](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md) + [6.1](../Chapter6-AgenticAI/6.1_llms_to_agents.md)) : ce qu'est un agent, par où les erreurs entrent dans la boucle, le système de badges, la déclaration d'usage. Étape 1 de l'arc de lecture donnée (sujet : opportunités et défis de l'IA dans votre sous-discipline, avec les deux lectures d'ancrage) ; ouverture des inscriptions au « pouls d'articles ». · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec03_working_with_agents.html)
- Mer. 7 oct. — À la rencontre des données : définitions et formats ([2.1](../Chapter2-DataManipulation/2.1_Data_Definitions.md)–[2.2](../Chapter2-DataManipulation/2.2_data_formats_rendered.ipynb)), la galerie de données, et un premier vrai jeu de données ([1.6](../Chapter1-GettingStarted/1.6_data_gallery.md)–[1.7](../Chapter1-GettingStarted/1.7_get_geodetic_gnss.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec04_meet_the_data.html)
- Ven. 9 oct. — Tableaux : pandas et préparation des *dataframes* ([2.3](../Chapter2-DataManipulation/2.3_pandas_rendered.ipynb)–[2.4](../Chapter2-DataManipulation/2.4_dataframes_prep.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec05_tables_that_tell_the_truth.html)

**Semaine 3 — Signaux** (HW1 à rendre lundi)
- Lun. 12 oct. — Tableaux multidimensionnels et données maillées ([2.5](../Chapter2-DataManipulation/2.5_Arrays.ipynb)) ; rééchantillonnage et données irrégulières ([2.6](../Chapter2-DataManipulation/2.6_resampling.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec06_sampling_resampling.html)
- Mer. 14 oct. — Considérations statistiques ([2.7](../Chapter2-DataManipulation/2.7_statistical_considerations.ipynb)) ; transformées spectrales ([2.8](../Chapter2-DataManipulation/2.8_data_spectral_transforms.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec07_statistics_spectra.html)
- Ven. 16 oct. — Filtrage, lacunes, erreurs d'horodatage : réparer des enregistrements réels ([2.9](../Chapter2-DataManipulation/2.9_filtering_data.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec08_repairing_records.html)

**Semaine 4 — Données prêtes pour l'IA, puis interroger l'IA** (étape 1 de l'arc à rendre mercredi)
- Lun. 19 oct. — Données synthétiques et le plancher de détection STA/LTA ([2.10](../Chapter2-DataManipulation/2.10_synthetic_noise.ipynb)) ; ingénierie des caractéristiques ([2.11](../Chapter2-DataManipulation/2.11_feature_engineering.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec09_synthetic_truth.html)
- Mer. 21 oct. — Réduction de dimension ([2.12](../Chapter2-DataManipulation/2.12_dimensionality_reduction.ipynb)) ; la liste de contrôle « prêt pour l'IA » et les jointures raster-station ([2.13](../Chapter2-DataManipulation/2.13_MLready_data.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec10_ai_ready_data.html)
- Ven. 23 oct. — **TP d'évaluation critique** ([6.2](../Chapter6-AgenticAI/6.2_critical_evaluation.ipynb)) — le point d'orgue du chapitre 2 : retourner vos nouvelles compétences en données contre les affirmations d'une IA et vérifier contre les données, non contre la vraisemblance. · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec11_verify_then_trust.html)

**Semaine 5 — Début du ML classique** (quiz du ch. 2, lun.–mer. ; propositions de projet à rendre vendredi)
- Lun. 26 oct. — Concepts de supervision ; classification et régression ([3.1](../Chapter3-MachineLearning/3.1_concepts_supervision.md)–[3.2](../Chapter3-MachineLearning/3.2_classification_regression.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec12_supervision_concepts.html)
- Mer. 28 oct. — Partitionnement (*clustering*) ([3.3](../Chapter3-MachineLearning/3.3_clustering.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec13_clustering.html)
- Ven. 30 oct. — Classification binaire et déséquilibre des classes ([3.4](../Chapter3-MachineLearning/3.4_binary_classification.ipynb)). **Propositions de projet à rendre.** · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec14_binary_imbalance.html)

**Semaine 6 — La classification, honnêtement** (étape 2 de l'arc à rendre mercredi)
- Lun. 2 nov. — Classification multiclasse ([3.5](../Chapter3-MachineLearning/3.5_multiclass_classification.ipynb)) ; comment fonctionnent le classement (*leaderboard*) et les ensembles de test cachés. **Ouverture du classement de classification.** · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec15_four_sources_leaderboard.html)
- Mer. 4 nov. — La régression logistique à la main ; vos probabilités sont-elles honnêtes ? ([3.6](../Chapter3-MachineLearning/3.6_logistic_regression.ipynb)). Devoir HW-CML donné. · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec16_honest_probabilities.html)
- Ven. 6 nov. — Arbres, forêts, ensembles ([3.7](../Chapter3-MachineLearning/3.7_randomForest_regression.ipynb) + [3.9](../Chapter3-MachineLearning/3.9_ensemble_learning.ipynb)) ; lire les importances sans se raconter d'histoires. [3.10](../Chapter3-MachineLearning/3.10_autoML.ipynb) donné en lecture inversée (vérifiée par le quiz du ch. 3). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec17_trees_forests_honestly.html)

**Semaine 7 — Les deux cours sur l'évaluation** (semaine courte — pas cours le mer. 11 nov. ; quiz du ch. 3, mar.–jeu. ; étape 3 de l'arc à rendre vendredi)
- Lun. 9 nov. — **Entraînement robuste** ([3.8](../Chapter3-MachineLearning/3.8_robust_training.ipynb)), séance entière : pourquoi les découpages aléatoires mentent ; découpages temporels, par groupes et spatiaux ; l'échelle des découpages sur des données à vérité plantée. · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec18_robust_training.html)
- Ven. 13 nov. — **Construire un jeu d'évaluation** ([6.3](../Chapter6-AgenticAI/6.3_build_an_eval_set.ipynb)) + déclaration et normes ([6.4](../Chapter6-AgenticAI/6.4_disclosure_and_norms.md)) : le même geste d'évaluation équitable, appliqué aux agents. Étape 4 de l'arc de lecture donnée — quatre semaines de marge. · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec19_eval_sets_for_agents.html)

**Semaine 8 — Point d'étape, puis apprentissage profond** (quiz du ch. 6, lun.–mer. ; HW-CML à rendre vendredi)
- Lun. 16 nov. — **Point d'étape de projet n° 1 : atelier d'audit des données.** Chaque équipe, cinq minutes : vos données face à la liste de contrôle 2.13, votre modèle de référence, la conception de votre découpage — tout frais de la séance 3.8.
- Mer. 18 nov. — Du perceptron aux MLP ([4.0](../Chapter4-DeepLearning/mlgeo_4.0_perceptrons.ipynb)–[4.2](../Chapter4-DeepLearning/mlgeo_4.2_MLP.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec21_neuron_to_network.html)
- Ven. 20 nov. — Les CNN : images, formes d'onde, et le modèle de référence classique ([4.3](../Chapter4-DeepLearning/mlgeo_4.3_CNN.ipynb)). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec22_convolution_detector.html)

**Semaine 9 — Modèles de séquences et TP d'entraînement** (semaine courte — Thanksgiving)
- Lun. 23 nov. — Modèles de séquences : RNN, LSTM, attention ([4.4](../Chapter4-DeepLearning/mlgeo_4.4_RNN.ipynb)). Devoir HW-DL donné. · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec23_networks_with_memory.html)
- Mer. 25 nov. — TP d'entraînement de modèles, séance I : curation des données et qualité des étiquettes ([4.5](../Chapter4-DeepLearning/mlgeo_4.5_ModelTraining.ipynb), sections 1 à 3). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec24_three_pillars_lab_I.html)

**Semaine 10 — Incertitude, prévision, chapitres inversés** (quiz du ch. 5, lun.–jeu. ; classement de classification fermé le 24 nov.)
- Lun. 30 nov. — TP d'entraînement de modèles, séance II : architecture, incertitude, calibration, comportement hors plage (4.5, sections 4 et 5). Le chapitre 5 ([5.1](../Chapter5-ModelWorkflows/5.1_reproducibility.md)–[5.5](../Chapter5-ModelWorkflows/5.5_data_at_scale.ipynb)) est donné en lecture inversée. · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec25_three_pillars_lab_II.html)
- Mer. 2 déc. — Confrontation de méthodes de prévision de séries temporelles, prévisions probabilistes, horizons de *skill* (score de compétence) ([4.10](../Chapter4-DeepLearning/mlgeo_4.10_timeseriesforecast.ipynb)). **Ouverture du classement de prévision.** Auto-encodeurs et PINN ([4.6](../Chapter4-DeepLearning/mlgeo_4.6_AutoEncoder.ipynb)–[4.7](../Chapter4-DeepLearning/mlgeo_4.7_PINN.ipynb)) en approfondissement (obligatoire en 569, facultatif en 469). · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec26_forecasting.html)
- Ven. 4 déc. — Discussion du chapitre 5 (30 min, à partir de la lecture inversée) + communiquer sa science : publics et conséquences en aval ([7.1](../Chapter7-UseCases/7.1_audience_translation.md)–[7.2](../Chapter7-UseCases/7.2_downstream_impact.md)). **HW-DL à rendre.** · [diapositives](https://geo-smart.github.io/mlgeo-book/slides/2026/lec27_ship_it_honestly.html)

**Semaine 11 — Cliniques, tampon, présentations** (quiz du ch. 4 fermé lundi ; classement de prévision fermé mercredi ; étape 4 de l'arc à rendre jeudi)
- Lun. 7 déc. — **Point d'étape de projet n° 2 : répétitions de présentation + clinique agent de relecture.** Répétez ; faites tourner votre agent de l'étape 4 sur le brouillon d'une autre équipe.
- Mer. 9 déc. — **Séance tampon.** Elle absorbe les décalages du trimestre ; si rien n'a glissé, discussion de synthèse sur la question de l'année — opportunités et défis de l'IA pour les géosciences, argumentés à partir des mesures faites par la classe elle-même (évaluations d'agents, statistiques d'accord entre notes des pairs) — ou approfondissement choisi par vote de la classe (démonstration d'évaluation d'agent en direct, PINN, données à grande échelle).
- Ven. 11 déc. — **Présentations finales, séance I.**

**Semaine d'examens** — Présentations, séance II, dans le créneau d'examen attribué par le service de scolarité (voir le *Time Schedule*) ; **rapport final et dépôt à rendre le mercredi 16 décembre**.

## Notes sur le rythme

- Le fil des agents est délibérément placé tôt : la culture de vérification (6.1/6.2) est installée avant que les habitudes de travail ne se forment, 6.3 arrive la semaine suivant 3.8 pour que l'évaluation équitable se lise comme une seule idée appliquée deux fois, et l'étape 4 de l'[arc de lecture](../Chapter6-AgenticAI/6.5_reading_arc.md) dispose d'un mois de marge. Le quiz du chapitre 6 suit 6.3/6.4 à la mi-novembre.
- Le carnet 4.5 est réparti sur deux séances de TP, à son point de reprise signalé ; ne tentez pas de le faire d'une traite.
- Les chapitres inversés (3.10, ch. 5) sont examinables via leurs quiz ; le temps de cours qu'ils auraient occupé finance les points d'étape et la séance tampon.
- Le point d'étape n° 1 remplace un rapport de mi-parcours écrit : un retour plus tôt, au moment où il peut encore changer le projet.
- Les étudiants de 469 suivent le parcours allégé indiqué dans chaque devoir : attentes de niveau Appliquer sur les résultats d'apprentissage relatifs à l'incertitude et aux agents, 4.6/4.7 facultatifs, et un rôle d'appui plutôt que de pilotage dans les projets finaux.
- Les sections non traitées en cours ne sont pas des lectures facultatives pour les 569, sauf mention contraire ; le cours sélectionne, le livre porte la profondeur.
