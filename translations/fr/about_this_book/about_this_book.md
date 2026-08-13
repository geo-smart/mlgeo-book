# L'apprentissage automatique en géosciences

Le cadre **GeoS**cience **MA**chine Learning **R**esources and **T**raining (GeoSMART) propose un parcours de formation au calcul scientifique open source, à la théorie du *machine learning* (apprentissage automatique), à ses outils et à son déploiement.

Ce livre accompagne le cours *Machine Learning in the Geosciences* (ESS 469/569) de l'Université de Washington. Le livre, les tutoriels et les devoirs vivent dans ce dépôt unique ; les étudiants exécutent les carnets (*notebooks*) localement ou sur le service cloud de leur choix.

Enseignants :

- Marine Denolle (mdenolle@uw.edu)
- Akshay Mehra (akmehra@uw.edu)

Ce projet est soutenu par l'équipe GeoSMART (Stefan Todoran, Nicoleta Cristea, Anthony Arendt, Scott Henderson, Ziheng Sun, Yiyu Ni, Akash Kharita).

## Vue d'ensemble

Le cours introduit l'apprentissage automatique en géosciences, les bases du calcul scientifique et la méthodologie du ML appliqué. Il travaille sur des jeux de données canoniques et thématiques en sismologie, océanographie, cryosphère, sciences planétaires, géologie et géodésie. Les méthodes enseignées comprennent le partitionnement (*clustering*) non supervisé, la régression logistique, la forêt aléatoire (*random forest*), les machines à vecteurs de support et l'apprentissage profond (*deep learning*) avec PyTorch.

Le cours repose sur trois piliers, plus un quatrième fil qui traverse toute l'édition 2026 :

1. **Des données prêtes pour l'IA** : transformer des observations géoscientifiques brutes en jeux de données dont un modèle peut apprendre.
2. **L'apprentissage automatique classique** : des méthodes fondées sur les caractéristiques (*features*), entraînées et évaluées honnêtement.
3. **L'apprentissage profond** : les réseaux de neurones en PyTorch, du perceptron aux architectures modernes.
4. **Travailler avec l'IA agentique** : en 2026, les étudiants écrivent du code aux côtés d'assistants IA qui lisent les dépôts, exécutent du code et proposent des modifications. Le cours en fait une compétence à enseigner, non un raccourci à réprimer. L'évaluation critique des sorties de l'IA, la traduction des résultats pour des publics différents et l'explicitation des conséquences en aval sont des compétences notées, au même titre que l'exactitude d'un modèle.

```{note}
**Portée et ancrage régional.** Ce livre a été écrit pour un cours précis, dans un lieu précis, et cela se voit. Le calendrier, le trimestre de dix semaines, la salle, les quiz sur Canvas, le classement (*leaderboard*) et la convention de nommage des dépôts sont ceux de l'Université de Washington. Les données aussi, pour l'essentiel : les carnets de sismologie travaillent sur des formes d'onde et des catalogues d'événements du Pacifique Nord-Ouest américain (miniPNW, un jeu d'événements déposé sur Zenodo, la station UW.RATT du Puget Sound), ceux de géodésie sur du GNSS de Cascadia, ceux de prévision sur le CO2 du Mauna Loa, et plusieurs exemples de tableaux et de grilles sur des listes de stations américaines et un champ de réanalyse nord-américain.

Rien de tout cela n'est imposé par la matière enseignée. Les méthodes, l'exigence d'évaluation et les modes d'échec sont les mêmes partout ; le terrain, non. Si vous enseignez ou lisez ce livre hors des États-Unis, prévoyez de substituer des données régionales et un contexte institutionnel local. La page [Adopter ce livre](adopting_this_book.md) recense ce qui relève de l'Université de Washington, quels carnets reposent sur des données américaines, et comment une substitution de jeu de données a été menée concrètement.

Cette édition française localise la prose — exemples, institutions, aléas, moyens de calcul — mais non les données : à l'exception du carnet 1.7 (GNSS), tous les carnets conservent les jeux de données de l'édition anglaise, si bien que les exercices entraînent des modèles sur des données du Pacifique Nord-Ouest américain. Les contributions de jeux de données régionaux sont bienvenues, selon la procédure décrite dans [`translations/README.md`](https://github.com/geo-smart/mlgeo-book/blob/main/translations/README.md).
```

# À qui s'adresse ce livre

Aux doctorants et aux étudiants avancés de licence et master, en géosciences et dans les disciplines voisines de l'ingénierie et de l'informatique — ainsi qu'aux enseignants, post-doctorants, ingénieurs de laboratoire et responsables de programmes qui les forment ou les recrutent. À l'issue du livre, l'étudiante ou l'étudiant sait mener un flux de données géoscientifiques brut et désordonné jusqu'à un modèle défendable, reproductible, évalué honnêtement — et sait dire exactement quelles parties un assistant IA a réalisées.

# Résultats d'apprentissage

Au terme du livre, les étudiants sauront :

| # | Résultat | Niveau de Bloom | Où dans le livre | Production notée |
|---|---------|-------------|-------------------|-----------------|
| 1 | Décrire les usages canoniques du ML en géosciences (découverte, automatisation, traitement du signal, émulation, prévision) et associer familles de méthodes, type de données, taille d'échantillon et question posée | Comprendre | introductions de chapitres, 3.1 | quiz chronométrés sur Canvas |
| 2 | Construire des environnements de calcul reproductibles (Git, pixi, carnets) et exécuter le même flux de travail sur un portable, un calculateur HPC ou une instance cloud | Appliquer | chapitre 1, 5.4 | devoir 1 : poste de travail |
| 3 | Transformer des flux de données brutes variés — séries temporelles de capteurs de l'échantillonnage quotidien à 100 Hz, imagerie géospatiale et champs maillés, observations tabulaires et ponctuelles — en jeux de données prêts pour l'IA, et réparer les pathologies réelles des instruments (lacunes, dérive, erreurs d'horodatage, échantillonnage irrégulier, bruit d'étiquetage) | Analyser / Créer | chapitre 2 | exercices du chapitre 2, volet données du projet final |
| 4 | Appliquer des transformations statistiques et de traitement du signal (filtrage, transformées de Fourier et en ondelettes, rééchantillonnage, réduction de dimension) et prédire leur effet sur ce qu'un modèle pourra apprendre en aval | Appliquer / Évaluer | 2.6–2.12 | exercices du chapitre 2 |
| 5 | Concevoir l'évaluation avant le modèle : choisir des modèles de référence (*baselines*) du domaine, construire des découpages conçus contre la fuite de données (temporels, spatiaux, par groupes) et rapporter les métriques avec leur incertitude | Évaluer / Créer | chapitre 3 (validation croisée spatiale et par groupes en 3.8), classement | classement (3.5, 4.10), devoirs |
| 6 | Construire et entraîner des modèles ML classiques et profonds (régression, forêts, *gradient boosting* ; MLP, CNN, RNN, *transformer*, auto-encodeur) avec scikit-learn et PyTorch, et diagnostiquer un entraînement sain ou défaillant | Appliquer / Créer / Analyser | chapitres 3–4 | devoirs ML classique et apprentissage profond, TP 4.5 |
| 7 | Exploiter la connaissance physique comme ressource : générer des données synthétiques d'entraînement et des jeux de référence à partir de modèles physiques, inscrire des contraintes physiques dans les fonctions de perte et les architectures, et valider contre une vérité terrain connue | Créer / Évaluer | 2.10, 4.7, `mlgeo_synth` | TP 4.5, exercice 6.3 (jeu d'évaluation) |
| 8 | Quantifier l'incertitude prédictive (bootstrap et ensembles, *MC dropout* à l'inférence, sorties par quantiles ou distributionnelles), vérifier la calibration et juger quand un modèle extrapole hors de sa distribution d'entraînement | Évaluer | tissé dans les chapitres 3–4 (3.8–3.9, 4.5, 4.10) | exercices des chapitres 3–4, projet final |
| 9 | Versionner et tracer données, modèles et expériences pour qu'un autre scientifique réexécute le pipeline et obtienne les mêmes nombres | Appliquer / Évaluer | chapitre 5 | dépôt du projet final (30 %) |
| 10 | Évaluer les agents IA et les sorties de LLM (grands modèles de langage) comme des instruments scientifiques : rédiger un cahier des charges de tâche, construire un jeu d'évaluation avec vérité terrain, noter les sorties et analyser les modes d'échec | Évaluer / Créer | chapitre 6 | exercice 6.3, agent de relecture (projet de synthèse) |
| 11 | Intégrer l'assistance de l'IA à la recherche sans en abandonner la maîtrise intellectuelle : déclarer l'usage, vérifier les sorties et défendre chaque choix méthodologique sans l'assistant | Appliquer / Évaluer | 1.8, 6.4 | déclarations d'usage, présentations |
| 12 | Traduire un même résultat pour des publics distincts et apprécier les usages et conséquences en aval d'un modèle déployé | Créer / Évaluer | chapitre 7 | livrables du projet final (7.1, 7.2) |

Les concepts de visualisation de données sont introduits et mobilisés tout au long du livre.

# Prérequis

**Prérequis** : MATH 207 et MATH 208, ou MATH 307 ou 308, ou AMATH 351 ou 352, CS160 ou CS163, ou l'accord de l'enseignant (cursus de l'Université de Washington ; l'équivalent français est une licence scientifique avec algèbre linéaire, équations différentielles et une initiation à la programmation).

**Compétences recommandées** : connaissance de Python, AMATH301, cours d'introduction aux sciences de la Terre. Des rappels de calcul scientifique font partie du cours.

# Programme

- **Partie I : des géodonnées prêtes pour l'IA** : les données géoscientifiques, leurs modalités et dimensions, leurs caractéristiques de base, l'extraction de caractéristiques, la réduction de dimension, et la mise en forme d'un jeu de données prêt pour l'IA.
- **Partie II : apprentissage automatique classique** : entraînement des modèles, évaluation, mesure de la généralisation, et bonnes pratiques pour un entraînement fiable des algorithmes classiques après ingénierie des caractéristiques (p. ex. K-moyennes, forêt aléatoire, k plus proches voisins).
- **Partie III : apprentissage profond** : les concepts fondamentaux — perceptrons et réseaux entièrement connectés, réseaux convolutifs et récurrents, un petit *transformer* pour la prévision de séquences, auto-encodeurs et réseaux de neurones informés par la physique — plus la pratique de l'entraînement : optimisation, régularisation, diagnostic des entraînements défaillants et incertitude des sorties.

Les chapitres suivants prolongent les piliers : flux de travail reproductibles à l'ère des agents (chapitre 5), construction et évaluation d'agents IA (chapitre 6), cas d'usage, traduction pour les publics et conséquences en aval (chapitre 7).

# Construction des compétences techniques

Tout au long du cours, les étudiants développent des compétences en shell, en contrôle de version avec git et GitHub, en programmation Python, en calcul haute performance et en visualisation de données en Python.

- _Shell_ : introduit tôt dans le cours, utilisé au besoin.
- _Contrôle de version_ : introduit tôt et utilisé à chaque séance.
- _Programmation Python_ : introduite progressivement. Nous détaillons l'usage de numpy, (geo)pandas et scikit-learn, avec PyTorch comme cadre d'apprentissage profond.
- _Visualisation en Python_ : introduite tôt avec Matplotlib et Plotly, utilisée dans chaque séance Python.
- _Calcul haute performance_ : utilisé dans la seconde moitié du cours et pendant le projet final.
- _Assistants IA agentiques_ : introduits au chapitre 1 (voir la [politique d'usage de l'IA du cours](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md)) et utilisés, avec déclaration, tout au long du cours.

# Lectures : de l'article au relecteur-agent

Les lectures suivent un arc en quatre étapes sur le trimestre :

1. **Revue de littérature assistée par IA** : les étudiants pilotent un assistant IA dans une revue de littérature sur un sujet imposé et vérifient eux-mêmes chaque référence.
2. **Anatomie des bons articles scientifiques** : disséquer des articles exemplaires — ce qui rend des méthodes défendables, des figures honnêtes et des affirmations étayées.
3. **Construire ses propres critères de qualité** : par la discussion en classe, chaque étudiant rédige une grille de qualité explicite dans son genre déclaré — article de revue ou livrable pour parties prenantes, en miroir des deux parcours du projet final.
4. **Construire un agent de relecture pré-soumission** : les étudiants transforment leur grille en agent de relecture et le testent avec la machinerie d'évaluation du chapitre 6 (cahier des charges, jeu d'évaluation, analyse des échecs) sur des articles aux forces et faiblesses connues.

Le calendrier hebdomadaire de cet arc figure dans le syllabus du cours.

# Infrastructure du cours

Ce livre contient tous les tutoriels et devoirs. Les étudiants travaillent dans VS Code ou JupyterLab avec un assistant IA agentique, conservent leur travail sur GitHub et gèrent leurs environnements logiciels avec [pixi](https://pixi.sh). Pour construire le livre localement :

```
pixi install
pixi run build
```

Chaque étudiant crée un dépôt de cours personnel nommé `MLGEO2026_UWNETID`, y copie les fichiers d'environnement de ce livre, et y conserve devoirs et travaux de projet sous contrôle de version.

# Licences

Le texte et les figures de ce livre sont sous licence [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) (CC-BY-4.0). Le code, y compris le code source des carnets, est sous [licence MIT](https://opensource.org/license/mit). Vous pouvez réutiliser et adapter les deux, avec attribution.
