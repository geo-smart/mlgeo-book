# Chapitre 5 : flux de travail, reproductibilité et rigueur à l'ère des agents

Les chapitres 1 à 4 vous ont appris à construire des modèles et à les évaluer équitablement. Ce chapitre vous apprend à conduire l'ensemble de l'opération de sorte que n'importe qui — vous-même dans six mois compris, et y compris un agent d'IA travaillant dans votre dépôt — puisse la réexécuter et obtenir la même réponse.

Le moment compte. En 2026, une grande partie du code d'un projet de recherche est rédigée par des agents. Cela augmente la valeur des pratiques de ce chapitre au lieu de la diminuer : un agent peut produire une analyse plausible, fausse et irreproductible plus vite que vous ne pouvez la lire. Environnements épinglés, transformations scriptées, expériences suivies et intégration continue : voilà comment vous gardez le contrôle d'un travail que vous n'avez pas tapé vous-même.

## Contenu du chapitre

1. **[5.1 Reproductibilité](5.1_reproducibility.md)** — Reproductibilité contre réplicabilité, et la pile de la reproductibilité : environnements épinglés, graines aléatoires, données brutes immuables, conteneurs et vérifications exécutables. La construction de ce livre lui-même sert d'étude de cas.
2. **[5.2 Suivi d'expériences](5.2_experiment_tracking.ipynb)** — Un TP. Vous construisez un outil minimal de suivi d'expériences en une trentaine de lignes et l'utilisez pour mener une petite étude d'hyperparamètres, puis vous voyez ce que MLflow et Weights & Biases ajoutent par-dessus les mêmes idées.
3. **[5.3 Versionnage des données et des modèles](5.3_data_model_versioning.md)** — Pourquoi Git échoue sur les données, ce que font DVC et ses semblables, la pratique minimale viable à base de sommes de contrôle, et comment versionner les modèles avec des fiches de modèle. En prime : recopier le motif d'intégration continue de ce livre dans le dépôt de votre projet.
4. **[5.4 Calculer au-delà du portable](5.4_compute_beyond_laptop.md)** — Quand quitter votre portable, l'échelle des options depuis le HPC de laboratoire jusqu'au cloud commercial, la discipline des coûts, et l'accès aux données optimisé pour le cloud.

## Objectifs d'apprentissage

À la fin de ce chapitre, vous saurez :

- énoncer la différence entre reproductibilité et réplicabilité, et dire laquelle des deux une vérification donnée met à l'épreuve ;
- épingler un environnement pour qu'un collaborateur (ou l'intégration continue, ou un agent) réexécute votre code avec les mêmes versions de bibliothèques ;
- suivre vos expériences pour que chaque nombre rapporté soit traçable jusqu'à une exécution, un *commit* et une version de jeu de données ;
- décider quand un projet demande plus de calcul qu'un portable, et choisir l'option adéquate la moins chère ;
- mettre en place une intégration continue qui exécute vos carnets (*notebooks*) à chaque *pull request*.

Ces compétences alimentent directement la grille du projet final ([critères de reproductibilité et de documentation](../about_this_book/1.10_MLGEO_FinalProject.md)).
