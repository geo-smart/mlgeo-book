# Vue d'ensemble du chapitre

## Chapitre 3 : l'apprentissage automatique classique en géosciences

Ce chapitre couvre le *machine learning* classique (apprentissage automatique classique, CML) pour les géosciences : des modèles qui apprennent à partir de tables de caractéristiques (*features*) plutôt que de formes d'onde ou d'images brutes. L'apprentissage automatique classique est rapide à construire, peu coûteux à exécuter et facile à interroger. C'est donc le bon endroit pour acquérir les habitudes qui se transposent à l'apprentissage profond : les modèles de référence (*baselines*), les découpages honnêtes des données, et une évaluation qui correspond à l'usage réel du modèle.

### Le parcours du chapitre

1. **Concepts** (3.1) — la taxonomie de la supervision de l'entraînement : apprentissage supervisé, non supervisé, semi-supervisé, auto-supervisé, par renforcement et actif, et où chacun apparaît en géosciences.
2. **Classification et régression** (3.2) — les deux types de problèmes supervisés, un premier flux de travail de bout en bout, et le découpage entraînement/validation/test.
3. **Partitionnement** (3.3) — la découverte de structure non supervisée : métriques de distance, un k-means écrit à la main, diagnostics de silhouette et du coude, partitionnement hiérarchique, et un exercice sur la sismicité volcanique.
4. **Classification binaire** (3.4) — la détection événement contre bruit, la comparaison de classifieurs, les métriques qui comptent quand les classes sont déséquilibrées (précision, rappel, courbes précision-rappel contre ROC), et deux leviers pour traiter le déséquilibre (pondération des classes, déplacement du seuil).
5. **Classification multiclasse** (3.5) — quatre types de sources sismiques, matrices de confusion par classe, ROC un-contre-tous, et l'exercice du classement (*leaderboard*) de la promotion.
6. **La régression logistique à la main** (3.6) — la leçon où la boîte noire s'ouvre : la fonction de perte, la descente de gradient, la différentiation automatique avec PyTorch, et un contrôle de calibration — diagrammes de fiabilité et score de Brier — sur les probabilités prédites.
7. **Arbres, forêts et *boosting*** (3.7) — arbres de décision, régression par forêt aléatoire, importance des caractéristiques et ses pièges (importance par permutation, dépendance partielle, caractéristiques corrélées), et le *gradient boosting* (renforcement par gradient) comme choix par défaut moderne sur données tabulaires.
8. **Entraînement robuste** (3.8) — la validation croisée pour données corrélées : pourquoi les découpages aléatoires mentent sur les séries autocorrélées ; découpages temporels, par groupes (site, événement) et spatiaux en laissant un groupe de côté (*leave-cluster-out*) ; StratifiedGroupKFold pour les petites collections de cas déséquilibrées ; modèles de référence par persistance et intervalles de confiance *bootstrap* sur les scores.
9. **Apprentissage ensembliste** (3.9) — vote, *bagging*, *boosting* et *stacking* (empilement), avec la dispersion des votes de l'ensemble comme première estimation de l'incertitude épistémique et le rappel par classe dans la comparaison des modèles.
10. **Ce qu'il est advenu de l'AutoML** (3.10) — une brève histoire de la recherche automatique de modèles, les morceaux qui ont survécu (optimisation d'hyperparamètres avec Optuna, bons réglages par défaut du *gradient boosting*), et un exercice d'évaluation critique d'un code de modélisation généré par l'IA.

La réduction de dimension (PCA, t-SNE) est traitée au chapitre 2.12 et sert ici d'étape de prétraitement — elle n'est pas réenseignée.

### Le fil de l'évaluation honnête

Une même discipline traverse tous les carnets de ce chapitre :

- **Le modèle de référence d'abord.** Avant tout modèle, établissez ce qu'obtient un prédicteur trivial : la classe majoritaire en classification, la moyenne historique en régression. Un modèle qui ne bat pas le modèle de référence n'a rien appris.
- **Ne jamais évaluer sur les données d'entraînement.** La qualité d'un modèle se mesure sur des données qu'il n'a pas vues. La validation croisée se déroule à l'intérieur de l'ensemble d'entraînement ; l'ensemble de test n'est touché qu'une fois.
- **Les découpages doivent respecter la structure des données.** Les séries autocorrélées et les données spatialement groupées exigent des découpages temporels, en blocs ou par groupes (3.8).
- **Les ensembles de test cachés existent.** L'enseignant détient des variantes régénérées des jeux de données du cours, avec des graines aléatoires privées. Des scores ajustés contre un ensemble de test public montreront un écart sur l'ensemble caché.

La leçon 3.5 met cette discipline en pratique avec un **classement (*leaderboard*) de la promotion** : les étudiants entraînent le classifieur de leur choix sur un découpage canonique d'un jeu réel de sources sismiques, soumettent leurs prédictions par *pull request*, et sont notés par l'intégration continue contre l'ensemble de test public et l'ensemble de test caché.

### Outils

Le chapitre utilise `scikit-learn` comme outil principal, `lightgbm` et le *gradient boosting* par histogrammes de scikit-learn pour les arbres *boostés*, `optuna` pour la recherche d'hyperparamètres, et `pytorch` en 3.6 pour introduire la différentiation automatique. Les jeux de données du cours proviennent du paquet `mlgeo_synth` (générateurs synthétiques à motivation physique), du dépôt de données du cours, et d'une archive Zenodo organisée d'événements sismiques du Nord-Ouest Pacifique.

### Objectifs d'apprentissage

À la fin de ce chapitre, vous saurez :

- Formuler un problème de géosciences comme une classification, une régression ou un partitionnement, et choisir un premier modèle approprié.
- Établir des modèles de référence et évaluer les modèles avec des métriques adaptées au problème, y compris pour des classes déséquilibrées.
- Concevoir des découpages et des schémas de validation croisée qui respectent les corrélations temporelles et spatiales.
- Entraîner, régler et comparer des ensembles d'arbres et d'autres modèles classiques, et rapporter les résultats honnêtement.
- Lire un script de modélisation généré par une machine et en trouver les défauts.

### Devoirs

- **Devoir** : un devoir noté (un problème de classification en géochimie sur roche totale) couvrant la préparation des données, la PCA, le partitionnement et la comparaison de modèles.
- **Jalon du projet final** : une consigne de jalon (3.20) appliquant ces méthodes au jeu de données de votre propre projet.
