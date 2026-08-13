# Jalon du projet final : apprentissage profond sur votre jeu de données prêt pour l'IA

**Objectif** : démontrer que vous savez implémenter, entraîner, diagnostiquer et évaluer de façon critique des modèles d'apprentissage profond sur votre propre jeu de données prêt pour l'IA, les comparer à l'apprentissage automatique classique, et livrer un logiciel reproductible.

Tout ce qui est exigé ci-dessous est enseigné avec du code fonctionnel dans ce chapitre. Quand une exigence nomme un carnet, réutilisez le motif de ce carnet sur vos propres données.

---

## 1. Préparation et exploration du jeu de données (10 %)

- **Utilisation de données prêtes pour l'IA (4 %)** : employez le jeu de données prêt pour l'IA que vous avez préparé plus tôt, avec un prétraitement cohérent pour tous les modèles. Décrivez les entrées, leur signification physique, leurs modalités et leurs dimensions.
- **Analyse exploratoire des données (3 %)** : visualisations et synthèses de la distribution des données et de leur structure temporelle ou spatiale.
- **Formulation du problème (3 %)** : définissez la tâche (régression ou classification) et mettez les données en forme pour chaque architecture (fenêtrage pour les modèles de séquences, remise en forme pour les CNN).

---

## 2. Comparaison avec l'apprentissage automatique classique (10 %)

- **Modèles de référence (5 %)** : rapportez vos résultats d'apprentissage automatique classique du jalon précédent (forêt aléatoire, *gradient boosting* — renforcement par gradient — ou équivalent). Peu de travail nouveau ; l'objectif est d'avoir une ligne de référence.
- **Comparaison des performances (5 %)** : comparez modèles classiques et modèles profonds avec les mêmes métriques sur les mêmes découpages.

---

## 3. Exploration des architectures (30 %)

- **Au moins trois architectures (8 %)** : implémentez et entraînez au moins trois modèles parmi : MLP, CNN 1-D ou 2-D, LSTM, encodeur *transformer*, auto-encodeur suivi d'une tête de classification ou de régression. Les cinq sont construits dans les carnets 4.1 à 4.6. Le U-Net est facultatif, pas exigé. Justifiez chaque choix au regard de vos données et de votre type de problème. Écrivez chaque architecture avec les dimensions de ses couches et ses fonctions d'activation.
- **Exploration des hyperparamètres (7 %)** : explorez systématiquement le taux d'apprentissage, la taille des couches et les autres hyperparamètres, en suivant le TP du carnet 4.5. Documentez chaque expérience ; un tableau d'exécutions vaut mieux que de la prose éparse.
- **Étude d'ablation (5 %)** — exigée : retirez un composant de votre meilleur modèle (un bloc de couches, le *dropout* — l'extinction aléatoire de neurones —, la normalisation par lots, un groupe de caractéristiques ou un terme de perte) et rapportez la variation de performance. Une ablation menée soigneusement suffit ; dites ce qu'elle vous apprend sur le composant.
- **Perte informée par la physique (4 %)** : ajoutez un terme de perte informé par la physique, ou autrement instruit par le domaine, là où votre problème s'y prête (le motif du carnet 4.7). Si votre problème ne s'y prête pas, dites pourquoi en un court paragraphe ; une réponse négative bien argumentée vaut la totalité des points.
- **Innovation (6 %)** : architectures hybrides, fonctions de perte sur mesure ou augmentation de données propre aux géosciences.

---

## 4. Évaluation des performances (20 %)

- **Évaluation quantitative (6 %)** : des métriques pour tous les modèles — exactitude, précision, rappel, score F1, RMSE ou mesures propres au domaine. Les problèmes multi-classes rapportent la précision et le rappel par classe. Indiquez l'optimiseur, le taux d'apprentissage et la taille de lot pour chaque modèle entraîné.
- **Généralisation et test hors distribution (7 %)** : évaluez sur des données inédites ou hors distribution et discutez le surapprentissage ou le sous-apprentissage.
- **Expérience au choix (4 %)** — choisissez-en UNE :
  - *Pré-entraînement ou entraînement à partir de zéro* : pré-entraînez un encodeur sur vos données sans étiquettes (auto-encodeur ou reconstruction masquée, motif du carnet 4.6), puis entraînez une sonde linéaire sur l'encodeur gelé — ou affinez l'encodeur entier — avec une petite fraction étiquetée, et comparez à un entraînement à partir de zéro sur la même fraction. Dites laquelle des deux voies vous avez suivie ; une sonde maintient les poids de l'encodeur fixes, un affinage les met à jour.
  - *Incertitude par ensemble profond* : entraînez votre meilleur modèle à partir de 5 graines aléatoires, rapportez la performance de la moyenne d'ensemble et la variance de prédiction par échantillon, et montrez sur quels échantillons l'ensemble diverge (motif du carnet 4.5).
- **Visualisation des résultats (3 %)** : matrices de confusion, courbes de perte en fonction de l'époque, cartes d'erreur ou équivalents.

---

## 5. Livraison logicielle et qualité du code (15 %)

- **Pratique d'entraînement standard (7 %)** : code modulaire, un carnet par section clairement délimitée. Traitez : (1) la préparation des données avec la description des ensembles d'entraînement, de validation et de test, (2) l'architecture et la conception du modèle, (3) la stratégie d'entraînement (taille de lot, optimiseur, planificateur) avec les courbes d'apprentissage, (4) l'évaluation et la généralisation.
- **Sauvegarde des résultats (4 %)** : enregistrez les poids du modèle, les journaux d'entraînement et les métriques de performance dans des fichiers CSV/JSON versionnés avec le dépôt.
- **Qualité du code et documentation (4 %)** : lisible, commenté, reproductible. Le README du dépôt indique comment exécuter les carnets, et dans quel ordre.

---

## 6. Rapport et interprétation (10 %)

- **Communication scientifique (3 %)** : un rapport clair et concis, avec des figures et des tableaux appropriés.
- **Apports pour le domaine (2 %)** : ce que les résultats signifient pour le problème géoscientifique : pertinence physique, limites des données, applications possibles.
- **Annexe de diagnostics d'entraînement (5 %)** — exigée : les courbes d'apprentissage de votre modèle final ET d'au moins une exécution défaillante ou imparfaite (perte divergente, surapprentissage, taux d'apprentissage qui rampe). Diagnostiquez l'exécution défaillante en 2 ou 3 phrases avec le vocabulaire du carnet 4.5. Les exécutions défaillantes sont la preuve d'un travail systématique, pas quelque chose à cacher.

---

## 7. Considérations computationnelles et éthiques (5 %)

- **Déclaration du calcul (3 %)** : temps d'entraînement, matériel employé et empreinte mémoire de chaque modèle, et comment le coût de calcul a influencé vos choix. Si vous avez utilisé des heures sur un calculateur national (Jean Zay ou une autre machine GENCI, un mésocentre), indiquez-le aussi.
- **Éthique et déclaration d'usage de l'IA (2 %)** : réfléchissez aux biais de vos données et à la transparence de vos prédictions. Incluez une déclaration d'usage de l'IA obligatoire, d'un paragraphe : quels assistants IA ou outils de génération de code vous avez employés, pour quelles parties du travail, et comment vous en avez vérifié la production. L'usage d'outils d'IA est autorisé ; ne pas le déclarer ne l'est pas.

---

**Total : 100 %**
