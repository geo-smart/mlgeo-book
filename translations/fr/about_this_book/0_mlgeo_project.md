# Le projet MLGeo

Cette page parcourt les étapes de conception d'un projet d'apprentissage automatique en géosciences et renvoie aux chapitres qui enseignent chacune d'elles.

## 1. Cadrer le projet

* Motiver le besoin d'apprentissage automatique dans votre projet scientifique.

Menez une revue de littérature sur les questions scientifiques ouvertes et sur les solutions proposées dans la littérature. Quelles seraient les étapes pour résoudre le problème à la main ? Quelles sont les limites des solutions actuelles ? Un nouvel algorithme de ML sera-t-il assez généralisable pour s'appliquer à dix autres problèmes de recherche ou plus ? Quel est le potentiel à cinq ou dix ans de ce problème particulier, compte tenu des nouvelles technologies, des nouvelles infrastructures de recherche, d'une nouvelle pertinence sociétale ? Existe-t-il des problèmes comparables pour lesquels les outils pourraient être réemployés ?

* Quel est l'état des données ?

Y a-t-il beaucoup de données, et des données étiquetées ? Dispose-t-on d'une expertise humaine ? S'agira-t-il d'un problème d'apprentissage supervisé ou non supervisé ? Les données sont-elles accessibles depuis des archives en accès libre respectant les principes FAIR (*findable, accessible, interoperable, reusable* — faciles à trouver, accessibles, interopérables, réutilisables) ? Quel serait leur DOI ? Existe-t-il des réglementations locales ou des conventions de collecte qui contraignent le partage des données ?


## 2. Organiser le projet — chapitre 1

Ouvrez un dépôt GitHub avec un `README.md`, créez une spécification d'environnement (pixi ou un fichier YML) et employez des noms de fichiers et de dossiers lisibles par un humain comme par une machine. Vérifiez que le nom du projet n'a pas déjà été utilisé.

## 3. Téléchargement des données — chapitre 1

Recensez les données, leurs informations, leurs étiquettes et leur provenance (y compris leur accessibilité depuis des archives en accès libre). Quel est le volume des données ? Quel format serait optimal pour une lecture depuis plusieurs langages (Python, C, R, MATLAB, Julia, etc.) ? Peut-il stocker des métadonnées ? Quelles sont ses performances en entrée/sortie ?

Les données sont-elles géospatiales, ou des séries temporelles ?

Trouvez une plateforme de calcul adaptée au stockage et aux entrées/sorties des données (cloud, grappe Linux de votre établissement, etc.).

Créez un carnet (*notebook*) Jupyter qui documente le téléchargement et le stockage des données.

## 4. Préparation des données — chapitre 2

* **Explorer les données**

Créez un carnet Jupyter pour une exploration préliminaire des données. Documentez :
- Le nom, le type de données
- Le bruit : de quel type est-il (stochastique, valeurs aberrantes, lacunes, etc.)
- La distribution des données : gaussienne, uniforme, logarithmique, etc.
- Les étiquettes (ou attributs cibles)


Visualisez un sous-ensemble des données.

Étudiez les corrélations élémentaires entre attributs.

Comment résoudriez-vous le problème à la main avec ces données ?

Repérez les transformations qui pourraient être utiles (STFT, CWT, PCA par exemple).

Sauvegardez les figures et carnets préliminaires. Documentez vos observations.

* **Préconditionnement des données — chapitre 2**

Copiez les données et travaillez sur ces copies.

Écrivez des fonctions pour toutes les transformations de données, afin qu'elles puissent être appelées automatiquement (et déboguées facilement). Ces fonctions serviront pour les ensembles d'entraînement, de validation et de test.

**Nettoyez les données** : corrigez ou supprimez les valeurs aberrantes, comblez les valeurs manquantes (zéro, moyenne, médiane), ou écartez des données (lorsqu'il y a trop de lacunes, par exemple).
Enregistrez la copie propre des données dans un fichier distinct.

Méfiez-vous des données synthétiques. Des données aléatoires jouets (du bruit tiré d'un générateur, sans physique derrière) vous apprennent peu : un algorithme qui fonctionne sur elles peut se comporter tout autrement sur des observations réelles, alors évitez-les au-delà d'un simple test de fumée. Les données synthétiques physiquement motivées, à vérité terrain documentée, sont une autre affaire. Parce que vous connaissez la vraie réponse, elles sont admissibles pour le développement de méthodes, l'étalonnage comparatif et les ensembles de test cachés. La section 2.10 et le paquet `mlgeo_synth` du livre génèrent ce type de données. Pour la science elle-même, préférez des données recueillies dans le monde réel.

* **Préparation des caractéristiques — chapitre 2**

**Écartez** les attributs inutiles pour la tâche.

**Transformez** les caractéristiques (*features*) — par STFT, par exemple.

Explorez des caractéristiques rapides et prometteuses (p. ex. le PGA pour les mouvements du sol).

**Mettez à l'échelle** les caractéristiques pour les standardiser ou les normaliser. Dans la plupart des cas, les algorithmes de ML ne fonctionneront pas bien sans normalisation des caractéristiques ou des données d'entrée. La mise à l'échelle n'est pas une obligation, mais elle améliore en général le comportement de l'entraînement.
+ *Mise à l'échelle min-max* : retire la valeur minimale, puis normalise par la valeur maximale de la distribution, de sorte que les amplitudes soient comprises entre 0 et 1. Elle convient lorsque les caractéristiques sont des nombres positifs. La fonction intégrée de scikit-learn ``sklearn.preprocessing.MinMaxScaler()`` réalise :

<code>X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))</code>

<code>X_scaled = X_std * (max - min) + min</code>
Scikit-learn dispose de fonctions intégrées pour effectuer la mise à l'échelle.

+ *Standardisation* : retire la moyenne et divise par l'écart-type. La distribution de sortie n'est pas bornée. Elle est plus stable que la mise à l'échelle min-max car moins sensible aux valeurs aberrantes. La fonction intégrée de scikit-learn est ````sklearn.preprocessing.StandardScaler()````

Il existe d'autres manières de normaliser les caractéristiques ou les données d'entrée.
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing



## 5. Réduction de dimension — chapitre 2

Explorez les moyens possibles de réduire la dimension des données (PCA, ICA).

Documentez la transformation des données dans des carnets. Réaffectez les attributs et étiquettes dans les nouvelles coordonnées.

## 6. Conception du modèle — chapitres 3 et 4

Trouvez le modèle de ***référence*** (*baseline*) que le projet de ML est censé battre. Au minimum, votre algorithme de ML doit battre la référence *aléatoire* ; sinon, il y a un problème dans la conception du modèle ou dans les données d'entrée.

Essayez plusieurs algorithmes. Le *no free lunch theorem* (Wolpert, 1995) : il n'existe pas de meilleur algorithme d'apprentissage dans l'absolu, seulement un algorithme très exact sur un jeu de données donné.

Le modèle doit avoir la **complexité minimale** requise pour **minimiser l'erreur attendue du modèle**.


## 7. Entraînement du modèle — chapitres 3 et 4

Séparez les données en trois ensembles : un ensemble d'entraînement pour ajuster le modèle, un ensemble de validation pour régler les hyperparamètres et guider la conception, et un ensemble de test que l'on ne touche qu'une fois, à la fin, pour rendre compte des performances.

Concevez le découpage avant de vous préoccuper de ses proportions. Les données géoscientifiques sont corrélées dans le temps et dans l'espace : un découpage aléatoire fuit généralement de l'information — des échantillons de la même tempête, du même séisme ou de la même station se retrouvent de part et d'autre de la frontière, et le score de test devient optimiste. Ajustez le découpage à la structure de corrélation des données :

- *Découpage temporel* : entraîner sur le passé, tester sur le futur, lorsque les échantillons sont ordonnés dans le temps.
- *Découpage spatial* : mettre de côté des régions entières, avec une zone tampon, lorsque les échantillons sont autocorrélés spatialement.
- *Découpage par groupes* : garder tous les échantillons d'un même événement, d'une même station ou d'un même site du même côté du découpage.

La [section 2.13](../Chapter2-DataManipulation/2.13_MLready_data.ipynb) fait des découpages de référence et des contrôles de fuite de données une partie de la définition même d'un jeu de données prêt pour l'IA ; la [section 3.8](../Chapter3-MachineLearning/3.8_robust_training.ipynb) enseigne la validation croisée attentive aux fuites, y compris ses variantes spatiale et par groupes. Une fois la conception du découpage à l'abri des fuites, les proportions sont secondaires (70/15/15 et 60/20/20 sont courants). La validation croisée sur la partie entraînement-validation — avec des plis qui respectent la même structure temporelle, spatiale ou de groupes — estime l'erreur attendue de l'algorithme d'apprentissage ainsi que sa dispersion.

Sauvegardez les résultats intermédiaires quand c'est possible.

Enregistrez les graines du générateur de nombres aléatoires afin de pouvoir reproduire les résultats.

Évitez d'écrire votre propre bibliothèque maison. Appuyez-vous sur des sources fiables.

Fournissez une bonne documentation, surtout en travail de groupe.

Commencez plus petit que l'exécution finale. Il est conseillé de ne pas utiliser plus de 25 % des ressources disponibles lors de la première conception du modèle.

Choisissez une mesure de performance.
