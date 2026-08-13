# Glossaire FR — terminologie de l'édition française

> **Tableau trilingue :** [`GLOSSARY.md`](GLOSSARY.md) donne, pour chaque
> terme anglais, le terme français *et* le terme espagnol côte à côte, avec
> l'indication des mots que chaque communauté conserve en anglais. C'est le
> tableau à consulter pour lire d'une langue à l'autre. La présente page reste
> la référence du traducteur francophone : elle porte le détail d'usage et
> l'historique des lignes contestées.

## Ce que cette page est — et ce qu'elle n'est pas

C'est un **guide d'usage**, pas une autorité qui prime sur les chapitres. Elle
décrit ce qu'écrivent les chercheurs francophones, contexte par contexte, pour
qu'un traducteur choisisse la bonne forme au bon endroit. Quand un chapitre et
ce tableau divergent, c'est le plus souvent le chapitre qui rapporte l'usage
réel et le tableau qu'il faut corriger : c'est ainsi qu'a été révisée la
plupart des lignes ci-dessous. Ouvrez une issue `[translation] term: …` plutôt
que de réécrire la prose d'un chapitre pour la faire coïncider avec une ligne.

Un mot anglais ne vaut pas un mot français. Plusieurs termes de ce livre
recouvrent deux ou trois concepts distincts — *workflow*, *pipeline*,
*repository*, *cluster*, *notebook*, *build* — et les fondre en un seul mot
français détruit une distinction dont le lecteur a besoin. Ces termes ont donc
**une ligne par sens**.

« EN conservé » = le terme anglais reste dans la prose française (usage
établi), glosé à la première occurrence de chaque chapitre. Ce n'est pas une
tolérance accordée à contrecœur : c'est ce que dit et écrit la communauté.
La règle de fond est : **gloser une fois par chapitre, puis rester cohérent
à l'intérieur du chapitre**. Les lignes marquées ⚑ sont contestées — ouvrez
une issue `[translation] term: …`. Les lignes marquées **erreur** ne sont pas
des préférences : s'y tromper change le sens de la phrase.

## Registre ML — la décision centrale

**Première occurrence substantielle de chaque chapitre :** « le *machine
learning* (**ML** ; apprentissage automatique) ».
**Ensuite :** **ML** ou *machine learning* pour le champ disciplinaire et la
pratique de recherche ; la famille « apprentissage… » pour les paradigmes
nommés — *apprentissage supervisé, non supervisé, auto-supervisé, par
renforcement, profond*.

C'est une **décision de registre, pas un jugement sur « apprentissage
automatique »**, qui reste correct, courant, et conservé comme glose. L'usage
universitaire et scientifique francophone est réellement mixte :

- le CNRS (Formation Entreprises) intitule une formation « Machine Learning et
  Deep Learning pour la vision par ordinateur » alors que le corps du
  descriptif écrit « apprentissage automatique » et « apprentissage profond » ;
- la page du M1 *Mathématiques et Apprentissage Statistique* de
  l'Université Paris-Saclay emploie *machine learning* et *data science* sans
  les traduire, à côté d'« apprentissage statistique » — un registre mixte
  assumé dans un diplôme francophone ;
- Inria publie en français aussi bien « Allier mathématiques et machine
  learning… » qu'« Apprentissage automatique et réseaux d'information » ;
- la biographie de chaire du Collège de France écrit « l'apprentissage
  automatique » (*machine learning*) : la forme française appariée au nom
  anglais du champ.

L'édition espagnole tranche **différemment et volontairement** : «aprendizaje
automático» reste sa forme de prose par défaut, avec le sigle ML enseigné
explicitement. Voir [`GLOSSARY_es.md`](GLOSSARY_es.md). Ce n'est pas une
incohérence à normaliser : les deux communautés n'écrivent pas pareil.

| English | Français | EN conservé ? | Note |
|---|---|---|---|
| machine learning | *machine learning* (**ML**) ; « apprentissage automatique » en glose | **oui** pour le champ ; non pour les paradigmes | voir la section ci-dessus. Les paradigmes restent français : apprentissage supervisé, non supervisé, auto-supervisé, par renforcement, profond |
| training set | ensemble d'entraînement | non | jamais « set d'entraînement » |
| validation set | ensemble de validation | non | |
| test set | ensemble de test | non | |
| hidden test set | ensemble de test caché | non | vocabulaire du cours |
| data leakage | fuite de données | non | **erreur** si non qualifié : toujours « de données ». Sans qualificatif, collision avec la fuite spectrale |
| spectral leakage | fuite spectrale | non | **erreur** si non qualifié : toujours « spectrale ». Étalement d'énergie entre bins fréquentiels dû à la troncature par la fenêtre — sans rapport avec la fuite de données |
| overfitting | surapprentissage | non | ⚑ « surajustement » existe ; on fixe surapprentissage |
| underfitting | sous-apprentissage | non | |
| cross-validation | validation croisée | non | |
| grouped cross-validation | validation croisée par groupes | non | |
| fold | pli | ⚑ | « pli » est correct mais rare ; alternative : « bloc de validation » |
| baseline | *baseline* (modèle de référence) | **oui**, glosé une fois par chapitre | les chercheurs disent *baseline* ; ensuite l'une ou l'autre forme, tenue dans tout le chapitre |
| persistence (forecast) | persistance | non | usage Météo-France |
| feature | « variable », « variable explicative » ou « caractéristique » selon la discipline | *feature* en pont vers le code | ne pas imposer « caractéristique » là où un statisticien dirait « variable explicative » ; identifiants (`n_features`, `feature_names`) inchangés |
| label | étiquette | non | |
| supervised learning | apprentissage supervisé | non | |
| unsupervised learning | apprentissage non supervisé | non | |
| self-supervised learning | apprentissage auto-supervisé | non | |
| reinforcement learning | apprentissage par renforcement | non | |
| active learning | apprentissage actif | non | |
| clustering | *clustering* ou « partitionnement », introduits ensemble | **oui**, au choix | ⚑ assoupli : *clustering* est normal dans la prose ML francophone. Introduire « le *clustering* (partitionnement) » une fois, puis s'y tenir dans le chapitre |
| classification | classification | non | |
| regression | régression | non | |
| random forest | forêt aléatoire | non | |
| gradient boosting | — | **oui** (*gradient boosting*) | glosé « renforcement par gradient » une fois |
| neural network | réseau de neurones | non | |
| deep learning | apprentissage profond | *deep learning* toléré première mention | |
| convolutional neural network (CNN) | réseau (de neurones) convolutif | sigle CNN conservé | |
| recurrent neural network (RNN) | réseau récurrent | sigle RNN conservé | |
| transformer | — | **oui** (*transformer*) | usage établi |
| autoencoder | auto-encodeur | non | |
| dropout | — | **oui** (*dropout*) | glosé « extinction aléatoire de neurones » une fois |
| MC dropout | — | **oui** | |
| deep ensemble | ensemble profond | ⚑ | ou « ensemble de réseaux » |
| calibration | calibration | non | ⚑ « étalonnage » réservé aux instruments |
| reliability diagram | diagramme de fiabilité | non | usage vérification |
| Brier score | score de Brier | « score » conservé | `mean((p − o)²)`, `o ∈ {0,1}` ; plus bas = meilleur. Leçon 3.6 |
| coverage (of intervals) | couverture (empirique) des intervalles | non | |
| expected calibration error (ECE) | erreur de calibration attendue | sigle ECE conservé | moyenne pondérée de `abs(exactitude(b) − confiance(b))` sur les paquets de confiance `b` ; le nombre dépend du découpage, il faut le dire. Leçon 4.5 |
| skill (forecast) | — | **oui** (*skill*) glosé « score de compétence » | usage opérationnel. `1 − score/score_réf` : un **nombre** calculé contre une prévision de référence (persistance, climatologie). Toujours nommer la référence |
| lead time | échéance | non | usage Météo-France |
| epistemic / aleatoric uncertainty | incertitude épistémique / aléatoire | non | |
| hyperparameter | hyperparamètre | non | |
| epoch | époque | non | |
| batch | lot | *batch* toléré | |
| learning rate | taux d'apprentissage | non | |
| loss function | fonction de perte | non | **erreur** : ni « fonction de coût » (réservé à *cost*), ni confusion avec l'objectif. L'erreur sur **un** exemple |
| objective (function) | fonction objectif | non | **erreur** si employé comme synonyme de *loss* : c'est la perte agrégée **plus** les termes de régularisation, ce que l'optimiseur minimise |
| cost | coût | non | la conséquence réelle d'une erreur (alerte manquée, évacuation inutile). Le mot n'est pas interdit ; la confusion l'est |
| split criterion (arbres) | critère de partition | non | Gini, entropie, erreur quadratique : ce sont des critères de partition, pas des « fonctions de coût » |
| gradient descent | descente de gradient | non | |
| accuracy | exactitude, ou « taux de bonnes classifications » | non | **erreur : jamais « précision »**. `(TP+TN)/(TP+TN+FP+FN)`, alors que *precision* = `TP/(TP+FP)`. Première leçon avec formule : 3.4 |
| precision | précision | non | **erreur** si employé pour *accuracy*. `TP/(TP+FP)`. Leçon 3.4 |
| recall | « rappel » ou « sensibilité » selon le domaine | *recall* possible en prose technique | `TP/(TP+FN)`. ⚑ assoupli : la formule fixe la quantité, le domaine choisit le mot — recherche d'information et ML disent « rappel », les lecteurs médicaux et détection attendent « sensibilité ». Énoncer la formule à la première occurrence, puis un seul mot par chapitre. Leçon 3.4 |
| F1 | score F1 | sigle F1 conservé | `2·P·R/(P+R)`, moyenne harmonique de la précision et du rappel |
| eval set (agents) | jeu d'évaluation | non | vocabulaire du chapitre 6 |
| agent | agent | non | |
| large language model (LLM) | grand modèle de langage | sigle LLM conservé | |
| prompt | — | **oui** (*prompt*) | glosé « consigne » une fois |
| pretraining / fine-tuning | pré-entraînement / affinage | *fine-tuning* toléré glosé | ⚑ |
| reproducibility | reproductibilité | non | |
| sampling rate | fréquence d'échantillonnage | non | |
| aliasing | repliement (de spectre) | *aliasing* toléré glosé | |
| leaderboard | classement | *leaderboard* toléré glosé | vocabulaire du cours |
| data card | fiche de données | non | |

## Un mot anglais, plusieurs sens — le tableau de contexte

Ces termes n'ont pas *une* traduction. La colonne « sens » décide ; c'est elle
qu'il faut lire en premier. Ces lignes remplacent d'anciennes équivalences
uniques (« workflow → flux de travail », « cluster → grappe », « notebook →
carnet ») que les chapitres contredisaient — et les chapitres avaient raison.

| Terme anglais | Sens visé | Forme française | À éviter dans ce sens |
|---|---|---|---|
| workflow | démarche de recherche, au niveau conceptuel | « démarche (de recherche) », « protocole », parfois « processus » | « flux de travail » par réflexe |
| workflow | calcul exécutable et reproductible | « workflow de calcul », « workflow reproductible » | prétendre que les chercheurs ne disent jamais *workflow* — `translations/fr/MANIFEST.yml` et les chapitres 5.3–5.4 disent le contraire |
| workflow | fichier d'automatisation CI | « workflow GitHub Actions » ; chemin `.github/workflows/` **jamais traduit** | « archive », « protocole » |
| workflow | circulation du travail entre personnes | « flux de travail » | l'employer comme traduction unique de tous les sens |
| pipeline | suite ordonnée de transformations du signal ou des données | « chaîne de traitement » | « démarche », « flux de travail » |
| pipeline | enchaînement de modélisation (hors objet d'API) | « chaîne de modélisation » | |
| pipeline | objet scikit-learn prétraitement + estimateur | *pipeline* ; « un *pipeline* scikit-learn (`Pipeline`) » à la première occurrence | traduire l'identifiant `Pipeline` |
| repository | projet Git ou GitHub versionné | « dépôt Git », « dépôt GitHub » | **erreur** : « archive ». « référentiel » seulement en citant l'interface |
| archive | dépôt de préservation et de diffusion (HAL, Zenodo) | « archive ouverte », « archive institutionnelle », « archive de données » | l'employer pour un projet Git actif |
| data repository | portail de données géré (Data Terra, centre de données national) | « entrepôt de données », « dépôt de données », ou le nom du portail | choisir d'après l'orthographe anglaise plutôt que d'après la fonction |
| object storage | stockage objet (S3, MinIO, buckets) | « stockage objet » | « archive », sauf si la préservation est bien la fonction |
| notebook | document interactif, prose orientée recherche | « notebook Jupyter (carnet) » à la première occurrence, puis « notebook » | imposer « carnet » comme seule forme écrite ; traduire `.ipynb` ou un nom de fichier |
| cluster | groupes issus d'un algorithme de partitionnement | « groupe », « classe », « partition » selon la méthode ; *cluster* en prose technique | ⚑ « grappe » comme défaut universel — ce n'est pas l'usage ML actuel |
| cluster | infrastructure de calcul | « cluster de calcul » (« grappe de calcul » possible) | laisser l'ambiguïté avec le sens ML dans un même chapitre |
| cloud | infrastructure distante | *cloud*, « informatique cloud » | « informatique en nuage » imposé ; les noms de produits (AWS, GCP, Azure) restent inchangés |
| HPC | calcul parallèle sur centre de calcul | « HPC (calcul haute performance) » une fois, puis HPC | traduire le sigle |
| job | exécution soumise à un ordonnanceur ou à la CI | *job* ; « tâche » quand il s'agit de travail en général | traduire les champs et commandes de l'ordonnanceur |
| build | étape de CI, empaquetage | *build* | forcer « construction », qui convient à la fabrication du livre publié |
| build | fabrication du livre publié | « construction (du livre) » | |
| dataset | jeu de données en prose | « jeu de données » | chasser *dataset* d'une phrase qui nomme un objet (`Dataset` Xarray, jeu HF) |
| benchmark | jeu de données de référence | « jeu de référence » | « banc d'essai » pour un simple jeu de données |
| benchmark | protocole ou suite de comparaison | « banc d'essai » | |
| feature engineering | construction de variables | *feature engineering* (construction de variables / de caractéristiques) une fois, puis l'une des deux | « ingénierie des caractéristiques » imposée partout : ça sonne traduit |
| inference | phase de déploiement / prédiction | « inférence » pour la phase, « prédiction » pour l'acte concret | employer « inférence » au sens statistique d'estimation sans le préciser |

## Identifiants — jamais traduits

Noms de fonctions, de classes et d'arguments (`Pipeline`, `.fit()`,
`DataLoader`, `StandardScaler`) ; noms de fichiers et chemins
(`environment.yml`, `pixi.lock`, `.github/workflows/`, `*.ipynb`) ; clés de
configuration YAML/TOML/JSON ; noms de colonnes utilisés par le code
(`target`, `label`, `uncertainty`, `pick_weight`) ; sorties exécutées et
rapports de modèle. Le patron d'écriture est « la variable cible `target` », et
jamais une colonne renommée qui casse le notebook.
