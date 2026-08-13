# Glossaire

Termes employés tout au long du livre, des deux côtés de son public : le vocabulaire de l'apprentissage automatique pour les étudiants de géosciences, et le vocabulaire des géosciences et du traitement du signal pour les étudiants d'informatique. Les entrées sont classées par ordre alphabétique du terme anglais, qui reste affiché en tête de chaque entrée : c'est sous ce terme que vous retrouverez la notion dans la littérature et dans le code.

Le [glossaire trilingue](https://github.com/geo-smart/mlgeo-book/blob/main/translations/GLOSSARY.md) donne chaque terme dans les trois langues côte à côte et indique quels mots anglais chaque communauté conserve dans sa prose : la francophone et l'hispanophone ne gardent pas les mêmes.

```{glossary}

Ablation — ablation
: Une expérience qui retire un composant d'un modèle ou d'un flux de travail (une caractéristique, un terme de la perte, une couche) puis réentraîne, de sorte que l'écart de score mesure ce que ce composant apporte. Le carnet 4.7 procède ainsi à l'ablation du terme physique de la perte d'un PINN.

Agent — agent
: Un grand modèle de langage (LLM) enveloppé dans une boucle capable d'appeler des outils (exécuter du code, lire des fichiers, chercher), d'observer les résultats et d'agir de nouveau en vue d'un but, au lieu de produire une réponse textuelle unique. Le chapitre 6 construit et évalue des agents pour des tâches de recherche.

AI-ready data — données prêtes pour l'IA
: Des données organisées pour qu'un modèle puisse les consommer directement et qu'un lecteur puisse leur faire confiance : unités et échantillonnage cohérents, provenance et licences documentées, étiquettes propres, formats lisibles par une machine. Le chapitre 2 transforme des téléchargements bruts en jeux de données prêts pour l'IA.

Baseline — modèle de référence (*baseline*)
: Le modèle crédible le plus simple pour une tâche — la persistance ou la climatologie en prévision, la classe majoritaire en classification, la régression linéaire pour des données tabulaires. Tout modèle complexe doit battre la référence pour justifier sa complexité.

Calibration — calibration
: L'accord entre la confiance annoncée par un modèle et la réalité : parmi les prédictions faites avec une probabilité de 80 %, environ 80 % devraient être correctes, et des intervalles de prédiction à 90 % devraient couvrir environ 90 % des issues. La calibration se mesure (diagrammes de fiabilité, couverture des intervalles) ; elle ne se suppose jamais.

Coda — coda
: En sismologie, la queue d'un enregistrement sismique après les arrivées principales P et S : des ondes diffusées dont l'amplitude décroît progressivement avec le temps. La forme de la coda aide à distinguer les types de source (la coda d'une explosion diffère de celle d'un séisme).

Corner frequency — fréquence coin
: La fréquence à laquelle le spectre d'amplitude d'un séisme passe d'un plateau à une décroissance. Elle varie à l'inverse de la durée de rupture, si bien que les séismes les plus forts ont les fréquences coin les plus basses ; c'est une caractéristique standard pour décrire les sources sismiques.

Data leakage — fuite de données
: Tout chemin par lequel une information indisponible au moment de la prédiction atteint le modèle pendant l'entraînement — ajuster une mise à l'échelle sur le jeu de données complet avant le découpage, des échantillons de test corrélés à des échantillons d'entraînement, des valeurs futures qui informent des prédictions passées. La fuite gonfle des scores qui s'effondrent ensuite sur des données véritablement nouvelles. Voir aussi *leakage (spatial and temporal)* et, pour le terme sans rapport issu du traitement du signal, *spectral leakage*.

Deep ensemble — ensemble profond
: Plusieurs copies d'un même réseau entraînées à partir de graines aléatoires différentes. Moyenner leurs prédictions améliore généralement l'exactitude, et leur désaccord sur un échantillon estime l'incertitude épistémique (carnet 4.5).

Dimensionality reduction — réduction de dimension
: Comprimer de nombreuses caractéristiques en un petit nombre tout en préservant la structure, soit pour visualiser les données (PCA, t-SNE, UMAP), soit pour fournir une représentation plus compacte à un modèle. Le goulot d'étranglement d'un auto-encodeur en est une forme apprise.

Epistemic vs. aleatoric uncertainty — incertitude épistémique et incertitude aléatoire
: L'incertitude épistémique vient de ce que le modèle ne sait pas — trop peu de données, des conditions jamais vues — et se réduit à mesure que les données s'accumulent ; les ensembles et le *MC dropout* l'estiment. L'incertitude aléatoire est le caractère aléatoire du processus ou de la mesure elle-même (bruit du capteur) et ne se réduit pas avec plus de données d'entraînement.

Eval set — jeu d'évaluation
: Une collection soignée de cas de test à réponses connues, assortie d'une règle de notation automatique, qui sert à mesurer un LLM ou un système d'agents au lieu de se fier à des impressions. Le carnet 6.3 en construit un à partir de données synthétiques à vérité terrain connue.

Expected calibration error (ECE) — erreur de calibration attendue
: Un résumé en un seul nombre d'un diagramme de fiabilité : l'écart moyen entre la confiance annoncée et l'exactitude observée, pondéré par le nombre de prédictions tombant dans chaque intervalle de confiance. Calculée aux côtés des diagrammes de fiabilité dans les carnets 3.6 et 4.5.

Feature engineering — ingénierie des caractéristiques
: Construire des entrées de modèle informatives à partir de données brutes en mobilisant la connaissance du domaine : rapports STA/LTA, statistiques spectrales, moyennes glissantes, kurtosis. L'apprentissage automatique classique (chapitre 3) en dépend ; les réseaux profonds apprennent au contraire de nombreuses caractéristiques directement à partir des données brutes.

Flicker noise — bruit de scintillement (bruit en 1/f)
: Un bruit dont la puissance croît vers les basses fréquences (à peu près en 1/f), courant dans les séries de positions GNSS et dans les instruments électroniques. Contrairement au bruit blanc, il ne s'atténue pas par moyennage sur des enregistrements plus longs : il biaise donc les estimations de tendance et d'incertitude si on le modélise comme blanc.

[Git](https://git-scm.com)
: Le système de contrôle de version employé tout au long du cours pour enregistrer, comparer et partager l'historique du code et du texte.

[GitHub](https://github.com)
: Le service d'hébergement où vivent les dépôts du cours, les jeux de données et les projets des étudiants, bâti autour de Git plus les issues et les *pull requests*.

Grouped cross-validation — validation croisée par groupes
: Une validation croisée qui garde dans le même pli tous les échantillons partageant un groupe — la même station, le même site de terrain ou le même séisme. Les scores mesurent alors la généralisation à de nouveaux groupes plutôt qu'une interpolation à l'intérieur de groupes déjà familiers (section 3.8).

Heteroscedastic noise — bruit hétéroscédastique
: Un bruit dont l'amplitude varie d'un échantillon à l'autre — un réseau mixte d'instruments de laboratoire et de terrain, ou des stations aux conditions de site différentes. Lorsque le niveau de bruit propre à chaque échantillon est connu, des fonctions de perte pondérées par la variance récupèrent l'exactitude que l'on abandonne en ignorant ces métadonnées (carnet 4.5).

Hidden test set — ensemble de test caché
: Des données mises de côté, jamais inspectées ni touchées pendant le développement du modèle, et notées une seule fois, à la fin. Le classement (*leaderboard*) de prévision de la classe (4.10) en utilise un ; y toucher à répétition le transforme en ensemble de validation.

Instrument response — réponse instrumentale
: La fonction de transfert d'un capteur et de sa chaîne d'enregistrement, qui relie le mouvement réel du sol aux *counts* enregistrés. Les enregistrements sismiques et géodésiques doivent en être corrigés avant que leurs amplitudes ne portent des unités physiques.

Interpolation vs extrapolation — interpolation et extrapolation
: Prédire à l'intérieur de la région couverte par les données d'entraînement, ou à l'extérieur. La plupart des modèles appris se dégradent brutalement, et parfois silencieusement, en extrapolation ; la distribution d'entraînement est un contrat, et la conception du découpage (3.8) ainsi que les contrôles hors plage (4.5) déterminent si un score parle de l'un ou de l'autre régime.

Jupyter notebook — carnet Jupyter
: Un document mêlant du code exécutable, ses sorties et un texte narratif ; c'est le format de la plupart des chapitres de ce livre.

Leakage (spatial and temporal) — fuite de données (spatiale et temporelle)
: Les deux formes de fuite qui dominent en données géoscientifiques. Un découpage aléatoire de données autocorrélées place des quasi-doublons de part et d'autre du découpage : pixels ou stations voisins (spatial), fenêtres temporelles adjacentes ou chevauchantes (temporel). Les remèdes sont des découpages spatiaux par blocs ou avec zone tampon, et des découpages qui respectent l'ordre du temps.

Linear probe — sonde linéaire
: Un classifieur formé d'un encodeur pré-entraîné gelé et d'une petite tête linéaire entraînée. Il mesure ce que portent les caractéristiques pré-entraînées à elles seules ; l'affinage (*fine-tuning*), lui, met aussi à jour les poids de l'encodeur. Le carnet 4.6 confronte une sonde linéaire à un entraînement depuis zéro.

Machine learning (ML) — *machine learning* (ML ; apprentissage automatique)
: Le champ disciplinaire dont traite ce livre : des méthodes qui apprennent une relation à partir de données plutôt que de la recevoir sous forme de règles écrites à la main. L'édition française écrit « le *machine learning* (ML ; apprentissage automatique) » à la première occurrence substantielle de chaque chapitre, puis **ML** ou *machine learning* pour le champ et la pratique de recherche ; les paradigmes nommés gardent la famille française — apprentissage supervisé, non supervisé, auto-supervisé, par renforcement, profond. C'est un choix de registre, aligné sur l'usage mixte des laboratoires et des formations francophones : « apprentissage automatique » reste parfaitement correct et sert de glose.

MASE
: *Mean absolute scaled error* — erreur absolue moyenne mise à l'échelle : l'erreur absolue moyenne d'une prévision divisée par celle d'un modèle de référence naïf (persistance ou naïf saisonnier) sur la même série. Un MASE inférieur à 1 bat la référence ; supérieur à 1, le modèle lui est inférieur (carnet 4.10).

MC dropout (Monte Carlo dropout) — *MC dropout*
: Exécuter un réseau entraîné de nombreuses fois en laissant le *dropout* actif au moment de la prédiction. Chaque passage échantillonne un sous-réseau légèrement différent, et la dispersion des sorties approche l'incertitude épistémique avec un seul modèle entraîné. Comparé face à face avec un ensemble profond dans le carnet 4.5.

[MyST](https://mystmd.org)
: *Markedly Structured Text*, la variante de Markdown et le système de construction avec lesquels ce livre est écrit et publié.

Physics-informed neural network (PINN) — réseau de neurones informé par la physique
: Un réseau entraîné en ajoutant à la fonction de perte, à côté de l'écart aux données, le résidu d'une équation gouvernante (loi de conservation, équation de diffusion), de sorte que les prédictions soient tirées vers des solutions physiquement cohérentes (carnet 4.7).

[pixi](https://pixi.sh)
: Le gestionnaire de paquets et d'environnements employé pour installer la pile logicielle du cours. Il résout les paquets de l'écosystème conda en un fichier de verrouillage, si bien que chaque étudiant et l'intégration continue construisent le même environnement.

[pooch](https://www.fatiando.org/pooch/)
: Une petite bibliothèque Python qui télécharge un fichier de données depuis une URL, le met en cache localement et vérifie sa somme de contrôle SHA256, de sorte qu'une analyse a démontrablement tourné sur le fichier voulu.

Spectral leakage — fuite spectrale
: En traitement du signal, l'étalement de l'énergie d'une fréquence vers les cases fréquentielles voisines lorsque la fenêtre analysée ne contient pas un nombre entier de cycles. Le fenêtrage (*tapering*) la réduit. Sans rapport avec la fuite de données (*data leakage*).

STA/LTA
: Moyenne à court terme sur moyenne à long terme de l'amplitude d'un signal : un rapport glissant qui bondit quand un transitoire arrive au-dessus du bruit de fond. Le déclencheur classique pour détecter les séismes dans des données sismiques continues, et une caractéristique construite standard.

Supervised, unsupervised, and self-supervised learning — apprentissage supervisé, non supervisé et auto-supervisé
: L'apprentissage supervisé ajuste des entrées à des étiquettes fournies par des humains ; l'apprentissage non supervisé trouve de la structure (groupes, plongements de faible dimension) sans étiquettes ; l'apprentissage auto-supervisé fabrique des étiquettes à partir des données elles-mêmes — masquer une partie de l'entrée et la prédire — de sorte que le pré-entraînement puisse exploiter des archives non étiquetées sans limite (carnet 4.6).

Tolerance-based reproducibility — reproductibilité à tolérance
: Déclarer qu'un résultat est reproduit lorsqu'une réexécution le retrouve à une tolérance numérique annoncée près, plutôt que bit à bit. L'arithmétique en virgule flottante, l'ordre d'exécution parallèle et les différences de matériel font de l'égalité exacte un mauvais test pour des pipelines scientifiques (chapitre 5).

Transformer — *transformer*
: L'architecture bâtie sur des couches d'auto-attention empilées avec des encodages de position, introduite par Vaswani et al. (2017). Elle est au fondement des grands modèles de langage et des systèmes de prévision actuels ; le carnet 4.4 en construit un petit à partir de ses pièces.

[Zarr](https://zarr.dev)
: Un format de stockage pour tableaux N-dimensionnels découpés en blocs et compressés, conçu pour que les magasins d'objets du cloud puissent servir des lectures partielles en parallèle. Le choix courant pour les grandes données géoscientifiques maillées, aux côtés de NetCDF.

```
