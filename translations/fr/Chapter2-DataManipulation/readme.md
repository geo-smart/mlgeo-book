# Vue d'ensemble du chapitre

## Chapitre 2 : Données géoscientifiques prêtes pour l'IA

Ce chapitre est le pilier « données prêtes pour l'IA » du cours. En géosciences, les projets de *machine learning* (apprentissage automatique) réussissent ou échouent sur la qualité de leurs données, et l'essentiel du travail se situe en amont de tout modèle : comprendre ce que sont les données, lire et écrire les formats standard, nettoyer des tables, remodeler des tableaux, rééchantillonner, caractériser des distributions, transformer et filtrer des signaux, générer des données synthétiques honnêtes, construire des caractéristiques (*features*) et réduire la dimension. Le chapitre se clôt par une leçon de synthèse qui définit opérationnellement les données prêtes pour l'IA — provenance, métadonnées, structures bien rangées (*tidy*), découpages de référence (*benchmark splits*) et contrôles de fuite de données — et cette définition est notée dans le projet final. Trois flux de données traversent le chapitre et reçoivent un traitement égal : les champs maillés (rasters climatiques, imagerie), les séries de capteurs régulières à haute cadence (sismogrammes à 100 Hz, marégraphes horaires, GNSS journalier) et les observations ponctuelles éparses et irrégulières (réseaux piézométriques multi-puits, campagnes de terrain) — parce que tout projet de géosciences finit par manipuler au moins deux d'entre eux.

### L'arc du chapitre

Les leçons s'enchaînent dans l'ordre :

1. **2.1 Définitions des données** — les modalités de données en géosciences ; tableaux vs tables de données ; formats courants et formats optimisés pour le cloud.
2. **2.2 Formats de données** — lecture et écriture pratiques de CSV, GeoJSON, GeoTIFF, netCDF, HDF5, Parquet et Zarr ; comparaison des tailles de fichiers sur disque.
3. **2.3 DataFrames pandas** — séries et tables de données, manipulation des dates, filtrage, regroupement, agrégation et cartographie de métadonnées de stations.
4. **2.4 Préparer les DataFrames** — nettoyage d'une table de géochimie sur roche totale : données manquantes, valeurs sentinelles, valeurs censurées à une limite de détection, absence de données informative, corrélations et distributions par classe.
5. **2.5 Tableaux** — tableaux NumPy et Xarray, indexation et remodelage, dimensions étiquetées, et un premier regard sur les tenseurs PyTorch.
6. **2.6 Rééchantillonnage** — rééchantillonnage statistique (randomisation, *bootstrap*, Monte-Carlo) pour l'incertitude, puis rééchantillonnage de signaux : décimation anti-repliement d'un enregistrement marégraphique, interpolation avec politique de lacunes d'une série GNSS dégradée, agrégation d'un réseau multi-puits irrégulier, et le *bootstrap* par blocs pour le bruit corrélé — sur données synthétiques et GNSS réelles tout du long.
7. **2.7 Considérations statistiques** — moments, distributions et loi de Gutenberg-Richter, sur données géochimiques synthétiques et réelles.
8. **2.8 Transformées spectrales** — transformées de Fourier et en ondelettes de sismogrammes et de champs 2D.
9. **2.9 Filtrage** — filtres passe-bas, passe-haut et passe-bande ; filtres à phase nulle vs causaux ; séparation de la tendance, du cycle saisonnier et du bruit ; filtrage à travers les lacunes et récupération d'une erreur d'horloge sur un sismogramme réel.
10. **2.10 Données synthétiques** — construction de sismogrammes synthétiques et de bruit à spectre ajusté ; une mesure travaillée du seuil de détection STA/LTA avec barres d'erreur binomiales ; quand les données synthétiques sont recevables en science.
11. **2.11 Ingénierie des caractéristiques** — caractéristiques faites main et automatisées pour les séries temporelles, sur un jeu de référence de formes d'onde sismiques réelles.
12. **2.12 Réduction de dimension** — ACP, EOF sur champs climatiques, ICA et t-SNE.
13. **2.13 Données prêtes pour l'IA (synthèse)** — la liste de contrôle opérationnelle : fiches de données, découpages de référence, la jointure raster-station pour les covariables maillées, la fuite par prétraitement, et les découpages corrects pour des données autocorrélées.

Le chapitre se termine par le **devoir du projet final pour ce pilier (2.20)** : construire un jeu de données prêt pour l'IA pour votre propre projet, avec une fiche de données et des découpages de référence, en suivant la liste de contrôle de 2.13.

### Objectifs d'apprentissage

À la fin de ce chapitre, vous saurez :

- Reconnaître les types, modalités et formats de données courants en géosciences, y compris les formats optimisés pour le cloud.
- Manipuler des données tabulaires avec Pandas et des tableaux avec NumPy et Xarray.
- Caractériser des données par moments statistiques, distributions et méthodes de rééchantillonnage.
- Réparer les pathologies d'instruments — lacunes, repliement de spectre (*aliasing*), erreurs de datation, valeurs censurées — la réparation étant notée contre une vérité de référence connue.
- Appliquer les transformées de Fourier et en ondelettes et concevoir des filtres numériques.
- Générer des données synthétiques de manière responsable et en déclarer l'usage.
- Construire des caractéristiques et réduire la dimension pour des tâches de ML en aval.
- Assembler un jeu de données prêt pour l'IA avec provenance documentée, fiche de données et découpages de référence sans fuite.

### Devoirs

- **Devoir final (2.20)** : construisez un jeu de données prêt pour l'IA pour votre projet final. Appliquez la liste de contrôle de 2.13 : documentez la provenance et la licence, rédigez une fiche de données, définissez des découpages de référence et démontrez que votre prétraitement ne fait fuir aucune information d'un découpage à l'autre.
