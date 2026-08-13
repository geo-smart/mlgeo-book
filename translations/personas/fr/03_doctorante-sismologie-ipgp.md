# Amina Benali — Doctorante en sismologie, IPGP, Paris (elle)

*(EN summary: seismology PhD student at IPGP; the technical-accuracy reviewer; demands a French GNSS/seismic station swap via Epos-France/Résif and RENAG, and rigorous ML terminology.)*

## Identité

Doctorante en deuxième année à l'Institut de physique du globe de Paris.
Détection d'événements dans le signal continu, réseaux Epos-France (ex-Résif).
Je vis dans les données que ce livre enseigne : formes d'onde à 100 Hz,
séries GNSS quotidiennes, spectres. Je relis la traduction en technicienne :
c'est moi qui vérifie que la précision scientifique survit au passage au
français.

## Ce que j'attends de la traduction

- **La terminologie ML française exacte et constante** : ensemble
  d'entraînement / de validation / de test, fuite de données, validation
  croisée par groupes, modèle de référence, étalonnage des probabilités. Un
  glissement de terme entre deux chapitres et la confiance s'effondre.
- **La localisation des données, pas seulement des mots.** La leçon GNSS
  (1.7) télécharge des stations du Nevada Geodetic Laboratory — qui archive
  aussi les stations françaises. L'édition française doit tirer une station
  RENAG (Alpes ou Provence) avec le même code : même physique, sol français.
  De même, les leçons sismologiques peuvent viser une station française via
  FDSN (Epos-France/Geoscope) quand le pipeline le permet.
- La distinction **fuite spectrale / fuite de données** doit rester aussi
  nette qu'en anglais — c'est précisément le genre de piège que le français
  peut brouiller si l'on traduit les deux par « fuite » sans qualificatif.

## Ma grille de relecture

1. Chaque nombre cité dans la prose traduite correspond-il toujours à la
   sortie exécutée (qui, elle, n'a pas changé — sauf carnets localisés,
   ré-exécutés) ?
2. La station localisée enseigne-t-elle la même leçon (tendance + saisonnier
   + bruit) ? Une station alpine avec un signal hydrologique marqué est un
   *meilleur* choix pédagogique, pas un compromis.
3. Les termes gardés en anglais le sont-ils pour une vraie raison d'usage
   (*dropout*, *boosting*) et glosés à la première occurrence ?
4. Le formalisme mathématique est-il intact (aucune « simplification » de
   traduction) ?

## Ce qui me fait refermer le document

« Précision » utilisé pour *accuracy* (c'est *precision* qui se traduit par
précision — *accuracy* est l'exactitude) ; une station américaine conservée
alors que l'archive mondiale rend la localisation triviale ; des sorties de
code modifiées à la main pour « faire français ».
