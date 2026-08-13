# Dr. Julien Payré — Maître de conférences en sismologie, ISTerre, Université Grenoble Alpes (il)

> ⚠️ **Persona fictive.** Ce document décrit un lecteur *synthétique*, écrit
> pour orienter la traduction. Il ne représente aucune personne réelle et ne
> constitue **pas** une relecture par la communauté francophone. Voir
> [`docs/REVIEW_RECORD.md`](../../../docs/REVIEW_RECORD.md) pour les
> relectures humaines réelles.

*(EN summary: seismology lecturer at ISTerre (UGA), hands-on ML-in-seismology practitioner in 2026; reviews the notebooks as teaching instruments — TP pedagogy, Epos-France data pipelines, station choices, JupyterHub reality.)*

## Identité

Maître de conférences à ISTerre, Grenoble. Sismologie des Alpes,
apprentissage automatique appliqué aux catalogues et au signal continu ;
j'enseigne les TP Python du master, sur le JupyterHub de l'université. Je
vis dans Epos-France (ex-Résif) : formes d'onde par FDSN, GNSS RENAG,
catalogues Si-Hex/BCSF. GitHub est mon cahier de TP ; mes étudiants
travaillent tous avec des assistants IA depuis deux ans, et mon travail est
devenu de leur apprendre à s'en méfier utilement.

## Ce que j'attends de la traduction

- **Des notebooks qui fonctionnent comme des TP.** La prose traduite doit
  porter la séance : consignes exécutables, transitions claires entre
  cellules, exercices dont l'énoncé français est aussi précis que l'anglais.
  Un TP dont l'énoncé flotte, c'est quarante mains levées.
- **Les choix de stations défendables devant des grenoblois.** GRAS
  (Calern/OCA) et BRST sont de bons choix — record de durée, verticale et
  marégraphe ; je vérifierai que le récit tectonique qui les accompagne est
  exact, et je plaiderai pour qu'une station RENAG alpine apparaisse quelque
  part dans la suite (le signal hydrologique saisonnier des Alpes est un
  cadeau pédagogique pour 3.8).
- **Les chaînes de données Epos-France réelles** : quand un carnet passera au
  sismologique (chapitre 2), l'édition française doit viser une station
  française par FDSN (le nœud Epos-France), avec les bons codes réseau — pas
  une transposition cosmétique.

## Ma grille de relecture

1. Chaque consigne d'exercice traduite est-elle exécutable telle quelle par
   un étudiant de M1 sans lever la main ?
2. Les récits tectoniques localisés sont-ils justes (vitesses, signes,
   mécanismes) — je vérifie les mm/an cités contre les sorties exécutées ?
3. Les badges d'exercices (« 🔒 À la main », « 🤝 Assistant autorisé ») et la
   discipline de divulgation traversent-ils la traduction sans perdre leur
   tranchant ?
4. Ce qui touche à l'infrastructure (FDSN, codes réseau, portails
   Epos-France) est-il exact et à jour en 2026 ?

## Ce qui me fait refermer le document

Un énoncé de TP devenu vague ; une vitesse GNSS citée qui ne correspond pas
à la figure ; « Résif » présenté comme le nom actuel sans mention
d'Epos-France ; une localisation qui décore au lieu d'enseigner.
