# Chapitre 6 : l'IA agentique pour la science

En 2023, les modèles de langage complétaient automatiquement du code. En 2026, les systèmes agentiques lisent des dépôts entiers, exécutent du code et en inspectent les résultats, cherchent dans la littérature et mènent des tâches de recherche en plusieurs étapes sous une supervision limitée. Entre ces deux dates, la compétence humaine utile s'est déplacée. Ce n'est plus le *prompting* — demander poliment, avec les bons mots, est un problème résolu. Les compétences qu'enseigne ce chapitre sont la **spécification** (énoncer une tâche assez précisément pour que la réussite soit vérifiable) et l'**évaluation** (la vérifier).

Rien ici ne contredit les chapitres 1 à 5 ; tout en dépend. La sortie d'un agent est une analyse non fiable, et vous savez déjà quoi faire d'une analyse non fiable : exiger un environnement reproductible (chapitre 5), une évaluation équitable contre une vérité mise de côté (chapitre 3), et un modèle de référence (*baseline*) qu'elle doit battre. Ce chapitre applique cette machinerie aux agents eux-mêmes.

Une note sur notre façon de travailler dans ce chapitre : les carnets n'appellent aucun service d'IA externe. Ils s'exécutent sur des artefacts fournis — réponses d'agents enregistrées, agents simulés dont les modes d'échec sont plantés à dessein — de sorte qu'ils s'exécutent en intégration continue et ne coûtent rien. Le raisonnement se transfère directement aux agents en ligne que vous utilisez dans votre projet, où s'applique la politique d'IA du cours : l'usage est autorisé, déclaré et vérifié, et vous devez pouvoir défendre chaque ligne que vous rendez ([chapitre 1.8](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md)).

## Contenu du chapitre

1. **[6.1 Des modèles de langage aux agents](6.1_llms_to_agents.md)** — Concepts : ce qu'est et ce que n'est pas un LLM (grand modèle de langage) fondé sur un *transformer*, pourquoi les LLM sont de mauvais calculateurs, la recherche documentaire (*retrieval*), l'usage d'outils, et l'hallucination comme mode d'échec distinct de l'erreur ordinaire.
2. **[6.2 Évaluation critique des sorties d'IA](6.2_critical_evaluation.ipynb)** — En pratique : vérifier les affirmations quantitatives d'un agent contre les données, attraper une citation fabriquée, et noter deux relectures d'IA pour voir les biais des juges à l'œuvre — puis échanger vos notes avec un binôme pour mesurer si deux évaluateurs appliquant la même grille sont seulement d'accord.
3. **[6.3 Construire un jeu d'évaluation](6.3_build_an_eval_set.ipynb)** — L'exercice noté central du chapitre : avant de confier une tâche à un agent, construisez le jeu d'évaluation qui la mesure. Traité de bout en bout sur une tâche de vitesse GNSS — cahier des charges, cas, notateur qui survit à une sortie malformée, tolérances dérivées empiriquement de réalisations synthétiques — puis étendu dans « Noter sans vérité calculable » aux tâches sans réponse exacte : la grille comme notateur, l'accord entre deux évaluateurs (pourcentage et kappa de Cohen), les contrôles du biais de juge, et les taux de réussite sur répétitions d'un agent stochastique. Une section finale facultative fait tourner la même machinerie contre un modèle à poids ouverts en ligne (OLMo 2 via Ollama) ; le chemin hors ligne, simulé ou enregistré, reste la référence notée. L'exercice se répète sur une tâche du domaine de votre propre projet.
4. **[6.4 Déclaration et normes](6.4_disclosure_and_norms.md)** — Attribution et déclaration pour la recherche assistée par IA : ce qu'attendent les revues, le format de déclaration du cours, la couche institutionnelle au-delà de la revue (politique de l'employeur ou du financeur, classification des données, conservation des transcriptions), et qui est propriétaire de l'exactitude (vous).
5. **[6.5 L'arc de lecture](6.5_reading_arc.md)** — Le contrat filé sur tout le trimestre : une revue de littérature assistée par IA avec journal de vérification des citations, l'anatomie des bons articles, une grille de qualité que vous rédigez dans le genre que vous déclarez, et un agent de relecture avant soumission construit à partir de cette grille et évalué avec la machinerie de ce chapitre.

## Objectifs d'apprentissage

À la fin de ce chapitre, vous savez :

- expliquer ce qu'est un agent (un LLM, un ensemble d'outils et une boucle sur les observations) et où, dans cette boucle, les erreurs entrent ;
- vérifier une affirmation produite par une IA contre les données sous-jacentes plutôt que contre votre impression de plausibilité ;
- concevoir et exécuter un jeu d'évaluation avec vérité terrain pour une tâche d'agent avant de déployer l'agent ;
- noter un agent dont la sortie est un jugement et non un nombre : vérifications par grille sur des défauts plantés, accord entre évaluateurs, et taux de réussite sur répétitions ;
- déclarer l'assistance par IA dans le format qu'exigent le cours, les politiques éditoriales actuelles et votre institution ;
- transformer vos propres exigences de qualité en un agent de relecture que vous avez mesuré, et défendre à la fois les exigences et la mesure.
