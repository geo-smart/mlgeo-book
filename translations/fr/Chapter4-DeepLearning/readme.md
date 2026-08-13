# Aperçu du chapitre

## Chapitre 4 : Apprentissage profond

Ce chapitre enseigne l'apprentissage profond (*deep learning*) en le construisant pièce par pièce, en PyTorch, sur des données géoscientifiques. Chaque architecture est implémentée, entraînée et diagnostiquée dans un carnet exécutable sur un ordinateur portable. Les modèles sont délibérément petits ; les idées ne le sont pas.

### Plan du chapitre

1. **Le perceptron** (4.0)
   - Un neurone artificiel unique, implémenté à partir de zéro
   - La règle d'apprentissage du perceptron et ses limites
   - La descente de gradient comparée aux moindres carrés ordinaires

2. **Un premier réseau de neurones** (4.1)
   - Les cinq étapes de tout script d'entraînement : jeu de données, modèle, perte, optimiseur, boucle d'entraînement
   - Classification multi-classes de sources sismiques à partir de caractéristiques tabulaires
   - Lire les courbes d'apprentissage

3. **Perceptrons multicouches** (4.2)
   - Profondeur, *dropout* (extinction aléatoire de neurones) et normalisation par lots
   - Sauvegarder les modèles, poser des *checkpoints* (points de sauvegarde) et les restaurer
   - PyTorch comparé au MLPClassifier de scikit-learn

4. **Réseaux de neurones convolutifs** (4.3)
   - Convolution et noyaux sur des images
   - LeNet sur MNIST, brièvement
   - Un CNN 2-D qui régresse des tendances de réchauffement sur un champ climatique synthétique, lu face à des références par moindres carrés
   - Un détecteur de séismes par CNN 1-D et son plancher de détection, confronté au déclencheur classique STA/LTA sur les mêmes traces
   - Une épreuve de réalité sur de vraies formes d'onde miniPNW : l'écart synthétique-réel mesuré, pas escamoté par des réglages
   - Lire et recoder un réseau publié

5. **Modèles de séquences** (4.4)
   - Fenêtres de contexte et horizons de prévision
   - Les RNN (réseaux récurrents) simples et pourquoi les gradients s'évanouissent
   - LSTM, auto-attention à partir de zéro, et un petit encodeur *transformer*
   - Tous comparés sur la même tâche de prévision, face aux modèles de référence par persistance et naïf saisonnier

6. **Les trois piliers du développement de modèles** (4.5)
   - Pilier 1 : curation des données d'entraînement — bruit d'étiquetage, désaccord structuré des étiquettes, déséquilibre des classes, bruit de capteur, qualité hétéroscédastique
   - Pilier 2 : architecture — largeur, profondeur, modèles de référence, ensembles profonds et *MC dropout* pour l'incertitude, calibration contre discrimination, comportement hors domaine
   - Pilier 3 : stratégies d'entraînement — taux d'apprentissage, taille de lot, arrêt précoce, planificateurs (*schedulers*)
   - Diagnostiquer des entraînements défaillants à partir de leurs courbes de perte
   - Recherche d'hyperparamètres avec Optuna

7. **Auto-encodeurs et auto-supervision** (4.6)
   - Auto-encodeurs denses, convolutifs et débruiteurs sur des spectrogrammes sismiques
   - Pré-entraînement par auto-encodeur masqué, porté vers un second domaine (champs climatiques maillés)
   - Réutiliser un encodeur pré-entraîné quand les étiquettes sont rares
   - Un sondage de transfert miniPNW : les caractéristiques traversent l'écart synthétique-réel, les frontières de décision non

8. **Apprentissage informé par la physique** (4.7)
   - Les contraintes physiques comme termes de perte
   - Une ablation sur une loi de refroidissement et un PINN de diffusion thermique 1-D
   - Une référence en différences finies qui bat le PINN sur le problème direct (~750 fois plus rapide, 100 fois plus précise)
   - Un PINN inverse qui retrouve la diffusivité à partir de 40 échantillons bruités, et un exercice « cassez le PINN » sur le déséquilibre des termes de perte
   - Où en sont les PINN en 2026, et les opérateurs neuronaux comme successeurs

9. **Concours de prévision de séries temporelles** (4.10)
   - Modèles de référence, SARIMA, *gradient boosting* (renforcement par gradient), LSTM et un encodeur *transformer* sur des séries géoscientifiques réelles
   - Des découpages temporels honnêtes et le MASE
   - Le classement (*leaderboard*) de prévision de la classe

10. **Jalon du projet final** (4.20)
    - Exploration d'architectures, évaluation et exigences de diagnostic pour le jalon « apprentissage profond »

L'apprentissage par transfert apparaît là où il sert : le carnet 4.6 se clôt sur un sondage linéaire (*linear probe*) d'un encodeur pré-entraîné gelé, ce qui est de l'apprentissage par transfert en miniature. Les grands modèles de langage et les agents IA sont traités au chapitre 6.

### Résultats d'apprentissage

À la fin de ce chapitre, vous saurez :
- Implémenter, entraîner et évaluer des réseaux de neurones en PyTorch, du neurone unique à l'encodeur *transformer*.
- Diagnostiquer les problèmes d'entraînement à partir des courbes d'apprentissage et les corriger.
- Quantifier comment la qualité des données, les choix d'architecture et la stratégie d'entraînement affectent chacun la performance du modèle.
- Estimer l'incertitude des prédictions avec des ensembles profonds.
- Recourir au pré-entraînement auto-supervisé quand les données étiquetées sont rares.
- Choisir des modèles de prévision et les comparer à des références, avec des découpages temporels sans fuite de données.
