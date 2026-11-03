# Notes

## SFT

SFT signifie Supervised Fine-Tuning, c'est-à-dire un fine-tuning supervisé. On montre au modèle des milliers d'exemples de la forme « entrée → bonne
réponse », et on l'entraîne à reproduire ces réponses. C'est de l'apprentissage par imitation.

Comment ça marche :

1. Chaque exemple est une conversation : prompt (description en langage naturel, éventuellement des exemples) → réponse attendue (la regex de référence).
2. Le modèle prédit la réponse token par token. À chaque position, on mesure la probabilité qu'il attribuait au bon token : c'est la loss, l'entropie
   croisée.
3. On ajuste les poids pour augmenter cette probabilité. La loss n'est calculée que sur la réponse : on n'apprend pas au modèle à réécrire le prompt.

C'est exactement l'objectif du pré-entraînement (prédire le token suivant), mais sur des données choisies et formatées pour une tâche précise.

Dans ce projet :

- Données : les exemples synthétiques générés par le modèle professeur et validés par exécution, mis au format de prompt commun à toutes les étapes.
- LoRA : on ne modifie pas les centaines de millions (ou milliards) de poids. On ajoute de petites matrices entraînables (environ 1 % des paramètres), ce qui tient en
  mémoire et donne un « adapter » de quelques dizaines de Mo.
- Ce qu'on en attend : que le modèle respecte le format de sortie (`/…/flags`, sans explication) et apprenne à écrire des regex justes,
  ancrées, sans sur-généraliser.

Sa limite, et la raison des étapes suivantes : le SFT n'apprend qu'à imiter. Le modèle ne voit jamais ses propres erreurs, et il ne sait pas qu'une
réponse est « presque juste » ou « complètement fausse ». Il reproduit aussi les défauts du dataset.

- DPO : on lui montre ses propres réponses, une bonne et une mauvaise, et il apprend à préférer la bonne.
- GRPO : il essaie, on exécute sa regex sur les tests cachés, et il est récompensé en proportion des tests réussis.

En pratique, on commence presque toujours par le SFT : il met le modèle dans le bon format et le bon domaine, et donne une base solide au DPO et au
RL.

## LoRA

LoRA signifie Low-Rank Adaptation. C'est une technique pour fine-tuner un modèle sans toucher à ses poids d'origine.

L'idée : un LLM est surtout composé de grandes matrices de poids. Par exemple, une projection d'attention de Qwen3-1.7B fait environ 2048 × 2048,
soit 4 millions de paramètres. Un fine-tuning classique modifie toute la matrice : W' = W + ΔW.

L'hypothèse de LoRA : la modification ΔW nécessaire pour adapter le modèle à une tâche est « simple ». On peut donc l'approximer par le produit de
deux petites matrices :

ΔW ≈ B × A avec B : 2048 × r et A : r × 2048, r petit (8, 16, 32…)

Avec r = 16, cela fait 2 × 2048 × 16 ≈ 65k paramètres au lieu de 4 millions, soit environ 60 fois moins. r s'appelle le rank (le rang).

Pendant l'entraînement :

- W est gelé : on ne calcule pas de gradient dessus, et on ne stocke pas d'état d'optimiseur pour lui.
- Seuls A et B sont entraînés. B est initialisé à zéro, donc au départ le modèle se comporte exactement comme l'original.
- La sortie d'une couche devient W·x + B·A·x.

Pourquoi c'est ce qu'il nous faut :

┌───────────────────────────────┬───────────────────────────────────────────────┬───────────────────────────────────────┐
│ │ Fine-tuning complet │ LoRA │
├───────────────────────────────┼───────────────────────────────────────────────┼───────────────────────────────────────┤
│ Paramètres entraînés │ 100 % │ ~0,1 à 1 % │
├───────────────────────────────┼───────────────────────────────────────────────┼───────────────────────────────────────┤
│ Mémoire (1.7B) │ ~20 Go et plus (poids, gradients, états Adam) │ ~5 Go │
├───────────────────────────────┼───────────────────────────────────────────────┼───────────────────────────────────────┤
│ Mémoire (4B) │ ne tient pas en 24 Go │ ~10 Go │
├───────────────────────────────┼───────────────────────────────────────────────┼───────────────────────────────────────┤
│ Ce qu'on sauvegarde │ tout le modèle (3,8 Go) │ un adapter de quelques dizaines de Mo │
├───────────────────────────────┼───────────────────────────────────────────────┼───────────────────────────────────────┤
│ Oubli des capacités générales │ risque plus élevé │ limité, car W reste intact │
└───────────────────────────────┴───────────────────────────────────────────────┴───────────────────────────────────────┘

Concrètement dans le projet :

- mlx_lm.lora produit un dossier adapters/sft/ qui ne contient que A et B.
- Pour évaluer, on charge le modèle de base plus l'adapter : eval.py --adapter adapters/sft.
- À la fin, mlx_lm.fuse fusionne l'adapter dans les poids (W + B·A) pour obtenir un modèle autonome.
- On peut aussi garder plusieurs adapters (SFT, DPO, GRPO) sur un seul modèle de base et comparer.

Les hyperparamètres qu'on réglera :

- rank r : la capacité de l'adapter. Trop petit, il n'apprend pas assez ; trop grand, il sur-apprend et coûte plus cher.
- alpha / scale : un facteur multiplicatif sur B·A, c'est-à-dire la force de l'adaptation.
- les couches ciblées : seulement l'attention, ou aussi les couches MLP, et sur combien de blocs.
- le learning rate : plus élevé qu'en fine-tuning complet, typiquement de 1e-5 à 1e-4.

La variante QLoRA fait la même chose sur un modèle de base quantizé en 4-bit, pour économiser encore de la mémoire. MLX la supporte, mais avec 24 Go
on n'en a pas besoin pour les tailles visées ici (0.6B et 1.7B).

# L'overfitting

L'overfitting (sur-apprentissage), c'est quand un modèle apprend par cœur ses exemples d'entraînement au lieu d'apprendre la règle générale derrière. Il devient excellent sur ce qu'il a vu, et moins bon sur ce qu'il n'a jamais vu.

# Le DPO

Le SFT ne montre au modèle que des bonnes réponses : il apprend « fais comme ça ». Le DPO (Direct Preference Optimization) lui montre des paires pour
un même prompt : une réponse préférée (chosen) et une réponse rejetée (rejected). Il apprend « plutôt ça que ça ».

La loss. Pour chaque paire, on regarde de combien le modèle entraîné a augmenté la probabilité de chaque réponse par rapport au modèle de référence.
Ce modèle de référence est le SFT, gelé.

marge = β × [ (log π(chosen) − log π_ref(chosen)) − (log π(rejected) − log π_ref(rejected)) ]
loss = −log σ(marge)

- La loss baisse quand le modèle fait monter chosen plus que rejected par rapport au modèle SFT.
- β règle la laisse. Un β petit autorise le modèle à s'éloigner du SFT. Un β grand le garde proche. Sans cette ancre, le modèle pourrait dériver, par
  exemple en dégradant le format pour gagner sur la marge.
- C'est le seul moyen de faire baisser des erreurs précises. Le SFT ne pénalise jamais directement une mauvaise réponse.
- Il n'y a pas besoin de modèle de récompense ni de RL. C'est un simple entraînement supervisé sur des paires, d'où le « Direct ».

## La génération des paires

On tire les paires du modèle SFT lui-même : on dit qu'elles sont on-policy.

1. Échantillonner. Pour chaque exemple du train, le modèle SFT génère 8 réponses à température 1, pour qu'elles varient.
2. Noter par exécution. Chaque regex passe les tests cachés dans Node. Le succès complet sépare les réponses justes des fausses.
3. Choisir.
   - chosen est la réponse juste la plus fréquente : celle que le modèle sait déjà produire, qu'on consolide.
   - rejected est l'erreur la plus fréquente : celle que le modèle fait le plus volontiers, donc la plus utile à corriger.
4. Filtrer. On écarte un exemple quand ses 8 réponses sont toutes justes (35 % : rien à corriger) ou toutes fausses (7 % : rien de juste à
   renforcer).

Pourquoi on-policy plutôt que la regex de référence contre une regex fausse inventée ? On corrige ainsi les erreurs que ce modèle commet vraiment, à
des probabilités où il les commet. Une paire hors-distribution apprendrait à éviter des erreurs qu'il ne fait pas.

Exemple réel (IPv6 complète) :

- chosen : /^(?:[0-9A-Fa-f]{4}:){7}[0-9A-Fa-f]{4}$/
- rejected : /^(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}$/, trop permissive (de 1 à 4 chiffres par groupe)

Le DPO va apprendre au modèle à préférer la première. C'est exactement le genre d'erreur « presque juste » qu'on voyait sur le benchmark.
