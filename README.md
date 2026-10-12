# Post-training d'un petit LLM en local : labellisation automatique des MR GitLab

Projet d'apprentissage : prendre un petit modèle open-weight et l'entraîner à labelliser les merge requests GitLab
comme le ferait l'équipe, en passant par les trois grandes étapes du post-training moderne : **SFT → DPO → GRPO**.
Tout tourne en local sur un Mac.

## 1. Objectif

**Tâche** : à partir d'une MR (titre, description, fichiers modifiés, extrait du diff) et de la liste des labels du
projet, choisir les labels qui s'appliquent. Si aucun label ne convient, ou si un label manque, en suggérer un à créer.

```
Labels   : bug, feature, refactoring, documentation, ci, dependencies, frontend, backend, …
MR       : « Fix NPE when the cart is empty » — 2 fichiers modifiés dans src/main/java/…/cart/
Réponse  : {"labels": ["bug", "backend"], "suggested": []}
```

**Pourquoi cette tâche** :

- **Un vrai cas d'usage** : les labels servent au tri, aux changelogs et aux métriques, mais sont posés de manière
  inégale. Un modèle généraliste ne connaît pas les conventions implicites d'une équipe ; un petit modèle fine-tuné
  peut les apprendre. C'est la vraie raison d'être du fine-tuning.
- **Récompense vérifiable** : choisir dans une liste fermée se note par comparaison d'ensembles (précision, rappel,
  F1), sans juge. La récompense est **graduée** entre 0 et 1, ce qui convient bien à GRPO.
- **Un vrai sujet de golden dataset** : les labels réels sont bruités (un label absent veut souvent dire « personne
  n'y a pensé », pas « faux »). Construire un jeu d'évaluation fiable est une partie centrale du projet.

## 2. Matériel et modèles

|                    |                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Machine            | Apple M4 Pro, 24 Go de RAM unifiée                                                                                     |
| Framework          | MLX, car pas de GPU NVIDIA (donc pas de CUDA)                                                                          |
| Modèle d'itération | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) : runs rapides pour expérimenter                           |
| Modèle final       | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) : on y rejoue le meilleur pipeline |
| Méthode            | **LoRA** uniquement. Un full fine-tune d'un 4B ne tient pas en 24 Go.                                                  |

Les mécanismes sont identiques entre 1.7B et 4B. On apprend sur le petit modèle, puis on confirme sur le gros.

## 3. Tooling

| Outil                                            | Rôle                                                                                                                                                                                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **uv**                                           | Gère la version de Python, l'environnement virtuel et les dépendances (`uv init`, `uv add`, `uv run`).                                                                                                                                        |
| **Python 3.12**                                  | Langage des scripts d'extraction, de préparation, d'évaluation et de reward.                                                                                                                                                                  |
| **glab**                                         | CLI GitLab. `glab api` sert à extraire les MR, leurs diffs et l'historique des labels, avec l'authentification déjà configurée.                                                                                                              |
| **mlx**                                          | Framework de calcul d'Apple (équivalent de PyTorch), optimisé pour la mémoire unifiée des puces Apple Silicon.                                                                                                                                |
| **mlx-lm**                                       | Charge et fait générer les LLM sous MLX. Fournit l'entraînement LoRA SFT (`mlx_lm.lora`), la fusion des adapters (`mlx_lm.fuse`), la conversion et la quantization (`mlx_lm.convert`) et un serveur HTTP compatible OpenAI (`mlx_lm.server`). |
| **mlx-lm-lora**                                  | Étend mlx-lm avec les entraînements par préférences et par RL : DPO, ORPO et **GRPO**.                                                                                                                                                        |
| **huggingface_hub** (CLI `hf`)                   | Télécharge les poids des modèles.                                                                                                                                                                                                             |
| **pandas**                                       | Analyse des données extraites et des résultats d'évaluation.                                                                                                                                                                                  |
| **Weights & Biases / TensorBoard** _(optionnel)_ | Suivi des courbes d'entraînement (loss, reward, KL).                                                                                                                                                                                          |
| **LM Studio / Ollama** _(optionnel)_             | Servent le modèle final pour le tester en conversation.                                                                                                                                                                                       |

## 4. Données

Les MR viennent de projets clients sur `gitlab.ekino.com`. Les titres, descriptions et labels peuvent apparaître tels
quels dans l'article, mais **jamais le nom du client ni celui du projet**. Les projets sont désignés par un alias
(« projet A », « projet B »…) dans tout ce qui sort de ce repo. Les données brutes restent dans `data/`, ignoré par git.

| Usage            | Source                                                     | Remarque                                                                      |
| ---------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Train (SFT)      | MR extraites, filtrées                                     | Labels réels, donc bruités : on le documente et on le mesure                  |
| Éval (golden)    | ~200 à 300 MR **relabellisées à la main**                  | Figé une fois constitué. Idéalement deux annotateurs pour mesurer leur accord |
| DPO              | **Généré par nous** à partir du modèle SFT (voir étape 6)  | Données _on-policy_                                                           |
| Suggestions      | Éval par **masquage** d'un label attendu (voir étape 9)    | Vérité terrain générée automatiquement                                        |

> ⚠️ On ne mélange jamais les données d'entraînement et d'évaluation. Le split se fait **par date** (ou par projet),
> jamais au hasard : une MR et son backport ne doivent pas se retrouver de part et d'autre.

## 5. Étapes

### Étape 0 : setup

- **Faire** : initialiser le projet avec uv, installer les dépendances, télécharger les modèles, s'authentifier sur GitLab.
- **Livrable** : un environnement qui charge Qwen3-1.7B et lui fait générer du texte, et un `glab` connecté.

```sh
uv sync                                         # crée .venv et installe les dépendances
uv run hf download Qwen/Qwen3-1.7B              # modèle d'itération (~4 Go)
uv run hf download Qwen/Qwen3-4B-Instruct-2507  # modèle final (~8 Go)
glab auth login --hostname gitlab.ekino.com
```

### Étape 1 : extraction des MR

- **Faire** : extraire les MR des projets retenus avec `scripts/extract_mrs.py`, puis analyser les labels avec
  `scripts/label_stats.py` pour choisir les projets et les labels exploitables.
- **Apprendre** : ce qu'on peut (ou non) déduire du contenu d'une MR, et la provenance des labels.
- **Livrable** : `data/raw/gitlab/<projet>/` et une liste de projets et de labels retenus.

```sh
uv run scripts/extract_mrs.py groupe/projet-a groupe/projet-b   # MR mergées, avec diffs, commits et historique des labels
uv run scripts/extract_mrs.py groupe/projet-a --limit 50        # test rapide
uv run scripts/label_stats.py                                   # couverture, fréquence, provenance de chaque label
```

Pour chaque MR, on garde l'objet brut de l'API, ses diffs (tronqués à 300 fichiers et 20 000 caractères par fichier),
ses commits et ses `resource_label_events`. Ces événements disent **qui** a posé chaque label et **quand** :

- un label posé par un bot (`project_…_bot_…`) vient d'une règle automatique. Soit on l'exclut, soit on le
  garde en sachant qu'on apprend une règle ;
- un label posé après le merge relève souvent du process (`workflow::`, `deployed`…) et ne se déduit pas du contenu.

### Étape 2 : construction du golden dataset

- **Faire** :
  1. **Filtrer les labels** : ne garder que ceux qui se déduisent du contenu de la MR, et qui sont assez fréquents.
  2. **Filtrer les MR** : projets disciplinés (bonne couverture de labels), MR non vides, pas de MR de bots
     (Renovate, Dependabot) en surnombre.
  3. **Choisir la représentation** : titre, description, chemins des fichiers, `diff --stat`, extrait du diff, messages
     de commits. Le format est défini une fois dans un module commun et partagé par toutes les étapes.
  4. **Splitter par date** : les MR les plus récentes servent à l'éval.
  5. **Relabelliser l'éval à la main** : 200 à 300 MR, sans voir les labels d'origine. Mesurer l'accord avec les labels
     d'origine (et entre deux annotateurs si possible) : c'est le plafond réaliste du modèle.
- **Apprendre** : le bruit d'étiquetage, l'accord inter-annotateurs, la décontamination, et pourquoi le jeu d'éval est
  le seul morceau qu'on ne délègue pas.
- **Livrable** : `data/sft/{train,valid}.jsonl` et `data/eval/golden.jsonl`.

```sh
uv run scripts/prepare_sft.py     # filtres + split → data/prepared/{train,valid,test}.jsonl et data/sft/{train,valid}.jsonl
uv run scripts/baseline_rules.py  # baseline à base de règles sur le test
```

Le référentiel `configs/labels.yaml` liste, par projet, les labels retenus et leur définition (GitLab n'en fournit
pas). Le prompt et la logique partagée (filtres, parsing, métriques) sont dans `scripts/common.py`.

Filtres appliqués :

| Filtre                                     | projet-a | projet-b | Pourquoi                                                                                   |
| ------------------------------------------ | -------- | -------- | ------------------------------------------------------------------------------------------ |
| MR non mergée                              | 271      | 735      | Souvent abandonnée ou remplacée                                                            |
| Avant le début de la labellisation         | 3 671    | –        | Avant 2023, moins de 12 % des MR de projet-a sont labellisées                              |
| MR automatique (dependabot, renovate)      | 57       | 325      | Labellisée par une règle du bot, pas par l'équipe. Reconnue à sa branche, car l'ancien compte du bot est devenu `ghost` |
| Aucun label du référentiel                 | 347      | 210      | Sans label, on ne distingue pas « aucun ne s'applique » de « personne n'a labellisé »     |
| **Gardées**                                | **1 525** | **1 658** | Split temporel 80 / 5 / 15 par projet → 2 546 train, 159 valid, 478 test               |

Les prompts font ~1 200 tokens en médiane, ~2 200 au p95 et ~2 900 au max (tokenizer Qwen3).

Premier exemple de bruit d'étiquetage : une mise à jour de `drupal/token` (un seul fichier, `composer.lock`) est
labellisée `chore` + `dependencies`, sans `php`, alors que d'autres MR identiques l'ont.

### Étape 3 : évaluation baseline

- **Faire** : écrire `scripts/eval.py`. Pour chaque MR du golden set, il construit le prompt, fait générer le modèle en
  décodage greedy, parse la réponse et la compare aux labels attendus.
- **Métriques** :
  - **F1 micro** (global) et **F1 macro** (moyenne par label, pour ne pas laisser les labels fréquents tout masquer) ;
  - **exact match** : part des MR dont l'ensemble de labels est exactement le bon ;
  - **labels invalides** : part des labels hors liste, et réponses impossibles à parser.
- **Comparaison** : faire passer le même golden set à un modèle frontier (même prompt). C'est le point de comparaison
  de l'article.
- **Baseline à base de règles** : préfixe conventional commits du titre, branches, fichiers touchés
  (`scripts/baseline_rules.py`). Sur le test (labels d'origine) : **F1 micro 0,66, F1 macro 0,50, exact match 0,38**.
  Elle est très bonne sur ce qui se lit dans le titre (`refactor` 0,90, `fix` 0,90, `release` 0,92) et nulle sur ce
  qui demande de comprendre la MR : `bugfix`, `technical task`, `security`, `typeScript`. C'est là que le modèle
  doit faire la différence.
- **Livrable** : le score du modèle brut, point de référence de tout le projet.

```sh
uv run scripts/eval.py --name baseline-1.7b                                      # 477 MR, ~5 min
uv run scripts/eval.py --name baseline-4b --model Qwen/Qwen3-4B-Instruct-2507    # ~12 min
uv run scripts/eval.py --name debug --n 16                                       # test rapide
```

Résultats sur le test (labels d'origine, 477 MR) :

| Run              | F1 micro  | F1 macro  | Exact match | Labels / MR (attendu : 1,56) | Hors liste |
| ---------------- | --------- | --------- | ----------- | ---------------------------- | ---------- |
| Règles           | **0,664** | **0,522** | **0,388**   | 1,48                         | 0 %        |
| Qwen3-1.7B brut  | 0,503     | 0,498     | 0,117       | 2,49                         | 0,3 %      |
| Qwen3-4B brut    | 0,622     | 0,526     | 0,216       | 1,74                         | 0,8 %      |

- Aucun modèle brut ne bat les règles en F1 micro. Le 1.7B met trop de labels (rappel élevé, précision faible).
- Le 4B applique ses propres conventions, pas celles de l'équipe : sur projet-a, il ne prédit jamais `featurefix`
  (F1 0,00) ni `technical task` (0,03) et range tous les correctifs en `bugfix`, malgré les définitions du prompt.
  C'est précisément ce que le fine-tuning doit corriger.
- Il est en revanche bien meilleur que les règles là où il faut lire le contenu : `security` 0,83 (règles : 0).
- Anecdote de format : avec un référentiel écrit `- feature: Nouvelle fonctionnalité.`, le 4B recopiait
  « feature: Nouvelle fonctionnalité. » comme nom de label (8,3 % de labels hors liste, F1 micro 0,566). Mettre les
  noms entre guillemets, comme dans le JSON attendu, a suffi (0,8 %, F1 micro 0,622).

### Étape 4 : SFT avec LoRA

- **Faire** : entraîner un adapter LoRA avec `mlx_lm.lora`, piloté par `configs/sft.yaml`. Ajuster `max_seq_length` à la
  représentation choisie à l'étape 2 : c'est elle qui dicte la mémoire nécessaire.
- **Apprendre** : ce que fait LoRA, comment lire les courbes train/valid, comment repérer l'overfitting.
- **Livrable** : `adapters/sft/` et le score SFT.

### Étape 5 : génération des paires de préférences

- **Faire** : pour chaque MR d'entraînement, échantillonner N réponses (par exemple 8, avec une température > 0) avec le
  modèle SFT, puis les noter (F1). La meilleure devient `chosen`, la pire `rejected`. On ne garde que les MR où l'écart
  de score est significatif.
- **Apprendre** : le sampling, la notion de données _on-policy_, et pourquoi les MR où toutes les réponses se valent
  n'apprennent rien.
- **Livrable** : `data/dpo/{train,valid}.jsonl`.

### Étape 6 : DPO

- **Faire** : entraîner avec mlx-lm-lora en mode DPO, en partant du modèle SFT, puis ré-évaluer.
- **Apprendre** : le rôle du modèle de référence, le paramètre `beta`, en quoi DPO diffère du SFT.
- **Livrable** : `adapters/dpo/` et le score DPO.

### Étape 7 : GRPO (RL avec récompense vérifiable)

- **Faire** : écrire `scripts/reward.py`, une récompense égale au F1 avec les labels attendus, une pénalité pour chaque
  label hors liste et pour une réponse mal formée. Entraîner avec mlx-lm-lora en mode GRPO, puis ré-évaluer.
- **Apprendre** : l'avantage relatif au groupe, le rôle du terme KL, et le _reward hacking_. Exemples à montrer : une
  reward fondée sur le rappel seul pousse le modèle à mettre tous les labels ; une pénalité trop forte le pousse à
  n'en mettre aucun.
- **Livrable** : `adapters/grpo/` et le score GRPO.

### Étape 8 : suggestions de labels à créer

- **Faire** : rendre la suggestion mesurable par **masquage**. À l'éval, on retire de la liste un label attendu : le
  modèle doit alors suggérer un label équivalent (`perf` quand `performance` a été retiré). On note la suggestion par
  similarité d'embeddings, ou par un juge LLM sur ce seul sous-ensemble. Le même masquage peut servir à générer des
  exemples d'entraînement.
- **Livrable** : un score de suggestion, et des exemples qualitatifs.

### Étape 9 : passage au 4B et packaging

- **Faire** : rejouer le meilleur pipeline sur Qwen3-4B-Instruct-2507, fusionner l'adapter dans le modèle (`mlx_lm.fuse`),
  le quantizer en 4-bit, puis le brancher sur GitLab : un script qui labellise une MR via `mlx_lm.server` et `glab`.
- **Livrable** : un modèle autonome, utilisable en local sur de vraies MR.

### Étape 10 : bilan

- **Livrable** : `results/summary.md` avec un tableau comparatif (F1 micro sur le golden set) :

| Modèle         | Baseline | SFT | DPO | GRPO |
| -------------- | -------- | --- | --- | ---- |
| Qwen3-1.7B     |          |     |     |      |
| Qwen3-4B       |          |     |     |      |
| Modèle frontier |          |  –  |  –  |  –   |

Ajouter quelques exemples qualitatifs : les erreurs corrigées à chaque étape, et les nouvelles erreurs apparues.

## 6. Structure du repo

```
.
├── README.md
├── pyproject.toml
├── configs/          # YAML d'entraînement (sft.yaml, dpo.yaml, grpo.yaml)
├── data/             # données extraites et préparées (raw/gitlab/, sft/, eval/, dpo/), ignorées par git
├── scripts/
│   ├── extract_mrs.py     # GitLab (glab) → data/raw/gitlab/
│   ├── label_stats.py     # couverture, fréquence et provenance des labels
│   ├── common.py          # filtres, prompt, parsing, métriques (partagé par toutes les étapes)
│   ├── prepare_sft.py     # MR → JSONL messages + split temporel
│   ├── baseline_rules.py  # baseline sans modèle (conventional commits, branches, fichiers)
│   ├── eval.py            # F1 sur le golden set
│   ├── generate_pairs.py  # échantillonnage + paires chosen/rejected
│   └── reward.py          # fonction de récompense GRPO
├── adapters/         # adapters LoRA entraînés (sft/, dpo/, grpo/)
└── results/          # scores et bilan
```

## 7. Ressources

- [glab api](https://gitlab.com/gitlab-org/cli/-/blob/main/docs/source/api/index.md) et l'API REST GitLab :
  [merge requests](https://docs.gitlab.com/api/merge_requests/), [labels](https://docs.gitlab.com/api/labels/),
  [resource label events](https://docs.gitlab.com/api/resource_label_events/)
- [mlx-lm : documentation LoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) : DPO, ORPO et GRPO sur MLX
- Papiers :
  - [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
  - [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
  - [DeepSeekMath](https://arxiv.org/abs/2402.03300), qui introduit GRPO (Shao et al., 2024)
