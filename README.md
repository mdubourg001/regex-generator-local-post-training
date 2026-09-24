# Post-training d'un petit LLM en local : Text-to-SQL

Projet d'apprentissage : prendre un petit modèle open-weight et l'améliorer sur une tâche précise, en passant par les
trois grandes étapes du post-training moderne : **SFT → DPO → GRPO**. Tout tourne en local sur un Mac.

## 1. Objectif

**Tâche** : à partir d'une question en langage naturel et du schéma d'une base, générer la requête SQL qui y répond.

```
Schéma   : CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary INT);
Question : Quel est le salaire moyen par département ?
Réponse  : SELECT dept, AVG(salary) FROM employees GROUP BY dept;
```

**Pourquoi Text-to-SQL** :

- **Récompense vérifiable** : on exécute la requête générée sur SQLite et on compare son résultat à celui de la requête de
  référence. Pas besoin de juge humain ni de reward model.
- **Métrique claire** : l'_execution accuracy_, c'est-à-dire le pourcentage de requêtes qui renvoient le bon résultat.
  On la mesure après chaque étape pour voir ce que chacune apporte.
- **Couverture complète** : la tâche se prête à la fois à l'imitation (SFT), aux préférences (DPO) et au RL (GRPO).

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
| **Python 3.12**                                  | Langage des scripts de préparation, d'évaluation et de reward.                                                                                                                                                                                |
| **mlx**                                          | Framework de calcul d'Apple (équivalent de PyTorch), optimisé pour la mémoire unifiée des puces Apple Silicon.                                                                                                                                |
| **mlx-lm**                                       | Charge et fait générer les LLM sous MLX. Fournit l'entraînement LoRA SFT (`mlx_lm.lora`), la fusion des adapters (`mlx_lm.fuse`), la conversion et la quantization (`mlx_lm.convert`) et un serveur HTTP compatible OpenAI (`mlx_lm.server`). |
| **mlx-lm-lora**                                  | Étend mlx-lm avec les entraînements par préférences et par RL : DPO, ORPO et **GRPO**.                                                                                                                                                        |
| **datasets**                                     | Télécharge et manipule les datasets du Hugging Face Hub (filtrage, split, export JSONL).                                                                                                                                                      |
| **huggingface_hub** (CLI `hf`)                   | Télécharge les poids des modèles et gère l'authentification au Hub.                                                                                                                                                                           |
| **sqlite3**                                      | Inclus dans Python. Exécute les requêtes générées : c'est le cœur de l'évaluation et de la reward.                                                                                                                                            |
| **sqlglot**                                      | Parse et normalise le SQL. Sert à vérifier la syntaxe et à comparer des requêtes.                                                                                                                                                             |
| **pandas**                                       | Manipule les résultats et compare les sorties de requêtes.                                                                                                                                                                                    |
| **Weights & Biases / TensorBoard** _(optionnel)_ | Suivi des courbes d'entraînement (loss, reward, KL).                                                                                                                                                                                          |
| **LM Studio / Ollama** _(optionnel)_             | Servent le modèle final pour le tester en conversation.                                                                                                                                                                                       |

## 4. Datasets

| Usage                        | Dataset                                                                                                                                          | Remarque                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| SFT                          | [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)                                               | ~100k exemples : schéma `CREATE TABLE`, question, SQL et explication    |
| SFT _(alternative)_          | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)                                                           | ~78k exemples, requêtes plus simples                                    |
| Éval + RL                    | [Spider](https://yale-lily.github.io/spider) ([`xlangai/spider`](https://huggingface.co/datasets/xlangai/spider) + les bases SQLite officielles) | Benchmark de référence, avec de vraies bases pour exécuter les requêtes |
| Éval difficile _(optionnel)_ | [BIRD](https://bird-bench.github.io/)                                                                                                            | Plus réaliste et plus dur, pour voir les limites                        |
| DPO                          | **Généré par nous** à partir du modèle SFT (voir étape 4)                                                                                        | Données _on-policy_                                                     |

> ⚠️ On ne mélange jamais les données d'entraînement et d'évaluation. On fige un sous-ensemble de Spider dev (par exemple
> 500 exemples) comme jeu de test, et on n'y touche plus.

## 5. Étapes

### Étape 0 : setup

- **Faire** : initialiser le projet avec uv, installer les dépendances, télécharger les modèles et les datasets.
- **Livrable** : un environnement qui charge Qwen3-1.7B et lui fait générer du texte.

```sh
uv sync                                         # crée .venv et installe les dépendances
uv run hf download Qwen/Qwen3-1.7B              # modèle d'itération (~4 Go)
uv run hf download Qwen/Qwen3-4B-Instruct-2507  # modèle final (~8 Go)
uv run scripts/download_data.py                 # gretel + Spider (questions et bases SQLite) → data/
uv run scripts/smoke_test.py                    # modèle → SQL → exécution sur une base Spider
```

### Étape 1 : évaluation baseline

- **Faire** : écrire `scripts/eval.py`. Pour chaque exemple de test, il construit le prompt (schéma + question), fait
  générer le modèle en décodage greedy, extrait le SQL, l'exécute sur la base SQLite et compare le résultat à celui de la
  requête de référence.
- **Apprendre** : le chat template du modèle, l'importance du format de prompt, le décodage (greedy ou sampling).
- **Livrable** : le score du modèle brut, c'est-à-dire le point de référence de tout le projet.

### Étape 2 : préparation des données SFT

- **Faire** : convertir gretel au format JSONL `messages` (system / user / assistant) attendu par mlx-lm, filtrer les
  exemples trop longs et créer les splits `train.jsonl` / `valid.jsonl`.
- **Apprendre** : le formatage des conversations, et pourquoi on ne calcule la loss que sur la réponse (masquage du prompt).
- **Livrable** : `data/sft/{train,valid}.jsonl`.

### Étape 3 : SFT avec LoRA

- **Faire** : entraîner un adapter LoRA avec `mlx_lm.lora`, piloté par un fichier `configs/sft.yaml`. Jouer sur les
  hyperparamètres clés (rank LoRA, learning rate, nombre d'itérations, batch size), puis ré-évaluer.
- **Apprendre** : ce que fait LoRA (des matrices de faible rang ajoutées aux couches d'attention), comment lire les courbes
  train/valid, comment repérer l'overfitting.
- **Livrable** : `adapters/sft/` et le score SFT.

### Étape 4 : génération des paires de préférences

- **Faire** : pour chaque question d'entraînement, échantillonner N requêtes (par exemple 8, avec une température > 0)
  avec le modèle SFT, puis les exécuter. Une requête correcte devient `chosen`, une incorrecte `rejected`. On ne garde que
  les questions qui ont au moins une réponse de chaque sorte.
- **Apprendre** : le sampling, la notion de données _on-policy_, et pourquoi les questions « faciles » (toutes les réponses
  correctes) ou « impossibles » (aucune correcte) n'apprennent rien.
- **Livrable** : `data/dpo/{train,valid}.jsonl`.

### Étape 5 : DPO

- **Faire** : entraîner avec mlx-lm-lora en mode DPO, en partant du modèle SFT, puis ré-évaluer.
- **Apprendre** : le rôle du modèle de référence, le paramètre `beta`, en quoi DPO diffère du SFT (on apprend à préférer
  une réponse plutôt qu'une autre, pas à imiter).
- **Livrable** : `adapters/dpo/` et le score DPO.

### Étape 6 : GRPO (RL avec récompense vérifiable)

- **Faire** : écrire `scripts/reward.py`, une récompense de 1 si l'exécution est correcte, avec un petit bonus si le format
  est respecté et une pénalité si le SQL est invalide. Entraîner avec mlx-lm-lora en mode GRPO, puis ré-évaluer.
- **Apprendre** : le RL sur une tâche vérifiable (façon DeepSeek-R1), l'avantage relatif au groupe (on compare les réponses
  d'un même prompt entre elles), le rôle du terme KL, et le _reward hacking_, c'est-à-dire un modèle qui exploite une
  faille de la reward.
- **Livrable** : `adapters/grpo/` et le score GRPO.

### Étape 7 : passage au 4B et packaging

- **Faire** : rejouer le meilleur pipeline sur Qwen3-4B-Instruct-2507, fusionner l'adapter dans le modèle (`mlx_lm.fuse`),
  le quantizer en 4-bit, puis le tester dans LM Studio ou via `mlx_lm.server`.
- **Livrable** : un modèle Text-to-SQL autonome, utilisable en local.

### Étape 8 : bilan

- **Livrable** : `results/summary.md` avec un tableau comparatif :

| Modèle     | Baseline | SFT | DPO | GRPO |
| ---------- | -------- | --- | --- | ---- |
| Qwen3-1.7B |          |     |     |      |
| Qwen3-4B   |          |     |     |      |

Ajouter quelques exemples qualitatifs : les erreurs corrigées à chaque étape, et les nouvelles erreurs apparues.

## 6. Structure du repo

```
.
├── README.md
├── pyproject.toml
├── configs/          # YAML d'entraînement (sft.yaml, dpo.yaml, grpo.yaml)
├── data/             # datasets préparés (sft/, dpo/, spider/)
├── scripts/
│   ├── prepare_sft.py     # gretel → JSONL messages
│   ├── eval.py            # execution accuracy sur Spider dev
│   ├── generate_pairs.py  # échantillonnage + paires chosen/rejected
│   └── reward.py          # fonction de récompense GRPO
├── adapters/         # adapters LoRA entraînés (sft/, dpo/, grpo/)
└── results/          # scores et bilan
```

## 7. Ressources

- [mlx-lm : documentation LoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) : DPO, ORPO et GRPO sur MLX
- Papiers :
  - [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
  - [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
  - [DeepSeekMath](https://arxiv.org/abs/2402.03300), qui introduit GRPO (Shao et al., 2024)
  - [Spider](https://arxiv.org/abs/1809.08887) (Yu et al., 2018)
