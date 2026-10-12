# Post-training d'un petit LLM en local : générer des regex JavaScript à partir du langage naturel

Projet d'apprentissage : prendre un petit modèle open-weight et l'entraîner à écrire des expressions régulières
JavaScript à partir d'une description en anglais, en passant par les trois grandes étapes du post-training moderne :
**SFT → DPO → GRPO**. Tout l'entraînement tourne en local sur un Mac.

Ce repo ne couvre que le post-training (données, entraînement, évaluation). La comparaison avec les modèles frontier,
l'export vers le navigateur, la démo et la publication sur Hugging Face viendront plus tard.

## 1. Objectif

**Tâche** : à partir d'une description en anglais, et éventuellement de quelques exemples de chaînes qui doivent (ou ne
doivent pas) matcher, produire une regex JavaScript sous forme de littéral `/…/flags`, sans explication.

```
Description : match a French mobile phone number, with or without spaces between pairs of digits
Exemples    : should match "06 12 34 56 78", should not match "01 12 34 56 78"
Réponse     : /^0[67](?: ?\d{2}){4}$/
```

**Pourquoi cette tâche** :

- **Un vrai besoin de dev**, que tout le monde comprend, et un cas où un petit modèle spécialisé a une chance d'être
  compétitif : la tâche est étroite, la sortie est courte.
- **Récompense vérifiable par exécution** : la regex produite est exécutée par un vrai moteur JS sur des tests cachés.
  Pas de juge, une récompense **graduée** entre 0 et 1, idéale pour GRPO.
- **Des données publiables** : tout est synthétique ou public, contrairement au premier cas étudié (labellisation de
  merge requests sur des projets client, abandonné).

**Thèse** : un modèle de 0,6 à 1,7 milliard de paramètres, post-entraîné en local, approche la qualité d'un gros modèle
pour un coût, une latence et une dépendance au cloud quasi nuls.

### Flavor ciblée

ECMAScript **2018**, exécutée par **Node 22** (version figée dans `.node-version`) : classes de caractères, quantificateurs,
groupes capturants et nommés, backreferences, lookahead **et lookbehind**, `\p{…}` (avec `u`), flags `g i m s u y`.
Hors périmètre : le flag `v` (ES2024), le flag `d` et les modifiers inline (ES2025).

## 2. Matériel et modèles

|                    |                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Machine            | Apple M4 Pro, 24 Go de RAM unifiée                                                                 |
| Framework          | MLX, car pas de GPU NVIDIA (donc pas de CUDA)                                                      |
| Modèle d'itération | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) : runs rapides pour expérimenter       |
| Modèle final       | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) : on y rejoue le meilleur pipeline    |
| Méthode            | **LoRA** uniquement                                                                                |
| Raisonnement       | Mode _thinking_ de Qwen3 **désactivé** (`enable_thinking=False`) : la latence prime                |

Les petites tailles sont un choix : à terme, le modèle doit pouvoir tourner dans un navigateur.

## 3. Tooling

| Outil                            | Rôle                                                                                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **uv**                           | Gère la version de Python, l'environnement virtuel et les dépendances (`uv sync`, `uv add`, `uv run`).                                               |
| **Python 3.12**                  | Langage des scripts de génération, de préparation, d'évaluation et de reward.                                                                       |
| **Node 22** (via fnm)            | Exécute les regex : c'est la seule façon d'avoir la vraie sémantique JS (le module `re` de Python n'a pas la même flavor).                           |
| **OpenRouter**                   | API des modèles open-weight qui génèrent les données (clé dans `OPENROUTER_API_KEY`).                                                               |
| **mlx / mlx-lm**                 | Chargement, génération, entraînement LoRA SFT (`mlx_lm.lora`), fusion des adapters (`mlx_lm.fuse`).                                                  |
| **mlx-lm-lora**                  | Étend mlx-lm avec les entraînements par préférences et par RL : DPO et **GRPO**.                                                                     |
| **huggingface_hub** (CLI `hf`)   | Télécharge les poids des modèles.                                                                                                                    |
| **pandas**                       | Analyse des données générées et des résultats d'évaluation.                                                                                          |

## 4. Format d'un exemple

Chaque exemple est une ligne JSONL. Les tests sont la vérité terrain : ils ne sont jamais montrés au modèle, sauf la
part tirée dans `examples` (et alors retirée des tests cachés).

```json
{
  "id": "dates-0042",
  "category": "dates-times",
  "difficulty": "medium",
  "register": "terse",
  "description": "ISO 8601 date, year between 1900 and 2099",
  "examples": { "match": ["2024-02-29"], "no_match": [] },
  "regex": "/^(?:19|20)\\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\\d|3[01])$/",
  "tests": {
    "match": ["1900-01-01", "2099-12-31", "2024-06-15", "1987-10-03", "2000-01-31"],
    "no_match": ["1899-12-31", "2100-01-01", "2024-13-01", "2024-6-15", "x2024-06-15", "2024-06-15 "]
  }
}
```

**Sémantique des tests** : une chaîne de `match` doit donner `regex.test(s) === true`, une chaîne de `no_match`
`false`. Les tâches d'extraction (trouver toutes les occurrences, groupes nommés) ajoutent des cas
`{"input": …, "matches": [...]}` comparés au résultat de `matchAll`.

**Prompt** : un message système fixe (« écris une regex JS ES2018, réponds uniquement par un littéral `/…/flags` »),
puis la description et, si présents, les exemples. Le format est défini une seule fois dans `scripts/common.py` et
partagé par toutes les étapes.

## 5. Données

| Usage              | Source                                                                              | Remarque                                                         |
| ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Train / valid      | Synthétiques : taxonomie → professeur **DeepSeek V3.x** (OpenRouter)                | ~10 000 exemples validés après filtrage                          |
| Held-out (dev)     | Split des données synthétiques                                                      | Pour itérer vite. Même distribution que le train                 |
| Benchmark          | ~150 à 300 cas, brouillon par **Qwen3-235B**, **relus et enrichis à la main**       | Fait foi. Figé une fois constitué, décontaminé contre le train   |
| DPO                | **Généré par nous** à partir du modèle SFT                                          | Données _on-policy_                                              |

Le benchmark est rédigé par un autre modèle que le professeur : sinon, il aurait la même distribution et le même style
que le train, et favoriserait artificiellement le modèle entraîné.

### Garde-fous sur les tests

Une regex de référence qui passe ses propres tests ne prouve rien si les tests sont faibles (`.*` passe tous les
positifs). Chaque exemple généré doit passer, dans Node :

1. **Tests dérivés de la description**, générés dans un appel séparé, sans voir la regex (sinon ils héritent de ses bugs).
2. **Minimum de tests** : au moins 5 positifs et 5 négatifs, dont au moins 3 négatifs « presque bons » (un caractère de
   trop, mauvais ancrage, mauvaise casse, valeur hors bornes…).
3. **La regex de référence passe tous les tests**, et la regex est valide en ES2018.
4. **Discrimination** : des regex triviales ou mutées (`.*`, la référence sans ancres, quantificateurs relâchés, classes
   élargies) doivent échouer à au moins un test. Sinon, l'exemple est rejeté.
5. **Second avis (sur un échantillon)** : Qwen3-235B écrit sa propre regex à partir de la description. Le taux de
   désaccord mesure le bruit du dataset.

### Composition

- **Taxonomie** (`configs/taxonomy.yaml`) : ~15 à 25 catégories (identifiants et formats, dates et heures, réseau,
  nombres et monnaies, chemins, logs, code, texte, Unicode, extraction, mots de passe…). Chacune liste des intentions
  types et les fonctionnalités de regex qu'elle sollicite, pour couvrir toute la flavor.
- **Difficulté** : 30 % simple, 50 % intermédiaire, 20 % piège.
- **Registres de description** : laconique façon dev (« match french phone numbers »), phrase complète, spécification
  détaillée, besoin métier indirect. Le registre est tiré au hasard, pour que le modèle n'apprenne pas le style du professeur.
- **Exemples dans l'entrée** : ~30 % des entrées contiennent 1 à 3 exemples.
- **Plus tard** : enrichir la distribution avec des intentions tirées de regex réelles (code GitHub). On n'en réutilise
  que l'intention, pas la regex.

### Coût et reproductibilité

Toutes les réponses brutes du professeur sont stockées dans `data/raw/` (prompt, modèle, tokens, coût). La génération
est **reprenable** : une relance ne refait pas les appels déjà payés. Un plafond configurable (8 $ par défaut, le compte
OpenRouter est plafonné à 10 $) arrête le script. Le coût réel est loggé.

> ⚠️ On ne mélange jamais train et évaluation. Le benchmark est décontaminé contre le train par similarité des
> descriptions.

## 6. Évaluation

Pour chaque cas : prompt → génération greedy → parsing du littéral `/…/flags` → exécution dans Node sur les tests cachés.

- **Succès complet** (métrique principale) : part des cas où **tous** les tests passent. En usage réel, une regex juste
  à 90 % est fausse.
- **Taux moyen de tests réussis** : pour diagnostiquer.
- **Sorties mal formées** : pas de littéral, ou `new RegExp(pattern, flags)` qui lève une erreur.
- **Timeouts** : chaque regex est exécutée avec une limite de temps, pour attraper le ReDoS (backtracking catastrophique).

Exécution : un **worker Node persistant** reçoit les regex par lots depuis Python. Un `RegExp` JS est synchrone et ne
s'interrompt pas, donc chaque évaluation tourne dans un `worker_thread` qu'on peut tuer au bout du délai.

## 7. Étapes

### Étape 0 : setup

- **Faire** : installer les dépendances, télécharger les modèles, vérifier Node et la clé OpenRouter.
- **Livrable** : un environnement qui charge Qwen3-0.6B et lui fait générer du texte.

```sh
uv sync                                 # crée .venv et installe les dépendances
uv run hf download Qwen/Qwen3-0.6B      # modèle d'itération
uv run hf download Qwen/Qwen3-1.7B      # modèle final
node --version                          # doit correspondre à .node-version
export OPENROUTER_API_KEY=…
```

### Étape 1 : harnais d'exécution et évaluation

- **Faire** : le worker Node (`scripts/regex_runner.mjs`), le format commun (`scripts/common.py` : prompt, parsing du
  littéral, métriques) et `scripts/eval.py`. On les valide sur une poignée de cas écrits à la main, dont des cas
  piégeux : littéral invalide, flag inconnu, regex ReDoS, sortie bavarde autour du littéral.
- **Apprendre** : ce qu'on mesure exactement, et pourquoi on exécute en JS plutôt qu'en Python.
- **Livrable** : une éval fiable, testée, qui tourne sur n'importe quel JSONL au format de la section 4.

### Étape 2 : taxonomie

- **Faire** : écrire `configs/taxonomy.yaml` (brouillon proposé, puis relu).
- **Livrable** : la taxonomie, avec la couverture de la flavor par catégorie.

### Étape 3 : génération des données synthétiques

- **Faire** : `scripts/generate.py`, qui fait générer descriptions, regex et tests au professeur, puis
  `scripts/validate.py`, qui applique les garde-fous. Commencer par un petit lot pour mesurer le taux de rejet et le
  coût, puis générer le volume.
- **Apprendre** : la distillation par un modèle professeur, le filtrage par exécution, le bruit d'un dataset synthétique.
- **Livrable** : `data/sft/{train,valid}.jsonl`, un held-out, et les statistiques de rejet par garde-fou.

### Étape 4 : benchmark

- **Faire** : faire rédiger un brouillon par Qwen3-235B, le relire et le corriger à la main, ajouter des pièges, puis le
  décontaminer contre le train et le figer.
- **Apprendre** : pourquoi le jeu d'éval est le seul morceau qu'on ne délègue pas.
- **Livrable** : `data/eval/benchmark.jsonl`.

### Étape 5 : baselines

- **Faire** : évaluer Qwen3-0.6B et Qwen3-1.7B bruts, en zero-shot (même prompt que l'entraînement) et en few-shot,
  sur le held-out et sur le benchmark.
- **Livrable** : le point de référence de tout le projet.

### Étape 6 : SFT avec LoRA

- **Faire** : entraîner un adapter avec `mlx_lm.lora`, piloté par `configs/sft.yaml`. Mesurer aussi l'effet du volume
  (2 000 vs 10 000 exemples).
- **Apprendre** : ce que fait LoRA, comment lire les courbes train/valid, comment repérer l'overfitting.
- **Livrable** : `adapters/sft/` et le score SFT.

### Étape 7 : génération des paires de préférences

- **Faire** : pour chaque exemple d'entraînement, échantillonner N réponses (par exemple 8, température > 0) avec le
  modèle SFT, les noter par exécution. La meilleure devient `chosen`, la pire `rejected`. On ne garde que les exemples
  où l'écart de score est significatif.
- **Apprendre** : le sampling, les données _on-policy_, et pourquoi les exemples où toutes les réponses se valent
  n'apprennent rien.
- **Livrable** : `data/dpo/{train,valid}.jsonl`.

### Étape 8 : DPO

- **Faire** : entraîner avec mlx-lm-lora en mode DPO, en partant du modèle SFT, puis ré-évaluer.
- **Apprendre** : le rôle du modèle de référence, le paramètre `beta`, en quoi DPO diffère du SFT.
- **Livrable** : `adapters/dpo/` et le score DPO.

### Étape 9 : GRPO (RL avec récompense vérifiable)

- **Faire** : écrire `scripts/reward.py`, puis entraîner avec mlx-lm-lora en mode GRPO et ré-évaluer. Reward :

  ```
  reward = 0,5 × taux_positifs × taux_négatifs + 0,5 × succès_complet
  reward = -1 si la sortie est mal formée ou part en timeout
  ```

  Le **produit** empêche de gagner sur un seul côté. Une petite pénalité de longueur pourra s'ajouter.
- **Apprendre** : l'avantage relatif au groupe, le rôle du terme KL, et le _reward hacking_. On essaie aussi deux
  variantes, pour montrer ce qui casse : le taux de tests réussis seul (qui pousse vers des regex trop permissives) et
  la reward binaire (trop clairsemée sur les cas difficiles).
- **Livrable** : `adapters/grpo/`, le score GRPO et l'étude des variantes de reward.

### Étape 10 : passage au 1.7B et bilan

- **Faire** : rejouer le meilleur pipeline sur Qwen3-1.7B.
- **Livrable** : `results/summary.md`, avec un tableau comparatif (succès complet sur le benchmark) et des exemples
  qualitatifs (les erreurs corrigées à chaque étape, et les nouvelles erreurs apparues).

| Modèle      | Brut (zero-shot) | Brut (few-shot) | SFT | DPO | GRPO |
| ----------- | ---------------- | --------------- | --- | --- | ---- |
| Qwen3-0.6B  |                  |                 |     |     |      |
| Qwen3-1.7B  |                  |                 |     |     |      |

### Plus tard (hors de ce repo pour l'instant)

- Comparaison avec des modèles frontier (qualité, latence, coût par requête).
- Fusion de l'adapter, export (ONNX ou MLC) et démo dans le navigateur (WebGPU).
- Publication du modèle, du dataset et du benchmark sur Hugging Face.

## 8. Structure du repo

```
.
├── README.md
├── NOTES.md          # notes de cours (SFT, LoRA, overfitting…)
├── pyproject.toml
├── .node-version
├── configs/          # taxonomy.yaml, sft.yaml, dpo.yaml, grpo.yaml
├── data/             # raw/, sft/, eval/, dpo/ (ignoré par git)
├── scripts/
│   ├── common.py          # prompt, parsing du littéral, métriques (partagé par toutes les étapes)
│   ├── regex_runner.mjs   # worker Node : exécute les regex sur les tests, avec timeout
│   ├── eval.py            # succès complet / taux de tests réussis sur un JSONL
│   ├── generate.py        # taxonomie → professeur (OpenRouter) → data/raw/
│   ├── validate.py        # garde-fous → data/sft/
│   ├── generate_pairs.py  # échantillonnage + paires chosen/rejected
│   └── reward.py          # fonction de récompense GRPO
├── adapters/         # adapters LoRA entraînés (sft/, dpo/, grpo/)
└── results/          # scores et bilan
```

## 9. Ressources

- [Expressions régulières JavaScript (MDN)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions)
  et la [spécification ECMAScript 2018](https://262.ecma-international.org/9.0/#sec-regexp-regular-expression-objects)
- [OpenRouter : documentation de l'API](https://openrouter.ai/docs)
- [mlx-lm : documentation LoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) : DPO, ORPO et GRPO sur MLX
- Papiers :
  - [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
  - [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
  - [DeepSeekMath](https://arxiv.org/abs/2402.03300), qui introduit GRPO (Shao et al., 2024)
