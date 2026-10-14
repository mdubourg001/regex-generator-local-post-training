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
`false`. Pour commencer, on ne fait que de la **validation** : les tâches d'extraction (trouver toutes les
occurrences, groupes nommés), qui demanderaient des tests comparés au résultat de `matchAll`, viendront plus tard.

**Prompt** : un message système fixe (« écris une regex JS ES2018, réponds uniquement par un littéral `/…/flags` »),
puis la description et, si présents, les exemples. Le format est défini une seule fois dans `scripts/common.py` et
partagé par toutes les étapes.

## 5. Données

| Usage              | Source                                                                              | Remarque                                                         |
| ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Train / valid      | Synthétiques : taxonomie → professeur **DeepSeek V4.1 Flash** (OpenRouter)          | ~10 000 exemples validés après filtrage                          |
| Held-out (dev)     | Split des données synthétiques                                                      | Pour itérer vite. Même distribution que le train                 |
| Benchmark          | ~210 cas **écrits à la main par Claude**, versionnés dans `benchmark/`               | Fait foi. Figé une fois constitué, décontaminé contre le train   |
| DPO                | **Généré par nous** à partir du modèle SFT                                          | Données _on-policy_                                              |

Le benchmark n'est pas produit par le professeur : sinon, il aurait la même distribution et le même style que le
train, et favoriserait artificiellement le modèle entraîné. Il est écrit par Claude, qui ne sert à rien d'autre dans le
projet. La comparaison avec les modèles frontier n'étant pas un objectif, ce n'est pas un biais gênant.

### Garde-fous sur les tests

Une regex de référence qui passe ses propres tests ne prouve rien si les tests sont faibles (`.*` passe tous les
positifs). Chaque exemple généré doit passer, dans Node :

1. **Tests dérivés de la description**, générés dans un appel séparé, sans voir la regex (sinon ils héritent de ses bugs).
2. **Minimum de tests** : au moins 5 positifs et 5 négatifs, dont au moins 3 négatifs « presque bons » (un caractère de
   trop, mauvais ancrage, mauvaise casse, valeur hors bornes…).
3. **La regex de référence passe tous les tests**, et la regex est valide en ES2018.
4. **Discrimination** : des regex triviales ou mutées (`.*`, la référence sans ancres, quantificateurs relâchés, classes
   élargies) doivent échouer à au moins un test. Sinon, l'exemple est rejeté.
5. **Décontamination** : on retire tout exemple qui recoupe le benchmark (même regex de référence, ou description
   proche au sens de la similarité de Jaccard sur mots et bigrammes ≥ 0,5).
6. **Second avis (sur un échantillon)** : Qwen3-235B écrit sa propre regex à partir de la description. Le taux de
   désaccord mesure le bruit du dataset.

### Composition

- **Taxonomie** (`configs/taxonomy.yaml`) : ~15 à 25 catégories (identifiants et formats, dates et heures, réseau,
  nombres et monnaies, chemins, logs, code, texte, Unicode, mots de passe…). Chacune liste des intentions
  types et les fonctionnalités de regex qu'elle sollicite, pour couvrir toute la flavor.
- **Difficulté** : 30 % simple, 50 % intermédiaire, 20 % piège.
- **Registres de description** : laconique façon dev (« match french phone numbers »), phrase complète, spécification
  détaillée, besoin métier indirect. Le registre est tiré au hasard, pour que le modèle n'apprenne pas le style du professeur.
- **Exemples dans l'entrée** : ~30 % des entrées contiennent 1 à 3 exemples.
- **Plus tard** : enrichir la distribution avec des intentions tirées de regex réelles (code GitHub). On n'en réutilise
  que l'intention, pas la regex.

### Coût et reproductibilité

Toutes les réponses brutes du professeur sont stockées dans `data/raw/` (prompt, modèle, tokens, coût). La génération
est **reprenable** : une relance ne refait pas les appels déjà payés. Un plafond configurable (9 $ par défaut, le compte
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

```sh
uv run pytest                                                                       # parsing, worker Node, métriques
uv run scripts/eval.py --name reference --data tests/fixtures/cases.jsonl --reference  # les références passent leurs tests
uv run scripts/eval.py --name smoke-0.6b --data tests/fixtures/cases.jsonl             # Qwen3-0.6B brut, bout en bout
```

Le parsing est strict : la réponse doit être exactement un littéral, éventuellement entouré d'un bloc de code, de
backticks ou de guillemets. Une réponse bavarde (« Here is the regex: /a/ ») est mal formée, car elle n'est pas collable dans du code.

### Étape 2 : taxonomie

- **Faire** : écrire `configs/taxonomy.yaml` (brouillon proposé, puis relu).
- **Livrable** : la taxonomie, avec la couverture de la flavor par catégorie.

### Étape 3 : génération des données synthétiques

- **Faire** : `scripts/generate.py`, qui fait générer descriptions, regex et tests au professeur, puis
  `scripts/validate.py`, qui applique les garde-fous. Commencer par un petit lot pour mesurer le taux de rejet et le
  coût, puis générer le volume.
- **Apprendre** : la distillation par un modèle professeur, le filtrage par exécution, le bruit d'un dataset synthétique.
- **Livrable** : `data/sft/{train,valid}.jsonl`, un held-out, et les statistiques de rejet par garde-fou.

```sh
echo 'OPENROUTER_API_KEY=sk-or-…' > .env       # ignoré par git
uv run scripts/generate.py --batches 20        # pilote : 100 besoins, mesure du coût et du taux de rejet
uv run scripts/validate.py                     # garde-fous → data/prepared/, data/sft/ + statistiques
uv run scripts/generate.py --batches 4000      # volume : reprend là où le pilote s'est arrêté
```

Les modèles, le plafond de dépense et les seuils des garde-fous sont dans `configs/generation.yaml`.

Pilote sur les mêmes 20 lots (100 besoins), avec ou sans raisonnement du professeur :

| DeepSeek V4.1 Flash        | Gardés | Désaccord regex / tests | Coût pour 1 000 gardés |
| -------------------------- | ------ | ----------------------- | ---------------------- |
| Sans raisonnement          | 52 %   | 37 %                    | 0,53 $                 |
| Raisonnement « low »       | 76 %   | 8 %                     | 2,98 $                 |
| Raisonnement par défaut    | 77 %   | 2 %                     | 4,46 $                 |

- Le raisonnement (6 000 à 11 000 tokens par appel) divise les désaccords par 5 à 20, mais coûte 6 à 8 fois plus cher
  par exemple gardé. Avec un budget de 9 $, on génère **sans raisonnement** et on laisse le filtre trier.
- Le filtre ne déforme pas la distribution : on garde à peu près la même part de cas simples, intermédiaires et
  difficiles.
- Les désaccords viennent des deux côtés : regex fausse (port `65536` accepté) ou tests faux (`9999` donné comme un PIN
  valide alors que « pas tous les chiffres identiques »).
- Premiers bugs du pipeline : le professeur écrivait parfois `"x".repeat(30)` dans le JSON des tests, et une réponse
  qui boucle n'a pas de fin sans `max_tokens` (OpenRouter garde la connexion ouverte, donc il faut aussi un délai total).

**Second avis** : Qwen3-235B écrit sa propre regex pour 200 exemples gardés, avec le prompt du modèle entraîné.

```sh
uv run scripts/second_opinion.py --n 200
```

Sur les 100 premiers lots : accord sur 51 % des exemples, 34 % de réponses mal formées (sorties cassées du
fournisseur). Sur 15 désaccords inspectés, la référence du professeur est juste dans 13 cas, et 2 sont ambigus. Le
bruit résiduel des données gardées est donc faible. Sur les données finales : accord sur 66,5 % des exemples, 88 % des
tests réussis, 8 % de réponses mal formées.

**Résultat** : 4 800 lots du plan principal, puis un complément ciblé de 700 lots sur les catégories où le filtre
rejette le plus (`logs`, `exclusions`, `multiline`, `dev-tooling`, `passwords`…) :

```sh
uv run scripts/generate.py --batches 700 --prefix t --categories logs,exclusions,multiline,…   # poids = répétitions
```

| Besoins générés | Gardés         | Train / valid / held-out | Coût   | Coût pour 1 000 gardés |
| --------------- | -------------- | ------------------------ | ------ | ---------------------- |
| 27 460          | **9 819** (36 %) | 8 810 / 538 / 471      | 6,05 $ | 0,62 $                 |

- Premier motif de rejet : le désaccord entre regex et tests (45 %). Il dépend beaucoup de la catégorie : 9 % de
  besoins gardés en `logs` (lignes longues aux règles imbriquées), 61 % en `identifiers`.
- Difficulté : 31 % simple, 51 % intermédiaire, 18 % difficile. 30 % des entrées ont des exemples.
- Couverture de la flavor : les fonctionnalités de base sont partout, mais les groupes nommés (0 %), le lookbehind
  (0,9 %), les flags `m` et `s` (< 1 %) et les regex non ancrées (1,8 %) sont rares : pour « contient X », le
  professeur écrit presque toujours `^.*X.*$`. Le benchmark dira si c'est un problème.

### Étape 4 : benchmark

- **Faire** : écrire ~10 cas par catégorie dans `benchmark/<catégorie>.yaml` (description réaliste, regex de
  référence, tests avec des pièges, et le piège principal dans `pitfall`), les contrôler mécaniquement, puis les figer.
  Le benchmark sollicite volontairement des fonctionnalités rares dans le train (groupes nommés, lookbehind, flags `m`
  et `s`, recherches non ancrées).
- **Apprendre** : pourquoi le jeu d'éval ne doit pas sortir du même moule que le train.
- **Livrable** : `benchmark/*.yaml` (source de vérité, versionnée) et `data/eval/benchmark.jsonl`.

```sh
uv run scripts/benchmark.py check    # référence vs tests dans Node, discrimination des tests, proximité avec le train
uv run scripts/benchmark.py build    # cas `status: ok` → data/eval/benchmark.jsonl (avec les exemples montrés)
```

**Résultat** : 210 cas (10 par catégorie ; 60 simples, 79 intermédiaires, 71 difficiles), 0 erreur aux contrôles.
Par rapport au train, le benchmark sollicite davantage les fonctionnalités rares : lookahead 20 % des regex,
non-ancré 10 %, backreferences 9 %, lookbehind 5 %, `\p{…}` 5 %, groupes nommés 3 %.

38 cas (18 %) ont une regex de référence identique à une regex du train : des besoins standard (IPv4, semver, date
ISO…) dont l'écriture canonique est unique. Plutôt que de retirer ces besoins fréquents du benchmark, on décontamine le
train (garde-fou 5) : 489 exemples de train, 33 de valid et 29 de held-out sont retirés. Il reste **9 268 exemples**
(8 321 / 505 / 442).

### Étape 5 : baselines

- **Faire** : évaluer Qwen3-0.6B et Qwen3-1.7B bruts, en zero-shot (même prompt que l'entraînement) et en few-shot,
  sur le held-out et sur le benchmark.
- **Livrable** : le point de référence de tout le projet.

```sh
uv run scripts/eval.py --name baseline-0.6b-benchmark --data data/eval/benchmark.jsonl
uv run scripts/eval.py --name baseline-0.6b-benchmark-fewshot --data data/eval/benchmark.jsonl --few-shot
uv run scripts/eval.py --name baseline-1.7b-heldout --data data/prepared/heldout.jsonl --model Qwen/Qwen3-1.7B
```

Le few-shot ajoute au prompt trois exemples fixes (`common.FEW_SHOT`), choisis pour ne recouper ni le benchmark ni le
train.

| Modèle brut          | Held-out : succès | tests réussis | mal formées | Benchmark : succès | tests réussis | mal formées |
| -------------------- | ----------------- | ------------- | ----------- | ------------------ | ------------- | ----------- |
| Qwen3-0.6B           | 2,5 %             | 46,5 %        | 24,4 %      | 0,0 %              | 40,6 %        | 29,0 %      |
| Qwen3-0.6B few-shot  | 2,3 %             | 43,8 %        | 27,2 %      | 0,9 %              | 39,7 %        | 33,8 %      |
| Qwen3-1.7B           | 2,3 %             | 26,3 %        | 62,7 %      | 0,9 %              | 25,4 %        | 60,5 %      |
| Qwen3-1.7B few-shot  | 5,0 %             | 53,6 %        | 22,0 %      | 3,3 %              | 50,8 %        | 20,5 %      |

- Les modèles bruts ne savent pas faire la tâche : au mieux 5 % de regex entièrement justes.
- Le 1.7B en zero-shot répond mal au format dans 60 % des cas (pattern sans délimiteurs, `/…/.test(str)`, texte
  autour). Le few-shot corrige surtout le format (20 % de sorties mal formées), et double le taux de tests réussis.
- Quand le format est bon, les regex sont fausses sur les détails : flag `i` ajouté sans raison (qui accepte `UserId`
  pour du camelCase), ancres oubliées, tirets en début ou fin de nom acceptés, `\p{L}` ignoré.
- À titre de repère, Qwen3-235B (second avis, étape 3) passe tous les tests sur 66,5 % des exemples du train.

### Étape 6 : SFT avec LoRA

- **Faire** : entraîner un adapter avec `mlx_lm.lora`, piloté par `configs/sft.yaml`. Mesurer aussi l'effet du volume
  (2 000 vs 10 000 exemples).
- **Apprendre** : ce que fait LoRA, comment lire les courbes train/valid, comment repérer l'overfitting.
- **Livrable** : `adapters/sft-0.6b/` et le score SFT.

```sh
uv run mlx_lm.lora --config configs/sft.yaml      # ~1 h sur le 0.6B
uv run scripts/eval.py --name sft-0.6b-benchmark --data data/eval/benchmark.jsonl --adapter adapters/sft-0.6b
```

**Recette** : LoRA de rang 16 (scale 2, dropout 0,05) sur l'attention et le MLP de toutes les couches, soit 1,7 % des
paramètres. Loss sur la réponse seulement (`mask_prompt`), AdamW, LR 2e-4 avec warmup de 100 pas puis décroissance
cosinus, batch effectif de 8, 2 epochs.

**Incident mémoire** : `mlx_lm` autorise le GPU à verrouiller jusqu'à 19 Go sur les 24 Go du Mac. Avec des batchs de 16,
puis de 8, le pic a atteint 18,3 Go : d'abord une erreur Metal, puis un redémarrage de la machine. L'essentiel vient des
logits (batch × longueur × 151 936 tokens de vocabulaire). Correctif : micro-batchs de 4 avec accumulation de gradient
sur 2 (même batch effectif) et gradient checkpointing, soit un pic à **4,6 Go**. Avant le run complet, une sonde sur les
160 exemples les plus longs a vérifié le pire cas.

**Courbe** : val loss 2,45 → 0,43 (100 mises à jour) → 0,20 (fin de la 1re epoch) → **0,164** (fin), train loss 0,10.
Pas de sur-apprentissage : la val loss baisse jusqu'au bout, en ralentissant avec la LR.

| Qwen3-0.6B               | Held-out : succès | tests réussis | mal formées | Benchmark : succès | tests réussis | mal formées |
| ------------------------ | ----------------- | ------------- | ----------- | ------------------ | ------------- | ----------- |
| Brut                     | 2,5 %             | 46,5 %        | 24,4 %      | 0,0 %              | 40,6 %        | 29,0 %      |
| **SFT**                  | **71,7 %**        | **92,0 %**    | 1,8 %       | **31,4 %**         | **74,2 %**    | 7,1 %       |

- Sur le held-out (même distribution que le train), le 0.6B passe de 2,5 % à 72 % de regex entièrement justes, au
  niveau de Qwen3-235B brut (66,5 % sur le train, étape 3).
- Le benchmark est plus dur et d'une autre distribution : 31 % de succès. Par difficulté : 67 % des cas simples, 27 %
  des intermédiaires, 7 % des difficiles. 23 des 144 échecs sont « de peu » (≥ 85 % des tests réussis).
- Erreurs typiques sur le benchmark :
  - une contrainte sémantique ignorée : un palindrome de 5 lettres devient `^[a-z]{5}$` (les backreferences sont rares
    dans le train), une ligne dupliquée devient « deux ou trois lignes quelconques » ;
  - une couverture partielle : mois abrégés seulement, quand les noms complets étaient aussi demandés ;
  - un détail du texte : `\s` au lieu d'une espace, encodage `%XX` non vérifié ;
  - des sorties mal formées (7 %) : groupe non fermé, ou boucle répétitive jusqu'à la limite de tokens.
- `logs` (0 %) et `multiline` (20 %), les catégories les moins représentées du train, sont aussi les plus faibles.

**Effet du volume** : même recette (`configs/sft-2k.yaml`, 2 epochs, warmup à 5 % du run) sur 2 000 exemples tirés au
hasard du train, avec le même valid.

```sh
mkdir -p data/sft-2k && cp data/sft/valid.jsonl data/sft-2k/
uv run python -c "import random; r = open('data/sft/train.jsonl').readlines(); random.Random(0).shuffle(r); open('data/sft-2k/train.jsonl', 'w').writelines(r[:2000])"
uv run mlx_lm.lora --config configs/sft-2k.yaml   # ~12 min
```

| Qwen3-0.6B SFT | Val loss | Held-out : succès | tests réussis | mal formées | Benchmark : succès | tests réussis | mal formées |
| -------------- | -------- | ----------------- | ------------- | ----------- | ------------------ | ------------- | ----------- |
| 2 000 exemples | 0,272    | 56,6 %            | 86,1 %        | 3,2 %       | 20,0 %             | 68,1 %        | 7,6 %       |
| 8 302 exemples | 0,164    | 71,7 %            | 92,0 %        | 1,8 %       | 31,4 %             | 74,2 %        | 7,1 %       |

- 2 000 exemples suffisent à apprendre le format : les sorties mal formées sont au même niveau. La justesse, elle,
  progresse encore nettement avec le volume : +15 points sur le held-out, +11 sur le benchmark (simples 43 → 67 %,
  intermédiaires 16 → 27 %, difficiles 4 → 7 %).
- Sur 2 000 exemples, la val loss plafonne dès la 2ᵉ epoch (0,283 à l'iter 700, 0,272 à la fin) et l'écart avec la
  train loss (~0,11 à 0,19) se creuse : le modèle commence à mémoriser plutôt qu'à généraliser.
- La courbe n'est pas saturée à 8 300 exemples : davantage de données aiderait probablement encore.

### Étape 7 : génération des paires de préférences

- **Faire** : pour chaque exemple d'entraînement, échantillonner N réponses (par exemple 8, température > 0) avec le
  modèle SFT, les noter par exécution. La meilleure devient `chosen`, la pire `rejected`. On ne garde que les exemples
  où l'écart de score est significatif.
- **Apprendre** : le sampling, les données _on-policy_, et pourquoi les exemples où toutes les réponses se valent
  n'apprennent rien.
- **Livrable** : `data/dpo/{train,valid}.jsonl`.

```sh
uv run scripts/pairs.py sample --split train --prompts-per-batch 4   # ~1 h, ~3 à 5 Go
uv run scripts/pairs.py sample --split valid --prompts-per-batch 4
uv run scripts/pairs.py build
```

**Recette** : 8 réponses par exemple du modèle SFT (température 1, 128 tokens au plus), notées par exécution avec la
reward de `scripts/reward.py`. `chosen` est la réponse juste la plus fréquente, `rejected` l'erreur la plus fréquente
(à égalité, la moins bien notée). Un exemple dont les 8 réponses sont toutes justes ou toutes fausses ne donne pas de
paire.

**Mémoire et débit** : la génération en parallèle de 16 prompts × 8 réponses montait à 8 à 10 Go, pour ~1 prompt/s.
Avec 4 × 8, le pic tombe à 3 à 5 Go et le débit double (~1,9 prompt/s) : un batch dure jusqu'à sa réponse la plus
longue, et sur 128 séquences il y en a presque toujours une qui boucle jusqu'à la limite de tokens.

| Échantillons du SFT 0.6B | Réponses justes | Mal formées | Distinctes / exemple | Toutes justes | Paire possible | Aucune juste | Paires |
| ------------------------ | --------------- | ----------- | -------------------- | ------------- | -------------- | ------------ | ------ |
| train (8 321 exemples)   | 65,9 %          | 4,4 %       | 4,7                  | 28,3 %        | 61,5 %         | 10,3 %       | 5 109  |
| valid (505 exemples)     | 59,2 %          | 5,5 %       | 4,8                  | 27,1 %        | 54,7 %         | 18,2 %       | 276    |

- Les paires couvrent toutes les difficultés : 59 % des exemples simples, 63 % des intermédiaires, 61 % des difficiles.
- Côté `rejected` : 3 193 réponses nettement fausses, 774 presque justes (reward ≥ 0,4) et 1 148 mal formées.
- Le valid, jamais vu au SFT, a moins de réponses justes que le train (59 % contre 66 %) : un léger effet de mémorisation.

### Étape 8 : DPO

- **Faire** : entraîner avec mlx-lm-lora en mode DPO, en partant du modèle SFT, puis ré-évaluer.
- **Apprendre** : le rôle du modèle de référence, le paramètre `beta`, en quoi DPO diffère du SFT.
- **Livrable** : `adapters/dpo/` et le score DPO.

**Mise en place** :

- le modèle SFT est fusionné (`mlx_lm.fuse` → `models/sft-0.6b`). Il sert de point de départ et de modèle de
  référence gelé : sans ça, mlx-lm-lora prendrait le modèle brut comme référence. Fusionné, il obtient 32,9 % au
  benchmark contre 31,4 % avec l'adapter (arrondi des poids, dans le bruit) ;
- `scripts/train_capped.py` lance mlx-lm-lora en plafonnant la mémoire (pic ~6 à 7 Go, contre 19 Go verrouillables
  par défaut) ;
- trois pièges de mlx-lm-lora 3.1.3 :
  - une valeur du YAML n'est appliquée que si l'option de ligne de commande vaut None. Les options qui ont un défaut
    (`--gradient-accumulation-steps`, `--dpo-cpo-loss-type`, `--delta`, et pour le GRPO `--temperature`,
    `--group-size`, `--max-completion-length`) ignorent le YAML. L'accumulation de gradient était silencieusement
    à 1 : batch de 2 et schedule de LR fini au premier quart du run ;
  - chaque batch est complété à la longueur de sa plus longue séquence, et le pas compilé (`mx.compile`) l'est pour
    chaque nouvelle forme : vers l'itération 200, `[metal::malloc] Resource limit (499000) exceeded`.
    `train_capped.py` complète les batchs à un multiple de 32 tokens ;
  - les batchs sont tirés au hasard avec remise : une « epoch » couvre ~63 % des paires. L'accuracy affichée compare
    les moyennes du batch, pas les paires une à une ;
  - `--dpo-cpo-loss-type` est rangé dans `DPOTrainingArgs.loss_type`, mais `train_dpo` lit son propre paramètre
    `loss_type`, toujours à "sigmoid" : hinge, ipo et dpop sont ignorés sans rien dire. `train_capped.py` le
    transmet et affiche la loss appliquée.

**1er essai, DPO simple** (`configs/dpo.yaml` : β 0,1, LR 1e-5, batch effectif 8, 2 555 micro-batchs) : les métriques
DPO sont excellentes (val loss 0,693 → 0,283, accuracy 96 %, marge 3,1) mais la génération s'effondre.

| Qwen3-0.6B   | Held-out : succès | tests réussis | mal formées | Benchmark : succès | tests réussis | mal formées |
| ------------ | ----------------- | ------------- | ----------- | ------------------ | ------------- | ----------- |
| SFT          | 71,7 %            | 92,0 %        | 1,8 %       | 31,4 %             | 74,2 %        | 7,1 %       |
| DPO simple   | 24,7 %            | 50,2 %        | 38,9 %      | 9,0 %              | 33,6 %        | 54,8 %      |

- Les sorties deviennent des regex cassées : `/  *?\/\/?  *? TODO? or FIXME???…/`, `(?:-[a-z0-9]+)**)??`.
- La loss ne voit que l'écart entre `chosen` et `rejected`. Le modèle l'a creusé en rendant **les deux** moins
  probables (rewards `chosen` négatifs tout le run) : la masse de probabilité part vers des tokens qu'aucune paire ne
  contient (_likelihood displacement_). Nos paires y sont exposées : `chosen` et `rejected` ne diffèrent souvent que
  de quelques caractères (`{4}` contre `{1,4}`).
- Ce n'est pas du sur-entraînement : le checkpoint de l'itération 500 est déjà cassé (13,3 % de succès, 31 % de
  sorties mal formées au benchmark). Les métriques de préférence ne suffisent pas : seule l'évaluation par
  génération et exécution révèle l'échec.

**2e essai, DPOP** (`configs/dpop.yaml`, DPO-Positive, Pal et al. 2024) : une pénalité empêche la log-probabilité de
`chosen` de passer sous celle de la référence, `loss = -log σ(β × marge) + δ × max(0, log π_ref(chosen) − log π(chosen))`,
avec δ = 5 (β × λ = 0,1 × 50 de l'article, la pénalité étant hors du sigmoïde dans mlx-lm-lora) et une LR de 5e-6.
Runs courts de 500 micro-batchs, comparés au benchmark :

| Benchmark après 500 micro-batchs | Succès | Tests réussis | Mal formées |
| -------------------------------- | ------ | ------------- | ----------- |
| SFT fusionné (point de départ)   | 32,9 % | 74,9 %        | 6,7 %       |
| DPO simple, LR 1e-5              | 13,3 % | 51,7 %        | 31,0 %      |
| DPO simple, LR 5e-6              | 20,0 % | 59,2 %        | 25,2 %      |
| DPOP, δ = 5, LR 5e-6             | 31,9 % | 75,1 %        | 6,7 %       |

- Une LR plus basse ralentit l'effondrement du DPO simple sans l'empêcher. Ce run devait être le premier DPOP : la
  loss demandée n'arrivait pas jusqu'à l'entraînement (dernier piège ci-dessus).
- DPOP n'effondre plus rien, mais n'apprend rien non plus : la marge reste à ~0 (val accuracy 32 à 34 %), et
  `chosen` comme `rejected` deviennent _plus_ probables, d'autant l'un que l'autre. La pénalité remonte `chosen`, et
  `rejected`, qui en partage presque tous les tokens, monte avec. Sur le benchmark : 2 cas gagnés, 1 perdu.
- Conclusion : sur des paires qui ne diffèrent que de quelques caractères, la préférence pèse très peu dans une
  log-probabilité de séquence. Sans ancre, le DPO creuse l'écart en dégradant tout ; avec une ancre forte, la
  pénalité écrase le signal. Pistes non explorées : une ancre plus faible (δ ≈ 1), retirer les `rejected` mal formés
  (22 % des paires, souvent des boucles de 128 tokens). Le GRPO, qui apprend directement de la reward d'exécution sur
  des réponses générées en ligne, est la suite logique.

### Étape 9 : GRPO (RL avec récompense vérifiable)

- **Faire** : entraîner par renforcement sur la reward d'exécution (`scripts/reward.py`), en partant du modèle SFT,
  puis ré-évaluer. Reward :

  ```
  reward = 0,5 × taux_positifs × taux_négatifs + 0,5 × succès_complet
  reward = -1 si la sortie est mal formée ou part en timeout
  ```

  Le **produit** empêche de gagner sur un seul côté.
- **Apprendre** : l'avantage relatif au groupe, le rôle du terme KL, et le _reward hacking_.
- **Livrable** : `adapters/grpo-0.6b-500/` et le score GRPO. L'étude des variantes de reward (taux de tests seul,
  reward binaire) n'est pas faite.

```sh
uv run scripts/grpo.py --steps 900        # ~3 h sur le 0.6B, pic 4,8 Go, reprend là où il s'est arrêté
uv run scripts/eval.py --name grpo-0.6b-500-benchmark --data data/eval/benchmark.jsonl --model models/sft-0.6b --adapter adapters/grpo-0.6b-500
```

**Pourquoi un script maison** : le GRPO de mlx-lm-lora 3.1.3 n'est pas utilisable ici :

- sa loss calcule les log-probabilités des réponses **sans le prompt devant** ;
- il clippe le ratio π / π_ref contre le modèle de référence (au lieu de la politique qui a échantillonné), avec
  ε = 1e-4 par défaut : le gradient est coupé dès que le modèle s'écarte du SFT ;
- il construit le prompt sans `enable_thinking=False`, donc différent de celui du SFT ;
- il retire le token de fin des réponses : le modèle n'apprend pas à s'arrêter, alors que les boucles jusqu'à
  128 tokens sont un de nos défauts ;
- son pas compilé est retracé pour chaque forme de batch, comme au DPO.

`scripts/grpo.py` (~250 lignes) fait le strict nécessaire. À chaque pas :

1. 8 prompts du train, 8 réponses chacun, à température 1 et sans top-p : on échantillonne exactement la politique
   qu'on entraîne ;
2. chaque réponse est exécutée sur les tests cachés et notée par `reward.py` ;
3. avantage = (reward − moyenne du groupe) / écart-type du groupe. Un groupe où toutes les réponses ont la même note
   n'apprend rien : il est sauté (en moyenne 5 groupes sur 8 sont gardés) ;
4. loss = −Σ avantage × log π(token) sur les tokens des réponses (token de fin compris), divisée par le nombre de
   tokens, plus β × KL vers le SFT (β = 0,04, estimateur k3). Un seul pas de gradient par génération : le ratio
   d'importance vaut 1, le clipping de PPO n'aurait aucun effet et n'est pas implémenté.

**Recette** : LoRA neuf (rang 16, sans dropout, mêmes couches que le SFT) sur le modèle SFT fusionné, Adam, LR 1e-5
constante après 10 pas de montée. Le modèle de référence est le même modèle, LoRA désactivé : pas de deuxième copie
en mémoire. Exemples : ceux du train dont les 8 réponses de l'étape 7 n'ont pas toutes la même reward (5 921, 71 %),
les autres n'apportent aucun gradient. 900 pas × 8 prompts = 7 200 tirages, soit ~1,2 passage.

**Mémoire** (pic de 9 Go à 3,4–4,8 Go) :

- la tête de vocabulaire (151 936 sorties) n'est appliquée qu'aux positions de la réponse, pas au prompt ;
- gradient checkpointing, micro-batchs de 4 séquences ;
- génération par paquets de 4 prompts × 8 réponses (cache KV de 64 séquences : ~3,5 Go). Au-dessus du plafond
  mémoire, MLX ralentit au lieu d'échouer : la génération passait de 5 à 13 s.

**Piège** : la référence doit être calculée dans le même graphe que la politique. Hors de `value_and_grad`, MLX
prend d'autres noyaux, les log-probabilités diffèrent de ~0,002 par token en bf16, et la KL n'est pas nulle avant le
premier pas.

**Entraînement** (moyennes par tranche de 100 pas, réponses échantillonnées à température 1 sur les prompts du
train) :

| Pas     | Reward | Réponses justes | Mal formées | KL    |
| ------- | ------ | --------------- | ----------- | ----- |
| 1–100   | 0,635  | 58,9 %          | 4,5 %       | 0,017 |
| 101–200 | 0,717  | 66,1 %          | 2,5 %       | 0,026 |
| 201–300 | 0,719  | 67,0 %          | 2,9 %       | 0,030 |
| 401–500 | 0,740  | 68,6 %          | 2,2 %       | 0,033 |
| 601–700 | 0,744  | 68,9 %          | 2,2 %       | 0,033 |
| 801–900 | 0,767  | 71,2 %          | 1,8 %       | 0,045 |

**Évaluation** (décodage glouton, comme toutes les évaluations ; « gagnés / perdus » compare cas par cas au SFT,
p du test de McNemar exact) :

| Qwen3-0.6B     | Held-out : succès | tests réussis | mal formées | Benchmark : succès | tests réussis | mal formées | gagnés / perdus (p) |
| -------------- | ----------------- | ------------- | ----------- | ------------------ | ------------- | ----------- | ------------------- |
| SFT            | 71,7 %            | 92,0 %        | 1,8 %       | 31,4 %             | 74,2 %        | 7,1 %       |                     |
| GRPO, pas 100  |                   |               |             | 30,5 %             | 75,4 %        | 5,2 %       |                     |
| GRPO, pas 200  |                   |               |             | 31,4 %             | 76,4 %        | 4,8 %       |                     |
| GRPO, pas 300  | 71,0 %            | 92,7 %        | 1,6 %       | 35,2 %             | 78,5 %        | 4,8 %       | 14 / 6 (0,12)       |
| **GRPO, pas 500** | **71,5 %**     | **93,2 %**    | **0,9 %**   | **35,2 %**         | **77,4 %**    | **4,8 %**   | 14 / 6 (0,12)       |
| GRPO, pas 700  |                   |               |             | 31,9 %             | 77,0 %        | 5,7 %       | 9 / 8 (1,0)         |
| GRPO, pas 900  | 72,4 %            | 93,1 %        | 0,7 %       | 32,9 %             | 78,1 %        | 4,8 %       | 11 / 8 (0,65)       |

- **Le GRPO améliore la fiabilité, pas encore la justesse.** Les sorties mal formées baissent (7,1 % → 4,8 % au
  benchmark, 1,8 % → 0,9 % au held-out) et les tests réussis gagnent 3 à 4 points à tous les checkpoints : les
  regex fausses sont plus souvent presque justes. Le succès complet gagne au mieux 3,8 points au benchmark, soit
  8 cas sur 210 : une tendance (p = 0,12), pas une preuve. Le held-out ne bouge pas (19 gagnés, 20 perdus au pas 500).
- **Le benchmark plafonne dès le pas 300**, puis redescend, alors que la reward d'entraînement monte encore un peu :
  au-delà, le modèle se spécialise sur la distribution du train sans mieux transférer.
- **Rien ne casse**, contrairement au DPO : la KL reste sous 0,05, aucune boucle, aucune dérive du format. C'est
  le seul post-entraînement après le SFT qui ne dégrade rien.
- **Checkpoint retenu : pas 500** (`adapters/grpo-0.6b-500`), à égalité avec le pas 300 au benchmark et meilleur au
  held-out sur les sorties mal formées. Le choisir sur le benchmark le flatte un peu.
- Le 0.6B a peu de marge : le GRPO renforce ce que le modèle réussit déjà parfois, et 10 % des exemples n'ont aucune
  réponse juste sur 8. Le 1.7B, qui en réussit plus, devrait en tirer davantage.

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
- Tâches d'extraction (`matchAll`, groupes nommés), en plus de la validation.
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
│   ├── grpo.py            # entraînement GRPO (étape 9)
│   └── reward.py          # fonction de récompense GRPO
├── benchmark/        # benchmark écrit à la main, un YAML par catégorie
├── tests/            # tests du harnais (pytest) et cas écrits à la main (fixtures/)
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
