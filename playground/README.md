# Regex playground

Teste les modèles SFT dans le navigateur : description en anglais → regex JS, puis test ligne à ligne façon regex101.
Inférence locale avec [transformers.js](https://huggingface.co/docs/transformers.js/index) sur WebGPU, en fp16,
génération greedy (comme `scripts/eval.py`). Chromium uniquement : Firefox et Safari ne sont pas pris en charge.

## Préparer les modèles

Les poids ONNX (plusieurs Go) ne sont pas versionnés. On les génère dans `public/models/` :

```sh
# depuis la racine du dépôt : fusion des adapters
uv run mlx_lm.fuse --model Qwen/Qwen3-0.6B --adapter-path adapters/sft-0.6b-plus --save-path models/sft-0.6b-plus
uv run mlx_lm.fuse --model Qwen/Qwen3-1.7B --adapter-path adapters/sft-1.7b --save-path models/sft-1.7b

# depuis playground/ : export ONNX fp16 (optimum), environnement isolé via les métadonnées PEP 723 du script
uv run scripts/export_models.py sft-0.6b-plus sft-1.7b
```

| Modèle        | Taille fp16 | Pic mémoire de l'export |
| ------------- | ----------- | ----------------------- |
| sft-0.6b-plus | 1,6 Go      | 6,3 Go                  |
| sft-1.7b      | ~4,7 Go     | à mesurer               |

Les tailles dépassent celles des safetensors : l'export ONNX dédouble la matrice d'embedding partagée avec `lm_head`.

Le 0.6B retenu est le SFT enrichi (étape 9 ter), le meilleur 0.6B au benchmark indépendant écrit par GPT (20,2 %
contre 17,4 % pour le SFT simple, non significatif : p = 0,065). `sft-0.6b` (SFT simple) s'exporte toujours.

Contrôle : sur 15 cas du benchmark, l'ONNX fp16 (ORT CPU) reproduit à l'identique 14 sorties MLX pour le SFT simple
et 13 pour le SFT enrichi ; les écarts viennent de l'arrondi fp16 / bf16.

## Lancer

```sh
pnpm install
pnpm dev
```

## Fonctionnement

- `src/prompt.ts` : port de `scripts/common.py` (prompt système, gabarit de chat Qwen3 sans thinking, parsing du
  littéral). Le prompt doit rester identique à celui de l'entraînement.
- `src/worker.ts` : chargement (à la demande, bouton « Load model ») et génération dans un Web Worker ; une nouvelle
  requête interrompt la génération en cours, et la regex s'affiche token par token. Le runtime onnxruntime-web est
  servi en local, pas depuis jsDelivr.
- Cache : les fichiers du modèle sont gardés dans le Cache API du navigateur (par origine), sous une clé qui inclut
  `playground_export_id` (écrit dans `config.json` par l'export). Un nouvel export invalide donc l'ancien cache, qui
  est supprimé. Rechargement du 0.6B : ~26 s sans cache, ~2,7 s avec (dont ~2 s d'initialisation WebGPU).
- `src/TestArea.tsx` : chaque ligne est testée avec `RegExp.prototype.test` (✓ / ✗), et les correspondances sont
  surlignées avec une copie de la regex en flag `g`.
