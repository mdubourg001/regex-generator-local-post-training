# Regex playground

Teste les modèles SFT dans le navigateur : description en anglais → regex JS, puis test ligne à ligne façon regex101.
Inférence locale avec [transformers.js](https://huggingface.co/docs/transformers.js/index), génération greedy
(comme `scripts/eval.py`). Chromium uniquement : Firefox et Safari ne sont pas pris en charge (non testés).

Modèle : SFT de Qwen3.5-0.8B (`configs/sft-qwen3.5-0.8b.yaml`), quantizé en 4 bits. Variante choisie au démarrage
(`src/variant.ts`), affichée dans l'en-tête (« Ready · WebGPU · q4f16 ») et sous la zone de test :

| Navigateur                                    | Variante | Backend    | Téléchargement |
| --------------------------------------------- | -------- | ---------- | -------------- |
| WebGPU avec la feature `shader-f16`           | q4f16    | WebGPU     | 584 Mo         |
| WebGPU sans `shader-f16`                      | q4       | WebGPU     | 648 Mo         |
| pas de `navigator.gpu`, ou aucun adaptateur   | q4       | WASM (CPU) | 648 Mo         |

Sur WASM, les embeddings viennent de `embed_tokens_uint8` (voir plus bas) : le build WASM d'onnxruntime-web n'a pas
l'opérateur `GatherBlockQuantized` d'`embed_tokens_q4`. Le WASM tourne sur un seul thread (pas de `crossOriginIsolated`,
donc pas de SharedArrayBuffer) : ~11 s pour `/^\d{5}$/`, ~35 s pour une regex de couleur hexadécimale (M-series), contre
~1 à 3 s sur WebGPU.

## Préparer le modèle

Les poids ONNX ne sont pas versionnés. On les génère dans `public/models/sft-qwen3.5-0.8b/` :

```sh
# depuis la racine du dépôt : fusion de l'adapter
uv run mlx_lm.fuse --model Qwen/Qwen3.5-0.8B --adapter-path adapters/sft-qwen3.5-0.8b --save-path models/sft-qwen3.5-0.8b

# depuis playground/ : environnement isolé via les métadonnées PEP 723 du script
uv run scripts/export_qwen35.py               # fp16, q4, q4f16, embed_tokens_uint8 et la carte du modèle (README.md)
uv run scripts/export_qwen35.py --check       # contrôle seul : requantifie la base et compare à onnx-community
uv run scripts/export_qwen35.py --wasm-embed  # ajoute seulement embed_tokens_uint8 à un export existant
```

Optimum ne sait pas exporter Qwen3.5 (couches Gated DeltaNet) : le script part du graphe
d'`onnx-community/Qwen3.5-0.8B-ONNX` (sans encodeur de vision) et y greffe les poids modifiés par le LoRA (détails en
tête du script). `embed_tokens_uint8` reprend les données q4 d'`embed_tokens_q4` à l'identique et les déquantifie avec
des opérateurs standard (Gather, BitwiseAnd, BitShift, Sub, Mul) : sorties identiques au bit près (ORT CPU).

| Variante                              | Taille, sans vision | Usage                         |
| ------------------------------------- | ------------------- | ----------------------------- |
| q4f16                                 | 584 Mo              | WebGPU avec `shader-f16`      |
| q4 (ou q4 + `embed_tokens_uint8`)     | 648 Mo              | WebGPU sans fp16, ou WASM     |
| fp16                                  | ~2 Go               | non publié, non utilisé       |

Scores (succès complet, `results/scores.jsonl`) : regex101 26,0 % en MLX bf16, 24,2 % en q4, 24,0 % en q4f16 (contre
22,8 % pour l'ancien SFT Qwen3-0.6B en fp16) ; held-out 73,8 %, 71,0 %, 71,7 %. Les réponses ONNX ont été générées en
Node (`scripts/generate.mjs`, ORT CPU) à partir des prompts du tokenizer Python. `src/prompt.ts` produit ces prompts
au caractère près (vérifié sur les 842 cas de regex101 et du held-out).

L'ancien SFT Qwen3-0.6B (`sft-0.6b-plus`, fp16, `scripts/export_models.py`) n'est plus proposé ; son repo
`maximedubourg/regex-sft-qwen3-0.6b-onnx` reste en ligne.

## Lancer

```sh
pnpm install
pnpm dev       # http://localhost:5299
```

Build de production en local (même build que Netlify, modèle chargé depuis le Hub) :

```sh
VITE_HUB_SFT_QWEN3_5_0_8B=maximedubourg/regex-sft-qwen3.5-0.8b-onnx@<sha> pnpm build && pnpm preview   # http://localhost:5298
```

## Fonctionnement

- `src/prompt.ts` : port de `scripts/common.py` (prompt système, gabarit de chat Qwen sans thinking, identique pour Qwen3.5, parsing du
  littéral). Le prompt doit rester identique à celui de l'entraînement.
- `src/worker.ts` : chargement (à la demande, bouton « Load model ») et génération dans un Web Worker ; une nouvelle
  requête interrompt la génération en cours, et la regex s'affiche token par token. Le runtime onnxruntime-web est
  servi en local, pas depuis jsDelivr.
- Cache : les fichiers du modèle sont gardés dans le Cache API du navigateur (par origine), sous une clé qui inclut
  `playground_export_id` (écrit dans `config.json` par l'export). Un nouvel export invalide donc l'ancien cache, qui
  est supprimé, comme tout autre cache du playground (dont celui de l'ancien 0.6B, 1,6 Go). Les variantes q4 et q4f16
  partagent la clé (fichiers distincts). Chargement du q4f16 depuis `public/models/` : ~22 s sans cache.
- `src/TestArea.tsx` : chaque ligne est testée avec `RegExp.prototype.test` (✓ / ✗), et les correspondances sont
  surlignées avec une copie de la regex en flag `g`.

## Déployer (Hugging Face + Netlify)

Le build public ne contient pas les poids (`public/` n'est pas copié dans `dist/`, ~26 Mo au total) : le navigateur
les télécharge depuis un repo **public** du Hub, à une révision figée.

1. Publier l'export (le dossier contient déjà la carte du modèle, `README.md`) :

   ```sh
   hf auth login                                                # token avec droit d'écriture
   hf repos create <owner>/regex-sft-qwen3.5-0.8b-onnx --no-private  # public : le navigateur n'envoie pas de token
   hf upload <owner>/regex-sft-qwen3.5-0.8b-onnx public/models/sft-qwen3.5-0.8b . \
     --include "*.json" --include "*.jinja" --include README.md \
     --include "onnx/*_q4.onnx*" --include "onnx/*_q4f16.onnx*" --include "onnx/embed_tokens_uint8.onnx*"   # pas le fp16
   ```

2. Récupérer le SHA du commit (page « Files and versions » du repo) et définir
   `VITE_HUB_SFT_QWEN3_5_0_8B=<owner>/regex-sft-qwen3.5-0.8b-onnx@<sha>` : dans `.env.local` pour tester en dev, et dans
   l'interface Netlify (ou `netlify.toml`) pour le build. Sans aucune variable `VITE_HUB_*`, le build échoue.
3. Netlify lit `netlify.toml` à la racine du dépôt (base `playground`, `pnpm build`, Node 22).

Figer la révision rend le build reproductible et sert de clé au cache navigateur : un nouvel upload ne change rien au
site tant que le SHA n'est pas mis à jour. En dev, une variable `VITE_HUB_*` fait charger ce modèle depuis le Hub au
lieu de `public/models/`.

Publié : `maximedubourg/regex-sft-qwen3.5-0.8b-onnx@7d3395acc8b0aa78c2ebd4fd7c26de365842aadb` (public, comme le 0.6B).

Données d'entraînement : ~1 100 des 9 418 exemples ont été écrits par Claude, dont les conditions d'utilisation
encadrent l'entraînement de modèles sur ses sorties (réserve déjà notée pour le 0.6B enrichi).
