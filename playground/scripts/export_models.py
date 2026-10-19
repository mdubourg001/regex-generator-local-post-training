# /// script
# requires-python = ">=3.12"
# dependencies = ["optimum-onnx[onnxruntime]==0.1.0", "transformers==4.57.6"]
# ///
"""Exporte les modèles SFT fusionnés (MLX) en ONNX fp16, au format attendu par transformers.js.

    uv run --project .. mlx_lm.fuse --model Qwen/Qwen3-1.7B --adapter-path ../adapters/sft-1.7b --save-path ../models/sft-1.7b
    uv run scripts/export_models.py sft-0.6b sft-1.7b

Sortie : public/models/<nom>/{config.json, generation_config.json, tokenizer*.json, onnx/model_fp16.onnx[_data*]}.

- Le tokenizer vient du modèle de base Qwen3 (identique) : celui sauvé par mlx_lm est au format transformers 5,
  que transformers 4.x ne relit pas.
- Les poids sont découpés en blocs de moins de 2 Go (`model_fp16.onnx_data`, `_data_1`…) : un navigateur ne peut
  pas allouer un ArrayBuffer plus gros, et le protobuf ONNX est limité à 2 Go.
- Pic mémoire mesuré : ~5 Go pour le 0.6B. La validation d'optimum est désactivée : elle charge une seconde copie du
  modèle, et son runtime Python suppose head_dim = hidden_size / heads (64), faux pour Qwen3 (128).
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "public" / "models"
CHUNK_BYTES = 1_800_000_000
BASES = {"sft-0.6b": "Qwen/Qwen3-0.6B", "sft-0.6b-plus": "Qwen/Qwen3-0.6B", "sft-1.7b": "Qwen/Qwen3-1.7B"}

# Provenance des données, reprise dans la carte du modèle (README.md du repo Hugging Face).
DATA = {
    "sft-0.6b": "~8,300 synthetic examples written by DeepSeek V4.1 Flash",
    "sft-0.6b-plus": "~8,300 synthetic examples written by DeepSeek V4.1 Flash, plus ~450 written by Claude "
    "on the weakest categories",
    "sft-1.7b": "~8,300 synthetic examples written by DeepSeek V4.1 Flash",
}

CARD = """---
base_model: {base}
license: apache-2.0
library_name: transformers.js
pipeline_tag: text-generation
tags: [onnx, regex, javascript]
---

# {name}: natural language → JavaScript regex

[{base}](https://huggingface.co/{base}) fine-tuned with LoRA (SFT) to answer a description in English with a
single JavaScript regex literal (ECMAScript 2018), such as `/^\\d{{5}}$/`. ONNX export in fp16 for
[Transformers.js](https://huggingface.co/docs/transformers.js/index) on WebGPU.

- **Data**: {data}. Every example is kept only if its regex passes test strings written in a separate call,
  without seeing the regex, and if trivial or mutated regexes fail at least one of them.
- **Prompt**: the system prompt and user format of the training (see `src/prompt.ts` in the playground), Qwen3 chat
  template with thinking disabled, greedy decoding.
- **Limits**: good on simple requests, weak on intermediate ones, and nearly always wrong on hard ones (lookarounds,
  backreferences, numeric ranges). Mistakes are often almost-right regexes: always test the result.
"""


def tensor_bytes(tensor: onnx.TensorProto, src_dir: Path) -> bytes:
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        return tensor.raw_data
    info = {e.key: e.value for e in tensor.external_data}
    with open(src_dir / info["location"], "rb") as f:
        f.seek(int(info.get("offset", 0)))
        return f.read(int(info["length"]))


def write_chunked(src: Path, dst_dir: Path, name: str) -> int:
    """Réécrit src avec ses poids répartis en blocs de moins de CHUNK_BYTES. Retourne le nombre de blocs."""
    model = onnx.load(str(src), load_external_data=False)
    chunk, size, files = 0, 0, [open(dst_dir / f"{name}_data", "wb")]
    for tensor in model.graph.initializer:
        data = tensor_bytes(tensor, src.parent)
        del tensor.external_data[:]
        if len(data) < 1024:  # petites constantes : restent dans le graphe
            tensor.raw_data, tensor.data_location = data, onnx.TensorProto.DEFAULT
            continue
        if size and size + len(data) > CHUNK_BYTES:
            chunk, size = chunk + 1, 0
            files.append(open(dst_dir / f"{name}_data_{chunk}", "wb"))
        files[-1].write(data)
        tensor.ClearField("raw_data")
        location = Path(files[-1].name).name
        for key, value in (("location", location), ("offset", str(size)), ("length", str(len(data)))):
            tensor.external_data.add(key=key, value=value)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        size += len(data)
    for f in files:
        f.close()
    onnx.save(model, str(dst_dir / name))
    return len(files)


def export(name: str) -> None:
    # Imports locaux : scripts/quantize_models.py réutilise write_chunked sans optimum ni torch.
    from huggingface_hub import snapshot_download
    from optimum.exporters.onnx import main_export

    fused = ROOT / "models" / name
    if not fused.exists():
        sys.exit(f"{fused} introuvable : fusionner l'adapter d'abord (voir l'en-tête du script)")
    dst = OUT / name
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "onnx").mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        main_export(str(fused), output=tmp, task="text-generation-with-past", dtype="fp16", device="cpu", do_validation=False)
        chunks = write_chunked(Path(tmp) / "model.onnx", dst / "onnx", "model_fp16.onnx")
        config = json.loads((Path(tmp) / "config.json").read_text())
        shutil.copy(Path(tmp) / "generation_config.json", dst)
    config["transformers.js_config"] = {"dtype": "fp16", "use_external_data_format": chunks}
    # Clé du cache navigateur (Cache API) : un nouvel export invalide les poids déjà mis en cache.
    config["playground_export_id"] = time.strftime("%Y%m%d-%H%M%S")
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    base = Path(snapshot_download(BASES[name], allow_patterns=["tokenizer.json", "tokenizer_config.json"]))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(base / f, dst / f)
    (dst / "README.md").write_text(CARD.format(name=name, base=BASES[name], data=DATA[name]))
    print(f"{name} → {dst} ({chunks} bloc(s))")


if __name__ == "__main__":
    for arg in sys.argv[1:] or BASES:
        export(arg)
