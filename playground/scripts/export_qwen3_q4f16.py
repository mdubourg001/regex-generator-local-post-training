# /// script
# requires-python = ">=3.12"
# dependencies = ["onnx==1.19.1", "onnxruntime==1.31.1", "numpy", "huggingface_hub"]
# ///
"""Exporte un SFT Qwen3 (fusionné par MLX) en ONNX q4f16 pour transformers.js, sans optimum.

    uv run --project .. mlx_lm.fuse --model Qwen/Qwen3-1.7B --adapter-path ../adapters/sft-1.7b --save-path ../models/sft-1.7b
    uv run scripts/export_qwen3_q4f16.py sft-1.7b           # → public/models/sft-1.7b/
    uv run scripts/export_qwen3_q4f16.py sft-1.7b --check   # contrôle seul : requantifie la base et compare

Même principe que export_qwen35.py : on part du graphe q4f16 d'onnx-community (GroupQueryAttention fusionné, bien plus
rapide sur WebGPU que l'export optimum non fusionné) et on remplace les 7 projections par couche touchées par le LoRA
(q/k/v/o_proj, gate/up/down_proj), requantifiées comme l'original : MatMulNBits 4 bits, blocs de 32, symétrique (pas
d'entrée zero_point), échelles fp16. Embeddings (partagés avec le lm_head) et normes : copiés tels quels.
Pic mémoire : le graphe chargé (~1,5 Go) plus une matrice à la fois ; les poids bf16 sont lus en memmap.
"""

import argparse
import json
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import snapshot_download
from onnx import numpy_helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits

sys.path.insert(0, str(Path(__file__).parent))
from export_models import write_chunked  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "public" / "models"
MODELS = {
    "sft-1.7b": ("Qwen/Qwen3-1.7B", "onnx-community/Qwen3-1.7B-ONNX", "3231b522c6e703b2f9b3cd8ea27426a1289a02f2"),
}
PROJ = {"attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "mlp": ("gate_proj", "up_proj", "down_proj")}
BLOCK = 32

CARD = """---
base_model: {base}
license: apache-2.0
library_name: transformers.js
pipeline_tag: text-generation
tags: [onnx, regex, javascript]
---

# {name}: natural language → JavaScript regex

[{base}](https://huggingface.co/{base}) fine-tuned with LoRA (SFT) to answer a description in English with a
single JavaScript regex literal (ECMAScript 2018), such as `/^\\d{{5}}$/`. ONNX export in q4f16 (`dtype: "q4f16"`) for
[Transformers.js](https://huggingface.co/docs/transformers.js/index) on WebGPU (requires the `shader-f16` feature).

The graph is the one of [{community}](https://huggingface.co/{community}), with the matrices touched by the LoRA
replaced by the fine-tuned ones, quantized the same way (MatMulNBits, 4 bits, blocks of 32, symmetric).

- **Data**: ~8,300 synthetic examples written by DeepSeek V4.1 Flash. Every example is kept only if its regex passes
  test strings written in a separate call, without seeing the regex, and if trivial or mutated regexes fail at least
  one of them.
- **Prompt**: the system prompt and user format of the training (see `src/prompt.ts` in the playground), Qwen3 chat
  template with thinking disabled, greedy decoding.
- **Scores** (share of regexes passing every hidden test; 4-bit score measured on an MLX 4-bit quantization of the
  same weights, groups of 32, as a proxy for this export):

  | Model                          | regex101 (400 real-world needs) | held-out (442 synthetic) |
  | ------------------------------ | ------------------------------- | ------------------------ |
  | this model, MLX bf16           | 26.5 %                          | 76.9 %                   |
  | this model, MLX 4-bit          | 25.8 %                          | 70.8 %                   |

- **Limits**: good on simple requests, weak on intermediate ones, and nearly always wrong on hard ones (lookarounds,
  backreferences, numeric ranges). Mistakes are often almost-right regexes: always test the result.
"""


class Bf16Reader:
    """Lit des safetensors (bf16, fp16 ou fp32) tenseur par tenseur, en memmap, en float32."""

    def __init__(self, folder: Path):
        self.where = {}
        for f in sorted(folder.glob("*.safetensors")):
            with open(f, "rb") as fh:
                (size,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(size))
            mm = np.memmap(f, dtype=np.uint8, mode="r", offset=8 + size)
            for k, v in header.items():
                if k != "__metadata__":
                    self.where[k] = (mm, v)

    def __call__(self, key: str) -> np.ndarray:
        mm, info = self.where[key]
        start, end = info["data_offsets"]
        raw = mm[start:end]
        if info["dtype"] == "BF16":
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
        return raw.view({"F16": np.float16, "F32": np.float32}[info["dtype"]]).astype(np.float32).reshape(info["shape"])


def quantize(w_kn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4 bits symétrique par blocs de 32 le long de K. w_kn : [K, N] fp16. Retourne Q4 [N, blocs, 16] et échelles."""
    k, n = w_kn.shape
    blocks = k // BLOCK
    assert blocks * BLOCK == k
    packed = np.zeros((n, blocks, BLOCK // 2), dtype=np.uint8)
    scales = np.zeros(n * blocks, dtype=np.float16)
    zp = np.zeros(n * ((blocks + 1) // 2), dtype=np.uint8)  # ignoré en symétrique
    quantize_matmul_4bits(packed, np.ascontiguousarray(w_kn), scales, zp, BLOCK, n, k, True)
    return packed, scales


def replace(tensor: onnx.TensorProto, array: np.ndarray) -> None:
    assert tuple(tensor.dims) == array.shape, (tensor.name, tuple(tensor.dims), array.shape)
    assert onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type) == array.dtype, (tensor.name, array.dtype)
    tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))


def patch(model: onnx.ModelProto, read: Bf16Reader, layers: int, check: bool) -> None:
    inits = {t.name: t for t in model.graph.initializer}
    count, worst, worst_scale = 0, 0.0, 0.0
    for i in range(layers):
        for block, projs in PROJ.items():
            hf_block = "self_attn" if block == "attn" else "mlp"
            for p in projs:
                name = f"model.layers.{i}.{block}.{p}.MatMul.weight"
                w = read(f"model.layers.{i}.{hf_block}.{p}.weight").T.astype(np.float16)  # [K, N]
                q, s = quantize(w)
                if check:
                    worst = max(worst, float((numpy_helper.to_array(inits[f"{name}_Q4"]) != q).mean()))
                    ref = numpy_helper.to_array(inits[f"{name}_scales"]).astype(np.float32)
                    worst_scale = max(worst_scale, float(np.abs(ref - s.astype(np.float32)).max() / np.abs(ref).max()))
                else:
                    replace(inits[f"{name}_Q4"], q)
                    replace(inits[f"{name}_scales"], s)
                count += 1
    matmuls = sum(n.op_type == "MatMulNBits" for n in model.graph.node)
    assert count == matmuls == layers * 7, (count, matmuls)
    if check:
        print(f"{count} matrices contrôlées : octets 4 bits différents ≤ {worst:.2%}, écart relatif des échelles ≤ {worst_scale:.1e}")
    else:
        print(f"{count} matrices remplacées")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", choices=MODELS)
    parser.add_argument("--check", action="store_true", help="requantifie la base et compare à onnx-community, sans écrire")
    args = parser.parse_args()
    base, community_repo, revision = MODELS[args.name]

    community = Path(snapshot_download(community_repo, revision=revision,
                                       allow_patterns=["*.json", "*.jinja", "onnx/model_q4f16.onnx"]))
    if args.check:
        folder = Path(snapshot_download(base, allow_patterns=["*.safetensors", "*.json"]))
    else:
        folder = ROOT / "models" / args.name
        if not folder.exists():
            sys.exit(f"{folder} introuvable : fusionner l'adapter d'abord (voir l'en-tête du script)")
    layers = json.loads((folder / "config.json").read_text())["num_hidden_layers"]

    model = onnx.load(str(community / "onnx" / "model_q4f16.onnx"))
    patch(model, Bf16Reader(folder), layers, args.check)
    if args.check:
        return

    dst = OUT / args.name
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "onnx").mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=dst) as tmp:
        onnx.save(model, f"{tmp}/model.onnx", save_as_external_data=True, location="model.onnx_data")
        del model
        chunks = write_chunked(Path(tmp) / "model.onnx", dst / "onnx", "model_q4f16.onnx")

    config = json.loads((community / "config.json").read_text())
    config["transformers.js_config"] = {"dtype": "q4f16", "kv_cache_dtype": {"q4f16": "float16"},
                                        "use_external_data_format": {"model_q4f16.onnx": chunks}}
    # Clé du cache navigateur (Cache API) : un nouvel export invalide les poids déjà mis en cache.
    config["playground_export_id"] = time.strftime("%Y%m%d-%H%M%S")
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for f in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (community / f).exists():
            shutil.copyfile(community / f, dst / f)
    (dst / "README.md").write_text(CARD.format(name=args.name, base=base, community=community_repo))
    print(f"{args.name} → {dst} ({chunks} bloc(s))")


if __name__ == "__main__":
    main()
