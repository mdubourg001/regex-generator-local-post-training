# /// script
# requires-python = ">=3.12"
# dependencies = ["onnx==1.19.1", "onnxruntime==1.31.1", "numpy", "safetensors", "huggingface_hub"]
# ///
"""Exporte le SFT Qwen3.5-0.8B (fusionné par MLX) en ONNX pour transformers.js, en fp16, q4 et q4f16.

    uv run --project .. mlx_lm.fuse --model Qwen/Qwen3.5-0.8B --adapter-path ../adapters/sft-qwen3.5-0.8b \
        --save-path ../models/sft-qwen3.5-0.8b
    uv run scripts/export_qwen35.py            # → public/models/sft-qwen3.5-0.8b/
    uv run scripts/export_qwen35.py --check    # contrôle seul : requantifie la base et compare à onnx-community
    uv run scripts/export_qwen35.py --wasm-embed  # ajoute seulement embed_tokens_uint8 (WASM) à un export existant

Optimum ne sait pas exporter Qwen3.5 (couches Gated DeltaNet hybrides). On part donc de l'export communautaire
`onnx-community/Qwen3.5-0.8B-ONNX`, dont le graphe garde les noms des poids, et on remplace les seules matrices modifiées
par le LoRA (in_proj_qkv, in_proj_z, out_proj, q/k/v/o_proj, gate/up/down_proj). Tout le reste (embeddings, lm_head,
conv1d, normes, in_proj_a/b) est copié tel quel. Hypothèses vérifiées par --check :
- chaque `model.layers.N.<gdn|attn|mlp>.X.MatMul.weight` du graphe fp16 est le poids HF de base transposé ;
- les tenseurs q4 / q4f16 (MatMulNBits, blocs de 32, asymétrique) se reproduisent avec `quantize_matmul_4bits`
  d'onnxruntime, depuis les poids fp32 (q4) ou fp16 (q4f16) : échelles identiques, ~0,3 % des valeurs 4 bits
  diffèrent d'un cran (égalités d'arrondi), erreur moyenne identique. Seule exception, des canaux de sortie quasi nuls
  (~1e-37, couche 0) : onnx-community leur met une échelle de 1,0, nous une échelle infime ; les deux déquantifient à 0.
"""

import argparse
import glob
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import snapshot_download
from onnx import numpy_helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).parent))
from export_models import write_chunked  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "public" / "models"
COMMUNITY = "onnx-community/Qwen3.5-0.8B-ONNX"
NAME = "sft-qwen3.5-0.8b"
TARGETS = ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DTYPES = ("fp16", "q4", "q4f16")
BLOCK = 32


CARD = """---
base_model: Qwen/Qwen3.5-0.8B
license: apache-2.0
library_name: transformers.js
pipeline_tag: text-generation
tags: [onnx, regex, javascript]
---

# sft-qwen3.5-0.8b: natural language → JavaScript regex

[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) fine-tuned with LoRA (SFT) to answer a description in
English, optionally with example strings, with a single JavaScript regex literal (ECMAScript 2018), such as
`/^\\d{5}$/`. Text-only ONNX export for [Transformers.js](https://huggingface.co/docs/transformers.js/index), in 4-bit
variants:

| Variant        | `dtype`                                          | Download | Use                                        |
| -------------- | ------------------------------------------------ | -------- | ------------------------------------------ |
| q4f16          | `"q4f16"`                                        | 584 MB   | WebGPU with the `shader-f16` feature       |
| q4             | `"q4"`                                           | 648 MB   | WebGPU without `shader-f16`                |
| q4, WASM (CPU) | `{ embed_tokens: "uint8", decoder_model_merged: "q4" }` | 648 MB   | no WebGPU                       |

The graph is the one of [onnx-community/Qwen3.5-0.8B-ONNX](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX)
(no vision encoder), with the matrices touched by the LoRA replaced by the fine-tuned ones, quantized the same way
(MatMulNBits, 4 bits, blocks of 32). The WASM build of onnxruntime-web has no `GatherBlockQuantized` kernel, used by
`embed_tokens_q4`: `embed_tokens_uint8` holds the same 4-bit weights (packed two per byte) and dequantizes them with
standard operators, with bit-identical outputs.

```js
import { AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";

const tokenizer = await AutoTokenizer.from_pretrained("maximedubourg/regex-sft-qwen3.5-0.8b-onnx");
const model = await AutoModelForCausalLM.from_pretrained("maximedubourg/regex-sft-qwen3.5-0.8b-onnx", {
  dtype: "q4f16", // see the table above
  device: "webgpu", // or "wasm"
});
```

- **Data**: about 9,400 synthetic examples (description → regex), mostly written by DeepSeek, plus about 1,100
  written by Claude. Every example is kept only if its regex passes test strings written in a separate call,
  without seeing the regex, and if trivial or mutated regexes fail at least one of them.
- **Prompt**: the system prompt and user format of the training (see `src/prompt.ts` in the playground), Qwen chat
  template with thinking disabled (the prompt ends with `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`),
  greedy decoding.
- **Scores** (share of regexes passing every hidden test):

  | Model                          | regex101 (400 real-world needs) | held-out (442 synthetic) |
  | ------------------------------ | ------------------------------- | ------------------------ |
  | this model, MLX bf16           | 26.0 %                          | 73.8 %                   |
  | this model, ONNX `q4f16`       | 24.0 %                          | 71.7 %                   |
  | this model, ONNX `q4`          | 24.2 %                          | 71.0 %                   |
  | previous SFT Qwen3-0.6B (fp16) | 22.8 %                          |                          |

  regex101: needs from the regex101 community library, with descriptions, reference regexes and tests rewritten by
  another model family. Held-out: same distribution as the training data, never trained on.
- **Limits**: good on simple requests (about 4 in 10 fully right on regex101), weak on intermediate ones (about 1 in
  8), and nearly always wrong on hard ones (lookarounds, backreferences, numeric ranges). Mistakes are often
  almost-right regexes: always test the result.
"""


def target(onnx_name: str) -> bool:
    return onnx_name.startswith("model.layers.") and onnx_name.endswith(".MatMul.weight") and onnx_name.split(".")[-3] in TARGETS


def hf_key(onnx_name: str, prefix: str) -> str:
    """model.layers.3.attn.q_proj.MatMul.weight → <prefix>layers.3.self_attn.q_proj.weight"""
    n = onnx_name.removeprefix("model.").replace(".MatMul.weight", ".weight")
    return prefix + n.replace(".gdn.", ".linear_attn.").replace(".attn.", ".self_attn.")


def load_weights(path: str) -> dict[str, np.ndarray]:
    weights = {}
    for f in glob.glob(f"{path}/*.safetensors"):
        weights.update(load_file(f))
    return weights


def quantize(w_kn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantification 4 bits par blocs de 32 le long de K, comme MatMul4BitsQuantizer (asymétrique).
    w_kn : [K, N] en float32 (q4) ou float16 (q4f16) ; les échelles ont le même type."""
    k, n = w_kn.shape
    blocks = (k + BLOCK - 1) // BLOCK
    assert blocks * BLOCK == k
    packed = np.zeros((n, blocks, BLOCK // 2), dtype=np.uint8)
    scales = np.zeros(n * blocks, dtype=w_kn.dtype)
    zp = np.zeros(n * ((blocks + 1) // 2), dtype=np.uint8)
    quantize_matmul_4bits(packed, np.ascontiguousarray(w_kn), scales, zp, BLOCK, n, k, False)
    return packed, scales.reshape(n, blocks), zp.reshape(n, -1)


def replace(tensor: onnx.TensorProto, array: np.ndarray) -> None:
    assert tuple(tensor.dims) == array.shape, (tensor.name, tuple(tensor.dims), array.shape)
    assert onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type) == array.dtype, (tensor.name, array.dtype)
    tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))


def patch(src: Path, weights: dict[str, np.ndarray], prefix: str, dtype: str, check: bool) -> onnx.ModelProto:
    """Charge le décodeur communautaire `dtype` et y remplace les matrices ciblées par `weights` (format HF [N, K])."""
    model = onnx.load(str(src / f"decoder_model_merged_{dtype}.onnx"))
    inits = {t.name: t for t in model.graph.initializer}
    # Graphe fp16 : `model.layers.3.attn.q_proj.MatMul.weight`. Graphes quantifiés : les points deviennent des `_`
    # (`model_layers_3_attn_q_proj_MatMul_weight` + `_quant`, `_scales`, `_zp`) ; on retrouve le nom pointé.
    if dtype == "fp16":
        pairs = [(n, n) for n in inits if target(n)]
    else:
        pairs = []
        for n in inits:
            if n.startswith("model_layers_") and n.endswith("_MatMul_weight_quant"):
                base = n.removesuffix("_quant")
                _, _, layer, block, rest = base.split("_", 4)
                dotted = f"model.layers.{layer}.{block}.{rest.removesuffix('_MatMul_weight')}.MatMul.weight"
                if target(dotted):
                    pairs.append((base, dotted))
    count, worst = 0, 0.0
    for name, onnx_name in pairs:
        w = weights[hf_key(onnx_name, prefix)].astype(np.float32).T  # [K, N]
        if dtype == "fp16":
            new = w.astype(np.float16)
            if check:
                worst = max(worst, float(np.abs(numpy_helper.to_array(inits[name]).astype(np.float32) - new).max()))
            else:
                replace(inits[name], new)
        else:
            q, s, z = quantize(w if dtype == "q4" else w.astype(np.float16))
            if check:  # part d'octets 4 bits différents (égalités d'arrondi : ~0,3 %)
                worst = max(worst, float((numpy_helper.to_array(inits[f"{name}_quant"]) != q).mean()))
            else:
                replace(inits[f"{name}_quant"], q)
                replace(inits[f"{name}_scales"], s.astype(onnx.helper.tensor_dtype_to_np_dtype(inits[f"{name}_scales"].data_type)))
                replace(inits[f"{name}_zp"], z)
        count += 1
    expected = 24 * 3 + 18 * 3 + 6 * 4  # MLP partout, 3 projections GDN sur 18 couches, 4 d'attention sur 6
    assert count == expected, (dtype, count, expected)
    print(f"{dtype}: {count} matrices {'contrôlées, écart max ' + format(worst, '.2e') if check else 'remplacées'}")
    return model


def write_wasm_embed(onnx_dir: Path) -> None:
    """embed_tokens_q4 sans GatherBlockQuantized, absent du build WASM d'onnxruntime-web : embed_tokens_uint8.onnx.

    Mêmes données q4 (table uint8 de 2 valeurs 4 bits par octet, échelles fp32 et zéros 4 bits par bloc de 32, copiés
    à l'identique dans embed_tokens_uint8.onnx_data), déquantifiées par des opérateurs standard après le Gather :
    (q - zp) * scale, demi-octet bas d'abord. Résultat identique au bit près à GatherBlockQuantized (vérifié sur ORT CPU).
    transformers.js le charge avec dtype {embed_tokens: "uint8", decoder_model_merged: "q4"}.
    """
    from onnx import TensorProto, helper

    src = onnx.load(str(onnx_dir / "embed_tokens_q4.onnx"), load_external_data=False)
    (node,) = src.graph.node
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    assert attrs == {"bits": 4, "block_size": BLOCK, "gather_axis": 0, "quantize_axis": 1}, attrs
    table, _, scales, zps = node.input
    inits = []
    for t in src.graph.initializer:
        t = onnx.TensorProto.FromString(t.SerializeToString())
        for e in t.external_data:
            if e.key == "location":
                e.value = "embed_tokens_uint8.onnx_data"
        inits.append(t)
    hidden = 2 * {t.name: t for t in inits}[table].dims[1]
    const = lambda name, values, dtype=TensorProto.INT64: helper.make_tensor(name, dtype, [len(values)], values)
    inits += [const("c_mask", [15], TensorProto.UINT8), const("c_shift", [4], TensorProto.UINT8),
              const("c_last", [-1]), const("c_blocks", [0, 0, hidden // BLOCK, BLOCK]),
              const("c_zshape", [0, 0, hidden // BLOCK, 1]), const("c_out", [0, 0, hidden])]

    def unpack(x: str) -> list:  # [..., n] uint8 → [..., 2n] float, demi-octet bas d'abord
        return [
            helper.make_node("BitwiseAnd", [x, "c_mask"], [f"{x}_lo"]),
            helper.make_node("BitShift", [x, "c_shift"], [f"{x}_hi"], direction="RIGHT"),
            helper.make_node("Unsqueeze", [f"{x}_lo", "c_last"], [f"{x}_lo1"]),
            helper.make_node("Unsqueeze", [f"{x}_hi", "c_last"], [f"{x}_hi1"]),
            helper.make_node("Concat", [f"{x}_lo1", f"{x}_hi1"], [f"{x}_pairs"], axis=-1),
            helper.make_node("Cast", [f"{x}_pairs"], [f"{x}_f"], to=TensorProto.FLOAT),
        ]

    nodes = [
        helper.make_node("Gather", [table, "input_ids"], ["q"], axis=0),  # [b, s, hidden / 2]
        helper.make_node("Gather", [scales, "input_ids"], ["s"], axis=0),  # [b, s, blocks]
        helper.make_node("Gather", [zps, "input_ids"], ["z"], axis=0),  # [b, s, blocks / 2]
        *unpack("q"), *unpack("z"),
        helper.make_node("Reshape", ["q_f", "c_blocks"], ["q_blocks"]),  # [b, s, blocks, 32]
        helper.make_node("Reshape", ["z_f", "c_zshape"], ["z1"]),  # [b, s, blocks / 2, 2] → [b, s, blocks, 1]
        helper.make_node("Unsqueeze", ["s", "c_last"], ["s1"]),  # [b, s, blocks, 1]
        helper.make_node("Sub", ["q_blocks", "z1"], ["centered"]),
        helper.make_node("Mul", ["centered", "s1"], ["scaled"]),
        helper.make_node("Reshape", ["scaled", "c_out"], ["inputs_embeds"]),
    ]
    graph = helper.make_graph(nodes, "embed_tokens_uint8", list(src.graph.input), list(src.graph.output), inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=src.ir_version)
    shutil.copyfile(onnx_dir / "embed_tokens_q4.onnx_data", onnx_dir / "embed_tokens_uint8.onnx_data")
    onnx.save(model, str(onnx_dir / "embed_tokens_uint8.onnx"))
    onnx.checker.check_model(str(onnx_dir / "embed_tokens_uint8.onnx"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare la base requantifiée à onnx-community, sans écrire")
    parser.add_argument("--dtypes", nargs="+", default=list(DTYPES), choices=DTYPES)
    parser.add_argument("--wasm-embed", action="store_true", help="ajoute seulement embed_tokens_uint8 à l'export existant")
    args = parser.parse_args()
    if args.wasm_embed:
        return write_wasm_embed(OUT / NAME / "onnx")

    community = Path(snapshot_download(COMMUNITY, allow_patterns=[
        "*.json", "*.jinja", *(f"onnx/{p}_{d}.onnx*" for d in args.dtypes for p in ("decoder_model_merged", "embed_tokens"))]))
    if args.check:
        weights, prefix = load_weights(snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=["*.safetensors"])), "model.language_model."
    else:
        fused = ROOT / "models" / NAME
        if not fused.exists():
            sys.exit(f"{fused} introuvable : fusionner l'adapter d'abord (voir l'en-tête du script)")
        weights, prefix = load_weights(str(fused)), "language_model.model."
        dst = OUT / NAME
        shutil.rmtree(dst, ignore_errors=True)
        (dst / "onnx").mkdir(parents=True)

    chunks = {}
    for dtype in args.dtypes:
        model = patch(community / "onnx", weights, prefix, dtype, args.check)
        if args.check:
            continue
        with tempfile.TemporaryDirectory(dir=dst) as tmp:
            onnx.save(model, f"{tmp}/decoder.onnx", save_as_external_data=True, location="decoder.onnx_data")
            del model
            chunks[f"decoder_model_merged_{dtype}"] = write_chunked(Path(tmp) / "decoder.onnx", dst / "onnx", f"decoder_model_merged_{dtype}.onnx")
        for f in (community / "onnx").glob(f"embed_tokens_{dtype}.onnx*"):  # embeddings non touchés par le LoRA
            shutil.copyfile(f, dst / "onnx" / f.name)
        if dtype == "q4":
            write_wasm_embed(dst / "onnx")
    if args.check:
        return

    config = json.loads((community / "config.json").read_text())
    # Clés sans suffixe de dtype (format d'onnx-community) : le plus grand nombre de blocs vaut pour tous les dtypes.
    config["transformers.js_config"] = {
        "use_external_data_format": {"decoder_model_merged": max(chunks.values()), "embed_tokens": 1},
        "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
    }
    config["playground_export_id"] = time.strftime("%Y%m%d-%H%M%S")
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for f in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "preprocessor_config.json", "processor_config.json"):
        if (community / f).exists():
            shutil.copyfile(community / f, dst / f)
    (dst / "README.md").write_text(CARD)  # carte du modèle, publiée avec l'export sur le Hub
    print(f"{NAME} → {dst}")


if __name__ == "__main__":
    main()
