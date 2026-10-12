"""Évalue un modèle (et éventuellement un adapter LoRA) sur la labellisation des MR.

Pour chaque MR : prompt (référentiel de labels + MR) → génération greedy → parsing du JSON → comparaison avec les labels
attendus (F1 micro/macro, exact match, labels hors liste, réponses mal formées).

    uv run scripts/eval.py --name baseline-1.7b
    uv run scripts/eval.py --name sft-1.7b --adapter adapters/sft
    uv run scripts/eval.py --name baseline-4b --model Qwen/Qwen3-4B-Instruct-2507
    uv run scripts/eval.py --name debug --n 20                                     # test rapide

Sorties :
    results/<name>.jsonl  détail par MR (réponse brute, labels prédits et attendus)
    results/scores.jsonl  une ligne de synthèse par run (pour le tableau final)
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import batch_generate, load
from tqdm import tqdm

from common import build_messages, load_config, parse_answer, score

PREPARED_DIR = Path("data/prepared")
RESULTS_DIR = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="nom du run, ex. baseline-1.7b")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default=None, help="dossier d'un adapter LoRA")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--n", type=int, default=None, help="nombre de MR (pour un test rapide)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8, help="les prompts font jusqu'à ~3k tokens")
    args = parser.parse_args()

    config = load_config()
    examples = [json.loads(line) for line in (PREPARED_DIR / f"{args.split}.jsonl").open()][: args.n]
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    # enable_thinking=False : désactive le mode « raisonnement » de Qwen3 (ignoré par les modèles qui ne l'ont pas).
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(e, config[e["project"]]["labels"]), add_generation_prompt=True, enable_thinking=False
        )
        for e in examples
    ]
    # Les prompts de longueur proche sont générés ensemble : moins de padding, donc moins de mémoire et de calcul.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [""] * len(prompts)
    start = time.monotonic()
    for i in tqdm(range(0, len(order), args.batch_size), desc="génération", unit="batch"):
        batch = order[i : i + args.batch_size]
        texts = batch_generate(model, tokenizer, [prompts[j] for j in batch], max_tokens=args.max_tokens).texts
        for j, text in zip(batch, texts):
            outputs[j] = text
    gen_time = time.monotonic() - start

    records = [
        {"id": e["id"], "project": e["project"], "title": e["title"], "labels": e["labels"], "output": out, "predicted": parse_answer(out)}
        for e, out in zip(examples, outputs)
    ]
    result = score(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / f"{args.name}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "name": args.name,
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in result.items() if k != "per_label"},
        "generation_seconds": round(gen_time, 1),
    }
    with (RESULTS_DIR / "scores.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"\n{args.name} sur {args.split} ({result['n']} MR), génération {gen_time:.0f}s")
    print(f"  F1 micro {result['f1_micro']:.3f} | F1 macro {result['f1_macro']:.3f} | exact match {result['exact_match']:.3f}")
    print(f"  labels hors liste {result['invalid_label_rate']:.1%} | réponses mal formées {result['malformed_rate']:.1%}")
    print(f"\n  {'label':28} {'F1':>5} {'préc.':>6} {'rappel':>6} {'support':>7}")
    for name, m in result["per_label"].items():
        print(f"  {name:28} {m['f1']:>5.2f} {m['precision']:>6.2f} {m['recall']:>6.2f} {m['support']:>7}")


if __name__ == "__main__":
    main()
