"""Execution accuracy sur un sous-ensemble fixe de Spider dev.

Pour chaque exemple : prompt (schéma + question) → génération greedy → extraction du SQL → exécution sur la base SQLite →
comparaison avec le résultat de la requête de référence.

Usage :
    uv run scripts/eval.py --name baseline-1.7b
    uv run scripts/eval.py --name sft-1.7b --adapter adapters/sft
    uv run scripts/eval.py --name baseline-4b --model Qwen/Qwen3-4B-Instruct-2507

Sorties :
    results/<name>.jsonl  détail par exemple (pour analyser les erreurs)
    results/scores.jsonl  une ligne de synthèse par run (pour le tableau final)
"""

import argparse
import json
import random
import sqlite3
import time
from collections import Counter

from datasets import load_from_disk
from mlx_lm import batch_generate, load
from tqdm import tqdm

from common import ROOT, build_prompt, execute, extract_sql, results_match, schema_of, spider_db_path

# Sous-ensemble figé : même seed et même taille = mêmes exemples à chaque run. Ne pas changer en cours de projet.
EVAL_SEED = 0
EVAL_SIZE = 500


def eval_subset(n: int):
    dev = load_from_disk(ROOT / "data" / "raw" / "spider")["validation"]
    indices = sorted(random.Random(EVAL_SEED).sample(range(len(dev)), EVAL_SIZE))[:n]
    return dev.select(indices)


def score(example: dict, output: str) -> dict:
    db_path = spider_db_path(example["db_id"])
    pred_sql = extract_sql(output)
    gold = execute(db_path, example["query"])
    try:
        pred = execute(db_path, pred_sql)
    except sqlite3.OperationalError as e:
        status = "timeout" if "interrupted" in str(e) else "sql_error"
        return {"status": status, "pred_sql": pred_sql, "error": str(e)}
    except (sqlite3.Error, sqlite3.Warning) as e:
        return {"status": "sql_error", "pred_sql": pred_sql, "error": str(e)}
    ordered = "order by" in example["query"].lower()
    return {"status": "correct" if results_match(pred, gold, ordered) else "wrong_result", "pred_sql": pred_sql}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="nom du run, ex. baseline-1.7b")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default=None, help="dossier d'un adapter LoRA")
    parser.add_argument("--n", type=int, default=EVAL_SIZE, help="nombre d'exemples (pour un test rapide)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    examples = eval_subset(args.n)
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    schemas: dict[str, str] = {}
    prompts = []
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = schema_of(spider_db_path(ex["db_id"]))
        prompts.append(build_prompt(tokenizer, schemas[ex["db_id"]], ex["question"]))

    start = time.monotonic()
    outputs: list[str] = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="génération", unit="batch"):
        outputs += batch_generate(model, tokenizer, prompts[i : i + args.batch_size], max_tokens=args.max_tokens).texts
    gen_time = time.monotonic() - start

    rows = []
    for ex, out in zip(examples, outputs):
        rows.append({"db_id": ex["db_id"], "question": ex["question"], "gold_sql": ex["query"], "output": out, **score(ex, out)})

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"{args.name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(r["status"] for r in rows)
    accuracy = counts["correct"] / len(rows)
    summary = {
        "name": args.name,
        "model": args.model,
        "adapter": args.adapter,
        "n": len(rows),
        "execution_accuracy": round(accuracy, 4),
        "statuses": dict(counts),
        "generation_seconds": round(gen_time, 1),
    }
    with open(results_dir / "scores.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"\n{args.name} : execution accuracy = {accuracy:.1%} ({counts['correct']}/{len(rows)})")
    for status, c in counts.most_common():
        print(f"  {status:<13} {c:>4}  ({c / len(rows):.1%})")
    print(f"  génération : {gen_time:.0f}s")


if __name__ == "__main__":
    main()
