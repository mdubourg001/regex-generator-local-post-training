"""Vérifie que la chaîne complète fonctionne : modèle → SQL → exécution SQLite.

Usage : uv run scripts/smoke_test.py [--model Qwen/Qwen3-1.7B]
"""

import argparse

from datasets import load_from_disk
from mlx_lm import generate, load

from common import ROOT, build_prompt, execute, extract_sql, results_match, schema_of, spider_db_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    example = load_from_disk(ROOT / "data" / "raw" / "spider")["validation"][0]
    db_path = spider_db_path(example["db_id"])

    model, tokenizer = load(args.model)
    prompt = build_prompt(tokenizer, schema_of(db_path), example["question"])
    sql = extract_sql(generate(model, tokenizer, prompt=prompt, max_tokens=256))

    print(f"Question   : {example['question']}")
    print(f"Référence  : {example['query']}")
    print(f"Généré     : {sql}")
    try:
        ok = results_match(execute(db_path, sql), execute(db_path, example["query"]), ordered=False)
    except Exception as e:
        print(f"Exécution  : ✗ erreur SQL ({e})")
        return
    print(f"Exécution  : {'✓ résultat correct' if ok else '✗ résultat différent'}")


if __name__ == "__main__":
    main()
