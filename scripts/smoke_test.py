"""Vérifie que la chaîne complète fonctionne : modèle → SQL → exécution SQLite.

Usage : uv run scripts/smoke_test.py [--model Qwen/Qwen3-1.7B]
"""

import argparse
import sqlite3
from pathlib import Path

from datasets import load_from_disk
from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parent.parent


def schema_of(db_path: Path) -> str:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    return "\n".join(r[0] for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    example = load_from_disk(ROOT / "data" / "raw" / "spider")["validation"][0]
    db_path = ROOT / "data" / "spider" / "database" / example["db_id"] / f"{example['db_id']}.sqlite"

    model, tokenizer = load(args.model)
    messages = [
        {"role": "system", "content": "Tu écris uniquement une requête SQLite, sans explication ni balise markdown."},
        {"role": "user", "content": f"Schéma :\n{schema_of(db_path)}\n\nQuestion : {example['question']}"},
    ]
    # enable_thinking=False : désactive le mode « raisonnement » de Qwen3 pour une réponse directe
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    sql = generate(model, tokenizer, prompt=prompt, max_tokens=256).strip()

    print(f"Question   : {example['question']}")
    print(f"Référence  : {example['query']}")
    print(f"Généré     : {sql}")
    with sqlite3.connect(db_path) as conn:
        expected = conn.execute(example["query"]).fetchall()
        try:
            got = conn.execute(sql).fetchall()
        except sqlite3.Error as e:
            print(f"Exécution  : ✗ erreur SQL ({e})")
            return
    print(f"Exécution  : {'✓ résultat correct' if sorted(got) == sorted(expected) else '✗ résultat différent'}")


if __name__ == "__main__":
    main()
