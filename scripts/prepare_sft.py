"""Prépare les données à partir des MR extraites : filtres, split par date, puis JSONL au format attendu par mlx-lm.

    uv run scripts/prepare_sft.py

Sorties :
- data/prepared/{train,valid,test}.jsonl : exemples structurés (titre, fichiers, diffs, labels…), utilisés par
  l'éval, la baseline à base de règles et la génération des paires DPO ;
- data/sft/{train,valid}.jsonl : conversations `messages` (system / user / assistant) pour `mlx_lm.lora`.

Le split est temporel, par projet : les MR les plus récentes forment le test (dont on tirera le golden set relabellisé
à la main), celles juste avant la validation, le reste l'entraînement. Un split aléatoire mettrait des MR quasi
identiques (un fix et son backport, une série de MR d'un même ticket) des deux côtés.
"""

import json
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from common import build_messages, load_config, load_examples

PREPARED_DIR = Path("data/prepared")
SFT_DIR = Path("data/sft")
VALID_RATIO = 0.05
TEST_RATIO = 0.15
TOKENIZER = "Qwen/Qwen3-1.7B"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    config = load_config()
    splits = {"train": [], "valid": [], "test": []}

    for project, cfg in config.items():
        stats = Counter()
        examples = sorted(load_examples(project, cfg, stats), key=lambda e: e["created_at"])
        n_test, n_valid = round(len(examples) * TEST_RATIO), round(len(examples) * VALID_RATIO)
        n_train = len(examples) - n_test - n_valid
        splits["train"] += examples[:n_train]
        splits["valid"] += examples[n_train : n_train + n_valid]
        splits["test"] += examples[n_train + n_valid :]

        print(f"\n## {cfg['alias']} ({project})")
        for reason, count in stats.most_common():
            print(f"  {reason:45} {count:>5}")
        print(f"  split : train {n_train}, valid {n_valid}, test {n_test} (test à partir du {examples[n_train + n_valid]['created_at'][:10]})")
        labels = Counter(label for e in examples for label in e["labels"])
        print("  labels : " + ", ".join(f"{name} {labels[name]}" for name in cfg["labels"]))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for split, examples in splits.items():
        write_jsonl(PREPARED_DIR / f"{split}.jsonl", examples)
        if split == "test":
            continue
        rows = [{"messages": build_messages(e, config[e["project"]]["labels"], with_answer=True)} for e in examples]
        write_jsonl(SFT_DIR / f"{split}.jsonl", rows)
        lengths = sorted(len(tokenizer.encode(tokenizer.apply_chat_template(r["messages"], tokenize=False))) for r in rows)
        print(
            f"\n{split}: {len(rows)} exemples, tokens médiane {statistics.median(lengths):.0f}, "
            f"p95 {lengths[int(len(lengths) * 0.95)]}, max {lengths[-1]}"
        )


if __name__ == "__main__":
    main()
