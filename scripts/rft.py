"""Jeu de données du rejection sampling fine-tuning (RFT, étape 9 bis), tiré des échantillons de l'étape 7.

Le modèle SFT a déjà généré 8 réponses par exemple du train, notées par exécution (`data/dpo/samples-train.jsonl`). On
garde les réponses entièrement justes, et on refait un SFT dessus : le modèle apprend sur ses propres réussites, sans
paire ni reward graduée. C'est la baseline la plus simple d'apprentissage sur ses propres sorties, celle que le GRPO
doit battre.

Pour chaque exemple, on garde jusqu'à `--per-example` réponses justes distinctes, les plus fréquentes d'abord (à
égalité, la plus courte). Plusieurs formulations justes d'une même regex apportent de la diversité ; les doublons
n'apprendraient qu'à renforcer la plus probable. Un exemple sans réponse juste ne donne rien (pas de repli sur la regex
de référence : ce serait refaire le SFT).

Le valid est celui du SFT (regex de référence) : la val loss reste comparable d'un run à l'autre.

    uv run scripts/rft.py
    uv run mlx_lm.lora --config configs/rft.yaml

Sortie : data/rft/{train,valid}.jsonl, au format {"messages": [...]} de mlx_lm.lora.
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from common import build_messages

PREPARED_DIR = Path("data/prepared")
SAMPLES = Path("data/dpo/samples-train.jsonl")
RFT_DIR = Path("data/rft")
MAX_PROMPT_TOKENS = 384  # avec 128 tokens de réponse, reste sous le max_seq_length de 512 du SFT


def keep(record: dict, per_example: int) -> list[str]:
    """Réponses justes distinctes d'un exemple, les plus fréquentes d'abord, `per_example` au plus."""
    right = Counter(o.strip() for o, ok in zip(record["outputs"], record["success"]) if ok)
    return sorted(right, key=lambda a: (-right[a], len(a), a))[:per_example]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-example", type=int, default=2, help="réponses justes distinctes gardées par exemple")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = {e["id"]: e for e in map(json.loads, (PREPARED_DIR / "train.jsonl").open())}
    rows, kept_per_example, too_long = [], Counter(), 0
    for record in map(json.loads, SAMPLES.open()):
        if record["prompt_tokens"] > MAX_PROMPT_TOKENS:
            too_long += 1
            continue
        answers = keep(record, args.per_example)
        kept_per_example[len(answers)] += 1
        messages = build_messages(examples[record["id"]])
        rows += [{"messages": messages + [{"role": "assistant", "content": a}]} for a in answers]
    random.Random(args.seed).shuffle(rows)  # les échantillons sont triés par longueur de prompt

    RFT_DIR.mkdir(parents=True, exist_ok=True)
    with (RFT_DIR / "train.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copy("data/sft/valid.jsonl", RFT_DIR / "valid.jsonl")

    n = sum(kept_per_example.values())
    print(f"{n} exemples ({too_long} prompts trop longs écartés) → {len(rows)} réponses dans {RFT_DIR / 'train.jsonl'}")
    for k in sorted(kept_per_example):
        print(f"  {k} réponse(s) gardée(s) : {kept_per_example[k] / n:.1%} des exemples")


if __name__ == "__main__":
    main()
