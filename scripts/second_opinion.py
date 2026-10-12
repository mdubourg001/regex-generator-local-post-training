"""Mesure le bruit des données gardées : un second modèle, d'une autre famille que le professeur, écrit sa propre regex
pour un échantillon d'exemples, à partir du même prompt que le modèle entraîné.

S'il rate les tests, soit il se trompe, soit l'exemple est bruité (regex et tests du professeur d'accord sur une
mauvaise interprétation). Le taux d'accord borne donc le bruit par le bas, et l'inspection des désaccords le précise.

    uv run scripts/second_opinion.py --n 200

Sortie : data/raw/second_opinion.jsonl (réponse brute, coût), puis le taux d'accord et des désaccords à inspecter.
"""

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from common import RegexRunner, build_messages, grade, score
from generate import CONFIG, RAW_DIR
from openrouter import OpenRouter

PREPARED_DIR = Path("data/prepared")
LOG = RAW_DIR / "second_opinion.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="taille de l'échantillon")
    parser.add_argument("--show", type=int, default=15, help="nombre de désaccords à afficher")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    examples = [json.loads(line) for split in ("train", "valid", "heldout") for line in (PREPARED_DIR / f"{split}.jsonl").open()]
    sample = random.Random(0).sample(examples, min(args.n, len(examples)))
    done = {r["id"]: r for r in map(json.loads, LOG.open())} if LOG.exists() else {}

    client, lock = OpenRouter(), threading.Lock()

    def ask(e: dict) -> None:
        response = client.chat(cfg["second_opinion"], build_messages(e), temperature=0)
        with lock, LOG.open("a") as f:
            record = {"id": e["id"], "model": cfg["second_opinion"], **response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            done[e["id"]] = record

    with ThreadPoolExecutor(cfg["concurrency"]) as pool:
        list(pool.map(ask, [e for e in sample if e["id"] not in done]))

    answers = [done[e["id"]]["content"] for e in sample]
    with RegexRunner() as runner:
        grades = grade(runner, sample, answers)
    result = score(grades)
    cost = sum(done[e["id"]]["usage"].get("cost", 0) for e in sample)
    print(f"{cfg['second_opinion']} sur {result['n']} exemples gardés ({cost:.3f} $)")
    print(f"  accord (tous les tests passent) {result['success_rate']:.1%} | tests réussis {result['mean_pass_rate']:.1%}")
    print(f"  mal formées {result['malformed_rate']:.1%} | timeouts {result['timeout_rate']:.1%}")

    disagreements = [(e, a, g) for e, a, g in zip(sample, answers, grades) if not g["success"] and g["status"] == "ok"]
    for e, answer, g in disagreements[: args.show]:
        print(f"\n[{e['id']} {e['category']}/{e['difficulty']}] {e['description']}")
        print(f"  professeur : {e['regex']}\n  second avis : {answer.strip()}")
        print(f"  valides refusées : {g['failed_match'][:4]}\n  invalides acceptées : {g['failed_no_match'][:4]}")


if __name__ == "__main__":
    main()
