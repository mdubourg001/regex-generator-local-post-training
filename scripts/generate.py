"""Génère les données synthétiques avec le modèle professeur (OpenRouter).

Pour chaque lot (une catégorie, `batch_size` besoins) :
  1. appel « descriptions » : le professeur invente des besoins concrets à partir des graines de la taxonomie, et écrit
     pour chacun une description (dans le registre demandé) et une regex de référence ;
  2. appel « tests » : à partir des seules descriptions (sans voir les regex, pour que les tests n'héritent pas de
     leurs bugs), il écrit des chaînes valides, des négatifs « presque bons » et d'autres négatifs.

Le plan est déterministe (le lot n a toujours la même catégorie et les mêmes graines) et chaque réponse est
journalisée : une relance ne refait pas les appels déjà payés, et augmenter --batches ne fait qu'ajouter des lots.

    uv run scripts/generate.py --batches 20      # pilote : 100 besoins
    uv run scripts/generate.py --batches 4000    # volume
    uv run scripts/generate.py --batches 700 --prefix t --categories logs,logs,multiline   # complément ciblé

Sorties (ignorées par git) :
    data/raw/descriptions.jsonl  un appel « descriptions » par ligne (réponse brute, besoins parsés, coût)
    data/raw/tests.jsonl         un appel « tests » par ligne
"""

import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from openrouter import OpenRouter, parse_json

CONFIG = Path("configs/generation.yaml")
TAXONOMY = Path("configs/taxonomy.yaml")
RAW_DIR = Path("data/raw")

DESCRIPTIONS_PROMPT = """\
You create training data for a model that writes JavaScript regular expressions (ECMAScript 2018) from \
natural-language descriptions.

For each slot, invent one concrete string validation need and write:
- "description": the need as a developer would express it, in the requested register. It must state every rule \
needed to decide whether a string is valid, except conventions any developer would assume for that format. Never \
mention regex syntax and never give example strings.
- "regex": a correct JavaScript regex literal /pattern/flags implementing the description.

Rules for the regex:
- It is used as regex.test(string) on the whole input. Anchor it with ^ and $, unless the need is explicitly about \
strings that merely contain something.
- ECMAScript 2018 only: flags among i, m, s, u. No g, y, d or v flags, no inline modifiers.
- Format only: never validate checksums (Luhn, IBAN keys…) or calendar facts such as leap years. When a format has \
such a checksum, the description says that only the format is checked.
- Prefer readable patterns, and avoid nested quantifiers that can backtrack catastrophically.

Use the seed intent as inspiration only: choose a specific variant (country, bounds, separators, case rules, optional \
parts…) so that the needs of a batch are clearly different from each other.

Difficulty levels:
{difficulties}

Registers:
{registers}

Answer with JSON only: {{"items": [{{"slot": 1, "description": "...", "regex": "/.../"}}, ...]}}"""

TESTS_PROMPT = """\
You write test cases for string validators. You only see the description of what is valid, never an implementation.

For each description, write:
- "match": {match} distinct strings that are valid according to the description, covering its variations and edge \
cases (bounds, optional parts, allowed alternatives…).
- "near_miss": {near_miss} distinct strings that are invalid but almost valid: one character too many or too few, a \
value just out of bounds, a wrong separator, the wrong case, extra leading or trailing text or whitespace…
- "other_no_match": {other_no_match} distinct strings that are clearly invalid.

Rules:
- Only include strings whose validity follows unambiguously from the description and the usual conventions of the \
format. Skip debatable cases.
- The whole string is validated, unless the description is explicitly about strings that contain something.
- Format only: no checksum validation and no calendar facts such as leap years, unless the description asks for it.
- Strings are tested exactly as written. Use JSON escapes (\\n, \\t, \\u00e9…) for special characters, and write \
every string out in full: no code such as "a".repeat(10) or "a" * 10.

Answer with JSON only: {{"items": [{{"slot": 1, "match": [...], "near_miss": [...], "other_no_match": [...]}}, ...]}}"""


def load_config() -> tuple[dict, dict]:
    return yaml.safe_load(CONFIG.read_text()), yaml.safe_load(TAXONOMY.read_text())


def plan(n_batches: int, cfg: dict, taxonomy: dict, categories: list[str] | None = None, prefix: str = "b") -> list[dict]:
    """Lots déterministes : le lot b dépend seulement de b et de la graine, pas du nombre total de lots.

    `categories` restreint le plan à ces catégories (dans cet ordre, répétitions comprises : une catégorie listée deux
    fois reçoit deux fois plus de lots), pour un complément ciblé. Il faut alors un autre `prefix`, qui distingue ses
    lots de ceux du plan principal.
    """
    by_id = {c["id"]: c for c in taxonomy["categories"]}
    cycle = [by_id[c] for c in categories] if categories else taxonomy["categories"]
    shares = taxonomy["difficulties"]["share"]
    batches = []
    for b in range(n_batches):
        rng = random.Random(f"{cfg['seed']}-{b}" if prefix == "b" else f"{cfg['seed']}-{prefix}-{b}")
        category = cycle[b % len(cycle)]
        slots = [
            {
                "id": f"{prefix}{b:05d}-{j}",
                "intent": rng.choice(category["intents"]),
                "difficulty": rng.choices(list(shares), weights=list(shares.values()))[0],
                "register": rng.choice(list(taxonomy["registers"])),
            }
            for j in range(cfg["batch_size"])
        ]
        batches.append({"batch": f"{prefix}{b:05d}", "category": category["id"], "slots": slots})
    return batches


def build_descriptions_messages(batch: dict, taxonomy: dict) -> list[dict]:
    levels = taxonomy["difficulties"]
    system = DESCRIPTIONS_PROMPT.format(
        difficulties="\n".join(f"- {name}: {levels[name]}" for name in levels["share"]),
        registers="\n".join(f"- {name}: {text}" for name, text in taxonomy["registers"].items()),
    )
    category = next(c for c in taxonomy["categories"] if c["id"] == batch["category"])
    slots = "\n".join(
        f"{j + 1}. seed intent: {s['intent']} | difficulty: {s['difficulty']} | register: {s['register']}"
        for j, s in enumerate(batch["slots"])
    )
    user = f"Category: {category['id']} ({category['description']})\n\nSlots:\n{slots}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_tests_messages(items: list[dict], cfg: dict) -> list[dict]:
    system = TESTS_PROMPT.format(**cfg["tests_requested"])
    user = "\n".join(f"{item['slot']}. {item['description']}" for item in items)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def expand_repeats(content: str) -> str:
    """Réécrit en entier les chaînes que le professeur écrit parfois comme du code : "a".repeat(30) ou "a" * 30."""

    def expand(m: re.Match) -> str:
        count = int(m[2] or m[3])
        return json.dumps(json.loads(f'"{m[1]}"', strict=False) * count) if count <= 1000 else m[0]

    return re.sub(r'"((?:[^"\\]|\\.)*)"\s*(?:\.repeat\((\d+)\)|\*\s*(\d+))', expand, content)


def parse_items(content: str, keys: list[str]) -> list[dict] | None:
    data = parse_json(expand_repeats(content))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    items = [i for i in data["items"] if isinstance(i, dict) and isinstance(i.get("slot"), int) and all(k in i for k in keys)]
    return items or None


def log_paths(raw_dir: Path) -> tuple[Path, Path]:
    return raw_dir / "descriptions.jsonl", raw_dir / "tests.jsonl"


def read_log(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["batch"]: r for r in map(json.loads, path.open())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, required=True, help="nombre total de lots du plan")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="dossier des journaux (pour une expérience à part)")
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], help="active le raisonnement du professeur")
    parser.add_argument("--categories", help="complément ciblé : catégories séparées par des virgules (répétitions = poids)")
    parser.add_argument("--prefix", default="b", help="préfixe des lots, différent de b pour un complément ciblé")
    args = parser.parse_args()

    cfg, taxonomy = load_config()
    cfg["reasoning"] = args.reasoning or cfg["reasoning"]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    descriptions_log, tests_log = log_paths(args.raw_dir)
    descriptions, tests = read_log(descriptions_log), read_log(tests_log)
    spent = sum(r["usage"].get("cost", 0) for r in [*descriptions.values(), *tests.values()])
    categories = args.categories.split(",") if args.categories else None
    if categories and args.prefix == "b":
        parser.error("un complément ciblé (--categories) demande un autre --prefix que b")
    todo = [b for b in plan(args.batches, cfg, taxonomy, categories, args.prefix) if b["batch"] not in tests]
    print(f"{len(todo)} lots à générer, {spent:.3f} $ déjà dépensés (plafond {cfg['budget_usd']} $)", flush=True)

    client = OpenRouter()
    lock = threading.Lock()
    state = {"spent": spent, "done": 0, "failed": 0}

    def call(messages: list[dict], temperature: float) -> dict | None:
        with lock:
            if state["spent"] >= cfg["budget_usd"]:
                return None
        response = client.chat(
            cfg["teacher"], messages, reasoning=cfg["reasoning"], temperature=temperature, max_tokens=cfg["max_tokens"]
        )
        with lock:
            state["spent"] += response["usage"].get("cost", 0)
        return response

    def log(path: Path, record: dict) -> None:
        with lock, path.open("a") as f:
            # Échappement ASCII : les tests peuvent contenir des demi-caractères UTF-16 isolés, non encodables en UTF-8.
            f.write(json.dumps(record) + "\n")

    def run(batch: dict) -> None:
        record = descriptions.get(batch["batch"])
        if record is None:
            response = call(build_descriptions_messages(batch, taxonomy), cfg["temperature_descriptions"])
            if response is None:
                return
            items = parse_items(response["content"], ["description", "regex"])
            record = {**batch, "model": cfg["teacher"], "reasoning": cfg["reasoning"], **response, "items": items}
            log(descriptions_log, record)
        if not record["items"]:
            return  # réponse illisible : comptée comme rejet par validate.py, pas relancée
        response = call(build_tests_messages(record["items"], cfg), cfg["temperature_tests"])
        if response is None:
            return
        items = parse_items(response["content"], ["match", "near_miss", "other_no_match"])
        record = {"batch": batch["batch"], "model": cfg["teacher"], "reasoning": cfg["reasoning"], **response, "items": items}
        log(tests_log, record)
        with lock:
            state["done"] += 1

    with ThreadPoolExecutor(cfg["concurrency"]) as pool:
        futures = [pool.submit(run, b) for b in todo]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as e:  # noqa: BLE001 — un lot en échec est retenté à la prochaine exécution
                state["failed"] += 1
                print(f"  échec : {e}", flush=True)
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} lots traités, {state['spent']:.3f} $", flush=True)

    stopped = state["spent"] >= cfg["budget_usd"]
    print(
        f"{state['done']} lots générés, {state['failed']} en échec, {state['spent']:.3f} $ dépensés au total"
        + (" — plafond atteint, génération arrêtée" if stopped else "")
    )


if __name__ == "__main__":
    main()
