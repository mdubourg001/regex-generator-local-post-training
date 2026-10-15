"""Compléments de données écrits par des agents Claude plutôt que par le professeur OpenRouter (étapes 9 ter et 9 quater).

Même découpage que generate.py, avec les mêmes prompts, mais les appels au professeur sont remplacés par des agents qui
lisent des fichiers et en écrivent d'autres :

1. `plan` : lots déterministes et prompts « descriptions », en `--shards` parts (`desc-shard<k>.json`,
   `descriptions-system.txt`). Chaque agent écrit `desc-out<k>.json` : {lot: [items]}.
2. `tests` : prompts « tests », construits à partir des seules descriptions (`test-shard<k>.json`,
   `tests-system.txt`). D'autres agents, qui ne voient pas les regex, écrivent `test-out<k>.json`. Avec
   `--second-opinion`, les mêmes descriptions servent aussi de prompts de second avis (`opinion-system.txt`) : encore
   d'autres agents écrivent leur propre regex dans `opinion-out<k>.json`.
3. `assemble` : journaux au format de generate.py dans `--raw-out`, plus `second_opinion.json` ({id: regex}) s'il y a
   des seconds avis. À passer à `validate.py --extra-raw-dir`.

Étape 9 ter (catégories faibles) : les options par défaut. Étape 9 quater (besoins difficiles, toutes catégories,
constructions imposées, second avis) :

    uv run scripts/claude_batches.py plan --work-dir <dossier> --prefix d --batches 200 --shards 8 --all-categories \\
        --difficulty hard=0.6,medium=0.4 --constructs 0.6
    uv run scripts/claude_batches.py tests --work-dir <dossier> --shards 8 --second-opinion
    uv run scripts/claude_batches.py assemble --work-dir <dossier> --prefix d --batches 200 --shards 8 --all-categories \\
        --difficulty hard=0.6,medium=0.4 --constructs 0.6 --raw-out data/raw-claude-hard
"""

import argparse
import json
import random
from pathlib import Path

from generate import build_descriptions_messages, build_tests_messages, load_config, plan

CATEGORIES = "logs,logs,logs,multiline,multiline,multiline,dev-tooling,files-paths,exclusions,text"

# Constructions rares, imposées à une part des besoins : la consigne générale (« privilégier… ») a peu porté à
# l'étape 9 ter. Chacune est décrite comme dans la taxonomie.
CONSTRUCTS = {
    "backreference": "a capturing group and a backreference such as \\1 (the need must require repeating an earlier part)",
    "named_groups": "a named group (?<name>...) and a named backreference \\k<name>",
    "lookbehind": "a lookbehind (?<=...) or negative lookbehind (?<!...)",
    "lookahead": "a lookahead (?=...) or negative lookahead (?!...) to combine or exclude constraints",
    "flag_m": "the m flag, with line-based rules on multi-line input",
    "flag_s": "the s flag, with a . that must cross line breaks",
    "unicode_property": "Unicode property escapes \\p{...} with the u flag",
    "unanchored": "a deliberately unanchored search: the string only has to contain the pattern",
    "numeric_range": "a numeric range expressed with alternations (for example 0-255 or 1-31)",
    "word_boundary": "word boundaries \\b",
}

OPINION_PROMPT = """\
You write JavaScript regular expressions (ECMAScript 2018). Each regex will validate strings with \
RegExp.prototype.test: it must accept every valid string and reject every other one. For each numbered description, \
write a single regex literal such as /^\\d+$/i.

Answer with JSON only: {"items": [{"slot": 1, "regex": "/.../"}, ...]}"""


def build_plan(args, cfg: dict, taxonomy: dict) -> list[dict]:
    categories = None if args.all_categories else args.categories.split(",")
    batches = plan(args.batches, cfg, taxonomy, categories, args.prefix)
    if args.difficulty:
        shares = {k: float(v) for k, v in (kv.split("=") for kv in args.difficulty.split(","))}
    for b in batches:
        for s in b["slots"]:
            rng = random.Random(f"claude-{s['id']}")
            if args.difficulty:
                s["difficulty"] = rng.choices(list(shares), weights=list(shares.values()))[0]
            if rng.random() < args.constructs:
                s["construct"] = rng.choice(list(CONSTRUCTS))
    return batches


def descriptions_messages(batch: dict, taxonomy: dict) -> list[dict]:
    """Prompt de generate.py, plus la construction imposée au bout de la ligne des besoins concernés."""
    system, user = build_descriptions_messages(batch, taxonomy)
    lines = user["content"].split("\n")
    for j, s in enumerate(batch["slots"]):
        if "construct" in s:
            i = lines.index(next(line for line in lines if line.startswith(f"{j + 1}. seed intent:")))
            lines[i] += f" | the natural solution must use: {CONSTRUCTS[s['construct']]}"
    return [system, {"role": "user", "content": "\n".join(lines)}]


def read_outputs(work: Path, name: str, shards: int) -> dict:
    out = {}
    for k in range(shards):
        out.update(json.loads((work / f"{name}{k}.json").read_text()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "tests", "assemble"])
    parser.add_argument("--work-dir", type=Path, required=True, help="dossier d'échange avec les agents")
    parser.add_argument("--batches", type=int, default=120)
    parser.add_argument("--shards", type=int, default=4, help="un agent par part")
    parser.add_argument("--prefix", default="c", help="préfixe des lots, distinct du plan principal (b)")
    parser.add_argument("--categories", default=CATEGORIES, help="catégories du cycle (répétitions = poids)")
    parser.add_argument("--all-categories", action="store_true", help="toutes les catégories de la taxonomie")
    parser.add_argument("--difficulty", help="parts imposées, ex. hard=0.6,medium=0.4 (sinon celles de la taxonomie)")
    parser.add_argument("--constructs", type=float, default=0.0, help="part des besoins avec une construction imposée")
    parser.add_argument("--second-opinion", action="store_true", help="tests : prépare aussi les prompts de second avis")
    parser.add_argument("--raw-out", type=Path, default=Path("data/raw-claude"))
    parser.add_argument("--model", default="claude-opus-5-5", help="noté dans les journaux")
    args = parser.parse_args()

    cfg, taxonomy = load_config()
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    if args.command == "plan":
        batches = build_plan(args, cfg, taxonomy)
        (work / "plan.json").write_text(json.dumps(batches, indent=1))
        (work / "descriptions-system.txt").write_text(descriptions_messages(batches[0], taxonomy)[0]["content"])
        for k in range(args.shards):
            shard = [{"batch": b["batch"], "user": descriptions_messages(b, taxonomy)[1]["content"]} for b in batches[k :: args.shards]]
            (work / f"desc-shard{k}.json").write_text(json.dumps(shard, indent=1))
        print(f"{len(batches)} lots en {args.shards} parts dans {work}")

    elif args.command == "tests":
        desc = read_outputs(work, "desc-out", args.shards)
        (work / "tests-system.txt").write_text(build_tests_messages(next(iter(desc.values())), cfg)[0]["content"])
        if args.second_opinion:
            (work / "opinion-system.txt").write_text(OPINION_PROMPT)
        ids = sorted(desc)
        for k in range(args.shards):
            shard = [{"batch": b, "user": build_tests_messages(desc[b], cfg)[1]["content"]} for b in ids[k :: args.shards]]
            (work / f"test-shard{k}.json").write_text(json.dumps(shard, indent=1))
        print(f"{len(ids)} lots, {sum(map(len, desc.values()))} besoins")

    else:
        batches = json.loads((work / "plan.json").read_text()) if (work / "plan.json").exists() else build_plan(args, cfg, taxonomy)
        desc, tests = read_outputs(work, "desc-out", args.shards), read_outputs(work, "test-out", args.shards)
        args.raw_out.mkdir(parents=True, exist_ok=True)
        common = {"model": args.model, "reasoning": False, "content": None, "usage": {}}
        with (args.raw_out / "descriptions.jsonl").open("w") as fd, (args.raw_out / "tests.jsonl").open("w") as ft:
            for b in batches:
                if b["batch"] in desc:
                    fd.write(json.dumps({**b, **common, "items": desc[b["batch"]]}) + "\n")
                if b["batch"] in tests:
                    ft.write(json.dumps({"batch": b["batch"], **common, "items": tests[b["batch"]]}) + "\n")
        print(f"{len(desc)} lots de descriptions, {len(tests)} lots de tests dans {args.raw_out}")
        if (work / "opinion-out0.json").exists():
            opinions = read_outputs(work, "opinion-out", args.shards)
            by_id = {f"{b}-{item['slot'] - 1}": item["regex"] for b, items in opinions.items() for item in items
                     if isinstance(item, dict) and isinstance(item.get("slot"), int) and isinstance(item.get("regex"), str)}
            (args.raw_out / "second_opinion.json").write_text(json.dumps(by_id, indent=1))
            print(f"{len(by_id)} seconds avis")


if __name__ == "__main__":
    main()
