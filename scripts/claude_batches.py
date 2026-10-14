"""Complément de données écrit par des agents Claude plutôt que par le professeur OpenRouter (étape 9 ter).

Même découpage que generate.py, avec les mêmes prompts, mais les appels au professeur sont remplacés par des agents qui
lisent des fichiers et en écrivent d'autres :

1. `plan` : lots déterministes (préfixe c, catégories ciblées) et prompts « descriptions », en 4 parts
   (`desc-shard<k>.json`, `descriptions-system.txt`). Chaque agent écrit `desc-out<k>.json` : {lot: [items]}.
2. `tests` : prompts « tests », construits à partir des seules descriptions (`test-shard<k>.json`,
   `tests-system.txt`). D'autres agents, qui ne voient pas les regex, écrivent `test-out<k>.json`.
3. `assemble` : journaux au format de generate.py dans data/raw-claude/, à passer à
   `validate.py --extra-raw-dir data/raw-claude`.

    uv run scripts/claude_batches.py plan --work-dir <dossier>
    uv run scripts/claude_batches.py tests --work-dir <dossier>
    uv run scripts/claude_batches.py assemble --work-dir <dossier>
"""

import argparse
import json
from pathlib import Path

from generate import build_descriptions_messages, build_tests_messages, load_config, plan

CATEGORIES = "logs,logs,logs,multiline,multiline,multiline,dev-tooling,files-paths,exclusions,text"
OUT = Path("data/raw-claude")
SHARDS = 4


def read_outputs(work: Path, name: str) -> dict:
    out = {}
    for k in range(SHARDS):
        out.update(json.loads((work / f"{name}{k}.json").read_text()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "tests", "assemble"])
    parser.add_argument("--work-dir", type=Path, required=True, help="dossier d'échange avec les agents")
    parser.add_argument("--batches", type=int, default=120)
    parser.add_argument("--categories", default=CATEGORIES)
    parser.add_argument("--model", default="claude-opus-5-5", help="noté dans les journaux")
    args = parser.parse_args()

    cfg, taxonomy = load_config()
    batches = plan(args.batches, cfg, taxonomy, args.categories.split(","), prefix="c")
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    if args.command == "plan":
        (work / "descriptions-system.txt").write_text(build_descriptions_messages(batches[0], taxonomy)[0]["content"])
        for k in range(SHARDS):
            shard = [{"batch": b["batch"], "user": build_descriptions_messages(b, taxonomy)[1]["content"]} for b in batches[k::SHARDS]]
            (work / f"desc-shard{k}.json").write_text(json.dumps(shard, indent=1))
        print(f"{len(batches)} lots en {SHARDS} parts dans {work}")

    elif args.command == "tests":
        desc = read_outputs(work, "desc-out")
        (work / "tests-system.txt").write_text(build_tests_messages(next(iter(desc.values())), cfg)[0]["content"])
        ids = sorted(desc)
        for k in range(SHARDS):
            shard = [{"batch": b, "user": build_tests_messages(desc[b], cfg)[1]["content"]} for b in ids[k::SHARDS]]
            (work / f"test-shard{k}.json").write_text(json.dumps(shard, indent=1))
        print(f"{len(ids)} lots, {sum(map(len, desc.values()))} besoins")

    else:
        desc, tests = read_outputs(work, "desc-out"), read_outputs(work, "test-out")
        OUT.mkdir(parents=True, exist_ok=True)
        common = {"model": args.model, "reasoning": False, "content": None, "usage": {}}
        with (OUT / "descriptions.jsonl").open("w") as fd, (OUT / "tests.jsonl").open("w") as ft:
            for b in batches:
                if b["batch"] in desc:
                    fd.write(json.dumps({**b, **common, "items": desc[b["batch"]]}) + "\n")
                if b["batch"] in tests:
                    ft.write(json.dumps({"batch": b["batch"], **common, "items": tests[b["batch"]]}) + "\n")
        print(f"{len(desc)} lots de descriptions, {len(tests)} lots de tests dans {OUT}")


if __name__ == "__main__":
    main()
