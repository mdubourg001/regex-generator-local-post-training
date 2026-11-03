"""Complément de train à partir de la bibliothèque communautaire de regex101 (dataset innovatorved/regex_dataset).

Les entrées sont des besoins réels, mais inutilisables telles quelles : une description sur deux est vide, les regex
sont souvent fausses ou en flavor PCRE, et `text` est un bloc de chaînes d'essai non étiquetées. On ne garde que le
besoin : un agent GPT (prompt dans `configs/regex101-prompt.md`) réécrit la description, une regex ES2018 et des tests.
Le benchmark est écrit par Claude : faire annoter par une autre famille de modèles évite d'apprendre son style.

    uv run scripts/regex101.py select     # source → parts à annoter (data/regex101/work/shard<k>.json)
    uv run scripts/regex101.py check      # contrôle des annotations out<k>.json, à relancer par l'agent
    uv run scripts/regex101.py split      # ~400 cas `ok` → évaluation réelle (benchmark-regex101/), le reste en réserve
    uv run scripts/regex101.py assemble   # réserve → journaux au format de generate.py

L'évaluation réelle se contrôle et se gèle comme les benchmarks (relire les cas, passer en `drop` ceux à écarter) :

    uv run scripts/benchmark.py check --dir benchmark-regex101
    uv run scripts/benchmark.py build --dir benchmark-regex101 --output data/eval/benchmark-regex101.jsonl

La réserve ne va au train que si l'évaluation réelle s'écarte des benchmarks LLM (README, bilan). Comme les autres
compléments, tous ses lots vont au train :

    uv run scripts/validate.py --extra-raw-dir data/raw-regex101 --prepared-dir data/prepared-regex101 \\
        --sft-dir data/sft-regex101
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

from common import RegexRunner, grade
from generate import CONFIG
from validate import check as guardrails
from validate import SIMILARITY_THRESHOLD, decontaminate, is_anchored, load_benchmark, similarity, words

SOURCE_URL = "https://raw.githubusercontent.com/innovatorved/regex_dataset/refs/heads/main/data.json"
DIR = Path("data/regex101")
WORK = DIR / "work"
RAW_OUT = Path("data/raw-regex101")
EVAL_DIR = Path("benchmark-regex101")
# Le train le plus large à ce jour : l'évaluation réelle ne doit recouper aucun de ses besoins.
TRAIN = [Path("data/prepared-plus2/train.jsonl"), Path("data/prepared-plus2/valid.jsonl")]
# `text` peut faire des dizaines de Ko (logs, pages HTML) : un extrait suffit à montrer le contexte.
MAX_TEXT_CHARS = 800


def select(source: Path, shard_size: int) -> None:
    """Garde les besoins de validation (regex d'origine ancrée), sans doublon de regex, en parts de `shard_size`."""
    entries = json.loads(source.read_text())
    seen, kept = set(), []
    for i, e in enumerate(entries):
        regex = (e.get("regex") or "").strip()
        if not regex or regex in seen or not (e.get("title") or "").strip() or not is_anchored(regex):
            continue
        seen.add(regex)
        kept.append(
            {
                "id": f"x{i:05d}",
                "title": e["title"].strip(),
                "description": (e.get("description") or "").strip(),
                "original_regex": regex,
                "text": (e.get("text") or "")[:MAX_TEXT_CHARS],
            }
        )
    WORK.mkdir(parents=True, exist_ok=True)
    shards = [kept[k : k + shard_size] for k in range(0, len(kept), shard_size)]
    for k, shard in enumerate(shards):
        (WORK / f"shard{k}.json").write_text(json.dumps(shard, indent=1, ensure_ascii=False))
    print(f"{len(entries)} entrées, {len(kept)} gardées, en {len(shards)} parts de {shard_size} dans {WORK}")


def read_annotations() -> dict[str, dict]:
    out = {}
    for path in sorted(WORK.glob("out*.json"), key=lambda p: int(p.stem[3:])):
        out |= json.loads(path.read_text())
    return out


def titles() -> dict[str, str]:
    return {e["id"]: e["title"] for path in WORK.glob("shard*.json") for e in json.loads(path.read_text())}


def as_example(id_: str, a: dict, title: str = "") -> dict:
    """Annotation → exemple au format de `validate.candidates`."""
    strings = lambda key: list(dict.fromkeys(s for s in a.get(key) or [] if isinstance(s, str)))  # noqa: E731
    near_miss = strings("near_miss")
    return {
        "id": f"{id_}-0",
        "batch": id_,
        "category": a.get("category"),
        "difficulty": a.get("difficulty"),
        "register": "regex101",
        "intent": title,
        "description": str(a.get("description", "")).strip(),
        "regex": str(a.get("regex", "")).strip(),
        "tests": {"match": strings("match"), "no_match": list(dict.fromkeys(near_miss + strings("other_no_match")))},
        "near_miss": near_miss,
    }


def check(shards: list[int] | None) -> None:
    """Applique à chaque annotation `ok` les garde-fous de validate.py, et donne le motif de rejet."""
    cfg = yaml.safe_load(CONFIG.read_text())
    paths = [WORK / f"out{k}.json" for k in shards] if shards else sorted(WORK.glob("out*.json"))
    annotations = {i: a for p in paths for i, a in json.loads(p.read_text()).items()}
    benchmark = load_benchmark()
    counts = {"ok": 0, "rejet": 0, "drop": 0}
    with RegexRunner() as runner:
        for id_, a in annotations.items():
            if a.get("status") == "drop":
                counts["drop"] += 1
                continue
            e = as_example(id_, a)
            problems = []
            if both := set(e["tests"]["match"]) & set(e["tests"]["no_match"]):
                problems.append(f"chaînes à la fois valides et invalides : {sorted(both)}")
            kept, rejected = guardrails([e], cfg, runner)
            if rejected:
                reason = next(iter(rejected))
                if reason == "référence en désaccord avec les tests":
                    g = grade(runner, [e], [e["regex"]])[0]
                    reason += f" : valides refusées {g['failed_match']}, invalides acceptées {g['failed_no_match']}"
                problems.append(reason)
            elif not decontaminate(kept, benchmark)[0]:
                # Sans montrer le cas du benchmark : l'agent ne doit pas le voir.
                problems.append("recoupe le benchmark : changer substantiellement le besoin, ou drop")
            counts["rejet" if problems else "ok"] += 1
            for p in problems:
                print(f"{id_}  {p}")
    print(f"\n{len(annotations)} annotations : " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def split(size: int, seed: int) -> None:
    """Tire `size` annotations `ok`, stratifiées par catégorie et difficulté, pour l'évaluation réelle.

    Les candidates proches d'un besoin du train actuel (même regex, ou description proche) restent en réserve : une
    évaluation réelle contaminée ne mesurerait plus la généralisation.
    """
    annotations = {i: a for i, a in read_annotations().items() if a.get("status") == "ok"}
    train = [json.loads(line) for path in TRAIN for line in path.open()]
    train_regexes = {e["regex"] for e in train}
    train_words = [words(e["description"]) for e in train]
    clean = {
        i: a for i, a in annotations.items()
        if a["regex"] not in train_regexes
        and not any(similarity(words(a["description"]), w) >= SIMILARITY_THRESHOLD for w in train_words)
    }
    strata = defaultdict(list)
    for i in sorted(clean):
        strata[(clean[i]["category"], clean[i]["difficulty"])].append(i)
    # Allocation proportionnelle, arrondie au plus fort reste.
    quotas = {k: size * len(v) / len(clean) for k, v in strata.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - counts[k], reverse=True)[: size - sum(counts.values())]:
        counts[k] += 1
    rng = random.Random(seed)
    chosen = sorted(i for k, ids in strata.items() for i in rng.sample(ids, counts[k]))

    EVAL_DIR.mkdir(exist_ok=True)
    by_category = defaultdict(list)
    for i in chosen:
        a = annotations[i]
        by_category[a["category"]].append(
            {
                "id": f"real-{i}",
                "difficulty": a["difficulty"],
                "status": "ok",
                "description": a["description"],
                "regex": a["regex"],
                "match": a["match"],
                "no_match": list(dict.fromkeys(a["near_miss"] + a["other_no_match"])),
            }
        )
    for category, cases in sorted(by_category.items()):
        header = f"# Évaluation réelle (regex101, annotée par GPT 6.1 Sol), catégorie {category}. Voir scripts/regex101.py.\n"
        body = yaml.safe_dump(cases, sort_keys=False, allow_unicode=False, width=10**6, default_flow_style=False)
        (EVAL_DIR / f"{category}.yaml").write_text(header + body)
    print(f"{len(annotations)} annotations `ok`, {len(annotations) - len(clean)} proches du train actuel (gardées en réserve)")
    print(f"{len(chosen)} cas dans {EVAL_DIR}, {len(annotations) - len(chosen)} en réserve pour le train")


def assemble() -> None:
    """Réserve (annotations `ok` hors évaluation réelle, et sans besoin proche d'elle) → journaux de generate.py."""
    annotations, title = read_annotations(), titles()
    held = load_benchmark(EVAL_DIR)
    held_ids = {c["id"].removeprefix("real-") for c in held}
    reserve = [as_example(i, a) for i, a in annotations.items() if a.get("status") == "ok" and i not in held_ids]
    kept, n_close = decontaminate(reserve, held)
    kept_ids = {e["batch"] for e in kept}
    annotations = {i: a for i, a in annotations.items() if i in kept_ids}
    print(f"{len(held_ids)} cas d'évaluation réelle exclus, {n_close} besoins proches d'eux écartés")
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    common = {"model": "openai/gpt-sol-6.1", "reasoning": True, "content": None, "usage": {}}
    n = 0
    with (RAW_OUT / "descriptions.jsonl").open("w") as fd, (RAW_OUT / "tests.jsonl").open("w") as ft:
        for id_, a in sorted(annotations.items()):
            if a.get("status") != "ok":
                continue
            e = as_example(id_, a, title.get(id_, ""))
            slot = {"id": e["id"], "intent": e["intent"], "difficulty": e["difficulty"], "register": e["register"]}
            item = {"slot": 1, "description": e["description"], "regex": e["regex"]}
            tests = {"slot": 1, **{k: a.get(k) or [] for k in ("match", "near_miss", "other_no_match")}}
            fd.write(json.dumps({"batch": id_, "category": e["category"], "slots": [slot], **common, "items": [item]}) + "\n")
            ft.write(json.dumps({"batch": id_, **common, "items": [tests]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} besoins en réserve dans {RAW_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["select", "check", "split", "assemble"])
    parser.add_argument("--source", type=Path, default=DIR / "source.json", help=f"select : copie locale de {SOURCE_URL}")
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--shards", type=int, nargs="*", help="check : numéros des parts (toutes par défaut)")
    parser.add_argument("--eval-size", type=int, default=400, help="split : taille de l'évaluation réelle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "select":
        select(args.source, args.shard_size)
    elif args.command == "check":
        check(args.shards)
    elif args.command == "split":
        split(args.eval_size, args.seed)
    else:
        assemble()


if __name__ == "__main__":
    main()
