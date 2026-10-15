"""Benchmark (étape 4) : cas écrits à la main par Claude, un fichier YAML par catégorie dans `benchmark/`.

    uv run scripts/benchmark.py check    # contrôles mécaniques, à relancer après chaque modification
    uv run scripts/benchmark.py build    # cas `status: ok` → data/eval/benchmark.jsonl, pour eval.py

Un autre benchmark, dans son propre dossier (même format, mêmes contrôles) :

    uv run scripts/benchmark.py check --dir benchmark-gpt
    uv run scripts/benchmark.py build --dir benchmark-gpt --output data/eval/benchmark-gpt.jsonl

`benchmark/<catégorie>.yaml` est la source de vérité, versionnée. Chaque cas : id, difficulty, status (`ok` gardé,
`todo` à revoir, `drop` écarté), description, regex de référence, chaînes `match` et `no_match`, et `pitfall` (ce
qu'une solution naïve rate).

Contrôles de `check` :
  - erreurs (bloquent `build`) : littéral invalide, référence qui rate un test, pas assez de tests, chaîne à la fois
    valide et invalide, id dupliqué ;
  - avertissements : tests peu discriminants (la référence sans ancres, aux quantificateurs relâchés ou aux classes
    élargies passe aussi), description proche d'une description du train, regex identique à une regex du train.
"""

import argparse
import json
from pathlib import Path

import yaml

from common import RegexRunner, parse_literal
from generate import CONFIG, TAXONOMY
from validate import (
    BENCHMARK_DIR,
    PREPARED_DIR,
    SIMILARITY_THRESHOLD,
    add_examples,
    load_benchmark,
    mutants,
    similarity,
    unanchored,
    words,
)

OUTPUT = Path("data/eval/benchmark.jsonl")


# --- Contrôles --------------------------------------------------------------------------------------------------------


def check(cases: list[dict], cfg: dict) -> dict[str, dict[str, list[str]]]:
    """Renvoie, par id, les erreurs et les avertissements de chaque cas (hors `drop`)."""
    required = cfg["tests_required"]
    train = [json.loads(line) for split in ("train", "valid") for line in (PREPARED_DIR / f"{split}.jsonl").open()]
    train_words = [(e["description"], words(e["description"])) for e in train]
    train_regexes = {e["regex"] for e in train}

    issues = {}
    ids = [c["id"] for c in cases]
    jobs, owners = [], []
    for c in cases:
        if c.get("status") == "drop":
            continue
        errors, warnings = [], []
        if ids.count(c["id"]) > 1:
            errors.append("id dupliqué")
        if len(c["match"]) < required["match"] + 1 or len(c["no_match"]) < required["no_match"] + 1:
            # +1 : un cas sur trois montre un exemple dans le prompt, retiré des tests cachés.
            errors.append(f"pas assez de tests (au moins {required['match'] + 1} de chaque côté)")
        if both := set(c["match"]) & set(c["no_match"]):
            errors.append(f"chaînes à la fois valides et invalides : {sorted(both)}")
        if c["regex"] in train_regexes:
            warnings.append("regex identique à une regex du train")
        score, nearest = max((similarity(words(c["description"]), w), d) for d, w in train_words)
        if score >= SIMILARITY_THRESHOLD:
            warnings.append(f"description proche du train ({score:.2f}) : {nearest[:150]}")
        parsed = parse_literal(c["regex"])
        if parsed is None:
            errors.append("regex hors format ES2018")
        else:
            pattern, flags = parsed
            tests = {"match": c["match"], "no_match": c["no_match"]}
            variants = [("reference", pattern)]
            if (stripped := unanchored(pattern)) is not None:
                variants.append(("référence sans ancres", stripped))
            variants += list(mutants(pattern).items())
            for kind, variant in variants:
                jobs.append({"pattern": variant, "flags": flags, **tests})
                owners.append((c, kind))
        issues[c["id"]] = {"errors": errors, "warnings": warnings}

    with RegexRunner() as runner:
        results = runner.run(jobs)
    for (c, kind), r in zip(owners, results):
        passed = "error" not in r and not r.get("timeout") and all(r["match"]) and all(r["no_match"])
        if kind == "reference":
            if "error" in r:
                issues[c["id"]]["errors"].append(f"regex invalide : {r['error']}")
            elif r.get("timeout"):
                issues[c["id"]]["errors"].append("timeout (ReDoS)")
            elif not passed:
                refused = [s for s, ok in zip(c["match"], r["match"]) if not ok]
                accepted = [s for s, ok in zip(c["no_match"], r["no_match"]) if not ok]
                issues[c["id"]]["errors"].append(f"référence en désaccord : valides refusées {refused}, invalides acceptées {accepted}")
        elif passed:
            issues[c["id"]]["warnings"].append(f"tests peu discriminants : {kind} passe aussi")
    return issues


# --- Gel --------------------------------------------------------------------------------------------------------------


def build(cases: list[dict], cfg: dict, share: float, output: Path = OUTPUT) -> None:
    issues = check(cases, cfg)
    kept = [c for c in cases if c.get("status") == "ok"]
    blocked = [c["id"] for c in kept if issues[c["id"]]["errors"]]
    if blocked:
        raise SystemExit(f"cas `ok` avec des erreurs, à corriger d'abord : {blocked}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for c in kept:
            example = {
                "id": c["id"],
                "category": c["category"],
                "difficulty": c["difficulty"],
                "description": c["description"],
                "regex": c["regex"],
                "tests": {"match": c["match"], "no_match": c["no_match"]},
            }
            f.write(json.dumps(add_examples(example, cfg, share), ensure_ascii=False) + "\n")
    todo = sum(c.get("status") == "todo" for c in cases)
    print(f"{len(kept)} cas écrits dans {output} ({todo} encore à relire, {len(cases) - len(kept) - todo} écartés)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "build"])
    parser.add_argument("--all", action="store_true", help="check : affiche aussi les cas sans problème détecté")
    parser.add_argument("--dir", type=Path, default=BENCHMARK_DIR, help="dossier des YAML du benchmark")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="build : JSONL écrit pour eval.py")
    args = parser.parse_args()
    cfg, taxonomy = yaml.safe_load(CONFIG.read_text()), yaml.safe_load(TAXONOMY.read_text())

    if args.command == "build":
        build(load_benchmark(args.dir), cfg, taxonomy["examples_share"], args.output)
    else:
        cases = load_benchmark(args.dir)
        issues = check(cases, cfg)
        for c in cases:
            found = issues.get(c["id"])
            if found and (found["errors"] or found["warnings"] or args.all):
                print(f"\n{c['id']} [{c.get('status')}]")
                for e in found["errors"]:
                    print(f"  ERREUR  {e}")
                for w in found["warnings"]:
                    print(f"  attention  {w}")
        n_errors = sum(bool(i["errors"]) for i in issues.values())
        n_warnings = sum(bool(i["warnings"]) and not i["errors"] for i in issues.values())
        status = {s: sum(c.get("status") == s for c in cases) for s in ("todo", "ok", "drop")}
        print(f"\n{len(issues)} cas contrôlés : {n_errors} avec erreurs, {n_warnings} avec avertissements seulement")
        print("statuts : " + ", ".join(f"{k} {v}" for k, v in status.items()))


if __name__ == "__main__":
    main()
