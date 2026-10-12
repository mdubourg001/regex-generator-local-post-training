"""Baseline sans modèle : quelques règles écrites à la main (préfixe conventional commits, branches, fichiers touchés).

C'est le score à battre. Les deux équipes écrivent leurs titres en conventional commits (`feat(...)`, `fix(...)`) : une
partie des labels se lit directement dans le titre. Le modèle n'a d'intérêt que s'il fait mieux que ces règles,
en particulier sur les labels qu'elles ne savent pas trancher (`featurefix` contre `bugfix`, `technical task`…).

    uv run scripts/baseline_rules.py               # sur data/prepared/test.jsonl
    uv run scripts/baseline_rules.py --split valid
"""

import argparse
import json
import re
from pathlib import Path

from common import score

PREPARED_DIR = Path("data/prepared")
RESULTS_DIR = Path("results")

# `[TICKET-1] feat(scope)!: …`, `Chore(deps): …`, `chore:(IPERIA-1): …` → `feat`, `chore`…
PREFIX_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([a-z]+)\s*[(:!]", re.IGNORECASE)

# Type conventional commits → label, par projet. `fix` donne `featurefix`, le cas le plus fréquent sur projet-a :
# une règle ne peut pas savoir si le bug est en production.
PREFIX_TO_LABELS = {
    "iperia__iperia-front": {
        "feat": ["feature"],
        "fix": ["featurefix"],
        "hotfix": ["hotfix"],
        "refactor": ["refactor"],
        "chore": ["chore"],
        "test": ["test"],
        "style": ["styles"],
    },
    "rgds__rgds-website-factory": {
        "feat": ["feature"],
        "fix": ["fix"],
        "hotfix": ["hotfix"],
        "refactor": ["refactor"],
        "chore": ["chore"],
        "test": ["test"],
        "ci": ["ci"],
        "build": ["build"],
    },
}


def predict(example: dict) -> list[str]:
    project = example["project"]
    title, source, target = example["title"], example["source_branch"], example["target_branch"]
    paths = [f["path"] for f in example["files"]]
    labels = set()

    if match := PREFIX_RE.match(title):
        labels.update(PREFIX_TO_LABELS[project].get(match.group(1).lower(), []))
    if source.startswith("hotfix/"):
        labels.add("hotfix")

    if project == "iperia__iperia-front":
        if not re.search(r"IPERIA-\d+", title + " " + source, re.IGNORECASE):
            labels.add("NOJIRA")
        if any(p.endswith((".scss", ".css")) for p in paths):
            labels.add("styles")

    if project == "rgds__rgds-website-factory":
        if source.startswith("release/") and target in ("master", "main", "develop"):
            labels = {"release"}
        if re.search(r"\b(bump|update|upgrade)\b.*\bfrom\b.*\bto\b", title, re.IGNORECASE) or "(deps" in title.lower():
            labels.add("dependencies")
            if any(p.startswith("composer.") for p in paths):
                labels.add("php")
            if any(p.startswith("package") for p in paths):
                labels.add("javascript")
        if any(p.startswith(".gitlab-ci") or p.startswith(".gitlab/") for p in paths):
            labels.add("ci")

    return sorted(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    examples = [json.loads(line) for line in (PREPARED_DIR / f"{args.split}.jsonl").open()]
    records = [{"id": e["id"], "project": e["project"], "labels": e["labels"], "predicted": predict(e)} for e in examples]
    result = score(records)

    print(f"rules sur {args.split} ({result['n']} MR)")
    print(f"  F1 micro {result['f1_micro']:.3f} | F1 macro {result['f1_macro']:.3f} | exact match {result['exact_match']:.3f}")
    print(f"\n  {'label':28} {'F1':>5} {'préc.':>6} {'rappel':>6} {'support':>7}")
    for name, m in result["per_label"].items():
        print(f"  {name:28} {m['f1']:>5.2f} {m['precision']:>6.2f} {m['recall']:>6.2f} {m['support']:>7}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / f"rules-{args.split}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"name": f"rules-{args.split}", **{k: v for k, v in result.items() if k != "per_label"}}
    with (RESULTS_DIR / "scores.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
