"""Statistiques sur les labels des MR extraites, pour décider quels projets et quels labels sont exploitables.

    uv run scripts/label_stats.py                 # tous les projets de data/raw/gitlab
    uv run scripts/label_stats.py --top 40

Pour chaque projet :
- couverture : part des MR qui ont au moins un label (un projet peu labellisé donne surtout des faux négatifs) ;
- fréquence de chaque label, et labels de la liste jamais utilisés ;
- provenance : part des poses faites par un bot (règle automatique, donc à apprendre telle quelle ou à exclure) ;
- timing : part des poses faites après le merge (souvent du process, pas du contenu : `workflow::`, `deployed`…).
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from common import RAW_DIR, is_bot


def parse_date(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def project_stats(project_dir: Path, top: int) -> None:
    labels = json.loads((project_dir / "labels.json").read_text())
    mrs = [json.loads(f.read_text()) for f in sorted((project_dir / "mrs").glob("*.json"))]
    if not mrs:
        return

    usage = Counter(label for mr in mrs for label in mr["labels"])
    added = Counter()
    by_bot = Counter()
    after_merge = Counter()
    for mr in mrs:
        merged_at = parse_date(mr.get("merged_at"))
        for event in mr.get("label_events", []):
            if event["action"] != "add" or not event.get("label"):
                continue
            name = event["label"]["name"]
            added[name] += 1
            if is_bot((event.get("user") or {}).get("username")):
                by_bot[name] += 1
            if merged_at and parse_date(event["created_at"]) > merged_at:
                after_merge[name] += 1

    labelled = sum(1 for mr in mrs if mr["labels"])
    per_mr = Counter(len(mr["labels"]) for mr in mrs)
    print(f"\n## {project_dir.name}")
    print(f"MR : {len(mrs)} | avec ≥ 1 label : {labelled} ({labelled / len(mrs):.0%}) | labels disponibles : {len(labels)}")
    print("Labels par MR : " + ", ".join(f"{k}→{v}" for k, v in sorted(per_mr.items())))

    print(f"\n{'label':40} {'MR':>6} {'%MR':>5} {'bot':>5} {'post-merge':>10}")
    for name, n in usage.most_common(top):
        bot = by_bot[name] / added[name] if added[name] else 0
        late = after_merge[name] / added[name] if added[name] else 0
        print(f"{name[:40]:40} {n:>6} {n / len(mrs):>5.0%} {bot:>5.0%} {late:>10.0%}")

    unused = sorted(label["name"] for label in labels if label["name"] not in usage)
    if unused:
        print(f"\nJamais utilisés ({len(unused)}) : " + ", ".join(unused))
    unknown = sorted(set(usage) - {label["name"] for label in labels})
    if unknown:
        print(f"Posés mais absents de la liste actuelle (supprimés/renommés ?) : " + ", ".join(unknown))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--top", type=int, default=25, help="nombre de labels affichés par projet")
    args = parser.parse_args()

    for project_dir in sorted(p for p in args.raw.iterdir() if (p / "labels.json").exists()):
        project_stats(project_dir, args.top)


if __name__ == "__main__":
    main()
