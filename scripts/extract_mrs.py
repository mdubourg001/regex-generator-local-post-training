"""Extrait les merge requests de projets GitLab via la CLI `glab`.

Pour chaque projet, on sauvegarde la liste de ses labels (avec descriptions) et, pour chaque MR, l'objet brut renvoyé
par l'API complété de ses diffs, de ses commits et de l'historique de ses labels (qui a posé quel label, et quand).
On garde tout en brut : le filtrage et la mise en forme du prompt se font plus tard, sans re-solliciter l'API.

    uv run scripts/extract_mrs.py groupe/projet-a groupe/projet-b
    uv run scripts/extract_mrs.py groupe/projet-a --since 2024-01-01 --limit 50   # test rapide

Sortie : data/raw/gitlab/<groupe__projet>/{project.json, labels.json, mrs/<iid>.json}.
Le script est reprenable : une MR déjà extraite et non modifiée depuis n'est pas re-téléchargée.

Prérequis : `glab auth login --hostname gitlab.ekino.com`.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlencode

DEFAULT_HOST = "gitlab.ekino.com"
OUT_DIR = Path("data/raw/gitlab")
PER_PAGE = 100

# Garde-fous de taille : certaines MR (lockfiles, code généré, migrations massives) font des dizaines de Mo de diff.
# Le prompt n'en verra qu'une petite partie, inutile de tout stocker.
MAX_FILES = 300  # fichiers de diff récupérés par MR
MAX_DIFF_CHARS = 20_000  # caractères gardés par fichier (le reste est tronqué et marqué `truncated`)
MAX_COMMITS = 100


class GlabError(RuntimeError):
    pass


def api(host: str, endpoint: str, retries: int = 4):
    """Appelle `glab api` et renvoie le JSON. Réessaie sur les erreurs transitoires (429, 5xx, réseau)."""
    for attempt in range(retries):
        proc = subprocess.run(
            ["glab", "api", "--hostname", host, endpoint], capture_output=True, text=True
        )
        if proc.returncode == 0:
            return json.loads(proc.stdout) if proc.stdout.strip() else None
        err = proc.stderr.strip()
        transient = any(code in err for code in ("429", "500", "502", "503", "504", "timeout", "connection"))
        if not transient or attempt == retries - 1:
            raise GlabError(f"{endpoint}: {err}")
        time.sleep(2**attempt)


def api_list(host: str, endpoint: str, params: dict | None = None, max_items: int | None = None) -> list:
    """Parcourt une ressource paginée page par page, en s'arrêtant dès qu'on a `max_items` éléments."""
    items, page = [], 1
    while True:
        query = urlencode({**(params or {}), "per_page": PER_PAGE, "page": page})
        batch = api(host, f"{endpoint}?{query}") or []
        items.extend(batch)
        if len(batch) < PER_PAGE or (max_items is not None and len(items) >= max_items):
            return items[:max_items] if max_items is not None else items
        page += 1


def fetch_diffs(host: str, mr_endpoint: str) -> tuple[list, bool]:
    diffs = api_list(host, f"{mr_endpoint}/diffs", max_items=MAX_FILES + 1)
    files_truncated = len(diffs) > MAX_FILES
    diffs = diffs[:MAX_FILES]
    for d in diffs:
        diff = d.get("diff") or ""
        if len(diff) > MAX_DIFF_CHARS:
            d["diff"] = diff[:MAX_DIFF_CHARS]
            d["truncated"] = True
    return diffs, files_truncated


def fetch_mr(host: str, project_endpoint: str, mr: dict, dest: Path) -> str:
    """Complète une MR (diffs, commits, événements de labels) et l'écrit dans `dest`. Renvoie un statut."""
    if dest.exists():
        cached = json.loads(dest.read_text())
        if cached.get("updated_at") == mr.get("updated_at"):
            return "cached"

    mr_endpoint = f"{project_endpoint}/merge_requests/{mr['iid']}"
    mr["diffs"], mr["diffs_files_truncated"] = fetch_diffs(host, mr_endpoint)
    mr["commits"] = api_list(host, f"{mr_endpoint}/commits", max_items=MAX_COMMITS)
    mr["label_events"] = api_list(host, f"{mr_endpoint}/resource_label_events")

    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(mr, ensure_ascii=False))
    tmp.rename(dest)  # écriture atomique : un Ctrl-C ne laisse pas de fichier à moitié écrit
    return "fetched"


def extract_project(host: str, path: str, args) -> None:
    project_endpoint = f"projects/{quote(path, safe='')}"
    out = args.out / path.replace("/", "__")
    (out / "mrs").mkdir(parents=True, exist_ok=True)

    project = api(host, project_endpoint)
    (out / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2))

    # Labels du projet ET des groupes parents : c'est la liste « autorisée » qu'on donnera au modèle.
    labels = api_list(host, f"{project_endpoint}/labels", {"include_ancestor_groups": "true"})
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))

    params = {"state": args.state, "order_by": "created_at", "sort": "desc"}
    if args.since:
        params["created_after"] = args.since
    mrs = api_list(host, f"{project_endpoint}/merge_requests", params, max_items=args.limit)
    print(f"{path}: {len(labels)} labels, {len(mrs)} MR ({args.state})", flush=True)

    counts = {"fetched": 0, "cached": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_mr, host, project_endpoint, mr, out / "mrs" / f"{mr['iid']}.json"): mr for mr in mrs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                counts[future.result()] += 1
            except GlabError as e:
                counts["error"] += 1
                print(f"  ! !{futures[future]['iid']}: {e}", flush=True)
            if i % 50 == 0 or i == len(mrs):
                print(f"  {i}/{len(mrs)} {counts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("projects", nargs="+", help="chemins complets des projets (groupe/sous-groupe/projet)")
    parser.add_argument("--hostname", default=DEFAULT_HOST)
    parser.add_argument("--state", default="merged", choices=["merged", "closed", "opened", "all"])
    parser.add_argument("--since", help="ne garder que les MR créées après cette date (ISO 8601)")
    parser.add_argument("--limit", type=int, help="nombre max de MR par projet (les plus récentes)")
    parser.add_argument("--workers", type=int, default=4, help="MR traitées en parallèle")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    for path in args.projects:
        try:
            extract_project(args.hostname, path, args)
        except GlabError as e:
            print(f"{path}: échec ({e})", flush=True)


if __name__ == "__main__":
    main()
