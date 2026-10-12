"""Briques partagées par toutes les étapes : filtrage des MR, format du prompt, parsing de la réponse et métriques.

Le format du prompt est défini ici une seule fois : préparation SFT, évaluation, génération des paires DPO et reward
GRPO l'importent. Changer ce fichier implique de ré-entraîner.
"""

import json
import re
from collections import Counter
from pathlib import Path

import yaml

RAW_DIR = Path("data/raw/gitlab")
LABELS_CONFIG = Path("configs/labels.yaml")

# Comptes techniques : bots de projet/groupe, compte du runner CI (qui lance dependabot/renovate).
# `ghost` (utilisateur supprimé) n'en fait pas partie : c'est autant un ancien développeur qu'un ancien compte de bot.
BOT_RE = re.compile(r"(^|[_\-.])bot($|[_\-.\d])|^project_\d+_bot|^group_\d+_bot|^gitlab.?ci$|renovate|dependabot", re.IGNORECASE)
# MR de mise à jour automatique, reconnues à leur branche quel que soit l'auteur (ancien compte devenu `ghost`).
AUTOMATED_BRANCH_RE = re.compile(r"^(dependabot|renovate)/")

# --- Budget du prompt -------------------------------------------------------------------------------------------------
# Un diff complet ne tient pas dans le contexte (médiane ~13k caractères sur projet-a). On garde un extrait.
MAX_DESCRIPTION_CHARS = 1000
MAX_COMMITS = 5
MAX_FILES = 30
MAX_DIFF_CHARS = 3000
MAX_DIFF_CHARS_PER_FILE = 800
# Fichiers listés mais jamais montrés dans l'extrait de diff : volumineux et peu informatifs.
NOISY_FILE_RE = re.compile(r"(\.lock|-lock\.json|lock\.yaml|\.snap|\.min\.(js|css)|\.svg|\.map)$")

SYSTEM_PROMPT = (
    "You label GitLab merge requests. Given the project's labels and a merge request, choose every label that applies, "
    'using only labels from the list. Answer with JSON only, in the form {"labels": ["label", ...]}.'
)


# --- Données ----------------------------------------------------------------------------------------------------------


def load_config() -> dict:
    return yaml.safe_load(LABELS_CONFIG.read_text())["projects"]


def is_bot(username: str | None) -> bool:
    return bool(username and BOT_RE.search(username))


def diff_line_counts(diff: str) -> tuple[int, int]:
    lines = diff.splitlines()
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    return added, removed


def to_example(mr: dict, project: str, alias: str, allowed: dict) -> dict:
    """Réduit une MR brute aux champs utiles au prompt, avec ses labels cibles (restreints au référentiel)."""
    files = []
    for d in mr["diffs"]:
        added, removed = diff_line_counts(d.get("diff") or "")
        status = "new" if d["new_file"] else "deleted" if d["deleted_file"] else "renamed" if d["renamed_file"] else ""
        files.append({"path": d["new_path"], "added": added, "removed": removed, "status": status, "diff": d.get("diff") or ""})
    commits = [c["title"] for c in mr["commits"] if not c["title"].startswith("Merge ")]
    return {
        "id": f"{alias}!{mr['iid']}",
        "project": project,
        "created_at": mr["created_at"],
        "title": mr["title"],
        "description": mr["description"] or "",
        "source_branch": mr["source_branch"],
        "target_branch": mr["target_branch"],
        "commits": commits,
        "files": files,
        "files_truncated": mr.get("diffs_files_truncated", False),
        "labels": sorted(label for label in mr["labels"] if label in allowed),
    }


def load_examples(project: str, cfg: dict, stats: Counter | None = None) -> list[dict]:
    """Charge les MR d'un projet et applique les filtres. `stats` compte les rejets par motif."""
    stats = stats if stats is not None else Counter()
    examples = []
    for path in sorted((RAW_DIR / project / "mrs").glob("*.json")):
        mr = json.loads(path.read_text())
        stats["total"] += 1
        if mr["state"] != "merged":
            stats["non mergée"] += 1
        elif is_bot(mr["author"]["username"]) or AUTOMATED_BRANCH_RE.match(mr["source_branch"]):
            stats["MR automatique (dependabot, renovate…)"] += 1
        elif mr["created_at"] < cfg["since"]:
            stats["avant la date de début de labellisation"] += 1
        elif not mr["diffs"]:
            stats["diff vide"] += 1
        else:
            example = to_example(mr, project, cfg["alias"], cfg["labels"])
            if not example["labels"]:
                # Sans label, impossible de distinguer « aucun ne s'applique » de « personne n'a labellisé ».
                stats["aucun label du référentiel"] += 1
            else:
                stats["gardée"] += 1
                examples.append(example)
    return examples


# --- Prompt -----------------------------------------------------------------------------------------------------------


def clean_description(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # commentaires des templates de MR
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_DESCRIPTION_CHARS] + (" […]" if len(text) > MAX_DESCRIPTION_CHARS else "")


def diff_excerpt(files: list[dict]) -> str:
    """Début du diff de chaque fichier, dans l'ordre, jusqu'à épuisement du budget."""
    parts, budget = [], MAX_DIFF_CHARS
    for f in files:
        if budget <= 0:
            break
        if NOISY_FILE_RE.search(f["path"]) or not f["diff"]:
            continue
        chunk = f["diff"][: min(MAX_DIFF_CHARS_PER_FILE, budget)]
        parts.append(f"--- {f['path']}\n{chunk.rstrip()}")
        budget -= len(chunk)
    return "\n".join(parts)


def build_user_prompt(example: dict, labels: dict) -> str:
    # Nom entre guillemets, comme dans la réponse attendue : avec `- nom: description`, Qwen3-4B recopiait
    # « feature: Nouvelle fonctionnalité. » comme nom de label (8 % de labels hors liste).
    label_list = "\n".join(f'- "{name}": {description}' for name, description in labels.items())
    files = example["files"]
    file_lines = [
        f"{f['path']} (+{f['added']} -{f['removed']}{', ' + f['status'] if f['status'] else ''})" for f in files[:MAX_FILES]
    ]
    if len(files) > MAX_FILES or example["files_truncated"]:
        file_lines.append(f"… and {max(len(files) - MAX_FILES, 0)}+ more files")
    commits = [c for c in dict.fromkeys(example["commits"]) if c != example["title"]][:MAX_COMMITS]

    sections = [
        f"## Labels\n{label_list}",
        "## Merge request",
        f"Title: {example['title']}",
        f"Branches: {example['source_branch']} → {example['target_branch']}",
    ]
    if description := clean_description(example["description"]):
        sections.append(f"Description:\n{description}")
    if commits:
        sections.append("Commits:\n" + "\n".join(f"- {c}" for c in commits))
    sections.append(f"Files changed ({len(files)}{'+' if example['files_truncated'] else ''}):\n" + "\n".join(file_lines))
    if excerpt := diff_excerpt(files):
        sections.append(f"Diff (excerpt):\n```diff\n{excerpt}\n```")
    return "\n\n".join(sections)


def build_messages(example: dict, labels: dict, with_answer: bool = False) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example, labels)},
    ]
    if with_answer:
        messages.append({"role": "assistant", "content": format_answer(example["labels"], labels)})
    return messages


def format_answer(predicted: list[str], labels: dict) -> str:
    # Ordre du référentiel : une seule façon d'écrire la bonne réponse, donc une cible SFT sans ambiguïté.
    return json.dumps({"labels": [name for name in labels if name in predicted]}, ensure_ascii=False)


def parse_answer(text: str) -> list[str] | None:
    """Extrait la liste de labels d'une réponse. Renvoie None si la réponse est mal formée."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        labels = json.loads(match.group(0)).get("labels")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return None
    return list(dict.fromkeys(labels))


# --- Métriques --------------------------------------------------------------------------------------------------------


def f1(predicted: set, gold: set) -> float:
    if not predicted and not gold:
        return 1.0
    tp = len(predicted & gold)
    return 2 * tp / (len(predicted) + len(gold))


def score(records: list[dict]) -> dict:
    """Agrège des prédictions `{"project", "labels" (gold), "predicted" (liste ou None si mal formée)}`.

    - f1_micro : sur l'ensemble des décisions, dominé par les labels fréquents ;
    - f1_macro : moyenne des F1 par (projet, label), chaque label compte autant ;
    - exact_match : part des MR dont l'ensemble prédit est exactement le bon ;
    - invalid_label_rate : part des labels prédits absents du référentiel ;
    - malformed_rate : part des réponses impossibles à parser.
    """
    config = load_config()
    tp, fp, fn = Counter(), Counter(), Counter()
    exact = malformed = invalid = n_predicted = 0
    for r in records:
        allowed = config[r["project"]]["labels"]
        if r["predicted"] is None:
            malformed += 1
        predicted = set(r["predicted"] or [])
        n_predicted += len(predicted)
        invalid += len(predicted - set(allowed))
        gold = set(r["labels"])
        exact += predicted == gold
        for label in allowed:
            key = (r["project"], label)
            tp[key] += label in predicted and label in gold
            fp[key] += label in predicted and label not in gold
            fn[key] += label not in predicted and label in gold

    per_label = {}
    for key in tp.keys() | fn.keys():
        support = tp[key] + fn[key]
        if support:
            per_label[key] = {
                "f1": 2 * tp[key] / (2 * tp[key] + fp[key] + fn[key]),
                "precision": tp[key] / (tp[key] + fp[key]) if tp[key] + fp[key] else 0.0,
                "recall": tp[key] / support,
                "support": support,
            }
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    n = len(records)
    return {
        "n": n,
        "f1_micro": 2 * total_tp / (2 * total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn else 1.0,
        "f1_macro": sum(v["f1"] for v in per_label.values()) / len(per_label) if per_label else 0.0,
        "exact_match": exact / n,
        "invalid_label_rate": invalid / n_predicted if n_predicted else 0.0,
        "malformed_rate": malformed / n,
        "per_label": {f"{config[p]['alias']}/{label}": v for (p, label), v in sorted(per_label.items())},
    }
