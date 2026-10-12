"""Briques partagées par toutes les étapes : format du prompt, parsing de la réponse, exécution des regex et métriques.

Le format du prompt est défini ici une seule fois : préparation SFT, évaluation, génération des paires DPO et reward
GRPO l'importent. Changer ce fichier implique de ré-entraîner.
"""

import json
import re
import subprocess
import threading
from pathlib import Path

RUNNER = Path(__file__).with_name("regex_runner.mjs")

# Flags ES2018. Node 22 accepte aussi `d` (ES2022) et `v` (ES2024), hors périmètre : on les refuse ici.
# Côté syntaxe, Node 22 refuse déjà les ajouts plus récents (modifiers inline, noms de groupes dupliqués).
ALLOWED_FLAGS = set("gimsuy")
LINE_TERMINATORS = re.compile("[\n\r  ]")

SYSTEM_PROMPT = (
    "You write JavaScript regular expressions (ECMAScript 2018). The regex will validate strings with "
    "RegExp.prototype.test: it must accept every valid string and reject every other one. Given a description, and "
    "sometimes example strings, answer with a single regex literal such as /^\\d+$/i, with no explanation."
)


# Exemples du mode few-shot (baselines des modèles bruts) : un cas ancré avec exemples, une recherche non ancrée avec
# flag, une liste. Choisis pour ne recouper ni le benchmark ni le train.
FEW_SHOT = [
    {
        "description": "French license plate: two uppercase letters, a hyphen, three digits, a hyphen, two uppercase letters",
        "examples": {"match": ["AB-123-CD"], "no_match": ["AB123CD"]},
        "regex": r"/^[A-Z]{2}-\d{3}-[A-Z]{2}$/",
    },
    {
        "description": "does the commit message contain the word wip, in any case, as a whole word",
        "regex": r"/\bwip\b/i",
    },
    {
        "description": "A semicolon-separated list of one to five two-letter uppercase country codes, with no spaces.",
        "regex": r"/^[A-Z]{2}(?:;[A-Z]{2}){0,4}$/",
    },
]


# --- Prompt -----------------------------------------------------------------------------------------------------------


def build_user_prompt(example: dict) -> str:
    # Chaînes en JSON : espaces de début et de fin, caractères invisibles et Unicode restent sans ambiguïté.
    sections = [f"Description: {example['description']}"]
    shown = example.get("examples") or {}
    if shown.get("match"):
        sections.append("Should match:\n" + "\n".join(f"- {json.dumps(s, ensure_ascii=False)}" for s in shown["match"]))
    if shown.get("no_match"):
        sections.append("Should not match:\n" + "\n".join(f"- {json.dumps(s, ensure_ascii=False)}" for s in shown["no_match"]))
    return "\n\n".join(sections)


def build_messages(example: dict, with_answer: bool = False, few_shot: bool = False) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in FEW_SHOT if few_shot else []:
        messages += [{"role": "user", "content": build_user_prompt(shot)}, {"role": "assistant", "content": shot["regex"]}]
    messages.append({"role": "user", "content": build_user_prompt(example)})
    if with_answer:
        messages.append({"role": "assistant", "content": example["regex"]})
    return messages


# --- Parsing ----------------------------------------------------------------------------------------------------------


def parse_literal(text: str) -> tuple[str, str] | None:
    """`/pattern/flags` → (pattern, flags). None si le texte n'est pas exactement un littéral regex JS ES2018.

    La syntaxe du pattern n'est pas vérifiée ici : c'est `new RegExp` qui le fait, côté Node.
    """
    if len(text) < 3 or text[0] != "/" or LINE_TERMINATORS.search(text):
        return None
    i, in_class = 1, False
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:  # dans une classe, `/` n'a pas besoin d'être échappé : /[/]/ est valide
            break
        i += 1
    else:
        return None
    pattern, flags = text[1:i], text[i + 1 :]
    if not pattern or not set(flags) <= ALLOWED_FLAGS or len(set(flags)) != len(flags):
        return None
    return pattern, flags


def parse_answer(text: str) -> tuple[str, str] | None:
    """Extrait le littéral d'une réponse du modèle. None si la réponse est mal formée.

    On tolère un bloc de code, des backticks ou des guillemets autour du littéral, mais pas de texte : une réponse
    bavarde est mal formée, même si elle contient une regex (« Here is the regex: /a/ » n'est pas collable dans du code).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if fence := re.fullmatch(r"```[\w-]*\n(.*?)\n?```", text, flags=re.DOTALL):
        text = fence.group(1).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "`\"'":
        text = text[1:-1].strip()
    return parse_literal(text)


# --- Exécution --------------------------------------------------------------------------------------------------------


class RegexRunner:
    """Client du worker Node `regex_runner.mjs`, à utiliser comme context manager."""

    def __init__(self, timeout_ms: int = 1000):
        self.timeout_ms = timeout_ms
        self.proc = subprocess.Popen(
            ["node", str(RUNNER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8"
        )

    def run(self, jobs: list[dict]) -> list[dict]:
        """Exécute des jobs `{"pattern", "flags", "match", "no_match"}` et renvoie les réponses du worker, dans l'ordre."""

        # Écriture dans un thread : sinon, si le worker remplit stdout pendant qu'on remplit stdin, les deux bloquent.
        def write() -> None:
            for i, job in enumerate(jobs):
                self.proc.stdin.write(json.dumps({"id": i, "timeout_ms": self.timeout_ms, **job}) + "\n")
            self.proc.stdin.flush()

        writer = threading.Thread(target=write)
        writer.start()
        results = [None] * len(jobs)
        for _ in jobs:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("le worker Node s'est arrêté")
            result = json.loads(line)
            results[result.pop("id")] = result
        writer.join()
        return results

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self) -> "RegexRunner":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# --- Métriques --------------------------------------------------------------------------------------------------------


def grade(runner: RegexRunner, examples: list[dict], answers: list[str]) -> list[dict]:
    """Exécute chaque réponse sur les tests cachés de son exemple.

    Statut `ok`, `malformed` (pas un littéral, flag hors ES2018, ou `new RegExp` qui lève une erreur) ou `timeout`.
    Une réponse mal formée ou en timeout a tous ses tests comptés comme ratés.
    """
    parsed = [parse_answer(a) for a in answers]
    jobs = [
        {"pattern": p[0], "flags": p[1], "match": e["tests"]["match"], "no_match": e["tests"]["no_match"]}
        for e, p in zip(examples, parsed)
        if p
    ]
    results = iter(runner.run(jobs))
    grades = []
    for e, p in zip(examples, parsed):
        result = next(results) if p else {"error": "not a single ES2018 regex literal"}
        n_match, n_no_match = len(e["tests"]["match"]), len(e["tests"]["no_match"])
        if "error" in result or result.get("timeout"):
            passed_match = passed_no_match = 0
        else:
            passed_match, passed_no_match = sum(result["match"]), sum(result["no_match"])
        grades.append(
            {
                "status": "timeout" if result.get("timeout") else "malformed" if "error" in result else "ok",
                "error": result.get("error"),
                "match_rate": passed_match / n_match,
                "no_match_rate": passed_no_match / n_no_match,
                "pass_rate": (passed_match + passed_no_match) / (n_match + n_no_match),
                "success": passed_match == n_match and passed_no_match == n_no_match,
                # Tests ratés, pour l'analyse d'erreurs.
                "failed_match": [s for s, ok in zip(e["tests"]["match"], result.get("match", [])) if not ok],
                "failed_no_match": [s for s, ok in zip(e["tests"]["no_match"], result.get("no_match", [])) if not ok],
            }
        )
    return grades


def score(grades: list[dict]) -> dict:
    """Agrège des notes de `grade` :

    - success_rate : part des cas où tous les tests passent (métrique principale) ;
    - mean_pass_rate : taux moyen de tests réussis ;
    - malformed_rate, timeout_rate : part des réponses mal formées ou en timeout.
    """
    n = len(grades)
    return {
        "n": n,
        "success_rate": sum(g["success"] for g in grades) / n,
        "mean_pass_rate": sum(g["pass_rate"] for g in grades) / n,
        "malformed_rate": sum(g["status"] == "malformed" for g in grades) / n,
        "timeout_rate": sum(g["status"] == "timeout" for g in grades) / n,
    }
