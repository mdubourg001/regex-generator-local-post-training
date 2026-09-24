"""Briques partagées : format de prompt, extraction et exécution du SQL.

Le format de prompt doit rester identique entre l'éval, le SFT et le RL : sinon on mesure un changement de prompt, pas
l'effet de l'entraînement.
"""

import re
import sqlite3
import time
from collections import Counter
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPIDER_DB_DIR = ROOT / "data" / "spider" / "database"

SYSTEM_PROMPT = "Tu écris uniquement une requête SQLite, sans explication ni balise markdown."


def spider_db_path(db_id: str) -> Path:
    return SPIDER_DB_DIR / db_id / f"{db_id}.sqlite"


def schema_of(db_path: Path) -> str:
    """Schéma de la base sous forme de CREATE TABLE, tel qu'on le donne au modèle."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    return "\n".join(r[0] for r in rows)


def build_messages(schema: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schéma :\n{schema}\n\nQuestion : {question}"},
    ]


def build_prompt(tokenizer, schema: str, question: str) -> list[int]:
    # enable_thinking=False : désactive le mode « raisonnement » de Qwen3 (ignoré par les modèles qui ne l'ont pas)
    return tokenizer.apply_chat_template(
        build_messages(schema, question), add_generation_prompt=True, enable_thinking=False
    )


def extract_sql(text: str) -> str:
    """Récupère la requête dans la sortie du modèle, même s'il n'a pas respecté la consigne."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    # Une seule instruction : on coupe au premier « ; »
    return text.strip().split(";")[0].strip()


def execute(db_path: Path, sql: str, timeout_s: float = 5.0) -> list[tuple]:
    """Exécute la requête en lecture seule, avec un timeout (le modèle peut générer des produits cartésiens énormes)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    # Certaines bases Spider contiennent du texte non-UTF-8
    conn.text_factory = lambda b: b.decode(errors="replace")
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _normalize(row: tuple) -> tuple:
    return tuple(round(v, 4) if isinstance(v, float) else v for v in row)


def _rows_equal(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    return pred == gold if ordered else Counter(pred) == Counter(gold)


def results_match(pred: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    """Compare deux résultats, comme l'éval officielle de Spider :
    - l'ordre des lignes ne compte que si la requête de référence a un ORDER BY ;
    - l'ordre des colonnes ne compte pas (`SELECT a, b` ≡ `SELECT b, a`).
    """
    pred, gold = [_normalize(r) for r in pred], [_normalize(r) for r in gold]
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    ncols = len(gold[0])
    if len(pred[0]) != ncols:
        return False
    if ncols > 6:  # au-delà, trop de permutations ; on compare telles quelles
        return _rows_equal(pred, gold, ordered)
    # Pour chaque ordre possible des colonnes prédites, on teste si les lignes correspondent
    return any(
        _rows_equal([tuple(r[i] for i in perm) for r in pred], gold, ordered)
        for perm in permutations(range(ncols))
    )
