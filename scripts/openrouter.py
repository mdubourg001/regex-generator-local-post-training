"""Client minimal de l'API OpenRouter (compatible OpenAI), avec le coût de chaque appel.

La clé est lue dans la variable d'environnement OPENROUTER_API_KEY, ou à défaut dans le fichier `.env` (ignoré par git).
"""

import json
import os
import re
import time
from pathlib import Path

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_FILE = Path(".env")


def api_key() -> str:
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip("\"'")
    raise SystemExit("OPENROUTER_API_KEY absente : l'exporter, ou l'écrire dans .env (OPENROUTER_API_KEY=sk-or-…)")


class OpenRouter:
    def __init__(self, deadline_s: float = 120):
        # Pendant la génération, OpenRouter envoie des espaces pour garder la connexion ouverte : le délai de lecture de
        # httpx ne se déclenche alors jamais. On impose donc aussi un délai total par appel.
        self.deadline_s = deadline_s
        self.client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key()}"})

    def post(self, body: dict) -> tuple[int, bytes]:
        start, chunks = time.monotonic(), []
        with self.client.stream("POST", API_URL, json=body) as response:
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                if time.monotonic() - start > self.deadline_s:
                    raise TimeoutError(f"pas de réponse complète après {self.deadline_s:.0f} s")
        return response.status_code, b"".join(chunks)

    def chat(self, model: str, messages: list[dict], reasoning: str | bool = False, retries: int = 4, **params) -> dict:
        """Renvoie `{"content", "usage"}`. `usage` contient les tokens et le coût réel (`cost`, en $).

        `reasoning` : False pour le désactiver, ou un niveau d'effort (`low`, `medium`, `high`).
        """
        reasoning_param = {"effort": reasoning} if reasoning else {"enabled": False}
        body = {"model": model, "messages": messages, "reasoning": reasoning_param, **params}
        error = None
        for attempt in range(retries + 1):
            if attempt:
                time.sleep(2**attempt)
            try:
                status, content = self.post(body)
            except (httpx.TransportError, TimeoutError) as e:
                error = e
                continue
            # Erreurs passagères : on réessaie. Les autres (clé invalide, requête mal formée) sont définitives.
            if status in (408, 429) or status >= 500:
                error = RuntimeError(f"HTTP {status}: {content[:300]!r}")
                continue
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {content[:300]!r}")
            try:
                # strict=False : certains fournisseurs laissent des caractères de contrôle bruts dans les chaînes.
                data = json.loads(content, strict=False)
            except json.JSONDecodeError as e:  # réponse tronquée ou corrompue : passagère
                error = e
                continue
            if not data.get("choices"):  # erreur du fournisseur renvoyée avec un statut 200
                error = RuntimeError(str(data.get("error", data))[:300])
                continue
            return {"content": data["choices"][0]["message"].get("content") or "", "usage": data.get("usage", {})}
        raise error


def parse_json(text: str):
    """Extrait le premier objet JSON d'une réponse, éventuellement entouré d'un bloc de code. None si illisible."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0), strict=False)  # tabulations ou retours à la ligne bruts dans les chaînes
    except json.JSONDecodeError:
        return None
