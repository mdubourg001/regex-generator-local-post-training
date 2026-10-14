"""Récompense d'une réponse, à partir de sa note `common.grade`. Sert à classer les échantillons des paires DPO (étape 7)
et de reward au GRPO (étape 9).

    reward = 0,5 × taux_positifs × taux_négatifs + 0,5 × succès_complet
    reward = -1 si la sortie est mal formée ou part en timeout

Le produit empêche de gagner sur un seul côté : `/.*/` accepte tous les positifs mais aucun négatif, et vaut 0. Le terme
de succès complet fait qu'une regex entièrement juste (1) vaut nettement plus qu'une regex presque juste (≤ 0,5).
"""


def reward(grade: dict) -> float:
    if grade["status"] != "ok":
        return -1.0
    return 0.5 * grade["match_rate"] * grade["no_match_rate"] + 0.5 * grade["success"]
