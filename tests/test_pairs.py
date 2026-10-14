import pytest

from pairs import pick
from reward import reward


def g(status="ok", match_rate=1.0, no_match_rate=1.0):
    return {"status": status, "match_rate": match_rate, "no_match_rate": no_match_rate,
            "success": status == "ok" and match_rate == no_match_rate == 1.0}


@pytest.mark.parametrize(
    "grade, expected",
    [
        (g(), 1.0),  # entièrement juste
        (g(match_rate=0.8), 0.4),  # presque juste : plafonnée à 0,5
        (g(no_match_rate=0.0), 0.0),  # trop permissive (/.*/) : le produit l'annule
        (g(match_rate=0.0), 0.0),  # trop stricte
        (g(status="malformed", match_rate=0.0, no_match_rate=0.0), -1.0),
        (g(status="timeout", match_rate=0.0, no_match_rate=0.0), -1.0),
    ],
)
def test_reward(grade, expected):
    assert reward(grade) == pytest.approx(expected)


def record(*answers):
    """answers : (sortie, succès, reward)."""
    return {"outputs": [a for a, _, _ in answers], "success": [s for _, s, _ in answers],
            "reward": [r for _, _, r in answers]}


def test_pick_most_frequent_right_and_wrong():
    r = record(("/a/", True, 1), ("/b/", True, 1), ("/b/", True, 1), ("/x/", False, 0.4), ("/y/", False, -1),
               ("/y/", False, -1), ("/y/\n", False, -1))
    assert pick(r) == ("/b/", "/y/")  # réponses comparées sans les espaces de fin


def test_pick_breaks_ties_on_lowest_reward():
    r = record(("/a/", True, 1), ("/x/", False, 0.4), ("/y/", False, 0.0))
    assert pick(r) == ("/a/", "/y/")


@pytest.mark.parametrize("success", [True, False])
def test_no_pair_without_contrast(success):
    assert pick(record(("/a/", success, 1 if success else 0), ("/b/", success, 1 if success else 0))) is None
