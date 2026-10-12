from collections import Counter
from pathlib import Path

import yaml

TAXONOMY = yaml.safe_load((Path(__file__).parents[1] / "configs" / "taxonomy.yaml").read_text())


def test_categories_are_unique_and_seeded():
    ids = [c["id"] for c in TAXONOMY["categories"]]
    assert len(ids) == len(set(ids))
    assert 15 <= len(ids) <= 25
    for c in TAXONOMY["categories"]:
        assert len(c["intents"]) >= 5, c["id"]


def test_every_feature_is_covered_and_declared():
    declared = set(TAXONOMY["features"])
    used = Counter(f for c in TAXONOMY["categories"] for f in c["features"])
    assert set(used) <= declared, set(used) - declared
    assert declared <= set(used), declared - set(used)


def test_difficulty_and_register_definitions():
    shares = TAXONOMY["difficulties"]["share"]
    assert abs(sum(shares.values()) - 1) < 1e-9
    assert set(shares) <= set(TAXONOMY["difficulties"])
    assert set(TAXONOMY["registers"]) == {"terse", "sentence", "spec", "business"}
