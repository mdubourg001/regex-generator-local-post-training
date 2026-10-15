import pytest
import yaml

from common import RegexRunner
from generate import CONFIG, TAXONOMY, build_descriptions_messages, build_tests_messages, parse_items, plan
from validate import (add_examples, check, decontaminate, features, is_anchored, mutants, scan, second_opinion, split_of,
                      unanchored)

CFG = yaml.safe_load(CONFIG.read_text())
TAX = yaml.safe_load(TAXONOMY.read_text())


@pytest.fixture(scope="module")
def runner():
    with RegexRunner(timeout_ms=500) as r:
        yield r


# --- Plan et prompts --------------------------------------------------------------------------------------------------


def test_plan_is_stable_when_extended():
    assert plan(10, CFG, TAX) == plan(30, CFG, TAX)[:10]
    batches = plan(len(TAX["categories"]), CFG, TAX)
    assert [b["category"] for b in batches] == [c["id"] for c in TAX["categories"]]
    assert all(len(b["slots"]) == CFG["batch_size"] for b in batches)


def test_targeted_plan_uses_its_own_ids_and_weights():
    batches = plan(6, CFG, TAX, categories=["logs", "logs", "multiline"], prefix="t")
    assert [b["category"] for b in batches] == ["logs", "logs", "multiline"] * 2
    assert batches[0]["batch"] == "t00000" and batches[0]["slots"][0]["id"] == "t00000-0"
    assert plan(3, CFG, TAX) == plan(3, CFG, TAX, prefix="b")


def test_tests_prompt_never_shows_the_regex():
    batch = plan(1, CFG, TAX)[0]
    assert "seed intent" in build_descriptions_messages(batch, TAX)[1]["content"]
    items = [{"slot": 1, "description": "a hex color", "regex": "/^#[0-9a-f]{6}$/"}]
    assert "#[0-9a-f]" not in "".join(m["content"] for m in build_tests_messages(items, CFG))


def test_parse_items():
    content = '```json\n{"items": [{"slot": 1, "description": "d", "regex": "/a/"}, {"slot": "x"}]}\n```'
    assert parse_items(content, ["description", "regex"]) == [{"slot": 1, "description": "d", "regex": "/a/"}]
    assert parse_items("not json", ["description"]) is None


def test_parse_items_expands_repeated_strings():
    content = '{"items": [{"slot": 1, "match": ["a".repeat(3), "\\u00e9" * 2, "ok"]}]}'
    assert parse_items(content, ["match"])[0]["match"] == ["aaa", "éé", "ok"]
    # Caractère de contrôle brut (tabulation) dans une chaîne répétée ou ordinaire.
    assert parse_items('{"items": [{"slot": 1, "match": ["a\tb" * 2, "c\td"]}]}', ["match"])[0]["match"] == ["a\tba\tb", "c\td"]


# --- Analyse d'un pattern ---------------------------------------------------------------------------------------------


def test_scan_ignores_escapes_and_class_contents():
    spans = scan(r"^\.[a.{]{2,3}x{2}\{3}.$")
    assert spans["classes"] == [(3, 8)]
    assert spans["braces"] == [(8, 13), (14, 17)]
    assert spans["dots"] == [(21, 22)]


@pytest.mark.parametrize(
    "pattern, expected",
    [("^a$", "a"), ("^a", "a"), ("a$", "a"), (r"a\$", None), (r"a\\$", r"a\\"), ("a", None), ("^(?:a|b)$", "(?:a|b)")],
)
def test_unanchored(pattern, expected):
    assert unanchored(pattern) == expected
    assert is_anchored(pattern) == (expected is not None)


def test_mutants():
    assert mutants(r"^[a-z]{2,4}\d{0,2}x{1,}$") == {
        "quantificateurs relâchés": r"^[a-z]+\d*x{1,}$",
        "classes élargies": r"^[\s\S]{2,4}\d{0,2}x{1,}$",
    }
    assert mutants("^abc$") == {}


def test_features():
    found = features(r"^(?<y>\d{4})-(?=\p{L})(?<!x)[a-z]\.\k<y>|$", "iu")
    assert {"anchors", "named_groups", "shorthand", "quantifiers", "lookahead", "lookbehind", "unicode_property"} <= found
    assert {"char_class", "escaping", "alternation", "flag_i"} <= found
    assert "unanchored" not in found and "flag_u" not in found
    assert features("error", "")  == {"unanchored"}


# --- Garde-fous -------------------------------------------------------------------------------------------------------


def example(regex: str, match: list[str], no_match: list[str], near_miss: list[str] | None = None, **extra) -> dict:
    near_miss = near_miss if near_miss is not None else no_match[:3]
    return {"id": "x", "batch": "b0", "description": regex, "regex": regex, "tests": {"match": match, "no_match": no_match},
            "near_miss": near_miss, **extra}


HEX_MATCH = ["#fff", "#000", "#abc", "#123", "#a1b"]
HEX_NO = ["fff", "#ffff", "#ff", "#ggg", " #fff"]


@pytest.mark.parametrize(
    "e, reason",
    [
        (example("^#[0-9a-f]{3}$", HEX_MATCH, HEX_NO), "regex hors format ES2018"),
        (example("/^#[0-9a-f]{3}$/", HEX_MATCH[:4], HEX_NO), "pas assez de tests"),
        (example("/^#[0-9a-f]{3}$/", HEX_MATCH, HEX_NO, near_miss=HEX_NO[:2]), "pas assez de tests"),
        (example("/^#[0-9a-f]{3}(/", HEX_MATCH, HEX_NO), "regex invalide"),
        (example("/^(a+)+$/", ["a"] * 5, ["a" * 40 + "b"] * 5), "timeout (ReDoS)"),
        (example("/^#[0-9a-f]{3}$/", HEX_MATCH, ["#FFF", *HEX_NO[1:]]), None),  # #FFF sans flag i : refusé, cohérent
        (example("/^#[0-9a-f]{3}$/i", HEX_MATCH, ["#FFF", *HEX_NO[1:]]), "référence en désaccord avec les tests"),
        # Aucun test avec du texte autour : la version sans ancres passe aussi.
        (example("/^#[0-9a-f]{3}$/", HEX_MATCH, ["fff", "#ff", "#ggg", "#gg", "#"]), "ancrage non testé"),
        # Aucun négatif sur les caractères : la classe élargie passe aussi.
        (example("/^#[0-9a-f]{3}$/", HEX_MATCH, ["fff", "#ffff", "#ff", " #fff", "#fff "]), "tests peu discriminants (classes élargies)"),
        # Aucun négatif sur la longueur : les quantificateurs relâchés passent aussi.
        (example("/^#[0-9a-f]{3}$/", HEX_MATCH, ["fff", "#ggg", "#xyz", " #fff", "#fff "]), "tests peu discriminants (quantificateurs relâchés)"),
    ],
)
def test_check_reasons(runner, e, reason):
    kept, rejected = check([e], CFG, runner)
    assert (list(rejected) or [None]) == [reason]
    assert len(kept) == (reason is None)


def test_check_rejects_duplicate_descriptions(runner):
    e = example("/^#[0-9a-f]{3}$/", HEX_MATCH, HEX_NO)
    kept, rejected = check([e, {**e, "id": "y", "description": e["description"].upper() + " "}], CFG, runner)
    assert len(kept) == 1 and rejected == {"doublon": 1}


# --- Exemples et split ------------------------------------------------------------------------------------------------


def test_add_examples_moves_strings_out_of_hidden_tests():
    required = CFG["tests_required"]
    e = example("/x/", [f"m{i}" for i in range(8)], [f"n{i}" for i in range(8)])
    with_examples = [add_examples({**e, "id": str(i)}, CFG, share=1.0) for i in range(200)]
    for w in with_examples:
        shown = w["examples"]
        assert 1 <= len(shown["match"]) and len(shown["match"]) + len(shown["no_match"]) <= CFG["max_examples"]
        assert not set(shown["match"]) & set(w["tests"]["match"])
        assert sorted(shown["match"] + w["tests"]["match"]) == sorted(e["tests"]["match"])
        assert len(w["tests"]["match"]) >= required["match"] and len(w["tests"]["no_match"]) >= required["no_match"]
    assert add_examples(e, CFG, share=0.0) == e


def test_split_is_by_batch_and_roughly_proportional():
    splits = [split_of(f"b{i:05d}", CFG["split"]) for i in range(4000)]
    assert splits[:50] == [split_of(f"b{i:05d}", CFG["split"]) for i in range(50)]
    assert 0.87 < splits.count("train") / len(splits) < 0.93


def test_decontaminate_drops_same_regex_or_close_description():
    benchmark = [{"description": "IPv4 address in dotted decimal notation", "regex": "/^ip$/"}]
    examples = [
        {"description": "anything else", "regex": "/^ip$/"},  # même regex
        {"description": "An IPv4 address in dotted decimal notation.", "regex": "/^x$/"},  # même besoin
        {"description": "French postal code", "regex": "/^\\d{5}$/"},
    ]
    kept, n = decontaminate(examples, benchmark)
    assert n == 2 and [e["description"] for e in kept] == ["French postal code"]


def test_second_opinion_keeps_agreements_and_unconcerned(runner):
    tests = {"match": ["abc", "abd"], "no_match": ["ab", "abcd"]}
    examples = [{"id": i, "tests": tests} for i in ("ok", "wrong", "missing", "malformed", "main")]
    opinions = {"ok": "/^ab[cd]$/", "wrong": "/^ab/", "missing": None, "malformed": "^ab[cd]$"}
    kept, n = second_opinion(examples, opinions, runner)
    assert [e["id"] for e in kept] == ["ok", "main"] and n == 3  # « main » n'a pas de second avis prévu
