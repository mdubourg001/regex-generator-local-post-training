import json
import time
from pathlib import Path

import pytest

from common import FEW_SHOT, RegexRunner, build_messages, grade, parse_answer, parse_literal, score

FIXTURES = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "cases.jsonl").open()]


@pytest.fixture(scope="module")
def runner():
    with RegexRunner(timeout_ms=500) as r:
        yield r


# --- Parsing ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/^a+$/", ("^a+$", "")),
        ("/abc/gimsuy", ("abc", "gimsuy")),
        (r"/a\/b/", (r"a\/b", "")),
        ("/[/]+/", ("[/]+", "")),  # `/` non échappé dans une classe
        (r"/[\]/]/", (r"[\]/]", "")),  # `]` échappé : la classe continue jusqu'au `]` suivant
        (r"/\\/", (r"\\", "")),
    ],
)
def test_parse_literal_valid(text, expected):
    assert parse_literal(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "//",  # commentaire JS, pas une regex
        "/abc",  # pas de `/` fermant
        r"/abc\/",  # le `/` final est échappé
        "/[/]",  # le `/` est dans la classe, donc pas de fermeture
        "/a/x",  # flag inconnu
        "/a/d",  # flag ES2022, hors périmètre
        "/a/v",  # flag ES2024, hors périmètre
        "/a/gg",  # flag dupliqué
        "/a/ g",  # espace entre le littéral et les flags
        "/a\nb/",  # retour à la ligne dans le littéral
        "^a+$",  # pattern sans délimiteurs
        "",
    ],
)
def test_parse_literal_invalid(text):
    assert parse_literal(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("  /a+/i\n", ("a+", "i")),
        ("<think>\n\n</think>\n\n/a+/", ("a+", "")),  # Qwen3, thinking désactivé
        ("```js\n/a+/\n```", ("a+", "")),
        ("```\n/a+/\n```", ("a+", "")),
        ("`/a+/`", ("a+", "")),
        ('"/a+/i"', ("a+", "i")),
        ("'/a+/'", ("a+", "")),
        ('"/a+/', None),  # guillemet non fermé
        ("Here is the regex: /a+/", None),  # réponse bavarde
        ("/a+/\n\nThis matches one or more a.", None),
        ("/a+/ or /b+/", None),
    ],
)
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


# --- Prompt -----------------------------------------------------------------------------------------------------------


def test_prompt_shows_examples_as_json_strings():
    example = {"description": "d", "examples": {"match": ["a b "], "no_match": ["é"]}, "regex": "/x/"}
    user = build_messages(example)[1]["content"]
    assert user == 'Description: d\n\nShould match:\n- "a b "\n\nShould not match:\n- "é"'


def test_prompt_without_examples():
    messages = build_messages({"description": "d", "regex": "/x/"}, with_answer=True)
    assert messages[1]["content"] == "Description: d"
    assert messages[2] == {"role": "assistant", "content": "/x/"}


# --- Exécution --------------------------------------------------------------------------------------------------------


def test_runner_results_are_per_test_and_in_order(runner):
    jobs = [
        {"pattern": "^a+$", "flags": "", "match": ["a", "aa", "b"], "no_match": ["b", "a"]},
        {"pattern": r"^\p{L}+$", "flags": "u", "match": ["Zoë", "李"], "no_match": ["1"]},
    ]
    assert runner.run(jobs) == [
        {"match": [True, True, False], "no_match": [True, False]},
        {"match": [True, True], "no_match": [True]},
    ]


def test_runner_resets_last_index_with_global_flag(runner):
    # Sans remise à zéro de lastIndex, le second test() sur "a" partirait de l'index 1 et échouerait.
    assert runner.run([{"pattern": "a", "flags": "g", "match": ["a", "a", "a"], "no_match": []}]) == [
        {"match": [True, True, True], "no_match": []}
    ]


def test_runner_reports_syntax_errors(runner):
    [result] = runner.run([{"pattern": "(a", "flags": "", "match": ["a"], "no_match": []}])
    assert "Invalid regular expression" in result["error"]


def test_runner_rejects_post_es2018_syntax(runner):
    jobs = [
        {"pattern": "(?i:a)", "flags": "", "match": ["a"], "no_match": []},  # modifiers inline (ES2025)
        {"pattern": "(?<x>a)|(?<x>b)", "flags": "", "match": ["a"], "no_match": []},  # noms dupliqués (ES2025)
    ]
    assert all("error" in r for r in runner.run(jobs))


def test_runner_times_out_on_catastrophic_backtracking_and_recovers(runner):
    redos = {"pattern": "^(a+)+$", "flags": "", "match": [], "no_match": ["a" * 40 + "b"]}
    start = time.monotonic()
    [result] = runner.run([redos])
    assert result == {"timeout": True}
    assert time.monotonic() - start < 2
    # Le worker bloqué a été remplacé : le runner fonctionne toujours.
    assert runner.run([{"pattern": "^a$", "flags": "", "match": ["a"], "no_match": []}]) == [
        {"match": [True], "no_match": []}
    ]


def test_runner_handles_many_jobs(runner):
    jobs = [{"pattern": f"^{i}$", "flags": "", "match": [str(i)], "no_match": [str(i + 1)]} for i in range(2000)]
    assert all(r == {"match": [True], "no_match": [True]} for r in runner.run(jobs))


# --- Métriques --------------------------------------------------------------------------------------------------------


def test_reference_regexes_pass_their_own_tests(runner):
    grades = grade(runner, FIXTURES, [e["regex"] for e in FIXTURES])
    assert [g["failed_match"] + g["failed_no_match"] for g in grades] == [[] for _ in FIXTURES]
    assert score(grades)["success_rate"] == 1.0


def test_fixtures_have_enough_discriminating_tests():
    for e in FIXTURES:
        assert len(e["tests"]["match"]) >= 5 and len(e["tests"]["no_match"]) >= 5, e["id"]
        shown = e.get("examples", {})
        hidden = set(e["tests"]["match"]) | set(e["tests"]["no_match"])
        assert not hidden & (set(shown.get("match", [])) | set(shown.get("no_match", []))), e["id"]


def test_grade_statuses_and_rates(runner):
    example = {"tests": {"match": ["ab", "aab"], "no_match": ["b", "abc"]}}
    answers = [
        "/^a+b$/",  # juste
        "/a+b/",  # pas ancrée : accepte "abc"
        "/.*/",  # trop permissive
        "Use /^a+b$/",  # bavarde
        "/(a/",  # syntaxe invalide
        "/^(a|a)+b$/",  # juste : vulnérable au ReDoS, mais pas sur ces tests courts
    ]
    grades = grade(runner, [example] * len(answers), answers)
    assert [g["status"] for g in grades] == ["ok", "ok", "ok", "malformed", "malformed", "ok"]
    assert [g["success"] for g in grades] == [True, False, False, False, False, True]
    assert grades[1]["failed_no_match"] == ["abc"]
    assert (grades[2]["match_rate"], grades[2]["no_match_rate"], grades[2]["pass_rate"]) == (1.0, 0.0, 0.5)
    assert grades[3]["pass_rate"] == 0.0

    result = score(grades)
    assert result["success_rate"] == pytest.approx(2 / 6)
    assert result["malformed_rate"] == pytest.approx(2 / 6)
    assert result["timeout_rate"] == 0.0


def test_grade_counts_timeouts_as_failures(runner):
    example = {"tests": {"match": ["a" * 40], "no_match": ["a" * 40 + "b"]}}
    [g] = grade(runner, [example], ["/^(a+)+$/"])
    assert (g["status"], g["success"], g["pass_rate"]) == ("timeout", False, 0.0)


def test_few_shot_prompt_and_answers_are_valid():
    messages = build_messages({"description": "d"}, few_shot=True)
    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * len(FEW_SHOT) + ["user"]
    assert all(parse_answer(shot["regex"]) for shot in FEW_SHOT)
    assert "Should match:" in messages[1]["content"]
