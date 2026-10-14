from rft import keep


def record(*answers):
    """answers : (sortie, succès)."""
    return {"outputs": [a for a, _ in answers], "success": [s for _, s in answers]}


def test_keep_distinct_right_answers_most_frequent_first():
    r = record(("/ab/", True), ("/b/", True), ("/b/\n", True), ("/x/", False), ("/x/", False))
    assert keep(r, 2) == ["/b/", "/ab/"]  # réponses comparées sans les espaces de fin


def test_keep_breaks_ties_on_shortest_and_caps():
    r = record(("/abc/", True), ("/a/", True), ("/ab/", True))
    assert keep(r, 2) == ["/a/", "/ab/"]


def test_keep_nothing_without_right_answer():
    assert keep(record(("/x/", False), ("/y/", False)), 2) == []
