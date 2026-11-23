"""Rapproche les descriptions de ce qu'on tape vraiment dans le playground : fautes de frappe, style relâché
(minuscules, sans ponctuation), exemples glissés dans la phrase.

Les transformations ne touchent que le texte de la description : la regex de référence et les tests cachés restent
les mêmes. Elles épargnent tout ce qui porte du sens littéral : passages entre guillemets, mots présents dans la regex
ou dans les exemples, sigles et mots en casse mixte.

    # SFT : train bruité, valid inchangé (comparable aux runs précédents)
    uv run scripts/noise.py sft --prepared-dir data/prepared-plus2 --out-prepared data/prepared-noisy --out-sft data/sft-noisy

    # Jeux d'éval bruités, appariés cas par cas avec leur version propre
    uv run scripts/noise.py eval data/prepared-plus2/heldout.jsonl data/eval/heldout-noisy.jsonl
    uv run scripts/noise.py eval data/eval/benchmark-regex101.jsonl data/eval/benchmark-regex101-noisy.jsonl

Chaque cas transformé garde la liste des transformations appliquées dans `noise`.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# Parts appliquées au train : environ une requête sur quatre avec des fautes (les études sur les requêtes web en
# trouvent 10 à 15 % avec une faute d'orthographe ; une description de plusieurs dizaines de mots tapée vite en a
# plus souvent), et dans celles-ci ~4 % des mots touchés, entre 1 et 3 fautes.
TRAIN = {"typos": 0.25, "casual": 0.15, "inline": 0.5}
# Éval : toujours des fautes, une fois sur deux le style relâché, les exemples toujours glissés dans la phrase.
EVAL = {"typos": 1.0, "casual": 0.5, "inline": 1.0}
TYPO_WORD_RATE = 0.04
MAX_TYPOS = 3

# Fautes d'orthographe courantes en anglais, prioritaires sur les fautes de frappe mécaniques.
MISSPELLINGS = {
    "separator": ["seperator", "separater"], "separators": ["seperators"], "separated": ["seperated"],
    "separate": ["seperate"], "length": ["lenght", "lengh"], "address": ["adress", "addres"],
    "addresses": ["adresses"], "character": ["charachter", "caracter", "charater"],
    "characters": ["charachters", "caracters", "charaters"], "optional": ["optionnal", "optinal"],
    "optionally": ["optionaly"], "allowed": ["alowed", "allowd"], "occurrence": ["occurence"],
    "occurrences": ["occurences"], "beginning": ["begining"], "preceded": ["preceeded"], "between": ["betwen"],
    "following": ["folowing"], "followed": ["folowed"], "exactly": ["exactely", "excatly"],
    "necessary": ["neccessary"], "hexadecimal": ["hexadecimel", "hexdecimal"], "whitespace": ["whitespce"],
    "consecutive": ["consecutve", "consequtive"], "uppercase": ["upercase"], "lowercase": ["lowecase"],
    "numbers": ["numbres"], "number": ["numer"], "digits": ["digts"], "digit": ["digi"],
    "string": ["strign"], "valid": ["vaild"], "without": ["whitout"], "which": ["wich"],
}

KEYBOARD = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBOURS = {}
for r, row in enumerate(KEYBOARD):
    for c, ch in enumerate(row):
        near = [row[i] for i in (c - 1, c + 1) if 0 <= i < len(row)]
        near += [KEYBOARD[rr][i] for rr in (r - 1, r + 1) if 0 <= rr < 3 for i in (c, c + 1) if i < len(KEYBOARD[rr])]
        NEIGHBOURS[ch] = near

# Passages à ne jamais modifier : entre guillemets doubles, backticks, ou apostrophes encadrant un littéral.
QUOTED = re.compile(r'"[^"\n]*"|`[^`\n]*`|(?<![A-Za-z])\'[^\'\n]+\'(?![A-Za-z])')
WORD = re.compile(r"(?<![\w.\\/@#$-])[A-Za-z]+(?![\w\\/@-]|\.\w)")
INLINE_ONE = ["{d}, e.g. {x}", "{d}, like {x}", "{d} (ex: {x})", "{d} - something like {x}", "{d}, for example {x}"]
INLINE_MANY = ["{d}, e.g. {xs}", "{d}, like {xs}", "{d} (ex: {xs})", "{d} - things like {xs}"]
NOT = [", but not {ys}", ", not {ys}", " ({ys} should fail)", ", {ys} is invalid"]


def protected_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in QUOTED.finditer(text)]


def literal_words(e: dict) -> set[str]:
    """Mots qui ont un sens littéral pour ce besoin : présents dans la regex ou dans un exemple montré."""
    shown = e.get("examples") or {}
    sources = [e["regex"], *shown.get("match", []), *shown.get("no_match", [])]
    return {w.lower() for s in sources for w in re.findall(r"[A-Za-z]{2,}", s)}


def editable_words(text: str, literals: set[str]) -> list[re.Match]:
    spans = protected_spans(text)
    return [
        m for m in WORD.finditer(text)
        if not any(a <= m.start() < b for a, b in spans) and m.group().lower() not in literals
    ]


def misspell(word: str, rng: random.Random) -> str:
    lower = word.lower()
    if lower in MISSPELLINGS and rng.random() < 0.7:
        out = rng.choice(MISSPELLINGS[lower])
        return out[0].upper() + out[1:] if word[0].isupper() else out
    # Jamais la première lettre : on la rate rarement.
    i = rng.randrange(1, len(word))
    kind = rng.choices(["neighbour", "drop", "swap", "double"], weights=[0.3, 0.3, 0.25, 0.15])[0]
    if kind == "neighbour" and word[i].lower() in NEIGHBOURS:
        c = rng.choice(NEIGHBOURS[word[i].lower()])
        return word[:i] + (c.upper() if word[i].isupper() else c) + word[i + 1:]
    if kind == "swap" and i < len(word) - 1 and word[i] != word[i + 1]:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    return word[:i] + word[i + 1:]


def add_typos(text: str, literals: set[str], rng: random.Random) -> str:
    # Mots de 4 lettres ou plus, en minuscules ou capitalisés : pas les sigles ni la casse mixte.
    candidates = [
        m for m in editable_words(text, literals)
        if len(m.group()) >= 4 and (m.group().islower() or m.group()[1:].islower())
    ]
    if not candidates:
        return text
    n = min(MAX_TYPOS, len(candidates), max(1, sum(rng.random() < TYPO_WORD_RATE for _ in candidates)))
    for m in sorted(rng.sample(candidates, n), key=lambda m: -m.start()):
        text = text[:m.start()] + misspell(m.group(), rng) + text[m.end():]
    return text


def casualize(text: str, literals: set[str]) -> str:
    """Minuscules sur les mots simplement capitalisés, et retrait de la ponctuation de phrase (. , ; : en fin de mot)."""
    for m in sorted(editable_words(text, literals), key=lambda m: -m.start()):
        w = m.group()
        if len(w) > 1 and w[0].isupper() and w[1:].islower() or w in ("A", "I"):
            text = text[:m.start()] + w.lower() + text[m.end():]
    spans = protected_spans(text)
    out = []
    for i, ch in enumerate(text):
        nxt = text[i + 1] if i + 1 < len(text) else ""
        prev = text[i - 1] if i else ""
        if (
            ch in ".,;:" and prev.isalpha() and (nxt == "" or nxt.isspace())
            and not any(a <= i < b for a, b in spans)
            and not re.search(r"\b(e\.g|i\.e|etc|vs)$", text[max(0, i - 4):i])
        ):
            continue
        out.append(ch)
    return "".join(out).rstrip()


def show(s: str, rng: random.Random) -> str:
    """Un exemple tel qu'on le taperait : nu s'il est sans ambiguïté, sinon entre guillemets (échappements JSON)."""
    plain = re.fullmatch(r"[\x21-\x7e -￿]+", s) and not re.search(r"[\"'`,()]|[.:;]$", s)
    return s if plain and rng.random() < 0.6 else json.dumps(s, ensure_ascii=False)


def inline_examples(e: dict, rng: random.Random) -> dict:
    shown = e.get("examples") or {}
    if not shown.get("match"):
        return e
    d = e["description"].rstrip()
    if "\n" in d:  # Spécification en liste : l'exemple va sur sa propre ligne.
        d += "\n" + rng.choice(["Example: {x}", "e.g. {x}", "Like {x}"]).format(x=", ".join(show(x, rng) for x in shown["match"]))
    else:
        d = d.rstrip(".")
        xs = [show(x, rng) for x in shown["match"]]
        template = rng.choice(INLINE_ONE) if len(xs) == 1 else rng.choice(INLINE_MANY)
        d = template.format(d=d, x=xs[0], xs=" or ".join(xs) if len(xs) == 2 else ", ".join(xs))
    if shown.get("no_match"):
        ys = [show(y, rng) for y in shown["no_match"]]
        d += rng.choice(NOT).format(ys=" or ".join(ys))
    return {**e, "description": d, "examples": {"match": [], "no_match": []}}


def perturb(e: dict, shares: dict, salt: str) -> dict:
    rng = random.Random(f"noise-{salt}-{e['id']}")
    literals, original = literal_words(e), e["description"]
    applied = []
    if rng.random() < shares["inline"] and (e.get("examples") or {}).get("match"):
        e, applied = inline_examples(e, rng), applied + ["inline"]
    description = e["description"]
    # Le registre terse est déjà relâché.
    if e.get("register") != "terse" and rng.random() < shares["casual"]:
        description, applied = casualize(description, literals), applied + ["casual"]
    if rng.random() < shares["typos"]:
        noisy = add_typos(description, literals, rng)
        if noisy != description:
            description, applied = noisy, applied + ["typos"]
    if not applied:
        return e
    return {**e, "description": description, "noise": applied, "clean_description": original}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(rows: list[dict]) -> None:
    counts = Counter(t for r in rows for t in r.get("noise", []))
    n = max(len(rows), 1)
    print(f"  {len(rows)} cas, dont {sum('noise' in r for r in rows) / n:.0%} transformés : "
          + ", ".join(f"{k} {v / n:.0%}" for k, v in counts.most_common()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sft = sub.add_parser("sft", help="bruite le train d'un dossier prepared et écrit le SFT")
    sft.add_argument("--prepared-dir", type=Path, required=True)
    sft.add_argument("--out-prepared", type=Path, required=True)
    sft.add_argument("--out-sft", type=Path, required=True)
    ev = sub.add_parser("eval", help="bruite tous les cas d'un jeu d'éval")
    ev.add_argument("src", type=Path)
    ev.add_argument("dst", type=Path)
    args = parser.parse_args()

    if args.command == "eval":
        rows = [perturb(e, EVAL, "eval") for e in read_jsonl(args.src)]
        write_jsonl(args.dst, rows)
        print(args.dst)
        report(rows)
        return

    from common import build_messages
    from validate import MAX_SFT_TOKENS, sft_lengths

    for name in ("train", "valid", "heldout"):
        rows = read_jsonl(args.prepared_dir / f"{name}.jsonl")
        if name == "train":
            rows = [perturb(e, TRAIN, "train") for e in rows]
            print(name)
            report(rows)
        write_jsonl(args.out_prepared / f"{name}.jsonl", rows)
        if name == "heldout":
            continue
        conversations = [build_messages(e, with_answer=True) for e in rows]
        lengths = sft_lengths(conversations)
        write_jsonl(args.out_sft / f"{name}.jsonl",
                    [{"messages": m} for m, n in zip(conversations, lengths) if n <= MAX_SFT_TOKENS])


if __name__ == "__main__":
    main()
