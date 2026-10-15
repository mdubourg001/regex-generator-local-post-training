"""Isolated schema, ES2018, reference, mutation, coverage and JSONL checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import subprocess
import yaml

ROOT = Path(__file__).resolve().parent
KEYS = {"id", "difficulty", "status", "description", "regex", "match", "no_match", "pitfall"}


def spans(pattern):
    found = {"classes": [], "braces": []}
    i, start = 0, None
    while i < len(pattern):
        char = pattern[i]
        if char == "\\":
            i += 2
            continue
        if start is not None:
            if char == "]":
                found["classes"].append((start, i + 1))
                start = None
        elif char == "[":
            start = i
        elif char == "{" and (m := re.match(r"\{\d+(?:,\d*)?\}", pattern[i:])):
            if m[0] not in ("{1,}", "{0,}"):
                found["braces"].append((i, i + len(m[0])))
            i += len(m[0])
            continue
        i += 1
    return found


def mutate(pattern, locations, replacement):
    for a, b in reversed(locations):
        pattern = pattern[:a] + replacement(pattern[a:b]) + pattern[b:]
    return pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "build"], default="check", nargs="?")
    args = parser.parse_args()
    cases, errors, jobs, owners = [], [], [], []
    for path in sorted(ROOT.glob("*.yaml")):
        text = path.read_text()
        expected = f"# Benchmark GPT, catégorie {path.stem}. Format et contrôles : scripts/benchmark.py."
        if not text.startswith(expected + "\n"):
            errors.append(f"{path.name}: comment header")
        group = yaml.safe_load(text)
        if len(group) != 20 or Counter(c["difficulty"] for c in group) != {"simple": 5, "medium": 8, "hard": 7}:
            errors.append(f"{path.name}: size or difficulty mix")
        for index, c in enumerate(group, 1):
            ident = c["id"]
            if set(c) != KEYS or ident != f"gpt-{path.stem}-{index:02d}" or c["status"] not in ("ok", "todo", "drop"):
                errors.append(f"{ident}: schema")
            c["category"] = path.stem
            cases.append(c)
            if c["status"] == "drop":
                continue
            for side in ("match", "no_match"):
                if len(c[side]) < 8 or len(c[side]) != len(set(c[side])) or not all(isinstance(x, str) for x in c[side]):
                    errors.append(f"{ident}: at least 8 unique strings required for {side}")
                if any(re.search("[\ud800-\udfff]", x) for x in c[side]):
                    errors.append(f"{ident}: lone UTF-16 surrogate in YAML value")
            if set(c["match"]) & set(c["no_match"]):
                errors.append(f"{ident}: conflicting tests")
            literal = c["regex"]
            m = re.fullmatch(r"/(.+)/([imsu]*)", literal)
            if not m or len(set(m[2])) != len(m[2]):
                errors.append(f"{ident}: flags or literal")
                continue
            pattern, flags = m.groups()
            if re.search(r"\(\?[ims-]+:", pattern):
                errors.append(f"{ident}: post-ES2018 inline modifiers")
            # Validate the exact JS literal too, not just the constructor spelling.
            literal_job = dict(pattern=pattern, flags=flags, match=c["match"], no_match=c["no_match"], literal=literal)
            variants = {"reference": pattern}
            if pattern.startswith("^"):
                # Remove ALL whole-input protections, including the strict end guard.
                unanchored = pattern[1:].removesuffix(r"$(?![\s\S])")
                variants["without anchors"] = unanchored
            # The final [\s\S] is an invariant strict-end guard, not a data class.
            scan = spans(pattern.removesuffix(r"$(?![\s\S])"))
            if scan["braces"]:
                variants["relaxed quantifiers"] = mutate(pattern, scan["braces"], lambda t: "*" if t.startswith("{0") else "+")
            if scan["classes"]:
                variants["widened classes"] = mutate(pattern, scan["classes"], lambda _: r"[\s\S]")
            for kind, p in variants.items():
                jobs.append(literal_job if kind == "reference" else dict(pattern=p, flags=flags, match=c["match"], no_match=c["no_match"]))
                owners.append((ident, kind))
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    proc = subprocess.run(["node", str(ROOT / "runner.cjs")], input=json.dumps(jobs), text=True, capture_output=True, timeout=120)
    if proc.returncode:
        raise SystemExit(proc.stderr)
    output = json.loads(proc.stdout)
    warnings = []
    for (ident, kind), result in zip(owners, output):
        if kind == "reference" and (result.get("error") or result["bad_match"] or result["bad_no_match"]):
            errors.append(f"{ident}: {json.dumps(result, ensure_ascii=False)}")
        elif kind != "reference" and not result.get("error") and not result["bad_match"] and not result["bad_no_match"]:
            warnings.append({"id": ident, "mutation": kind})
    counts = Counter()
    for c in cases:
        p, f = c["regex"][1:].rsplit("/", 1)
        semantic = p.removesuffix(r"$(?![\s\S])")
        counts["lookahead"] += bool(re.search(r"\(\?[=!]", semantic))
        counts["backreference"] += bool(re.search(r"\\[1-9]|\\k<", p))
        counts["lookbehind"] += bool(re.search(r"\(\?<[=!]", p))
        counts["multiline_or_explicit_lines"] += "m" in f or r"\n" in p or r"\r" in p
        counts["dotall"] += "s" in f
        counts["unicode_property"] += bool(re.search(r"\\[pP]\{", p))
        counts["contains"] += not p.startswith("^")
        counts["numeric_range_alternation"] += "|" in p and bool(re.search(r"\[[0-9]-[0-8]\]|\[[1-9]-9\]", p))
    report = dict(cases=len(cases), difficulties=dict(Counter(c["difficulty"] for c in cases)), statuses=dict(Counter(c["status"] for c in cases)), coverage=dict(counts), errors=errors, mutation_warnings=warnings, isolation="No project modules, training data, taxonomy or other benchmark read. Training overlap checks unavailable.")
    (ROOT / "check-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)
    if args.command == "build":
        with (ROOT / "benchmark-gpt.jsonl").open("w") as handle:
            for c in cases:
                if c["status"] != "ok":
                    continue
                e = {k: c[k] for k in ("id", "category", "difficulty", "description", "regex")}
                e["tests"] = {s: c[s].copy() for s in ("match", "no_match")}
                rng = random.Random(f"examples-{c['id']}")
                if rng.random() < 1 / 3:
                    shown = {"match": [], "no_match": []}
                    for _ in range(rng.randint(1, 3)):
                        side = "match" if not shown["match"] or rng.random() < .6 else "no_match"
                        pool = e["tests"][side]
                        if len(pool) > 5:
                            shown[side].append(pool.pop(rng.randrange(len(pool))))
                    e["examples"] = shown
                encoded = json.dumps(e, ensure_ascii=False).replace("\u2028", r"\u2028").replace("\u2029", r"\u2029")
                handle.write(encoded + "\n")
        print("Built benchmark-gpt/benchmark-gpt.jsonl")


if __name__ == "__main__":
    main()
