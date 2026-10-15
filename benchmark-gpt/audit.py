"""Independent spec probes and final export invariants; only reads our own files."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import unicodedata
import yaml

ROOT = Path(__file__).resolve().parent
groups = {p.stem: yaml.safe_load(p.read_text()) for p in ROOT.glob("*.yaml")}
cases = [c for group in groups.values() for c in group]
by_id = {c["id"]: c for c in cases}

# Conservative manual inventory: only cases whose bounded numeric portion itself
# is expressed with alternatives, not cases with unrelated unit/currency alternatives.
range_inventory = {
    "numbers": [6,8,10,11,16,20],
    "money": [6,10,11,20],
    "dates-times": [6,7,8,9,10,12,13,14,15,18,19,20],
    "identifiers": [6,9,12,20],
    "web": [8],
    "network": [6,8,9,11,15,18,20],
    "email-accounts": [6,9,12],
    "phone-numbers": [6],
    "postal-official": [6,9,13,19],
    "versions": [6,7,8,9,10,13,20],
    "code-syntax": [8,12],
    "dev-tooling": [6,8,11,13],
    "logs": [6,8,9,10,11,19],
    "text": [8],
    "structured-data": [6,8,12,19],
    "multiline": [7],
    "exclusions": [9],
    "measurements": [6,7,8,11,12,19,20],
}
range_ids = [f"gpt-{cat}-{n:02d}" for cat, indices in range_inventory.items() for n in indices]
assert all("|" in by_id[ident]["regex"] for ident in range_ids), [ident for ident in range_ids if "|" not in by_id[ident]["regex"]]


def letter_clusters(text):
    clusters, marks = 0, 0
    for ch in text:
        category = unicodedata.category(ch)
        if category.startswith("L"):
            clusters += 1
            marks = 0
        elif category.startswith("M") and clusters and marks < 2:
            marks += 1
        else:
            return None
    return clusters


def cluster_name(text):
    pieces = text.split("-")
    counts = [letter_clusters(p) for p in pieces]
    return len(pieces) <= 2 and all(n is not None and n > 0 for n in counts) and 2 <= sum(counts) <= 12


jobs, owners, checks = [], [], 0
cluster_strings = ["ab", "éx", "á̈b", "á̈̃b", "́ab", "a-b", "-ab", "ab-", "a--b", "ab-cd-ef", "é", "abcdefghijkl", "abcdefghijklm", "😀a", "東京-大阪", "ab1", "𐐀𐐨"]
for c in cases:
    desc = c["description"]
    if "Length counts letter clusters" in desc:
        prefix = desc.split("the literal prefix ", 1)[1].split(" followed by ", 1)[0] if "the literal prefix " in desc else ""
        suffix = desc.split("Append the exact literal suffix ",1)[1].split(" after the name",1)[0] if "Append the exact literal suffix " in desc else ""
        suffix = suffix.replace("LF followed by ", "\n")
        def oracle(value):
            if not value.startswith(prefix) or not value.endswith(suffix):
                return False
            name = value[len(prefix):-len(suffix)] if suffix else value[len(prefix):]
            return cluster_name(name)
        for side in ("match", "no_match"):
            for value in c[side]:
                assert bool(oracle(value)) == (side == "match"), (c["id"], side, value)
                checks += 1
        valid = [prefix+s+suffix for s in cluster_strings if cluster_name(s)]
        invalid = [prefix+s+suffix for s in cluster_strings if not cluster_name(s)]
        p, flags = c["regex"][1:].rsplit("/",1)
        jobs.append(dict(pattern=p, flags=flags, match=valid, no_match=invalid))
        owners.append(c["id"])
    if "Only the body counts toward its length limit" in desc:
        opening = desc.split("literal opening wrapper ",1)[1].split(",",1)[0]
        closing = desc.split("literal closing wrapper ",1)[1].split(".",1)[0]
        marker = desc.split("lowercase substring ",1)[1].split(".",1)[0]
        def oracle(value):
            if not value.startswith(opening) or not value.endswith(closing):
                return False
            content = value[len(opening):-len(closing)]
            if ":" not in content:
                return False
            label, remainder = content.split(":",1)
            if not remainder.endswith(":"+label):
                return False
            body = remainder[:-len(label)-1]
            return (2 <= len(label) <= 6 and all("a" <= ch <= "z" for ch in label)
                    and 1 <= len(body) <= 40 and marker in body
                    and ("virus" not in body if c["id"] == "gpt-exclusions-17" else True))
        for side in ("match", "no_match"):
            for value in c[side]:
                assert oracle(value) == (side == "match"), (c["id"], side, value)
                checks += 1
        probes = []
        for label in ["a", "aa", "abcdef", "abcdefg", "AA", marker]:
            for body in ["", "abc", marker, "\n"+marker, marker+":"+marker,
                         "😀"*(40-len(marker))+marker, "😀"*(41-len(marker))+marker,
                         "virus"+marker, marker+"\r\n"]:
                probes.append(opening+label+":"+body+":"+label+closing)
        p, flags = c["regex"][1:].rsplit("/",1)
        jobs.append(dict(pattern=p, flags=flags, match=[s for s in probes if oracle(s)], no_match=[s for s in probes if not oracle(s)]))
        owners.append(c["id"])

results = subprocess.run(["node",str(ROOT/"runner.cjs")],input=json.dumps(jobs),text=True,capture_output=True,check=True)
for ident, result in zip(owners,json.loads(results.stdout)):
    assert not result.get("error") and not result["bad_match"] and not result["bad_no_match"], (ident,result)

report = dict(spec_assertions=checks, extra_spec_probe_strings=sum(len(j[side]) for j in jobs for side in ("match","no_match")),
              numeric_range_alternation_cases=len(range_ids), numeric_range_ids=range_ids,
              node_version=subprocess.check_output(["node","--version"],text=True).strip(),
              total_tests=sum(len(c[s]) for c in cases for s in ("match","no_match")),
              minimum_positive_tests=min(len(c["match"]) for c in cases),
              minimum_negative_tests=min(len(c["no_match"]) for c in cases))
if (ROOT/"benchmark-gpt.jsonl").exists():
    # JSONL is delimited by LF, not Python's broader Unicode splitlines set.
    exported = [json.loads(line) for line in (ROOT/"benchmark-gpt.jsonl").read_text().split("\n") if line]
    assert len(exported) == 420
    for e in exported:
        c = by_id[e["id"]]
        assert e["description"] == c["description"] and e["regex"] == c["regex"]
        shown = e.get("examples",{})
        assert not shown or 1 <= sum(len(v) for v in shown.values()) <= 3
        for side in ("match","no_match"):
            assert len(e["tests"][side]) >= 5
            assert set(e["tests"][side]).isdisjoint(shown.get(side,[]))
            assert set(e["tests"][side]) | set(shown.get(side,[])) == set(c[side])
    report["exported_cases"] = len(exported)
    report["cases_with_examples"] = sum("examples" in e for e in exported)
(ROOT/"audit-report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "numeric_range_ids"},indent=2))
