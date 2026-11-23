"""Classe les échecs d'un run de dev (wsp/runs/<name>.jsonl). Aucune génération : re-note des variantes contrefactuelles.

uv run wsp/diagnose.py e00-base
"""

import collections
import re
import sys

import wsplib
from wsplib import parse_answer


def malformed_kind(out, error):
    t = out.strip()
    if len(t) > 250 or re.search(r"(.{3,25}?)\1{4,}", t):
        return "boucle (répétition jusqu'à 128 tokens)"
    core = re.sub(r"^```[\w-]*\n|\n?```$", "", t).strip().strip("`\"'").strip()
    if "/" not in core[:1]:
        return "pas un littéral : chaîne exemple ou pattern sans /…/"
    if error and "not a single" not in error:
        return "littéral qui ne compile pas (new RegExp lève)"
    if re.search(r"/\.(test|i|g)\b", core) or re.search(r"/[a-z]*\.test", core):
        return "suffixe cassé : /.i, .test"
    return "littéral cassé : / final absent, / non échappé, texte autour"


def main(name):
    rows = wsplib.read_jsonl(wsplib.WSP / "runs" / f"{name}.jsonl")
    ex = {e["id"]: e for e in wsplib.dev_set()}
    examples = [ex[r["id"]] for r in rows]
    n = len(rows)

    def regrade(transform):
        outs = []
        for r in rows:
            p = parse_answer(r["output"])
            outs.append(transform(*p) if p else "")
        return wsplib.grade_outputs(examples, outs)

    lit = lambda p, f: f"/{p}/{f}"
    no_i = regrade(lambda p, f: lit(p, f.replace("i", "")))
    anchored = regrade(lambda p, f: lit(("" if p.startswith("^") else "^") + "(?:" + p.lstrip("^").removesuffix("$") + ")$" if not (p.startswith("^") and p.endswith("$")) else p, f))
    both = regrade(lambda p, f: lit(("" if p.startswith("^") else "^") + "(?:" + p.lstrip("^").removesuffix("$") + ")$" if not (p.startswith("^") and p.endswith("$")) else p, f.replace("i", "")))

    cats, kinds = collections.Counter(), collections.Counter()
    for i, r in enumerate(rows):
        if r["success"]:
            cats["0. succès"] += 1
        elif r["status"] != "ok":
            cats["1. malformé"] += 1
            kinds[malformed_kind(r["output"], r["error"])] += 1
        elif no_i[i]["success"]:
            cats["2. habitude : flag i en trop (juste sans lui)"] += 1
        elif anchored[i]["success"] or both[i]["success"]:
            cats["3. ancrage : juste une fois ancré ^…$"] += 1
        elif r["match_rate"] == 1:
            cats["4. sur-généralisation (accepte tout le valide, et trop)"] += 1
        elif r["no_match_rate"] == 1:
            cats["5. sous-généralisation (ne refuse rien à tort, rate du valide)"] += 1
        else:
            cats["6. mauvaise sémantique (rate des deux côtés)"] += 1

    print(f"{name} : {n} cas")
    for k, v in sorted(cats.items()):
        print(f"  {k:62} {v:4}  {v / n:.1%}")
    print("  détail des malformés :")
    for k, v in kinds.most_common():
        print(f"    {k:60} {v:4}  {v / n:.1%}")

    ok = [r for r in rows if r["status"] == "ok"]
    parsed = [parse_answer(r["output"]) for r in ok]
    refs = [parse_answer(r["reference"]) for r in ok]
    stat = lambda f: sum(map(f, parsed)) / len(ok)
    print(f"  habitudes, sur {len(ok)} sorties bien formées (référence entre parenthèses) :")
    print(f"    flag i présent            {stat(lambda p: 'i' in p[1]):.1%}  ({sum('i' in p[1] for p in refs) / len(ok):.1%})")
    print(f"    ancre ^ au début          {stat(lambda p: p[0].startswith('^')):.1%}  ({sum(p[0].startswith('^') for p in refs) / len(ok):.1%})")
    print(f"    ancre $ à la fin          {stat(lambda p: p[0].endswith('$')):.1%}  ({sum(p[0].endswith('$') for p in refs) / len(ok):.1%})")
    print(f"    copie exacte de /^\\d+$/i  {stat(lambda p: p == (chr(94) + chr(92) + 'd+$', 'i')):.1%}")
    wrap = collections.Counter("fence" if r["output"].strip().startswith("```") else "guillemets" if r["output"].strip()[:1] in "\"'`" else "nu" for r in rows)
    print(f"  enveloppe (tous les cas) : {dict(wrap)}")
    escaping = sum(1 for r in rows if r["status"] != "ok" and r["error"] and "not a single" not in r["error"])
    print(f"  erreurs de compilation (échappement, quantificateur, groupe) : {escaping}")
    print(f"  pass_rate moyen {sum(r['pass_rate'] for r in rows) / n:.3f} | sans i : succès {sum(g['success'] for g in no_i)} | ancré : {sum(g['success'] for g in anchored)} | les deux : {sum(g['success'] for g in both)}")


if __name__ == "__main__":
    main(sys.argv[1])
