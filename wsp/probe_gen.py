"""Sonde de génération : 64 cas du pool de sonde (hors dev), mêmes réglages que l'éval. Ne compte pas dans le budget.
Sert à vérifier le mécanisme d'une édition (flag, format), pas à choisir des coefficients.

uv run wsp/probe_gen.py e01-blind-flag[,autre] [--show]
"""
import json
import sys
import numpy as np
import mlx.core as mx
import wsplib
from wsplib import parse_answer, score

edits = [e for e in sys.argv[1].split(",") if e and e != "base"]
mx.set_memory_limit(6 * 1024**3)
m, tok = wsplib.load(edits=edits)
I, EOS = tok.encode("i")[0], tok.encode("<|im_end|>")[0]
# marge du flag, référence forcée
pool = [e for e in wsplib.probe_pool() if e["regex"].endswith("$/")][:48]
mg, top = [], []
for e in pool:
    ids = wsplib.prompts_for(tok, [e])[0] + tok.encode(e["regex"])
    lg, _ = wsplib.forward(m, mx.array([ids]))
    l = lg[0, -1].astype(mx.float32)
    mg.append(float(l[I] - l[EOS])); top.append(tok.decode([int(mx.argmax(l))]))
import collections
print(f"flag : marge logit(i)−logit(eos) {np.mean(mg):+.2f} ; token suivant le / final : {collections.Counter(top).most_common(5)}")
pool = wsplib.probe_pool()[:64]
outs = wsplib.generate(m, tok, pool)
g = wsplib.grade_outputs(pool, outs)
s = score(g)
parsed = [p for p in map(parse_answer, outs) if p]
print(f"{'+'.join(edits) or 'base':40} succès {sum(x['success'] for x in g):2}/64  tests {s['mean_pass_rate']:.3f}  malformés {s['malformed_rate']:.1%}  flag i {sum('i' in p[1] for p in parsed)}/{len(parsed)}  \\d {sum(chr(92) + 'd' in p[0] for p in parsed)}/{len(parsed)}  ^…$ {sum(p[0][0] == '^' and p[0][-1] == '$' for p in parsed)}/{len(parsed)}")
import re
loops = sum(1 for o in outs if len(o) > 250 or re.search(r"(.{3,25}?)\1{4,}", o))
print(f"  boucles {loops}/64 ; longueur médiane {int(np.median([len(o) for o in outs]))} caractères")
if "--show" in sys.argv:
    for e, o, x in zip(pool, outs, g):
        print(" ", "OK " if x["success"] else "MAL" if x["status"] != "ok" else "   ", repr(o[:100]), "|| REF", e["regex"][:60])
