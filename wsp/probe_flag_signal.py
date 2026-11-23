"""Sonde 6 : la marge logit(i) − logit(eos) après le / fermant dépend-elle de la description ?

Références forcées (jusqu'au / fermant). 40 cas de train dont la référence porte le flag i, 40 sans flag.
"""
import sys
import numpy as np
import mlx.core as mx
import wsplib
from wsplib import parse_answer

edits = [e for e in sys.argv[1].split(",") if e and e != "base"] if len(sys.argv) > 1 else []
m, tok = wsplib.load(edits=edits)
I, EOS = tok.encode("i")[0], tok.encode("<|im_end|>")[0]
train = wsplib.read_jsonl(wsplib.ROOT / "data/prepared/train.jsonl")
with_i = [e for e in train if parse_answer(e["regex"])[1] == "i"][:40]
without = [e for e in train if parse_answer(e["regex"])[1] == ""][:: 200][:40]
res = {}
for name, group in (("réf. avec flag i", with_i), ("réf. sans flag", without)):
    mg = []
    for e in group:
        ids = wsplib.prompts_for(tok, [e])[0] + tok.encode("/" + parse_answer(e["regex"])[0] + "/")
        lg, _ = wsplib.forward(m, mx.array([ids]))
        l = lg[0, -1].astype(mx.float32)
        mg.append(float(l[I] - l[EOS]))
    res[name] = np.array(mg)
    print(f"{name:18} marge i−eos : moyenne {np.mean(mg):+.2f}, écart-type {np.std(mg):.2f}, déciles 10/50/90 {np.percentile(mg, 10):+.1f} {np.percentile(mg, 50):+.1f} {np.percentile(mg, 90):+.1f}")
a, b = res["réf. avec flag i"], res["réf. sans flag"]
thr = (np.median(a) + np.median(b)) / 2
print(f"séparation au seuil médian {thr:+.2f} : {((a > thr).mean() + (b <= thr).mean()) / 2:.0%} de bien classés")
