"""Sonde 5 : la copie du corps de l'exemple. Après `/^\\` forcé, le modèle prédit-il `d` parce que l'exemple dit `/^\\d+$/i` ?

48 cas du pool. On force `/` `^\\` puis, pour la suite `d` → `+` → `$/`, on mesure la proba du token de l'exemple et
quelles têtes regardent le token correspondant du prompt système (induction).
"""
import sys
import collections
import numpy as np
import mlx.core as mx
import wsplib

edits = [e for e in sys.argv[1].split(",") if e and e != "base"] if len(sys.argv) > 1 else []
m, tok = wsplib.load(edits=edits)
pool = wsplib.probe_pool()[:48]
needs_d = [chr(92) + "d" in e["regex"] for e in pool]
steps = [(["/", "^\\"], "d", 58), (["/", "^\\", "d"], "+", 59), (["/", "^\\", "d", "+"], "$/", 60)]
for forced, nxt, src in steps:
    f = [tok.encode(s)[0] for s in forced]; t = tok.encode(nxt)[0]
    A, pr, top = 0, [], collections.Counter()
    for e in pool:
        ids = wsplib.prompts_for(tok, [e])[0] + f
        o = wsplib.forward_heads(m, mx.array([ids]), len(ids) - 1)
        p = mx.softmax(o["logits"]); pr.append(float(p[t])); top[tok.decode([int(mx.argmax(p))])] += 1
        A = A + np.array(o["pats"][:, :, src].astype(mx.float32))
    A /= len(pool)
    heads = sorted(((A[l, h], l, h) for l in range(28) for h in range(16)), reverse=True)[:14]
    extra = f" (réf. avec \\d : {np.mean([p for p, n in zip(pr, needs_d) if n]):.2f}, sans : {np.mean([p for p, n in zip(pr, needs_d) if not n]):.2f})" if nxt == "d" else ""
    print(f"après {''.join(forced)!r} : P({nxt!r}) = {np.mean(pr):.2f}{extra} ; top-1 {top.most_common(4)}")
    print(f"   têtes → pos {src} : " + " ".join(f"{l}.{h}:{v:.2f}" for v, l, h in heads))
