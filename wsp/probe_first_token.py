"""Sonde 9 : le choix du premier token change-t-il le corps de la regex ?

64 exemples de train. On force le premier token de la réponse (`/` seul, `/^` d'un bloc, `"/`, ```` ```javascript\\n/ ````)
puis on laisse le modèle continuer en greedy. Même prompt, mêmes poids : seule l'amorce change.
"""
import sys
import mlx.core as mx
from mlx_lm import batch_generate
import wsplib
from wsplib import parse_answer, score

edits = [e for e in sys.argv[1].split(",") if e and e != "base"]
m, tok = wsplib.load(edits=edits)
ex = wsplib.read_jsonl(wsplib.ROOT / "data/prepared/train.jsonl")[7::130][:64]
prompts = wsplib.prompts_for(tok, ex)
for forced in ["/", "/^", '"/', "```javascript\n/"]:
    f = tok.encode(forced)
    outs = []
    for i in range(0, 64, 16):
        outs += [forced + t for t in batch_generate(m, tok, [p + f for p in prompts[i : i + 16]], max_tokens=127).texts]
    g = wsplib.grade_outputs(ex, outs)
    s = score(g)
    parsed = [p for p in map(parse_answer, outs) if p]
    lazy = sum(p[0] in ("^\\d+$", "^\\w+$", "^\\S+$") or p[0].startswith("^\\d+") or p[0].startswith("^\\w+") for p in parsed)
    print(f"amorce {forced!r:20} succès {sum(x['success'] for x in g):2}/64  tests {s['mean_pass_rate']:.3f}  malformés {s['malformed_rate']:.1%}  corps commençant par ^\\d+ ou ^\\w+ : {lazy}/{len(parsed)}  \\d : {sum(chr(92) + 'd' in p[0] for p in parsed)}  classe [...] : {sum('[' in p[0] for p in parsed)}  longueur médiane {sorted(len(p[0]) for p in parsed)[len(parsed) // 2]}")
