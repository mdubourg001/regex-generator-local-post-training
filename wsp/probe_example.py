"""Sonde 2 : à quel point la sortie dépend-elle de l'exemple `/^\\d+$/i` du prompt système ?

Intervention au niveau du prompt (sonde causale, pas une édition) sur 64 cas du pool de sonde : on change l'exemple
et on regarde le flag i, la copie de \\d+, les malformés et le taux de tests réussis.
"""
import collections
import mlx.core as mx
from mlx_lm import batch_generate

import wsplib
import common
from wsplib import parse_answer, score

m, tok = wsplib.load()
pool = wsplib.probe_pool()[:64]
VARIANTS = {
    "original /^\\d+$/i": "/^\\d+$/i",
    "sans flag /^\\d+$/": "/^\\d+$/",
    "autre corps /^[a-z]+$/i": "/^[a-z]+$/i",
    "neutre /^…$/": "/^…$/",
}
for name, lit in VARIANTS.items():
    sysp = common.SYSTEM_PROMPT.replace("/^\\d+$/i", lit)
    assert lit in sysp
    prompts = [tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": common.build_user_prompt(e)}], add_generation_prompt=True, enable_thinking=False) for e in pool]
    outs = []
    for i in range(0, 64, 16):
        outs += batch_generate(m, tok, prompts[i : i + 16], max_tokens=128).texts
    g = wsplib.grade_outputs(pool, outs)
    s = score(g)
    parsed = [p for p in map(parse_answer, outs) if p]
    print(f"{name:28} succès {sum(x['success'] for x in g):2}/64  tests {s['mean_pass_rate']:.3f}  malformés {s['malformed_rate']:.1%}  flag i {sum('i' in p[1] for p in parsed)}/{len(parsed)}  contient \\d {sum(chr(92) + 'd' in p[0] for p in parsed)}/{len(parsed)}  ^…$ {sum(p[0][0] == '^' and p[0][-1] == '$' for p in parsed)}/{len(parsed)}")
