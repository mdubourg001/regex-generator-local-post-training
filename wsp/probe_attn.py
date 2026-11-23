"""Sonde 3 : depuis une position de la réponse, quelles têtes regardent quels tokens de l'exemple du prompt système ?

Référence forcée sur 32 cas du pool. Requêtes : après `$/` final (flag), et moyenne sur toutes les positions de la réponse.
"""
import sys
import numpy as np
import mlx.core as mx
import wsplib

edits = [e for e in sys.argv[1].split(",") if e] if len(sys.argv) > 1 else []
m, tok = wsplib.load(edits=edits)
pool = [e for e in wsplib.probe_pool() if e["regex"].endswith("$/")][:32]
EX = list(range(57, 62))  # ' /^\', 'd', '+', '$/', 'i'
A_flag, A_body, n_body = 0, 0, 0
for e in pool:
    p = wsplib.prompts_for(tok, [e])[0]
    ids = p + tok.encode(e["regex"])
    o = wsplib.forward_heads(m, mx.array([ids]), len(ids) - 1)
    A_flag = A_flag + np.array(o["pats"][:, :, 57:62].astype(mx.float32))
    for pos in range(len(p), len(ids) - 1, 3):
        o = wsplib.forward_heads(m, mx.array([ids]), pos)
        A_body = A_body + np.array(o["pats"][:, :, 57:62].astype(mx.float32)); n_body += 1
A_flag /= len(pool); A_body /= n_body
names = ["' /^\\'", "d", "+", "$/", "i"]
for title, A in (("requête = `$/` final (prédiction du flag)", A_flag), ("requête = positions du corps de la réponse", A_body)):
    print(title)
    for j, nm in enumerate(names):
        top = sorted(((A[l, h, j], l, h) for l in range(28) for h in range(16)), reverse=True)[:12]
        print(f"  → {nm:6} " + " ".join(f"{l}.{h}:{v:.2f}" for v, l, h in top))
np.savez(wsplib.WSP / "probes" / "attn_example.npz", flag=A_flag, body=A_body)
