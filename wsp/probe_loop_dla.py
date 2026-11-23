"""Sonde 8 : dans une boucle, qui pousse le token répété contre le meilleur token de fermeture ?

Mêmes 192 prompts de train que la sonde 7. Aux positions en boucle (une sur trois, après la 2e répétition) :
logit(token réellement émis) − logit(meilleur token fermant parmi `$/`, `)$/`, `}$/`, `/`, `]$/`), décomposé par
tête et par MLP. Pour chaque tête : part de son attention posée exactement une période plus tôt (auto-induction).
"""
import re
import sys
import numpy as np
import mlx.core as mx
import wsplib

edits = [e for e in sys.argv[1].split(",") if e and e != "base"]
m, tok = wsplib.load(edits=edits)
batch = wsplib.read_jsonl(wsplib.ROOT / "data/prepared/train.jsonl")[3::43][:192]
outs = wsplib.generate(m, tok, batch)
LOOP = re.compile(r"(.{3,25}?)\1{4,}")
CLOSE = [tok.encode(s)[0] for s in ["$/", ")$/", "}$/", "/", "]$/"]]
H = np.zeros((28, 16)); M = np.zeros(28); P = np.zeros((28, 16)); n = 0; margins = []
for e, o in zip(batch, outs):
    mt = LOOP.search(o)
    if not mt:
        continue
    p = wsplib.prompts_for(tok, [e])[0]
    a = tok.encode(o)
    ids = p + a
    ends = np.cumsum([len(tok.decode([t])) for t in a])
    start = mt.start() + 2 * len(mt.group(1))
    idx = [len(p) + i for i, c in enumerate(ends) if c > start][:-1]
    for pos in idx[2:40:6]:
        nxt = ids[pos + 1]
        # période en tokens : distance à l'occurrence précédente du même couple (token courant, token suivant)
        back = next((k for k in range(1, 40) if ids[pos - k] == ids[pos] and ids[pos - k + 1] == nxt), None)
        if back is None:
            continue
        o_ = wsplib.forward_heads(m, mx.array([ids[: pos + 1]]), pos)
        best = CLOSE[int(np.argmax(np.array(o_["logits"][mx.array(CLOSE)])))]
        d = wsplib.dla_direction(m, o_["final"], nxt, best)
        H += np.array(o_["heads"] @ d); M += np.array(o_["mlps"] @ d)
        pats = np.array(o_["pats"].astype(mx.float32))
        P += pats[:, :, pos - back + 1]  # la tête regarde le token qui suivait l'occurrence précédente
        margins.append(float(o_["logits"][nxt] - o_["logits"][best])); n += 1
H /= n; M /= n; P /= n
print(f"{n} positions de boucle ; marge token répété − meilleure fermeture : {np.mean(margins):+.2f} (min {np.min(margins):+.1f})")
for l in range(12, 28):
    print(f"  L{l:2} attn {H[l].sum():+6.2f}  mlp {M[l]:+6.2f}")
flat = sorted(((H[l, h], l, h) for l in range(28) for h in range(16)), reverse=True)
print("têtes pro-répétition :", [(f"L{l}H{h}", round(float(v), 2), f"attn période {P[l, h]:.2f}") for v, l, h in flat[:14]])
ind = sorted(((P[l, h], l, h) for l in range(28) for h in range(16)), reverse=True)[:14]
print("têtes d'auto-induction (attention une période plus tôt) :", [(f"L{l}H{h}", round(float(v), 2), f"dla {H[l, h]:+.2f}") for v, l, h in ind])
np.savez(wsplib.WSP / "probes" / "loop_dla.npz", H=H, M=M, P=P)
