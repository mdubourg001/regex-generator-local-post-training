"""Sonde 1 : qui écrit le flag `i` après le / fermant ?

48 cas du pool de sonde (hors dev) dont la référence finit par `$/` sans flag. On force la référence et on regarde la
prédiction juste après le dernier token : logit(i) − logit(<|im_end|>), décomposé par tête et par MLP.
"""
import numpy as np
import mlx.core as mx
import wsplib

m, tok = wsplib.load()
I, EOS = tok.encode("i")[0], tok.encode("<|im_end|>")[0]
pool = [e for e in wsplib.probe_pool() if e["regex"].endswith("$/")][:48]
H, M, P, margin, pi = 0, 0, 0, [], []
sys_i = None
for e in pool:
    ids = wsplib.prompts_for(tok, [e])[0] + tok.encode(e["regex"])
    if sys_i is None:
        sys_i = [k for k in range(80) if tok.decode([ids[k]]) == "$/"][0] + 1  # le `i` de l'exemple du prompt système
    o = wsplib.forward_heads(m, mx.array([ids]), len(ids) - 1)
    d = wsplib.dla_direction(m, o["final"], I, EOS)
    H = H + np.array(o["heads"] @ d); M = M + np.array(o["mlps"] @ d)
    P = P + np.array(o["pats"][:, :, sys_i - 1 : sys_i + 1].astype(mx.float32).sum(-1))
    pr = mx.softmax(o["logits"])
    margin.append(float(o["logits"][I] - o["logits"][EOS])); pi.append((float(pr[I]), float(pr[EOS])))
n = len(pool)
H, M, P = H / n, M / n, P / n
print(f"n={n}  marge moyenne logit(i)-logit(eos) = {np.mean(margin):+.2f}   P(i)={np.mean([a for a,_ in pi]):.2f}  P(eos)={np.mean([b for _,b in pi]):.2f}")
print("contribution par couche (attention | MLP) :")
for l in range(28):
    print(f"  L{l:2}  attn {H[l].sum():+6.2f}  mlp {M[l]:+6.2f}   têtes: " + " ".join(f"{x:+.1f}" for x in H[l]))
flat = sorted(((H[l, h], l, h) for l in range(28) for h in range(16)), reverse=True)
print("têtes les plus pro-i :", [(f"L{l}H{h}", round(float(v), 2), f"attn→ex {P[l, h]:.2f}") for v, l, h in flat[:10]])
print("têtes les plus anti-i :", [(f"L{l}H{h}", round(float(v), 2), f"attn→ex {P[l, h]:.2f}") for v, l, h in flat[-5:]])
att = sorted(((P[l, h], l, h) for l in range(28) for h in range(16)), reverse=True)[:12]
print("têtes qui regardent le `$/i` du prompt système :", [(f"L{l}H{h}", round(float(v), 2), f"dla {H[l, h]:+.2f}") for v, l, h in att])
np.savez(wsplib.WSP / "probes" / "flag.npz", H=H, M=M, P=P)
