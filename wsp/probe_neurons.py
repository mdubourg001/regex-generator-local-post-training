"""Sonde 4 : quels neurones MLP écrivent logit(i) − logit(eos) à la position du flag, et que font-ils ailleurs ?

Référence forcée, 48 cas du pool finissant par `$/`. Pour chaque couche tardive : contribution de chaque neurone
(activation × colonne de down_proj · direction), activation moyenne à la position du flag et ailleurs dans la réponse,
et logit lens de la colonne de sortie (tokens promus).
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import wsplib

m, tok = wsplib.load()
I, EOS = tok.encode("i")[0], tok.encode("<|im_end|>")[0]
pool = [e for e in wsplib.probe_pool() if e["regex"].endswith("$/")][:48]
LAYERS = list(range(18, 28))
acts_flag = {l: 0 for l in LAYERS}; acts_else = {l: 0 for l in LAYERS}; contrib = {l: 0 for l in LAYERS}; n_else = 0

def hidden(layer, h):
    x = layer.post_attention_layernorm(h)
    return (nn.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)).astype(mx.float32)

for e in pool:
    p = wsplib.prompts_for(tok, [e])[0]
    ids = p + tok.encode(e["regex"])
    _, cap = wsplib.forward(m, mx.array([ids]), want=[f"resid_mid.{l}" for l in LAYERS] + ["final"])
    d = wsplib.dla_direction(m, cap["final"][0, -1].astype(mx.float32), I, EOS)
    for l in LAYERS:
        a = hidden(m.model.layers[l], cap[f"resid_mid.{l}"])[0]  # (L, 3072)
        Wd = m.model.layers[l].mlp.down_proj.weight.astype(mx.float32)  # (1024, 3072)
        acts_flag[l] = acts_flag[l] + np.array(a[-1]); acts_else[l] = acts_else[l] + np.array(a[len(p) - 1 : -1].mean(0))
        contrib[l] = contrib[l] + np.array(a[-1] * (d @ Wd))
n = len(pool)
E = m.model.embed_tokens.weight.astype(mx.float32); g = m.model.norm.weight.astype(mx.float32)
save = {}
for l in LAYERS:
    c = contrib[l] / n; af = acts_flag[l] / n; ae = acts_else[l] / n
    save[f"c{l}"] = c; save[f"af{l}"] = af; save[f"ae{l}"] = ae
    order = np.argsort(-c)
    print(f"L{l} MLP total {c.sum():+.2f} ; top5 pro-i = {c[order[:5]].sum():+.2f}, top5 anti-i = {c[order[-5:]].sum():+.2f}")
    for idx in list(order[:4]) + list(order[-3:]):
        col = m.model.layers[l].mlp.down_proj.weight[:, int(idx)].astype(mx.float32)
        lens = E @ (col * g)
        top = [tok.decode([t]) for t in mx.argsort(-lens)[:8].tolist()]; bot = [tok.decode([t]) for t in mx.argsort(lens)[:5].tolist()]
        print(f"    n{idx:4} contrib {c[idx]:+.2f}  act flag {af[idx]:+.2f} / ailleurs {ae[idx]:+.2f}  promeut {top}  réprime {bot}")
np.savez(wsplib.WSP / "probes" / "neurons_flag.npz", **save)
