"""Sonde 7 : existe-t-il une « direction de boucle » dans le flux résiduel, et qui l'écrit ?

3 lots de 64 prompts de train. On génère avec la pile de format, on repère les sorties en boucle (un motif de 3 à 25
caractères répété ≥ 5 fois) et, dans chacune, les tokens situés après la 2e répétition. Différence de moyennes du
résiduel (par couche) entre ces positions et les positions de réponse hors boucle. On mesure la cohérence de cette
direction d'une boucle à l'autre, qui l'écrit (attention ou MLP, par couche) et ce qu'elle promeut (logit lens).
"""
import re
import sys
import numpy as np
import mlx.core as mx
import wsplib

edits = [e for e in sys.argv[1].split(",") if e and e != "base"]
m, tok = wsplib.load(edits=edits)
train = wsplib.read_jsonl(wsplib.ROOT / "data/prepared/train.jsonl")
batch = train[3::43][:192]
outs = wsplib.generate(m, tok, batch)
LOOP = re.compile(r"(.{3,25}?)\1{4,}")
want = [f"{k}.{l}" for l in range(28) for k in ("attn_out", "mlp_out", "resid_post")]
loop_sum, norm_sum, n_loop, n_norm = {}, {}, 0, 0
per_loop = []
kept = 0
for e, o in zip(batch, outs):
    mt = LOOP.search(o)
    if not mt and kept >= 40 and len(o) < 250:
        continue
    p = wsplib.prompts_for(tok, [e])[0]
    a = tok.encode(o)
    ids = p + a
    # position en caractères de chaque token de la réponse
    ends = np.cumsum([len(tok.decode([t])) for t in a])
    if mt:
        start = mt.start() + 2 * len(mt.group(1))
        idx = [len(p) + i for i, c in enumerate(ends) if c > start][:-1]
    elif len(o) < 250:
        idx = list(range(len(p), len(ids))); kept += 1
    else:
        continue
    if not idx:
        continue
    _, cap = wsplib.forward(m, mx.array([ids]), want=want)
    vec = {k: np.array(cap[k][0].astype(mx.float32))[idx] for k in want}
    tgt, cnt = (loop_sum, "loop") if mt else (norm_sum, "norm")
    for k in want:
        tgt[k] = tgt.get(k, 0) + vec[k].sum(0)
    if mt:
        n_loop += len(idx); per_loop.append({l: vec[f"resid_post.{l}"].mean(0) for l in (15, 19, 23, 27)})
    else:
        n_norm += len(idx)
print(f"{len(per_loop)} sorties en boucle sur {len(batch)} ({n_loop} positions), {kept} sorties normales ({n_norm} positions)")
E = m.model.embed_tokens.weight.astype(mx.float32); g = m.model.norm.weight.astype(mx.float32)
dirs = {}
for l in range(28):
    d = loop_sum[f"resid_post.{l}"] / n_loop - norm_sum[f"resid_post.{l}"] / n_norm
    base = np.linalg.norm(norm_sum[f"resid_post.{l}"] / n_norm)
    dirs[l] = d
    dh = d / np.linalg.norm(d)
    wa = (loop_sum[f"attn_out.{l}"] / n_loop - norm_sum[f"attn_out.{l}"] / n_norm)
    wm = (loop_sum[f"mlp_out.{l}"] / n_loop - norm_sum[f"mlp_out.{l}"] / n_norm)
    line = f"L{l:2} |d|={np.linalg.norm(d):6.2f} ({np.linalg.norm(d) / base:.2f} × moyenne)"
    if l in (15, 19, 23, 27):
        ref = np.stack([x[l] for x in per_loop]) - norm_sum[f"resid_post.{l}"] / n_norm
        cos = ref @ dh / np.linalg.norm(ref, axis=1)
        line += f"  cohérence cos par boucle : moyenne {cos.mean():.2f}, min {cos.min():.2f}"
    print(line)
final = dirs[27] / np.linalg.norm(dirs[27])
print("écriture de la direction finale (L27) par couche : attention | MLP")
for l in range(28):
    wa = (loop_sum[f"attn_out.{l}"] / n_loop - norm_sum[f"attn_out.{l}"] / n_norm) @ final
    wm = (loop_sum[f"mlp_out.{l}"] / n_loop - norm_sum[f"mlp_out.{l}"] / n_norm) @ final
    print(f"  L{l:2} attn {wa:+7.2f}  mlp {wm:+7.2f}")
lens = E @ (mx.array(final) * g)
print("logit lens de la direction : promeut", [tok.decode([t]) for t in mx.argsort(-lens)[:20].tolist()])
print("                             réprime", [tok.decode([t]) for t in mx.argsort(lens)[:20].tolist()])
np.savez(wsplib.WSP / "probes" / "loop_dirs.npz", **{f"L{l}": dirs[l] for l in range(28)})
