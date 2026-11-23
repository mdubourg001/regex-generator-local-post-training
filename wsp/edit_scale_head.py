"""Multiplie la sortie de têtes d'attention par un facteur (0 = ablation) : colonnes de o_proj de la tête.

uv run wsp/edit_scale_head.py --name e04-ablate-16.14 --heads 16.14 --scale 0
"""
import argparse
import numpy as np
import mlx.core as mx
import wsplib

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--heads", required=True)
parser.add_argument("--scale", type=float, required=True)
args = parser.parse_args()
m, tok = wsplib.load()
out = {}
for hd in args.heads.split(","):
    l, h = map(int, hd.split("."))
    name = f"model.layers.{l}.self_attn.o_proj.weight"
    W = out.get(f"full:{name}")
    if W is None:
        W = np.array(m.model.layers[l].self_attn.o_proj.weight.astype(mx.float32))
    W[:, 128 * h : 128 * (h + 1)] *= args.scale
    out[f"full:{name}"] = W
    print(f"L{l}H{h} × {args.scale} : {128 * 1024} poids")
np.savez(wsplib.WSP / "edits" / f"{args.name}.npz", **out)
