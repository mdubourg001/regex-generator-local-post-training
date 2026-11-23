"""Édition « mémoire » de rang 1 dans un MLP : à un type de position donné, écrire une direction choisie.

Lecture du MLP comme mémoire clé → valeur : W_down += α · d ⊗ cᵀ.

- clé c (espace des 3072 neurones) : différence de moyennes des activations cachées entre les positions visées et les
  positions de contrôle, orthogonalisée à la moyenne de contrôle (réponse moyenne nulle hors cible), normalisée pour
  valoir 1 sur la moyenne des positions visées. Calculée sur 64 exemples de train (références forcées).
- valeur d (résiduel) : moyenne des embeddings des tokens à favoriser − moyenne de ceux à défavoriser, normée.
- α : fixé analytiquement pour déplacer la marge de logits de `--shift`, norme finale gelée à sa valeur mesurée.
  Pas de recherche : `--shift` est la marge observée par la sonde, plus une réserve.

Positions (--where) :
  first : dernier token du prompt (prédiction du 1er token de la réponse)
  end   : dernier token d'une référence sans flag (prédiction attendue : fin de séquence), sous ses deux découpages
          en tokens : `…$/` d'un bloc, ou `…$` puis `/` seul
  close : référence sans flag privée de son / final (prédiction attendue : le token `/` seul)
Contrôle : toutes les autres positions de la réponse, plus un échantillon de positions du prompt.

uv run wsp/edit_memory.py --name e03-first --where first --layer 27 --plus "/,/^" --minus '"/,'"'"'/,`,```,",\\x20/' --shift 8
"""
import argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import wsplib

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--where", required=True, choices=["first", "end", "close"])
parser.add_argument("--layer", type=int, default=27)
parser.add_argument("--plus", required=True, help="tokens à favoriser, séparés par des espaces")
parser.add_argument("--minus", default="", help="tokens à défavoriser, séparés par des espaces ; ␠ = espace")
parser.add_argument("--shift", type=float, required=True, help="déplacement visé de la marge de logits")
parser.add_argument("--top", type=int, default=0, help="clé creuse : ne garder que les N neurones les plus discriminants (z-score cible vs contrôle)")
parser.add_argument("--edits", default="")
args = parser.parse_args()

m, tok = wsplib.load(edits=[e for e in args.edits.split(",") if e])
layer = m.model.layers[args.layer]


def one(s):
    ids = tok.encode(s.replace("␠", " ").replace("⏎", "\n"))
    assert len(ids) == 1, (s, ids)
    return ids[0]


plus, minus = [one(s) for s in args.plus.split()], [one(s) for s in args.minus.split()]


def collect(examples):
    """Activations cachées du MLP aux positions visées et de contrôle, et rms du résiduel final aux positions visées."""
    keys, ctrl, rms = [], [], []
    SLASH = tok.encode("/")[0]
    for e in examples:
        p = wsplib.prompts_for(tok, [e])[0]
        full = p + tok.encode(e["regex"])
        cut = p + tok.encode(e["regex"][:-1])
        no_flag = e["regex"].endswith("/")
        # (ids, position visée ou None) ; la première variante fournit aussi les positions de contrôle
        variants = [(full, len(p) - 1 if args.where == "first" else len(full) - 1 if args.where == "end" and no_flag else None)]
        if args.where == "end" and no_flag:
            variants.append((cut + [SLASH], len(cut)))
        if args.where == "close" and no_flag:
            variants.append((cut, len(cut) - 1))
        for v, (ids, target) in enumerate(variants):
            _, cap = wsplib.forward(m, mx.array([ids]), want=[f"resid_mid.{args.layer}", "final"])
            x = layer.post_attention_layernorm(cap[f"resid_mid.{args.layer}"])
            a = np.array((nn.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x))[0].astype(mx.float32))
            fin = np.array(cap["final"][0].astype(mx.float32))
            if target is not None:
                keys.append(a[target]); rms.append(np.sqrt((fin[target] ** 2).mean()))
            if v == 0:
                for pos in list(range(len(p) - 1, len(ids))) + list(range(70, len(p) - 1, 7)):
                    if pos != target:
                        ctrl.append(a[pos])
    return np.stack(keys), np.stack(ctrl), float(np.mean(rms))


train = wsplib.read_jsonl(wsplib.ROOT / "data/prepared/train.jsonl")[:: 8321 // 64][:64]
K, C, rms = collect(train)
mk, mc = K.mean(0), C.mean(0)
c = mk - mc
if args.top:  # neurones dont l'activation distingue le mieux la cible du contrôle
    z = np.abs(c) / (K.std(0) + C.std(0) + 1e-6)
    keep = np.argsort(-z)[: args.top]
    mask = np.zeros_like(c); mask[keep] = 1
    c, mc = c * mask, mc * mask
    print(f"clé creuse : {args.top} neurones, z de {z[keep].min():.1f} à {z[keep].max():.1f} ; les 8 premiers : {keep[:8].tolist()}")
c -= (c @ mc) / (mc @ mc) * mc
c /= mk @ c

E = np.array(m.model.embed_tokens.weight[mx.array(plus + minus)].astype(mx.float32))
g = np.array(m.model.norm.weight.astype(mx.float32))
diff = E[: len(plus)].mean(0) - (E[len(plus) :].mean(0) if minus else 0)  # sans --minus : hausse du logit des tokens favorisés contre tous les autres
d = diff / np.linalg.norm(diff)
per_alpha = diff @ (g * d) / rms  # marge de logits gagnée par unité de α, norme finale gelée
alpha = args.shift / per_alpha
print(f"L{args.layer} down_proj, rang 1 : |d|=1, α={alpha:.2f} (résiduel final rms {rms:.2f}, soit ×{alpha / (rms * 32):.2f} sa norme), {len(K)} clés, {len(C)} contrôles")

# Sélectivité, sur des prompts jamais utilisés pour la clé (pool de sonde).
K2, C2, _ = collect(wsplib.probe_pool()[64:96])
fk, fc = K2 @ c, C2 @ c
print(f"réponse de la clé — cible : moyenne {fk.mean():.2f}, min {fk.min():.2f} ; contrôle : moyenne {fc.mean():+.3f}, écart-type {fc.std():.3f}, max |.| {np.abs(fc).max():.2f}, part > 0,3 : {(np.abs(fc) > 0.3).mean():.1%}")

name = f"model.layers.{args.layer}.mlp.down_proj.weight"
np.savez(wsplib.WSP / "edits" / f"{args.name}.npz", **{f"U:{name}": (alpha * d)[:, None].astype(np.float32), f"V:{name}": c[None].astype(np.float32)})
