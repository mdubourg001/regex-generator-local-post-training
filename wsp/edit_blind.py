"""Édition « aveugle » : rend des têtes d'attention aveugles au contenu propre d'un token fixe du prompt système.

Le prompt système est identique pour tous les cas : le résiduel à une position p du prompt système est donc un vecteur
constant x_l[p] à chaque couche l (attention causale). Pour une tête donnée, on retire de v_proj la composante
spécifique de ce vecteur :

    u = x_l[p] − moyenne des x_l sur les autres positions du prompt système ;  û = u / |u|
    W_V[lignes de la tête kv] ← W_V − (W_V û) ûᵀ          (rang 1 par bloc 128 × 1024)

La tête continue de lire cette position, mais n'y trouve plus que le contenu « moyen » d'un token du prompt système.
Aucune donnée d'entraînement n'intervient : û vient d'une seule passe forward sur le prompt système.

uv run wsp/edit_blind.py --name e01-blind-flag --pos 61 --heads 16.14,19.5,...
"""
import argparse
import numpy as np
import mlx.core as mx
import wsplib
import common

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--pos", required=True, help="positions du prompt système, ex. 61 ou 58,59")
parser.add_argument("--heads", required=True, help="têtes couche.tête, ex. 16.14,19.5")
parser.add_argument("--keep", default="", help="positions dont la lecture doit rester intacte : û leur est orthogonalisé")
parser.add_argument("--edits", default="", help="éditions déjà appliquées avant de calculer û")
args = parser.parse_args()

m, tok = wsplib.load(edits=[e for e in args.edits.split(",") if e])
ids = tok.apply_chat_template([{"role": "system", "content": common.SYSTEM_PROMPT}], add_generation_prompt=False, enable_thinking=False)
end = ids.index(tok.encode("<|im_end|>")[0])
print("tokens visés :", [(p, tok.decode([ids[int(p)]])) for p in args.pos.split(",")])
layers = sorted({int(h.split(".")[0]) for h in args.heads.split(",")})
_, cap = wsplib.forward(m, mx.array([ids[: end + 1]]), want=[f"resid_pre.{l}" for l in layers])
out = {}
for l in layers:
    x = np.array(m.model.layers[l].input_layernorm(cap[f"resid_pre.{l}"])[0].astype(mx.float32))
    W = np.array(m.model.layers[l].self_attn.v_proj.weight.astype(mx.float32))
    kvs = sorted({int(h.split(".")[1]) // 2 for h in args.heads.split(",") if int(h.split(".")[0]) == l})
    U, V = [], []
    for p in map(int, args.pos.split(",")):
        others = [q for q in range(3, end) if q != p]
        mean = x[others].mean(0)
        u = x[p] - mean
        if args.keep:  # û ⟂ contenu propre des positions à préserver : W_V (x_q − moyenne) reste exactement inchangé
            K = np.stack([x[int(q)] - mean for q in args.keep.split(",")])
            Q, _ = np.linalg.qr(K.T)
            kept = np.linalg.norm(u - Q @ (Q.T @ u)) / np.linalg.norm(u)
            u = u - Q @ (Q.T @ u)
            print(f"  L{l} pos {p} : part de u conservée après orthogonalisation {kept:.2f}")
        u /= np.linalg.norm(u)
        col = np.zeros((1024, 1), np.float32)
        for kv in kvs:
            rows = slice(128 * kv, 128 * (kv + 1))
            col[rows, 0] = -(W[rows] @ u)
        W = W + col @ u[None]  # les positions suivantes sont projetées sur le W déjà édité
        U.append(col); V.append(u[None])
    name = f"model.layers.{l}.self_attn.v_proj.weight"
    out[f"U:{name}"] = np.concatenate(U, 1)
    out[f"V:{name}"] = np.concatenate(V, 0)
    print(f"L{l} têtes kv {kvs} : {len(kvs) * 128 * 1024} poids, rang {len(U)}")
np.savez(wsplib.WSP / "edits" / f"{args.name}.npz", **out)
