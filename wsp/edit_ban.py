"""Bannit des tokens de la sortie : baisse uniforme de leur logit dans les réponses, par édition de leur ligne
d'embedding (liée à l'unembedding).

Hypothèse : ces tokens fusionnent le / fermant avec autre chose (`/.`, `/,`, `/i`…) ; aucune réponse valide
`/…/flags` ne les contient, car un / dans un pattern s'écrit `\\/` ou dans une classe.

    μ̂ = direction moyenne du résiduel final normalisé (g ⊙ h / rms) sur les positions de réponse (64 exemples de train)
    E_t ← E_t − β μ̂      donc      logit_t ← logit_t − β (n · μ̂)
    β = shift / moyenne de (n · μ̂)

uv run wsp/edit_ban.py --name e11-ban --tokens "/. /, /i" --shift 8
"""
import argparse
import numpy as np
import mlx.core as mx
import wsplib

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--tokens", default="")
parser.add_argument("--slash-family", action="store_true", help="tous les tokens qui contiennent un / non échappé suivi d'autre chose, sauf `/^`")
parser.add_argument("--keep-wrappers", action="store_true", help="épargne les tokens où le / est suivi d'un guillemet, d'un backtick ou d'un saut de ligne (fermeture d'une réponse enveloppée, tolérée par le parseur)")
parser.add_argument("--shift", type=float, required=True)
parser.add_argument("--edits", default="")
args = parser.parse_args()
m, tok = wsplib.load(edits=[e for e in args.edits.split(",") if e])
ids_ban = []
for s in args.tokens.split():
    t = tok.encode(s)
    assert len(t) == 1, (s, t)
    ids_ban.append(t[0])
if args.slash_family:
    import re
    # Un / non échappé puis au moins un caractère : dans une réponse valide, seul le token d'ouverture peut avoir
    # cette forme. On garde `/^` (ouverture ancrée, favorisée par l'édition « first ») ; l'ouverture non ancrée passe
    # par le token `/` seul.
    wrap = re.compile("/[\"'`\n]")
    fam = [i for i in range(151643) if re.search(r"(?<!\\)/.", s := tok.decode([i])) and s != "/^" and not (args.keep_wrappers and wrap.search(s))]
    print(f"famille slash : {len(fam)} tokens, ex. {[tok.decode([i]) for i in fam[:40]]}")
    ids_ban += fam
g = np.array(m.model.norm.weight.astype(mx.float32))


def normed(examples):
    out = []
    for e in examples:
        p = wsplib.prompts_for(tok, [e])[0]
        ids = p + tok.encode(e["regex"])
        _, cap = wsplib.forward(m, mx.array([ids]), want=["final"])
        h = np.array(cap["final"][0, len(p) - 1 :].astype(mx.float32))
        out.append(g * h / np.sqrt((h**2).mean(-1, keepdims=True)))
    return np.concatenate(out)


N = normed(wsplib.read_jsonl(wsplib.ROOT / "data/prepared/train.jsonl")[:: 8321 // 64][:64])
mu = N.mean(0); mu /= np.linalg.norm(mu)
proj = N @ mu
beta = args.shift / proj.mean()
N2 = normed(wsplib.probe_pool()[64:96]) @ mu
print(f"n·μ̂ sur les réponses : train {proj.mean():.1f} ± {proj.std():.1f} ; pool de sonde {N2.mean():.1f} ± {N2.std():.1f}, min {N2.min():.1f} → baisse de logit {beta * N2.mean():.1f} ± {beta * N2.std():.1f}")
E = np.array(m.model.embed_tokens.weight[mx.array(ids_ban)].astype(mx.float32))
print(f"β = {beta:.3f} ; norme des lignes {np.linalg.norm(E, axis=1).round(2)} ; {len(ids_ban)} lignes × 1024 = {len(ids_ban) * 1024} poids")
U = np.zeros((151936, 1), np.float32); U[ids_ban, 0] = -beta
np.savez(wsplib.WSP / "edits" / f"{args.name}.npz", **{"U:model.embed_tokens.weight": U, "V:model.embed_tokens.weight": mu[None].astype(np.float32)})
